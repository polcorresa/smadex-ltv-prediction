#!/usr/bin/env python3
"""Lightweight holdout evaluator that reports RMSLE (Kaggle metric).

Usage:
    uv run python scripts/evaluate_holdout.py --split val --limit-rows 20000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

# Ensure local imports resolve
sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.inference.predictor import FastPredictor
from src.utils.logger import setup_logger
from src.utils.metrics import evaluate_predictions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on a labeled split without rerunning validation."
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the configuration file used for training/inference."
    )
    parser.add_argument(
        "--split",
        choices=["train", "val"],
        default="val",
        help="Which labeled split to score (defaults to validation window)."
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fraction (0,1] applied before computing to reduce memory usage."
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=None,
        help="Limit number of Dask partitions loaded from disk."
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Hard cap on number of rows evaluated after loading/sampling."
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50000,
        help="Number of rows to process per chunk (memory-efficient)."
    )
    parser.add_argument(
        "--use-chunked",
        action="store_true",
        default=True,
        help="Use chunked processing to avoid RAM issues (default: True)."
    )
    parser.add_argument(
        "--start-dt",
        default=None,
        help="Override the split start datetime (format: YYYY-MM-DD-HH-MM)."
    )
    parser.add_argument(
        "--end-dt",
        default=None,
        help="Override the split end datetime (format: YYYY-MM-DD-HH-MM)."
    )
    return parser.parse_args()


def _resolve_window(config: dict[str, Any], split: str, start: str | None, end: str | None) -> tuple[str, str]:
    data_cfg = config.get("data", {})

    default_start_key = f"{split}_start"
    default_end_key = f"{split}_end"
    if default_start_key not in data_cfg or default_end_key not in data_cfg:
        raise KeyError(
            f"Config missing '{default_start_key}'/'{default_end_key}' entries under data.*; "
            "cannot determine evaluation window."
        )

    resolved_start = start or data_cfg[default_start_key]
    resolved_end = end or data_cfg[default_end_key]

    assert resolved_start <= resolved_end, "Start datetime must be <= end datetime"
    return resolved_start, resolved_end


def _load_labeled_split_chunked(
    config: dict[str, Any],
    split: str,
    max_partitions: int | None,
    chunk_size: int,
    start_dt: str | None,
    end_dt: str | None
):
    """
    Generator that yields chunks of labeled data for evaluation.
    
    Yields:
        (chunk_df, y_true_chunk) tuples
    """
    loader = DataLoader(config)
    resolved_start, resolved_end = _resolve_window(config, split, start_dt, end_dt)
    
    for chunk_df, _ in loader.iter_train_chunks(
        chunk_size=chunk_size,
        validation_split=False,
        start_dt=resolved_start,
        end_dt=resolved_end,
        max_partitions=max_partitions
    ):
        if "iap_revenue_d7" not in chunk_df.columns:
            raise ValueError("Chunk lacks 'iap_revenue_d7'; ensure you selected a labeled split.")
        
        y_true_chunk = chunk_df["iap_revenue_d7"].astype(float).values
        yield chunk_df, y_true_chunk


def _load_labeled_split(
    config: dict[str, Any],
    split: str,
    sample_frac: float | None,
    max_partitions: int | None,
    limit_rows: int | None,
    start_dt: str | None,
    end_dt: str | None
) -> pd.DataFrame:
    loader = DataLoader(config)
    resolved_start, resolved_end = _resolve_window(config, split, start_dt, end_dt)

    ddf, _ = loader.load_train(
        validation_split=False,
        sample_frac=sample_frac or 1.0,
        max_partitions=max_partitions,
        start_dt=resolved_start,
        end_dt=resolved_end
    )
    df = ddf.compute()

    if limit_rows is not None and limit_rows > 0 and len(df) > limit_rows:
        df = df.head(limit_rows).reset_index(drop=True)

    if "iap_revenue_d7" not in df.columns:
        raise ValueError("Loaded dataframe lacks 'iap_revenue_d7'; ensure you selected a labeled split.")

    return df


def main() -> None:
    args = _parse_args()

    logger = setup_logger("holdout_evaluation", "logs/holdout_evaluation.log")
    logger.info("Smadex LTV Prediction - Holdout Evaluation")
    logger.info("=" * 80)
    logger.info(
        "Config=%s | split=%s | sample_frac=%s | max_partitions=%s | limit_rows=%s | chunk_size=%s | use_chunked=%s",
        args.config,
        args.split,
        args.sample_frac,
        args.max_partitions,
        args.limit_rows,
        args.chunk_size,
        args.use_chunked
    )

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    predictor = FastPredictor(args.config)
    
    if args.use_chunked:
        logger.info("Using chunked evaluation (memory-efficient)")
        
        # Accumulate predictions and true values
        all_y_true = []
        all_predictions = []
        total_rows = 0
        chunk_count = 0
        
        for chunk_df, y_true_chunk in _load_labeled_split_chunked(
            config=config,
            split=args.split,
            max_partitions=args.max_partitions,
            chunk_size=args.chunk_size,
            start_dt=args.start_dt,
            end_dt=args.end_dt
        ):
            chunk_count += 1
            chunk_rows = len(chunk_df)
            total_rows += chunk_rows
            
            # Apply limit_rows if specified
            if args.limit_rows is not None and total_rows > args.limit_rows:
                rows_to_take = args.limit_rows - (total_rows - chunk_rows)
                if rows_to_take <= 0:
                    break
                chunk_df = chunk_df.head(rows_to_take)
                y_true_chunk = y_true_chunk[:rows_to_take]
            
            logger.info(f"Processing chunk {chunk_count}: {len(chunk_df)} rows (total: {total_rows:,})")
            
            # Predict on chunk
            predictions_chunk = predictor.predict(chunk_df.copy())
            
            # Accumulate
            all_y_true.extend(y_true_chunk.tolist())
            all_predictions.extend(predictions_chunk.tolist())
            
            # Log progress every 10 chunks
            if chunk_count % 10 == 0:
                logger.info(f"  Processed {chunk_count} chunks, {total_rows:,} total rows")
            
            # Check if we've reached limit
            if args.limit_rows is not None and total_rows >= args.limit_rows:
                logger.info(f"Reached limit_rows={args.limit_rows}, stopping")
                break
        
        logger.info(f"Loaded {len(all_y_true):,} rows for evaluation from {chunk_count} chunks")
        
        y_true = np.array(all_y_true)
        predictions = np.array(all_predictions)
        
    else:
        logger.info("Using standard evaluation (loads all data)")
        df = _load_labeled_split(
            config=config,
            split=args.split,
            sample_frac=args.sample_frac,
            max_partitions=args.max_partitions,
            limit_rows=args.limit_rows,
            start_dt=args.start_dt,
            end_dt=args.end_dt
        )
        logger.info("Loaded %d rows for evaluation", len(df))

        y_true = df["iap_revenue_d7"].astype(float).values
        predictions = predictor.predict(df.copy())

    # Log prediction diagnostics
    logger.info("\n📊 Prediction diagnostics:")
    logger.info("  True values:")
    logger.info("    Mean: $%.4f", y_true.mean())
    logger.info("    Zeros: %d (%.2f%%)", (y_true == 0).sum(), (y_true == 0).mean() * 100)
    logger.info("  Predictions:")
    logger.info("    Mean: $%.4f", predictions.mean())
    logger.info("    Std: $%.4f", predictions.std())
    logger.info("    Min: $%.4f", predictions.min())
    logger.info("    Max: $%.4f", predictions.max())
    logger.info("    Zeros: %d (%.2f%%)", (predictions == 0).sum(), (predictions == 0).mean() * 100)
    logger.info("    Unique (rounded to 2 decimals): %d", len(np.unique(predictions.round(2))))

    metrics = evaluate_predictions(y_true, predictions)

    logger.info("\n📏 Holdout metrics:")
    for metric, value in metrics.to_dict().items():
        logger.info("  %s: %.6f", metric.upper(), value)

    logger.info("\n🏆 Kaggle metric (RMSLE): %.6f", metrics.rmsle)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
