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
        "Config=%s | split=%s | sample_frac=%s | max_partitions=%s | limit_rows=%s",
        args.config,
        args.split,
        args.sample_frac,
        args.max_partitions,
        args.limit_rows
    )

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

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

    predictor = FastPredictor(args.config)
    predictions = predictor.predict(df.copy())

    metrics = evaluate_predictions(y_true, predictions)

    logger.info("\n📏 Holdout metrics:")
    for metric, value in metrics.to_dict().items():
        logger.info("  %s: %.6f", metric.upper(), value)

    logger.info("\n🏆 Kaggle metric (RMSLE): %.6f", metrics.rmsle)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
