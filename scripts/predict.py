"""CLI helper to generate submission files with optional runtime controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predictor import FastPredictor
from src.utils.logger import setup_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LTV submission CSV")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to configuration YAML used for preprocessing and model loading"
    )
    parser.add_argument(
        "--output",
        default="data/submissions/submission.csv",
        help="Where to store the generated submission CSV"
    )
    parser.add_argument(
        "--max-partitions",
        type=int,
        default=None,
        help="Optional cap on number of Dask partitions loaded from disk"
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional fraction (0,1] to randomly sample from the computed test dataframe"
    )
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Optional hard cap on the number of rows scored after sampling"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logger = setup_logger('prediction', 'logs/prediction.log')
    logger.info("Smadex LTV Prediction - Inference Script")
    logger.info("=" * 80)
    logger.info(
        "Options ➜ config=%s | output=%s | max_partitions=%s | sample_frac=%s | limit_rows=%s",
        args.config,
        args.output,
        args.max_partitions,
        args.sample_frac,
        args.limit_rows
    )
    
    predictor = FastPredictor(args.config)
    submission = predictor.predict_test_set(
        max_partitions=args.max_partitions,
        sample_frac=args.sample_frac,
        limit_rows=args.limit_rows
    )
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Sample predictions:\n{submission.head()}")
    
    logger.info("Prediction statistics:")
    logger.info(f"  Mean: {submission['iap_revenue_d7'].mean():.4f}")
    logger.info(f"  Median: {submission['iap_revenue_d7'].median():.4f}")
    logger.info(f"  Std: {submission['iap_revenue_d7'].std():.4f}")
    logger.info(f"  Min: {submission['iap_revenue_d7'].min():.4f}")
    logger.info(f"  Max: {submission['iap_revenue_d7'].max():.4f}")
    logger.info(f"  % Zeros: {(submission['iap_revenue_d7'] == 0).mean() * 100:.2f}%")


if __name__ == '__main__':
    main()
