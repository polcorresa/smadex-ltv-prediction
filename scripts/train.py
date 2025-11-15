"""Entrypoint for training the Smadex LTV pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import TrainingPipeline
from src.utils.logger import setup_logger


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LTV pipeline")
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Path to the YAML config controlling data footprint and hyperparameters"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    logger = setup_logger('training', 'logs/training.log')
    logger.info("Smadex LTV Prediction - Training Script")
    logger.info("=" * 80)
    logger.info("Using configuration: %s", args.config)

    pipeline = TrainingPipeline(args.config)
    pipeline.run()
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()