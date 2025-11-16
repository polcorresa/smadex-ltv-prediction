"""Entrypoint for training the Smadex LTV pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

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

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    data_cfg = config.get('data', {})
    training_cfg = config.get('training', {})
    sampling_cfg = training_cfg.get('sampling', {})

    logger.info(
        "Data window -> train: %s→%s | val: %s→%s | model: %s→%s",
        data_cfg.get('train_start'),
        data_cfg.get('train_end'),
        data_cfg.get('val_start'),
        data_cfg.get('val_end'),
        data_cfg.get('model_start'),
        data_cfg.get('model_end')
    )
    logger.info(
        "Chunked loading: %s | chunk_size: %s | batch_size: %s",
        training_cfg.get('use_chunked_loading', False),
        training_cfg.get('chunk_size', 'auto'),
        training_cfg.get('batch_size')
    )
    logger.info(
        "Sampling -> train_frac=%s | val_frac=%s | max_train_parts=%s | max_val_parts=%s",
        sampling_cfg.get('train_frac', sampling_cfg.get('frac')),
        sampling_cfg.get('val_frac'),
        sampling_cfg.get('max_train_partitions'),
        sampling_cfg.get('max_val_partitions')
    )
    logger.info(
        "Split strategy: %s | n_folds: %s | use_gpu: %s",
        training_cfg.get('split', {}).get('strategy'),
        training_cfg.get('n_folds'),
        training_cfg.get('use_gpu')
    )

    pipeline = TrainingPipeline(args.config)
    logger.info("Training pipeline initialized; starting stage execution...")
    pipeline.run()
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()
