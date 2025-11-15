"""
Main training script
Usage: python scripts/train.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.training.trainer import TrainingPipeline
from src.utils.logger import setup_logger


def main():
    # Setup logging
    logger = setup_logger('training', 'logs/training.log')
    
    logger.info("Smadex LTV Prediction - Training Script")
    logger.info("=" * 80)
    
    # Initialize pipeline with TEST configuration
    logger.info("Using TEST configuration: config/config_test.yaml")
    pipeline = TrainingPipeline('config/config_test.yaml')
    
    # Run training
    pipeline.run()
    
    logger.info("Training completed successfully!")


if __name__ == '__main__':
    main()