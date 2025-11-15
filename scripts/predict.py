"""
Generate submission file
Usage: python scripts/predict.py
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.inference.predictor import FastPredictor
from src.utils.logger import setup_logger


def main():
    # Setup logging
    logger = setup_logger('prediction', 'logs/prediction.log')
    
    logger.info("Smadex LTV Prediction - Inference Script")
    logger.info("=" * 80)
    
    # Initialize predictor
    predictor = FastPredictor('config/config.yaml')
    
    # Generate predictions
    submission = predictor.predict_test_set()
    
    # Save submission
    output_path = 'data/submissions/submission.csv'
    submission.to_csv(output_path, index=False)
    
    logger.info(f"Submission saved to {output_path}")
    logger.info(f"Submission shape: {submission.shape}")
    logger.info(f"Sample predictions:\n{submission.head()}")
    
    # Statistics
    logger.info(f"Prediction statistics:")
    logger.info(f"  Mean: {submission['iap_revenue_d7'].mean():.4f}")
    logger.info(f"  Median: {submission['iap_revenue_d7'].median():.4f}")
    logger.info(f"  Std: {submission['iap_revenue_d7'].std():.4f}")
    logger.info(f"  Min: {submission['iap_revenue_d7'].min():.4f}")
    logger.info(f"  Max: {submission['iap_revenue_d7'].max():.4f}")
    logger.info(f"  % Zeros: {(submission['iap_revenue_d7'] == 0).mean() * 100:.2f}%")


if __name__ == '__main__':
    main()