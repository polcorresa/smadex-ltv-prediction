"""Memory-efficient chunked prediction for large test sets."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
import gc

sys.path.append(str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.inference.predictor import FastPredictor
from src.utils.logger import setup_logger

logger = setup_logger('predict_chunked', 'logs/predict_chunked.log')


class ChunkedPredictor:
    """Memory-efficient predictor that processes data in chunks."""
    
    def __init__(self, config_path: str):
        """Initialize predictor with config."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_loader = DataLoader(self.config)
        self.fast_predictor = FastPredictor(config_path)
    
    def predict_chunk(self, chunk_df: pd.DataFrame) -> np.ndarray:
        """Predict on a single chunk."""
        return self.fast_predictor.predict(chunk_df.copy())
    
    def predict_test_chunked(
        self,
        output_path: str = 'data/submissions/submission_chunked.csv',
        chunk_size: int | None = None,
        max_partitions: int | None = None
    ):
        """
        Predict on test set using chunked processing.
        
        Args:
            output_path: Where to save submission file
            chunk_size: Override default chunk size
            max_partitions: Limit number of partitions
        """
        logger.info("=" * 80)
        logger.info("CHUNKED PREDICTION ON TEST SET")
        logger.info("=" * 80)
        
        chunk_size = chunk_size or self.config.get('training', {}).get('chunk_size', 50000)
        logger.info(f"Processing in chunks of {chunk_size} rows")
        
        # Initialize results storage
        all_row_ids = []
        all_predictions = []
        total_rows = 0
        chunk_count = 0
        
        # Iterate over test chunks
        for chunk_df in self.data_loader.iter_test_chunks(
            chunk_size=chunk_size,
            max_partitions=max_partitions
        ):
            chunk_count += 1
            chunk_rows = len(chunk_df)
            total_rows += chunk_rows
            
            logger.info(f"Processing chunk {chunk_count}: {chunk_rows} rows (total: {total_rows:,})")
            
            # Store row_ids
            all_row_ids.extend(chunk_df['row_id'].tolist())
            
            # Predict
            predictions = self.predict_chunk(chunk_df)
            all_predictions.extend(predictions.tolist())
            
            # Force garbage collection
            del chunk_df
            gc.collect()
            
            # Log progress
            if chunk_count % 10 == 0:
                logger.info(f"  Processed {chunk_count} chunks, {total_rows:,} total rows")
        
        # Create submission DataFrame
        logger.info(f"\nCreating submission with {len(all_row_ids):,} predictions...")
        submission = pd.DataFrame({
            'row_id': all_row_ids,
            'iap_revenue_d7': all_predictions
        })
        
        # Log statistics
        logger.info(f"\nPrediction statistics:")
        logger.info(f"  Mean: ${submission['iap_revenue_d7'].mean():.4f}")
        logger.info(f"  Median: ${submission['iap_revenue_d7'].median():.4f}")
        logger.info(f"  Std: ${submission['iap_revenue_d7'].std():.4f}")
        logger.info(f"  Min: ${submission['iap_revenue_d7'].min():.4f}")
        logger.info(f"  Max: ${submission['iap_revenue_d7'].max():.4f}")
        
        # Save submission
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(output_path, index=False)
        
        logger.info(f"\nSubmission saved to {output_path}")
        logger.info(f"Total rows: {len(submission):,}")
        logger.info("=" * 80)
        
        return submission


def main():
    """Run chunked prediction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chunked prediction for test set")
    parser.add_argument(
        '--config',
        default='config/config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--output',
        default='data/submissions/submission_chunked.csv',
        help='Output submission file path'
    )
    parser.add_argument(
        '--chunk-size',
        type=int,
        default=None,
        help='Override chunk size'
    )
    parser.add_argument(
        '--max-partitions',
        type=int,
        default=None,
        help='Maximum number of partitions to process'
    )
    
    args = parser.parse_args()
    
    predictor = ChunkedPredictor(args.config)
    predictor.predict_test_chunked(
        output_path=args.output,
        chunk_size=args.chunk_size,
        max_partitions=args.max_partitions
    )


if __name__ == '__main__':
    main()


