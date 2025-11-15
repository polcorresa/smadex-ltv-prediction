"""
Dask-based data loader for large parquet datasets
"""
import dask
import dask.dataframe as dd
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
import logging

# Disable string conversion
dask.config.set({"dataframe.convert-string": False})

logger = logging.getLogger(__name__)


class DataLoader:
    """Efficient data loader using Dask for distributed computation"""
    
    def __init__(self, config: dict):
        self.config = config
        self.train_path = Path(config['data']['train_path'])
        self.test_path = Path(config['data']['test_path'])
    
    def load_train(
        self, 
        validation_split: bool = True
    ) -> Tuple[dd.DataFrame, Optional[dd.DataFrame]]:
        """
        Load training data with optional temporal validation split
        
        Returns:
            train_ddf, val_ddf (or train_ddf, None)
        """
        logger.info("Loading training data...")
        
        if validation_split:
            # Train split (Oct 1-6)
            train_filters = [
                ("datetime", ">=", self.config['data']['train_start']),
                ("datetime", "<=", self.config['data']['train_end'])
            ]
            
            # Validation split (Oct 7)
            val_filters = [
                ("datetime", ">=", self.config['data']['val_start']),
                ("datetime", "<=", self.config['data']['val_end'])
            ]
            
            train_ddf = dd.read_parquet(
                self.train_path,
                filters=train_filters,
                engine='pyarrow'
            )
            
            val_ddf = dd.read_parquet(
                self.train_path,
                filters=val_filters,
                engine='pyarrow'
            )
            
            logger.info(f"Train partitions: {train_ddf.npartitions}")
            logger.info(f"Validation partitions: {val_ddf.npartitions}")
            
            return train_ddf, val_ddf
        
        else:
            # Full training set
            ddf = dd.read_parquet(
                self.train_path,
                engine='pyarrow'
            )
            return ddf, None
    
    def load_test(self) -> dd.DataFrame:
        """Load test data (Oct 8-12)"""
        logger.info("Loading test data...")
        
        test_filters = [
            ("datetime", ">=", self.config['data']['test_start']),
            ("datetime", "<=", self.config['data']['test_end'])
        ]
        
        ddf = dd.read_parquet(
            self.test_path,
            filters=test_filters,
            engine='pyarrow'
        )
        
        logger.info(f"Test partitions: {ddf.npartitions}")
        return ddf
    
    def compute_batch(
        self, 
        ddf: dd.DataFrame, 
        batch_size: int = 10000
    ) -> pd.DataFrame:
        """
        Compute Dask DataFrame to Pandas in batches
        Useful for memory-constrained environments
        """
        n_partitions = ddf.npartitions
        results = []
        
        for i in range(n_partitions):
            partition = ddf.get_partition(i).compute()
            results.append(partition)
            
            if len(results) * len(partition) >= batch_size:
                yield pd.concat(results, ignore_index=True)
                results = []
        
        if results:
            yield pd.concat(results, ignore_index=True)