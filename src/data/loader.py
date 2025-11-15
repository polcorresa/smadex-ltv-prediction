"""
Dask-based data loader for large parquet datasets with optimizations.

Following ArjanCodes best practices:
- Complete type hints
- Precondition/postcondition assertions
- Clear parameter validation
"""
from __future__ import annotations

import dask
import dask.dataframe as dd
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import logging
import psutil

# Disable string conversion
dask.config.set({"dataframe.convert-string": False})

# Dask performance optimizations
dask.config.set({
    # Optimize task fusion for better graph efficiency
    "optimization.fuse.active": True,
    "optimization.fuse.ave-width": 10,
    
    # Better memory management
    "dataframe.shuffle.method": "tasks",  # More memory efficient than "disk"
    
    # Optimize string handling
    "dataframe.convert-string": False,  # Keep as object dtype for performance
})

logger = logging.getLogger(__name__)


class DataLoader:
    """Efficient data loader using Dask for distributed computation with optimizations."""
    
    def __init__(self, config: dict) -> None:
        """
        Initialize data loader.
        
        Args:
            config: Configuration dictionary with data paths
        """
        assert config is not None, "Config must not be None"
        assert 'data' in config, "Config must contain 'data' key"
        assert 'train_path' in config['data'], "Config must contain train_path"
        assert 'test_path' in config['data'], "Config must contain test_path"
        
        self.config = config
        self.train_path = Path(config['data']['train_path'])
        self.test_path = Path(config['data']['test_path'])
        
        assert self.train_path.exists(), f"Train path does not exist: {self.train_path}"
        assert self.test_path.exists(), f"Test path does not exist: {self.test_path}"
        
        # Calculate optimal blocksize based on available memory
        # Target: 100-200 MB per partition for good parallelism
        available_memory = psutil.virtual_memory().available
        n_cores = psutil.cpu_count(logical=False) or 4
        
        # Reserve 20% for overhead, distribute across cores with 2-3x buffer
        self.optimal_blocksize = min(
            200 * 1024 * 1024,  # Max 200MB per partition
            int(available_memory * 0.8 / (n_cores * 2.5))
        )
        
        assert self.optimal_blocksize > 0, "Block size must be positive"
        
        logger.info(f"Initialized DataLoader with blocksize={self.optimal_blocksize / 1024 / 1024:.1f}MB, cores={n_cores}")
        
    def _get_partition_paths(self, base_path: Path, start_dt: str = None, end_dt: str = None) -> List[Path]:
        """
        Get list of partition directories between start and end datetimes
        
        Args:
            base_path: Base directory containing datetime partitions
            start_dt: Start datetime string (e.g., "2025-10-01-00-00")
            end_dt: End datetime string (e.g., "2025-10-05-23-00")
            
        Returns:
            List of partition paths
        """
        all_partitions = sorted(base_path.glob("datetime=*"))
        
        if start_dt is None and end_dt is None:
            return all_partitions
        
        filtered = []
        for partition in all_partitions:
            # Extract datetime from directory name: "datetime=2025-10-01-00-00"
            dt_str = partition.name.split("=")[1]
            
            if start_dt and dt_str < start_dt:
                continue
            if end_dt and dt_str > end_dt:
                continue
                
            filtered.append(partition)
        
        return filtered
    
    def _load_sample_from_partitions(
        self, 
        partitions: List[Path], 
        n_rows: int = 1000
    ) -> pd.DataFrame:
        """
        Load a sample of rows from partition directories
        
        Args:
            partitions: List of partition directories
            n_rows: Number of rows to sample
            
        Returns:
            DataFrame with sampled rows
        """
        dfs = []
        rows_collected = 0
        
        for partition_path in partitions:
            if rows_collected >= n_rows:
                break
                
            # Find parquet files in this partition
            parquet_files = list(partition_path.glob("*.parquet"))
            
            if not parquet_files:
                continue
            
            # Read first file in partition
            df = pd.read_parquet(parquet_files[0], engine='pyarrow')
            
            # Add datetime column from partition name
            dt_str = partition_path.name.split("=")[1]
            df['datetime'] = dt_str
            
            rows_needed = n_rows - rows_collected
            df_sample = df.head(rows_needed)
            dfs.append(df_sample)
            rows_collected += len(df_sample)
        
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def load_train_sample(
        self, 
        n_rows: int = 1000,
        validation_split: bool = True
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Load a sample of training data without loading the entire dataset
        
        Args:
            n_rows: Number of rows to sample per split
            validation_split: Whether to split into train/val
            
        Returns:
            train_df, val_df (or train_df, None)
        """
        logger.info(f"Loading training sample ({n_rows} rows)...")
        
        if validation_split:
            train_start = self.config['data']['train_start']
            train_end = self.config['data']['train_end']
            val_start = self.config['data']['val_start']
            val_end = self.config['data']['val_end']
            
            # Get partition paths for train and val
            train_partitions = self._get_partition_paths(self.train_path, train_start, train_end)
            val_partitions = self._get_partition_paths(self.train_path, val_start, val_end)
            
            logger.info(f"Found {len(train_partitions)} train partitions, {len(val_partitions)} val partitions")
            
            # Load samples
            train_df = self._load_sample_from_partitions(train_partitions, n_rows)
            val_df = self._load_sample_from_partitions(val_partitions, n_rows)
            
            logger.info(f"Loaded train: {len(train_df)} rows, val: {len(val_df)} rows")
            
            return train_df, val_df
        else:
            # Load sample from all training data
            all_partitions = self._get_partition_paths(self.train_path)
            train_df = self._load_sample_from_partitions(all_partitions, n_rows)
            
            logger.info(f"Loaded {len(train_df)} rows")
            return train_df, None
    
    def load_test_sample(self, n_rows: int = 1000) -> pd.DataFrame:
        """
        Load a sample of test data without loading the entire dataset
        
        Args:
            n_rows: Number of rows to sample
            
        Returns:
            DataFrame with sampled test data
        """
        logger.info(f"Loading test sample ({n_rows} rows)...")
        
        test_start = self.config['data']['test_start']
        test_end = self.config['data']['test_end']
        
        # Get partition paths for test
        test_partitions = self._get_partition_paths(self.test_path, test_start, test_end)
        
        logger.info(f"Found {len(test_partitions)} test partitions")
        
        # Load sample
        test_df = self._load_sample_from_partitions(test_partitions, n_rows)
        
        logger.info(f"Loaded {len(test_df)} rows")
        
        return test_df
    
    def _datetime_to_filter(self, dt_str: str, operator: str = "==") -> List:
        """
        Convert datetime string to filter format for parquet partitions
        
        Args:
            dt_str: Datetime string like "2025-10-01-00-00"
            operator: Comparison operator
        
        Returns:
            Filter list for read_parquet
        """
        return [("datetime", operator, dt_str)]
    
    def load_train(
        self, 
        validation_split: bool = True,
        sample_frac: float = 1.0,
        max_partitions: Optional[int] = None
    ) -> Tuple[dd.DataFrame, Optional[dd.DataFrame]]:
        """
        Load training data with optimal Dask configuration
        
        Args:
            validation_split: Whether to split into train/val based on config dates
            sample_frac: Fraction of data to sample (for memory management)
            max_partitions: Maximum number of partitions to load (for memory management)
        
        Returns:
            train_ddf, val_ddf (or train_ddf, None)
        """
        logger.info("Loading training data with Dask optimizations...")
        
        if validation_split:
            train_start = self.config['data']['train_start']
            train_end = self.config['data']['train_end']
            val_start = self.config['data']['val_start']
            val_end = self.config['data']['val_end']
            
            logger.info(f"Train period: {train_start} to {train_end}")
            logger.info(f"Val period: {val_start} to {val_end}")
            
            # Get partition paths
            train_partitions = self._get_partition_paths(self.train_path, train_start, train_end)
            val_partitions = self._get_partition_paths(self.train_path, val_start, val_end)
            
            # Apply max_partitions limit
            if max_partitions:
                train_partitions = train_partitions[:max_partitions]
                val_partitions = val_partitions[:max_partitions]
            
            logger.info(f"Loading {len(train_partitions)} train partitions, {len(val_partitions)} val partitions")
            
            # Build file paths for all parquet files in partitions
            train_files = []
            for partition in train_partitions:
                train_files.extend(partition.glob("*.parquet"))
            
            val_files = []
            for partition in val_partitions:
                val_files.extend(partition.glob("*.parquet"))
            
            # Load with Dask - OPTIMIZED
            train_ddf = dd.read_parquet(
                [str(f) for f in train_files],
                engine='pyarrow',
                blocksize=self.optimal_blocksize,  # Optimal partition size
                aggregate_files=True,  # Combine small files
                calculate_divisions=False,  # Skip expensive division calculation
                index=False,  # Don't set index (faster)
                gather_statistics=False  # Skip metadata gathering
            )
            
            val_ddf = dd.read_parquet(
                [str(f) for f in val_files],
                engine='pyarrow',
                blocksize=self.optimal_blocksize,
                aggregate_files=True,
                calculate_divisions=False,
                index=False,
                gather_statistics=False
            )
            
            # Apply dtype optimizations for categorical columns
            train_ddf = self._optimize_dtypes(train_ddf)
            val_ddf = self._optimize_dtypes(val_ddf)
            
            # Apply sampling if requested
            if sample_frac < 1.0:
                train_ddf = train_ddf.sample(frac=sample_frac, random_state=self.config['training']['random_state'])
                val_ddf = val_ddf.sample(frac=sample_frac, random_state=self.config['training']['random_state'])
            
            logger.info(f"Train partitions: {train_ddf.npartitions}, Val partitions: {val_ddf.npartitions}")
            logger.info(f"Estimated memory per partition: {self.optimal_blocksize / 1024 / 1024:.1f}MB")
            
            return train_ddf, val_ddf
        
        else:
            # Full training set
            all_partitions = self._get_partition_paths(self.train_path)
            
            if max_partitions:
                all_partitions = all_partitions[:max_partitions]
            
            logger.info(f"Loading {len(all_partitions)} partitions")
            
            # Build file paths
            all_files = []
            for partition in all_partitions:
                all_files.extend(partition.glob("*.parquet"))
            
            ddf = dd.read_parquet(
                [str(f) for f in all_files],
                engine='pyarrow',
                blocksize=self.optimal_blocksize,
                aggregate_files=True,
                calculate_divisions=False,
                index=False,
                gather_statistics=False
            )
            
            ddf = self._optimize_dtypes(ddf)
            
            if sample_frac < 1.0:
                ddf = ddf.sample(frac=sample_frac, random_state=self.config['training']['random_state'])
            
            logger.info(f"Loaded partitions: {ddf.npartitions}")
            return ddf, None
    
    def load_test(self, max_partitions: Optional[int] = None) -> dd.DataFrame:
        """
        Load test data using Dask with optimizations
        
        Args:
            max_partitions: Maximum number of partitions to load (for memory management)
            
        Returns:
            Dask DataFrame with test data
        """
        logger.info("Loading test data with Dask optimizations...")
        
        test_start = self.config['data']['test_start']
        test_end = self.config['data']['test_end']
        
        logger.info(f"Test period: {test_start} to {test_end}")
        
        # Get partition paths
        test_partitions = self._get_partition_paths(self.test_path, test_start, test_end)
        
        if max_partitions:
            test_partitions = test_partitions[:max_partitions]
        
        logger.info(f"Loading {len(test_partitions)} test partitions")
        
        # Build file paths
        test_files = []
        for partition in test_partitions:
            test_files.extend(partition.glob("*.parquet"))
        
        ddf = dd.read_parquet(
            [str(f) for f in test_files],
            engine='pyarrow',
            blocksize=self.optimal_blocksize,
            aggregate_files=True,
            calculate_divisions=False,
            index=False,
            gather_statistics=False
        )
        
        ddf = self._optimize_dtypes(ddf)
        
        logger.info(f"Test partitions: {ddf.npartitions}")
        logger.info(f"Estimated memory per partition: {self.optimal_blocksize / 1024 / 1024:.1f}MB")
        return ddf
    
    def _optimize_dtypes(self, ddf: dd.DataFrame) -> dd.DataFrame:
        """
        Optimize data types for memory efficiency
        
        Convert low-cardinality string columns to categorical
        Keep revenue/numeric columns as-is (don't downcast)
        """
        # Categorical candidates (low cardinality expected)
        categorical_candidates = [
            'dev_ctry', 'dev_region', 'dev_city',
            'dev_os', 'dev_osv', 'dev_language',
            'dev_brand', 'dev_model', 'dev_name',
            'tg_adv_id', 'tg_ssp_id', 'tg_pub_id'
        ]
        
        # Check which columns exist and convert to categorical
        for col in categorical_candidates:
            if col in ddf.columns:
                ddf[col] = ddf[col].astype('category')
        
        return ddf
    
    def persist_optimized(
        self,
        ddf: dd.DataFrame,
        scheduler: str = 'threads'
    ) -> dd.DataFrame:
        """
        Persist DataFrame in memory with optimal scheduler
        
        Args:
            ddf: Dask DataFrame to persist
            scheduler: 'threads' (default, good for numeric) or 'processes' (for text-heavy)
            
        Returns:
            Persisted DataFrame
        """
        logger.info(f"Persisting {ddf.npartitions} partitions with {scheduler} scheduler...")
        
        with dask.config.set(scheduler=scheduler):
            ddf = ddf.persist()
        
        return ddf
    
    def compute_batch(
        self, 
        ddf: dd.DataFrame, 
        batch_size: int = 10000
    ) -> pd.DataFrame:
        """
        Compute Dask DataFrame to Pandas in batches (OPTIMIZED)
        Useful for memory-constrained environments
        
        Uses map_partitions for better efficiency
        """
        logger.info(f"Computing {ddf.npartitions} partitions in batches...")
        
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
    
    def repartition_optimized(
        self,
        ddf: dd.DataFrame,
        target_partition_size_mb: int = 100
    ) -> dd.DataFrame:
        """
        Repartition DataFrame for optimal partition size
        
        Args:
            ddf: Input DataFrame
            target_partition_size_mb: Target size per partition in MB
            
        Returns:
            Repartitioned DataFrame
        """
        # Estimate current partition size
        sample = ddf.head(1000)
        row_size_bytes = sample.memory_usage(deep=True).sum() / len(sample)
        
        total_rows = len(ddf)
        target_rows_per_partition = int(
            (target_partition_size_mb * 1024 * 1024) / row_size_bytes
        )
        
        optimal_npartitions = max(1, total_rows // target_rows_per_partition)
        
        if abs(ddf.npartitions - optimal_npartitions) > ddf.npartitions * 0.2:
            logger.info(f"Repartitioning from {ddf.npartitions} to {optimal_npartitions} partitions")
            ddf = ddf.repartition(npartitions=optimal_npartitions)
        else:
            logger.info(f"Partition size already optimal ({ddf.npartitions} partitions)")
        
        return ddf