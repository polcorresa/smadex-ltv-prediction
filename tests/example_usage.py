"""
Quick examples of using the data loader
"""
import yaml
from pathlib import Path
from src.data.loader import DataLoader

# Load configuration
config_path = Path("config/config.yaml")
with open(config_path) as f:
    config = yaml.safe_load(f)

# Initialize loader
loader = DataLoader(config)

# =============================================================================
# Example 1: Load small samples for exploration (safe for memory)
# =============================================================================
print("Example 1: Loading small samples...")
train_df, val_df = loader.load_train_sample(n_rows=100, validation_split=True)
test_df = loader.load_test_sample(n_rows=100)

print(f"Train shape: {train_df.shape}")
print(f"Val shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nTrain columns: {train_df.columns.tolist()}")

# =============================================================================
# Example 2: Load with Dask for training (with memory limits)
# =============================================================================
print("\n" + "="*80)
print("Example 2: Loading with Dask (limited partitions)...")

# Use config parameters for sampling
train_frac = config['training']['sampling']['train_frac']  # 0.2 = 20%
max_train_parts = config['training']['sampling']['max_train_partitions']  # 5

train_ddf, val_ddf = loader.load_train(
    validation_split=True,
    sample_frac=train_frac,
    max_partitions=max_train_parts
)

print(f"Train Dask DataFrame: {train_ddf.npartitions} partitions")
print(f"Val Dask DataFrame: {val_ddf.npartitions} partitions")

# Compute a small batch to see the data
print("\nComputing first 1000 rows...")
train_sample = train_ddf.head(1000)
print(f"Sample shape: {train_sample.shape}")

# =============================================================================
# Example 3: Process in batches to avoid memory issues
# =============================================================================
print("\n" + "="*80)
print("Example 3: Processing in batches...")

batch_size = config['training']['batch_size']
batch_count = 0

for batch_df in loader.compute_batch(train_ddf, batch_size=batch_size):
    print(f"Processing batch {batch_count + 1}: {batch_df.shape[0]} rows")
    
    # Your processing code here
    # Example: Extract features and targets
    target_cols = ['buyer_d1', 'buyer_d7', 'buyer_d14']
    feature_cols = [col for col in batch_df.columns if col not in target_cols]
    
    X_batch = batch_df[feature_cols]
    y_batch = batch_df[target_cols]
    
    batch_count += 1
    if batch_count >= 2:  # Only process first 2 batches for demo
        break

print(f"\nProcessed {batch_count} batches")

# =============================================================================
# Example 4: Load test data for inference
# =============================================================================
print("\n" + "="*80)
print("Example 4: Loading test data...")

max_test_parts = config['training']['sampling']['max_val_partitions']  # 2

test_ddf = loader.load_test(max_partitions=max_test_parts)
print(f"Test Dask DataFrame: {test_ddf.npartitions} partitions")

# Compute first batch
test_sample = test_ddf.head(1000)
print(f"Test sample shape: {test_sample.shape}")
print(f"Test columns: {len(test_sample.columns)} (no targets)")

# =============================================================================
# Example 5: Check target distributions
# =============================================================================
print("\n" + "="*80)
print("Example 5: Analyzing targets...")

train_sample_df, _ = loader.load_train_sample(n_rows=10000, validation_split=False)

target_cols = ['buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28']
for col in target_cols:
    if col in train_sample_df.columns:
        pos_rate = train_sample_df[col].mean()
        print(f"{col}: {pos_rate:.2%} positive rate")

# Revenue targets
revenue_cols = ['iap_revenue_d7', 'iap_revenue_d14', 'iap_revenue_d28']
for col in revenue_cols:
    if col in train_sample_df.columns:
        non_zero = (train_sample_df[col].notna() & (train_sample_df[col] > 0)).sum()
        pct = non_zero / len(train_sample_df) * 100
        print(f"{col}: {non_zero} non-zero values ({pct:.2f}%)")

print("\n" + "="*80)
print("All examples complete!")
