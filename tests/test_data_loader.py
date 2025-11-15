"""
Test script for data loader - displays data structure with examples
"""
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def display_dataframe_info(df: pd.DataFrame, name: str, n_rows: int = 15):
    """Display comprehensive information about a dataframe"""
    print_section(f"{name} - Overview")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumn names ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print_section(f"{name} - Data Types")
    print(df.dtypes.to_string())
    
    print_section(f"{name} - Missing Values")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({
        'Column': missing.index,
        'Missing': missing.values,
        'Percentage': missing_pct.values
    })
    missing_df = missing_df[missing_df['Missing'] > 0].sort_values('Missing', ascending=False)
    if len(missing_df) > 0:
        print(missing_df.to_string(index=False))
    else:
        print("No missing values found!")
    
    print_section(f"{name} - Sample Data (first {min(n_rows, len(df))} rows)")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 50)
    print(df.head(n_rows).to_string())
    
    print_section(f"{name} - Numerical Statistics")
    print(df.describe().to_string())
    
    # Check for target columns (only in train)
    target_cols = [col for col in df.columns if 'buyer' in col or 'iap_revenue' in col or 'revenue' in col]
    if target_cols:
        print_section(f"{name} - Target Columns")
        print(f"Found {len(target_cols)} target columns:")
        for col in target_cols:
            if col in df.columns:
                print(f"\n  {col}:")
                print(f"    Type: {df[col].dtype}")
                print(f"    Unique values: {df[col].nunique()}")
                if pd.api.types.is_numeric_dtype(df[col]):
                    print(f"    Mean: {df[col].mean():.4f}")
                    print(f"    Std: {df[col].std():.4f}")
                    print(f"    Min: {df[col].min()}")
                    print(f"    Max: {df[col].max()}")
                print(f"    Value counts:\n{df[col].value_counts().head(10).to_string()}")


def main():
    """Main test function"""
    print_section("Data Loader Test")
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    print(f"Config loaded from: {config_path}")
    print(f"Train path: {config['data']['train_path']}")
    print(f"Test path: {config['data']['test_path']}")
    
    # Initialize loader
    loader = DataLoader(config)
    
    # Test 1: Load a small sample from training data
    print_section("Test 1: Loading Training Data Sample")
    try:
        train_df, val_df = loader.load_train_sample(n_rows=15, validation_split=True)
        
        if train_df is not None:
            display_dataframe_info(train_df, "TRAINING DATA", n_rows=15)
        
        if val_df is not None:
            display_dataframe_info(val_df, "VALIDATION DATA", n_rows=15)
            
    except Exception as e:
        print(f"Error loading training data: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 2: Load a small sample from test data
    print_section("Test 2: Loading Test Data Sample")
    try:
        test_df = loader.load_test_sample(n_rows=15)
        
        if test_df is not None:
            display_dataframe_info(test_df, "TEST DATA", n_rows=15)
            
    except Exception as e:
        print(f"Error loading test data: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Compare train and test schemas
    print_section("Test 3: Schema Comparison")
    try:
        train_df, _ = loader.load_train_sample(n_rows=100, validation_split=False)
        test_df = loader.load_test_sample(n_rows=100)
        
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        
        common_cols = train_cols & test_cols
        train_only = train_cols - test_cols
        test_only = test_cols - train_cols
        
        print(f"Common columns: {len(common_cols)}")
        print(f"Train-only columns: {len(train_only)}")
        if train_only:
            print(f"  {sorted(train_only)}")
        
        print(f"\nTest-only columns: {len(test_only)}")
        if test_only:
            print(f"  {sorted(test_only)}")
            
    except Exception as e:
        print(f"Error comparing schemas: {e}")
        import traceback
        traceback.print_exc()
    
    print_section("Tests Complete")


if __name__ == "__main__":
    main()
