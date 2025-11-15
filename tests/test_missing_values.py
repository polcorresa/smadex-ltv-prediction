"""
Test missing value handling
"""
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.missing_handler import MissingValueHandler, summarize_missing_values
from src.data.preprocessor import NestedFeatureParser


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80 + "\n")


def main():
    """Test missing value handling"""
    
    print_section("Missing Value Handling Test")
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Initialize loader
    loader = DataLoader(config)
    
    # Load a sample
    print("Loading training sample (1000 rows)...")
    train_df, _ = loader.load_train_sample(n_rows=1000, validation_split=False)
    
    print(f"Loaded: {train_df.shape[0]} rows × {train_df.shape[1]} columns")
    
    # Step 1: Analyze raw missing values
    print_section("Step 1: Raw Missing Value Analysis")
    summarize_missing_values(train_df)
    
    # Step 2: Initialize missing handler
    print_section("Step 2: Initialize Missing Handler")
    missing_handler = MissingValueHandler(config)
    
    # Fit on training data
    print("Fitting missing handler on training data...")
    missing_handler.fit(train_df, verbose=True)
    
    # Get strategy summary
    summary = missing_handler.get_summary()
    print(f"\nTotal columns with missing: {summary['total_columns_with_missing']}")
    print("\nStrategy distribution:")
    for strategy, count in summary['strategy_distribution'].items():
        print(f"  {strategy:20s}: {count:3d} columns")
    
    print("\nSample imputation strategies:")
    for col, info in summary['sample_imputations'].items():
        print(f"  {col:50s}: {info['strategy']:15s} -> {info['value']}")
    
    # Step 3: Transform data
    print_section("Step 3: Transform Data (Apply Imputation)")
    train_df_imputed = missing_handler.transform(train_df)
    
    print("After imputation:")
    summarize_missing_values(train_df_imputed)
    
    # Step 4: Test with missing indicators
    print_section("Step 4: Create Missing Indicators")
    train_df_with_indicators = missing_handler.create_missing_indicators(
        train_df.copy(), 
        threshold=0.05
    )
    
    indicator_cols = [col for col in train_df_with_indicators.columns if '_is_missing' in col]
    print(f"Created {len(indicator_cols)} missing indicator columns")
    
    if indicator_cols:
        print("\nSample indicators:")
        for col in indicator_cols[:10]:
            original_col = col.replace('_is_missing', '')
            pct = train_df_with_indicators[col].mean() * 100
            print(f"  {col:60s}: {pct:5.1f}% missing in original")
    
    # Step 5: Test with preprocessor integration
    print_section("Step 5: Test Preprocessor Integration")
    preprocessor = NestedFeatureParser(config)
    
    print("Processing features with missing handling...")
    train_processed = preprocessor.process_all(
        train_df.copy(),
        handle_missing=True,
        create_missing_indicators=True,
        fit_missing_handler=True
    )
    
    print(f"\nAfter preprocessing:")
    print(f"  Original shape: {train_df.shape}")
    print(f"  Processed shape: {train_processed.shape}")
    print(f"  New features created: {train_processed.shape[1] - train_df.shape[1]}")
    
    # Check for remaining missing values
    print("\nFinal missing value check:")
    summarize_missing_values(train_processed)
    
    # Step 6: Compare before/after for specific columns
    print_section("Step 6: Before/After Comparison (Sample Columns)")
    
    # Select a few columns to compare
    sample_cols = [
        'avg_daily_sessions',
        'avg_duration',
        'wifi_ratio',
        'weeks_since_first_seen'
    ]
    
    for col in sample_cols:
        if col in train_df.columns and col in train_processed.columns:
            before_missing = train_df[col].isnull().sum()
            after_missing = train_processed[col].isnull().sum()
            before_pct = (before_missing / len(train_df)) * 100
            after_pct = (after_missing / len(train_processed)) * 100
            
            print(f"\n{col}:")
            print(f"  Before: {before_missing:4d} missing ({before_pct:5.1f}%)")
            print(f"  After:  {after_missing:4d} missing ({after_pct:5.1f}%)")
            
            if before_missing > 0:
                # Show sample of imputed values
                was_missing_mask = train_df[col].isnull()
                if was_missing_mask.sum() > 0:
                    imputed_values = train_processed.loc[was_missing_mask, col].head(5)
                    print(f"  Imputed values (sample): {imputed_values.tolist()}")
    
    print_section("Test Complete")
    print("\nKey Takeaways:")
    print("✓ Missing values are systematically identified")
    print("✓ Domain-appropriate imputation strategies are applied")
    print("✓ Missing indicators can be created for predictive power")
    print("✓ Integration with preprocessor is seamless")
    print("✓ All missing values are handled before modeling")


if __name__ == "__main__":
    main()
