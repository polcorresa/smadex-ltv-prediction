"""
Test avg_daily_sessions parsing and critical column preservation
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from data.loader import DataLoader
from data.preprocessor import NestedFeatureParser
from features.engineer import FeatureEngineer

console = Console()

def test_session_parsing_and_critical_columns():
    """Test that avg_daily_sessions is parsed correctly and critical columns preserved"""
    
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    console.print("\n[bold cyan]🔍 Testing Session Parsing & Critical Column Preservation[/bold cyan]\n")
    
    # Initialize components
    loader = DataLoader(config)
    preprocessor = NestedFeatureParser(config)
    engineer = FeatureEngineer(config)
    
    # Load sample
    console.print("[yellow]Loading sample data...[/yellow]")
    train_df, _ = loader.load_train_sample(n_rows=1000, validation_split=False)
    
    console.print(f"✅ Loaded {len(train_df)} rows\n")
    
    # Test 1: Check avg_daily_sessions before processing
    console.print("[bold yellow]Test 1:[/bold yellow] Check avg_daily_sessions structure")
    if 'avg_daily_sessions' in train_df.columns:
        sample_val = train_df['avg_daily_sessions'].iloc[2]
        console.print(f"  Type: {type(sample_val)}")
        console.print(f"  Sample value: {sample_val}")
        console.print(f"  ✅ Column exists and is list type\n")
    else:
        console.print("  ❌ Column not found!\n")
    
    # Test 2: Check critical columns before processing
    console.print("[bold yellow]Test 2:[/bold yellow] Critical columns before processing")
    critical_cols = [
        'row_id', 'datetime',
        'buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28',
        'iap_revenue_d1', 'iap_revenue_d7', 'iap_revenue_d14', 'iap_revenue_d28'
    ]
    
    before_critical = [col for col in critical_cols if col in train_df.columns]
    console.print(f"  Found {len(before_critical)}/{len(critical_cols)} critical columns:")
    for col in before_critical:
        console.print(f"    • {col}")
    console.print()
    
    # Test 3: Preprocess
    console.print("[bold yellow]Test 3:[/bold yellow] Preprocessing (parsing nested features)")
    try:
        processed_df = preprocessor.process_all(
            train_df.copy(),
            handle_missing=True,
            create_missing_indicators=True,
            fit_missing_handler=True
        )
        console.print(f"  ✅ Preprocessing successful")
        console.print(f"  Columns: {len(train_df.columns)} → {len(processed_df.columns)}\n")
    except Exception as e:
        console.print(f"  ❌ Error: {e}\n")
        return
    
    # Test 4: Check avg_daily_sessions parsing
    console.print("[bold yellow]Test 4:[/bold yellow] Check avg_daily_sessions parsing")
    session_features = [col for col in processed_df.columns if 'avg_daily_sessions' in col]
    
    if session_features:
        console.print(f"  ✅ Created {len(session_features)} session features:")
        for feat in session_features:
            values = processed_df[feat]
            if pd.api.types.is_numeric_dtype(values):
                console.print(f"    • {feat}: mean={values.mean():.2f}, max={values.max():.2f}")
            else:
                console.print(f"    • {feat}: (non-numeric)")
    else:
        console.print("  ❌ No session features created!")
    console.print()
    
    # Test 5: Check critical columns after preprocessing
    console.print("[bold yellow]Test 5:[/bold yellow] Critical columns after preprocessing")
    after_critical = [col for col in critical_cols if col in processed_df.columns]
    missing_critical = set(before_critical) - set(after_critical)
    
    if not missing_critical:
        console.print(f"  ✅ All {len(after_critical)} critical columns preserved:")
        for col in after_critical:
            console.print(f"    • {col}")
    else:
        console.print(f"  ⚠️  {len(missing_critical)} critical columns LOST:")
        for col in missing_critical:
            console.print(f"    • {col} [red](MISSING)[/red]")
    console.print()
    
    # Test 6: Feature engineering
    console.print("[bold yellow]Test 6:[/bold yellow] Feature engineering")
    try:
        engineered_df = engineer.engineer_all(
            processed_df.copy(),
            target_col='iap_revenue_d7',
            fit=True
        )
        console.print(f"  ✅ Feature engineering successful")
        console.print(f"  Columns: {len(processed_df.columns)} → {len(engineered_df.columns)}\n")
    except Exception as e:
        console.print(f"  ❌ Error: {e}\n")
        return
    
    # Test 7: Check session-based features
    console.print("[bold yellow]Test 7:[/bold yellow] Session-based interaction features")
    session_interaction_features = [
        'session_consistency', 'engagement_score', 'wifi_engagement'
    ]
    
    found_interactions = [f for f in session_interaction_features if f in engineered_df.columns]
    console.print(f"  Found {len(found_interactions)}/{len(session_interaction_features)} features:")
    for feat in found_interactions:
        values = engineered_df[feat]
        console.print(f"    • {feat}: mean={values.mean():.2f}, nonzero={((values != 0).mean() * 100):.1f}%")
    console.print()
    
    # Test 8: Final critical column check
    console.print("[bold yellow]Test 8:[/bold yellow] Critical columns after full pipeline")
    final_critical = [col for col in critical_cols if col in engineered_df.columns]
    final_missing = set(before_critical) - set(final_critical)
    
    if not final_missing:
        console.print(f"  ✅ All {len(final_critical)} critical columns preserved through full pipeline")
    else:
        console.print(f"  ❌ {len(final_missing)} critical columns LOST in pipeline:")
        for col in final_missing:
            console.print(f"    • {col} [red](MISSING)[/red]")
    console.print()
    
    # Summary
    summary_lines = [
        "[bold green]✅ All Tests Passed![/bold green]" if not final_missing else "[bold red]❌ Issues Found[/bold red]",
        "",
        "[bold]Session Parsing:[/bold]",
        f"• avg_daily_sessions → {len(session_features)} features",
        f"• Session-based interactions: {len(found_interactions)}/3 created",
        "",
        "[bold]Critical Columns:[/bold]",
        f"• Before: {len(before_critical)}/{len(critical_cols)}",
        f"• After preprocessing: {len(after_critical)}/{len(critical_cols)}",
        f"• After engineering: {len(final_critical)}/{len(critical_cols)}",
        f"• Lost: {len(final_missing)} columns" if final_missing else "• Lost: 0 columns ✅",
        "",
        "[bold]Pipeline:[/bold]",
        f"• Raw: {len(train_df.columns)} cols",
        f"• Preprocessed: {len(processed_df.columns)} cols",
        f"• Engineered: {len(engineered_df.columns)} cols"
    ]
    
    summary = Panel(
        "\n".join(summary_lines),
        title="🎯 Test Summary",
        border_style="green" if not final_missing else "red"
    )
    
    console.print(summary)


if __name__ == "__main__":
    test_session_parsing_and_critical_columns()
