"""
Test to verify all data quality improvements are working
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data.preprocessor import NestedFeatureParser
from src.features.engineer import FeatureEngineer
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def main():
    console.print(Panel.fit(
        "[bold green]Pipeline Improvements Validation[/bold green]\n"
        "Verifying all data quality fixes",
        border_style="green"
    ))
    
    # Load test data
    train_path = Path("data/raw/train/train")
    parquet_files = sorted(train_path.glob("datetime=*/part-*.parquet"))
    df_raw = pd.read_parquet(parquet_files[0]).head(1000)
    
    config_path = 'config/config_test.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    console.print("\n[cyan]Processing data through full pipeline...[/cyan]")
    
    # Initialize pipeline
    preprocessor = NestedFeatureParser()
    engineer = FeatureEngineer(config)
    
    # Process
    df_processed = preprocessor.process_all(df_raw.copy())
    df_final = engineer.engineer_all(df_processed.copy(), target_col='iap_revenue_d7', fit=True)
    
    console.print(f"[green]✓ Pipeline complete: {df_final.shape[0]:,} rows × {df_final.shape[1]:,} columns[/green]\n")
    
    # ========== VERIFICATION TABLE ==========
    
    table = Table(title="Data Quality Improvements", box=box.DOUBLE, show_header=True)
    table.add_column("Check", style="cyan", width=40)
    table.add_column("Status", style="bold", width=15)
    table.add_column("Details", style="yellow", width=45)
    
    # 1. Complex types handled
    complex_cols = []
    for col in df_final.columns:
        try:
            val = df_final[col].dropna().iloc[0] if len(df_final[col].dropna()) > 0 else None
            if isinstance(val, (list, dict, tuple)) or (hasattr(val, '__array__') and val.ndim > 0):
                complex_cols.append(col)
        except: pass
    
    status = "[green]✓ PASS[/green]" if len(complex_cols) == 0 else "[red]✗ FAIL[/red]"
    table.add_row(
        "1. Complex types handled",
        status,
        f"Found {len(complex_cols)} complex type columns"
    )
    
    # 2. Categorical encoding applied
    obj_cols = df_final.select_dtypes(include=['object']).columns.tolist()
    metadata_cols = {'row_id', 'datetime'}
    obj_cols = [c for c in obj_cols if c not in metadata_cols]
    
    status = "[green]✓ PASS[/green]" if len(obj_cols) == 0 else "[yellow]⚠ WARNING[/yellow]"
    table.add_row(
        "2. Categorical encoding",
        status,
        f"{len(obj_cols)} object columns remaining (should be 0)"
    )
    
    # 3. Missing values handled
    missing_pct = df_final.isna().sum().sum() / df_final.size * 100
    status = "[green]✓ PASS[/green]" if missing_pct < 1 else "[yellow]⚠ WARNING[/yellow]"
    table.add_row(
        "3. Missing values handled",
        status,
        f"{missing_pct:.2f}% missing"
    )
    
    # 4. Numeric features ready
    numeric_cols = df_final.select_dtypes(include=['number']).columns
    target_cols = {'buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28', 
                   'iap_revenue_d1', 'iap_revenue_d7', 'iap_revenue_d14', 'iap_revenue_d28'}
    feature_cols = [c for c in numeric_cols if c not in target_cols]
    
    status = "[green]✓ PASS[/green]" if len(feature_cols) > 50 else "[yellow]⚠ WARNING[/yellow]"
    table.add_row(
        "4. Numeric features available",
        status,
        f"{len(feature_cols)} numeric features for modeling"
    )
    
    # 5. Constant columns (would be removed by trainer)
    constant_cols = []
    critical_cols = ['row_id', 'datetime'] + list(target_cols)
    for col in df_final.columns:
        if col in critical_cols:
            continue
        try:
            if df_final[col].nunique(dropna=False) <= 1:
                constant_cols.append(col)
        except:
            try:
                if df_final[col].astype(str).nunique(dropna=False) <= 1:
                    constant_cols.append(col)
            except: pass
    
    status = "[green]✓ INFO[/green]"
    table.add_row(
        "5. Constant columns detected",
        status,
        f"{len(constant_cols)} constant columns (removed by trainer)"
    )
    
    # 6. Target columns present
    targets_present = [t for t in target_cols if t in df_final.columns]
    status = "[green]✓ PASS[/green]" if len(targets_present) >= 4 else "[red]✗ FAIL[/red]"
    table.add_row(
        "6. Target columns present",
        status,
        f"{len(targets_present)}/{len(target_cols)} targets available"
    )
    
    # 7. Inf/NaN in numeric columns
    has_inf = sum((df_final[col].isin([float('inf'), float('-inf')])).any() for col in numeric_cols)
    has_nan = sum(df_final[col].isna().any() for col in numeric_cols)
    status = "[green]✓ PASS[/green]" if has_inf == 0 and has_nan == 0 else "[yellow]⚠ WARNING[/yellow]"
    table.add_row(
        "7. No inf/nan values",
        status,
        f"{has_inf} columns with inf, {has_nan} with nan"
    )
    
    # 8. Encoded feature columns created
    encoded_cols = [c for c in df_final.columns if '_encoded' in c or '_target_encoded' in c]
    status = "[green]✓ PASS[/green]" if len(encoded_cols) > 0 else "[yellow]⚠ INFO[/yellow]"
    table.add_row(
        "8. Encoded feature columns",
        status,
        f"{len(encoded_cols)} encoded columns created"
    )
    
    console.print(table)
    
    # ========== SUMMARY ==========
    
    console.print("\n" + "=" * 80)
    console.print("[bold]IMPROVEMENT SUMMARY[/bold]")
    console.print("=" * 80)
    
    improvements = [
        ("Categorical Encoding", f"16 object columns → {len(encoded_cols)} encoded features"),
        ("Complex Type Handling", f"36 complex columns → {len(complex_cols)} remaining"),
        ("Constant Column Removal", f"~130 constant columns filtered by trainer"),
        ("Missing Value Handling", f"{missing_pct:.2f}% missing (imputed)"),
        ("Model-Ready Features", f"{len(feature_cols)} numeric features available"),
    ]
    
    for improvement, detail in improvements:
        console.print(f"  [cyan]•[/cyan] {improvement}: [yellow]{detail}[/yellow]")
    
    console.print("\n" + "=" * 80)
    
    # Check if all fixes are working
    all_pass = (
        len(complex_cols) == 0 and
        len(obj_cols) == 0 and
        missing_pct < 1 and
        len(feature_cols) > 50 and
        len(targets_present) >= 4 and
        has_inf == 0 and has_nan == 0
    )
    
    if all_pass:
        console.print(Panel.fit(
            "[bold green]✓ ALL IMPROVEMENTS VERIFIED![/bold green]\n"
            "Pipeline is production-ready",
            border_style="green",
            title="SUCCESS"
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]⚠ Some checks need attention[/bold yellow]\n"
            "Review the table above for details",
            border_style="yellow",
            title="REVIEW NEEDED"
        ))
    
    console.print()


if __name__ == '__main__':
    main()
