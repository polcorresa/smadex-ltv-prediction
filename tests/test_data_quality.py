"""
Comprehensive data quality check for model-ready dataset
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
        "[bold cyan]Data Quality Check for Model Training[/bold cyan]\n"
        "Ensuring dataset is model-ready",
        border_style="cyan"
    ))
    
    # Load and process data
    train_path = Path("data/raw/train/train")
    parquet_files = sorted(train_path.glob("datetime=*/part-*.parquet"))
    df_raw = pd.read_parquet(parquet_files[0]).head(1000)
    
    config_path = 'config/config_test.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    preprocessor = NestedFeatureParser()
    engineer = FeatureEngineer(config)
    
    console.print("\n[cyan]Processing data through full pipeline...[/cyan]")
    df_processed = preprocessor.process_all(df_raw.copy())
    df_final = engineer.engineer_all(df_processed.copy(), target_col='iap_revenue_d7', fit=True)
    
    console.print(f"[green]✓ Final dataset: {df_final.shape[0]:,} rows × {df_final.shape[1]:,} columns[/green]\n")
    
    # ========== CHECKS ==========
    
    # 1. Data Types
    console.print("="*80, style="cyan")
    console.print(" CHECK 1: Data Types", style="bold cyan")
    console.print("="*80 + "\n", style="cyan")
    
    type_table = Table(title="Column Types Distribution", box=box.ROUNDED)
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", justify="right", style="green")
    type_table.add_column("Status", style="yellow")
    
    numeric_cols = len(df_final.select_dtypes(include=['number']).columns)
    object_cols = len(df_final.select_dtypes(include=['object']).columns)
    bool_cols = len(df_final.select_dtypes(include=['bool']).columns)
    
    type_table.add_row("Numeric", str(numeric_cols), "✓ Good for modeling")
    type_table.add_row("Object", str(object_cols), "⚠ Need encoding" if object_cols > 20 else "✓ Manageable")
    type_table.add_row("Boolean", str(bool_cols), "✓ Ready")
    
    console.print(type_table)
    
    # 2. Complex Types Check
    console.print("\n" + "="*80, style="cyan")
    console.print(" CHECK 2: Complex/Unhashable Types", style="bold cyan")
    console.print("="*80 + "\n", style="cyan")
    
    complex_cols = []
    for col in df_final.columns:
        try:
            val = df_final[col].dropna().iloc[0] if len(df_final[col].dropna()) > 0 else None
            if isinstance(val, (list, dict, tuple)) or (hasattr(val, '__array__') and val.ndim > 0):
                complex_cols.append((col, type(val).__name__))
        except: pass
    
    if complex_cols:
        console.print(f"[red]✗ Found {len(complex_cols)} columns with complex types:[/red]")
        for col, typ in complex_cols[:10]:
            console.print(f"  • {col}: {typ}")
    else:
        console.print("[green]✓ No complex types found - all columns are model-ready[/green]")
    
    # 3. Missing Values
    console.print("\n" + "="*80, style="cyan")
    console.print(" CHECK 3: Missing Values", style="bold cyan")
    console.print("="*80 + "\n", style="cyan")
    
    missing_pct = df_final.isna().sum().sum() / df_final.size * 100
    cols_with_missing = (df_final.isna().sum() > 0).sum()
    
    missing_table = Table(box=box.ROUNDED)
    missing_table.add_column("Metric", style="cyan")
    missing_table.add_column("Value", justify="right", style="yellow")
    missing_table.add_column("Status", style="green")
    
    missing_table.add_row("Overall missing %", f"{missing_pct:.2f}%", "✓ Good" if missing_pct < 1 else "⚠ High")
    missing_table.add_row("Columns with missing", f"{cols_with_missing}/{len(df_final.columns)}", "✓ Good" if cols_with_missing < 5 else "⚠ Many")
    
    console.print(missing_table)
    
    if cols_with_missing > 0:
        console.print("\n[yellow]Columns with missing values:[/yellow]")
        missing_cols = df_final.columns[df_final.isna().any()].tolist()
        for col in missing_cols:
            pct = df_final[col].isna().sum() / len(df_final) * 100
            console.print(f"  • {col}: {pct:.2f}%")
    
    # 4. Target Columns
    console.print("\n" + "="*80, style="cyan")
    console.print(" CHECK 4: Target Columns", style="bold cyan")
    console.print("="*80 + "\n", style="cyan")
    
    targets = {
        'buyer_d1': 'binary',
        'buyer_d7': 'binary',
        'buyer_d14': 'binary',
        'buyer_d28': 'binary',
        'iap_revenue_d7': 'continuous',
        'iap_revenue_d14': 'continuous',
        'iap_revenue_d28': 'continuous'
    }
    
    target_table = Table(title="Target Variables", box=box.ROUNDED)
    target_table.add_column("Target", style="cyan")
    target_table.add_column("Type", style="magenta")
    target_table.add_column("Missing", justify="right", style="red")
    target_table.add_column("Mean", justify="right", style="green")
    target_table.add_column("Status", style="yellow")
    
    for target, ttype in targets.items():
        if target in df_final.columns:
            missing = df_final[target].isna().sum()
            mean = df_final[target].mean()
            status = "✓ Ready" if missing == 0 else f"✗ {missing} missing"
            target_table.add_row(target, ttype, str(missing), f"{mean:.4f}", status)
        else:
            target_table.add_row(target, ttype, "N/A", "N/A", "✗ NOT FOUND")
    
    console.print(target_table)
    
    # 5. Inf/NaN in Numeric Columns
    console.print("\n" + "="*80, style="cyan")
    console.print(" CHECK 5: Inf/NaN in Numeric Columns", style="bold cyan")
    console.print("="*80 + "\n", style="cyan")
    
    numeric_cols = df_final.select_dtypes(include=['number']).columns
    has_inf = sum((df_final[col].isin([float('inf'), float('-inf')])).any() for col in numeric_cols)
    has_nan = sum(df_final[col].isna().any() for col in numeric_cols)
    
    if has_inf == 0 and has_nan == 0:
        console.print("[green]✓ No inf or nan values in numeric columns[/green]")
    else:
        console.print(f"[yellow]Columns with inf: {has_inf}[/yellow]")
        console.print(f"[yellow]Columns with nan: {has_nan}[/yellow]")
    
    # 6. Missing Indicator Flags
    console.print("\n" + "="*80, style="cyan")
    console.print(" CHECK 6: Missing Indicator Flags", style="bold cyan")
    console.print("="*80 + "\n", style="cyan")
    
    flag_cols = [c for c in df_final.columns if '_is_missing' in c]
    console.print(f"[cyan]Total flag columns: {len(flag_cols)}[/cyan]")
    
    if flag_cols and 'buyer_d7' in df_final.columns:
        # Check which flags have meaningful correlation
        high_corr = []
        for flag in flag_cols:
            corr = abs(df_final[flag].corr(df_final['buyer_d7']))
            if corr > 0.05:
                high_corr.append((flag, corr))
        
        if high_corr:
            console.print(f"\n[green]✓ {len(high_corr)} flags have correlation > 0.05 with buyer_d7[/green]")
            console.print("Top 5:")
            for flag, corr in sorted(high_corr, key=lambda x: x[1], reverse=True)[:5]:
                console.print(f"  • {flag}: {corr:.3f}")
        else:
            console.print("[yellow]⚠ No flags have meaningful correlation with target[/yellow]")
            console.print("[yellow]  Consider removing flag columns to reduce dimensionality[/yellow]")
    
    # 7. Constant Columns
    console.print("\n" + "="*80, style="cyan")
    console.print(" CHECK 7: Constant Columns", style="bold cyan")
    console.print("="*80 + "\n", style="cyan")
    
    constant_cols = []
    for col in df_final.columns:
        try:
            if df_final[col].apply(str).nunique() <= 1:
                constant_cols.append(col)
        except:
            pass
    
    if constant_cols:
        console.print(f"[yellow]⚠ Found {len(constant_cols)} constant columns (should be removed):[/yellow]")
        for col in constant_cols[:10]:
            console.print(f"  • {col}")
        if len(constant_cols) > 10:
            console.print(f"  ... and {len(constant_cols)-10} more")
    else:
        console.print("[green]✓ No constant columns found[/green]")
    
    # 8. Object Columns (need encoding)
    console.print("\n" + "="*80, style="cyan")
    console.print(" CHECK 8: Object Columns (need encoding)", style="bold cyan")
    console.print("="*80 + "\n", style="cyan")
    
    obj_cols = df_final.select_dtypes(include=['object']).columns.tolist()
    # Exclude metadata columns
    metadata_cols = {'row_id', 'datetime'}
    obj_cols_for_encoding = [c for c in obj_cols if c not in metadata_cols]
    
    if obj_cols_for_encoding:
        console.print(f"[yellow]⚠ {len(obj_cols_for_encoding)} object columns need encoding:[/yellow]")
        
        enc_table = Table(box=box.SIMPLE)
        enc_table.add_column("Column", style="cyan")
        enc_table.add_column("Cardinality", justify="right", style="yellow")
        enc_table.add_column("Recommendation", style="green")
        
        for col in obj_cols_for_encoding[:15]:
            try:
                cardinality = df_final[col].apply(str).nunique()
                if cardinality <= 10:
                    rec = "One-hot encoding"
                elif cardinality <= 50:
                    rec = "Target encoding"
                else:
                    rec = "Frequency/Hash encoding"
                enc_table.add_row(col, str(cardinality), rec)
            except:
                enc_table.add_row(col, "ERROR", "Manual review")
        
        console.print(enc_table)
        if len(obj_cols_for_encoding) > 15:
            console.print(f"  ... and {len(obj_cols_for_encoding)-15} more")
    else:
        console.print("[green]✓ All categorical columns already encoded[/green]")
    
    # Final Summary
    console.print("\n" + "="*80, style="bold cyan")
    console.print(" FINAL SUMMARY", style="bold cyan")
    console.print("="*80 + "\n", style="bold cyan")
    
    issues = []
    if complex_cols:
        issues.append(f"✗ {len(complex_cols)} columns with complex types")
    if missing_pct > 1:
        issues.append(f"⚠ {missing_pct:.2f}% missing values")
    if has_inf > 0:
        issues.append(f"✗ {has_inf} columns with inf values")
    if constant_cols:
        issues.append(f"⚠ {len(constant_cols)} constant columns")
    if len(obj_cols_for_encoding) > 0:
        issues.append(f"⚠ {len(obj_cols_for_encoding)} object columns need encoding")
    
    if not issues:
        console.print(Panel.fit(
            "[bold green]✓ Dataset is MODEL-READY![/bold green]\n"
            "All checks passed. Data can be used for training.",
            border_style="green"
        ))
    else:
        console.print(Panel.fit(
            "[bold yellow]⚠ Dataset needs attention:[/bold yellow]\n" + "\n".join(f"  {issue}" for issue in issues),
            border_style="yellow"
        ))
        
        console.print("\n[bold]Recommendations:[/bold]")
        if complex_cols:
            console.print("  1. Add missing columns to preprocessor parsing")
        if len(obj_cols_for_encoding) > 0:
            console.print("  2. Add categorical encoding to feature engineering")
        if constant_cols:
            console.print("  3. Remove constant columns before training")
        if len(flag_cols) > 30 and not high_corr:
            console.print("  4. Consider removing low-value missing indicators")
    
    console.print()


if __name__ == '__main__':
    main()
