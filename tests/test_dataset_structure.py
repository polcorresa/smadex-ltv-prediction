"""
Test to visualize the complete dataset structure after preprocessing
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.data.loader import DataLoader
from src.data.preprocessor import NestedFeatureParser
from src.features.engineer import FeatureEngineer
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

def print_section(title: str):
    """Print a section separator"""
    console.print(f"\n{'='*80}", style="bold cyan")
    console.print(f" {title}", style="bold cyan")
    console.print(f"{'='*80}\n", style="bold cyan")


def analyze_dataframe_structure(df: pd.DataFrame, name: str):
    """Comprehensive analysis of dataframe structure"""
    
    print_section(f"{name} - Overview")
    
    # Basic stats
    console.print(f"[bold]Shape:[/bold] {df.shape[0]:,} rows × {df.shape[1]:,} columns")
    console.print(f"[bold]Memory usage:[/bold] {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    console.print(f"[bold]Missing cells:[/bold] {df.isna().sum().sum():,} ({df.isna().sum().sum() / df.size * 100:.2f}%)")
    
    # Column types
    print_section(f"{name} - Column Types Distribution")
    
    type_counts = df.dtypes.value_counts()
    type_table = Table(title="Data Types", box=box.ROUNDED)
    type_table.add_column("Type", style="cyan")
    type_table.add_column("Count", justify="right", style="green")
    type_table.add_column("Percentage", justify="right", style="yellow")
    
    for dtype, count in type_counts.items():
        pct = count / len(df.columns) * 100
        type_table.add_row(str(dtype), str(count), f"{pct:.1f}%")
    
    console.print(type_table)
    
    # Numeric columns statistics
    print_section(f"{name} - Numeric Columns Summary")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    console.print(f"[bold]Numeric columns:[/bold] {len(numeric_cols)}")
    
    if len(numeric_cols) > 0:
        stats_table = Table(title="Numeric Column Statistics", box=box.ROUNDED)
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", justify="right", style="green")
        
        numeric_df = df[numeric_cols]
        stats_table.add_row("Mean of means", f"{numeric_df.mean().mean():.4f}")
        stats_table.add_row("Mean of stds", f"{numeric_df.std().mean():.4f}")
        stats_table.add_row("Columns with zeros", f"{(numeric_df == 0).all().sum()}")
        stats_table.add_row("Columns with missing", f"{numeric_df.isna().any().sum()}")
        stats_table.add_row("Max value", f"{numeric_df.max().max():.4f}")
        stats_table.add_row("Min value", f"{numeric_df.min().min():.4f}")
        
        console.print(stats_table)
    
    # Categorical columns
    print_section(f"{name} - Categorical Columns")
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    console.print(f"[bold]Categorical columns:[/bold] {len(cat_cols)}")
    
    if len(cat_cols) > 0:
        cat_table = Table(title="Categorical Columns Details", box=box.ROUNDED, show_lines=True)
        cat_table.add_column("Column", style="cyan", no_wrap=True)
        cat_table.add_column("Type", style="magenta")
        cat_table.add_column("Non-null", justify="right", style="green")
        cat_table.add_column("Sample Value", style="yellow")
        
        for col in cat_cols[:15]:  # Show first 15
            try:
                # Check if column contains hashable types
                first_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if isinstance(first_val, (list, dict, tuple)):
                    col_type = type(first_val).__name__
                    sample = str(first_val)[:30] + "..." if len(str(first_val)) > 30 else str(first_val)
                else:
                    col_type = "str/categorical"
                    sample = str(df[col].mode()[0])[:30] if len(df[col].mode()) > 0 else "N/A"
                
                non_null = df[col].notna().sum()
                cat_table.add_row(
                    col[:40], 
                    col_type,
                    str(non_null),
                    sample
                )
            except Exception as e:
                # Fallback for problematic columns
                cat_table.add_row(col[:40], "complex", str(df[col].notna().sum()), "[error reading]")
        
        console.print(cat_table)
        if len(cat_cols) > 15:
            console.print(f"[dim]... and {len(cat_cols) - 15} more categorical columns[/dim]")
    
    # Feature groups analysis
    print_section(f"{name} - Feature Groups")
    
    feature_groups = {
        'Revenue features': [c for c in df.columns if 'revenue' in c.lower()],
        'Buyer features': [c for c in df.columns if 'buyer' in c.lower() or 'buy' in c.lower()],
        'Session features': [c for c in df.columns if 'session' in c.lower()],
        'Bundle features': [c for c in df.columns if 'bundle' in c.lower()],
        'Category features': [c for c in df.columns if 'category' in c.lower()],
        'Device features': [c for c in df.columns if 'dev_' in c.lower()],
        'Temporal features': [c for c in df.columns if any(x in c.lower() for x in ['hour', 'day', 'weekday', 'weekend', 'ts'])],
        'Missing indicators': [c for c in df.columns if '_is_missing' in c.lower()],
        'Aggregated features': [c for c in df.columns if any(x in c for x in ['_mean', '_std', '_max', '_min', '_count'])],
    }
    
    group_table = Table(title="Feature Groups", box=box.ROUNDED)
    group_table.add_column("Group", style="cyan")
    group_table.add_column("Count", justify="right", style="green")
    group_table.add_column("Examples", style="yellow")
    
    for group_name, cols in feature_groups.items():
        if cols:
            examples = ", ".join(cols[:3])
            if len(cols) > 3:
                examples += f", ... (+{len(cols)-3} more)"
            group_table.add_row(group_name, str(len(cols)), examples)
    
    console.print(group_table)
    
    # Target columns
    print_section(f"{name} - Target Columns")
    
    target_cols = [c for c in df.columns if any(x in c for x in ['iap_revenue_d', 'buyer_d', 'retention_d'])]
    if target_cols:
        target_table = Table(title="Target Variables", box=box.ROUNDED)
        target_table.add_column("Column", style="cyan")
        target_table.add_column("Type", style="magenta")
        target_table.add_column("Non-null", justify="right", style="green")
        target_table.add_column("Mean/Mode", justify="right", style="yellow")
        
        for col in target_cols:
            if col in df.columns:
                dtype = str(df[col].dtype)
                non_null = df[col].notna().sum()
                if df[col].dtype in [np.number, 'float64', 'int64']:
                    summary = f"{df[col].mean():.4f}"
                else:
                    mode_val = df[col].mode()[0] if len(df[col].mode()) > 0 else "N/A"
                    summary = str(mode_val)[:20]
                target_table.add_row(col, dtype, str(non_null), summary)
        
        console.print(target_table)
    else:
        console.print("[yellow]No target columns found in this dataset[/yellow]")


def main():
    console.print(Panel.fit(
        "[bold cyan]Dataset Structure Visualization Test[/bold cyan]\n"
        "Analyzing raw → processed → engineered data",
        border_style="cyan"
    ))
    
    # Load config
    config_path = 'config/config_test.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    console.print(f"\n[bold]Loading data with test configuration...[/bold]")
    
    # Initialize components
    preprocessor = NestedFeatureParser()
    engineer = FeatureEngineer(config)
    
    # Load raw data (sample) - Direct parquet read for speed
    console.print("[cyan]Loading training data (first 1000 rows directly from parquet)...[/cyan]")
    train_path = Path("data/raw/train/train")
    
    # Find first available partition file
    parquet_files = sorted(train_path.glob("datetime=*/part-*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {train_path}")
    
    # Read first file with row limit
    console.print(f"[dim]Reading from: {parquet_files[0].parent.name}/{parquet_files[0].name}[/dim]")
    train_df = pd.read_parquet(parquet_files[0], engine='pyarrow')
    
    # Sample if more than 1000 rows
    if len(train_df) > 1000:
        train_df = train_df.head(1000)
    
    console.print(f"[green]✓ Loaded {len(train_df)} rows[/green]")
    
    # STAGE 1: Raw data
    analyze_dataframe_structure(train_df, "STAGE 1: Raw Data")
    
    # STAGE 2: After preprocessing
    console.print("\n[cyan]Preprocessing features...[/cyan]")
    train_processed = preprocessor.process_all(train_df.copy())
    analyze_dataframe_structure(train_processed, "STAGE 2: After Preprocessing")
    
    # STAGE 3: After feature engineering
    console.print("\n[cyan]Engineering features...[/cyan]")
    train_engineered = engineer.engineer_all(
        train_processed.copy(),
        target_col='iap_revenue_d7',
        fit=True
    )
    analyze_dataframe_structure(train_engineered, "STAGE 3: After Feature Engineering")
    
    # Final summary comparison
    print_section("Pipeline Transformation Summary")
    
    summary_table = Table(title="Data Transformation Pipeline", box=box.DOUBLE)
    summary_table.add_column("Stage", style="cyan", no_wrap=True)
    summary_table.add_column("Rows", justify="right", style="green")
    summary_table.add_column("Columns", justify="right", style="yellow")
    summary_table.add_column("Memory (MB)", justify="right", style="magenta")
    summary_table.add_column("Missing %", justify="right", style="red")
    
    stages = [
        ("Raw Data", train_df),
        ("After Preprocessing", train_processed),
        ("After Engineering", train_engineered)
    ]
    
    for stage_name, df in stages:
        rows = df.shape[0]
        cols = df.shape[1]
        memory = df.memory_usage(deep=True).sum() / 1024**2
        missing_pct = df.isna().sum().sum() / df.size * 100
        summary_table.add_row(
            stage_name,
            f"{rows:,}",
            f"{cols:,}",
            f"{memory:.2f}",
            f"{missing_pct:.2f}%"
        )
    
    console.print(summary_table)
    
    # Column growth analysis
    print_section("Feature Creation Breakdown")
    
    growth_table = Table(title="Column Growth by Stage", box=box.ROUNDED)
    growth_table.add_column("Transition", style="cyan")
    growth_table.add_column("Added", justify="right", style="green")
    growth_table.add_column("Removed", justify="right", style="red")
    growth_table.add_column("Net Change", justify="right", style="yellow")
    
    # Raw → Preprocessed
    raw_cols = set(train_df.columns)
    processed_cols = set(train_processed.columns)
    added_1 = len(processed_cols - raw_cols)
    removed_1 = len(raw_cols - processed_cols)
    growth_table.add_row(
        "Raw → Preprocessed",
        f"+{added_1}",
        f"-{removed_1}",
        f"{added_1 - removed_1:+d}"
    )
    
    # Preprocessed → Engineered
    engineered_cols = set(train_engineered.columns)
    added_2 = len(engineered_cols - processed_cols)
    removed_2 = len(processed_cols - engineered_cols)
    growth_table.add_row(
        "Preprocessed → Engineered",
        f"+{added_2}",
        f"-{removed_2}",
        f"{added_2 - removed_2:+d}"
    )
    
    # Total
    total_added = len(engineered_cols - raw_cols)
    total_removed = len(raw_cols - engineered_cols)
    growth_table.add_row(
        "[bold]Total (Raw → Final)[/bold]",
        f"[bold]+{total_added}[/bold]",
        f"[bold]-{total_removed}[/bold]",
        f"[bold]{total_added - total_removed:+d}[/bold]"
    )
    
    console.print(growth_table)
    
    console.print("\n[bold green]✓ Dataset structure analysis complete![/bold green]\n")


if __name__ == '__main__':
    main()
