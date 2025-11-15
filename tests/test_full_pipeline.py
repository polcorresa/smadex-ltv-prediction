"""
Full pipeline test: Load → Preprocess → Display
Shows complete data transformation with fancy terminal output
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import DataLoader
from src.data.preprocessor import NestedFeatureParser
from src.features.engineer import FeatureEngineer

# Initialize Rich console
console = Console()


def create_fancy_header(title: str, subtitle: str = None):
    """Create a fancy header panel"""
    text = Text(title, style="bold magenta", justify="center")
    if subtitle:
        text.append("\n" + subtitle, style="italic cyan")
    
    panel = Panel(
        text,
        box=box.DOUBLE,
        border_style="bright_blue",
        padding=(1, 2)
    )
    console.print(panel)


def create_stats_table(df: pd.DataFrame, title: str):
    """Create a fancy statistics table"""
    table = Table(
        title=f"📊 {title}",
        box=box.ROUNDED,
        title_style="bold yellow",
        header_style="bold cyan",
        show_lines=True
    )
    
    table.add_column("Metric", style="green", width=30)
    table.add_column("Value", style="white", justify="right", width=20)
    
    # Calculate stats
    stats = [
        ("Rows", f"{df.shape[0]:,}"),
        ("Columns", f"{df.shape[1]:,}"),
        ("Total Cells", f"{df.shape[0] * df.shape[1]:,}"),
        ("Missing Cells", f"{df.isnull().sum().sum():,}"),
        ("Missing %", f"{(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.2f}%"),
        ("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"),
    ]
    
    for metric, value in stats:
        table.add_row(metric, value)
    
    return table


def create_sample_table(df: pd.DataFrame, n_rows: int = 15):
    """Create a fancy sample data table"""
    table = Table(
        title=f"🔍 Sample Data (First {n_rows} Rows)",
        box=box.SIMPLE_HEAD,
        title_style="bold magenta",
        header_style="bold yellow",
        show_lines=False,
        row_styles=["", "dim"]
    )
    
    # Select interesting columns to display
    priority_cols = [
        'row_id', 'datetime', 'country', 'dev_os', 'advertiser_category',
        'buyer_d1', 'buyer_d7', 'iap_revenue_d7', 
        'avg_daily_sessions', 'wifi_ratio', 'weeks_since_first_seen'
    ]
    
    # Filter to available columns
    display_cols = [col for col in priority_cols if col in df.columns]
    
    # Add fallback columns if we don't have enough
    if len(display_cols) < 8:
        for col in df.columns:
            if col not in display_cols and len(display_cols) < 12:
                display_cols.append(col)
    
    # Add index column
    table.add_column("#", style="dim", width=4, justify="right")
    
    # Add data columns
    for col in display_cols[:10]:  # Limit to 10 columns for readability
        # Determine width based on column name
        width = min(max(len(col) + 2, 12), 20)
        table.add_column(col, width=width, overflow="fold")
    
    # Add rows
    sample_df = df.head(n_rows)
    for idx, row in sample_df.iterrows():
        row_data = [str(idx + 1)]
        
        for col in display_cols[:10]:
            value = row[col]
            
            # Format based on type
            try:
                if pd.isna(value):
                    formatted = "[dim]NULL[/dim]"
                elif isinstance(value, (list, dict, tuple, np.ndarray)):
                    formatted = "[dim]complex[/dim]"
                elif isinstance(value, float):
                    if value == 0:
                        formatted = "[dim]0[/dim]"
                    else:
                        formatted = f"{value:.2f}" if abs(value) < 1000 else f"{value:.0f}"
                elif isinstance(value, str):
                    if len(value) > 18:
                        formatted = value[:15] + "..."
                    else:
                        formatted = value
                else:
                    formatted = str(value)
            except:
                # Fallback for any problematic values
                formatted = "[dim]error[/dim]"
            
            row_data.append(formatted)
        
        table.add_row(*row_data)
    
    return table


def create_column_summary(df: pd.DataFrame):
    """Create a summary of column types and statistics"""
    table = Table(
        title="📋 Column Summary (Top 20)",
        box=box.SIMPLE,
        title_style="bold cyan",
        header_style="bold green",
        show_lines=True
    )
    
    table.add_column("Column", style="yellow", width=40)
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Non-Null", style="green", justify="right", width=10)
    table.add_column("Unique", style="magenta", justify="right", width=10)
    table.add_column("Sample", style="white", width=30)
    
    # Get column info
    for col in df.columns[:20]:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        n_unique = df[col].nunique()
        
        # Get sample value
        sample_vals = df[col].dropna().head(1)
        if len(sample_vals) > 0:
            sample = str(sample_vals.iloc[0])
            if len(sample) > 28:
                sample = sample[:25] + "..."
        else:
            sample = "[dim]all null[/dim]"
        
        table.add_row(col, dtype, str(non_null), str(n_unique), sample)
    
    if len(df.columns) > 20:
        table.add_row(
            f"[dim]... {len(df.columns) - 20} more columns ...[/dim]",
            "", "", "", ""
        )
    
    return table


def create_target_distribution(df: pd.DataFrame):
    """Create target variable distribution table"""
    target_cols = ['buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28',
                   'iap_revenue_d7', 'iap_revenue_d14', 'iap_revenue_d28']
    
    available_targets = [col for col in target_cols if col in df.columns]
    
    if not available_targets:
        return None
    
    table = Table(
        title="🎯 Target Variable Distribution",
        box=box.ROUNDED,
        title_style="bold red",
        header_style="bold yellow",
        show_lines=True
    )
    
    table.add_column("Target", style="yellow", width=25)
    table.add_column("Type", style="cyan", width=12)
    table.add_column("Positive Rate", style="green", justify="right", width=15)
    table.add_column("Mean Value", style="magenta", justify="right", width=15)
    
    for col in available_targets:
        if col.startswith('buyer_'):
            # Binary classification target
            pos_rate = df[col].mean() * 100
            mean_val = f"{pos_rate:.2f}%"
            target_type = "Binary"
            pos_rate_str = f"{pos_rate:.2f}%"
        else:
            # Revenue target
            mean_val = df[col].mean()
            pos_rate = (df[col] > 0).mean() * 100
            target_type = "Revenue"
            pos_rate_str = f"{pos_rate:.2f}%"
            mean_val = f"${mean_val:.2f}"
        
        table.add_row(col, target_type, pos_rate_str, mean_val)
    
    return table


def main():
    """Run complete pipeline test"""
    
    # Header
    create_fancy_header(
        "🚀 SMADEX LTV PREDICTION - FULL PIPELINE TEST",
        "Load → Preprocess → Missing Value Handling → Feature Engineering"
    )
    
    # Load config
    console.print("\n[bold cyan]Loading configuration...[/bold cyan]")
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    console.print("[green]✓[/green] Configuration loaded\n")
    
    # Initialize components
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Initializing pipeline components...", total=None)
        loader = DataLoader(config)
        preprocessor = NestedFeatureParser(config)
        feature_engineer = FeatureEngineer(config)
        time.sleep(0.5)  # For effect
        progress.update(task, completed=True)
    
    console.print("[green]✓[/green] Pipeline components initialized\n")
    
    # Step 1: Load Raw Data
    console.print(Panel.fit(
        "[bold]STEP 1: LOAD RAW DATA[/bold]",
        border_style="blue"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Loading training sample (1000 rows)...", total=None)
        train_df, _ = loader.load_train_sample(n_rows=1000, validation_split=False)
        progress.update(task, completed=True)
    
    console.print(create_stats_table(train_df, "Raw Data Statistics"))
    console.print()
    
    # Step 2: Preprocess Data
    console.print(Panel.fit(
        "[bold]STEP 2: PREPROCESS & HANDLE MISSING VALUES[/bold]",
        border_style="yellow"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Processing nested features...", total=None)
        time.sleep(0.3)
        progress.update(task, description="[cyan]Handling missing values...")
        train_processed = preprocessor.process_all(
            train_df,
            handle_missing=True,
            create_missing_indicators=True,
            fit_missing_handler=True
        )
        progress.update(task, completed=True)
    
    console.print(create_stats_table(train_processed, "Preprocessed Data Statistics"))
    console.print()
    
    # Step 3: Feature Engineering
    console.print(Panel.fit(
        "[bold]STEP 3: FEATURE ENGINEERING[/bold]",
        border_style="green"
    ))
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Creating interaction features...", total=None)
        time.sleep(0.2)
        progress.update(task, description="[cyan]Encoding categorical features...")
        time.sleep(0.2)
        progress.update(task, description="[cyan]Creating temporal features...")
        
        # Determine target column
        target_col = 'iap_revenue_d7' if 'iap_revenue_d7' in train_processed.columns else None
        
        train_final = feature_engineer.engineer_all(
            train_processed,
            target_col=target_col,
            fit=True
        )
        progress.update(task, completed=True)
    
    console.print(create_stats_table(train_final, "Final Feature Set Statistics"))
    console.print()
    
    # Step 4: Show Sample Data
    console.print(Panel.fit(
        "[bold]STEP 4: FINAL DATASET SAMPLE[/bold]",
        border_style="magenta"
    ))
    
    console.print(create_sample_table(train_final, n_rows=15))
    console.print()
    
    # Step 5: Column Summary
    console.print(Panel.fit(
        "[bold]STEP 5: COLUMN DETAILS[/bold]",
        border_style="cyan"
    ))
    
    console.print(create_column_summary(train_final))
    console.print()
    
    # Step 6: Target Distribution (if available)
    target_table = create_target_distribution(train_final)
    if target_table:
        console.print(Panel.fit(
            "[bold]STEP 6: TARGET VARIABLES[/bold]",
            border_style="red"
        ))
        console.print(target_table)
        console.print()
    
    # Summary
    create_fancy_header(
        "✅ PIPELINE TEST COMPLETE",
        f"Processed {train_df.shape[0]} rows → {train_final.shape[1]} features ready for modeling"
    )
    
    # Key metrics
    console.print("\n[bold]📈 Transformation Summary:[/bold]")
    metrics_table = Table(box=box.SIMPLE, show_header=False, padding=(0, 2))
    metrics_table.add_column(style="cyan")
    metrics_table.add_column(style="white")
    
    metrics_table.add_row("Original Columns:", f"{train_df.shape[1]}")
    metrics_table.add_row("Final Columns:", f"{train_final.shape[1]}")
    metrics_table.add_row("New Features Created:", f"{train_final.shape[1] - train_df.shape[1]}")
    metrics_table.add_row("Missing Values (Before):", f"{(train_df.isnull().sum().sum() / (train_df.shape[0] * train_df.shape[1]) * 100):.2f}%")
    metrics_table.add_row("Missing Values (After):", f"{(train_final.isnull().sum().sum() / (train_final.shape[0] * train_final.shape[1]) * 100):.2f}%")
    
    console.print(metrics_table)
    console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Test interrupted by user[/yellow]")
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {e}")
        import traceback
        console.print(traceback.format_exc())
