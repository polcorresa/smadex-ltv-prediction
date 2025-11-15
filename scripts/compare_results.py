#!/usr/bin/env python3
"""
Quick Results Comparison
Extracts and compares key metrics from validation logs

Usage: uv run python scripts/compare_results.py
"""
import re
from pathlib import Path
from typing import Dict, Optional


def extract_metrics_from_log(log_file: Path) -> Optional[Dict]:
    """Extract key metrics from a validation log file"""
    if not log_file.exists():
        return None
    
    metrics = {}
    
    with open(log_file, 'r') as f:
        content = f.read()
        
        # Extract validation RMSLE (Full Pipeline)
        match = re.search(r'🎭 FULL PIPELINE.*?Validation RMSLE:\s+([\d.]+)', content, re.DOTALL)
        if match:
            metrics['rmsle'] = float(match.group(1))
        
        # Extract validation AUC (Stage 1)
        match = re.search(r'🎯 STAGE 1.*?Validation AUC:\s+([\d.]+)', content, re.DOTALL)
        if match:
            metrics['auc'] = float(match.group(1))
        
        # Extract execution time
        match = re.search(r'Total execution time:\s+([\d.]+)\s+minutes', content)
        if match:
            metrics['time_min'] = float(match.group(1))
        
        # Extract data sizes
        match = re.search(r'TRAINING SET.*?Rows:\s+([\d,]+)', content, re.DOTALL)
        if match:
            metrics['train_rows'] = int(match.group(1).replace(',', ''))
        
        match = re.search(r'VALIDATION SET.*?Rows:\s+([\d,]+)', content, re.DOTALL)
        if match:
            metrics['val_rows'] = int(match.group(1).replace(',', ''))
        
        # Extract buyer rate
        match = re.search(r'VALIDATION SET.*?Buyers \(D7\):\s+[\d,]+\s+\(([\d.]+)%\)', content, re.DOTALL)
        if match:
            metrics['buyer_rate'] = float(match.group(1))
    
    return metrics if metrics else None


def format_number(val: float, decimals: int = 3) -> str:
    """Format number with fixed decimals"""
    if val is None:
        return "N/A"
    return f"{val:.{decimals}f}"


def format_int(val: int) -> str:
    """Format integer with commas"""
    if val is None:
        return "N/A"
    return f"{val:,}"


def calculate_improvement(current: float, baseline: float) -> str:
    """Calculate improvement percentage"""
    if baseline is None or current is None:
        return "N/A"
    improvement = ((baseline - current) / baseline) * 100
    sign = "+" if improvement > 0 else ""
    return f"{sign}{improvement:.1f}%"


def main():
    print("\n" + "=" * 80)
    print("📊 MODEL VALIDATION RESULTS COMPARISON")
    print("=" * 80 + "\n")
    
    configs = ['small', 'medium', 'large', 'xlarge']
    results = {}
    
    # Extract metrics from each log
    for config in configs:
        log_file = Path(f'logs/validation_config_test_{config}.log')
        results[config] = extract_metrics_from_log(log_file)
    
    # Check if we have any results
    if not any(results.values()):
        print("❌ No validation logs found!")
        print("   Run: uv run python scripts/test_model_validation.py config/config_test_small.yaml")
        return
    
    # Print header
    print(f"{'Config':<10} {'RMSLE':<10} {'Improv':<10} {'AUC':<10} {'Time':<10} {'Train Rows':<12} {'Val Rows':<12}")
    print("-" * 80)
    
    # Get baseline RMSLE
    baseline_rmsle = None
    for config in configs:
        if results[config] and 'rmsle' in results[config]:
            baseline_rmsle = results[config]['rmsle']
            break
    
    # Print results
    for config in configs:
        metrics = results[config]
        
        if metrics is None:
            print(f"{config.upper():<10} {'Not run yet':<50}")
            continue
        
        rmsle = format_number(metrics.get('rmsle'), 4)
        improvement = calculate_improvement(metrics.get('rmsle'), baseline_rmsle)
        auc = format_number(metrics.get('auc'), 4)
        time_str = f"{format_number(metrics.get('time_min'), 2)}m" if metrics.get('time_min') else "N/A"
        train_rows = format_int(metrics.get('train_rows'))
        val_rows = format_int(metrics.get('val_rows'))
        
        print(f"{config.upper():<10} {rmsle:<10} {improvement:<10} {auc:<10} {time_str:<10} {train_rows:<12} {val_rows:<12}")
    
    print("\n" + "=" * 80)
    
    # Print detailed comparison if we have multiple results
    run_configs = [c for c in configs if results[c] is not None]
    
    if len(run_configs) >= 2:
        print("\n📈 INSIGHTS:\n")
        
        # Find best RMSLE
        best_config = min(run_configs, key=lambda c: results[c].get('rmsle', float('inf')))
        best_rmsle = results[best_config]['rmsle']
        print(f"   🏆 Best RMSLE: {best_rmsle:.4f} ({best_config.upper()})")
        
        # Find fastest
        fastest_config = min(run_configs, key=lambda c: results[c].get('time_min', float('inf')))
        fastest_time = results[fastest_config]['time_min']
        print(f"   ⚡ Fastest: {fastest_time:.2f} min ({fastest_config.upper()})")
        
        # Check for diminishing returns
        if len(run_configs) >= 3:
            improvements = []
            for i in range(1, len(run_configs)):
                curr_config = run_configs[i]
                prev_config = run_configs[i-1]
                
                curr_rmsle = results[curr_config].get('rmsle')
                prev_rmsle = results[prev_config].get('rmsle')
                
                if curr_rmsle and prev_rmsle:
                    improv = ((prev_rmsle - curr_rmsle) / prev_rmsle) * 100
                    improvements.append(improv)
            
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                print(f"   📊 Average improvement per step: {avg_improvement:.1f}%")
                
                if len(improvements) >= 2 and improvements[-1] < improvements[0] / 2:
                    print(f"   ⚠️  Diminishing returns detected! Consider stopping here.")
        
        # Time warning
        slowest_time = max(results[c].get('time_min', 0) for c in run_configs)
        if slowest_time > 5:
            print(f"   ⚠️  WARNING: Slowest config exceeded 5 min limit ({slowest_time:.1f} min)")
        
    print("\n" + "=" * 80)
    print("\n💡 TIP: Run next config with:")
    
    # Suggest next config to run
    for config in configs:
        if results[config] is None:
            print(f"   uv run python scripts/test_model_validation.py config/config_test_{config}.yaml")
            break
    else:
        print("   All configs completed! ✅")
    
    print("")


if __name__ == '__main__':
    main()
