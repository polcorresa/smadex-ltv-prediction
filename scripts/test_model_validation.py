"""
Model Validation Test Script
=============================
Train on a small subset of data and validate on a different subset.
Measures model performance with comprehensive metrics and fancy logging.

Usage: 
    uv run python scripts/test_model_validation.py [config_file]
    
Examples:
    uv run python scripts/test_model_validation.py config/config_test_small.yaml
    uv run python scripts/test_model_validation.py config/config_test_medium.yaml
    uv run python scripts/test_model_validation.py config/config_test_large.yaml
    uv run python scripts/test_model_validation.py config/config_test_xlarge.yaml
"""
import sys
from pathlib import Path
import time
import numpy as np
import pandas as pd
from dataclasses import asdict

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import setup_logger, log_section, log_subsection, log_metric
from src.utils.metrics import evaluate_predictions
from src.training.trainer import TrainingPipeline
from src.models.buyer_classifier import BuyerClassifier
from src.models.revenue_regressor import ODMNRevenueRegressor
from src.models.ensemble import StackingEnsemble


def print_header(logger):
    """Print fancy header"""
    logger.info("")
    logger.info("╔" + "═" * 78 + "╗")
    logger.info("║" + " " * 78 + "║")
    logger.info("║" + "  🚀 SMADEX LTV PREDICTION - MODEL VALIDATION TEST  ".center(78) + "║")
    logger.info("║" + " " * 78 + "║")
    logger.info("╚" + "═" * 78 + "╝")
    logger.info("")


def print_data_info(logger, train_df, val_df):
    """Print detailed data information"""
    log_section(logger, "📊 DATASET INFORMATION")
    
    # Training data
    logger.info("")
    logger.info("  📚 TRAINING SET")
    logger.info(f"     • Rows: {len(train_df):,}")
    logger.info(f"     • Columns: {len(train_df.columns):,}")
    logger.info(f"     • Memory: {train_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if 'buyer_d7' in train_df.columns:
        buyer_count = train_df['buyer_d7'].sum()
        buyer_rate = train_df['buyer_d7'].mean() * 100
        logger.info(f"     • Buyers (D7): {buyer_count:,} ({buyer_rate:.2f}%)")
    
    if 'iap_revenue_d7' in train_df.columns:
        revenue_stats = train_df['iap_revenue_d7'].describe()
        logger.info(f"     • Avg Revenue (D7): ${revenue_stats['mean']:.4f}")
        logger.info(f"     • Max Revenue (D7): ${revenue_stats['max']:.4f}")
        non_zero = (train_df['iap_revenue_d7'] > 0).sum()
        logger.info(f"     • Non-zero revenue: {non_zero:,} ({non_zero/len(train_df)*100:.2f}%)")
    
    # Validation data
    logger.info("")
    logger.info("  🎯 VALIDATION SET")
    logger.info(f"     • Rows: {len(val_df):,}")
    logger.info(f"     • Columns: {len(val_df.columns):,}")
    logger.info(f"     • Memory: {val_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    if 'buyer_d7' in val_df.columns:
        buyer_count = val_df['buyer_d7'].sum()
        buyer_rate = val_df['buyer_d7'].mean() * 100
        logger.info(f"     • Buyers (D7): {buyer_count:,} ({buyer_rate:.2f}%)")
    
    if 'iap_revenue_d7' in val_df.columns:
        revenue_stats = val_df['iap_revenue_d7'].describe()
        logger.info(f"     • Avg Revenue (D7): ${revenue_stats['mean']:.4f}")
        logger.info(f"     • Max Revenue (D7): ${revenue_stats['max']:.4f}")
        non_zero = (val_df['iap_revenue_d7'] > 0).sum()
        logger.info(f"     • Non-zero revenue: {non_zero:,} ({non_zero/len(val_df)*100:.2f}%)")
    
    logger.info("")
    logger.info("=" * 80)


def evaluate_stage1(logger, pipeline, train_df, val_df, feature_cols):
    """Evaluate Stage 1: Buyer Classification"""
    log_section(logger, "🎯 STAGE 1: BUYER CLASSIFICATION EVALUATION")
    
    start_time = time.time()
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['buyer_d7'].astype(int).values
    
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['buyer_d7'].astype(int).values
    
    # Get predictions
    logger.info("")
    logger.info("  🔮 Generating predictions...")
    y_train_pred_proba = pipeline.buyer_model.predict_proba(X_train)
    y_val_pred_proba = pipeline.buyer_model.predict_proba(X_val)
    
    y_train_pred = (y_train_pred_proba > 0.5).astype(int)
    y_val_pred = (y_val_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    logger.info("")
    log_subsection(logger, "📈 TRAINING SET METRICS")
    log_metric(logger, "Accuracy", accuracy_score(y_train, y_train_pred), ".4f")
    log_metric(logger, "Precision", precision_score(y_train, y_train_pred, zero_division=0), ".4f")
    log_metric(logger, "Recall", recall_score(y_train, y_train_pred, zero_division=0), ".4f")
    log_metric(logger, "F1 Score", f1_score(y_train, y_train_pred, zero_division=0), ".4f")
    log_metric(logger, "ROC AUC", roc_auc_score(y_train, y_train_pred_proba), ".4f")
    
    logger.info("")
    log_subsection(logger, "📊 VALIDATION SET METRICS")
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred, zero_division=0)
    val_recall = recall_score(y_val, y_val_pred, zero_division=0)
    val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
    val_auc = roc_auc_score(y_val, y_val_pred_proba)
    
    log_metric(logger, "Accuracy", val_accuracy, ".4f")
    log_metric(logger, "Precision", val_precision, ".4f")
    log_metric(logger, "Recall", val_recall, ".4f")
    log_metric(logger, "F1 Score", val_f1, ".4f")
    log_metric(logger, "ROC AUC", val_auc, ".4f")
    
    elapsed = time.time() - start_time
    logger.info("")
    logger.info(f"  ⏱️  Stage 1 evaluation completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    
    return {
        'train': {
            'accuracy': accuracy_score(y_train, y_train_pred),
            'auc': roc_auc_score(y_train, y_train_pred_proba)
        },
        'val': {
            'accuracy': val_accuracy,
            'auc': val_auc
        }
    }


def evaluate_stage2(logger, pipeline, train_df, val_df, feature_cols):
    """Evaluate Stage 2: Revenue Regression"""
    log_section(logger, "💰 STAGE 2: REVENUE REGRESSION EVALUATION")
    
    start_time = time.time()
    
    # Filter to buyers only
    train_buyers = train_df[train_df['buyer_d7'] == 1].copy()
    val_buyers = val_df[val_df['buyer_d7'] == 1].copy()
    
    logger.info("")
    logger.info(f"  📊 Training buyers: {len(train_buyers):,}")
    logger.info(f"  📊 Validation buyers: {len(val_buyers):,}")
    
    if len(train_buyers) == 0 or len(val_buyers) == 0:
        logger.warning("")
        logger.warning("  ⚠️  Not enough buyers for revenue evaluation!")
        logger.info("=" * 80)
        return None
    
    # Prepare data
    X_train = train_buyers[feature_cols].fillna(0)
    y_train = train_buyers['iap_revenue_d7'].values
    
    X_val = val_buyers[feature_cols].fillna(0)
    y_val = val_buyers['iap_revenue_d7'].values
    
    # Get predictions
    logger.info("")
    logger.info("  🔮 Generating revenue predictions...")
    train_preds = pipeline.revenue_model.predict(X_train, enforce_order=True)
    val_preds = pipeline.revenue_model.predict(X_val, enforce_order=True)
    
    # Evaluate D7 predictions
    y_train_pred = train_preds.d7
    y_val_pred = val_preds.d7
    
    logger.info("")
    log_subsection(logger, "📈 TRAINING SET METRICS (D7 Revenue)")
    train_metrics = evaluate_predictions(y_train, y_train_pred)
    for metric, value in asdict(train_metrics).items():
        log_metric(logger, metric.upper(), value, ".6f")
    
    logger.info("")
    log_subsection(logger, "📊 VALIDATION SET METRICS (D7 Revenue)")
    val_metrics = evaluate_predictions(y_val, y_val_pred)
    for metric, value in asdict(val_metrics).items():
        log_metric(logger, metric.upper(), value, ".6f")
    
    # Additional statistics
    logger.info("")
    log_subsection(logger, "📉 PREDICTION STATISTICS")
    logger.info(f"  Training:")
    logger.info(f"     • Mean actual: ${y_train.mean():.4f}")
    logger.info(f"     • Mean predicted: ${y_train_pred.mean():.4f}")
    logger.info(f"     • Std actual: ${y_train.std():.4f}")
    logger.info(f"     • Std predicted: ${y_train_pred.std():.4f}")
    
    logger.info(f"  Validation:")
    logger.info(f"     • Mean actual: ${y_val.mean():.4f}")
    logger.info(f"     • Mean predicted: ${y_val_pred.mean():.4f}")
    logger.info(f"     • Std actual: ${y_val.std():.4f}")
    logger.info(f"     • Std predicted: ${y_val_pred.std():.4f}")
    
    elapsed = time.time() - start_time
    logger.info("")
    logger.info(f"  ⏱️  Stage 2 evaluation completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    
    return {
        'train': train_metrics,
        'val': val_metrics
    }


def evaluate_full_pipeline(logger, pipeline, train_df, val_df, feature_cols):
    """Evaluate full pipeline (buyer classification + revenue regression)"""
    log_section(logger, "🎭 FULL PIPELINE EVALUATION")
    
    start_time = time.time()
    
    # Prepare data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df['iap_revenue_d7'].values
    
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df['iap_revenue_d7'].values
    
    # Get predictions from full pipeline
    logger.info("")
    logger.info("  🔮 Generating full pipeline predictions...")
    
    # Stage 1: Buyer probability
    buyer_proba_train = pipeline.buyer_model.predict_proba(X_train)
    buyer_proba_val = pipeline.buyer_model.predict_proba(X_val)
    
    # Stage 2: Revenue predictions
    revenue_preds_train = pipeline.revenue_model.predict(X_train, enforce_order=True)
    revenue_preds_val = pipeline.revenue_model.predict(X_val, enforce_order=True)
    
    # Combine: buyer_proba * revenue
    y_train_pred = buyer_proba_train * revenue_preds_train.d7
    y_val_pred = buyer_proba_val * revenue_preds_val.d7
    
    logger.info("")
    log_subsection(logger, "📈 TRAINING SET METRICS (Full Pipeline)")
    train_metrics = evaluate_predictions(y_train, y_train_pred)
    for metric, value in asdict(train_metrics).items():
        log_metric(logger, metric.upper(), value, ".6f")
    
    logger.info("")
    log_subsection(logger, "📊 VALIDATION SET METRICS (Full Pipeline)")
    val_metrics = evaluate_predictions(y_val, y_val_pred)
    for metric, value in asdict(val_metrics).items():
        log_metric(logger, metric.upper(), value, ".6f")
    
    # Prediction distribution
    logger.info("")
    log_subsection(logger, "📉 PREDICTION DISTRIBUTION")
    
    logger.info("  Training:")
    logger.info(f"     • Predicted zeros: {(y_train_pred == 0).sum():,} ({(y_train_pred == 0).mean() * 100:.2f}%)")
    logger.info(f"     • Predicted > 0: {(y_train_pred > 0).sum():,} ({(y_train_pred > 0).mean() * 100:.2f}%)")
    logger.info(f"     • Mean prediction: ${y_train_pred.mean():.4f}")
    logger.info(f"     • Median prediction: ${np.median(y_train_pred):.4f}")
    logger.info(f"     • Max prediction: ${y_train_pred.max():.4f}")
    
    logger.info("  Validation:")
    logger.info(f"     • Predicted zeros: {(y_val_pred == 0).sum():,} ({(y_val_pred == 0).mean() * 100:.2f}%)")
    logger.info(f"     • Predicted > 0: {(y_val_pred > 0).sum():,} ({(y_val_pred > 0).mean() * 100:.2f}%)")
    logger.info(f"     • Mean prediction: ${y_val_pred.mean():.4f}")
    logger.info(f"     • Median prediction: ${np.median(y_val_pred):.4f}")
    logger.info(f"     • Max prediction: ${y_val_pred.max():.4f}")
    
    # Actual distribution for comparison
    logger.info("")
    logger.info("  Actual values (for comparison):")
    logger.info(f"     • Training zeros: {(y_train == 0).sum():,} ({(y_train == 0).mean() * 100:.2f}%)")
    logger.info(f"     • Validation zeros: {(y_val == 0).sum():,} ({(y_val == 0).mean() * 100:.2f}%)")
    
    elapsed = time.time() - start_time
    logger.info("")
    logger.info(f"  ⏱️  Full pipeline evaluation completed in {elapsed:.2f} seconds")
    logger.info("=" * 80)
    
    return {
        'train': train_metrics,
        'val': val_metrics
    }


def print_summary(logger, results, total_time):
    """Print final summary"""
    log_section(logger, "📋 VALIDATION SUMMARY")
    
    logger.info("")
    logger.info("  🎯 STAGE 1: BUYER CLASSIFICATION")
    if 'stage1' in results and results['stage1']:
        logger.info(f"     Training AUC:    {results['stage1']['train']['auc']:.4f}")
        logger.info(f"     Validation AUC:  {results['stage1']['val']['auc']:.4f}")
        logger.info(f"     Training Acc:    {results['stage1']['train']['accuracy']:.4f}")
        logger.info(f"     Validation Acc:  {results['stage1']['val']['accuracy']:.4f}")
    
    logger.info("")
    logger.info("  💰 STAGE 2: REVENUE REGRESSION")
    if 'stage2' in results and results['stage2']:
        logger.info(f"     Training RMSLE:     {results['stage2']['train'].rmsle:.6f}")
        logger.info(f"     Validation RMSLE:   {results['stage2']['val'].rmsle:.6f}")
        logger.info(f"     Training RMSE:      {results['stage2']['train'].rmse:.6f}")
        logger.info(f"     Validation RMSE:    {results['stage2']['val'].rmse:.6f}")
    else:
        logger.info(f"     ⚠️  Insufficient data for evaluation")
    
    logger.info("")
    logger.info("  🎭 FULL PIPELINE")
    if 'full_pipeline' in results:
        logger.info(f"     Training RMSLE:     {results['full_pipeline']['train'].rmsle:.6f}")
        logger.info(f"     Validation RMSLE:   {results['full_pipeline']['val'].rmsle:.6f}")
        logger.info(f"     Training RMSE:      {results['full_pipeline']['train'].rmse:.6f}")
        logger.info(f"     Validation RMSE:    {results['full_pipeline']['val'].rmse:.6f}")
        logger.info(f"     Validation R²:      {results['full_pipeline']['val'].r2:.6f}")
    
    logger.info("")
    logger.info(f"  ⏱️  Total execution time: {total_time / 60:.2f} minutes")
    logger.info("")
    logger.info("  ✅ VALIDATION TEST COMPLETE!")
    logger.info("")
    logger.info("=" * 80)


def main():
    """Main execution"""
    total_start = time.time()
    
    # Get config file from command line or use default
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'config/config_test.yaml'
    
    # Setup logging with config-specific log file
    config_name = Path(config_file).stem
    log_file = f'logs/validation_{config_name}.log'
    logger = setup_logger('validation_test', log_file)
    
    print_header(logger)
    
    try:
        # Initialize pipeline
        log_section(logger, "🔧 INITIALIZATION")
        logger.info("")
        logger.info(f"  📝 Loading configuration: {config_file}")
        pipeline = TrainingPipeline(config_file)
        logger.info("  ✅ Pipeline initialized successfully")
        logger.info("")
        logger.info("=" * 80)
        
        # Step 1: Prepare data
        logger.info("")
        train_df, val_df = pipeline.prepare_data()
        
        # Print data info
        print_data_info(logger, train_df, val_df)
        
        # Step 2: Train models
        log_section(logger, "🏋️  MODEL TRAINING")
        logger.info("")
        
        train_start = time.time()
        
        logger.info("  🎯 Training Stage 1: Buyer Classifier...")
        pipeline.train_stage1_buyer(train_df, val_df)
        logger.info("  ✅ Stage 1 complete")
        logger.info("")
        
        logger.info("  💰 Training Stage 2: Revenue Regressor...")
        pipeline.train_stage2_revenue(train_df, val_df)
        logger.info("  ✅ Stage 2 complete")
        logger.info("")
        
        train_elapsed = time.time() - train_start
        logger.info(f"  ⏱️  Training completed in {train_elapsed:.2f} seconds")
        logger.info("=" * 80)
        
        # Step 3: Evaluate models
        logger.info("")
        
        # Get feature columns (numeric only, excluding targets)
        excluded_cols = {
            'row_id', 'datetime', 'iap_revenue_d7', 'iap_revenue_d1', 'iap_revenue_d14',
            'iap_revenue_d28', 'buyer_d1', 'buyer_d7', 'buyer_d14', 'buyer_d28',
            'buy_d7', 'buy_d14', 'buy_d28', 'registration',
            'retention_d1', 'retention_d7', 'retention_d14'
        }
        feature_cols = [col for col in train_df.columns if col not in excluded_cols]
        feature_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"  📊 Using {len(feature_cols)} features for evaluation")
        logger.info("")
        
        # Store results
        results = {}
        
        # Evaluate Stage 1
        results['stage1'] = evaluate_stage1(logger, pipeline, train_df, val_df, feature_cols)
        logger.info("")
        
        # Evaluate Stage 2
        results['stage2'] = evaluate_stage2(logger, pipeline, train_df, val_df, feature_cols)
        logger.info("")
        
        # Evaluate full pipeline
        results['full_pipeline'] = evaluate_full_pipeline(logger, pipeline, train_df, val_df, feature_cols)
        logger.info("")
        
        # Print final summary
        total_time = time.time() - total_start
        print_summary(logger, results, total_time)
        
    except Exception as e:
        logger.error("")
        logger.error("=" * 80)
        logger.error(f"  ❌ ERROR: {str(e)}")
        logger.error("=" * 80)
        import traceback
        logger.error(traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
