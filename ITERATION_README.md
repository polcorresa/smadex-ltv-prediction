# 🔄 Data Iteration Workflow - Quick Start

## 📋 Overview

Manually iterate through increasing amounts of training data to find the optimal balance between performance and training time (< 5 min).

## 🚀 Quick Start

### 1. Run First Iteration (SMALL - Baseline)
```bash
uv run python scripts/test_model_validation.py config/config_test_small.yaml
```
**Expected**: ~15 seconds, RMSLE ~0.57

### 2. Run Second Iteration (MEDIUM)
```bash
uv run python scripts/test_model_validation.py config/config_test_medium.yaml
```
**Expected**: ~1-2 minutes, RMSLE improvement

### 3. Run Third Iteration (LARGE)
```bash
uv run python scripts/test_model_validation.py config/config_test_large.yaml
```
**Expected**: ~3-4 minutes, further improvement

### 4. Run Fourth Iteration (XLARGE - Maximum)
```bash
uv run python scripts/test_model_validation.py config/config_test_xlarge.yaml
```
**Expected**: ~4-5 minutes, best performance

### 5. Compare Results
```bash
uv run python scripts/compare_results.py
```
See all metrics side-by-side with improvement percentages!

## 📊 Configuration Summary

| Config | Time Range | Sample % | Estimators | Est. Time |
|--------|-----------|----------|------------|-----------|
| **SMALL** | 3h train, 1h val | 10% / 20% | 50 | ~15s |
| **MEDIUM** | 12h train, 3h val | 15% / 30% | 100 | ~1-2m |
| **LARGE** | 24h train, 6h val | 20% / 40% | 150 | ~3-4m |
| **XLARGE** | 48h train, 12h val | 25% / 50% | 200 | ~4-5m |

## 📈 Example Output

```bash
$ uv run python scripts/compare_results.py

================================================================================
📊 MODEL VALIDATION RESULTS COMPARISON
================================================================================

Config     RMSLE      Improv     AUC        Time       Train Rows   Val Rows    
--------------------------------------------------------------------------------
SMALL      0.5721     baseline   0.9950     0.23m      24,688       25,131      
MEDIUM     0.5234     +8.5%      0.9965     1.45m      98,456       75,320      
LARGE      0.4891     +14.5%     0.9972     3.12m      195,234      156,789     
XLARGE     0.4756     +16.9%     0.9978     4.67m      387,921      298,456     

================================================================================

📈 INSIGHTS:

   🏆 Best RMSLE: 0.4756 (XLARGE)
   ⚡ Fastest: 0.23 min (SMALL)
   📊 Average improvement per step: 5.3%
   ⚠️  Diminishing returns detected! Consider stopping here.

================================================================================
```

## 🔧 Customizing Between Iterations

If a config is too fast or too slow, edit the YAML file:

**To increase data** (if < 2 min):
```yaml
training:
  sampling:
    train_frac: 0.2  # Increase
```

**To decrease data** (if > 5 min):
```yaml
training:
  sampling:
    train_frac: 0.1  # Decrease
```

## 🎲 Randomized Train/Validation Split

Smaller temporal windows could starve the buyer-only regressor, so every test config now enables a stratified random split inside the combined modeling window. Inspect the new `training.split` block:

```yaml
training:
  split:
    strategy: "stratified_random"  # switch back to "temporal" to disable
    model_start: "2025-10-01-00-00"  # window that feeds both splits
    model_end: "2025-10-01-03-00"
    train_fraction: 0.8             # 80% rows for training
    stratify_column: "buyer_d7"     # keeps buyer mix balanced
```

- The loader first gathers every row between `model_start` and `model_end`.
- Rows are shuffled (seeded) and split with `sklearn.train_test_split`.
- Stratification keeps the buyer rate similar in train and validation, giving the revenue regressor enough positive examples.
- Set `strategy: "temporal"` if you specifically need chronological holdouts.

## 📁 Files Created

```
config/
├── config_test_small.yaml     ← Baseline
├── config_test_medium.yaml    ← 4x more data
├── config_test_large.yaml     ← 8x more data
└── config_test_xlarge.yaml    ← 16x more data

scripts/
├── test_model_validation.py   ← Main validation script
└── compare_results.py         ← Results comparison tool

logs/
├── validation_config_test_small.log
├── validation_config_test_medium.log
├── validation_config_test_large.log
└── validation_config_test_xlarge.log
```

## 💡 Tips

1. ✅ **Always start with SMALL** to verify everything works
2. ⏱️ **Check time after each run** - stop if > 5 minutes
3. 📊 **Run compare_results.py** after each iteration to track progress
4. 🎯 **Look for plateau** - if improvement < 2%, you've found your limit
5. 💾 **Logs are saved** - you can always review past runs

## 🎓 What You'll Learn

- How much data do you actually need?
- Where are the diminishing returns?
- What's the optimal time/performance trade-off?
- Which hyperparameters work best for your data size?

## 🆘 Troubleshooting

**"Out of memory"**
→ Reduce `train_frac` in the config file

**"Takes too long"**
→ Reduce `n_estimators` or `train_frac`

**"RMSLE got worse"**
→ Might be overfitting or bad config, check logs

**"Can't find logs"**
→ They're in `logs/validation_config_test_*.log`

## 📚 Full Documentation

See `docs/ITERATION_GUIDE.md` for detailed protocol and tracking templates.

---

**Ready?** Start with: `uv run python scripts/test_model_validation.py config/config_test_small.yaml` 🚀
