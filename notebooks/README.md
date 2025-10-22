# Training Pipeline Notebooks

This directory contains Jupyter notebooks for the Trino Query Predictor training pipeline.

## Overview

The notebooks implement an end-to-end ML training pipeline from data loading through model deployment. They integrate production code from `query_predictor/` to ensure train-serve parity.

## Notebooks

### 00_setup_and_config.ipynb
**Purpose**: Environment setup and code packaging

**Tasks**:
- Validate Python and S3 access
- Load training configuration
- Package `query_predictor` module for Spark
- Upload to S3 with versioning
- Generate Spark configuration for other notebooks

**Duration**: ~5 minutes
**Required**: Yes (run first)

---

### 01_data_loading.ipynb
**Purpose**: Data loading, labeling, filtering, and sampling

**Pipeline**:
1. Load raw Trino logs from S3 parquet
2. Apply labeling (CPU ≥300s OR Memory >10GB OR resource errors)
3. Apply filters (NON-CURATED, query types, dedup)
4. Apply boundary-focused sampling (5:1 small:heavy ratio)
5. Save processed data to S3

**Key Features**:
- Continuous distance-based boundary sampling (POC v2.1.0)
- S3 checkpointing for fault tolerance
- Optional analysis at each stage
- Target 5:1 class balance

**Duration**: ~30-45 minutes (full dataset)
**Output**: Processed parquet dataset partitioned by `is_heavy`

---

### 02_zero_cost_analysis.ipynb
**Purpose**: Validate Stage 0 zero-cost SQL categories

**Tasks**:
- Analyze SQL categories using production `QueryClassifier`
- Validate <1% heavy rate for safe categories
- Ensure no heavy queries incorrectly filtered
- Save validated categories for inference

**Duration**: ~15 minutes
**Optional**: Yes (recommended for production deployment)

---

### 03_feature_engineering.ipynb
**Purpose**: Extract 78 + 17 + 1000 features

**Pipeline**:
1. Time-based splits (30 train / 7 val / 7 test days)
2. Extract 78 base features (production `FeatureExtractor`)
3. Extract 17 historical features (cold-start handling)
4. Build TF-IDF vocabulary (training data only, 500-1000 terms)
5. Extract TF-IDF features for all splits
6. Validate feature parity (training vs inference)
7. Save feature datasets to S3

**Duration**: ~45-60 minutes
**Output**: Feature datasets (train/val/test) + TF-IDF vectorizer

---

### 04_model_training.ipynb
**Purpose**: XGBoost training with ONNX export

**Pipeline**:
1. Load feature datasets
2. Convert to NumPy arrays
3. 5-fold stratified cross-validation
4. Train XGBoost with early stopping
5. Optimize threshold (100:1 FN:FP cost ratio)
6. Evaluate on test set
7. Check PRD compliance (recall ≥98%, FNR ≤2%)
8. Export to ONNX with validation
9. Save model artifacts to S3

**Duration**: ~30-45 minutes
**Output**: ONNX model + XGBoost model + metrics + metadata

---

### 05_offline_replay.ipynb (PLANNED)
**Purpose**: Historical validation on recent data

**Tasks**:
- Load trained model
- Generate predictions on last 14 days
- Calculate production-like metrics
- Detect data/model drift
- Compare with online metrics

**Duration**: ~20 minutes
**Optional**: Yes (recommended for production monitoring)

---

## Quick Start

### Prerequisites
- AWS credentials configured (`~/.aws/credentials`)
- S3 bucket access: `uip-datalake-bucket-prod`
- EMR cluster or SageMaker Studio with PySpark kernel
- Python 3.8+

### Setup

1. **Configure training parameters** (optional):
   ```bash
   vim ../config/training_config.yaml
   # Adjust dates, thresholds, ratios as needed
   ```

2. **Run notebook 00** (setup):
   - Open `00_setup_and_config.ipynb`
   - Run all cells
   - Copy the Spark configuration from output

3. **Run notebook 01** (data loading):
   - Open `01_data_loading.ipynb`
   - Paste Spark configuration in first cell
   - Run all cells
   - Wait ~30-45 minutes

4. **Run remaining notebooks** (when implemented):
   - 02 (zero-cost analysis) - optional
   - 03 (feature engineering) - required
   - 04 (model training) - required
   - 05 (offline replay) - optional

---

## Configuration

All notebooks read from `../config/training_config.yaml`:

### Key Parameters

```yaml
# Data Loading
data_loading:
  start_date: "2025-06-01"
  end_date: "2025-07-15"
  cpu_threshold_seconds: 300      # Heavy if ≥5 minutes
  memory_threshold_gb: 10         # Heavy if >10 GB

# Boundary Sampling
boundary_sampling:
  balance_ratio: 5.0              # 5:1 small:heavy
  max_boost: 2.0                  # Max sampling at boundary
  min_multiplier: 0.05            # Min sampling far from boundary

# Features
features:
  base_feature_count: 78          # Production extractors
  historical_feature_count: 17    # Cold-start features
  tfidf_vocab_size: 1000          # 500 or 1000

# Model
model:
  algorithm: xgboost
  n_estimators: 100
  cv_folds: 5
  cost_fn: 100.0                  # Cost of missing heavy query
  cost_fp: 1.0                    # Cost of over-routing
```

---

## Architecture

### Code Organization

```
trino-query-predictor/
├── notebooks/                   # Training notebooks (THIS DIRECTORY)
│   ├── 00_setup_and_config.ipynb
│   ├── 01_data_loading.ipynb
│   └── README.md
│
├── query_predictor/
│   ├── core/                    # Production code (shared)
│   │   ├── featurizer/          # 78 base features
│   │   ├── types/               # Data models
│   │   └── classifier/          # Zero-cost filter
│   │
│   ├── training/                # Training-specific modules
│   │   ├── data_loader.py       # ✅ IMPLEMENTED
│   │   ├── checkpoint_manager.py # ✅ IMPLEMENTED
│   │   ├── boundary_sampler.py  # ✅ IMPLEMENTED
│   │   ├── dataframe_analyzer.py # ✅ IMPLEMENTED
│   │   ├── tfidf_pipeline.py    # ✅ IMPLEMENTED
│   │   ├── parity_validator.py  # ✅ IMPLEMENTED
│   │   ├── model_trainer.py     # ✅ IMPLEMENTED
│   │   ├── prd_checker.py       # ✅ IMPLEMENTED
│   │   └── onnx_validator.py    # ✅ IMPLEMENTED
│   │
│   └── service/                 # Inference service
│
└── config/
    ├── service_config.yaml      # Service config
    └── training_config.yaml     # Training config (✅ IMPLEMENTED)
```

### Key Principles

1. **Shared Featurization**: Training uses same `FeatureExtractor` as inference
2. **No Data Leakage**: TF-IDF vocabulary built only on training data
3. **Time-based Splits**: Mimics production deployment (past → future)
4. **Fault Tolerance**: S3 checkpointing for long-running jobs
5. **Feature Parity**: Automated validation of training vs inference features

---

## Implementation Status

### Phase 1: Foundation ✅ COMPLETE
- [x] Training modules (data_loader, checkpoint_manager, boundary_sampler, dataframe_analyzer)
- [x] Configuration file (training_config.yaml)
- [x] Notebook 00 (setup)
- [x] Notebook 01 (data loading)
- [x] Notebooks README

### Phase 2: Features ✅ COMPLETE
- [x] TF-IDF pipeline module
- [x] Parity validator module
- [x] Unit tests for TF-IDF and parity validator
- [x] Notebook 02 (zero-cost analysis)
- [x] Notebook 03 (feature engineering)

### Phase 3: Training ✅ COMPLETE
- [x] Model trainer module
- [x] PRD checker module
- [x] ONNX validator module
- [x] Notebook 04 (model training)

### Phase 4: Testing & Docs (PLANNED)
- [ ] Unit tests for model training modules
- [ ] Integration tests
- [ ] Notebook 05 (offline replay)
- [ ] Training guide documentation

---

## Tips

### Performance
- Set `analysis: enabled: false` in config for faster execution
- Use `sample_fraction: 0.1` for quick testing (10% of data)
- Enable `checkpointing` to resume from failures

### Debugging
- Enable analysis mode: `config['analysis']['enabled'] = True`
- Check checkpoints: `checkpoint_mgr.list_checkpoints()`
- View sampling stats: `sampler.get_sampling_stats(df)`

### S3 Paths
All outputs go to: `s3://uip-datalake-bucket-prod/sf_trino/trino_query_predictor/`
- Processed data: `processed_data/{date_range}/`
- Checkpoints: `checkpoints/`
- Features: `features/`
- Models: `models/`

---

## Troubleshooting

### Import Errors
```python
# If "No module named 'query_predictor'" error:
# 1. Check notebook 00 ran successfully
# 2. Verify S3 upload completed
# 3. Check Spark config has correct pyFiles path
```

### Memory Errors
```yaml
# Increase Spark resources in training_config.yaml:
spark:
  driver_memory: 32G    # Increase from 16G
  executor_memory: 40G  # Increase from 20G
```

### S3 Access Errors
```bash
# Check AWS credentials:
aws s3 ls s3://uip-datalake-bucket-prod/sf_trino/trino_query_predictor/

# If permission denied, ensure IAM role has S3 access
```

---

## Next Steps After Phase 1

1. **Validate Data Loading**:
   - Run notebook 01 end-to-end
   - Verify output in S3
   - Check class balance achieved (5:1)

2. **Implement Phase 2** (Features):
   - Create `tfidf_pipeline.py`
   - Create `parity_validator.py`
   - Implement notebooks 02-03

3. **Implement Phase 3** (Training):
   - Create `model_trainer.py`
   - Create `prd_checker.py`
   - Create `onnx_validator.py`
   - Implement notebook 04

---

## Contact

For questions or issues:
- See main repository README
- Check implementation plan document
- Review POC notebooks in `/Users/pmannem/workspace/query-prediction/qp/code/notebooks/`
