# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**trino-query-predictor** is a production ML service that classifies Trino SQL queries as "small" or "heavy" for smart cluster routing. The system uses a two-stage approach: zero-cost filtering (Stage 0) for simple queries and ML inference (Stage 1) for complex queries.

## Development Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package in development mode
pip install -e . --no-deps

# Install core dependencies (lightweight, no cmake required)
pip install Flask flask-cors waitress python-json-logger pyyaml

# For full dependencies including ML (requires cmake for onnxsim)
pip install -r requirements.txt
```

## Essential Commands

### Testing
```bash
source venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/unit/featurizer/ -v          # Featurizer tests (220+ tests)
python -m pytest tests/unit/training/ -v            # Training pipeline (37 tests)
python -m pytest tests/integration/ -v              # Integration tests

# Run with coverage
pytest --cov=query_predictor --cov-report=html tests/

# CI test command (used by Strata CI/CD)
make strata-test
```

### Running the Service
```bash
source venv/bin/activate
python -m query_predictor.service
# Service runs on port 8000

# Test endpoints
curl http://localhost:8000/manage/health/liveness
curl http://localhost:8000/manage/health/readiness
```

### Docker
```bash
docker build -t trino-query-predictor .
docker run -p 8000:8000 -e QUERY_PREDICTOR_S3_BUCKET=your-bucket trino-query-predictor
```

## Architecture Principles

### 1. Shared Featurization (Train-Serve Parity)
The `query_predictor/core/featurizer/` module is **SHARED** between training and inference to prevent train-serve skew. This is the most critical architectural decision.

```python
# Both training and inference use identical FeatureExtractor
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor

# In training (PySpark UDF)
extractor = FeatureExtractor(config)
base_udf = extractor.create_spark_udf()
train_df = train_df.withColumn('base_features', base_udf(F.struct(...)))

# In inference (single query)
extractor = FeatureExtractor(config)
features = extractor.extract(query_data)
```

**When modifying featurizers**: Changes to any extractor in `query_predictor/core/featurizer/extractors/` affect BOTH training and inference. Always:
1. Run both featurizer tests AND training pipeline tests
2. Update `FeatureExtractor.VERSION` if feature counts change
3. Ensure extractor order in `_initialize_extractors()` remains unchanged (critical for feature ordering)

### 2. Feature Configuration System
The system supports multiple feature configurations:
- **78 features (production)**: 9 base extractors, `enable_historical_features=False`
- **95 features (enhanced)**: Base + 17 historical features, `enable_historical_features=True`
- **345 features (training)**: Base + historical + 250 SQL-aware TF-IDF

**When changing feature count**: Update `config/service_config.yaml`:
```yaml
featurizer:
  expected_feature_count: 78  # or 95
  enable_historical_features: false  # or true
```

### 3. Distributed TF-IDF (Training Only)
`SparkMLTfidfPipeline` in `query_predictor/training/spark_ml_tfidf_pipeline.py` implements SQL-aware TF-IDF with:
- **Binary mode**: Presence/absence instead of term frequency
- **SQL keyword filtering**: Removes 40+ common keywords (SELECT, FROM, WHERE, etc.)
- **SQL normalization**: Replaces literals/numbers with placeholders
- **Distributed vocabulary building**: No `.collect()` - uses Spark ML directly

**When modifying TF-IDF**: Changes must preserve sklearn compatibility for inference while using Spark ML for training. Run `tests/unit/training/test_spark_ml_tfidf_pipeline.py` (27 tests).

## Code Organization

```
query_predictor/
├── core/                           # SHARED code (training + inference)
│   ├── featurizer/                 # ⚠️ SHARED: Changes affect both pipelines
│   │   ├── feature_extractor.py    # Main orchestrator (facade pattern)
│   │   ├── base.py                 # Abstract base for all extractors
│   │   ├── parsers/                # AST (sqlglot) and regex parsers
│   │   └── extractors/             # 9 specialized extractors (SOLID design)
│   │       ├── sql_extractor.py           # 15 features
│   │       ├── table_join_extractor.py    # 12 features
│   │       ├── where_extractor.py         # 10 features
│   │       ├── aggregation_extractor.py   # 8 features
│   │       ├── ast_extractor.py           # 10 features (with 50ms timeout)
│   │       ├── context_extractor.py       # 8 features
│   │       ├── query_type_extractor.py    # 5 features
│   │       ├── set_operation_extractor.py # 4 features
│   │       ├── null_aware_extractor.py    # 6 features
│   │       └── historical_extractor.py    # 17 features (optional)
│   │
│   └── types/                      # Data models and schemas
│       ├── query_data.py           # QueryData dataclass
│       ├── feature_spec.py         # FeatureSpec for enable/disable
│       ├── historical_stats.py     # Historical statistics schema
│       └── ast_metrics.py          # AST parsing metrics
│
├── training/                       # Training-only modules
│   ├── spark_ml_tfidf_pipeline.py  # Distributed TF-IDF (training) → sklearn (inference)
│   ├── parity_validator.py         # Train-serve parity validation
│   ├── historical_stats_computer.py # Compute historical statistics
│   ├── checkpoint_manager.py       # S3 checkpointing for Spark
│   └── [other training modules]
│
├── service/                        # Inference service
│   ├── app.py                      # Flask + Waitress entry point
│   ├── routes.py                   # API endpoints
│   └── schemas.py                  # Pydantic request/response models
│
└── utils/                          # Shared utilities
    ├── s3_utils.py                 # S3 handler with 24hr caching
    ├── logging_utils.py            # JSON logging
    └── config.py                   # Configuration management
```

## Critical Implementation Details

### Extractor Order is Sacred
Extractor initialization order in `FeatureExtractor._initialize_extractors()` determines feature vector order. **Never change the order** without retraining models:

```python
# query_predictor/core/featurizer/feature_extractor.py
extractors = [
    SQLFeatureExtractor(...),           # Features 0-14
    TableJoinExtractor(...),            # Features 15-26
    WhereClauseExtractor(...),          # Features 27-36
    AggregationExtractor(...),          # Features 37-44
    ASTFeatureExtractor(...),           # Features 45-54
    ContextExtractor(...),              # Features 55-62
    QueryTypeExtractor(...),            # Features 63-67
    SetOperationExtractor(...),         # Features 68-71
    NullAwareExtractor(...),            # Features 72-77
    # Historical is LAST (78-94) when enabled
]
```

### NULL Handling is Explicit
The system handles 3.8M NULL catalogs and 4.1M NULL schemas. `NullAwareExtractor` provides 6 features for NULL detection and inference. **Never use `.isnull()` checks** - use the safe utilities in `query_predictor/core/featurizer/utils.py`:

```python
from query_predictor.core.featurizer.utils import safe_float, safe_int, is_null_value

# ✅ Correct
value = safe_float(data.get('cpu_seconds'), default=0.0)
is_null = is_null_value(data.get('catalog'))

# ❌ Incorrect - doesn't handle edge cases
value = float(data.get('cpu_seconds') or 0.0)  # fails on "null" string
```

### AST Parsing has Timeouts
`ASTParser` uses sqlglot with a 50ms timeout to prevent hanging on pathological queries. The parser has fallback strategies:

```python
# query_predictor/core/featurizer/parsers/ast_parser.py
# Returns ASTMetrics with parsed_successfully=False on timeout
# Extractors check parsed_successfully before using AST
```

**When debugging AST issues**: Check `ast_metrics.parsed_successfully` and `ast_metrics.parse_error_msg` before assuming AST is valid.

### S3 Caching is Automatic
`S3Utils` provides 24-hour local caching with stale-cache fallback. **Do not bypass** the cache for model/feature_spec loading:

```python
# ✅ Correct - uses cache
from query_predictor.utils.s3_utils import S3Utils
s3 = S3Utils(config)
content = s3.read_s3_file("s3://bucket/model.onnx")

# ❌ Incorrect - no caching
import boto3
s3 = boto3.client('s3')
obj = s3.get_object(Bucket='bucket', Key='model.onnx')
```

## Training Pipeline (Jupyter Notebooks)

Located in `notebooks/` for SageMaker JupyterLab with PySpark:

1. **00_setup_and_config.ipynb** - Package code, upload to S3 for Spark
2. **01_data_loading.ipynb** - Load Trino logs, label, filter, sample (30-45 min)
3. **02_zero_cost_analysis.ipynb** - Validate zero-cost SQL categories (15 min)
4. **03_feature_engineering.ipynb** - Extract 345 features, build TF-IDF vocab (45-60 min)
5. **04_model_training.ipynb** - Train XGBoost, export ONNX (30-45 min)

**Key principle**: TF-IDF vocabulary is built ONLY on training data (no data leakage). Time-based splits: 30 days train / 7 days val / 7 days test.

## Common Workflows

### Adding a New Feature Extractor

1. Create extractor in `query_predictor/core/featurizer/extractors/new_extractor.py`:
   ```python
   from query_predictor.core.featurizer.base import BaseFeatureExtractor

   class NewExtractor(BaseFeatureExtractor):
       def __init__(self, config, feature_spec=None):
           super().__init__(config, feature_spec)

       def extract(self, query_data, ast_metrics, sql_parsed):
           # Return list of floats
           return [1.0, 2.0, 3.0]

       def get_feature_names(self):
           return ['new_feature_1', 'new_feature_2', 'new_feature_3']
   ```

2. Add to `FeatureExtractor._initialize_extractors()` (AT THE END to preserve order):
   ```python
   extractors = [
       # ... existing extractors ...
       NewExtractor(self.config, self.feature_spec),  # ADD HERE
   ]
   ```

3. Update expected feature count in `config/service_config.yaml`:
   ```yaml
   featurizer:
     expected_feature_count: 81  # was 78, added 3
   ```

4. Add tests in `tests/unit/featurizer/test_extractors.py`

5. Run parity validation: `python -m pytest tests/unit/training/test_parity_validator.py -v`

### Modifying TF-IDF Parameters

1. Update `config/training_config.yaml`:
   ```yaml
   tfidf:
     vocab_size: 250        # Default 250 (was 1000)
     min_df: 100           # Minimum document frequency
     max_df: 0.80          # Maximum document frequency
     use_binary: true      # Binary presence/absence
   ```

2. Retrain model (re-run notebooks 03 and 04)

3. Validate: Check `tests/unit/training/test_spark_ml_tfidf_pipeline.py` passes

### Running Tests Before PR

```bash
# Run all tests with coverage
source venv/bin/activate
pytest --cov=query_predictor --cov-report=html tests/

# Ensure Strata CI test passes
make strata-test

# Check specific areas
python -m pytest tests/unit/featurizer/ -v          # After featurizer changes
python -m pytest tests/unit/training/ -v            # After training changes
python -m pytest tests/integration/ -v              # After major changes
```

## Configuration Files

- `config/service_config.yaml` - Service settings (port, workers, timeouts, feature toggles)
- `config/training_config.yaml` - Training pipeline settings (dates, thresholds, Spark config)
- `pyrightconfig.json` - Type checking configuration (Python 3.13)

## Environment Variables

Key variables for service deployment:
```bash
# S3 Configuration (overrides config/service_config.yaml)
QUERY_PREDICTOR_S3_BUCKET=uip-datalake-bucket-prod
QUERY_PREDICTOR_S3_PREFIX=query_predictor
QUERY_PREDICTOR_MODEL_FILE=model_v20251005.onnx
QUERY_PREDICTOR_FEATURE_SPEC_FILE=feature_spec_v20251005.json
QUERY_PREDICTOR_HISTORICAL_STATS_FILE=historical_stats_v20251005.json

# Feature Toggles
QUERY_PREDICTOR_ZERO_COST_ENABLED=true
QUERY_PREDICTOR_HISTORICAL_FEATURES_ENABLED=false

# Model Configuration
QUERY_PREDICTOR_THRESHOLD=0.5
QUERY_PREDICTOR_MODEL_VERSION=20241005
```

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Overall p99 latency | <500ms | End-to-end request |
| Stage 0 p99 (zero-cost) | <10ms | Simple SQL parsing |
| Stage 1 p99 (ML) | <200ms | Feature extraction + inference |
| Feature extraction p99 | <100ms | Within Stage 1 |
| Throughput | 180 RPS | Sustained load |
| Success rate | >99.95% | With fallback to "default" |

## Common Issues

### Import Errors After Changes
```bash
# Reinstall package in editable mode
source venv/bin/activate
pip install -e . --no-deps
```

### Feature Count Mismatch
Check `config/service_config.yaml` matches extractor count:
```python
# Count features programmatically
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor
config = {'ast_timeout_ms': 50, 'enable_historical_features': False}
extractor = FeatureExtractor(config)
print(len(extractor.feature_names))  # Should match expected_feature_count
```

### Train-Serve Parity Failures
Run parity validator:
```bash
python -m pytest tests/unit/training/test_parity_validator.py -v -s
```

Check for:
- Extractor order changes
- Feature name changes
- Config mismatches between training and inference

## Deployment

- **CI/CD**: Strata pipeline (`.strata.yml`)
- **Test command**: `make strata-test`
- **Base image**: RHEL9 Python3
- **Container user**: cronutil (uid 7447)
- **Health checks**: `/manage/health/liveness` and `/manage/health/readiness`
- **Metrics**: `/metrics` (Prometheus format)

## Key Files to Review

When getting started:
1. `README.md` - Comprehensive overview with all details
2. `query_predictor/core/featurizer/feature_extractor.py` - Main orchestrator
3. `config/service_config.yaml` - Service configuration
4. `notebooks/README.md` - Training pipeline guide
5. This file (CLAUDE.md) - Architecture and workflows