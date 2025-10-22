# Trino Query Predictor

## Project Overview

**trino-query-predictor** is a Python service for smart routing in the Trino ecosystem that classifies Trino queries as "small" or "heavy" at submit time to enable cost-effective cluster routing.

This repository contains:
- **Prediction Service** (`query_predictor/`): Flask REST API for real-time query classification
- **Training Pipeline** (`query_predictor/training/`): PySpark-based feature extraction and model training
- **Training Notebooks** (`notebooks/`): 5 Jupyter notebooks for SageMaker JupyterLab
- **Deployment** (Docker + Strata CI/CD): RHEL9 Python3 container deployed to Falcon

### Key Architectural Decision: Shared Featurization

The `query_predictor/core/featurizer/` module is **SHARED** between training and inference to prevent train-serve skew. Training code will import from the same featurizer used by the prediction service.

```python
# In prediction service
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor

config = {'ast_timeout_ms': 50, 'enable_historical_features': False}
extractor = FeatureExtractor(config)
features = extractor.extract(query_data)  # Returns 78 or 95 floats

# In training pipeline
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor  # SAME module
extract_udf = extractor.create_spark_udf()  # PySpark UDF wrapper
```

**Architecture:**
- **Modular Design:** 9-10 specialized extractors following SOLID principles (10 when historical features enabled)
- **Error Isolation:** Per-extractor error handling prevents cascade failures
- **NULL-Aware:** Explicit handling for NULL catalogs (3.8M) and schemas (4.1M)
- **Cold-Start Handling:** Historical features provide user/catalog/schema patterns for new entities
- **Performance:** AST parsing with 50ms timeout, S3 caching (24hr TTL)
- **Feature Control:** FeatureSpec for dynamic enable/disable based on importance

## Development Commands

### Environment Setup
```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install package in development mode (without problematic dependencies)
pip install -e . --no-deps

# Install core dependencies only (some ML deps require cmake)
pip install Flask flask-cors waitress python-json-logger pyyaml

# For full dependencies (requires cmake for onnxsim):
# pip install -r requirements.txt
```

### Testing
```bash
# Activate virtual environment first
source venv/bin/activate

# Run all tests
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/unit/featurizer/ -v          # Featurizer tests (220+ tests)
python -m pytest tests/unit/training/ -v            # Training pipeline tests (37 tests)
python -m pytest tests/unit/test_schemas.py -v      # Schema model tests
python -m pytest tests/unit/test_routes.py -v       # Service endpoint tests
python -m pytest tests/integration/ -v              # Integration tests

# Run tests with coverage
pytest --cov=query_predictor --cov-report=html tests/

# Run Strata CI test suite (used in CI/CD)
make strata-test
```

### Running the Service
```bash
# Activate virtual environment first
source venv/bin/activate

# Run Flask service locally (port 8000)
python -m query_predictor.service

# Test health endpoints
curl http://localhost:8000/manage/health/liveness
curl http://localhost:8000/manage/health/readiness

# Test prediction endpoint
curl -X POST http://localhost:8000/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"query": "SELECT * FROM table", "user": "test@example.com"}'
```

### Docker
```bash
# Build Docker image
docker build -t trino-query-predictor .

# Run container
docker run -p 8000:8000 \
  -e QUERY_PREDICTOR_S3_BUCKET=your-bucket \
  trino-query-predictor
```

## Architecture & Structure

### Two-Stage Inference Pipeline

The service uses a two-stage approach to classify queries:

```
HTTP Request → /v1/predict
  ↓
[Request Validation] (Pydantic)
  ↓
Stage 0: Zero-Cost Filter (SQLGlot)
  ├─→ Zero-cost query (DDL, DESCRIBE, etc.)? → Return "small" (skip ML)
  ├─→ Parse error? → Continue to Stage 1
  └─→ Not zero-cost? → Stage 1
        ↓
Stage 1: ML Inference
  ├─→ FeatureExtractor.extract() → 78 features (current)
  ├─→ ONNX Runtime inference
  ├─→ Apply threshold → "small" or "heavy"
  └─→ Error? → Fallback to "default"
        ↓
Response: {prediction, confidence, stage, model_version}
```

**Stage 0 (Zero-Cost Filter):**
- Uses SQLGlot for intelligent SQL parsing
- Routes ~15-20% of queries immediately without ML
- Categories: DDL, DESCRIBE, SHOW, EXPLAIN, SET SESSION
- Target latency: <10ms p99

**Stage 1 (ML Inference):**
- Feature extraction:
  - **78 features (production)**: 72 base + 6 NULL-aware (when historical and TF-IDF disabled)
  - **95 features (with historical)**: 78 base + 17 historical (when TF-IDF disabled)
  - **345 features (full training)**: 78 base + 17 historical + 250 TF-IDF (training pipeline with SQL-aware optimizations)
- Historical features: User/catalog/schema query patterns for cold-start handling
- TF-IDF features: SQL-aware distributed vocabulary with binary mode, keyword filtering, and normalization
- ONNX Runtime for inference
- Target latency: <200ms p99
- Fallback to "default" cluster on any error

### Repository Structure

```
trino-query-predictor/
├── query_predictor/              # Main Python package
│   ├── core/                     # Shared ML components
│   │   ├── types/                # Data types and models
│   │   │   ├── query_data.py     # QueryData dataclass
│   │   │   ├── feature_spec.py   # FeatureSpec, FeatureGroup
│   │   │   ├── ast_metrics.py    # ASTMetrics dataclass
│   │   │   └── historical_stats.py  # HistoricalStatsSchema
│   │   └── featurizer/           # Feature extraction (SHARED)
│   │       ├── base.py           # Abstract base extractor
│   │       ├── utils.py          # Safe utility functions
│   │       ├── feature_extractor.py  # Main orchestrator
│   │       ├── parsers/
│   │       │   ├── ast_parser.py     # sqlglot with timeout
│   │       │   └── sql_parser.py     # Regex pattern extraction
│   │       └── extractors/
│   │           ├── sql_extractor.py  # 15 features
│   │           ├── table_join_extractor.py  # 12 features
│   │           ├── where_extractor.py  # 10 features
│   │           ├── aggregation_extractor.py  # 8 features
│   │           ├── ast_extractor.py  # 10 features
│   │           ├── context_extractor.py  # 8 features
│   │           ├── query_type_extractor.py  # 5 features
│   │           ├── set_operation_extractor.py  # 4 features
│   │           ├── null_aware_extractor.py  # 6 features
│   │           └── historical_extractor.py  # 17 features (optional)
│   │
│   ├── training/                 # Training pipeline (PySpark)
│   │   ├── checkpoint_manager.py # Spark checkpointing
│   │   ├── historical_stats_computer.py  # Compute historical stats
│   │   ├── spark_ml_tfidf_pipeline.py   # Distributed TF-IDF (1000 features)
│   │   └── parity_validator.py  # Train-serve parity validation
│   │
│   ├── service/                  # Prediction service
│   │   ├── __init__.py           # Flask + Waitress entry point
│   │   ├── routes.py             # API route handlers
│   │   └── schemas.py            # Pydantic schemas
│   │
│   └── utils/                    # Service utilities
│       ├── logging_utils.py      # Structured JSON logging
│       ├── config.py             # Configuration management
│       └── s3_utils.py           # S3 handler with caching
│
├── notebooks/                    # Training notebooks (SageMaker)
│   ├── 00_setup.ipynb            # Environment setup + code upload
│   ├── 01_data_loading.ipynb    # Load and preprocess Trino logs
│   ├── 02_eda.ipynb              # Exploratory data analysis
│   ├── 03_feature_engineering.ipynb  # Extract 1095 features
│   └── 04_model_training.ipynb  # Train XGBoost + ONNX export
│
├── tests/
│   ├── unit/                     # Unit tests
│   │   ├── featurizer/           # 220+ tests
│   │   │   ├── test_utils.py
│   │   │   ├── test_models.py
│   │   │   ├── test_parsers.py
│   │   │   ├── test_extractors.py
│   │   │   ├── test_feature_extractor.py
│   │   │   └── test_historical_extractor.py
│   │   ├── training/             # Training pipeline tests
│   │   │   ├── test_spark_ml_tfidf_pipeline.py  # 18 tests
│   │   │   └── test_parity_validator.py         # 19 tests
│   │   ├── test_schemas.py       # Pydantic tests
│   │   ├── test_config.py        # Config tests
│   │   └── test_routes.py        # Route tests
│   └── integration/              # Integration tests
│       └── test_feature_extraction_pipeline.py
│
├── docs/                         # Documentation
│   ├── PRD-prediction-service.md
│   └── implementation-plan.md
│
├── config/                       # Configuration
│   ├── service_config.yaml       # Service + featurizer settings
│   └── training_config.yaml      # Training pipeline settings
│
├── Dockerfile                    # RHEL9 Python3
├── requirements.txt              # Dependencies
└── setup.py                      # Package setup
```


## Getting Started

### Prerequisites

- Python 3.8+
- AWS CLI configured (for S3 access)
- Docker (for containerized deployment)

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd trino-query-predictor

# Set up development environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Start service locally
python -m query_predictor.service
```

### Using the Featurizer

#### Basic Usage (78 features)

```python
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor

# Initialize without historical features
config = {
    'ast_timeout_ms': 50,
    'enable_historical_features': False  # Default: 78 features
}
extractor = FeatureExtractor(config)

# Extract features
query_data = {
    'query': 'SELECT * FROM orders WHERE date > "2024-01-01"',
    'user': 'john.doe',
    'catalog': 'hive',
    'schema': 'default',
    'hour': 14
}

features = extractor.extract(query_data)
# Returns: List[float] with 78 features

# Get feature names
feature_names = extractor.feature_names
# Returns: ['query_length', 'token_count', 'has_select_star', ...]

# Get extractor summary
summary = extractor.get_extractor_summary()
# Returns: {'version': '3.0.0', 'total_extractors': 9, ...}
```

#### With Historical Features (95 features)

```python
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor
from query_predictor.core.types.historical_stats import HistoricalStatsSchema

# Load historical stats from S3
historical_stats = HistoricalStatsSchema.load_from_s3(
    "s3://bucket/historical_stats_v20251005.json"
)

# Initialize with historical features
config = {
    'ast_timeout_ms': 50,
    'enable_historical_features': True  # Enable 17 additional features
}
extractor = FeatureExtractor(
    config,
    historical_stats=historical_stats.to_dict()
)

# Extract features
features = extractor.extract(query_data)
# Returns: List[float] with 95 features (78 base + 17 historical)

# Historical features include:
# - User features (6): query patterns, heavy rate, resource usage
# - Catalog features (6): catalog-level query patterns
# - Schema features (4): schema-level query patterns
# - Cold-start indicator (1): flag for unknown users/catalogs/schemas
```

### Using FeatureSpec for Feature Control

```python
from query_predictor.core.types.feature_spec import FeatureSpec

# Load from S3 (with automatic caching)
spec = FeatureSpec.load_from_s3("s3://bucket/feature_spec_v20251005.json")

# Disable low-importance features
spec.importance_threshold = 0.01  # Disable features < 1% importance

# Manual disable
spec.disabled_features.add("query_length")

# Save back to S3
spec.save_to_s3("s3://bucket/feature_spec_updated.json")

# Use with extractor
extractor = FeatureExtractor(config, feature_spec=spec)
features = extractor.extract(query_data)
```

### Development Workflow

Structured implementation covering:
- Service infrastructure 
- Training pipeline 
- Automation and monitoring 

## Configuration

### Environment Variables

```bash
# Service Configuration
QUERY_PREDICTOR_CONFIG=/app/config/service_config.yaml
QUERY_PREDICTOR_PORT=8000
QUERY_PREDICTOR_LOG_LEVEL=INFO

# AWS & S3 Configuration
AWS_DEFAULT_REGION=us-west-2
AWS_ROLE_ARN=arn:aws:iam::123456789:role/query-predictor

# S3 Bucket and Path Configuration (overrides config/service_config.yaml)
# Option 1: Override individual components (recommended)
QUERY_PREDICTOR_S3_BUCKET=uip-datalake-bucket-prod
QUERY_PREDICTOR_S3_PREFIX=query_predictor
QUERY_PREDICTOR_MODEL_FILE=model_v20251005.onnx
QUERY_PREDICTOR_FEATURE_SPEC_FILE=feature_spec_v20251005.json
QUERY_PREDICTOR_HISTORICAL_STATS_FILE=historical_stats_v20251005.json

# Option 2: Override full S3 paths (takes precedence over components)
# QUERY_PREDICTOR_MODEL_PATH=s3://my-bucket/my-prefix/model.onnx
# QUERY_PREDICTOR_FEATURE_SPEC_PATH=s3://my-bucket/my-prefix/feature_spec.json
# QUERY_PREDICTOR_HISTORICAL_STATS_PATH=s3://my-bucket/my-prefix/historical_stats.json

# Model Configuration
QUERY_PREDICTOR_MODEL_VERSION=20241005
QUERY_PREDICTOR_THRESHOLD=0.5

# Feature Toggles
QUERY_PREDICTOR_ZERO_COST_ENABLED=true
QUERY_PREDICTOR_HISTORICAL_FEATURES_ENABLED=false
```

### Configuration Files

- `config/service_config.yaml` - Service settings (port, workers, timeouts)
- `config/training_config.yaml` - Training pipeline settings

## Feature Extraction

The `FeatureExtractor` class uses a modular architecture with specialized extractors. The system supports multiple feature configurations:

- **Production (78 features)**: Base features only (fast, <100ms p99)
- **Enhanced (95 features)**: Base + historical features (with cold-start handling)
- **Training (345 features)**: Base + historical + SQL-aware TF-IDF (full optimized set)

### Base Feature Extractors (78 features)
- **SQLFeatureExtractor** (15 features): query_length, token_count, has_select_star, column_count, etc.
- **TableJoinExtractor** (12 features): table_count, join_count, with_clause_count, cte_complexity, etc.
- **WhereClauseExtractor** (10 features): where_condition_count, in_clause_count, max_in_list_size, etc.
- **AggregationExtractor** (8 features): group_by_count, count_function_count, window_function_count, etc.
- **ASTFeatureExtractor** (10 features): ast_depth, ast_node_count, case_when_count, parse_timeout, etc.
- **ContextExtractor** (8 features): user_hash, catalog_hash, hour_sin, hour_cos, is_business_hours, etc.
- **QueryTypeExtractor** (5 features): is_select_query, is_insert_query, is_create_table_as, etc.
- **SetOperationExtractor** (4 features): union_count, except_count, intersect_count, has_multiple_statements
- **NullAwareExtractor** (6 features): is_catalog_null, is_schema_null, inferred_catalog_count, etc.

### Historical Features (17 features, optional)
- **User Features** (6): query_count, heavy_rate, avg_cpu_seconds, p90_cpu_seconds, avg_memory_gb, catalog_diversity
- **Catalog Features** (6): query_count, heavy_rate, avg_cpu_seconds, p90_cpu_seconds, avg_memory_gb, user_diversity
- **Schema Features** (4): query_count, heavy_rate, avg_cpu_seconds, avg_memory_gb
- **Cold-Start Indicator** (1): Flags unknown users/catalogs/schemas

### SQL-Aware TF-IDF Features (250 features, training only)
- **Binary mode**: Presence/absence instead of term frequency (better for SQL)
- **SQL keyword filtering**: Removes 40+ common SQL keywords (SELECT, FROM, WHERE, etc.)
- **SQL normalization**: Replaces literals, numbers, dates with placeholders
- **Distributed vocabulary building**: Spark ML CountVectorizer + IDF (no driver collection)
- **Table name preservation**: Keeps underscores in identifiers (e.g., `table_name`)
- **Optimized parameters**: vocab_size=250, min_df=100, max_df=0.80

**SQL Normalization Examples**:
```sql
-- Before normalization:
SELECT * FROM users WHERE id = 12345 AND name = 'John Doe' AND date > '2025-01-15'

-- After normalization:
SELECT * FROM users WHERE id =  NUMERIC  AND name =  STRING_LITERAL  AND date >  STRING_LITERAL

-- Tokens extracted (after keyword filtering):
['users', 'numeric', 'string_literal']
```

See `SparkMLTfidfPipeline` documentation below for implementation details.

**Key Design Principles:**
- SQLGlot AST parsing with 50ms timeout + fallback strategies
- Handles NULL catalog/schema gracefully (3.8M NULL catalogs, 4.1M NULL schemas)
- Error isolation: Per-extractor error handling prevents cascade failures
- Train-serve parity: Shared featurization code for training and inference
- FeatureSpec: Dynamic feature enable/disable based on importance scores
- Modular design: Easy to add/remove feature groups

## Performance Targets

| Metric | Target | Stage |
|--------|--------|-------|
| Overall p99 latency | <500ms | End-to-end |
| Stage 0 p99 latency | <10ms | Zero-cost filter |
| Stage 1 p99 latency | <200ms | ML inference |
| Feature extraction p99 | <100ms | Stage 1 component |
| Throughput | 180 RPS | Sustained load |
| Success rate | >99.95% | With fallback |

## Deployment

### Strata CI/CD Pipeline

- **Test command:** `make strata-test` (configured in `.strata.yml`)
- **Docker image:** Built from `Dockerfile` (RHEL9 Python3 base)
- **Promotion:** Automatically promotes to Falcon on merge to `master`
- **Production branch:** `master`

### Docker Image Details

- **Base:** `sfdc_rhel9_python3:118` (with `@ACCEPT_3PP_RISK` annotation)
- **Build deps:** gcc, gcc-c++, make, python3-devel, libgomp
- **Port:** 8000 (Flask via Waitress)
- **User:** cronutil (uid 7447)

### Health Checks

- **Liveness:** `GET /manage/health/liveness` - Service is alive
- **Readiness:** `GET /manage/health/readiness` - Service is ready (model loaded, featurizer initialized)

## API Endpoints

### GET /manage/health/liveness
Basic health check - service is alive

**Response:**
```json
{"status": "ok"}
```

### GET /manage/health/readiness
Readiness check - service is ready to accept requests

**Response:**
```json
{
  "status": "ready",
  "checks": {
    "featurizer_initialized": true
  }
}
```

### POST /v1/predict
Classify a Trino query as "HEAVY" or "LIGHT"

**Request:**
```json
{
  "query": "SELECT * FROM large_table WHERE date > '2024-01-01'",
  "user": "abc",
  "catalog": "hive",
  "schema": "default",
  "hour": 14,
  "clientInfo": "trino-python-client",
  "sessionProperties": {}
}
```

**Response:**
```json
{
  "prediction": "HEAVY",
  "confidence": 0.87,
  "probability": {"HEAVY": 0.87, "LIGHT": 0.13},
  "model_version": "1.0.0",
  "featurizer_version": "3.0.0",
  "feature_count": 78,
  "processing_time_ms": 45.2,
  "feature_extraction_time_ms": 38.1,
  "model_inference_time_ms": 7.1,
  "warnings": []
}
```

### GET /v1/info
Service and model information

### GET /metrics
Prometheus metrics (requests, latency, errors, fallbacks)

## Model Training Pipeline

The training pipeline is implemented in 5 Jupyter notebooks designed for SageMaker JupyterLab with PySpark:

### Training Notebooks

**00_setup.ipynb** - Environment Setup
- Package query_predictor code as zip
- Upload to S3 for Spark distribution
- Configure Spark session with proper memory settings
- Validates environment and dependencies

**01_data_loading.ipynb** - Data Loading & Preprocessing
- Load Trino query logs from S3
- Apply data quality filters (execution time, memory, completeness)
- Create is_heavy labels (cpu_time ≥ 300s OR memory ≥ 25GB)
- Save processed datasets to S3 for reuse

**02_eda.ipynb** - Exploratory Data Analysis
- Analyze query patterns and resource usage
- Validate data quality and class distribution
- Identify NULL handling requirements (3.8M NULL catalogs, 4.1M NULL schemas)
- Document feature engineering requirements

**03_feature_engineering.ipynb** - Feature Extraction
- Create time-based splits (30 train / 7 val / 7 test days)
- Extract 78 base features using production FeatureExtractor
- Extract 17 historical features from training statistics
- Build SQL-aware TF-IDF vocabulary (250 features) with distributed approach
- Validate train-serve parity (<0.5% mismatch tolerance)
- Save feature datasets + TF-IDF vectorizer to S3

**04_model_training.ipynb** - Model Training & Export
- Train XGBoost with class imbalance handling
- Hyperparameter tuning with validation set
- Export to ONNX with parity validation
- Target: Heavy recall ≥98%, FNR ≤1%

### Training Architecture

**Shared Featurization** - Training and inference use identical feature extraction:
```python
# In training (notebook 03)
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(config)
base_udf = extractor.create_spark_udf()  # PySpark UDF
train_df = train_df.withColumn('base_features', base_udf(F.struct(...)))

# In inference (production service)
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor

extractor = FeatureExtractor(config)
features = extractor.extract(query_data)  # Same logic, different wrapper
```

**Feature Pipeline** (345 total features):
1. **Base Features (78)**: Production FeatureExtractor with 9 specialized extractors
2. **Historical Features (17)**: User/catalog/schema statistics with cold-start handling
3. **SQL-Aware TF-IDF Features (250)**: Optimized distributed vocabulary with binary mode and keyword filtering

### Distributed SQL-Aware TF-IDF with SparkMLTfidfPipeline

**Problem**: Original sklearn TF-IDF required `.collect()` which exceeded 8GB driver memory limit.

**Solution**: `SparkMLTfidfPipeline` uses Spark ML for distributed vocabulary building with SQL-specific optimizations:

```python
from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

# Training: SQL-aware distributed vocabulary building (NO COLLECT!)
tfidf_config = {
    'tfidf_vocab_size': 250,       # Optimized for SQL queries
    'min_df': 100,                 # Filter rare terms
    'max_df': 0.80,                # Filter very common terms
    'use_binary': True,            # Binary presence/absence (better for SQL)
    'filter_sql_keywords': True,   # Remove SQL keywords
    'normalize_sql': True          # Normalize literals and numbers
}
tfidf_pipeline = SparkMLTfidfPipeline(tfidf_config)

# Fit on DataFrame directly - fully distributed
tfidf_pipeline.fit_on_dataframe(train_df, query_column='query')

# Create Spark UDF for distributed transformation
tfidf_udf = tfidf_pipeline.create_spark_udf()
train_features = train_df.withColumn('tfidf_features', tfidf_udf(F.col('query')))

# Save for inference
tfidf_pipeline.save('s3://bucket/tfidf_vectorizer.pkl')
```

**Inference**: sklearn-compatible for lightweight single-query transformation:

```python
# Load pipeline
pipeline = SparkMLTfidfPipeline.load('s3://bucket/tfidf_vectorizer.pkl')

# Transform single query (no Spark needed)
features = pipeline.transform_single("SELECT * FROM table")  # Returns np.ndarray
```

**SQL-Aware Optimizations**:
- **Binary TF-IDF**: Uses presence/absence instead of term counts (better for SQL query classification)
- **SQL keyword filtering**: Removes 40+ common keywords (SELECT, FROM, WHERE, JOIN, etc.) that don't differentiate query types
- **SQL normalization**: Replaces literals, numbers, and dates with placeholders to reduce vocabulary noise
- **Table name preservation**: Keeps underscores in identifiers (e.g., `user_events`, `fact_table`)
- **Aggressive filtering**: min_df=100 (must appear in ≥100 queries), max_df=0.80 (filter very common terms)

**Key Features**:
- Uses Spark ML (CountVectorizer + IDF) for distributed training
- Extracts vocabulary and IDF weights after fitting
- Initializes sklearn TfidfVectorizer for inference compatibility
- Scales to any dataset size without driver memory limits
- Preserves vocabulary for interpretability (unlike HashingTF)
- Train-serve parity through vocabulary extraction and shared preprocessing

### Train-Serve Parity Validation

`ParityValidator` ensures training features match inference features:

```python
from query_predictor.training.parity_validator import ParityValidator

validator = ParityValidator(config={'tolerance': 1e-6})

# Validate training vs inference features
parity_result = validator.validate_parity(
    training_features=train_features,  # From Spark DataFrame
    inference_featurizer=extractor,    # Production FeatureExtractor
    tfidf_pipeline=tfidf_pipeline,
    sample_queries=sample_queries,
    n_samples=100
)

if not parity_result['passed']:
    raise ValueError("Train-serve skew detected!")
```

**Validation Checks**:
- Feature count match (345 expected: 78 + 17 + 250)
- Numerical parity (tolerance: 1e-6)
- Mismatch rate < 0.5%
- Reports detailed differences for debugging
- Validates SQL normalization consistency across training UDF and inference

### Time-Based Splits

Training uses chronological splits to simulate production deployment:

```python
# 44-day total period
Train:      First 30 days  (T-44 → T-14)
Validation: Next 7 days    (T-14 → T-7)
Test:       Last 7 days    (T-7 → T)
```

**Critical**: TF-IDF vocabulary is built ONLY on training data to prevent data leakage.

### Training Test Coverage

**test_spark_ml_tfidf_pipeline.py** (27 tests):
- Initialization and configuration with SQL-aware defaults
- SQL normalization (literals, numbers, dates, timestamps)
- SQL keyword filtering
- Binary vs count mode
- Distributed fitting without data collection
- Single-query transformation (inference)
- Save/load round-trip with new config fields
- Feature metadata with SQL-aware fields
- Spark UDF creation
- Edge cases (empty queries, special characters, case insensitivity, table names)

**test_parity_validator.py** (19 tests):
- Feature count validation
- Numerical parity checks
- Mismatch rate computation
- Report generation
- Error handling

## Documentation
- **ML Design Doc:** https://salesforce.quip.com/97O0AqbFLDPh (internal)

## Testing Strategy

- **Unit tests:** 260+ tests covering featurizers, training pipeline, service routes
- **Integration tests:** End-to-end feature extraction and train-serve parity
- **Performance tests:** Validate latency targets (<500ms p99)
- **Load tests:** 180 RPS sustained throughput

```bash
# Run all tests
source venv/bin/activate
python -m pytest tests/ -v

# Run specific test suites
python -m pytest tests/unit/featurizer/ -v              # 220+ tests
python -m pytest tests/unit/training/ -v                # 37 tests (TF-IDF + parity)
python -m pytest tests/integration/ -v                  # Integration tests

# Run with coverage
pytest --cov=query_predictor --cov-report=html tests/

# Run Strata CI test suite
make strata-test
```

**Test Coverage**:
- **Featurizer**: 220+ tests for 9 extractors, parsers, and utilities
- **Training Pipeline**: 46 tests (27 SQL-aware TF-IDF + 19 parity validation)
- **Service**: 40+ tests for routes, schemas, and config
- **Integration**: End-to-end feature extraction and parity validation

## Common Issues & Solutions

### Build Failures

If `make strata-test` fails with missing dependencies:
```bash
# For RHEL systems
make build-deps  # Installs gcc, python3-devel, libgomp

# For local development, install core dependencies only
source venv/bin/activate
pip install -e . --no-deps
pip install Flask flask-cors waitress python-json-logger pyyaml
```

### Import Errors

If you see `ModuleNotFoundError: No module named 'query_predictor'`:
```bash
source venv/bin/activate
pip install -e . --no-deps  # Install package in development mode
```

### IDE Navigation Issues

If "Go to Declaration" and other IDE features don't work:
```bash
# Ensure package is installed in editable mode
source venv/bin/activate
pip install -e . --no-deps

# Reload VS Code/Cursor window
# Cmd+Shift+P -> "Developer: Reload Window"
```

### Docker Build Failures

If Docker build fails on RHEL9 base image, ensure `@ACCEPT_3PP_RISK` annotation is present in Dockerfile for third-party dependencies.

## Key Principles

- **Shared featurization** - Training and inference use the same `FeatureExtractor` class to prevent train-serve skew
- **SOLID architecture** - Modular design with single responsibility extractors
- **Error isolation** - Multi-layer error handling prevents cascade failures
- **NULL-aware design** - Explicit handling for NULL catalogs/schemas
- **Service-first approach** - Service infrastructure is built before adding ML components
- **Incremental integration** - Components are added with full context, tests, and integration code
- **Performance targets** - p99 latency <500ms, 180 RPS sustained throughput
- **CI/CD** - All PRs must pass `make strata-test` before merging to master

## Current Features

**Architecture** (25+ files, ~2500 lines):
- Modular OOP design with 9 specialized feature extractors
- Abstract base class with dependency injection
- Main orchestrator (facade pattern)
- Production S3 handler with caching
- Training pipeline with PySpark integration

**Feature Extraction** (78-345 features):
- **Base Features (78)**: 9 specialized extractors for SQL analysis
  - SQLFeatureExtractor (15), TableJoinExtractor (12), WhereClauseExtractor (10)
  - AggregationExtractor (8), ASTFeatureExtractor (10), ContextExtractor (8)
  - QueryTypeExtractor (5), SetOperationExtractor (4), NullAwareExtractor (6)
- **Historical Features (17)**: User/catalog/schema statistics with cold-start handling
- **SQL-Aware TF-IDF Features (250)**: Optimized distributed vocabulary with binary mode and keyword filtering

**Training Pipeline**:
- 5 Jupyter notebooks for SageMaker JupyterLab
- SparkMLTfidfPipeline for distributed TF-IDF (scales to any dataset size)
- HistoricalStatsComputer for user/catalog/schema statistics
- ParityValidator for train-serve parity validation
- CheckpointManager for Spark DataFrame checkpointing
- Time-based splits with no data leakage

**Key Features**:
- FeatureSpec system for dynamic feature enable/disable
- S3 integration with 24-hour local caching
- AST parsing with 50ms timeout + fallback strategies
- NULL-aware design (handles 3.8M NULL catalogs, 4.1M NULL schemas)
- Train-serve parity through shared featurization code
- Distributed TF-IDF without driver memory limits
- 5-layer error handling (extractor, orchestrator, parser, S3, service)

**Testing** (260+ unit tests):
- **Featurizer** (220+ tests):
  - test_utils.py (35 tests) - Safe utility functions
  - test_models.py (30 tests) - Data models and FeatureSpec
  - test_parsers.py (25 tests) - AST and SQL parsers
  - test_extractors.py (60 tests) - All 9 extractors
  - test_feature_extractor.py (30 tests) - Main orchestrator
  - test_historical_extractor.py (40 tests) - Historical features
- **Training Pipeline** (37 tests):
  - test_spark_ml_tfidf_pipeline.py (18 tests) - Distributed TF-IDF
  - test_parity_validator.py (19 tests) - Train-serve parity
- **Service** (40+ tests):
  - test_schemas.py - Pydantic schemas
  - test_routes.py - API endpoints
  - test_config.py - Configuration

**Documentation**:
- Comprehensive README with training pipeline documentation
- Model training notebooks with inline documentation
- SparkMLTfidfPipeline usage examples
- Train-serve parity validation guide

**Configuration**:
- service_config.yaml with featurizer and model settings
- training_config.yaml for training pipeline
- s3_cache configuration
- Feature toggles for historical features and TF-IDF

## Contributing

### PR Guidelines

- Each PR should have clear purpose and manageable scope
- Include unit tests and integration tests
- Add performance benchmarks for latency-sensitive components
- Update documentation as needed
- Ensure `make strata-test` passes

## Support

- **Documentation:** See `docs/` directory for detailed specifications
- **Issues:** Report issues via GitHub Issues
- **ML Design:** https://salesforce.quip.com/97O0AqbFLDPh (internal)
