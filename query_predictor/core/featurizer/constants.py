"""
Feature extraction constants.

Centralizes all normalization values and thresholds used across extractors.
"""

# ============================================================================
# Query Size Limits
# ============================================================================

# Maximum query length (1MB = 1,000,000 characters)
MAX_QUERY_LENGTH = 1_000_000

# ============================================================================
# AST Feature Normalization Constants
# ============================================================================

# AST depth normalization (typical queries: 3-8 depth, complex queries: 10+)
AST_DEPTH_NORMALIZER = 10.0

# AST width normalization (typical queries: 10-50 nodes, complex: 100+)
AST_WIDTH_NORMALIZER = 100.0

# AST node count normalization (typical: 50-300 nodes, heavy: 500+)
AST_NODE_COUNT_NORMALIZER = 500.0

# Predicate complexity (CTE + subquery count, typical: 1-5, complex: 10+)
PREDICATE_COMPLEXITY_NORMALIZER = 10.0

# CASE WHEN count (typical: 0-2, complex: 5+)
CASE_WHEN_COUNT_NORMALIZER = 5.0

# CASE WHEN max branches (typical: 2-5, complex: 10+)
CASE_WHEN_MAX_BRANCHES_NORMALIZER = 10.0

# UNION count (typical: 0-1, complex: 5+)
UNION_COUNT_NORMALIZER = 5.0

# UNION ALL count (typical: 0-2, complex: 5+)
UNION_ALL_COUNT_NORMALIZER = 5.0

# WITH clause count (typical: 0-1, complex: 3+)
WITH_COUNT_NORMALIZER = 3.0

# CTE count (typical: 0-2, complex: 5+)
CTE_COUNT_NORMALIZER = 5.0

# LATERAL VIEW count (typical: 0-1, complex: 3+, Spark/Hive specific)
LATERAL_VIEW_COUNT_NORMALIZER = 3.0

# Window function count (typical: 0-2, complex: 5+)
WINDOW_FUNC_COUNT_NORMALIZER = 5.0

# DISTINCT count (typical: 0-3, complex: 10+)
DISTINCT_COUNT_NORMALIZER = 10.0

# HAVING count (typical: 0-1, complex: 3+)
HAVING_COUNT_NORMALIZER = 3.0

# COALESCE/NULLIF count (typical: 0-5, complex: 10+)
COALESCE_NULL_IF_COUNT_NORMALIZER = 10.0

# ============================================================================
# AST Parsing Configuration
# ============================================================================

# Default timeout for AST parsing operations (50 ms)
DEFAULT_AST_TIMEOUT_MS = 50

# ============================================================================
# Historical Feature Normalization Constants
# ============================================================================
# These constants MUST match between training and inference for train-serve parity

# Query count normalization (p95 from production data analysis)
# Typical users: 10-1000 queries, power users: 10k+
HISTORICAL_QUERY_COUNT_NORMALIZER = 10000.0

# CPU time normalization (5 minutes = 300 seconds)
# Typical queries: 1-60s, heavy queries: 300s+
HISTORICAL_CPU_SECONDS_NORMALIZER = 300.0

# Memory normalization (50GB)
# Typical queries: 1-10GB, heavy queries: 50GB+
HISTORICAL_MEMORY_GB_NORMALIZER = 50.0

# Diversity normalization (100 unique entities)
# Typical users: 1-10 catalogs, power users: 100+
HISTORICAL_DIVERSITY_NORMALIZER = 100.0

# ============================================================================
# Cold-Start Default Values
# ============================================================================
# Used when user/catalog/schema is unknown (not in historical stats)
# These are conservative estimates based on production data analysis

COLD_START_DEFAULTS = {
    # Query volume defaults
    'query_count': 1,
    'heavy_rate': 0.1,  # Conservative 10% heavy rate

    # CPU time defaults (seconds)
    'avg_cpu_seconds': 30.0,      # 30 seconds average
    'p50_cpu_seconds': 15.0,      # 15 seconds median
    'p75_cpu_seconds': 50.0,      # 50 seconds p75
    'p90_cpu_seconds': 100.0,     # 100 seconds p90
    'p95_cpu_seconds': 200.0,     # 200 seconds p95
    'cpu_stddev': 50.0,           # 50 seconds standard deviation

    # Memory defaults (GB)
    'avg_memory_gb': 5.0,         # 5GB average
    'p95_memory_gb': 15.0,        # 15GB p95
    'p90_memory_gb': 12.0,        # 12GB p90

    # Diversity defaults
    'unique_catalogs': 1,         # Single catalog
    'unique_schemas': 1,          # Single schema
    'unique_users': 1,            # Single user
}
