"""
Historical statistics computation for training pipeline.

Computes per-user, per-catalog, and per-schema statistics from training query logs
using PySpark. These stats are used for cold-start handling in production.

Uses shared schema (HistoricalStatsSchema) to ensure train-serve parity.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

try:
    from pyspark.sql import DataFrame
    from pyspark.sql import functions as F
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from pyspark.sql import DataFrame
    else:
        DataFrame = Any  # Type hint fallback

from query_predictor.core.types.historical_stats import (
    HistoricalStatsSchema,
    UserStats,
    CatalogStats,
    SchemaStats
)

logger = logging.getLogger(__name__)


class HistoricalStatsComputer:
    """
    Compute historical statistics from training query logs.

    Uses PySpark for scalable computation across millions of queries.
    Outputs HistoricalStatsSchema compatible with inference service.

    Example usage:
        computer = HistoricalStatsComputer(version='1.0.0')
        stats_schema = computer.compute(training_df, date_range)
        stats_schema.save_to_s3('s3://bucket/historical_stats_v20251005.json')
    """

    def __init__(self, version: str = '1.0.0'):
        """
        Initialize stats computer.

        Args:
            version: Version string for output stats file
        """
        if not PYSPARK_AVAILABLE:
            raise RuntimeError(
                "PySpark is not available. Install with: pip install pyspark"
            )

        self.version = version
        logger.info(f"Initialized HistoricalStatsComputer v{version}")

    def compute(
        self,
        df: DataFrame,
        date_range: Optional[Dict[str, str]] = None
    ) -> HistoricalStatsSchema:
        """
        Compute complete historical statistics from training data.

        Args:
            df: PySpark DataFrame with training queries
                Required columns: query, user, catalog, schema, cpuTime,
                                peakUserMemoryBytes, is_heavy
            date_range: Optional date range dict {'start': '2025-06-01', 'end': '2025-07-15'}

        Returns:
            HistoricalStatsSchema with all computed stats

        Raises:
            ValueError: If required columns are missing
        """
        # Validate input DataFrame
        self._validate_dataframe(df)

        logger.info(f"Computing historical stats from {df.count():,} queries")

        # Set default date range if not provided
        if date_range is None:
            date_range = {
                'start': 'unknown',
                'end': 'unknown'
            }

        # Compute overall metrics
        total_queries = df.count()
        heavy_count = df.filter(F.col('is_heavy') == 1).count()
        heavy_rate_overall = heavy_count / total_queries if total_queries > 0 else 0.0

        logger.info(f"Overall heavy rate: {heavy_rate_overall:.2%} "
                   f"({heavy_count:,} / {total_queries:,})")

        # Compute per-entity stats
        logger.info("Computing user statistics...")
        users = self._compute_user_stats(df)

        logger.info("Computing catalog statistics...")
        catalogs = self._compute_catalog_stats(df)

        logger.info("Computing schema statistics...")
        schemas = self._compute_schema_stats(df)

        # Create validated schema
        stats_schema = HistoricalStatsSchema(
            version=self.version,
            created_at=datetime.now().isoformat(),
            data_date_range=date_range,
            users=users,
            catalogs=catalogs,
            schemas=schemas,
            total_queries_analyzed=total_queries,
            heavy_rate_overall=heavy_rate_overall
        )

        logger.info(f"Computed stats for {len(users)} users, "
                   f"{len(catalogs)} catalogs, {len(schemas)} schemas")

        return stats_schema

    def _validate_dataframe(self, df: DataFrame):
        """Validate DataFrame has required columns."""
        required_columns = {
            'query', 'user', 'catalog', 'schema',
            'cpuTime', 'peakUserMemoryBytes', 'is_heavy'
        }

        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"DataFrame missing required columns: {missing_columns}"
            )

    def _compute_user_stats(self, df: DataFrame) -> Dict[str, UserStats]:
        """
        Compute per-user statistics.

        Computes 9 metrics per user matching UserStats schema exactly.
        """
        user_stats_df = df.groupBy('user').agg(
            # Query volume
            F.count('*').alias('query_count'),

            # Heavy rate
            F.avg(
                F.when(F.col('is_heavy') == 1, 1.0).otherwise(0.0)
            ).alias('heavy_rate'),

            # CPU time metrics (convert milliseconds to seconds)
            F.avg(F.col('cpuTime') / 1000.0).alias('avg_cpu_seconds'),
            F.expr('percentile_approx(cpuTime / 1000.0, 0.9)').alias('p90_cpu_seconds'),
            F.stddev(F.col('cpuTime') / 1000.0).alias('cpu_stddev'),

            # Memory metrics (convert bytes to GB)
            F.avg(F.col('peakUserMemoryBytes') / (1024**3)).alias('avg_memory_gb'),
            F.expr('percentile_approx(peakUserMemoryBytes / POW(1024, 3), 0.9)').alias('p90_memory_gb'),

            # Diversity metrics
            F.countDistinct('catalog').alias('unique_catalogs'),
            F.countDistinct('schema').alias('unique_schemas')
        ).collect()

        # Convert to UserStats objects (validates field names match schema)
        users = {}
        for row in user_stats_df:
            users[row['user']] = UserStats(
                query_count=int(row['query_count']),
                heavy_rate=float(row['heavy_rate']),
                avg_cpu_seconds=float(row['avg_cpu_seconds']),
                p90_cpu_seconds=float(row['p90_cpu_seconds']),
                avg_memory_gb=float(row['avg_memory_gb']),
                p90_memory_gb=float(row['p90_memory_gb']),
                unique_catalogs=int(row['unique_catalogs']),
                unique_schemas=int(row['unique_schemas']),
                cpu_stddev=float(row['cpu_stddev'] or 0.0)  # Handle NULL stddev
            )

        logger.info(f"Computed stats for {len(users)} users")
        return users

    def _compute_catalog_stats(self, df: DataFrame) -> Dict[str, CatalogStats]:
        """
        Compute per-catalog statistics.

        Handles NULL catalogs by replacing with 'NULL' string key.
        Computes 11 metrics per catalog matching CatalogStats schema.
        """
        # Replace NULL catalog with 'NULL' string for grouping
        df_with_catalog = df.withColumn(
            'catalog_key',
            F.when(
                F.col('catalog').isNull() | (F.col('catalog') == ''),
                F.lit('NULL')
            ).otherwise(F.col('catalog'))
        )

        catalog_stats_df = df_with_catalog.groupBy('catalog_key').agg(
            # Query volume
            F.count('*').alias('query_count'),

            # Heavy rate
            F.avg(
                F.when(F.col('is_heavy') == 1, 1.0).otherwise(0.0)
            ).alias('heavy_rate'),

            # CPU time metrics (multiple percentiles)
            F.avg(F.col('cpuTime') / 1000.0).alias('avg_cpu_seconds'),
            F.expr('percentile_approx(cpuTime / 1000.0, 0.5)').alias('p50_cpu_seconds'),
            F.expr('percentile_approx(cpuTime / 1000.0, 0.75)').alias('p75_cpu_seconds'),
            F.expr('percentile_approx(cpuTime / 1000.0, 0.9)').alias('p90_cpu_seconds'),
            F.expr('percentile_approx(cpuTime / 1000.0, 0.95)').alias('p95_cpu_seconds'),

            # Memory metrics
            F.avg(F.col('peakUserMemoryBytes') / (1024**3)).alias('avg_memory_gb'),
            F.expr('percentile_approx(peakUserMemoryBytes / POW(1024, 3), 0.95)').alias('p95_memory_gb'),

            # Diversity metrics
            F.countDistinct('user').alias('unique_users'),
            F.countDistinct('schema').alias('unique_schemas')
        ).collect()

        # Convert to CatalogStats objects
        catalogs = {}
        for row in catalog_stats_df:
            catalogs[row['catalog_key']] = CatalogStats(
                query_count=int(row['query_count']),
                heavy_rate=float(row['heavy_rate']),
                avg_cpu_seconds=float(row['avg_cpu_seconds']),
                p50_cpu_seconds=float(row['p50_cpu_seconds']),
                p75_cpu_seconds=float(row['p75_cpu_seconds']),
                p90_cpu_seconds=float(row['p90_cpu_seconds']),
                p95_cpu_seconds=float(row['p95_cpu_seconds']),
                avg_memory_gb=float(row['avg_memory_gb']),
                p95_memory_gb=float(row['p95_memory_gb']),
                unique_users=int(row['unique_users']),
                unique_schemas=int(row['unique_schemas'])
            )

        logger.info(f"Computed stats for {len(catalogs)} catalogs "
                   f"(including 'NULL' for missing catalogs)")
        return catalogs

    def _compute_schema_stats(self, df: DataFrame) -> Dict[str, SchemaStats]:
        """
        Compute per-schema statistics.

        Handles NULL schemas by replacing with 'NULL' string key.
        Computes 11 metrics per schema matching SchemaStats schema.
        """
        # Replace NULL schema with 'NULL' string for grouping
        df_with_schema = df.withColumn(
            'schema_key',
            F.when(
                F.col('schema').isNull() | (F.col('schema') == ''),
                F.lit('NULL')
            ).otherwise(F.col('schema'))
        )

        schema_stats_df = df_with_schema.groupBy('schema_key').agg(
            # Query volume
            F.count('*').alias('query_count'),

            # Heavy rate
            F.avg(
                F.when(F.col('is_heavy') == 1, 1.0).otherwise(0.0)
            ).alias('heavy_rate'),

            # CPU time metrics (multiple percentiles)
            F.avg(F.col('cpuTime') / 1000.0).alias('avg_cpu_seconds'),
            F.expr('percentile_approx(cpuTime / 1000.0, 0.5)').alias('p50_cpu_seconds'),
            F.expr('percentile_approx(cpuTime / 1000.0, 0.75)').alias('p75_cpu_seconds'),
            F.expr('percentile_approx(cpuTime / 1000.0, 0.9)').alias('p90_cpu_seconds'),
            F.expr('percentile_approx(cpuTime / 1000.0, 0.95)').alias('p95_cpu_seconds'),

            # Memory metrics
            F.avg(F.col('peakUserMemoryBytes') / (1024**3)).alias('avg_memory_gb'),
            F.expr('percentile_approx(peakUserMemoryBytes / POW(1024, 3), 0.95)').alias('p95_memory_gb'),

            # Diversity metrics
            F.countDistinct('user').alias('unique_users'),
            F.countDistinct('catalog').alias('unique_catalogs')
        ).collect()

        # Convert to SchemaStats objects
        schemas = {}
        for row in schema_stats_df:
            schemas[row['schema_key']] = SchemaStats(
                query_count=int(row['query_count']),
                heavy_rate=float(row['heavy_rate']),
                avg_cpu_seconds=float(row['avg_cpu_seconds']),
                p50_cpu_seconds=float(row['p50_cpu_seconds']),
                p75_cpu_seconds=float(row['p75_cpu_seconds']),
                p90_cpu_seconds=float(row['p90_cpu_seconds']),
                p95_cpu_seconds=float(row['p95_cpu_seconds']),
                avg_memory_gb=float(row['avg_memory_gb']),
                p95_memory_gb=float(row['p95_memory_gb']),
                unique_users=int(row['unique_users']),
                unique_catalogs=int(row['unique_catalogs'])
            )

        logger.info(f"Computed stats for {len(schemas)} schemas "
                   f"(including 'NULL' for missing schemas)")
        return schemas

    def compute_and_save(
        self,
        df: DataFrame,
        s3_path: str,
        date_range: Optional[Dict[str, str]] = None,
        s3_handler=None
    ) -> HistoricalStatsSchema:
        """
        Compute stats and save to S3 in one operation.

        Args:
            df: Training data DataFrame
            s3_path: S3 path for output (e.g., s3://bucket/historical_stats_v20251005.json)
            date_range: Optional date range dict
            s3_handler: Optional S3Handler instance

        Returns:
            Computed HistoricalStatsSchema
        """
        logger.info(f"Computing and saving historical stats to {s3_path}")

        # Compute stats
        stats_schema = self.compute(df, date_range)

        # Save to S3
        stats_schema.save_to_s3(s3_path, s3_handler)

        logger.info(f"Successfully saved historical stats v{self.version} to {s3_path}")
        return stats_schema
