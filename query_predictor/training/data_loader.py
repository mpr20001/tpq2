"""
Data loading pipeline for training.

Loads Trino query logs from S3, applies labeling, filtering, and prepares
data for feature engineering.
"""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime

try:
    from pyspark.sql import DataFrame, SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType, DoubleType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False
    DataFrame = Any
    SparkSession = Any

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and prepare Trino query logs for model training.

    Pipeline stages:
    1. Load raw data from S3 parquet
    2. Apply labeling (CPU/memory/error thresholds)
    3. Apply filters (NON-CURATED, query types, dedup)

    Example:
        loader = DataLoader(spark, config)
        df_raw = loader.load_raw_data(s3_path, start_date, end_date)
        df_labeled = loader.apply_labeling(df_raw)
        df_filtered = loader.apply_filters(df_labeled)
    """

    def __init__(self, spark: SparkSession, config: Dict[str, Any]):
        """
        Initialize data loader.

        Args:
            spark: Active SparkSession
            config: Configuration dictionary with data loading settings
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required for DataLoader")

        self.spark = spark
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Extract configuration
        self.cpu_threshold = config.get('cpu_threshold_seconds', 300)
        self.memory_threshold = config.get('memory_threshold_gb', 10)
        self.heavy_error_types = config.get('heavy_error_types', [
            'EXCEEDED_TIME_LIMIT',
            'EXCEEDED_SCAN_LIMIT',
            'EXCEEDED_CPU_LIMIT'
        ])
        self.require_non_curated = config.get('require_non_curated', True)
        self.exclude_query_types = config.get('exclude_query_types', [
            'DATA_DEFINITION',
            'DESCRIBE',
            'EXPLAIN'
        ])

    def load_raw_data(
        self,
        s3_path: str,
        start_date: str,
        end_date: str,
        sample_fraction: Optional[float] = None
    ) -> DataFrame:
        """
        Load raw Trino query logs from S3 parquet.

        Args:
            s3_path: S3 base path (e.g., s3://bucket/trino-logs/)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            sample_fraction: Optional sampling fraction (0.0-1.0)

        Returns:
            PySpark DataFrame with raw query logs
        """
        self.logger.info(f"Loading raw data from {s3_path}")
        self.logger.info(f"Date range: {start_date} to {end_date}")

        # Read parquet files
        df = self.spark.read.parquet(s3_path)

        # Apply date filtering
        df = df.filter(
            (F.col('createTime') >= F.lit(start_date)) &
            (F.col('createTime') < F.lit(end_date))
        )

        # Optional sampling
        if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
            self.logger.info(f"Applying sampling: {sample_fraction:.1%}")
            df = df.sample(fraction=sample_fraction, seed=42)

        count = df.count()
        self.logger.info(f"Loaded {count:,} raw queries")

        return df

    def apply_labeling(
        self,
        df: DataFrame,
        cpu_threshold_seconds: Optional[int] = None,
        memory_threshold_gb: Optional[int] = None,
        heavy_error_types: Optional[List[str]] = None
    ) -> DataFrame:
        """
        Label queries as HEAVY (1) or SMALL (0).

        Criteria (OR logic):
        - CPU time >= threshold seconds
        - Peak memory > threshold GB
        - Error in heavy_error_types list

        Args:
            df: Input DataFrame
            cpu_threshold_seconds: CPU threshold (default from config)
            memory_threshold_gb: Memory threshold (default from config)
            heavy_error_types: List of error types that indicate heavy (default from config)

        Returns:
            DataFrame with 'is_heavy' column (0 or 1)
        """
        cpu_thresh = cpu_threshold_seconds or self.cpu_threshold
        mem_thresh = memory_threshold_gb or self.memory_threshold
        error_types = heavy_error_types or self.heavy_error_types

        self.logger.info(f"Applying labeling with CPU threshold: {cpu_thresh}s, Memory threshold: {mem_thresh}GB")

        # Convert units
        # cpuTime is in milliseconds, convert to seconds
        df = df.withColumn(
            'cpu_time_seconds',
            F.when(F.col('cpuTime').isNotNull(), F.col('cpuTime') / 1000.0).otherwise(0.0)
        )

        # peakUserMemoryBytes is in bytes, convert to GB
        df = df.withColumn(
            'memory_gb',
            F.when(F.col('peakUserMemoryBytes').isNotNull(),
                   F.col('peakUserMemoryBytes') / (1024**3)).otherwise(0.0)
        )

        # Apply labeling criteria
        df = df.withColumn(
            'is_heavy',
            F.when(
                (F.col('cpu_time_seconds') >= cpu_thresh) |
                (F.col('memory_gb') > mem_thresh) |
                (F.col('errorName').isin(error_types)),
                1
            ).otherwise(0).cast(IntegerType())
        )

        # Log distribution
        heavy_count = df.filter(F.col('is_heavy') == 1).count()
        total_count = df.count()
        heavy_pct = (heavy_count / total_count * 100) if total_count > 0 else 0

        self.logger.info(f"Labeling complete: {heavy_count:,} heavy ({heavy_pct:.2f}%), "
                        f"{total_count - heavy_count:,} small ({100-heavy_pct:.2f}%)")

        return df

    def apply_filters(self, df: DataFrame) -> DataFrame:
        """
        Apply data quality filters.

        Filters applied:
        1. NON-CURATED queries only (if enabled)
        2. Exclude specific query types (DDL, DESCRIBE, EXPLAIN)
        3. Remove empty/invalid SQL
        4. Deduplicate by queryId (keep first occurrence)

        Args:
            df: Input DataFrame with labels

        Returns:
            Filtered DataFrame
        """
        self.logger.info("Applying data quality filters")
        initial_count = df.count()

        # Filter 1: NON-CURATED queries only
        if self.require_non_curated:
            df = df.filter(F.col('clientInfo').contains('NON-CURATED'))
            after_curated = df.count()
            self.logger.info(f"After NON-CURATED filter: {after_curated:,} queries "
                           f"(removed {initial_count - after_curated:,})")

        # Filter 2: Exclude query types
        if self.exclude_query_types:
            df = df.filter(~F.col('queryType').isin(self.exclude_query_types))
            after_query_type = df.count()
            self.logger.info(f"After query type filter: {after_query_type:,} queries")

        # Filter 3: Remove empty/invalid SQL
        df = df.filter(
            F.col('query').isNotNull() &
            (F.length(F.trim(F.col('query'))) > 0)
        )
        after_null = df.count()
        self.logger.info(f"After null/empty filter: {after_null:,} queries")

        # Filter 4: Deduplicate by queryId (keep first)
        df = df.dropDuplicates(['queryId'])
        final_count = df.count()

        self.logger.info(f"After deduplication: {final_count:,} queries")
        self.logger.info(f"Total removed: {initial_count - final_count:,} "
                        f"({(initial_count - final_count) / initial_count * 100:.1f}%)")

        return df

    def validate_schema(self, df: DataFrame) -> bool:
        """
        Validate that required columns exist in DataFrame.

        Args:
            df: DataFrame to validate

        Returns:
            True if all required columns exist

        Raises:
            ValueError: If required columns are missing
        """
        required_columns = [
            'queryId', 'query', 'user', 'catalog', 'schema',
            'cpuTime', 'peakUserMemoryBytes', 'errorName',
            'queryType', 'createTime', 'endTime', 'clientInfo'
        ]

        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        self.logger.info(f"Schema validation passed: all {len(required_columns)} required columns present")
        return True

    def get_summary_stats(self, df: DataFrame) -> Dict[str, Any]:
        """
        Get summary statistics for a DataFrame.

        Args:
            df: DataFrame with 'is_heavy' column

        Returns:
            Dictionary with count, heavy count, small count, and ratio
        """
        total = df.count()
        heavy = df.filter(F.col('is_heavy') == 1).count()
        small = total - heavy
        ratio = small / heavy if heavy > 0 else 0

        return {
            'total_queries': total,
            'heavy_queries': heavy,
            'small_queries': small,
            'heavy_percentage': (heavy / total * 100) if total > 0 else 0,
            'small_to_heavy_ratio': ratio
        }
