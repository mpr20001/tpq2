"""
Historical feature extractor.

Extracts 17 features from precomputed historical statistics:
- 6 user features
- 6 catalog features
- 4 schema features
- 1 cold-start indicator

Uses shared HistoricalStatsSchema to ensure train-serve parity.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING, Callable

from query_predictor.core.featurizer.base import BaseFeatureExtractor
from query_predictor.core.featurizer import constants, utils
from query_predictor.core.types.historical_stats import HistoricalStatsSchema

# Optional PySpark dependency for distributed feature extraction
try:
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, FloatType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

if TYPE_CHECKING:
    from query_predictor.core.types.query_data import QueryData
    from query_predictor.core.types.feature_spec import FeatureSpec

logger = logging.getLogger(__name__)


class HistoricalFeatureExtractor(BaseFeatureExtractor):
    """
    Extract historical features from precomputed statistics.

    Features:
    - Uses shared HistoricalStatsSchema (train-serve parity)
    - Cold-start handling for unknown entities
    - NULL catalog/schema handling with 'NULL' key
    - Shared normalization constants

    Example:
        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=stats_dict
        )
        features = extractor.extract(query_data)
        assert len(features) == 17
    """

    def __init__(
        self,
        config: Dict[str, Any],
        historical_stats: Dict[str, Any],
        feature_spec: Optional['FeatureSpec'] = None
    ):
        """
        Initialize historical feature extractor.

        Args:
            config: Configuration dictionary
            historical_stats: Historical stats dict (from HistoricalStatsSchema.to_dict())
            feature_spec: Optional FeatureSpec for enable/disable control
        """
        super().__init__(config, feature_spec)

        # Load and validate historical stats using shared schema
        self.stats_schema = HistoricalStatsSchema.from_dict(historical_stats)

        logger.info(f"Initialized HistoricalFeatureExtractor v{self.stats_schema.version}")
        logger.info(f"Historical stats: {len(self.stats_schema.users)} users, "
                   f"{len(self.stats_schema.catalogs)} catalogs, "
                   f"{len(self.stats_schema.schemas)} schemas")

    def extract(self, query_data: 'QueryData') -> List[float]:
        """
        Extract 17 historical features.

        Args:
            query_data: QueryData with user, catalog, schema

        Returns:
            List of 17 float features
        """
        try:
            user = query_data.user
            catalog = query_data.catalog  # Can be None
            schema = query_data.schema    # Can be None

            # Get stats (returns None if not found)
            user_stats = self.stats_schema.get_user_stats(user)
            catalog_stats = self.stats_schema.get_catalog_stats(catalog)
            schema_stats = self.stats_schema.get_schema_stats(schema)

            # Determine if cold-start (all three are unknown)
            is_cold_start = (
                user_stats is None and
                catalog_stats is None and
                schema_stats is None
            )

            # Extract feature groups
            user_features = self._extract_user_features(user_stats)
            catalog_features = self._extract_catalog_features(catalog_stats)
            schema_features = self._extract_schema_features(schema_stats)
            cold_start_feature = [float(is_cold_start)]

            # Combine all features (order matters for train-serve parity!)
            features = (
                user_features +
                catalog_features +
                schema_features +
                cold_start_feature
            )

            assert len(features) == 17, f"Expected 17 features, got {len(features)}"
            return features

        except Exception as e:
            logger.error(f"Error extracting historical features: {e}", exc_info=True)
            # Return safe defaults on error
            return [0.0] * 17

    def _extract_user_features(self, user_stats) -> List[float]:
        """
        Extract 6 user features.

        Features:
        1. query_count (normalized)
        2. heavy_rate (0-1)
        3. avg_cpu_seconds (normalized)
        4. p90_cpu_seconds (normalized)
        5. avg_memory_gb (normalized)
        6. catalog_diversity (unique_catalogs normalized)
        """
        if user_stats is None:
            # Cold-start: use shared defaults
            defaults = constants.COLD_START_DEFAULTS
            return [
                utils.normalize_value(defaults['query_count'],
                                     constants.HISTORICAL_QUERY_COUNT_NORMALIZER),
                defaults['heavy_rate'],
                utils.normalize_value(defaults['avg_cpu_seconds'],
                                     constants.HISTORICAL_CPU_SECONDS_NORMALIZER),
                utils.normalize_value(defaults['p90_cpu_seconds'],
                                     constants.HISTORICAL_CPU_SECONDS_NORMALIZER),
                utils.normalize_value(defaults['avg_memory_gb'],
                                     constants.HISTORICAL_MEMORY_GB_NORMALIZER),
                utils.normalize_value(defaults['unique_catalogs'],
                                     constants.HISTORICAL_DIVERSITY_NORMALIZER)
            ]

        # Use actual user stats (field names match UserStats schema exactly!)
        return [
            utils.normalize_value(user_stats.query_count,
                                 constants.HISTORICAL_QUERY_COUNT_NORMALIZER),
            user_stats.heavy_rate,
            utils.normalize_value(user_stats.avg_cpu_seconds,
                                 constants.HISTORICAL_CPU_SECONDS_NORMALIZER),
            utils.normalize_value(user_stats.p90_cpu_seconds,
                                 constants.HISTORICAL_CPU_SECONDS_NORMALIZER),
            utils.normalize_value(user_stats.avg_memory_gb,
                                 constants.HISTORICAL_MEMORY_GB_NORMALIZER),
            utils.normalize_value(user_stats.unique_catalogs,
                                 constants.HISTORICAL_DIVERSITY_NORMALIZER)
        ]

    def _extract_catalog_features(self, catalog_stats) -> List[float]:
        """
        Extract 6 catalog features.

        Features:
        1. query_count (normalized)
        2. heavy_rate (0-1)
        3. avg_cpu_seconds (normalized)
        4. p90_cpu_seconds (normalized)
        5. avg_memory_gb (normalized)
        6. user_diversity (unique_users normalized)
        """
        if catalog_stats is None:
            # Cold-start: use shared defaults
            defaults = constants.COLD_START_DEFAULTS
            return [
                utils.normalize_value(defaults['query_count'],
                                     constants.HISTORICAL_QUERY_COUNT_NORMALIZER),
                defaults['heavy_rate'],
                utils.normalize_value(defaults['avg_cpu_seconds'],
                                     constants.HISTORICAL_CPU_SECONDS_NORMALIZER),
                utils.normalize_value(defaults['p90_cpu_seconds'],
                                     constants.HISTORICAL_CPU_SECONDS_NORMALIZER),
                utils.normalize_value(defaults['avg_memory_gb'],
                                     constants.HISTORICAL_MEMORY_GB_NORMALIZER),
                utils.normalize_value(defaults['unique_users'],
                                     constants.HISTORICAL_DIVERSITY_NORMALIZER)
            ]

        # Use actual catalog stats (field names match CatalogStats schema exactly!)
        return [
            utils.normalize_value(catalog_stats.query_count,
                                 constants.HISTORICAL_QUERY_COUNT_NORMALIZER),
            catalog_stats.heavy_rate,
            utils.normalize_value(catalog_stats.avg_cpu_seconds,
                                 constants.HISTORICAL_CPU_SECONDS_NORMALIZER),
            utils.normalize_value(catalog_stats.p90_cpu_seconds,
                                 constants.HISTORICAL_CPU_SECONDS_NORMALIZER),
            utils.normalize_value(catalog_stats.avg_memory_gb,
                                 constants.HISTORICAL_MEMORY_GB_NORMALIZER),
            utils.normalize_value(catalog_stats.unique_users,
                                 constants.HISTORICAL_DIVERSITY_NORMALIZER)
        ]

    def _extract_schema_features(self, schema_stats) -> List[float]:
        """
        Extract 4 schema features.

        Features:
        1. query_count (normalized)
        2. heavy_rate (0-1)
        3. avg_cpu_seconds (normalized)
        4. avg_memory_gb (normalized)
        """
        if schema_stats is None:
            # Cold-start: use shared defaults
            defaults = constants.COLD_START_DEFAULTS
            return [
                utils.normalize_value(defaults['query_count'],
                                     constants.HISTORICAL_QUERY_COUNT_NORMALIZER),
                defaults['heavy_rate'],
                utils.normalize_value(defaults['avg_cpu_seconds'],
                                     constants.HISTORICAL_CPU_SECONDS_NORMALIZER),
                utils.normalize_value(defaults['avg_memory_gb'],
                                     constants.HISTORICAL_MEMORY_GB_NORMALIZER)
            ]

        # Use actual schema stats (field names match SchemaStats schema exactly!)
        return [
            utils.normalize_value(schema_stats.query_count,
                                 constants.HISTORICAL_QUERY_COUNT_NORMALIZER),
            schema_stats.heavy_rate,
            utils.normalize_value(schema_stats.avg_cpu_seconds,
                                 constants.HISTORICAL_CPU_SECONDS_NORMALIZER),
            utils.normalize_value(schema_stats.avg_memory_gb,
                                 constants.HISTORICAL_MEMORY_GB_NORMALIZER)
        ]

    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of 17 feature names.

        Order MUST match extract() output for train-serve parity!
        """
        return [
            # User features (6)
            'user_query_count_norm',
            'user_heavy_rate',
            'user_avg_cpu_seconds_norm',
            'user_p90_cpu_seconds_norm',
            'user_avg_memory_gb_norm',
            'user_catalog_diversity_norm',

            # Catalog features (6)
            'catalog_query_count_norm',
            'catalog_heavy_rate',
            'catalog_avg_cpu_seconds_norm',
            'catalog_p90_cpu_seconds_norm',
            'catalog_avg_memory_gb_norm',
            'catalog_user_diversity_norm',

            # Schema features (4)
            'schema_query_count_norm',
            'schema_heavy_rate',
            'schema_avg_cpu_seconds_norm',
            'schema_avg_memory_gb_norm',

            # Cold-start indicator (1)
            'is_cold_start'
        ]

    def create_spark_udf(self) -> Callable:
        """
        Create PySpark UDF for distributed historical feature extraction.

        Returns:
            UDF that transforms (user, catalog, schema) struct → 17 historical features

        Raises:
            ImportError: If PySpark not available

        Example:
            extractor = HistoricalFeatureExtractor(config, historical_stats)
            udf = extractor.create_spark_udf()
            df = df.withColumn('historical_features', udf(F.struct(
                F.col('user'), F.col('catalog'), F.col('schema')
            )))
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required for create_spark_udf()")

        # Capture stats_schema in closure for thread safety
        stats_schema = self.stats_schema

        @F.udf(returnType=ArrayType(FloatType()))
        def historical_udf(row) -> List[float]:
            """Transform user/catalog/schema to historical features."""
            if row is None:
                return [0.0] * 17

            try:
                # Extract fields from Row
                user = row.user if hasattr(row, 'user') else ''
                catalog = row.catalog if hasattr(row, 'catalog') else None
                schema = row.schema if hasattr(row, 'schema') else None

                # Get stats (returns None if not found)
                user_stats = stats_schema.get_user_stats(user)
                catalog_stats = stats_schema.get_catalog_stats(catalog)
                schema_stats = stats_schema.get_schema_stats(schema)

                # Determine if cold-start
                is_cold_start = (
                    user_stats is None and
                    catalog_stats is None and
                    schema_stats is None
                )

                # Extract feature groups (using private methods)
                user_features = self._extract_user_features(user_stats)
                catalog_features = self._extract_catalog_features(catalog_stats)
                schema_features = self._extract_schema_features(schema_stats)
                cold_start_feature = [float(is_cold_start)]

                # Combine all features
                features = (
                    user_features +
                    catalog_features +
                    schema_features +
                    cold_start_feature
                )

                return features

            except Exception as e:
                # Return zero features on error
                logger.error(f"Error in historical feature extraction UDF: {e}")
                return [0.0] * 17

        return historical_udf
