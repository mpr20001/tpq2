"""
Main feature extractor orchestrator.

Combines all specialized extractors to produce complete feature vectors.
"""

import logging
from typing import List, Dict, Any, Optional, Callable

from query_predictor.core.types.feature_spec import FeatureSpec
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer.parsers import ASTParser, SQLPatternParser
from query_predictor.core.featurizer.extractors.sql_extractor import SQLFeatureExtractor
from query_predictor.core.featurizer.extractors.table_join_extractor import TableJoinExtractor
from query_predictor.core.featurizer.extractors.where_extractor import WhereClauseExtractor
from query_predictor.core.featurizer.extractors.aggregation_extractor import AggregationExtractor
from query_predictor.core.featurizer.extractors.ast_extractor import ASTFeatureExtractor
from query_predictor.core.featurizer.extractors.context_extractor import ContextExtractor
from query_predictor.core.featurizer.extractors.query_type_extractor import QueryTypeExtractor
from query_predictor.core.featurizer.extractors.set_operation_extractor import SetOperationExtractor
from query_predictor.core.featurizer.extractors.null_aware_extractor import NullAwareExtractor
from query_predictor.core.featurizer.extractors.historical_extractor import HistoricalFeatureExtractor
from query_predictor.core.featurizer import constants
from query_predictor.common import validators

# Optional PySpark dependency for distributed feature extraction
try:
    from pyspark.sql import functions as F
    from pyspark.sql.types import ArrayType, FloatType
    PYSPARK_AVAILABLE = True
except ImportError:
    PYSPARK_AVAILABLE = False

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Main feature extractor orchestrator.

    Coordinates all specialized extractors to produce complete feature vectors.
    - 78 features (base) when historical features disabled
    - 95 features (base + historical) when historical features enabled

    Features:
    - Dependency injection for all components
    - FeatureSpec integration for enable/disable
    - Per-extractor error isolation
    - Comprehensive logging
    - Conditional historical features support
    - Support for PySpark UDFs (future)
    """

    VERSION = "3.0.0"

    def __init__(self,
                 config: Dict[str, Any],
                 historical_stats: Optional[Dict] = None,
                 feature_spec: Optional[FeatureSpec] = None):
        """
        Initialize feature extractor.

        Args:
            config: Configuration dictionary
            historical_stats: Historical statistics (for next PR)
            feature_spec: Optional FeatureSpec for feature control
        """
        self.config = config
        self.historical_stats = historical_stats or {}
        self.feature_spec = feature_spec

        # Initialize parsers (shared across extractors)
        self.ast_parser = ASTParser(timeout_ms=config.get('ast_timeout_ms', 50))
        self.sql_parser = SQLPatternParser()

        # Initialize extractors
        self.extractors = self._initialize_extractors()

        # Build feature spec if not provided
        if self.feature_spec is None:
            self.feature_spec = FeatureSpec.from_extractors(self.extractors, self.VERSION)

        # Cache feature names
        self.feature_names = self.feature_spec.active_feature_names

        logger.info(f"FeatureExtractor v{self.VERSION} initialized with "
                   f"{len(self.feature_names)} active features")

    def _initialize_extractors(self) -> List:
        """
        Initialize all extractors in correct order.

        Order is critical for train-serve parity!
        Returns 9 extractors (78 features) or 10 extractors (95 features).
        """
        extractors = [
            SQLFeatureExtractor(self.config, self.feature_spec),
            TableJoinExtractor(self.config, self.feature_spec),
            WhereClauseExtractor(self.config, self.feature_spec),
            AggregationExtractor(self.config, self.feature_spec),
            ASTFeatureExtractor(self.config, self.feature_spec, self.ast_parser),
            ContextExtractor(self.config, self.feature_spec),
            QueryTypeExtractor(self.config, self.feature_spec),
            SetOperationExtractor(self.config, self.feature_spec),
            NullAwareExtractor(self.config, self.feature_spec),
        ]

        # Conditionally add historical features (17 features)
        # Total becomes 95 features when enabled
        if self.config.get('enable_historical_features', False):
            if not self.historical_stats:
                logger.warning("Historical features enabled but no historical_stats provided. "
                             "Skipping HistoricalFeatureExtractor.")
            else:
                logger.info("Adding HistoricalFeatureExtractor (17 features)")
                extractors.append(
                    HistoricalFeatureExtractor(
                        self.config,
                        self.historical_stats,
                        self.feature_spec
                    )
                )

        return extractors

    def _validate_input(self, query_data: Dict[str, Any]) -> str:
        """
        Validate input query data and return query string.
        
        Args:
            query_data: Dictionary with query, user, catalog, schema, hour, etc.
            
        Returns:
            The validated query string
            
        Raises:
            ValueError: If validation fails
        """
        # Validate input type
        validators.validate_query_data_dict(query_data)
        
        # Extract and validate query field
        query = query_data.get('query', '')
        validators.validate_query_not_empty(query)
        validators.validate_query_length(query, constants.MAX_QUERY_LENGTH)
        
        return query

    def extract(self, query_data: Dict[str, Any]) -> List[float]:
        """
        Extract features from query data.

        Args:
            query_data: Dictionary with query, user, catalog, schema, hour, etc.

        Returns:
            List of float features:
            - 78 features if historical features disabled
            - 95 features if historical features enabled

        Raises:
            ValueError: If input validation fails (query empty, too long, or invalid type)

        Note:
            Handles errors per-extractor to prevent cascade failures.
        """
        # Validate input
        query = self._validate_input(query_data)

        # Convert dict to QueryData for type safety and validation
        try:
            query_data_obj = QueryData(
                query=query,
                user=query_data.get('user', ''),
                catalog=query_data.get('catalog'),
                schema=query_data.get('schema'),
                hour=query_data.get('hour', 12),
                client_info=query_data.get('clientInfo', '') or query_data.get('client_info', ''),
                session_properties=query_data.get('sessionProperties', {}) or query_data.get('session_properties', {})
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Invalid query data: {e}")
            raise ValueError(f"Failed to construct QueryData: {e}")

        features = []

        try:
            for extractor in self.extractors:
                try:
                    # Use safe extraction with spec filtering
                    extractor_features = extractor.extract_with_spec(query_data_obj)
                    features.extend(extractor_features)

                except Exception as e:
                    # Log error but continue with zero features
                    logger.error(f"Error in {extractor.__class__.__name__}: {e}")
                    zero_features = [0.0] * extractor.feature_count
                    features.extend(zero_features)

            # Validate total feature count
            expected_count = len(self.feature_names)
            actual_count = len(features)

            if actual_count != expected_count:
                logger.warning(f"Feature count mismatch: expected {expected_count}, got {actual_count}")
                # Pad or truncate to match expected
                if actual_count < expected_count:
                    features.extend([0.0] * (expected_count - actual_count))
                else:
                    features = features[:expected_count]

            return features

        except Exception as e:
            logger.error(f"Critical error in feature extraction: {e}", exc_info=True)
            # Return all zeros as fallback
            return [0.0] * len(self.feature_names)

    def get_feature_importance_report(self) -> Dict[str, Any]:
        """
        Get report of feature importance and disabled features.

        Returns:
            Dictionary with feature importance information
        """
        if not self.feature_spec or not self.feature_spec.feature_importance:
            return {"error": "No feature importance data available"}

        # Sort by importance
        sorted_features = sorted(
            self.feature_spec.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return {
            'version': self.VERSION,
            'total_features': self.feature_spec.total_feature_count,
            'active_features': self.feature_spec.feature_count,
            'disabled_features': list(self.feature_spec.disabled_features),
            'top_10_features': sorted_features[:10],
            'bottom_10_features': sorted_features[-10:],
            'feature_groups': {
                name: {
                    'enabled': group.enabled,
                    'feature_count': len(group.feature_names)
                }
                for name, group in self.feature_spec.feature_groups.items()
            }
        }

    def get_extractor_summary(self) -> Dict[str, Any]:
        """
        Get summary of all extractors and their feature counts.

        Returns:
            Dictionary with extractor information
        """
        return {
            'version': self.VERSION,
            'total_extractors': len(self.extractors),
            'extractors': [
                {
                    'name': ext.__class__.__name__,
                    'feature_count': ext.feature_count,
                    'features': ext.get_active_feature_names()
                }
                for ext in self.extractors
            ],
            'total_active_features': len(self.feature_names)
        }

    @property
    def feature_count(self) -> int:
        """Number of features produced."""
        return len(self.feature_names)

    def create_spark_udf(self) -> Callable:
        """
        Create PySpark UDF for distributed feature extraction.

        Returns:
            UDF that transforms query data struct → feature array (List[float])

        Raises:
            ImportError: If PySpark not available

        Example:
            extractor = FeatureExtractor(config)
            udf = extractor.create_spark_udf()
            df = df.withColumn('features', udf(F.struct(
                F.col('query'), F.col('user'), F.col('catalog'),
                F.col('schema'), F.col('hour'), F.col('clientInfo')
            )))
        """
        if not PYSPARK_AVAILABLE:
            raise ImportError("PySpark is required for create_spark_udf()")

        # Capture self.extract and feature_count in closure
        extract_func = self.extract
        expected_feature_count = self.feature_count

        @F.udf(returnType=ArrayType(FloatType()))
        def feature_udf(row) -> List[float]:
            """Transform PySpark Row to features."""
            if row is None:
                return [0.0] * expected_feature_count

            try:
                # Convert PySpark Row to dict
                query_data = {
                    'query': row.query if hasattr(row, 'query') else '',
                    'user': row.user if hasattr(row, 'user') else '',
                    'catalog': row.catalog if hasattr(row, 'catalog') else None,
                    'schema': row.schema if hasattr(row, 'schema') else None,
                    'hour': row.hour if hasattr(row, 'hour') else 12,
                    'clientInfo': row.clientInfo if hasattr(row, 'clientInfo') else ''
                }

                # Extract features using production logic
                features = extract_func(query_data)
                return features

            except Exception as e:
                # Return zero features on error
                logger.error(f"Error in feature extraction UDF: {e}")
                return [0.0] * expected_feature_count

        return feature_udf
