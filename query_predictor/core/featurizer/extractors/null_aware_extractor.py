"""NULL-aware feature extractor - 6 features."""

import logging
from typing import List, Dict, Any

from query_predictor.core.featurizer.base import BaseFeatureExtractor
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer.parsers import SQLPatternParser

logger = logging.getLogger(__name__)


class NullAwareExtractor(BaseFeatureExtractor):
    """
    Extract NULL-aware features for handling NULL catalogs/schemas (6 features).

    Handles the reality that many queries have NULL catalog (3.8M) or NULL schema (4.1M).
    Extracts information from SQL text when metadata is NULL.

    Features:
    - is_catalog_null, is_schema_null
    - has_qualified_tables
    - inferred_catalog_count, inferred_schema_count
    - table_qualification_level
    """

    FEATURE_NAMES = [
        'is_catalog_null', 'is_schema_null', 'has_qualified_tables',
        'inferred_catalog_count', 'inferred_schema_count',
        'table_qualification_level'
    ]

    def __init__(self, config: Dict[str, Any], feature_spec=None):
        super().__init__(config, feature_spec)
        self.sql_parser = SQLPatternParser()

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def extract(self, query_data: QueryData) -> List[float]:
        try:
            features = []

            # Check if catalog/schema are NULL
            catalog = query_data.catalog
            schema = query_data.schema

            features.append(1.0 if catalog is None or catalog == '' else 0.0)  # is_catalog_null
            features.append(1.0 if schema is None or schema == '' else 0.0)  # is_schema_null

            # Extract qualified table references from query
            query = query_data.query
            if not query:
                features.extend([0.0, 0.0, 0.0, 0.0])
                return features

            # Get qualified tables
            fully_qualified, schema_qualified = self.sql_parser.extract_qualified_tables(query)

            # Check if has qualified tables
            has_qualified = 1.0 if (len(fully_qualified) > 0 or len(schema_qualified) > 0) else 0.0
            features.append(has_qualified)  # has_qualified_tables

            # Infer catalogs and schemas from query
            inferred_catalogs, inferred_schemas = self.sql_parser.infer_catalog_schema(query)

            features.append(float(len(inferred_catalogs)))  # inferred_catalog_count
            features.append(float(len(inferred_schemas)))  # inferred_schema_count

            # Calculate qualification level
            # 0.0 = unqualified only
            # 1.0 = schema-qualified
            # 2.0 = fully-qualified
            all_tables = self.sql_parser.extract_tables(query)
            total_tables = len(all_tables)

            if total_tables > 0:
                if len(fully_qualified) > 0:
                    qualification_level = 2.0
                elif len(schema_qualified) > 0:
                    qualification_level = 1.0
                else:
                    qualification_level = 0.0
            else:
                qualification_level = 0.0

            features.append(qualification_level)  # table_qualification_level

            return features

        except Exception as e:
            logger.error(f"Error extracting NULL-aware features: {e}")
            return [0.0] * len(self.FEATURE_NAMES)
