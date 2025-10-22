"""Query type feature extractor - 5 features."""

import logging
from typing import List, Dict, Any

from query_predictor.core.featurizer.base import BaseFeatureExtractor
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer import utils

logger = logging.getLogger(__name__)


class QueryTypeExtractor(BaseFeatureExtractor):
    """
    Extract query type features (5 features).

    Features:
    - is_select_query, is_insert_query, is_create_table_as
    - is_delete_query, is_update_query
    """

    FEATURE_NAMES = [
        'is_select_query', 'is_insert_query', 'is_create_table_as',
        'is_delete_query', 'is_update_query'
    ]

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def extract(self, query_data: QueryData) -> List[float]:
        query = query_data.query

        if not query:
            return [0.0] * len(self.FEATURE_NAMES)

        try:
            features = []
            query_upper = utils.safe_upper(query).strip()

            features.append(1.0 if query_upper.startswith('SELECT') else 0.0)  # is_select_query
            features.append(1.0 if query_upper.startswith('INSERT') else 0.0)  # is_insert_query
            features.append(1.0 if 'CREATE TABLE' in query_upper and 'AS SELECT' in query_upper else 0.0)  # is_create_table_as
            features.append(1.0 if query_upper.startswith('DELETE') else 0.0)  # is_delete_query
            features.append(1.0 if query_upper.startswith('UPDATE') else 0.0)  # is_update_query

            return features

        except Exception as e:
            logger.error(f"Error extracting query type features: {e}")
            return [0.0] * len(self.FEATURE_NAMES)
