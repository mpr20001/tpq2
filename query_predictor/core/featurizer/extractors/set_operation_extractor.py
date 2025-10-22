"""Set operation feature extractor - 4 features."""

import logging
from typing import List, Dict, Any

from query_predictor.core.featurizer.base import BaseFeatureExtractor
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer import utils

logger = logging.getLogger(__name__)


class SetOperationExtractor(BaseFeatureExtractor):
    """
    Extract set operation features (4 features).

    Features:
    - except_count, intersect_count, set_operation_total, has_multiple_statements
    """

    FEATURE_NAMES = [
        'except_count', 'intersect_count', 'set_operation_total',
        'has_multiple_statements'
    ]

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def extract(self, query_data: QueryData) -> List[float]:
        query = query_data.query

        if not query:
            return [0.0] * len(self.FEATURE_NAMES)

        try:
            features = []
            query_upper = utils.safe_upper(query)

            except_count = utils.safe_count(query_upper, 'EXCEPT')
            intersect_count = utils.safe_count(query_upper, 'INTERSECT')
            union_count = utils.safe_count(query_upper, 'UNION')

            features.append(float(except_count))  # except_count
            features.append(float(intersect_count))  # intersect_count
            features.append(float(except_count + intersect_count + union_count))  # set_operation_total
            features.append(1.0 if utils.safe_count(query, ';') > 1 else 0.0)  # has_multiple_statements

            return features

        except Exception as e:
            logger.error(f"Error extracting set operation features: {e}")
            return [0.0] * len(self.FEATURE_NAMES)
