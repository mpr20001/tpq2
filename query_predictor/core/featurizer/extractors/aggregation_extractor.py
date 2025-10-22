"""Aggregation feature extractor"""

import re
import logging
from typing import List, Dict, Any

from query_predictor.core.featurizer.base import BaseFeatureExtractor
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer import utils

logger = logging.getLogger(__name__)


class AggregationExtractor(BaseFeatureExtractor):
    """
    Extract aggregation features.

    Features:
    - group_by_count, having_clause_present
    - count_function_count, sum_avg_function_count, min_max_function_count
    - distinct_aggregation_count, window_function_count, unbounded_window_present
    """

    FEATURE_NAMES = [
        'group_by_count', 'having_clause_present', 'count_function_count',
        'sum_avg_function_count', 'min_max_function_count',
        'distinct_aggregation_count', 'window_function_count',
        'unbounded_window_present'
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

            # GROUP BY features
            group_by_match = re.search(r'GROUP BY\s+([^;]+?)(?:HAVING|ORDER BY|LIMIT|$)',
                                     query_upper, re.DOTALL)
            if group_by_match:
                group_by_cols = len(group_by_match.group(1).split(','))
            else:
                group_by_cols = 0
            features.append(float(group_by_cols))  # group_by_count

            features.append(1.0 if 'HAVING' in query_upper else 0.0)  # having_clause_present

            # Aggregation functions
            features.append(float(utils.safe_count(query_upper, 'COUNT(')))  # count_function_count
            sum_avg_count = utils.safe_count(query_upper, 'SUM(') + utils.safe_count(query_upper, 'AVG(')
            features.append(float(sum_avg_count))  # sum_avg_function_count
            min_max_count = utils.safe_count(query_upper, 'MIN(') + utils.safe_count(query_upper, 'MAX(')
            features.append(float(min_max_count))  # min_max_function_count
            features.append(float(utils.safe_count(query_upper, 'DISTINCT')))  # distinct_aggregation_count

            # Window functions
            features.append(float(utils.safe_count(query_upper, 'OVER (')))  # window_function_count
            features.append(1.0 if 'UNBOUNDED' in query_upper else 0.0)  # unbounded_window_present

            return features

        except Exception as e:
            logger.error(f"Error extracting aggregation features: {e}")
            return [0.0] * len(self.FEATURE_NAMES)
