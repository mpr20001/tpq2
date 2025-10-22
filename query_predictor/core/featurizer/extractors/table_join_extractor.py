"""Table and join feature extractor - 12 features."""

import re
import logging
from typing import List, Dict, Any

from query_predictor.core.featurizer.base import BaseFeatureExtractor
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer import utils
from query_predictor.core.featurizer.parsers import SQLPatternParser

logger = logging.getLogger(__name__)


class TableJoinExtractor(BaseFeatureExtractor):
    """
    Extract table and join features (12 features).

    Features:
    - table_count, join_count, left_join_count, inner_join_count
    - cross_join_present, full_outer_join_present, self_join_count
    - with_clause_count, nested_with_depth, recursive_cte_present
    - subquery_in_from_count, correlated_subquery_likely
    """

    FEATURE_NAMES = [
        'table_count', 'join_count', 'left_join_count', 'inner_join_count',
        'cross_join_present', 'full_outer_join_present', 'self_join_count',
        'with_clause_count', 'nested_with_depth', 'recursive_cte_present',
        'subquery_in_from_count', 'correlated_subquery_likely'
    ]

    def __init__(self, config: Dict[str, Any], feature_spec=None):
        super().__init__(config, feature_spec)
        self.sql_parser = SQLPatternParser()

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def extract(self, query_data: QueryData) -> List[float]:
        query = query_data.query

        if not query:
            return [0.0] * len(self.FEATURE_NAMES)

        try:
            features = []
            query_upper = utils.safe_upper(query)

            # Table count using parser
            tables = self.sql_parser.extract_tables(query)
            features.append(float(len(tables)))  # table_count

            # Join counts
            features.append(float(utils.safe_count(query_upper, 'JOIN')))  # join_count
            features.append(float(utils.safe_count(query_upper, 'LEFT JOIN')))  # left_join_count
            features.append(float(utils.safe_count(query_upper, 'INNER JOIN')))  # inner_join_count
            features.append(1.0 if 'CROSS JOIN' in query_upper else 0.0)  # cross_join_present
            features.append(1.0 if 'FULL OUTER JOIN' in query_upper else 0.0)  # full_outer_join_present

            # Self join detection (simplified)
            features.append(0.0)  # self_join_count (would need proper parsing)

            # WITH clause features
            features.append(float(utils.safe_count(query_upper, 'WITH')))  # with_clause_count
            features.append(0.0)  # nested_with_depth (would need proper parsing)
            features.append(1.0 if 'RECURSIVE' in query_upper else 0.0)  # recursive_cte_present

            # Subquery features
            features.append(float(utils.safe_count(query_upper, 'FROM (')))  # subquery_in_from_count

            # Correlated subquery detection (simplified)
            where_pos = query_upper.find('WHERE')
            if where_pos != -1 and 'SELECT' in query_upper[where_pos:]:
                features.append(1.0)  # correlated_subquery_likely
            else:
                features.append(0.0)

            return features

        except Exception as e:
            logger.error(f"Error extracting table/join features: {e}")
            return [0.0] * len(self.FEATURE_NAMES)
