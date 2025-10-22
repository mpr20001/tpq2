"""WHERE clause feature extractor - 10 features."""

import logging
from typing import List, Dict, Any

from query_predictor.core.featurizer.base import BaseFeatureExtractor
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer import utils
from query_predictor.core.featurizer.parsers import SQLPatternParser

logger = logging.getLogger(__name__)


class WhereClauseExtractor(BaseFeatureExtractor):
    """
    Extract WHERE clause features (10 features).

    Features:
    - where_condition_count, where_and_count, where_or_count, where_not_count
    - in_clause_count, in_list_total_items, max_in_list_size
    - between_count, like_pattern_count, exists_count
    """

    FEATURE_NAMES = [
        'where_condition_count', 'where_and_count', 'where_or_count',
        'where_not_count', 'in_clause_count', 'in_list_total_items',
        'max_in_list_size', 'between_count', 'like_pattern_count', 'exists_count'
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

            # Extract WHERE clause
            where_clause = self.sql_parser.extract_where_clause(query)
            where_upper = utils.safe_upper(where_clause)

            # Condition counts
            condition_count = (utils.safe_count(where_clause, '=') +
                             utils.safe_count(where_clause, '<') +
                             utils.safe_count(where_clause, '>'))
            features.append(float(condition_count))  # where_condition_count
            features.append(float(utils.safe_count(where_upper, ' AND ')))  # where_and_count
            features.append(float(utils.safe_count(where_upper, ' OR ')))  # where_or_count
            features.append(float(utils.safe_count(where_upper, ' NOT ')))  # where_not_count

            # IN clause features
            in_clauses = self.sql_parser.extract_in_clauses(where_clause)
            features.append(float(len(in_clauses)))  # in_clause_count

            total_items = 0
            max_items = 0
            for clause in in_clauses:
                items = len(clause.split(','))
                total_items += items
                max_items = max(max_items, items)

            features.append(float(total_items))  # in_list_total_items
            features.append(float(max_items))  # max_in_list_size

            # Other patterns
            features.append(float(utils.safe_count(where_upper, 'BETWEEN')))  # between_count
            features.append(float(utils.safe_count(where_upper, 'LIKE')))  # like_pattern_count
            features.append(float(utils.safe_count(where_upper, 'EXISTS')))  # exists_count

            return features

        except Exception as e:
            logger.error(f"Error extracting WHERE features: {e}")
            return [0.0] * len(self.FEATURE_NAMES)
