"""SQL structure feature extractor - 15 features."""

import re
import logging
from typing import List, Dict, Any

from query_predictor.core.featurizer.base import BaseFeatureExtractor
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer import utils

logger = logging.getLogger(__name__)


class SQLFeatureExtractor(BaseFeatureExtractor):
    """
    Extract basic SQL structure features (15 features).

    Features:
    - query_length, token_count, line_count
    - has_select_star, has_distinct, has_order_by, has_limit, limit_value
    - has_offset, comment_ratio, parentheses_depth
    - quoted_strings_count, numeric_literals_count
    - wildcard_count, semicolon_count
    """

    FEATURE_NAMES = [
        'query_length', 'token_count', 'line_count', 'has_select_star',
        'has_distinct', 'has_order_by', 'has_limit', 'limit_value',
        'has_offset', 'comment_ratio', 'parentheses_depth',
        'quoted_strings_count', 'numeric_literals_count',
        'wildcard_count', 'semicolon_count'
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

            # Basic metrics
            features.append(float(utils.safe_len(query)))  # query_length
            features.append(float(len(query.split())))  # token_count
            features.append(float(utils.safe_count(query, '\n') + 1))  # line_count

            # SQL patterns
            features.append(1.0 if 'SELECT *' in query_upper else 0.0)  # has_select_star
            features.append(1.0 if 'DISTINCT' in query_upper else 0.0)  # has_distinct
            features.append(1.0 if 'ORDER BY' in query_upper else 0.0)  # has_order_by
            features.append(1.0 if 'LIMIT' in query_upper else 0.0)  # has_limit

            # Extract limit value
            limit_match = re.search(r'LIMIT\s+(\d+)', query_upper)
            features.append(float(limit_match.group(1)) if limit_match else 0.0)  # limit_value

            features.append(1.0 if 'OFFSET' in query_upper else 0.0)  # has_offset

            # Comment ratio
            comment_chars = utils.safe_count(query, '--') * 20 + utils.safe_count(query, '/*') * 30
            features.append(utils.safe_divide(comment_chars, max(len(query), 1)))  # comment_ratio

            # Parentheses depth
            features.append(float(self._calculate_parentheses_depth(query)))  # parentheses_depth

            # String and literal counts
            features.append(float(utils.safe_count(query, "'") // 2))  # quoted_strings_count
            features.append(float(len(re.findall(r'\b\d+\b', query))))  # numeric_literals_count
            features.append(float(utils.safe_count(query, '%')))  # wildcard_count
            features.append(float(utils.safe_count(query, ';')))  # semicolon_count

            return features

        except Exception as e:
            logger.error(f"Error extracting SQL features: {e}")
            return [0.0] * len(self.FEATURE_NAMES)

    def _calculate_parentheses_depth(self, query: str) -> int:
        """Calculate maximum nesting depth of parentheses."""
        try:
            max_depth = 0
            current_depth = 0
            for char in query:
                if char == '(':
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == ')':
                    current_depth = max(0, current_depth - 1)
            return max_depth
        except:
            return 0
