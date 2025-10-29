"""AST feature extractor - 10 features."""

import logging
from typing import List, Dict, Any

from query_predictor.core.featurizer.base import BaseFeatureExtractor
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer import utils
from query_predictor.core.featurizer import constants
from query_predictor.core.featurizer.parsers import ASTParser

logger = logging.getLogger(__name__)


class ASTFeatureExtractor(BaseFeatureExtractor):
    """
    Extract AST features using sqlglot parser (10 features).

    Features (MUST match training notebook order for parity):
    - ast_depth, ast_breadth, ast_with_count, ast_cte_count
    - ast_lateral_view_count, ast_window_func_count, ast_distinct_count
    - ast_having_count, ast_case_when_count, ast_coalesce_null_if_count

    NOTE: Feature order is CRITICAL for train-serve parity!
    """

    FEATURE_NAMES = [
        'ast_depth',                 # Index 45 - Tree depth
        'ast_breadth',               # Index 46 - Tree width/breadth
        'ast_with_count',            # Index 47 - WITH clause count
        'ast_cte_count',             # Index 48 - CTE count
        'ast_lateral_view_count',    # Index 49 - LATERAL VIEW count
        'ast_window_func_count',     # Index 50 - Window function count
        'ast_distinct_count',        # Index 51 - DISTINCT count
        'ast_having_count',          # Index 52 - HAVING clause count
        'ast_case_when_count',       # Index 53 - CASE WHEN count
        'ast_coalesce_null_if_count' # Index 54 - COALESCE/NULLIF count
    ]

    def __init__(self, config: Dict[str, Any], feature_spec=None, ast_parser=None):
        super().__init__(config, feature_spec)
        timeout_ms = config.get('ast_timeout_ms', 50)
        self.ast_parser = ast_parser or ASTParser(timeout_ms=timeout_ms)

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def extract(self, query_data: QueryData) -> List[float]:
        query = query_data.query

        if not query:
            return [0.0] * len(self.FEATURE_NAMES)

        try:
            # Try AST parsing with timeout
            success, ast, metrics = self.ast_parser.parse(query)

            if success and metrics.parse_success:
                # Return features in EXACT order expected by training
                # CRITICAL: This order MUST match FEATURE_NAMES for parity!
                return [
                    # Index 45: ast_depth
                    utils.normalize_value(metrics.depth, constants.AST_DEPTH_NORMALIZER),
                    # Index 46: ast_breadth (width)
                    utils.normalize_value(metrics.breadth, constants.AST_WIDTH_NORMALIZER),
                    # Index 47: ast_with_count
                    utils.normalize_value(metrics.with_count, constants.WITH_COUNT_NORMALIZER),
                    # Index 48: ast_cte_count
                    utils.normalize_value(metrics.cte_count, constants.CTE_COUNT_NORMALIZER),
                    # Index 49: ast_lateral_view_count
                    utils.normalize_value(metrics.lateral_view_count, constants.LATERAL_VIEW_COUNT_NORMALIZER),
                    # Index 50: ast_window_func_count
                    utils.normalize_value(metrics.window_func_count, constants.WINDOW_FUNC_COUNT_NORMALIZER),
                    # Index 51: ast_distinct_count
                    utils.normalize_value(metrics.distinct_count, constants.DISTINCT_COUNT_NORMALIZER),
                    # Index 52: ast_having_count
                    utils.normalize_value(metrics.having_count, constants.HAVING_COUNT_NORMALIZER),
                    # Index 53: ast_case_when_count
                    utils.normalize_value(metrics.case_when_count, constants.CASE_WHEN_COUNT_NORMALIZER),
                    # Index 54: ast_coalesce_null_if_count
                    utils.normalize_value(metrics.coalesce_null_if_count, constants.COALESCE_NULL_IF_COUNT_NORMALIZER)
                ]

            elif metrics.parse_timeout:
                # Timeout - return simplified features
                return self._fallback_features(query, timeout=True)

            else:
                # Parse failed - return simplified features
                return self._fallback_features(query, failed=True)

        except Exception as e:
            logger.error(f"Error extracting AST features: {e}")
            return self._fallback_features(query, failed=True)

    def _fallback_features(self, query: str, timeout: bool = False, failed: bool = False) -> List[float]:
        """
        Fallback features when AST parsing fails or times out.

        Uses simple heuristics based on query structure.
        Returns 10 features in the EXACT order expected by training.
        """
        try:
            query_upper = utils.safe_upper(query)

            # Estimate features using heuristics (in training order!)
            # Index 45: ast_depth
            depth = utils.normalize_value(
                utils.safe_count(query, '(') + utils.safe_count(query_upper, 'SELECT'),
                constants.AST_DEPTH_NORMALIZER
            )

            # Index 46: ast_breadth (width)
            breadth = utils.normalize_value(
                utils.safe_count(query, ',') + 1,
                constants.AST_WIDTH_NORMALIZER
            )

            # Index 47: ast_with_count
            with_count = utils.normalize_value(
                utils.safe_count(query_upper, 'WITH '),
                constants.WITH_COUNT_NORMALIZER
            )

            # Index 48: ast_cte_count (approximated as WITH count)
            cte_count = with_count  # Same as WITH for heuristic

            # Index 49: ast_lateral_view_count
            lateral_view_count = utils.normalize_value(
                utils.safe_count(query_upper, 'LATERAL VIEW'),
                constants.LATERAL_VIEW_COUNT_NORMALIZER
            )

            # Index 50: ast_window_func_count
            window_func_count = utils.normalize_value(
                utils.safe_count(query_upper, 'OVER (') +
                utils.safe_count(query_upper, 'ROW_NUMBER') +
                utils.safe_count(query_upper, 'RANK'),
                constants.WINDOW_FUNC_COUNT_NORMALIZER
            )

            # Index 51: ast_distinct_count
            distinct_count = utils.normalize_value(
                utils.safe_count(query_upper, 'DISTINCT'),
                constants.DISTINCT_COUNT_NORMALIZER
            )

            # Index 52: ast_having_count
            having_count = utils.normalize_value(
                utils.safe_count(query_upper, 'HAVING'),
                constants.HAVING_COUNT_NORMALIZER
            )

            # Index 53: ast_case_when_count
            case_when_count = utils.normalize_value(
                utils.safe_count(query_upper, 'CASE WHEN'),
                constants.CASE_WHEN_COUNT_NORMALIZER
            )

            # Index 54: ast_coalesce_null_if_count
            coalesce_null_if_count = utils.normalize_value(
                utils.safe_count(query_upper, 'COALESCE') +
                utils.safe_count(query_upper, 'NULLIF'),
                constants.COALESCE_NULL_IF_COUNT_NORMALIZER
            )

            # Return in EXACT training order!
            return [
                depth,                    # 45
                breadth,                  # 46
                with_count,               # 47
                cte_count,                # 48
                lateral_view_count,       # 49
                window_func_count,        # 50
                distinct_count,           # 51
                having_count,             # 52
                case_when_count,          # 53
                coalesce_null_if_count    # 54
            ]

        except Exception as e:
            logger.error(f"Error in fallback features: {e}")
            return [0.0] * len(self.FEATURE_NAMES)
