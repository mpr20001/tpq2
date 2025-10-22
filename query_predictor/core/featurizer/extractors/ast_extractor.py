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

    Features:
    - ast_depth, ast_width, ast_node_count, predicate_complexity
    - case_when_count, case_when_max_branches
    - union_count, union_all_count
    - ast_parse_timeout, ast_parse_failed
    """

    FEATURE_NAMES = [
        'ast_depth', 'ast_width', 'ast_node_count', 'predicate_complexity',
        'case_when_count', 'case_when_max_branches', 'union_count',
        'union_all_count', 'ast_parse_timeout', 'ast_parse_failed'
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
                # Use actual AST metrics (normalized)
                return [
                    utils.normalize_value(metrics.depth, constants.AST_DEPTH_NORMALIZER),
                    utils.normalize_value(metrics.node_count, constants.AST_WIDTH_NORMALIZER),
                    utils.normalize_value(metrics.node_count, constants.AST_NODE_COUNT_NORMALIZER),
                    utils.normalize_value(metrics.cte_count + metrics.subquery_count, constants.PREDICATE_COMPLEXITY_NORMALIZER),
                    utils.normalize_value(metrics.case_when_count, constants.CASE_WHEN_COUNT_NORMALIZER),
                    utils.normalize_value(metrics.max_branches, constants.CASE_WHEN_MAX_BRANCHES_NORMALIZER),
                    utils.normalize_value(metrics.union_count, constants.UNION_COUNT_NORMALIZER),
                    utils.normalize_value(metrics.union_all_count, constants.UNION_ALL_COUNT_NORMALIZER),
                    0.0,  # ast_parse_timeout
                    0.0   # ast_parse_failed
                ]

            elif metrics.parse_timeout:
                # Timeout - return simplified features with timeout flag
                return self._fallback_features(query, timeout=True)

            else:
                # Parse failed - return simplified features with failed flag
                return self._fallback_features(query, failed=True)

        except Exception as e:
            logger.error(f"Error extracting AST features: {e}")
            return self._fallback_features(query, failed=True)

    def _fallback_features(self, query: str, timeout: bool = False, failed: bool = False) -> List[float]:
        """
        Fallback features when AST parsing fails or times out.

        Uses simple heuristics based on query structure.
        """
        try:
            query_upper = utils.safe_upper(query)

            # Estimate AST complexity from query structure (using same normalizers as actual AST parsing)
            depth = utils.normalize_value(
                utils.safe_count(query, '(') + utils.safe_count(query_upper, 'SELECT'),
                constants.AST_DEPTH_NORMALIZER
            )
            width = utils.normalize_value(
                utils.safe_count(query, ',') + 1,
                constants.AST_WIDTH_NORMALIZER
            )
            node_count = utils.normalize_value(
                len(query.split()),
                constants.AST_NODE_COUNT_NORMALIZER
            )
            predicate_complexity = utils.normalize_value(
                utils.safe_count(query_upper, 'AND') + utils.safe_count(query_upper, 'OR'),
                constants.PREDICATE_COMPLEXITY_NORMALIZER
            )

            # CASE WHEN features (heuristic)
            case_when_count = utils.normalize_value(
                utils.safe_count(query_upper, 'CASE WHEN'),
                constants.CASE_WHEN_COUNT_NORMALIZER
            )
            max_branches = utils.normalize_value(
                utils.safe_count(query_upper, 'WHEN'),
                constants.CASE_WHEN_MAX_BRANCHES_NORMALIZER
            ) if case_when_count > 0 else 0.0

            # UNION features (heuristic)
            union_count = utils.normalize_value(
                utils.safe_count(query_upper, 'UNION') - utils.safe_count(query_upper, 'UNION ALL'),
                constants.UNION_COUNT_NORMALIZER
            )
            union_all_count = utils.normalize_value(
                utils.safe_count(query_upper, 'UNION ALL'),
                constants.UNION_ALL_COUNT_NORMALIZER
            )

            # Flags
            parse_timeout = 1.0 if timeout else 0.0
            parse_failed = 1.0 if failed else 0.0

            return [
                depth, width, node_count, predicate_complexity,
                case_when_count, max_branches, union_count, union_all_count,
                parse_timeout, parse_failed
            ]

        except Exception as e:
            logger.error(f"Error in fallback features: {e}")
            return [0.0] * len(self.FEATURE_NAMES)
