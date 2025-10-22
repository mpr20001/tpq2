"""
AST parser with timeout protection for SQL queries.

Uses sqlglot for intelligent SQL parsing with:
- Timeout protection (prevents hanging on complex queries)
- Thread pool management for concurrent parsing
- Graceful error handling
"""

import logging
import threading
import atexit
import weakref
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Import sqlglot for AST parsing
import sqlglot
from sqlglot import Expression
from sqlglot.expressions import CTE, Subquery, Join, Union, Case


from query_predictor.core.types.ast_metrics import ASTMetrics


class ASTParser:
    """
    Thread-safe AST parser with timeout handling.

    Features:
    - Timeout protection (default 50ms)
    - Thread pool for concurrent parsing
    - Singleton thread pool (shared across instances)
    - Graceful error handling
    - Proper resource cleanup via shutdown() or automatic cleanup on exit

    Thread Pool Lifecycle:
    - Thread pool is created on first use (lazy initialization)
    - Shared across all ASTParser instances for efficiency
    - Can be explicitly shutdown via shutdown() class method
    - Automatically shutdown on program exit via atexit handler

    Note:
        The singleton pattern is used for efficiency in multi-threaded environments.
    """

    # Shared thread pool for all instances
    _executor = None
    _executor_lock = threading.Lock()
    _executor_shutdown = False
    _cleanup_registered = False

    def __init__(self, timeout_ms: int = 50):
        """
        Initialize AST parser.

        Args:
            timeout_ms: Timeout in milliseconds for parsing
        """
        self.timeout_ms = timeout_ms
        self.timeout_seconds = timeout_ms / 1000.0

        # Initialize executor if needed
        if ASTParser._executor is None:
            with ASTParser._executor_lock:
                if ASTParser._executor is None:
                    ASTParser._executor = ThreadPoolExecutor(
                        max_workers=2,
                        thread_name_prefix="ast_parser"
                    )
                    ASTParser._executor_shutdown = False
                    # Register cleanup on program exit (only once)
                    if not ASTParser._cleanup_registered:
                        atexit.register(ASTParser.shutdown)
                        ASTParser._cleanup_registered = True

    @classmethod
    def shutdown(cls, wait: bool = True):
        """
        Shutdown the shared thread pool.

        This method can be called explicitly to cleanup resources,
        or will be called automatically on program exit.

        Args:
            wait: If True, wait for all pending futures to complete

        Note:
            After shutdown, creating new ASTParser instances will
            create a new thread pool.
        """
        with cls._executor_lock:
            if cls._executor is not None and not cls._executor_shutdown:
                # Use print instead of logger during shutdown to avoid I/O on closed file
                try:
                    print("Shutting down AST parser thread pool", flush=True)
                except:
                    pass  # Ignore if stdout is also closed
                
                try:
                    cls._executor.shutdown(wait=wait)
                    cls._executor_shutdown = True
                except Exception as e:
                    # Use print for error reporting during shutdown
                    try:
                        print(f"Error shutting down AST parser thread pool: {e}", flush=True)
                    except:
                        pass  # Ignore if stdout is also closed
                finally:
                    cls._executor = None

    def parse(self, query: str) -> Tuple[bool, Optional[Expression], ASTMetrics]:
        """
        Parse SQL query with timeout protection.

        Args:
            query: SQL query string

        Returns:
            Tuple of (success, parsed_ast, metrics)
        """
        if not query or not isinstance(query, str):
            return False, None, ASTMetrics(parse_success=False, parse_timeout=False, node_count=0, depth=0, cte_count=0, subquery_count=0, case_when_count=0, max_branches=0, union_count=0, union_all_count=0)

        # Check if executor has been shutdown
        if ASTParser._executor is None or ASTParser._executor_shutdown:
            logger.debug("AST parser thread pool has been shutdown, skipping parse")
            return False, None, ASTMetrics(parse_success=False, parse_timeout=False, node_count=0, depth=0, cte_count=0, subquery_count=0, case_when_count=0, max_branches=0, union_count=0, union_all_count=0)

        try:
            # Submit parse job to thread pool
            future = ASTParser._executor.submit(self._parse_internal, query)

            # Wait for result with timeout
            ast, metrics = future.result(timeout=self.timeout_seconds)
            return True, ast, metrics

        except FutureTimeoutError:
            logger.debug(f"AST parsing timeout after {self.timeout_ms}ms")
            metrics = ASTMetrics(parse_success=False, parse_timeout=True, node_count=0, depth=0, cte_count=0, subquery_count=0, case_when_count=0, max_branches=0, union_count=0, union_all_count=0)
            return False, None, metrics

        except Exception as e:
            logger.debug(f"AST parsing failed: {e}")
            return False, None, ASTMetrics(parse_success=False, parse_timeout=False, node_count=0, depth=0, cte_count=0, subquery_count=0, case_when_count=0, max_branches=0, union_count=0, union_all_count=0)

    def _parse_internal(self, query: str) -> Tuple[Optional[Expression], ASTMetrics]:
        """
        Internal parse method (runs in thread pool).

        Args:
            query: SQL query string

        Returns:
            Tuple of (ast, metrics)
        """
        try:
            # Parse query
            ast = sqlglot.parse_one(query, dialect='trino')

            if ast is None:
                return None, ASTMetrics(parse_success=False, parse_timeout=False, node_count=0, depth=0, cte_count=0, subquery_count=0, case_when_count=0, max_branches=0, union_count=0, union_all_count=0)

            # Extract metrics
            metrics = self._extract_metrics(ast)
            metrics.parse_success = True

            return ast, metrics

        except Exception as e:
            logger.debug(f"sqlglot parse error: {e}")
            return None, ASTMetrics(parse_success=False, parse_timeout=False, node_count=0, depth=0, cte_count=0, subquery_count=0, case_when_count=0, max_branches=0, union_count=0, union_all_count=0)

    def _extract_metrics(self, ast: Expression) -> ASTMetrics:
        """
        Extract metrics from parsed AST.

        Args:
            ast: Parsed sqlglot Expression

        Returns:
            ASTMetrics with extracted values
        """
        try:
            metrics = ASTMetrics(
                parse_success=True,
                parse_timeout=False,
                node_count=0,
                depth=0,
                cte_count=0,
                subquery_count=0,
                case_when_count=0,
                max_branches=0,
                union_count=0,
                union_all_count=0
            )

            # Calculate depth
            metrics.depth = self._calculate_depth(ast)

            # Count nodes
            metrics.node_count = sum(1 for _ in ast.walk())

            # Count specific node types
            metrics.cte_count = len(list(ast.find_all(CTE)))
            metrics.subquery_count = len(list(ast.find_all(Subquery)))
            # Note: join_count is not tracked in ASTMetrics, only in SQLPatternParser

            # CASE WHEN analysis
            case_nodes = list(ast.find_all(Case))
            metrics.case_when_count = len(case_nodes)
            if case_nodes:
                metrics.max_branches = max(
                    len(case.args.get('ifs', []))
                    for case in case_nodes
                )

            # UNION analysis
            unions = list(ast.find_all(Union))
            for union in unions:
                if union.args.get('distinct', True):
                    metrics.union_count += 1
                else:
                    metrics.union_all_count += 1

            return metrics

        except Exception as e:
            logger.debug(f"Error extracting AST metrics: {e}")
            return ASTMetrics(
                parse_success=False,
                parse_timeout=False,
                node_count=0,
                depth=0,
                cte_count=0,
                subquery_count=0,
                case_when_count=0,
                max_branches=0,
                union_count=0,
                union_all_count=0
            )

    def _calculate_depth(self, node: Expression, current_depth: int = 0) -> int:
        """
        Calculate maximum depth of AST tree.

        Args:
            node: Current AST node
            current_depth: Current depth level

        Returns:
            Maximum depth of tree
        """
        try:
            if not hasattr(node, 'args') or not node.args:
                return current_depth

            max_child_depth = current_depth

            for arg_value in node.args.values():
                if isinstance(arg_value, Expression):
                    child_depth = self._calculate_depth(arg_value, current_depth + 1)
                    max_child_depth = max(max_child_depth, child_depth)
                elif isinstance(arg_value, list):
                    for item in arg_value:
                        if isinstance(item, Expression):
                            child_depth = self._calculate_depth(item, current_depth + 1)
                            max_child_depth = max(max_child_depth, child_depth)

            return max_child_depth

        except Exception as e:
            logger.debug(f"Error calculating AST depth: {e}")
            return current_depth
