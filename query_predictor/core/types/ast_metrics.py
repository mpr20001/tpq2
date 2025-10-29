"""
AST parsing metrics and statistics.

Provides:
- ASTMetrics: AST parsing metrics and statistics
"""

from dataclasses import dataclass

@dataclass
class ASTMetrics:
    """
    AST parsing metrics and statistics.

    Tracks parsing success, complexity metrics, and structural features.

    NOTE: These fields MUST match the features expected by training for parity!
    """
    parse_success: bool
    parse_timeout: bool
    node_count: int
    depth: int
    breadth: int  # Maximum breadth/width of AST tree
    with_count: int  # WITH clause count
    cte_count: int
    lateral_view_count: int  # LATERAL VIEW count (Hive/Spark SQL)
    window_func_count: int  # Window function count (ROW_NUMBER, RANK, etc.)
    distinct_count: int  # DISTINCT keyword count
    having_count: int  # HAVING clause count
    subquery_count: int
    case_when_count: int
    coalesce_null_if_count: int  # COALESCE and NULLIF function count
    max_branches: int
    union_count: int
    union_all_count: int
