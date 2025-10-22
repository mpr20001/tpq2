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
    """
    parse_success: bool
    parse_timeout: bool
    node_count: int
    depth: int
    cte_count: int
    subquery_count: int
    case_when_count: int
    max_branches: int
    union_count: int
    union_all_count: int
