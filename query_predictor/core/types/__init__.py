"""
Data types and type definitions for the query predictor.

Provides:
- QueryData: Validated query data for feature extraction
- FeatureSpec: Feature specification and management
- FeatureGroup: Logical grouping of features
- ASTMetrics: AST parsing metrics and statistics
"""

from .query_data import QueryData
from .feature_group import FeatureGroup
from .feature_spec import FeatureSpec
from .ast_metrics import ASTMetrics

__all__ = ['QueryData', 'FeatureGroup', 'FeatureSpec', 'ASTMetrics']
