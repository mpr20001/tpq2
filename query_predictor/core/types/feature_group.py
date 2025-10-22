"""
Feature group management.

Provides:
- FeatureGroup: Logical grouping of features
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any


@dataclass
class FeatureGroup:
    """
    Represents a logical group of features (e.g., SQL features, AST features).

    Used by FeatureSpec to enable/disable entire feature groups.
    """
    name: str
    feature_names: List[str]
    enabled: bool = True
    importance_scores: Dict[str, float] = field(default_factory=dict)
    description: str = ""

    @property
    def active_features(self) -> List[str]:
        """Return only enabled features."""
        if not self.enabled:
            return []
        return self.feature_names

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            'name': self.name,
            'feature_names': self.feature_names,
            'enabled': self.enabled,
            'importance_scores': self.importance_scores,
            'description': self.description
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureGroup':
        """Deserialize from dictionary."""
        return cls(
            name=data['name'],
            feature_names=data['feature_names'],
            enabled=data.get('enabled', True),
            importance_scores=data.get('importance_scores', {}),
            description=data.get('description', '')
        )
