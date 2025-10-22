"""
Feature specification and management.

Provides:
- FeatureSpec: Complete feature specification with S3 save/load
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any
from datetime import datetime
import logging

from .feature_group import FeatureGroup

logger = logging.getLogger(__name__)


@dataclass
class FeatureSpec:
    """
    Complete feature specification with S3-native save/load.

    Features:
    - Feature group management (enable/disable groups)
    - Individual feature enable/disable
    - Feature importance tracking
    - S3 persistence with caching
    - Version control for train-serve parity
    """
    version: str
    created_at: str
    model_version: Optional[str] = None

    feature_groups: Dict[str, FeatureGroup] = field(default_factory=dict)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    disabled_features: Set[str] = field(default_factory=set)

    importance_threshold: float = 0.0
    enable_historical_features: bool = False

    def __post_init__(self):
        """Apply importance threshold if set."""
        if self.importance_threshold > 0 and self.feature_importance:
            self._apply_importance_threshold()

    @property
    def active_feature_names(self) -> List[str]:
        """Get ordered list of all active features."""
        features = []
        for group_name in self._get_ordered_group_names():
            if group_name not in self.feature_groups:
                continue

            group = self.feature_groups[group_name]
            if not group.enabled:
                continue

            for feature in group.feature_names:
                if feature not in self.disabled_features:
                    features.append(feature)

        return features

    @property
    def feature_count(self) -> int:
        """Total number of active features."""
        return len(self.active_feature_names)

    @property
    def total_feature_count(self) -> int:
        """Total features if all were enabled."""
        return sum(len(g.feature_names) for g in self.feature_groups.values())

    def _get_ordered_group_names(self) -> List[str]:
        """
        Return feature groups in extraction order.

        Order matters for train-serve parity!
        """
        order = [
            'sql', 'tablejoin', 'whereclause', 'aggregation', 'ast',
            'context', 'querytype', 'setoperation', 'nullaware',
            'historical'
        ]
        return [name for name in order if name in self.feature_groups]

    def enable_feature(self, feature_name: str):
        """Enable individual feature."""
        self.disabled_features.discard(feature_name)
        logger.info(f"Enabled feature: {feature_name}")

    def disable_feature(self, feature_name: str):
        """Disable individual feature."""
        self.disabled_features.add(feature_name)
        logger.info(f"Disabled feature: {feature_name}")

    def enable_group(self, group_name: str):
        """Enable entire feature group."""
        if group_name in self.feature_groups:
            self.feature_groups[group_name].enabled = True
            logger.info(f"Enabled feature group: {group_name}")

    def disable_group(self, group_name: str):
        """Disable entire feature group."""
        if group_name in self.feature_groups:
            self.feature_groups[group_name].enabled = False
            logger.info(f"Disabled feature group: {group_name}")

    def _apply_importance_threshold(self):
        """Disable features below importance threshold."""
        for feature, importance in self.feature_importance.items():
            if importance < self.importance_threshold:
                self.disabled_features.add(feature)

        logger.info(f"Applied importance threshold {self.importance_threshold}: "
                   f"{len(self.disabled_features)} features disabled")

    def prune_low_importance_features(self, threshold: float) -> Dict[str, List[str]]:
        """
        Remove features with importance < threshold.

        Args:
            threshold: Importance threshold (0.0 to 1.0)

        Returns:
            Dictionary of {group_name: [disabled_features]}
        """
        if not self.feature_importance:
            logger.warning("No feature importance data available for pruning")
            return {}

        disabled_by_group = {}

        for group_name, group in self.feature_groups.items():
            disabled = []
            for feature in group.feature_names:
                importance = self.feature_importance.get(feature, 1.0)
                if importance < threshold:
                    disabled.append(feature)
                    self.disabled_features.add(feature)

            if disabled:
                disabled_by_group[group_name] = disabled

        total_disabled = sum(len(v) for v in disabled_by_group.values())
        logger.info(f"Pruned {total_disabled} features with importance < {threshold}")

        return disabled_by_group

    def get_feature_importance(self, feature_name: str) -> Optional[float]:
        """Get importance score for a feature."""
        return self.feature_importance.get(feature_name)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for JSON."""
        return {
            'version': self.version,
            'created_at': self.created_at,
            'model_version': self.model_version,
            'feature_groups': {
                name: group.to_dict()
                for name, group in self.feature_groups.items()
            },
            'feature_importance': self.feature_importance,
            'disabled_features': list(self.disabled_features),
            'importance_threshold': self.importance_threshold,
            'enable_historical_features': self.enable_historical_features,
            'active_feature_count': self.feature_count,
            'total_feature_count': self.total_feature_count,
            'active_features': self.active_feature_names
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureSpec':
        """Deserialize from dictionary."""
        try:
            feature_groups = {}
            for name, group_data in data.get('feature_groups', {}).items():
                feature_groups[name] = FeatureGroup.from_dict(group_data)

            return cls(
                version=data['version'],
                created_at=data['created_at'],
                model_version=data.get('model_version'),
                feature_groups=feature_groups,
                feature_importance=data.get('feature_importance', {}),
                disabled_features=set(data.get('disabled_features', [])),
                importance_threshold=data.get('importance_threshold', 0.0),
                enable_historical_features=data.get('enable_historical_features', False)
            )
        except Exception as e:
            logger.error(f"Error deserializing FeatureSpec: {e}")
            raise ValueError(f"Invalid FeatureSpec data: {e}")

    @classmethod
    def from_extractors(cls, extractors: List, version: str = "3.0.0") -> 'FeatureSpec':
        """
        Build FeatureSpec from list of extractors.

        Args:
            extractors: List of BaseFeatureExtractor instances
            version: Feature spec version

        Returns:
            New FeatureSpec instance
        """
        feature_groups = {}

        for extractor in extractors:
            # Normalize group name
            group_name = extractor.__class__.__name__.replace('Extractor', '').replace('Feature', '').lower()

            feature_groups[group_name] = FeatureGroup(
                name=group_name,
                feature_names=extractor.get_feature_names(),
                enabled=True,
                description=extractor.__doc__ or ""
            )

        return cls(
            version=version,
            created_at=datetime.now().isoformat(),
            feature_groups=feature_groups
        )

    # S3-native save/load methods
    def save_to_s3(self, s3_path: str, s3_handler=None):
        """
        Save FeatureSpec to S3.

        Args:
            s3_path: S3 path (s3://bucket/key)
            s3_handler: Optional S3Handler instance

        Raises:
            RuntimeError: If S3 save fails
        """
        try:
            from query_predictor.utils.s3_utils import get_s3_handler

            handler = s3_handler or get_s3_handler()
            data = self.to_dict()
            handler.save_json(data, s3_path, cache_locally=True)
            logger.info(f"Saved FeatureSpec v{self.version} to {s3_path}")

        except Exception as e:
            logger.error(f"Failed to save FeatureSpec to S3: {e}")
            raise RuntimeError(f"Failed to save FeatureSpec: {e}")

    @classmethod
    def load_from_s3(cls, s3_path: str, s3_handler=None, use_cache: bool = True) -> 'FeatureSpec':
        """
        Load FeatureSpec from S3.

        Args:
            s3_path: S3 path (s3://bucket/key)
            s3_handler: Optional S3Handler instance
            use_cache: Whether to use local cache

        Returns:
            Loaded FeatureSpec instance

        Raises:
            RuntimeError: If S3 load fails
        """
        try:
            from query_predictor.utils.s3_utils import get_s3_handler

            handler = s3_handler or get_s3_handler()
            data = handler.load_json(s3_path, use_cache=use_cache)
            spec = cls.from_dict(data)
            logger.info(f"Loaded FeatureSpec v{spec.version} from {s3_path} "
                       f"({spec.feature_count}/{spec.total_feature_count} active features)")
            return spec

        except Exception as e:
            logger.error(f"Failed to load FeatureSpec from S3: {e}")
            raise RuntimeError(f"Failed to load FeatureSpec: {e}")
