"""
Base classes for feature extractors.

Provides abstract base class and common functionality for all feature extractors.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, TYPE_CHECKING
import logging

from query_predictor.core.types.query_data import QueryData

if TYPE_CHECKING:
    from query_predictor.core.types.feature_spec import FeatureSpec

logger = logging.getLogger(__name__)


class BaseFeatureExtractor(ABC):
    """
    Abstract base class for all feature extractors.

    All extractors must implement:
    - extract(): Extract features from query data
    - get_feature_names(): Return ordered list of feature names

    Features:
    - Error isolation (extractors never crash the pipeline)
    - FeatureSpec integration for enable/disable
    - Type safety with hints
    - Logging support
    """

    def __init__(self, config: Dict[str, Any], feature_spec: Optional['FeatureSpec'] = None):
        """
        Initialize feature extractor.

        Args:
            config: Configuration dictionary
            feature_spec: Optional FeatureSpec for enable/disable control
        """
        self.config = config
        self.feature_spec = feature_spec
        self._all_feature_names = self.get_feature_names()

    @abstractmethod
    def extract(self, query_data: QueryData) -> List[float]:
        """
        Extract features from query data.

        Args:
            query_data: QueryData object with query, user, catalog, schema, etc.

        Returns:
            List of float features

        Note:
            Should handle all errors internally and return zero features on failure.
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Get ordered list of feature names this extractor produces.

        Returns:
            List of feature name strings
        """
        pass

    def get_active_feature_names(self) -> List[str]:
        """
        Get list of currently active (enabled) feature names.

        Returns:
            List of active feature names based on FeatureSpec
        """
        if self.feature_spec is None:
            return self._all_feature_names

        group_name = self.__class__.__name__.replace('Extractor', '').replace('Feature', '').lower()

        # Check if group is enabled
        if group_name in self.feature_spec.feature_groups:
            group = self.feature_spec.feature_groups[group_name]
            if not group.enabled:
                return []

            # Filter out disabled features
            return [f for f in group.feature_names
                   if f not in self.feature_spec.disabled_features]

        return self._all_feature_names

    def extract_with_spec(self, query_data: QueryData) -> List[float]:
        """
        Extract features with FeatureSpec filtering applied.

        Args:
            query_data: QueryData object

        Returns:
            Filtered list of features (only active ones)
        """
        try:
            all_features = self.extract(query_data)

            if self.feature_spec is None:
                return all_features

            # Get active feature names
            active_names = self.get_active_feature_names()

            # If all features are active, return as-is
            if len(active_names) == len(all_features):
                return all_features

            # Filter to active features only
            indices = [i for i, name in enumerate(self._all_feature_names)
                      if name in active_names]

            return [all_features[i] for i in indices]

        except Exception as e:
            logger.error(f"Error in extract_with_spec for {self.__class__.__name__}: {e}")
            # Return zero features on failure
            return [0.0] * len(self.get_active_feature_names())

    @property
    def feature_count(self) -> int:
        """Number of features this extractor produces (active only)."""
        return len(self.get_active_feature_names())

    def validate_feature_count(self, features: List[float]) -> bool:
        """
        Validate that feature count matches expected.

        Args:
            features: List of extracted features

        Returns:
            True if count matches expected
        """
        expected = len(self._all_feature_names)
        actual = len(features)

        if actual != expected:
            logger.warning(f"{self.__class__.__name__}: Expected {expected} features, got {actual}")
            return False

        return True

    def safe_extract(self, query_data: QueryData) -> List[float]:
        """
        Extract features with error handling and fallback.

        Args:
            query_data: QueryData object

        Returns:
            List of features (zero features on any error)
        """
        try:
            features = self.extract(query_data)

            # Validate count
            if not self.validate_feature_count(features):
                logger.error(f"{self.__class__.__name__}: Feature count mismatch, returning zeros")
                return [0.0] * len(self._all_feature_names)

            return features

        except Exception as e:
            logger.error(f"Error in {self.__class__.__name__}.extract(): {e}", exc_info=True)
            # Return zero features as fallback
            return [0.0] * len(self._all_feature_names)
