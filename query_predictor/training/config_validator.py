"""
Training configuration validator.

Validates training_config.yaml structure and values to catch configuration
errors early before starting long-running jobs.
"""

import logging
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validate training configuration dictionary.

    Checks:
    - Required sections exist
    - Required keys within sections exist
    - Data types are correct
    - Value ranges are valid
    - Date formats are valid
    - S3 paths are well-formed

    Example:
        validator = ConfigValidator()
        errors = validator.validate(config)
        if errors:
            raise ValueError(f"Invalid config: {errors}")
    """

    # Required top-level sections
    REQUIRED_SECTIONS = [
        'data_loading',
        'features',
        'model',
        'prd_requirements',
        'time_splits',
        'spark',
        's3'
    ]

    # Optional sections
    OPTIONAL_SECTIONS = [
        'boundary_sampling',
        'checkpointing',
        'validation',
        'logging',
        'analysis'
    ]

    def __init__(self):
        """Initialize config validator."""
        self.logger = logging.getLogger(__name__)

    def validate(self, config: Dict[str, Any], strict: bool = False) -> List[str]:
        """
        Validate configuration dictionary.

        Args:
            config: Configuration dictionary to validate
            strict: If True, treats warnings as errors

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Check required sections
        errors.extend(self._validate_required_sections(config))

        # Validate individual sections
        errors.extend(self._validate_data_loading(config.get('data_loading', {})))
        errors.extend(self._validate_boundary_sampling(config.get('boundary_sampling', {})))
        errors.extend(self._validate_features(config.get('features', {})))
        errors.extend(self._validate_model(config.get('model', {})))
        errors.extend(self._validate_prd_requirements(config.get('prd_requirements', {})))
        errors.extend(self._validate_time_splits(config.get('time_splits', {})))
        errors.extend(self._validate_spark(config.get('spark', {})))
        errors.extend(self._validate_validation(config.get('validation', {})))
        errors.extend(self._validate_s3(config.get('s3', {})))

        return errors

    def _validate_required_sections(self, config: Dict[str, Any]) -> List[str]:
        """Check that all required sections exist."""
        errors = []
        for section in self.REQUIRED_SECTIONS:
            if section not in config:
                errors.append(f"Missing required section: '{section}'")
        return errors

    def _validate_data_loading(self, section: Dict[str, Any]) -> List[str]:
        """Validate data_loading section."""
        errors = []

        # Required keys
        required_keys = ['start_date', 'end_date', 'cpu_threshold_seconds', 'memory_threshold_gb']
        for key in required_keys:
            if key not in section:
                errors.append(f"data_loading: Missing required key '{key}'")

        # Validate dates
        if 'start_date' in section:
            try:
                datetime.strptime(section['start_date'], '%Y-%m-%d')
            except ValueError:
                errors.append(f"data_loading: Invalid start_date format (expected YYYY-MM-DD)")

        if 'end_date' in section:
            try:
                end_date = datetime.strptime(section['end_date'], '%Y-%m-%d')
                if 'start_date' in section:
                    start_date = datetime.strptime(section['start_date'], '%Y-%m-%d')
                    if end_date <= start_date:
                        errors.append(f"data_loading: end_date must be after start_date")
            except ValueError:
                errors.append(f"data_loading: Invalid end_date format (expected YYYY-MM-DD)")

        # Validate thresholds
        if 'cpu_threshold_seconds' in section:
            if not isinstance(section['cpu_threshold_seconds'], (int, float)):
                errors.append(f"data_loading: cpu_threshold_seconds must be numeric")
            elif section['cpu_threshold_seconds'] <= 0:
                errors.append(f"data_loading: cpu_threshold_seconds must be positive")

        if 'memory_threshold_gb' in section:
            if not isinstance(section['memory_threshold_gb'], (int, float)):
                errors.append(f"data_loading: memory_threshold_gb must be numeric")
            elif section['memory_threshold_gb'] <= 0:
                errors.append(f"data_loading: memory_threshold_gb must be positive")

        return errors

    def _validate_boundary_sampling(self, section: Dict[str, Any]) -> List[str]:
        """Validate boundary_sampling section."""
        errors = []

        if 'balance_ratio' in section:
            if not isinstance(section['balance_ratio'], (int, float)):
                errors.append(f"boundary_sampling: balance_ratio must be numeric")
            elif section['balance_ratio'] <= 0:
                errors.append(f"boundary_sampling: balance_ratio must be positive")

        if 'boundary_sampling_max_boost' in section:
            if not isinstance(section['boundary_sampling_max_boost'], (int, float)):
                errors.append(f"boundary_sampling: boundary_sampling_max_boost must be numeric")
            elif section['boundary_sampling_max_boost'] <= 0:
                errors.append(f"boundary_sampling: boundary_sampling_max_boost must be positive")

        return errors

    def _validate_features(self, section: Dict[str, Any]) -> List[str]:
        """Validate features section."""
        errors = []

        # Required keys
        required_keys = ['base_feature_count', 'tfidf_vocab_size', 'total_features']
        for key in required_keys:
            if key not in section:
                errors.append(f"features: Missing required key '{key}'")

        # Validate counts
        for key in ['base_feature_count', 'historical_feature_count', 'tfidf_vocab_size', 'total_features']:
            if key in section:
                if not isinstance(section[key], int):
                    errors.append(f"features: {key} must be an integer")
                elif section[key] <= 0:
                    errors.append(f"features: {key} must be positive")

        # Validate TF-IDF params
        if 'ngram_range' in section:
            if not isinstance(section['ngram_range'], list) or len(section['ngram_range']) != 2:
                errors.append(f"features: ngram_range must be a list of 2 integers")
            else:
                if section['ngram_range'][0] < 1 or section['ngram_range'][1] < section['ngram_range'][0]:
                    errors.append(f"features: invalid ngram_range values")

        if 'min_df' in section:
            if not isinstance(section['min_df'], int) or section['min_df'] < 1:
                errors.append(f"features: min_df must be integer >= 1")

        if 'max_df' in section:
            if not isinstance(section['max_df'], (int, float)):
                errors.append(f"features: max_df must be numeric")
            elif not (0 < section['max_df'] <= 1.0):
                errors.append(f"features: max_df must be between 0 and 1")

        return errors

    def _validate_model(self, section: Dict[str, Any]) -> List[str]:
        """Validate model section."""
        errors = []

        # Required keys
        required_keys = ['algorithm', 'n_estimators', 'cost_fn', 'cost_fp']
        for key in required_keys:
            if key not in section:
                errors.append(f"model: Missing required key '{key}'")

        # Validate hyperparameters
        if 'n_estimators' in section:
            if not isinstance(section['n_estimators'], int) or section['n_estimators'] < 1:
                errors.append(f"model: n_estimators must be positive integer")

        if 'max_depth' in section:
            if not isinstance(section['max_depth'], int) or section['max_depth'] < 1:
                errors.append(f"model: max_depth must be positive integer")

        if 'learning_rate' in section:
            if not isinstance(section['learning_rate'], (int, float)):
                errors.append(f"model: learning_rate must be numeric")
            elif not (0 < section['learning_rate'] <= 1.0):
                errors.append(f"model: learning_rate must be between 0 and 1")

        if 'subsample' in section:
            if not isinstance(section['subsample'], (int, float)):
                errors.append(f"model: subsample must be numeric")
            elif not (0 < section['subsample'] <= 1.0):
                errors.append(f"model: subsample must be between 0 and 1")

        if 'cv_folds' in section:
            if not isinstance(section['cv_folds'], int) or section['cv_folds'] < 2:
                errors.append(f"model: cv_folds must be integer >= 2")

        # Validate costs
        if 'cost_fn' in section:
            if not isinstance(section['cost_fn'], (int, float)) or section['cost_fn'] <= 0:
                errors.append(f"model: cost_fn must be positive numeric")

        if 'cost_fp' in section:
            if not isinstance(section['cost_fp'], (int, float)) or section['cost_fp'] <= 0:
                errors.append(f"model: cost_fp must be positive numeric")

        return errors

    def _validate_prd_requirements(self, section: Dict[str, Any]) -> List[str]:
        """Validate prd_requirements section."""
        errors = []

        # Required keys
        required_keys = ['target_heavy_recall', 'target_fnr', 'target_f1', 'target_roc_auc']
        for key in required_keys:
            if key not in section:
                errors.append(f"prd_requirements: Missing required key '{key}'")

        # Validate metrics (all should be between 0 and 1)
        for key in required_keys:
            if key in section:
                if not isinstance(section[key], (int, float)):
                    errors.append(f"prd_requirements: {key} must be numeric")
                elif not (0 <= section[key] <= 1.0):
                    errors.append(f"prd_requirements: {key} must be between 0 and 1")

        return errors

    def _validate_time_splits(self, section: Dict[str, Any]) -> List[str]:
        """Validate time_splits section."""
        errors = []

        # Required keys
        required_keys = ['train_days', 'val_days', 'test_days']
        for key in required_keys:
            if key not in section:
                errors.append(f"time_splits: Missing required key '{key}'")

        # Validate all are positive integers
        for key in required_keys:
            if key in section:
                if not isinstance(section[key], int) or section[key] < 1:
                    errors.append(f"time_splits: {key} must be positive integer")

        return errors

    def _validate_spark(self, section: Dict[str, Any]) -> List[str]:
        """Validate spark section."""
        errors = []

        # Validate memory strings
        for key in ['driver_memory', 'executor_memory']:
            if key in section:
                if not isinstance(section[key], str):
                    errors.append(f"spark: {key} must be string (e.g., '16G')")
                elif not section[key][-1] in ['G', 'M']:
                    errors.append(f"spark: {key} must end with 'G' or 'M'")

        # Validate core counts
        for key in ['driver_cores', 'executor_cores', 'min_executors', 'max_executors']:
            if key in section:
                if not isinstance(section[key], int) or section[key] < 1:
                    errors.append(f"spark: {key} must be positive integer")

        return errors

    def _validate_validation(self, section: Dict[str, Any]) -> List[str]:
        """Validate validation section."""
        errors = []

        # Validate tolerances
        if 'parity_tolerance' in section:
            if not isinstance(section['parity_tolerance'], (int, float)) or section['parity_tolerance'] <= 0:
                errors.append(f"validation: parity_tolerance must be positive numeric")

        if 'onnx_tolerance' in section:
            if not isinstance(section['onnx_tolerance'], (int, float)) or section['onnx_tolerance'] <= 0:
                errors.append(f"validation: onnx_tolerance must be positive numeric")

        # Validate sample counts
        if 'parity_samples' in section:
            if not isinstance(section['parity_samples'], int) or section['parity_samples'] < 1:
                errors.append(f"validation: parity_samples must be positive integer")

        if 'onnx_validation_samples' in section:
            if not isinstance(section['onnx_validation_samples'], int) or section['onnx_validation_samples'] < 1:
                errors.append(f"validation: onnx_validation_samples must be positive integer")

        # Validate opset version
        if 'onnx_opset_version' in section:
            if not isinstance(section['onnx_opset_version'], int) or section['onnx_opset_version'] < 1:
                errors.append(f"validation: onnx_opset_version must be positive integer")

        return errors

    def _validate_s3(self, section: Dict[str, Any]) -> List[str]:
        """Validate s3 section."""
        errors = []

        # Required keys
        required_keys = ['bucket', 'prefix']
        for key in required_keys:
            if key not in section:
                errors.append(f"s3: Missing required key '{key}'")

        # Validate bucket name format (basic check)
        if 'bucket' in section:
            if not isinstance(section['bucket'], str) or not section['bucket']:
                errors.append(f"s3: bucket must be non-empty string")

        if 'prefix' in section:
            if not isinstance(section['prefix'], str):
                errors.append(f"s3: prefix must be string")

        return errors

    def generate_report(self, config: Dict[str, Any]) -> str:
        """
        Generate validation report.

        Args:
            config: Configuration to validate

        Returns:
            Formatted validation report
        """
        errors = self.validate(config)

        if not errors:
            return "✅ Configuration valid - all checks passed"

        report = ["❌ Configuration validation failed:\n"]
        for i, error in enumerate(errors, 1):
            report.append(f"  {i}. {error}")

        return "\n".join(report)

    def validate_and_raise(self, config: Dict[str, Any]):
        """
        Validate config and raise ValueError if invalid.

        Args:
            config: Configuration to validate

        Raises:
            ValueError: If configuration is invalid
        """
        errors = self.validate(config)
        if errors:
            error_msg = "\n".join(errors)
            raise ValueError(f"Invalid training configuration:\n{error_msg}")
