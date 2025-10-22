"""Configuration management for the query predictor service."""

import os
import yaml
from typing import Dict, Any
from query_predictor.common.exceptions import ConfigurationError


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file and environment variables.

    Environment variables override YAML config values.

    Args:
        config_path: Path to YAML config file.
                     Defaults to QUERY_PREDICTOR_CONFIG env var or
                     'config/service_config.yaml'

    Returns:
        Dictionary containing merged configuration

    Raises:
        ConfigurationError: If config file is missing or invalid
    """
    # Determine config file path
    if config_path is None:
        config_path = os.environ.get(
            'QUERY_PREDICTOR_CONFIG',
            'config/service_config.yaml'
        )

    # Load YAML configuration
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ConfigurationError(f"Configuration file not found: {config_path}")
    except yaml.YAMLError as e:
        raise ConfigurationError(f"Invalid YAML configuration: {e}")

    if config is None:
        raise ConfigurationError(f"Empty configuration file: {config_path}")

    # Apply environment variable overrides
    config = _apply_env_overrides(config)

    # Validate required fields
    _validate_config(config)

    return config


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """Apply environment variable overrides to configuration."""

    # Service overrides
    if 'QUERY_PREDICTOR_PORT' in os.environ:
        config.setdefault('service', {})['port'] = int(os.environ['QUERY_PREDICTOR_PORT'])

    if 'QUERY_PREDICTOR_LOG_LEVEL' in os.environ:
        config.setdefault('logging', {})['level'] = os.environ['QUERY_PREDICTOR_LOG_LEVEL']

    # AWS overrides
    if 'AWS_DEFAULT_REGION' in os.environ:
        config.setdefault('aws', {})['region'] = os.environ['AWS_DEFAULT_REGION']

    # S3 bucket and path overrides
    model_config = config.setdefault('model', {})

    # Override S3 bucket/prefix components
    if 'QUERY_PREDICTOR_S3_BUCKET' in os.environ:
        model_config['s3_bucket'] = os.environ['QUERY_PREDICTOR_S3_BUCKET']

    if 'QUERY_PREDICTOR_S3_PREFIX' in os.environ:
        model_config['s3_prefix'] = os.environ['QUERY_PREDICTOR_S3_PREFIX']

    if 'QUERY_PREDICTOR_MODEL_FILE' in os.environ:
        model_config['model_file'] = os.environ['QUERY_PREDICTOR_MODEL_FILE']

    if 'QUERY_PREDICTOR_FEATURE_SPEC_FILE' in os.environ:
        model_config['feature_spec_file'] = os.environ['QUERY_PREDICTOR_FEATURE_SPEC_FILE']

    if 'QUERY_PREDICTOR_HISTORICAL_STATS_FILE' in os.environ:
        model_config['historical_stats_file'] = os.environ['QUERY_PREDICTOR_HISTORICAL_STATS_FILE']

    # Reconstruct full S3 paths from components (support override)
    s3_bucket = model_config.get('s3_bucket', 'uip-datalake-bucket-prod')
    s3_prefix = model_config.get('s3_prefix', 'query_predictor')
    model_file = model_config.get('model_file', 'model_v20251005.onnx')
    feature_spec_file = model_config.get('feature_spec_file', 'feature_spec_v20251005.json')
    historical_stats_file = model_config.get('historical_stats_file', 'historical_stats_v20251005.json')

    # Allow direct path override (takes precedence)
    if 'QUERY_PREDICTOR_MODEL_PATH' not in os.environ:
        model_config['model_path'] = f"s3://{s3_bucket}/{s3_prefix}/{model_file}"
    else:
        model_config['model_path'] = os.environ['QUERY_PREDICTOR_MODEL_PATH']

    if 'QUERY_PREDICTOR_FEATURE_SPEC_PATH' not in os.environ:
        model_config['feature_spec_path'] = f"s3://{s3_bucket}/{s3_prefix}/{feature_spec_file}"
    else:
        model_config['feature_spec_path'] = os.environ['QUERY_PREDICTOR_FEATURE_SPEC_PATH']

    if 'QUERY_PREDICTOR_HISTORICAL_STATS_PATH' not in os.environ:
        model_config['historical_stats_path'] = f"s3://{s3_bucket}/{s3_prefix}/{historical_stats_file}"
    else:
        model_config['historical_stats_path'] = os.environ['QUERY_PREDICTOR_HISTORICAL_STATS_PATH']

    # Model overrides
    if 'QUERY_PREDICTOR_MODEL_VERSION' in os.environ:
        model_config['model_version'] = os.environ['QUERY_PREDICTOR_MODEL_VERSION']

    if 'QUERY_PREDICTOR_THRESHOLD' in os.environ:
        model_config['threshold'] = float(os.environ['QUERY_PREDICTOR_THRESHOLD'])

    # Feature toggles
    if 'QUERY_PREDICTOR_ZERO_COST_ENABLED' in os.environ:
        enabled = os.environ['QUERY_PREDICTOR_ZERO_COST_ENABLED'].lower() == 'true'
        config.setdefault('zero_cost_filter', {})['enabled'] = enabled

    if 'QUERY_PREDICTOR_HISTORICAL_FEATURES_ENABLED' in os.environ:
        enabled = os.environ['QUERY_PREDICTOR_HISTORICAL_FEATURES_ENABLED'].lower() == 'true'
        config.setdefault('featurizer', {})['enable_historical_features'] = enabled

    return config


def _validate_config(config: Dict[str, Any]) -> None:
    """
    Validate required configuration fields.

    Raises:
        ConfigurationError: If required fields are missing
    """
    required_sections = ['service', 'logging']

    for section in required_sections:
        if section not in config:
            raise ConfigurationError(f"Missing required config section: {section}")

    # Validate service section
    service_config = config['service']
    if 'name' not in service_config:
        raise ConfigurationError("Missing required field: service.name")
    if 'version' not in service_config:
        raise ConfigurationError("Missing required field: service.version")

    # Validate logging section
    logging_config = config['logging']
    if 'level' not in logging_config:
        raise ConfigurationError("Missing required field: logging.level")
