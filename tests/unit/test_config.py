"""Unit tests for configuration management."""

import pytest
import os
import tempfile
import yaml
from query_predictor.common.config import load_config, _apply_env_overrides, _validate_config
from query_predictor.common.exceptions import ConfigurationError


def test_load_config_success(tmp_path):
    """Test successful configuration loading from YAML."""
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        'service': {
            'name': 'test-service',
            'version': '1.0.0',
            'port': 8000
        },
        'logging': {
            'level': 'INFO',
            'format': 'json',
            'output': 'stdout'
        }
    }

    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    config = load_config(str(config_file))

    assert config['service']['name'] == 'test-service'
    assert config['service']['version'] == '1.0.0'
    assert config['service']['port'] == 8000
    assert config['logging']['level'] == 'INFO'


def test_load_config_file_not_found():
    """Test configuration loading fails with missing file."""
    with pytest.raises(ConfigurationError) as exc_info:
        load_config('/nonexistent/config.yaml')

    assert "Configuration file not found" in str(exc_info.value)


def test_load_config_invalid_yaml(tmp_path):
    """Test configuration loading fails with invalid YAML."""
    config_file = tmp_path / "invalid_config.yaml"

    with open(config_file, 'w') as f:
        f.write("invalid: yaml: content: [")

    with pytest.raises(ConfigurationError) as exc_info:
        load_config(str(config_file))

    assert "Invalid YAML configuration" in str(exc_info.value)


def test_load_config_empty_file(tmp_path):
    """Test configuration loading fails with empty file."""
    config_file = tmp_path / "empty_config.yaml"
    config_file.write_text("")

    with pytest.raises(ConfigurationError) as exc_info:
        load_config(str(config_file))

    assert "Empty configuration file" in str(exc_info.value)


def test_apply_env_overrides():
    """Test environment variable overrides."""
    config = {
        'service': {'port': 8000},
        'logging': {'level': 'INFO'},
        'model': {'threshold': 0.5}
    }

    # Set environment variables
    os.environ['QUERY_PREDICTOR_PORT'] = '9000'
    os.environ['QUERY_PREDICTOR_LOG_LEVEL'] = 'DEBUG'
    os.environ['QUERY_PREDICTOR_THRESHOLD'] = '0.7'

    try:
        config = _apply_env_overrides(config)

        assert config['service']['port'] == 9000
        assert config['logging']['level'] == 'DEBUG'
        assert config['model']['threshold'] == 0.7
    finally:
        # Clean up environment variables
        del os.environ['QUERY_PREDICTOR_PORT']
        del os.environ['QUERY_PREDICTOR_LOG_LEVEL']
        del os.environ['QUERY_PREDICTOR_THRESHOLD']


def test_apply_env_overrides_feature_toggles():
    """Test feature toggle environment variable overrides."""
    config = {}

    os.environ['QUERY_PREDICTOR_ZERO_COST_ENABLED'] = 'true'
    os.environ['QUERY_PREDICTOR_HISTORICAL_FEATURES_ENABLED'] = 'false'

    try:
        config = _apply_env_overrides(config)

        assert config['zero_cost_filter']['enabled'] is True
        assert config['featurizer']['enable_historical_features'] is False
    finally:
        del os.environ['QUERY_PREDICTOR_ZERO_COST_ENABLED']
        del os.environ['QUERY_PREDICTOR_HISTORICAL_FEATURES_ENABLED']


def test_validate_config_success():
    """Test successful configuration validation."""
    config = {
        'service': {
            'name': 'test-service',
            'version': '1.0.0'
        },
        'logging': {
            'level': 'INFO'
        }
    }

    # Should not raise exception
    _validate_config(config)


def test_validate_config_missing_section():
    """Test validation fails with missing section."""
    config = {
        'logging': {'level': 'INFO'}
        # Missing 'service' section
    }

    with pytest.raises(ConfigurationError) as exc_info:
        _validate_config(config)

    assert "Missing required config section: service" in str(exc_info.value)


def test_validate_config_missing_field():
    """Test validation fails with missing field."""
    config = {
        'service': {
            'version': '1.0.0'
            # Missing 'name' field
        },
        'logging': {'level': 'INFO'}
    }

    with pytest.raises(ConfigurationError) as exc_info:
        _validate_config(config)

    assert "Missing required field: service.name" in str(exc_info.value)
