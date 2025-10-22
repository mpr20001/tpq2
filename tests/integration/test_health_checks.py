"""Integration tests for health check endpoints."""

import pytest
from query_predictor.service.app import create_app
import tempfile
import yaml


@pytest.fixture
def test_config(tmp_path):
    """Create a test configuration file."""
    config_file = tmp_path / "test_config.yaml"
    config_data = {
        'service': {
            'name': 'test-query-predictor',
            'version': '1.0.0-test',
            'port': 8000,
            'workers': 2
        },
        'logging': {
            'level': 'DEBUG',
            'format': 'json',
            'output': 'stdout'
        }
    }

    with open(config_file, 'w') as f:
        yaml.dump(config_data, f)

    return str(config_file)


@pytest.fixture
def client(test_config):
    """Create a test client for the Flask app."""
    app = create_app(test_config)
    app.config['TESTING'] = True

    with app.test_client() as client:
        yield client


def test_liveness_endpoint(client):
    """Test liveness endpoint returns 200."""
    response = client.get('/manage/health/liveness')

    assert response.status_code == 200

    data = response.get_json()
    assert data['status'] == 'UP'
    assert 'service' in data


def test_readiness_endpoint(client):
    """Test readiness endpoint returns 200."""
    response = client.get('/manage/health/readiness')

    assert response.status_code == 200

    data = response.get_json()
    assert data['ready'] is True
    assert data['status'] == 'UP'
    assert 'service' in data
    assert 'version' in data
    assert 'checks' in data

    # all checks should be False (not implemented yet)
    checks = data['checks']
    assert checks['model_loaded'] is False
    assert checks['featurizer_ready'] is False
    assert checks['classifier_ready'] is False


def test_info_endpoint(client):
    """Test info endpoint returns service metadata."""
    response = client.get('/v1/info')

    assert response.status_code == 200

    data = response.get_json()
    assert data['service'] == 'test-query-predictor'
    assert data['version'] == '1.0.0-test'
    assert data['status'] == 'skeleton'
    assert data['phase'] == 'Service skeleton'
    assert 'endpoints' in data


def test_predict_endpoint_not_implemented(client):
    """Test predict endpoint returns 501 (not implemented)."""
    response = client.post('/v1/predict', json={
        'query': 'SELECT * FROM table',
        'user': 'test@example.com'
    })

    assert response.status_code == 501

    data = response.get_json()
    assert 'error' in data
    assert 'Not implemented yet' in data['error']


def test_liveness_multiple_requests(client):
    """Test liveness endpoint handles multiple requests."""
    for _ in range(10):
        response = client.get('/manage/health/liveness')
        assert response.status_code == 200


def test_readiness_multiple_requests(client):
    """Test readiness endpoint handles multiple requests."""
    for _ in range(10):
        response = client.get('/manage/health/readiness')
        assert response.status_code == 200
