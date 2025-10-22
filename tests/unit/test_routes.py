"""Unit tests for API route handlers with comprehensive error handling tests."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from flask import Flask, jsonify
import logging

from query_predictor.service.routes import register_routes


@pytest.fixture
def app():
    """Create a Flask app for testing."""
    app = Flask(__name__)
    app.config['TESTING'] = True
    app.config['service'] = {
        'name': 'test-service',
        'version': '1.0.0'
    }
    return app


@pytest.fixture
def client(app):
    """Create a test client."""
    register_routes(app)
    return app.test_client()


class TestLivenessEndpoint:
    """Test cases for the liveness endpoint."""

    def test_liveness_success(self, client):
        """Test successful liveness check."""
        response = client.get('/manage/health/liveness')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'UP'
        assert data['service'] == 'test-service'

    def test_liveness_with_missing_service_config(self, app):
        """Test liveness with missing service config."""
        app.config.pop('service', None)
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/manage/health/liveness')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'UP'
        assert data['service'] == 'unknown'

    def test_liveness_with_empty_service_config(self, app):
        """Test liveness with empty service config."""
        app.config['service'] = {}
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/manage/health/liveness')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'UP'
        assert data['service'] == 'unknown'

    def test_liveness_config_access_error(self, app):
        """Test liveness when config access raises an exception."""
        # Set up config that will cause an error when accessed
        app.config['service'] = None
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/manage/health/liveness')
        
        assert response.status_code == 500
        data = response.get_json()
        assert data['status'] == 'DOWN'
        assert 'error' in data

    @patch('query_predictor.service.routes.logger')
    def test_liveness_error_logging(self, mock_logger, app):
        """Test that errors are properly logged."""
        # Set up config that will cause an error when accessed
        app.config['service'] = None
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/manage/health/liveness')
        
        assert response.status_code == 500
        mock_logger.error.assert_called_once()
        assert "Error in liveness check" in mock_logger.error.call_args[0][0]


class TestReadinessEndpoint:
    """Test cases for the readiness endpoint."""

    def test_readiness_success(self, client):
        """Test successful readiness check."""
        response = client.get('/manage/health/readiness')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['ready'] is True
        assert data['status'] == 'UP'
        assert data['service'] == 'test-service'
        assert data['version'] == '1.0.0'
        assert 'checks' in data
        assert data['checks']['model_loaded'] is False
        assert data['checks']['featurizer_ready'] is False
        assert data['checks']['classifier_ready'] is False

    def test_readiness_with_missing_service_config(self, app):
        """Test readiness with missing service config."""
        app.config.pop('service', None)
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/manage/health/readiness')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['ready'] is True
        assert data['status'] == 'UP'
        assert data['service'] == 'unknown'
        assert data['version'] == 'unknown'

    def test_readiness_with_partial_service_config(self, app):
        """Test readiness with partial service config."""
        app.config['service'] = {'name': 'partial-service'}
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/manage/health/readiness')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['service'] == 'partial-service'
        assert data['version'] == 'unknown'

    def test_readiness_config_access_error(self, app):
        """Test readiness when config access raises an exception."""
        # Set up config that will cause an error when accessed
        app.config['service'] = None
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/manage/health/readiness')
        
        assert response.status_code == 500
        data = response.get_json()
        assert data['ready'] is False
        assert data['status'] == 'DOWN'
        assert 'error' in data

    @patch('query_predictor.service.routes.logger')
    def test_readiness_error_logging(self, mock_logger, app):
        """Test that errors are properly logged."""
        # Set up config that will cause an error when accessed
        app.config['service'] = None
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/manage/health/readiness')
        
        assert response.status_code == 500
        mock_logger.error.assert_called_once()
        assert "Error in readiness check" in mock_logger.error.call_args[0][0]


class TestPredictEndpoint:
    """Test cases for the predict endpoint."""

    def test_predict_not_implemented(self, client):
        """Test predict endpoint returns 501 (not implemented)."""
        response = client.post('/v1/predict', json={
            'query': 'SELECT * FROM table',
            'user': 'test@example.com'
        })
        
        assert response.status_code == 501
        data = response.get_json()
        assert 'error' in data
        assert 'Not implemented yet' in data['error']
        assert 'message' in data

    def test_predict_with_invalid_json(self, client):
        """Test predict endpoint with invalid JSON."""
        response = client.post('/v1/predict', 
                             data='invalid json',
                             content_type='application/json')
        
        # Our handler still gets called and returns 501 (not implemented)
        assert response.status_code == 501

    @patch('query_predictor.service.routes.logger')
    def test_predict_error_logging(self, mock_logger, app):
        """Test that errors are properly logged in predict endpoint."""
        with patch('query_predictor.service.routes.logger') as mock_logger:
            # Mock the logger.warning to raise an exception
            mock_logger.warning.side_effect = Exception("Logging failed")
            register_routes(app)
            client = app.test_client()
            
            response = client.post('/v1/predict', json={'query': 'test'})
            
            assert response.status_code == 500
            mock_logger.error.assert_called_once()
            assert "Error in prediction endpoint" in mock_logger.error.call_args[0][0]

    def test_predict_with_empty_payload(self, client):
        """Test predict endpoint with empty payload."""
        response = client.post('/v1/predict', json={})
        
        assert response.status_code == 501
        data = response.get_json()
        assert 'error' in data

    def test_predict_with_malformed_payload(self, client):
        """Test predict endpoint with malformed payload."""
        response = client.post('/v1/predict', json={'invalid': 'data'})
        
        assert response.status_code == 501
        data = response.get_json()
        assert 'error' in data


class TestInfoEndpoint:
    """Test cases for the info endpoint."""

    def test_info_success(self, client):
        """Test successful info endpoint."""
        response = client.get('/v1/info')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['service'] == 'test-service'
        assert data['version'] == '1.0.0'
        assert data['status'] == 'skeleton'
        assert data['phase'] == 'Service skeleton'
        assert 'endpoints' in data
        assert len(data['endpoints']) == 4

    def test_info_with_missing_service_config(self, app):
        """Test info with missing service config."""
        app.config.pop('service', None)
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/v1/info')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['service'] == 'trino-query-predictor'  # default value
        assert data['version'] == '1.0.0'  # default value

    def test_info_with_partial_service_config(self, app):
        """Test info with partial service config."""
        app.config['service'] = {'name': 'partial-service'}
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/v1/info')
        
        assert response.status_code == 200
        data = response.get_json()
        assert data['service'] == 'partial-service'
        assert data['version'] == '1.0.0'  # default value

    def test_info_config_access_error(self, app):
        """Test info when config access raises an exception."""
        # Set up config that will cause an error when accessed
        app.config['service'] = None
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/v1/info')
        
        assert response.status_code == 500
        data = response.get_json()
        assert 'error' in data
        assert 'Internal server error' in data['error']

    @patch('query_predictor.service.routes.logger')
    def test_info_error_logging(self, mock_logger, app):
        """Test that errors are properly logged."""
        # Set up config that will cause an error when accessed
        app.config['service'] = None
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/v1/info')
        
        assert response.status_code == 500
        mock_logger.error.assert_called_once()
        assert "Error in info endpoint" in mock_logger.error.call_args[0][0]


class TestErrorHandlingEdgeCases:
    """Test edge cases and error scenarios."""

    def test_all_endpoints_with_none_config(self, app):
        """Test all endpoints when config is None."""
        app.config['service'] = None
        register_routes(app)
        client = app.test_client()
        
        # All endpoints should return 500 when config is None
        endpoints = [
            '/manage/health/liveness',
            '/manage/health/readiness', 
            '/v1/info'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 500

    def test_all_endpoints_with_malformed_config(self, app):
        """Test all endpoints with malformed config."""
        app.config['service'] = "not a dict"
        register_routes(app)
        client = app.test_client()
        
        # All endpoints should return 500 when config is malformed
        endpoints = [
            '/manage/health/liveness',
            '/manage/health/readiness',
            '/v1/info'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 500

    def test_concurrent_requests(self, client):
        """Test that endpoints handle concurrent requests properly."""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get('/manage/health/liveness')
            results.append(response.status_code)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)

    def test_endpoints_with_different_http_methods(self, client):
        """Test that endpoints reject unsupported HTTP methods."""
        # Test POST on GET-only endpoints
        response = client.post('/manage/health/liveness')
        assert response.status_code == 405  # Method Not Allowed
        
        response = client.post('/manage/health/readiness')
        assert response.status_code == 405
        
        response = client.post('/v1/info')
        assert response.status_code == 405
        
        # Test GET on POST-only endpoint
        response = client.get('/v1/predict')
        assert response.status_code == 405

    def test_endpoints_with_malformed_requests(self, client):
        """Test endpoints with various malformed requests."""
        # Test with invalid content types
        response = client.post('/v1/predict', 
                             data='test',
                             content_type='text/plain')
        # Should still return 501 (not implemented) rather than crash
        
        # Test with very large payload
        large_data = {'query': 'x' * 10000}
        response = client.post('/v1/predict', json=large_data)
        assert response.status_code == 501


class TestLoggingBehavior:
    """Test logging behavior in different scenarios."""

    @patch('query_predictor.service.routes.logger')
    def test_successful_requests_dont_log_errors(self, mock_logger, client):
        """Test that successful requests don't log errors."""
        client.get('/manage/health/liveness')
        client.get('/manage/health/readiness')
        client.get('/v1/info')
        client.post('/v1/predict', json={'query': 'test'})
        
        # No error logging should occur for successful requests
        mock_logger.error.assert_not_called()

    @patch('query_predictor.service.routes.logger')
    def test_error_logging_includes_exception_details(self, mock_logger, app):
        """Test that error logging includes exception details."""
        # Set up config that will cause an error when accessed
        app.config['service'] = None
        register_routes(app)
        client = app.test_client()
        
        response = client.get('/manage/health/liveness')
        
        assert response.status_code == 500
        mock_logger.error.assert_called_once()
        logged_message = mock_logger.error.call_args[0][0]
        assert "Error in liveness check" in logged_message
        assert "'NoneType' object has no attribute 'get'" in logged_message


class TestResponseFormat:
    """Test response format consistency."""

    def test_error_responses_have_consistent_format(self, app):
        """Test that error responses have consistent format."""
        # Set up config that will cause an error when accessed
        app.config['service'] = None
        register_routes(app)
        client = app.test_client()
        
        # Test all endpoints return consistent error format
        endpoints = [
            '/manage/health/liveness',
            '/manage/health/readiness',
            '/v1/info'
        ]
        
        for endpoint in endpoints:
            response = client.get(endpoint)
            assert response.status_code == 500
            data = response.get_json()
            assert 'error' in data
            assert isinstance(data['error'], str)
            assert len(data['error']) > 0

    def test_success_responses_have_expected_fields(self, client):
        """Test that success responses have expected fields."""
        # Test liveness
        response = client.get('/manage/health/liveness')
        data = response.get_json()
        assert 'status' in data
        assert 'service' in data
        
        # Test readiness
        response = client.get('/manage/health/readiness')
        data = response.get_json()
        assert 'ready' in data
        assert 'status' in data
        assert 'service' in data
        assert 'version' in data
        assert 'checks' in data
        
        # Test info
        response = client.get('/v1/info')
        data = response.get_json()
        assert 'service' in data
        assert 'version' in data
        assert 'status' in data
        assert 'phase' in data
        assert 'endpoints' in data
        
        # Test predict
        response = client.post('/v1/predict', json={'query': 'test'})
        data = response.get_json()
        assert 'error' in data
        assert 'message' in data
