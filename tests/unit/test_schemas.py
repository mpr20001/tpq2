"""Unit tests for service schema models."""

import pytest
from pydantic import ValidationError
from query_predictor.service.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthCheckResponse,
    ErrorResponse,
    validate_s3_path,
    create_error_response
)


class TestPredictionRequest:
    """Test PredictionRequest model."""

    def test_valid_request(self):
        request = PredictionRequest(
            query="SELECT * FROM table",
            user="test_user",
            catalog="hive",
            schema="default",
            hour=12,
            clientInfo="trino-cli",
            sessionProperties={"key": "value"}
        )
        assert request.query == "SELECT * FROM table"
        assert request.user == "test_user"
        assert request.catalog == "hive"
        assert request.schema_name == "default"
        assert request.hour == 12

    def test_minimal_request(self):
        request = PredictionRequest(
            query="SELECT 1",
            user=None,
            catalog=None,
            schema=None,
            hour=None,
            clientInfo=None
        )
        assert request.query == "SELECT 1"
        assert request.user is None
        assert request.catalog is None
        assert request.schema_name is None
        assert request.hour is not None  # Should default to current hour

    def test_empty_query_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequest(
                query="",
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )

    def test_whitespace_query_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequest(
                query="   ",
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )

    def test_very_long_query_raises_error(self):
        long_query = "SELECT * FROM table WHERE " + " AND ".join([f"col{i} = {i}" for i in range(100000)])
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            PredictionRequest(
                query=long_query,
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )

    def test_invalid_hour_raises_error(self):
        with pytest.raises(ValidationError):
            PredictionRequest(
                query="SELECT 1", 
                hour=25,
                user=None,
                catalog=None,
                schema=None,
                clientInfo=None
            )

        with pytest.raises(ValidationError):
            PredictionRequest(
                query="SELECT 1", 
                hour=-1,
                user=None,
                catalog=None,
                schema=None,
                clientInfo=None
            )

    def test_user_normalization(self):
        request = PredictionRequest(
            query="SELECT 1", 
            user="Test_User ",
            catalog=None,
            schema=None,
            hour=None,
            clientInfo=None
        )
        assert request.user == "test_user"

    def test_null_catalog_schema(self):
        request = PredictionRequest(
            query="SELECT * FROM table",
            catalog=None,
            schema=None,
            user=None,
            hour=None,
            clientInfo=None
        )
        assert request.catalog is None
        assert request.schema_name is None

    def test_empty_string_catalog_schema_converted_to_none(self):
        request = PredictionRequest(
            query="SELECT * FROM table",
            catalog="",
            schema="",
            user=None,
            hour=None,
            clientInfo=None
        )
        assert request.catalog is None
        assert request.schema_name is None

    def test_to_query_data(self):
        request = PredictionRequest(
            query="SELECT * FROM table",
            user="test_user",
            catalog="hive",
            schema="default",
            hour=12,
            clientInfo="cli",
            sessionProperties={"key": "value"}
        )
        query_data = request.to_query_data()

        assert query_data.query == "SELECT * FROM table"
        assert query_data.user == "test_user"
        assert query_data.catalog == "hive"
        assert query_data.schema == "default"
        assert query_data.hour == 12
        assert query_data.client_info == "cli"
        assert query_data.session_properties["key"] == "value"

    def test_to_query_data_with_none_user(self):
        request = PredictionRequest(
            query="SELECT 1", 
            user=None,
            catalog=None,
            schema=None,
            hour=None,
            clientInfo=None
        )
        query_data = request.to_query_data()

        assert query_data.user == "unknown"


class TestPredictionResponse:
    """Test PredictionResponse model."""

    def test_valid_response(self):
        response = PredictionResponse(
            prediction="HEAVY",
            confidence=0.92,
            probability={"HEAVY": 0.92, "LIGHT": 0.08},
            model_version="1.0.0",
            featurizer_version="3.0.0",
            feature_count=78,
            processing_time_ms=45.2,
            feature_extraction_time_ms=38.1,
            model_inference_time_ms=7.1,
            query_info=None
        )
        assert response.prediction == "HEAVY"
        assert response.confidence == 0.92
        assert response.feature_count == 78

    def test_confidence_validation(self):
        # Valid confidence
        response = PredictionResponse(
            prediction="HEAVY",
            confidence=0.5,
            probability={"HEAVY": 0.5, "LIGHT": 0.5},
            model_version="1.0.0",
            featurizer_version="3.0.0",
            feature_count=78,
            processing_time_ms=10.0,
            feature_extraction_time_ms=5.0,
            model_inference_time_ms=5.0,
            query_info=None
        )
        assert response.confidence == 0.5

        # Invalid confidence > 1
        with pytest.raises(ValidationError):
            PredictionResponse(
                prediction="HEAVY",
                confidence=1.5,
                probability={},
                model_version="1.0.0",
                featurizer_version="3.0.0",
                feature_count=78,
                processing_time_ms=10.0,
                feature_extraction_time_ms=5.0,
                model_inference_time_ms=5.0,
                query_info=None
            )

        # Invalid confidence < 0
        with pytest.raises(ValidationError):
            PredictionResponse(
                prediction="HEAVY",
                confidence=-0.1,
                probability={},
                model_version="1.0.0",
                featurizer_version="3.0.0",
                feature_count=78,
                processing_time_ms=10.0,
                feature_extraction_time_ms=5.0,
                model_inference_time_ms=5.0,
                query_info=None
            )

    def test_with_warnings(self):
        response = PredictionResponse(
            prediction="HEAVY",
            confidence=0.5,
            probability={"HEAVY": 0.5},
            model_version="1.0.0",
            featurizer_version="3.0.0",
            feature_count=78,
            processing_time_ms=10.0,
            feature_extraction_time_ms=5.0,
            model_inference_time_ms=5.0,
            query_info=None,
            warnings=["Low confidence", "AST parsing timeout"]
        )
        assert len(response.warnings) == 2

    def test_with_query_info(self):
        response = PredictionResponse(
            prediction="HEAVY",
            confidence=0.9,
            probability={"HEAVY": 0.9},
            model_version="1.0.0",
            featurizer_version="3.0.0",
            feature_count=78,
            processing_time_ms=10.0,
            feature_extraction_time_ms=5.0,
            model_inference_time_ms=5.0,
            query_info={"table_count": 2, "join_count": 1}
        )
        assert response.query_info["table_count"] == 2


class TestHealthCheckResponse:
    """Test HealthCheckResponse model."""

    def test_healthy_response(self):
        response = HealthCheckResponse(
            status="healthy",
            version="1.0.0",
            featurizer_version="3.0.0",
            model_version="1.0.0",
            uptime_seconds=3600.5,
            timestamp="2024-01-15T10:30:00Z",
            s3_connection=True,
            model_loaded=True,
            feature_spec_loaded=True
        )
        assert response.status == "healthy"
        assert response.s3_connection is True
        assert response.model_loaded is True

    def test_unhealthy_response(self):
        response = HealthCheckResponse(
            status="unhealthy",
            version="1.0.0",
            featurizer_version="3.0.0",
            model_version=None,
            uptime_seconds=100.0,
            timestamp="2024-01-15T10:30:00Z",
            s3_connection=False,
            model_loaded=False,
            feature_spec_loaded=True
        )
        assert response.status == "unhealthy"
        assert response.s3_connection is False
        assert response.model_loaded is False


class TestErrorResponse:
    """Test ErrorResponse model."""

    def test_simple_error(self):
        error = ErrorResponse(
            error="ValidationError",
            message="Query cannot be empty",
            timestamp="2024-01-15T10:30:00Z"
        )
        assert error.error == "ValidationError"
        assert error.message == "Query cannot be empty"

    def test_error_with_details(self):
        error = ErrorResponse(
            error="ValidationError",
            message="Query cannot be empty",
            details={"field": "query", "provided_value": ""},
            timestamp="2024-01-15T10:30:00Z",
            request_id="req-123"
        )
        assert error.details["field"] == "query"
        assert error.request_id == "req-123"


class TestValidationUtilities:
    """Test validation utility functions."""

    def test_validate_s3_path_valid(self):
        assert validate_s3_path("s3://bucket/key/file.json") is True
        assert validate_s3_path("s3://my-bucket/path/to/file.onnx") is True

    def test_validate_s3_path_invalid(self):
        assert validate_s3_path("") is False
        assert validate_s3_path("http://bucket/key") is False
        assert validate_s3_path("s3://bucket") is False  # No key
        assert validate_s3_path("bucket/key") is False  # Missing s3://

    def test_create_error_response(self):
        error = create_error_response(
            error_type="ValidationError",
            message="Test error",
            details={"field": "test"},
            request_id="req-123"
        )
        assert error.error == "ValidationError"
        assert error.message == "Test error"
        assert error.details["field"] == "test"
        assert error.request_id == "req-123"
        assert error.timestamp is not None

    def test_create_error_response_minimal(self):
        error = create_error_response(
            error_type="InternalError",
            message="Something went wrong"
        )
        assert error.error == "InternalError"
        assert error.message == "Something went wrong"
        assert error.details is None
        assert error.request_id is None


class TestEdgeCasesAndErroneousQueries:
    """Test edge cases and erroneous queries."""

    def test_malformed_sql_queries(self):
        """Test various malformed SQL queries that should still be accepted."""
        # These are syntactically invalid but should be accepted for feature extraction
        malformed_queries = [
            "SELECT * FROM",  # Missing table name
            "SELECT * FROM table WHERE",  # Incomplete WHERE clause
            "SELECT * FROM table WHERE col =",  # Missing value
            "SELECT * FROM table WHERE col = 'unclosed string",  # Unclosed string
            "SELECT * FROM table WHERE col = 'value' AND",  # Incomplete AND clause
            "SELECT * FROM table WHERE col = 'value' OR",  # Incomplete OR clause
            "SELECT * FROM table WHERE col = 'value' AND (",  # Unclosed parenthesis
            "SELECT * FROM table WHERE col = 'value' AND (col2 = 'value2'",  # Unclosed parenthesis
            "SELECT * FROM table WHERE col = 'value' AND col2 = 'value2' AND",  # Trailing AND
            "SELECT * FROM table WHERE col = 'value' OR col2 = 'value2' OR",  # Trailing OR
        ]
        
        for query in malformed_queries:
            # These should be accepted by validation (syntax errors are handled by the parser)
            request = PredictionRequest(
                query=query,
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )
            assert request.query == query

    def test_sql_injection_patterns(self):
        """Test SQL injection patterns that should be logged but accepted."""
        injection_queries = [
            "SELECT * FROM users; DROP TABLE users;",
            "SELECT * FROM users WHERE id = 1; DELETE FROM users;",
            "SELECT * FROM users WHERE id = 1 UNION SELECT password FROM users;",
            "SELECT * FROM users WHERE id = 1 -- comment",
            "SELECT * FROM users WHERE id = 1 /* comment */",
            "SELECT * FROM users WHERE id = 1; EXEC xp_cmdshell('dir');",
            "SELECT * FROM users WHERE id = CAST(1 AS VARCHAR)",
            "SELECT * FROM users WHERE id = CONVERT(VARCHAR, 1)",
        ]
        
        for query in injection_queries:
            # These should be accepted but logged as warnings
            request = PredictionRequest(
                query=query,
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )
            assert request.query == query

    def test_unicode_and_special_characters(self):
        """Test queries with unicode and special characters."""
        unicode_queries = [
            "SELECT * FROM table WHERE name = 'José'",
            "SELECT * FROM table WHERE name = '中文'",
            "SELECT * FROM table WHERE name = '🚀'",
            "SELECT * FROM table WHERE name = 'café'",
            "SELECT * FROM table WHERE name = 'Müller'",
            "SELECT * FROM table WHERE name = 'Здравствуй'",
            "SELECT * FROM table WHERE name = 'مرحبا'",
            "SELECT * FROM table WHERE name = 'שלום'",
        ]
        
        for query in unicode_queries:
            request = PredictionRequest(
                query=query,
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )
            assert request.query == query

    def test_very_long_queries(self):
        """Test queries approaching the size limit."""
        # Create a query just under the 1MB limit
        long_query = "SELECT * FROM table WHERE " + " AND ".join([f"col{i} = {i}" for i in range(10000)])
        
        # This should be accepted (under 1MB)
        request = PredictionRequest(
            query=long_query,
            user=None,
            catalog=None,
            schema=None,
            hour=None,
            clientInfo=None
        )
        assert len(request.query) < 1000000  # Should be under 1MB
        
        # Test query that exceeds the limit
        very_long_query = "SELECT * FROM table WHERE " + " AND ".join([f"col{i} = {i}" for i in range(50000)])
        with pytest.raises(ValidationError, match="exceeds maximum length"):
            PredictionRequest(
                query=very_long_query,
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )

    def test_boundary_values(self):
        """Test boundary values for numeric fields."""
        # Test hour boundaries
        request = PredictionRequest(
            query="SELECT 1", 
            hour=0,
            user=None,
            catalog=None,
            schema=None,
            clientInfo=None
        )
        assert request.hour == 0
        
        request = PredictionRequest(
            query="SELECT 1", 
            hour=23,
            user=None,
            catalog=None,
            schema=None,
            clientInfo=None
        )
        assert request.hour == 23
        
        # Test invalid hours
        with pytest.raises(ValidationError):
            PredictionRequest(
                query="SELECT 1", 
                hour=24,
                user=None,
                catalog=None,
                schema=None,
                clientInfo=None
            )
        
        with pytest.raises(ValidationError):
            PredictionRequest(
                query="SELECT 1", 
                hour=-1,
                user=None,
                catalog=None,
                schema=None,
                clientInfo=None
            )

    def test_session_properties_edge_cases(self):
        """Test session properties with various data types and edge cases."""
        # Valid session properties
        valid_props = {
            "string_prop": "value",
            "int_prop": 42,
            "float_prop": 3.14,
            "bool_prop": True,
            "null_prop": None,
            "empty_string": "",
            "unicode_prop": "café"
        }
        
        request = PredictionRequest(
            query="SELECT 1",
            sessionProperties=valid_props,
            user=None,
            catalog=None,
            schema=None,
            hour=None,
            clientInfo=None
        )
        assert request.sessionProperties == valid_props
        
        # Test invalid session properties
        with pytest.raises(ValidationError):
            PredictionRequest(
                query="SELECT 1",
                sessionProperties={"key": [1, 2, 3]},  # List not allowed
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )
        
        with pytest.raises(ValidationError):
            PredictionRequest(
                query="SELECT 1",
                sessionProperties={123: "value"},  # Non-string key
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )
        
        with pytest.raises(ValidationError):
            PredictionRequest(
                query="SELECT 1",
                sessionProperties={"key": {"nested": "dict"}},  # Nested dict not allowed
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )

    def test_session_properties_size_limit(self):
        """Test session properties size limit."""
        # Create a large session properties dict
        large_props = {f"key{i}": f"value{i}" * 100 for i in range(1000)}
        
        with pytest.raises(ValidationError, match="Session properties exceed maximum size"):
            PredictionRequest(
                query="SELECT 1",
                sessionProperties=large_props,
                user=None,
                catalog=None,
                schema=None,
                hour=None,
                clientInfo=None
            )

    def test_client_info_edge_cases(self):
        """Test client info with various edge cases."""
        # Valid client info
        request = PredictionRequest(
            query="SELECT 1",
            clientInfo="trino-cli/1.0",
            user=None,
            catalog=None,
            schema=None,
            hour=None
        )
        assert request.clientInfo == "trino-cli/1.0"
        
        # Test client info length limit
        long_client_info = "a" * 501
        with pytest.raises(ValidationError, match="too long"):
            PredictionRequest(
                query="SELECT 1",
                clientInfo=long_client_info,
                user=None,
                catalog=None,
                schema=None,
                hour=None
            )

    def test_catalog_schema_edge_cases(self):
        """Test catalog and schema with various edge cases."""
        # Empty strings should be converted to None
        request = PredictionRequest(
            query="SELECT 1",
            catalog="",
            schema=""
        )
        assert request.catalog is None
        assert request.schema_name is None
        
        # Whitespace should be stripped
        request = PredictionRequest(
            query="SELECT 1",
            catalog="  hive  ",
            schema="  default  "
        )
        assert request.catalog == "hive"
        assert request.schema_name == "default"
        
        # None should remain None
        request = PredictionRequest(
            query="SELECT 1",
            catalog=None,
            schema=None
        )
        assert request.catalog is None
        assert request.schema_name is None

    def test_user_normalization_edge_cases(self):
        """Test user field normalization with edge cases."""
        # Test case normalization
        request = PredictionRequest(query="SELECT 1", user="Test_User")
        assert request.user == "test_user"
        
        # Test whitespace handling
        request = PredictionRequest(query="SELECT 1", user="  Test_User  ")
        assert request.user == "test_user"
        
        # Test None user
        request = PredictionRequest(query="SELECT 1", user=None)
        assert request.user is None
        
        # Test empty string user (should be normalized to None)
        request = PredictionRequest(query="SELECT 1", user="")
        assert request.user == ""  # Empty string is normalized but not converted to None

    def test_malformed_json_handling(self):
        """Test handling of malformed JSON data."""
        # This test would typically be handled at the API level,
        # but we can test the validation models handle various data types
        
        # Test with non-string query
        with pytest.raises(ValidationError):
            PredictionRequest(query=123)
        
        # Test with non-dict session properties
        with pytest.raises(ValidationError):
            PredictionRequest(query="SELECT 1", sessionProperties="not_a_dict")
        
        # Test with non-int hour
        with pytest.raises(ValidationError):
            PredictionRequest(query="SELECT 1", hour="not_an_int")

    def test_very_long_field_values(self):
        """Test very long field values."""
        # Test very long catalog name
        long_catalog = "a" * 1000
        request = PredictionRequest(
            query="SELECT 1",
            catalog=long_catalog
        )
        assert request.catalog == long_catalog
        
        # Test very long schema name
        long_schema = "a" * 1000
        request = PredictionRequest(
            query="SELECT 1",
            schema=long_schema
        )
        assert request.schema_name == long_schema

    def test_special_sql_keywords(self):
        """Test queries with special SQL keywords and reserved words."""
        special_queries = [
            "SELECT * FROM `table` WHERE `where` = 'value'",
            "SELECT * FROM table WHERE `order` = 'value'",
            "SELECT * FROM table WHERE `group` = 'value'",
            "SELECT * FROM table WHERE `having` = 'value'",
            "SELECT * FROM table WHERE `limit` = 'value'",
            "SELECT * FROM table WHERE `offset` = 'value'",
            "SELECT * FROM table WHERE `distinct` = 'value'",
            "SELECT * FROM table WHERE `all` = 'value'",
        ]
        
        for query in special_queries:
            request = PredictionRequest(query=query)
            assert request.query == query

    def test_nested_queries_and_ctes(self):
        """Test complex nested queries and CTEs."""
        complex_queries = [
            "WITH cte AS (SELECT * FROM table1) SELECT * FROM cte",
            "SELECT * FROM (SELECT * FROM table1) AS subquery",
            "SELECT * FROM table1 WHERE col IN (SELECT col FROM table2)",
            "SELECT * FROM table1 WHERE EXISTS (SELECT 1 FROM table2 WHERE table2.id = table1.id)",
            "SELECT * FROM table1 WHERE col = (SELECT col FROM table2 LIMIT 1)",
        ]
        
        for query in complex_queries:
            request = PredictionRequest(query=query)
            assert request.query == query

    def test_queries_with_comments(self):
        """Test queries with various comment styles."""
        comment_queries = [
            "SELECT * FROM table -- single line comment",
            "SELECT * FROM table /* multi line comment */",
            "SELECT * FROM table /* multi\nline\ncomment */",
            "SELECT * FROM table /* nested /* comment */ comment */",
            "SELECT * FROM table WHERE col = 'value' -- comment at end",
            "SELECT * FROM table WHERE col = 'value' /* comment at end */",
        ]
        
        for query in comment_queries:
            request = PredictionRequest(query=query)
            assert request.query == query
