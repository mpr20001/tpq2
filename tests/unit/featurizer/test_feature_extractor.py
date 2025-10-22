"""Unit tests for main FeatureExtractor orchestrator."""

import pytest
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor
from query_predictor.core.types.feature_spec import FeatureSpec


@pytest.fixture
def config():
    """Fixture for common config."""
    return {
        'ast_timeout_ms': 50,
        'enable_historical_features': False
    }


@pytest.fixture
def simple_query_data():
    """Fixture for simple query data."""
    return {
        'query': 'SELECT * FROM table',
        'user': 'test_user',
        'catalog': 'hive',
        'schema': 'default',
        'hour': 12
    }


class TestFeatureExtractorInit:
    """Test FeatureExtractor initialization."""

    def test_init_without_feature_spec(self, config):
        extractor = FeatureExtractor(config)

        assert extractor.feature_spec is not None
        assert extractor.feature_count == 78  # 78 features
        assert len(extractor.feature_names) == 78
        assert len(extractor.extractors) == 9  # 9 extractors

    def test_init_with_custom_feature_spec(self, config):
        # Create a custom feature spec that disables some features
        from query_predictor.core.types.feature_group import FeatureGroup
        from datetime import datetime, timezone

        feature_spec = FeatureSpec(
            version="3.0.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_groups={
                "sql": FeatureGroup(
                    name="sql",
                    feature_names=["query_length"],
                    enabled=False  # Disable this group
                )
            },
            feature_importance={},
            disabled_features=set(),
            importance_threshold=0.0
        )

        extractor = FeatureExtractor(config, feature_spec=feature_spec)
        assert extractor.feature_spec == feature_spec

    def test_version(self, config):
        extractor = FeatureExtractor(config)
        assert extractor.VERSION == "3.0.0"


class TestFeatureExtractorExtract:
    """Test FeatureExtractor extract method."""

    def test_extract_simple_query(self, config, simple_query_data):
        extractor = FeatureExtractor(config)
        features = extractor.extract(simple_query_data)

        assert len(features) == 78
        assert all(isinstance(f, float) for f in features)

    def test_extract_complex_query(self, config):
        extractor = FeatureExtractor(config)
        query_data = {
            'query': '''
                WITH cte AS (
                    SELECT user_id, COUNT(*) as cnt
                    FROM orders
                    WHERE status IN ('completed', 'pending')
                    GROUP BY user_id
                    HAVING COUNT(*) > 5
                )
                SELECT u.name, cte.cnt
                FROM users u
                JOIN cte ON u.id = cte.user_id
                WHERE u.created_date > '2024-01-01'
                ORDER BY cte.cnt DESC
                LIMIT 100
            ''',
            'user': 'test_user',
            'catalog': 'hive',
            'schema': 'default',
            'hour': 14
        }
        features = extractor.extract(query_data)

        assert len(features) == 78
        assert all(isinstance(f, float) for f in features)
        # Should have non-zero features for this complex query
        assert sum(features) > 0

    def test_extract_null_catalog_schema(self, config):
        extractor = FeatureExtractor(config)
        query_data = {
            'query': 'SELECT * FROM catalog.schema.table',
            'user': 'test_user',
            'catalog': None,  # NULL catalog
            'schema': None,   # NULL schema
            'hour': 12
        }
        features = extractor.extract(query_data)

        assert len(features) == 78
        # NULL-aware features should be set
        # Last 6 features are from NullAwareExtractor
        assert features[-6] == 1.0  # is_catalog_null
        assert features[-5] == 1.0  # is_schema_null

    def test_extract_empty_query(self, config):
        extractor = FeatureExtractor(config)
        query_data = {
            'query': '',
            'user': 'test_user',
            'hour': 12
        }
        # Empty query should raise ValueError
        with pytest.raises(ValueError, match="Query field is required"):
            extractor.extract(query_data)

    def test_extract_whitespace_only_query(self, config):
        extractor = FeatureExtractor(config)
        query_data = {
            'query': '   \n  \t  ',
            'user': 'test_user',
            'hour': 12
        }
        # Whitespace-only query should raise ValueError
        with pytest.raises(ValueError, match="Query cannot be empty or whitespace-only"):
            extractor.extract(query_data)

    def test_extract_query_too_long(self, config):
        extractor = FeatureExtractor(config)
        # Create a query that exceeds 1MB
        long_query = "SELECT * FROM table WHERE " + " OR ".join([f"col{i} = {i}" for i in range(100000)])
        query_data = {
            'query': long_query,
            'user': 'test_user',
            'hour': 12
        }
        # Query too long should raise ValueError with helpful message
        with pytest.raises(ValueError, match="Query exceeds maximum length"):
            extractor.extract(query_data)

    def test_extract_with_error_isolation(self, config):
        extractor = FeatureExtractor(config)

        # Create query data that might cause errors in some extractors
        query_data = {
            'query': 'INVALID SQL THAT MIGHT BREAK THINGS',
            'user': 'test_user',
            'hour': 12
        }
        features = extractor.extract(query_data)

        # Should still return 78 features (with zeros for failed extractors)
        assert len(features) == 78
        assert all(isinstance(f, float) for f in features)

    def test_extract_feature_count_mismatch_handling(self, config):
        extractor = FeatureExtractor(config)

        # Normal extraction should work
        query_data = {
            'query': 'SELECT * FROM table',
            'user': 'test_user',
            'hour': 12
        }
        features = extractor.extract(query_data)

        # Should always return exactly the expected feature count
        assert len(features) == extractor.feature_count


class TestFeatureExtractorReports:
    """Test FeatureExtractor reporting methods."""

    def test_get_extractor_summary(self, config):
        extractor = FeatureExtractor(config)
        summary = extractor.get_extractor_summary()

        assert summary['version'] == '3.0.0'
        assert summary['total_extractors'] == 9
        assert summary['total_active_features'] == 78
        assert len(summary['extractors']) == 9

        # Check each extractor has required fields
        for ext_info in summary['extractors']:
            assert 'name' in ext_info
            assert 'feature_count' in ext_info
            assert 'features' in ext_info

    def test_get_feature_importance_report_no_importance(self, config):
        extractor = FeatureExtractor(config)
        report = extractor.get_feature_importance_report()

        # Without importance data, should return error
        assert 'error' in report

    def test_get_feature_importance_report_with_importance(self, config):
        from query_predictor.core.types.feature_spec import FeatureSpec
        from query_predictor.core.types.feature_group import FeatureGroup
        from datetime import datetime, timezone

        # Create feature spec with importance
        feature_spec = FeatureSpec(
            version="3.0.0",
            created_at=datetime.now(timezone.utc).isoformat(),
            feature_groups={
                "sql": FeatureGroup(
                    name="sql",
                    feature_names=["query_length", "token_count"],
                    enabled=True
                )
            },
            feature_importance={
                "query_length": 0.9,
                "token_count": 0.1
            },
            disabled_features=set(),
            importance_threshold=0.0
        )

        extractor = FeatureExtractor(config, feature_spec=feature_spec)
        report = extractor.get_feature_importance_report()

        assert 'version' in report
        assert 'total_features' in report
        assert 'active_features' in report
        assert 'top_10_features' in report
        assert 'bottom_10_features' in report


class TestFeatureExtractorFeatureCount:
    """Test FeatureExtractor feature count property."""

    def test_feature_count_property(self, config):
        extractor = FeatureExtractor(config)
        assert extractor.feature_count == 78

    def test_feature_count_matches_extraction(self, config, simple_query_data):
        extractor = FeatureExtractor(config)
        features = extractor.extract(simple_query_data)

        assert len(features) == extractor.feature_count
        assert len(features) == len(extractor.feature_names)


class TestFeatureExtractorOrderConsistency:
    """Test that feature extraction order is consistent (train-serve parity)."""

    def test_extractor_order(self, config):
        extractor = FeatureExtractor(config)

        # Check that extractors are in the expected order
        expected_names = [
            'SQLFeatureExtractor',
            'TableJoinExtractor',
            'WhereClauseExtractor',
            'AggregationExtractor',
            'ASTFeatureExtractor',
            'ContextExtractor',
            'QueryTypeExtractor',
            'SetOperationExtractor',
            'NullAwareExtractor'
        ]

        actual_names = [ext.__class__.__name__ for ext in extractor.extractors]
        assert actual_names == expected_names

    def test_feature_order_consistency(self, config, simple_query_data):
        # Extract features twice and ensure order is identical
        extractor1 = FeatureExtractor(config)
        extractor2 = FeatureExtractor(config)

        features1 = extractor1.extract(simple_query_data)
        features2 = extractor2.extract(simple_query_data)

        assert features1 == features2


class TestFeatureExtractorEdgeCases:
    """Test edge cases and error conditions."""

    def test_missing_required_fields(self, config):
        extractor = FeatureExtractor(config)

        # Missing query field should raise ValueError
        query_data = {
            'user': 'test_user',
            'hour': 12
        }
        with pytest.raises(ValueError, match="Query field is required"):
            extractor.extract(query_data)

    def test_invalid_input_type(self, config):
        extractor = FeatureExtractor(config)

        # Non-dict input should raise ValueError
        with pytest.raises(ValueError, match="query_data must be a dictionary"):
            extractor.extract("not a dict")

        with pytest.raises(ValueError, match="query_data must be a dictionary"):
            extractor.extract(None)

        with pytest.raises(ValueError, match="query_data must be a dictionary"):
            extractor.extract(["list", "not", "dict"])

    def test_none_values(self, config):
        extractor = FeatureExtractor(config)

        query_data = {
            'query': 'SELECT * FROM table',
            'user': None,
            'catalog': None,
            'schema': None,
            'hour': 12
        }
        features = extractor.extract(query_data)

        assert len(features) == 78

    def test_very_long_query(self, config):
        extractor = FeatureExtractor(config)

        # Create a very long query
        long_query = "SELECT " + ", ".join([f"col{i}" for i in range(1000)]) + " FROM table"

        query_data = {
            'query': long_query,
            'user': 'test_user',
            'hour': 12
        }
        features = extractor.extract(query_data)

        assert len(features) == 78

    def test_special_characters_in_query(self, config):
        extractor = FeatureExtractor(config)

        query_data = {
            'query': "SELECT * FROM table WHERE col = 'value with \" quotes and \n newlines'",
            'user': 'test_user',
            'hour': 12
        }
        features = extractor.extract(query_data)

        assert len(features) == 78


class TestFeatureExtractorSparkUDF:
    """Test FeatureExtractor Spark UDF creation."""

    def test_create_udf_requires_pyspark(self, config):
        """Test that creating UDF without PySpark raises error."""
        from query_predictor.core import featurizer

        extractor = FeatureExtractor(config)

        # Mock PySpark as unavailable
        original_pyspark = featurizer.feature_extractor.PYSPARK_AVAILABLE
        try:
            featurizer.feature_extractor.PYSPARK_AVAILABLE = False

            with pytest.raises(ImportError, match="PySpark is required"):
                extractor.create_spark_udf()
        finally:
            featurizer.feature_extractor.PYSPARK_AVAILABLE = original_pyspark

    def test_udf_returns_callable(self, config):
        """Test that create_spark_udf returns a callable."""
        extractor = FeatureExtractor(config)

        # This will fail if PySpark not available, but that's expected in unit tests
        try:
            udf = extractor.create_spark_udf()
            assert callable(udf)
        except ImportError:
            pytest.skip("PySpark not available in unit test environment")

    def test_udf_output_matches_extract(self, config, simple_query_data):
        """Test that UDF output matches extract() for parity."""
        extractor = FeatureExtractor(config)

        try:
            # Create UDF
            udf = extractor.create_spark_udf()

            # Extract features normally
            expected_features = extractor.extract(simple_query_data)

            # Create a mock PySpark Row
            from unittest.mock import Mock
            row = Mock()
            row.query = simple_query_data['query']
            row.user = simple_query_data['user']
            row.catalog = simple_query_data.get('catalog')
            row.schema = simple_query_data.get('schema')
            row.hour = simple_query_data['hour']
            row.clientInfo = simple_query_data.get('clientInfo', '')

            # Call UDF function directly (without Spark context)
            udf_features = udf.func(row)

            # Verify match
            assert len(udf_features) == len(expected_features)
            assert udf_features == expected_features

        except ImportError:
            pytest.skip("PySpark not available in unit test environment")
