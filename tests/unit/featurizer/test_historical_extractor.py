"""
Unit tests for HistoricalFeatureExtractor.

Tests historical feature extraction with known stats, cold-start, and NULL handling.
"""

import pytest
from query_predictor.core.featurizer.extractors.historical_extractor import HistoricalFeatureExtractor
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.types.historical_stats import (
    HistoricalStatsSchema,
    UserStats,
    CatalogStats,
    SchemaStats
)


class TestHistoricalFeatureExtractor:
    """Test HistoricalFeatureExtractor with various scenarios."""

    @pytest.fixture
    def sample_historical_stats(self):
        """Create sample historical stats for testing."""
        user1_stats = UserStats(
            query_count=1000,
            heavy_rate=0.15,
            avg_cpu_seconds=60.0,
            p90_cpu_seconds=180.0,
            avg_memory_gb=12.0,
            p90_memory_gb=25.0,
            unique_catalogs=3,
            unique_schemas=5,
            cpu_stddev=30.0
        )

        user2_stats = UserStats(
            query_count=50,
            heavy_rate=0.05,
            avg_cpu_seconds=20.0,
            p90_cpu_seconds=50.0,
            avg_memory_gb=5.0,
            p90_memory_gb=10.0,
            unique_catalogs=1,
            unique_schemas=2,
            cpu_stddev=10.0
        )

        hive_catalog_stats = CatalogStats(
            query_count=5000,
            heavy_rate=0.12,
            avg_cpu_seconds=55.0,
            p50_cpu_seconds=30.0,
            p75_cpu_seconds=75.0,
            p90_cpu_seconds=150.0,
            p95_cpu_seconds=250.0,
            avg_memory_gb=10.0,
            p95_memory_gb=20.0,
            unique_users=100,
            unique_schemas=20
        )

        null_catalog_stats = CatalogStats(
            query_count=500,
            heavy_rate=0.08,
            avg_cpu_seconds=25.0,
            p50_cpu_seconds=15.0,
            p75_cpu_seconds=35.0,
            p90_cpu_seconds=60.0,
            p95_cpu_seconds=100.0,
            avg_memory_gb=5.0,
            p95_memory_gb=10.0,
            unique_users=30,
            unique_schemas=5
        )

        prod_schema_stats = SchemaStats(
            query_count=3000,
            heavy_rate=0.10,
            avg_cpu_seconds=50.0,
            p50_cpu_seconds=28.0,
            p75_cpu_seconds=70.0,
            p90_cpu_seconds=140.0,
            p95_cpu_seconds=230.0,
            avg_memory_gb=9.0,
            p95_memory_gb=18.0,
            unique_users=80,
            unique_catalogs=5
        )

        null_schema_stats = SchemaStats(
            query_count=400,
            heavy_rate=0.07,
            avg_cpu_seconds=22.0,
            p50_cpu_seconds=12.0,
            p75_cpu_seconds=32.0,
            p90_cpu_seconds=55.0,
            p95_cpu_seconds=95.0,
            avg_memory_gb=4.5,
            p95_memory_gb=9.0,
            unique_users=25,
            unique_catalogs=3
        )

        stats_schema = HistoricalStatsSchema(
            version='1.0.0',
            created_at='2025-01-15T10:00:00',
            data_date_range={'start': '2025-06-01', 'end': '2025-07-15'},
            users={
                'user1@example.com': user1_stats,
                'user2@example.com': user2_stats
            },
            catalogs={
                'hive': hive_catalog_stats,
                'NULL': null_catalog_stats
            },
            schemas={
                'production': prod_schema_stats,
                'NULL': null_schema_stats
            },
            total_queries_analyzed=10000,
            heavy_rate_overall=0.11
        )

        return stats_schema.to_dict()

    @pytest.fixture
    def extractor(self, sample_historical_stats):
        """Create HistoricalFeatureExtractor with sample stats."""
        return HistoricalFeatureExtractor(
            config={},
            historical_stats=sample_historical_stats
        )

    def test_initialization(self, sample_historical_stats):
        """Test extractor initialization with historical stats."""
        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=sample_historical_stats
        )

        assert extractor.stats_schema is not None
        assert extractor.stats_schema.version == '1.0.0'
        assert len(extractor.stats_schema.users) == 2
        assert len(extractor.stats_schema.catalogs) == 2
        assert len(extractor.stats_schema.schemas) == 2

    def test_feature_names(self, extractor):
        """Test that extractor returns correct feature names."""
        feature_names = extractor.get_feature_names()

        assert len(feature_names) == 17

        # Check user features
        assert 'user_query_count_norm' in feature_names
        assert 'user_heavy_rate' in feature_names
        assert 'user_avg_cpu_seconds_norm' in feature_names
        assert 'user_p90_cpu_seconds_norm' in feature_names
        assert 'user_avg_memory_gb_norm' in feature_names
        assert 'user_catalog_diversity_norm' in feature_names

        # Check catalog features
        assert 'catalog_query_count_norm' in feature_names
        assert 'catalog_heavy_rate' in feature_names

        # Check schema features
        assert 'schema_query_count_norm' in feature_names
        assert 'schema_heavy_rate' in feature_names

        # Check cold-start indicator
        assert 'is_cold_start' in feature_names

    def test_extract_known_entities(self, extractor):
        """Test extraction with known user, catalog, and schema."""
        query_data = QueryData(
            query='SELECT * FROM table',
            user='user1@example.com',
            catalog='hive',
            schema='production',
            hour=14
        )

        features = extractor.extract(query_data)

        # Verify feature count
        assert len(features) == 17

        # All features should be floats
        assert all(isinstance(f, float) for f in features)

        # No NaN or inf
        assert all(f == f for f in features)  # NaN check
        assert all(abs(f) != float('inf') for f in features)  # inf check

        # Should NOT be cold-start (all entities known)
        assert features[-1] == 0.0  # is_cold_start = False

        # User features should be non-zero (known user with 1000 queries)
        assert features[0] > 0.0  # user_query_count_norm
        assert features[1] == 0.15  # user_heavy_rate

    def test_extract_null_catalog(self, extractor):
        """Test extraction with NULL catalog."""
        query_data = QueryData(
            query='SELECT * FROM table',
            user='user1@example.com',
            catalog=None,  # NULL catalog
            schema='production',
            hour=14
        )

        features = extractor.extract(query_data)

        assert len(features) == 17

        # Should NOT be cold-start (user and schema are known)
        assert features[-1] == 0.0

        # Catalog features should use 'NULL' catalog stats
        # catalog_query_count_norm (index 6)
        assert features[6] > 0.0  # Should have NULL catalog stats

    def test_extract_null_schema(self, extractor):
        """Test extraction with NULL schema."""
        query_data = QueryData(
            query='SELECT * FROM table',
            user='user1@example.com',
            catalog='hive',
            schema=None,  # NULL schema
            hour=14
        )

        features = extractor.extract(query_data)

        assert len(features) == 17

        # Should NOT be cold-start (user and catalog are known)
        assert features[-1] == 0.0

        # Schema features should use 'NULL' schema stats
        # schema_query_count_norm (index 12)
        assert features[12] > 0.0  # Should have NULL schema stats

    def test_extract_cold_start_all_unknown(self, extractor):
        """Test extraction with all unknown entities (cold-start)."""
        query_data = QueryData(
            query='SELECT * FROM table',
            user='unknown.user@example.com',
            catalog='unknown_catalog',
            schema='unknown_schema',
            hour=14
        )

        features = extractor.extract(query_data)

        assert len(features) == 17

        # Should be cold-start (all entities unknown)
        assert features[-1] == 1.0  # is_cold_start = True

        # Features should use cold-start defaults (non-zero)
        assert all(f >= 0.0 for f in features)

        # User features should use defaults
        # user_query_count_norm (index 0) - default is 1, normalized by 10000
        assert features[0] > 0.0
        # user_heavy_rate (index 1) - default is 0.1
        assert features[1] == 0.1

    def test_extract_partial_cold_start(self, extractor):
        """Test extraction with some known, some unknown entities."""
        query_data = QueryData(
            query='SELECT * FROM table',
            user='user1@example.com',  # Known
            catalog='unknown_catalog',  # Unknown
            schema='unknown_schema',    # Unknown
            hour=14
        )

        features = extractor.extract(query_data)

        assert len(features) == 17

        # Should NOT be cold-start (user is known)
        assert features[-1] == 0.0

        # User features should use actual stats
        assert features[0] > 0.0  # user_query_count_norm
        assert features[1] == 0.15  # user_heavy_rate (from user1 stats)

        # Catalog and schema features should use defaults
        # These should be non-zero but use default values
        assert features[6] > 0.0  # catalog_query_count_norm (default)
        assert features[7] == 0.1  # catalog_heavy_rate (default)

    def test_feature_normalization(self, extractor):
        """Test that features are properly normalized."""
        query_data = QueryData(
            query='SELECT * FROM table',
            user='user1@example.com',
            catalog='hive',
            schema='production',
            hour=14
        )

        features = extractor.extract(query_data)

        # Normalized features should be in [0, 1] range (or slightly above for outliers)
        for i, feature in enumerate(features):
            # Heavy rate features should be in [0, 1]
            if 'heavy_rate' in extractor.get_feature_names()[i]:
                assert 0.0 <= feature <= 1.0, \
                    f"Feature {extractor.get_feature_names()[i]} out of range: {feature}"

            # is_cold_start should be 0 or 1
            if extractor.get_feature_names()[i] == 'is_cold_start':
                assert feature in [0.0, 1.0]

    def test_feature_consistency(self, extractor):
        """Test that same input produces same output (deterministic)."""
        query_data = QueryData(
            query='SELECT * FROM table',
            user='user1@example.com',
            catalog='hive',
            schema='production',
            hour=14
        )

        features1 = extractor.extract(query_data)
        features2 = extractor.extract(query_data)
        features3 = extractor.extract(query_data)

        # All extractions should be identical
        assert features1 == features2
        assert features2 == features3

    def test_extract_with_different_users(self, extractor):
        """Test that different users produce different features."""
        query_data1 = QueryData(
            query='SELECT * FROM table',
            user='user1@example.com',  # Heavy user (1000 queries, 15% heavy)
            catalog='hive',
            schema='production',
            hour=14
        )

        query_data2 = QueryData(
            query='SELECT * FROM table',
            user='user2@example.com',  # Light user (50 queries, 5% heavy)
            catalog='hive',
            schema='production',
            hour=14
        )

        features1 = extractor.extract(query_data1)
        features2 = extractor.extract(query_data2)

        # User features should differ
        assert features1[0] != features2[0]  # user_query_count_norm
        assert features1[1] != features2[1]  # user_heavy_rate

        # user1 should have higher heavy_rate
        assert features1[1] > features2[1]

    def test_error_handling(self, sample_historical_stats):
        """Test that extractor handles errors gracefully."""
        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=sample_historical_stats
        )

        # Extract with minimal query data (should not crash)
        query_data = QueryData(
            query='SELECT 1',
            user='test@example.com',
            hour=12
        )

        features = extractor.extract(query_data)

        # Should return 17 features (not crash)
        assert len(features) == 17
        assert all(isinstance(f, float) for f in features)

    def test_invalid_historical_stats(self):
        """Test that extractor validates historical stats schema."""
        invalid_stats = {
            'version': '1.0.0',
            # Missing required fields
        }

        with pytest.raises(ValueError):
            HistoricalFeatureExtractor(
                config={},
                historical_stats=invalid_stats
            )

    def test_empty_historical_stats(self):
        """Test extractor with empty historical stats (all cold-start)."""
        empty_stats = HistoricalStatsSchema(
            version='1.0.0',
            created_at='2025-01-15T10:00:00',
            data_date_range={'start': '2025-06-01', 'end': '2025-07-15'},
            users={},       # No users
            catalogs={},    # No catalogs
            schemas={},     # No schemas
            total_queries_analyzed=1,  # Must be positive for validation
            heavy_rate_overall=0.0
        )

        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=empty_stats.to_dict()
        )

        query_data = QueryData(
            query='SELECT * FROM table',
            user='any.user@example.com',
            catalog='any_catalog',
            schema='any_schema',
            hour=14
        )

        features = extractor.extract(query_data)

        # All queries should be cold-start
        assert features[-1] == 1.0  # is_cold_start = True

        # All features should use defaults
        assert len(features) == 17
        assert all(f >= 0.0 for f in features)


class TestHistoricalFeatureExtractorIntegration:
    """Integration tests for historical feature extractor."""

    def test_train_serve_parity_schema(self):
        """Test that training output schema matches inference input."""
        # Simulate training output
        user_stats = UserStats(
            query_count=100,
            heavy_rate=0.1,
            avg_cpu_seconds=50.0,
            p90_cpu_seconds=150.0,
            avg_memory_gb=10.0,
            p90_memory_gb=20.0,
            unique_catalogs=2,
            unique_schemas=3,
            cpu_stddev=25.0
        )

        stats_schema = HistoricalStatsSchema(
            version='1.0.0',
            created_at='2025-01-15T10:00:00',
            data_date_range={'start': '2025-06-01', 'end': '2025-07-15'},
            users={'test.user': user_stats},
            catalogs={},
            schemas={},
            total_queries_analyzed=100,
            heavy_rate_overall=0.1
        )

        # Serialize (what training saves to S3)
        stats_dict = stats_schema.to_dict()

        # Deserialize (what inference loads from S3)
        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=stats_dict
        )

        # Extract features (should work without errors)
        query_data = QueryData(
            query='SELECT * FROM table',
            user='test.user',
            hour=12
        )

        features = extractor.extract(query_data)

        # Verify success
        assert len(features) == 17
        assert all(isinstance(f, float) for f in features)

    def test_feature_name_order_matches_extract(self):
        """Test that feature names order matches extract() output."""
        stats_schema = HistoricalStatsSchema(
            version='1.0.0',
            created_at='2025-01-15T10:00:00',
            data_date_range={'start': '2025-06-01', 'end': '2025-07-15'},
            users={},
            catalogs={},
            schemas={},
            total_queries_analyzed=1,  # Must be positive for validation
            heavy_rate_overall=0.0
        )

        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=stats_schema.to_dict()
        )

        feature_names = extractor.get_feature_names()

        # Feature names order is critical for train-serve parity
        assert feature_names[0] == 'user_query_count_norm'
        assert feature_names[6] == 'catalog_query_count_norm'
        assert feature_names[12] == 'schema_query_count_norm'
        assert feature_names[16] == 'is_cold_start'

        # Verify count
        assert len(feature_names) == 17


class TestHistoricalFeatureExtractorSparkUDF:
    """Test HistoricalFeatureExtractor Spark UDF creation."""

    @pytest.fixture
    def sample_stats(self):
        """Sample historical stats for testing."""
        user_stats = UserStats(
            query_count=1000,
            heavy_rate=0.15,
            avg_cpu_seconds=60.0,
            p90_cpu_seconds=180.0,
            avg_memory_gb=12.0,
            p90_memory_gb=25.0,
            unique_catalogs=3,
            unique_schemas=5,
            cpu_stddev=30.0
        )

        catalog_stats = CatalogStats(
            query_count=5000,
            heavy_rate=0.12,
            avg_cpu_seconds=55.0,
            p50_cpu_seconds=30.0,
            p75_cpu_seconds=75.0,
            p90_cpu_seconds=150.0,
            p95_cpu_seconds=250.0,
            avg_memory_gb=10.0,
            p95_memory_gb=20.0,
            unique_users=100,
            unique_schemas=20
        )

        schema_stats = SchemaStats(
            query_count=3000,
            heavy_rate=0.10,
            avg_cpu_seconds=50.0,
            p50_cpu_seconds=28.0,
            p75_cpu_seconds=70.0,
            p90_cpu_seconds=140.0,
            p95_cpu_seconds=230.0,
            avg_memory_gb=9.0,
            p95_memory_gb=18.0,
            unique_users=80,
            unique_catalogs=5
        )

        stats_schema = HistoricalStatsSchema(
            version='1.0.0',
            created_at='2025-01-15T10:00:00',
            data_date_range={'start': '2025-06-01', 'end': '2025-07-15'},
            users={'test.user@example.com': user_stats},
            catalogs={'hive': catalog_stats},
            schemas={'production': schema_stats},
            total_queries_analyzed=10000,
            heavy_rate_overall=0.12
        )

        return stats_schema.to_dict()

    def test_create_udf_requires_pyspark(self, sample_stats):
        """Test that creating UDF without PySpark raises error."""
        from query_predictor.core.featurizer.extractors import historical_extractor

        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=sample_stats
        )

        # Mock PySpark as unavailable
        original_pyspark = historical_extractor.PYSPARK_AVAILABLE
        try:
            historical_extractor.PYSPARK_AVAILABLE = False

            with pytest.raises(ImportError, match="PySpark is required"):
                extractor.create_spark_udf()
        finally:
            historical_extractor.PYSPARK_AVAILABLE = original_pyspark

    def test_udf_returns_callable(self, sample_stats):
        """Test that create_spark_udf returns a callable."""
        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=sample_stats
        )

        try:
            udf = extractor.create_spark_udf()
            assert callable(udf)
        except ImportError:
            pytest.skip("PySpark not available in unit test environment")

    def test_udf_output_matches_extract(self, sample_stats):
        """Test that UDF output matches extract() for parity."""
        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=sample_stats
        )

        try:
            # Create UDF
            udf = extractor.create_spark_udf()

            # Prepare query data
            query_data = QueryData(
                query='SELECT * FROM table',
                user='test.user@example.com',
                catalog='hive',
                schema='production',
                hour=14
            )

            # Extract features normally
            expected_features = extractor.extract(query_data)

            # Create a mock PySpark Row
            from unittest.mock import Mock
            row = Mock()
            row.user = query_data.user
            row.catalog = query_data.catalog
            row.schema = query_data.schema

            # Call UDF function directly (without Spark context)
            udf_features = udf.func(row)

            # Verify match
            assert len(udf_features) == len(expected_features)
            assert len(udf_features) == 17
            assert udf_features == expected_features

        except ImportError:
            pytest.skip("PySpark not available in unit test environment")
