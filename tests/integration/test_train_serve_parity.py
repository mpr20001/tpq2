"""
Train-serve parity integration tests.

Critical tests ensuring training output is directly compatible with inference input.
Tests the complete flow: HistoricalStatsComputer → S3 JSON → HistoricalFeatureExtractor
"""

import pytest
from unittest.mock import Mock
from query_predictor.core.types.historical_stats import (
    HistoricalStatsSchema,
    UserStats,
    CatalogStats,
    SchemaStats
)
from query_predictor.core.featurizer.extractors.historical_extractor import HistoricalFeatureExtractor
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor
from query_predictor.core.types.query_data import QueryData


class TestTrainServeParity:
    """
    Critical tests for train-serve parity.

    These tests verify that:
    1. Training output schema matches inference input expectations
    2. Field names are identical between training and inference
    3. Normalization constants are shared
    4. Serialization/deserialization preserves data
    """

    @pytest.fixture
    def mock_training_stats(self):
        """Simulate stats computed by training pipeline."""
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

        return stats_schema

    def test_training_output_directly_usable_by_inference(self, mock_training_stats):
        """
        CRITICAL: Training output must be directly usable by inference.

        This is the most important parity test - it verifies the complete flow:
        1. Training computes stats (HistoricalStatsComputer)
        2. Stats are serialized to JSON (to_dict)
        3. JSON is saved to S3
        4. Inference loads JSON from S3 (from_dict)
        5. Inference uses stats for feature extraction (HistoricalFeatureExtractor)
        """
        # Step 1: Simulate training output (serialization)
        stats_dict = mock_training_stats.to_dict()

        # Verify dict structure (what gets saved to S3)
        assert 'version' in stats_dict
        assert 'users' in stats_dict
        assert 'catalogs' in stats_dict
        assert 'schemas' in stats_dict

        # Step 2: Simulate inference input (deserialization)
        loaded_schema = HistoricalStatsSchema.from_dict(stats_dict)

        # Verify deserialization worked
        assert loaded_schema.version == mock_training_stats.version
        assert len(loaded_schema.users) == len(mock_training_stats.users)

        # Step 3: Use stats in inference extractor
        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=stats_dict
        )

        # Step 4: Extract features (should work without errors)
        query_data = QueryData(
            query='SELECT * FROM table',
            user='test.user@example.com',
            catalog='hive',
            schema='production',
            hour=14
        )

        features = extractor.extract(query_data)

        # Step 5: Verify features are valid
        assert len(features) == 17
        assert all(isinstance(f, float) for f in features)
        assert all(f == f for f in features)  # No NaN
        assert all(abs(f) != float('inf') for f in features)  # No inf

        # Step 6: Verify features are reasonable (not all zeros)
        assert sum(features) > 0.0

    def test_field_names_match_between_training_and_inference(self):
        """
        CRITICAL: Field names must match exactly.

        If training computes 'avg_cpu_seconds', inference MUST use 'avg_cpu_seconds'.
        Mismatched field names will cause deserialization to fail.
        """
        # Training produces UserStats with these fields
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

        # Get field names from dataclass
        training_field_names = set(user_stats.__dict__.keys())

        # Expected field names (what inference expects)
        expected_field_names = {
            'query_count', 'heavy_rate', 'avg_cpu_seconds', 'p90_cpu_seconds',
            'avg_memory_gb', 'p90_memory_gb', 'unique_catalogs', 'unique_schemas',
            'cpu_stddev'
        }

        # Verify exact match
        assert training_field_names == expected_field_names, \
            f"Field name mismatch! " \
            f"Missing: {expected_field_names - training_field_names}, " \
            f"Extra: {training_field_names - expected_field_names}"

    def test_normalization_constants_shared(self):
        """
        CRITICAL: Normalization constants must be shared.

        Training and inference must use the same normalizers, otherwise features will have
        different distributions (train-serve skew).
        """
        from query_predictor.core.featurizer import constants

        # These constants MUST exist and be shared
        assert hasattr(constants, 'HISTORICAL_QUERY_COUNT_NORMALIZER')
        assert hasattr(constants, 'HISTORICAL_CPU_SECONDS_NORMALIZER')
        assert hasattr(constants, 'HISTORICAL_MEMORY_GB_NORMALIZER')
        assert hasattr(constants, 'HISTORICAL_DIVERSITY_NORMALIZER')

        # Verify they are positive floats
        assert constants.HISTORICAL_QUERY_COUNT_NORMALIZER > 0.0
        assert constants.HISTORICAL_CPU_SECONDS_NORMALIZER > 0.0
        assert constants.HISTORICAL_MEMORY_GB_NORMALIZER > 0.0
        assert constants.HISTORICAL_DIVERSITY_NORMALIZER > 0.0

    def test_cold_start_defaults_shared(self):
        """
        CRITICAL: Cold-start defaults must be shared.

        Training and inference must use the same defaults for unknown entities.
        """
        from query_predictor.core.featurizer import constants

        # Cold-start defaults must exist
        assert hasattr(constants, 'COLD_START_DEFAULTS')

        defaults = constants.COLD_START_DEFAULTS

        # Verify required fields
        required_fields = [
            'query_count', 'heavy_rate', 'avg_cpu_seconds', 'p90_cpu_seconds',
            'avg_memory_gb', 'unique_catalogs', 'unique_schemas', 'unique_users'
        ]

        for field in required_fields:
            assert field in defaults, f"Missing cold-start default: {field}"

    def test_feature_count_78_vs_95(self, mock_training_stats):
        """
        Test that feature count is 78 without historical, 95 with historical.
        """
        stats_dict = mock_training_stats.to_dict()

        # Test without historical features (78 features)
        config_without_historical = {
            'enable_historical_features': False,
            'ast_timeout_ms': 50
        }

        extractor_without = FeatureExtractor(
            config=config_without_historical,
            historical_stats=None
        )

        query_data = QueryData(
            query='SELECT * FROM table',
            user='test.user@example.com',
            hour=14
        )

        features_without = extractor_without.extract({'query': 'SELECT * FROM table', 'user': 'test', 'hour': 14})
        assert len(features_without) == 78, f"Expected 78 features without historical, got {len(features_without)}"

        # Test with historical features (95 features)
        config_with_historical = {
            'enable_historical_features': True,
            'ast_timeout_ms': 50
        }

        extractor_with = FeatureExtractor(
            config=config_with_historical,
            historical_stats=stats_dict
        )

        features_with = extractor_with.extract({'query': 'SELECT * FROM table', 'user': 'test', 'hour': 14})
        assert len(features_with) == 95, f"Expected 95 features with historical, got {len(features_with)}"

    def test_feature_extraction_deterministic(self, mock_training_stats):
        """
        Test that feature extraction is deterministic (same input → same output).
        """
        stats_dict = mock_training_stats.to_dict()

        config = {
            'enable_historical_features': True,
            'ast_timeout_ms': 50
        }

        extractor = FeatureExtractor(
            config=config,
            historical_stats=stats_dict
        )

        query_dict = {
            'query': 'SELECT * FROM users WHERE id > 100',
            'user': 'test.user@example.com',
            'catalog': 'hive',
            'schema': 'production',
            'hour': 14
        }

        # Extract features multiple times
        features1 = extractor.extract(query_dict)
        features2 = extractor.extract(query_dict)
        features3 = extractor.extract(query_dict)

        # All should be identical (deterministic)
        assert features1 == features2
        assert features2 == features3

    def test_null_catalog_schema_handling(self, mock_training_stats):
        """
        Test that NULL catalog/schema are handled with 'NULL' key.
        """
        # Add NULL catalog and schema to stats
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

        mock_training_stats.catalogs['NULL'] = null_catalog_stats
        mock_training_stats.schemas['NULL'] = SchemaStats(
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

        stats_dict = mock_training_stats.to_dict()

        config = {
            'enable_historical_features': True,
            'ast_timeout_ms': 50
        }

        extractor = FeatureExtractor(
            config=config,
            historical_stats=stats_dict
        )

        # Test with NULL catalog and schema
        query_dict = {
            'query': 'SELECT * FROM table',
            'user': 'test.user@example.com',
            'catalog': None,  # NULL
            'schema': None,   # NULL
            'hour': 14
        }

        features = extractor.extract(query_dict)

        # Should work without errors
        assert len(features) == 95
        assert all(isinstance(f, float) for f in features)

        # Historical features should use 'NULL' stats (not cold-start)
        # is_cold_start should be False (index -1)
        assert features[-1] == 0.0  # Not cold-start

    def test_version_compatibility(self, mock_training_stats):
        """
        Test that version information is preserved through serialization.
        """
        stats_dict = mock_training_stats.to_dict()

        # Verify version is in serialized data
        assert 'version' in stats_dict
        assert stats_dict['version'] == '1.0.0'

        # Deserialize
        loaded_schema = HistoricalStatsSchema.from_dict(stats_dict)

        # Version should be preserved
        assert loaded_schema.version == '1.0.0'

        # Extractor should be able to access version
        extractor = HistoricalFeatureExtractor(
            config={},
            historical_stats=stats_dict
        )

        assert extractor.stats_schema.version == '1.0.0'
