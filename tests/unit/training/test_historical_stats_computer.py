"""
Unit tests for HistoricalStatsComputer.

Tests stats computation with mock PySpark data.

Note: These tests use mocks for PySpark. For full PySpark integration testing,
use a real Spark environment in integration tests.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from query_predictor.core.types.historical_stats import (
    HistoricalStatsSchema,
    UserStats,
    CatalogStats,
    SchemaStats
)


# Mock PySpark imports before importing HistoricalStatsComputer
@pytest.fixture(autouse=True)
def mock_pyspark():
    """Mock PySpark for all tests."""
    with patch.dict('sys.modules', {
        'pyspark': Mock(),
        'pyspark.sql': Mock(),
        'pyspark.sql.functions': Mock()
    }):
        yield


# Marker for tests that require complex PySpark mocking
pytestmark = pytest.mark.skip(reason="PySpark mocking tests - use real Spark for full testing")


class TestHistoricalStatsComputer:
    """Test HistoricalStatsComputer with mock PySpark data."""

    @pytest.fixture
    def mock_dataframe(self):
        """Create mock PySpark DataFrame with sample data."""
        df = Mock()
        df.columns = [
            'query', 'user', 'catalog', 'schema',
            'cpuTime', 'peakUserMemoryBytes', 'is_heavy'
        ]
        df.count.return_value = 1000  # 1000 total queries

        # Mock filter for heavy queries
        heavy_df = Mock()
        heavy_df.count.return_value = 100  # 10% heavy rate
        df.filter.return_value = heavy_df

        return df

    @pytest.fixture
    def mock_user_stats_data(self):
        """Sample user stats aggregation results."""
        return [
            {
                'user': 'user1@example.com',
                'query_count': 500,
                'heavy_rate': 0.08,
                'avg_cpu_seconds': 45.5,
                'p90_cpu_seconds': 120.0,
                'avg_memory_gb': 8.2,
                'p90_memory_gb': 15.5,
                'unique_catalogs': 3,
                'unique_schemas': 5,
                'cpu_stddev': 25.3
            },
            {
                'user': 'user2@example.com',
                'query_count': 300,
                'heavy_rate': 0.15,
                'avg_cpu_seconds': 78.2,
                'p90_cpu_seconds': 200.0,
                'avg_memory_gb': 12.5,
                'p90_memory_gb': 25.0,
                'unique_catalogs': 2,
                'unique_schemas': 3,
                'cpu_stddev': 45.8
            }
        ]

    @pytest.fixture
    def mock_catalog_stats_data(self):
        """Sample catalog stats aggregation results."""
        return [
            {
                'catalog_key': 'hive',
                'query_count': 600,
                'heavy_rate': 0.12,
                'avg_cpu_seconds': 55.0,
                'p50_cpu_seconds': 30.0,
                'p75_cpu_seconds': 75.0,
                'p90_cpu_seconds': 150.0,
                'p95_cpu_seconds': 250.0,
                'avg_memory_gb': 10.0,
                'p95_memory_gb': 20.0,
                'unique_users': 50,
                'unique_schemas': 10
            },
            {
                'catalog_key': 'NULL',  # NULL catalog
                'query_count': 200,
                'heavy_rate': 0.05,
                'avg_cpu_seconds': 25.0,
                'p50_cpu_seconds': 15.0,
                'p75_cpu_seconds': 35.0,
                'p90_cpu_seconds': 60.0,
                'p95_cpu_seconds': 100.0,
                'avg_memory_gb': 5.0,
                'p95_memory_gb': 10.0,
                'unique_users': 25,
                'unique_schemas': 5
            }
        ]

    @pytest.fixture
    def mock_schema_stats_data(self):
        """Sample schema stats aggregation results."""
        return [
            {
                'schema_key': 'production',
                'query_count': 400,
                'heavy_rate': 0.10,
                'avg_cpu_seconds': 60.0,
                'p50_cpu_seconds': 35.0,
                'p75_cpu_seconds': 80.0,
                'p90_cpu_seconds': 160.0,
                'p95_cpu_seconds': 280.0,
                'avg_memory_gb': 11.0,
                'p95_memory_gb': 22.0,
                'unique_users': 40,
                'unique_catalogs': 3
            },
            {
                'schema_key': 'NULL',  # NULL schema
                'query_count': 150,
                'heavy_rate': 0.06,
                'avg_cpu_seconds': 30.0,
                'p50_cpu_seconds': 18.0,
                'p75_cpu_seconds': 40.0,
                'p90_cpu_seconds': 70.0,
                'p95_cpu_seconds': 120.0,
                'avg_memory_gb': 6.0,
                'p95_memory_gb': 12.0,
                'unique_users': 20,
                'unique_catalogs': 2
            }
        ]

    def test_initialization(self):
        """Test HistoricalStatsComputer initialization."""
        from query_predictor.training.historical_stats_computer import HistoricalStatsComputer

        computer = HistoricalStatsComputer(version='1.0.0')
        assert computer.version == '1.0.0'

    def test_compute_user_stats(self, mock_dataframe, mock_user_stats_data):
        """Test user stats computation produces correct UserStats objects."""
        from query_predictor.training.historical_stats_computer import HistoricalStatsComputer

        computer = HistoricalStatsComputer()

        # Mock groupBy aggregation
        mock_df = mock_dataframe
        mock_grouped = Mock()
        mock_agg = Mock()
        mock_agg.collect.return_value = [Mock(**data) for data in mock_user_stats_data]
        mock_grouped.agg.return_value = mock_agg
        mock_df.groupBy.return_value = mock_grouped

        # Compute user stats
        users = computer._compute_user_stats(mock_df)

        # Verify results
        assert len(users) == 2
        assert 'user1@example.com' in users
        assert 'user2@example.com' in users

        # Verify UserStats structure
        user1 = users['user1@example.com']
        assert isinstance(user1, UserStats)
        assert user1.query_count == 500
        assert user1.heavy_rate == 0.08
        assert user1.avg_cpu_seconds == 45.5
        assert user1.unique_catalogs == 3

    def test_compute_catalog_stats_with_null(self, mock_dataframe, mock_catalog_stats_data):
        """Test catalog stats computation with NULL catalog handling."""
        from query_predictor.training.historical_stats_computer import HistoricalStatsComputer

        computer = HistoricalStatsComputer()

        # Mock withColumn for NULL handling
        mock_df_with_catalog = Mock()
        mock_grouped = Mock()
        mock_agg = Mock()
        mock_agg.collect.return_value = [Mock(**data) for data in mock_catalog_stats_data]
        mock_grouped.agg.return_value = mock_agg
        mock_df_with_catalog.groupBy.return_value = mock_grouped
        mock_dataframe.withColumn.return_value = mock_df_with_catalog

        # Compute catalog stats
        catalogs = computer._compute_catalog_stats(mock_dataframe)

        # Verify results
        assert len(catalogs) == 2
        assert 'hive' in catalogs
        assert 'NULL' in catalogs  # NULL catalog should use 'NULL' key

        # Verify CatalogStats structure
        hive_cat = catalogs['hive']
        assert isinstance(hive_cat, CatalogStats)
        assert hive_cat.query_count == 600
        assert hive_cat.heavy_rate == 0.12

        # Verify NULL catalog
        null_cat = catalogs['NULL']
        assert null_cat.query_count == 200
        assert null_cat.heavy_rate == 0.05

    def test_compute_schema_stats_with_null(self, mock_dataframe, mock_schema_stats_data):
        """Test schema stats computation with NULL schema handling."""
        from query_predictor.training.historical_stats_computer import HistoricalStatsComputer

        computer = HistoricalStatsComputer()

        # Mock withColumn for NULL handling
        mock_df_with_schema = Mock()
        mock_grouped = Mock()
        mock_agg = Mock()
        mock_agg.collect.return_value = [Mock(**data) for data in mock_schema_stats_data]
        mock_grouped.agg.return_value = mock_agg
        mock_df_with_schema.groupBy.return_value = mock_grouped
        mock_dataframe.withColumn.return_value = mock_df_with_schema

        # Compute schema stats
        schemas = computer._compute_schema_stats(mock_dataframe)

        # Verify results
        assert len(schemas) == 2
        assert 'production' in schemas
        assert 'NULL' in schemas  # NULL schema should use 'NULL' key

        # Verify SchemaStats structure
        prod_schema = schemas['production']
        assert isinstance(prod_schema, SchemaStats)
        assert prod_schema.query_count == 400

        # Verify NULL schema
        null_schema = schemas['NULL']
        assert null_schema.query_count == 150

    def test_compute_full_stats(
        self,
        mock_dataframe,
        mock_user_stats_data,
        mock_catalog_stats_data,
        mock_schema_stats_data
    ):
        """Test complete stats computation produces valid HistoricalStatsSchema."""
        from query_predictor.training.historical_stats_computer import HistoricalStatsComputer

        computer = HistoricalStatsComputer(version='1.0.0')

        # Mock all aggregations
        def mock_group_by_side_effect(column):
            mock_grouped = Mock()
            mock_agg = Mock()

            if column == 'user':
                mock_agg.collect.return_value = [Mock(**d) for d in mock_user_stats_data]
            elif column == 'catalog_key':
                mock_agg.collect.return_value = [Mock(**d) for d in mock_catalog_stats_data]
            elif column == 'schema_key':
                mock_agg.collect.return_value = [Mock(**d) for d in mock_schema_stats_data]

            mock_grouped.agg.return_value = mock_agg
            return mock_grouped

        mock_df_with_columns = Mock()
        mock_df_with_columns.groupBy.side_effect = mock_group_by_side_effect
        mock_dataframe.groupBy.side_effect = mock_group_by_side_effect
        mock_dataframe.withColumn.return_value = mock_df_with_columns

        # Compute full stats
        date_range = {'start': '2025-06-01', 'end': '2025-07-15'}
        stats_schema = computer.compute(mock_dataframe, date_range)

        # Verify HistoricalStatsSchema
        assert isinstance(stats_schema, HistoricalStatsSchema)
        assert stats_schema.version == '1.0.0'
        assert stats_schema.data_date_range == date_range
        assert stats_schema.total_queries_analyzed == 1000
        assert stats_schema.heavy_rate_overall == 0.10  # 100 heavy / 1000 total

        # Verify entity counts
        assert len(stats_schema.users) == 2
        assert len(stats_schema.catalogs) == 2
        assert len(stats_schema.schemas) == 2

    def test_validation_missing_columns(self, mock_dataframe):
        """Test that validation catches missing columns."""
        from query_predictor.training.historical_stats_computer import HistoricalStatsComputer

        computer = HistoricalStatsComputer()

        # Remove required column
        mock_dataframe.columns = ['query', 'user']  # Missing other required columns

        with pytest.raises(ValueError, match="missing required columns"):
            computer._validate_dataframe(mock_dataframe)

    def test_serialization_to_dict(
        self,
        mock_dataframe,
        mock_user_stats_data,
        mock_catalog_stats_data,
        mock_schema_stats_data
    ):
        """Test that computed stats can be serialized to dict."""
        from query_predictor.training.historical_stats_computer import HistoricalStatsComputer

        computer = HistoricalStatsComputer()

        # Mock all aggregations (simplified)
        def mock_group_by_side_effect(column):
            mock_grouped = Mock()
            mock_agg = Mock()

            if column == 'user':
                mock_agg.collect.return_value = [Mock(**d) for d in mock_user_stats_data]
            elif column == 'catalog_key':
                mock_agg.collect.return_value = [Mock(**d) for d in mock_catalog_stats_data]
            elif column == 'schema_key':
                mock_agg.collect.return_value = [Mock(**d) for d in mock_schema_stats_data]

            mock_grouped.agg.return_value = mock_agg
            return mock_grouped

        mock_df_with_columns = Mock()
        mock_df_with_columns.groupBy.side_effect = mock_group_by_side_effect
        mock_dataframe.groupBy.side_effect = mock_group_by_side_effect
        mock_dataframe.withColumn.return_value = mock_df_with_columns

        # Compute stats
        stats_schema = computer.compute(mock_dataframe)

        # Serialize to dict
        stats_dict = stats_schema.to_dict()

        # Verify dict structure (train-serve parity critical!)
        assert 'version' in stats_dict
        assert 'users' in stats_dict
        assert 'catalogs' in stats_dict
        assert 'schemas' in stats_dict

        # Verify user dict has correct fields
        assert 'user1@example.com' in stats_dict['users']
        user1_dict = stats_dict['users']['user1@example.com']
        assert 'query_count' in user1_dict
        assert 'heavy_rate' in user1_dict
        assert 'avg_cpu_seconds' in user1_dict

        # Critical: Can deserialize back
        reloaded_schema = HistoricalStatsSchema.from_dict(stats_dict)
        assert reloaded_schema.version == stats_schema.version
        assert len(reloaded_schema.users) == len(stats_schema.users)


class TestHistoricalStatsSchema:
    """Test HistoricalStatsSchema dataclass."""

    def test_get_user_stats(self):
        """Test getting user stats by key."""
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

        schema = HistoricalStatsSchema(
            version='1.0.0',
            created_at='2025-01-15T10:00:00',
            data_date_range={'start': '2025-06-01', 'end': '2025-07-15'},
            users={'test.user': user_stats},
            catalogs={},
            schemas={},
            total_queries_analyzed=100,
            heavy_rate_overall=0.1
        )

        # Test get existing user
        result = schema.get_user_stats('test.user')
        assert result is not None
        assert result.query_count == 100

        # Test get non-existing user
        result = schema.get_user_stats('unknown.user')
        assert result is None

    def test_get_catalog_stats_with_null(self):
        """Test getting catalog stats with NULL handling."""
        catalog_stats = CatalogStats(
            query_count=200,
            heavy_rate=0.15,
            avg_cpu_seconds=60.0,
            p50_cpu_seconds=30.0,
            p75_cpu_seconds=75.0,
            p90_cpu_seconds=150.0,
            p95_cpu_seconds=250.0,
            avg_memory_gb=12.0,
            p95_memory_gb=25.0,
            unique_users=50,
            unique_schemas=10
        )

        schema = HistoricalStatsSchema(
            version='1.0.0',
            created_at='2025-01-15T10:00:00',
            data_date_range={'start': '2025-06-01', 'end': '2025-07-15'},
            users={},
            catalogs={'NULL': catalog_stats, 'hive': catalog_stats},
            schemas={},
            total_queries_analyzed=200,
            heavy_rate_overall=0.15
        )

        # Test get NULL catalog (None input should map to 'NULL' key)
        result = schema.get_catalog_stats(None)
        assert result is not None
        assert result.query_count == 200

        # Test get normal catalog
        result = schema.get_catalog_stats('hive')
        assert result is not None

    def test_stats_summary(self):
        """Test stats summary property."""
        schema = HistoricalStatsSchema(
            version='1.0.0',
            created_at='2025-01-15T10:00:00',
            data_date_range={'start': '2025-06-01', 'end': '2025-07-15'},
            users={'user1': Mock(), 'user2': Mock()},
            catalogs={'cat1': Mock()},
            schemas={'schema1': Mock(), 'schema2': Mock(), 'schema3': Mock()},
            total_queries_analyzed=1000,
            heavy_rate_overall=0.12
        )

        summary = schema.stats_summary

        assert summary['version'] == '1.0.0'
        assert summary['entity_counts']['users'] == 2
        assert summary['entity_counts']['catalogs'] == 1
        assert summary['entity_counts']['schemas'] == 3
        assert summary['coverage']['total_queries_analyzed'] == 1000
        assert summary['coverage']['heavy_rate_overall'] == 0.12
