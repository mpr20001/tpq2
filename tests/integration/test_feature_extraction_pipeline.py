"""
Integration tests for feature extraction pipeline.

Tests end-to-end feature extraction with realistic production scenarios.
"""

import pytest
from query_predictor.core.featurizer.feature_extractor import FeatureExtractor
from query_predictor.core.featurizer import constants


class TestFeatureExtractionPipeline:
    """Test complete feature extraction pipeline."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor with production-like config."""
        config = {
            'ast_timeout_ms': constants.DEFAULT_AST_TIMEOUT_MS,
            'enable_historical_features': False
        }
        return FeatureExtractor(config)

    def test_end_to_end_simple_query(self, extractor):
        """Test simple SELECT query end-to-end."""
        query_data = {
            'query': 'SELECT * FROM users WHERE created_date > "2024-01-01"',
            'user': 'test.user@example.com',
            'catalog': 'hive',
            'schema': 'production',
            'hour': 14
        }

        features = extractor.extract(query_data)

        # Verify feature count
        assert len(features) == 78

        # Verify all features are floats
        assert all(isinstance(f, float) for f in features)

        # Verify no NaN or inf (most important check for production)
        assert all(f == f for f in features)  # Check for NaN (NaN != NaN)
        assert all(abs(f) != float('inf') for f in features)  # Check for inf

        # Verify we have some non-zero features (sanity check)
        assert sum(abs(f) for f in features) > 0.0

    def test_end_to_end_complex_query(self, extractor):
        """Test complex query with JOINs, CTEs, and aggregations."""
        query_data = {
            'query': '''
                WITH user_stats AS (
                    SELECT
                        user_id,
                        COUNT(*) as order_count,
                        SUM(total_amount) as total_spent,
                        MAX(order_date) as last_order_date
                    FROM orders
                    WHERE order_date >= DATE_SUB(CURRENT_DATE, 30)
                    GROUP BY user_id
                    HAVING COUNT(*) > 5
                ),
                product_rankings AS (
                    SELECT
                        product_id,
                        product_name,
                        COUNT(DISTINCT order_id) as order_frequency,
                        ROW_NUMBER() OVER (PARTITION BY category ORDER BY revenue DESC) as rank
                    FROM order_items oi
                    JOIN products p ON oi.product_id = p.id
                    GROUP BY product_id, product_name, category
                )
                SELECT
                    u.user_id,
                    u.username,
                    us.order_count,
                    us.total_spent,
                    pr.product_name as favorite_product
                FROM users u
                INNER JOIN user_stats us ON u.id = us.user_id
                LEFT JOIN orders o ON u.id = o.user_id AND o.order_date = us.last_order_date
                LEFT JOIN order_items oi ON o.id = oi.order_id
                LEFT JOIN product_rankings pr ON oi.product_id = pr.product_id AND pr.rank = 1
                WHERE u.status = 'active'
                ORDER BY us.total_spent DESC
                LIMIT 100
            ''',
            'user': 'data.analyst@example.com',
            'catalog': 'hive',
            'schema': 'analytics',
            'hour': 10
        }

        features = extractor.extract(query_data)

        # Verify feature count
        assert len(features) == 78

        # Complex query should have non-zero features
        assert sum(features) > 10.0  # Expecting many active features

        # Verify specific features for complex query
        feature_names = extractor.feature_names

        # Should have CTEs (with_clause_count > 0)
        with_idx = [i for i, name in enumerate(feature_names) if 'with_clause_count' in name]
        if with_idx:
            assert features[with_idx[0]] > 0.0

        # Should have JOINs
        join_idx = [i for i, name in enumerate(feature_names) if 'join_count' in name]
        if join_idx:
            assert features[join_idx[0]] > 0.0

    def test_end_to_end_null_catalog_schema(self, extractor):
        """Test query with NULL catalog/schema."""
        query_data = {
            'query': 'SELECT catalog.schema.table.* FROM catalog.schema.table LIMIT 10',
            'user': 'etl.job@example.com',
            'catalog': None,  # NULL catalog
            'schema': None,   # NULL schema
            'hour': 3
        }

        features = extractor.extract(query_data)

        # Verify feature count
        assert len(features) == 78

        # NULL-aware features should be set (last 6 features)
        assert features[-6] == 1.0  # is_catalog_null
        assert features[-5] == 1.0  # is_schema_null

    def test_end_to_end_timeout_scenario(self, extractor):
        """Test query that might timeout during AST parsing."""
        # Create an extremely complex query that might timeout
        complex_conditions = " OR ".join([f"col{i} = {i}" for i in range(1000)])
        query_data = {
            'query': f'SELECT * FROM huge_table WHERE {complex_conditions}',
            'user': 'batch.processor@example.com',
            'catalog': 'hive',
            'schema': 'batch',
            'hour': 2
        }

        features = extractor.extract(query_data)

        # Should still return 78 features (graceful degradation)
        assert len(features) == 78

        # Should have fallback features (non-zero despite timeout)
        assert sum(features) > 0.0

    def test_end_to_end_validation_errors(self, extractor):
        """Test that validation errors are properly raised."""

        # Empty query
        with pytest.raises(ValueError, match="Query field is required"):
            extractor.extract({'query': '', 'user': 'test', 'hour': 12})

        # Whitespace-only query
        with pytest.raises(ValueError, match="whitespace-only"):
            extractor.extract({'query': '   \n  ', 'user': 'test', 'hour': 12})

        # Query too long (exceed 1MB = 1,000,000 characters)
        # Create a query with ~1.1M characters
        long_query = "SELECT " + ", ".join([f"column_name_{i}_with_long_name" for i in range(50000)])
        assert len(long_query) > constants.MAX_QUERY_LENGTH  # Ensure it's actually too long
        with pytest.raises(ValueError, match="exceeds maximum length"):
            extractor.extract({'query': long_query, 'user': 'test', 'hour': 12})

        # Non-dict input
        with pytest.raises(ValueError, match="must be a dictionary"):
            extractor.extract("not a dict")

    def test_end_to_end_error_isolation(self, extractor):
        """Test that extractor errors are isolated and don't cascade."""
        query_data = {
            'query': 'COMPLETELY INVALID SQL THAT BREAKS EVERYTHING;;;;',
            'user': 'error.test@example.com',
            'catalog': 'test',
            'schema': 'test',
            'hour': 15
        }

        features = extractor.extract(query_data)

        # Should still return 78 features (error isolation)
        assert len(features) == 78

        # Should return valid floats (not NaN or inf)
        assert all(isinstance(f, float) for f in features)
        assert all(not (f != f) for f in features)  # Check for NaN

    def test_end_to_end_feature_consistency(self, extractor):
        """Test that same query produces same features (deterministic)."""
        query_data = {
            'query': 'SELECT user_id, COUNT(*) FROM events GROUP BY user_id',
            'user': 'consistency.test@example.com',
            'catalog': 'analytics',
            'schema': 'events',
            'hour': 12
        }

        # Extract features multiple times
        features1 = extractor.extract(query_data)
        features2 = extractor.extract(query_data)
        features3 = extractor.extract(query_data)

        # All should be identical (deterministic)
        assert features1 == features2
        assert features2 == features3

    def test_end_to_end_production_queries(self, extractor):
        """Test with realistic production query patterns."""
        production_queries = [
            # ETL query
            {
                'query': 'INSERT INTO target_table SELECT * FROM source_table WHERE date = "2024-01-15"',
                'user': 'etl.user',
                'catalog': 'hive',
                'schema': 'etl',
                'hour': 2
            },
            # Analytics query
            {
                'query': '''
                    SELECT
                        DATE_TRUNC('day', timestamp) as date,
                        COUNT(DISTINCT user_id) as dau,
                        COUNT(*) as events
                    FROM events
                    WHERE timestamp >= DATE_SUB(CURRENT_DATE, 7)
                    GROUP BY 1
                    ORDER BY 1
                ''',
                'user': 'analyst.user',
                'catalog': 'analytics',
                'schema': 'events',
                'hour': 10
            },
            # Metadata query
            {
                'query': 'SHOW TABLES FROM schema',
                'user': 'metadata.user',
                'catalog': 'system',
                'schema': 'information_schema',
                'hour': 14
            },
            # DDL query
            {
                'query': 'CREATE TABLE new_table AS SELECT * FROM old_table WHERE partition_date > "2024-01-01"',
                'user': 'ddl.user',
                'catalog': 'hive',
                'schema': 'staging',
                'hour': 5
            }
        ]

        for query_data in production_queries:
            features = extractor.extract(query_data)

            # All should return 78 features
            assert len(features) == 78

            # All should have valid floats
            assert all(isinstance(f, float) for f in features)

            # Should have some non-zero features
            assert sum(features) > 0.0


class TestConcurrentExtraction:
    """Test concurrent feature extraction (thread safety)."""

    @pytest.fixture
    def extractor(self):
        """Create feature extractor."""
        config = {
            'ast_timeout_ms': constants.DEFAULT_AST_TIMEOUT_MS,
            'enable_historical_features': False
        }
        return FeatureExtractor(config)

    def test_concurrent_extraction(self, extractor):
        """Test concurrent extraction from multiple threads."""
        import concurrent.futures

        query_data_list = [
            {
                'query': f'SELECT * FROM table{i} WHERE id > {i}',
                'user': f'user{i}',
                'catalog': 'hive',
                'schema': 'test',
                'hour': i % 24
            }
            for i in range(10)
        ]

        # Extract features concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(extractor.extract, qd) for qd in query_data_list]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should return 78 features
        assert all(len(features) == 78 for features in results)

        # All should have valid floats
        assert all(
            all(isinstance(f, float) for f in features)
            for features in results
        )
