"""Unit tests for feature extractors."""

import pytest
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer.extractors.sql_extractor import SQLFeatureExtractor
from query_predictor.core.featurizer.extractors.table_join_extractor import TableJoinExtractor
from query_predictor.core.featurizer.extractors.where_extractor import WhereClauseExtractor
from query_predictor.core.featurizer.extractors.aggregation_extractor import AggregationExtractor
from query_predictor.core.featurizer.extractors.ast_extractor import ASTFeatureExtractor
from query_predictor.core.featurizer.extractors.context_extractor import ContextExtractor
from query_predictor.core.featurizer.extractors.query_type_extractor import QueryTypeExtractor
from query_predictor.core.featurizer.extractors.set_operation_extractor import SetOperationExtractor
from query_predictor.core.featurizer.extractors.null_aware_extractor import NullAwareExtractor


@pytest.fixture
def config():
    """Fixture for common config."""
    return {'ast_timeout_ms': 50}


class TestSQLFeatureExtractor:
    """Test SQLFeatureExtractor."""

    def test_simple_query(self, config):
        extractor = SQLFeatureExtractor(config)
        query_data = QueryData(query='SELECT * FROM table', user='test')
        features = extractor.extract(query_data)

        assert len(features) == extractor.feature_count
        assert all(isinstance(f, float) for f in features)

    def test_complex_query(self, config):
        extractor = SQLFeatureExtractor(config)
        query_data = QueryData(
            query='SELECT col1, col2, COUNT(*) FROM table WHERE id > 10 GROUP BY col1, col2',
            user='test'
        )
        features = extractor.extract(query_data)

        assert len(features) == extractor.feature_count
        # query_length should be > 0
        assert features[0] > 0

    def test_select_star(self, config):
        extractor = SQLFeatureExtractor(config)
        query_data = QueryData(query='SELECT * FROM table', user='test')
        features = extractor.extract(query_data)

        # has_select_star should be 1.0
        feature_names = extractor.get_feature_names()
        has_select_star_idx = feature_names.index('has_select_star')
        assert features[has_select_star_idx] == 1.0

    def test_empty_query(self, config):
        extractor = SQLFeatureExtractor(config)
        # Test that empty queries are rejected by QueryData validation
        with pytest.raises(ValueError, match="Query cannot be empty"):
            QueryData(query='', user='test')


class TestTableJoinExtractor:
    """Test TableJoinExtractor."""

    def test_single_table(self, config):
        extractor = TableJoinExtractor(config)
        query_data = QueryData(query='SELECT * FROM table1', user='test')
        features = extractor.extract(query_data)

        assert len(features) == extractor.feature_count
        # table_count should be at least 1
        assert features[0] >= 1.0

    def test_join_query(self, config):
        extractor = TableJoinExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table1 JOIN table2 ON table1.id = table2.id',
            user='test'
        )
        features = extractor.extract(query_data)

        # join_count should be > 0
        feature_names = extractor.get_feature_names()
        join_count_idx = feature_names.index('join_count')
        assert features[join_count_idx] > 0

    def test_left_join(self, config):
        extractor = TableJoinExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table1 LEFT JOIN table2 ON table1.id = table2.id',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        left_join_idx = feature_names.index('left_join_count')
        assert features[left_join_idx] > 0

    def test_with_clause(self, config):
        extractor = TableJoinExtractor(config)
        query_data = QueryData(
            query='WITH cte AS (SELECT * FROM table1) SELECT * FROM cte',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        with_clause_idx = feature_names.index('with_clause_count')
        assert features[with_clause_idx] > 0


class TestWhereClauseExtractor:
    """Test WhereClauseExtractor."""

    def test_simple_where(self, config):
        extractor = WhereClauseExtractor(config)
        query_data = QueryData(query='SELECT * FROM table WHERE id = 1', user='test')
        features = extractor.extract(query_data)

        assert len(features) == extractor.feature_count
        # where_condition_count should be > 0
        assert features[0] > 0

    def test_in_clause(self, config):
        extractor = WhereClauseExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table WHERE id IN (1, 2, 3, 4, 5)',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        in_clause_idx = feature_names.index('in_clause_count')
        max_in_size_idx = feature_names.index('max_in_list_size')

        assert features[in_clause_idx] > 0
        assert features[max_in_size_idx] >= 5.0

    def test_and_or(self, config):
        extractor = WhereClauseExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table WHERE id > 10 AND status = "active" OR type = "special"',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        and_idx = feature_names.index('where_and_count')
        or_idx = feature_names.index('where_or_count')

        assert features[and_idx] > 0
        assert features[or_idx] > 0

    def test_no_where(self, config):
        extractor = WhereClauseExtractor(config)
        query_data = QueryData(query='SELECT * FROM table', user='test')
        features = extractor.extract(query_data)

        # All features should be 0
        assert all(f == 0.0 for f in features)


class TestAggregationExtractor:
    """Test AggregationExtractor."""

    def test_group_by(self, config):
        extractor = AggregationExtractor(config)
        query_data = QueryData(
            query='SELECT col1, COUNT(*) FROM table GROUP BY col1',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        group_by_idx = feature_names.index('group_by_count')
        assert features[group_by_idx] >= 1.0

    def test_having(self, config):
        extractor = AggregationExtractor(config)
        query_data = QueryData(
            query='SELECT col1, COUNT(*) FROM table GROUP BY col1 HAVING COUNT(*) > 10',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        having_idx = feature_names.index('having_clause_present')
        assert features[having_idx] == 1.0

    def test_aggregation_functions(self, config):
        extractor = AggregationExtractor(config)
        query_data = QueryData(
            query='SELECT COUNT(*), SUM(amount), AVG(price), MIN(date), MAX(date) FROM table',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        count_idx = feature_names.index('count_function_count')
        sum_avg_idx = feature_names.index('sum_avg_function_count')
        min_max_idx = feature_names.index('min_max_function_count')

        assert features[count_idx] >= 1.0
        assert features[sum_avg_idx] >= 2.0  # SUM + AVG
        assert features[min_max_idx] >= 2.0  # MIN + MAX

    def test_window_functions(self, config):
        extractor = AggregationExtractor(config)
        query_data = QueryData(
            query='SELECT ROW_NUMBER() OVER (PARTITION BY col1 ORDER BY col2) FROM table',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        window_idx = feature_names.index('window_function_count')
        assert features[window_idx] >= 1.0


class TestASTFeatureExtractor:
    """Test ASTFeatureExtractor."""

    def test_simple_query(self, config):
        extractor = ASTFeatureExtractor(config)
        query_data = QueryData(query='SELECT * FROM table', user='test')
        features = extractor.extract(query_data)

        assert len(features) == extractor.feature_count
        # ast_parse_failed should be 0 for valid query
        assert features[-1] == 0.0

    def test_complex_query_metrics(self, config):
        extractor = ASTFeatureExtractor(config)
        query_data = QueryData(
            query='''
                WITH cte AS (SELECT * FROM table1)
                SELECT * FROM cte WHERE id IN (
                    SELECT id FROM table2 WHERE status = 'active'
                )
            ''',
            user='test'
        )
        features = extractor.extract(query_data)

        # ast_depth should be > 0
        assert features[0] > 0

    def test_case_when(self, config):
        extractor = ASTFeatureExtractor(config)
        query_data = QueryData(
            query='''
                SELECT CASE
                    WHEN status = 'active' THEN 1
                    WHEN status = 'pending' THEN 2
                    ELSE 0
                END FROM table
            ''',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        case_when_idx = feature_names.index('case_when_count')
        assert features[case_when_idx] > 0

    def test_invalid_query_fallback(self, config):
        extractor = ASTFeatureExtractor(config)
        query_data = QueryData(query='SELECT FROM', user='test')  # This query fails to parse
        features = extractor.extract(query_data)

        # Should return fallback features
        assert len(features) == extractor.feature_count
        # ast_parse_failed should be 1.0
        assert features[-1] == 1.0


class TestContextExtractor:
    """Test ContextExtractor."""

    def test_user_hash(self, config):
        extractor = ContextExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table',
            user='test_user',
            catalog='hive',
            schema='default',
            hour=12
        )
        features = extractor.extract(query_data)

        assert len(features) == extractor.feature_count
        # user_hash should be between 0 and 1
        assert 0.0 <= features[0] <= 1.0

    def test_hour_encoding(self, config):
        extractor = ContextExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table',
            user='test_user',
            hour=0  # Midnight
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        hour_sin_idx = feature_names.index('hour_sin')
        hour_cos_idx = feature_names.index('hour_cos')

        # At midnight: sin should be ~0, cos should be ~1
        assert abs(features[hour_sin_idx] - 0.0) < 0.1
        assert abs(features[hour_cos_idx] - 1.0) < 0.1

    def test_business_hours(self, config):
        extractor = ContextExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table',
            user='test_user',
            hour=14  # 2 PM on weekday
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        business_hours_idx = feature_names.index('is_business_hours')

        # Depends on day of week, but should be valid
        assert features[business_hours_idx] in [0.0, 1.0]


class TestQueryTypeExtractor:
    """Test QueryTypeExtractor."""

    def test_select_query(self, config):
        extractor = QueryTypeExtractor(config)
        query_data = QueryData(query='SELECT * FROM table', user='test')
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        select_idx = feature_names.index('is_select_query')
        assert features[select_idx] == 1.0

    def test_insert_query(self, config):
        extractor = QueryTypeExtractor(config)
        query_data = QueryData(query='INSERT INTO table VALUES (1, 2, 3)', user='test')
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        insert_idx = feature_names.index('is_insert_query')
        assert features[insert_idx] == 1.0

    def test_create_table_as(self, config):
        extractor = QueryTypeExtractor(config)
        query_data = QueryData(
            query='CREATE TABLE new_table AS SELECT * FROM old_table',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        ctas_idx = feature_names.index('is_create_table_as')
        assert features[ctas_idx] == 1.0


class TestSetOperationExtractor:
    """Test SetOperationExtractor."""

    def test_union(self, config):
        extractor = SetOperationExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table1 UNION SELECT * FROM table2',
            user='test'
        )
        features = extractor.extract(query_data)

        # Should detect UNION (counted in set_operation_total)
        feature_names = extractor.get_feature_names()
        total_idx = feature_names.index('set_operation_total')
        assert features[total_idx] > 0

    def test_except(self, config):
        extractor = SetOperationExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table1 EXCEPT SELECT * FROM table2',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        except_idx = feature_names.index('except_count')
        assert features[except_idx] > 0

    def test_intersect(self, config):
        extractor = SetOperationExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table1 INTERSECT SELECT * FROM table2',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        intersect_idx = feature_names.index('intersect_count')
        assert features[intersect_idx] > 0

    def test_multiple_statements(self, config):
        extractor = SetOperationExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table1; SELECT * FROM table2;',
            user='test'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        multi_stmt_idx = feature_names.index('has_multiple_statements')
        assert features[multi_stmt_idx] == 1.0


class TestNullAwareExtractor:
    """Test NullAwareExtractor."""

    def test_null_catalog(self, config):
        extractor = NullAwareExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table',
            user='test',
            catalog=None,
            schema='default'
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        catalog_null_idx = feature_names.index('is_catalog_null')
        assert features[catalog_null_idx] == 1.0

    def test_null_schema(self, config):
        extractor = NullAwareExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM table',
            user='test',
            catalog='hive',
            schema=None
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        schema_null_idx = feature_names.index('is_schema_null')
        assert features[schema_null_idx] == 1.0

    def test_qualified_tables(self, config):
        extractor = NullAwareExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM catalog.schema.table',
            user='test',
            catalog=None,
            schema=None
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        has_qualified_idx = feature_names.index('has_qualified_tables')
        assert features[has_qualified_idx] == 1.0

    def test_inferred_catalog_schema(self, config):
        extractor = NullAwareExtractor(config)
        query_data = QueryData(
            query='SELECT * FROM hive.default.users JOIN hive.sales.orders',
            user='test',
            catalog=None,
            schema=None
        )
        features = extractor.extract(query_data)

        feature_names = extractor.get_feature_names()
        catalog_count_idx = feature_names.index('inferred_catalog_count')
        schema_count_idx = feature_names.index('inferred_schema_count')

        # Should infer at least 1 catalog and schema
        assert features[catalog_count_idx] >= 1.0
        assert features[schema_count_idx] >= 1.0

    def test_qualification_level(self, config):
        extractor = NullAwareExtractor(config)

        # Fully qualified
        query_data = QueryData(
            query='SELECT * FROM catalog.schema.table',
            user='test'
        )
        features = extractor.extract(query_data)
        feature_names = extractor.get_feature_names()
        qual_level_idx = feature_names.index('table_qualification_level')
        assert features[qual_level_idx] == 2.0

        # Schema qualified
        query_data = QueryData(
            query='SELECT * FROM schema.table',
            user='test'
        )
        features = extractor.extract(query_data)
        assert features[qual_level_idx] == 1.0

        # Unqualified
        query_data = QueryData(
            query='SELECT * FROM table',
            user='test'
        )
        features = extractor.extract(query_data)
        assert features[qual_level_idx] == 0.0
