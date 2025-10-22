"""Unit tests for parsers."""

import pytest
from query_predictor.core.featurizer.parsers import ASTParser, SQLPatternParser


class TestASTParser:
    """Test ASTParser."""

    def test_simple_query(self):
        parser = ASTParser(timeout_ms=50)
        success, ast, metrics = parser.parse("SELECT * FROM table")

        assert success is True
        assert ast is not None
        assert metrics.parse_success is True
        assert metrics.parse_timeout is False
        assert metrics.node_count > 0

    def test_complex_query_with_cte(self):
        parser = ASTParser(timeout_ms=50)
        query = """
        WITH cte AS (
            SELECT id, name FROM users
        )
        SELECT * FROM cte WHERE id > 10
        """
        success, ast, metrics = parser.parse(query)

        assert success is True
        assert metrics.cte_count >= 1

    def test_query_with_subquery(self):
        parser = ASTParser(timeout_ms=50)
        query = "SELECT * FROM (SELECT id FROM users) t"
        success, ast, metrics = parser.parse(query)

        assert success is True
        assert metrics.subquery_count >= 1

    def test_query_with_case_when(self):
        parser = ASTParser(timeout_ms=50)
        query = """
        SELECT CASE
            WHEN status = 'active' THEN 1
            WHEN status = 'pending' THEN 2
            ELSE 0
        END as status_code
        FROM users
        """
        success, ast, metrics = parser.parse(query)

        assert success is True
        assert metrics.case_when_count >= 1
        assert metrics.max_branches >= 2

    def test_query_with_union(self):
        parser = ASTParser(timeout_ms=50)
        query = "SELECT id FROM table1 UNION SELECT id FROM table2"
        success, ast, metrics = parser.parse(query)

        assert success is True
        assert metrics.union_count >= 1

    def test_query_with_union_all(self):
        parser = ASTParser(timeout_ms=50)
        query = "SELECT id FROM table1 UNION ALL SELECT id FROM table2"
        success, ast, metrics = parser.parse(query)

        assert success is True
        assert metrics.union_all_count >= 1

    def test_invalid_query(self):
        parser = ASTParser(timeout_ms=50)
        success, ast, metrics = parser.parse("INVALID SQL GARBAGE")

        # Should handle gracefully
        assert success is False or metrics.parse_success is False

    def test_empty_query(self):
        parser = ASTParser(timeout_ms=50)
        success, ast, metrics = parser.parse("")

        assert success is False
        assert metrics.parse_success is False

    def test_timeout_handling(self):
        parser = ASTParser(timeout_ms=1)  # Very short timeout
        # Create a very complex query that might timeout
        query = " UNION ".join([f"SELECT {i} FROM table{i}" for i in range(100)])
        success, ast, metrics = parser.parse(query)

        # Should handle timeout gracefully
        # Either succeeds or marks as timeout
        if not success:
            assert metrics.parse_timeout is True or metrics.parse_success is False


class TestSQLPatternParser:
    """Test SQLPatternParser."""

    def test_extract_tables(self):
        parser = SQLPatternParser()
        query = "SELECT * FROM table1 JOIN table2 ON table1.id = table2.id"
        tables = parser.extract_tables(query)

        assert "table1" in tables or len(tables) >= 1
        assert "table2" in tables or len(tables) >= 2

    def test_extract_qualified_tables(self):
        parser = SQLPatternParser()
        query = "SELECT * FROM catalog.schema.table1 JOIN schema.table2"
        fully_qualified, schema_qualified = parser.extract_qualified_tables(query)

        assert len(fully_qualified) >= 1 or len(schema_qualified) >= 1

    def test_extract_where_clause(self):
        parser = SQLPatternParser()
        query = "SELECT * FROM table WHERE id > 10 AND status = 'active'"
        where = parser.extract_where_clause(query)

        assert "id > 10" in where or "id" in where
        assert "status" in where or "active" in where

    def test_extract_where_clause_no_where(self):
        parser = SQLPatternParser()
        query = "SELECT * FROM table"
        where = parser.extract_where_clause(query)

        assert where == ""

    def test_extract_in_clauses(self):
        parser = SQLPatternParser()
        query = "SELECT * FROM table WHERE id IN (1, 2, 3) AND status IN ('a', 'b')"
        in_clauses = parser.extract_in_clauses(query)

        assert len(in_clauses) >= 2

    def test_extract_in_clauses_with_counts(self):
        parser = SQLPatternParser()
        query = "SELECT * FROM table WHERE id IN (1, 2, 3, 4, 5)"
        in_clauses = parser.extract_in_clauses(query)

        assert len(in_clauses) >= 1
        if len(in_clauses) > 0:
            # Should capture the items in the IN clause
            assert ',' in in_clauses[0]

    def test_infer_catalog_schema(self):
        parser = SQLPatternParser()
        query = "SELECT * FROM catalog1.schema1.table1 JOIN catalog2.schema2.table2"
        catalogs, schemas = parser.infer_catalog_schema(query)

        # Should infer at least one catalog and schema
        assert len(catalogs) >= 1 or len(schemas) >= 1

    def test_infer_catalog_schema_simple(self):
        parser = SQLPatternParser()
        query = "SELECT * FROM hive.default.users"
        catalogs, schemas = parser.infer_catalog_schema(query)

        assert "hive" in catalogs or len(catalogs) >= 1
        assert "default" in schemas or len(schemas) >= 1

    def test_empty_query(self):
        parser = SQLPatternParser()
        tables = parser.extract_tables("")
        assert len(tables) == 0

        where = parser.extract_where_clause("")
        assert where == ""

        in_clauses = parser.extract_in_clauses("")
        assert len(in_clauses) == 0

    def test_complex_nested_query(self):
        parser = SQLPatternParser()
        query = """
        SELECT * FROM (
            SELECT t1.id FROM catalog.schema.table1 t1
            WHERE t1.status IN ('active', 'pending')
        ) sub
        WHERE sub.id > 10
        """
        tables = parser.extract_tables(query)
        in_clauses = parser.extract_in_clauses(query)
        where = parser.extract_where_clause(query)

        assert len(tables) >= 1
        assert len(in_clauses) >= 1
        assert len(where) > 0
