"""
SQL pattern parser using regex for fast extraction.

Provides regex-based extraction for common SQL patterns:
- Table references (FROM/JOIN clauses)
- WHERE clause extraction
- IN clauses
- Qualified table names (catalog.schema.table)
"""

import re
import logging
from typing import List, Tuple, Optional, Set

logger = logging.getLogger(__name__)


class SQLPatternParser:
    """
    Fast regex-based SQL pattern parser.

    Uses regex patterns for quick extraction without full parsing overhead.
    Suitable for feature extraction where exact parse is not required.
    """

    # Compiled regex patterns for performance
    FROM_PATTERN = re.compile(r'FROM\s+([a-zA-Z0-9_\.]+)', re.IGNORECASE)
    JOIN_PATTERN = re.compile(r'JOIN\s+([a-zA-Z0-9_\.]+)', re.IGNORECASE)
    FULLY_QUALIFIED_PATTERN = re.compile(r'([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)')
    SCHEMA_QUALIFIED_PATTERN = re.compile(r'(?:FROM|JOIN)\s+([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)', re.IGNORECASE)
    IN_CLAUSE_PATTERN = re.compile(r'IN\s*\(([^)]+)\)', re.IGNORECASE)
    WHERE_PATTERN = re.compile(r'WHERE\s+(.*?)(?:GROUP BY|ORDER BY|LIMIT|HAVING|UNION|EXCEPT|INTERSECT|$)', re.IGNORECASE | re.DOTALL)
    USE_CATALOG_PATTERN = re.compile(r'USE\s+([a-zA-Z0-9_]+)\.([a-zA-Z0-9_]+)', re.IGNORECASE)
    USE_SCHEMA_PATTERN = re.compile(r'USE\s+([a-zA-Z0-9_]+)(?:\s|$|;)', re.IGNORECASE)

    def __init__(self):
        """Initialize SQL pattern parser."""
        pass

    def extract_tables(self, query: str) -> List[str]:
        """
        Extract table names from FROM and JOIN clauses.

        Args:
            query: SQL query string

        Returns:
            List of table names (may include qualified names like schema.table)
        """
        if not query or not isinstance(query, str):
            return []

        try:
            tables = set()

            # Extract from FROM clauses
            tables.update(self.FROM_PATTERN.findall(query))

            # Extract from JOIN clauses
            tables.update(self.JOIN_PATTERN.findall(query))

            return list(tables)

        except Exception as e:
            logger.debug(f"Error extracting tables: {e}")
            return []

    def extract_qualified_tables(self, query: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str]]]:
        """
        Extract fully-qualified and schema-qualified table references.

        Args:
            query: SQL query string

        Returns:
            Tuple of (fully_qualified_tables, schema_qualified_tables)
            - fully_qualified: [(catalog, schema, table), ...]
            - schema_qualified: [(schema, table), ...]
        """
        if not query or not isinstance(query, str):
            return [], []

        try:
            # Extract fully qualified (catalog.schema.table)
            fully_qualified = self.FULLY_QUALIFIED_PATTERN.findall(query)

            # Extract schema qualified (schema.table) - exclude fully qualified
            # Remove fully qualified patterns from query first
            query_without_fully = query
            for match in fully_qualified:
                pattern = f"{match[0]}\\.{match[1]}\\.{match[2]}"
                query_without_fully = re.sub(pattern, '', query_without_fully)

            schema_qualified = self.SCHEMA_QUALIFIED_PATTERN.findall(query_without_fully)

            return fully_qualified, schema_qualified

        except Exception as e:
            logger.debug(f"Error extracting qualified tables: {e}")
            return [], []

    def extract_where_clause(self, query: str) -> str:
        """
        Extract WHERE clause content.

        Args:
            query: SQL query string

        Returns:
            WHERE clause content (empty string if not found)
        """
        if not query or not isinstance(query, str):
            return ""

        try:
            match = self.WHERE_PATTERN.search(query)
            return match.group(1).strip() if match else ""

        except Exception as e:
            logger.debug(f"Error extracting WHERE clause: {e}")
            return ""

    def extract_in_clauses(self, where_clause: str) -> List[str]:
        """
        Extract IN clause contents from WHERE clause.

        Args:
            where_clause: WHERE clause string

        Returns:
            List of IN clause contents
        """
        if not where_clause or not isinstance(where_clause, str):
            return []

        try:
            return self.IN_CLAUSE_PATTERN.findall(where_clause)

        except Exception as e:
            logger.debug(f"Error extracting IN clauses: {e}")
            return []

    def extract_use_statements(self, query: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract catalog and schema from USE statements.

        Args:
            query: SQL query string

        Returns:
            Tuple of (catalog, schema) - either may be None
        """
        if not query or not isinstance(query, str):
            return None, None

        try:
            catalog = None
            schema = None

            # Check for USE catalog.schema
            use_catalog_match = self.USE_CATALOG_PATTERN.search(query)
            if use_catalog_match:
                catalog = use_catalog_match.group(1)
                schema = use_catalog_match.group(2)
            else:
                # Check for USE schema
                use_schema_match = self.USE_SCHEMA_PATTERN.search(query)
                if use_schema_match:
                    schema = use_schema_match.group(1)

            return catalog, schema

        except Exception as e:
            logger.debug(f"Error extracting USE statements: {e}")
            return None, None

    def infer_catalog_schema(self, query: str) -> Tuple[Set[str], Set[str]]:
        """
        Infer catalogs and schemas mentioned in query.

        Args:
            query: SQL query string

        Returns:
            Tuple of (inferred_catalogs, inferred_schemas)
        """
        if not query or not isinstance(query, str):
            return set(), set()

        try:
            inferred_catalogs = set()
            inferred_schemas = set()

            # Get qualified tables
            fully_qualified, schema_qualified = self.extract_qualified_tables(query)

            # Extract catalogs and schemas from fully qualified
            for catalog, schema, table in fully_qualified:
                inferred_catalogs.add(catalog)
                inferred_schemas.add(schema)

            # Extract schemas from schema qualified
            for schema, table in schema_qualified:
                inferred_schemas.add(schema)

            # Check USE statements
            use_catalog, use_schema = self.extract_use_statements(query)
            if use_catalog:
                inferred_catalogs.add(use_catalog)
            if use_schema:
                inferred_schemas.add(use_schema)

            return inferred_catalogs, inferred_schemas

        except Exception as e:
            logger.debug(f"Error inferring catalog/schema: {e}")
            return set(), set()

    def has_keyword(self, query: str, keyword: str, case_sensitive: bool = False) -> bool:
        """
        Check if query contains keyword.

        Args:
            query: SQL query string
            keyword: Keyword to search for
            case_sensitive: Whether search is case-sensitive

        Returns:
            True if keyword found
        """
        if not query or not isinstance(query, str) or not keyword:
            return False

        try:
            if case_sensitive:
                return keyword in query
            else:
                return keyword.upper() in query.upper()

        except Exception as e:
            logger.debug(f"Error checking keyword: {e}")
            return False

    def count_pattern(self, query: str, pattern: str) -> int:
        """
        Count occurrences of pattern in query.

        Args:
            query: SQL query string
            pattern: Pattern to count

        Returns:
            Number of occurrences
        """
        if not query or not isinstance(query, str) or not pattern:
            return 0

        try:
            return query.upper().count(pattern.upper())

        except Exception as e:
            logger.debug(f"Error counting pattern: {e}")
            return 0
