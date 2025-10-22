"""Unit tests for shared validators."""

import pytest
from query_predictor.common import validators
from query_predictor.core.featurizer import constants


class TestValidateQueryNotEmpty:
    """Test validate_query_not_empty function."""

    def test_valid_query(self):
        """Test valid query passes."""
        validators.validate_query_not_empty("SELECT * FROM table")
        # Should not raise

    def test_empty_string_raises(self):
        """Test empty string raises error."""
        with pytest.raises(ValueError, match="Query field is required"):
            validators.validate_query_not_empty("")

    def test_whitespace_only_raises(self):
        """Test whitespace-only string raises error."""
        with pytest.raises(ValueError, match="cannot be empty or whitespace-only"):
            validators.validate_query_not_empty("   ")

    def test_none_raises(self):
        """Test None raises error."""
        with pytest.raises(ValueError, match="Query field is required"):
            validators.validate_query_not_empty(None)

    def test_non_string_raises(self):
        """Test non-string raises error."""
        with pytest.raises(ValueError, match="Query field is required"):
            validators.validate_query_not_empty(123)


class TestValidateQueryLength:
    """Test validate_query_length function."""

    def test_valid_length(self):
        """Test valid length passes."""
        validators.validate_query_length("SELECT 1", constants.MAX_QUERY_LENGTH)
        # Should not raise

    def test_exceeds_max_length(self):
        """Test query exceeding max length raises error."""
        long_query = "SELECT " + "x" * constants.MAX_QUERY_LENGTH
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validators.validate_query_length(long_query, constants.MAX_QUERY_LENGTH)

    def test_exactly_max_length(self):
        """Test query at exactly max length passes."""
        query = "x" * constants.MAX_QUERY_LENGTH
        validators.validate_query_length(query, constants.MAX_QUERY_LENGTH)
        # Should not raise

    def test_custom_max_length(self):
        """Test with custom max length."""
        short_query = "SELECT 1"
        validators.validate_query_length(short_query, 10)
        # Should not raise

        long_query = "SELECT " + "x" * 20
        with pytest.raises(ValueError, match="exceeds maximum length"):
            validators.validate_query_length(long_query, 10)


class TestValidateQueryDataDict:
    """Test validate_query_data_dict function."""

    def test_valid_dict(self):
        """Test valid dictionary passes."""
        validators.validate_query_data_dict({"query": "SELECT 1"})
        # Should not raise

    def test_string_raises(self):
        """Test string raises error."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            validators.validate_query_data_dict("not a dict")

    def test_none_raises(self):
        """Test None raises error."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            validators.validate_query_data_dict(None)

    def test_list_raises(self):
        """Test list raises error."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            validators.validate_query_data_dict(["list"])

    def test_int_raises(self):
        """Test integer raises error."""
        with pytest.raises(ValueError, match="must be a dictionary"):
            validators.validate_query_data_dict(123)

    def test_empty_dict_passes(self):
        """Test empty dictionary passes."""
        validators.validate_query_data_dict({})
        # Should not raise
