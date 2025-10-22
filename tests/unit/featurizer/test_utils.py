"""Unit tests for featurizer utility functions."""

import pytest
import math
from query_predictor.core.featurizer import utils


class TestSafeDivide:
    """Test safe_divide function."""

    def test_normal_division(self):
        assert utils.safe_divide(10, 2) == 5.0
        assert utils.safe_divide(7, 2) == 3.5

    def test_divide_by_zero(self):
        assert utils.safe_divide(10, 0) == 0.0
        assert utils.safe_divide(0, 0) == 0.0

    def test_negative_numbers(self):
        assert utils.safe_divide(-10, 2) == -5.0
        assert utils.safe_divide(10, -2) == -5.0


class TestNormalizeValue:
    """Test normalize_value function."""

    def test_normal_normalization(self):
        assert utils.normalize_value(5, 10) == 0.5
        assert utils.normalize_value(10, 10) == 1.0
        assert utils.normalize_value(0, 10) == 0.0

    def test_max_at_one(self):
        # Values >= max should return 1.0
        assert utils.normalize_value(15, 10) == 1.0
        assert utils.normalize_value(100, 10) == 1.0

    def test_zero_max(self):
        assert utils.normalize_value(5, 0) == 0.0


class TestSafeUpper:
    """Test safe_upper function."""

    def test_normal_uppercase(self):
        assert utils.safe_upper("hello") == "HELLO"
        assert utils.safe_upper("MiXeD") == "MIXED"

    def test_empty_string(self):
        assert utils.safe_upper("") == ""

    def test_none_value(self):
        assert utils.safe_upper(None) == ""

    def test_non_string(self):
        assert utils.safe_upper(123) == ""


class TestSafeCount:
    """Test safe_count function."""

    def test_normal_count(self):
        assert utils.safe_count("hello world", "o") == 2
        assert utils.safe_count("aaa", "a") == 3

    def test_substring_count(self):
        assert utils.safe_count("SELECT * FROM table", "SELECT") == 1
        assert utils.safe_count("JOIN table1 JOIN table2", "JOIN") == 2

    def test_empty_string(self):
        assert utils.safe_count("", "a") == 0

    def test_not_found(self):
        assert utils.safe_count("hello", "z") == 0

    def test_none_value(self):
        assert utils.safe_count(None, "a") == 0


class TestHashString:
    """Test hash_string function."""

    def test_normal_hash(self):
        hash1 = utils.hash_string("test")
        hash2 = utils.hash_string("test")
        assert hash1 == hash2  # Same input -> same hash
        assert 0.0 <= hash1 <= 1.0

    def test_different_strings(self):
        hash1 = utils.hash_string("test1")
        hash2 = utils.hash_string("test2")
        assert hash1 != hash2  # Different inputs -> different hashes

    def test_empty_string(self):
        hash_val = utils.hash_string("")
        assert 0.0 <= hash_val <= 1.0

    def test_none_value(self):
        hash_val = utils.hash_string(None)
        assert hash_val == 0.0


class TestEncodeHourCyclical:
    """Test encode_hour_cyclical function."""

    def test_midnight(self):
        sin_h, cos_h = utils.encode_hour_cyclical(0)
        assert abs(sin_h - 0.0) < 0.01
        assert abs(cos_h - 1.0) < 0.01

    def test_6am(self):
        sin_h, cos_h = utils.encode_hour_cyclical(6)
        assert abs(sin_h - 1.0) < 0.01
        assert abs(cos_h - 0.0) < 0.01

    def test_noon(self):
        sin_h, cos_h = utils.encode_hour_cyclical(12)
        assert abs(sin_h - 0.0) < 0.01
        assert abs(cos_h - (-1.0)) < 0.01

    def test_6pm(self):
        sin_h, cos_h = utils.encode_hour_cyclical(18)
        assert abs(sin_h - (-1.0)) < 0.01
        assert abs(cos_h - 0.0) < 0.01

    def test_cyclical_property(self):
        # Hour 23 and hour 0 should be close
        sin_23, cos_23 = utils.encode_hour_cyclical(23)
        sin_0, cos_0 = utils.encode_hour_cyclical(0)
        # They should be close in value
        assert abs(sin_23 - sin_0) < 0.3

    def test_invalid_hour(self):
        # Invalid hours should default to noon (12)
        sin_h, cos_h = utils.encode_hour_cyclical(25)
        sin_12, cos_12 = utils.encode_hour_cyclical(12)
        assert abs(sin_h - sin_12) < 0.01
        assert abs(cos_h - cos_12) < 0.01


class TestIsBusinessHours:
    """Test is_business_hours function."""

    def test_business_hours(self):
        # 9am-5pm are business hours
        assert utils.is_business_hours(9) == 1.0
        assert utils.is_business_hours(12) == 1.0
        assert utils.is_business_hours(16) == 1.0

    def test_non_business_hours(self):
        # Outside 9am-5pm
        assert utils.is_business_hours(8) == 0.0
        assert utils.is_business_hours(18) == 0.0
        assert utils.is_business_hours(0) == 0.0
        assert utils.is_business_hours(23) == 0.0

    def test_edge_cases(self):
        # Hour 17 (5pm) should be business hours
        assert utils.is_business_hours(17) == 1.0
        # Hour 8 should NOT be business hours
        assert utils.is_business_hours(8) == 0.0


class TestSafeLen:
    """Test safe_len function."""

    def test_normal_string(self):
        assert utils.safe_len("hello") == 5
        assert utils.safe_len("") == 0

    def test_none_value(self):
        assert utils.safe_len(None) == 0


class TestSafeFloat:
    """Test safe_float function."""

    def test_normal_conversion(self):
        assert utils.safe_float(5) == 5.0
        assert utils.safe_float(3.14) == 3.14
        assert utils.safe_float("10") == 10.0

    def test_invalid_conversion(self):
        assert utils.safe_float("invalid") == 0.0
        assert utils.safe_float(None) == 0.0

    def test_custom_default(self):
        assert utils.safe_float("invalid", default=99.0) == 99.0


class TestClamp:
    """Test clamp function."""

    def test_normal_clamp(self):
        assert utils.clamp(0.5, 0.0, 1.0) == 0.5
        assert utils.clamp(1.5, 0.0, 1.0) == 1.0
        assert utils.clamp(-0.5, 0.0, 1.0) == 0.0

    def test_custom_range(self):
        assert utils.clamp(5, 0, 10) == 5
        assert utils.clamp(15, 0, 10) == 10
        assert utils.clamp(-5, 0, 10) == 0
