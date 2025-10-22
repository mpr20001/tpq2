"""
Shared utility functions for feature extraction.

Provides safe, NULL-aware utility functions used across extractors.
"""

import math
from typing import Tuple, Optional


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with zero-denominator handling.

    Args:
        numerator: Numerator value
        denominator: Denominator value
        default: Value to return if denominator is zero

    Returns:
        numerator / denominator, or default if denominator is zero
    """
    try:
        if denominator == 0 or not math.isfinite(denominator):
            return default
        result = numerator / denominator
        return result if math.isfinite(result) else default
    except:
        return default


def normalize_value(value: float, max_value: float, default: float = 0.0) -> float:
    """
    Normalize value to [0, 1] range.

    Args:
        value: Value to normalize
        max_value: Maximum value for normalization
        default: Value to return on error

    Returns:
        Normalized value between 0 and 1
    """
    try:
        if max_value == 0:
            return default
        normalized = min(value / max_value, 1.0)
        return normalized if math.isfinite(normalized) else default
    except:
        return default


def hash_string(text: str, modulo: int = 1000000) -> float:
    """
    Hash string to normalized float value.

    Args:
        text: String to hash
        modulo: Modulo value for hash (determines range)

    Returns:
        Normalized hash value between 0 and 1
    """
    try:
        if not text or not isinstance(text, str):
            return 0.0
        hash_value = abs(hash(text)) % modulo
        return hash_value / modulo
    except:
        return 0.0


def encode_hour_cyclical(hour: int) -> Tuple[float, float]:
    """
    Encode hour of day as cyclical features (sin, cos).

    This preserves the cyclical nature of time (23:00 is close to 00:00).

    Args:
        hour: Hour of day (0-23)

    Returns:
        Tuple of (sin_encoded, cos_encoded)
    """
    try:
        if not isinstance(hour, int) or not (0 <= hour <= 23):
            hour = 12  # Default to noon

        angle = 2 * math.pi * hour / 24
        sin_val = math.sin(angle)
        cos_val = math.cos(angle)

        return sin_val, cos_val
    except:
        return 0.0, 1.0  # Default to midnight


def safe_len(text: Optional[str]) -> int:
    """
    Get length of string with NULL safety.

    Args:
        text: String to measure

    Returns:
        Length of string, or 0 if NULL/None
    """
    try:
        return len(text) if text is not None else 0
    except:
        return 0


def safe_upper(text: Optional[str]) -> str:
    """
    Convert string to uppercase with NULL safety.

    Args:
        text: String to convert

    Returns:
        Uppercase string, or empty string if NULL/None
    """
    try:
        return text.upper() if text is not None else ""
    except:
        return ""


def safe_count(text: Optional[str], substring: str) -> int:
    """
    Count occurrences of substring with NULL safety.

    Args:
        text: String to search in
        substring: Substring to count

    Returns:
        Count of occurrences, or 0 if text is NULL
    """
    try:
        if text is None or not isinstance(text, str):
            return 0
        return text.count(substring)
    except:
        return 0


def safe_float(value, default: float = 0.0) -> float:
    """
    Convert value to float with error handling.

    Args:
        value: Value to convert
        default: Default value on conversion error

    Returns:
        Float value or default
    """
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except:
        return default


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """
    Clamp value to specified range.

    Args:
        value: Value to clamp
        min_val: Minimum value
        max_val: Maximum value

    Returns:
        Clamped value
    """
    try:
        return max(min_val, min(value, max_val))
    except:
        return min_val


def is_business_hours(hour: int) -> float:
    """
    Check if hour is during business hours (9-17).

    Args:
        hour: Hour of day (0-23)

    Returns:
        1.0 if business hours, 0.0 otherwise
    """
    try:
        if not isinstance(hour, int):
            return 0.0
        return 1.0 if 9 <= hour <= 17 else 0.0
    except:
        return 0.0
