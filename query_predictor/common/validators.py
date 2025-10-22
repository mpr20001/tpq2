"""Shared validation functions for query predictor."""

def validate_query_not_empty(query) -> None:
    """Validate query is not empty or whitespace-only."""
    if not query or not isinstance(query, str):
        raise ValueError("Query field is required and must be a non-empty string")
    if not query.strip():
        raise ValueError("Query cannot be empty or whitespace-only")


def validate_query_length(query, max_length: int) -> None:
    """Validate query does not exceed maximum length."""
    query_length = len(query)
    if query_length > max_length:
        raise ValueError(
            f"Query exceeds maximum length of {max_length:,} characters "
            f"(got {query_length:,}). Consider simplifying the query or breaking it into smaller parts."
        )


def validate_query_data_dict(query_data) -> None:
    """Validate query_data is a dictionary."""
    if not isinstance(query_data, dict):
        raise ValueError(f"query_data must be a dictionary, got {type(query_data).__name__}")
