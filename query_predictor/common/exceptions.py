"""Custom exceptions for the query predictor service."""


class QueryPredictorError(Exception):
    """Base exception for all query predictor errors."""
    pass


class ConfigurationError(QueryPredictorError):
    """Raised when configuration loading or validation fails."""
    pass


class ModelLoadError(QueryPredictorError):
    """Raised when model loading fails."""
    pass


class FeatureExtractionError(QueryPredictorError):
    """Raised when feature extraction fails."""
    pass


class PredictionError(QueryPredictorError):
    """Raised when prediction fails."""
    pass


class ValidationError(QueryPredictorError):
    """Raised when request validation fails."""
    pass
