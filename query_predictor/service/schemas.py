"""
Pydantic schemas for API request/response models.

Defines the API contract for all service endpoints using Pydantic models.
"""

import logging
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from query_predictor.core.types.query_data import QueryData
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from query_predictor.common import validators
from query_predictor.core.featurizer import constants

logger = logging.getLogger(__name__)


class PredictionRequest(BaseModel):
    """
    Request model for query prediction endpoint.

    Validates incoming prediction requests and handles NULL catalog/schema.
    """
    query: str = Field(..., description="SQL query text to predict", min_length=1)
    user: Optional[str] = Field(None, description="Username executing the query")
    catalog: Optional[str] = Field(None, description="Trino catalog (nullable)")
    schema_name: Optional[str] = Field(None, description="Trino schema (nullable)", alias="schema")
    hour: Optional[int] = Field(None, description="Hour of day (0-23)", ge=0, le=23)
    clientInfo: Optional[str] = Field(None, description="Client information")
    sessionProperties: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Session properties")

    @field_validator('query')
    @classmethod
    def validate_query(cls, v):
        """Validate query is not empty after stripping."""
        # Use shared validators
        validators.validate_query_not_empty(v)
        validators.validate_query_length(v, constants.MAX_QUERY_LENGTH)
        
        return v

    @model_validator(mode='after')
    def set_default_hour(self):
        """Set current hour if not provided."""
        if self.hour is None:
            self.hour = datetime.now().hour
        return self

    @field_validator('user', mode='before')
    @classmethod
    def normalize_user(cls, v):
        """Normalize user to lowercase."""
        if v:
            return v.lower().strip()
        return v

    @field_validator('catalog', 'schema_name', mode='before')
    @classmethod
    def normalize_catalog_schema(cls, v):
        """Normalize catalog/schema, treating empty strings as None."""
        if v is None or v == '':
            return None
        return v.strip()

    @field_validator('sessionProperties', mode='before')
    @classmethod
    def validate_session_properties(cls, v):
        """Validate session properties."""
        if v is None:
            return {}
        if not isinstance(v, dict):
            raise ValueError("Session properties must be a dictionary")
        
        # Check for reasonable size limit
        if len(str(v)) > 10000:  # 10KB limit for session properties
            raise ValueError("Session properties exceed maximum size")
        
        # Validate keys and values
        for key, value in v.items():
            if not isinstance(key, str):
                raise ValueError("Session property keys must be strings")
            if len(key) > 100:
                raise ValueError("Session property key too long")
            if not isinstance(value, (str, int, float, bool, type(None))):
                raise ValueError("Session property values must be strings, numbers, booleans, or null")
        
        return v

    @field_validator('clientInfo', mode='before')
    @classmethod
    def validate_client_info(cls, v):
        """Validate client info."""
        if v is None:
            return v
        if len(v) > 500:  # Reasonable limit for client info
            raise ValueError("Client info too long")
        return v.strip() if isinstance(v, str) else v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query": "SELECT * FROM orders WHERE created_date > '2024-01-01'",
                "user": "john.doe",
                "catalog": "hive",
                "schema": "sales",
                "hour": 14,
                "clientInfo": "trino-cli/1.0",
                "sessionProperties": {"query_max_memory": "10GB"}
            }
        }
    )

    def to_query_data(self) -> 'QueryData':
        """
        Convert to QueryData object for feature extraction.

        Returns:
            QueryData object with validated fields
        """
        from query_predictor.core.types.query_data import QueryData
        
        return QueryData(
            query=self.query,
            user=self.user or 'unknown',
            catalog=self.catalog,  # Can be None
            schema=self.schema_name,    # Can be None
            hour=self.hour or 12,  # Default to noon if None
            client_info=self.clientInfo or '',
            session_properties=self.sessionProperties or {}
        )


class PredictionResponse(BaseModel):
    """
    Response model for query prediction endpoint.

    Returns prediction results with metadata and diagnostics.
    """
    prediction: str = Field(..., description="Predicted class (HEAVY/LIGHT)")
    confidence: float = Field(..., description="Model confidence (0.0-1.0)", ge=0.0, le=1.0)
    probability: Dict[str, float] = Field(..., description="Class probabilities")

    # Metadata
    model_version: str = Field(..., description="Model version used")
    featurizer_version: str = Field(..., description="Featurizer version used")
    feature_count: int = Field(..., description="Number of features extracted")

    # Diagnostics
    processing_time_ms: float = Field(..., description="Total processing time in milliseconds")
    feature_extraction_time_ms: float = Field(..., description="Feature extraction time in milliseconds")
    model_inference_time_ms: float = Field(..., description="Model inference time in milliseconds")

    # Optional enrichment
    query_info: Optional[Dict[str, Any]] = Field(None, description="Additional query information")
    warnings: Optional[List[str]] = Field(default_factory=list, description="Any warnings during processing")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "HEAVY",
                "confidence": 0.92,
                "probability": {"HEAVY": 0.92, "LIGHT": 0.08},
                "model_version": "1.0.0",
                "featurizer_version": "3.0.0",
                "feature_count": 78,
                "processing_time_ms": 45.2,
                "feature_extraction_time_ms": 38.1,
                "model_inference_time_ms": 7.1,
                "query_info": {
                    "table_count": 2,
                    "join_count": 1,
                    "has_aggregation": False
                },
                "warnings": []
            }
        }
    )


class HealthCheckResponse(BaseModel):
    """
    Response model for health check endpoint.
    """
    status: str = Field(..., description="Service status (healthy/unhealthy)")
    version: str = Field(..., description="Service version")
    featurizer_version: str = Field(..., description="Featurizer version")
    model_version: Optional[str] = Field(None, description="Model version if loaded")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")
    timestamp: str = Field(..., description="Current timestamp")

    # Component health
    s3_connection: bool = Field(..., description="S3 connection status")
    model_loaded: bool = Field(..., description="Model loaded status")
    feature_spec_loaded: bool = Field(..., description="Feature spec loaded status")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "featurizer_version": "3.0.0",
                "model_version": "1.0.0",
                "uptime_seconds": 3600.5,
                "timestamp": "2024-01-15T10:30:00Z",
                "s3_connection": True,
                "model_loaded": True,
                "feature_spec_loaded": True
            }
        }
    )


class ErrorResponse(BaseModel):
    """
    Standard error response model.
    """
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: str = Field(..., description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request ID for tracing")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "error": "ValidationError",
                "message": "Query cannot be empty",
                "details": {"field": "query", "provided_value": ""},
                "timestamp": "2024-01-15T10:30:00Z",
                "request_id": "req-123-456-789"
            }
        }
    )


# Validation utilities

def validate_s3_path(path: str) -> bool:
    """
    Validate S3 path format.

    Args:
        path: S3 path to validate

    Returns:
        True if valid, False otherwise
    """
    if not path:
        return False
    if not path.startswith('s3://'):
        return False
    if len(path.split('/')) < 4:  # s3://bucket/key
        return False
    return True


def create_error_response(error_type: str, message: str,
                         details: Optional[Dict] = None,
                         request_id: Optional[str] = None) -> ErrorResponse:
    """
    Create standardized error response.

    Args:
        error_type: Type of error
        message: Human-readable error message
        details: Optional additional details
        request_id: Optional request ID for tracing

    Returns:
        ErrorResponse object
    """
    return ErrorResponse(
        error=error_type,
        message=message,
        details=details,
        timestamp=datetime.now(timezone.utc).isoformat(),
        request_id=request_id
    )
