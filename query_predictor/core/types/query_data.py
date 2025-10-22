"""
Data types for feature extraction.

Provides:
- QueryData: Validated query data for feature extraction
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class QueryData:
    """
    Validated query data for feature extraction.

    All extractors receive this standardized format.
    """
    query: str
    user: str
    catalog: Optional[str] = None
    schema: Optional[str] = None
    hour: int = 12
    client_info: str = ""
    session_properties: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate and normalize data."""
        # Ensure query is string
        if not isinstance(self.query, str):
            self.query = str(self.query) if self.query is not None else ""

        # Validate query is not empty
        if not self.query.strip():
            raise ValueError("Query cannot be empty")

        # Ensure user is string
        if not isinstance(self.user, str):
            self.user = str(self.user) if self.user is not None else ""

        # Validate hour range
        if not isinstance(self.hour, int) or not (0 <= self.hour <= 23):
            raise ValueError("Hour must be between 0 and 23")

        # Ensure session_properties is dict
        if self.session_properties is None:
            self.session_properties = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'query': self.query,
            'user': self.user,
            'catalog': self.catalog,
            'schema': self.schema,
            'hour': self.hour,
            'clientInfo': self.client_info,
            'sessionProperties': self.session_properties,
            'session_properties': self.session_properties  # Include both formats for compatibility
        }
