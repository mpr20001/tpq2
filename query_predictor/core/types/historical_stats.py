"""
Shared schema for historical statistics.

Ensures train-serve parity by defining a single source of truth for:
- UserStats: Per-user query patterns and resource usage
- CatalogStats: Per-catalog query patterns
- SchemaStats: Per-schema query patterns
- HistoricalStatsSchema: Complete stats with versioning

Both training (HistoricalStatsComputer) and inference (HistoricalFeatureExtractor)
use these exact same dataclasses to prevent schema drift.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserStats:
    """
    Per-user historical statistics.

    Used for cold-start handling and user-specific feature extraction.
    Computed from training query logs using PySpark aggregations.
    """
    query_count: int
    heavy_rate: float
    avg_cpu_seconds: float
    p90_cpu_seconds: float
    avg_memory_gb: float
    p90_memory_gb: float
    unique_catalogs: int
    unique_schemas: int
    cpu_stddev: float


@dataclass
class CatalogStats:
    """
    Per-catalog historical statistics.

    Handles NULL catalogs using 'NULL' string key.
    Computed from training query logs using PySpark aggregations.
    """
    query_count: int
    heavy_rate: float
    avg_cpu_seconds: float
    p50_cpu_seconds: float
    p75_cpu_seconds: float
    p90_cpu_seconds: float
    p95_cpu_seconds: float
    avg_memory_gb: float
    p95_memory_gb: float
    unique_users: int
    unique_schemas: int


@dataclass
class SchemaStats:
    """
    Per-schema historical statistics.

    Handles NULL schemas using 'NULL' string key.
    Computed from training query logs using PySpark aggregations.
    """
    query_count: int
    heavy_rate: float
    avg_cpu_seconds: float
    p50_cpu_seconds: float
    p75_cpu_seconds: float
    p90_cpu_seconds: float
    p95_cpu_seconds: float
    avg_memory_gb: float
    p95_memory_gb: float
    unique_users: int
    unique_catalogs: int


@dataclass
class HistoricalStatsSchema:
    """
    Complete historical statistics schema with versioning.

    This is the top-level container for all historical stats.
    Saved to S3 as JSON and loaded by inference service at startup.

    Versioning ensures train-serve parity:
    - Stats version must match model version
    - Created_at tracks data freshness
    - Data_date_range documents training data period
    """
    version: str
    created_at: str
    data_date_range: Dict[str, str]  # {'start': '2025-06-01', 'end': '2025-07-15'}

    users: Dict[str, UserStats]
    catalogs: Dict[str, CatalogStats]
    schemas: Dict[str, SchemaStats]

    # Metadata for validation and monitoring
    total_queries_analyzed: int
    heavy_rate_overall: float

    # Optional table pattern stats (future enhancement)
    table_patterns: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate stats after initialization."""
        if self.total_queries_analyzed <= 0:
            raise ValueError("total_queries_analyzed must be positive")

        if not (0.0 <= self.heavy_rate_overall <= 1.0):
            raise ValueError(f"heavy_rate_overall must be in [0, 1], got {self.heavy_rate_overall}")

        logger.info(f"HistoricalStatsSchema v{self.version}: "
                   f"{len(self.users)} users, "
                   f"{len(self.catalogs)} catalogs, "
                   f"{len(self.schemas)} schemas")

    @property
    def stats_summary(self) -> Dict[str, Any]:
        """Get summary statistics for monitoring."""
        return {
            'version': self.version,
            'created_at': self.created_at,
            'data_date_range': self.data_date_range,
            'entity_counts': {
                'users': len(self.users),
                'catalogs': len(self.catalogs),
                'schemas': len(self.schemas)
            },
            'coverage': {
                'total_queries_analyzed': self.total_queries_analyzed,
                'heavy_rate_overall': self.heavy_rate_overall
            }
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to JSON-compatible dictionary.

        Used by training pipeline to save to S3.
        """
        return {
            'version': self.version,
            'created_at': self.created_at,
            'data_date_range': self.data_date_range,
            'users': {k: v.__dict__ for k, v in self.users.items()},
            'catalogs': {k: v.__dict__ for k, v in self.catalogs.items()},
            'schemas': {k: v.__dict__ for k, v in self.schemas.items()},
            'total_queries_analyzed': self.total_queries_analyzed,
            'heavy_rate_overall': self.heavy_rate_overall,
            'table_patterns': self.table_patterns
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HistoricalStatsSchema':
        """
        Deserialize from dictionary with validation.

        Used by inference service to load from S3.
        Raises ValueError if schema doesn't match expected structure.
        """
        try:
            # Deserialize nested dataclasses
            users = {k: UserStats(**v) for k, v in data.get('users', {}).items()}
            catalogs = {k: CatalogStats(**v) for k, v in data.get('catalogs', {}).items()}
            schemas = {k: SchemaStats(**v) for k, v in data.get('schemas', {}).items()}

            return cls(
                version=data['version'],
                created_at=data['created_at'],
                data_date_range=data['data_date_range'],
                users=users,
                catalogs=catalogs,
                schemas=schemas,
                total_queries_analyzed=data['total_queries_analyzed'],
                heavy_rate_overall=data['heavy_rate_overall'],
                table_patterns=data.get('table_patterns')
            )
        except KeyError as e:
            raise ValueError(f"Missing required field in HistoricalStatsSchema: {e}")
        except TypeError as e:
            raise ValueError(f"Invalid field type in HistoricalStatsSchema: {e}")
        except Exception as e:
            logger.error(f"Error deserializing HistoricalStatsSchema: {e}")
            raise ValueError(f"Failed to deserialize HistoricalStatsSchema: {e}")

    def get_user_stats(self, user: str) -> Optional[UserStats]:
        """Get stats for user, returns None if not found."""
        return self.users.get(user)

    def get_catalog_stats(self, catalog: Optional[str]) -> Optional[CatalogStats]:
        """Get stats for catalog, handles None by using 'NULL' key."""
        key = 'NULL' if catalog is None else catalog
        return self.catalogs.get(key)

    def get_schema_stats(self, schema: Optional[str]) -> Optional[SchemaStats]:
        """Get stats for schema, handles None by using 'NULL' key."""
        key = 'NULL' if schema is None else schema
        return self.schemas.get(key)

    def save_to_s3(self, s3_path: str, s3_handler=None):
        """
        Save historical stats to S3.

        Args:
            s3_path: S3 path (s3://bucket/key)
            s3_handler: Optional S3Handler instance
        """
        try:
            from query_predictor.utils.s3_utils import get_s3_handler

            handler = s3_handler or get_s3_handler()
            data = self.to_dict()
            handler.save_json(data, s3_path, cache_locally=True)
            logger.info(f"Saved HistoricalStatsSchema v{self.version} to {s3_path}")

        except Exception as e:
            logger.error(f"Failed to save HistoricalStatsSchema to S3: {e}")
            raise RuntimeError(f"Failed to save historical stats: {e}")

    @classmethod
    def load_from_s3(cls, s3_path: str, s3_handler=None, use_cache: bool = True) -> 'HistoricalStatsSchema':
        """
        Load historical stats from S3.

        Args:
            s3_path: S3 path (s3://bucket/key)
            s3_handler: Optional S3Handler instance
            use_cache: Whether to use local cache

        Returns:
            Loaded HistoricalStatsSchema instance
        """
        try:
            from query_predictor.utils.s3_utils import get_s3_handler

            handler = s3_handler or get_s3_handler()
            data = handler.load_json(s3_path, use_cache=use_cache)
            stats = cls.from_dict(data)
            logger.info(f"Loaded HistoricalStatsSchema v{stats.version} from {s3_path}")
            return stats

        except Exception as e:
            logger.error(f"Failed to load HistoricalStatsSchema from S3: {e}")
            raise RuntimeError(f"Failed to load historical stats: {e}")
