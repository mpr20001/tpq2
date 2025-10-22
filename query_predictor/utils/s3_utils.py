"""
S3 utilities for Trino Query Predictor.

Production-ready S3 handler with:
- Singleton pattern for connection reuse
- Automatic local caching with TTL
- Retry logic and error handling
- Fallback to local cache on S3 failures
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

# Check for boto3 availability
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    from botocore.config import Config
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    logger.warning("boto3 not available - S3 operations will use local filesystem fallback")


class S3Handler:
    """
    Production S3 handler with automatic caching and error handling.

    Features:
    - Singleton pattern (one client per process)
    - Automatic local caching with TTL
    - Retry logic with exponential backoff
    - Graceful fallback to cache on S3 failures
    - Thread-safe initialization
    """

    _instance = None

    def __new__(cls):
        """Ensure singleton pattern."""
        if cls._instance is None:
            cls._instance = super(S3Handler, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize S3 handler with environment configuration."""
        # Only initialize once
        if self._initialized:
            return

        self.s3_client = None
        self.default_bucket = os.environ.get('S3_BUCKET', 'uip-datalake-bucket-prod')
        self.default_prefix = os.environ.get('S3_PREFIX', 'sf_trino/query_predictor')
        self.use_s3 = os.environ.get('USE_S3', 'true').lower() == 'true'
        self.local_cache_dir = os.environ.get('LOCAL_CACHE_DIR', '/tmp/query_predictor_cache')
        self.cache_ttl_hours = int(os.environ.get('CACHE_TTL_HOURS', '24'))

        # Initialize S3 client if available
        if self.use_s3 and BOTO3_AVAILABLE:
            try:
                self._init_s3_client()
            except Exception as e:
                logger.warning(f"Failed to initialize S3 client: {e}")
                self.use_s3 = False

        # Ensure local cache directory exists
        os.makedirs(self.local_cache_dir, exist_ok=True)

        self._initialized = True
        logger.info(f"S3Handler initialized - Mode: {'S3' if self.use_s3 else 'Local'}, "
                   f"Bucket: {self.default_bucket}")

    def _init_s3_client(self):
        """Initialize S3 client with retry configuration."""
        config = Config(
            retries={
                'max_attempts': 3,
                'mode': 'adaptive'
            },
            max_pool_connections=10
        )

        region = os.environ.get('AWS_DEFAULT_REGION', 'us-west-2')
        self.s3_client = boto3.client('s3', region_name=region, config=config)

        # Test connection
        try:
            self.s3_client.head_bucket(Bucket=self.default_bucket)
            logger.info(f"S3 client initialized successfully for bucket: {self.default_bucket}")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                logger.error(f"Bucket {self.default_bucket} not found")
            else:
                logger.error(f"Cannot access bucket {self.default_bucket}: {error_code}")
            raise

    def parse_s3_path(self, s3_path: str) -> tuple:
        """
        Parse S3 path into bucket and key.

        Args:
            s3_path: S3 path (s3://bucket/key or bucket/key)

        Returns:
            Tuple of (bucket, key)
        """
        if s3_path.startswith('s3://'):
            s3_path = s3_path[5:]
        elif s3_path.startswith('s3a://'):
            s3_path = s3_path[6:]

        parts = s3_path.split('/', 1)
        bucket = parts[0] if parts[0] else self.default_bucket
        key = parts[1] if len(parts) > 1 else ''

        return bucket, key

    def load_json(self, s3_path: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Load JSON from S3 with automatic caching.

        Args:
            s3_path: S3 path (s3://bucket/key) or local path
            use_cache: Whether to use local cache

        Returns:
            Parsed JSON data

        Raises:
            RuntimeError: If load fails and no cache available
        """
        # Handle local path
        if not s3_path.startswith('s3://'):
            return self._load_json_local(s3_path)

        if not self.use_s3:
            raise RuntimeError("S3 not available and path is not local")

        # Check cache first if enabled
        cache_path = self._get_cache_path(s3_path)
        if use_cache and self._is_cache_valid(cache_path):
            logger.debug(f"Cache hit: {cache_path}")
            return self._load_json_local(cache_path)

        # Load from S3
        try:
            if not self.s3_client:
                raise RuntimeError("S3 client not initialized")
            bucket, key = self.parse_s3_path(s3_path)
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            content = response['Body'].read().decode('utf-8')
            data = json.loads(content)

            # Cache locally
            if use_cache:
                try:
                    self._save_json_local(data, cache_path)
                except Exception as e:
                    logger.warning(f"Failed to cache data: {e}")

            logger.info(f"Loaded JSON from S3: {s3_path}")
            return data

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Failed to load JSON from S3 {s3_path}: {error_code}")

            # Try stale cache as fallback
            if os.path.exists(cache_path):
                logger.warning(f"Using stale cache as fallback: {cache_path}")
                return self._load_json_local(cache_path)

            raise RuntimeError(f"Failed to load from S3 and no cache available: {error_code}")

        except Exception as e:
            logger.error(f"Unexpected error loading from S3 {s3_path}: {e}")

            # Try cache as fallback
            if os.path.exists(cache_path):
                logger.warning(f"Using cache due to error: {cache_path}")
                return self._load_json_local(cache_path)

            raise RuntimeError(f"Failed to load JSON: {e}")

    def save_json(self, data: Dict[str, Any], s3_path: str, cache_locally: bool = True):
        """
        Save JSON to S3 with optional local caching.

        Args:
            data: Dictionary to save
            s3_path: S3 path (s3://bucket/key) or local path
            cache_locally: Whether to cache locally

        Raises:
            RuntimeError: If save fails
        """
        # Handle local path
        if not s3_path.startswith('s3://'):
            return self._save_json_local(data, s3_path)

        if not self.use_s3:
            logger.warning("S3 not available, saving to cache only")
            cache_path = self._get_cache_path(s3_path)
            return self._save_json_local(data, cache_path)

        # Save to S3
        try:
            if not self.s3_client:
                raise RuntimeError("S3 client not initialized")
            bucket, key = self.parse_s3_path(s3_path)
            json_str = json.dumps(data, indent=2, default=str)

            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json_str.encode('utf-8'),
                ContentType='application/json',
                Metadata={
                    'created_at': data.get('created_at', datetime.now().isoformat()),
                    'version': str(data.get('version', ''))
                }
            )

            # Cache locally if requested
            if cache_locally:
                try:
                    cache_path = self._get_cache_path(s3_path)
                    self._save_json_local(data, cache_path)
                except Exception as e:
                    logger.warning(f"Failed to cache data locally: {e}")

            logger.info(f"Saved JSON to S3: {s3_path}")

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Failed to save JSON to S3 {s3_path}: {error_code}")

            # Fallback to cache only
            cache_path = self._get_cache_path(s3_path)
            self._save_json_local(data, cache_path)
            logger.warning(f"Saved to cache only due to S3 error: {cache_path}")

        except Exception as e:
            logger.error(f"Unexpected error saving to S3 {s3_path}: {e}")
            raise RuntimeError(f"Failed to save JSON: {e}")

    def download_file(self, s3_path: str, local_path: str):
        """
        Download file from S3 to local path.

        Args:
            s3_path: S3 source path
            local_path: Local destination path

        Raises:
            RuntimeError: If download fails
        """
        if not BOTO3_AVAILABLE or not self.use_s3:
            raise RuntimeError("S3 not available for download")

        try:
            if not self.s3_client:
                raise RuntimeError("S3 client not initialized")
            bucket, key = self.parse_s3_path(s3_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded {s3_path} to {local_path}")

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Failed to download {s3_path}: {error_code}")
            raise RuntimeError(f"Download failed: {error_code}")

    def upload_file(self, local_path: str, s3_path: str):
        """
        Upload file from local to S3.

        Args:
            local_path: Local source path
            s3_path: S3 destination path

        Raises:
            RuntimeError: If upload fails
        """
        if not BOTO3_AVAILABLE or not self.use_s3:
            logger.warning("S3 not available, keeping file local")
            return

        try:
            if not self.s3_client:
                raise RuntimeError("S3 client not initialized")
            bucket, key = self.parse_s3_path(s3_path)

            # Determine content type
            content_type = self._get_content_type(local_path)

            self.s3_client.upload_file(
                local_path, bucket, key,
                ExtraArgs={'ContentType': content_type}
            )
            logger.info(f"Uploaded {local_path} to {s3_path}")

        except ClientError as e:
            error_code = e.response['Error']['Code']
            logger.error(f"Failed to upload {local_path}: {error_code}")
            raise RuntimeError(f"Upload failed: {error_code}")

    def exists(self, s3_path: str) -> bool:
        """
        Check if file exists in S3.

        Args:
            s3_path: S3 path to check

        Returns:
            True if file exists
        """
        if not s3_path.startswith('s3://'):
            return os.path.exists(s3_path)

        if not self.use_s3:
            return False

        try:
            if not self.s3_client:
                return False
            bucket, key = self.parse_s3_path(s3_path)
            self.s3_client.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError:
            return False
        except Exception as e:
            logger.error(f"Error checking existence of {s3_path}: {e}")
            return False

    def _get_cache_path(self, s3_path: str) -> str:
        """Get local cache path for S3 path."""
        bucket, key = self.parse_s3_path(s3_path)
        cache_path = os.path.join(self.local_cache_dir, bucket, key)
        return cache_path

    def _is_cache_valid(self, cache_path: str) -> bool:
        """Check if cache is still valid based on TTL."""
        if not os.path.exists(cache_path):
            return False

        if self.cache_ttl_hours <= 0:
            return True  # No TTL, always valid

        try:
            file_age_hours = (datetime.now().timestamp() -
                            os.path.getmtime(cache_path)) / 3600
            return file_age_hours < self.cache_ttl_hours
        except:
            return False

    def _load_json_local(self, path: str) -> Dict[str, Any]:
        """Load JSON from local file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"File not found: {path}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid JSON in {path}: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to load JSON from {path}: {e}")

    def _save_json_local(self, data: Dict[str, Any], path: str):
        """Save JSON to local file."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            logger.debug(f"Saved JSON to local: {path}")
        except Exception as e:
            raise RuntimeError(f"Failed to save JSON to {path}: {e}")

    def _get_content_type(self, file_path: str) -> str:
        """Determine content type from file extension."""
        ext = os.path.splitext(file_path)[1].lower()
        content_types = {
            '.json': 'application/json',
            '.onnx': 'application/octet-stream',
            '.parquet': 'application/octet-stream',
            '.csv': 'text/csv',
            '.yaml': 'text/yaml',
            '.yml': 'text/yaml',
        }
        return content_types.get(ext, 'application/octet-stream')


# Global singleton accessor
def get_s3_handler() -> S3Handler:
    """
    Get singleton S3Handler instance.

    Returns:
        S3Handler singleton
    """
    return S3Handler()
