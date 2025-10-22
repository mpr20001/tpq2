"""
Unit tests for SparkMLTfidfPipeline.

Tests Spark ML TF-IDF feature extraction with distributed fitting and SQL-aware optimizations.
"""

import pytest
import numpy as np
import tempfile
import pickle


@pytest.fixture
def tfidf_config():
    """Standard TF-IDF configuration (SQL-aware defaults)."""
    return {
        'tfidf_vocab_size': 50,  # Small vocab for testing
        'min_df': 1,             # Low for test data
        'max_df': 0.95,
        'use_binary': True,
        'filter_sql_keywords': True,
        'normalize_sql': True
    }


@pytest.fixture
def tfidf_config_basic():
    """Basic TF-IDF configuration without SQL optimizations."""
    return {
        'tfidf_vocab_size': 50,
        'min_df': 1,
        'max_df': 0.95,
        'use_binary': False,
        'filter_sql_keywords': False,
        'normalize_sql': False
    }


@pytest.fixture
def sample_queries():
    """Sample SQL queries for training."""
    return [
        "SELECT * FROM users WHERE id = 1",
        "SELECT name, email FROM customers ORDER BY name",
        "SELECT COUNT(*) FROM orders GROUP BY user_id",
        "SELECT a.*, b.name FROM table_a a JOIN table_b b ON a.id = b.id",
        "SELECT user_id, SUM(amount) FROM transactions WHERE date > '2025-01-01' GROUP BY user_id",
        "SELECT * FROM products WHERE category IN ('electronics', 'books')",
        "SELECT DISTINCT catalog FROM tables",
        "SELECT MAX(value), MIN(value), AVG(value) FROM metrics",
        "SELECT * FROM logs WHERE timestamp > NOW() - INTERVAL '1 day'",
        "SELECT user_id, COUNT(*) as cnt FROM events GROUP BY user_id HAVING cnt > 10"
    ]


class TestSparkMLTfidfPipelineInitialization:
    """Test SparkMLTfidfPipeline initialization."""

    def test_basic_initialization(self, tfidf_config):
        """Test basic initialization with config."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)

        assert pipeline.vocab_size == 50
        assert pipeline.min_df == 1
        assert pipeline.max_df == 0.95
        assert pipeline.use_binary is True
        assert pipeline.filter_sql_keywords is True
        assert pipeline.normalize_sql is True
        assert not pipeline.is_fitted
        assert len(pipeline.vocabulary) == 0

    def test_initialization_with_defaults(self):
        """Test initialization with default values."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        config = {}  # Empty config should use defaults
        pipeline = SparkMLTfidfPipeline(config)

        assert pipeline.vocab_size == 250  # New default
        assert pipeline.min_df == 100  # New default
        assert pipeline.max_df == 0.80  # New default
        assert pipeline.use_binary is True  # New default
        assert pipeline.filter_sql_keywords is True  # New default
        assert pipeline.normalize_sql is True  # New default

    def test_pyspark_import_error(self):
        """Test that ImportError is raised if PySpark not available."""
        from query_predictor.training import spark_ml_tfidf_pipeline

        # Temporarily mock PySpark as unavailable
        original_pyspark = spark_ml_tfidf_pipeline.PYSPARK_AVAILABLE
        try:
            spark_ml_tfidf_pipeline.PYSPARK_AVAILABLE = False

            with pytest.raises(ImportError, match="PySpark is required"):
                spark_ml_tfidf_pipeline.SparkMLTfidfPipeline({})
        finally:
            spark_ml_tfidf_pipeline.PYSPARK_AVAILABLE = original_pyspark

    def test_sklearn_import_error(self):
        """Test that ImportError is raised if sklearn not available."""
        from query_predictor.training import spark_ml_tfidf_pipeline

        # Temporarily mock sklearn as unavailable
        original_sklearn = spark_ml_tfidf_pipeline.SKLEARN_AVAILABLE
        try:
            spark_ml_tfidf_pipeline.SKLEARN_AVAILABLE = False

            with pytest.raises(ImportError, match="sklearn is required"):
                spark_ml_tfidf_pipeline.SparkMLTfidfPipeline({})
        finally:
            spark_ml_tfidf_pipeline.SKLEARN_AVAILABLE = original_sklearn


class TestSQLNormalization:
    """Test SQL query normalization."""

    def test_normalize_literals(self):
        """Test that string literals are normalized."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        query = "SELECT * FROM users WHERE name = 'John Doe' AND email = \"test@example.com\""
        normalized = SparkMLTfidfPipeline._normalize_sql_query(query)

        assert "John Doe" not in normalized
        assert "test@example.com" not in normalized
        assert "STRING_LITERAL" in normalized

    def test_normalize_numbers(self):
        """Test that numbers are normalized."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        query = "SELECT * FROM users WHERE id = 12345 AND age > 25"
        normalized = SparkMLTfidfPipeline._normalize_sql_query(query)

        assert "12345" not in normalized
        assert "25" not in normalized
        assert "NUMERIC" in normalized

    def test_normalize_dates_in_quotes(self):
        """Test that dates in quotes become STRING_LITERAL (correct behavior)."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        # Dates in quotes are string literals
        query = "SELECT * FROM logs WHERE date = '2025-01-15' OR date = '01/15/2025'"
        normalized = SparkMLTfidfPipeline._normalize_sql_query(query)

        # Dates in quotes should be replaced as STRING_LITERAL
        assert "2025-01-15" not in normalized
        assert "01/15/2025" not in normalized
        assert "STRING_LITERAL" in normalized

    def test_normalize_dates_unquoted(self):
        """Test that unquoted dates are normalized to DATE_VALUE."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        # Unquoted date patterns (e.g., in date arithmetic)
        query = "SELECT * FROM logs WHERE date_col > 2025-01-15"
        normalized = SparkMLTfidfPipeline._normalize_sql_query(query)

        assert "2025-01-15" not in normalized
        assert "DATE_VALUE" in normalized

    def test_normalize_timestamps_in_quotes(self):
        """Test that timestamps in quotes become STRING_LITERAL (correct behavior)."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        # Timestamps in quotes are string literals
        query = "SELECT * FROM logs WHERE timestamp > '2025-01-15 10:30:00'"
        normalized = SparkMLTfidfPipeline._normalize_sql_query(query)

        # Timestamps in quotes should be replaced as STRING_LITERAL
        assert "2025-01-15 10:30:00" not in normalized
        assert "STRING_LITERAL" in normalized

    def test_normalize_timestamps_unquoted(self):
        """Test that unquoted timestamps are normalized to TIMESTAMP_VALUE."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        # Unquoted timestamp pattern
        query = "SELECT * FROM logs WHERE ts > 2025-01-15 10:30:00"
        normalized = SparkMLTfidfPipeline._normalize_sql_query(query)

        assert "2025-01-15 10:30:00" not in normalized
        assert "TIMESTAMP_VALUE" in normalized

    def test_normalize_empty_query(self):
        """Test that empty query returns empty string."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        assert SparkMLTfidfPipeline._normalize_sql_query("") == ""
        assert SparkMLTfidfPipeline._normalize_sql_query(None) == ""


class TestSparkMLTfidfPipelineTransformSingle:
    """Test single-query transformation (inference mode)."""

    def test_transform_single_without_spark(self, tfidf_config, sample_queries):
        """Test transform_single using sklearn (no Spark needed for inference)."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        # For unit test, we'll mock the fitted state manually
        pipeline = SparkMLTfidfPipeline(tfidf_config)

        # Manually set fitted state and vocabulary (simulating a loaded pipeline)
        pipeline.vocabulary = ['users', 'customers', 'orders', 'products', 'transactions', 'metrics', 'events']
        pipeline.idf_weights = np.array([1.0, 1.2, 1.5, 1.1, 1.3, 1.4, 1.6])
        pipeline.is_fitted = True

        # Initialize sklearn vectorizer
        pipeline._init_sklearn_vectorizer()

        # Test transformation
        query = "SELECT * FROM users WHERE id = 1"
        features = pipeline.transform_single(query)

        assert isinstance(features, np.ndarray)
        assert features.dtype == np.float32
        assert features.shape == (len(pipeline.vocabulary),)
        assert np.all(features >= 0)  # TF-IDF scores are non-negative

    def test_transform_with_sql_keywords_filtered(self, tfidf_config):
        """Test that SQL keywords are filtered out."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)

        # Set fitted state with vocabulary (no SQL keywords)
        pipeline.vocabulary = ['users', 'customers', 'products']
        pipeline.idf_weights = np.array([1.0, 1.0, 1.0])
        pipeline.is_fitted = True
        pipeline._init_sklearn_vectorizer()

        # Query with many SQL keywords
        query = "SELECT * FROM users WHERE id = 1"
        features = pipeline.transform_single(query)

        # Should still extract 'users' even though query has many keywords
        assert features[0] > 0  # 'users' should have non-zero weight

    def test_transform_empty_query(self, tfidf_config):
        """Test transforming empty query returns zero vector."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)

        # Set minimal fitted state
        pipeline.vocabulary = ['users', 'products', 'orders']
        pipeline.idf_weights = np.array([1.0, 1.0, 1.0])
        pipeline.is_fitted = True
        pipeline._init_sklearn_vectorizer()

        features = pipeline.transform_single("")

        assert isinstance(features, np.ndarray)
        assert features.shape == (len(pipeline.vocabulary),)
        assert np.all(features == 0)

    def test_transform_before_fit_raises_error(self, tfidf_config):
        """Test that transform before fit raises error."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)

        with pytest.raises(ValueError, match="Must call fit_on_dataframe"):
            pipeline.transform_single("SELECT * FROM users")


class TestSparkMLTfidfPipelineSaveLoad:
    """Test save/load functionality."""

    def test_save_requires_fit(self, tfidf_config):
        """Test that save before fit raises error."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            with pytest.raises(ValueError, match="Must call fit_on_dataframe"):
                pipeline.save(tmp.name)

    def test_save_and_load_round_trip(self, tfidf_config):
        """Test save and load preserves pipeline state."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        # Create and manually fit pipeline
        pipeline = SparkMLTfidfPipeline(tfidf_config)
        pipeline.vocabulary = ['users', 'products', 'orders', 'customers', 'transactions']
        pipeline.idf_weights = np.array([1.0, 1.2, 1.5, 1.1, 1.3])
        pipeline.is_fitted = True
        pipeline._init_sklearn_vectorizer()

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            # Save
            pipeline.save(tmp.name)

            # Load
            loaded_pipeline = SparkMLTfidfPipeline.load(tmp.name)

            # Verify state preserved
            assert loaded_pipeline.is_fitted
            assert len(loaded_pipeline.vocabulary) == len(pipeline.vocabulary)
            assert loaded_pipeline.vocabulary == pipeline.vocabulary
            assert loaded_pipeline.vocab_size == pipeline.vocab_size
            assert loaded_pipeline.use_binary == pipeline.use_binary
            assert loaded_pipeline.filter_sql_keywords == pipeline.filter_sql_keywords
            assert loaded_pipeline.normalize_sql == pipeline.normalize_sql

            # Verify transforms produce same results
            test_query = "SELECT * FROM users WHERE id = 1"
            original_features = pipeline.transform_single(test_query)
            loaded_features = loaded_pipeline.transform_single(test_query)

            np.testing.assert_array_almost_equal(original_features, loaded_features)

    def test_saved_pickle_structure(self, tfidf_config):
        """Test that saved pickle has expected structure."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)
        pipeline.vocabulary = ['users', 'products', 'orders']
        pipeline.idf_weights = np.array([1.0, 1.0, 1.0])
        pipeline.is_fitted = True
        pipeline._init_sklearn_vectorizer()

        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp:
            pipeline.save(tmp.name)

            # Load pickle and verify structure
            with open(tmp.name, 'rb') as f:
                save_data = pickle.load(f)

            assert 'vocabulary' in save_data
            assert 'idf_weights' in save_data
            assert 'vocab_size' in save_data
            assert 'min_df' in save_data
            assert 'max_df' in save_data
            assert 'use_binary' in save_data
            assert 'filter_sql_keywords' in save_data
            assert 'normalize_sql' in save_data
            assert 'sklearn_vectorizer' in save_data


class TestSparkMLTfidfPipelineMetadata:
    """Test feature metadata methods."""

    def test_get_metadata_requires_fit(self, tfidf_config):
        """Test that get_feature_metadata before fit raises error."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)

        with pytest.raises(ValueError, match="Must call fit_on_dataframe"):
            pipeline.get_feature_metadata()

    def test_get_feature_metadata(self, tfidf_config):
        """Test get_feature_metadata returns correct structure."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)
        pipeline.vocabulary = ['users', 'products', 'orders', 'customers']
        pipeline.idf_weights = np.array([1.0, 1.2, 1.5, 1.1])
        pipeline.is_fitted = True

        metadata = pipeline.get_feature_metadata()

        assert isinstance(metadata, dict)
        assert 'vocab_size' in metadata
        assert 'max_features' in metadata
        assert 'min_df' in metadata
        assert 'max_df' in metadata
        assert 'use_binary' in metadata
        assert 'filter_sql_keywords' in metadata
        assert 'normalize_sql' in metadata
        assert 'feature_names' in metadata
        assert 'is_fitted' in metadata
        assert 'method' in metadata

        assert metadata['is_fitted'] is True
        assert metadata['vocab_size'] == len(metadata['feature_names'])
        assert metadata['method'] == 'spark_ml_countvectorizer_optimized'
        assert metadata['use_binary'] is True

    def test_get_top_features(self, tfidf_config):
        """Test get_top_features returns correct format."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)
        pipeline.vocabulary = ['users', 'products', 'orders', 'customers', 'transactions']
        pipeline.idf_weights = np.array([1.0, 1.2, 1.5, 1.3, 1.4])
        pipeline.is_fitted = True
        pipeline._init_sklearn_vectorizer()

        query = "SELECT * FROM users WHERE id = 1"
        top_features = pipeline.get_top_features(query, top_n=3)

        assert isinstance(top_features, list)
        assert len(top_features) <= 3

        for feature_name, score in top_features:
            assert isinstance(feature_name, str)
            assert feature_name.startswith("tfidf_")
            assert isinstance(score, float)
            assert score > 0  # Only non-zero features should be returned

    def test_get_top_features_requires_fit(self, tfidf_config):
        """Test that get_top_features before fit raises error."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)

        with pytest.raises(ValueError, match="Must call fit_on_dataframe"):
            pipeline.get_top_features("SELECT * FROM users", top_n=5)


class TestSparkMLTfidfPipelineUDF:
    """Test Spark UDF creation."""

    def test_create_udf_requires_fit(self, tfidf_config):
        """Test that creating UDF before fit raises error."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)

        with pytest.raises(ValueError, match="Must call fit_on_dataframe"):
            pipeline.create_spark_udf()

    def test_udf_returns_callable(self, tfidf_config):
        """Test that create_spark_udf returns a callable."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)
        pipeline.vocabulary = ['users', 'products', 'orders']
        pipeline.idf_weights = np.array([1.0, 1.0, 1.0])
        pipeline.is_fitted = True

        # This will fail if PySpark not available, but that's expected in unit tests
        try:
            udf = pipeline.create_spark_udf()
            assert callable(udf)
        except ImportError:
            pytest.skip("PySpark not available in unit test environment")


class TestSparkMLTfidfPipelineEdgeCases:
    """Test edge cases and error handling."""

    def test_transform_with_special_characters(self, tfidf_config):
        """Test transform handles queries with special characters."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)
        pipeline.vocabulary = ['users', 'products', 'table_name', 'column_value']
        pipeline.idf_weights = np.array([1.0, 1.2, 1.5, 1.1])
        pipeline.is_fitted = True
        pipeline._init_sklearn_vectorizer()

        query_with_special = "SELECT /* comment */ * FROM `table_name` WHERE col = 'value'"
        features = pipeline.transform_single(query_with_special)

        assert isinstance(features, np.ndarray)
        assert features.shape == (len(pipeline.vocabulary),)
        # table_name should be preserved (underscores kept)
        assert features[2] > 0  # table_name present

    def test_case_insensitivity(self, tfidf_config):
        """Test that TF-IDF is case-insensitive."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        pipeline = SparkMLTfidfPipeline(tfidf_config)
        pipeline.vocabulary = ['users', 'products', 'customers']
        pipeline.idf_weights = np.array([1.0, 1.2, 1.5])
        pipeline.is_fitted = True
        pipeline._init_sklearn_vectorizer()

        query_lower = "select * from users"
        query_upper = "SELECT * FROM USERS"
        query_mixed = "SeLeCt * FrOm UsErS"

        features_lower = pipeline.transform_single(query_lower)
        features_upper = pipeline.transform_single(query_upper)
        features_mixed = pipeline.transform_single(query_mixed)

        # All should produce identical features
        np.testing.assert_array_almost_equal(features_lower, features_upper)
        np.testing.assert_array_almost_equal(features_lower, features_mixed)

    def test_binary_mode_vs_count_mode(self):
        """Test binary mode produces different results than count mode."""
        from query_predictor.training.spark_ml_tfidf_pipeline import SparkMLTfidfPipeline

        # Binary mode
        config_binary = {'tfidf_vocab_size': 10, 'min_df': 1, 'max_df': 0.95, 'use_binary': True,
                        'filter_sql_keywords': False, 'normalize_sql': False}
        pipeline_binary = SparkMLTfidfPipeline(config_binary)
        pipeline_binary.vocabulary = ['users', 'products']
        pipeline_binary.idf_weights = np.array([1.0, 1.0])
        pipeline_binary.is_fitted = True
        pipeline_binary._init_sklearn_vectorizer()

        # Count mode
        config_count = {'tfidf_vocab_size': 10, 'min_df': 1, 'max_df': 0.95, 'use_binary': False,
                       'filter_sql_keywords': False, 'normalize_sql': False}
        pipeline_count = SparkMLTfidfPipeline(config_count)
        pipeline_count.vocabulary = ['users', 'products']
        pipeline_count.idf_weights = np.array([1.0, 1.0])
        pipeline_count.is_fitted = True
        pipeline_count._init_sklearn_vectorizer()

        # Query with repeated term
        query = "users users users products"

        features_binary = pipeline_binary.transform_single(query)
        features_count = pipeline_count.transform_single(query)

        # Binary should treat all as 1, count should weight by frequency
        # After normalization they'll differ
        assert not np.array_equal(features_binary, features_count)
