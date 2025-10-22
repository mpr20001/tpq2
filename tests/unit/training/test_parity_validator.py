"""
Unit tests for ParityValidator.

Tests feature parity validation between training and inference.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock


@pytest.fixture
def sample_training_features():
    """Sample training features (100 samples x 1095 features)."""
    np.random.seed(42)
    return np.random.rand(100, 1095).astype(np.float32)


@pytest.fixture
def sample_queries():
    """Sample query data dictionaries."""
    return [
        {
            'query': f"SELECT * FROM table{i} WHERE id = {i}",
            'user': f'user{i}@example.com',
            'catalog': 'hive',
            'schema': 'production',
            'hour': i % 24,
            'clientInfo': 'jdbc'
        }
        for i in range(100)
    ]


@pytest.fixture
def mock_feature_extractor():
    """Mock FeatureExtractor for inference."""
    mock_extractor = Mock()

    # Mock extract() to return 95 base + historical features
    def mock_extract(query_data):
        np.random.seed(hash(query_data['query']) % 2**32)
        return np.random.rand(95).astype(np.float32)

    mock_extractor.extract.side_effect = mock_extract
    return mock_extractor


@pytest.fixture
def mock_tfidf_pipeline():
    """Mock TfidfPipeline for TF-IDF features."""
    mock_pipeline = Mock()

    # Mock transform_single() to return 1000 TF-IDF features
    def mock_transform(query):
        np.random.seed(hash(query) % 2**32)
        return np.random.rand(1000).astype(np.float32)

    mock_pipeline.transform_single.side_effect = mock_transform
    mock_pipeline.vocab_size = 1000
    return mock_pipeline


class TestParityValidatorInitialization:
    """Test ParityValidator initialization."""

    def test_default_initialization(self):
        """Test initialization with default tolerance."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator()
        assert validator.tolerance == 1e-6

    def test_custom_tolerance(self):
        """Test initialization with custom tolerance."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-3)
        assert validator.tolerance == 1e-3


class TestValidateParityMain:
    """Test main validate_parity() method."""

    def test_parity_validation_perfect_match(self, sample_queries):
        """Test validation when training and inference features match perfectly."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)
        n_samples = 10

        # Create mock extractors that return identical features
        np.random.seed(42)
        training_features = np.random.rand(n_samples, 1095).astype(np.float32)

        mock_extractor = Mock()
        mock_tfidf = Mock()

        # Return matching features
        for i in range(n_samples):
            base_hist = training_features[i][:95]
            tfidf = training_features[i][95:]

            # Set up mock to return exact features
            def make_extract(feat):
                return lambda q: feat.copy()

            def make_tfidf(feat):
                return lambda q: feat.copy()

        # Simpler approach: mock to return exact same features
        mock_extractor.extract.side_effect = [
            training_features[i][:95] for i in range(n_samples)
        ]
        mock_tfidf.transform_single.side_effect = [
            training_features[i][95:] for i in range(n_samples)
        ]

        result = validator.validate_parity(
            training_features=training_features,
            inference_featurizer=mock_extractor,
            tfidf_pipeline=mock_tfidf,
            sample_queries=sample_queries[:n_samples],
            n_samples=n_samples
        )

        assert result['passed'] is True
        assert result['mismatch_rate'] == 0.0
        assert result['mismatches'] == 0
        assert result['samples_tested'] == n_samples
        assert result['max_difference'] < validator.tolerance

    def test_parity_validation_with_mismatches(self, sample_queries):
        """Test validation when features have small mismatches."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)
        n_samples = 10

        # Create training features
        np.random.seed(42)
        training_features = np.random.rand(n_samples, 1095).astype(np.float32)

        # Create inference features with small differences
        inference_features = training_features.copy()
        inference_features[0, 0] += 1e-5  # Small difference > tolerance
        inference_features[1, 10] += 1e-4  # Another mismatch

        mock_extractor = Mock()
        mock_tfidf = Mock()

        # Return inference features (with mismatches)
        mock_extractor.extract.side_effect = [
            inference_features[i][:95] for i in range(n_samples)
        ]
        mock_tfidf.transform_single.side_effect = [
            inference_features[i][95:] for i in range(n_samples)
        ]

        result = validator.validate_parity(
            training_features=training_features,
            inference_featurizer=mock_extractor,
            tfidf_pipeline=mock_tfidf,
            sample_queries=sample_queries[:n_samples],
            n_samples=n_samples
        )

        # 2 mismatches out of 10 = 20% mismatch rate (fails <0.5% threshold)
        assert result['passed'] is False
        assert result['mismatch_rate'] == 20.0
        assert result['mismatches'] == 2
        assert result['samples_tested'] == n_samples
        assert result['max_difference'] >= 1e-4

    def test_parity_validation_with_extraction_errors(self, sample_queries):
        """Test validation handles extraction errors gracefully."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)
        n_samples = 5

        # Create training features
        np.random.seed(42)
        training_features = np.random.rand(n_samples, 1095).astype(np.float32)

        mock_extractor = Mock()
        mock_tfidf = Mock()

        # Make some extractions fail
        def failing_extract(query_data):
            if query_data['query'].startswith("SELECT * FROM table0"):
                raise ValueError("Extraction failed")
            return np.random.rand(95).astype(np.float32)

        mock_extractor.extract.side_effect = failing_extract
        mock_tfidf.transform_single.return_value = np.random.rand(1000).astype(np.float32)

        result = validator.validate_parity(
            training_features=training_features,
            inference_featurizer=mock_extractor,
            tfidf_pipeline=mock_tfidf,
            sample_queries=sample_queries[:n_samples],
            n_samples=n_samples
        )

        # Should count extraction errors as mismatches
        assert result['mismatches'] >= 1
        assert 'details' in result
        assert any('error' in detail for detail in result['details'])

    def test_parity_validation_limits_samples(self, sample_queries):
        """Test that validation limits to min of available samples."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator()

        # Request 100 samples but only provide 5
        training_features = np.random.rand(5, 1095).astype(np.float32)

        mock_extractor = Mock()
        mock_tfidf = Mock()
        mock_extractor.extract.return_value = np.random.rand(95).astype(np.float32)
        mock_tfidf.transform_single.return_value = np.random.rand(1000).astype(np.float32)

        result = validator.validate_parity(
            training_features=training_features,
            inference_featurizer=mock_extractor,
            tfidf_pipeline=mock_tfidf,
            sample_queries=sample_queries[:5],
            n_samples=100  # Request more than available
        )

        # Should only test 5 samples
        assert result['samples_tested'] == 5


class TestValidateParitySimple:
    """Test validate_parity_simple() method."""

    def test_simple_validation_perfect_match(self):
        """Test simple validation with identical features."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)

        training_features = np.random.rand(50, 1095).astype(np.float32)
        inference_features = training_features.copy()

        result = validator.validate_parity_simple(training_features, inference_features)

        assert result['passed'] is True
        assert result['mismatch_rate'] == 0.0
        assert result['mismatches'] == 0
        assert result['samples_tested'] == 50

    def test_simple_validation_with_mismatches(self):
        """Test simple validation with some differences."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)

        training_features = np.random.rand(10, 1095).astype(np.float32)
        inference_features = training_features.copy()

        # Add mismatches to 2 samples (20%)
        inference_features[0, 0] += 1e-3
        inference_features[1, 10] += 1e-4

        result = validator.validate_parity_simple(training_features, inference_features)

        assert result['passed'] is False
        assert result['mismatch_rate'] == 20.0
        assert result['mismatches'] == 2

    def test_simple_validation_shape_mismatch(self):
        """Test that shape mismatch raises error."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator()

        training_features = np.random.rand(10, 1095).astype(np.float32)
        inference_features = np.random.rand(10, 500).astype(np.float32)  # Wrong shape

        with pytest.raises(ValueError, match="Shape mismatch"):
            validator.validate_parity_simple(training_features, inference_features)


class TestValidateBaseFeatures:
    """Test validate_base_features_only() method."""

    def test_validate_base_features(self, sample_queries):
        """Test validation of base features only."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)
        n_samples = 10

        # Create base features (78 + 17 = 95)
        training_base = np.random.rand(n_samples, 95).astype(np.float32)

        mock_extractor = Mock()
        # Return matching base features
        mock_extractor.extract.side_effect = [
            training_base[i] for i in range(n_samples)
        ]

        result = validator.validate_base_features_only(
            training_base=training_base,
            inference_featurizer=mock_extractor,
            sample_queries=sample_queries[:n_samples]
        )

        assert result['passed'] is True
        assert result['mismatch_rate'] == 0.0
        assert result['samples_tested'] == n_samples
        assert result['feature_type'] == 'base_only (78 + 17)'


class TestValidateTfidfFeatures:
    """Test validate_tfidf_features_only() method."""

    def test_validate_tfidf_features(self):
        """Test validation of TF-IDF features only."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)
        n_samples = 10

        queries = [f"SELECT * FROM table{i}" for i in range(n_samples)]
        training_tfidf = np.random.rand(n_samples, 1000).astype(np.float32)

        mock_tfidf = Mock()
        # Return matching TF-IDF features
        mock_tfidf.transform_single.side_effect = [
            training_tfidf[i] for i in range(n_samples)
        ]

        result = validator.validate_tfidf_features_only(
            training_tfidf=training_tfidf,
            tfidf_pipeline=mock_tfidf,
            queries=queries
        )

        assert result['passed'] is True
        assert result['mismatch_rate'] == 0.0
        assert result['samples_tested'] == n_samples
        assert '1000 features' in result['feature_type']


class TestGenerateReport:
    """Test report generation."""

    def test_generate_report_passed(self):
        """Test report generation for passed validation."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)

        validation_result = {
            'passed': True,
            'mismatch_rate': 0.0,
            'max_difference': 1e-8,
            'samples_tested': 100,
            'mismatches': 0,
            'tolerance': 1e-6
        }

        report = validator.generate_report(validation_result)

        assert isinstance(report, str)
        assert "✅ PASSED" in report
        assert "Samples Tested:  100" in report
        assert "Mismatches:      0" in report
        assert "0.00%" in report  # Mismatch rate

    def test_generate_report_failed(self):
        """Test report generation for failed validation."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)

        validation_result = {
            'passed': False,
            'mismatch_rate': 5.0,
            'max_difference': 1e-3,
            'samples_tested': 100,
            'mismatches': 5,
            'tolerance': 1e-6
        }

        report = validator.generate_report(validation_result)

        assert isinstance(report, str)
        assert "❌ FAILED" in report
        assert "Samples Tested:  100" in report
        assert "Mismatches:      5" in report
        assert "5.00%" in report  # Mismatch rate

    def test_generate_report_with_details(self):
        """Test report generation includes mismatch details."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)

        validation_result = {
            'passed': False,
            'mismatch_rate': 2.0,
            'max_difference': 1e-3,
            'samples_tested': 100,
            'mismatches': 2,
            'tolerance': 1e-6,
            'details': [
                {
                    'sample_index': 0,
                    'max_diff': 1e-3,
                    'num_mismatches': 5,
                    'mismatch_indices': [0, 1, 2, 3, 4]
                },
                {
                    'sample_index': 5,
                    'error': 'Feature extraction failed'
                }
            ]
        }

        report = validator.generate_report(validation_result)

        assert "Mismatch Details" in report
        assert "Sample 0" in report
        assert "Sample 5" in report
        assert "Error: Feature extraction failed" in report

    def test_generate_report_limits_details(self):
        """Test that report limits mismatch details to first 10."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator()

        # Create 50 mismatch details
        details = [
            {
                'sample_index': i,
                'max_diff': 1e-3,
                'num_mismatches': 5,
                'mismatch_indices': [0, 1, 2]
            }
            for i in range(50)
        ]

        validation_result = {
            'passed': False,
            'mismatch_rate': 50.0,
            'max_difference': 1e-3,
            'samples_tested': 100,
            'mismatches': 50,
            'tolerance': 1e-6,
            'details': details
        }

        report = validator.generate_report(validation_result)

        # Should show "first 10" in report
        assert "first 10" in report.lower()


class TestParityValidatorEdgeCases:
    """Test edge cases and error handling."""

    def test_tolerance_boundary(self):
        """Test that differences exactly at tolerance are handled correctly."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)

        training_features = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        inference_features = np.array([[1.0000001, 2.0, 3.0]], dtype=np.float32)

        # Difference is 1e-7, which is less than tolerance 1e-6
        result = validator.validate_parity_simple(training_features, inference_features)

        assert result['passed'] is True

        # Now test with difference exactly at tolerance
        inference_features = np.array([[1.000001, 2.0, 3.0]], dtype=np.float32)
        result = validator.validate_parity_simple(training_features, inference_features)

        # Difference is exactly 1e-6, should pass (using >, not >=)
        assert result['passed'] is True

        # Test with difference just above tolerance
        inference_features = np.array([[1.0000011, 2.0, 3.0]], dtype=np.float32)
        result = validator.validate_parity_simple(training_features, inference_features)

        # Difference is >1e-6, should fail
        assert result['passed'] is False

    def test_all_zeros_features(self):
        """Test validation with all-zero features."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)

        training_features = np.zeros((10, 100), dtype=np.float32)
        inference_features = np.zeros((10, 100), dtype=np.float32)

        result = validator.validate_parity_simple(training_features, inference_features)

        assert result['passed'] is True
        assert result['max_difference'] == 0.0

    def test_nan_handling(self):
        """Test that NaN values are detected as mismatches."""
        from query_predictor.training.parity_validator import ParityValidator

        validator = ParityValidator(tolerance=1e-6)

        training_features = np.random.rand(10, 100).astype(np.float32)
        inference_features = training_features.copy()
        inference_features[0, 0] = np.nan  # Introduce NaN

        result = validator.validate_parity_simple(training_features, inference_features)

        # NaN should cause mismatch
        assert result['passed'] is False
        assert result['mismatches'] >= 1

    def test_large_tolerance(self):
        """Test validation with large tolerance."""
        from query_predictor.training.parity_validator import ParityValidator

        # Very large tolerance
        validator = ParityValidator(tolerance=1.0)

        training_features = np.random.rand(10, 100).astype(np.float32)
        inference_features = training_features + 0.5  # Add large difference

        result = validator.validate_parity_simple(training_features, inference_features)

        # With tolerance=1.0, differences of 0.5 should pass
        assert result['passed'] is True
