"""
Feature parity validation between training and inference.

Ensures that features extracted during training match those extracted during
inference to prevent train-serve skew and silent performance degradation.
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np

logger = logging.getLogger(__name__)


class ParityValidator:
    """
    Validate feature parity between training and inference.

    Prevents train-serve skew by ensuring features are identical between
    training (notebook) and inference (production service).

    Example:
        validator = ParityValidator(tolerance=1e-6)

        result = validator.validate_parity(
            training_features=train_features_array,
            inference_featurizer=production_featurizer,
            tfidf_pipeline=trained_tfidf,
            sample_queries=sample_data,
            n_samples=100
        )

        if result['passed']:
            print(f"✅ Parity validation passed: {result['mismatch_rate']:.2f}% mismatch")
        else:
            print(f"❌ Parity validation failed: {result['mismatch_rate']:.2f}% mismatch")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, tolerance: Optional[float] = None):
        """
        Initialize parity validator.

        Args:
            config: Configuration dictionary with validation settings.
                   If provided, reads tolerance from config['validation']['parity_tolerance']
            tolerance: Maximum acceptable absolute difference between features.
                      If provided, overrides config value.
                      Default: 1e-6 for floating point precision
        """
        # Load from config if provided
        if config is not None:
            validation_config = config.get('validation', {})
            default_tolerance = validation_config.get('parity_tolerance', 1e-6)
        else:
            default_tolerance = 1e-6

        # tolerance parameter takes precedence over config
        self.tolerance = tolerance if tolerance is not None else default_tolerance
        self.logger = logging.getLogger(__name__)

    def validate_parity(
        self,
        training_features: np.ndarray,
        inference_featurizer: Any,
        tfidf_pipeline: Any,
        sample_queries: List[Dict[str, Any]],
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Compare training vs inference features on sample queries.

        Args:
            training_features: Features extracted during training (n_samples, n_features)
            inference_featurizer: Production FeatureExtractor instance
            tfidf_pipeline: Trained TfidfPipeline instance
            sample_queries: List of query data dictionaries with keys:
                           ['query', 'user', 'catalog', 'schema', 'hour', 'clientInfo']
            n_samples: Number of samples to test (default: 100)

        Returns:
            Dictionary with validation results:
            - passed: True if mismatch_rate < 0.5%
            - mismatch_rate: Percentage of samples with mismatches
            - max_difference: Maximum absolute difference found
            - samples_tested: Number of samples tested
            - mismatches: Number of mismatched samples
            - details: List of mismatch details (if any)
        """
        self.logger.info(f"Validating feature parity on {n_samples} samples...")

        mismatches = 0
        max_diff = 0.0
        mismatch_details = []

        # Limit to available samples
        n_samples = min(n_samples, len(sample_queries), training_features.shape[0])

        for i in range(n_samples):
            query_data = sample_queries[i]
            training_feat = training_features[i]

            try:
                # Extract base + historical features using inference featurizer
                base_historical_feat = inference_featurizer.extract(query_data)

                # Extract TF-IDF features
                tfidf_feat = tfidf_pipeline.transform_single(query_data['query'])

                # Combine: base + historical + tfidf
                inference_feat = np.concatenate([base_historical_feat, tfidf_feat])

                # Check for NaN values first
                has_nan_training = np.any(np.isnan(training_feat))
                has_nan_inference = np.any(np.isnan(inference_feat))

                if has_nan_training or has_nan_inference:
                    # NaN is always a mismatch
                    mismatches += 1
                    max_diff = max(max_diff, np.inf)

                    nan_indices = np.where(np.isnan(training_feat) | np.isnan(inference_feat))[0]

                    mismatch_details.append({
                        'sample_index': i,
                        'max_diff': float('inf'),
                        'num_mismatches': len(nan_indices),
                        'mismatch_indices': nan_indices[:10].tolist(),
                        'query_preview': query_data['query'][:100] if query_data.get('query') else None,
                        'error': 'NaN values detected'
                    })
                    continue

                # Compare
                diffs = np.abs(inference_feat - training_feat)
                sample_max_diff = np.max(diffs)
                max_diff = max(max_diff, sample_max_diff)

                # Check if any difference exceeds tolerance
                if np.any(diffs > self.tolerance):
                    mismatches += 1

                    # Find mismatched features
                    mismatch_indices = np.where(diffs > self.tolerance)[0]

                    mismatch_details.append({
                        'sample_index': i,
                        'max_diff': float(sample_max_diff),
                        'num_mismatches': len(mismatch_indices),
                        'mismatch_indices': mismatch_indices[:10].tolist(),  # First 10
                        'query_preview': query_data['query'][:100] if query_data.get('query') else None
                    })

            except Exception as e:
                self.logger.error(f"Failed to validate sample {i}: {e}")
                mismatches += 1
                mismatch_details.append({
                    'sample_index': i,
                    'error': str(e)
                })

        mismatch_rate = (mismatches / n_samples) * 100
        passed = mismatch_rate < 0.5  # Success threshold: <0.5% mismatch

        result = {
            'passed': passed,
            'mismatch_rate': mismatch_rate,
            'max_difference': max_diff,
            'samples_tested': n_samples,
            'mismatches': mismatches,
            'tolerance': self.tolerance
        }

        # Only include details if there are mismatches
        if mismatch_details:
            result['details'] = mismatch_details[:20]  # First 20 mismatches

        # Log results
        if passed:
            self.logger.info(f"✅ Parity validation PASSED: {mismatch_rate:.2f}% mismatch rate")
            self.logger.info(f"   Max difference: {max_diff:.9f}")
        else:
            self.logger.warning(f"❌ Parity validation FAILED: {mismatch_rate:.2f}% mismatch rate")
            self.logger.warning(f"   Mismatches: {mismatches}/{n_samples}")
            self.logger.warning(f"   Max difference: {max_diff:.9f}")

        return result

    def validate_parity_simple(
        self,
        training_features: np.ndarray,
        inference_features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Simple parity validation without feature extraction.

        Useful for testing when you already have both feature arrays.

        Args:
            training_features: Features from training (n_samples, n_features)
            inference_features: Features from inference (n_samples, n_features)

        Returns:
            Dictionary with validation results (same as validate_parity)
        """
        if training_features.shape != inference_features.shape:
            raise ValueError(f"Shape mismatch: training {training_features.shape} "
                           f"vs inference {inference_features.shape}")

        n_samples = training_features.shape[0]
        mismatches = 0
        max_diff = 0.0
        mismatch_details = []

        for i in range(n_samples):
            training_feat = training_features[i]
            inference_feat = inference_features[i]

            # Check for NaN values first
            has_nan_training = np.any(np.isnan(training_feat))
            has_nan_inference = np.any(np.isnan(inference_feat))

            if has_nan_training or has_nan_inference:
                # NaN is always a mismatch
                mismatches += 1
                max_diff = max(max_diff, np.inf)

                nan_indices = np.where(np.isnan(training_feat) | np.isnan(inference_feat))[0]

                mismatch_details.append({
                    'sample_index': i,
                    'max_diff': float('inf'),
                    'num_mismatches': len(nan_indices),
                    'mismatch_indices': nan_indices[:10].tolist(),
                    'error': 'NaN values detected'
                })
                continue

            diffs = np.abs(inference_feat - training_feat)
            sample_max_diff = np.max(diffs)
            max_diff = max(max_diff, sample_max_diff)

            if np.any(diffs > self.tolerance):
                mismatches += 1
                mismatch_indices = np.where(diffs > self.tolerance)[0]

                mismatch_details.append({
                    'sample_index': i,
                    'max_diff': float(sample_max_diff),
                    'num_mismatches': len(mismatch_indices),
                    'mismatch_indices': mismatch_indices[:10].tolist()
                })

        mismatch_rate = (mismatches / n_samples) * 100
        passed = mismatch_rate < 0.5

        result = {
            'passed': passed,
            'mismatch_rate': mismatch_rate,
            'max_difference': max_diff,
            'samples_tested': n_samples,
            'mismatches': mismatches,
            'tolerance': self.tolerance
        }

        if mismatch_details:
            result['details'] = mismatch_details[:20]

        return result

    def generate_report(self, validation_result: Dict[str, Any]) -> str:
        """
        Generate detailed parity validation report.

        Args:
            validation_result: Result dictionary from validate_parity()

        Returns:
            Formatted report string
        """
        status = "✅ PASSED" if validation_result['passed'] else "❌ FAILED"

        report = f"""
{'='*70}
FEATURE PARITY VALIDATION REPORT
{'='*70}

Status: {status}

Summary:
  Samples Tested:  {validation_result['samples_tested']}
  Mismatches:      {validation_result['mismatches']}
  Mismatch Rate:   {validation_result['mismatch_rate']:.2f}%
  Max Difference:  {validation_result['max_difference']:.9f}
  Tolerance:       {validation_result['tolerance']:.9f}

"""

        if 'details' in validation_result and validation_result['details']:
            report += "Mismatch Details (first 10):\n"
            for detail in validation_result['details'][:10]:
                report += f"\n  Sample {detail['sample_index']}:\n"
                if 'error' in detail:
                    report += f"    Error: {detail['error']}\n"
                else:
                    report += f"    Max diff: {detail['max_diff']:.9f}\n"
                    report += f"    Num mismatches: {detail['num_mismatches']}\n"
                    if 'mismatch_indices' in detail:
                        indices = detail['mismatch_indices']
                        report += f"    Feature indices: {indices}\n"

        report += "\n" + "="*70 + "\n"

        return report

    def validate_base_features_only(
        self,
        training_base: np.ndarray,
        inference_featurizer: Any,
        sample_queries: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate only base features (78 + 17) without TF-IDF.

        Useful for debugging feature extraction issues.

        Args:
            training_base: Base features from training (n_samples, 95)
            inference_featurizer: Production FeatureExtractor
            sample_queries: Query data dictionaries

        Returns:
            Validation result dictionary
        """
        n_samples = min(len(sample_queries), training_base.shape[0])
        mismatches = 0
        max_diff = 0.0

        for i in range(n_samples):
            training_feat = training_base[i]
            inference_feat = inference_featurizer.extract(sample_queries[i])

            # Check for NaN values
            has_nan = np.any(np.isnan(training_feat)) or np.any(np.isnan(inference_feat))
            if has_nan:
                mismatches += 1
                max_diff = max(max_diff, np.inf)
                continue

            diffs = np.abs(inference_feat - training_feat)
            max_diff = max(max_diff, np.max(diffs))

            if np.any(diffs > self.tolerance):
                mismatches += 1

        mismatch_rate = (mismatches / n_samples) * 100

        return {
            'passed': mismatch_rate < 0.5,
            'mismatch_rate': mismatch_rate,
            'max_difference': max_diff,
            'samples_tested': n_samples,
            'mismatches': mismatches,
            'feature_type': 'base_only (78 + 17)'
        }

    def validate_tfidf_features_only(
        self,
        training_tfidf: np.ndarray,
        tfidf_pipeline: Any,
        queries: List[str]
    ) -> Dict[str, Any]:
        """
        Validate only TF-IDF features without base features.

        Useful for debugging TF-IDF extraction issues.

        Args:
            training_tfidf: TF-IDF features from training (n_samples, vocab_size)
            tfidf_pipeline: Trained TfidfPipeline
            queries: List of SQL query strings

        Returns:
            Validation result dictionary
        """
        n_samples = min(len(queries), training_tfidf.shape[0])
        mismatches = 0
        max_diff = 0.0

        for i in range(n_samples):
            training_feat = training_tfidf[i]
            inference_feat = tfidf_pipeline.transform_single(queries[i])

            # Check for NaN values
            has_nan = np.any(np.isnan(training_feat)) or np.any(np.isnan(inference_feat))
            if has_nan:
                mismatches += 1
                max_diff = max(max_diff, np.inf)
                continue

            diffs = np.abs(inference_feat - training_feat)
            max_diff = max(max_diff, np.max(diffs))

            if np.any(diffs > self.tolerance):
                mismatches += 1

        mismatch_rate = (mismatches / n_samples) * 100

        return {
            'passed': mismatch_rate < 0.5,
            'mismatch_rate': mismatch_rate,
            'max_difference': max_diff,
            'samples_tested': n_samples,
            'mismatches': mismatches,
            'feature_type': f'tfidf_only ({training_tfidf.shape[1]} features)'
        }
