"""
ONNX model export and validation.

Converts XGBoost models to ONNX format and validates prediction parity.
"""

import logging
import numpy as np
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

try:
    import xgboost as xgb
    import onnx
    import onnxruntime as ort
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    onnx = None
    ort = None

logger = logging.getLogger(__name__)


@dataclass
class ONNXValidationResult:
    """ONNX validation results."""
    passed: bool
    max_difference: float
    mean_difference: float
    mismatch_rate: float
    samples_tested: int
    tolerance: float


class ONNXValidator:
    """
    Convert XGBoost to ONNX and validate prediction parity.

    Ensures that ONNX model produces identical predictions to XGBoost
    to prevent inference-time degradation.

    Example:
        validator = ONNXValidator(tolerance=1e-5)
        validator.export_to_onnx(xgb_model, 'model.onnx', input_size=1095)
        result = validator.validate_onnx(xgb_model, 'model.onnx', X_test)
        if result.passed:
            print("ONNX export validated")
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None, tolerance: Optional[float] = None):
        """
        Initialize ONNX validator.

        Args:
            config: Configuration dictionary with validation settings.
                   If provided, reads tolerance and opset from config['validation']
            tolerance: Maximum acceptable absolute difference between
                      XGBoost and ONNX predictions. If provided, overrides config value.
                      Default: 1e-5
        """
        if not ONNX_AVAILABLE:
            raise ImportError(
                "ONNX dependencies required: onnx, onnxruntime, skl2onnx. "
                "Install with: pip install onnx onnxruntime skl2onnx"
            )

        # Load from config if provided
        if config is not None:
            validation_config = config.get('validation', {})
            default_tolerance = validation_config.get('onnx_tolerance', 1e-5)
            self.default_opset_version = validation_config.get('onnx_opset_version', 12)
        else:
            default_tolerance = 1e-5
            self.default_opset_version = 12

        # tolerance parameter takes precedence over config
        self.tolerance = tolerance if tolerance is not None else default_tolerance
        self.logger = logging.getLogger(__name__)

    def export_to_onnx(
        self,
        model: xgb.XGBClassifier,
        output_path: str,
        input_size: int,
        opset_version: Optional[int] = None
    ):
        """
        Export XGBoost model to ONNX format.

        Args:
            model: Trained XGBClassifier
            output_path: Path to save ONNX model
            input_size: Number of input features
            opset_version: ONNX opset version. If None, uses value from config (default: 12)
        """
        # Use default opset from config if not provided
        if opset_version is None:
            opset_version = self.default_opset_version

        self.logger.info(f"Exporting XGBoost model to ONNX...")
        self.logger.info(f"  Output path: {output_path}")
        self.logger.info(f"  Input features: {input_size}")
        self.logger.info(f"  Opset version: {opset_version}")

        # Define initial types for ONNX conversion
        initial_type = [('float_input', FloatTensorType([None, input_size]))]

        # Convert to ONNX
        try:
            onnx_model = convert_sklearn(
                model,
                initial_types=initial_type,
                target_opset=opset_version
            )

            # Save ONNX model
            with open(output_path, 'wb') as f:
                f.write(onnx_model.SerializeToString())

            self.logger.info(f"✅ ONNX model exported successfully")

            # Verify ONNX model is valid
            onnx.checker.check_model(onnx_model)
            self.logger.info(f"✅ ONNX model validation passed")

        except Exception as e:
            self.logger.error(f"❌ ONNX export failed: {e}")
            raise

    def validate_onnx(
        self,
        xgb_model: xgb.XGBClassifier,
        onnx_path: str,
        X_test: np.ndarray,
        n_samples: int = 1000
    ) -> ONNXValidationResult:
        """
        Validate ONNX model predictions match XGBoost.

        Args:
            xgb_model: Original XGBoost model
            onnx_path: Path to ONNX model
            X_test: Test features for validation
            n_samples: Number of samples to test (default: 1000)

        Returns:
            ONNXValidationResult with validation metrics
        """
        self.logger.info(f"\nValidating ONNX predictions...")
        self.logger.info(f"  Tolerance: {self.tolerance}")
        self.logger.info(f"  Samples: {n_samples}")

        # Limit samples
        n_samples = min(n_samples, X_test.shape[0])
        X_sample = X_test[:n_samples]

        # Get XGBoost predictions
        xgb_proba = xgb_model.predict_proba(X_sample)[:, 1]

        # Get ONNX predictions
        onnx_proba = self._predict_onnx(onnx_path, X_sample)

        # Compare predictions
        differences = np.abs(xgb_proba - onnx_proba)
        max_diff = np.max(differences)
        mean_diff = np.mean(differences)

        # Count mismatches
        mismatches = np.sum(differences > self.tolerance)
        mismatch_rate = (mismatches / n_samples) * 100

        # Validation passes if mismatch rate < 0.5%
        passed = mismatch_rate < 0.5

        result = ONNXValidationResult(
            passed=passed,
            max_difference=float(max_diff),
            mean_difference=float(mean_diff),
            mismatch_rate=float(mismatch_rate),
            samples_tested=n_samples,
            tolerance=self.tolerance
        )

        # Log results
        if passed:
            self.logger.info(f"✅ ONNX validation PASSED")
            self.logger.info(f"   Max difference: {max_diff:.9f}")
            self.logger.info(f"   Mean difference: {mean_diff:.9f}")
            self.logger.info(f"   Mismatch rate: {mismatch_rate:.2f}%")
        else:
            self.logger.warning(f"❌ ONNX validation FAILED")
            self.logger.warning(f"   Max difference: {max_diff:.9f}")
            self.logger.warning(f"   Mean difference: {mean_diff:.9f}")
            self.logger.warning(f"   Mismatch rate: {mismatch_rate:.2f}%")
            self.logger.warning(f"   Mismatches: {mismatches}/{n_samples}")

        return result

    def _predict_onnx(self, onnx_path: str, X: np.ndarray) -> np.ndarray:
        """
        Get predictions from ONNX model.

        Args:
            onnx_path: Path to ONNX model
            X: Input features

        Returns:
            Predicted probabilities (n_samples,)
        """
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)

        # Get input name
        input_name = session.get_inputs()[0].name

        # Get predictions
        # ONNX returns probabilities for both classes
        onnx_output = session.run(None, {input_name: X.astype(np.float32)})

        # Extract probability for positive class (heavy queries)
        # Output format: [probabilities array, label array]
        probabilities = onnx_output[1]  # Second output is probabilities

        # Handle different output formats
        if len(probabilities.shape) == 2:
            # Format: (n_samples, 2) - take positive class
            return probabilities[:, 1]
        else:
            # Format: (n_samples,) - already positive class
            return probabilities

    def compare_predictions_detailed(
        self,
        xgb_model: xgb.XGBClassifier,
        onnx_path: str,
        X_test: np.ndarray,
        threshold: float,
        n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Detailed comparison of XGBoost vs ONNX predictions.

        Args:
            xgb_model: Original XGBoost model
            onnx_path: Path to ONNX model
            X_test: Test features
            threshold: Classification threshold
            n_samples: Number of samples to compare

        Returns:
            Dictionary with detailed comparison metrics
        """
        n_samples = min(n_samples, X_test.shape[0])
        X_sample = X_test[:n_samples]

        # Get predictions
        xgb_proba = xgb_model.predict_proba(X_sample)[:, 1]
        onnx_proba = self._predict_onnx(onnx_path, X_sample)

        xgb_pred = (xgb_proba >= threshold).astype(int)
        onnx_pred = (onnx_proba >= threshold).astype(int)

        # Probability differences
        prob_diffs = np.abs(xgb_proba - onnx_proba)

        # Prediction agreement
        pred_agreement = np.sum(xgb_pred == onnx_pred) / n_samples

        # Classification differences
        classification_mismatches = np.sum(xgb_pred != onnx_pred)

        return {
            'samples_compared': n_samples,
            'threshold': threshold,
            'probability_differences': {
                'max': float(np.max(prob_diffs)),
                'mean': float(np.mean(prob_diffs)),
                'std': float(np.std(prob_diffs)),
                'percentile_95': float(np.percentile(prob_diffs, 95)),
                'percentile_99': float(np.percentile(prob_diffs, 99))
            },
            'classification_agreement': {
                'agreement_rate': float(pred_agreement),
                'mismatches': int(classification_mismatches),
                'mismatch_rate': float(classification_mismatches / n_samples * 100)
            }
        }

    def generate_validation_report(self, result: ONNXValidationResult) -> str:
        """
        Generate human-readable validation report.

        Args:
            result: ONNX validation result

        Returns:
            Formatted report string
        """
        status = "✅ PASSED" if result.passed else "❌ FAILED"

        report = f"""
{'='*70}
ONNX VALIDATION REPORT
{'='*70}

Status: {status}

Summary:
  Samples Tested:  {result.samples_tested}
  Mismatch Rate:   {result.mismatch_rate:.2f}%
  Max Difference:  {result.max_difference:.9f}
  Mean Difference: {result.mean_difference:.9f}
  Tolerance:       {result.tolerance:.9f}

"""

        if not result.passed:
            report += """
⚠️  WARNING: ONNX predictions differ significantly from XGBoost!
   This may cause performance degradation in production.
   Consider:
   - Adjusting ONNX opset version
   - Reviewing feature preprocessing
   - Checking ONNX runtime version compatibility
"""

        report += "="*70 + "\n"

        return report

    def get_onnx_metadata(self, onnx_path: str) -> Dict[str, Any]:
        """
        Extract metadata from ONNX model.

        Args:
            onnx_path: Path to ONNX model

        Returns:
            Dictionary with model metadata
        """
        model = onnx.load(onnx_path)

        # Get input/output info
        inputs = [
            {
                'name': inp.name,
                'type': inp.type.WhichOneof('value'),
                'shape': [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
            }
            for inp in model.graph.input
        ]

        outputs = [
            {
                'name': out.name,
                'type': out.type.WhichOneof('value'),
                'shape': [dim.dim_value for dim in out.type.tensor_type.shape.dim]
            }
            for out in model.graph.output
        ]

        metadata = {
            'opset_version': model.opset_import[0].version if model.opset_import else None,
            'producer_name': model.producer_name,
            'graph_name': model.graph.name,
            'inputs': inputs,
            'outputs': outputs,
            'num_nodes': len(model.graph.node)
        }

        return metadata
