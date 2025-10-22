"""
XGBoost model trainer with threshold optimization.

Handles model training, cross-validation, threshold optimization, and evaluation.
"""

import logging
import numpy as np
import pickle
from typing import Dict, Any, Tuple, Optional, List
from dataclasses import dataclass

try:
    import xgboost as xgb
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import (
        precision_recall_curve,
        roc_auc_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    xgb = None
    StratifiedKFold = None

logger = logging.getLogger(__name__)


@dataclass
class ThresholdMetrics:
    """Metrics for a specific classification threshold."""
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    fnr: float
    tpr: float
    tn: int
    fp: int
    fn: int
    tp: int
    total_cost: float


@dataclass
class CVResults:
    """Cross-validation results."""
    fold_scores: List[Dict[str, float]]
    mean_metrics: Dict[str, float]
    std_metrics: Dict[str, float]
    best_iteration: int


@dataclass
class TrainingResults:
    """Complete training results."""
    model: Any
    cv_results: CVResults
    optimal_threshold: float
    threshold_metrics: ThresholdMetrics
    test_metrics: Dict[str, float]
    feature_importance: Optional[Dict[str, float]] = None


class ModelTrainer:
    """
    XGBoost model trainer with threshold optimization.

    Features:
    - 5-fold stratified cross-validation
    - Early stopping to prevent overfitting
    - Custom threshold optimization (100:1 FN:FP cost ratio)
    - Comprehensive evaluation metrics
    - Feature importance extraction

    Example:
        trainer = ModelTrainer(config)
        results = trainer.train(X_train, y_train, X_val, y_val)
        test_metrics = trainer.evaluate(results.model, X_test, y_test, results.optimal_threshold)
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model trainer.

        Args:
            config: Training configuration with keys:
                - model: XGBoost hyperparameters
                - prd_requirements: Target metrics
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn and xgboost are required for ModelTrainer")

        self.config = config
        self.model_config = config.get('model', {})

        # XGBoost hyperparameters
        self.n_estimators = self.model_config.get('n_estimators', 100)
        self.max_depth = self.model_config.get('max_depth', 6)
        self.learning_rate = self.model_config.get('learning_rate', 0.1)
        self.subsample = self.model_config.get('subsample', 0.8)
        self.colsample_bytree = self.model_config.get('colsample_bytree', 0.8)

        # Cross-validation
        self.cv_folds = self.model_config.get('cv_folds', 5)

        # Threshold optimization
        self.cost_fn = self.model_config.get('cost_fn', 100.0)  # Cost of false negative
        self.cost_fp = self.model_config.get('cost_fp', 1.0)    # Cost of false positive

        self.logger = logging.getLogger(__name__)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> TrainingResults:
        """
        Train XGBoost model with cross-validation and threshold optimization.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training labels (n_samples,)
            X_val: Validation features (n_samples, n_features)
            y_val: Validation labels (n_samples,)
            feature_names: Optional feature names for importance analysis

        Returns:
            TrainingResults with trained model, CV results, and optimal threshold
        """
        self.logger.info("Starting model training...")
        self.logger.info(f"  Train samples: {X_train.shape[0]:,}")
        self.logger.info(f"  Val samples: {X_val.shape[0]:,}")
        self.logger.info(f"  Features: {X_train.shape[1]:,}")
        self.logger.info(f"  Class balance (train): {np.mean(y_train):.2%} heavy")

        # Cross-validation
        self.logger.info(f"\nRunning {self.cv_folds}-fold cross-validation...")
        cv_results = self._cross_validate(X_train, y_train)

        self.logger.info(f"\nCV Results:")
        for metric, value in cv_results.mean_metrics.items():
            std = cv_results.std_metrics[metric]
            self.logger.info(f"  {metric}: {value:.4f} ± {std:.4f}")

        # Train final model on full training set
        self.logger.info(f"\nTraining final model...")
        model = self._train_model(
            X_train, y_train,
            X_val, y_val,
            num_boost_round=cv_results.best_iteration
        )

        # Get predictions on validation set
        y_val_proba = model.predict_proba(X_val)[:, 1]

        # Optimize threshold
        self.logger.info(f"\nOptimizing classification threshold...")
        self.logger.info(f"  Cost FN: {self.cost_fn}")
        self.logger.info(f"  Cost FP: {self.cost_fp}")

        optimal_threshold, threshold_metrics = self._optimize_threshold(
            y_val, y_val_proba
        )

        self.logger.info(f"\nOptimal threshold: {optimal_threshold:.4f}")
        self.logger.info(f"  Precision: {threshold_metrics.precision:.4f}")
        self.logger.info(f"  Recall: {threshold_metrics.recall:.4f}")
        self.logger.info(f"  F1: {threshold_metrics.f1:.4f}")
        self.logger.info(f"  FNR: {threshold_metrics.fnr:.4f}")

        # Extract feature importance
        feature_importance = None
        if feature_names:
            feature_importance = self._get_feature_importance(model, feature_names)

        # Evaluate on validation set with optimal threshold
        test_metrics = self.evaluate(model, X_val, y_val, optimal_threshold)

        return TrainingResults(
            model=model,
            cv_results=cv_results,
            optimal_threshold=optimal_threshold,
            threshold_metrics=threshold_metrics,
            test_metrics=test_metrics,
            feature_importance=feature_importance
        )

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> CVResults:
        """
        Perform stratified k-fold cross-validation.

        Args:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,)

        Returns:
            CVResults with fold scores and aggregated metrics
        """
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        fold_scores = []
        best_iterations = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            self.logger.info(f"  Fold {fold_idx + 1}/{self.cv_folds}...")

            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # Train model with early stopping
            model = self._train_model(
                X_fold_train, y_fold_train,
                X_fold_val, y_fold_val,
                num_boost_round=self.n_estimators
            )

            # Get best iteration
            best_iteration = model.best_iteration if hasattr(model, 'best_iteration') else self.n_estimators
            best_iterations.append(best_iteration)

            # Evaluate
            y_pred_proba = model.predict_proba(X_fold_val)[:, 1]

            # Calculate metrics at default threshold (0.5)
            y_pred = (y_pred_proba >= 0.5).astype(int)

            fold_metrics = {
                'roc_auc': roc_auc_score(y_fold_val, y_pred_proba),
                'precision': precision_score(y_fold_val, y_pred, zero_division=0),
                'recall': recall_score(y_fold_val, y_pred, zero_division=0),
                'f1': f1_score(y_fold_val, y_pred, zero_division=0)
            }

            fold_scores.append(fold_metrics)

        # Aggregate metrics
        mean_metrics = {
            metric: np.mean([fold[metric] for fold in fold_scores])
            for metric in fold_scores[0].keys()
        }

        std_metrics = {
            metric: np.std([fold[metric] for fold in fold_scores])
            for metric in fold_scores[0].keys()
        }

        # Use median of best iterations
        best_iteration = int(np.median(best_iterations))

        return CVResults(
            fold_scores=fold_scores,
            mean_metrics=mean_metrics,
            std_metrics=std_metrics,
            best_iteration=best_iteration
        )

    def _train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        num_boost_round: int
    ) -> xgb.XGBClassifier:
        """
        Train single XGBoost model with early stopping.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            num_boost_round: Maximum number of boosting rounds

        Returns:
            Trained XGBClassifier
        """
        model = xgb.XGBClassifier(
            n_estimators=num_boost_round,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective='binary:logistic',
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )

        # Train with early stopping
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=10,
            verbose=False
        )

        return model

    def _optimize_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray
    ) -> Tuple[float, ThresholdMetrics]:
        """
        Optimize classification threshold using cost function.

        Args:
            y_true: True labels
            y_proba: Predicted probabilities

        Returns:
            Tuple of (optimal_threshold, threshold_metrics)
        """
        # Generate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

        # Calculate cost for each threshold
        min_cost = float('inf')
        optimal_threshold = 0.5
        optimal_metrics = None

        for i, threshold in enumerate(thresholds):
            y_pred = (y_proba >= threshold).astype(int)

            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

            # Calculate cost
            cost = (fn * self.cost_fn) + (fp * self.cost_fp)

            if cost < min_cost:
                min_cost = cost
                optimal_threshold = threshold

                # Calculate metrics
                precision = precisions[i]
                recall = recalls[i]
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                tpr = recall

                optimal_metrics = ThresholdMetrics(
                    threshold=threshold,
                    precision=precision,
                    recall=recall,
                    f1=f1,
                    fpr=fpr,
                    fnr=fnr,
                    tpr=tpr,
                    tn=int(tn),
                    fp=int(fp),
                    fn=int(fn),
                    tp=int(tp),
                    total_cost=cost
                )

        return optimal_threshold, optimal_metrics

    def _get_feature_importance(
        self,
        model: xgb.XGBClassifier,
        feature_names: List[str]
    ) -> Dict[str, float]:
        """
        Extract feature importance from trained model.

        Args:
            model: Trained XGBClassifier
            feature_names: List of feature names

        Returns:
            Dictionary mapping feature names to importance scores
        """
        importance_values = model.feature_importances_

        # Sort by importance
        importance_dict = {
            name: float(score)
            for name, score in zip(feature_names, importance_values)
        }

        # Sort descending
        importance_dict = dict(sorted(
            importance_dict.items(),
            key=lambda x: x[1],
            reverse=True
        ))

        return importance_dict

    def evaluate(
        self,
        model: xgb.XGBClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
        threshold: float
    ) -> Dict[str, Any]:
        """
        Evaluate model on test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold

        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_proba)

        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

        metrics = {
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'fnr': fnr,
            'fpr': fpr,
            'confusion_matrix': {
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp)
            }
        }

        return metrics

    def save_model(self, model: xgb.XGBClassifier, path: str):
        """
        Save trained model to disk.

        Args:
            model: Trained XGBClassifier
            path: File path to save to
        """
        with open(path, 'wb') as f:
            pickle.dump(model, f)

        self.logger.info(f"Model saved: {path}")

    @staticmethod
    def load_model(path: str) -> xgb.XGBClassifier:
        """
        Load trained model from disk.

        Args:
            path: File path to load from

        Returns:
            Loaded XGBClassifier
        """
        with open(path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Model loaded: {path}")
        return model
