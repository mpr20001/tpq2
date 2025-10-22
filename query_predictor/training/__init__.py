"""
Training pipeline components for Trino Query Predictor.

This module contains components for training the ML model:
- Historical stats computation (HistoricalStatsComputer)
- Feature engineering pipeline
- Model training and evaluation
- ONNX model export
"""

from query_predictor.training.historical_stats_computer import HistoricalStatsComputer

__all__ = ['HistoricalStatsComputer']
