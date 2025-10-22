"""
Trino Query Predictor Service

A production-ready REST API service that classifies incoming Trino SQL queries
as "small" or "heavy" at submit time using a pre-trained ONNX model.
"""

__version__ = "1.0.0"
__service__ = "trino-query-predictor"
