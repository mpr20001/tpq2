"""API route handlers for the query predictor service."""

from flask import jsonify, current_app
import logging

logger = logging.getLogger(__name__)


def register_routes(app):
    """
    Register all API routes with the Flask app.

    Args:
        app: Flask application instance
    """

    @app.route('/manage/health/liveness', methods=['GET'])
    def liveness():
        """
        Liveness probe endpoint.

        Returns 200 if the service is alive (process is running).
        Used by Kubernetes to restart unhealthy pods.
        """
        try:
            service_name = current_app.config.get('service', {}).get('name', 'unknown')
            return jsonify({
                "status": "UP",
                "service": service_name
            }), 200
        except Exception as e:
            logger.error(f"Error in liveness check: {e}")
            return jsonify({
                "status": "DOWN",
                "error": "Internal error"
            }), 500

    @app.route('/manage/health/readiness', methods=['GET'])
    def readiness():
        """
        Readiness probe endpoint.

        Returns 200 if the service is ready to accept traffic.
        Used by Kubernetes to determine if pod should receive requests.

        TODO: Add checks for:
        - Model loaded
        - Featurizer ready
        - Historical stats loaded
        """
        try:
            service_config = current_app.config.get('service', {})

            return jsonify({
                "ready": True,
                "status": "UP",
                "service": service_config.get('name', 'unknown'),
                "version": service_config.get('version', 'unknown'),
                "checks": {
                    "model_loaded": False,  # TODO: 
                    "featurizer_ready": False,  # TODO:
                    "classifier_ready": False  # TODO: 
                }
            }), 200
        except Exception as e:
            logger.error(f"Error in readiness check: {e}")
            return jsonify({
                "ready": False,
                "status": "DOWN",
                "error": "Internal error"
            }), 500

    @app.route('/v1/predict', methods=['POST'])
    def predict():
        """
        Prediction endpoint - placeholder implementation.

        Will be fully implemented after:
        - Featurizer integration
        - Classifier integration
        - ONNX model loading
        """
        try:
            logger.warning("Prediction endpoint called but not yet implemented")
            return jsonify({
                "error": "Not implemented yet",
                "message": "Prediction endpoint will be available later"
            }), 501
        except Exception as e:
            logger.error(f"Error in prediction endpoint: {e}")
            return jsonify({
                "error": "Internal server error",
                "message": "An unexpected error occurred"
            }), 500

    @app.route('/v1/info', methods=['GET'])
    def info():
        """
        Service information endpoint.

        Returns metadata about the service, model, and configuration.
        """
        try:
            service_config = current_app.config.get('service', {})

            return jsonify({
                "service": service_config.get('name', 'trino-query-predictor'),
                "version": service_config.get('version', '1.0.0'),
                "status": "skeleton",
                "phase": "Service skeleton",
                "endpoints": {
                    "/manage/health/liveness": "Liveness probe",
                    "/manage/health/readiness": "Readiness probe",
                    "/v1/predict": "Query prediction (not implemented)",
                    "/v1/info": "Service information"
                }
            }), 200
        except Exception as e:
            logger.error(f"Error in info endpoint: {e}")
            return jsonify({
                "error": "Internal server error",
                "message": "Unable to retrieve service information"
            }), 500

    logger.info("Routes registered successfully")
