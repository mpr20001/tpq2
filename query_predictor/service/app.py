"""
Main application entry point for the Trino Query Predictor service.

This module creates and configures the Flask application, sets up logging,
and serves the application using Waitress WSGI server.
"""

import logging
from flask import Flask
from waitress import serve

from query_predictor.common.config import load_config
from query_predictor.common.exceptions import ConfigurationError
from query_predictor.service.routes import register_routes
from query_predictor.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def create_app(config_path: str = None) -> Flask:
    """
    Create and configure the Flask application.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configured Flask application instance

    Raises:
        ConfigurationError: If configuration loading fails
    """
    app = Flask(__name__)

    # Load configuration
    try:
        config = load_config(config_path)
        app.config.update(config)
    except ConfigurationError as e:
        print(f"FATAL: Configuration error: {e}")
        raise

    # Set up logging
    setup_logging(config.get('logging', {'level': 'INFO', 'format': 'json', 'output': 'stdout'}))

    logger.info(
        "Flask application created",
        extra={
            'service': config.get('service', {}).get('name', 'unknown'),
            'version': config.get('service', {}).get('version', 'unknown')
        }
    )

    # Register routes
    register_routes(app)

    return app


def main():
    """
    Main entry point for running the service.

    Creates the Flask app and serves it using Waitress WSGI server.
    """
    try:
        app = create_app()

        # Get service configuration
        service_config = app.config.get('service', {})
        port = service_config.get('port', 8000)
        workers = service_config.get('workers', 4)

        logger.info(
            "Starting Trino Query Predictor service",
            extra={
                'port': port,
                'workers': workers,
                'service': service_config.get('name', 'unknown'),
                'version': service_config.get('version', 'unknown')
            }
        )

        # Serve with Waitress
        serve(
            app,
            host='0.0.0.0',
            port=port,
            threads=workers,
            url_scheme='http'
        )

    except Exception as e:
        logger.error(f"Failed to start service: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
