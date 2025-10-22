"""Context and temporal feature extractor - 8 features."""

import logging
from typing import List, Dict, Any

from query_predictor.core.featurizer.base import BaseFeatureExtractor
from query_predictor.core.types.query_data import QueryData
from query_predictor.core.featurizer import utils

logger = logging.getLogger(__name__)


class ContextExtractor(BaseFeatureExtractor):
    """
    Extract context and temporal features (8 features).

    Features:
    - user_hash, catalog_hash, schema_hash
    - hour_of_day, hour_sin, hour_cos
    - is_business_hours, client_info_length
    """

    FEATURE_NAMES = [
        'user_hash', 'catalog_hash', 'schema_hash', 'hour_of_day',
        'hour_sin', 'hour_cos', 'is_business_hours', 'client_info_length'
    ]

    def get_feature_names(self) -> List[str]:
        return self.FEATURE_NAMES

    def extract(self, query_data: QueryData) -> List[float]:
        try:
            features = []

            # Hash user/catalog/schema for anonymization
            user = str(query_data.user)
            catalog = str(query_data.catalog) if query_data.catalog is not None else ''
            schema = str(query_data.schema) if query_data.schema is not None else ''

            features.append(utils.hash_string(user, modulo=1000000))  # user_hash
            features.append(utils.hash_string(catalog, modulo=10000))  # catalog_hash
            features.append(utils.hash_string(schema, modulo=10000))  # schema_hash

            # Temporal features
            hour = query_data.hour

            features.append(hour / 24.0)  # hour_of_day
            hour_sin, hour_cos = utils.encode_hour_cyclical(hour)
            features.append(hour_sin)  # hour_sin
            features.append(hour_cos)  # hour_cos
            features.append(utils.is_business_hours(hour))  # is_business_hours

            # Client info
            client_info = query_data.client_info
            features.append(float(utils.safe_len(client_info)))  # client_info_length

            return features

        except Exception as e:
            logger.error(f"Error extracting context features: {e}")
            return [0.0] * len(self.FEATURE_NAMES)
