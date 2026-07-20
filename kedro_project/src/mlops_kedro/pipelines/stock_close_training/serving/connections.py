from __future__ import annotations

from .connections_class import *  # noqa: F403
from .connections_class import FeatureStoreConnections

feature_store_connections = FeatureStoreConnections()
_ensure_feature_repo_on_path = feature_store_connections._ensure_feature_repo_on_path
_timescale_connection_kwargs = feature_store_connections._timescale_connection_kwargs
_schema_name = feature_store_connections._schema_name
