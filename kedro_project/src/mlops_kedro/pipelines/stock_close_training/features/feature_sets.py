from __future__ import annotations

from .feature_sets_class import *  # noqa: F403
from .feature_sets_class import StockCloseFeatureSetConfig

stock_close_feature_set_config = StockCloseFeatureSetConfig()
stock_price_indicator_features_path = stock_close_feature_set_config.stock_price_indicator_features_path
