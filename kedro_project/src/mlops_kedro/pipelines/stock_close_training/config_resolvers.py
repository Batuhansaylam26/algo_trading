from __future__ import annotations

from .config_resolvers_class import *  # noqa: F403
from .config_resolvers_class import StockCloseConfigResolver

stock_close_config_resolver = StockCloseConfigResolver()
_ordered_unique = stock_close_config_resolver._ordered_unique
configured_list = stock_close_config_resolver.configured_list
resolve_column_config = stock_close_config_resolver.resolve_column_config
