from __future__ import annotations

from .bronze_class import *  # noqa: F403
from .bronze_class import BronzeStockPriceAssets

stock_prices = asset(
    io_manager_key="delta_io_manager",
    key_prefix=["bronze"],
    compute_kind="polars",
    group_name="ingestion",
)(BronzeStockPriceAssets.stock_prices)

stock_prices_weekly = asset(
    io_manager_key="delta_io_manager",
    key_prefix=["bronze"],
    compute_kind="polars",
    group_name="ingestion",
)(BronzeStockPriceAssets.stock_prices_weekly)
