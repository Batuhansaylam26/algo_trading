from __future__ import annotations

from dagster import AssetIn, asset

from .silver_class import SilverStockPriceAssets


STOCK_PRICE_SCHEMA = SilverStockPriceAssets.stock_price_schema()

stock_prices = asset(
    ins={"bronze_stock_prices": AssetIn(key=["bronze", "stock_prices"])},
    io_manager_key="delta_io_manager",
    key_prefix=["silver"],
    compute_kind="polars",
    group_name="staging",
)(SilverStockPriceAssets.stock_prices)

stock_prices_weekly = asset(
    ins={
        "bronze_stock_prices_weekly": AssetIn(
            key=["bronze", "stock_prices_weekly"]
        )
    },
    io_manager_key="delta_io_manager",
    key_prefix=["silver"],
    compute_kind="polars",
    group_name="staging",
)(SilverStockPriceAssets.stock_prices_weekly)
