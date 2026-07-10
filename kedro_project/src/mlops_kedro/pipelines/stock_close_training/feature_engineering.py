import os
from functools import partial
from typing import Any

import polars as pl
from deltalake import DeltaTable, write_deltalake

from .features.conditions import calculate_conditions_for_symbol
from .features.feature_engineering_utils import (
    add_model_training_tier_columns_for_symbol,
    drop_rows_with_missing_model_features,
    map_by_symbol,
    prepare_feature_source,
    to_stock_close_model_dataset,
    with_created_timestamp,
)
from .features.indicators import calculate_indicators_for_symbol
from .features.strategy import classify_strategy_for_symbol


def delta_storage_options() -> dict[str, str]:
    endpoint = os.getenv("DELTA_LAKE_S3_ENDPOINT_URL", "http://host.docker.internal:9000")
    return {
        "AWS_ACCESS_KEY_ID": os.getenv("DELTA_LAKE_S3_ACCESS_KEY", "admin"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("DELTA_LAKE_S3_SECRET_KEY", "admin1234"),
        "AWS_REGION": os.getenv("AWS_REGION", "us-east-1"),
        "AWS_ENDPOINT_URL": endpoint,
        "AWS_ALLOW_HTTP": os.getenv("AWS_ALLOW_HTTP", "true"),
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
        "AWS_S3_FORCE_PATH_STYLE": "true",
    }


def silver_stock_prices_path(bucket: str) -> str:
    return f"s3://{bucket}/silver/stock_prices"


def stock_price_indicator_features_path(bucket: str) -> str:
    return f"s3://{bucket}/feature_engineering/stock_price_indicators"


def read_delta_table(path: str) -> pl.DataFrame:
    table = DeltaTable(path, storage_options=delta_storage_options())
    return pl.from_arrow(table.to_pyarrow_table())


def write_delta_table(path: str, df: pl.DataFrame, mode: str = "overwrite") -> None:
    schema_mode = "overwrite" if mode == "overwrite" else None
    write_deltalake(
        path,
        df.to_arrow(),
        mode=mode,
        schema_mode=schema_mode,
        storage_options=delta_storage_options(),
    )


def read_silver_stock_prices(bucket: str) -> pl.DataFrame:
    return read_delta_table(silver_stock_prices_path(bucket))


def build_stock_close_model_dataset(
    silver_stock_prices: pl.DataFrame,
    columns_config: dict[str, list[str]],
    time_encoding_config: dict[str, Any],
) -> pl.DataFrame:
    return to_stock_close_model_dataset(
        silver_stock_prices,
        columns_config["close_model_dataset"],
        time_encoding_config,
    )


def build_stock_price_indicator_features(
    silver_stock_prices: pl.DataFrame,
    columns_config: dict[str, list[str]],
    time_encoding_config: dict[str, Any],
) -> pl.DataFrame:
    feature_source = prepare_feature_source(silver_stock_prices, time_encoding_config)
    feature_enriched_prices = map_by_symbol(
        feature_source.drop_nulls(["symbol", "date"]).sort(["symbol", "date"]),
        partial(
            add_model_training_tier_columns_for_symbol,
            time_encoding_config=time_encoding_config,
        ),
    )
    indicators = map_by_symbol(
        feature_enriched_prices,
        calculate_indicators_for_symbol,
    )
    return with_created_timestamp(
        drop_rows_with_missing_model_features(indicators, columns_config["model_ready"])
    ).select(columns_config["indicator_features"])


def build_conventional_gap_trading_features(
    stock_price_indicator_features: pl.DataFrame,
    columns_config: dict[str, list[str]],
) -> pl.DataFrame:
    conditions = map_by_symbol(
        stock_price_indicator_features,
        partial(
            calculate_conditions_for_symbol,
            condition_columns=columns_config["condition"],
        ),
    )
    strategy_features = map_by_symbol(
        conditions,
        partial(
            classify_strategy_for_symbol,
            required_condition_columns=columns_config["strategy_required_conditions"],
        ),
    )
    return strategy_features.select(columns_config["conventional_gap_trading"])
