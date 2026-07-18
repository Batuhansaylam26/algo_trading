from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .loaders_close import (
    load_stock_close_model_dataset_from_feast,
    load_stock_close_model_dataset_from_feast_online,
    load_stock_close_model_dataset_from_redis,
    load_stock_close_model_dataset_from_timescale,
)
from .loaders_model import (
    get_online_model_features,
    load_stock_model_training_dataset_from_feast_online,
    load_stock_tier1_model_dataset_from_feast_online,
    load_stock_tier2_model_dataset_from_feast_online,
    load_stock_tier3_model_dataset_from_feast_online,
)
from .publishers import (
    publish_close_model_dataset,
    publish_conventional_gap_trading,
    publish_model_features,
    publish_pecnet_preprocessed_training_data,
)


@dataclass(slots=True)
class FeatureStoreService:
    def publish_close_model_dataset(self, df: pl.DataFrame) -> dict[str, object]:
        return publish_close_model_dataset(df)

    def publish_model_features(self, df: pl.DataFrame) -> dict[str, object]:
        return publish_model_features(df)

    def publish_conventional_gap_trading(self, df: pl.DataFrame) -> dict[str, object]:
        return publish_conventional_gap_trading(df)

    def publish_pecnet_preprocessed_training_data(
        self,
        df: pl.DataFrame,
    ) -> dict[str, object]:
        return publish_pecnet_preprocessed_training_data(df)

    def load_model_training_dataset_from_online_store(
        self,
        feature_columns: list[str],
    ) -> pl.DataFrame:
        return load_stock_model_training_dataset_from_feast_online(feature_columns)

    def load_close_model_dataset_from_offline_store(self) -> pl.DataFrame:
        return load_stock_close_model_dataset_from_feast()

    def load_close_model_dataset_from_online_store(self) -> pl.DataFrame:
        return load_stock_close_model_dataset_from_feast_online()

    def load_close_model_dataset_from_redis(self) -> pl.DataFrame:
        return load_stock_close_model_dataset_from_redis()

    def load_close_model_dataset_from_timescale(self) -> pl.DataFrame:
        return load_stock_close_model_dataset_from_timescale()

    def load_tier1_model_dataset_from_online_store(self) -> pl.DataFrame:
        return load_stock_tier1_model_dataset_from_feast_online()

    def load_tier2_model_dataset_from_online_store(self) -> pl.DataFrame:
        return load_stock_tier2_model_dataset_from_feast_online()

    def load_tier3_model_dataset_from_online_store(self) -> pl.DataFrame:
        return load_stock_tier3_model_dataset_from_feast_online()

    def get_online_model_features(
        self,
        symbol: str,
        date,
        feature_columns: list[str],
    ) -> dict:
        return get_online_model_features(symbol, date, feature_columns)
