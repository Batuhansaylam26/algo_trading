from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl

from .feature_engineering_oop import StockCloseFeatureEngineering
from .ml.mlforecast import make_train_test_split
from .node_utils import (
    _as_bool,
    _as_int,
    _bucket,
    _columns,
    _feature_columns_for_tier,
    _indicator_features_path,
    _log_step,
)
from .serving import FeatureStoreService


class StockCloseDataNodes:
    def __init__(self, feature_store: FeatureStoreService | None = None) -> None:
        self.feature_store = feature_store or FeatureStoreService()

    def configure_feature_engineering(
        self,
        delta_lake_params: dict[str, Any] | None,
        columns_params: dict[str, Any] | None,
        time_encoding_params: dict[str, Any] | None,
    ) -> StockCloseFeatureEngineering:
        return StockCloseFeatureEngineering(
            columns_config=_columns(columns_params),
            time_encoding_config=time_encoding_params or {},
            bucket=_bucket(delta_lake_params),
        )

    def load_silver_stock_prices(
        self,
        feature_engineering: StockCloseFeatureEngineering,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        silver_stock_prices = feature_engineering.read_silver_stock_prices()
        metadata = {
            "silver_rows": len(silver_stock_prices),
            "symbols": silver_stock_prices["symbol"].n_unique()
            if "symbol" in silver_stock_prices.columns
            else 0,
            "min_date": silver_stock_prices["date"].min()
            if "date" in silver_stock_prices.columns
            else None,
            "max_date": silver_stock_prices["date"].max()
            if "date" in silver_stock_prices.columns
            else None,
        }
        _log_step("load_silver_stock_prices", **metadata)
        return silver_stock_prices, metadata

    def load_silver_stock_prices_weekly(
        self,
        feature_engineering: StockCloseFeatureEngineering,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        silver_stock_prices_weekly = feature_engineering.read_silver_stock_prices_weekly()
        metadata = {
            "silver_weekly_rows": len(silver_stock_prices_weekly),
            "symbols": silver_stock_prices_weekly["symbol"].n_unique()
            if "symbol" in silver_stock_prices_weekly.columns
            else 0,
            "min_date": silver_stock_prices_weekly["date"].min()
            if "date" in silver_stock_prices_weekly.columns
            else None,
            "max_date": silver_stock_prices_weekly["date"].max()
            if "date" in silver_stock_prices_weekly.columns
            else None,
        }
        _log_step("load_silver_stock_prices_weekly", **metadata)
        return silver_stock_prices_weekly, metadata

    def prepare_close_model_dataset(
        self,
        feature_engineering: StockCloseFeatureEngineering,
        silver_stock_prices: pl.DataFrame,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        model_dataset = feature_engineering.build_stock_close_model_dataset(
            silver_stock_prices,
        )
        metadata = {
            "silver_rows": len(silver_stock_prices),
            "close_model_rows": len(model_dataset),
            "symbols": model_dataset["unique_id"].n_unique(),
            "min_ds": model_dataset["ds"].min(),
            "max_ds": model_dataset["ds"].max(),
        }
        _log_step("prepare_close_model_dataset", **metadata)
        return model_dataset, metadata

    def publish_close_model_dataset(
        self,
        stock_close_model_dataset: pl.DataFrame,
    ) -> dict[str, Any]:
        metadata = self.feature_store.publish_close_model_dataset(
            stock_close_model_dataset
        )
        _log_step("publish_close_model_dataset", **metadata)
        return metadata

    def prepare_indicator_features(
        self,
        feature_engineering: StockCloseFeatureEngineering,
        feature_engineering_params: dict[str, Any] | None,
        silver_stock_prices: pl.DataFrame,
        silver_stock_prices_weekly: pl.DataFrame,
    ) -> tuple[pl.DataFrame, dict[str, Any], pl.DataFrame, dict[str, Any]]:
        feature_engineering_params = feature_engineering_params or {}
        indicator_features_path = _indicator_features_path(
            {"bucket": feature_engineering.bucket},
            feature_engineering_params,
        )
        if not _as_bool(
            feature_engineering_params.get("publish_indicator_features"),
            True,
        ):
            metadata = {
                "publish_indicator_features": False,
                "indicator_feature_rows": 0,
                "indicator_features_path": indicator_features_path,
            }
            model_feature_metadata = {
                "publish_model_features": False,
                "model_feature_rows": 0,
                "reason": "publish_indicator_features is false",
            }
            _log_step("prepare_indicator_features", **metadata)
            return pl.DataFrame(), metadata, pl.DataFrame(), model_feature_metadata

        (
            indicator_features,
            stock_model_features,
        ) = feature_engineering.build_stock_feature_sets(
            silver_stock_prices,
            silver_stock_prices_weekly,
        )
        feature_engineering.write_delta_table(indicator_features_path, indicator_features)
        metadata = {
            "publish_indicator_features": True,
            "silver_rows": len(silver_stock_prices),
            "silver_weekly_rows": len(silver_stock_prices_weekly),
            "indicator_feature_rows": len(indicator_features),
            "indicator_features_path": indicator_features_path,
        }
        model_feature_metadata = {
            "publish_model_features": True,
            "model_feature_rows": len(stock_model_features),
        }
        _log_step("prepare_indicator_features", **metadata)
        return indicator_features, metadata, stock_model_features, model_feature_metadata

    def load_indicator_features(
        self,
        feature_engineering: StockCloseFeatureEngineering,
        feature_engineering_params: dict[str, Any] | None,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        indicator_features_path = _indicator_features_path(
            {"bucket": feature_engineering.bucket},
            feature_engineering_params,
        )
        indicator_features = StockCloseFeatureEngineering.read_delta_table(
            indicator_features_path,
        )
        metadata = {
            "indicator_feature_rows": len(indicator_features),
            "indicator_features_path": indicator_features_path,
        }
        _log_step("load_indicator_features", **metadata)
        return indicator_features, metadata

    def publish_indicator_model_features(
        self,
        stock_model_features: pl.DataFrame,
        model_feature_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if stock_model_features.is_empty():
            metadata = {
                "skipped": True,
                "reason": "stock_model_features is empty",
            }
            _log_step("publish_indicator_model_features", **metadata)
            return metadata

        metadata = self.feature_store.publish_model_features(stock_model_features)
        metadata["model_feature_rows"] = model_feature_metadata.get(
            "model_feature_rows",
            len(stock_model_features),
        )
        _log_step("publish_indicator_model_features", **metadata)
        return metadata

    def prepare_conventional_gap_trading(
        self,
        stock_price_indicator_features: pl.DataFrame,
        conventional_gap_trading_params: dict[str, Any] | None,
        feature_engineering: StockCloseFeatureEngineering,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        conventional_gap_trading_params = conventional_gap_trading_params or {}
        if not _as_bool(
            conventional_gap_trading_params.get("publish_to_timescale"),
            True,
        ):
            metadata = {
                "publish_conventional_gap_trading": False,
                "conventional_gap_trading_rows": 0,
            }
            _log_step("prepare_conventional_gap_trading", **metadata)
            return pl.DataFrame(), metadata

        if stock_price_indicator_features.is_empty():
            metadata = {
                "publish_conventional_gap_trading": True,
                "indicator_feature_rows": 0,
                "conventional_gap_trading_rows": 0,
                "signals": {},
            }
            _log_step("prepare_conventional_gap_trading", **metadata)
            return pl.DataFrame(), metadata

        conventional_gap_trading = (
            feature_engineering.build_conventional_gap_trading_features(
                stock_price_indicator_features,
            )
        )
        metadata = {
            "publish_conventional_gap_trading": True,
            "indicator_feature_rows": len(stock_price_indicator_features),
            "conventional_gap_trading_rows": len(conventional_gap_trading),
            "signals": (
                conventional_gap_trading.get_column("Gap_Type")
                .value_counts()
                .to_dict(as_series=False)
                if "Gap_Type" in conventional_gap_trading.columns
                else {}
            ),
        }
        _log_step("prepare_conventional_gap_trading", **metadata)
        return conventional_gap_trading, metadata

    def publish_conventional_gap_trading(
        self,
        conventional_gap_trading: pl.DataFrame,
        conventional_gap_trading_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not conventional_gap_trading_metadata["publish_conventional_gap_trading"]:
            metadata = {
                "skipped": True,
                "reason": "publish_conventional_gap_trading is false",
            }
            _log_step("publish_conventional_gap_trading", **metadata)
            return metadata

        metadata = self.feature_store.publish_conventional_gap_trading(
            conventional_gap_trading,
        )
        _log_step("publish_conventional_gap_trading", **metadata)
        return metadata

    def load_model_training_dataset(
        self,
        columns_params: dict[str, Any] | None,
        training_params: dict[str, Any] | None,
        *,
        tier_name: str,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        training_params = training_params or {}
        data_source = training_params.get("data_source", "feast_online")
        if data_source != "feast_online":
            raise ValueError(
                "Stock close model training currently expects Feast online data. "
                f"Got data_source={data_source!r}."
            )

        feature_columns = _feature_columns_for_tier(columns_params, tier_name)
        training_dataset = self.feature_store.load_model_training_dataset_from_online_store(
            feature_columns,
        )
        metadata = {
            "tier": tier_name,
            "training_data_source": data_source,
            "training_rows": len(training_dataset),
            "symbols": training_dataset["unique_id"].n_unique()
            if "unique_id" in training_dataset.columns
            else 0,
            "feature_columns": feature_columns,
        }
        _log_step(f"load_{tier_name}_training_dataset", **metadata)
        return training_dataset, metadata

    def load_model_training_dataset_after_feature_publish(
        self,
        columns_params: dict[str, Any] | None,
        training_params: dict[str, Any] | None,
        model_feature_publish_metadata: dict[str, Any],
        *,
        tier_name: str,
    ) -> tuple[pl.DataFrame, dict[str, Any]]:
        _ = model_feature_publish_metadata
        return self.load_model_training_dataset(
            columns_params,
            training_params,
            tier_name=tier_name,
        )

    def train_test_split_for_tier(
        self,
        stock_close_training_dataset: pl.DataFrame,
        mlforecast_params: dict[str, Any] | None,
        *,
        tier_name: str,
    ) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
        mlforecast_params = mlforecast_params or {}
        test_horizon = _as_int(mlforecast_params.get("test_horizon"), 5)
        split = make_train_test_split(
            stock_close_training_dataset,
            test_horizon=test_horizon,
        )
        metadata = {
            "tier": tier_name,
            "train_rows": len(split["train"]),
            "test_rows": len(split["test"]),
            "test_horizon": test_horizon,
        }
        _log_step(f"{tier_name}_train_test_split", **metadata)
        return split, metadata


stock_close_data_nodes = StockCloseDataNodes()
