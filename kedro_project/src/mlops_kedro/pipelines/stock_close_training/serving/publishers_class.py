from __future__ import annotations

import polars as pl

from .constants import (
    CLOSE_MODEL_TIME_FEATURE_COLUMNS,
    CONVENTIONAL_GAP_TRADING_COLUMNS,
    FOURIER_TIME_ENCODING_COLUMNS,
    PECNET_PREPROCESSED_COLUMNS,
    TIER_1_FEATURE_COLUMNS,
    TIER_3_FEATURE_COLUMNS,
    TIER_5_FEATURE_COLUMNS,
    TIER_6_FEATURE_COLUMNS,
    TIMESCALE_CLOSE_MODEL_DATASET_TABLE,
    TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE,
    TIMESCALE_PECNET_PREPROCESSED_TABLE,
    TIMESCALE_TABLE,
)
from .definitions import (
    _apply_close_model_dataset_definition_and_push,
    _apply_feast_definitions_and_push,
    _apply_pecnet_preprocessed_definition_and_push,
)
from .writers import (
    _write_close_model_dataset_to_timescale,
    _write_conventional_gap_trading_to_timescale,
    _write_model_features_to_timescale,
    _write_pecnet_preprocessed_to_timescale,
)


class FeatureStorePublisher:

    @staticmethod
    def publish_close_model_dataset(df: pl.DataFrame) -> dict[str, object]:
        timescale_rows = _write_close_model_dataset_to_timescale(df)
        feast_online_rows = _apply_close_model_dataset_definition_and_push(df)

        return {
            "timescale_table": TIMESCALE_CLOSE_MODEL_DATASET_TABLE,
            "timescale_rows": timescale_rows,
            "feast_online_rows": feast_online_rows,
            "feast_registry_applied": True,
            "feast_feature_view": "stock_close_model_dataset",
            "model_columns": ["unique_id", "ds", "y", *CLOSE_MODEL_TIME_FEATURE_COLUMNS],
        }

    @staticmethod
    def publish_model_features(df: pl.DataFrame) -> dict[str, object]:
        timescale_rows = _write_model_features_to_timescale(df)
        feast_online_rows = _apply_feast_definitions_and_push(df)

        return {
            "timescale_table": TIMESCALE_TABLE,
            "timescale_rows": timescale_rows,
            "feast_online_rows": feast_online_rows,
            "feast_feature_view": "stock_model_features",
            "tier_1_features": TIER_1_FEATURE_COLUMNS,
            "tier_2_time_features": [
                "calendar_gap_days",
                *FOURIER_TIME_ENCODING_COLUMNS,
            ],
            "tier_3_features": TIER_3_FEATURE_COLUMNS,
            "tier_5_features": TIER_5_FEATURE_COLUMNS,
            "tier_6_features": TIER_6_FEATURE_COLUMNS,
        }

    @staticmethod
    def publish_conventional_gap_trading(df: pl.DataFrame) -> dict[str, object]:
        timescale_rows = _write_conventional_gap_trading_to_timescale(df)
        return {
            "timescale_table": TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE,
            "timescale_rows": timescale_rows,
            "columns": CONVENTIONAL_GAP_TRADING_COLUMNS,
        }

    @staticmethod
    def publish_pecnet_preprocessed_training_data(
        df: pl.DataFrame,
    ) -> dict[str, object]:
        timescale_rows = _write_pecnet_preprocessed_to_timescale(df)
        feast_online_rows = _apply_pecnet_preprocessed_definition_and_push(df)
        return {
            "timescale_table": TIMESCALE_PECNET_PREPROCESSED_TABLE,
            "timescale_rows": timescale_rows,
            "feast_online_rows": feast_online_rows,
            "feast_feature_view": "pecnet_preprocessed_training_data",
            "columns": PECNET_PREPROCESSED_COLUMNS,
        }
