from __future__ import annotations

import os
from pathlib import Path

from ..features.feature_sets import (
    CLOSE_MODEL_DATASET_COLUMNS,
    CLOSE_MODEL_TIME_FEATURE_COLUMNS,
    CONDITION_COLUMNS,
    CONVENTIONAL_GAP_TRADING_COLUMNS,
    FEAST_OFFLINE_COLUMNS,
    FOURIER_TIME_ENCODING_COLUMNS,
    MODEL_TIER_FEATURE_COLUMNS,
    TIER_1_FEATURE_COLUMNS,
    TIER_2_FEATURE_COLUMNS,
    TIER_3_FEATURE_COLUMNS,
    TIER_5_FEATURE_COLUMNS,
    TIER_6_FEATURE_COLUMNS,
)


FEATURE_REPO_DIR = Path(
    os.getenv(
        "FEATURE_REPO_DIR",
        str(Path(__file__).resolve().parents[6] / "feature_repo"),
    )
).resolve()
TIMESCALE_TABLE = "feature_store.stock_model_features"
TIMESCALE_CLOSE_MODEL_DATASET_TABLE = "feature_store.stock_close_model_dataset"
TIMESCALE_CONVENTIONAL_GAP_TRADING_TABLE = "feature_store.conventional_gap_trading"
TIMESCALE_PECNET_PREPROCESSED_TABLE = (
    "feature_store.pecnet_preprocessed_training_data"
)
TIMESCALE_WRITE_BATCH_SIZE = int(os.getenv("TIMESCALE_WRITE_BATCH_SIZE", "500"))
TIMESCALE_DAILY_FILL_FREQ = os.getenv("TIMESCALE_DAILY_FILL_FREQ", "B")

PECNET_PREPROCESSED_COLUMNS = [
    "row_key",
    "tier",
    "symbol",
    "event_timestamp",
    "split",
    "split_index",
    "variable_name",
    "variable_index",
    "sample_index",
    "step_index",
    "value",
    "target_y",
    "created_timestamp",
]
PECNET_PREPROCESSED_FEAST_COLUMNS = [
    "row_key",
    "event_timestamp",
    "created_timestamp",
    "value",
    "target_y",
    "sample_index",
    "step_index",
    "variable_index",
    "split_index",
]
