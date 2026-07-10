import os
from pathlib import Path

import yaml

from ..config_resolvers import resolve_column_config


def _kedro_project_dir() -> Path:
    return Path(
        os.getenv(
            "KEDRO_PROJECT_DIR",
            str(Path(__file__).resolve().parents[5]),
        )
    )


def _load_data_preprocessing_columns() -> dict[str, list[str]]:
    parameters_path = (
        _kedro_project_dir() / "conf" / "base" / "parameters_data_preprocessing.yml"
    )
    parameters = yaml.safe_load(parameters_path.read_text()) or {}
    data_preprocessing = parameters.get("stock_close_data_preprocessing", {})
    return resolve_column_config(data_preprocessing.get("columns", {}))


def _column_config() -> dict[str, list[str]]:
    return _load_data_preprocessing_columns()


_COLUMNS = _column_config()

ENTITY_COLUMNS = _COLUMNS["entity"]
PRICE_COLUMNS = _COLUMNS["price"]
ANALYTICS_CALENDAR_COLUMNS = _COLUMNS["analytics_calendar"]
TARGET_COLUMNS = _COLUMNS["target"]
TIER_1_FEATURE_COLUMNS = _COLUMNS["tier_1_features"]
FOURIER_TIME_ENCODING_COLUMNS = _COLUMNS["fourier_time_encoding"]
MODEL_TIME_FEATURE_COLUMNS = _COLUMNS["model_time_features"]
TIME_FEATURE_COLUMNS = MODEL_TIME_FEATURE_COLUMNS
TIER_2_FEATURE_COLUMNS = _COLUMNS["tier_2_features"]
MODEL_TIER_FEATURE_COLUMNS = {
    "tier1": TIER_1_FEATURE_COLUMNS,
    "tier2": TIER_2_FEATURE_COLUMNS,
}
MODEL_TIER_NAMES = tuple(MODEL_TIER_FEATURE_COLUMNS.keys())
INDICATOR_COLUMNS = _COLUMNS["indicator"]
CONDITION_COLUMNS = _COLUMNS["condition"]
STRATEGY_LABEL_COLUMNS = _COLUMNS["strategy_label"]
OUTPUT_AUDIT_COLUMNS = _COLUMNS["output_audit"]
INDICATOR_FEATURE_COLUMNS = _COLUMNS["indicator_features"]
CONVENTIONAL_GAP_TRADING_COLUMNS = _COLUMNS["conventional_gap_trading"]
STRATEGY_FEATURE_COLUMNS = CONVENTIONAL_GAP_TRADING_COLUMNS
FEATURE_COLUMNS = INDICATOR_FEATURE_COLUMNS
CLOSE_MODEL_TIME_FEATURE_COLUMNS = _COLUMNS["close_model_time_features"]
CLOSE_MODEL_DATASET_COLUMNS = _COLUMNS["close_model_dataset"]
FEAST_ENTITY_COLUMNS = [
    *ENTITY_COLUMNS,
    *OUTPUT_AUDIT_COLUMNS,
]
FEAST_OFFLINE_COLUMNS = [
    *FEAST_ENTITY_COLUMNS,
    *TIER_2_FEATURE_COLUMNS,
]
