from ..config_resolvers import resolve_column_config


_COLUMNS = resolve_column_config({})

ENTITY_COLUMNS = _COLUMNS["entity"]
PRICE_COLUMNS = _COLUMNS["price"]
ANALYTICS_CALENDAR_COLUMNS = _COLUMNS["analytics_calendar"]
TARGET_COLUMNS = _COLUMNS["target"]
TIER_1_FEATURE_COLUMNS = _COLUMNS["tier_1_features"]
FOURIER_TIME_ENCODING_COLUMNS = _COLUMNS["fourier_time_encoding"]
MODEL_TIME_FEATURE_COLUMNS = _COLUMNS["model_time_features"]
TIME_FEATURE_COLUMNS = MODEL_TIME_FEATURE_COLUMNS
TIER_2_FEATURE_COLUMNS = _COLUMNS["tier_2_features"]
TIER_3_FEATURE_COLUMNS = _COLUMNS["tier_3_features"]
TIER_4_FEATURE_COLUMNS = _COLUMNS["tier_4_features"]
TIER_5_FEATURE_COLUMNS = _COLUMNS["tier_5_features"]
MODEL_FEATURE_COLUMNS = _COLUMNS["model_features"]
MODEL_TIER_FEATURE_COLUMNS = {
    "tier1": TIER_1_FEATURE_COLUMNS,
    "tier2": TIER_2_FEATURE_COLUMNS,
    "tier3": TIER_3_FEATURE_COLUMNS,
    "tier4": TIER_4_FEATURE_COLUMNS,
    "tier5": TIER_5_FEATURE_COLUMNS,
}
MODEL_TIER_NAMES = ("tier1", "tier2", "tier3")
PECNET_ONLY_TIER_NAMES = ("tier4", "tier5")
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
    *MODEL_FEATURE_COLUMNS,
]


def stock_price_indicator_features_path(bucket: str) -> str:
    return f"s3://{bucket}/feature_engineering/stock_price_indicators"
