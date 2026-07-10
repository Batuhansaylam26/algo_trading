from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int64


TIER_1_FEATURES = [
    "prev_open",
    "prev_close",
    "prev_high",
    "prev_low",
    "prev_volume",
]

TIER_2_TIME_FEATURES = [
    "calendar_gap_days",
    "month_sin_1",
    "month_cos_1",
    "month_sin_2",
    "month_cos_2",
    "day_sin_1",
    "day_cos_1",
    "day_sin_2",
    "day_cos_2",
    "day_of_year_sin_1",
    "day_of_year_cos_1",
    "day_of_year_sin_2",
    "day_of_year_cos_2",
]

TIER_2_FEATURES = [
    *TIER_1_FEATURES,
    *TIER_2_TIME_FEATURES,
]


ticker = Entity(name="ticker", join_keys=["symbol"])

stock_model_source = PostgreSQLSource(
    name="stock_model_features_source",
    query="SELECT * FROM feature_store.stock_model_features",
    timestamp_field="date",
    created_timestamp_column="created_timestamp",
)

stock_model_features_view = FeatureView(
    name="stock_model_features",
    entities=[ticker],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="prev_open", dtype=Float64),
        Field(name="prev_close", dtype=Float64),
        Field(name="prev_high", dtype=Float64),
        Field(name="prev_low", dtype=Float64),
        Field(name="prev_volume", dtype=Float64),
        Field(name="calendar_gap_days", dtype=Int64),
        Field(name="month_sin_1", dtype=Float64),
        Field(name="month_cos_1", dtype=Float64),
        Field(name="month_sin_2", dtype=Float64),
        Field(name="month_cos_2", dtype=Float64),
        Field(name="day_sin_1", dtype=Float64),
        Field(name="day_cos_1", dtype=Float64),
        Field(name="day_sin_2", dtype=Float64),
        Field(name="day_cos_2", dtype=Float64),
        Field(name="day_of_year_sin_1", dtype=Float64),
        Field(name="day_of_year_cos_1", dtype=Float64),
        Field(name="day_of_year_sin_2", dtype=Float64),
        Field(name="day_of_year_cos_2", dtype=Float64),
    ],
    online=True,
    source=stock_model_source,
    tags={"layer": "feature_serving", "team": "dataops_mlops"},
)

stock_model_tier_1_feature_service = FeatureService(
    name="stock_model_tier_1_features_v1",
    features=[stock_model_features_view[TIER_1_FEATURES]],
)

stock_model_tier_2_feature_service = FeatureService(
    name="stock_model_tier_2_features_v1",
    features=[stock_model_features_view[TIER_2_FEATURES]],
)


stock_feature_row = Entity(name="stock_feature_row", join_keys=["feature_key"])

stock_model_tier2_dataset_source = PostgreSQLSource(
    name="stock_model_tier2_dataset_source",
    query="""
        SELECT
            *,
            to_char(
                "date" AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            ) AS date_key,
            symbol || '|' || to_char(
                "date" AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            ) AS feature_key
        FROM feature_store.stock_model_features
    """,
    timestamp_field="date",
    created_timestamp_column="created_timestamp",
)

stock_model_tier2_dataset_view = FeatureView(
    name="stock_model_tier2_dataset",
    entities=[stock_feature_row],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="prev_open", dtype=Float64),
        Field(name="prev_close", dtype=Float64),
        Field(name="prev_high", dtype=Float64),
        Field(name="prev_low", dtype=Float64),
        Field(name="prev_volume", dtype=Float64),
        Field(name="calendar_gap_days", dtype=Int64),
        Field(name="month_sin_1", dtype=Float64),
        Field(name="month_cos_1", dtype=Float64),
        Field(name="month_sin_2", dtype=Float64),
        Field(name="month_cos_2", dtype=Float64),
        Field(name="day_sin_1", dtype=Float64),
        Field(name="day_cos_1", dtype=Float64),
        Field(name="day_sin_2", dtype=Float64),
        Field(name="day_cos_2", dtype=Float64),
        Field(name="day_of_year_sin_1", dtype=Float64),
        Field(name="day_of_year_cos_1", dtype=Float64),
        Field(name="day_of_year_sin_2", dtype=Float64),
        Field(name="day_of_year_cos_2", dtype=Float64),
    ],
    online=True,
    source=stock_model_tier2_dataset_source,
    tags={"layer": "model_training", "team": "dataops_mlops"},
)

stock_model_tier2_dataset_service = FeatureService(
    name="stock_model_tier2_dataset_v1",
    features=[stock_model_tier2_dataset_view[TIER_2_FEATURES]],
)


stock_series = Entity(name="stock_series", join_keys=["series_key"])

stock_close_model_dataset_source = PostgreSQLSource(
    name="stock_close_model_dataset_source",
    query="""
        SELECT
            *,
            to_char(
                ds AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            ) AS ds_key,
            unique_id || '|' || to_char(
                ds AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            ) AS series_key
        FROM feature_store.stock_close_model_dataset
    """,
    timestamp_field="ds",
    created_timestamp_column="created_timestamp",
)

stock_close_model_dataset_view = FeatureView(
    name="stock_close_model_dataset",
    entities=[stock_series],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="y", dtype=Float64),
        Field(name="month_sin_1", dtype=Float64),
        Field(name="month_cos_1", dtype=Float64),
        Field(name="day_sin_1", dtype=Float64),
        Field(name="day_cos_1", dtype=Float64),
        Field(name="day_of_year_sin_1", dtype=Float64),
        Field(name="day_of_year_cos_1", dtype=Float64),
    ],
    online=True,
    source=stock_close_model_dataset_source,
    tags={"layer": "model_training", "team": "dataops_mlops"},
)

stock_close_model_dataset_service = FeatureService(
    name="stock_close_model_dataset_v1",
    features=[
        stock_close_model_dataset_view[
            [
                "y",
                "month_sin_1",
                "month_cos_1",
                "day_sin_1",
                "day_cos_1",
                "day_of_year_sin_1",
                "day_of_year_cos_1",
            ]
        ]
    ],
)
