from __future__ import annotations

import pandas as pd
import polars as pl
import psycopg2
from feast import FeatureStore

from .connections import _ensure_feature_repo_on_path, _timescale_connection_kwargs
from .constants import (
    FEATURE_REPO_DIR,
    MODEL_TIER_FEATURE_COLUMNS,
    TIER_2_FEATURE_COLUMNS,
    TIMESCALE_TABLE,
    TIMESCALE_WRITE_BATCH_SIZE,
)
from .definitions import (
    _apply_close_model_dataset_definition,
    _apply_model_feature_definitions,
)
from .loaders_close import load_stock_close_model_dataset_from_feast_online
from .transforms import _model_feature_date_key, _model_feature_key


def _empty_model_training_dataset_frame(
    feature_columns: list[str],
) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "unique_id": pl.Utf8,
            "ds": pl.Datetime("us"),
            "y": pl.Float64,
            **{column: pl.Float64 for column in feature_columns},
        }
    )

def _read_tier2_online_entity_rows() -> pd.DataFrame:
    query = f"""
        SELECT symbol, "date"
        FROM {TIMESCALE_TABLE}
        ORDER BY symbol, "date";
    """

    with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
        entity_df = pd.read_sql_query(query, connection)

    if entity_df.empty:
        return entity_df

    entity_df["date"] = pd.to_datetime(entity_df["date"], utc=True)
    entity_df["date_key"] = _model_feature_date_key(entity_df["date"])
    entity_df["feature_key"] = _model_feature_key(
        entity_df["symbol"],
        entity_df["date_key"],
    )
    return entity_df[["feature_key", "symbol", "date"]]

def _tier2_online_entity_batches(entity_df: pd.DataFrame):
    for start in range(0, len(entity_df), TIMESCALE_WRITE_BATCH_SIZE):
        batch_df = entity_df.iloc[start : start + TIMESCALE_WRITE_BATCH_SIZE].copy()
        yield batch_df, batch_df[["feature_key"]].to_dict("records")

def load_stock_model_training_dataset_from_feast_online(
    feature_columns: list[str],
) -> pl.DataFrame:
    _apply_close_model_dataset_definition()
    _apply_model_feature_definitions()
    close_dataset = load_stock_close_model_dataset_from_feast_online()
    entity_df = _read_tier2_online_entity_rows()

    if entity_df.empty or close_dataset.is_empty():
        return _empty_model_training_dataset_frame(feature_columns)

    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    feature_refs = [
        f"stock_model_tier2_dataset:{feature_name}"
        for feature_name in feature_columns
    ]
    frames = []
    for batch_df, entity_rows in _tier2_online_entity_batches(entity_df):
        online_features = store.get_online_features(
            features=feature_refs,
            entity_rows=entity_rows,
        ).to_df()
        if not online_features.empty:
            online_features = online_features.merge(
                batch_df[["feature_key", "symbol", "date"]],
                on="feature_key",
                how="left",
            )
            frames.append(pl.from_pandas(online_features))

    if not frames:
        return _empty_model_training_dataset_frame(feature_columns)

    model_features = (
        pl.concat(frames, how="vertical_relaxed")
        .select(
            pl.col("symbol").cast(pl.Utf8).alias("unique_id"),
            pl.col("date").cast(pl.Datetime("us"), strict=False).alias("ds"),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in feature_columns
            ],
        )
        .sort(["unique_id", "ds"])
    )

    return (
        close_dataset.join(
            model_features,
            on=["unique_id", "ds"],
            how="inner",
        )
        .select(["unique_id", "ds", "y", *feature_columns])
        .drop_nulls(["unique_id", "ds", "y", *feature_columns])
        .unique(subset=["unique_id", "ds"], keep="last", maintain_order=True)
        .sort(["unique_id", "ds"])
    )

def load_stock_tier1_model_dataset_from_feast_online() -> pl.DataFrame:
    return load_stock_model_training_dataset_from_feast_online(
        MODEL_TIER_FEATURE_COLUMNS["tier1"]
    )

def load_stock_tier2_model_dataset_from_feast_online() -> pl.DataFrame:
    return load_stock_model_training_dataset_from_feast_online(
        MODEL_TIER_FEATURE_COLUMNS["tier2"]
    )

def load_stock_tier3_model_dataset_from_feast_online() -> pl.DataFrame:
    return load_stock_model_training_dataset_from_feast_online(
        MODEL_TIER_FEATURE_COLUMNS["tier3"]
    )

def get_online_model_features(
    symbols: list[str],
    feature_columns: list[str] | None = None,
) -> pl.DataFrame:
    if not symbols:
        return pl.DataFrame()

    _ensure_feature_repo_on_path()
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
    feature_columns = feature_columns or TIER_2_FEATURE_COLUMNS
    online_features = store.get_online_features(
        features=[
            f"stock_model_features:{feature_name}"
            for feature_name in feature_columns
        ],
        entity_rows=[
            {"symbol": symbol}
            for symbol in sorted(set(symbols))
        ],
    ).to_df()

    return pl.from_pandas(online_features)
