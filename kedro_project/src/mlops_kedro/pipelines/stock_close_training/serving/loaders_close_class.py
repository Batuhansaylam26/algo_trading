from __future__ import annotations

import pandas as pd
import polars as pl
import psycopg2
from feast import FeatureStore

from .connections import _timescale_connection_kwargs
from .constants import (
    CLOSE_MODEL_TIME_FEATURE_COLUMNS,
    FEATURE_REPO_DIR,
    TIMESCALE_CLOSE_MODEL_DATASET_TABLE,
    TIMESCALE_WRITE_BATCH_SIZE,
)
from .definitions import _apply_close_model_dataset_definition
from .transforms import _close_model_ds_key, _close_model_series_key


class CloseModelFeatureLoader:

    @staticmethod
    def _read_close_model_entity_rows() -> pd.DataFrame:
        query = f"""
            SELECT unique_id, ds
            FROM {TIMESCALE_CLOSE_MODEL_DATASET_TABLE}
            ORDER BY unique_id, ds;
        """

        try:
            with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
                entity_df = pd.read_sql_query(query, connection)
        except psycopg2.Error as error:
            if "does not exist" in str(error):
                return pd.DataFrame()
            raise

        if entity_df.empty:
            return entity_df

        entity_df["ds"] = pd.to_datetime(entity_df["ds"], utc=True)
        entity_df["ds_key"] = _close_model_ds_key(entity_df["ds"])
        entity_df["series_key"] = _close_model_series_key(
            entity_df["unique_id"],
            entity_df["ds_key"],
        )
        entity_df["event_timestamp"] = entity_df["ds"]
        return entity_df[["series_key", "unique_id", "ds", "event_timestamp"]]

    @staticmethod
    def _read_close_model_online_entity_rows() -> pd.DataFrame:
        query = f"""
            SELECT unique_id, ds
            FROM {TIMESCALE_CLOSE_MODEL_DATASET_TABLE}
            ORDER BY unique_id, ds;
        """

        with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
            entity_df = pd.read_sql_query(query, connection)

        if entity_df.empty:
            return entity_df

        entity_df["ds"] = pd.to_datetime(entity_df["ds"], utc=True)
        entity_df["ds_key"] = _close_model_ds_key(entity_df["ds"])
        entity_df["series_key"] = _close_model_series_key(
            entity_df["unique_id"],
            entity_df["ds_key"],
        )
        return entity_df[["series_key", "unique_id", "ds"]]

    @staticmethod
    def load_stock_close_model_dataset_from_feast() -> pl.DataFrame:
        _apply_close_model_dataset_definition()
        entity_df = CloseModelFeatureLoader._read_close_model_entity_rows()

        if entity_df.empty:
            return pl.DataFrame(
                schema={
                    "unique_id": pl.Utf8,
                    "ds": pl.Datetime("us"),
                    "y": pl.Float64,
                    "month_sin_1": pl.Float64,
                    "month_cos_1": pl.Float64,
                    "day_sin_1": pl.Float64,
                    "day_cos_1": pl.Float64,
                    "day_of_year_sin_1": pl.Float64,
                    "day_of_year_cos_1": pl.Float64,
                }
            )

        store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
        historical_features = store.get_historical_features(
            entity_df=entity_df[["series_key", "event_timestamp"]],
            features=[
                f"stock_close_model_dataset:{feature_name}"
                for feature_name in ["y", *CLOSE_MODEL_TIME_FEATURE_COLUMNS]
            ],
        ).to_df()

        if "event_timestamp" in historical_features:
            historical_features = historical_features.drop(columns=["event_timestamp"])

        historical_features = historical_features.merge(
            entity_df[["series_key", "unique_id", "ds"]],
            on="series_key",
            how="left",
        )

        return (
            pl.from_pandas(historical_features)
            .select(
                pl.col("unique_id").cast(pl.Utf8),
                pl.col("ds").cast(pl.Datetime("us"), strict=False),
                pl.col("y").cast(pl.Float64, strict=False),
                pl.col("month_sin_1").cast(pl.Float64, strict=False),
                pl.col("month_cos_1").cast(pl.Float64, strict=False),
                pl.col("day_sin_1").cast(pl.Float64, strict=False),
                pl.col("day_cos_1").cast(pl.Float64, strict=False),
                pl.col("day_of_year_sin_1").cast(pl.Float64, strict=False),
                pl.col("day_of_year_cos_1").cast(pl.Float64, strict=False),
            )
            .sort(["unique_id", "ds"])
        )

    @staticmethod
    def _empty_close_model_dataset_frame() -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "unique_id": pl.Utf8,
                "ds": pl.Datetime("us"),
                "y": pl.Float64,
                "month_sin_1": pl.Float64,
                "month_cos_1": pl.Float64,
                "day_sin_1": pl.Float64,
                "day_cos_1": pl.Float64,
                "day_of_year_sin_1": pl.Float64,
                "day_of_year_cos_1": pl.Float64,
            }
        )

    @staticmethod
    def _online_entity_batches(entity_df: pd.DataFrame):
        for start in range(0, len(entity_df), TIMESCALE_WRITE_BATCH_SIZE):
            batch_df = entity_df.iloc[start : start + TIMESCALE_WRITE_BATCH_SIZE].copy()
            yield batch_df, batch_df[["series_key"]].to_dict("records")

    @staticmethod
    def load_stock_close_model_dataset_from_feast_online() -> pl.DataFrame:
        _apply_close_model_dataset_definition()
        entity_df = CloseModelFeatureLoader._read_close_model_online_entity_rows()
        if entity_df.empty:
            return CloseModelFeatureLoader._empty_close_model_dataset_frame()

        store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))
        feature_refs = [
            f"stock_close_model_dataset:{feature_name}"
            for feature_name in ["y", *CLOSE_MODEL_TIME_FEATURE_COLUMNS]
        ]
        frames = []
        for batch_df, entity_rows in CloseModelFeatureLoader._online_entity_batches(entity_df):
            online_features = store.get_online_features(
                features=feature_refs,
                entity_rows=entity_rows,
            ).to_df()
            if not online_features.empty:
                online_features = online_features.merge(
                    batch_df[["series_key", "unique_id", "ds"]],
                    on="series_key",
                    how="left",
                )
                frames.append(pl.from_pandas(online_features))

        if not frames:
            return CloseModelFeatureLoader._empty_close_model_dataset_frame()

        return (
            pl.concat(frames, how="vertical_relaxed")
            .select(
                pl.col("unique_id").cast(pl.Utf8),
                pl.col("ds").cast(pl.Datetime("us"), strict=False),
                pl.col("y").cast(pl.Float64, strict=False),
                pl.col("month_sin_1").cast(pl.Float64, strict=False),
                pl.col("month_cos_1").cast(pl.Float64, strict=False),
                pl.col("day_sin_1").cast(pl.Float64, strict=False),
                pl.col("day_cos_1").cast(pl.Float64, strict=False),
                pl.col("day_of_year_sin_1").cast(pl.Float64, strict=False),
                pl.col("day_of_year_cos_1").cast(pl.Float64, strict=False),
            )
            .sort(["unique_id", "ds"])
        )

    @staticmethod
    def load_stock_close_model_dataset_from_redis() -> pl.DataFrame:
        return CloseModelFeatureLoader.load_stock_close_model_dataset_from_feast_online()

    @staticmethod
    def load_stock_close_model_dataset_from_timescale() -> pl.DataFrame:
        columns = ["unique_id", "ds", "y", *CLOSE_MODEL_TIME_FEATURE_COLUMNS]
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        query = f"""
            SELECT {quoted_columns}
            FROM {TIMESCALE_CLOSE_MODEL_DATASET_TABLE}
            ORDER BY unique_id, ds;
        """

        rows = []
        try:
            with psycopg2.connect(**_timescale_connection_kwargs()) as connection:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    while batch := cursor.fetchmany(TIMESCALE_WRITE_BATCH_SIZE):
                        rows.extend(batch)
        except psycopg2.Error as error:
            if "does not exist" in str(error):
                return CloseModelFeatureLoader._empty_close_model_dataset_frame()
            raise

        if not rows:
            return CloseModelFeatureLoader._empty_close_model_dataset_frame()

        return (
            pl.DataFrame(rows, schema=columns, orient="row")
            .select(
                pl.col("unique_id").cast(pl.Utf8),
                pl.col("ds").cast(pl.Datetime("us"), strict=False),
                pl.col("y").cast(pl.Float64, strict=False),
                pl.col("month_sin_1").cast(pl.Float64, strict=False),
                pl.col("month_cos_1").cast(pl.Float64, strict=False),
                pl.col("day_sin_1").cast(pl.Float64, strict=False),
                pl.col("day_cos_1").cast(pl.Float64, strict=False),
                pl.col("day_of_year_sin_1").cast(pl.Float64, strict=False),
                pl.col("day_of_year_cos_1").cast(pl.Float64, strict=False),
            )
            .sort(["unique_id", "ds"])
        )
