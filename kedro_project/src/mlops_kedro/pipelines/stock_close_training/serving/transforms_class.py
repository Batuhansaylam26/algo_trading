from __future__ import annotations

from datetime import datetime, timezone
from math import pi

import pandas as pd
import polars as pl

from .constants import (
    CLOSE_MODEL_DATASET_COLUMNS,
    FEAST_OFFLINE_COLUMNS,
    PECNET_PREPROCESSED_FEAST_COLUMNS,
    TIMESCALE_DAILY_FILL_FREQ,
    TIMESCALE_WRITE_BATCH_SIZE,
)


class FeatureStoreTransforms:

    @staticmethod
    def _to_pandas_for_feature_store(df: pl.DataFrame) -> pd.DataFrame:
        pdf = df.select(FEAST_OFFLINE_COLUMNS).to_pandas()
        pdf["date"] = pd.to_datetime(pdf["date"], utc=True)
        pdf["created_timestamp"] = pd.to_datetime(pdf["created_timestamp"], utc=True)
        return pdf.where(pd.notnull(pdf), None)

    @staticmethod
    def _model_feature_date_key(date: pd.Series) -> pd.Series:
        return pd.to_datetime(date, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    @staticmethod
    def _model_feature_key(symbol: pd.Series, date_key: pd.Series) -> pd.Series:
        return symbol.astype(str) + "|" + date_key.astype(str)

    @staticmethod
    def _to_pandas_for_tier2_feature_dataset(df: pl.DataFrame) -> pd.DataFrame:
        pdf = df.select(FEAST_OFFLINE_COLUMNS).to_pandas()
        pdf["date"] = pd.to_datetime(pdf["date"], utc=True)
        pdf["date_key"] = FeatureStoreTransforms._model_feature_date_key(pdf["date"])
        pdf["feature_key"] = FeatureStoreTransforms._model_feature_key(pdf["symbol"], pdf["date_key"])
        pdf["created_timestamp"] = pd.to_datetime(pdf["created_timestamp"], utc=True)
        return pdf.where(pd.notnull(pdf), None)

    @staticmethod
    def _iter_polars_row_batches(
        df: pl.DataFrame,
        columns: list[str],
        batch_size: int = TIMESCALE_WRITE_BATCH_SIZE,
    ):
        selected = df.select(columns)
        for batch in selected.iter_slices(n_rows=batch_size):
            rows = list(batch.iter_rows(named=False))
            if rows:
                yield rows

    @staticmethod
    def _fill_interval() -> str:
        if TIMESCALE_DAILY_FILL_FREQ.upper() == "B":
            return "1d"
        return TIMESCALE_DAILY_FILL_FREQ

    @staticmethod
    def _time_encoding_expressions(
        time_column: str,
        output_columns: list[str],
    ) -> list[pl.Expr]:
        expressions = []
        time_parts = {
            "month": (pl.col(time_column).dt.month(), 12.0),
            "day": (pl.col(time_column).dt.day(), 31.0),
            "day_of_year": (pl.col(time_column).dt.ordinal_day(), 366.0),
        }

        for column_name, (time_expr, period) in time_parts.items():
            for harmonic in (1, 2):
                angle = 2.0 * pi * harmonic * time_expr.cast(pl.Float64) / period
                sin_column = f"{column_name}_sin_{harmonic}"
                cos_column = f"{column_name}_cos_{harmonic}"
                if sin_column in output_columns:
                    expressions.append(angle.sin().alias(sin_column))
                if cos_column in output_columns:
                    expressions.append(angle.cos().alias(cos_column))

        return expressions

    @staticmethod
    def _fill_daily_gaps(
        df: pl.DataFrame,
        *,
        id_column: str,
        time_column: str,
        output_columns: list[str],
        preserve_calendar_gap_days: bool = False,
    ) -> pl.DataFrame:
        if df.is_empty():
            return df.select(output_columns)

        source = (
            df.select(output_columns)
            .with_columns(
                pl.col(time_column)
                .cast(pl.Datetime("us"), strict=False)
                .dt.truncate("1d")
                .alias(time_column)
            )
            .with_row_index("_source_order")
            .sort([id_column, time_column, "_source_order"])
            .unique(subset=[id_column, time_column], keep="last", maintain_order=True)
            .drop("_source_order")
            .with_columns(pl.lit(True).alias("_source_row"))
        )

        date_grid = (
            source.group_by(id_column)
            .agg(
                pl.datetime_ranges(
                    pl.col(time_column).min(),
                    pl.col(time_column).max(),
                    interval=FeatureStoreTransforms._fill_interval(),
                ).alias(time_column)
            )
            .explode(time_column)
        )

        if TIMESCALE_DAILY_FILL_FREQ.upper() == "B":
            date_grid = date_grid.filter(pl.col(time_column).dt.weekday() <= 5)

        filled = (
            date_grid.join(source, on=[id_column, time_column], how="left")
            .with_columns(pl.col("_source_row").is_null().alias("_synthetic_row"))
            .sort([id_column, time_column])
        )
        if preserve_calendar_gap_days and "calendar_gap_days" in output_columns:
            filled = (
                filled.with_columns(
                    (~pl.col("_synthetic_row")).alias("_actual_row")
                )
                .with_columns(
                    pl.col("_actual_row")
                    .cast(pl.Int64)
                    .cum_sum()
                    .over(id_column)
                    .alias("_actual_segment")
                )
                .with_columns(
                    (
                        pl.col(time_column)
                        .cum_count()
                        .over([id_column, "_actual_segment"])
                        - 1
                    )
                    .cast(pl.Int64)
                    .alias("_business_gap_run")
                )
            )

        fill_columns = [
            column
            for column in output_columns
            if column not in {id_column, time_column, "created_timestamp"}
            and not (
                preserve_calendar_gap_days
                and column == "calendar_gap_days"
            )
        ]
        fill_expressions = [
            pl.col(column)
            .forward_fill()
            .backward_fill()
            .over(id_column)
            .alias(column)
            for column in fill_columns
        ]

        if preserve_calendar_gap_days and "calendar_gap_days" in output_columns:
            fill_expressions.append(
                pl.when(pl.col("_synthetic_row"))
                .then(pl.col("_business_gap_run"))
                .when(pl.col("_business_gap_run").shift(1).over(id_column) > 0)
                .then(pl.col("_business_gap_run").shift(1).over(id_column))
                .otherwise(pl.col("calendar_gap_days"))
                .fill_null(0)
                .cast(pl.Int32)
                .alias("calendar_gap_days")
            )

        if "created_timestamp" in output_columns:
            fill_expressions.append(
                pl.lit(datetime.now(timezone.utc)).alias("created_timestamp")
            )

        filled = filled.with_columns(fill_expressions)
        time_expressions = FeatureStoreTransforms._time_encoding_expressions(time_column, output_columns)
        if time_expressions:
            filled = filled.with_columns(time_expressions)

        return filled.select(output_columns)

    @staticmethod
    def _fill_model_feature_daily_gaps(df: pl.DataFrame) -> pl.DataFrame:
        return FeatureStoreTransforms._fill_daily_gaps(
            df,
            id_column="symbol",
            time_column="date",
            output_columns=FEAST_OFFLINE_COLUMNS,
            preserve_calendar_gap_days=True,
        )

    @staticmethod
    def _fill_close_model_dataset_daily_gaps(df: pl.DataFrame) -> pl.DataFrame:
        return FeatureStoreTransforms._fill_daily_gaps(
            df,
            id_column="unique_id",
            time_column="ds",
            output_columns=CLOSE_MODEL_DATASET_COLUMNS,
        )

    @staticmethod
    def _to_pandas_for_close_model_dataset(df: pl.DataFrame) -> pd.DataFrame:
        pdf = df.select(CLOSE_MODEL_DATASET_COLUMNS).to_pandas()
        pdf["ds"] = pd.to_datetime(pdf["ds"], utc=True)
        pdf["ds_key"] = FeatureStoreTransforms._close_model_ds_key(pdf["ds"])
        pdf["series_key"] = FeatureStoreTransforms._close_model_series_key(pdf["unique_id"], pdf["ds_key"])
        pdf["created_timestamp"] = pd.to_datetime(pdf["created_timestamp"], utc=True)
        return pdf.where(pd.notnull(pdf), None)

    @staticmethod
    def _close_model_ds_key(ds: pd.Series) -> pd.Series:
        return pd.to_datetime(ds, utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")

    @staticmethod
    def _close_model_series_key(unique_id: pd.Series, ds_key: pd.Series) -> pd.Series:
        return unique_id.astype(str) + "|" + ds_key.astype(str)

    @staticmethod
    def _to_pandas_for_pecnet_preprocessed(df: pl.DataFrame) -> pd.DataFrame:
        pdf = df.select(PECNET_PREPROCESSED_FEAST_COLUMNS).to_pandas()
        pdf["event_timestamp"] = pd.to_datetime(pdf["event_timestamp"], utc=True)
        pdf["created_timestamp"] = pd.to_datetime(pdf["created_timestamp"], utc=True)
        return pdf.where(pd.notnull(pdf), None)
