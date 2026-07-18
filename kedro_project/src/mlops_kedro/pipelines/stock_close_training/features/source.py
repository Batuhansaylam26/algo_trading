from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import polars as pl

from .time_encoding import FourierTimeEncoder


@dataclass(slots=True)
class StockPriceFeatureSourceBuilder:
    time_encoder: FourierTimeEncoder

    def prepare(self, df: pl.DataFrame) -> pl.DataFrame:
        return self.time_encoder.add(
            df.select(
                pl.col("symbol").cast(pl.Utf8).str.to_uppercase(),
                pl.col("date").cast(pl.Datetime("us"), strict=False),
                pl.col("open").cast(pl.Float64, strict=False),
                pl.col("high").cast(pl.Float64, strict=False),
                pl.col("low").cast(pl.Float64, strict=False),
                pl.col("close").cast(pl.Float64, strict=False),
                pl.col("volume").cast(pl.Float64, strict=False),
            ),
            date_column="date",
            harmonics_key="close_model_harmonics",
            drop_date_parts=True,
        )

    @staticmethod
    def map_by_symbol(df: pl.DataFrame, function) -> pl.DataFrame:
        return df.group_by("symbol", maintain_order=True).map_groups(function)

    @staticmethod
    def add_calendar_gap_days(df_symbol: pl.DataFrame) -> pl.DataFrame:
        date_gaps = (
            df_symbol.select(pl.col("date").dt.date().alias("_event_date"))
            .unique()
            .sort("_event_date")
        )

        date_index = pl.col("_event_date").cast(pl.Int32)
        gap_expr = date_index.diff().fill_null(1) - 1
        date_gaps = date_gaps.with_columns(
            pl.when(gap_expr > 0)
            .then(gap_expr)
            .otherwise(0)
            .cast(pl.Int32)
            .alias("calendar_gap_days")
        )

        return (
            df_symbol.with_columns(pl.col("date").dt.date().alias("_event_date"))
            .join(date_gaps, on="_event_date", how="left")
            .drop("_event_date")
        )

    def add_time_encoding_for_symbol(self, df_symbol: pl.DataFrame) -> pl.DataFrame:
        return self.time_encoder.add(
            self.add_calendar_gap_days(df_symbol.sort("date")),
            date_column="date",
            harmonics_key="source_harmonics",
        ).sort("date")

    def add_model_training_tier_columns_for_symbol(
        self,
        df_symbol: pl.DataFrame,
    ) -> pl.DataFrame:
        return self.add_time_encoding_for_symbol(df_symbol).with_columns(
            pl.col("close").alias("target_close"),
            pl.col("open").shift(1).alias("prev_open"),
            pl.col("high").shift(1).alias("prev_high"),
            pl.col("low").shift(1).alias("prev_low"),
            pl.col("volume").shift(1).alias("prev_volume"),
        )

    @staticmethod
    def with_created_timestamp(df: pl.DataFrame) -> pl.DataFrame:
        created_timestamp = datetime.now(timezone.utc)
        return df.with_columns(pl.lit(created_timestamp).alias("created_timestamp"))

    @staticmethod
    def drop_rows_with_missing_model_features(
        df: pl.DataFrame,
        model_ready_columns: list[str],
    ) -> pl.DataFrame:
        return df.drop_nulls(model_ready_columns).filter(
            pl.all_horizontal(pl.col(model_ready_columns).is_finite())
        )
