from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from .source import StockPriceFeatureSourceBuilder
from .time_encoding import FourierTimeEncoder


@dataclass(slots=True)
class CloseModelDatasetBuilder:
    columns_config: dict[str, list[str]]
    time_encoding_config: dict[str, Any]
    time_encoder: FourierTimeEncoder

    def build(self, df: pl.DataFrame) -> pl.DataFrame:
        model_dataset = (
            df.select(
                pl.col("symbol").cast(pl.Utf8).alias("unique_id"),
                pl.col("date")
                .cast(pl.Datetime("us"), strict=False)
                .dt.truncate("1d")
                .alias("ds"),
                pl.col("close").cast(pl.Float64, strict=False).alias("y"),
            )
            .drop_nulls(["unique_id", "ds", "y"])
            .unique(subset=["unique_id", "ds"], keep="last", maintain_order=True)
            .sort(["unique_id", "ds"])
        )
        return StockPriceFeatureSourceBuilder.with_created_timestamp(
            self.time_encoder.add(
                self.fill_business_day_gaps(
                    model_dataset,
                    self.time_encoding_config.get("close_model_freq", "B"),
                ),
                date_column="ds",
                harmonics_key="close_model_harmonics",
                drop_date_parts=True,
            )
        ).select(self.columns_config["close_model_dataset"])

    @staticmethod
    def fill_business_day_gaps(
        df: pl.DataFrame,
        close_model_freq: str,
    ) -> pl.DataFrame:
        if df.is_empty():
            return df

        interval = "1d" if close_model_freq.upper() == "B" else close_model_freq
        source = (
            df.with_row_index("_source_order")
            .sort(["unique_id", "ds", "_source_order"])
            .unique(subset=["unique_id", "ds"], keep="last", maintain_order=True)
            .drop("_source_order")
        )
        date_grid = (
            source.group_by("unique_id")
            .agg(
                pl.datetime_ranges(
                    pl.col("ds").min(),
                    pl.col("ds").max(),
                    interval=interval,
                ).alias("ds")
            )
            .explode("ds", empty_as_null=True)
        )
        if close_model_freq.upper() == "B":
            date_grid = date_grid.filter(pl.col("ds").dt.weekday() <= 5)

        return (
            date_grid.join(source, on=["unique_id", "ds"], how="left")
            .sort(["unique_id", "ds"])
            .with_columns(
                pl.col("y")
                .forward_fill()
                .backward_fill()
                .over("unique_id")
                .alias("y")
            )
            .select(["unique_id", "ds", "y"])
        )
