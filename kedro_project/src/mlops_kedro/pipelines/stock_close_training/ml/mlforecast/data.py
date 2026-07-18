from __future__ import annotations

import pandas as pd
import polars as pl

from ..common import (
    model_id_columns,
    non_feature_columns,
    split_train_test_by_horizon,
)


class MLForecastDataPreparer:
    def to_dataset(self, df: pl.DataFrame) -> pl.DataFrame:
        exogenous_columns = [
            column for column in df.columns if column not in non_feature_columns()
        ]

        if {"unique_id", "ds", "y"}.issubset(set(df.columns)):
            return (
                df.select(
                    pl.col("unique_id").cast(pl.Utf8),
                    pl.col("ds").cast(pl.Datetime("us"), strict=False),
                    pl.col("y").cast(pl.Float64, strict=False),
                    *[
                        pl.col(column).cast(pl.Float64, strict=False)
                        for column in exogenous_columns
                    ],
                )
                .sort(["unique_id", "ds"])
            )

        if not {"symbol", "date", "close"}.issubset(set(df.columns)):
            raise ValueError(
                "MLForecast training data needs either unique_id/ds/y "
                "or symbol/date/close columns."
            )

        return (
            df.select(
                pl.col("symbol").cast(pl.Utf8).alias("unique_id"),
                pl.col("date").cast(pl.Datetime("us"), strict=False).alias("ds"),
                pl.col("close").cast(pl.Float64, strict=False).alias("y"),
                *[
                    pl.col(column).cast(pl.Float64, strict=False)
                    for column in exogenous_columns
                ],
            )
            .sort(["unique_id", "ds"])
        )

    def fill_business_day_gaps(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        feature_columns = [
            column for column in df.columns if column not in model_id_columns()
        ]
        source = (
            df.with_row_index("_source_order")
            .sort(["unique_id", "ds", "_source_order"])
            .unique(subset=["unique_id", "ds"], keep="last", maintain_order=True)
            .drop("_source_order")
        )
        filled = (
            self._business_day_grid(source)
            .join(source, on=["unique_id", "ds"], how="left")
            .with_columns(pl.col("y").is_null().alias("_synthetic_row"))
            .sort(["unique_id", "ds"])
        )
        if "calendar_gap_days" in feature_columns:
            filled = self._add_business_gap_run(filled)

        fill_columns = ["y", *feature_columns]
        filled = filled.with_columns(
            [
                pl.col(column)
                .forward_fill()
                .backward_fill()
                .over("unique_id")
                .alias(column)
                for column in fill_columns
                if column != "calendar_gap_days"
            ]
        )
        if "calendar_gap_days" in feature_columns:
            filled = self._fill_calendar_gap_days(filled)

        return (
            filled.drop(
                [
                    column
                    for column in [
                        "_synthetic_row",
                        "_source_row",
                        "_actual_row",
                        "_actual_segment",
                        "_business_gap_run",
                    ]
                    if column in filled.columns
                ]
            )
            .drop_nulls(["unique_id", "ds", "y", *feature_columns])
            .filter(pl.all_horizontal(pl.col(["y", *feature_columns]).is_finite()))
            .select(["unique_id", "ds", "y", *feature_columns])
            .sort(["unique_id", "ds"])
        )

    def to_frame(self, df: pl.DataFrame) -> pd.DataFrame:
        model_df = self.fill_business_day_gaps(self.to_dataset(df))
        return model_df.to_pandas()

    def make_train_test_split(
        self,
        dataset: pl.DataFrame,
        test_horizon: int,
    ) -> dict[str, pd.DataFrame]:
        model_df = self.to_frame(dataset)
        train_df, test_df = split_train_test_by_horizon(model_df, test_horizon)
        return {
            "full": model_df,
            "train": train_df,
            "test": test_df,
        }

    @staticmethod
    def _business_day_grid(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.group_by("unique_id")
            .agg(
                pl.datetime_ranges(
                    pl.col("ds").min(),
                    pl.col("ds").max(),
                    interval="1d",
                ).alias("ds")
            )
            .explode("ds")
            .filter(pl.col("ds").dt.weekday() <= 5)
        )

    @staticmethod
    def _add_business_gap_run(filled: pl.DataFrame) -> pl.DataFrame:
        return (
            filled.with_columns((~pl.col("_synthetic_row")).alias("_actual_row"))
            .with_columns(
                pl.col("_actual_row")
                .cast(pl.Int64)
                .cum_sum()
                .over("unique_id")
                .alias("_actual_segment")
            )
            .with_columns(
                (
                    pl.col("ds").cum_count().over(["unique_id", "_actual_segment"])
                    - 1
                )
                .cast(pl.Int64)
                .alias("_business_gap_run")
            )
        )

    @staticmethod
    def _fill_calendar_gap_days(filled: pl.DataFrame) -> pl.DataFrame:
        return filled.with_columns(
            pl.when(pl.col("_synthetic_row"))
            .then(pl.col("_business_gap_run"))
            .when(pl.col("_business_gap_run").shift(1).over("unique_id") > 0)
            .then(pl.col("_business_gap_run").shift(1).over("unique_id"))
            .otherwise(pl.col("calendar_gap_days"))
            .fill_null(0)
            .cast(pl.Float64)
            .alias("calendar_gap_days")
        )


def to_mlforecast_dataset(df: pl.DataFrame) -> pl.DataFrame:
    return MLForecastDataPreparer().to_dataset(df)

def _business_day_grid(df: pl.DataFrame) -> pl.DataFrame:
    return MLForecastDataPreparer._business_day_grid(df)

def fill_mlforecast_business_day_gaps(df: pl.DataFrame) -> pl.DataFrame:
    return MLForecastDataPreparer().fill_business_day_gaps(df)

def to_mlforecast_frame(df: pl.DataFrame) -> pd.DataFrame:
    return MLForecastDataPreparer().to_frame(df)

def make_train_test_split(
    dataset: pl.DataFrame,
    test_horizon: int,
) -> dict[str, pd.DataFrame]:
    return MLForecastDataPreparer().make_train_test_split(dataset, test_horizon)
