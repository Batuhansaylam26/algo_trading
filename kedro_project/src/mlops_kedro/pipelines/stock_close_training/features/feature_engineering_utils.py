from datetime import datetime, timezone
from math import pi
from typing import Any

import polars as pl

def _harmonics(time_encoding_config: dict[str, Any], key: str) -> tuple[int, ...]:
    return tuple(time_encoding_config.get(key, [1, 2]))


def _period(time_encoding_config: dict[str, Any], column: str) -> float:
    return float(time_encoding_config.get("periods", {}).get(column))


def prepare_feature_source(
    df: pl.DataFrame,
    time_encoding_config: dict[str, Any],
) -> pl.DataFrame:
    return (
        df.select(
            pl.col("symbol").cast(pl.Utf8).str.to_uppercase(),
            pl.col("date").cast(pl.Datetime("us"), strict=False),
            pl.col("open").cast(pl.Float64, strict=False),
            pl.col("high").cast(pl.Float64, strict=False),
            pl.col("low").cast(pl.Float64, strict=False),
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
        )
        .with_columns(
            pl.col("date").dt.month().cast(pl.Int8).alias("month"),
            pl.col("date").dt.day().cast(pl.Int8).alias("day"),
            pl.col("date").dt.ordinal_day().cast(pl.Int16).alias("day_of_year"),
        )
        .with_columns(
            [
                *fourier_terms(
                    "month",
                    period=_period(time_encoding_config, "month"),
                    harmonics=_harmonics(time_encoding_config, "close_model_harmonics"),
                ),
                *fourier_terms(
                    "day",
                    period=_period(time_encoding_config, "day"),
                    harmonics=_harmonics(time_encoding_config, "close_model_harmonics"),
                ),
                *fourier_terms(
                    "day_of_year",
                    period=_period(time_encoding_config, "day_of_year"),
                    harmonics=_harmonics(time_encoding_config, "close_model_harmonics"),
                ),
            ]
        )
        .drop(["month", "day", "day_of_year"])
    )


def map_by_symbol(df: pl.DataFrame, function) -> pl.DataFrame:
    return df.group_by("symbol", maintain_order=True).map_groups(function)


def fourier_terms(
    column: str,
    period: float,
    harmonics: tuple[int, ...],
):
    expressions = []
    for harmonic in harmonics:
        angle = 2.0 * pi * harmonic * pl.col(column).cast(pl.Float64) / period
        expressions.extend(
            [
                angle.sin().alias(f"{column}_sin_{harmonic}"),
                angle.cos().alias(f"{column}_cos_{harmonic}"),
            ]
        )
    return expressions


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


def add_time_encoding_for_symbol(
    df_symbol: pl.DataFrame,
    time_encoding_config: dict[str, Any],
) -> pl.DataFrame:
    return (
        add_calendar_gap_days(df_symbol.sort("date"))
        .with_columns(
            pl.col("date").dt.month().cast(pl.Int8).alias("month"),
            pl.col("date").dt.day().cast(pl.Int8).alias("day"),
        )
        .with_columns(
            pl.col("date")
            .dt.ordinal_day()
            .cast(pl.Int16)
            .alias("day_of_year")
        )
        .with_columns(
            [
                *fourier_terms(
                    "month",
                    period=_period(time_encoding_config, "month"),
                    harmonics=_harmonics(time_encoding_config, "source_harmonics"),
                ),
                *fourier_terms(
                    "day",
                    period=_period(time_encoding_config, "day"),
                    harmonics=_harmonics(time_encoding_config, "source_harmonics"),
                ),
                *fourier_terms(
                    "day_of_year",
                    period=_period(time_encoding_config, "day_of_year"),
                    harmonics=_harmonics(time_encoding_config, "source_harmonics"),
                ),
            ]
        )
        .sort("date")
    )


def add_model_training_tier_columns_for_symbol(
    df_symbol: pl.DataFrame,
    time_encoding_config: dict[str, Any],
) -> pl.DataFrame:
    return add_time_encoding_for_symbol(df_symbol, time_encoding_config).with_columns(
        pl.col("close").alias("target_close"),
        pl.col("open").shift(1).alias("prev_open"),
        pl.col("close").shift(1).alias("prev_close"),
        pl.col("high").shift(1).alias("prev_high"),
        pl.col("low").shift(1).alias("prev_low"),
        pl.col("volume").shift(1).alias("prev_volume"),
    )


def with_created_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    created_timestamp = datetime.now(timezone.utc)
    return df.with_columns(pl.lit(created_timestamp).alias("created_timestamp"))


def drop_rows_with_missing_model_features(
    df: pl.DataFrame,
    model_ready_columns: list[str],
) -> pl.DataFrame:
    return df.drop_nulls(model_ready_columns).filter(
        pl.all_horizontal(pl.col(model_ready_columns).is_finite())
    )


def _add_close_model_time_encoding(
    df: pl.DataFrame,
    time_encoding_config: dict[str, Any],
) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.col("ds").dt.month().cast(pl.Int8).alias("month"),
            pl.col("ds").dt.day().cast(pl.Int8).alias("day"),
            pl.col("ds").dt.ordinal_day().cast(pl.Int16).alias("day_of_year"),
        )
        .with_columns(
            [
                *fourier_terms(
                    "month",
                    period=_period(time_encoding_config, "month"),
                    harmonics=_harmonics(time_encoding_config, "close_model_harmonics"),
                ),
                *fourier_terms(
                    "day",
                    period=_period(time_encoding_config, "day"),
                    harmonics=_harmonics(time_encoding_config, "close_model_harmonics"),
                ),
                *fourier_terms(
                    "day_of_year",
                    period=_period(time_encoding_config, "day_of_year"),
                    harmonics=_harmonics(time_encoding_config, "close_model_harmonics"),
                ),
            ]
        )
        .drop(["month", "day", "day_of_year"])
    )


def _fill_close_model_business_day_gaps(
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
        .explode("ds")
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


def to_stock_close_model_dataset(
    df: pl.DataFrame,
    output_columns: list[str],
    time_encoding_config: dict[str, Any],
) -> pl.DataFrame:
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
    return with_created_timestamp(
        _add_close_model_time_encoding(
            _fill_close_model_business_day_gaps(
                model_dataset,
                time_encoding_config.get("close_model_freq", "B"),
            ),
            time_encoding_config,
        )
    ).select(output_columns)
