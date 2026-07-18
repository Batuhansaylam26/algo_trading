import polars as pl
import pandera.polars as pa
from dagster import (
    AssetExecutionContext,
    AssetIn,
    Failure,
    MetadataValue,
    asset,
)
from pandera.errors import SchemaErrors
from pandera.engines.polars_engine import DateTime
from pandera.polars import PolarsData


def _column_gt_zero(data: PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(data.key) > 0)


def _column_ge_zero(data: PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col(data.key) >= 0)


def _high_gte_low(data: PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(pl.col("high") >= pl.col("low"))


def _unique_symbol_date(data: PolarsData) -> pl.LazyFrame:
    return data.lazyframe.select(
        pl.struct(["symbol", "date"]).is_duplicated().not_().all()
    )


STOCK_PRICE_SCHEMA = pa.DataFrameSchema(
    {
        "symbol": pa.Column(str, nullable=False),
        "date": pa.Column(DateTime(time_zone_agnostic=True), nullable=False),
        "open": pa.Column(float, checks=pa.Check(_column_gt_zero), nullable=False),
        "high": pa.Column(float, checks=pa.Check(_column_gt_zero), nullable=False),
        "low": pa.Column(float, checks=pa.Check(_column_gt_zero), nullable=False),
        "close": pa.Column(float, checks=pa.Check(_column_gt_zero), nullable=False),
        "volume": pa.Column(int, checks=pa.Check(_column_ge_zero), nullable=False),
    },
    checks=[
        pa.Check(_high_gte_low, name="high_gte_low"),
        pa.Check(_unique_symbol_date, name="unique_symbol_date"),
    ],
    coerce=True,
    strict=False,
)


def _standardize_stock_prices(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        pl.col("symbol").cast(pl.Utf8).str.to_uppercase(),
        pl.col("date").cast(pl.Datetime("us"), strict=False),
        pl.col("open").cast(pl.Float64, strict=False),
        pl.col("high").cast(pl.Float64, strict=False),
        pl.col("low").cast(pl.Float64, strict=False),
        pl.col("close").cast(pl.Float64, strict=False),
        pl.col("volume").cast(pl.Int64, strict=False),
    )


def _deduplicate_stock_prices(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.sort(["symbol", "date"])
        .unique(
            subset=["symbol", "date"],
            keep="last",
            maintain_order=True,
        )
        .sort(["symbol", "date"])
    )


def _add_calendar_columns(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        pl.col("date").dt.year().cast(pl.Int32).alias("year"),
        pl.col("date").dt.month().cast(pl.Int8).alias("month"),
        pl.col("date").dt.day().cast(pl.Int8).alias("day"),
    )


@asset(
    ins={"bronze_stock_prices": AssetIn(key=["bronze", "stock_prices"])},
    io_manager_key="delta_io_manager",
    key_prefix=["silver"],
    compute_kind="polars",
    group_name="staging",
)
def stock_prices(
    context: AssetExecutionContext,
    bronze_stock_prices: pl.DataFrame,
) -> pl.DataFrame:
    standardized = _standardize_stock_prices(bronze_stock_prices)
    deduplicated = _deduplicate_stock_prices(standardized)
    duplicate_count = len(standardized) - len(deduplicated)

    try:
        validated = STOCK_PRICE_SCHEMA.validate(deduplicated, lazy=True)
    except SchemaErrors as exc:
        raise Failure(
            description="Silver stock_prices validation failed.",
            metadata={
                "pandera_errors": MetadataValue.md(f"```text\n{exc}\n```"),
            },
        ) from exc

    silver_df = _add_calendar_columns(validated)

    context.add_output_metadata(
        {
            "input_rows": len(bronze_stock_prices),
            "silver_rows": len(silver_df),
            "duplicates_removed": duplicate_count,
            "calendar_columns": ["year", "month", "day"],
            "delta_path": "s3://delta-lake-bucket/silver/stock_prices",
        }
    )

    return silver_df


@asset(
    ins={
        "bronze_stock_prices_weekly": AssetIn(
            key=["bronze", "stock_prices_weekly"]
        )
    },
    io_manager_key="delta_io_manager",
    key_prefix=["silver"],
    compute_kind="polars",
    group_name="staging",
)
def stock_prices_weekly(
    context: AssetExecutionContext,
    bronze_stock_prices_weekly: pl.DataFrame,
) -> pl.DataFrame:
    standardized = _standardize_stock_prices(bronze_stock_prices_weekly)
    deduplicated = _deduplicate_stock_prices(standardized)
    duplicate_count = len(standardized) - len(deduplicated)

    try:
        validated = STOCK_PRICE_SCHEMA.validate(deduplicated, lazy=True)
    except SchemaErrors as exc:
        raise Failure(
            description="Silver stock_prices_weekly validation failed.",
            metadata={
                "pandera_errors": MetadataValue.md(f"```text\n{exc}\n```"),
            },
        ) from exc

    silver_df = _add_calendar_columns(validated)

    context.add_output_metadata(
        {
            "input_rows": len(bronze_stock_prices_weekly),
            "silver_rows": len(silver_df),
            "duplicates_removed": duplicate_count,
            "calendar_columns": ["year", "month", "day"],
            "delta_path": "s3://delta-lake-bucket/silver/stock_prices_weekly",
        }
    )

    return silver_df
