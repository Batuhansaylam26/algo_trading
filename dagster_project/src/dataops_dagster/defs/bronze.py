import pandas as pd
import polars as pl
from dagster import AssetExecutionContext, MetadataValue, asset
from yahooquery import Ticker


TICKER_LIST = ["AAPL", "BMW.DE"]
DEFAULT_PERIOD = "max"


def _load_yahooquery_history(
    *,
    period: str,
    interval: str,
    asset_name: str,
) -> pl.DataFrame:
    ticker = Ticker(TICKER_LIST)
    pandas_df = ticker.history(period=period, interval=interval).reset_index()

    if pandas_df.empty:
        raise ValueError(f"YahooQuery returned no rows for {asset_name}.")

    pandas_df["date"] = pd.to_datetime(pandas_df["date"], utc=True).dt.tz_localize(None)
    ingested_at = pd.Timestamp.now(tz="UTC").tz_localize(None).to_pydatetime()

    return pl.from_pandas(pandas_df).with_columns(
        pl.lit("yahooquery").alias("_source"),
        pl.lit(interval).alias("_interval"),
        pl.lit(ingested_at).alias("_ingested_at"),
    )


def _stock_price_metadata(
    df: pl.DataFrame,
    *,
    interval: str,
    delta_path: str,
) -> dict:
    return {
        "rows": len(df),
        "interval": interval,
        "symbols": MetadataValue.json(sorted(df["symbol"].unique().to_list())),
        "delta_path": delta_path,
    }


@asset(
    io_manager_key="delta_io_manager",
    key_prefix=["bronze"],
    compute_kind="polars",
    group_name="ingestion",
)
def stock_prices(context: AssetExecutionContext) -> pl.DataFrame:
    df = _load_yahooquery_history(
        period=DEFAULT_PERIOD,
        interval="1d",
        asset_name="stock_prices",
    )

    context.add_output_metadata(
        _stock_price_metadata(
            df,
            interval="1d",
            delta_path="s3://delta-lake-bucket/bronze/stock_prices",
        )
    )

    return df


@asset(
    io_manager_key="delta_io_manager",
    key_prefix=["bronze"],
    compute_kind="polars",
    group_name="ingestion",
)
def stock_prices_weekly(context: AssetExecutionContext) -> pl.DataFrame:
    df = _load_yahooquery_history(
        period=DEFAULT_PERIOD,
        interval="1wk",
        asset_name="stock_prices_weekly",
    )

    context.add_output_metadata(
        _stock_price_metadata(
            df,
            interval="1wk",
            delta_path="s3://delta-lake-bucket/bronze/stock_prices_weekly",
        )
    )

    return df
