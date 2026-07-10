import pandas as pd
import polars as pl
from dagster import AssetExecutionContext, MetadataValue, asset
from yahooquery import Ticker


TICKER_LIST = ["AAPL", "BMW.DE"]


@asset(
    io_manager_key="delta_io_manager",
    key_prefix=["bronze"],
    compute_kind="polars",
    group_name="ingestion",
)
def stock_prices(context: AssetExecutionContext) -> pl.DataFrame:
    ticker = Ticker(TICKER_LIST)
    pandas_df = ticker.history(period="max", interval="1d").reset_index()

    if pandas_df.empty:
        raise ValueError("YahooQuery returned no rows for stock_prices.")

    pandas_df["date"] = pd.to_datetime(pandas_df["date"], utc=True).dt.tz_localize(None)

    ingested_at = pd.Timestamp.now(tz="UTC").tz_localize(None).to_pydatetime()

    df = pl.from_pandas(pandas_df).with_columns(
        pl.lit("yahooquery").alias("_source"),
        pl.lit(ingested_at).alias("_ingested_at"),
    )

    context.add_output_metadata(
        {
            "rows": len(df),
            "symbols": MetadataValue.json(sorted(df["symbol"].unique().to_list())),
            "delta_path": "s3://delta-lake-bucket/bronze/stock_prices",
        }
    )

    return df
