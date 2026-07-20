import pandas as pd
import polars as pl
from dagster import AssetExecutionContext, MetadataValue, asset
from yahooquery import Ticker


TICKER_LIST = ["AAPL", "BMW.DE"]
DEFAULT_PERIOD = "max"


class BronzeStockPriceAssets:
    @staticmethod
    def load_yahooquery_history(
        *,
        period: str,
        interval: str,
        asset_name: str,
    ) -> pl.DataFrame:
        ticker = Ticker(TICKER_LIST)
        pandas_df = ticker.history(period=period, interval=interval).reset_index()

        if pandas_df.empty:
            raise ValueError(f"YahooQuery returned no rows for {asset_name}.")

        pandas_df["date"] = pd.to_datetime(
            pandas_df["date"],
            utc=True,
        ).dt.tz_localize(None)
        ingested_at = pd.Timestamp.now(tz="UTC").tz_localize(None).to_pydatetime()

        return pl.from_pandas(pandas_df).with_columns(
            pl.lit("yahooquery").alias("_source"),
            pl.lit(interval).alias("_interval"),
            pl.lit(ingested_at).alias("_ingested_at"),
        )

    @staticmethod
    def stock_price_metadata(
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

    @staticmethod
    def stock_prices(context: AssetExecutionContext) -> pl.DataFrame:
        df = BronzeStockPriceAssets.load_yahooquery_history(
            period=DEFAULT_PERIOD,
            interval="1d",
            asset_name="BronzeStockPriceAssets.stock_prices",
        )

        context.add_output_metadata(
            BronzeStockPriceAssets.stock_price_metadata(
                df,
                interval="1d",
                delta_path="s3://delta-lake-bucket/bronze/BronzeStockPriceAssets.stock_prices",
            )
        )

        return df

    @staticmethod
    def stock_prices_weekly(context: AssetExecutionContext) -> pl.DataFrame:
        df = BronzeStockPriceAssets.load_yahooquery_history(
            period=DEFAULT_PERIOD,
            interval="1wk",
            asset_name="BronzeStockPriceAssets.stock_prices_weekly",
        )

        context.add_output_metadata(
            BronzeStockPriceAssets.stock_price_metadata(
                df,
                interval="1wk",
                delta_path="s3://delta-lake-bucket/bronze/BronzeStockPriceAssets.stock_prices_weekly",
            )
        )

        return df
