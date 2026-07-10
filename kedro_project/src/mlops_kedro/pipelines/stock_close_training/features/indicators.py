import polars as pl
import polars_ta.talib as pl_ta


def calculate_indicators_for_symbol(df_symbol: pl.DataFrame) -> pl.DataFrame:
    expressions = [
        pl_ta.RSI(pl.col("close"), timeperiod=14).alias("RSI"),
        pl_ta.SMA(pl.col("close"), timeperiod=20).alias("SMA_Short"),
        pl_ta.SMA(pl.col("close"), timeperiod=50).alias("SMA_Long"),
        pl_ta.SMA(pl.col("volume"), timeperiod=20).alias("Vol_SMA"),
        pl.col("high")
        .cast(pl.Float64)
        .shift(1)
        .rolling_max(window_size=20)
        .alias("Range_high"),
        pl.col("low")
        .cast(pl.Float64)
        .shift(1)
        .rolling_min(window_size=20)
        .alias("Range_low"),
        pl_ta.ADX(
            pl.col("high"),
            pl.col("low"),
            pl.col("close"),
            timeperiod=14,
        ).alias("ADX"),
    ]

    return df_symbol.with_columns(expressions)
