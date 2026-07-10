import polars as pl


def calculate_conditions_for_symbol(
    df_symbol: pl.DataFrame,
    condition_columns: list[str],
) -> pl.DataFrame:
    expressions = [
        (pl.col("low") > pl.col("high").shift(1)).alias("gap_up"),
        (pl.col("high") < pl.col("low").shift(1)).alias("gap_down"),
        (pl.col("open") > pl.col("close").shift(1)).alias(
            "exhaustion_gap_up"
        ),
        (pl.col("open") < pl.col("close").shift(1)).alias(
            "exhaustion_gap_down"
        ),
        (pl.col("volume") > (1.5 * pl.col("Vol_SMA"))).alias("high_vol"),
        (pl.col("volume") > (2.0 * pl.col("Vol_SMA"))).alias("extreme_vol"),
        (pl.col("low") > pl.col("Range_high")).alias("breakout_up"),
        (pl.col("high") < pl.col("Range_low")).alias("breakout_down"),
        (pl.col("ADX") < 25).alias("consolidation"),
        (pl.col("SMA_Short") > pl.col("SMA_Long")).alias("uptrend"),
        (pl.col("SMA_Short") < pl.col("SMA_Long")).alias("downtrend"),
        (pl.col("RSI") > 70).alias("overbought"),
        (pl.col("RSI") < 30).alias("oversold"),
    ]

    return df_symbol.with_columns(expressions).with_columns(
        pl.col(condition_columns).fill_null(False)
    )
