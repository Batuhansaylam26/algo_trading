import polars as pl


class GapStrategyClassifier:
    def __init__(self, required_condition_columns: list[str]) -> None:
        self.required_condition_columns = required_condition_columns

    def classify_for_symbol(self, df_symbol: pl.DataFrame) -> pl.DataFrame:
        if not all(
            column in df_symbol.columns for column in self.required_condition_columns
        ):
            return df_symbol

        return df_symbol.with_columns(self.expression())

    @staticmethod
    def expression() -> pl.Expr:
        return (
            pl.when(
                pl.col("gap_up")
                & pl.col("breakout_up")
                & pl.col("high_vol")
                & pl.col("consolidation").shift(1).fill_null(False)
            )
            .then(pl.lit("Breakaway Up"))
            .when(
                pl.col("gap_down")
                & pl.col("breakout_down")
                & pl.col("high_vol")
                & pl.col("consolidation").shift(1).fill_null(False)
            )
            .then(pl.lit("Breakaway Down"))
            .when(
                pl.col("exhaustion_gap_up")
                & pl.col("uptrend")
                & pl.col("overbought")
                & pl.col("extreme_vol")
                & ~pl.col("consolidation")
            )
            .then(pl.lit("Exhaustion Up"))
            .when(
                pl.col("exhaustion_gap_down")
                & pl.col("downtrend")
                & pl.col("oversold")
                & pl.col("extreme_vol")
                & ~pl.col("consolidation")
            )
            .then(pl.lit("Exhaustion Down"))
            .when(
                pl.col("gap_up")
                & pl.col("uptrend")
                & ~pl.col("breakout_up")
                & ~pl.col("overbought")
                & ~pl.col("consolidation")
            )
            .then(pl.lit("Runaway Up"))
            .when(
                pl.col("gap_down")
                & pl.col("downtrend")
                & ~pl.col("breakout_down")
                & ~pl.col("oversold")
                & ~pl.col("consolidation")
            )
            .then(pl.lit("Runaway Down"))
            .when(pl.col("gap_up") & ~pl.col("breakout_up") & ~pl.col("high_vol"))
            .then(pl.lit("Common Up"))
            .when(pl.col("gap_down") & ~pl.col("breakout_down") & ~pl.col("high_vol"))
            .then(pl.lit("Common Down"))
            .otherwise(pl.lit("None"))
            .alias("Gap_Type")
        )



    @staticmethod
    def classify_strategy_for_symbol(
        df_symbol: pl.DataFrame,
        required_condition_columns: list[str],
    ) -> pl.DataFrame:
        return GapStrategyClassifier(required_condition_columns).classify_for_symbol(
            df_symbol
        )
