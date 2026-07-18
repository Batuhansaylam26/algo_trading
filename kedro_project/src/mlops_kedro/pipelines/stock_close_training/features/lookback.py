from __future__ import annotations

import re
from dataclasses import dataclass

import polars as pl


_LOOKBACK_PATTERN = re.compile(r"^(daily|weekly)_(?P<column>.+)_lag_(?P<lag>\d+)$")


@dataclass(slots=True)
class LookbackFeatureBuilder:
    columns_config: dict[str, list[str]]

    def add_daily_lookbacks(self, df: pl.DataFrame) -> pl.DataFrame:
        expressions = [
            pl.col(source_column)
            .shift(lag)
            .over("symbol")
            .alias(output_column)
            for output_column, source_column, lag in self._lookback_specs("daily")
            if source_column in df.columns
        ]
        return df.with_columns(expressions) if expressions else df

    def attach_weekly_lookbacks(
        self,
        daily_df: pl.DataFrame,
        weekly_df: pl.DataFrame | None,
    ) -> pl.DataFrame:
        weekly_columns = self.columns_config.get("weekly_lookback_features", [])
        if not weekly_columns:
            return daily_df
        if weekly_df is None or weekly_df.is_empty():
            return self._ensure_columns(daily_df, weekly_columns)

        weekly_features = self._build_weekly_features(weekly_df)
        if weekly_features.is_empty():
            return self._ensure_columns(daily_df, weekly_columns)

        return (
            daily_df.sort(["symbol", "date"])
            .join_asof(
                weekly_features.sort(["symbol", "available_date"]),
                left_on="date",
                right_on="available_date",
                by="symbol",
                strategy="backward",
            )
            .drop("available_date")
        )

    def _build_weekly_features(self, weekly_df: pl.DataFrame) -> pl.DataFrame:
        weekly_columns = self.columns_config.get("weekly_lookback_features", [])
        weekly = (
            weekly_df.select(
                pl.col("symbol").cast(pl.Utf8).str.to_uppercase(),
                pl.col("date").cast(pl.Datetime("us"), strict=False),
                pl.col("open").cast(pl.Float64, strict=False),
                pl.col("high").cast(pl.Float64, strict=False),
                pl.col("low").cast(pl.Float64, strict=False),
                pl.col("close").cast(pl.Float64, strict=False),
                pl.col("volume").cast(pl.Float64, strict=False),
            )
            .drop_nulls(["symbol", "date", "open", "high", "low", "close", "volume"])
            .unique(subset=["symbol", "date"], keep="last", maintain_order=True)
            .sort(["symbol", "date"])
        )
        days_until_friday = (pl.lit(5) - pl.col("date").dt.weekday()) % 7
        weekly = weekly.with_columns(
            (pl.col("date") + pl.duration(days=days_until_friday + 3)).alias(
                "available_date"
            )
        )
        expressions = [
            pl.col(source_column)
            .shift(lag - 1)
            .over("symbol")
            .alias(output_column)
            for output_column, source_column, lag in self._lookback_specs("weekly")
            if source_column in weekly.columns
        ]
        if expressions:
            weekly = weekly.with_columns(expressions)
        return weekly.select(["symbol", "available_date", *weekly_columns]).drop_nulls(
            weekly_columns
        )

    def _lookback_specs(self, prefix: str) -> list[tuple[str, str, int]]:
        specs = []
        for output_column in self.columns_config.get(f"{prefix}_lookback_features", []):
            match = _LOOKBACK_PATTERN.match(output_column)
            if match is None:
                continue
            specs.append(
                (
                    output_column,
                    match.group("column"),
                    int(match.group("lag")),
                )
            )
        return specs

    @staticmethod
    def _ensure_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
        missing_columns = [column for column in columns if column not in df.columns]
        if not missing_columns:
            return df
        return df.with_columns(
            [pl.lit(None, dtype=pl.Float64).alias(column) for column in missing_columns]
        )
