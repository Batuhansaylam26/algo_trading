from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Any

import polars as pl


@dataclass(slots=True)
class FourierTimeEncoder:
    config: dict[str, Any]

    def harmonics(self, key: str) -> tuple[int, ...]:
        return tuple(self.config.get(key, [1, 2]))

    def period(self, column: str) -> float:
        return float(self.config.get("periods", {}).get(column))

    def terms(
        self,
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

    def add(
        self,
        df: pl.DataFrame,
        *,
        date_column: str,
        harmonics_key: str,
        drop_date_parts: bool = False,
    ) -> pl.DataFrame:
        encoded = (
            df.with_columns(
                pl.col(date_column).dt.month().cast(pl.Int8).alias("month"),
                pl.col(date_column).dt.day().cast(pl.Int8).alias("day"),
                pl.col(date_column)
                .dt.ordinal_day()
                .cast(pl.Int16)
                .alias("day_of_year"),
            )
            .with_columns(
                [
                    *self.terms(
                        "month",
                        period=self.period("month"),
                        harmonics=self.harmonics(harmonics_key),
                    ),
                    *self.terms(
                        "day",
                        period=self.period("day"),
                        harmonics=self.harmonics(harmonics_key),
                    ),
                    *self.terms(
                        "day_of_year",
                        period=self.period("day_of_year"),
                        harmonics=self.harmonics(harmonics_key),
                    ),
                ]
            )
        )
        if drop_date_parts:
            return encoded.drop(["month", "day", "day_of_year"])
        return encoded
