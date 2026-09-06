from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import polars as pl

from .indicators import TechnicalIndicatorCalculator
from .lookback import LookbackFeatureBuilder
from .source import StockPriceFeatureSourceBuilder


@dataclass(slots=True)
class StockPriceIndicatorFeatureBuilder:
    MARKET_INDEX_SYMBOL_MAP: ClassVar[dict[str, str]] = {
        "AAPL": "^GSPC",
        "BMW.DE": "^GDAXI",
    }
    MARKET_INDEX_CONTEXT_COLUMN: ClassVar[str] = "market_index_prev_close"

    columns_config: dict[str, list[str]]
    source_builder: StockPriceFeatureSourceBuilder
    lookback_builder: LookbackFeatureBuilder

    def _build_enriched_prices(
        self,
        silver_stock_prices: pl.DataFrame,
        silver_stock_prices_weekly: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        feature_source = self.source_builder.prepare(silver_stock_prices)
        feature_enriched_prices = self.source_builder.map_by_symbol(
            feature_source.drop_nulls(["symbol", "date"]).sort(["symbol", "date"]),
            self.source_builder.add_model_training_tier_columns_for_symbol,
        )
        feature_enriched_prices = self.lookback_builder.add_daily_lookbacks(
            feature_enriched_prices
        )
        feature_enriched_prices = self.lookback_builder.attach_weekly_lookbacks(
            feature_enriched_prices,
            silver_stock_prices_weekly,
        )
        feature_enriched_prices = self._attach_market_index_context(
            feature_enriched_prices,
        )
        return feature_enriched_prices

    def _select_indicator_features(self, enriched_prices: pl.DataFrame) -> pl.DataFrame:
        indicator_calculator = TechnicalIndicatorCalculator()
        indicators = self.source_builder.map_by_symbol(
            self._target_asset_rows(enriched_prices),
            indicator_calculator.calculate_for_symbol,
        )
        return StockPriceFeatureSourceBuilder.with_created_timestamp(
            StockPriceFeatureSourceBuilder.drop_rows_with_missing_model_features(
                indicators,
                self.columns_config["indicator_ready"],
            )
        ).select(self.columns_config["indicator_features"])

    def _select_model_features(self, enriched_prices: pl.DataFrame) -> pl.DataFrame:
        return StockPriceFeatureSourceBuilder.with_created_timestamp(
            StockPriceFeatureSourceBuilder.drop_rows_with_missing_model_features(
                self._target_asset_rows(enriched_prices),
                self.columns_config["model_ready"],
            )
        ).select(
            [
                *self.columns_config["entity"],
                *self.columns_config["output_audit"],
                *self.columns_config["model_features"],
            ]
        )

    def _attach_market_index_context(self, enriched_prices: pl.DataFrame) -> pl.DataFrame:
        market_index_features = self.columns_config.get("market_index_features", [])
        if self.MARKET_INDEX_CONTEXT_COLUMN not in market_index_features:
            return enriched_prices

        index_symbols = list(self.MARKET_INDEX_SYMBOL_MAP.values())
        index_features = (
            enriched_prices.filter(pl.col("symbol").is_in(index_symbols))
            .sort(["symbol", "date"])
            .with_columns(
                pl.col("close")
                .shift(1)
                .over("symbol")
                .alias(self.MARKET_INDEX_CONTEXT_COLUMN)
            )
            .select(
                pl.col("symbol").alias("_market_index_symbol"),
                "date",
                self.MARKET_INDEX_CONTEXT_COLUMN,
            )
        )
        if index_features.is_empty():
            return self._ensure_market_index_context_column(enriched_prices)

        symbol_map = pl.DataFrame(
            {
                "symbol": list(self.MARKET_INDEX_SYMBOL_MAP.keys()),
                "_market_index_symbol": list(self.MARKET_INDEX_SYMBOL_MAP.values()),
            }
        )
        return (
            enriched_prices.join(symbol_map, on="symbol", how="left")
            .join(index_features, on=["_market_index_symbol", "date"], how="left")
            .drop("_market_index_symbol")
        )

    def _target_asset_rows(self, df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("symbol").is_in(list(self.MARKET_INDEX_SYMBOL_MAP)))

    def _ensure_market_index_context_column(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.MARKET_INDEX_CONTEXT_COLUMN in df.columns:
            return df
        return df.with_columns(
            pl.lit(None, dtype=pl.Float64).alias(self.MARKET_INDEX_CONTEXT_COLUMN)
        )

    def build(
        self,
        silver_stock_prices: pl.DataFrame,
        silver_stock_prices_weekly: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        return self._select_indicator_features(
            self._build_enriched_prices(silver_stock_prices, silver_stock_prices_weekly)
        )

    def build_model_features(
        self,
        silver_stock_prices: pl.DataFrame,
        silver_stock_prices_weekly: pl.DataFrame | None = None,
    ) -> pl.DataFrame:
        return self._select_model_features(
            self._build_enriched_prices(silver_stock_prices, silver_stock_prices_weekly)
        )

    def build_feature_sets(
        self,
        silver_stock_prices: pl.DataFrame,
        silver_stock_prices_weekly: pl.DataFrame | None = None,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        enriched_prices = self._build_enriched_prices(
            silver_stock_prices,
            silver_stock_prices_weekly,
        )
        return (
            self._select_indicator_features(enriched_prices),
            self._select_model_features(enriched_prices),
        )
