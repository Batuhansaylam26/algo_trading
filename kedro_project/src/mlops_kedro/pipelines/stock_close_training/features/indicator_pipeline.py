from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .indicators import TechnicalIndicatorCalculator
from .lookback import LookbackFeatureBuilder
from .source import StockPriceFeatureSourceBuilder


@dataclass(slots=True)
class StockPriceIndicatorFeatureBuilder:
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
        return feature_enriched_prices

    def _select_indicator_features(self, enriched_prices: pl.DataFrame) -> pl.DataFrame:
        indicator_calculator = TechnicalIndicatorCalculator()
        indicators = self.source_builder.map_by_symbol(
            enriched_prices,
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
                enriched_prices,
                self.columns_config["model_ready"],
            )
        ).select(
            [
                *self.columns_config["entity"],
                *self.columns_config["output_audit"],
                *self.columns_config["model_features"],
            ]
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
