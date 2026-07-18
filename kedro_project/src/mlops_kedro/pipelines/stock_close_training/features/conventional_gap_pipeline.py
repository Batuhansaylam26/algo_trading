from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from .conditions import GapConditionCalculator
from .source import StockPriceFeatureSourceBuilder
from .strategy import GapStrategyClassifier


@dataclass(slots=True)
class ConventionalGapTradingFeatureBuilder:
    columns_config: dict[str, list[str]]

    def build(
        self,
        stock_price_indicator_features: pl.DataFrame,
    ) -> pl.DataFrame:
        condition_calculator = GapConditionCalculator(
            self.columns_config["condition"],
        )
        strategy_classifier = GapStrategyClassifier(
            self.columns_config["strategy_required_conditions"],
        )
        conditions = StockPriceFeatureSourceBuilder.map_by_symbol(
            stock_price_indicator_features,
            condition_calculator.calculate_for_symbol,
        )
        strategy_features = StockPriceFeatureSourceBuilder.map_by_symbol(
            conditions,
            strategy_classifier.classify_for_symbol,
        )
        return strategy_features.select(self.columns_config["conventional_gap_trading"])
