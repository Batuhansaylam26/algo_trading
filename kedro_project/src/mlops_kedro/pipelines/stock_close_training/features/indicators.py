from __future__ import annotations

from .indicators_class import *  # noqa: F403
from .indicators_class import TechnicalIndicatorCalculator

technical_indicator_calculator = TechnicalIndicatorCalculator()
calculate_indicators_for_symbol = technical_indicator_calculator.calculate_indicators_for_symbol
