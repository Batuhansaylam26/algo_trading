from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(slots=True)
class ForecastPerformanceResult:
    predictions: pd.DataFrame
    regression_metrics: pd.DataFrame
    long_direction_metrics: pd.DataFrame
