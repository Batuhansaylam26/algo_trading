from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .spec import StatsForecastSpecBuilder
from .training import StatsForecastTrainer


@dataclass(slots=True)
class StatsForecastService:
    default_freq: str = "B"

    def build_spec(
        self,
        *,
        freq: str | None = None,
        seasonal_length: int = 5,
        validation_horizon: int = 1,
        test_horizon: int = 5,
        conformal_n_windows: int = 3,
        level: list[int] | None = None,
        models: list[dict[str, Any]] | None = None,
        verbose: bool = True,
        tier_name: str = "tier1",
    ) -> dict[str, Any]:
        return StatsForecastSpecBuilder().build_spec(
            freq=freq or self.default_freq,
            seasonal_length=seasonal_length,
            validation_horizon=validation_horizon,
            test_horizon=test_horizon,
            conformal_n_windows=conformal_n_windows,
            level=level,
            models=models,
            verbose=verbose,
            tier_name=tier_name,
        )

    def train_from_split(
        self,
        train_test_split: dict[str, pd.DataFrame],
        *,
        model_spec: dict[str, Any],
    ) -> dict[str, Any]:
        return StatsForecastTrainer().train_from_split(
            train_test_split,
            model_spec=model_spec,
        )
