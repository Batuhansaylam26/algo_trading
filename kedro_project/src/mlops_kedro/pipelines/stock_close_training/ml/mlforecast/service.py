from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from .spec import MLForecastSpecBuilder
from .training import (
    MLForecastTrainer,
    train_auto_mlforecast_models,
)


@dataclass(slots=True)
class MLForecastService:
    default_freq: str = "B"

    def build_spec(
        self,
        *,
        freq: str | None = None,
        validation_horizon: int = 1,
        test_horizon: int = 5,
        n_windows: int = 3,
        n_trials: int = 20,
        level: list[int] | None = None,
        verbose: bool = True,
        models: list[str] | None = None,
        tier_name: str = "tier1",
    ) -> dict[str, Any]:
        return MLForecastSpecBuilder().build_spec(
            freq=freq or self.default_freq,
            validation_horizon=validation_horizon,
            test_horizon=test_horizon,
            n_windows=n_windows,
            n_trials=n_trials,
            level=level,
            verbose=verbose,
            models=models,
            tier_name=tier_name,
        )

    def train_from_split(
        self,
        train_test_split: dict[str, pd.DataFrame],
        *,
        model_spec: dict[str, Any] | None = None,
        freq: str | None = None,
        validation_horizon: int = 1,
        test_horizon: int = 5,
        n_windows: int = 3,
        n_trials: int = 20,
        level: list[int] | None = None,
        verbose: bool = True,
        study_kwargs: dict[str, Any] | None = None,
        optimize_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return MLForecastTrainer().train_from_split(
            train_test_split,
            model_spec=model_spec,
            freq=freq or self.default_freq,
            validation_horizon=validation_horizon,
            test_horizon=test_horizon,
            n_windows=n_windows,
            n_trials=n_trials,
            level=level,
            verbose=verbose,
            study_kwargs=study_kwargs,
            optimize_kwargs=optimize_kwargs,
        )

    def train(
        self,
        dataset: pl.DataFrame,
        *,
        freq: str | None = None,
        validation_horizon: int = 1,
        test_horizon: int = 5,
        n_windows: int = 3,
        n_trials: int = 20,
        level: list[int] | None = None,
        study_kwargs: dict[str, Any] | None = None,
        optimize_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return train_auto_mlforecast_models(
            dataset,
            freq=freq or self.default_freq,
            validation_horizon=validation_horizon,
            test_horizon=test_horizon,
            n_windows=n_windows,
            n_trials=n_trials,
            level=level,
            study_kwargs=study_kwargs,
            optimize_kwargs=optimize_kwargs,
        )
