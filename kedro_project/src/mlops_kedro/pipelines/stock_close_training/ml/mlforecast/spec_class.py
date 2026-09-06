from __future__ import annotations

import logging
from typing import Any

import optuna

from ..runtime import cpu_count_from_env


LOGGER = logging.getLogger(__name__)


class MLForecastSpecBuilder:
    def build_auto_mlforecast(
        self,
        freq: str = "B",
        lags: list[int] | None = None,
    ) -> Any:
        try:
            from mlforecast.auto import AutoMLForecast
        except OSError as exc:
            if "libgomp.so.1" not in str(exc):
                raise
            raise RuntimeError(
                "MLForecast cannot start because LightGBM needs libgomp.so.1. "
                "Rebuild the devcontainer, or run inside the container: "
                "`apt-get update && apt-get install -y libgomp1`."
            ) from exc

        from .models import (
            MLForecastModelFactory,
            build_auto_models,
            fit_config,
            init_config,
        )

        init_config_factory = (
            MLForecastModelFactory.fixed_init_config(lags) if lags else init_config
        )

        return AutoMLForecast(
            models=build_auto_models(),
            freq=freq,
            init_config=init_config_factory,
            fit_config=fit_config,
            num_threads=cpu_count_from_env("MLFORECAST_NUM_THREADS"),
            reuse_cv_splits=True,
        )

    def build_spec(
        self,
        *,
        freq: str = "B",
        lags: list[int] | None = None,
        n_windows: int = 3,
        n_trials: int = 20,
        level: list[int] | None = None,
        verbose: bool = True,
        models: list[str] | None = None,
        tier_name: str = "tier1",
    ) -> dict[str, Any]:
        return {
            "tier_name": tier_name,
            "freq": freq,
            "lags": lags,
            "n_windows": n_windows,
            "n_trials": n_trials,
            "level": level or [80, 95],
            "verbose": verbose,
            "models": models,
        }

    @staticmethod
    def _trial_logger(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        LOGGER.info(
            "Optuna trial finished | study=%s trial=%s state=%s value=%s params=%s",
            study.study_name,
            trial.number,
            trial.state.name,
            trial.value,
            trial.params,
        )
