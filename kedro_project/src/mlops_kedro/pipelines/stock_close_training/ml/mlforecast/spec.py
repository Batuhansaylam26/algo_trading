from __future__ import annotations

import logging
from typing import Any

import optuna

from ..runtime import cpu_count_from_env


LOGGER = logging.getLogger(__name__)


class MLForecastSpecBuilder:
    def build_auto_mlforecast(self, freq: str = "B") -> Any:
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

        from .models import build_auto_models, fit_config, init_config

        return AutoMLForecast(
            models=build_auto_models(),
            freq=freq,
            init_config=init_config,
            fit_config=fit_config,
            num_threads=cpu_count_from_env("MLFORECAST_NUM_THREADS"),
            reuse_cv_splits=True,
        )

    def build_spec(
        self,
        *,
        freq: str = "B",
        validation_horizon: int = 1,
        test_horizon: int = 5,
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
            "validation_horizon": validation_horizon,
            "test_horizon": test_horizon,
            "n_windows": n_windows,
            "n_trials": n_trials,
            "level": level or [80, 95],
            "verbose": verbose,
            "models": models,
        }


def build_auto_mlforecast(freq: str = "B") -> Any:
    return MLForecastSpecBuilder().build_auto_mlforecast(freq=freq)

def build_auto_mlforecast_spec(
    *,
    freq: str = "B",
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
        freq=freq,
        validation_horizon=validation_horizon,
        test_horizon=test_horizon,
        n_windows=n_windows,
        n_trials=n_trials,
        level=level,
        verbose=verbose,
        models=models,
        tier_name=tier_name,
    )

def _trial_logger(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    LOGGER.info(
        "Optuna trial finished | study=%s trial=%s state=%s value=%s params=%s",
        study.study_name,
        trial.number,
        trial.state.name,
        trial.value,
        trial.params,
    )
