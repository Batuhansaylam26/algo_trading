from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")

import mlflow
import mlforecast.flavor
import optuna
import pandas as pd
import polars as pl
from mlforecast.utils import PredictionIntervals

from ..common import (
    configure_mlflow_tracking,
    tier_experiment_name,
    validation_reference_frame,
)
from ..performance import ForecastPerformanceMeasurement
from ..runtime import filter_sklearn_parallel_warnings
from .data import make_train_test_split
from .spec import _trial_logger, build_auto_mlforecast
from .training_config import MLForecastTrainingConfig


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class MLForecastTrainer:
    def train_from_split(
        self,
        train_test_split: dict[str, pd.DataFrame],
        *,
        model_spec: dict[str, Any] | None = None,
        freq: str = "B",
        validation_horizon: int = 1,
        test_horizon: int = 5,
        n_windows: int = 3,
        n_trials: int = 20,
        level: list[int] | None = None,
        verbose: bool = True,
        study_kwargs: dict[str, Any] | None = None,
        optimize_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        filter_sklearn_parallel_warnings()
        config = MLForecastTrainingConfig.from_inputs(
            model_spec=model_spec,
            freq=freq,
            validation_horizon=validation_horizon,
            test_horizon=test_horizon,
            n_windows=n_windows,
            n_trials=n_trials,
            level=level,
            verbose=verbose,
        )
        study_kwargs, optimize_kwargs = self._configure_optimization(
            config,
            study_kwargs,
            optimize_kwargs,
        )
        train_df = train_test_split["train"]
        test_df = train_test_split["test"]
        auto_mlf = build_auto_mlforecast(freq=config.freq)
        model_names = (
            model_spec.get("models")
            if model_spec and model_spec.get("models")
            else list(auto_mlf.models.keys())
        )

        LOGGER.info(
            "Starting AutoMLForecast training | rows=%s test_rows=%s models=%s "
            "freq=%s n_windows=%s validation_horizon=%s n_trials=%s",
            len(train_df),
            len(test_df),
            model_names,
            config.freq,
            config.n_windows,
            config.validation_horizon,
            config.n_trials,
        )

        configure_mlflow_tracking(experiment_name=tier_experiment_name(config.tier_name))
        with mlflow.start_run(
            run_name=f"stock-close-{config.tier_name}-automlforecast",
            nested=True,
        ):
            dynamic_feature_columns = self._dynamic_feature_columns(train_df)
            evaluator = ForecastPerformanceMeasurement(
                model_family="mlforecast",
                tier_name=config.tier_name,
                levels=config.level or [80, 95],
            )
            self._log_training_inputs(
                config=config,
                train_df=train_df,
                test_df=test_df,
                dynamic_feature_columns=dynamic_feature_columns,
                evaluator=evaluator,
            )
            auto_mlf.fit(
                df=train_df,
                n_windows=config.n_windows,
                h=config.validation_horizon,
                num_samples=config.n_trials,
                id_col="unique_id",
                time_col="ds",
                target_col="y",
                study_kwargs=study_kwargs,
                optimize_kwargs=optimize_kwargs,
                prediction_intervals=PredictionIntervals(
                    n_windows=config.n_windows,
                    h=config.validation_horizon,
                ),
            )

            LOGGER.info("AutoMLForecast fit completed. Generating test predictions.")
            predictions = auto_mlf.predict(
                **self._predict_kwargs(
                    config=config,
                    test_df=test_df,
                    dynamic_feature_columns=dynamic_feature_columns,
                )
            )
            result = evaluator.measure(
                train_df=train_df,
                test_df=test_df,
                predictions=predictions,
            )
            LOGGER.info(
                "Regression metrics:\n%s",
                result.regression_metrics.to_string(index=False),
            )
            LOGGER.info(
                "Long-only directional metrics:\n%s",
                result.long_direction_metrics.to_string(index=False),
            )
            evaluator.log_result(train_df=train_df, result=result)
            self._log_models(auto_mlf, config.tier_name)

        return {
            "model": auto_mlf,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "predictions": result.predictions,
            "regression_metrics": result.regression_metrics,
            "long_direction_metrics": result.long_direction_metrics,
        }

    @staticmethod
    def _configure_optimization(
        config: MLForecastTrainingConfig,
        study_kwargs: dict[str, Any] | None,
        optimize_kwargs: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        os.environ["MLFORECAST_VERBOSE"] = "1" if config.verbose else "0"
        optuna.logging.set_verbosity(
            optuna.logging.INFO if config.verbose else optuna.logging.WARNING
        )
        resolved_study_kwargs = study_kwargs or {
            "study_name": "stock_close_automlforecast",
        }
        resolved_study_kwargs.pop("direction", None)
        resolved_optimize_kwargs = optimize_kwargs or {}
        callbacks = list(resolved_optimize_kwargs.get("callbacks", []))
        if config.verbose:
            callbacks.append(_trial_logger)
        resolved_optimize_kwargs["callbacks"] = callbacks
        return resolved_study_kwargs, resolved_optimize_kwargs

    @staticmethod
    def _dynamic_feature_columns(train_df: pd.DataFrame) -> list[str]:
        return [
            column
            for column in train_df.columns
            if column not in {"unique_id", "ds", "y"}
        ]

    @staticmethod
    def _log_training_inputs(
        *,
        config: MLForecastTrainingConfig,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dynamic_feature_columns: list[str],
        evaluator: ForecastPerformanceMeasurement,
    ) -> None:
        mlflow.log_params(
            {
                "freq": config.freq,
                "validation_horizon": config.validation_horizon,
                "test_horizon": config.test_horizon,
                "n_windows": config.n_windows,
                "n_trials": config.n_trials,
                "verbose": config.verbose,
                "tier_name": config.tier_name,
                "model_input_columns": "unique_id,ds,y",
                "dynamic_features": ",".join(dynamic_feature_columns),
            }
        )
        evaluator.log_datasets(
            train_df=train_df,
            validation_df=validation_reference_frame(
                train_df,
                validation_horizon=config.validation_horizon,
                n_windows=config.n_windows,
            ),
            test_df=test_df,
        )

    @staticmethod
    def _predict_kwargs(
        *,
        config: MLForecastTrainingConfig,
        test_df: pd.DataFrame,
        dynamic_feature_columns: list[str],
    ) -> dict[str, Any]:
        predict_kwargs: dict[str, Any] = {"h": config.test_horizon, "level": config.level}
        if dynamic_feature_columns:
            predict_kwargs["X_df"] = test_df.drop(columns=["y"])
        return predict_kwargs

    @staticmethod
    def _log_models(auto_mlf, tier_name: str) -> None:
        for model_name, fitted_model in auto_mlf.models_.items():
            mlforecast.flavor.log_model(
                fitted_model,
                name=f"stock_close_{tier_name}_{model_name}",
                artifact_path=None,
            )


def train_auto_mlforecast_models_from_split(
    train_test_split: dict[str, pd.DataFrame],
    *,
    model_spec: dict[str, Any] | None = None,
    freq: str = "B",
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
        freq=freq,
        validation_horizon=validation_horizon,
        test_horizon=test_horizon,
        n_windows=n_windows,
        n_trials=n_trials,
        level=level,
        verbose=verbose,
        study_kwargs=study_kwargs,
        optimize_kwargs=optimize_kwargs,
    )


def train_auto_mlforecast_models(
    dataset: pl.DataFrame,
    *,
    freq: str = "B",
    validation_horizon: int = 1,
    test_horizon: int = 5,
    n_windows: int = 3,
    n_trials: int = 20,
    level: list[int] | None = None,
    study_kwargs: dict[str, Any] | None = None,
    optimize_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_test_split = make_train_test_split(
        dataset,
        test_horizon=test_horizon,
    )
    return train_auto_mlforecast_models_from_split(
        train_test_split,
        freq=freq,
        validation_horizon=validation_horizon,
        test_horizon=test_horizon,
        n_windows=n_windows,
        n_trials=n_trials,
        level=level,
        study_kwargs=study_kwargs,
        optimize_kwargs=optimize_kwargs,
    )
