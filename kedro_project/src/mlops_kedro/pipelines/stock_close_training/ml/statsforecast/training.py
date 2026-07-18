from __future__ import annotations

import logging
import pickle
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from statsforecast import StatsForecast

from ..common import (
    configure_mlflow_tracking,
    tier_experiment_name,
    validation_reference_frame,
)
from ..performance import ForecastPerformanceMeasurement
from ..runtime import cpu_count_from_env
from .models import build_statsforecast_models
from .training_config import StatsForecastTrainingConfig


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class StatsForecastTrainer:
    def train_from_split(
        self,
        train_test_split: dict[str, pd.DataFrame],
        *,
        model_spec: dict[str, Any],
    ) -> dict[str, Any]:
        config = StatsForecastTrainingConfig.from_spec(model_spec)
        train_df = train_test_split["train"]
        test_df = train_test_split["test"]
        dynamic_feature_columns = self._dynamic_feature_columns(train_df)
        models = build_statsforecast_models(
            seasonal_length=config.seasonal_length,
            horizon=config.validation_horizon,
            conformal_n_windows=config.conformal_n_windows,
            models_config=config.models or [],
        )

        LOGGER.info(
            "Starting StatsForecast training | rows=%s test_rows=%s models=%s "
            "freq=%s seasonal_length=%s conformal_n_windows=%s",
            len(train_df),
            len(test_df),
            [getattr(model, "alias", model.__class__.__name__) for model in models],
            config.freq,
            config.seasonal_length,
            config.conformal_n_windows,
        )

        configure_mlflow_tracking(experiment_name=tier_experiment_name(config.tier_name))
        with mlflow.start_run(
            run_name=f"stock-close-{config.tier_name}-statsforecast",
            nested=True,
        ):
            evaluator = ForecastPerformanceMeasurement(
                model_family="statsforecast",
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
            fitted_sf = self._fit(train_df, config, models)
            predictions = self._predict(
                model=fitted_sf,
                config=config,
                test_df=test_df,
                dynamic_feature_columns=dynamic_feature_columns,
            )
            result = evaluator.measure(
                train_df=train_df,
                test_df=test_df,
                predictions=predictions,
            )
            LOGGER.info("StatsForecast regression metrics:\n%s", result.regression_metrics)
            LOGGER.info("StatsForecast long metrics:\n%s", result.long_direction_metrics)
            evaluator.log_result(train_df=train_df, result=result)
            self._log_model(fitted_sf, config.tier_name)

        return {
            "model": fitted_sf,
            "model_names": ForecastPerformanceMeasurement.model_names_from_predictions(
                result.predictions,
            ),
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "predictions": result.predictions,
            "regression_metrics": result.regression_metrics,
            "long_direction_metrics": result.long_direction_metrics,
        }

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
        config: StatsForecastTrainingConfig,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        dynamic_feature_columns: list[str],
        evaluator: ForecastPerformanceMeasurement,
    ) -> None:
        mlflow.log_params(
            {
                "tier_name": config.tier_name,
                "freq": config.freq,
                "seasonal_length": config.seasonal_length,
                "validation_horizon": config.validation_horizon,
                "test_horizon": config.test_horizon,
                "conformal_n_windows": config.conformal_n_windows,
                "verbose": config.verbose,
                "models": ",".join(
                    model_config.get("alias", model_config["class_name"])
                    for model_config in config.models or []
                ),
                "dynamic_features": ",".join(dynamic_feature_columns),
            }
        )
        evaluator.log_datasets(
            train_df=train_df,
            validation_df=validation_reference_frame(
                train_df,
                validation_horizon=config.validation_horizon,
                n_windows=config.conformal_n_windows,
            ),
            test_df=test_df,
        )

    @staticmethod
    def _fit(
        train_df: pd.DataFrame,
        config: StatsForecastTrainingConfig,
        models: list[Any],
    ) -> StatsForecast:
        model = StatsForecast(
            models=models,
            freq=config.freq,
            n_jobs=cpu_count_from_env("MODEL_N_JOBS"),
            verbose=config.verbose,
        )
        return model.fit(df=train_df)

    @staticmethod
    def _predict(
        *,
        model: StatsForecast,
        config: StatsForecastTrainingConfig,
        test_df: pd.DataFrame,
        dynamic_feature_columns: list[str],
    ) -> pd.DataFrame:
        predict_kwargs: dict[str, Any] = {
            "h": config.test_horizon,
            "level": config.level,
        }
        if dynamic_feature_columns:
            predict_kwargs["X_df"] = test_df.drop(columns=["y"])

        try:
            predictions = model.predict(**predict_kwargs)
        except Exception:
            if "X_df" not in predict_kwargs:
                raise
            LOGGER.warning(
                "StatsForecast prediction with X_df failed; retrying without "
                "dynamic features.",
                exc_info=True,
            )
            predictions = model.predict(h=config.test_horizon, level=config.level)

        if hasattr(predictions, "to_pandas"):
            predictions = predictions.to_pandas()
        return predictions

    @staticmethod
    def _log_model(model: StatsForecast, tier_name: str) -> None:
        temp_dir = tempfile.mkdtemp(prefix="statsforecast_")
        model_path = Path(temp_dir) / "statsforecast_model.pkl"
        with model_path.open("wb") as file_obj:
            pickle.dump(model, file_obj)
        mlflow.log_artifact(
            str(model_path),
            artifact_path=f"statsforecast/{tier_name}/model",
        )


def train_statsforecast_models_from_split(
    train_test_split: dict[str, pd.DataFrame],
    *,
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    return StatsForecastTrainer().train_from_split(
        train_test_split,
        model_spec=model_spec,
    )
