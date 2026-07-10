from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from statsforecast import StatsForecast

from .metrics import long_only_directional_metrics
from .plots import log_forecast_plots
from .runtime import cpu_count_from_env
from .stats_models import build_statsforecast_models
from .common import (
    _prediction_frame,
    _regression_metrics,
    configure_mlflow_tracking,
    log_mlflow_datasets,
    tier_experiment_name,
    validation_reference_frame,
)


LOGGER = logging.getLogger(__name__)


def build_statsforecast_spec(
    *,
    freq: str = "B",
    seasonal_length: int = 5,
    validation_horizon: int = 1,
    test_horizon: int = 5,
    conformal_n_windows: int = 3,
    level: list[int] | None = None,
    models: list[dict[str, Any]] | None = None,
    verbose: bool = True,
    tier_name: str = "tier1",
) -> dict[str, Any]:
    return {
        "tier_name": tier_name,
        "freq": freq,
        "seasonal_length": seasonal_length,
        "validation_horizon": validation_horizon,
        "test_horizon": test_horizon,
        "conformal_n_windows": conformal_n_windows,
        "level": level or [80, 95],
        "models": models
        or [
            {"class_name": "AutoARIMA", "alias": "AutoARIMA"},
            {"class_name": "AutoETS", "alias": "AutoETS"},
            {"class_name": "AutoTheta", "alias": "AutoTheta"},
            {"class_name": "AutoRegressive", "alias": "AutoRegressive"},
        ],
        "verbose": verbose,
    }


def _model_names_from_predictions(joined_df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in joined_df.columns
        if column not in {"unique_id", "ds", "y"}
        and "-lo-" not in column
        and "-hi-" not in column
    ]


def _pickle_statsforecast_model(model: StatsForecast) -> str:
    temp_dir = tempfile.mkdtemp(prefix="statsforecast_")
    model_path = Path(temp_dir) / "statsforecast_model.pkl"
    with model_path.open("wb") as file_obj:
        pickle.dump(model, file_obj)
    return str(model_path)


def train_statsforecast_models_from_split(
    train_test_split: dict[str, pd.DataFrame],
    *,
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    freq = model_spec["freq"]
    seasonal_length = model_spec["seasonal_length"]
    validation_horizon = model_spec["validation_horizon"]
    test_horizon = model_spec["test_horizon"]
    conformal_n_windows = model_spec["conformal_n_windows"]
    level = model_spec.get("level") or [80, 95]
    verbose = model_spec.get("verbose", True)
    tier_name = model_spec.get("tier_name", "tier1")

    train_df = train_test_split["train"]
    test_df = train_test_split["test"]
    dynamic_feature_columns = [
        column for column in train_df.columns if column not in {"unique_id", "ds", "y"}
    ]
    models = build_statsforecast_models(
        seasonal_length=seasonal_length,
        horizon=validation_horizon,
        conformal_n_windows=conformal_n_windows,
        models_config=model_spec["models"],
    )

    LOGGER.info(
        "Starting StatsForecast training | rows=%s test_rows=%s models=%s "
        "freq=%s seasonal_length=%s conformal_n_windows=%s",
        len(train_df),
        len(test_df),
        [getattr(model, "alias", model.__class__.__name__) for model in models],
        freq,
        seasonal_length,
        conformal_n_windows,
    )

    configure_mlflow_tracking(experiment_name=tier_experiment_name(tier_name))
    with mlflow.start_run(run_name=f"stock-close-{tier_name}-statsforecast", nested=True):
        mlflow.log_params(
            {
                "tier_name": tier_name,
                "freq": freq,
                "seasonal_length": seasonal_length,
                "validation_horizon": validation_horizon,
                "test_horizon": test_horizon,
                "conformal_n_windows": conformal_n_windows,
                "verbose": verbose,
                "models": ",".join(
                    model_config.get("alias", model_config["class_name"])
                    for model_config in model_spec["models"]
                ),
                "dynamic_features": ",".join(dynamic_feature_columns),
            }
        )
        log_mlflow_datasets(
            train_df=train_df,
            validation_df=validation_reference_frame(
                train_df,
                validation_horizon=validation_horizon,
                n_windows=conformal_n_windows,
            ),
            test_df=test_df,
            dataset_prefix=f"stock_close_{tier_name}_statsforecast",
            artifact_prefix=f"statsforecast/{tier_name}",
        )

        sf = StatsForecast(
            models=models,
            freq=freq,
            n_jobs=cpu_count_from_env("MODEL_N_JOBS"),
            verbose=verbose,
        )
        fitted_sf = sf.fit(df=train_df)

        predict_kwargs: dict[str, Any] = {"h": test_horizon, "level": level}
        if dynamic_feature_columns:
            predict_kwargs["X_df"] = test_df.drop(columns=["y"])

        try:
            predictions = fitted_sf.predict(**predict_kwargs)
        except Exception:
            if "X_df" not in predict_kwargs:
                raise
            LOGGER.warning(
                "StatsForecast prediction with X_df failed; retrying without "
                "dynamic features.",
                exc_info=True,
            )
            predictions = fitted_sf.predict(h=test_horizon, level=level)

        if hasattr(predictions, "to_pandas"):
            predictions = predictions.to_pandas()
        joined_df = _prediction_frame(test_df, predictions)
        regression_df = _regression_metrics(joined_df)
        long_direction_df = long_only_directional_metrics(joined_df, train_df)

        LOGGER.info("StatsForecast regression metrics:\n%s", regression_df)
        LOGGER.info("StatsForecast long metrics:\n%s", long_direction_df)

        mlflow.log_table(
            joined_df,
            f"statsforecast/{tier_name}/evaluation/predictions.json",
        )
        mlflow.log_table(
            regression_df,
            f"statsforecast/{tier_name}/evaluation/regression_metrics.json",
        )
        mlflow.log_table(
            long_direction_df,
            f"statsforecast/{tier_name}/evaluation/long_only_direction_metrics.json",
        )
        log_forecast_plots(
            train_df=train_df,
            joined_df=joined_df,
            levels=level,
            artifact_prefix=f"statsforecast/{tier_name}/plots",
        )

        for _, row in regression_df.iterrows():
            mlflow.log_metric(f"statsforecast.{row['model']}.test.mae", float(row["mae"]))
            mlflow.log_metric(f"statsforecast.{row['model']}.test.rmse", float(row["rmse"]))

        for _, row in long_direction_df.iterrows():
            if row[["long_accuracy", "long_precision", "long_recall"]].isna().any():
                continue
            mlflow.log_metric(
                f"statsforecast.{row['model']}.long.accuracy",
                float(row["long_accuracy"]),
            )
            mlflow.log_metric(
                f"statsforecast.{row['model']}.long.precision",
                float(row["long_precision"]),
            )
            mlflow.log_metric(
                f"statsforecast.{row['model']}.long.recall",
                float(row["long_recall"]),
            )

        model_path = _pickle_statsforecast_model(fitted_sf)
        mlflow.log_artifact(
            model_path,
            artifact_path=f"statsforecast/{tier_name}/model",
        )

    return {
        "model": fitted_sf,
        "model_names": _model_names_from_predictions(joined_df),
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "predictions": joined_df,
        "regression_metrics": regression_df,
        "long_direction_metrics": long_direction_df,
    }
