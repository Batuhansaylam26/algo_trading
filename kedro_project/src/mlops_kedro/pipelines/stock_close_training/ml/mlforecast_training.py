from __future__ import annotations

import logging
import os
from typing import Any

os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")

import mlflow
import optuna
import pandas as pd
import polars as pl

from .common import (
    _prediction_frame,
    _regression_metrics,
    configure_mlflow_tracking,
    log_mlflow_datasets,
    model_id_columns,
    non_feature_columns,
    split_train_test_by_horizon,
    tier_experiment_name,
    validation_reference_frame,
)
from .metrics import long_only_directional_metrics
from .plots import log_forecast_plots
from .runtime import cpu_count_from_env


LOGGER = logging.getLogger(__name__)


def to_mlforecast_dataset(df: pl.DataFrame) -> pl.DataFrame:
    exogenous_columns = [
        column for column in df.columns if column not in non_feature_columns()
    ]

    if {"unique_id", "ds", "y"}.issubset(set(df.columns)):
        return (
            df.select(
                pl.col("unique_id").cast(pl.Utf8),
                pl.col("ds").cast(pl.Datetime("us"), strict=False),
                pl.col("y").cast(pl.Float64, strict=False),
                *[
                    pl.col(column).cast(pl.Float64, strict=False)
                    for column in exogenous_columns
                ],
            )
            .sort(["unique_id", "ds"])
        )

    if not {"symbol", "date", "close"}.issubset(set(df.columns)):
        raise ValueError(
            "MLForecast training data needs either unique_id/ds/y "
            "or symbol/date/close columns."
        )

    return (
        df.select(
            pl.col("symbol").cast(pl.Utf8).alias("unique_id"),
            pl.col("date").cast(pl.Datetime("us"), strict=False).alias("ds"),
            pl.col("close").cast(pl.Float64, strict=False).alias("y"),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in exogenous_columns
            ],
        )
        .sort(["unique_id", "ds"])
    )


def _business_day_grid(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.group_by("unique_id")
        .agg(
            pl.datetime_ranges(
                pl.col("ds").min(),
                pl.col("ds").max(),
                interval="1d",
            ).alias("ds")
        )
        .explode("ds")
        .filter(pl.col("ds").dt.weekday() <= 5)
    )


def fill_mlforecast_business_day_gaps(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    feature_columns = [
        column for column in df.columns if column not in model_id_columns()
    ]
    source = (
        df.with_row_index("_source_order")
        .sort(["unique_id", "ds", "_source_order"])
        .unique(subset=["unique_id", "ds"], keep="last", maintain_order=True)
        .drop("_source_order")
    )
    filled = (
        _business_day_grid(source)
        .join(source, on=["unique_id", "ds"], how="left")
        .with_columns(pl.col("y").is_null().alias("_synthetic_row"))
        .sort(["unique_id", "ds"])
    )
    if "calendar_gap_days" in feature_columns:
        filled = (
            filled.with_columns(
                (~pl.col("_synthetic_row")).alias("_actual_row")
            )
            .with_columns(
                pl.col("_actual_row")
                .cast(pl.Int64)
                .cum_sum()
                .over("unique_id")
                .alias("_actual_segment")
            )
            .with_columns(
                (
                    pl.col("ds")
                    .cum_count()
                    .over(["unique_id", "_actual_segment"])
                    - 1
                )
                .cast(pl.Int64)
                .alias("_business_gap_run")
            )
        )

    fill_columns = ["y", *feature_columns]
    filled = filled.with_columns(
        [
            pl.col(column)
            .forward_fill()
            .backward_fill()
            .over("unique_id")
            .alias(column)
            for column in fill_columns
            if column != "calendar_gap_days"
        ]
    )
    if "calendar_gap_days" in feature_columns:
        filled = filled.with_columns(
            pl.when(pl.col("_synthetic_row"))
            .then(pl.col("_business_gap_run"))
            .when(pl.col("_business_gap_run").shift(1).over("unique_id") > 0)
            .then(pl.col("_business_gap_run").shift(1).over("unique_id"))
            .otherwise(pl.col("calendar_gap_days"))
            .fill_null(0)
            .cast(pl.Float64)
            .alias("calendar_gap_days")
        )

    return (
        filled.drop(
            [
                column
                for column in [
                    "_synthetic_row",
                    "_source_row",
                    "_actual_row",
                    "_actual_segment",
                    "_business_gap_run",
                ]
                if column in filled.columns
            ]
        )
        .drop_nulls(["unique_id", "ds", "y", *feature_columns])
        .filter(pl.all_horizontal(pl.col(["y", *feature_columns]).is_finite()))
        .select(["unique_id", "ds", "y", *feature_columns])
        .sort(["unique_id", "ds"])
    )


def to_mlforecast_frame(df: pl.DataFrame) -> pd.DataFrame:
    model_df = fill_mlforecast_business_day_gaps(to_mlforecast_dataset(df))
    return model_df.to_pandas()


def make_train_test_split(
    dataset: pl.DataFrame,
    test_horizon: int,
) -> dict[str, pd.DataFrame]:
    model_df = to_mlforecast_frame(dataset)
    train_df, test_df = split_train_test_by_horizon(model_df, test_horizon)
    return {
        "full": model_df,
        "train": train_df,
        "test": test_df,
    }


def build_auto_mlforecast(freq: str = "B") -> Any:
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

    from .mlforecast_models import build_auto_models, fit_config, init_config

    return AutoMLForecast(
        models=build_auto_models(),
        freq=freq,
        init_config=init_config,
        fit_config=fit_config,
        num_threads=cpu_count_from_env("MLFORECAST_NUM_THREADS"),
        reuse_cv_splits=True,
    )


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


def _trial_logger(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
    LOGGER.info(
        "Optuna trial finished | study=%s trial=%s state=%s value=%s params=%s",
        study.study_name,
        trial.number,
        trial.state.name,
        trial.value,
        trial.params,
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
    tier_name = "tier1"
    if model_spec:
        tier_name = model_spec.get("tier_name", tier_name)
        freq = model_spec.get("freq", freq)
        validation_horizon = model_spec.get(
            "validation_horizon",
            validation_horizon,
        )
        test_horizon = model_spec.get("test_horizon", test_horizon)
        n_windows = model_spec.get("n_windows", n_windows)
        n_trials = model_spec.get("n_trials", n_trials)
        level = model_spec.get("level", level)
        verbose = model_spec.get("verbose", verbose)

    level = level or [80, 95]
    os.environ["MLFORECAST_VERBOSE"] = "1" if verbose else "0"
    optuna.logging.set_verbosity(
        optuna.logging.INFO if verbose else optuna.logging.WARNING
    )
    study_kwargs = study_kwargs or {
        "study_name": "stock_close_automlforecast",
    }
    study_kwargs.pop("direction", None)
    optimize_kwargs = optimize_kwargs or {}
    callbacks = list(optimize_kwargs.get("callbacks", []))
    if verbose:
        callbacks.append(_trial_logger)
    optimize_kwargs["callbacks"] = callbacks

    train_df = train_test_split["train"]
    test_df = train_test_split["test"]
    auto_mlf = build_auto_mlforecast(freq=freq)
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
        freq,
        n_windows,
        validation_horizon,
        n_trials,
    )

    configure_mlflow_tracking(experiment_name=tier_experiment_name(tier_name))

    with mlflow.start_run(
        run_name=f"stock-close-{tier_name}-automlforecast",
        nested=True,
    ):
        dynamic_feature_columns = [
            column
            for column in train_df.columns
            if column not in {"unique_id", "ds", "y"}
        ]

        mlflow.log_params(
            {
                "freq": freq,
                "validation_horizon": validation_horizon,
                "test_horizon": test_horizon,
                "n_windows": n_windows,
                "n_trials": n_trials,
                "verbose": verbose,
                "tier_name": tier_name,
                "model_input_columns": "unique_id,ds,y",
                "dynamic_features": ",".join(dynamic_feature_columns),
            }
        )
        log_mlflow_datasets(
            train_df=train_df,
            validation_df=validation_reference_frame(
                train_df,
                validation_horizon=validation_horizon,
                n_windows=n_windows,
            ),
            test_df=test_df,
            dataset_prefix=f"stock_close_{tier_name}_mlforecast",
            artifact_prefix=f"mlforecast/{tier_name}",
        )

        from mlforecast.utils import PredictionIntervals

        auto_mlf.fit(
            df=train_df,
            n_windows=n_windows,
            h=validation_horizon,
            num_samples=n_trials,
            id_col="unique_id",
            time_col="ds",
            target_col="y",
            study_kwargs=study_kwargs,
            optimize_kwargs=optimize_kwargs,
            prediction_intervals=PredictionIntervals(
                n_windows=n_windows,
                h=validation_horizon,
            ),
        )

        LOGGER.info("AutoMLForecast fit completed. Generating test predictions.")

        predict_kwargs = {"h": test_horizon, "level": level}
        if dynamic_feature_columns:
            predict_kwargs["X_df"] = test_df.drop(columns=["y"])

        predictions = auto_mlf.predict(**predict_kwargs)
        joined_df = _prediction_frame(test_df, predictions)
        regression_df = _regression_metrics(joined_df)
        long_direction_df = long_only_directional_metrics(joined_df, train_df)

        LOGGER.info("Regression metrics:\n%s", regression_df.to_string(index=False))
        LOGGER.info(
            "Long-only directional metrics:\n%s",
            long_direction_df.to_string(index=False),
        )

        mlflow.log_table(
            joined_df,
            f"mlforecast/{tier_name}/evaluation/predictions.json",
        )
        mlflow.log_table(
            regression_df,
            f"mlforecast/{tier_name}/evaluation/regression_metrics.json",
        )
        mlflow.log_table(
            long_direction_df,
            f"mlforecast/{tier_name}/evaluation/long_only_direction_metrics.json",
        )
        log_forecast_plots(
            train_df=train_df,
            joined_df=joined_df,
            levels=level,
            artifact_prefix=f"mlforecast/{tier_name}/plots",
        )

        for _, row in regression_df.iterrows():
            mlflow.log_metric(f"{row['model']}.test.mae", float(row["mae"]))
            mlflow.log_metric(f"{row['model']}.test.rmse", float(row["rmse"]))

        for _, row in long_direction_df.iterrows():
            if row[["long_accuracy", "long_precision", "long_recall"]].isna().any():
                continue

            mlflow.log_metric(
                f"{row['model']}.long.accuracy",
                float(row["long_accuracy"]),
            )
            mlflow.log_metric(
                f"{row['model']}.long.precision",
                float(row["long_precision"]),
            )
            mlflow.log_metric(
                f"{row['model']}.long.recall",
                float(row["long_recall"]),
            )
            mlflow.log_metric(
                f"{row['model']}.long.signal_rate",
                float(row["long_signal_rate"]),
            )

        import mlforecast.flavor

        for model_name, fitted_model in auto_mlf.models_.items():
            mlforecast.flavor.log_model(
                fitted_model,
                name=f"stock_close_{tier_name}_{model_name}",
                artifact_path=None,
            )

    return {
        "model": auto_mlf,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "predictions": joined_df,
        "regression_metrics": regression_df,
        "long_direction_metrics": long_direction_df,
    }


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
