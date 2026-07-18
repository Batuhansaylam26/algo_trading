from __future__ import annotations

import os

import mlflow
import numpy as np
import pandas as pd
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")

DEFAULT_MLFLOW_TRACKING_URI = "http://host.docker.internal:5001"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "stock_close_training"
DEFAULT_MLFLOW_REQUEST_TIMEOUT = "60"
DEFAULT_MLFLOW_REQUEST_MAX_RETRIES = "1"
DEFAULT_MLFLOW_REQUEST_BACKOFF_FACTOR = "1"
DEFAULT_MLFLOW_TIER_EXPERIMENT_PREFIX = "stock_close"


def model_id_columns() -> list[str]:
    return ["unique_id", "ds", "y"]


def non_feature_columns() -> set[str]:
    return {
        "unique_id",
        "ds",
        "y",
        "symbol",
        "date",
        "close",
        "created_timestamp",
    }


def tier_experiment_name(tier_name: str) -> str:
    prefix = os.getenv(
        "MLFLOW_TIER_EXPERIMENT_PREFIX",
        DEFAULT_MLFLOW_TIER_EXPERIMENT_PREFIX,
    )
    return f"{prefix}_{tier_name}"


def configure_mlflow_tracking(
    tracking_uri: str | None = None,
    experiment_name: str | None = None,
) -> None:
    tracking_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI",
        DEFAULT_MLFLOW_TRACKING_URI,
    )
    experiment_name = experiment_name or os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        DEFAULT_MLFLOW_EXPERIMENT_NAME,
    )

    os.environ.setdefault(
        "MLFLOW_HTTP_REQUEST_TIMEOUT",
        DEFAULT_MLFLOW_REQUEST_TIMEOUT,
    )
    os.environ.setdefault(
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
        DEFAULT_MLFLOW_REQUEST_MAX_RETRIES,
    )
    os.environ.setdefault(
        "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR",
        DEFAULT_MLFLOW_REQUEST_BACKOFF_FACTOR,
    )

    mlflow.set_tracking_uri(tracking_uri)

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as exc:
        client = MlflowClient(tracking_uri=tracking_uri)
        deleted_experiment = next(
            (
                experiment
                for experiment in client.search_experiments(
                    view_type=ViewType.DELETED_ONLY,
                )
                if experiment.name == experiment_name
            ),
            None,
        )
        if deleted_experiment is not None:
            client.restore_experiment(deleted_experiment.experiment_id)
            mlflow.set_experiment(experiment_name)
            return

        raise RuntimeError(
            "MLflow experiment setup failed from Kedro. "
            f"tracking_uri={tracking_uri!r} experiment={experiment_name!r}. "
            "Check the original MLflow exception below and the mlflow container logs."
        ) from exc


def split_train_test_by_horizon(
    df: pd.DataFrame,
    test_horizon: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    test_index = ordered.groupby("unique_id", observed=True).tail(test_horizon).index
    test_df = ordered.loc[test_index].copy()
    train_df = ordered.drop(test_index).copy()
    return train_df, test_df


def validation_reference_frame(
    train_df: pd.DataFrame,
    *,
    validation_horizon: int,
    n_windows: int,
) -> pd.DataFrame:
    row_count = max(0, int(validation_horizon)) * max(0, int(n_windows))
    if row_count == 0 or train_df.empty:
        return train_df.iloc[0:0].copy()

    return (
        train_df.sort_values(["unique_id", "ds"])
        .groupby("unique_id", observed=True)
        .tail(row_count)
        .reset_index(drop=True)
    )


def log_mlflow_datasets(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    dataset_prefix: str,
    artifact_prefix: str,
    validation_df: pd.DataFrame | None = None,
) -> None:
    datasets = [
        ("train", "training", train_df),
        ("validation", "validation", validation_df),
        ("test", "evaluation", test_df),
    ]
    for split_name, context, dataset_df in datasets:
        if dataset_df is None or dataset_df.empty:
            continue

        dataset_name = f"{dataset_prefix}_{split_name}".replace("/", "_")
        mlflow.log_input(
            mlflow.data.from_pandas(dataset_df, name=dataset_name),
            context=context,
        )
        if _log_dataset_tables_enabled():
            mlflow.log_table(
                dataset_df,
                f"{artifact_prefix}/datasets/{split_name}.json",
            )


def _log_dataset_tables_enabled() -> bool:
    return os.getenv("MLFLOW_LOG_DATASET_TABLES", "0").lower() in {
        "1",
        "true",
        "yes",
    }


def _prediction_frame(
    test_df: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    expected_dates = (
        test_df[["unique_id", "ds"]]
        .sort_values(["unique_id", "ds"])
        .reset_index(drop=True)
    )
    pred_df = predictions.sort_values(["unique_id", "ds"]).reset_index(drop=True)
    pred_df["_forecast_step"] = pred_df.groupby(
        "unique_id",
        observed=True,
    ).cumcount()
    expected_dates["_forecast_step"] = expected_dates.groupby(
        "unique_id",
        observed=True,
    ).cumcount()

    aligned = (
        pred_df.drop(columns=["ds"])
        .merge(
            expected_dates,
            on=["unique_id", "_forecast_step"],
            how="left",
            validate="one_to_one",
        )
        .drop(columns=["_forecast_step"])
    )
    return aligned.merge(
        test_df[["unique_id", "ds", "y"]],
        on=["unique_id", "ds"],
        how="left",
        validate="one_to_one",
    )


def _regression_metrics(joined_df: pd.DataFrame) -> pd.DataFrame:
    models = [
        column
        for column in joined_df.columns
        if column
        not in {
            "unique_id",
            "ds",
            "y",
            "previous_actual_close",
            "actual_long",
        }
        and "-lo-" not in column
        and "-hi-" not in column
    ]
    rows = []
    for model in models:
        valid_rows = joined_df[["y", model]].dropna()
        if valid_rows.empty:
            continue

        mse = mean_squared_error(valid_rows["y"], valid_rows[model])
        non_zero_actuals = valid_rows[valid_rows["y"] != 0]
        mape = (
            float(
                np.mean(
                    np.abs(
                        (non_zero_actuals["y"] - non_zero_actuals[model])
                        / non_zero_actuals["y"]
                    )
                )
                * 100.0
            )
            if not non_zero_actuals.empty
            else None
        )
        rows.append(
            {
                "model": model,
                "mae": mean_absolute_error(valid_rows["y"], valid_rows[model]),
                "rmse": mse**0.5,
                "mape": mape,
                "r2": (
                    r2_score(valid_rows["y"], valid_rows[model])
                    if len(valid_rows) >= 2
                    else None
                ),
                "rows": len(valid_rows),
            }
        )
    return pd.DataFrame(rows)
