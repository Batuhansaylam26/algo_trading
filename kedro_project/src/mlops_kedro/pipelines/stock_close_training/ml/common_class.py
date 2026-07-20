from __future__ import annotations

import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "admin")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "admin1234")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

DEFAULT_MLFLOW_TRACKING_URI = "http://127.0.0.1:5001"
DEFAULT_MLFLOW_EXPERIMENT_NAME = "stock_close_training"
DEFAULT_MLFLOW_REQUEST_TIMEOUT = "60"
DEFAULT_MLFLOW_REQUEST_MAX_RETRIES = "1"
DEFAULT_MLFLOW_REQUEST_BACKOFF_FACTOR = "1"
DEFAULT_MLFLOW_SERVER_READY_TIMEOUT = "30"
DEFAULT_MLFLOW_EXPERIMENT_SETUP_RETRIES = "3"
DEFAULT_MLFLOW_TIER_EXPERIMENT_PREFIX = "stock_close"


class MlCommon:

    @staticmethod
    def resolve_local_service_url(value: str | None, *, port: int) -> str:
        if not value or value.lower() == "auto":
            host = "host.docker.internal" if MlCommon._running_in_container() else "127.0.0.1"
            return f"http://{host}:{port}"
        if MlCommon._running_in_container():
            return value.replace("127.0.0.1", "host.docker.internal").replace(
                "localhost",
                "host.docker.internal",
            )
        return value.replace("host.docker.internal", "127.0.0.1")

    @staticmethod
    def _running_in_container() -> bool:
        return Path("/.dockerenv").exists() or Path("/workspaces").exists()

    @staticmethod
    def model_id_columns() -> list[str]:
        return ["unique_id", "ds", "y"]

    @staticmethod
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

    @staticmethod
    def tier_experiment_name(tier_name: str) -> str:
        prefix = os.getenv(
            "MLFLOW_TIER_EXPERIMENT_PREFIX",
            DEFAULT_MLFLOW_TIER_EXPERIMENT_PREFIX,
        )
        return f"{prefix}_{tier_name}"

    @staticmethod
    def configure_mlflow_tracking(
        tracking_uri: str | None = None,
        experiment_name: str | None = None,
    ) -> None:
        tracking_uri = MlCommon.resolve_local_service_url(
            tracking_uri
            or os.getenv(
                "MLFLOW_TRACKING_URI",
                DEFAULT_MLFLOW_TRACKING_URI,
            ),
            port=5001,
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

        ready_timeout = int(
            os.getenv("MLFLOW_SERVER_READY_TIMEOUT", DEFAULT_MLFLOW_SERVER_READY_TIMEOUT)
        )
        setup_retries = int(
            os.getenv(
                "MLFLOW_EXPERIMENT_SETUP_RETRIES",
                DEFAULT_MLFLOW_EXPERIMENT_SETUP_RETRIES,
            )
        )

        last_error: Exception | None = None
        for attempt in range(1, setup_retries + 1):
            try:
                MlCommon.wait_for_mlflow_server(tracking_uri, timeout_seconds=ready_timeout)
                mlflow.set_experiment(experiment_name)
                return
            except Exception as exc:
                last_error = exc
                if attempt < setup_retries:
                    time.sleep(min(10, attempt * 2))
                    continue

        raise RuntimeError(
            "MLflow experiment setup failed from Kedro. "
            f"tracking_uri={tracking_uri!r} experiment={experiment_name!r}. "
            "Make sure the MLflow stack is running with "
            "`docker compose up -d postgres-mlflow minio create-bucket mlflow`, "
            "then check `docker compose ps mlflow postgres-mlflow minio` and "
            "`docker logs mlflow_server --tail 120`."
        ) from last_error

    @staticmethod
    def wait_for_mlflow_server(tracking_uri: str, *, timeout_seconds: int) -> None:
        deadline = time.monotonic() + max(1, timeout_seconds)
        base_uri = tracking_uri.rstrip("/")
        health_urls = [f"{base_uri}/health", base_uri]
        last_error: Exception | None = None

        while time.monotonic() < deadline:
            for health_url in health_urls:
                try:
                    request = Request(health_url, method="GET")
                    with urlopen(request, timeout=3) as response:
                        if response.status < 500:
                            return
                except HTTPError as exc:
                    if exc.code < 500:
                        return
                    last_error = exc
                except (OSError, URLError) as exc:
                    last_error = exc
            time.sleep(1)

        raise RuntimeError(
            "MLflow tracking server is not ready. "
            f"tracking_uri={tracking_uri!r} timeout={timeout_seconds}s."
        ) from last_error

    @staticmethod
    def split_train_test_by_horizon(
        df: pd.DataFrame,
        test_horizon: int,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ordered = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)
        test_index = ordered.groupby("unique_id", observed=True).tail(test_horizon).index
        test_df = ordered.loc[test_index].copy()
        train_df = ordered.drop(test_index).copy()
        return train_df, test_df

    @staticmethod
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

    @staticmethod
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
            if MlCommon._log_dataset_tables_enabled():
                mlflow.log_table(
                    dataset_df,
                    f"{artifact_prefix}/datasets/{split_name}.json",
                )

    @staticmethod
    def _log_dataset_tables_enabled() -> bool:
        return os.getenv("MLFLOW_LOG_DATASET_TABLES", "0").lower() in {
            "1",
            "true",
            "yes",
        }

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _regression_metrics_by_unique_id(joined_df: pd.DataFrame) -> pd.DataFrame:
        frames = []
        for unique_id, ticker_df in joined_df.groupby("unique_id", observed=True):
            metrics = MlCommon._regression_metrics(ticker_df)
            if metrics.empty:
                continue

            metrics.insert(0, "unique_id", unique_id)
            frames.append(metrics)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
