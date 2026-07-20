from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config_resolvers import resolve_column_config
from .features.feature_sets import stock_price_indicator_features_path


class StockCloseNodeUtils:

    @staticmethod
    def _log_step(step_name: str, **metadata: Any) -> None:
        print(f"\n[{step_name}]")
        for key, value in metadata.items():
            print(f"  {key}: {value}")

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes"}
        return bool(value)

    @staticmethod
    def _as_int(value: Any, default: int) -> int:
        return int(default if value is None else value)

    @staticmethod
    def _running_in_container() -> bool:
        return Path("/.dockerenv").exists() or Path("/workspaces").exists()

    @staticmethod
    def _resolve_local_service_url(value: Any, *, port: int) -> str:
        value = str(value or "auto")
        if value.lower() == "auto":
            host = "host.docker.internal" if StockCloseNodeUtils._running_in_container() else "127.0.0.1"
            return f"http://{host}:{port}"
        if StockCloseNodeUtils._running_in_container():
            return value.replace("127.0.0.1", "host.docker.internal").replace(
                "localhost",
                "host.docker.internal",
            )
        return value.replace("host.docker.internal", "127.0.0.1")

    @staticmethod
    def _bucket(delta_lake_params: dict[str, Any] | None) -> str:
        return (delta_lake_params or {}).get("bucket", "delta-lake-bucket")

    @staticmethod
    def _columns(columns_params: dict[str, Any] | None) -> dict[str, list[str]]:
        return resolve_column_config(columns_params or {})

    @staticmethod
    def _indicator_features_path(
        delta_lake_params: dict[str, Any] | None,
        feature_engineering_params: dict[str, Any] | None,
    ) -> str:
        feature_engineering_params = feature_engineering_params or {}
        return feature_engineering_params.get(
            "indicator_features_path",
            stock_price_indicator_features_path(StockCloseNodeUtils._bucket(delta_lake_params)),
        )

    @staticmethod
    def _model_tier_feature_columns(
        columns_params: dict[str, Any] | None,
    ) -> dict[str, list[str]]:
        columns_config = StockCloseNodeUtils._columns(columns_params)
        return {
            tier_name.removesuffix("_features").replace("_", ""): feature_columns
            for tier_name, feature_columns in columns_config.items()
            if tier_name.startswith("tier_") and tier_name.endswith("_features")
        }

    @staticmethod
    def _apply_training_environment(
        *,
        mlflow_params: dict[str, Any] | None = None,
        runtime_params: dict[str, Any] | None = None,
        mlforecast_params: dict[str, Any] | None = None,
    ) -> None:
        mlflow_params = mlflow_params or {}
        runtime_params = runtime_params or {}
        mlforecast_params = mlforecast_params or {}

        os.environ["MLFLOW_TRACKING_URI"] = str(
            StockCloseNodeUtils._resolve_local_service_url(mlflow_params.get("tracking_uri"), port=5001)
        )
        os.environ["MLFLOW_EXPERIMENT_NAME"] = str(
            mlflow_params.get("experiment_name", "stock_close_training")
        )
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = str(
            StockCloseNodeUtils._resolve_local_service_url(mlflow_params.get("s3_endpoint_url"), port=9000)
        )
        os.environ["AWS_ACCESS_KEY_ID"] = str(
            mlflow_params.get("aws_access_key_id", "admin")
        )
        os.environ["AWS_SECRET_ACCESS_KEY"] = str(
            mlflow_params.get("aws_secret_access_key", "admin1234")
        )
        os.environ["AWS_DEFAULT_REGION"] = str(
            mlflow_params.get("aws_default_region", "us-east-1")
        )
        os.environ["MLFLOW_TIER_EXPERIMENT_PREFIX"] = str(
            mlflow_params.get("tier_experiment_prefix", "stock_close")
        )
        os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(
            mlflow_params.get("request_timeout", 60)
        )
        os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = str(
            mlflow_params.get("request_max_retries", 1)
        )
        os.environ["MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR"] = str(
            mlflow_params.get("request_backoff_factor", 1)
        )
        os.environ["MLFLOW_SERVER_READY_TIMEOUT"] = str(
            mlflow_params.get("server_ready_timeout", 30)
        )
        os.environ["MLFLOW_EXPERIMENT_SETUP_RETRIES"] = str(
            mlflow_params.get("experiment_setup_retries", 3)
        )
        os.environ["MLFORECAST_NUM_THREADS"] = str(
            runtime_params.get("mlforecast_num_threads", 1)
        )
        os.environ["MODEL_N_JOBS"] = str(runtime_params.get("model_n_jobs", 1))
        os.environ["MODEL_MAX_ESTIMATORS"] = str(
            runtime_params.get("model_max_estimators", 300)
        )
        os.environ["PECNET_N_JOBS"] = str(runtime_params.get("pecnet_n_jobs", 1))
        os.environ["PECNET_TORCH_THREADS_PER_WORKER"] = str(
            runtime_params.get("pecnet_torch_threads_per_worker", 1)
        )
        os.environ["PECNET_TORCH_DEVICE"] = str(
            runtime_params.get(
                "pecnet_torch_device",
                os.environ.get("PECNET_TORCH_DEVICE", "auto"),
            )
        )
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
            "1" if StockCloseNodeUtils._as_bool(runtime_params.get("pecnet_mps_fallback"), True) else "0"
        )
        os.environ["LOCAL_ARTIFACT_DIR"] = str(
            runtime_params.get(
                "local_artifact_dir",
                os.environ.get(
                    "LOCAL_ARTIFACT_DIR",
                    "/workspaces/yahooquery_lakehouse_revamp/artifacts/stock_close_training",
                ),
            )
        )
        os.environ["LOCAL_ARTIFACTS_ENABLED"] = (
            "1" if StockCloseNodeUtils._as_bool(runtime_params.get("local_artifacts_enabled"), True) else "0"
        )
        os.environ["MODEL_ESTIMATOR_VERBOSE"] = (
            "1" if StockCloseNodeUtils._as_bool(mlforecast_params.get("estimator_verbose"), False) else "0"
        )

    @staticmethod
    def _feature_columns_for_tier(
        columns_params: dict[str, Any] | None,
        tier_name: str,
    ) -> list[str]:
        model_tier_feature_columns = StockCloseNodeUtils._model_tier_feature_columns(columns_params)
        try:
            return model_tier_feature_columns[tier_name]
        except KeyError as exc:
            raise ValueError(
                f"Unknown model tier {tier_name!r}. "
                f"Expected one of {sorted(model_tier_feature_columns)}."
            ) from exc

    @staticmethod
    def _merge_tier_overrides(
        base_params: dict[str, Any] | None,
        overrides_by_tier: dict[str, dict[str, Any]] | None,
        tier_name: str,
    ) -> dict[str, Any]:
        return {
            **(base_params or {}),
            **((overrides_by_tier or {}).get(tier_name) or {}),
        }
