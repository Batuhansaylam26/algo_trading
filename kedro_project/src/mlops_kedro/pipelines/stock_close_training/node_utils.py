from __future__ import annotations

import os
from typing import Any

from .config_resolvers import resolve_column_config
from .features.feature_sets import stock_price_indicator_features_path


def _log_step(step_name: str, **metadata: Any) -> None:
    print(f"\n[{step_name}]")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes"}
    return bool(value)

def _as_int(value: Any, default: int) -> int:
    return int(default if value is None else value)

def _bucket(delta_lake_params: dict[str, Any] | None) -> str:
    return (delta_lake_params or {}).get("bucket", "delta-lake-bucket")

def _columns(columns_params: dict[str, Any] | None) -> dict[str, list[str]]:
    return resolve_column_config(columns_params or {})

def _indicator_features_path(
    delta_lake_params: dict[str, Any] | None,
    feature_engineering_params: dict[str, Any] | None,
) -> str:
    feature_engineering_params = feature_engineering_params or {}
    return feature_engineering_params.get(
        "indicator_features_path",
        stock_price_indicator_features_path(_bucket(delta_lake_params)),
    )

def _model_tier_feature_columns(
    columns_params: dict[str, Any] | None,
) -> dict[str, list[str]]:
    columns_config = _columns(columns_params)
    return {
        tier_name.removesuffix("_features").replace("_", ""): feature_columns
        for tier_name, feature_columns in columns_config.items()
        if tier_name.startswith("tier_") and tier_name.endswith("_features")
    }

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
        mlflow_params.get("tracking_uri", "http://host.docker.internal:5001")
    )
    os.environ["MLFLOW_EXPERIMENT_NAME"] = str(
        mlflow_params.get("experiment_name", "stock_close_training")
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
    os.environ["MLFORECAST_NUM_THREADS"] = str(
        runtime_params.get("mlforecast_num_threads", 1)
    )
    os.environ["MODEL_N_JOBS"] = str(runtime_params.get("model_n_jobs", 1))
    os.environ["MODEL_MAX_ESTIMATORS"] = str(
        runtime_params.get("model_max_estimators", 300)
    )
    os.environ["MODEL_ESTIMATOR_VERBOSE"] = (
        "1" if _as_bool(mlforecast_params.get("estimator_verbose"), False) else "0"
    )

def _feature_columns_for_tier(
    columns_params: dict[str, Any] | None,
    tier_name: str,
) -> list[str]:
    model_tier_feature_columns = _model_tier_feature_columns(columns_params)
    try:
        return model_tier_feature_columns[tier_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model tier {tier_name!r}. "
            f"Expected one of {sorted(model_tier_feature_columns)}."
        ) from exc

def _merge_tier_overrides(
    base_params: dict[str, Any] | None,
    overrides_by_tier: dict[str, dict[str, Any]] | None,
    tier_name: str,
) -> dict[str, Any]:
    return {
        **(base_params or {}),
        **((overrides_by_tier or {}).get(tier_name) or {}),
    }
