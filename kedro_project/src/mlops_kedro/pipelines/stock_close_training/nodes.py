import os
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl
import yaml

from .config_resolvers import resolve_column_config
from .feature_engineering import (
    build_conventional_gap_trading_features,
    build_stock_close_model_dataset,
    build_stock_price_indicator_features,
    read_delta_table,
    read_silver_stock_prices,
    stock_price_indicator_features_path,
    write_delta_table,
)
from .ml.mlforecast_training import (
    build_auto_mlforecast_spec,
    make_train_test_split,
    train_auto_mlforecast_models_from_split,
)
from .ml.pecnet_training import (
    build_pecnet_spec,
    train_pecnet_models_from_split,
)
from .ml.stats_training import (
    build_statsforecast_spec,
    train_statsforecast_models_from_split,
)
from .serving.feast_store import (
    load_stock_model_training_dataset_from_feast_online,
    publish_close_model_dataset as publish_close_model_dataset_to_store,
    publish_conventional_gap_trading as publish_conventional_gap_trading_to_store,
    publish_model_features,
)


def _log_step(step_name: str, **metadata: Any) -> None:
    print(f"\n[{step_name}]")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


def _bool_from_env(env_name: str, default: bool) -> bool:
    value = os.getenv(env_name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes"}


def _int_from_env(env_name: str, default: int) -> int:
    value = os.getenv(env_name)
    if value is None:
        return default
    return int(value)


def _int_from_envs(env_names: tuple[str, ...], default: int) -> tuple[int, str]:
    for env_name in env_names:
        value = os.getenv(env_name)
        if value is not None:
            return int(value), f"env:{env_name}"
    return int(default), "parameters"


def _set_env_default(env_name: str, value: Any) -> None:
    os.environ.setdefault(env_name, str(value))


def _valid_wandb_api_key(value: Any) -> bool:
    return isinstance(value, str) and len(value.strip()) >= 40


def _wandb_base_url(value: Any) -> str:
    if not isinstance(value, str):
        return ""

    base_url = value.strip()
    local_wandb_hosts = (
        "localhost:8080",
        "127.0.0.1:8080",
        "host.docker.internal:8080",
    )
    if any(host in base_url for host in local_wandb_hosts):
        return ""
    return base_url


def _load_local_credentials() -> dict[str, Any]:
    env_credentials_path = os.getenv("KEDRO_CREDENTIALS_PATH")
    candidates = [
        Path(env_credentials_path) if env_credentials_path else None,
        Path.cwd() / "conf" / "local" / "credentials.yml",
        Path(__file__).resolve().parents[4] / "conf" / "local" / "credentials.yml",
        Path("/workspaces/yahooquery_lakehouse_revamp/kedro_project/conf/local/credentials.yml"),
    ]
    for path in candidates:
        if path is None:
            continue
        if not path.exists():
            continue
        with path.open() as file:
            return yaml.safe_load(file) or {}
    return {}


def start_config(
    data_preprocessing_parameters: dict[str, Any] | None = None,
    machine_learning_parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    data_preprocessing_parameters = data_preprocessing_parameters or {}
    machine_learning_parameters = machine_learning_parameters or {}
    credentials = _load_local_credentials()
    wandb_credentials = credentials.get("wandb", {})
    delta_lake = data_preprocessing_parameters.get("delta_lake", {})
    feature_engineering = data_preprocessing_parameters.get("feature_engineering", {})
    conventional_gap_trading = data_preprocessing_parameters.get(
        "conventional_gap_trading",
        {},
    )
    time_encoding = data_preprocessing_parameters.get("time_encoding", {})
    columns_config = resolve_column_config(
        data_preprocessing_parameters.get("columns", {})
    )
    training = machine_learning_parameters.get("training", {})
    mlflow_config = machine_learning_parameters.get("mlflow", {})
    mlforecast_config = machine_learning_parameters.get("mlforecast", {})
    statsforecast_config = machine_learning_parameters.get("statsforecast", {})
    pecnet_config = machine_learning_parameters.get("pecnet", {})
    validation_horizon, validation_horizon_source = _int_from_envs(
        ("MODEL_VALIDATION_HORIZON", "MLFORECAST_VALIDATION_HORIZON"),
        training.get(
            "validation_horizon",
            mlforecast_config.get("validation_horizon", 1),
        ),
    )
    test_horizon, test_horizon_source = _int_from_envs(
        ("MODEL_TEST_HORIZON", "MLFORECAST_TEST_HORIZON"),
        training.get(
            "test_horizon",
            mlforecast_config.get("test_horizon", 3),
        ),
    )
    configured_wandb_base_url = _wandb_base_url(
        os.getenv(
            "WANDB_BASE_URL",
            pecnet_config.get("wandb_base_url", ""),
        )
    )
    runtime_config = machine_learning_parameters.get("runtime", {})

    config = {
        "bucket": os.getenv(
            "DELTA_LAKE_S3_BUCKET",
            delta_lake.get("bucket", "delta-lake-bucket"),
        ),
        "training_data_source": os.getenv(
            "MODEL_TRAINING_DATA_SOURCE",
            training.get("data_source", "feast_online"),
        ),
        "publish_indicator_features": _bool_from_env(
            "PUBLISH_INDICATOR_FEATURES",
            feature_engineering.get("publish_indicator_features", True),
        ),
        "publish_conventional_gap_trading": _bool_from_env(
            "PUBLISH_CONVENTIONAL_GAP_TRADING",
            conventional_gap_trading.get("publish_to_timescale", True),
        ),
        "mlflow_tracking_uri": os.getenv(
            "MLFLOW_TRACKING_URI",
            mlflow_config.get("tracking_uri", "http://host.docker.internal:5001"),
        ),
        "mlflow_experiment_name": os.getenv(
            "MLFLOW_EXPERIMENT_NAME",
            mlflow_config.get("experiment_name", "stock_close_training"),
        ),
        "mlflow_tier_experiment_prefix": os.getenv(
            "MLFLOW_TIER_EXPERIMENT_PREFIX",
            mlflow_config.get("tier_experiment_prefix", "stock_close"),
        ),
        "mlflow_request_timeout": _int_from_env(
            "MLFLOW_HTTP_REQUEST_TIMEOUT",
            mlflow_config.get("request_timeout", 10),
        ),
        "mlflow_request_max_retries": _int_from_env(
            "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
            mlflow_config.get("request_max_retries", 1),
        ),
        "mlflow_request_backoff_factor": _int_from_env(
            "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR",
            mlflow_config.get("request_backoff_factor", 1),
        ),
        "freq": os.getenv("MLFORECAST_FREQ", mlforecast_config.get("freq", "B")),
        "validation_horizon": validation_horizon,
        "validation_horizon_source": validation_horizon_source,
        "test_horizon": test_horizon,
        "test_horizon_source": test_horizon_source,
        "n_windows": _int_from_env(
            "MLFORECAST_N_WINDOWS",
            mlforecast_config.get("n_windows", 1),
        ),
        "n_trials": _int_from_env(
            "MLFORECAST_N_TRIALS",
            mlforecast_config.get("n_trials", 1),
        ),
        "verbose": _bool_from_env(
            "MLFORECAST_VERBOSE",
            mlforecast_config.get("verbose", True),
        ),
        "mlforecast_models": mlforecast_config.get("models"),
        "mlforecast_num_threads": _int_from_env(
            "MLFORECAST_NUM_THREADS",
            runtime_config.get("mlforecast_num_threads", 1),
        ),
        "model_n_jobs": _int_from_env(
            "MODEL_N_JOBS",
            runtime_config.get("model_n_jobs", 1),
        ),
        "model_max_estimators": _int_from_env(
            "MODEL_MAX_ESTIMATORS",
            runtime_config.get("model_max_estimators", 300),
        ),
        "statsforecast": {
            "enabled": _bool_from_env(
                "STATSFORECAST_ENABLED",
                statsforecast_config.get("enabled", True),
            ),
            "freq": os.getenv(
                "STATSFORECAST_FREQ",
                statsforecast_config.get("freq", mlforecast_config.get("freq", "B")),
            ),
            "seasonal_length": _int_from_env(
                "STATSFORECAST_SEASONAL_LENGTH",
                statsforecast_config.get("seasonal_length", 5),
            ),
            "conformal_n_windows": _int_from_env(
                "STATSFORECAST_CONFORMAL_N_WINDOWS",
                statsforecast_config.get(
                    "conformal_n_windows",
                    mlforecast_config.get("n_windows", 3),
                ),
            ),
            "level": statsforecast_config.get("level", [80, 95]),
            "models": statsforecast_config.get("models"),
            "verbose": _bool_from_env(
                "STATSFORECAST_VERBOSE",
                statsforecast_config.get("verbose", True),
            ),
        },
        "pecnet": {
            "enabled": _bool_from_env(
                "PECNET_ENABLED",
                pecnet_config.get("enabled", True),
            ),
            "feature_columns_by_tier": pecnet_config.get("feature_columns_by_tier"),
            "preprocess_params": pecnet_config.get("preprocess_params"),
            "hyperparams": pecnet_config.get("hyperparams"),
            "wandb_project": os.getenv(
                "WANDB_PROJECT",
                pecnet_config.get("wandb_project", "stock-close-pecnet"),
            ),
            "wandb_mode": os.getenv(
                "WANDB_MODE",
                pecnet_config.get("wandb_mode", "offline"),
            ),
            "wandb_base_url": configured_wandb_base_url,
        },
        "time_encoding": time_encoding,
        "columns": columns_config,
        "model_tier_feature_columns": {
            "tier1": columns_config["tier_1_features"],
            "tier2": columns_config["tier_2_features"],
        },
    }
    config["indicator_features_path"] = feature_engineering.get(
        "indicator_features_path",
        stock_price_indicator_features_path(config["bucket"]),
    )
    os.environ["MLFLOW_TRACKING_URI"] = config["mlflow_tracking_uri"]
    os.environ["MLFLOW_EXPERIMENT_NAME"] = config["mlflow_experiment_name"]
    os.environ["MLFLOW_TIER_EXPERIMENT_PREFIX"] = config[
        "mlflow_tier_experiment_prefix"
    ]
    _set_env_default(
        "MLFLOW_HTTP_REQUEST_TIMEOUT",
        config["mlflow_request_timeout"],
    )
    _set_env_default(
        "MLFLOW_HTTP_REQUEST_MAX_RETRIES",
        config["mlflow_request_max_retries"],
    )
    _set_env_default(
        "MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR",
        config["mlflow_request_backoff_factor"],
    )
    _set_env_default("MLFORECAST_NUM_THREADS", config["mlforecast_num_threads"])
    _set_env_default("MODEL_N_JOBS", config["model_n_jobs"])
    _set_env_default("MODEL_MAX_ESTIMATORS", config["model_max_estimators"])
    env_wandb_api_key = os.getenv("WANDB_API_KEY")
    credentials_wandb_api_key = wandb_credentials.get("api_key")
    wandb_api_key = (
        env_wandb_api_key
        if _valid_wandb_api_key(env_wandb_api_key)
        else credentials_wandb_api_key
    )
    config["pecnet"]["wandb_api_key_loaded"] = _valid_wandb_api_key(wandb_api_key)
    if wandb_api_key:
        os.environ["WANDB_API_KEY"] = str(wandb_api_key)
    config["pecnet"]["wandb_base_url"] = configured_wandb_base_url
    if config["pecnet"].get("wandb_base_url"):
        os.environ["WANDB_BASE_URL"] = str(config["pecnet"]["wandb_base_url"])
    else:
        os.environ.pop("WANDB_BASE_URL", None)
    _log_step("start", **config)
    return config


def _feature_columns_for_tier(
    run_config: dict[str, Any],
    tier_name: str,
) -> list[str]:
    try:
        return run_config["model_tier_feature_columns"][tier_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown model tier {tier_name!r}. "
            f"Expected one of {sorted(run_config['model_tier_feature_columns'])}."
        ) from exc


def prepare_close_model_dataset(
    run_config: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    silver_stock_prices = read_silver_stock_prices(run_config["bucket"])
    model_dataset = build_stock_close_model_dataset(
        silver_stock_prices,
        run_config["columns"],
        run_config["time_encoding"],
    )
    metadata = {
        "silver_rows": len(silver_stock_prices),
        "close_model_rows": len(model_dataset),
        "symbols": model_dataset["unique_id"].n_unique(),
        "min_ds": model_dataset["ds"].min(),
        "max_ds": model_dataset["ds"].max(),
    }
    _log_step("prepare_close_model_dataset", **metadata)
    return model_dataset, metadata


def publish_close_model_dataset(stock_close_model_dataset: pl.DataFrame) -> dict[str, Any]:
    metadata = publish_close_model_dataset_to_store(stock_close_model_dataset)
    _log_step("publish_close_model_dataset", **metadata)
    return metadata


def prepare_indicator_features(
    run_config: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if not run_config["publish_indicator_features"]:
        indicator_features = pl.DataFrame()
        metadata = {
            "publish_indicator_features": False,
            "indicator_feature_rows": 0,
            "indicator_features_path": run_config["indicator_features_path"],
        }
        _log_step("prepare_indicator_features", **metadata)
        return indicator_features, metadata

    silver_stock_prices = read_silver_stock_prices(run_config["bucket"])
    indicator_features = build_stock_price_indicator_features(
        silver_stock_prices,
        run_config["columns"],
        run_config["time_encoding"],
    )
    write_delta_table(run_config["indicator_features_path"], indicator_features)
    metadata = {
        "publish_indicator_features": True,
        "silver_rows": len(silver_stock_prices),
        "indicator_feature_rows": len(indicator_features),
        "indicator_features_path": run_config["indicator_features_path"],
    }
    _log_step("prepare_indicator_features", **metadata)
    return indicator_features, metadata


def load_indicator_features(
    run_config: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    indicator_features = read_delta_table(run_config["indicator_features_path"])
    metadata = {
        "indicator_feature_rows": len(indicator_features),
        "indicator_features_path": run_config["indicator_features_path"],
    }
    _log_step("load_indicator_features", **metadata)
    return indicator_features, metadata


def publish_indicator_model_features(
    stock_price_indicator_features: pl.DataFrame,
    indicator_feature_metadata: dict[str, Any],
) -> dict[str, Any]:
    if stock_price_indicator_features.is_empty():
        metadata = {
            "skipped": True,
            "reason": "stock_price_indicator_features is empty",
        }
        _log_step("publish_indicator_model_features", **metadata)
        return metadata

    metadata = publish_model_features(stock_price_indicator_features)
    metadata["indicator_feature_rows"] = indicator_feature_metadata.get(
        "indicator_feature_rows",
        len(stock_price_indicator_features),
    )
    _log_step("publish_indicator_model_features", **metadata)
    return metadata


def prepare_conventional_gap_trading(
    stock_price_indicator_features: pl.DataFrame,
    run_config: dict[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    if not run_config["publish_conventional_gap_trading"]:
        conventional_gap_trading = pl.DataFrame()
        metadata = {
            "publish_conventional_gap_trading": False,
            "conventional_gap_trading_rows": 0,
        }
        _log_step("prepare_conventional_gap_trading", **metadata)
        return conventional_gap_trading, metadata

    if stock_price_indicator_features.is_empty():
        conventional_gap_trading = pl.DataFrame()
        metadata = {
            "publish_conventional_gap_trading": True,
            "indicator_feature_rows": 0,
            "conventional_gap_trading_rows": 0,
            "signals": {},
        }
        _log_step("prepare_conventional_gap_trading", **metadata)
        return conventional_gap_trading, metadata

    conventional_gap_trading = build_conventional_gap_trading_features(
        stock_price_indicator_features,
        run_config["columns"],
    )
    metadata = {
        "publish_conventional_gap_trading": True,
        "indicator_feature_rows": len(stock_price_indicator_features),
        "conventional_gap_trading_rows": len(conventional_gap_trading),
        "signals": (
            conventional_gap_trading.get_column("Gap_Type")
            .value_counts()
            .to_dict(as_series=False)
            if "Gap_Type" in conventional_gap_trading.columns
            else {}
        ),
    }
    _log_step("prepare_conventional_gap_trading", **metadata)
    return conventional_gap_trading, metadata


def publish_conventional_gap_trading(
    conventional_gap_trading: pl.DataFrame,
    conventional_gap_trading_metadata: dict[str, Any],
) -> dict[str, Any]:
    if not conventional_gap_trading_metadata["publish_conventional_gap_trading"]:
        metadata = {
            "skipped": True,
            "reason": "publish_conventional_gap_trading is false",
        }
        _log_step("publish_conventional_gap_trading", **metadata)
        return metadata

    metadata = publish_conventional_gap_trading_to_store(conventional_gap_trading)
    _log_step("publish_conventional_gap_trading", **metadata)
    return metadata


def load_model_training_dataset(
    run_config: dict[str, Any],
    *,
    tier_name: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    feature_columns = _feature_columns_for_tier(run_config, tier_name)
    training_dataset = load_stock_model_training_dataset_from_feast_online(
        feature_columns
    )
    metadata = {
        "tier": tier_name,
        "training_data_source": "feast_online",
        "training_rows": len(training_dataset),
        "symbols": training_dataset["unique_id"].n_unique()
        if "unique_id" in training_dataset.columns
        else 0,
        "feature_columns": feature_columns,
    }
    _log_step(f"load_{tier_name}_training_dataset", **metadata)
    return training_dataset, metadata


def load_model_training_dataset_after_feature_publish(
    run_config: dict[str, Any],
    model_feature_publish_metadata: dict[str, Any],
    *,
    tier_name: str,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    return load_model_training_dataset(run_config, tier_name=tier_name)


def train_test_split_for_tier(
    stock_close_training_dataset: pl.DataFrame,
    run_config: dict[str, Any],
    *,
    tier_name: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    split = make_train_test_split(
        stock_close_training_dataset,
        test_horizon=run_config["test_horizon"],
    )
    metadata = {
        "tier": tier_name,
        "train_rows": len(split["train"]),
        "test_rows": len(split["test"]),
        "test_horizon": run_config["test_horizon"],
    }
    _log_step(f"{tier_name}_train_test_split", **metadata)
    return split, metadata


def build_model_spec(
    run_config: dict[str, Any],
    *,
    tier_name: str = "tier1",
) -> dict[str, Any]:
    model_spec = build_auto_mlforecast_spec(
        freq=run_config["freq"],
        validation_horizon=run_config["validation_horizon"],
        test_horizon=run_config["test_horizon"],
        n_windows=run_config["n_windows"],
        n_trials=run_config["n_trials"],
        verbose=run_config["verbose"],
        models=run_config["mlforecast_models"],
        tier_name=tier_name,
    )
    _log_step(f"build_{tier_name}_mlforecast_model_spec", **model_spec)
    return model_spec


def train_models(
    stock_close_train_test_split: dict[str, pd.DataFrame],
    stock_close_model_spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    result = train_auto_mlforecast_models_from_split(
        stock_close_train_test_split,
        model_spec=stock_close_model_spec,
    )
    regression_metrics = result["regression_metrics"]
    long_direction_metrics = result["long_direction_metrics"]
    predictions = result["predictions"]

    best_model = None
    if not regression_metrics.empty:
        best_model = (
            regression_metrics.sort_values("rmse", ascending=True).iloc[0]["model"]
        )

    metadata = {
        "tier": stock_close_model_spec.get("tier_name", "tier1"),
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "best_model": best_model,
        "regression_metric_rows": len(regression_metrics),
        "long_direction_metric_rows": len(long_direction_metrics),
        "prediction_rows": len(predictions),
    }
    _log_step("train_models", **metadata)
    return regression_metrics, long_direction_metrics, predictions, metadata


def build_statsforecast_model_spec_for_tier(
    run_config: dict[str, Any],
    *,
    tier_name: str,
) -> dict[str, Any]:
    stats_config = run_config["statsforecast"]
    model_spec = build_statsforecast_spec(
        freq=stats_config["freq"],
        seasonal_length=stats_config["seasonal_length"],
        validation_horizon=run_config["validation_horizon"],
        test_horizon=run_config["test_horizon"],
        conformal_n_windows=stats_config["conformal_n_windows"],
        level=stats_config["level"],
        models=stats_config["models"],
        verbose=stats_config["verbose"],
        tier_name=tier_name,
    )
    model_spec["enabled"] = stats_config["enabled"]
    _log_step(f"build_{tier_name}_statsforecast_model_spec", **model_spec)
    return model_spec


def train_statsforecast_models(
    stock_close_train_test_split: dict[str, pd.DataFrame],
    stock_close_statsforecast_model_spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not stock_close_statsforecast_model_spec.get("enabled", True):
        empty = pd.DataFrame()
        metadata = {"skipped": True, "reason": "statsforecast disabled"}
        _log_step("train_statsforecast_models", **metadata)
        return empty, empty, empty, metadata

    result = train_statsforecast_models_from_split(
        stock_close_train_test_split,
        model_spec=stock_close_statsforecast_model_spec,
    )
    regression_metrics = result["regression_metrics"]
    long_direction_metrics = result["long_direction_metrics"]
    predictions = result["predictions"]
    best_model = None
    if not regression_metrics.empty:
        best_model = (
            regression_metrics.sort_values("rmse", ascending=True).iloc[0]["model"]
        )

    metadata = {
        "tier": stock_close_statsforecast_model_spec.get("tier_name", "tier1"),
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "best_model": best_model,
        "regression_metric_rows": len(regression_metrics),
        "long_direction_metric_rows": len(long_direction_metrics),
        "prediction_rows": len(predictions),
    }
    _log_step("train_statsforecast_models", **metadata)
    return regression_metrics, long_direction_metrics, predictions, metadata


def build_pecnet_model_spec_for_tier(
    run_config: dict[str, Any],
    *,
    tier_name: str,
) -> dict[str, Any]:
    pecnet_config = run_config["pecnet"]
    feature_columns_by_tier = pecnet_config.get("feature_columns_by_tier") or {}
    feature_columns = feature_columns_by_tier.get(
        tier_name,
        _feature_columns_for_tier(run_config, tier_name),
    )
    model_spec = build_pecnet_spec(
        enabled=pecnet_config["enabled"],
        test_horizon=run_config["test_horizon"],
        feature_columns=feature_columns,
        preprocess_params=pecnet_config["preprocess_params"],
        hyperparams=pecnet_config["hyperparams"],
        wandb_project=pecnet_config["wandb_project"],
        wandb_mode=pecnet_config["wandb_mode"],
        tier_name=tier_name,
    )
    _log_step(f"build_{tier_name}_pecnet_model_spec", **model_spec)
    return model_spec


def train_pecnet_models(
    stock_close_pecnet_train_test_split: dict[str, pd.DataFrame],
    stock_close_pecnet_model_spec: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    result = train_pecnet_models_from_split(
        stock_close_pecnet_train_test_split,
        model_spec=stock_close_pecnet_model_spec,
    )
    regression_metrics = result["regression_metrics"]
    long_direction_metrics = result["long_direction_metrics"]
    predictions = result["predictions"]
    best_model = None
    if not regression_metrics.empty and "rmse" in regression_metrics.columns:
        best_model = (
            regression_metrics.sort_values("rmse", ascending=True).iloc[0]["model"]
        )

    metadata = {
        "tier": stock_close_pecnet_model_spec.get("tier_name", "tier1"),
        "train_rows": result["train_rows"],
        "test_rows": result["test_rows"],
        "best_model": best_model,
        "regression_metric_rows": len(regression_metrics),
        "long_direction_metric_rows": len(long_direction_metrics),
        "prediction_rows": len(predictions),
        "models": list(result.get("models", {}).keys()),
    }
    _log_step("train_pecnet_models", **metadata)
    return regression_metrics, long_direction_metrics, predictions, metadata


def summarize_training(*metadata_items: dict[str, Any]) -> dict[str, Any]:
    summary = {"sections": list(metadata_items)}
    _log_step("summarize_training", sections=len(metadata_items))
    return summary


def summarize_machine_learning(*metadata_items: dict[str, Any]) -> dict[str, Any]:
    summary = {"sections": list(metadata_items)}
    _log_step("summarize_machine_learning", sections=len(metadata_items))
    return summary
