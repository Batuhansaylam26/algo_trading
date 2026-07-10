from __future__ import annotations

import logging
import os
import re
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import polars as pl

from .metrics import long_only_directional_metrics
from .plots import log_forecast_plots, pecnet_prediction_figure
from .common import (
    _regression_metrics,
    configure_mlflow_tracking,
    log_mlflow_datasets,
    non_feature_columns,
    split_train_test_by_horizon,
    tier_experiment_name,
)


LOGGER = logging.getLogger(__name__)


def build_pecnet_spec(
    *,
    enabled: bool = True,
    test_horizon: int = 5,
    feature_columns: list[str] | None = None,
    preprocess_params: dict[str, Any] | None = None,
    hyperparams: dict[str, Any] | None = None,
    wandb_project: str = "stock-close-pecnet",
    wandb_mode: str = "offline",
    tier_name: str = "tier1",
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "tier_name": tier_name,
        "test_horizon": test_horizon,
        "feature_columns": feature_columns or [],
        "preprocess_params": preprocess_params or {},
        "hyperparams": hyperparams or {},
        "wandb_project": wandb_project,
        "wandb_mode": wandb_mode,
    }


def to_pecnet_frame(df: pl.DataFrame) -> pd.DataFrame:
    available_feature_columns = [
        column for column in df.columns if column not in non_feature_columns()
    ]
    return (
        df.select(
            pl.col("unique_id").cast(pl.Utf8),
            pl.col("ds").cast(pl.Datetime("us"), strict=False),
            pl.col("y").cast(pl.Float64, strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in available_feature_columns
            ],
        )
        .drop_nulls(["unique_id", "ds", "y", *available_feature_columns])
        .unique(subset=["unique_id", "ds"], keep="last", maintain_order=True)
        .sort(["unique_id", "ds"])
        .to_pandas()
    )


def make_pecnet_train_test_split(
    dataset: pl.DataFrame,
    *,
    test_horizon: int,
) -> dict[str, pd.DataFrame]:
    model_df = to_pecnet_frame(dataset)
    train_df, test_df = split_train_test_by_horizon(model_df, test_horizon)
    return {
        "full": model_df,
        "train": train_df,
        "test": test_df,
    }


def _resolve_pecnetframework_path() -> Path:
    candidates = [
        "/opt/pecnetframework",
        os.getenv("PECNETFRAMEWORK_PATH"),
        str(Path(__file__).resolve().parents[6] / "pecnetframework"),
        "/workspaces/yahooquery_lakehouse_revamp/pecnetframework",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser().resolve()
        if (path / "pecnet").exists():
            return path
    raise FileNotFoundError(
        "pecnetframework klasoru bulunamadi. PECNETFRAMEWORK_PATH env var ile "
        "klasoru goster veya repo root altina pecnetframework koy."
    )


def _load_pecnet_runtime():
    pecnet_path = _resolve_pecnetframework_path()
    if str(pecnet_path) not in sys.path:
        sys.path.insert(0, str(pecnet_path))

    from pecnet.network import PecnetBuilder  # noqa: PLC0415
    from pecnet.models.BasicNN import BasicNN  # noqa: PLC0415
    from pecnet.preprocessing.DataPreprocessor import DataPreprocessor  # noqa: PLC0415
    from pecnet.utils import Utility  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import wandb  # noqa: PLC0415

    return Utility, PecnetBuilder, DataPreprocessor, BasicNN, torch, wandb


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


def _validate_wandb_auth(wandb_mode: str) -> None:
    if wandb_mode.lower() not in {"online", "run"}:
        return

    api_key = os.getenv("WANDB_API_KEY", "")
    if len(api_key.strip()) < 40:
        raise RuntimeError(
            "WANDB_API_KEY is missing or too short for W&B's Python SDK. "
            "Use a 40+ character key from the W&B UI, or set WANDB_MODE=offline."
        )


def _ticker_test_ratio(row_count: int, test_horizon: int) -> float:
    if row_count <= test_horizon:
        raise ValueError(
            f"PECNet needs more rows than test_horizon. rows={row_count}, "
            f"test_horizon={test_horizon}"
        )
    return min(max(test_horizon / row_count, 0.01), 0.5)


def _preprocess_ticker(
    *,
    ticker_df: pd.DataFrame,
    ticker: str,
    feature_columns: list[str],
    preprocess_params: dict[str, Any],
    test_horizon: int,
    data_preprocessor_cls,
) -> dict[str, Any]:
    dp = data_preprocessor_cls()
    dp.reset()

    ticker_df = ticker_df.sort_values("ds").copy()
    test_ratio = _ticker_test_ratio(len(ticker_df), test_horizon)
    params = {
        **preprocess_params,
        "test_ratio": test_ratio,
    }

    target_series = ticker_df["y"].to_numpy(dtype=float)
    X_train_target, X_test_target, y_train, y_test = dp.preprocess(
        data=target_series,
        **params,
    )

    feature_X_trains = []
    feature_X_tests = []
    available_feature_columns = [
        column for column in feature_columns if column in ticker_df.columns
    ]
    for column in available_feature_columns:
        X_train_feature, X_test_feature, _, _ = dp.preprocess(
            data=ticker_df[column].to_numpy(dtype=float),
            **params,
        )
        feature_X_trains.append(X_train_feature)
        feature_X_tests.append(X_test_feature)

    return {
        "ticker": ticker,
        "target_series": target_series,
        "dates": ticker_df["ds"].reset_index(drop=True),
        "X_train_target": X_train_target,
        "X_test_target": X_test_target,
        "y_train": y_train,
        "y_test": y_test,
        "feature_X_trains": feature_X_trains,
        "feature_X_tests": feature_X_tests,
        "feature_names": available_feature_columns,
        "preprocess_params": params,
        "test_ratio": test_ratio,
    }


def _iter_pecnet_basic_models(pecnet) -> list[tuple[str, Any]]:
    models = []
    for variable_index, variable_network in enumerate(pecnet.variable_networks):
        for model_index, model in enumerate(variable_network.models):
            model_name = getattr(
                model,
                "network_name",
                f"Variable_{variable_index}_Network_{model_index}",
            )
            models.append((model_name, model))

    if pecnet.error_network is not None:
        for model_index, model in enumerate(pecnet.error_network.models):
            model_name = getattr(model, "network_name", f"ErrorNetwork_{model_index}")
            models.append((model_name, model))

    if pecnet.final_network is not None:
        for model_index, model in enumerate(pecnet.final_network.models):
            model_name = getattr(model, "network_name", f"FinalNetwork_{model_index}")
            models.append((model_name, model))

    return models


def _pecnet_epoch_metrics_frame(
    *,
    pecnet,
    ticker: str,
    tier_name: str,
) -> pd.DataFrame:
    rows = []
    for network_name, model in _iter_pecnet_basic_models(pecnet):
        for epoch, loss in enumerate(getattr(model, "loss_log", []), start=1):
            rows.append(
                {
                    "tier": tier_name,
                    "ticker": ticker,
                    "network": network_name,
                    "epoch": epoch,
                    "train_loss": float(loss),
                }
            )
    return pd.DataFrame(rows)


def _define_pecnet_wandb_metrics(run) -> None:
    run.define_metric("pecnet/step")
    run.define_metric("pecnet/epoch", step_metric="pecnet/step")
    run.define_metric("pecnet/train_loss", step_metric="pecnet/step")
    run.define_metric("pecnet/tier1/train_loss", step_metric="pecnet/step")
    run.define_metric("pecnet/tier2/train_loss", step_metric="pecnet/step")


@contextmanager
def _wandb_live_pecnet_epoch_logging(
    *,
    basic_nn_cls,
    run,
    ticker: str,
    tier_name: str,
):
    original_fit = basic_nn_cls.fit
    global_step = {"value": 0}
    tier_safe = _safe_name(tier_name)
    ticker_safe = _safe_name(ticker)

    def patched_fit(self, input_values, target_values):
        original_loss_log = getattr(self, "loss_log", [])
        network_name = str(getattr(self, "network_name", "Network"))
        network_safe = _safe_name(network_name)

        class WandbLossLog(list):
            def append(loss_log, value):
                super().append(value)
                epoch = len(loss_log)
                global_step["value"] += 1
                train_loss = float(value)
                run.log(
                    {
                        "pecnet/step": global_step["value"],
                        "pecnet/epoch": epoch,
                        "pecnet/train_loss": train_loss,
                        f"pecnet/{tier_safe}/train_loss": train_loss,
                        f"pecnet/{tier_safe}/{ticker_safe}/train_loss": train_loss,
                        (
                            f"pecnet/{tier_safe}/{ticker_safe}/"
                            f"{network_safe}/train_loss"
                        ): train_loss,
                    },
                    step=global_step["value"],
                )

        live_loss_log = WandbLossLog(original_loss_log)
        self.loss_log = live_loss_log
        try:
            return original_fit(self, input_values, target_values)
        finally:
            if isinstance(original_loss_log, list):
                original_loss_log[:] = list(live_loss_log)
            self.loss_log = original_loss_log

    basic_nn_cls.fit = patched_fit
    try:
        yield
    finally:
        basic_nn_cls.fit = original_fit


def _log_pecnet_epoch_metrics_to_wandb(
    *,
    run,
    wandb_module,
    pecnet,
    ticker: str,
    tier_name: str,
) -> pd.DataFrame:
    epoch_metrics = _pecnet_epoch_metrics_frame(
        pecnet=pecnet,
        ticker=ticker,
        tier_name=tier_name,
    )
    if epoch_metrics.empty:
        return epoch_metrics

    run.log(
        {
            "pecnet_epoch_metrics_table": wandb_module.Table(
                dataframe=epoch_metrics,
            ),
        }
    )

    tier_safe = _safe_name(tier_name)
    ticker_safe = _safe_name(ticker)
    for network_name, network_metrics in epoch_metrics.groupby("network", sort=False):
        network_safe = _safe_name(str(network_name))
        run.summary[
            f"pecnet/{tier_safe}/{ticker_safe}/{network_safe}/final_train_loss"
        ] = float(network_metrics.iloc[-1]["train_loss"])
        run.summary[
            f"pecnet/{tier_safe}/{ticker_safe}/{network_safe}/min_train_loss"
        ] = float(network_metrics["train_loss"].min())

    return epoch_metrics


def _save_pecnet_model_file(
    *,
    pecnet,
    torch_module,
    ticker: str,
    tier_name: str,
) -> Path:
    tier_safe = _safe_name(tier_name)
    ticker_safe = _safe_name(ticker)
    model_dir = Path(tempfile.mkdtemp(prefix=f"pecnet_{tier_safe}_{ticker_safe}_"))
    model_path = model_dir / f"pecnet_{tier_safe}_{ticker_safe}.pt"
    torch_module.save(pecnet, model_path)
    return model_path


def _log_pecnet_model_to_wandb(
    *,
    run,
    wandb_module,
    model_path: Path,
    ticker: str,
    tier_name: str,
) -> None:
    tier_safe = _safe_name(tier_name)
    ticker_safe = _safe_name(ticker)
    artifact = wandb_module.Artifact(
        name=f"pecnet-{tier_safe}-{ticker_safe}",
        type="model",
        metadata={
            "tier": tier_name,
            "ticker": ticker,
            "model_format": "torch_pickle",
        },
    )
    artifact.add_file(str(model_path), name=model_path.name)
    run.log_artifact(artifact, aliases=["latest", tier_safe])


def _train_one_ticker(
    *,
    ticker_data: dict[str, Any],
    ticker_train_df: pd.DataFrame,
    ticker_test_df: pd.DataFrame,
    hyperparams: dict[str, Any],
    utility,
    pecnet_builder_cls,
    basic_nn_cls,
    torch_module,
    wandb_module,
    wandb_project: str,
    wandb_mode: str,
    tier_name: str,
) -> tuple[Any, pd.DataFrame, pd.DataFrame]:
    ticker = ticker_data["ticker"]
    utility.set_seed(hyperparams.get("seed", 42))
    utility.set_hyperparameters(
        learning_rate=hyperparams["learning_rate"],
        epoch_size=hyperparams["epoch_size"],
        batch_size=hyperparams["batch_size"],
        hidden_units_sizes=hyperparams["hidden_units_sizes"],
    )

    run = wandb_module.init(
        project=wandb_project,
        name=f"pecnet-{_safe_name(tier_name)}-{_safe_name(ticker)}",
        mode=wandb_mode,
        reinit=True,
        config={
            **hyperparams,
            "tier_name": tier_name,
            "ticker": ticker,
            "feature_columns": ticker_data["feature_names"],
            "test_ratio": ticker_data["test_ratio"],
        },
    )
    with run:
        _define_pecnet_wandb_metrics(run)
        with _wandb_live_pecnet_epoch_logging(
            basic_nn_cls=basic_nn_cls,
            run=run,
            ticker=ticker,
            tier_name=tier_name,
        ):
            builder = pecnet_builder_cls()
            builder.add_variable_network(
                ticker_data["X_train_target"],
                ticker_data["y_train"],
            )
            for X_train_feature in ticker_data["feature_X_trains"]:
                builder.add_variable_network(
                    X_train_feature,
                    builder.pecnet.get_target_values_for_current_variable_network(),
                )

            pecnet = builder.add_error_network().add_final_network().build()
        predictions = pecnet.predict(
            ticker_data["X_test_target"],
            *ticker_data["feature_X_tests"],
            test_target=ticker_data["y_test"],
        )

        if torch_module.is_tensor(predictions):
            predictions_array = predictions.detach().cpu().numpy().reshape(-1)
        else:
            predictions_array = np.asarray(predictions, dtype=float).reshape(-1)

        prediction_dates = (
            ticker_test_df[["unique_id", "ds"]]
            .sort_values(["unique_id", "ds"])
            .tail(len(predictions_array))
            .reset_index(drop=True)
        )
        prediction_dates["PECNet"] = predictions_array[-len(prediction_dates) :]
        joined_df = prediction_dates.merge(
            ticker_test_df[["unique_id", "ds", "y"]],
            on=["unique_id", "ds"],
            how="left",
            validate="one_to_one",
        )
        regression_df = _regression_metrics(joined_df)
        long_direction_df = long_only_directional_metrics(joined_df, ticker_train_df)
        epoch_metrics = _log_pecnet_epoch_metrics_to_wandb(
            run=run,
            wandb_module=wandb_module,
            pecnet=pecnet,
            ticker=ticker,
            tier_name=tier_name,
        )
        LOGGER.info(
            "Logged PECNet epoch metrics to W&B | tier=%s ticker=%s rows=%s",
            tier_name,
            ticker,
            len(epoch_metrics),
        )
        wandb_model_path = _save_pecnet_model_file(
            pecnet=pecnet,
            torch_module=torch_module,
            ticker=ticker,
            tier_name=tier_name,
        )
        _log_pecnet_model_to_wandb(
            run=run,
            wandb_module=wandb_module,
            model_path=wandb_model_path,
            ticker=ticker,
            tier_name=tier_name,
        )
        LOGGER.info(
            "Logged PECNet model artifact to W&B | tier=%s ticker=%s path=%s",
            tier_name,
            ticker,
            wandb_model_path,
        )

        for _, row in regression_df.iterrows():
            model_safe = _safe_name(str(row["model"]))
            mae = float(row["mae"])
            rmse = float(row["rmse"])
            run.log(
                {
                    f"{row['model']}/mae": mae,
                    f"{row['model']}/rmse": rmse,
                    f"pecnet/eval/{model_safe}/mae": mae,
                    f"pecnet/eval/{model_safe}/rmse": rmse,
                }
            )
            run.summary[f"pecnet/eval/{model_safe}/mae"] = mae
            run.summary[f"pecnet/eval/{model_safe}/rmse"] = rmse
        for _, row in long_direction_df.iterrows():
            if row[["long_accuracy", "long_precision", "long_recall"]].isna().any():
                continue
            model_safe = _safe_name(str(row["model"]))
            long_accuracy = float(row["long_accuracy"])
            long_precision = float(row["long_precision"])
            long_recall = float(row["long_recall"])
            run.log(
                {
                    f"{row['model']}/long_accuracy": long_accuracy,
                    f"{row['model']}/long_precision": long_precision,
                    f"{row['model']}/long_recall": long_recall,
                    f"pecnet/eval/{model_safe}/long_accuracy": long_accuracy,
                    f"pecnet/eval/{model_safe}/long_precision": long_precision,
                    f"pecnet/eval/{model_safe}/long_recall": long_recall,
                }
            )
            run.summary[f"pecnet/eval/{model_safe}/long_accuracy"] = long_accuracy
            run.summary[f"pecnet/eval/{model_safe}/long_precision"] = long_precision
            run.summary[f"pecnet/eval/{model_safe}/long_recall"] = long_recall

    return pecnet, joined_df, pd.concat(
        [
            regression_df.assign(metric_family="regression"),
            long_direction_df.assign(metric_family="long_direction"),
        ],
        ignore_index=True,
    )


def train_pecnet_models_from_split(
    train_test_split: dict[str, pd.DataFrame],
    *,
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    if not model_spec.get("enabled", True) or train_test_split["full"].empty:
        empty = pd.DataFrame()
        return {
            "models": {},
            "train_rows": len(train_test_split["train"]),
            "test_rows": len(train_test_split["test"]),
            "predictions": empty,
            "regression_metrics": empty,
            "long_direction_metrics": empty,
        }

    tier_name = model_spec.get("tier_name", "tier1")
    Utility, PecnetBuilder, DataPreprocessor, BasicNN, torch, wandb = (
        _load_pecnet_runtime()
    )
    _validate_wandb_auth(model_spec["wandb_mode"])
    configure_mlflow_tracking(experiment_name=tier_experiment_name(tier_name))

    full_df = train_test_split["full"]
    train_df = train_test_split["train"]
    test_df = train_test_split["test"]
    feature_columns = model_spec["feature_columns"]
    preprocess_params = model_spec["preprocess_params"]
    hyperparams = model_spec["hyperparams"]

    all_models = {}
    prediction_frames = []
    regression_frames = []
    long_direction_frames = []

    with mlflow.start_run(run_name=f"stock-close-{tier_name}-pecnet", nested=True):
        mlflow.log_params(
            {
                "tier_name": tier_name,
                "test_horizon": model_spec["test_horizon"],
                "feature_columns": ",".join(feature_columns),
                "wandb_project": model_spec["wandb_project"],
                "wandb_mode": model_spec["wandb_mode"],
                **{
                    f"pecnet.{key}": value
                    for key, value in hyperparams.items()
                    if not isinstance(value, (list, dict, tuple))
                },
            }
        )
        mlflow.log_dict(preprocess_params, f"pecnet/{tier_name}/preprocess_params.json")
        mlflow.log_dict(hyperparams, f"pecnet/{tier_name}/hyperparams.json")
        log_mlflow_datasets(
            train_df=train_df,
            test_df=test_df,
            dataset_prefix=f"stock_close_{tier_name}_pecnet",
            artifact_prefix=f"pecnet/{tier_name}",
        )

        for ticker, ticker_df in full_df.groupby("unique_id", observed=True):
            ticker_train_df = train_df[train_df["unique_id"] == ticker].copy()
            ticker_test_df = test_df[test_df["unique_id"] == ticker].copy()
            if ticker_train_df.empty or ticker_test_df.empty:
                continue

            with mlflow.start_run(
                run_name=f"pecnet-{tier_name}-{_safe_name(str(ticker))}",
                nested=True,
            ):
                log_mlflow_datasets(
                    train_df=ticker_train_df,
                    test_df=ticker_test_df,
                    dataset_prefix=(
                        f"stock_close_{tier_name}_{_safe_name(str(ticker))}_pecnet"
                    ),
                    artifact_prefix=(
                        f"pecnet/{tier_name}/tickers/{_safe_name(str(ticker))}"
                    ),
                )
                ticker_data = _preprocess_ticker(
                    ticker_df=ticker_df,
                    ticker=str(ticker),
                    feature_columns=feature_columns,
                    preprocess_params=preprocess_params,
                    test_horizon=model_spec["test_horizon"],
                    data_preprocessor_cls=DataPreprocessor,
                )
                pecnet, joined_df, combined_metrics = _train_one_ticker(
                    ticker_data=ticker_data,
                    ticker_train_df=ticker_train_df,
                    ticker_test_df=ticker_test_df,
                    hyperparams=hyperparams,
                    utility=Utility,
                    pecnet_builder_cls=PecnetBuilder,
                    basic_nn_cls=BasicNN,
                    torch_module=torch,
                    wandb_module=wandb,
                    wandb_project=model_spec["wandb_project"],
                    wandb_mode=model_spec["wandb_mode"],
                    tier_name=tier_name,
                )

                regression_df = combined_metrics[
                    combined_metrics["metric_family"] == "regression"
                ].drop(columns=["metric_family"])
                long_direction_df = combined_metrics[
                    combined_metrics["metric_family"] == "long_direction"
                ].drop(columns=["metric_family"])

                all_models[str(ticker)] = pecnet
                prediction_frames.append(joined_df)
                regression_frames.append(regression_df)
                long_direction_frames.append(long_direction_df)

                mlflow.log_table(joined_df, f"pecnet/{tier_name}/predictions/{ticker}.json")
                mlflow.log_table(
                    regression_df,
                    f"pecnet/{tier_name}/evaluation/{ticker}_regression.json",
                )
                mlflow.log_table(
                    long_direction_df,
                    f"pecnet/{tier_name}/evaluation/{ticker}_long_direction.json",
                )
                log_forecast_plots(
                    train_df=ticker_train_df,
                    joined_df=joined_df,
                    levels=None,
                    artifact_prefix=f"pecnet/{tier_name}/plots/{_safe_name(str(ticker))}",
                )

                figure, comparison_df = pecnet_prediction_figure(
                    predictions=joined_df["PECNet"].to_numpy(),
                    actual=joined_df["y"].to_numpy(),
                    dates=joined_df["ds"],
                    model_name=f"PECNet {ticker}",
                )
                mlflow.log_figure(
                    figure,
                    artifact_file=(
                        f"pecnet/{tier_name}/plots/"
                        f"{_safe_name(str(ticker))}/comparison.png"
                    ),
                )
                plt.close(figure)
                mlflow.log_table(
                    comparison_df,
                    f"pecnet/{tier_name}/predictions/"
                    f"{_safe_name(str(ticker))}_comparison.json",
                )

                model_dir = Path(tempfile.mkdtemp(prefix="pecnet_"))
                model_path = model_dir / f"pecnet_{_safe_name(str(ticker))}.pt"
                try:
                    torch.save(pecnet, model_path)
                    mlflow.log_artifact(
                        str(model_path),
                        artifact_path=(
                            f"pecnet/{tier_name}/models/{_safe_name(str(ticker))}"
                        ),
                    )
                except Exception:
                    LOGGER.warning("Could not serialize PECNet model.", exc_info=True)

        predictions = (
            pd.concat(prediction_frames, ignore_index=True)
            if prediction_frames
            else pd.DataFrame()
        )
        regression_metrics = (
            pd.concat(regression_frames, ignore_index=True)
            if regression_frames
            else pd.DataFrame()
        )
        long_direction_metrics = (
            pd.concat(long_direction_frames, ignore_index=True)
            if long_direction_frames
            else pd.DataFrame()
        )
        mlflow.log_table(predictions, f"pecnet/{tier_name}/predictions/all_predictions.json")
        mlflow.log_table(
            regression_metrics,
            f"pecnet/{tier_name}/evaluation/all_regression_metrics.json",
        )
        mlflow.log_table(
            long_direction_metrics,
            f"pecnet/{tier_name}/evaluation/all_long_direction_metrics.json",
        )

    return {
        "models": all_models,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "predictions": predictions,
        "regression_metrics": regression_metrics,
        "long_direction_metrics": long_direction_metrics,
    }
