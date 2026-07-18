from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ...common import _regression_metrics
from ...metrics import long_only_directional_metrics
from ..runtime import _safe_name
from ..selection import _build_pecnet_variables
from ..tracking import (
    _define_pecnet_wandb_metrics,
    _log_pecnet_epoch_metrics_to_wandb,
    _log_pecnet_model_to_wandb,
    _save_pecnet_model_file,
    _wandb_live_pecnet_epoch_logging,
)


LOGGER = logging.getLogger(__name__)


def _train_one_ticker(
    *,
    ticker_data: dict[str, Any],
    ticker_train_df: pd.DataFrame,
    ticker_test_df: pd.DataFrame,
    hyperparams: dict[str, Any],
    utility,
    pecnet_builder_cls,
    basic_nn_cls,
    feature_selector_cls,
    torch_module,
    wandb_module,
    wandb_project: str,
    wandb_mode: str,
    tier_name: str,
    selection_params: dict[str, Any],
) -> tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
            "selection_params": selection_params,
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
            builder, selected_X_test, selection_df = _build_pecnet_variables(
                builder=builder,
                ticker_data=ticker_data,
                tier_name=tier_name,
                feature_selector_cls=feature_selector_cls,
                selection_params=selection_params,
            )
            pecnet = builder.add_error_network().add_final_network().build()
        predictions = pecnet.predict(
            *selected_X_test,
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
        if not selection_df.empty:
            run.log(
                {
                    "pecnet_feature_selection": wandb_module.Table(
                        dataframe=selection_df
                    )
                }
            )
            for _, row in selection_df.iterrows():
                order = int(row["selection_order"])
                feature_safe = _safe_name(str(row["feature_name"]))
                correlation = row.get("correlation")
                if pd.notna(correlation):
                    run.summary[
                        f"pecnet/selection/{order}/{feature_safe}/correlation"
                    ] = float(correlation)

        for _, row in regression_df.iterrows():
            model_safe = _safe_name(str(row["model"]))
            metric_payload = {}
            for metric_name in ["mae", "rmse", "mape", "r2"]:
                if pd.isna(row.get(metric_name)):
                    continue
                metric_value = float(row[metric_name])
                metric_payload[f"{row['model']}/{metric_name}"] = metric_value
                metric_payload[
                    f"pecnet/eval/{model_safe}/{metric_name}"
                ] = metric_value
                run.summary[f"pecnet/eval/{model_safe}/{metric_name}"] = metric_value
            if metric_payload:
                run.log(metric_payload)
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

    return (
        pecnet,
        joined_df,
        pd.concat(
            [
                regression_df.assign(metric_family="regression"),
                long_direction_df.assign(metric_family="long_direction"),
            ],
            ignore_index=True,
        ),
        selection_df,
    )
