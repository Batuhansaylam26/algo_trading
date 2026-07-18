from __future__ import annotations

from contextlib import contextmanager
from typing import Any

import mlflow
import pandas as pd

from ..runtime import _safe_name


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


@contextmanager
def _mlflow_live_pecnet_epoch_logging(
    *,
    basic_nn_cls,
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

        class MlflowLossLog(list):
            def append(loss_log, value):
                super().append(value)
                epoch = len(loss_log)
                global_step["value"] += 1
                train_loss = float(value)
                mlflow.log_metric(
                    "pecnet.train_loss",
                    train_loss,
                    step=global_step["value"],
                )
                mlflow.log_metric(
                    f"pecnet.{tier_safe}.train_loss",
                    train_loss,
                    step=global_step["value"],
                )
                mlflow.log_metric(
                    f"pecnet.{tier_safe}.{ticker_safe}.train_loss",
                    train_loss,
                    step=global_step["value"],
                )
                mlflow.log_metric(
                    f"pecnet.{tier_safe}.{ticker_safe}.{network_safe}.train_loss",
                    train_loss,
                    step=global_step["value"],
                )
                mlflow.log_metric(
                    f"pecnet.{tier_safe}.{ticker_safe}.{network_safe}.epoch",
                    float(epoch),
                    step=global_step["value"],
                )

        live_loss_log = MlflowLossLog(original_loss_log)
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


def _log_pecnet_epoch_metrics_to_mlflow(
    *,
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

    ticker_safe = _safe_name(ticker)
    mlflow.log_table(
        epoch_metrics,
        f"pecnet/{tier_name}/epoch_metrics/{ticker_safe}.json",
    )
    for network_name, network_metrics in epoch_metrics.groupby("network", sort=False):
        network_safe = _safe_name(str(network_name))
        mlflow.log_metric(
            f"pecnet.{ticker_safe}.{network_safe}.final_train_loss",
            float(network_metrics.iloc[-1]["train_loss"]),
        )
        mlflow.log_metric(
            f"pecnet.{ticker_safe}.{network_safe}.min_train_loss",
            float(network_metrics["train_loss"].min()),
        )

    return epoch_metrics
