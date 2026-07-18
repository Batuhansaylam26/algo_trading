from __future__ import annotations

import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

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

def _define_pecnet_wandb_metrics(run) -> None:
    run.define_metric("pecnet/step")
    run.define_metric("pecnet/epoch", step_metric="pecnet/step")
    run.define_metric("pecnet/train_loss", step_metric="pecnet/step")
    run.define_metric("pecnet/tier1/train_loss", step_metric="pecnet/step")
    run.define_metric("pecnet/tier2/train_loss", step_metric="pecnet/step")
    run.define_metric("pecnet/tier3/train_loss", step_metric="pecnet/step")
    run.define_metric("pecnet/tier4/train_loss", step_metric="pecnet/step")

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
