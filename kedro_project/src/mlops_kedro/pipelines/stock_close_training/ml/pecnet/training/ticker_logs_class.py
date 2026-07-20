from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import mlflow
import pandas as pd

from ..runtime import _safe_name
from ..selection import _log_feature_selection_heatmap


LOGGER = logging.getLogger(__name__)


class PecnetTickerLogWriter:

    @staticmethod
    def log_ticker_selection_metrics(
        *,
        selection_df: pd.DataFrame,
        tier_name: str,
        ticker_safe: str,
    ) -> None:
        mlflow.log_table(
            selection_df,
            f"pecnet/{tier_name}/feature_selection/{ticker_safe}.json",
        )
        _log_feature_selection_heatmap(
            selection_df,
            artifact_file=(
                f"pecnet/{tier_name}/feature_selection/"
                f"{ticker_safe}_correlation_heatmap.png"
            ),
            title=f"PECNet {tier_name} {ticker_safe} selected feature correlations",
            index_column="selection_order",
        )
        for _, row in selection_df.iterrows():
            correlation = row.get("correlation")
            if pd.isna(correlation):
                continue
            order = int(row["selection_order"])
            mlflow.log_metric(
                f"pecnet.{ticker_safe}.selection.{order}.correlation",
                float(correlation),
            )
            mlflow.log_metric(
                f"pecnet.{ticker_safe}.selection.{order}.abs_correlation",
                float(row["abs_correlation"]),
            )

    @staticmethod
    def log_ticker_model_metrics(
        *,
        regression_df: pd.DataFrame,
        long_direction_df: pd.DataFrame,
        ticker_safe: str,
    ) -> None:
        for _, row in regression_df.iterrows():
            model_safe = _safe_name(str(row["model"]))
            for metric_name in ["mae", "rmse", "mape", "r2"]:
                if pd.isna(row.get(metric_name)):
                    continue
                mlflow.log_metric(
                    f"pecnet.{ticker_safe}.{model_safe}.test.{metric_name}",
                    float(row[metric_name]),
                )

        for _, row in long_direction_df.iterrows():
            if row[["long_accuracy", "long_precision", "long_recall"]].isna().any():
                continue
            model_safe = _safe_name(str(row["model"]))
            for metric_name in [
                "long_accuracy",
                "long_precision",
                "long_recall",
                "long_signal_rate",
            ]:
                if pd.isna(row.get(metric_name)):
                    continue
                mlflow.log_metric(
                    f"pecnet.{ticker_safe}.{model_safe}.{metric_name}",
                    float(row[metric_name]),
                )

    @staticmethod
    def log_ticker_model_artifact(
        *,
        pecnet,
        torch_module,
        tier_name: str,
        ticker_safe: str,
    ) -> None:
        model_dir = Path(tempfile.mkdtemp(prefix="pecnet_"))
        model_path = model_dir / f"pecnet_{ticker_safe}.pt"
        try:
            torch_module.save(pecnet, model_path)
            mlflow.log_artifact(
                str(model_path),
                artifact_path=f"pecnet/{tier_name}/models/{ticker_safe}",
            )
        except Exception:
            LOGGER.warning("Could not serialize PECNet model.", exc_info=True)
