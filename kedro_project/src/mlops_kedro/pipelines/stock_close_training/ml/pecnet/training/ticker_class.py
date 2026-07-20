from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from ...common import _regression_metrics
from ...metrics import long_only_directional_metrics
from ..selection import _build_pecnet_variables
from ..tracking import (
    _log_pecnet_epoch_metrics_to_mlflow,
    _mlflow_live_pecnet_epoch_logging,
)


LOGGER = logging.getLogger(__name__)


class PecnetTickerTrainer:

    @staticmethod
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

        with _mlflow_live_pecnet_epoch_logging(
            basic_nn_cls=basic_nn_cls,
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
        predictions_array = PecnetTickerTrainer._as_prediction_array(predictions, torch_module)

        evaluation_predictions = PecnetTickerTrainer._drop_tomorrow_prediction(predictions_array)
        prediction_dates = (
            ticker_test_df[["unique_id", "ds", "y"]]
            .sort_values(["unique_id", "ds"])
            .tail(len(evaluation_predictions))
            .reset_index(drop=True)
        )

        joined_df = prediction_dates.copy()
        joined_df["PECNet"] = (
            evaluation_predictions[-len(prediction_dates) :]
            if len(prediction_dates)
            else np.asarray([], dtype=float)
        )
        regression_df = PecnetTickerTrainer._ticker_metric_frame(
            _regression_metrics(joined_df),
            unique_id=ticker,
        )
        long_direction_df = PecnetTickerTrainer._ticker_metric_frame(
            long_only_directional_metrics(joined_df, ticker_train_df),
            unique_id=ticker,
        )
        epoch_metrics = _log_pecnet_epoch_metrics_to_mlflow(
            pecnet=pecnet,
            ticker=ticker,
            tier_name=tier_name,
        )
        LOGGER.info(
            "Logged PECNet epoch metrics to MLflow | tier=%s ticker=%s rows=%s",
            tier_name,
            ticker,
            len(epoch_metrics),
        )

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

    @staticmethod
    def _ticker_metric_frame(metrics: pd.DataFrame, *, unique_id: str) -> pd.DataFrame:
        if metrics.empty:
            return metrics

        scoped_metrics = metrics.copy()
        scoped_metrics.insert(0, "unique_id", unique_id)
        return scoped_metrics

    @staticmethod
    def _drop_tomorrow_prediction(predictions: np.ndarray) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=float).reshape(-1)
        if len(predictions) <= 1:
            return predictions
        return predictions[:-1]

    @staticmethod
    def _as_prediction_array(predictions: Any, torch_module) -> np.ndarray:
        if torch_module.is_tensor(predictions):
            return predictions.detach().cpu().numpy().reshape(-1)
        return np.asarray(predictions, dtype=float).reshape(-1)
