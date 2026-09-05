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
        data_preprocessor_cls,
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
            hidden_units_strategy=hyperparams.get("hidden_units_strategy"),
            optimizer_name=hyperparams.get("optimizer_name", "adam"),
            momentum=hyperparams.get("momentum", 0.0),
            activation=hyperparams.get("activation", "gelu"),
            use_layer_norm=hyperparams.get("use_layer_norm", True),
            epoch_size_by_network_name=hyperparams.get("epoch_size_by_network_name"),
        )
        paper_pec_wnn = PecnetTickerTrainer._is_paper_pec_wnn(
            tier_name=tier_name,
            selection_params=selection_params,
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
            if paper_pec_wnn:
                pecnet = builder.add_error_network().build()
            else:
                pecnet = builder.add_error_network().add_final_network().build()

        if paper_pec_wnn:
            predictions = PecnetTickerTrainer._predict_paper_pec_wnn(
                pecnet=pecnet,
                selected_X_test=selected_X_test,
                test_target=ticker_data["y_test"],
                data_preprocessor_cls=data_preprocessor_cls,
            )
        else:
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
    def _is_paper_pec_wnn(
        *,
        tier_name: str,
        selection_params: dict[str, Any],
    ) -> bool:
        strategy_by_tier = selection_params.get("strategy_by_tier", {})
        strategy = strategy_by_tier.get(
            tier_name,
            selection_params.get("strategy", "all_features"),
        )
        return strategy == "paper_pec_wnn"

    @staticmethod
    def _predict_paper_pec_wnn(
        *,
        pecnet,
        selected_X_test: list[np.ndarray],
        test_target: np.ndarray,
        data_preprocessor_cls,
    ) -> np.ndarray:
        data_preprocessor = data_preprocessor_cls()
        data_preprocessor.switch_mode("test")
        pecnet.mode = "test"

        for variable_network in pecnet.variable_networks:
            variable_network.switch_mode("test")
        pecnet.error_network.switch_mode("test")

        for index, variable_network in enumerate(pecnet.variable_networks):
            if index == 0:
                variable_network.init_network(selected_X_test[index], test_target)
            else:
                y_target = pecnet.get_target_values_for_current_variable_network(index)
                pre_comp = pecnet.get_comp_preds_for_current_variable_network(index)
                variable_network.init_network(selected_X_test[index], y_target, pre_comp)

        pecnet.error_network.init_network(
            pecnet.get_shifted_compensated_errors(),
            pecnet.get_last_compensated_predictions(),
        )
        compensated_predictions = pecnet.error_network.get_compensated_error_predictions()
        return PecnetTickerTrainer._denormalize_paper_predictions(
            compensated_predictions,
            data_preprocessor,
        )

    @staticmethod
    def _denormalize_paper_predictions(
        predictions: np.ndarray,
        data_preprocessor,
    ) -> np.ndarray:
        predictions = np.asarray(predictions, dtype=float).reshape(-1, 1)
        denormalization_term = np.asarray(
            data_preprocessor.get_final_denormalization_term(),
            dtype=float,
        ).reshape(-1, 1)
        if len(denormalization_term) > len(predictions):
            denormalization_term = denormalization_term[-len(predictions):]
        if len(denormalization_term) and len(denormalization_term) == len(predictions):
            denormalization_term[-1] = (
                data_preprocessor.generate_final_denormalization_term_for_last_pred_element()
            )

        denormalized = predictions + denormalization_term
        if data_preprocessor.target_normalizer:
            denormalized = data_preprocessor.target_normalizer.inverse_transform(
                denormalized
            )
        if data_preprocessor.target_scaler is not None:
            return data_preprocessor.target_scaler.unscale1D(denormalized)
        return denormalized

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
