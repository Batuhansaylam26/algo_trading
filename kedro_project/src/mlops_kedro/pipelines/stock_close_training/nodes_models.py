from __future__ import annotations

from typing import Any

import pandas as pd

from .ml.mlforecast import (
    MLForecastService,
)
from .ml.pecnet import PecnetService
from .ml.root_performance import RootModelPerformanceEvaluator
from .ml.statsforecast import StatsForecastService
from .node_utils import (
    _apply_training_environment,
    _as_bool,
    _as_int,
    _feature_columns_for_tier,
    _log_step,
    _merge_tier_overrides,
)


class StockCloseModelNodes:
    def __init__(
        self,
        mlforecast: MLForecastService | None = None,
        statsforecast: StatsForecastService | None = None,
        pecnet: PecnetService | None = None,
        root_performance: RootModelPerformanceEvaluator | None = None,
    ) -> None:
        self.mlforecast = mlforecast or MLForecastService()
        self.statsforecast = statsforecast or StatsForecastService()
        self.pecnet = pecnet or PecnetService()
        self.root_performance = root_performance or RootModelPerformanceEvaluator()

    def build_model_spec(
        self,
        mlforecast_params: dict[str, Any] | None,
        mlflow_params: dict[str, Any] | None,
        runtime_params: dict[str, Any] | None,
        *,
        tier_name: str = "tier1",
    ) -> dict[str, Any]:
        mlforecast_params = mlforecast_params or {}
        _apply_training_environment(
            mlflow_params=mlflow_params,
            runtime_params=runtime_params,
            mlforecast_params=mlforecast_params,
        )
        model_spec = self.mlforecast.build_spec(
            freq=mlforecast_params.get("freq", "B"),
            validation_horizon=_as_int(
                mlforecast_params.get("validation_horizon"),
                1,
            ),
            test_horizon=_as_int(mlforecast_params.get("test_horizon"), 5),
            n_windows=_as_int(mlforecast_params.get("n_windows"), 3),
            n_trials=_as_int(mlforecast_params.get("n_trials"), 20),
            verbose=_as_bool(mlforecast_params.get("verbose"), True),
            models=mlforecast_params.get("models"),
            tier_name=tier_name,
        )
        model_spec["_mlflow"] = mlflow_params or {}
        model_spec["_runtime"] = runtime_params or {}
        model_spec["_mlforecast"] = mlforecast_params
        _log_step(f"build_{tier_name}_mlforecast_model_spec", **model_spec)
        return model_spec

    def train_models(
        self,
        stock_close_train_test_split: dict[str, pd.DataFrame],
        stock_close_model_spec: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        _apply_training_environment(
            mlflow_params=stock_close_model_spec.get("_mlflow"),
            runtime_params=stock_close_model_spec.get("_runtime"),
            mlforecast_params=stock_close_model_spec.get("_mlforecast"),
        )
        result = self.mlforecast.train_from_split(
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
        self,
        statsforecast_params: dict[str, Any] | None,
        mlforecast_params: dict[str, Any] | None,
        mlflow_params: dict[str, Any] | None,
        runtime_params: dict[str, Any] | None,
        *,
        tier_name: str,
    ) -> dict[str, Any]:
        statsforecast_params = statsforecast_params or {}
        mlforecast_params = mlforecast_params or {}
        _apply_training_environment(
            mlflow_params=mlflow_params,
            runtime_params=runtime_params,
            mlforecast_params=mlforecast_params,
        )
        model_spec = self.statsforecast.build_spec(
            freq=statsforecast_params.get("freq", mlforecast_params.get("freq", "B")),
            seasonal_length=_as_int(statsforecast_params.get("seasonal_length"), 5),
            validation_horizon=_as_int(
                mlforecast_params.get("validation_horizon"),
                1,
            ),
            test_horizon=_as_int(mlforecast_params.get("test_horizon"), 5),
            conformal_n_windows=_as_int(
                statsforecast_params.get("conformal_n_windows"),
                3,
            ),
            level=statsforecast_params.get("level", [80, 95]),
            models=statsforecast_params.get("models"),
            verbose=_as_bool(statsforecast_params.get("verbose"), True),
            tier_name=tier_name,
        )
        model_spec["enabled"] = _as_bool(statsforecast_params.get("enabled"), True)
        model_spec["_mlflow"] = mlflow_params or {}
        model_spec["_runtime"] = runtime_params or {}
        model_spec["_mlforecast"] = mlforecast_params
        _log_step(f"build_{tier_name}_statsforecast_model_spec", **model_spec)
        return model_spec

    def train_statsforecast_models(
        self,
        stock_close_train_test_split: dict[str, pd.DataFrame],
        stock_close_statsforecast_model_spec: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        _apply_training_environment(
            mlflow_params=stock_close_statsforecast_model_spec.get("_mlflow"),
            runtime_params=stock_close_statsforecast_model_spec.get("_runtime"),
            mlforecast_params=stock_close_statsforecast_model_spec.get("_mlforecast"),
        )
        if not stock_close_statsforecast_model_spec.get("enabled", True):
            empty = pd.DataFrame()
            metadata = {"skipped": True, "reason": "statsforecast disabled"}
            _log_step("train_statsforecast_models", **metadata)
            return empty, empty, empty, metadata

        result = self.statsforecast.train_from_split(
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
        self,
        pecnet_params: dict[str, Any] | None,
        mlforecast_params: dict[str, Any] | None,
        mlflow_params: dict[str, Any] | None,
        runtime_params: dict[str, Any] | None,
        columns_params: dict[str, Any] | None,
        *,
        tier_name: str,
    ) -> dict[str, Any]:
        pecnet_params = pecnet_params or {}
        mlforecast_params = mlforecast_params or {}
        _apply_training_environment(
            mlflow_params=mlflow_params,
            runtime_params=runtime_params,
            mlforecast_params=mlforecast_params,
        )
        feature_columns_by_tier = pecnet_params.get("feature_columns_by_tier") or {}
        feature_columns = feature_columns_by_tier.get(
            tier_name,
            _feature_columns_for_tier(columns_params, tier_name),
        )
        model_spec = self.pecnet.build_spec(
            enabled=_as_bool(pecnet_params.get("enabled"), True),
            test_horizon=_as_int(mlforecast_params.get("test_horizon"), 5),
            feature_columns=feature_columns,
            preprocess_params=_merge_tier_overrides(
                pecnet_params.get("preprocess_params"),
                pecnet_params.get("preprocess_params_by_tier"),
                tier_name,
            ),
            hyperparams=_merge_tier_overrides(
                pecnet_params.get("hyperparams"),
                pecnet_params.get("hyperparams_by_tier"),
                tier_name,
            ),
            selection_params=pecnet_params.get("selection_params", {}),
            tier_name=tier_name,
        )
        model_spec["_mlflow"] = mlflow_params or {}
        model_spec["_runtime"] = runtime_params or {}
        model_spec["_mlforecast"] = mlforecast_params
        _log_step(f"build_{tier_name}_pecnet_model_spec", **model_spec)
        return model_spec

    def train_pecnet_models(
        self,
        stock_close_pecnet_train_test_split: dict[str, pd.DataFrame],
        stock_close_pecnet_model_spec: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        _apply_training_environment(
            mlflow_params=stock_close_pecnet_model_spec.get("_mlflow"),
            runtime_params=stock_close_pecnet_model_spec.get("_runtime"),
            mlforecast_params=stock_close_pecnet_model_spec.get("_mlforecast"),
        )
        result = self.pecnet.train_from_split(
            stock_close_pecnet_train_test_split,
            model_spec=stock_close_pecnet_model_spec,
        )
        regression_metrics = result["regression_metrics"]
        long_direction_metrics = result["long_direction_metrics"]
        predictions = result["predictions"]
        feature_selection = result.get("feature_selection", pd.DataFrame())
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
            "feature_selection_rows": len(feature_selection),
            "preprocessed_store_rows": result.get("preprocessed_store_rows", 0),
            "models": list(result.get("models", {}).keys()),
        }
        _log_step("train_pecnet_models", **metadata)
        return (
            regression_metrics,
            long_direction_metrics,
            predictions,
            feature_selection,
            metadata,
        )

    def summarize_training(self, *metadata_items: dict[str, Any]) -> dict[str, Any]:
        summary = {"sections": list(metadata_items)}
        _log_step("summarize_training", sections=len(metadata_items))
        return summary

    def evaluate_root_model_performance(
        self,
        mlflow_params: dict[str, Any] | None = None,
        **model_outputs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        result = self.root_performance.evaluate(
            mlflow_params=mlflow_params,
            **model_outputs,
        )
        _log_step("evaluate_root_model_performance", **result[-1])
        return result

    def summarize_machine_learning(
        self,
        *metadata_items: dict[str, Any],
    ) -> dict[str, Any]:
        summary = {"sections": list(metadata_items)}
        _log_step("summarize_machine_learning", sections=len(metadata_items))
        return summary


stock_close_model_nodes = StockCloseModelNodes()
