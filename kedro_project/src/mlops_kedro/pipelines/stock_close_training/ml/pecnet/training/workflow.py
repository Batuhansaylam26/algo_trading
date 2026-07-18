from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd

from ...common import configure_mlflow_tracking, tier_experiment_name
from ..runtime import (
    _configure_torch_threads,
    _load_pecnet_runtime,
    _safe_name,
)
from .performance import PecnetPerformanceMeasurement
from .ticker import _train_one_ticker


@dataclass(slots=True)
class PecnetTrainingWorkflow:
    def train_from_split(
        self,
        train_test_split: dict[str, pd.DataFrame],
        *,
        model_spec: dict[str, Any],
    ) -> dict[str, Any]:
        if not model_spec.get("enabled", True) or train_test_split["full"].empty:
            return self._empty_result(train_test_split)

        runtime = self._load_runtime(model_spec)
        tier_name = model_spec.get("tier_name", "tier1")
        configure_mlflow_tracking(experiment_name=tier_experiment_name(tier_name))
        workflow_state = self._workflow_state(train_test_split, model_spec)
        performance = PecnetPerformanceMeasurement(tier_name=tier_name)

        with mlflow.start_run(run_name=f"stock-close-{tier_name}-pecnet", nested=True):
            performance.log_parent_metadata(
                model_spec=model_spec,
                train_df=workflow_state["train_df"],
                test_df=workflow_state["test_df"],
                feature_columns=workflow_state["feature_columns"],
                preprocess_params=workflow_state["preprocess_params"],
                hyperparams=workflow_state["hyperparams"],
                selection_params=workflow_state["selection_params"],
                torch_thread_config=runtime["torch_thread_config"],
            )
            outputs = self._train_tickers(
                workflow_state=workflow_state,
                model_spec=model_spec,
                runtime=runtime,
                performance=performance,
            )
            performance.log_parent_outputs(
                predictions=outputs["predictions"],
                regression_metrics=outputs["regression_metrics"],
                long_direction_metrics=outputs["long_direction_metrics"],
                feature_selection=outputs["feature_selection"],
            )

        return {
            "models": outputs["models"],
            "train_rows": len(workflow_state["train_df"]),
            "test_rows": len(workflow_state["test_df"]),
            "predictions": outputs["predictions"],
            "regression_metrics": outputs["regression_metrics"],
            "long_direction_metrics": outputs["long_direction_metrics"],
            "feature_selection": outputs["feature_selection"],
            "preprocessed_store_rows": outputs["preprocessed_store_rows"],
        }

    @staticmethod
    def _empty_result(train_test_split: dict[str, pd.DataFrame]) -> dict[str, Any]:
        empty = pd.DataFrame()
        return {
            "models": {},
            "train_rows": len(train_test_split["train"]),
            "test_rows": len(train_test_split["test"]),
            "predictions": empty,
            "regression_metrics": empty,
            "long_direction_metrics": empty,
        }

    @staticmethod
    def _load_runtime(model_spec: dict[str, Any]) -> dict[str, Any]:
        Utility, PecnetBuilder, DataPreprocessor, BasicNN, FeatureSelector, torch = (
            _load_pecnet_runtime()
        )
        torch_thread_config = _configure_torch_threads(torch)
        return {
            "utility": Utility,
            "pecnet_builder_cls": PecnetBuilder,
            "data_preprocessor_cls": DataPreprocessor,
            "basic_nn_cls": BasicNN,
            "feature_selector_cls": FeatureSelector,
            "torch": torch,
            "torch_thread_config": torch_thread_config,
        }

    @staticmethod
    def _workflow_state(
        train_test_split: dict[str, pd.DataFrame],
        model_spec: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "full_df": train_test_split["full"],
            "train_df": train_test_split["train"],
            "test_df": train_test_split["test"],
            "feature_columns": model_spec["feature_columns"],
            "preprocess_params": model_spec["preprocess_params"],
            "hyperparams": model_spec["hyperparams"],
            "selection_params": model_spec.get("selection_params", {}),
            "tier_name": model_spec.get("tier_name", "tier1"),
        }

    def _train_tickers(
        self,
        *,
        workflow_state: dict[str, Any],
        model_spec: dict[str, Any],
        runtime: dict[str, Any],
        performance: PecnetPerformanceMeasurement,
    ) -> dict[str, Any]:
        outputs = self._empty_ticker_outputs()
        for ticker, ticker_df in workflow_state["full_df"].groupby(
            "unique_id",
            observed=True,
        ):
            ticker_train_df = workflow_state["train_df"][
                workflow_state["train_df"]["unique_id"] == ticker
            ].copy()
            ticker_test_df = workflow_state["test_df"][
                workflow_state["test_df"]["unique_id"] == ticker
            ].copy()
            if ticker_train_df.empty or ticker_test_df.empty:
                continue

            with mlflow.start_run(
                run_name=(
                    f"pecnet-{workflow_state['tier_name']}-{_safe_name(str(ticker))}"
                ),
                nested=True,
            ):
                self._train_one_ticker(
                    ticker=str(ticker),
                    ticker_df=ticker_df,
                    ticker_train_df=ticker_train_df,
                    ticker_test_df=ticker_test_df,
                    workflow_state=workflow_state,
                    model_spec=model_spec,
                    runtime=runtime,
                    performance=performance,
                    outputs=outputs,
                )

        outputs["predictions"] = _concat_or_empty(outputs.pop("prediction_frames"))
        outputs["regression_metrics"] = _concat_or_empty(
            outputs.pop("regression_frames")
        )
        outputs["long_direction_metrics"] = _concat_or_empty(
            outputs.pop("long_direction_frames")
        )
        outputs["feature_selection"] = _concat_or_empty(
            outputs.pop("selection_frames")
        )
        return outputs

    @staticmethod
    def _empty_ticker_outputs() -> dict[str, Any]:
        return {
            "models": {},
            "prediction_frames": [],
            "regression_frames": [],
            "long_direction_frames": [],
            "selection_frames": [],
            "preprocessed_store_rows": 0,
        }

    @staticmethod
    def _train_one_ticker(
        *,
        ticker: str,
        ticker_df: pd.DataFrame,
        ticker_train_df: pd.DataFrame,
        ticker_test_df: pd.DataFrame,
        workflow_state: dict[str, Any],
        model_spec: dict[str, Any],
        runtime: dict[str, Any],
        performance: PecnetPerformanceMeasurement,
        outputs: dict[str, Any],
    ) -> None:
        ticker_data, store_rows = performance.prepare_ticker_inputs(
            ticker_df=ticker_df,
            ticker_train_df=ticker_train_df,
            ticker_test_df=ticker_test_df,
            ticker=ticker,
            feature_columns=workflow_state["feature_columns"],
            preprocess_params=workflow_state["preprocess_params"],
            test_horizon=model_spec["test_horizon"],
            data_preprocessor_cls=runtime["data_preprocessor_cls"],
        )
        outputs["preprocessed_store_rows"] += store_rows
        pecnet, joined_df, combined_metrics, selection_df = _train_one_ticker(
            ticker_data=ticker_data,
            ticker_train_df=ticker_train_df,
            ticker_test_df=ticker_test_df,
            hyperparams=workflow_state["hyperparams"],
            utility=runtime["utility"],
            pecnet_builder_cls=runtime["pecnet_builder_cls"],
            basic_nn_cls=runtime["basic_nn_cls"],
            feature_selector_cls=runtime["feature_selector_cls"],
            torch_module=runtime["torch"],
            tier_name=workflow_state["tier_name"],
            selection_params=workflow_state["selection_params"],
        )
        regression_df, long_direction_df = performance.split_combined_metrics(
            combined_metrics
        )

        outputs["models"][ticker] = pecnet
        outputs["prediction_frames"].append(joined_df)
        outputs["regression_frames"].append(regression_df)
        outputs["long_direction_frames"].append(long_direction_df)
        if not selection_df.empty:
            outputs["selection_frames"].append(selection_df)

        performance.log_ticker_results(
            ticker=ticker,
            ticker_train_df=ticker_train_df,
            joined_df=joined_df,
            regression_df=regression_df,
            long_direction_df=long_direction_df,
            selection_df=selection_df,
            pecnet=pecnet,
            torch_module=runtime["torch"],
        )


def train_pecnet_models_from_split(
    train_test_split: dict[str, pd.DataFrame],
    *,
    model_spec: dict[str, Any],
) -> dict[str, Any]:
    return PecnetTrainingWorkflow().train_from_split(
        train_test_split,
        model_spec=model_spec,
    )


def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
