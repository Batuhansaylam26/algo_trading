from __future__ import annotations

import logging
import os
from contextlib import nullcontext
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd
import polars as pl

from ...common import configure_mlflow_tracking, tier_experiment_name
from ...runtime import cpu_count_from_env
from ..runtime import (
    _configure_torch_threads,
    _load_pecnet_runtime,
    _safe_name,
)
from .performance import PecnetPerformanceMeasurement
from .ticker import _train_one_ticker


LOGGER = logging.getLogger(__name__)


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

        outputs = self._train_tickers(
            workflow_state=workflow_state,
            model_spec=model_spec,
            runtime=runtime,
            performance=performance,
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
        ticker_jobs = PecnetTrainingWorkflow._ticker_jobs(
            workflow_state=workflow_state,
            model_spec=model_spec,
        )
        n_jobs = PecnetTrainingWorkflow._pecnet_worker_count(model_spec, len(ticker_jobs))
        worker_threads = PecnetTrainingWorkflow._pecnet_worker_torch_threads(model_spec, n_jobs)

        if n_jobs == 1:
            for ticker_job in ticker_jobs:
                result = PecnetTrainingWorkflow._run_ticker_job(
                    ticker_job=ticker_job,
                    runtime=runtime,
                    performance=performance,
                    start_child_run=True,
                )
                PecnetTrainingWorkflow._append_ticker_result(outputs, result)
        else:
            LOGGER.info(
                "Training PECNet tickers in parallel | tier=%s tickers=%s "
                "workers=%s torch_threads_per_worker=%s",
                workflow_state["tier_name"],
                len(ticker_jobs),
                n_jobs,
                worker_threads,
            )
            for ticker_job in ticker_jobs:
                ticker_job["torch_threads"] = worker_threads

            with ProcessPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(PecnetTrainingWorkflow._run_ticker_job_in_child_process, ticker_job): (
                        ticker_job["ticker"]
                    )
                    for ticker_job in ticker_jobs
                }
                for future in as_completed(futures):
                    ticker = futures[future]
                    try:
                        result = future.result()
                    except Exception:
                        LOGGER.exception("PECNet ticker training failed: %s", ticker)
                        raise
                    PecnetTrainingWorkflow._append_ticker_result(outputs, result)
            PecnetTrainingWorkflow._publish_deferred_preprocessed_inputs(
                outputs=outputs,
                performance=performance,
            )

        outputs["predictions"] = PecnetTrainingWorkflow._concat_or_empty(outputs.pop("prediction_frames"))
        outputs["regression_metrics"] = PecnetTrainingWorkflow._concat_or_empty(
            outputs.pop("regression_frames")
        )
        outputs["long_direction_metrics"] = PecnetTrainingWorkflow._concat_or_empty(
            outputs.pop("long_direction_frames")
        )
        outputs["feature_selection"] = PecnetTrainingWorkflow._concat_or_empty(
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
            "preprocessed_frames": [],
            "preprocessed_run_ids": {},
            "preprocessed_store_rows": 0,
        }























    @staticmethod
    def train_pecnet_models_from_split(
        train_test_split: dict[str, pd.DataFrame],
        *,
        model_spec: dict[str, Any],
    ) -> dict[str, Any]:
        return PecnetTrainingWorkflow().train_from_split(
            train_test_split,
            model_spec=model_spec,
        )

    @staticmethod
    def _concat_or_empty(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _ticker_jobs(
        *,
        workflow_state: dict[str, Any],
        model_spec: dict[str, Any],
    ) -> list[dict[str, Any]]:
        jobs = []
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

            jobs.append(
                {
                    "ticker": str(ticker),
                    "ticker_df": ticker_df.copy(),
                    "ticker_train_df": ticker_train_df,
                    "ticker_test_df": ticker_test_df,
                    "feature_columns": workflow_state["feature_columns"],
                    "preprocess_params": workflow_state["preprocess_params"],
                    "hyperparams": workflow_state["hyperparams"],
                    "selection_params": workflow_state["selection_params"],
                    "tier_name": workflow_state["tier_name"],
                    "test_horizon": model_spec["test_horizon"],
                    "mlflow_params": model_spec.get("_mlflow", {}),
                }
            )
        return jobs

    @staticmethod
    def _run_ticker_job(
        *,
        ticker_job: dict[str, Any],
        runtime: dict[str, Any],
        performance: PecnetPerformanceMeasurement,
        start_child_run: bool,
        publish_preprocessed_to_store: bool = True,
    ) -> dict[str, Any]:
        if start_child_run:
            run_context = mlflow.start_run(
                run_name=f"pecnet-{ticker_job['tier_name']}-{_safe_name(ticker_job['ticker'])}",
            )
        else:
            run_context = nullcontext()

        with run_context:
            performance.log_ticker_metadata(
                ticker=ticker_job["ticker"],
                feature_columns=ticker_job["feature_columns"],
                preprocess_params=ticker_job["preprocess_params"],
                hyperparams=ticker_job["hyperparams"],
                selection_params=ticker_job["selection_params"],
                test_horizon=ticker_job["test_horizon"],
                torch_thread_config=runtime["torch_thread_config"],
            )
            run_id = None
            active_run = mlflow.active_run()
            if active_run is not None:
                run_id = active_run.info.run_id

            ticker_data, store_rows, preprocessed_df = performance.prepare_ticker_inputs(
                ticker_df=ticker_job["ticker_df"],
                ticker_train_df=ticker_job["ticker_train_df"],
                ticker_test_df=ticker_job["ticker_test_df"],
                ticker=ticker_job["ticker"],
                feature_columns=ticker_job["feature_columns"],
                preprocess_params=ticker_job["preprocess_params"],
                test_horizon=ticker_job["test_horizon"],
                data_preprocessor_cls=runtime["data_preprocessor_cls"],
                publish_preprocessed_to_store=publish_preprocessed_to_store,
            )
            pecnet, joined_df, combined_metrics, selection_df = _train_one_ticker(
                ticker_data=ticker_data,
                ticker_train_df=ticker_job["ticker_train_df"],
                ticker_test_df=ticker_job["ticker_test_df"],
                hyperparams=ticker_job["hyperparams"],
                utility=runtime["utility"],
                pecnet_builder_cls=runtime["pecnet_builder_cls"],
                basic_nn_cls=runtime["basic_nn_cls"],
                feature_selector_cls=runtime["feature_selector_cls"],
                torch_module=runtime["torch"],
                tier_name=ticker_job["tier_name"],
                selection_params=ticker_job["selection_params"],
            )
            regression_df, long_direction_df = performance.split_combined_metrics(
                combined_metrics
            )
            performance.log_ticker_results(
                ticker=ticker_job["ticker"],
                ticker_train_df=ticker_job["ticker_train_df"],
                joined_df=joined_df,
                regression_df=regression_df,
                long_direction_df=long_direction_df,
                selection_df=selection_df,
                pecnet=pecnet,
                torch_module=runtime["torch"],
            )

        return {
            "ticker": ticker_job["ticker"],
            "prediction_frame": joined_df,
            "regression_frame": regression_df,
            "long_direction_frame": long_direction_df,
            "selection_frame": selection_df,
            "preprocessed_frame": preprocessed_df,
            "preprocessed_store_rows": int(store_rows),
            "mlflow_run_id": run_id,
        }

    @staticmethod
    def _run_ticker_job_in_child_process(ticker_job: dict[str, Any]) -> dict[str, Any]:
        os.environ["MODEL_N_JOBS"] = str(ticker_job["torch_threads"])
        configure_mlflow_tracking(
            tracking_uri=ticker_job["mlflow_params"].get("tracking_uri"),
            experiment_name=tier_experiment_name(ticker_job["tier_name"]),
        )
        runtime = PecnetTrainingWorkflow._load_runtime({})
        performance = PecnetPerformanceMeasurement(tier_name=ticker_job["tier_name"])
        return PecnetTrainingWorkflow._run_ticker_job(
            ticker_job=ticker_job,
            runtime=runtime,
            performance=performance,
            start_child_run=True,
            publish_preprocessed_to_store=False,
        )

    @staticmethod
    def _append_ticker_result(outputs: dict[str, Any], result: dict[str, Any]) -> None:
        outputs["models"][result["ticker"]] = "logged_to_mlflow"
        outputs["prediction_frames"].append(result["prediction_frame"])
        outputs["regression_frames"].append(result["regression_frame"])
        outputs["long_direction_frames"].append(result["long_direction_frame"])
        if not result["selection_frame"].empty:
            outputs["selection_frames"].append(result["selection_frame"])
        preprocessed_frame = result.get("preprocessed_frame")
        if (
            result["preprocessed_store_rows"] == 0
            and preprocessed_frame is not None
            and not preprocessed_frame.empty
        ):
            outputs["preprocessed_frames"].append(preprocessed_frame)
            outputs["preprocessed_run_ids"][result["ticker"]] = result.get("mlflow_run_id")
        outputs["preprocessed_store_rows"] += result["preprocessed_store_rows"]

    @staticmethod
    def _publish_deferred_preprocessed_inputs(
        *,
        outputs: dict[str, Any],
        performance: PecnetPerformanceMeasurement,
    ) -> None:
        frames = outputs.pop("preprocessed_frames", [])
        run_ids = outputs.pop("preprocessed_run_ids", {})
        if not frames:
            return

        preprocessed_df = pd.concat(frames, ignore_index=True)
        metadata = PecnetTrainingWorkflow._publish_pecnet_preprocessed_frame(preprocessed_df)
        outputs["preprocessed_store_rows"] += int(metadata.get("timescale_rows", 0))

        for ticker, ticker_df in preprocessed_df.groupby("symbol", observed=True):
            ticker_metadata = {
                **metadata,
                "timescale_rows": len(ticker_df),
                "feast_online_rows": len(ticker_df),
                "published_in_parent_process": True,
            }
            run_id = run_ids.get(str(ticker))
            if run_id:
                with mlflow.start_run(run_id=run_id):
                    performance.log_preprocessed_store_metadata(
                        ticker=str(ticker),
                        store_metadata=ticker_metadata,
                    )

    @staticmethod
    def _publish_pecnet_preprocessed_frame(
        preprocessed_df: pd.DataFrame,
    ) -> dict[str, object]:
        from ....serving.feast_store import (  # noqa: PLC0415
            publish_pecnet_preprocessed_training_data,
        )

        return publish_pecnet_preprocessed_training_data(pl.from_pandas(preprocessed_df))

    @staticmethod
    def _pecnet_worker_count(model_spec: dict[str, Any], ticker_count: int) -> int:
        if ticker_count <= 1:
            return 1

        runtime_params = model_spec.get("_runtime") or {}
        requested = runtime_params.get("pecnet_n_jobs", os.getenv("PECNET_N_JOBS", 1))
        return min(PecnetTrainingWorkflow._resolve_worker_count(requested), ticker_count)

    @staticmethod
    def _pecnet_worker_torch_threads(model_spec: dict[str, Any], n_jobs: int) -> int:
        runtime_params = model_spec.get("_runtime") or {}
        configured = runtime_params.get(
            "pecnet_torch_threads_per_worker",
            os.getenv("PECNET_TORCH_THREADS_PER_WORKER"),
        )
        if configured is not None:
            return PecnetTrainingWorkflow._resolve_worker_count(configured)

        return max(1, cpu_count_from_env("MODEL_N_JOBS") // max(1, n_jobs))

    @staticmethod
    def _resolve_worker_count(value: Any) -> int:
        requested = int(value)
        if requested <= 0:
            return os.cpu_count() or 1
        return max(1, requested)
