"""
Parallel Walk-Forward Execution Engine
=====================================
Executes walk-forward block training and inference across multiple GPUs
using ProcessPoolExecutor for process-level isolation (preventing DataPreprocessor
singleton state corruption) and live terminal progress tracking.
"""

from __future__ import annotations

import io
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import numpy as np
import pandas as pd

from mlops_kedro.pipelines.stock_close_training.ml.pecnet.runtime_class import (
    PecnetRuntime,
)
from mlops_kedro.pipelines.stock_close_training.ml.pecnet.selection.builder_class import (
    PecnetSelectionBuilder,
)
from mlops_kedro.walk_forward.contract import PecnetRequirementsContract
from mlops_kedro.walk_forward.splitter import (
    WalkForwardIndexBlock,
    WalkForwardSplitter,
)

LOGGER = logging.getLogger(__name__)


def _run_single_block_ticker_worker(job_args: dict[str, Any]) -> dict[str, Any]:
    """Child process worker for a single (block, ticker) job.

    Ensures CUDA device pinning, quiet logging, and fast-failing contract validation.
    """
    block_id = job_args["block_id"]
    ticker = job_args["ticker"]
    worker_id = job_args["worker_id"]
    num_gpus = job_args["num_gpus"]

    log_buffer = io.StringIO()

    try:
        with redirect_stdout(log_buffer), redirect_stderr(log_buffer):
            # 1. Round-robin GPU pinning
            if num_gpus > 0:
                gpu_id = worker_id % num_gpus
                os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                torch_device = f"cuda:{gpu_id}"
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                torch_device = "cpu"

            os.environ["PECNET_TORCH_DEVICE"] = torch_device

            # 2. Load runtime directly without triggering top-level ml package imports
            Utility, PecnetBuilder, DataPreprocessor, BasicNN, FeatureSelector, torch = (
                PecnetRuntime._load_pecnet_runtime()
            )
            DataPreprocessor().reset()

            # Configure Torch thread concurrency
            torch_thread_config = PecnetRuntime._configure_torch_threads(torch)

            # 3. Extract datasets explicitly (padded vs target evaluation dataframes)
            ticker_padded_train_df = job_args["ticker_padded_train_df"].copy()
            ticker_train_df = job_args["ticker_train_df"].copy()
            ticker_padded_test_df = job_args["ticker_padded_test_df"].copy()
            ticker_test_df = job_args["ticker_test_df"].copy()

            feature_columns = job_args["feature_columns"]
            preprocess_params = job_args["preprocess_params"].copy()
            hyperparams = job_args["hyperparams"]
            selection_params = job_args["selection_params"]
            tier_name = job_args["tier_name"]

            # 4. Instantiate & Validate Bar Requirement Contract
            contract = PecnetRequirementsContract.from_preprocess_params(preprocess_params)

            requested_train_samples = job_args["requested_train_samples"]
            requested_test_samples = job_args["requested_test_samples"]

            contract.validate_bar_counts(
                provided_train_bars=len(ticker_padded_train_df),
                provided_test_bars=len(ticker_padded_test_df),
                requested_train_samples=requested_train_samples,
                requested_test_samples=requested_test_samples,
            )

            test_ratio = contract.compute_concatenated_test_ratio(
                train_bars=len(ticker_padded_train_df),
                test_bars=len(ticker_padded_test_df),
            )
            preprocess_params["test_ratio"] = test_ratio

            ticker_combined_df = pd.concat(
                [ticker_padded_train_df, ticker_padded_test_df],
                ignore_index=True,
            )

            # 5. Single-pass aligned preprocessing
            dp = DataPreprocessor()
            dp.reset()

            target_full_series = ticker_combined_df["y"].to_numpy(dtype=float)
            X_train_target, X_test_target, y_train, y_test = dp.preprocess(
                data=target_full_series,
                profile="target",
                fit=True,
                **preprocess_params,
            )

            feature_X_trains = []
            feature_X_tests = []
            available_feature_columns = [
                col for col in feature_columns if col in ticker_combined_df.columns
            ]

            for col in available_feature_columns:
                X_train_feat, X_test_feat, _, _ = dp.preprocess(
                    data=ticker_combined_df[col].to_numpy(dtype=float),
                    profile=f"feature:{col}",
                    fit=True,
                    **preprocess_params,
                )
                feature_X_trains.append(X_train_feat)
                feature_X_tests.append(X_test_feat)

            ticker_data = {
                "ticker": ticker,
                "X_train_target": X_train_target,
                "X_test_target": X_test_target,
                "y_train": y_train,
                "y_test": y_test,
                "feature_X_trains": feature_X_trains,
                "feature_X_tests": feature_X_tests,
                "feature_names": available_feature_columns,
            }

            # 6. Configure hyperparams & build model
            Utility.set_seed(hyperparams["seed"])
            Utility.set_hyperparameters(
                learning_rate=hyperparams["learning_rate"],
                epoch_size=hyperparams["epoch_size"],
                batch_size=hyperparams["batch_size"],
                hidden_units_sizes=hyperparams["hidden_units_sizes"],
            )

            builder = PecnetBuilder()
            builder, selected_X_test, _ = PecnetSelectionBuilder._build_pecnet_variables(
                builder=builder,
                ticker_data=ticker_data,
                tier_name=tier_name,
                feature_selector_cls=FeatureSelector,
                selection_params=selection_params,
            )
            pecnet = builder.add_error_network().add_final_network().build()

            # 7. Execute inference on padded test window
            raw_predictions = pecnet.predict(*selected_X_test, test_target=y_test)

            if torch.is_tensor(raw_predictions):
                preds_array = raw_predictions.detach().cpu().numpy().reshape(-1)
            else:
                preds_array = np.asarray(raw_predictions, dtype=float).reshape(-1)

            if len(preds_array) > 1:
                preds_array = preds_array[:-1]

            # 8. Trim warmup padding: keep ONLY actual test block prediction rows
            eval_test_len = len(ticker_test_df)
            clean_test_preds = (
                preds_array[-eval_test_len:] if eval_test_len > 0 else np.array([])
            )

            res_df = ticker_test_df[["ds", "unique_id", "y"]].copy()
            res_df = res_df.reset_index(drop=True)
            res_df["prediction"] = clean_test_preds
            res_df["block_id"] = block_id
            res_df["tier"] = tier_name

            return {
                "block_id": block_id,
                "ticker": ticker,
                "torch_device": torch_device,
                "predictions_df": res_df,
            }

    except Exception:
        worker_log = log_buffer.getvalue()
        if worker_log:
            sys.stderr.write(
                f"\n{'='*80}\n[WORKER CRASH LOG | Block {block_id} | Ticker {ticker}]\n"
                f"{worker_log}\n{'='*80}\n"
            )
            sys.stderr.flush()
        raise


class ParallelWalkForwardRunner:
    """Orchestrates parallel multi-GPU execution of walk-forward model blocks."""

    def __init__(
        self,
        num_gpus: int,
        max_workers: int,
    ) -> None:
        if num_gpus < 0:
            raise ValueError(f"num_gpus cannot be negative, got {num_gpus}")
        if max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {max_workers}")

        self.num_gpus = num_gpus
        self.max_workers = max_workers

    def run_walk_forward(
        self,
        full_df: pd.DataFrame,
        model_spec: dict[str, Any],
        splitter: WalkForwardSplitter,
    ) -> pd.DataFrame:
        """Runs walk-forward model training and produces a continuous OOS prediction DataFrame."""
        if full_df.empty:
            raise ValueError("Input full_df cannot be empty.")

        preprocess_params = model_spec["preprocess_params"]
        hyperparams = model_spec["hyperparams"]
        selection_params = model_spec.get("selection_params", {})
        tier_name = model_spec["tier_name"]
        feature_columns = model_spec["feature_columns"]

        contract = PecnetRequirementsContract.from_preprocess_params(preprocess_params)

        work_df = full_df.copy()
        if "ds" in work_df.columns:
            work_df["ds"] = pd.to_datetime(work_df["ds"], utc=True)
            work_df = work_df.sort_values("ds").reset_index(drop=True)

        unique_tickers = work_df["unique_id"].unique()

        jobs = []
        worker_counter = 0

        # Process per ticker to slice exact integer indices
        for ticker in unique_tickers:
            ticker_df = work_df[work_df["unique_id"] == ticker].reset_index(drop=True)
            total_sequence_len = len(ticker_df)

            index_blocks = splitter.split_indices(
                total_sequence_length=total_sequence_len,
                padding_length=contract.padding_length,
            )

            for block in index_blocks:
                t_padded_train = ticker_df.iloc[block.padded_train_slice].copy()
                t_train = ticker_df.iloc[block.train_slice].copy()
                t_padded_test = ticker_df.iloc[block.padded_test_slice].copy()
                t_test = ticker_df.iloc[block.test_slice].copy()

                if t_padded_train.empty or t_test.empty:
                    continue

                req_train_samples = block.train_slice.stop - block.train_slice.start
                req_test_samples = block.test_slice.stop - block.test_slice.start

                jobs.append(
                    {
                        "block_id": block.block_id,
                        "ticker": str(ticker),
                        "worker_id": worker_counter,
                        "num_gpus": self.num_gpus,
                        "padding_length": contract.padding_length,
                        "requested_train_samples": req_train_samples,
                        "requested_test_samples": req_test_samples,
                        "ticker_padded_train_df": t_padded_train,
                        "ticker_train_df": t_train,
                        "ticker_padded_test_df": t_padded_test,
                        "ticker_test_df": t_test,
                        "feature_columns": feature_columns,
                        "preprocess_params": preprocess_params,
                        "hyperparams": hyperparams,
                        "selection_params": selection_params,
                        "tier_name": tier_name,
                    }
                )
                worker_counter += 1

        total_jobs = len(jobs)
        LOGGER.info(
            "Launching parallel walk-forward execution | jobs=%d max_workers=%d gpus=%d",
            total_jobs,
            self.max_workers,
            self.num_gpus,
        )

        print("\n" + "=" * 80)
        print(f"[Walk-Forward Execution Engine] Started {total_jobs} tasks across {self.num_gpus} GPUs ({self.max_workers} max workers)")
        print("=" * 80)

        start_time = time.time()
        completed_count = 0
        all_results = []

        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(_run_single_block_ticker_worker, job): (
                    job["block_id"],
                    job["ticker"],
                )
                for job in jobs
            }

            for future in as_completed(futures):
                b_id, ticker = futures[future]
                completed_count += 1
                try:
                    res = future.result()
                    all_results.append(res["predictions_df"])

                    elapsed = time.time() - start_time
                    percent = (completed_count / total_jobs) * 100.0
                    device_str = res.get("torch_device", "cpu")

                    print(
                        f"[{completed_count:3d}/{total_jobs:3d}] ({percent:5.1f}%) | "
                        f"Block {b_id:2d} | Ticker: {ticker:<6s} | Device: {device_str:<6s} | "
                        f"Elapsed: {elapsed:5.1f}s",
                        flush=True,
                    )

                except Exception:
                    LOGGER.exception(
                        "Walk-forward block job failed | block_id=%d ticker=%s",
                        b_id,
                        ticker,
                    )
                    raise

        print("-" * 80)
        print(f"[Walk-Forward Execution Engine] All {total_jobs} tasks completed in {time.time() - start_time:.2f}s")
        print("=" * 80 + "\n")

        if not all_results:
            return pd.DataFrame()

        stitched_oos = (
            pd.concat(all_results, ignore_index=True)
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )

        LOGGER.info(
            "Stitched %d out-of-sample prediction rows across tasks.",
            len(stitched_oos),
        )
        return stitched_oos
