from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd

from ...common import log_mlflow_datasets
from ...local_artifacts import LightweightArtifactStore
from ...plots import log_forecast_plots
from ..data import PecnetDataPreprocessor
from ..runtime import _safe_name
from ..selection import _max_selected_features_for_tier
from .ticker_logs import (
    log_ticker_model_artifact,
    log_ticker_model_metrics,
    log_ticker_selection_metrics,
)


@dataclass(slots=True)
class PecnetPerformanceMeasurement:
    tier_name: str

    def prepare_ticker_inputs(
        self,
        *,
        ticker_df: pd.DataFrame,
        ticker_train_df: pd.DataFrame,
        ticker_test_df: pd.DataFrame,
        ticker: str,
        feature_columns: list[str],
        preprocess_params: dict[str, Any],
        test_horizon: int,
        data_preprocessor_cls,
        publish_preprocessed_to_store: bool = True,
    ) -> tuple[dict[str, Any], int, pd.DataFrame]:
        ticker_safe = _safe_name(str(ticker))
        log_mlflow_datasets(
            train_df=ticker_train_df,
            test_df=ticker_test_df,
            dataset_prefix=f"stock_close_{self.tier_name}_{ticker_safe}_pecnet",
            artifact_prefix=f"pecnet/{self.tier_name}/tickers/{ticker_safe}",
        )
        ticker_data, store_metadata, preprocessed_df = PecnetDataPreprocessor(
            data_preprocessor_cls,
        ).prepare_ticker_inputs(
            ticker_df=ticker_df,
            ticker_train_df=ticker_train_df,
            ticker_test_df=ticker_test_df,
            ticker=str(ticker),
            tier_name=self.tier_name,
            feature_columns=feature_columns,
            preprocess_params=preprocess_params,
            test_horizon=test_horizon,
            publish_to_store=publish_preprocessed_to_store,
        )
        self.log_preprocessed_store_metadata(
            ticker=ticker,
            store_metadata=store_metadata,
        )
        return ticker_data, int(store_metadata.get("timescale_rows", 0)), preprocessed_df

    def log_preprocessed_store_metadata(
        self,
        *,
        ticker: str,
        store_metadata: dict[str, object],
    ) -> None:
        ticker_safe = _safe_name(str(ticker))
        mlflow.log_dict(
            store_metadata,
            f"pecnet/{self.tier_name}/tickers/{ticker_safe}/preprocessed/"
            "store_metadata.json",
        )

    def log_ticker_metadata(
        self,
        *,
        ticker: str,
        feature_columns: list[str],
        preprocess_params: dict[str, Any],
        hyperparams: dict[str, Any],
        selection_params: dict[str, Any],
        test_horizon: int,
        torch_thread_config: dict[str, Any],
    ) -> None:
        PecnetPerformanceMeasurement.log_ticker_run_metadata(
            ticker=ticker,
            tier_name=self.tier_name,
            feature_columns=feature_columns,
            preprocess_params=preprocess_params,
            hyperparams=hyperparams,
            selection_params=selection_params,
            test_horizon=test_horizon,
            torch_thread_config=torch_thread_config,
        )

    def log_ticker_results(
        self,
        *,
        ticker: str,
        ticker_train_df: pd.DataFrame,
        joined_df: pd.DataFrame,
        regression_df: pd.DataFrame,
        long_direction_df: pd.DataFrame,
        selection_df: pd.DataFrame,
        pecnet,
        torch_module,
    ) -> None:
        PecnetPerformanceMeasurement.log_ticker_results_for_tier(
            ticker=ticker,
            tier_name=self.tier_name,
            ticker_train_df=ticker_train_df,
            joined_df=joined_df,
            regression_df=regression_df,
            long_direction_df=long_direction_df,
            selection_df=selection_df,
            pecnet=pecnet,
            torch_module=torch_module,
        )






    @staticmethod
    def log_ticker_run_metadata(
        *,
        ticker: str,
        tier_name: str,
        feature_columns: list[str],
        preprocess_params: dict[str, Any],
        hyperparams: dict[str, Any],
        selection_params: dict[str, Any],
        test_horizon: int,
        torch_thread_config: dict[str, Any],
    ) -> None:
        ticker_safe = _safe_name(str(ticker))
        params = {
            "tier_name": tier_name,
            "ticker": str(ticker),
            "test_horizon": test_horizon,
            "feature_columns": ",".join(feature_columns),
            "pecnet.selection_strategy": (
                selection_params.get("strategy_by_tier", {}).get(
                    tier_name,
                    selection_params.get("strategy", "all_features"),
                )
            ),
            "pecnet.correlation_threshold": selection_params.get(
                "correlation_threshold",
                "",
            )
            or "",
            "pecnet.max_selected_features": selection_params.get(
                "max_selected_features",
                "",
            )
            or "",
            "pecnet.max_selected_features_for_tier": (
                _max_selected_features_for_tier(selection_params, tier_name) or ""
            ),
            "pecnet.torch_num_threads": torch_thread_config["torch_num_threads"],
            "pecnet.torch_num_interop_threads": (
                torch_thread_config["torch_num_interop_threads"] or ""
            ),
            "pecnet.torch_device": torch_thread_config.get("torch_device", ""),
            "pecnet.torch_device_requested": torch_thread_config.get(
                "torch_device_requested",
                "",
            ),
            "pecnet.torch_mps_available": torch_thread_config.get(
                "torch_mps_available",
                False,
            ),
            "pecnet.torch_cuda_available": torch_thread_config.get(
                "torch_cuda_available",
                False,
            ),
            **{
                f"pecnet.{key}": value
                for key, value in hyperparams.items()
                if not isinstance(value, (list, dict, tuple))
            },
        }
        artifact_prefix = f"pecnet/{tier_name}/tickers/{ticker_safe}/params"
        mlflow.log_params(params)
        mlflow.log_dict(preprocess_params, f"{artifact_prefix}/preprocess_params.json")
        mlflow.log_dict(hyperparams, f"{artifact_prefix}/hyperparams.json")
        mlflow.log_dict(selection_params, f"{artifact_prefix}/selection_params.json")
        store = LightweightArtifactStore()
        store.save_params(params, f"{artifact_prefix}/training_params.json")
        store.save_params(preprocess_params, f"{artifact_prefix}/preprocess_params.json")
        store.save_params(hyperparams, f"{artifact_prefix}/hyperparams.json")
        store.save_params(selection_params, f"{artifact_prefix}/selection_params.json")

    @staticmethod
    def split_combined_metrics(
        combined_metrics: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        regression_df = combined_metrics[
            combined_metrics["metric_family"] == "regression"
        ].drop(columns=["metric_family"])
        long_direction_df = combined_metrics[
            combined_metrics["metric_family"] == "long_direction"
        ].drop(columns=["metric_family"])
        return regression_df, long_direction_df

    @staticmethod
    def log_ticker_results_for_tier(
        *,
        ticker: str,
        tier_name: str,
        ticker_train_df: pd.DataFrame,
        joined_df: pd.DataFrame,
        regression_df: pd.DataFrame,
        long_direction_df: pd.DataFrame,
        selection_df: pd.DataFrame,
        pecnet,
        torch_module,
    ) -> None:
        ticker_safe = _safe_name(str(ticker))
        mlflow.log_table(joined_df, f"pecnet/{tier_name}/predictions/{ticker}.json")
        mlflow.log_table(
            regression_df,
            f"pecnet/{tier_name}/evaluation/{ticker}_regression.json",
        )
        mlflow.log_table(
            long_direction_df,
            f"pecnet/{tier_name}/evaluation/{ticker}_long_direction.json",
        )
        store = LightweightArtifactStore()
        store.save_metrics(
            regression_df,
            f"pecnet/{tier_name}/tickers/{ticker_safe}/metrics/regression_metrics.csv",
        )
        store.save_metrics(
            long_direction_df,
            f"pecnet/{tier_name}/tickers/{ticker_safe}/metrics/long_direction_metrics.csv",
        )
        store.save_metrics(
            selection_df,
            f"pecnet/{tier_name}/tickers/{ticker_safe}/metrics/feature_selection.csv",
        )

        if not selection_df.empty:
            log_ticker_selection_metrics(
                selection_df=selection_df,
                tier_name=tier_name,
                ticker_safe=ticker_safe,
            )
        log_ticker_model_metrics(
            regression_df=regression_df,
            long_direction_df=long_direction_df,
            ticker_safe=ticker_safe,
        )
        log_forecast_plots(
            train_df=ticker_train_df,
            joined_df=joined_df,
            levels=None,
            artifact_prefix=f"pecnet/{tier_name}/plots/{ticker_safe}",
        )
        log_ticker_model_artifact(
            pecnet=pecnet,
            torch_module=torch_module,
            tier_name=tier_name,
            ticker_safe=ticker_safe,
        )
