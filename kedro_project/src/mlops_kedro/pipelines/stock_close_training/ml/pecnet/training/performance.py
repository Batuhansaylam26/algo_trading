from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd

from ...common import log_mlflow_datasets
from ...plots import log_forecast_plots
from ..data import PecnetDataPreprocessor
from ..runtime import _safe_name
from ..selection import _log_feature_selection_heatmap, _max_selected_features_for_tier
from .ticker_logs import (
    log_ticker_model_artifact,
    log_ticker_model_metrics,
    log_ticker_prediction_plot,
    log_ticker_selection_metrics,
)


@dataclass(slots=True)
class PecnetPerformanceMeasurement:
    tier_name: str

    def log_parent_metadata(
        self,
        *,
        model_spec: dict[str, Any],
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: list[str],
        preprocess_params: dict[str, Any],
        hyperparams: dict[str, Any],
        selection_params: dict[str, Any],
        torch_thread_config: dict[str, int | None],
    ) -> None:
        log_parent_run_metadata(
            model_spec=model_spec,
            train_df=train_df,
            test_df=test_df,
            feature_columns=feature_columns,
            preprocess_params=preprocess_params,
            hyperparams=hyperparams,
            selection_params=selection_params,
            tier_name=self.tier_name,
            torch_thread_config=torch_thread_config,
        )

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
    ) -> tuple[dict[str, Any], int]:
        ticker_safe = _safe_name(str(ticker))
        log_mlflow_datasets(
            train_df=ticker_train_df,
            test_df=ticker_test_df,
            dataset_prefix=f"stock_close_{self.tier_name}_{ticker_safe}_pecnet",
            artifact_prefix=f"pecnet/{self.tier_name}/tickers/{ticker_safe}",
        )
        ticker_data, store_metadata = PecnetDataPreprocessor(
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
        )
        mlflow.log_dict(
            store_metadata,
            f"pecnet/{self.tier_name}/tickers/{ticker_safe}/preprocessed/"
            "store_metadata.json",
        )
        return ticker_data, int(store_metadata.get("timescale_rows", 0))

    @staticmethod
    def split_combined_metrics(
        combined_metrics: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        return split_combined_metrics(combined_metrics)

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
        log_ticker_results(
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

    def log_parent_outputs(
        self,
        *,
        predictions: pd.DataFrame,
        regression_metrics: pd.DataFrame,
        long_direction_metrics: pd.DataFrame,
        feature_selection: pd.DataFrame,
    ) -> None:
        log_parent_run_outputs(
            tier_name=self.tier_name,
            predictions=predictions,
            regression_metrics=regression_metrics,
            long_direction_metrics=long_direction_metrics,
            feature_selection=feature_selection,
        )


def log_parent_run_metadata(
    *,
    model_spec: dict[str, Any],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
    preprocess_params: dict[str, Any],
    hyperparams: dict[str, Any],
    selection_params: dict[str, Any],
    tier_name: str,
    torch_thread_config: dict[str, int | None],
) -> None:
    mlflow.log_params(
        {
            "tier_name": tier_name,
            "test_horizon": model_spec["test_horizon"],
            "feature_columns": ",".join(feature_columns),
            "wandb_project": model_spec["wandb_project"],
            "wandb_mode": model_spec["wandb_mode"],
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
            **{
                f"pecnet.{key}": value
                for key, value in hyperparams.items()
                if not isinstance(value, (list, dict, tuple))
            },
        }
    )
    mlflow.log_dict(preprocess_params, f"pecnet/{tier_name}/preprocess_params.json")
    mlflow.log_dict(hyperparams, f"pecnet/{tier_name}/hyperparams.json")
    mlflow.log_dict(selection_params, f"pecnet/{tier_name}/selection_params.json")
    log_mlflow_datasets(
        train_df=train_df,
        test_df=test_df,
        dataset_prefix=f"stock_close_{tier_name}_pecnet",
        artifact_prefix=f"pecnet/{tier_name}",
    )

def prepare_ticker_inputs(
    *,
    ticker_df: pd.DataFrame,
    ticker_train_df: pd.DataFrame,
    ticker_test_df: pd.DataFrame,
    ticker: str,
    tier_name: str,
    feature_columns: list[str],
    preprocess_params: dict[str, Any],
    test_horizon: int,
    data_preprocessor_cls,
) -> tuple[dict[str, Any], int]:
    return PecnetPerformanceMeasurement(tier_name).prepare_ticker_inputs(
        ticker_df=ticker_df,
        ticker_train_df=ticker_train_df,
        ticker_test_df=ticker_test_df,
        ticker=ticker,
        feature_columns=feature_columns,
        preprocess_params=preprocess_params,
        test_horizon=test_horizon,
        data_preprocessor_cls=data_preprocessor_cls,
    )

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

def log_ticker_results(
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
    log_ticker_prediction_plot(
        joined_df=joined_df,
        tier_name=tier_name,
        ticker=str(ticker),
        ticker_safe=ticker_safe,
    )
    log_ticker_model_artifact(
        pecnet=pecnet,
        torch_module=torch_module,
        tier_name=tier_name,
        ticker_safe=ticker_safe,
    )

def log_parent_run_outputs(
    *,
    tier_name: str,
    predictions: pd.DataFrame,
    regression_metrics: pd.DataFrame,
    long_direction_metrics: pd.DataFrame,
    feature_selection: pd.DataFrame,
) -> None:
    mlflow.log_table(predictions, f"pecnet/{tier_name}/predictions/all_predictions.json")
    mlflow.log_table(
        regression_metrics,
        f"pecnet/{tier_name}/evaluation/all_regression_metrics.json",
    )
    mlflow.log_table(
        long_direction_metrics,
        f"pecnet/{tier_name}/evaluation/all_long_direction_metrics.json",
    )
    mlflow.log_table(
        feature_selection,
        f"pecnet/{tier_name}/feature_selection/all_feature_selection.json",
    )
    _log_feature_selection_heatmap(
        feature_selection,
        artifact_file=f"pecnet/{tier_name}/feature_selection/all_correlation_heatmap.png",
        title=f"PECNet {tier_name} selected feature correlations",
        index_column="ticker",
    )
