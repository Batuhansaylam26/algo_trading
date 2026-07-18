from __future__ import annotations

from dataclasses import dataclass

import mlflow
import pandas as pd

from .common import _prediction_frame, _regression_metrics, log_mlflow_datasets
from .metrics import long_only_directional_metrics
from .performance_result import ForecastPerformanceResult
from .plots import log_forecast_plots


@dataclass(slots=True)
class ForecastPerformanceMeasurement:
    model_family: str
    tier_name: str
    levels: list[int]

    @property
    def artifact_prefix(self) -> str:
        return f"{self.model_family}/{self.tier_name}"

    @property
    def metric_prefix(self) -> str:
        if self.model_family == "mlforecast":
            return ""
        return f"{self.model_family}."

    def log_datasets(
        self,
        *,
        train_df: pd.DataFrame,
        validation_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        log_mlflow_datasets(
            train_df=train_df,
            validation_df=validation_df,
            test_df=test_df,
            dataset_prefix=f"stock_close_{self.tier_name}_{self.model_family}",
            artifact_prefix=self.artifact_prefix,
        )

    def measure(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> ForecastPerformanceResult:
        joined_df = _prediction_frame(test_df, predictions)
        regression_df = _regression_metrics(joined_df)
        long_direction_df = long_only_directional_metrics(joined_df, train_df)
        return ForecastPerformanceResult(
            predictions=joined_df,
            regression_metrics=regression_df,
            long_direction_metrics=long_direction_df,
        )

    def log_result(
        self,
        *,
        train_df: pd.DataFrame,
        result: ForecastPerformanceResult,
    ) -> None:
        mlflow.log_table(
            result.predictions,
            f"{self.artifact_prefix}/evaluation/predictions.json",
        )
        mlflow.log_table(
            result.regression_metrics,
            f"{self.artifact_prefix}/evaluation/regression_metrics.json",
        )
        mlflow.log_table(
            result.long_direction_metrics,
            f"{self.artifact_prefix}/evaluation/long_only_direction_metrics.json",
        )
        log_forecast_plots(
            train_df=train_df,
            joined_df=result.predictions,
            levels=self.levels,
            artifact_prefix=f"{self.artifact_prefix}/plots",
        )
        self._log_regression_metrics(result.regression_metrics)
        self._log_long_direction_metrics(result.long_direction_metrics)

    def _log_regression_metrics(self, regression_df: pd.DataFrame) -> None:
        for _, row in regression_df.iterrows():
            for metric_name in ["mae", "rmse", "mape", "r2"]:
                if pd.isna(row.get(metric_name)):
                    continue
                mlflow.log_metric(
                    f"{self.metric_prefix}{row['model']}.test.{metric_name}",
                    float(row[metric_name]),
                )

    def _log_long_direction_metrics(
        self,
        long_direction_df: pd.DataFrame,
    ) -> None:
        for _, row in long_direction_df.iterrows():
            if row[["long_accuracy", "long_precision", "long_recall"]].isna().any():
                continue

            mlflow.log_metric(
                f"{self.metric_prefix}{row['model']}.long.accuracy",
                float(row["long_accuracy"]),
            )
            mlflow.log_metric(
                f"{self.metric_prefix}{row['model']}.long.precision",
                float(row["long_precision"]),
            )
            mlflow.log_metric(
                f"{self.metric_prefix}{row['model']}.long.recall",
                float(row["long_recall"]),
            )
            mlflow.log_metric(
                f"{self.metric_prefix}{row['model']}.long.signal_rate",
                float(row["long_signal_rate"]),
            )

    @staticmethod
    def model_names_from_predictions(joined_df: pd.DataFrame) -> list[str]:
        return [
            column
            for column in joined_df.columns
            if column not in {"unique_id", "ds", "y"}
            and "-lo-" not in column
            and "-hi-" not in column
        ]
