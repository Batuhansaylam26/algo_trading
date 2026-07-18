from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import mlflow
import pandas as pd

from .common import _regression_metrics, configure_mlflow_tracking, log_mlflow_datasets
from .metrics import long_only_directional_metrics, model_prediction_columns


_PREDICTION_INPUT_PATTERN = re.compile(
    r"^(?P<tier>tier\d+)_(?P<family>mlforecast|statsforecast|pecnet)_predictions$"
)


@dataclass(slots=True)
class RootModelPerformanceEvaluator:
    artifact_prefix: str = "root_performance"

    def evaluate(
        self,
        mlflow_params: dict[str, Any] | None = None,
        **inputs: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        mlflow_params = mlflow_params or {}
        configure_mlflow_tracking(
            tracking_uri=mlflow_params.get("tracking_uri"),
            experiment_name=mlflow_params.get("experiment_name"),
        )

        predictions, regression_metrics, long_metrics, split_metadata = (
            self._measure_all(inputs)
        )

        with mlflow.start_run(run_name="stock-close-root-performance") as run:
            self._log_split_inputs(inputs, split_metadata)
            self._log_tables(
                predictions=predictions,
                regression_metrics=regression_metrics,
                long_metrics=long_metrics,
            )
            self._log_metrics(regression_metrics, long_metrics)
            mlflow.log_dict(split_metadata, f"{self.artifact_prefix}/split_metadata.json")
            metadata = self._metadata(
                run_id=run.info.run_id,
                predictions=predictions,
                regression_metrics=regression_metrics,
                long_metrics=long_metrics,
                split_metadata=split_metadata,
            )

        return predictions, regression_metrics, long_metrics, metadata

    def _measure_all(
        self,
        inputs: dict[str, Any],
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        prediction_frames = []
        regression_frames = []
        long_frames = []
        split_metadata = {}

        for input_name, prediction_df in sorted(inputs.items()):
            match = _PREDICTION_INPUT_PATTERN.match(input_name)
            if match is None:
                continue

            tier_name = match.group("tier")
            model_family = match.group("family")
            split = inputs[f"{tier_name}_train_test_split"]
            train_df = split["train"]
            test_df = split["test"]
            tier_metadata = split_metadata.setdefault(
                tier_name,
                self._split_metadata(tier_name, train_df, test_df),
            )

            if prediction_df is None or prediction_df.empty:
                continue

            joined_df = self._align_predictions_to_test(
                predictions=prediction_df,
                test_df=test_df,
            )
            prediction_columns = model_prediction_columns(joined_df)
            self._record_output_metadata(
                tier_metadata=tier_metadata,
                model_family=model_family,
                joined_df=joined_df,
                prediction_columns=prediction_columns,
            )
            regression_df = _regression_metrics(joined_df).assign(
                tier=tier_name,
                model_family=model_family,
            )
            long_df = long_only_directional_metrics(joined_df, train_df).assign(
                tier=tier_name,
                model_family=model_family,
            )

            prediction_frames.append(
                joined_df.assign(tier=tier_name, model_family=model_family)
            )
            regression_frames.append(regression_df)
            long_frames.append(long_df)

        return (
            self._concat(prediction_frames),
            self._ordered_metrics(self._concat(regression_frames)),
            self._ordered_metrics(self._concat(long_frames)),
            split_metadata,
        )

    @staticmethod
    def _align_predictions_to_test(
        *,
        predictions: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> pd.DataFrame:
        test_actuals = (
            test_df[["unique_id", "ds", "y"]]
            .sort_values(["unique_id", "ds"])
            .reset_index(drop=True)
        )
        prediction_columns = model_prediction_columns(predictions)
        prediction_values = (
            predictions[["unique_id", "ds", *prediction_columns]]
            .sort_values(["unique_id", "ds"])
            .drop_duplicates(["unique_id", "ds"], keep="last")
            .reset_index(drop=True)
        )
        return test_actuals.merge(
            prediction_values,
            on=["unique_id", "ds"],
            how="left",
            validate="one_to_one",
        )

    def _log_split_inputs(
        self,
        inputs: dict[str, Any],
        split_metadata: dict[str, Any],
    ) -> None:
        for tier_name in sorted(split_metadata):
            split = inputs[f"{tier_name}_train_test_split"]
            log_mlflow_datasets(
                train_df=split["train"],
                test_df=split["test"],
                dataset_prefix=f"stock_close_{tier_name}_root_performance",
                artifact_prefix=f"{self.artifact_prefix}/{tier_name}",
            )

    def _log_tables(
        self,
        *,
        predictions: pd.DataFrame,
        regression_metrics: pd.DataFrame,
        long_metrics: pd.DataFrame,
    ) -> None:
        mlflow.log_table(predictions, f"{self.artifact_prefix}/predictions.json")
        mlflow.log_table(
            regression_metrics,
            f"{self.artifact_prefix}/regression_metrics.json",
        )
        mlflow.log_table(
            long_metrics,
            f"{self.artifact_prefix}/long_direction_metrics.json",
        )

    def _log_metrics(
        self,
        regression_metrics: pd.DataFrame,
        long_metrics: pd.DataFrame,
    ) -> None:
        for _, row in regression_metrics.iterrows():
            for metric_name in ["mae", "rmse", "mape", "r2"]:
                metric_value = row.get(metric_name)
                if pd.isna(metric_value):
                    continue
                mlflow.log_metric(
                    self._metric_name(row, "test", metric_name),
                    float(metric_value),
                )

        for _, row in long_metrics.iterrows():
            for metric_name, column in {
                "accuracy": "long_accuracy",
                "precision": "long_precision",
                "recall": "long_recall",
                "signal_rate": "long_signal_rate",
            }.items():
                metric_value = row.get(column)
                if pd.isna(metric_value):
                    continue
                mlflow.log_metric(
                    self._metric_name(row, "long", metric_name),
                    float(metric_value),
                )

    @staticmethod
    def _metric_name(
        row: pd.Series,
        metric_group: str,
        metric_name: str,
    ) -> str:
        return ".".join(
            [
                "root",
                _safe_metric_component(row["tier"]),
                _safe_metric_component(row["model_family"]),
                _safe_metric_component(row["model"]),
                metric_group,
                metric_name,
            ]
        )

    @staticmethod
    def _split_metadata(
        tier_name: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> dict[str, Any]:
        return {
            "tier": tier_name,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "symbols": sorted(test_df["unique_id"].dropna().unique().tolist()),
            "train_min_ds": str(train_df["ds"].min()) if not train_df.empty else None,
            "train_max_ds": str(train_df["ds"].max()) if not train_df.empty else None,
            "test_min_ds": str(test_df["ds"].min()) if not test_df.empty else None,
            "test_max_ds": str(test_df["ds"].max()) if not test_df.empty else None,
            "model_outputs": {},
        }

    @staticmethod
    def _record_output_metadata(
        *,
        tier_metadata: dict[str, Any],
        model_family: str,
        joined_df: pd.DataFrame,
        prediction_columns: list[str],
    ) -> None:
        missing_rows = (
            int(joined_df[prediction_columns].isna().all(axis=1).sum())
            if prediction_columns
            else len(joined_df)
        )
        tier_metadata["model_outputs"][model_family] = {
            "models": prediction_columns,
            "test_rows": len(joined_df),
            "rows_without_any_prediction": missing_rows,
        }

    @staticmethod
    def _metadata(
        *,
        run_id: str,
        predictions: pd.DataFrame,
        regression_metrics: pd.DataFrame,
        long_metrics: pd.DataFrame,
        split_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "mlflow_root_run_id": run_id,
            "evaluated_tiers": sorted(split_metadata),
            "prediction_rows": len(predictions),
            "regression_metric_rows": len(regression_metrics),
            "long_direction_metric_rows": len(long_metrics),
        }

    @staticmethod
    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    @staticmethod
    def _ordered_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
        if metrics.empty:
            return metrics
        leading_columns = ["tier", "model_family", "model"]
        remaining_columns = [
            column for column in metrics.columns if column not in leading_columns
        ]
        return metrics[leading_columns + remaining_columns]


def _safe_metric_component(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"
