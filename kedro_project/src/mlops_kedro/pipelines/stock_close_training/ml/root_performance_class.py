from __future__ import annotations

import json
import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

from .common import (
    _regression_metrics_by_unique_id,
    configure_mlflow_tracking,
    log_mlflow_datasets,
    tier_experiment_name,
)
from ..features.feature_sets import MODEL_TIER_NAMES, PECNET_ONLY_TIER_NAMES
from .local_artifacts import LightweightArtifactStore
from .metrics import long_only_directional_metrics_by_unique_id, model_prediction_columns


LOGGER = logging.getLogger(__name__)

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
            store = LightweightArtifactStore()
            store.save_params(
                mlflow_params,
                f"{self.artifact_prefix}/params/mlflow_params.json",
            )
            store.save_params(
                split_metadata,
                f"{self.artifact_prefix}/params/split_metadata.json",
            )
            metadata = self._metadata(
                run_id=run.info.run_id,
                predictions=predictions,
                regression_metrics=regression_metrics,
                long_metrics=long_metrics,
                split_metadata=split_metadata,
            )

        return predictions, regression_metrics, long_metrics, metadata

    def evaluate_from_mlflow(
        self,
        mlflow_params: dict[str, Any] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        mlflow_params = mlflow_params or {}
        configure_mlflow_tracking(
            tracking_uri=mlflow_params.get("tracking_uri"),
            experiment_name=mlflow_params.get("experiment_name"),
        )

        predictions, regression_metrics, long_metrics, source_metadata = (
            self._load_latest_outputs_from_mlflow()
        )

        with mlflow.start_run(run_name="stock-close-root-performance-from-mlflow") as run:
            self._log_tables(
                predictions=predictions,
                regression_metrics=regression_metrics,
                long_metrics=long_metrics,
            )
            self._log_metrics(regression_metrics, long_metrics)
            mlflow.log_dict(source_metadata, f"{self.artifact_prefix}/source_metadata.json")
            store = LightweightArtifactStore()
            store.save_params(
                mlflow_params,
                f"{self.artifact_prefix}/params/mlflow_params.json",
            )
            store.save_params(
                source_metadata,
                f"{self.artifact_prefix}/params/source_metadata.json",
            )
            metadata = self._metadata(
                run_id=run.info.run_id,
                predictions=predictions,
                regression_metrics=regression_metrics,
                long_metrics=long_metrics,
                split_metadata={},
            )
            metadata["source"] = "mlflow"
            metadata["evaluated_model_outputs"] = sorted(
                source_metadata["model_outputs"]
            )
            metadata["evaluated_tiers"] = sorted(
                {
                    output_name.split("_", maxsplit=1)[0]
                    for output_name in source_metadata["model_outputs"]
                }
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
            regression_df = _regression_metrics_by_unique_id(joined_df).assign(
                tier=tier_name,
                model_family=model_family,
            )
            long_df = long_only_directional_metrics_by_unique_id(
                joined_df,
                train_df,
            ).assign(
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

    def _load_latest_outputs_from_mlflow(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
        client = MlflowClient()
        prediction_frames = []
        regression_frames = []
        long_frames = []
        source_metadata: dict[str, Any] = {"source": "mlflow", "model_outputs": {}}

        for tier_name, model_family in self._expected_model_outputs():
            loaded = (
                self._load_latest_pecnet_outputs(client, tier_name)
                if model_family == "pecnet"
                else self._load_latest_family_outputs(client, tier_name, model_family)
            )
            if loaded is None:
                continue

            predictions, regression_metrics, long_metrics, metadata = loaded
            prediction_frames.append(predictions)
            regression_frames.append(regression_metrics)
            long_frames.append(long_metrics)
            source_metadata["model_outputs"][f"{tier_name}_{model_family}"] = metadata

        return (
            self._concat(prediction_frames),
            self._ordered_metrics(self._concat(regression_frames)),
            self._ordered_metrics(self._concat(long_frames)),
            source_metadata,
        )

    @staticmethod
    def _expected_model_outputs() -> list[tuple[str, str]]:
        outputs: list[tuple[str, str]] = []
        for tier_name in MODEL_TIER_NAMES:
            outputs.extend(
                [
                    (tier_name, "mlforecast"),
                    (tier_name, "statsforecast"),
                    (tier_name, "pecnet"),
                ]
            )
        for tier_name in PECNET_ONLY_TIER_NAMES:
            outputs.append((tier_name, "pecnet"))
        return outputs

    def _load_latest_family_outputs(
        self,
        client: MlflowClient,
        tier_name: str,
        model_family: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
        run_name = self._family_run_name(tier_name, model_family)
        run = self._latest_run_by_name(client, tier_name, run_name)
        if run is None:
            LOGGER.info("No MLflow run found for %s %s", tier_name, model_family)
            return None

        artifact_prefix = f"{model_family}/{tier_name}/evaluation"
        try:
            predictions = self._read_artifact_table(
                client,
                run.info.run_id,
                f"{artifact_prefix}/predictions.json",
            )
            regression_metrics = self._read_artifact_table(
                client,
                run.info.run_id,
                f"{artifact_prefix}/regression_metrics.json",
            )
            long_metrics = self._read_artifact_table(
                client,
                run.info.run_id,
                f"{artifact_prefix}/long_only_direction_metrics.json",
            )
        except Exception:
            LOGGER.warning(
                "Could not load MLflow artifacts for %s %s run_id=%s",
                tier_name,
                model_family,
                run.info.run_id,
                exc_info=True,
            )
            return None

        return (
            self._with_source_columns(predictions, tier_name, model_family),
            self._with_source_columns(regression_metrics, tier_name, model_family),
            self._with_source_columns(long_metrics, tier_name, model_family),
            self._run_metadata(run, row_count=len(predictions)),
        )

    def _load_latest_pecnet_outputs(
        self,
        client: MlflowClient,
        tier_name: str,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]] | None:
        runs = self._latest_pecnet_ticker_runs(client, tier_name)
        if not runs:
            LOGGER.info("No PECNet ticker MLflow runs found for %s", tier_name)
            return None

        prediction_frames = []
        regression_frames = []
        long_frames = []
        ticker_metadata = {}

        for ticker, run in runs.items():
            try:
                prediction_frames.append(
                    self._read_artifact_table(
                        client,
                        run.info.run_id,
                        f"pecnet/{tier_name}/predictions/{ticker}.json",
                    )
                )
                regression_frames.append(
                    self._read_artifact_table(
                        client,
                        run.info.run_id,
                        f"pecnet/{tier_name}/evaluation/{ticker}_regression.json",
                    )
                )
                long_frames.append(
                    self._read_artifact_table(
                        client,
                        run.info.run_id,
                        f"pecnet/{tier_name}/evaluation/{ticker}_long_direction.json",
                    )
                )
                ticker_metadata[ticker] = self._run_metadata(run)
            except Exception:
                LOGGER.warning(
                    "Could not load PECNet MLflow artifacts for %s %s run_id=%s",
                    tier_name,
                    ticker,
                    run.info.run_id,
                    exc_info=True,
                )

        predictions = self._concat(prediction_frames)
        if predictions.empty:
            return None

        regression_metrics = self._concat(regression_frames)
        long_metrics = self._concat(long_frames)
        metadata = {
            "run_count": len(ticker_metadata),
            "tickers": sorted(ticker_metadata),
            "ticker_runs": ticker_metadata,
            "row_count": len(predictions),
        }
        return (
            self._with_source_columns(predictions, tier_name, "pecnet"),
            self._with_source_columns(regression_metrics, tier_name, "pecnet"),
            self._with_source_columns(long_metrics, tier_name, "pecnet"),
            metadata,
        )

    @staticmethod
    def _family_run_name(tier_name: str, model_family: str) -> str:
        suffix = "automlforecast" if model_family == "mlforecast" else model_family
        return f"stock-close-{tier_name}-{suffix}"

    @staticmethod
    def _latest_run_by_name(
        client: MlflowClient,
        tier_name: str,
        run_name: str,
    ) -> Run | None:
        experiment = client.get_experiment_by_name(tier_experiment_name(tier_name))
        if experiment is None:
            return None
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=(
                "attributes.status = 'FINISHED' "
                f"and tags.mlflow.runName = '{run_name}'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        return runs[0] if runs else None

    @staticmethod
    def _latest_pecnet_ticker_runs(
        client: MlflowClient,
        tier_name: str,
    ) -> dict[str, Run]:
        experiment = client.get_experiment_by_name(tier_experiment_name(tier_name))
        if experiment is None:
            return {}
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=500,
        )
        prefix = f"pecnet-{tier_name}-"
        latest: dict[str, Run] = {}
        for run in runs:
            run_name = run.data.tags.get("mlflow.runName", "")
            if not run_name.startswith(prefix):
                continue
            ticker = run.data.params.get("ticker") or run_name.removeprefix(prefix)
            ticker = str(ticker)
            latest.setdefault(ticker, run)
        return latest

    @staticmethod
    def _read_artifact_table(
        client: MlflowClient,
        run_id: str,
        artifact_path: str,
    ) -> pd.DataFrame:
        with tempfile.TemporaryDirectory() as tmp_dir:
            local_path = Path(client.download_artifacts(run_id, artifact_path, tmp_dir))
            with local_path.open(encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
        if isinstance(payload, dict) and {"columns", "data"}.issubset(payload):
            return pd.DataFrame(payload["data"], columns=payload["columns"])
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return pd.DataFrame(payload["data"])
        if isinstance(payload, dict) and all(
            isinstance(value, list) for value in payload.values()
        ):
            return pd.DataFrame(payload)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        return pd.DataFrame()

    @staticmethod
    def _with_source_columns(
        frame: pd.DataFrame,
        tier_name: str,
        model_family: str,
    ) -> pd.DataFrame:
        if frame.empty:
            return frame
        enriched = frame.copy()
        enriched["tier"] = tier_name
        enriched["model_family"] = model_family
        return enriched

    @staticmethod
    def _run_metadata(run: Run, row_count: int | None = None) -> dict[str, Any]:
        metadata = {
            "run_id": run.info.run_id,
            "run_name": run.data.tags.get("mlflow.runName"),
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
        }
        if row_count is not None:
            metadata["row_count"] = row_count
        return metadata

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
        store = LightweightArtifactStore()
        store.save_metrics(
            regression_metrics,
            f"{self.artifact_prefix}/metrics/regression_metrics.csv",
        )
        store.save_metrics(
            long_metrics,
            f"{self.artifact_prefix}/metrics/long_direction_metrics.csv",
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
                RootModelPerformanceEvaluator._safe_metric_component(row["tier"]),
                RootModelPerformanceEvaluator._safe_metric_component(row["model_family"]),
                RootModelPerformanceEvaluator._safe_metric_component(row["unique_id"]),
                RootModelPerformanceEvaluator._safe_metric_component(row["model"]),
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
        leading_columns = ["tier", "model_family", "unique_id", "model"]
        remaining_columns = [
            column for column in metrics.columns if column not in leading_columns
        ]
        return metrics[leading_columns + remaining_columns]



    @staticmethod
    def _safe_metric_component(value: Any) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"
