from __future__ import annotations

import json
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

try:
    from .environment_class import LocalMlflowEnvironment
except ImportError:
    from environment_class import LocalMlflowEnvironment


@dataclass(slots=True)
class MlflowForecastRepository:
    """Loads model forecasts and metadata from the project's MLflow runs."""

    DEFAULT_FAMILIES: ClassVar[tuple[str, ...]] = ("pecnet",)
    TIER_PATTERN: ClassVar[re.Pattern[str]] = re.compile(r"^tier\d+$")

    tracking_uri: str | None = None
    experiment_prefix: str = "stock_close"
    local_artifact_roots: list[Path] = field(default_factory=list)

    def __post_init__(self) -> None:
        LocalMlflowEnvironment.apply_defaults()
        self.tracking_uri = LocalMlflowEnvironment.normalize_service_url(
            self.tracking_uri,
            port=5001,
        )
        self._mlflow_module().set_tracking_uri(self.tracking_uri)
        if not self.local_artifact_roots:
            project_root = Path.cwd()
            self.local_artifact_roots = [
                project_root / "artifacts",
                project_root / "kedro_project" / "artifacts",
            ]

    @property
    def client(self):
        from mlflow.tracking import MlflowClient

        return MlflowClient(tracking_uri=self.tracking_uri)

    def discover_candidates(
        self,
        *,
        ticker: str | None = None,
        tiers: list[str] | None = None,
        families: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        selected_tiers = tiers if tiers is not None else self.available_tiers()
        selected_families = (
            families if families is not None else list(self.DEFAULT_FAMILIES)
        )
        candidates: list[dict[str, Any]] = []

        for tier in selected_tiers:
            for family in selected_families:
                if family == "pecnet":
                    candidates.extend(self._pecnet_candidates(tier=tier, ticker=ticker))
                else:
                    candidates.extend(
                        self._family_candidates(
                            tier=tier,
                            family=family,
                            ticker=ticker,
                        )
                    )
        return candidates

    def available_tiers(self) -> list[str]:
        try:
            experiments = self.client.search_experiments()
        except Exception:
            return [f"tier{index}" for index in range(1, 10)]

        prefix = f"{self.experiment_prefix}_"
        tiers = [
            experiment.name.removeprefix(prefix)
            for experiment in experiments
            if experiment.name.startswith(prefix)
        ]
        tiers = [tier for tier in tiers if self._is_tier_name(tier)]
        return sorted(tiers, key=self._tier_sort_key)

    def download_artifact(self, run_id: str, artifact_path: str) -> Path:
        temp_dir = tempfile.mkdtemp(prefix="ucb_mlflow_artifact_")
        return Path(self.client.download_artifacts(run_id, artifact_path, temp_dir))

    def load_artifact_table(self, run_id: str, artifact_path: str) -> pd.DataFrame:
        try:
            local_path = self.download_artifact(run_id, artifact_path)
        except Exception:
            return pd.DataFrame()
        return self._read_table(local_path)

    def _family_candidates(
        self,
        *,
        tier: str,
        family: str,
        ticker: str | None,
    ) -> list[dict[str, Any]]:
        run = self._latest_family_run(tier=tier, family=family)
        if run is None:
            return []

        predictions = self.load_artifact_table(
            run.info.run_id,
            f"{family}/{tier}/evaluation/predictions.json",
        )
        metrics = self.load_artifact_table(
            run.info.run_id,
            f"{family}/{tier}/evaluation/regression_metrics.json",
        )
        if predictions.empty:
            return []

        predictions = self._normalize_prediction_frame(predictions)
        if ticker:
            predictions = predictions[predictions["unique_id"].astype(str) == str(ticker)]
            metrics = self._filter_metrics(metrics, ticker=ticker)

        model_names = self._model_columns(predictions, metrics)
        return [
            self._candidate_payload(
                run_id=run.info.run_id,
                run_name=run.data.tags.get("mlflow.runName", ""),
                tier=tier,
                family=family,
                model=model_name,
                unique_id_value=unique_id,
                predictions=predictions[predictions["unique_id"].astype(str) == unique_id],
                metrics=self._filter_metrics(metrics, ticker=unique_id, model=model_name),
                registerable_model_uri=self._registerable_model_uri(run, model_name),
            )
            for unique_id in sorted(predictions["unique_id"].astype(str).unique())
            for model_name in model_names
            if model_name in predictions.columns
        ]

    def _pecnet_candidates(self, *, tier: str, ticker: str | None) -> list[dict[str, Any]]:
        runs = self._latest_pecnet_ticker_runs(tier=tier)
        if ticker:
            runs = {key: value for key, value in runs.items() if str(key) == str(ticker)}

        candidates: list[dict[str, Any]] = []
        for unique_id, run in runs.items():
            predictions = self.load_artifact_table(
                run.info.run_id,
                f"pecnet/{tier}/predictions/{unique_id}.json",
            )
            if predictions.empty:
                continue
            predictions = self._normalize_prediction_frame(predictions)
            metrics = self.load_artifact_table(
                run.info.run_id,
                f"pecnet/{tier}/evaluation/{unique_id}_regression.json",
            )
            model_uri = f"runs:/{run.info.run_id}/pecnet/{tier}/models/{self._safe_name(unique_id)}"
            candidates.append(
                self._candidate_payload(
                    run_id=run.info.run_id,
                    run_name=run.data.tags.get("mlflow.runName", ""),
                    tier=tier,
                    family="pecnet",
                    model="PECNet",
                    unique_id_value=unique_id,
                    predictions=predictions[predictions["unique_id"].astype(str) == str(unique_id)],
                    metrics=metrics,
                    registerable_model_uri=None,
                    raw_model_uri=model_uri,
                )
            )
        return candidates

    def _latest_family_run(self, *, tier: str, family: str):
        experiment = self.client.get_experiment_by_name(f"{self.experiment_prefix}_{tier}")
        if experiment is None:
            return None

        run_name = self._family_run_name(tier=tier, family=family)
        runs = self.client.search_runs(
            [experiment.experiment_id],
            filter_string=(
                "attributes.status = 'FINISHED' "
                f"and tags.mlflow.runName = '{run_name}'"
            ),
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        return runs[0] if runs else None

    def _latest_pecnet_ticker_runs(self, *, tier: str) -> dict[str, Any]:
        experiment = self.client.get_experiment_by_name(f"{self.experiment_prefix}_{tier}")
        if experiment is None:
            return {}

        runs = self.client.search_runs(
            [experiment.experiment_id],
            filter_string="attributes.status = 'FINISHED'",
            order_by=["attributes.start_time DESC"],
            max_results=500,
        )
        prefix = f"pecnet-{tier}-"
        latest: dict[str, Any] = {}
        for run in runs:
            run_name = run.data.tags.get("mlflow.runName", "")
            if not run_name.startswith(prefix):
                continue
            ticker = str(run.data.params.get("ticker") or run_name.removeprefix(prefix))
            latest.setdefault(ticker, run)
        return latest

    @staticmethod
    def _candidate_payload(
        *,
        run_id: str,
        run_name: str,
        tier: str,
        family: str,
        model: str,
        unique_id_value: str,
        predictions: pd.DataFrame,
        metrics: pd.DataFrame,
        registerable_model_uri: str | None,
        raw_model_uri: str | None = None,
    ) -> dict[str, Any]:
        return {
            "arm_id": f"{unique_id_value}|{tier}|{family}|{model}",
            "run_id": run_id,
            "run_name": run_name,
            "tier": tier,
            "family": family,
            "model": model,
            "unique_id": unique_id_value,
            "prediction_column": model,
            "predictions": predictions.copy(),
            "metrics": metrics.copy(),
            "registerable_model_uri": registerable_model_uri,
            "raw_model_uri": raw_model_uri,
        }

    @staticmethod
    def _normalize_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
        output = frame.copy()
        if "date" in output.columns and "ds" not in output.columns:
            output = output.rename(columns={"date": "ds"})
        if "symbol" in output.columns and "unique_id" not in output.columns:
            output = output.rename(columns={"symbol": "unique_id"})
        output["ds"] = pd.to_datetime(output["ds"], errors="coerce")
        output["unique_id"] = output["unique_id"].astype(str)
        return output.dropna(subset=["unique_id", "ds", "y"])

    @staticmethod
    def _model_columns(predictions: pd.DataFrame, metrics: pd.DataFrame) -> list[str]:
        excluded = {
            "unique_id",
            "ds",
            "y",
            "_tier",
            "_family",
            "tier",
            "family",
            "source",
            "run_id",
        }
        metric_models = []
        if not metrics.empty and "model" in metrics.columns:
            metric_models = [str(model) for model in metrics["model"].dropna().unique()]
        candidates = [
            column
            for column in predictions.columns
            if column not in excluded
            and "-lo-" not in column
            and "-hi-" not in column
            and not column.endswith("_lower")
            and not column.endswith("_upper")
        ]
        if metric_models:
            ordered = [model for model in metric_models if model in candidates]
            return ordered or candidates
        return candidates

    @staticmethod
    def _filter_metrics(
        metrics: pd.DataFrame,
        *,
        ticker: str,
        model: str | None = None,
    ) -> pd.DataFrame:
        if metrics.empty:
            return metrics
        output = metrics.copy()
        if "unique_id" in output.columns:
            output = output[output["unique_id"].astype(str) == str(ticker)]
        if model is not None and "model" in output.columns:
            output = output[output["model"].astype(str) == str(model)]
        return output.reset_index(drop=True)

    @staticmethod
    def _family_run_name(*, tier: str, family: str) -> str:
        suffix = "automlforecast" if family == "mlforecast" else family
        return f"stock-close-{tier}-{suffix}"

    @staticmethod
    def _registerable_model_uri(run: Any, model_name: str) -> str | None:
        history = run.data.tags.get("mlflow.log-model.history")
        if not history:
            return None
        try:
            entries = json.loads(history)
        except json.JSONDecodeError:
            return None
        for entry in entries:
            artifact_path = entry.get("artifact_path")
            if not artifact_path:
                continue
            if model_name.lower() in artifact_path.lower():
                return f"runs:/{run.info.run_id}/{artifact_path}"
        if entries and entries[0].get("artifact_path"):
            return f"runs:/{run.info.run_id}/{entries[0]['artifact_path']}"
        return None

    @staticmethod
    def _read_table(path: Path) -> pd.DataFrame:
        if path.is_dir():
            json_files = sorted(path.glob("*.json"))
            csv_files = sorted(path.glob("*.csv"))
            if json_files:
                path = json_files[0]
            elif csv_files:
                path = csv_files[0]
        if path.suffix == ".csv":
            return pd.read_csv(path)
        with path.open(encoding="utf-8") as file_obj:
            payload = json.load(file_obj)
        if isinstance(payload, dict) and {"columns", "data"}.issubset(payload):
            return pd.DataFrame(payload["data"], columns=payload["columns"])
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            return pd.DataFrame(payload["data"])
        if isinstance(payload, dict) and all(isinstance(value, list) for value in payload.values()):
            return pd.DataFrame(payload)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        return pd.DataFrame()

    @staticmethod
    def _tier_sort_key(tier: str) -> tuple[int, str]:
        number = "".join(character for character in tier if character.isdigit())
        return (int(number) if number else 9999, tier)

    @classmethod
    def _is_tier_name(cls, tier: str) -> bool:
        return bool(cls.TIER_PATTERN.fullmatch(str(tier)))

    @staticmethod
    def _safe_name(value: str) -> str:
        return str(value).replace(".", "_").replace("/", "_").replace(" ", "_")

    @staticmethod
    def _mlflow_module():
        import mlflow

        return mlflow
