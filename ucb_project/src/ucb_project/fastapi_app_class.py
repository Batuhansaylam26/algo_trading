from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .mlflow_forecast_repository_class import MlflowForecastRepository
from .pecnet_torch_predictor_class import PecnetTorchPredictor
from .selection_workflow_class import UcbSelectionWorkflow


@dataclass(slots=True)
class UcbFastApiApp:
    """Creates the FastAPI application for adaptive model selection."""

    tracking_uri: str | None = None
    experiment_prefix: str = "stock_close"

    def create(self):
        from fastapi import FastAPI

        repository = MlflowForecastRepository(
            tracking_uri=self.tracking_uri,
            experiment_prefix=self.experiment_prefix,
        )
        workflow = UcbSelectionWorkflow(repository=repository)
        pecnet_predictor = PecnetTorchPredictor(repository=repository)
        app = FastAPI(title="Stock Close UCB1-Tuned Selector", version="0.1.0")

        @app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @app.post("/backtest")
        def backtest(payload: dict[str, Any]) -> dict[str, Any]:
            return workflow.backtest(
                ticker=payload["ticker"],
                tiers=self._split(payload.get("tiers")),
                families=self._split(payload.get("families")),
                recency_mode=payload.get("recency_mode", "sliding_window"),
                window_size=int(payload.get("window_size", 60)),
                discount_factor=float(payload.get("discount_factor", 0.97)),
                exploration_scale=float(payload.get("exploration_scale", 1.0)),
                reward_mode=payload.get("reward_mode", "negative_rmse"),
                direction_weight=float(payload.get("direction_weight", 0.0)),
                date_policy=payload.get("date_policy", "common"),
                start_date=payload.get("start_date", "2026-02-03"),
            )

        @app.post("/select")
        def select(payload: dict[str, Any]) -> dict[str, Any]:
            result = backtest(payload)
            return {
                "ticker": result["ticker"],
                "winner": result["winner"],
                "candidate_count": result["candidate_count"],
                "event_count": result["event_count"],
            }

        @app.post("/predict-logged")
        def predict_logged(payload: dict[str, Any]) -> dict[str, Any]:
            return workflow.latest_logged_prediction(
                ticker=payload["ticker"],
                tiers=self._split(payload.get("tiers")),
                families=self._split(payload.get("families")),
                recency_mode=payload.get("recency_mode", "sliding_window"),
                window_size=int(payload.get("window_size", 60)),
                discount_factor=float(payload.get("discount_factor", 0.97)),
                exploration_scale=float(payload.get("exploration_scale", 1.0)),
                reward_mode=payload.get("reward_mode", "negative_rmse"),
                direction_weight=float(payload.get("direction_weight", 0.0)),
                date_policy=payload.get("date_policy", "common"),
                start_date=payload.get("start_date", "2026-02-03"),
            )

        @app.post("/register")
        def register(payload: dict[str, Any]) -> dict[str, Any]:
            return workflow.register_winner(
                ticker=payload["ticker"],
                tiers=self._split(payload.get("tiers")),
                families=self._split(payload.get("families")),
                recency_mode=payload.get("recency_mode", "sliding_window"),
                window_size=int(payload.get("window_size", 60)),
                discount_factor=float(payload.get("discount_factor", 0.97)),
                exploration_scale=float(payload.get("exploration_scale", 1.0)),
                reward_mode=payload.get("reward_mode", "negative_rmse"),
                direction_weight=float(payload.get("direction_weight", 0.0)),
                date_policy=payload.get("date_policy", "common"),
                start_date=payload.get("start_date", "2026-02-03"),
                min_mean_reward=float(payload.get("min_mean_reward", 0.98)),
            )

        @app.post("/pecnet/predict-from-artifact")
        def pecnet_predict(payload: dict[str, Any]) -> dict[str, Any]:
            return pecnet_predictor.predict_from_artifact(
                run_id=payload["run_id"],
                model_artifact_path=payload["model_artifact_path"],
                input_artifact_path=payload.get("input_artifact_path"),
                arrays=payload.get("arrays"),
                device=payload.get("device", "auto"),
            )

        return app

    @staticmethod
    def _split(value: str | list[str] | None) -> list[str] | None:
        if value is None or value == "":
            return None
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        return [item.strip() for item in str(value).split(",") if item.strip()]
