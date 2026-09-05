from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RegistryPromoter:
    """Registers a selected winner when the artifact is a registerable MLflow model."""

    registered_model_prefix: str = "stock_close_ucb"
    min_mean_reward: float = 0.98
    experiment_name: str = "stock_close_ucb_selection"

    def promote_if_successful(self, *, backtest_result: dict[str, Any]) -> dict[str, Any]:
        mlflow = self._mlflow_module()
        winner = backtest_result["winner"]
        model_uri = winner.get("registerable_model_uri")
        mean_reward = float(winner.get("mean_reward", 0.0))
        model_name = self._registered_name(winner)

        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run(run_name=f"ucb-selection-{winner['unique_id']}") as run:
            mlflow.log_params(
                {
                    "winner_arm_id": winner["arm_id"],
                    "winner_tier": winner["tier"],
                    "winner_family": winner["family"],
                    "winner_model": winner["model"],
                    "winner_unique_id": winner["unique_id"],
                    "min_mean_reward": self.min_mean_reward,
                }
            )
            mlflow.log_metric("winner_mean_reward", mean_reward)
            mlflow.log_dict(backtest_result, "ucb/selection_result.json")

            if mean_reward < self.min_mean_reward:
                return {
                    "registered": False,
                    "reason": "winner_mean_reward_below_threshold",
                    "selection_run_id": run.info.run_id,
                    "winner": winner,
                }
            if not model_uri:
                return {
                    "registered": False,
                    "reason": "winner_has_no_registerable_mlflow_model_uri",
                    "selection_run_id": run.info.run_id,
                    "winner": winner,
                }

            registered = mlflow.register_model(model_uri=model_uri, name=model_name)
            return {
                "registered": True,
                "registered_model_name": model_name,
                "registered_model_version": registered.version,
                "selection_run_id": run.info.run_id,
                "winner": winner,
            }

    def _registered_name(self, winner: dict[str, Any]) -> str:
        pieces = [
            self.registered_model_prefix,
            winner["unique_id"],
            winner["tier"],
            winner["family"],
            winner["model"],
        ]
        return "_".join(self._safe_piece(piece) for piece in pieces)

    @staticmethod
    def _safe_piece(value: object) -> str:
        return str(value).replace(".", "_").replace("/", "_").replace(" ", "_")

    @staticmethod
    def _mlflow_module():
        import mlflow

        return mlflow
