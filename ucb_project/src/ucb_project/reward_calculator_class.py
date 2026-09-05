from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import numpy as np


@dataclass(slots=True)
class RewardCalculator:
    """Converts a realized forecast error into a UCB reward."""

    SUPPORTED_MODES: ClassVar[tuple[str, ...]] = (
        "inverse_mape",
        "negative_mape",
        "negative_mae",
        "negative_rmse",
        "negative_squared_error",
        "direction",
        "hybrid",
    )

    reward_mode: str = "negative_rmse"
    direction_weight: float = 0.0

    def score(
        self,
        *,
        actual: float,
        prediction: float,
        previous_actual: float | None = None,
    ) -> dict[str, float | bool | None]:
        actual_value = float(actual)
        prediction_value = float(prediction)
        y_true = np.array([actual_value], dtype=float)
        y_pred = np.array([prediction_value], dtype=float)
        abs_error = float(mean_absolute_error(y_true, y_pred))
        squared_error = float(mean_squared_error(y_true, y_pred))
        rmse = float(np.sqrt(squared_error))
        abs_pct_error = float(mean_absolute_percentage_error(y_true, y_pred))
        direction_correct = self._direction_correct(actual_value, prediction_value, previous_actual)
        loss_name = self._loss_name()
        loss = self._loss_value(
            abs_error=abs_error,
            rmse=rmse,
            squared_error=squared_error,
            abs_pct_error=abs_pct_error,
        )

        if self.reward_mode == "inverse_mape":
            reward = 1.0 / (1.0 + abs_pct_error)
        elif self.reward_mode == "negative_mape":
            reward = -abs_pct_error
        elif self.reward_mode == "negative_mae":
            reward = -abs_error
        elif self.reward_mode == "negative_rmse":
            reward = -rmse
        elif self.reward_mode == "negative_squared_error":
            reward = -squared_error
        elif self.reward_mode == "direction":
            reward = 1.0 if direction_correct else 0.0
        elif self.reward_mode == "hybrid":
            accuracy_reward = 1.0 / (1.0 + abs_pct_error)
            direction_reward = 1.0 if direction_correct else 0.0
            reward = (1.0 - self.direction_weight) * accuracy_reward
            reward += self.direction_weight * direction_reward
        else:
            raise ValueError(
                f"reward_mode must be one of {', '.join(self.SUPPORTED_MODES)}."
            )

        if not np.isfinite(reward):
            reward = 0.0
        return {
            "reward": float(reward),
            "loss": float(loss),
            "loss_name": loss_name,
            "abs_error": float(abs_error),
            "rmse": float(rmse),
            "squared_error": float(squared_error),
            "abs_pct_error": float(abs_pct_error),
            "direction_correct": direction_correct,
        }

    @staticmethod
    def _direction_correct(
        actual: float,
        prediction: float,
        previous_actual: float | None,
    ) -> bool | None:
        if previous_actual is None:
            return None
        actual_direction = np.sign(actual - float(previous_actual))
        predicted_direction = np.sign(prediction - float(previous_actual))
        return bool(actual_direction == predicted_direction)

    def _loss_name(self) -> str:
        if self.reward_mode in {"inverse_mape", "negative_mape", "hybrid"}:
            return "absolute_percentage_error"
        if self.reward_mode == "negative_mae":
            return "absolute_error"
        if self.reward_mode == "negative_rmse":
            return "root_squared_error"
        if self.reward_mode == "negative_squared_error":
            return "squared_error"
        if self.reward_mode == "direction":
            return "direction_error"
        return "forecast_error"

    def _loss_value(
        self,
        *,
        abs_error: float,
        rmse: float,
        squared_error: float,
        abs_pct_error: float,
    ) -> float:
        if self.reward_mode in {"inverse_mape", "negative_mape", "hybrid"}:
            return abs_pct_error
        if self.reward_mode == "negative_mae":
            return abs_error
        if self.reward_mode == "negative_rmse":
            return rmse
        if self.reward_mode == "negative_squared_error":
            return squared_error
        if self.reward_mode == "direction":
            return 1.0 if abs_error > 0 else 0.0
        return abs_pct_error

