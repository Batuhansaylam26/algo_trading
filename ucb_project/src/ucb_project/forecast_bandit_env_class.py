from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - dependency guard for import-only paths
    gym = None
    spaces = None

from .reward_calculator_class import RewardCalculator


@dataclass(slots=True)
class ForecastBanditEnv(gym.Env if gym is not None else object):
    """Gymnasium environment that reveals one forecast timestamp per step."""

    candidates: list[dict[str, Any]]
    reward_calculator: RewardCalculator
    date_policy: str = "common"
    start_date: str | pd.Timestamp | None = "2026-02-03"
    dates: list[pd.Timestamp] = field(init=False)
    action_space: Any = field(init=False)
    observation_space: Any = field(init=False)
    current_index: int = field(default=0, init=False)
    previous_actual: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if gym is None or spaces is None:
            raise ImportError(
                "gymnasium is required for ForecastBanditEnv. "
                "Install it with `pip install -e ucb_project`."
            )
        if not self.candidates:
            raise ValueError("ForecastBanditEnv requires at least one candidate arm.")

        self.dates = self._evaluation_dates()
        if not self.dates:
            raise ValueError("No evaluation dates found for the selected candidates.")

        arm_count = len(self.candidates)
        self.action_space = spaces.Discrete(arm_count)
        self.observation_space = spaces.Dict(
            {
                "step": spaces.Box(0, np.iinfo(np.int32).max, shape=(1,), dtype=np.int32),
                "timestamp": spaces.Box(
                    0,
                    np.iinfo(np.int64).max,
                    shape=(1,),
                    dtype=np.int64,
                ),
                "available_actions": spaces.MultiBinary(arm_count),
                "previous_actual": spaces.Box(
                    -np.inf,
                    np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

    @property
    def current_step(self) -> int:
        return self.current_index + 1

    @property
    def current_date(self) -> pd.Timestamp | None:
        if self.current_index >= len(self.dates):
            return None
        return self.dates[self.current_index]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self.current_index = 0
        self.previous_actual = None
        return self._observation(), self._info_without_reward()

    def step(
        self,
        action: int,
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        if self.current_date is None:
            raise RuntimeError("Cannot step after the environment has terminated.")
        if not self.action_space.contains(action):
            raise ValueError(f"Action {action} is outside the action space.")

        selected = self.candidates[int(action)]
        date_value = self.current_date
        prediction_row = self._row_for_date(selected["predictions"], date_value)
        if prediction_row is None:
            raise ValueError(
                f"Arm {selected['arm_id']} has no forecast for {date_value.date()}."
            )

        prediction = float(prediction_row[selected["prediction_column"]])
        actual = float(prediction_row["y"])
        scored = self.reward_calculator.score(
            actual=actual,
            prediction=prediction,
            previous_actual=self.previous_actual,
        )
        reward = float(scored["reward"])
        info = {
            "step": self.current_step,
            "ds": str(date_value.date()),
            "timestamp": pd.Timestamp(date_value),
            "action": int(action),
            "arm_id": selected["arm_id"],
            "unique_id": selected["unique_id"],
            "tier": selected["tier"],
            "family": selected["family"],
            "model": selected["model"],
            "actual": actual,
            "prediction": prediction,
            **scored,
        }

        self.previous_actual = actual
        self.current_index += 1
        terminated = self.current_index >= len(self.dates)
        return self._observation(), reward, terminated, False, info

    def available_arm_ids(self) -> list[str]:
        date_value = self.current_date
        if date_value is None:
            return []
        return [
            candidate["arm_id"]
            for candidate in self.candidates
            if self._row_for_date(candidate["predictions"], date_value) is not None
        ]

    def action_for_arm(self, arm_id: str) -> int:
        for index, candidate in enumerate(self.candidates):
            if candidate["arm_id"] == arm_id:
                return index
        raise ValueError(f"Unknown arm_id: {arm_id}")

    def _observation(self) -> dict[str, np.ndarray]:
        date_value = self.current_date
        timestamp = 0 if date_value is None else int(pd.Timestamp(date_value).timestamp())
        previous_actual = np.nan if self.previous_actual is None else self.previous_actual
        return {
            "step": np.array([self.current_step], dtype=np.int32),
            "timestamp": np.array([timestamp], dtype=np.int64),
            "available_actions": self._available_action_mask(),
            "previous_actual": np.array([previous_actual], dtype=np.float32),
        }

    def _available_action_mask(self) -> np.ndarray:
        available = set(self.available_arm_ids())
        return np.array(
            [candidate["arm_id"] in available for candidate in self.candidates],
            dtype=np.int8,
        )

    def _info_without_reward(self) -> dict[str, Any]:
        date_value = self.current_date
        return {
            "step": self.current_step,
            "ds": None if date_value is None else str(date_value.date()),
            "available_arm_ids": self.available_arm_ids(),
        }

    def _evaluation_dates(self) -> list[pd.Timestamp]:
        date_sets = [
            set(pd.to_datetime(candidate["predictions"]["ds"]).dropna())
            for candidate in self.candidates
        ]
        if self.date_policy == "common":
            dates = set.intersection(*date_sets)
        elif self.date_policy == "union":
            dates = set.union(*date_sets)
        else:
            raise ValueError("date_policy must be common or union.")

        output = sorted(pd.Timestamp(date) for date in dates)
        if self.start_date is None:
            return output

        start = pd.Timestamp(self.start_date)
        return [date for date in output if date >= start]

    @staticmethod
    def _row_for_date(frame: pd.DataFrame, date_value: pd.Timestamp) -> pd.Series | None:
        rows = frame[pd.to_datetime(frame["ds"]) == pd.Timestamp(date_value)]
        if rows.empty:
            return None
        return rows.iloc[0]
