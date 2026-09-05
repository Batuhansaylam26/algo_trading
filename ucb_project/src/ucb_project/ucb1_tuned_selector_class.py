from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable


@dataclass(slots=True)
class UCB1TunedSelector:
    """Recency-aware UCB1-Tuned selector for non-stationary rewards."""

    arm_ids: list[str]
    window_size: int = 5
    recency_mode: str = "sliding_window"
    discount_factor: float = 0.97
    exploration_scale: float = 1.0
    reward_history: dict[str, list[tuple[int, float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.reward_history = {arm_id: [] for arm_id in self.arm_ids}
        if self.window_size <= 0:
            raise ValueError("window_size must be positive.")
        if not 0 < self.discount_factor <= 1:
            raise ValueError("discount_factor must be in (0, 1].")
        if self.recency_mode not in {"sliding_window", "discounted", "all"}:
            raise ValueError("recency_mode must be sliding_window, discounted, or all.")

    def select_arm(
        self,
        *,
        step: int,
        available_arm_ids: Iterable[str] | None = None,
    ) -> str:
        available = list(self.arm_ids if available_arm_ids is None else available_arm_ids)
        if not available:
            raise ValueError("No available arms to select from.")

        for arm_id in available:
            if not self.reward_history.get(arm_id):
                return arm_id

        scores = {arm_id: self.score(arm_id, step=step) for arm_id in available}
        return max(scores, key=scores.get)

    def update(self, arm_id: str, reward: float, *, step: int) -> None:
        if arm_id not in self.reward_history:
            self.reward_history[arm_id] = []
        self.reward_history[arm_id].append((int(step), float(reward)))

    def best_arm(self, *, step: int) -> str:
        means = {
            arm_id: self._stats(arm_id, step=step)["mean_reward"]
            for arm_id in self.arm_ids
            if self._stats(arm_id, step=step)["n"] > 0
        }
        if not means:
            return self.arm_ids[0]
        return max(means, key=means.get)

    def score(self, arm_id: str, *, step: int) -> float:
        return self.score_components(arm_id, step=step)["ucb_score"]

    def score_components(self, arm_id: str, *, step: int) -> dict[str, float | int]:
        stats = self._stats(arm_id, step=step)
        n = int(stats["n"])
        if n <= 0:
            return {
                **stats,
                "exploitation_reward": stats["mean_reward"],
                "tuned_variance": 0.0,
                "exploration_bonus": float("inf"),
                "ucb_score": float("inf"),
            }

        log_t = math.log(max(step, 2))
        tuned_variance = float(stats["variance"]) + math.sqrt((2.0 * log_t) / n)
        raw_exploration = math.sqrt((log_t / n) * min(0.25, tuned_variance))
        exploration_bonus = self.exploration_scale * raw_exploration
        exploitation_reward = float(stats["mean_reward"])
        return {
            **stats,
            "exploitation_reward": exploitation_reward,
            "tuned_variance": tuned_variance,
            "exploration_bonus": exploration_bonus,
            "ucb_score": exploitation_reward + exploration_bonus,
        }

    def summary(self, *, step: int) -> list[dict[str, float | int | str]]:
        rows = []
        for arm_id in self.arm_ids:
            stats = self.score_components(arm_id, step=step)
            rows.append(
                {
                    "arm_id": arm_id,
                    "n": stats["n"],
                    "all_time_n": len(self.reward_history.get(arm_id, [])),
                    "mean_reward": stats["mean_reward"],
                    "variance": stats["variance"],
                    "exploitation_reward": stats["exploitation_reward"],
                    "exploration_bonus": stats["exploration_bonus"],
                    "tuned_variance": stats["tuned_variance"],
                    "ucb_score": stats["ucb_score"],
                }
            )
        return sorted(rows, key=lambda row: float(row["ucb_score"]), reverse=True)

    def _stats(self, arm_id: str, *, step: int) -> dict[str, float | int]:
        rewards = self._recent_rewards(arm_id, step=step)
        if not rewards:
            return {"n": 0, "mean_reward": 0.0, "variance": 0.0}

        if self.recency_mode == "discounted":
            return self._discounted_stats(arm_id, step=step)

        n = len(rewards)
        mean_reward = sum(rewards) / n
        second_moment = sum(reward * reward for reward in rewards) / n
        variance = max(0.0, second_moment - mean_reward * mean_reward)
        return {"n": n, "mean_reward": mean_reward, "variance": variance}

    def _recent_rewards(self, arm_id: str, *, step: int) -> list[float]:
        rows = self.reward_history.get(arm_id, [])
        if self.recency_mode == "all":
            return [reward for _, reward in rows]
        if self.recency_mode == "discounted":
            return [reward for _, reward in rows]

        recent = [
            reward
            for reward_step, reward in rows
            if int(step) - int(reward_step) < self.window_size
        ]
        return recent or [reward for _, reward in rows[-1:]]

    def _discounted_stats(self, arm_id: str, *, step: int) -> dict[str, float | int]:
        rows = self.reward_history.get(arm_id, [])
        weights = [self.discount_factor ** max(0, int(step) - reward_step) for reward_step, _ in rows]
        weight_sum = sum(weights)
        if weight_sum <= 0:
            return {"n": 0, "mean_reward": 0.0, "variance": 0.0}

        rewards = [reward for _, reward in rows]
        mean_reward = sum(weight * reward for weight, reward in zip(weights, rewards)) / weight_sum
        second_moment = sum(weight * reward * reward for weight, reward in zip(weights, rewards)) / weight_sum
        variance = max(0.0, second_moment - mean_reward * mean_reward)
        effective_n = max(1, int(round(weight_sum)))
        return {
            "n": effective_n,
            "mean_reward": mean_reward,
            "variance": variance,
        }
