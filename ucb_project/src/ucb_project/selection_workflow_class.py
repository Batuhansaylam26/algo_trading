from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .forecast_bandit_env_class import ForecastBanditEnv
from .mlflow_forecast_repository_class import MlflowForecastRepository
from .registry_promoter_class import RegistryPromoter
from .reward_calculator_class import RewardCalculator
from .ucb1_tuned_selector_class import UCB1TunedSelector


@dataclass(slots=True)
class UcbSelectionWorkflow:
    """Replays logged forecasts and selects model configurations with UCB1-Tuned."""

    repository: MlflowForecastRepository

    def backtest(
        self,
        *,
        ticker: str,
        tiers: list[str] | None = None,
        families: list[str] | None = None,
        recency_mode: str = "sliding_window",
        window_size: int = 60,
        discount_factor: float = 0.97,
        exploration_scale: float = 1.0,
        reward_mode: str = "negative_rmse",
        direction_weight: float = 0.0,
        date_policy: str = "common",
        start_date: str | None = "2026-02-03",
    ) -> dict[str, Any]:
        candidates = self.repository.discover_candidates(
            ticker=ticker,
            tiers=tiers,
            families=families,
        )
        candidates = [candidate for candidate in candidates if not candidate["predictions"].empty]
        if not candidates:
            raise ValueError(f"No forecast candidates found for ticker={ticker}.")

        arm_ids = [candidate["arm_id"] for candidate in candidates]
        selector = UCB1TunedSelector(
            arm_ids=arm_ids,
            window_size=window_size,
            recency_mode=recency_mode,
            discount_factor=discount_factor,
            exploration_scale=exploration_scale,
        )
        reward_calculator = RewardCalculator(
            reward_mode=reward_mode,
            direction_weight=direction_weight,
        )
        rows = self._replay_forecasts(
            candidates=candidates,
            selector=selector,
            reward_calculator=reward_calculator,
            date_policy=date_policy,
            start_date=start_date,
        )
        summary = pd.DataFrame(selector.summary(step=max(len(rows), 1)))
        metrics = self._candidate_metrics(
            candidates,
            reward_calculator,
            start_date=start_date,
        )
        winner = self._winner_payload(candidates, selector, summary)

        return {
            "ticker": ticker,
            "tiers": tiers,
            "families": families,
            "recency_mode": recency_mode,
            "window_size": window_size,
            "discount_factor": discount_factor,
            "exploration_scale": exploration_scale,
            "reward_mode": reward_mode,
            "date_policy": date_policy,
            "start_date": start_date,
            "candidate_count": len(candidates),
            "event_count": len(rows),
            "winner": winner,
            "selection_events": rows,
            "ucb_summary": summary.to_dict(orient="records"),
            "candidate_metrics": metrics.to_dict(orient="records"),
        }

    def register_winner(
        self,
        *,
        ticker: str,
        tiers: list[str] | None = None,
        families: list[str] | None = None,
        min_mean_reward: float = 0.98,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = self.backtest(ticker=ticker, tiers=tiers, families=families, **kwargs)
        promoter = RegistryPromoter(min_mean_reward=min_mean_reward)
        return promoter.promote_if_successful(backtest_result=result)

    def latest_logged_prediction(
        self,
        *,
        ticker: str,
        tiers: list[str] | None = None,
        families: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result = self.backtest(ticker=ticker, tiers=tiers, families=families, **kwargs)
        winner = result["winner"]
        candidates = self.repository.discover_candidates(
            ticker=ticker,
            tiers=tiers,
            families=families,
        )
        selected = next(
            candidate for candidate in candidates if candidate["arm_id"] == winner["arm_id"]
        )
        frame = selected["predictions"].sort_values("ds").tail(1)
        if frame.empty:
            raise ValueError(f"No logged prediction row found for {winner['arm_id']}.")
        row = frame.iloc[0]
        return {
            "ticker": ticker,
            "winner": winner,
            "latest_logged_prediction": {
                "ds": str(pd.Timestamp(row["ds"]).date()),
                "actual": float(row["y"]),
                "prediction": float(row[selected["prediction_column"]]),
            },
        }

    def _replay_forecasts(
        self,
        *,
        candidates: list[dict[str, Any]],
        selector: UCB1TunedSelector,
        reward_calculator: RewardCalculator,
        date_policy: str,
        start_date: str | None,
    ) -> list[dict[str, Any]]:
        env = ForecastBanditEnv(
            candidates=candidates,
            reward_calculator=reward_calculator,
            date_policy=date_policy,
            start_date=start_date,
        )
        env.reset()
        rows: list[dict[str, Any]] = []
        terminated = False
        truncated = False

        while not terminated and not truncated:
            available_arm_ids = env.available_arm_ids()
            if not available_arm_ids:
                break
            selected_arm = selector.select_arm(
                step=env.current_step,
                available_arm_ids=available_arm_ids,
            )
            action = env.action_for_arm(selected_arm)
            _, reward, terminated, truncated, info = env.step(action)
            selector.update(selected_arm, float(reward), step=int(info["step"]))
            rows.append(
                {
                    key: value
                    for key, value in info.items()
                    if key not in {"timestamp", "action"}
                }
            )
        return rows

    @staticmethod
    def _evaluation_dates(candidates: list[dict[str, Any]], *, date_policy: str) -> list[pd.Timestamp]:
        date_sets = [
            set(pd.to_datetime(candidate["predictions"]["ds"]).dropna())
            for candidate in candidates
        ]
        if not date_sets:
            return []
        if date_policy == "common":
            dates = set.intersection(*date_sets)
        elif date_policy == "union":
            dates = set.union(*date_sets)
        else:
            raise ValueError("date_policy must be common or union.")
        return sorted(pd.Timestamp(date) for date in dates)

    @staticmethod
    def _row_for_date(frame: pd.DataFrame, date_value: pd.Timestamp) -> pd.Series | None:
        rows = frame[pd.to_datetime(frame["ds"]) == pd.Timestamp(date_value)]
        if rows.empty:
            return None
        return rows.iloc[0]

    @staticmethod
    def _candidate_metrics(
        candidates: list[dict[str, Any]],
        reward_calculator: RewardCalculator,
        start_date: str | None = "2026-02-03",
    ) -> pd.DataFrame:
        rows = []
        for candidate in candidates:
            frame = candidate["predictions"].sort_values("ds").copy()
            if start_date is not None:
                frame = frame[pd.to_datetime(frame["ds"]) >= pd.Timestamp(start_date)]
            pred_col = candidate["prediction_column"]
            valid = frame[["y", pred_col]].dropna()
            if valid.empty:
                continue
            actual = valid["y"].astype(float)
            prediction = valid[pred_col].astype(float)
            error = prediction - actual
            abs_error = error.abs()
            abs_pct = abs_error / actual.abs().clip(lower=1e-12)
            rewards = []
            losses = []
            previous_actual = None
            for actual_value, prediction_value in zip(actual, prediction):
                scored = reward_calculator.score(
                    actual=float(actual_value),
                    prediction=float(prediction_value),
                    previous_actual=previous_actual,
                )
                rewards.append(float(scored["reward"]))
                losses.append(float(scored["loss"]))
                previous_actual = float(actual_value)
            ss_res = float(np.square(error).sum())
            ss_tot = float(np.square(actual - actual.mean()).sum())
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
            rows.append(
                {
                    "arm_id": candidate["arm_id"],
                    "unique_id": candidate["unique_id"],
                    "tier": candidate["tier"],
                    "family": candidate["family"],
                    "model": candidate["model"],
                    "rows": int(len(valid)),
                    "mae": float(abs_error.mean()),
                    "rmse": float(np.sqrt(np.mean(np.square(error)))),
                    "mape": float(abs_pct.mean() * 100.0),
                    "r2": r2,
                    "mean_loss": float(np.mean(losses)),
                    "mean_reward": float(np.mean(rewards)),
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).sort_values(
            ["mean_reward", "rmse"],
            ascending=[False, True],
        )

    @staticmethod
    def _winner_payload(
        candidates: list[dict[str, Any]],
        selector: UCB1TunedSelector,
        summary: pd.DataFrame,
    ) -> dict[str, Any]:
        if summary.empty:
            winner_arm = selector.best_arm(step=1)
        else:
            selected_summary = summary[summary["n"].astype(int) > 0]
            if selected_summary.empty:
                winner_arm = selector.best_arm(step=1)
            else:
                winner_arm = str(
                    selected_summary.sort_values(
                        ["mean_reward", "ucb_score"],
                        ascending=[False, False],
                    ).iloc[0]["arm_id"]
                )
        candidate = next(candidate for candidate in candidates if candidate["arm_id"] == winner_arm)
        winner_summary = (
            summary[summary["arm_id"] == winner_arm].iloc[0].to_dict()
            if not summary.empty
            else {}
        )
        return {
            "arm_id": winner_arm,
            "unique_id": candidate["unique_id"],
            "tier": candidate["tier"],
            "family": candidate["family"],
            "model": candidate["model"],
            "run_id": candidate["run_id"],
            "run_name": candidate["run_name"],
            "registerable_model_uri": candidate.get("registerable_model_uri"),
            "raw_model_uri": candidate.get("raw_model_uri"),
            **winner_summary,
        }
