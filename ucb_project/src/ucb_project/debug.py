from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
from pathlib import Path
from typing import Any


PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
for path in [str(SRC_DIR), str(PACKAGE_DIR)]:
    if path not in sys.path:
        sys.path.insert(0, path)

import pandas as pd
from ucb_project.forecast_bandit_env_class import ForecastBanditEnv
from ucb_project.mlflow_forecast_repository_class import MlflowForecastRepository
from ucb_project.reward_calculator_class import RewardCalculator
from ucb_project.ucb1_tuned_selector_class import UCB1TunedSelector




def main() -> None:
    args = _parse_args()
    try:
        result = run_debug(args)
    except Exception as exc:
        print("\n" + "=" * 100)
        print("DEBUG FAILED")
        print("=" * 100)
        print(f"{type(exc).__name__}: {exc}")
        print("\nFull traceback:")
        traceback.print_exc()
        print("\nQuick checks:")
        print("- Is MLflow running at the printed MLFLOW_TRACKING_URI?")
        print("- Are MinIO credentials exported or available from defaults?")
        print("- Do the selected tiers/families have FINISHED MLflow runs?")
        raise

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\nSaved debug result JSON: {output_path}")


def run_debug(args: argparse.Namespace) -> dict[str, Any]:
    _print_header("UCB DEBUG CONFIG")
    _print_environment(args)

    repository = MlflowForecastRepository(
        tracking_uri=args.tracking_uri,
        experiment_prefix=args.experiment_prefix,
    )

    tiers = _string_to_list(args.tiers)
    families = _string_to_list(args.families)
    print(f"Resolved tracking_uri: {repository.tracking_uri}")
    print(f"Requested ticker     : {args.ticker}")
    print(f"Requested tiers      : {tiers or 'auto from MLflow experiments'}")
    print(f"Requested families   : {families or list(repository.DEFAULT_FAMILIES)}")

    _print_header("DISCOVERING CANDIDATE ARMS")
    available_tiers = repository.available_tiers()
    print(f"Available tiers from MLflow: {available_tiers}")
    candidates = repository.discover_candidates(
        ticker=args.ticker,
        tiers=tiers,
        families=families,
    )
    candidates = [candidate for candidate in candidates if not candidate["predictions"].empty]
    print(f"Candidate arm count: {len(candidates)}")
    _print_candidates(candidates)
    if not candidates:
        raise ValueError("No candidates found. Check ticker, tiers, families, and MLflow artifacts.")

    _print_header("EVALUATION DATES")
    reward_calculator = RewardCalculator(
        reward_mode=args.reward_mode,
        direction_weight=args.direction_weight,
    )
    env = ForecastBanditEnv(
        candidates=candidates,
        reward_calculator=reward_calculator,
        date_policy=args.date_policy,
        start_date=args.start_date,
    )
    dates = env.dates
    if args.max_steps > 0:
        dates = dates[: args.max_steps]
        env.dates = dates
    print(f"date_policy      : {args.date_policy}")
    print(f"start_date       : {args.start_date or '(none)'}")
    print(f"evaluation dates : {len(dates)}")
    if dates:
        print(f"first date       : {pd.Timestamp(dates[0]).date()}")
        print(f"last date        : {pd.Timestamp(dates[-1]).date()}")
    else:
        raise ValueError("No overlapping evaluation dates found.")

    _print_header("UCB1-TUNED REPLAY")
    selector = UCB1TunedSelector(
        arm_ids=[candidate["arm_id"] for candidate in candidates],
        window_size=args.window_size,
        recency_mode=args.recency_mode,
        discount_factor=args.discount_factor,
        exploration_scale=args.exploration_scale,
    )
    print(f"recency_mode      : {args.recency_mode}")
    print(f"window_size       : {args.window_size}")
    print(f"discount_factor   : {args.discount_factor}")
    print(f"exploration_scale : {args.exploration_scale}")
    print(f"reward_mode       : {args.reward_mode}")
    print("-" * 100)

    events = _replay_with_prints(
        env=env,
        candidates=candidates,
        selector=selector,
        print_every=args.print_every,
        top_scores=args.top_scores,
    )

    _print_header("FINAL UCB SUMMARY")
    summary = selector.summary(step=max(1, len(events)))
    _print_table(summary[: args.top_scores])

    _print_header("CANDIDATE ERROR METRICS")
    metrics = _candidate_metrics(
        candidates,
        reward_calculator,
        start_date=args.start_date,
    )
    _print_table(metrics[: args.top_scores])

    winner = _winner_payload(candidates, selector, summary)
    _print_header("WINNER")
    _print_table([winner])
    print("\nDone.")
    return {
        "ticker": args.ticker,
        "candidate_count": len(candidates),
        "event_count": len(events),
        "winner": winner,
        "ucb_summary": summary,
        "candidate_metrics": metrics,
        "selection_events": events,
    }


def _replay_with_prints(
    *,
    env: ForecastBanditEnv,
    candidates: list[dict[str, Any]],
    selector: UCB1TunedSelector,
    print_every: int,
    top_scores: int,
) -> list[dict[str, Any]]:
    env.reset()
    events: list[dict[str, Any]] = []
    terminated = False
    truncated = False

    while not terminated and not truncated:
        step = env.current_step
        available_arm_ids = env.available_arm_ids()
        available = [
            candidate
            for candidate in candidates
            if candidate["arm_id"] in available_arm_ids
        ]
        if not available:
            break

        score_rows = _score_rows(selector, available, step=step)
        selected_arm = selector.select_arm(
            step=step,
            available_arm_ids=available_arm_ids,
        )
        selected_before_n = len(selector.reward_history.get(selected_arm, []))
        action = env.action_for_arm(selected_arm)
        _, reward, terminated, truncated, info = env.step(action)
        selector.update(selected_arm, float(reward), step=int(info["step"]))
        event = {
            key: value
            for key, value in info.items()
            if key not in {"timestamp", "action"}
        }
        event.update(
            {
                "selection_reason": "initial_exploration" if selected_before_n == 0 else "ucb_score",
            }
        )
        events.append(event)

        if step == 1 or step % print_every == 0 or terminated:
            _print_step(event, score_rows[:top_scores])
    return events


def _print_step(event: dict[str, Any], score_rows: list[dict[str, Any]]) -> None:
    print(
        "step={step:>4} date={ds} reason={selection_reason:<19} "
        "selected={arm_id} actual={actual:.4f} pred={prediction:.4f} "
        "ape={abs_pct_error:.4%} loss={loss:.6f} reward={reward:.6f}".format(**event)
    )
    print("  top UCB scores before update:")
    for score in score_rows:
        score_value = "inf" if math.isinf(float(score["ucb_score"])) else f"{score['ucb_score']:.6f}"
        print(
            f"    {score_value:>10} | n={score['n']:<4} "
            f"exploit={score['exploitation_reward']:.6f} "
            f"explore={score['exploration_bonus']:.6f} "
            f"var={score['variance']:.6f} | {score['arm_id']}"
        )


def _score_rows(
    selector: UCB1TunedSelector,
    candidates: list[dict[str, Any]],
    *,
    step: int,
) -> list[dict[str, Any]]:
    rows = []
    for candidate in candidates:
        arm_id = candidate["arm_id"]
        stats = selector.score_components(arm_id, step=step)
        rows.append(
            {
                "arm_id": arm_id,
                "n": stats["n"],
                "mean_reward": stats["mean_reward"],
                "variance": stats["variance"],
                "exploitation_reward": stats["exploitation_reward"],
                "exploration_bonus": stats["exploration_bonus"],
                "tuned_variance": stats["tuned_variance"],
                "ucb_score": stats["ucb_score"],
            }
        )
    return sorted(rows, key=lambda row: float(row["ucb_score"]), reverse=True)


def _print_candidates(candidates: list[dict[str, Any]]) -> None:
    rows = []
    for candidate in candidates:
        frame = candidate["predictions"].sort_values("ds")
        rows.append(
            {
                "arm_id": candidate["arm_id"],
                "run_name": candidate["run_name"],
                "run_id": str(candidate["run_id"])[:10],
                "rows": len(frame),
                "first_ds": str(pd.Timestamp(frame["ds"].iloc[0]).date()) if len(frame) else None,
                "last_ds": str(pd.Timestamp(frame["ds"].iloc[-1]).date()) if len(frame) else None,
                "pred_col": candidate["prediction_column"],
                "registerable": bool(candidate.get("registerable_model_uri")),
                "raw_model": bool(candidate.get("raw_model_uri")),
            }
        )
    _print_table(rows)


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


def _row_for_date(frame: pd.DataFrame, date_value: pd.Timestamp):
    rows = frame[pd.to_datetime(frame["ds"]) == pd.Timestamp(date_value)]
    if rows.empty:
        return None
    return rows.iloc[0]


def _candidate_metrics(
    candidates: list[dict[str, Any]],
    reward_calculator: RewardCalculator,
    start_date: str | None = "2026-02-03",
) -> list[dict[str, Any]]:
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
        ss_res = float(error.pow(2).sum())
        ss_tot = float((actual - actual.mean()).pow(2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        rows.append(
            {
                "arm_id": candidate["arm_id"],
                "rows": int(len(valid)),
                "mae": round(float(abs_error.mean()), 6),
                "rmse": round(float((error.pow(2).mean()) ** 0.5), 6),
                "mape": round(float(abs_pct.mean() * 100.0), 6),
                "r2": round(float(r2), 6),
                "mean_loss": round(float(sum(losses) / len(losses)), 6),
                "mean_reward": round(float(sum(rewards) / len(rewards)), 6),
            }
        )
    return sorted(rows, key=lambda row: (-float(row["mean_reward"]), float(row["rmse"])))


def _winner_payload(
    candidates: list[dict[str, Any]],
    selector: UCB1TunedSelector,
    summary: list[dict[str, Any]],
) -> dict[str, Any]:
    selected_summary = [row for row in summary if int(row["n"]) > 0]
    if selected_summary:
        winner_arm = str(
            sorted(
                selected_summary,
                key=lambda row: (float(row["mean_reward"]), float(row["ucb_score"])),
                reverse=True,
            )[0]["arm_id"]
        )
    else:
        winner_arm = selector.best_arm(step=1)
    candidate = next(candidate for candidate in candidates if candidate["arm_id"] == winner_arm)
    winner_summary = next(row for row in summary if row["arm_id"] == winner_arm)
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


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("(empty)")
        return
    columns = list(rows[0].keys())
    widths = {
        column: min(
            max(len(str(column)), *(len(_format_cell(row.get(column))) for row in rows)),
            48,
        )
        for column in columns
    }
    print(" | ".join(str(column).ljust(widths[column]) for column in columns))
    print("-+-".join("-" * widths[column] for column in columns))
    for row in rows:
        print(
            " | ".join(
                _format_cell(row.get(column))[: widths[column]].ljust(widths[column])
                for column in columns
            )
        )


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        return f"{value:.6f}"
    return str(value)


def _print_environment(args: argparse.Namespace) -> None:
    print(f"cwd                  : {Path.cwd()}")
    print(f"python               : {sys.executable}")
    print(f"tracking_uri arg     : {args.tracking_uri or '(default)'}")
    print(f"MLFLOW_TRACKING_URI  : {os.getenv('MLFLOW_TRACKING_URI')}")
    print(f"MLFLOW_S3_ENDPOINT   : {os.getenv('MLFLOW_S3_ENDPOINT_URL')}")
    print(f"AWS_ACCESS_KEY_ID    : {os.getenv('AWS_ACCESS_KEY_ID')}")
    print(f"AWS_DEFAULT_REGION   : {os.getenv('AWS_DEFAULT_REGION')}")


def _print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def _string_to_list(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verbose UCB1-Tuned debug runner for MLflow forecast artifacts."
    )
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--tiers", default="tier1,tier2,tier3,tier4,tier5,tier6,tier7,tier8")
    parser.add_argument("--families")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-prefix", default="stock_close")
    parser.add_argument("--date-policy", choices=["common", "union"], default="common")
    parser.add_argument("--start-date", default="2026-02-03")
    parser.add_argument("--recency-mode", choices=["sliding_window", "discounted", "all"], default="sliding_window")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--discount-factor", type=float, default=0.97)
    parser.add_argument("--exploration-scale", type=float, default=1.0)
    parser.add_argument("--reward-mode", choices=RewardCalculator.SUPPORTED_MODES, default="negative_rmse")
    parser.add_argument("--direction-weight", type=float, default=0.0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--top-scores", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--output-json")
    return parser.parse_args()


if __name__ == "__main__":
    main()
