from __future__ import annotations

import argparse
import json
from typing import Any

try:
    from .reward_calculator_class import RewardCalculator
except ImportError:
    from reward_calculator_class import RewardCalculator


def main() -> None:
    parser = argparse.ArgumentParser(description="Recency-aware UCB1-Tuned model selector.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    backtest_parser = subparsers.add_parser("backtest")
    _add_selection_args(backtest_parser)

    register_parser = subparsers.add_parser("register")
    _add_selection_args(register_parser)
    register_parser.add_argument("--min-mean-reward", type=float, default=0.98)

    serve_parser = subparsers.add_parser("serve")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8088)
    serve_parser.add_argument("--tracking-uri")
    serve_parser.add_argument("--experiment-prefix", default="stock_close")

    pecnet_parser = subparsers.add_parser("pecnet-predict")
    pecnet_parser.add_argument("--run-id", required=True)
    pecnet_parser.add_argument("--model-artifact-path", required=True)
    pecnet_parser.add_argument("--input-artifact-path")
    pecnet_parser.add_argument("--device", default="auto")
    pecnet_parser.add_argument("--tracking-uri")

    args = parser.parse_args()
    if args.command == "serve":
        _serve(args)
    elif args.command == "backtest":
        _print_json(_workflow(args).backtest(**_selection_kwargs(args)))
    elif args.command == "register":
        _print_json(
            _workflow(args).register_winner(
                **_selection_kwargs(args),
                min_mean_reward=args.min_mean_reward,
            )
        )
    elif args.command == "pecnet-predict":
        _pecnet_predict(args)


def _add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--tiers")
    parser.add_argument("--families")
    parser.add_argument("--recency-mode", default="sliding_window")
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--discount-factor", type=float, default=0.97)
    parser.add_argument("--exploration-scale", type=float, default=1.0)
    parser.add_argument(
        "--reward-mode",
        choices=RewardCalculator.SUPPORTED_MODES,
        default="negative_rmse",
    )
    parser.add_argument("--direction-weight", type=float, default=0.0)
    parser.add_argument("--date-policy", default="common")
    parser.add_argument("--start-date", default="2026-02-03")
    parser.add_argument("--tracking-uri")
    parser.add_argument("--experiment-prefix", default="stock_close")


def _workflow(args: argparse.Namespace):
    from .mlflow_forecast_repository_class import MlflowForecastRepository
    from .selection_workflow_class import UcbSelectionWorkflow

    repository = MlflowForecastRepository(
        tracking_uri=args.tracking_uri,
        experiment_prefix=args.experiment_prefix,
    )
    return UcbSelectionWorkflow(repository=repository)


def _selection_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "ticker": args.ticker,
        "tiers": _split(args.tiers),
        "families": _split(args.families),
        "recency_mode": args.recency_mode,
        "window_size": args.window_size,
        "discount_factor": args.discount_factor,
        "exploration_scale": args.exploration_scale,
        "reward_mode": args.reward_mode,
        "direction_weight": args.direction_weight,
        "date_policy": args.date_policy,
        "start_date": args.start_date,
    }


def _serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .fastapi_app_class import UcbFastApiApp

    app = UcbFastApiApp(
        tracking_uri=args.tracking_uri,
        experiment_prefix=args.experiment_prefix,
    ).create()
    uvicorn.run(app, host=args.host, port=args.port)


def _pecnet_predict(args: argparse.Namespace) -> None:
    from .mlflow_forecast_repository_class import MlflowForecastRepository
    from .pecnet_torch_predictor_class import PecnetTorchPredictor

    repository = MlflowForecastRepository(tracking_uri=args.tracking_uri)
    predictor = PecnetTorchPredictor(repository=repository)
    result = predictor.predict_from_artifact(
        run_id=args.run_id,
        model_artifact_path=args.model_artifact_path,
        input_artifact_path=args.input_artifact_path,
        device=args.device,
    )
    _print_json(result)


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
