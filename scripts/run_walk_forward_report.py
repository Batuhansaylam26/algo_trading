"""
Walk-Forward Engine CLI Report Runner
====================================
CLI entrypoint executing parallel walk-forward model training across GPUs
and generating report_suite statistical metrics and visualization artifacts.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from mlops_kedro.walk_forward import (
    ParallelWalkForwardRunner,
    ReportSuiteAdapter,
    WalkForwardSplitter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
LOGGER = logging.getLogger("run_walk_forward_report")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONF_BASE = PROJECT_ROOT / "kedro_project" / "conf" / "base"


def load_kedro_parameters() -> dict[str, Any]:
    """Loads machine learning parameters from Kedro configuration files."""
    ml_params_path = CONF_BASE / "parameters_machine_learning.yml"

    if not ml_params_path.exists():
        raise FileNotFoundError(f"Required parameter file not found: {ml_params_path}")

    with open(ml_params_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if "stock_close_machine_learning" not in data:
        raise KeyError("Missing root key 'stock_close_machine_learning' in parameters_machine_learning.yml")

    return data["stock_close_machine_learning"]


def build_model_spec_for_tier(ml_params: dict[str, Any], tier_name: str) -> dict[str, Any]:
    """Extracts exact model specification and feature set for a tier."""
    if "tiers" not in ml_params or tier_name not in ml_params["tiers"]:
        raise KeyError(f"Tier '{tier_name}' not defined in parameters_machine_learning.yml tiers configuration.")

    tier_cfg = ml_params["tiers"][tier_name]

    if "features" not in tier_cfg:
        raise KeyError(f"Missing 'features' list for tier '{tier_name}'.")
    if "pecnet_preprocess_params" not in ml_params:
        raise KeyError("Missing 'pecnet_preprocess_params' in parameters_machine_learning.yml.")
    if "pecnet_hyperparams" not in ml_params:
        raise KeyError("Missing 'pecnet_hyperparams' in parameters_machine_learning.yml.")

    feature_columns = list(tier_cfg["features"])
    preprocess_params = dict(ml_params["pecnet_preprocess_params"])
    hyperparams = dict(ml_params["pecnet_hyperparams"])
    selection_params = dict(ml_params.get("pecnet_selection_params", {}))

    return {
        "tier_name": tier_name,
        "feature_columns": feature_columns,
        "preprocess_params": preprocess_params,
        "hyperparams": hyperparams,
        "selection_params": selection_params,
    }


def load_silver_prices_data() -> pd.DataFrame:
    """Loads stock price dataset from Delta Lake or local CSV fallback."""
    minio_host = os.getenv("MINIO_HOST", "127.0.0.1")
    minio_port = os.getenv("MINIO_PORT", "9000")
    endpoint = f"http://{minio_host}:{minio_port}"

    storage_options = {
        "AWS_ACCESS_KEY_ID": os.getenv("AWS_ACCESS_KEY_ID", "admin"),
        "AWS_SECRET_ACCESS_KEY": os.getenv("AWS_SECRET_ACCESS_KEY", "admin1234"),
        "AWS_REGION": "us-east-1",
        "AWS_ENDPOINT_URL": endpoint,
        "AWS_ALLOW_HTTP": "true",
        "AWS_S3_ALLOW_UNSAFE_RENAME": "true",
        "AWS_S3_FORCE_PATH_STYLE": "true",
    }

    try:
        from deltalake import DeltaTable

        table = DeltaTable("s3://dataops/silver/stock_prices", storage_options=storage_options)
        df = table.to_pandas()
        LOGGER.info("Loaded silver stock prices from Delta Lake (MinIO).")
        return df
    except Exception as exc:
        LOGGER.info("Delta Lake unavailable (%s). Checking local CSV datasets...", exc)

    candidates = [
        PROJECT_ROOT.parent / "market_data_10y.csv",
        PROJECT_ROOT / "market_data_10y.csv",
        PROJECT_ROOT / "data" / "market_data_10y.csv",
    ]
    for csv_path in candidates:
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            LOGGER.info("Loaded market price data from CSV: %s", csv_path)
            return df

    raise FileNotFoundError("Could not find silver_stock_prices dataset in MinIO Delta Lake or local CSV files.")


def main():
    ml_params = load_kedro_parameters()

    if "walk_forward" not in ml_params:
        raise KeyError("Missing mandatory 'walk_forward' configuration section in parameters_machine_learning.yml")

    wf_config = ml_params["walk_forward"]

    required_wf_keys = [
        "training_prediction_count",
        "testing_prediction_count",
        "num_gpus",
        "max_workers",
        "bootstrap_reps",
        "output_dir",
    ]
    for k in required_wf_keys:
        if k not in wf_config:
            raise KeyError(f"Missing mandatory walk_forward parameter '{k}' in parameters_machine_learning.yml")

    parser = argparse.ArgumentParser(description="Run Walk-Forward Backtest & report_suite Evaluation")
    parser.add_argument("--tier", type=str, required=True, help="Feature tier to evaluate (e.g. tier4, tier5, tier6)")
    parser.add_argument("--training-prediction-count", type=int, default=wf_config["training_prediction_count"], help="Target training prediction count per block")
    parser.add_argument("--testing-prediction-count", type=int, default=wf_config["testing_prediction_count"], help="Target out-of-sample testing prediction count per block")
    parser.add_argument("--num-gpus", type=int, default=wf_config["num_gpus"], help="Number of GPUs to distribute work across")
    parser.add_argument("--max-workers", type=int, default=wf_config["max_workers"], help="Maximum process workers in ProcessPoolExecutor")
    parser.add_argument("--bootstrap-reps", type=int, default=wf_config["bootstrap_reps"], help="Stationary bootstrap repetitions for report_suite")
    parser.add_argument("--output-dir", type=str, default=wf_config["output_dir"], help="Output directory for CSV metrics and plots")
    parser.add_argument("--alpha", type=float, default=0.05, help="Alpha level for confidence intervals")
    parser.add_argument("--skip-training", action="store_true", help="Skip model training and load cached OOS predictions from disk if available")

    args = parser.parse_args()

    LOGGER.info("=" * 60)
    LOGGER.info("Walk-Forward Engine & report_suite Integration Runner")
    LOGGER.info("Tier: %s | GPUs: %d | Workers: %d | Train Count: %d | Test Count: %d", args.tier, args.num_gpus, args.max_workers, args.training_prediction_count, args.testing_prediction_count)
    LOGGER.info("=" * 60)

    # 1. Build Model Spec
    model_spec = build_model_spec_for_tier(ml_params, args.tier)

    # 2. Load Historical Data (Silver Stock Prices)
    LOGGER.info("Loading Silver Stock Prices data...")
    silver_df = load_silver_prices_data()

    if "date" in silver_df.columns and "ds" not in silver_df.columns:
        silver_df["ds"] = pd.to_datetime(silver_df["date"], utc=True)
    if "Date" in silver_df.columns and "ds" not in silver_df.columns:
        silver_df["ds"] = pd.to_datetime(silver_df["Date"], utc=True)

    if "symbol" in silver_df.columns and "unique_id" not in silver_df.columns:
        silver_df["unique_id"] = silver_df["symbol"]
    if "Ticker" in silver_df.columns and "unique_id" not in silver_df.columns:
        silver_df["unique_id"] = silver_df["Ticker"]

    if "close" in silver_df.columns and "y" not in silver_df.columns:
        silver_df["y"] = silver_df["close"]
    if "Close" in silver_df.columns and "y" not in silver_df.columns:
        silver_df["y"] = silver_df["Close"]

    LOGGER.info("Loaded %d price records for %d unique instruments.", len(silver_df), silver_df["unique_id"].nunique())

    # Prediction Cache File Path
    pred_dir = Path(args.output_dir) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    cache_path = pred_dir / f"oos_predictions_{args.tier}.csv"

    if args.skip_training and cache_path.exists():
        LOGGER.info("Skipping model training as requested. Loading cached OOS predictions from: %s", cache_path)
        stitched_oos = pd.read_csv(cache_path)
        stitched_oos["ds"] = pd.to_datetime(stitched_oos["ds"], utc=True)
    else:
        # 3. Partition Data into Walk-Forward Blocks
        splitter = WalkForwardSplitter(
            training_prediction_count=args.training_prediction_count,
            testing_prediction_count=args.testing_prediction_count,
        )

        # 4. Execute Parallel Walk-Forward Runner across GPUs
        runner = ParallelWalkForwardRunner(
            num_gpus=args.num_gpus,
            max_workers=args.max_workers,
        )
        stitched_oos = runner.run_walk_forward(silver_df, model_spec=model_spec, splitter=splitter)

        if stitched_oos.empty:
            LOGGER.error("Walk-forward run produced empty out-of-sample predictions. Exiting.")
            sys.exit(1)

        LOGGER.info("Successfully generated %d continuous out-of-sample predictions.", len(stitched_oos))

        stitched_oos.to_csv(cache_path, index=False)
        LOGGER.info("Saved OOS predictions cache to: %s", cache_path)

    # 5. Run report_suite Evaluation & Metric Export
    adapter = ReportSuiteAdapter(output_dir=args.output_dir)
    report_results = adapter.evaluate_and_generate_report(
        stitched_oos_df=stitched_oos,
        prices_df=silver_df,
        bootstrap_reps=args.bootstrap_reps,
        alpha=args.alpha,
        experiment_name=f"stock_close_walk_forward_{args.tier}",
    )

    LOGGER.info("Walk-forward backtest & metric report generation complete!")
    LOGGER.info("Metrics saved to: %s/metrics/", report_results["output_dir"])


if __name__ == "__main__":
    main()
