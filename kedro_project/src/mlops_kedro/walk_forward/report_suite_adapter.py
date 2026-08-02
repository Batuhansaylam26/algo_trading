"""
report_suite Package Adapter for Native algo_trading Integration
================================================================
Adapts report_suite evaluation capabilities to algo_trading data contracts,
exporting metrics DataFrames, PNG plots, and logging artifacts to MLflow.
"""

from __future__ import annotations

import importlib
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mlops_kedro.pipelines.stock_close_training.ml.local_artifacts import (
    LightweightArtifactStore,
)

LOGGER = logging.getLogger(__name__)


def _load_report_suite_modules():
    """Imports report_suite submodules directly from src package."""
    import report_suite.bootstrap as rs_bs
    import report_suite.data as rs_data
    import report_suite.example_run as rs_run
    import report_suite.report_metrics as rs_metrics
    import report_suite.strategy as rs_strat

    return rs_data, rs_strat, rs_metrics, rs_bs, rs_run



class ReportSuiteAdapter:
    """Integrates report_suite package for quantitative evaluation and native algo_trading metric export."""

    def __init__(self, output_dir: Path | str) -> None:
        if output_dir is None:
            raise ValueError("output_dir must be explicitly provided (no default fallback allowed).")
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.artifact_store = LightweightArtifactStore(root_dir=self.output_dir)

    def evaluate_and_generate_report(
        self,
        stitched_oos_df: pd.DataFrame,
        prices_df: pd.DataFrame,
        bootstrap_reps: int,
        alpha: float,
        experiment_name: str,
    ) -> dict[str, Any]:
        """Runs report_suite evaluation on out-of-sample walk-forward predictions."""
        if stitched_oos_df.empty:
            raise ValueError("stitched_oos_df cannot be empty.")
        if prices_df.empty:
            raise ValueError("prices_df cannot be empty.")
        if bootstrap_reps <= 0:
            raise ValueError(f"bootstrap_reps must be positive, got {bootstrap_reps}")
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be between 0 and 1, got {alpha}")

        rs_data, rs_strat, rs_metrics, rs_bs, rs_run = _load_report_suite_modules()

        LOGGER.info("Starting report_suite evaluation on %d OOS prediction rows...", len(stitched_oos_df))

        # 1. Format price matrix: (Index=date, Columns=tickers)
        prices_df = prices_df.copy()
        prices_df["ds"] = pd.to_datetime(prices_df["ds"], utc=True)
        price_matrix = prices_df.pivot(index="ds", columns="unique_id", values="y").sort_index()

        # Intersect to common valid range
        price_matrix = rs_data.intersect_to_common_range(price_matrix)

        # 2. Format signal matrix from predictions
        pred_df = stitched_oos_df.copy()
        pred_df["ds"] = pd.to_datetime(pred_df["ds"], utc=True)

        signal_raw = pred_df.pivot(index="ds", columns="unique_id", values="prediction").sort_index()

        # Reindex signals to match price matrix
        signal_raw = signal_raw.reindex(index=price_matrix.index, columns=price_matrix.columns).fillna(0.0)

        # Convert continuous predictions into discrete long/short/cash signals (-1, 0, +1)
        signals = np.where(signal_raw > 0, 1.0, np.where(signal_raw < 0, -1.0, 0.0))
        signals_df = pd.DataFrame(signals, index=price_matrix.index, columns=price_matrix.columns)

        # 3. Build returns and strategy metrics
        returns_df = rs_strat.calculate_log_returns(price_matrix)
        returns_df = rs_strat.demean_log_returns(returns_df)

        strat_returns = rs_strat.apply_strategy_signals(signals_df, returns_df)
        portfolio_returns = rs_strat.compute_equal_weighted_portfolio(strat_returns)

        # Generate DataFrames
        per_instrument_df = rs_run.build_per_instrument_table(strat_returns)
        cross_instrument_df = rs_run.build_cross_instrument_table(per_instrument_df, portfolio_returns)

        # 4. Stationary Block Bootstrap
        LOGGER.info("Running Stationary Block Bootstrap (reps=%d)...", bootstrap_reps)
        bootstrap_ci_df = rs_run.build_bootstrap_table(
            strat_returns=strat_returns,
            portfolio_returns=portfolio_returns,
            num_samples=bootstrap_reps,
            alpha=alpha,
        )

        # 5. Save Artifacts to Local ArtifactStore
        metrics_dir = self.output_dir / "metrics"
        plots_dir = self.output_dir / "plots"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        per_instrument_path = metrics_dir / "per_instrument_metrics.csv"
        cross_summary_path = metrics_dir / "cross_instrument_summary.csv"
        bootstrap_ci_path = metrics_dir / "bootstrap_confidence_intervals.csv"

        per_instrument_df.to_csv(per_instrument_path)
        cross_instrument_df.to_csv(cross_summary_path)
        bootstrap_ci_df.to_csv(bootstrap_ci_path, index=False)

        portfolio_metrics_path = metrics_dir / "portfolio_summary_metrics.csv"
        portfolio_metrics_series = rs_metrics.compute_all_metrics(portfolio_returns)
        portfolio_metrics_df = pd.DataFrame({
            "metric": portfolio_metrics_series.index,
            "value": portfolio_metrics_series.values,
        })
        portfolio_metrics_df.to_csv(portfolio_metrics_path, index=False)

        # Generate individual plots
        saved_plot_paths = rs_run.save_individual_plots(
            strat_returns=strat_returns,
            portfolio_returns=portfolio_returns,
            bootstrap_ci_df=bootstrap_ci_df,
            output_dir=plots_dir,
        )

        # 6. Log to MLflow
        try:
            import mlflow

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name="walk_forward_report_suite"):
                # Log summary metrics
                for key, val in portfolio_metrics_series.items():
                    if isinstance(val, (int, float, np.number)) and not np.isnan(val):
                        mlflow.log_metric(f"portfolio_{key}", float(val))

                # Log tables
                mlflow.log_table(portfolio_metrics_df, artifact_file="portfolio_metrics.json")
                mlflow.log_table(cross_instrument_df.reset_index(), artifact_file="cross_instrument_summary.json")
                mlflow.log_table(bootstrap_ci_df, artifact_file="bootstrap_confidence_intervals.json")

                # Log directory artifacts
                mlflow.log_artifacts(str(self.output_dir))
                LOGGER.info("Successfully logged metrics and artifacts to MLflow experiment '%s'.", experiment_name)
        except Exception as exc:
            LOGGER.warning("MLflow logging skipped or encountered an error: %s", exc)

        LOGGER.info(
            "report_suite evaluation complete! Metrics CSVs saved to %s, plots saved to %s",
            metrics_dir,
            plots_dir,
        )

        return {
            "per_instrument_df": per_instrument_df,
            "cross_instrument_df": cross_instrument_df,
            "bootstrap_ci_df": bootstrap_ci_df,
            "portfolio_metrics": portfolio_metrics_series,
            "output_dir": str(self.output_dir),
            "saved_plot_paths": saved_plot_paths,
        }
