"""
Run Report — Multi-Instrument Strategy Evaluation
==================================================
Provides modular plotters and DataFrame table builders for per-instrument,
portfolio-level, and stationary bootstrap evaluations.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import quantstats as qs

from .bootstrap import StrategyBootstrap
from .data import (
    compute_demeaned_log_returns,
    download_prices,
    intersect_to_common_range,
)
from .report_metrics import (
    ALL_METRICS,
    BOOTSTRAP_SAFE_METRICS,
    compute_all_metrics,
    compute_metrics,
)
from .strategy import apply_signals, equal_weight_average, perfect_lookforward_signals

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "JNJ", "PG", "KO",
    "XOM", "JPM", "WMT", "PFE", "GE",
]


def _safe_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(name)).strip("_")


def build_per_instrument_table(
    strategy_returns: pd.DataFrame,
) -> pd.DataFrame:
    """Builds a DataFrame of metrics for each instrument (Columns: Tickers, Index: Metric Keys)."""
    results: dict[str, dict[str, float | None]] = {}
    for ticker in strategy_returns.columns:
        results[ticker] = compute_all_metrics(strategy_returns[ticker])

    df = pd.DataFrame(results)
    df.index.name = "metric"
    return df


def build_cross_instrument_table(
    per_inst_df: pd.DataFrame,
    avg_returns: pd.Series,
) -> pd.DataFrame:
    """Build a cross-instrument summary table comparing average single-asset metrics
    against equal-weighted portfolio metrics.
    """
    single_asset_mean = per_inst_df.mean(axis=1)
    portfolio_metrics = compute_all_metrics(avg_returns)

    rows: dict[str, dict[str, float | None]] = {}
    for m in ALL_METRICS:
        val_single = single_asset_mean.get(m.key)
        val_port = portfolio_metrics.get(m.key)

        if val_single is not None and (isinstance(val_single, float) and np.isnan(val_single)):
            val_single = None
        if val_port is not None and (isinstance(val_port, float) and np.isnan(val_port)):
            val_port = None

        rows[m.key] = {
            "metric_label": m.label,
            "metric_format": m.fmt,
            "single_asset_mean": val_single,
            "equal_weight_portfolio": val_port,
        }

    df = pd.DataFrame(rows).T
    df.index.name = "metric"
    return df


def build_bootstrap_table(
    bs: StrategyBootstrap,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Builds a structured DataFrame of bootstrap point estimates and percentile CIs."""
    rows = []
    for inst in bs.df.columns:
        for m in BOOTSTRAP_SAFE_METRICS:
            lo, hi = bs.percentile_ci(inst, m.key, alpha)
            pe = bs.point_estimates.get((inst, m.key))
            rows.append({
                "unique_id": inst,
                "metric": m.key,
                "metric_label": m.label,
                "point_estimate": pe,
                "ci_lower": lo,
                "ci_upper": hi,
                "alpha": alpha,
            })
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Individual Standalone Plotters
# ---------------------------------------------------------------------------
def plot_single_equity_curve(
    series: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    """Saves a standalone equity curve plot for a single instrument or portfolio."""
    fig, ax = plt.subplots(figsize=(10, 5))
    cum = np.exp(series.cumsum())

    ax.plot(cum.index, cum.values, color="#1e40af", linewidth=2.0, label="Cumulative Return")
    ax.set_yscale("log")
    ax.set_title(f"Cumulative Equity: {title}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Growth of $1 (Log Scale)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)


def plot_single_drawdown(
    series: pd.Series,
    title: str,
    output_path: Path,
) -> None:
    """Saves a standalone drawdown series plot for a single instrument or portfolio."""
    fig, ax = plt.subplots(figsize=(10, 5))
    wealth = np.exp(series.cumsum())
    running_max = np.maximum.accumulate(wealth)
    drawdown = (wealth / running_max - 1.0) * 100.0

    ax.fill_between(drawdown.index, drawdown.values, 0, color="#dc2626", alpha=0.4, label="Drawdown %")
    ax.plot(drawdown.index, drawdown.values, color="#991b1b", linewidth=1.2)
    ax.set_title(f"Drawdown Series: {title}", fontsize=12, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)


def plot_bootstrap_distribution(
    vals: pd.Series,
    metric_label: str,
    instrument_name: str,
    point_est: float | None,
    ci_low: float | None,
    ci_high: float | None,
    output_path: Path,
) -> None:
    """Saves a standalone histogram for a single bootstrap metric distribution."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(vals.dropna(), bins=40, alpha=0.75, color="#2563eb", edgecolor="white")

    if point_est is not None and not np.isnan(point_est):
        ax.axvline(point_est, color="black", lw=2, label=f"Point Est ({point_est:.4f})")

    if ci_low is not None and ci_high is not None and not (np.isnan(ci_low) or np.isnan(ci_high)):
        ax.axvline(ci_low, color="#dc2626", ls="--", lw=1.5, label=f"CI Low ({ci_low:.4f})")
        ax.axvline(ci_high, color="#dc2626", ls="--", lw=1.5, label=f"CI High ({ci_high:.4f})")

    ax.set_title(f"Bootstrap Distribution: {instrument_name} - {metric_label}", fontsize=11, fontweight="bold")
    ax.set_xlabel("Metric Value")
    ax.set_ylabel("Frequency")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=200)
    plt.close(fig)


# Legacy multi-plot wrappers for backward compatibility
def plot_equity_curves(
    strategy_returns: pd.DataFrame,
    avg_returns: pd.Series,
    output_path: Path,
) -> None:
    plot_single_equity_curve(avg_returns, "Equal-Weight Portfolio", output_path)


def plot_drawdown_series(
    strategy_returns: pd.DataFrame,
    avg_returns: pd.Series,
    output_path: Path,
) -> None:
    plot_single_drawdown(avg_returns, "Equal-Weight Portfolio", output_path)
