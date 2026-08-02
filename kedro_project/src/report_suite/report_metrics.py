"""
Report Metrics
==============
Two registries of metric definitions:

- **BOOTSTRAP_SAFE_METRICS** — order-independent statistics that are valid
  on resampled (out-of-order) return series.
- **FULL_SAMPLE_ONLY_METRICS** — path-dependent or calendar-dependent
  statistics that should only be computed on the actual sample.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import quantstats as qs

PERIODS_PER_YEAR = 252
VAR_CONFIDENCE = 0.95


@dataclass(frozen=True)
class MetricDef:
    key: str
    label: str
    fmt: str  # "percent", "ratio", "number", "integer"
    fn: Callable[[pd.Series], Any]


def _safe(fn: Callable[..., Any], *args: Any, **kwargs: Any) -> float | None:
    """Call *fn* and return a finite float, or None on failure."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = fn(*args, **kwargs)
    except (ArithmeticError, ValueError, IndexError, TypeError):
        return None
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return round(f, 8) if math.isfinite(f) else None


# ---------------------------------------------------------------------------
# Bootstrap-safe metrics (strictly linear / sample-unbiased statistics)
# ---------------------------------------------------------------------------
BOOTSTRAP_SAFE_METRICS: list[MetricDef] = [
    MetricDef(
        "avg_log_return",
        "Avg Log Return",
        "number",
        lambda r: _safe(lambda s: s.mean(), r),
    ),
    MetricDef(
        "win_rate",
        "Win Rate",
        "percent",
        lambda r: _safe(qs.stats.win_rate, r, aggregate=None, prepare_returns=False),
    ),
]


# ---------------------------------------------------------------------------
# Full-sample-only metrics (nonlinear, ratio, quantile, or path dependent)
# ---------------------------------------------------------------------------
def _log_max_drawdown(r: pd.Series) -> float:
    wealth = np.exp(np.cumsum(r.to_numpy()))
    running_max = np.maximum.accumulate(wealth)
    dd = wealth / running_max - 1.0
    return float(dd.min())


FULL_SAMPLE_ONLY_METRICS: list[MetricDef] = [
    MetricDef(
        "volatility",
        "Volatility (Ann.)",
        "percent",
        lambda r: _safe(
            qs.stats.volatility,
            r,
            periods=PERIODS_PER_YEAR,
            annualize=True,
            prepare_returns=False,
        ),
    ),
    MetricDef(
        "sharpe",
        "Sharpe",
        "ratio",
        lambda r: _safe(
            qs.stats.sharpe,
            r,
            rf=0.0,
            periods=PERIODS_PER_YEAR,
            annualize=True,
        ),
    ),
    MetricDef(
        "sortino",
        "Sortino",
        "ratio",
        lambda r: _safe(
            qs.stats.sortino,
            r,
            rf=0.0,
            periods=PERIODS_PER_YEAR,
            annualize=True,
        ),
    ),
    MetricDef(
        "skew",
        "Skew",
        "ratio",
        lambda r: _safe(qs.stats.skew, r, prepare_returns=False),
    ),
    MetricDef(
        "kurtosis",
        "Kurtosis",
        "ratio",
        lambda r: _safe(qs.stats.kurtosis, r, prepare_returns=False),
    ),
    MetricDef(
        "value_at_risk",
        "VaR (95%)",
        "percent",
        lambda r: _safe(
            qs.stats.value_at_risk,
            r,
            confidence=VAR_CONFIDENCE,
            prepare_returns=False,
        ),
    ),
    MetricDef(
        "cvar",
        "CVaR (95%)",
        "percent",
        lambda r: _safe(
            qs.stats.conditional_value_at_risk,
            r,
            confidence=VAR_CONFIDENCE,
            prepare_returns=False,
        ),
    ),
    MetricDef(
        "payoff_ratio",
        "Payoff Ratio",
        "ratio",
        lambda r: _safe(qs.stats.payoff_ratio, r, prepare_returns=False),
    ),
    MetricDef(
        "profit_factor",
        "Profit Factor",
        "ratio",
        lambda r: _safe(qs.stats.profit_factor, r, prepare_returns=False),
    ),
    MetricDef(
        "kelly",
        "Kelly Criterion",
        "percent",
        lambda r: _safe(qs.stats.kelly_criterion, r, prepare_returns=False),
    ),
    MetricDef(
        "max_drawdown",
        "Max Drawdown",
        "percent",
        lambda r: _safe(_log_max_drawdown, r),
    ),
    MetricDef(
        "avg_drawdown",
        "Avg Drawdown",
        "percent",
        lambda r: _safe(
            lambda s: qs.stats.drawdown_details(
                qs.stats.to_drawdown_series(s)
            )["max drawdown"].mean()
            / 100.0,
            r,
        ),
    ),
    MetricDef(
        "longest_dd_days",
        "Longest DD Days",
        "integer",
        lambda r: _safe(
            lambda s: qs.stats.drawdown_details(
                qs.stats.to_drawdown_series(s)
            )["days"].max(),
            r,
        ),
    ),
    MetricDef(
        "calmar",
        "Calmar",
        "ratio",
        lambda r: _safe(
            qs.stats.calmar, r, prepare_returns=False, periods=PERIODS_PER_YEAR
        ),
    ),
    MetricDef(
        "recovery_factor",
        "Recovery Factor",
        "ratio",
        lambda r: _safe(qs.stats.recovery_factor, r, rf=0.0, prepare_returns=False),
    ),
    MetricDef(
        "ulcer_index",
        "Ulcer Index",
        "ratio",
        lambda r: _safe(qs.stats.ulcer_index, r),
    ),
    MetricDef(
        "consec_wins",
        "Max Consec. Wins",
        "integer",
        lambda r: _safe(qs.stats.consecutive_wins, r, prepare_returns=False),
    ),
    MetricDef(
        "consec_losses",
        "Max Consec. Losses",
        "integer",
        lambda r: _safe(qs.stats.consecutive_losses, r, prepare_returns=False),
    ),
    MetricDef(
        "gain_pain",
        "Gain/Pain Ratio",
        "ratio",
        lambda r: _safe(qs.stats.gain_to_pain_ratio, r, rf=0.0, resolution="D"),
    ),
    MetricDef(
        "best_day",
        "Best Day",
        "percent",
        lambda r: _safe(qs.stats.best, r, aggregate=None, prepare_returns=False),
    ),
    MetricDef(
        "worst_day",
        "Worst Day",
        "percent",
        lambda r: _safe(qs.stats.worst, r, aggregate=None, prepare_returns=False),
    ),
]


ALL_METRICS: list[MetricDef] = BOOTSTRAP_SAFE_METRICS + FULL_SAMPLE_ONLY_METRICS


def compute_metrics(
    returns: pd.Series,
    metric_list: list[MetricDef],
) -> dict[str, float | None]:
    """Evaluate every metric in *metric_list* on *returns*."""
    return {m.key: m.fn(returns) for m in metric_list}


def compute_all_metrics(returns: pd.Series) -> dict[str, float | None]:
    """Evaluate both safe and full-sample metrics on *returns*."""
    return compute_metrics(returns, ALL_METRICS)
