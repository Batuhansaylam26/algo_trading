"""
Strategy Layer
==============
Generate trading signals and apply them to demeaned log returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def perfect_lookforward_signals(demeaned_log_returns: pd.DataFrame) -> pd.DataFrame:
    """Generate perfect-foresight signals.

    signal[t] = sign(return[t+1])
    """
    signals = np.sign(demeaned_log_returns.shift(-1))
    return signals


def apply_signals(
    demeaned_log_returns: pd.DataFrame,
    signals: pd.DataFrame,
) -> pd.DataFrame:
    """Apply signals to returns with a one-day causal shift.

    applied_signal[t] = signal[t-1]
    strategy_return[t] = applied_signal[t] × demeaned_log_return[t]
    """
    applied = signals.shift(1)
    strategy_returns = applied * demeaned_log_returns
    strategy_returns = strategy_returns.dropna()
    return strategy_returns


def equal_weight_average(strategy_returns: pd.DataFrame) -> pd.Series:
    """Row-wise arithmetic mean of log strategy returns across instruments."""
    avg = strategy_returns.mean(axis=1)
    avg.name = "avg_strategy_return"
    return avg
