"""
Data Layer
==========
Download close prices via yfinance, intersect to the maximum common date range
across all tickers, and compute demeaned log returns per instrument.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(tickers: list[str]) -> pd.DataFrame:
    """Download daily close prices for *tickers* over their full available history.

    Returns a DataFrame with a DatetimeIndex and one column per ticker.
    """
    raw = yf.download(
        tickers,
        period="max",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if raw is None or cast(pd.DataFrame, raw).empty:
        raise RuntimeError(f"No price data returned for {tickers!r}.")

    data = cast(pd.DataFrame, raw)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"]
    else:
        prices = data[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def intersect_to_common_range(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep only dates where **every** ticker has a valid positive price."""
    valid = prices.dropna()
    valid = valid[(valid > 0).all(axis=1)]

    if valid.empty:
        raise RuntimeError("No common date range found across all tickers.")

    per_ticker_first = valid.apply(lambda col: col.first_valid_index())
    per_ticker_last = valid.apply(lambda col: col.last_valid_index())
    common_start = per_ticker_first.max()
    common_end = per_ticker_last.min()

    trimmed = valid.loc[common_start:common_end]

    if len(trimmed) < 10:
        raise RuntimeError(
            f"Common date range {common_start} – {common_end} has only "
            f"{len(trimmed)} observations."
        )

    return trimmed


def compute_demeaned_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute demeaned log returns for each instrument column.

    For each column:
        log_return[t] = log(price[t] / price[t-1])
        demeaned[t]   = log_return[t] − mean(log_return)
    """
    log_returns = np.log(prices / prices.shift(1))
    log_returns = log_returns.dropna()
    demeaned = log_returns - log_returns.mean()
    return demeaned
