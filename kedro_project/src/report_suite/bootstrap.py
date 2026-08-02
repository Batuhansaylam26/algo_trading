"""
Bootstrap Layer
===============
1-D stationary bootstrap on a pre-averaged strategy return series.

Uses the Politis & Romano (1994) stationary bootstrap with automatic block
length selection via Politis & White (2004).  Only bootstrap-safe (order-
independent) metrics are evaluated on each resample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch.bootstrap import StationaryBootstrap, optimal_block_length

from .report_metrics import BOOTSTRAP_SAFE_METRICS, compute_metrics


class StrategyBootstrap:
    """Multivariate stationary bootstrap for individual instruments and the average series."""

    def __init__(self, strategy_returns: pd.DataFrame):
        # Store df and add Avg column
        self.df = strategy_returns.copy()
        if "Avg" not in self.df.columns:
            self.df["Avg"] = strategy_returns.mean(axis=1)

        self.X = self.df.to_numpy()

        # Data-driven block length selection across all columns (conservative max)
        opt = optimal_block_length(self.X)
        self.block_length: float = max(float(opt["stationary"].max()), 1.0)

        # Precompute point estimates for all columns
        self.point_estimates: dict[tuple[str, str], float | None] = {}
        for col in self.df.columns:
            pe_vals = compute_metrics(self.df[col], BOOTSTRAP_SAFE_METRICS)
            for m_key, val in pe_vals.items():
                self.point_estimates[(col, m_key)] = val

        self.boot_df: pd.DataFrame | None = None

    def run(self, reps: int = 2000, seed: int | None = None) -> pd.DataFrame:
        """Run the stationary bootstrap and compute metrics for each instrument on each resample.

        Returns a DataFrame with MultiIndex columns: (Instrument, Metric)
        """
        rng = np.random.default_rng(seed)
        bs = StationaryBootstrap(self.block_length, self.X, seed=rng)

        records: list[dict[tuple[str, str], float | None]] = []
        cols = self.df.columns

        for data, _ in bs.bootstrap(reps):
            x_star = data[0]
            # Build resampled DataFrame
            resampled_df = pd.DataFrame(x_star, columns=cols)
            row_record = {}
            for col in cols:
                metrics_vals = compute_metrics(resampled_df[col], BOOTSTRAP_SAFE_METRICS)
                for m_key, val in metrics_vals.items():
                    row_record[(col, m_key)] = val
            records.append(row_record)

        self.boot_df = pd.DataFrame(records)
        self.boot_df.columns = pd.MultiIndex.from_tuples(
            self.boot_df.columns, names=["Instrument", "Metric"]
        )
        return self.boot_df

    def percentile_ci(
        self, instrument: str, name: str, alpha: float = 0.05
    ) -> tuple[float, float]:
        """Return the (alpha/2, 1-alpha/2) percentile confidence interval for a specific instrument and metric."""
        if self.boot_df is None:
            raise RuntimeError("Call .run() first.")
        lo, hi = self.boot_df[(instrument, name)].quantile([alpha / 2, 1 - alpha / 2])
        return float(lo), float(hi)

    def prob_below(self, instrument: str, name: str, threshold: float = 0.0) -> float:
        """P(metric <= threshold) under the bootstrap distribution."""
        if self.boot_df is None:
            raise RuntimeError("Call .run() first.")
        return float((self.boot_df[(instrument, name)] <= threshold).mean())
