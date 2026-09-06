from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class StockCloseTierMetadata:
    DEFINITIONS: dict[str, dict[str, str]] = {
        "tier1": {
            "name": "Historical baseline",
            "feature_representation": "Previous OHLCV-style historical features.",
            "hypothesis": "Recent price and volume history provides a baseline signal.",
            "benchmark_role": "model_family_benchmark",
            "comparison_scope": "mlforecast_statsforecast_pecnet",
        },
        "tier2": {
            "name": "Calendar-aware baseline",
            "feature_representation": (
                "Tier 1 features plus calendar gap days and Fourier time encodings."
            ),
            "hypothesis": (
                "Business-day gaps and cyclical calendar effects add useful temporal context."
            ),
            "benchmark_role": "model_family_benchmark",
            "comparison_scope": "mlforecast_statsforecast_pecnet",
        },
        "tier3": {
            "name": "Residual-correlation PECNet selection",
            "feature_representation": (
                "Calendar-aware candidate features; PECNet selects features by residual correlation."
            ),
            "hypothesis": (
                "Features correlated with prediction errors can improve PECNet compensation."
            ),
            "benchmark_role": "pecnet_configuration_benchmark",
            "comparison_scope": "pecnet_only",
        },
        "tier4": {
            "name": "Thesis-inspired PECNet",
            "feature_representation": "Wavelet/error-compensated PECNet structure.",
            "hypothesis": (
                "Cascaded error compensation can model residual structures left by the main network."
            ),
            "benchmark_role": "pecnet_configuration_benchmark",
            "comparison_scope": "pecnet_only",
        },
        "tier5": {
            "name": "Multi-timeframe feature set",
            "feature_representation": "Daily and weekly historical context.",
            "hypothesis": (
                "Combining short-term daily behavior with weekly context improves robustness."
            ),
            "benchmark_role": "model_family_benchmark",
            "comparison_scope": "mlforecast_statsforecast_pecnet",
        },
        "tier6": {
            "name": "Weekly-context feature set with calendar features",
            "feature_representation": (
                "Daily target context, weekly close context, calendar gap days, and Fourier encodings."
            ),
            "hypothesis": (
                "Weekly regime information and calendar-aware context help explain changing dynamics."
            ),
            "benchmark_role": "model_family_benchmark",
            "comparison_scope": "mlforecast_statsforecast_pecnet",
        },
        "tier7": {
            "name": "Paper-style fixed chronological feature set",
            "feature_representation": (
                "Calendar/time features with fixed 2019-2024 train and "
                "2025-latest test design; ML/Stats AutoRegressive lags use "
                "1,2,3,4,5,10,15,20 and PECNet uses sampling periods 1 and 5."
            ),
            "hypothesis": (
                "A fixed chronological split tests out-of-sample behavior over a recent regime."
            ),
            "benchmark_role": "model_family_benchmark",
            "comparison_scope": "mlforecast_statsforecast_pecnet",
        },
        "tier8": {
            "name": "Paper-inspired PEC-WNN feature set with market index context",
            "feature_representation": (
                "Tier 7 calendar/time context plus market-index previous close; "
                "AAPL uses ^GSPC and BMW.DE uses ^GDAXI as exogenous context."
            ),
            "hypothesis": (
                "External market-index context can improve long chronological forecasting stability."
            ),
            "benchmark_role": "model_family_benchmark",
            "comparison_scope": "mlforecast_statsforecast_pecnet",
        },
        "all_tiers": {
            "name": "Root performance comparison",
            "feature_representation": (
                "Aggregated predictions from all configured tiers and model families."
            ),
            "hypothesis": (
                "A root evaluation run compares model outputs under a common metric format."
            ),
            "benchmark_role": "root_performance_measurement",
            "comparison_scope": "all_available_outputs",
        },
    }

    @classmethod
    def definition(cls, tier_name: str) -> dict[str, str]:
        return cls.DEFINITIONS.get(
            tier_name,
            {
                "name": tier_name,
                "feature_representation": "Custom stock-close forecasting tier.",
                "hypothesis": "Custom experiment configuration.",
                "benchmark_role": "custom",
                "comparison_scope": "custom",
            },
        )

    @classmethod
    def mlflow_tags(
        cls,
        *,
        tier_name: str,
        model_family: str,
        train_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
    ) -> dict[str, str]:
        definition = cls.definition(tier_name)
        tags: dict[str, str] = {
            "stock_close.workflow": "forecasting",
            "stock_close.tier": tier_name,
            "stock_close.model_family": model_family,
            "stock_close.tier.name": definition["name"],
            "stock_close.tier.feature_representation": definition[
                "feature_representation"
            ],
            "stock_close.tier.hypothesis": definition["hypothesis"],
            "stock_close.benchmark.role": definition["benchmark_role"],
            "stock_close.comparison.scope": definition["comparison_scope"],
            "stock_close.ucb_candidate": str(model_family == "pecnet").lower(),
        }
        tags.update(cls._dataset_tags(train_df=train_df, test_df=test_df))
        return tags

    @classmethod
    def run_note(
        cls,
        *,
        tier_name: str,
        model_family: str,
        train_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
    ) -> str:
        definition = cls.definition(tier_name)
        split_summary = cls._split_summary(train_df=train_df, test_df=test_df)
        return "\n".join(
            [
                f"Model family: {model_family}",
                f"Tier: {tier_name} - {definition['name']}",
                f"Feature representation: {definition['feature_representation']}",
                f"Hypothesis: {definition['hypothesis']}",
                f"Benchmark role: {definition['benchmark_role']}",
                f"Comparison scope: {definition['comparison_scope']}",
                split_summary,
            ]
        ).strip()

    @classmethod
    def context_payload(
        cls,
        *,
        tier_name: str,
        model_family: str,
        train_df: pd.DataFrame | None = None,
        test_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        return {
            "tier": tier_name,
            "model_family": model_family,
            "definition": cls.definition(tier_name),
            "dataset": cls._dataset_tags(train_df=train_df, test_df=test_df),
            "note": cls.run_note(
                tier_name=tier_name,
                model_family=model_family,
                train_df=train_df,
                test_df=test_df,
            ),
        }

    @classmethod
    def _dataset_tags(
        cls,
        *,
        train_df: pd.DataFrame | None,
        test_df: pd.DataFrame | None,
    ) -> dict[str, str]:
        tags: dict[str, str] = {}
        symbols = cls._symbols(train_df, test_df)
        if symbols:
            tags["stock_close.dataset.symbols"] = ",".join(symbols)
        tags.update(cls._frame_tags(train_df, prefix="stock_close.train"))
        tags.update(cls._frame_tags(test_df, prefix="stock_close.test"))
        return tags

    @staticmethod
    def _symbols(*frames: pd.DataFrame | None) -> list[str]:
        symbols: set[str] = set()
        for frame in frames:
            if frame is None or frame.empty or "unique_id" not in frame.columns:
                continue
            symbols.update(str(value) for value in frame["unique_id"].dropna().unique())
        return sorted(symbols)

    @staticmethod
    def _frame_tags(df: pd.DataFrame | None, *, prefix: str) -> dict[str, str]:
        if df is None or df.empty or "ds" not in df.columns:
            return {}
        import pandas as pd

        ds = pd.to_datetime(df["ds"], errors="coerce", utc=True).dropna()
        if ds.empty:
            return {f"{prefix}.rows": str(len(df))}
        return {
            f"{prefix}.rows": str(len(df)),
            f"{prefix}.start": str(ds.min().date()),
            f"{prefix}.end": str(ds.max().date()),
        }

    @classmethod
    def _split_summary(
        cls,
        *,
        train_df: pd.DataFrame | None,
        test_df: pd.DataFrame | None,
    ) -> str:
        train_tags = cls._frame_tags(train_df, prefix="train")
        test_tags = cls._frame_tags(test_df, prefix="test")
        if not train_tags and not test_tags:
            return "Split: not provided."
        return (
            "Split: "
            f"train {train_tags.get('train.start', '?')} to "
            f"{train_tags.get('train.end', '?')} "
            f"({train_tags.get('train.rows', '0')} rows); "
            f"test {test_tags.get('test.start', '?')} to "
            f"{test_tags.get('test.end', '?')} "
            f"({test_tags.get('test.rows', '0')} rows)."
        )
