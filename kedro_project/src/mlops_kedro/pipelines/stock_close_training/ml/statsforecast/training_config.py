from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class StatsForecastTrainingConfig:
    tier_name: str = "tier1"
    freq: str = "B"
    seasonal_length: int = 5
    autoregressive_lags: list[int] | None = None
    validation_horizon: int = 1
    test_horizon: int = 5
    conformal_n_windows: int = 3
    level: list[int] | None = None
    models: list[dict[str, Any]] | None = None
    verbose: bool = True

    @classmethod
    def from_spec(
        cls,
        model_spec: dict[str, Any],
    ) -> "StatsForecastTrainingConfig":
        return cls(
            tier_name=model_spec.get("tier_name", "tier1"),
            freq=model_spec["freq"],
            seasonal_length=model_spec["seasonal_length"],
            autoregressive_lags=model_spec.get("autoregressive_lags"),
            validation_horizon=model_spec.get("validation_horizon", 1),
            test_horizon=model_spec.get("test_horizon", 5),
            conformal_n_windows=model_spec["conformal_n_windows"],
            level=model_spec.get("level") or [80, 95],
            models=model_spec["models"],
            verbose=model_spec.get("verbose", True),
        )
