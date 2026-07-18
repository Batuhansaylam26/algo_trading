from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MLForecastTrainingConfig:
    tier_name: str = "tier1"
    freq: str = "B"
    validation_horizon: int = 1
    test_horizon: int = 5
    n_windows: int = 3
    n_trials: int = 20
    level: list[int] | None = None
    verbose: bool = True

    @classmethod
    def from_inputs(
        cls,
        *,
        model_spec: dict[str, Any] | None,
        freq: str,
        validation_horizon: int,
        test_horizon: int,
        n_windows: int,
        n_trials: int,
        level: list[int] | None,
        verbose: bool,
    ) -> "MLForecastTrainingConfig":
        if not model_spec:
            return cls(
                freq=freq,
                validation_horizon=validation_horizon,
                test_horizon=test_horizon,
                n_windows=n_windows,
                n_trials=n_trials,
                level=level or [80, 95],
                verbose=verbose,
            )

        return cls(
            tier_name=model_spec.get("tier_name", "tier1"),
            freq=model_spec.get("freq", freq),
            validation_horizon=model_spec.get(
                "validation_horizon",
                validation_horizon,
            ),
            test_horizon=model_spec.get("test_horizon", test_horizon),
            n_windows=model_spec.get("n_windows", n_windows),
            n_trials=model_spec.get("n_trials", n_trials),
            level=model_spec.get("level", level) or [80, 95],
            verbose=model_spec.get("verbose", verbose),
        )
