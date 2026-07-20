from __future__ import annotations

from typing import Any


class StatsForecastSpecBuilder:
    def build_spec(
        self,
        *,
        freq: str = "B",
        seasonal_length: int = 5,
        validation_horizon: int = 1,
        test_horizon: int = 5,
        conformal_n_windows: int = 3,
        level: list[int] | None = None,
        models: list[dict[str, Any]] | None = None,
        verbose: bool = True,
        tier_name: str = "tier1",
    ) -> dict[str, Any]:
        return {
            "tier_name": tier_name,
            "freq": freq,
            "seasonal_length": seasonal_length,
            "validation_horizon": validation_horizon,
            "test_horizon": test_horizon,
            "conformal_n_windows": conformal_n_windows,
            "level": level or [80, 95],
            "models": models
            or [
                {"class_name": "AutoARIMA", "alias": "AutoARIMA"},
                {"class_name": "AutoETS", "alias": "AutoETS"},
                {"class_name": "AutoTheta", "alias": "AutoTheta"},
                {"class_name": "AutoRegressive", "alias": "AutoRegressive"},
            ],
            "verbose": verbose,
        }



    @staticmethod
    def build_statsforecast_spec(
        *,
        freq: str = "B",
        seasonal_length: int = 5,
        validation_horizon: int = 1,
        test_horizon: int = 5,
        conformal_n_windows: int = 3,
        level: list[int] | None = None,
        models: list[dict[str, Any]] | None = None,
        verbose: bool = True,
        tier_name: str = "tier1",
    ) -> dict[str, Any]:
        return StatsForecastSpecBuilder().build_spec(
            freq=freq,
            seasonal_length=seasonal_length,
            validation_horizon=validation_horizon,
            test_horizon=test_horizon,
            conformal_n_windows=conformal_n_windows,
            level=level,
            models=models,
            verbose=verbose,
            tier_name=tier_name,
        )
