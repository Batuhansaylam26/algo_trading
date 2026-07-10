from __future__ import annotations

from typing import Any

import statsforecast.models as stats_models
from statsforecast.utils import ConformalIntervals


SUPPORTED_STATSFORECAST_MODELS = {
    "AutoARIMA",
    "AutoCES",
    "AutoETS",
    "AutoMFLES",
    "AutoRegressive",
    "AutoTBATS",
    "AutoTheta",
}


def _model_class(class_name: str):
    if class_name not in SUPPORTED_STATSFORECAST_MODELS:
        raise ValueError(
            f"Unsupported StatsForecast model {class_name!r}. "
            f"Supported models: {sorted(SUPPORTED_STATSFORECAST_MODELS)}"
        )
    try:
        return getattr(stats_models, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"StatsForecast does not expose {class_name!r} in this installation."
        ) from exc


def build_statsforecast_models(
    *,
    seasonal_length: int,
    horizon: int,
    conformal_n_windows: int,
    models_config: list[dict[str, Any]],
) -> list[Any]:
    def prediction_intervals() -> ConformalIntervals:
        return ConformalIntervals(
            h=horizon,
            n_windows=conformal_n_windows,
        )

    def defaults_for(class_name: str) -> dict[str, Any]:
        if class_name in {
            "AutoARIMA",
            "AutoCES",
            "AutoETS",
            "AutoMFLES",
            "AutoTBATS",
            "AutoTheta",
        }:
            return {
                "season_length": seasonal_length,
                "prediction_intervals": prediction_intervals(),
            }
        if class_name == "AutoRegressive":
            return {
                "lags": sorted({1, seasonal_length}),
                "prediction_intervals": prediction_intervals(),
            }
        return {}

    built_models = []
    for model_config in models_config:
        class_name = model_config["class_name"]
        alias = model_config.get("alias", class_name)
        kwargs = model_config.get("kwargs") or {}
        model_class = _model_class(class_name)
        built_models.append(
            model_class(
                **defaults_for(class_name),
                **kwargs,
                alias=alias,
            )
        )

    return built_models
