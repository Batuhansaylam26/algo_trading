from __future__ import annotations

from .models_class import *  # noqa: F403
from .models_class import StatsForecastModelFactory

stats_forecast_model_factory = StatsForecastModelFactory()
_model_class = stats_forecast_model_factory._model_class
build_statsforecast_models = stats_forecast_model_factory.build_statsforecast_models
