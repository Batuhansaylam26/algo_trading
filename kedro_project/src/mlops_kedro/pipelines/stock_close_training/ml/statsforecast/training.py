from __future__ import annotations

from .training_class import *  # noqa: F403
from .training_class import StatsForecastTrainer

stats_forecast_trainer = StatsForecastTrainer()
train_statsforecast_models_from_split = stats_forecast_trainer.train_statsforecast_models_from_split
