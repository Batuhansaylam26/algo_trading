from __future__ import annotations

from .training_class import *  # noqa: F403
from .training_class import MLForecastTrainer

mlforecast_trainer = MLForecastTrainer()
train_auto_mlforecast_models_from_split = mlforecast_trainer.train_auto_mlforecast_models_from_split
train_auto_mlforecast_models = mlforecast_trainer.train_auto_mlforecast_models
