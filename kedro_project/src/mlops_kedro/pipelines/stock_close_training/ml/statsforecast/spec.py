from __future__ import annotations

from .spec_class import *  # noqa: F403
from .spec_class import StatsForecastSpecBuilder

stats_forecast_spec_builder = StatsForecastSpecBuilder()
build_statsforecast_spec = stats_forecast_spec_builder.build_statsforecast_spec
