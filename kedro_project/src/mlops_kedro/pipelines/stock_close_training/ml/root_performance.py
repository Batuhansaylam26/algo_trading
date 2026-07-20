from __future__ import annotations

from .root_performance_class import *  # noqa: F403
from .root_performance_class import RootModelPerformanceEvaluator

root_model_performance_evaluator = RootModelPerformanceEvaluator()
_safe_metric_component = root_model_performance_evaluator._safe_metric_component
