from __future__ import annotations

from .runtime_class import *  # noqa: F403
from .runtime_class import TrainingRuntime

training_runtime = TrainingRuntime()
cpu_count_from_env = training_runtime.cpu_count_from_env
bool_from_env = training_runtime.bool_from_env
filter_sklearn_parallel_warnings = training_runtime.filter_sklearn_parallel_warnings
