from __future__ import annotations

from .main_class import *  # noqa: F403
from .main_class import RuntimeBootstrapRunner

runtime_bootstrap_runner = RuntimeBootstrapRunner()
main = runtime_bootstrap_runner.main
