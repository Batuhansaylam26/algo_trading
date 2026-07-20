from __future__ import annotations

from .pipeline_registry_class import *  # noqa: F403
from .pipeline_registry_class import KedroPipelineRegistry

kedro_pipeline_registry = KedroPipelineRegistry()
register_pipelines = kedro_pipeline_registry.register_pipelines
