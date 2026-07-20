from __future__ import annotations

from .definitions_class import *  # noqa: F403
from .definitions_class import DagsterDefinitionsFactory

_running_in_container = DagsterDefinitionsFactory.running_in_container
_local_service_url = DagsterDefinitionsFactory.local_service_url
defs = DagsterDefinitionsFactory.build_definitions()
