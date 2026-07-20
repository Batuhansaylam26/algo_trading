from __future__ import annotations

from .dbt_assets_class import *  # noqa: F403
from .dbt_assets_class import DagsterDbtAssets

_run_dbt_parse = DagsterDbtAssets.run_dbt_parse
_ensure_manifest_path = DagsterDbtAssets.ensure_manifest_path
dbt_models = dbt_assets(
    manifest=DagsterDbtAssets.ensure_manifest_path(),
)(DagsterDbtAssets.dbt_models)
