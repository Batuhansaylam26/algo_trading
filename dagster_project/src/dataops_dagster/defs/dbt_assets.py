import os
from pathlib import Path
import subprocess

import dagster as dg
from dagster_dbt import DbtCliResource, dbt_assets


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DBT_PROJECT_DIR = Path(
    os.getenv("DBT_PROJECT_DIR", PROJECT_ROOT / "dbt_project")
).resolve()
DBT_PROFILES_DIR = Path(os.getenv("DBT_PROFILES_DIR", DBT_PROJECT_DIR)).resolve()
DBT_MANIFEST_PATH = Path(
    os.getenv("DBT_MANIFEST_PATH", DBT_PROJECT_DIR / "target" / "manifest.json")
).resolve()
LOCAL_DBT_MANIFEST_PATH = (
    PROJECT_ROOT / "dbt_project" / "target" / "manifest.json"
).resolve()


def _run_dbt_parse() -> None:
    subprocess.run(
        [
            "dbt",
            "parse",
            "--profiles-dir",
            str(DBT_PROFILES_DIR),
            "--project-dir",
            str(DBT_PROJECT_DIR),
        ],
        check=True,
    )


def _ensure_manifest_path() -> Path:
    if DBT_MANIFEST_PATH.exists():
        return DBT_MANIFEST_PATH

    parse_on_load = os.getenv("DBT_PARSE_ON_LOAD", "1").lower() in {
        "1",
        "true",
        "yes",
    }
    if parse_on_load and DBT_PROJECT_DIR.exists():
        _run_dbt_parse()
        if DBT_MANIFEST_PATH.exists():
            return DBT_MANIFEST_PATH

    if LOCAL_DBT_MANIFEST_PATH.exists():
        return LOCAL_DBT_MANIFEST_PATH

    return DBT_MANIFEST_PATH


@dbt_assets(
    manifest=_ensure_manifest_path(),
)
def dbt_models(context: dg.AssetExecutionContext, dbt: DbtCliResource):
    yield from dbt.cli(["build"], context=context).stream()
