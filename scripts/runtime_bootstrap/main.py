from __future__ import annotations

from .copying import (
    refresh_pecnet_runtime,
    refresh_tree_from_source,
    reset_directory,
    sync_kedro_runtime_overlay,
    try_runtime_step,
)
from .paths import (
    DAGSTER_APP_RUNTIME,
    DAGSTER_HOME,
    DAGSTER_SOURCE,
    DBT_RUNTIME,
    DUCKDB_RUNTIME,
    KEDRO_RUNTIME,
    KEDRO_SOURCE,
    PECNET_RUNTIME,
    REQUIREMENTS_RUNTIME,
    FEATURE_REPO_RUNTIME,
)
from .writers import (
    prepare_duckdb_runtime,
    write_dagster_home,
    write_dbt_runtime_project,
    write_feature_repo_runtime,
    write_requirements,
)


def main() -> None:
    write_feature_repo_runtime()
    kedro_ready = refresh_tree_from_source(KEDRO_SOURCE, KEDRO_RUNTIME, "Kedro runtime")
    kedro_overlay_ready = try_runtime_step(
        "Kedro runtime overlay copy",
        sync_kedro_runtime_overlay,
    )
    pecnet_ready = refresh_pecnet_runtime()

    dbt_ready = try_runtime_step(
        "dbt runtime project write",
        lambda: (reset_directory(DBT_RUNTIME), write_dbt_runtime_project()),
    )
    DAGSTER_APP_RUNTIME.mkdir(parents=True, exist_ok=True)
    dagster_ready = try_runtime_step(
        "Dagster app source copy",
        lambda: refresh_tree_from_source(
            DAGSTER_SOURCE / "src",
            DAGSTER_APP_RUNTIME / "src",
            "Dagster app source",
        ),
    )

    write_dagster_home()
    prepare_duckdb_runtime()
    requirements_ready = try_runtime_step("runtime requirements write", write_requirements)

    if dbt_ready:
        print(f"Wrote dbt runtime project to {DBT_RUNTIME}")
    else:
        print(f"Skipped dbt runtime project at {DBT_RUNTIME}")
    if dagster_ready:
        print(f"Copied Dagster app source to {DAGSTER_APP_RUNTIME}")
    else:
        print(f"Skipped Dagster app source at {DAGSTER_APP_RUNTIME}")
    if kedro_ready:
        print(f"Wrote Kedro runtime project to {KEDRO_RUNTIME}")
    else:
        print(f"Kept existing Kedro runtime project at {KEDRO_RUNTIME}")
    if kedro_overlay_ready:
        print("Synced selected Kedro runtime config and ML files")
    else:
        print("Skipped selected Kedro runtime config and ML file sync")
    if pecnet_ready:
        print(f"Wrote PecNet framework runtime to {PECNET_RUNTIME}")
    else:
        print(f"Kept existing PecNet framework runtime at {PECNET_RUNTIME}")
    print(f"Wrote Feast runtime repo to {FEATURE_REPO_RUNTIME}")
    print(f"Wrote Dagster runtime config to {DAGSTER_HOME}")
    print(f"Prepared DuckDB runtime at {DUCKDB_RUNTIME}")
    if requirements_ready:
        print(f"Wrote runtime requirements to {REQUIREMENTS_RUNTIME}")
    else:
        print(f"Skipped runtime requirements at {REQUIREMENTS_RUNTIME}")
