from __future__ import annotations

import errno
import shutil
import tomllib

from .copying import copy_tree, reset_directory
from .paths import (
    DAGSTER_HOME,
    DAGSTER_SOURCE,
    DBT_RUNTIME,
    DBT_SOURCE,
    DUCKDB_RUNTIME,
    FEATURE_REPO_RUNTIME,
    FEATURE_REPO_SOURCE,
    LEGACY_DUCKDB_RUNTIME,
    REQUIREMENTS_RUNTIME,
)
from .templates_dbt import DBT_RUNTIME_FILES
from .templates_feast import FEATURE_STORE_YAML, STOCK_FEATURES_PY









class RuntimeBootstrapWriter:

    @staticmethod
    def runtime_dependencies() -> list[str]:
        try:
            pyproject = tomllib.loads((DAGSTER_SOURCE / "pyproject.toml").read_text())
            return pyproject["project"]["dependencies"]
        except OSError as error:
            if error.errno != errno.EDEADLK:
                raise
            print("Using embedded runtime requirements: source pyproject hit EDEADLK")
            return [
                "dagster",
                "dagster-webserver",
                "dagster-dbt",
                "dagster-deltalake",
                "dagster-deltalake-polars",
                "dbt-core",
                "dbt-duckdb",
                "duckdb",
                "deltalake",
                "boto3",
                "feast[postgres,redis]",
                "pandas",
                "pandera[polars]",
                "polars",
                "polars-ta",
                "pyarrow",
                "protobuf>=6.33.5,<7",
                "grpcio==1.76.0",
                "grpcio-health-checking==1.76.0",
                "python-dotenv",
                "mlflow",
                "kedro==1.5.0",
                "kedro-viz",
                "mlforecast",
                "statsforecast",
                "optuna",
                "scikit-learn",
                "lightgbm",
                "xgboost",
                "catboost",
                "utilsforecast",
                "matplotlib",
                "seaborn",
                "tabulate",
                "torch",
                "PyWavelets",
                "psycopg[binary]",
                "psycopg-pool",
                "psycopg2-binary",
                "redis",
                "yahooquery",
            ]

    @staticmethod
    def write_requirements() -> None:
        REQUIREMENTS_RUNTIME.write_text("\n".join(RuntimeBootstrapWriter.runtime_dependencies()) + "\n")

    @staticmethod
    def write_dbt_runtime_project() -> None:
        try:
            copy_tree(DBT_SOURCE, DBT_RUNTIME)
            return
        except OSError as error:
            if error.errno != errno.EDEADLK:
                raise
            print("Falling back to embedded dbt project: source copy hit EDEADLK")
            reset_directory(DBT_RUNTIME)

        for relative_path, content in DBT_RUNTIME_FILES.items():
            path = DBT_RUNTIME / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)

    @staticmethod
    def write_feature_repo_runtime() -> None:
        reset_directory(FEATURE_REPO_RUNTIME)
        try:
            copy_tree(FEATURE_REPO_SOURCE, FEATURE_REPO_RUNTIME)
            (FEATURE_REPO_RUNTIME / "data").mkdir(parents=True, exist_ok=True)
            return
        except OSError as error:
            if error.errno != errno.EDEADLK:
                raise
            print("Falling back to embedded Feast repo: source copy hit EDEADLK")
            reset_directory(FEATURE_REPO_RUNTIME)

        (FEATURE_REPO_RUNTIME / "data").mkdir(parents=True, exist_ok=True)
        (FEATURE_REPO_RUNTIME / "feature_store.yaml").write_text(FEATURE_STORE_YAML)
        (FEATURE_REPO_RUNTIME / "stock_features.py").write_text(STOCK_FEATURES_PY)

    @staticmethod
    def write_dagster_home() -> None:
        DAGSTER_HOME.mkdir(parents=True, exist_ok=True)
        (DAGSTER_HOME / "dagster.yaml").write_text("telemetry:\n  enabled: false\n")

    @staticmethod
    def prepare_duckdb_runtime() -> None:
        DUCKDB_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
        if not DUCKDB_RUNTIME.exists() and LEGACY_DUCKDB_RUNTIME.exists():
            shutil.copy2(LEGACY_DUCKDB_RUNTIME, DUCKDB_RUNTIME)
