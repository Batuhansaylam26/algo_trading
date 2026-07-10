from pathlib import Path
import errno
import shutil
import time
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DBT_SOURCE = PROJECT_ROOT / "dbt_project"
DAGSTER_SOURCE = PROJECT_ROOT / "dagster_project"
FEATURE_REPO_SOURCE = PROJECT_ROOT / "feature_repo"
KEDRO_SOURCE = PROJECT_ROOT / "kedro_project"
PECNET_SOURCE = PROJECT_ROOT / "pecnetframework"

DBT_RUNTIME = Path("/opt/dbt_project")
DAGSTER_APP_RUNTIME = Path("/opt/dataops_app")
DAGSTER_HOME = Path("/opt/dagster_home")
FEATURE_REPO_RUNTIME = Path("/opt/feature_repo")
KEDRO_RUNTIME = Path("/opt/kedro_project")
PECNET_RUNTIME = Path("/opt/pecnetframework")
REQUIREMENTS_RUNTIME = Path("/opt/requirements-runtime.txt")
DUCKDB_RUNTIME = Path(
    "/workspaces/yahooquery_lakehouse_revamp/database/duckdb/dataops_mlops.duckdb"
)
LEGACY_DUCKDB_RUNTIME = Path("/opt/dataops_mlops.duckdb")

EXCLUDED_DIRS = {
    ".dbt",
    ".git",
    ".idea",
    ".pytest_cache",
    "__pycache__",
    "dbt_packages",
    "logs",
    "target",
}
EXCLUDED_SUFFIXES = {".pyc"}
EXCLUDED_FILES = {".DS_Store"}
DEADLOCK_RETRY_COUNT = 5
DEADLOCK_RETRY_SECONDS = 0.25
PECNET_DEADLOCK_RETRY_COUNT = 30

PECNET_RUNTIME_FILES = [
    "pecnet/__init__.py",
    "pecnet/models/BasicNN.py",
    "pecnet/models/__init__.py",
    "pecnet/network/ErrorNetwork.py",
    "pecnet/network/FinalNetwork.py",
    "pecnet/network/ModelLoader.py",
    "pecnet/network/Pecnet.py",
    "pecnet/network/PecnetBuilder.py",
    "pecnet/network/VariableNetwork.py",
    "pecnet/network/__init__.py",
    "pecnet/preprocessing/DataPreprocessor.py",
    "pecnet/preprocessing/Imputators.py",
    "pecnet/preprocessing/Normalizers.py",
    "pecnet/preprocessing/PreprocessArtifacts.py",
    "pecnet/preprocessing/__init__.py",
    "pecnet/utils/FeatureSelector.py",
    "pecnet/utils/Utility.py",
    "pecnet/utils/__init__.py",
]

KEDRO_RUNTIME_OVERLAY_FILES = [
    "conf/base/parameters_data_preprocessing.yml",
    "conf/base/parameters_machine_learning.yml",
    "conf/local/credentials.yml",
    "src/mlops_kedro/pipelines/stock_close_training/nodes.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/common.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/mlforecast_training.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet_training.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/stats_training.py",
]

DBT_RUNTIME_FILES = {
    "dbt_project.yml": """name: "dbt_project"
version: "1.0.0"
profile: "dataops_mlops"

model-paths: ["models"]
analysis-paths: ["analyses"]
test-paths: ["tests"]
seed-paths: ["seeds"]
macro-paths: ["macros"]
snapshot-paths: ["snapshots"]

clean-targets:
  - "target"
  - "dbt_packages"

models:
  dbt_project:
    query_layer:
      +materialized: view
      +schema: query
    marts:
      +materialized: view
      +schema: marts
""",
    "profiles.yml": """dataops_mlops:
  target: dev
  outputs:
    dev:
      type: duckdb
      path: "{{ env_var('DBT_DUCKDB_PATH', './dataops_mlops.duckdb') }}"
      threads: 4

      extensions:
        - httpfs
        - delta

      secrets:
        - type: s3
          provider: config
          key_id: "{{ env_var('DELTA_LAKE_S3_ACCESS_KEY', 'admin') }}"
          secret: "{{ env_var('DELTA_LAKE_S3_SECRET_KEY', 'admin1234') }}"
          region: us-east-1
          endpoint: host.docker.internal:9000
          use_ssl: false
          url_style: path
          scope: "s3://{{ env_var('DELTA_LAKE_S3_BUCKET', 'delta-lake-bucket') }}"
""",
    "models/query_layer/read_silver_stock_prices.sql": """{{ config(materialized="view", schema="query") }}

-- depends_on: {{ source("silver", "stock_prices") }}

select
    symbol,
    date,
    extract(year from date)::integer as year,
    extract(month from date)::integer as month,
    extract(day from date)::integer as day,
    open,
    high,
    low,
    close,
    volume
from delta_scan('s3://{{ env_var("DELTA_LAKE_S3_BUCKET", "delta-lake-bucket") }}/silver/stock_prices')
""",
    "models/query_layer/schema.yml": """version: 2

models:
  - name: read_silver_stock_prices
    description: "Query-layer DuckDB view over the validated silver Delta table."
    columns:
      - name: symbol
        description: "Ticker symbol."
      - name: date
        description: "Quote timestamp in UTC without timezone."
      - name: year
        description: "Year extracted from the quote timestamp."
      - name: month
        description: "Month extracted from the quote timestamp."
      - name: day
        description: "Day of month extracted from the quote timestamp."
      - name: open
        description: "Open price."
      - name: high
        description: "High price."
      - name: low
        description: "Low price."
      - name: close
        description: "Close price."
      - name: volume
        description: "Traded volume."
""",
    "models/query_layer/sources.yml": """version: 2

sources:
  - name: silver
    description: "Validated and calendar-enriched Delta Lake silver layer written by Polars and Pandera."
    tables:
      - name: stock_prices
        description: "Cleaned, deduplicated, validated, and calendar-enriched intraday stock prices."
        meta:
          external_location: "s3://delta-lake-bucket/silver/stock_prices"
          format: "delta"
          dagster:
            asset_key:
              - silver
              - stock_prices
""",
    "models/marts/stock_price_daily.sql": """{{ config(materialized="view", schema="marts") }}

select
    symbol,
    year,
    month,
    day,
    cast(date as date) as trading_date,
    min(low) as low,
    max(high) as high,
    avg(open) as avg_open,
    avg(close) as avg_close,
    sum(volume) as total_volume,
    count(*) as bar_count
from {{ ref("read_silver_stock_prices") }}
group by 1, 2, 3, 4, 5
""",
    "models/marts/schema.yml": """version: 2

models:
  - name: stock_price_daily
    description: "Daily stock-price mart built from the silver query-layer view."
    columns:
      - name: symbol
        description: "Ticker symbol."
      - name: year
        description: "Trading year."
      - name: month
        description: "Trading month."
      - name: day
        description: "Trading day of month."
      - name: trading_date
        description: "Trading date."
      - name: low
        description: "Daily low."
      - name: high
        description: "Daily high."
      - name: avg_open
        description: "Average intraday open price."
      - name: avg_close
        description: "Average intraday close price."
      - name: total_volume
        description: "Total daily volume."
      - name: bar_count
        description: "Number of intraday bars in the day."
""",
}

FEATURE_STORE_YAML = """project: dataops_mlops
registry: data/registry.db
provider: local

offline_store:
  type: postgres
  host: host.docker.internal
  port: 5432
  database: dataops
  db_schema: feature_store
  user: dataops
  password: dataops
  sslmode: disable
  entity_select_mode: temp_table

online_store:
  type: redis
  connection_string: "host.docker.internal:6379"

entity_key_serialization_version: 3
"""

STOCK_FEATURES_PY = '''from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import (
    PostgreSQLSource,
)
from feast.types import Float64, Int64


TIER_1_FEATURES = [
    "prev_open",
    "prev_close",
    "prev_high",
    "prev_low",
    "prev_volume",
]

TIER_2_TIME_FEATURES = [
    "calendar_gap_days",
    "month_sin_1",
    "month_cos_1",
    "month_sin_2",
    "month_cos_2",
    "day_sin_1",
    "day_cos_1",
    "day_sin_2",
    "day_cos_2",
    "day_of_year_sin_1",
    "day_of_year_cos_1",
    "day_of_year_sin_2",
    "day_of_year_cos_2",
]

TIER_2_FEATURES = [
    *TIER_1_FEATURES,
    *TIER_2_TIME_FEATURES,
]


ticker = Entity(name="ticker", join_keys=["symbol"])

stock_model_source = PostgreSQLSource(
    name="stock_model_features_source",
    query="SELECT * FROM feature_store.stock_model_features",
    timestamp_field="date",
    created_timestamp_column="created_timestamp",
)

stock_model_features_view = FeatureView(
    name="stock_model_features",
    entities=[ticker],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="prev_open", dtype=Float64),
        Field(name="prev_close", dtype=Float64),
        Field(name="prev_high", dtype=Float64),
        Field(name="prev_low", dtype=Float64),
        Field(name="prev_volume", dtype=Float64),
        Field(name="calendar_gap_days", dtype=Int64),
        Field(name="month_sin_1", dtype=Float64),
        Field(name="month_cos_1", dtype=Float64),
        Field(name="month_sin_2", dtype=Float64),
        Field(name="month_cos_2", dtype=Float64),
        Field(name="day_sin_1", dtype=Float64),
        Field(name="day_cos_1", dtype=Float64),
        Field(name="day_sin_2", dtype=Float64),
        Field(name="day_cos_2", dtype=Float64),
        Field(name="day_of_year_sin_1", dtype=Float64),
        Field(name="day_of_year_cos_1", dtype=Float64),
        Field(name="day_of_year_sin_2", dtype=Float64),
        Field(name="day_of_year_cos_2", dtype=Float64),
    ],
    online=True,
    source=stock_model_source,
    tags={"layer": "feature_serving", "team": "dataops_mlops"},
)

stock_model_tier_1_feature_service = FeatureService(
    name="stock_model_tier_1_features_v1",
    features=[stock_model_features_view[TIER_1_FEATURES]],
)

stock_model_tier_2_feature_service = FeatureService(
    name="stock_model_tier_2_features_v1",
    features=[stock_model_features_view[TIER_2_FEATURES]],
)


stock_feature_row = Entity(name="stock_feature_row", join_keys=["feature_key"])

stock_model_tier2_dataset_source = PostgreSQLSource(
    name="stock_model_tier2_dataset_source",
    query="""
        SELECT
            *,
            to_char(
                "date" AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            ) AS date_key,
            symbol || '|' || to_char(
                "date" AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            ) AS feature_key
        FROM feature_store.stock_model_features
    """,
    timestamp_field="date",
    created_timestamp_column="created_timestamp",
)

stock_model_tier2_dataset_view = FeatureView(
    name="stock_model_tier2_dataset",
    entities=[stock_feature_row],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="prev_open", dtype=Float64),
        Field(name="prev_close", dtype=Float64),
        Field(name="prev_high", dtype=Float64),
        Field(name="prev_low", dtype=Float64),
        Field(name="prev_volume", dtype=Float64),
        Field(name="calendar_gap_days", dtype=Int64),
        Field(name="month_sin_1", dtype=Float64),
        Field(name="month_cos_1", dtype=Float64),
        Field(name="month_sin_2", dtype=Float64),
        Field(name="month_cos_2", dtype=Float64),
        Field(name="day_sin_1", dtype=Float64),
        Field(name="day_cos_1", dtype=Float64),
        Field(name="day_sin_2", dtype=Float64),
        Field(name="day_cos_2", dtype=Float64),
        Field(name="day_of_year_sin_1", dtype=Float64),
        Field(name="day_of_year_cos_1", dtype=Float64),
        Field(name="day_of_year_sin_2", dtype=Float64),
        Field(name="day_of_year_cos_2", dtype=Float64),
    ],
    online=True,
    source=stock_model_tier2_dataset_source,
    tags={"layer": "model_training", "team": "dataops_mlops"},
)

stock_model_tier2_dataset_service = FeatureService(
    name="stock_model_tier2_dataset_v1",
    features=[stock_model_tier2_dataset_view[TIER_2_FEATURES]],
)


stock_series = Entity(name="stock_series", join_keys=["series_key"])

stock_close_model_dataset_source = PostgreSQLSource(
    name="stock_close_model_dataset_source",
    query="""
        SELECT
            *,
            to_char(
                ds AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            ) AS ds_key,
            unique_id || '|' || to_char(
                ds AT TIME ZONE 'UTC',
                'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'
            ) AS series_key
        FROM feature_store.stock_close_model_dataset
    """,
    timestamp_field="ds",
    created_timestamp_column="created_timestamp",
)

stock_close_model_dataset_view = FeatureView(
    name="stock_close_model_dataset",
    entities=[stock_series],
    ttl=timedelta(days=3650),
    schema=[
        Field(name="y", dtype=Float64),
        Field(name="month_sin_1", dtype=Float64),
        Field(name="month_cos_1", dtype=Float64),
        Field(name="day_sin_1", dtype=Float64),
        Field(name="day_cos_1", dtype=Float64),
        Field(name="day_of_year_sin_1", dtype=Float64),
        Field(name="day_of_year_cos_1", dtype=Float64),
    ],
    online=True,
    source=stock_close_model_dataset_source,
    tags={"layer": "model_training", "team": "dataops_mlops"},
)

stock_close_model_dataset_service = FeatureService(
    name="stock_close_model_dataset_v1",
    features=[
        stock_close_model_dataset_view[
            [
                "y",
                "month_sin_1",
                "month_cos_1",
                "day_sin_1",
                "day_cos_1",
                "day_of_year_sin_1",
                "day_of_year_cos_1",
            ]
        ]
    ],
)
'''


def should_skip(path: Path) -> bool:
    return (
        any(part in EXCLUDED_DIRS for part in path.parts)
        or path.name in EXCLUDED_FILES
        or path.suffix in EXCLUDED_SUFFIXES
    )


def reset_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError as error:
            if error.errno == errno.EBUSY:
                print(f"Skipping busy runtime path during reset: {child}")
                continue
            raise


def copy_file_with_retry(
    source: Path,
    target: Path,
    *,
    retry_count: int = DEADLOCK_RETRY_COUNT,
) -> None:
    for attempt in range(1, retry_count + 1):
        try:
            shutil.copy2(source, target)
            return
        except OSError as error:
            if error.errno != errno.EDEADLK or attempt == retry_count:
                raise
            time.sleep(DEADLOCK_RETRY_SECONDS * attempt)


def copy_tree(source: Path, target: Path) -> None:
    for path in source.rglob("*"):
        relative_path = path.relative_to(source)
        if should_skip(relative_path):
            continue
        target_path = target / relative_path
        if path.is_dir():
            target_path.mkdir(parents=True, exist_ok=True)
            continue
        target_path.parent.mkdir(parents=True, exist_ok=True)
        copy_file_with_retry(path, target_path)


def copy_selected_files(
    source: Path,
    target: Path,
    relative_paths: list[str],
    *,
    retry_count: int = DEADLOCK_RETRY_COUNT,
) -> None:
    for relative_path in relative_paths:
        source_path = source / relative_path
        target_path = target / relative_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        copy_file_with_retry(source_path, target_path, retry_count=retry_count)


def has_runtime_contents(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def refresh_tree_from_source(source: Path, target: Path, description: str) -> bool:
    staging = target.parent / f".{target.name}.next"
    reset_directory(staging)

    try:
        copy_tree(source, staging)
    except OSError as error:
        if error.errno != errno.EDEADLK:
            raise
        reset_directory(staging)
        if has_runtime_contents(target):
            print(f"Kept existing {description}: source copy hit EDEADLK")
            return False
        raise

    reset_directory(target)
    for child in staging.iterdir():
        shutil.move(str(child), target / child.name)
    staging.rmdir()
    return True


def refresh_pecnet_runtime() -> bool:
    staging = PECNET_RUNTIME.parent / f".{PECNET_RUNTIME.name}.next"
    reset_directory(staging)

    try:
        copy_selected_files(
            PECNET_SOURCE,
            staging,
            PECNET_RUNTIME_FILES,
            retry_count=PECNET_DEADLOCK_RETRY_COUNT,
        )
    except OSError as error:
        if error.errno != errno.EDEADLK:
            raise
        reset_directory(staging)
        if has_runtime_contents(PECNET_RUNTIME):
            print("Kept existing PecNet framework runtime: source copy hit EDEADLK")
            return False
        raise

    reset_directory(PECNET_RUNTIME)
    for child in staging.iterdir():
        shutil.move(str(child), PECNET_RUNTIME / child.name)
    staging.rmdir()
    return True


def try_runtime_step(description: str, function) -> bool:
    try:
        result = function()
    except OSError as error:
        if error.errno == errno.EDEADLK:
            print(f"Skipped {description}: resource deadlock avoided")
            return False
        raise
    return bool(result) if result is not None else True


def sync_kedro_runtime_overlay() -> None:
    existing_files = [
        relative_path
        for relative_path in KEDRO_RUNTIME_OVERLAY_FILES
        if (KEDRO_SOURCE / relative_path).exists()
    ]
    copy_selected_files(KEDRO_SOURCE, KEDRO_RUNTIME, existing_files)


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
            "wandb",
            "yahooquery",
        ]


def write_requirements() -> None:
    REQUIREMENTS_RUNTIME.write_text("\n".join(runtime_dependencies()) + "\n")


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


def write_dagster_home() -> None:
    DAGSTER_HOME.mkdir(parents=True, exist_ok=True)
    (DAGSTER_HOME / "dagster.yaml").write_text("telemetry:\n  enabled: false\n")


def prepare_duckdb_runtime() -> None:
    DUCKDB_RUNTIME.parent.mkdir(parents=True, exist_ok=True)
    if not DUCKDB_RUNTIME.exists() and LEGACY_DUCKDB_RUNTIME.exists():
        shutil.copy2(LEGACY_DUCKDB_RUNTIME, DUCKDB_RUNTIME)


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


if __name__ == "__main__":
    main()
