from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

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
DUCKDB_RUNTIME = Path("/opt/duckdb/dataops_mlops.duckdb")
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
    "conf/base/catalog.yml",
    "conf/base/parameters_data_preprocessing.yml",
    "conf/base/parameters_machine_learning.yml",
    "src/mlops_kedro/pipelines/stock_close_training/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/config_resolvers.py",
    "src/mlops_kedro/pipelines/stock_close_training/feature_engineering_oop.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/conditions.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/conventional_gap_pipeline.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/feature_sets.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/indicator_pipeline.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/indicators.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/lookback.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/model_dataset.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/pipelines.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/source.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/strategy.py",
    "src/mlops_kedro/pipelines/stock_close_training/features/time_encoding.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/common.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/metrics.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/mlforecast/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/mlforecast/data.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/mlforecast/models.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/mlforecast/service.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/mlforecast/spec.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/mlforecast/training.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/mlforecast/training_config.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/performance.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/performance_result.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/data/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/data/preprocessor.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/runtime.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/service.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/selection/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/selection/builder.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/selection/plots.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/selection/thesis.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/spec.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/tracking/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/tracking/wandb.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/training/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/training/performance.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/training/ticker.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/training/ticker_logs.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/pecnet/training/workflow.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/plots.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/runtime.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/statsforecast/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/statsforecast/models.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/statsforecast/service.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/statsforecast/spec.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/statsforecast/training.py",
    "src/mlops_kedro/pipelines/stock_close_training/ml/statsforecast/training_config.py",
    "src/mlops_kedro/pipelines/stock_close_training/model_matrix.py",
    "src/mlops_kedro/pipelines/stock_close_training/node_utils.py",
    "src/mlops_kedro/pipelines/stock_close_training/nodes.py",
    "src/mlops_kedro/pipelines/stock_close_training/nodes_data.py",
    "src/mlops_kedro/pipelines/stock_close_training/nodes_models.py",
    "src/mlops_kedro/pipelines/stock_close_training/pipeline.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/__init__.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/connections.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/constants.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/definitions.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/feast_store.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/loaders_close.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/loaders_model.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/publishers.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/schemas.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/service.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/transforms.py",
    "src/mlops_kedro/pipelines/stock_close_training/serving/writers.py",
]
