from __future__ import annotations

from .common_class import *  # noqa: F403
from .common_class import MlCommon

ml_common = MlCommon()
resolve_local_service_url = ml_common.resolve_local_service_url
_running_in_container = ml_common._running_in_container
model_id_columns = ml_common.model_id_columns
non_feature_columns = ml_common.non_feature_columns
tier_experiment_name = ml_common.tier_experiment_name
configure_mlflow_tracking = ml_common.configure_mlflow_tracking
wait_for_mlflow_server = ml_common.wait_for_mlflow_server
split_train_test_by_horizon = ml_common.split_train_test_by_horizon
validation_reference_frame = ml_common.validation_reference_frame
log_mlflow_datasets = ml_common.log_mlflow_datasets
_log_dataset_tables_enabled = ml_common._log_dataset_tables_enabled
_prediction_frame = ml_common._prediction_frame
_regression_metrics = ml_common._regression_metrics
_regression_metrics_by_unique_id = ml_common._regression_metrics_by_unique_id
