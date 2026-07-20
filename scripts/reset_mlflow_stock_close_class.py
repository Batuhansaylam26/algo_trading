from __future__ import annotations

import os

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient


TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://host.docker.internal:5001")
EXPERIMENT_PREFIXES = ("stock_close",)
MODEL_NAME_MARKERS = ("stock", "pecnet", "mlforecast", "statsforecast")












class MlflowStockCloseResetter:

    @staticmethod
    def _should_reset_experiment(name: str) -> bool:
        return name != "Default" and name.startswith(EXPERIMENT_PREFIXES)

    @staticmethod
    def _should_delete_registered_model(name: str) -> bool:
        lowered = name.lower()
        return any(marker in lowered for marker in MODEL_NAME_MARKERS)

    @staticmethod
    def _stock_close_experiments(client: MlflowClient):
        return [
            experiment
            for experiment in client.search_experiments(view_type=ViewType.ACTIVE_ONLY)
            if MlflowStockCloseResetter._should_reset_experiment(experiment.name)
        ]

    @staticmethod
    def main() -> None:
        client = MlflowClient(tracking_uri=TRACKING_URI)

        deleted_model_versions = 0
        deleted_registered_models = []
        for registered_model in client.search_registered_models():
            if not MlflowStockCloseResetter._should_delete_registered_model(registered_model.name):
                continue

            model_versions = client.search_model_versions(
                filter_string=f"name = '{registered_model.name}'"
            )
            for model_version in model_versions:
                client.delete_model_version(
                    name=registered_model.name,
                    version=model_version.version,
                )
                deleted_model_versions += 1

            client.delete_registered_model(registered_model.name)
            deleted_registered_models.append(registered_model.name)

        restored_experiments = []
        for experiment in client.search_experiments(view_type=ViewType.DELETED_ONLY):
            if not MlflowStockCloseResetter._should_reset_experiment(experiment.name):
                continue
            client.restore_experiment(experiment.experiment_id)
            restored_experiments.append(experiment.name)

        deleted_logged_models = []
        stock_experiments = MlflowStockCloseResetter._stock_close_experiments(client)
        stock_experiment_ids = [experiment.experiment_id for experiment in stock_experiments]
        if stock_experiment_ids:
            page_token = None
            while True:
                logged_models = client.search_logged_models(
                    experiment_ids=stock_experiment_ids,
                    max_results=1000,
                    page_token=page_token,
                )
                for logged_model in logged_models:
                    client.delete_logged_model(logged_model.model_id)
                    deleted_logged_models.append(logged_model.name)

                page_token = getattr(logged_models, "token", None)
                if not page_token:
                    break

        deleted_runs = 0
        reset_experiments = []
        for experiment in MlflowStockCloseResetter._stock_close_experiments(client):
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                run_view_type=ViewType.ACTIVE_ONLY,
                max_results=50000,
            )
            for run in runs:
                client.delete_run(run.info.run_id)
                deleted_runs += 1

            reset_experiments.append(experiment.name)

        print(f"tracking_uri={TRACKING_URI}")
        print(f"deleted_registered_models={deleted_registered_models}")
        print(f"deleted_model_versions={deleted_model_versions}")
        print(f"deleted_logged_models={deleted_logged_models}")
        print(f"restored_experiments={restored_experiments}")
        print(f"reset_experiments={reset_experiments}")
        print(f"deleted_runs={deleted_runs}")
