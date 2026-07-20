from kedro.pipeline import Pipeline

from mlops_kedro.pipelines.stock_close_training.pipeline import (
    create_conventional_gap_trading_pipeline,
    create_feature_engineering_pipeline,
    create_mlflow_root_performance_pipeline,
    create_machine_learning_pipeline,
    create_pipeline,
)


class KedroPipelineRegistry:

    @staticmethod
    def register_pipelines() -> dict[str, Pipeline]:
        stock_close_training = create_pipeline()
        feature_engineering = create_feature_engineering_pipeline()
        conventional_gap_trading = create_conventional_gap_trading_pipeline()
        machine_learning = create_machine_learning_pipeline()
        mlflow_root_performance = create_mlflow_root_performance_pipeline()
        return {
            "__default__": stock_close_training,
            "stock_close_training": stock_close_training,
            "feature_engineering": feature_engineering,
            "conventional_gap_trading": conventional_gap_trading,
            "machine_learning": machine_learning,
            "mlflow_root_performance": mlflow_root_performance,
        }
