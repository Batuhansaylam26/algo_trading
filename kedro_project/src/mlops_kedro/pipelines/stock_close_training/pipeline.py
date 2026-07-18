from kedro.pipeline import Pipeline, node

from .model_matrix import (
    model_matrix_nodes,
    model_matrix_summary_inputs,
    model_tiers,
    pecnet_only_tiers,
)
from .nodes import (
    stock_close_data_nodes,
    stock_close_model_nodes,
)


DATA_PREPROCESSING = "params:stock_close_data_preprocessing"

DELTA_LAKE = f"{DATA_PREPROCESSING}.delta_lake"
FEATURE_ENGINEERING = f"{DATA_PREPROCESSING}.feature_engineering"
CONVENTIONAL_GAP_TRADING = f"{DATA_PREPROCESSING}.conventional_gap_trading"
TIME_ENCODING = f"{DATA_PREPROCESSING}.time_encoding"
COLUMNS = f"{DATA_PREPROCESSING}.columns"


def _feature_engineering_context_nodes(*, load_silver: bool) -> list:
    nodes = [
        node(
            func=stock_close_data_nodes.configure_feature_engineering,
            inputs=[
                DELTA_LAKE,
                COLUMNS,
                TIME_ENCODING,
            ],
            outputs="stock_close_feature_engineering",
            name="configure_feature_engineering",
        ),
    ]
    if load_silver:
        nodes.append(
            node(
                func=stock_close_data_nodes.load_silver_stock_prices,
                inputs="stock_close_feature_engineering",
                outputs=[
                    "silver_stock_prices",
                    "silver_stock_prices_metadata",
                ],
                name="load_silver_stock_prices",
            )
        )
        nodes.append(
            node(
                func=stock_close_data_nodes.load_silver_stock_prices_weekly,
                inputs="stock_close_feature_engineering",
                outputs=[
                    "silver_stock_prices_weekly",
                    "silver_stock_prices_weekly_metadata",
                ],
                name="load_silver_stock_prices_weekly",
            )
        )
    return nodes


def _close_model_dataset_nodes() -> list:
    return [
        node(
            func=stock_close_data_nodes.prepare_close_model_dataset,
            inputs=[
                "stock_close_feature_engineering",
                "silver_stock_prices",
            ],
            outputs=[
                "stock_close_model_dataset",
                "close_model_dataset_metadata",
            ],
            name="prepare_close_model_dataset",
        ),
        node(
            func=stock_close_data_nodes.publish_close_model_dataset,
            inputs="stock_close_model_dataset",
            outputs="close_model_publish_metadata",
            name="publish_close_model_dataset",
        ),
    ]


def _indicator_feature_nodes() -> list:
    return [
        node(
            func=stock_close_data_nodes.prepare_indicator_features,
            inputs=[
                "stock_close_feature_engineering",
                FEATURE_ENGINEERING,
                "silver_stock_prices",
                "silver_stock_prices_weekly",
            ],
            outputs=[
                "stock_price_indicator_features",
                "indicator_feature_metadata",
                "stock_model_features",
                "stock_model_feature_metadata",
            ],
            name="prepare_indicator_features",
        ),
        node(
            func=stock_close_data_nodes.publish_indicator_model_features,
            inputs=[
                "stock_model_features",
                "stock_model_feature_metadata",
            ],
            outputs="model_feature_publish_metadata",
            name="publish_indicator_model_features",
        ),
    ]


def _load_indicator_feature_nodes() -> list:
    return [
        node(
            func=stock_close_data_nodes.load_indicator_features,
            inputs=[
                "stock_close_feature_engineering",
                FEATURE_ENGINEERING,
            ],
            outputs=[
                "stock_price_indicator_features",
                "indicator_feature_metadata",
            ],
            name="load_indicator_features",
        ),
    ]


def _conventional_gap_trading_nodes() -> list:
    return [
        node(
            func=stock_close_data_nodes.prepare_conventional_gap_trading,
            inputs=[
                "stock_price_indicator_features",
                CONVENTIONAL_GAP_TRADING,
                "stock_close_feature_engineering",
            ],
            outputs=[
                "conventional_gap_trading",
                "conventional_gap_trading_metadata",
            ],
            name="prepare_conventional_gap_trading",
        ),
        node(
            func=stock_close_data_nodes.publish_conventional_gap_trading,
            inputs=[
                "conventional_gap_trading",
                "conventional_gap_trading_metadata",
            ],
            outputs="conventional_gap_trading_publish_metadata",
            name="publish_conventional_gap_trading",
        ),
    ]


def _machine_learning_nodes(
    summary_output: str,
    *,
    wait_for_feature_publish: bool = False,
) -> list:
    return [
        *model_matrix_nodes(wait_for_feature_publish=wait_for_feature_publish),
        node(
            func=stock_close_model_nodes.evaluate_root_model_performance,
            inputs=_root_model_performance_inputs(),
            outputs=[
                "root_model_performance_predictions",
                "root_model_performance_regression_metrics",
                "root_model_performance_long_direction_metrics",
                "root_model_performance_metadata",
            ],
            name="evaluate_root_model_performance",
        ),
        node(
            func=stock_close_model_nodes.summarize_machine_learning,
            inputs=[
                *model_matrix_summary_inputs(),
                "root_model_performance_metadata",
            ],
            outputs=summary_output,
            name="summarize_machine_learning",
        ),
    ]


def _root_model_performance_inputs() -> dict[str, str]:
    inputs = {"mlflow_params": "params:stock_close_machine_learning.mlflow"}
    for tier_name in model_tiers():
        inputs[f"{tier_name}_train_test_split"] = (
            f"stock_close_{tier_name}_train_test_split"
        )
        for model_family in ["mlforecast", "statsforecast", "pecnet"]:
            inputs[f"{tier_name}_{model_family}_predictions"] = (
                f"stock_close_{tier_name}_{model_family}_predictions"
            )
    for tier_name in pecnet_only_tiers():
        inputs[f"{tier_name}_train_test_split"] = (
            f"stock_close_{tier_name}_train_test_split"
        )
        inputs[f"{tier_name}_pecnet_predictions"] = (
            f"stock_close_{tier_name}_pecnet_predictions"
        )
    return inputs


def create_feature_engineering_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            *_feature_engineering_context_nodes(load_silver=True),
            *_close_model_dataset_nodes(),
            *_indicator_feature_nodes(),
        ]
    )


def create_conventional_gap_trading_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            *_feature_engineering_context_nodes(load_silver=False),
            *_load_indicator_feature_nodes(),
            *_conventional_gap_trading_nodes(),
        ]
    )


def create_machine_learning_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            *_machine_learning_nodes("machine_learning_summary"),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            *_feature_engineering_context_nodes(load_silver=True),
            *_close_model_dataset_nodes(),
            *_indicator_feature_nodes(),
            *_conventional_gap_trading_nodes(),
            *_machine_learning_nodes(
                "machine_learning_summary",
                wait_for_feature_publish=True,
            ),
            node(
                func=stock_close_model_nodes.summarize_training,
                inputs=[
                    "close_model_dataset_metadata",
                    "close_model_publish_metadata",
                    "indicator_feature_metadata",
                    "model_feature_publish_metadata",
                    "conventional_gap_trading_metadata",
                    "conventional_gap_trading_publish_metadata",
                    "machine_learning_summary",
                ],
                outputs="stock_close_training_summary",
                name="summarize_training",
            ),
        ]
    )
