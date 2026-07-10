from kedro.pipeline import Pipeline, node

from .model_matrix import model_matrix_nodes, model_matrix_summary_inputs
from .nodes import (
    load_indicator_features,
    prepare_close_model_dataset,
    prepare_conventional_gap_trading,
    prepare_indicator_features,
    publish_close_model_dataset,
    publish_conventional_gap_trading,
    publish_indicator_model_features,
    start_config,
    summarize_machine_learning,
    summarize_training,
)


def _start_node():
    return node(
        func=start_config,
        inputs=[
            "params:stock_close_data_preprocessing",
            "params:stock_close_machine_learning",
        ],
        outputs="run_config",
        name="start",
    )


def _close_model_dataset_nodes() -> list:
    return [
        node(
            func=prepare_close_model_dataset,
            inputs="run_config",
            outputs=[
                "stock_close_model_dataset",
                "close_model_dataset_metadata",
            ],
            name="prepare_close_model_dataset",
        ),
        node(
            func=publish_close_model_dataset,
            inputs="stock_close_model_dataset",
            outputs="close_model_publish_metadata",
            name="publish_close_model_dataset",
        ),
    ]


def _indicator_feature_nodes() -> list:
    return [
        node(
            func=prepare_indicator_features,
            inputs="run_config",
            outputs=[
                "stock_price_indicator_features",
                "indicator_feature_metadata",
            ],
            name="prepare_indicator_features",
        ),
        node(
            func=publish_indicator_model_features,
            inputs=[
                "stock_price_indicator_features",
                "indicator_feature_metadata",
            ],
            outputs="model_feature_publish_metadata",
            name="publish_indicator_model_features",
        ),
    ]


def _load_indicator_feature_nodes() -> list:
    return [
        node(
            func=load_indicator_features,
            inputs="run_config",
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
            func=prepare_conventional_gap_trading,
            inputs=[
                "stock_price_indicator_features",
                "run_config",
            ],
            outputs=[
                "conventional_gap_trading",
                "conventional_gap_trading_metadata",
            ],
            name="prepare_conventional_gap_trading",
        ),
        node(
            func=publish_conventional_gap_trading,
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
            func=summarize_machine_learning,
            inputs=model_matrix_summary_inputs(),
            outputs=summary_output,
            name="summarize_machine_learning",
        ),
    ]


def create_feature_engineering_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            _start_node(),
            *_close_model_dataset_nodes(),
            *_indicator_feature_nodes(),
        ]
    )


def create_conventional_gap_trading_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            _start_node(),
            *_load_indicator_feature_nodes(),
            *_conventional_gap_trading_nodes(),
        ]
    )


def create_machine_learning_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            _start_node(),
            *_machine_learning_nodes("machine_learning_summary"),
        ]
    )


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            _start_node(),
            *_close_model_dataset_nodes(),
            *_indicator_feature_nodes(),
            *_conventional_gap_trading_nodes(),
            *_machine_learning_nodes(
                "machine_learning_summary",
                wait_for_feature_publish=True,
            ),
            node(
                func=summarize_training,
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
