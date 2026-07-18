from __future__ import annotations

from functools import partial

from kedro.pipeline import node

from .features.feature_sets import MODEL_TIER_NAMES, PECNET_ONLY_TIER_NAMES
from .nodes import (
    stock_close_data_nodes,
    stock_close_model_nodes,
)


DATA_PREPROCESSING = "params:stock_close_data_preprocessing"
MACHINE_LEARNING = "params:stock_close_machine_learning"

COLUMNS = f"{DATA_PREPROCESSING}.columns"
TRAINING = f"{MACHINE_LEARNING}.training"
MLFLOW = f"{MACHINE_LEARNING}.mlflow"
MLFORECAST = f"{MACHINE_LEARNING}.mlforecast"
STATSFORECAST = f"{MACHINE_LEARNING}.statsforecast"
PECNET = f"{MACHINE_LEARNING}.pecnet"
RUNTIME = f"{MACHINE_LEARNING}.runtime"


def model_tiers() -> tuple[str, ...]:
    return MODEL_TIER_NAMES


def pecnet_only_tiers() -> tuple[str, ...]:
    return PECNET_ONLY_TIER_NAMES


def _dataset(tier_name: str, name: str) -> str:
    return f"stock_close_{tier_name}_{name}"


def tier_machine_learning_nodes(
    tier_name: str,
    *,
    wait_for_feature_publish: bool = False,
) -> list:
    load_inputs = (
        [COLUMNS, TRAINING, "model_feature_publish_metadata"]
        if wait_for_feature_publish
        else [COLUMNS, TRAINING]
    )
    load_func = (
        stock_close_data_nodes.load_model_training_dataset_after_feature_publish
        if wait_for_feature_publish
        else stock_close_data_nodes.load_model_training_dataset
    )

    return [
        node(
            func=partial(load_func, tier_name=tier_name),
            inputs=load_inputs,
            outputs=[
                _dataset(tier_name, "training_dataset"),
                _dataset(tier_name, "training_dataset_metadata"),
            ],
            name=f"load_{tier_name}_training_dataset",
        ),
        node(
            func=partial(
                stock_close_data_nodes.train_test_split_for_tier,
                tier_name=tier_name,
            ),
            inputs=[
                _dataset(tier_name, "training_dataset"),
                MLFORECAST,
            ],
            outputs=[
                _dataset(tier_name, "train_test_split"),
                _dataset(tier_name, "train_test_split_metadata"),
            ],
            name=f"{tier_name}_train_test_split",
        ),
        *_mlforecast_nodes(tier_name),
        *_statsforecast_nodes(tier_name),
        *_pecnet_nodes(tier_name),
    ]


def pecnet_only_tier_machine_learning_nodes(
    tier_name: str,
    *,
    wait_for_feature_publish: bool = False,
) -> list:
    load_inputs = (
        [COLUMNS, TRAINING, "model_feature_publish_metadata"]
        if wait_for_feature_publish
        else [COLUMNS, TRAINING]
    )
    load_func = (
        stock_close_data_nodes.load_model_training_dataset_after_feature_publish
        if wait_for_feature_publish
        else stock_close_data_nodes.load_model_training_dataset
    )

    return [
        node(
            func=partial(load_func, tier_name=tier_name),
            inputs=load_inputs,
            outputs=[
                _dataset(tier_name, "training_dataset"),
                _dataset(tier_name, "training_dataset_metadata"),
            ],
            name=f"load_{tier_name}_training_dataset",
        ),
        node(
            func=partial(
                stock_close_data_nodes.train_test_split_for_tier,
                tier_name=tier_name,
            ),
            inputs=[
                _dataset(tier_name, "training_dataset"),
                MLFORECAST,
            ],
            outputs=[
                _dataset(tier_name, "train_test_split"),
                _dataset(tier_name, "train_test_split_metadata"),
            ],
            name=f"{tier_name}_train_test_split",
        ),
        *_pecnet_nodes(tier_name),
    ]


def model_matrix_nodes(*, wait_for_feature_publish: bool = False) -> list:
    nodes = []
    for tier_name in model_tiers():
        nodes.extend(
            tier_machine_learning_nodes(
                tier_name,
                wait_for_feature_publish=wait_for_feature_publish,
            )
        )
    for tier_name in pecnet_only_tiers():
        nodes.extend(
            pecnet_only_tier_machine_learning_nodes(
                tier_name,
                wait_for_feature_publish=wait_for_feature_publish,
            )
        )
    return nodes


def model_matrix_summary_inputs() -> list[str]:
    inputs = []
    for tier_name in model_tiers():
        inputs.extend(
            [
                _dataset(tier_name, "training_dataset_metadata"),
                _dataset(tier_name, "train_test_split_metadata"),
                _dataset(tier_name, "mlforecast_training_metadata"),
                _dataset(tier_name, "statsforecast_training_metadata"),
                _dataset(tier_name, "pecnet_training_metadata"),
            ]
        )
    for tier_name in pecnet_only_tiers():
        inputs.extend(
            [
                _dataset(tier_name, "training_dataset_metadata"),
                _dataset(tier_name, "train_test_split_metadata"),
                _dataset(tier_name, "pecnet_training_metadata"),
            ]
        )
    return inputs


def _mlforecast_nodes(tier_name: str) -> list:
    return [
        node(
            func=partial(
                stock_close_model_nodes.build_model_spec,
                tier_name=tier_name,
            ),
            inputs=[
                MLFORECAST,
                MLFLOW,
                RUNTIME,
            ],
            outputs=_dataset(tier_name, "mlforecast_model_spec"),
            name=f"build_{tier_name}_mlforecast_model_spec",
        ),
        node(
            func=stock_close_model_nodes.train_models,
            inputs=[
                _dataset(tier_name, "train_test_split"),
                _dataset(tier_name, "mlforecast_model_spec"),
            ],
            outputs=[
                _dataset(tier_name, "mlforecast_regression_metrics"),
                _dataset(tier_name, "mlforecast_long_direction_metrics"),
                _dataset(tier_name, "mlforecast_predictions"),
                _dataset(tier_name, "mlforecast_training_metadata"),
            ],
            name=f"train_{tier_name}_mlforecast_models",
        ),
    ]


def _statsforecast_nodes(tier_name: str) -> list:
    return [
        node(
            func=partial(
                stock_close_model_nodes.build_statsforecast_model_spec_for_tier,
                tier_name=tier_name,
            ),
            inputs=[
                STATSFORECAST,
                MLFORECAST,
                MLFLOW,
                RUNTIME,
            ],
            outputs=_dataset(tier_name, "statsforecast_model_spec"),
            name=f"build_{tier_name}_statsforecast_model_spec",
        ),
        node(
            func=stock_close_model_nodes.train_statsforecast_models,
            inputs=[
                _dataset(tier_name, "train_test_split"),
                _dataset(tier_name, "statsforecast_model_spec"),
            ],
            outputs=[
                _dataset(tier_name, "statsforecast_regression_metrics"),
                _dataset(tier_name, "statsforecast_long_direction_metrics"),
                _dataset(tier_name, "statsforecast_predictions"),
                _dataset(tier_name, "statsforecast_training_metadata"),
            ],
            name=f"train_{tier_name}_statsforecast_models",
        ),
    ]


def _pecnet_nodes(tier_name: str) -> list:
    return [
        node(
            func=partial(
                stock_close_model_nodes.build_pecnet_model_spec_for_tier,
                tier_name=tier_name,
            ),
            inputs=[
                PECNET,
                MLFORECAST,
                MLFLOW,
                RUNTIME,
                COLUMNS,
            ],
            outputs=_dataset(tier_name, "pecnet_model_spec"),
            name=f"build_{tier_name}_pecnet_model_spec",
        ),
        node(
            func=stock_close_model_nodes.train_pecnet_models,
            inputs=[
                _dataset(tier_name, "train_test_split"),
                _dataset(tier_name, "pecnet_model_spec"),
            ],
            outputs=[
                _dataset(tier_name, "pecnet_regression_metrics"),
                _dataset(tier_name, "pecnet_long_direction_metrics"),
                _dataset(tier_name, "pecnet_predictions"),
                _dataset(tier_name, "pecnet_feature_selection"),
                _dataset(tier_name, "pecnet_training_metadata"),
            ],
            name=f"train_{tier_name}_pecnet_models",
        ),
    ]
