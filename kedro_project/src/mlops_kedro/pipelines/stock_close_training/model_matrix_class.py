from __future__ import annotations

from collections.abc import Callable
from functools import partial, update_wrapper
from typing import Any

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


class StockCloseModelMatrix:

    @staticmethod
    def model_tiers() -> tuple[str, ...]:
        return MODEL_TIER_NAMES

    @staticmethod
    def pecnet_only_tiers() -> tuple[str, ...]:
        return PECNET_ONLY_TIER_NAMES

    @staticmethod
    def _dataset(tier_name: str, name: str) -> str:
        return f"stock_close_{tier_name}_{name}"

    @staticmethod
    def _named_partial(function: Callable[..., Any], /, name: str, **kwargs: Any):
        wrapped = partial(function, **kwargs)
        update_wrapper(wrapped, function)
        wrapped.__name__ = name
        return wrapped

    @staticmethod
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
                func=StockCloseModelMatrix._named_partial(
                    load_func,
                    name=f"load_{tier_name}_training_dataset_fn",
                    tier_name=tier_name,
                ),
                inputs=load_inputs,
                outputs=[
                    StockCloseModelMatrix._dataset(tier_name, "training_dataset"),
                    StockCloseModelMatrix._dataset(tier_name, "training_dataset_metadata"),
                ],
                name=f"load_{tier_name}_training_dataset",
            ),
            node(
                func=StockCloseModelMatrix._named_partial(
                    stock_close_data_nodes.train_test_split_for_tier,
                    name=f"{tier_name}_train_test_split_fn",
                    tier_name=tier_name,
                ),
                inputs=[
                    StockCloseModelMatrix._dataset(tier_name, "training_dataset"),
                    MLFORECAST,
                ],
                outputs=[
                    StockCloseModelMatrix._dataset(tier_name, "train_test_split"),
                    StockCloseModelMatrix._dataset(tier_name, "train_test_split_metadata"),
                ],
                name=f"{tier_name}_train_test_split",
            ),
            *StockCloseModelMatrix._mlforecast_nodes(tier_name),
            *StockCloseModelMatrix._statsforecast_nodes(tier_name),
            *StockCloseModelMatrix._pecnet_nodes(tier_name),
        ]

    @staticmethod
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
                func=StockCloseModelMatrix._named_partial(
                    load_func,
                    name=f"load_{tier_name}_training_dataset_fn",
                    tier_name=tier_name,
                ),
                inputs=load_inputs,
                outputs=[
                    StockCloseModelMatrix._dataset(tier_name, "training_dataset"),
                    StockCloseModelMatrix._dataset(tier_name, "training_dataset_metadata"),
                ],
                name=f"load_{tier_name}_training_dataset",
            ),
            node(
                func=StockCloseModelMatrix._named_partial(
                    stock_close_data_nodes.train_test_split_for_tier,
                    name=f"{tier_name}_train_test_split_fn",
                    tier_name=tier_name,
                ),
                inputs=[
                    StockCloseModelMatrix._dataset(tier_name, "training_dataset"),
                    MLFORECAST,
                ],
                outputs=[
                    StockCloseModelMatrix._dataset(tier_name, "train_test_split"),
                    StockCloseModelMatrix._dataset(tier_name, "train_test_split_metadata"),
                ],
                name=f"{tier_name}_train_test_split",
            ),
            *StockCloseModelMatrix._pecnet_nodes(tier_name),
        ]

    @staticmethod
    def model_matrix_nodes(*, wait_for_feature_publish: bool = False) -> list:
        nodes = []
        for tier_name in StockCloseModelMatrix.model_tiers():
            nodes.extend(
                StockCloseModelMatrix.tier_machine_learning_nodes(
                    tier_name,
                    wait_for_feature_publish=wait_for_feature_publish,
                )
            )
        for tier_name in StockCloseModelMatrix.pecnet_only_tiers():
            nodes.extend(
                StockCloseModelMatrix.pecnet_only_tier_machine_learning_nodes(
                    tier_name,
                    wait_for_feature_publish=wait_for_feature_publish,
                )
            )
        return nodes

    @staticmethod
    def model_matrix_summary_inputs() -> list[str]:
        inputs = []
        for tier_name in StockCloseModelMatrix.model_tiers():
            inputs.extend(
                [
                    StockCloseModelMatrix._dataset(tier_name, "training_dataset_metadata"),
                    StockCloseModelMatrix._dataset(tier_name, "train_test_split_metadata"),
                    StockCloseModelMatrix._dataset(tier_name, "mlforecast_training_metadata"),
                    StockCloseModelMatrix._dataset(tier_name, "statsforecast_training_metadata"),
                    StockCloseModelMatrix._dataset(tier_name, "pecnet_training_metadata"),
                ]
            )
        for tier_name in StockCloseModelMatrix.pecnet_only_tiers():
            inputs.extend(
                [
                    StockCloseModelMatrix._dataset(tier_name, "training_dataset_metadata"),
                    StockCloseModelMatrix._dataset(tier_name, "train_test_split_metadata"),
                    StockCloseModelMatrix._dataset(tier_name, "pecnet_training_metadata"),
                ]
            )
        return inputs

    @staticmethod
    def _mlforecast_nodes(tier_name: str) -> list:
        return [
            node(
                func=StockCloseModelMatrix._named_partial(
                    stock_close_model_nodes.build_model_spec,
                    name=f"build_{tier_name}_mlforecast_model_spec_fn",
                    tier_name=tier_name,
                ),
                inputs=[
                    MLFORECAST,
                    MLFLOW,
                    RUNTIME,
                ],
                outputs=StockCloseModelMatrix._dataset(tier_name, "mlforecast_model_spec"),
                name=f"build_{tier_name}_mlforecast_model_spec",
            ),
            node(
                func=stock_close_model_nodes.train_models,
                inputs=[
                    StockCloseModelMatrix._dataset(tier_name, "train_test_split"),
                    StockCloseModelMatrix._dataset(tier_name, "mlforecast_model_spec"),
                ],
                outputs=[
                    StockCloseModelMatrix._dataset(tier_name, "mlforecast_regression_metrics"),
                    StockCloseModelMatrix._dataset(tier_name, "mlforecast_long_direction_metrics"),
                    StockCloseModelMatrix._dataset(tier_name, "mlforecast_predictions"),
                    StockCloseModelMatrix._dataset(tier_name, "mlforecast_training_metadata"),
                ],
                name=f"train_{tier_name}_mlforecast_models",
            ),
        ]

    @staticmethod
    def _statsforecast_nodes(tier_name: str) -> list:
        return [
            node(
                func=StockCloseModelMatrix._named_partial(
                    stock_close_model_nodes.build_statsforecast_model_spec_for_tier,
                    name=f"build_{tier_name}_statsforecast_model_spec_fn",
                    tier_name=tier_name,
                ),
                inputs=[
                    STATSFORECAST,
                    MLFORECAST,
                    MLFLOW,
                    RUNTIME,
                ],
                outputs=StockCloseModelMatrix._dataset(tier_name, "statsforecast_model_spec"),
                name=f"build_{tier_name}_statsforecast_model_spec",
            ),
            node(
                func=stock_close_model_nodes.train_statsforecast_models,
                inputs=[
                    StockCloseModelMatrix._dataset(tier_name, "train_test_split"),
                    StockCloseModelMatrix._dataset(tier_name, "statsforecast_model_spec"),
                ],
                outputs=[
                    StockCloseModelMatrix._dataset(tier_name, "statsforecast_regression_metrics"),
                    StockCloseModelMatrix._dataset(tier_name, "statsforecast_long_direction_metrics"),
                    StockCloseModelMatrix._dataset(tier_name, "statsforecast_predictions"),
                    StockCloseModelMatrix._dataset(tier_name, "statsforecast_training_metadata"),
                ],
                name=f"train_{tier_name}_statsforecast_models",
            ),
        ]

    @staticmethod
    def _pecnet_nodes(tier_name: str) -> list:
        return [
            node(
                func=StockCloseModelMatrix._named_partial(
                    stock_close_model_nodes.build_pecnet_model_spec_for_tier,
                    name=f"build_{tier_name}_pecnet_model_spec_fn",
                    tier_name=tier_name,
                ),
                inputs=[
                    PECNET,
                    MLFORECAST,
                    MLFLOW,
                    RUNTIME,
                    COLUMNS,
                ],
                outputs=StockCloseModelMatrix._dataset(tier_name, "pecnet_model_spec"),
                name=f"build_{tier_name}_pecnet_model_spec",
            ),
            node(
                func=stock_close_model_nodes.train_pecnet_models,
                inputs=[
                    StockCloseModelMatrix._dataset(tier_name, "train_test_split"),
                    StockCloseModelMatrix._dataset(tier_name, "pecnet_model_spec"),
                ],
                outputs=[
                    StockCloseModelMatrix._dataset(tier_name, "pecnet_regression_metrics"),
                    StockCloseModelMatrix._dataset(tier_name, "pecnet_long_direction_metrics"),
                    StockCloseModelMatrix._dataset(tier_name, "pecnet_predictions"),
                    StockCloseModelMatrix._dataset(tier_name, "pecnet_feature_selection"),
                    StockCloseModelMatrix._dataset(tier_name, "pecnet_training_metadata"),
                ],
                name=f"train_{tier_name}_pecnet_models",
            ),
        ]
