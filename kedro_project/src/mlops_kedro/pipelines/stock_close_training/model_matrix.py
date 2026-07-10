from __future__ import annotations

from functools import partial

from kedro.pipeline import node

from .nodes import (
    build_model_spec,
    build_pecnet_model_spec_for_tier,
    build_statsforecast_model_spec_for_tier,
    load_model_training_dataset,
    load_model_training_dataset_after_feature_publish,
    train_models,
    train_pecnet_models,
    train_statsforecast_models,
    train_test_split_for_tier,
)


def model_tiers() -> tuple[str, ...]:
    return ("tier1", "tier2")


def _dataset(tier_name: str, name: str) -> str:
    return f"stock_close_{tier_name}_{name}"


def tier_machine_learning_nodes(
    tier_name: str,
    *,
    wait_for_feature_publish: bool = False,
) -> list:
    load_inputs = (
        ["run_config", "model_feature_publish_metadata"]
        if wait_for_feature_publish
        else "run_config"
    )
    load_func = (
        load_model_training_dataset_after_feature_publish
        if wait_for_feature_publish
        else load_model_training_dataset
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
            func=partial(train_test_split_for_tier, tier_name=tier_name),
            inputs=[
                _dataset(tier_name, "training_dataset"),
                "run_config",
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


def model_matrix_nodes(*, wait_for_feature_publish: bool = False) -> list:
    nodes = []
    for tier_name in model_tiers():
        nodes.extend(
            tier_machine_learning_nodes(
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
    return inputs


def _mlforecast_nodes(tier_name: str) -> list:
    return [
        node(
            func=partial(build_model_spec, tier_name=tier_name),
            inputs="run_config",
            outputs=_dataset(tier_name, "mlforecast_model_spec"),
            name=f"build_{tier_name}_mlforecast_model_spec",
        ),
        node(
            func=train_models,
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
            func=partial(build_statsforecast_model_spec_for_tier, tier_name=tier_name),
            inputs="run_config",
            outputs=_dataset(tier_name, "statsforecast_model_spec"),
            name=f"build_{tier_name}_statsforecast_model_spec",
        ),
        node(
            func=train_statsforecast_models,
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
            func=partial(build_pecnet_model_spec_for_tier, tier_name=tier_name),
            inputs="run_config",
            outputs=_dataset(tier_name, "pecnet_model_spec"),
            name=f"build_{tier_name}_pecnet_model_spec",
        ),
        node(
            func=train_pecnet_models,
            inputs=[
                _dataset(tier_name, "train_test_split"),
                _dataset(tier_name, "pecnet_model_spec"),
            ],
            outputs=[
                _dataset(tier_name, "pecnet_regression_metrics"),
                _dataset(tier_name, "pecnet_long_direction_metrics"),
                _dataset(tier_name, "pecnet_predictions"),
                _dataset(tier_name, "pecnet_training_metadata"),
            ],
            name=f"train_{tier_name}_pecnet_models",
        ),
    ]
