from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .plots import _log_feature_selection_heatmap


def _candidate_variable_inputs(
    ticker_data: dict[str, Any],
) -> tuple[list[str], list[np.ndarray], list[np.ndarray]]:
    return (
        ["target_history", *ticker_data["feature_names"]],
        [ticker_data["X_train_target"], *ticker_data["feature_X_trains"]],
        [ticker_data["X_test_target"], *ticker_data["feature_X_tests"]],
    )

def _selection_row(
    *,
    ticker: str,
    tier_name: str,
    strategy: str,
    order: int,
    feature_index: int,
    feature_name: str,
    correlation: float | None,
    reference_name: str,
    threshold: float | None,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "tier": tier_name,
        "strategy": strategy,
        "selection_order": order,
        "feature_index": feature_index,
        "feature_name": feature_name,
        "reference_name": reference_name,
        "correlation": correlation,
        "abs_correlation": abs(correlation) if correlation is not None else None,
        "threshold": threshold,
    }

def _max_selected_features_for_tier(
    selection_params: dict[str, Any],
    tier_name: str,
) -> int | None:
    by_tier = selection_params.get("max_selected_features_by_tier") or {}
    value = by_tier.get(tier_name, selection_params.get("max_selected_features"))
    if value is None:
        return None
    return int(value)

def _build_all_feature_pecnet_variables(
    *,
    builder,
    ticker_data: dict[str, Any],
    tier_name: str,
) -> tuple[Any, list[np.ndarray], pd.DataFrame]:
    candidate_names, _, candidate_X_test = _candidate_variable_inputs(ticker_data)
    builder.add_variable_network(
        ticker_data["X_train_target"],
        ticker_data["y_train"],
    )
    rows = [
        _selection_row(
            ticker=ticker_data["ticker"],
            tier_name=tier_name,
            strategy="all_features",
            order=1,
            feature_index=0,
            feature_name=candidate_names[0],
            correlation=None,
            reference_name="target_y",
            threshold=None,
        )
    ]
    for feature_order, (feature_name, X_train_feature) in enumerate(
        zip(
            ticker_data["feature_names"],
            ticker_data["feature_X_trains"],
            strict=False,
        ),
        start=2,
    ):
        builder.add_variable_network(
            X_train_feature,
            builder.pecnet.get_target_values_for_current_variable_network(),
        )
        rows.append(
            _selection_row(
                ticker=ticker_data["ticker"],
                tier_name=tier_name,
                strategy="all_features",
                order=feature_order,
                feature_index=feature_order - 1,
                feature_name=feature_name,
                correlation=None,
                reference_name="residual_error",
                threshold=None,
            )
        )

    return builder, candidate_X_test, pd.DataFrame(rows)

def _build_feature_selector_pecnet_variables(
    *,
    builder,
    ticker_data: dict[str, Any],
    tier_name: str,
    feature_selector_cls,
    selection_params: dict[str, Any],
    strategy_name: str,
) -> tuple[Any, list[np.ndarray], pd.DataFrame]:
    candidate_names, candidate_X_train, candidate_X_test = _candidate_variable_inputs(
        ticker_data
    )
    threshold = float(selection_params.get("correlation_threshold", 0.08))
    max_selected_features = _max_selected_features_for_tier(
        selection_params,
        tier_name,
    )
    force_first = bool(selection_params.get("force_include_best_if_first", True))
    selector = feature_selector_cls(threshold=threshold)
    reference = ticker_data["y_train"]
    rows = []
    initial_network = True

    while True:
        if max_selected_features is not None and len(rows) >= max_selected_features:
            break

        selected_index = selector.select_next(
            candidate_X_train,
            reference,
            force_include_best_if_first=force_first and initial_network,
        )
        if selected_index is None:
            break

        builder.add_variable_network(
            candidate_X_train[selected_index],
            ticker_data["y_train"] if initial_network else reference,
        )
        correlation = selector.get_last_corr_score()
        rows.append(
            _selection_row(
                ticker=ticker_data["ticker"],
                tier_name=tier_name,
                strategy=strategy_name,
                order=len(rows) + 1,
                feature_index=selected_index,
                feature_name=candidate_names[selected_index],
                correlation=float(correlation) if correlation is not None else None,
                reference_name="target_y" if initial_network else "residual_error",
                threshold=threshold,
            )
        )
        reference = builder.pecnet.get_target_values_for_current_variable_network()
        initial_network = False

    if not rows:
        builder.add_variable_network(
            ticker_data["X_train_target"],
            ticker_data["y_train"],
        )
        rows.append(
            _selection_row(
                ticker=ticker_data["ticker"],
                tier_name=tier_name,
                strategy=f"{strategy_name}_fallback",
                order=1,
                feature_index=0,
                feature_name="target_history",
                correlation=None,
                reference_name="target_y",
                threshold=threshold,
            )
        )

    selected_X_test = [
        candidate_X_test[int(row["feature_index"])]
        for row in rows
    ]
    return builder, selected_X_test, pd.DataFrame(rows)

def _build_residual_correlation_pecnet_variables(
    *,
    builder,
    ticker_data: dict[str, Any],
    tier_name: str,
    feature_selector_cls,
    selection_params: dict[str, Any],
) -> tuple[Any, list[np.ndarray], pd.DataFrame]:
    return _build_feature_selector_pecnet_variables(
        builder=builder,
        ticker_data=ticker_data,
        tier_name=tier_name,
        feature_selector_cls=feature_selector_cls,
        selection_params=selection_params,
        strategy_name="residual_correlation",
    )

def _build_pecnet_variables(
    *,
    builder,
    ticker_data: dict[str, Any],
    tier_name: str,
    feature_selector_cls,
    selection_params: dict[str, Any],
) -> tuple[Any, list[np.ndarray], pd.DataFrame]:
    strategy_by_tier = selection_params.get("strategy_by_tier", {})
    strategy = strategy_by_tier.get(
        tier_name,
        selection_params.get("strategy", "all_features"),
    )
    if strategy == "residual_correlation":
        return _build_residual_correlation_pecnet_variables(
            builder=builder,
            ticker_data=ticker_data,
            tier_name=tier_name,
            feature_selector_cls=feature_selector_cls,
            selection_params=selection_params,
        )

    if strategy == "thesis_wavelet_error_compensated_fusion":
        from .thesis import (  # noqa: PLC0415
            _build_thesis_wavelet_error_compensated_pecnet_variables,
        )

        return _build_thesis_wavelet_error_compensated_pecnet_variables(
            builder=builder,
            ticker_data=ticker_data,
            tier_name=tier_name,
            feature_selector_cls=feature_selector_cls,
            selection_params=selection_params,
        )

    return _build_all_feature_pecnet_variables(
        builder=builder,
        ticker_data=ticker_data,
        tier_name=tier_name,
    )
