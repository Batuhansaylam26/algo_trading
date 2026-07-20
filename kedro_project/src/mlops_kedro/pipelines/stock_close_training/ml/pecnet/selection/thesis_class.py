from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .builder import (
    _candidate_variable_inputs,
    _max_selected_features_for_tier,
    _selection_row,
)


class PecnetThesisSelector:

    @staticmethod
    def _build_thesis_wavelet_error_compensated_pecnet_variables(
        *,
        builder,
        ticker_data: dict[str, Any],
        tier_name: str,
        feature_selector_cls,
        selection_params: dict[str, Any],
    ) -> tuple[Any, list[np.ndarray], pd.DataFrame]:
        candidate_names, candidate_X_train, candidate_X_test = _candidate_variable_inputs(
            ticker_data
        )
        builder.add_variable_network(
            ticker_data["X_train_target"],
            ticker_data["y_train"],
        )
        rows = [
            _selection_row(
                ticker=ticker_data["ticker"],
                tier_name=tier_name,
                strategy="thesis_wavelet_error_compensated_fusion",
                order=1,
                feature_index=0,
                feature_name="target_history",
                correlation=None,
                reference_name="target_y",
                threshold=None,
            )
        ]

        external_X_train = candidate_X_train[1:]
        external_X_test = candidate_X_test[1:]
        external_names = candidate_names[1:]
        if not external_X_train:
            return builder, [candidate_X_test[0]], pd.DataFrame(rows)

        threshold = float(selection_params.get("correlation_threshold", 0.08))
        max_selected_features = _max_selected_features_for_tier(
            selection_params,
            tier_name,
        )
        selector = feature_selector_cls(threshold=threshold)
        reference = builder.pecnet.get_target_values_for_current_variable_network()

        while True:
            if max_selected_features is not None and len(rows) >= max_selected_features:
                break

            selected_external_index = selector.select_next(
                external_X_train,
                reference,
                force_include_best_if_first=False,
            )
            if selected_external_index is None:
                break

            builder.add_variable_network(
                external_X_train[selected_external_index],
                reference,
            )
            correlation = selector.get_last_corr_score()
            rows.append(
                _selection_row(
                    ticker=ticker_data["ticker"],
                    tier_name=tier_name,
                    strategy="thesis_wavelet_error_compensated_fusion",
                    order=len(rows) + 1,
                    feature_index=selected_external_index + 1,
                    feature_name=external_names[selected_external_index],
                    correlation=float(correlation) if correlation is not None else None,
                    reference_name="residual_error",
                    threshold=threshold,
                )
            )
            reference = builder.pecnet.get_target_values_for_current_variable_network()

        selected_X_test = [
            candidate_X_test[int(row["feature_index"])]
            for row in rows
        ]
        return builder, selected_X_test, pd.DataFrame(rows)
