from typing import Any


def configured_list(mapping: dict[str, Any], key: str) -> list[str]:
    return list(mapping.get(key, []))


def resolve_column_config(columns: dict[str, Any]) -> dict[str, list[str]]:
    entity_columns = configured_list(columns, "entity")
    price_columns = configured_list(columns, "price")
    analytics_calendar_columns = configured_list(columns, "analytics_calendar")
    target_columns = configured_list(columns, "target")
    tier_1_feature_columns = configured_list(columns, "tier_1_features")
    model_time_feature_columns = configured_list(columns, "model_time_features")
    indicator_columns = configured_list(columns, "indicator")
    condition_columns = configured_list(columns, "condition")
    strategy_label_columns = configured_list(columns, "strategy_label")
    output_audit_columns = configured_list(columns, "output_audit")
    close_model_time_feature_columns = configured_list(
        columns,
        "close_model_time_features",
    )
    tier_2_feature_columns = list(
        columns.get(
            "tier_2_features",
            [*tier_1_feature_columns, *model_time_feature_columns],
        )
    )
    indicator_feature_columns = list(
        columns.get(
            "indicator_features",
            [
                *entity_columns,
                *output_audit_columns,
                *price_columns,
                *analytics_calendar_columns,
                *target_columns,
                *tier_1_feature_columns,
                *model_time_feature_columns,
                *indicator_columns,
            ],
        )
    )
    conventional_gap_trading_columns = list(
        columns.get(
            "conventional_gap_trading",
            [
                *indicator_feature_columns,
                *condition_columns,
                *strategy_label_columns,
            ],
        )
    )
    close_model_dataset_columns = list(
        columns.get(
            "close_model_dataset",
            [
                "unique_id",
                "ds",
                "y",
                *close_model_time_feature_columns,
                *output_audit_columns,
            ],
        )
    )
    return {
        "entity": entity_columns,
        "price": price_columns,
        "analytics_calendar": analytics_calendar_columns,
        "target": target_columns,
        "tier_1_features": tier_1_feature_columns,
        "tier_2_features": tier_2_feature_columns,
        "fourier_time_encoding": configured_list(columns, "fourier_time_encoding"),
        "model_time_features": model_time_feature_columns,
        "indicator": indicator_columns,
        "condition": condition_columns,
        "strategy_required_conditions": configured_list(
            columns,
            "strategy_required_conditions",
        ),
        "strategy_label": strategy_label_columns,
        "output_audit": output_audit_columns,
        "indicator_features": indicator_feature_columns,
        "conventional_gap_trading": conventional_gap_trading_columns,
        "close_model_time_features": close_model_time_feature_columns,
        "close_model_dataset": close_model_dataset_columns,
        "model_ready": list(
            columns.get(
                "model_ready",
                [
                    *tier_1_feature_columns,
                    *model_time_feature_columns,
                    *indicator_columns,
                ],
            )
        ),
    }
