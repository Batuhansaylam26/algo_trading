from typing import Any




DEFAULT_COLUMN_CONFIG: dict[str, list[str]] = {
    "entity": ["symbol", "date"],
    "price": ["open", "high", "low", "close", "volume"],
    "analytics_calendar": ["month", "day", "day_of_year"],
    "target": ["target_close"],
    "tier_1_features": ["prev_open", "prev_high", "prev_low", "prev_volume"],
    "fourier_time_encoding": [
        "month_sin_1",
        "month_cos_1",
        "month_sin_2",
        "month_cos_2",
        "day_sin_1",
        "day_cos_1",
        "day_sin_2",
        "day_cos_2",
        "day_of_year_sin_1",
        "day_of_year_cos_1",
        "day_of_year_sin_2",
        "day_of_year_cos_2",
    ],
    "model_time_features": [
        "calendar_gap_days",
        "month_sin_1",
        "month_cos_1",
        "month_sin_2",
        "month_cos_2",
        "day_sin_1",
        "day_cos_1",
        "day_sin_2",
        "day_cos_2",
        "day_of_year_sin_1",
        "day_of_year_cos_1",
        "day_of_year_sin_2",
        "day_of_year_cos_2",
    ],
    "indicator": [
        "RSI",
        "SMA_Short",
        "SMA_Long",
        "Vol_SMA",
        "Range_high",
        "Range_low",
        "ADX",
    ],
    "condition": [
        "gap_up",
        "gap_down",
        "exhaustion_gap_up",
        "exhaustion_gap_down",
        "high_vol",
        "extreme_vol",
        "breakout_up",
        "breakout_down",
        "consolidation",
        "uptrend",
        "downtrend",
        "overbought",
        "oversold",
    ],
    "strategy_required_conditions": [
        "gap_up",
        "gap_down",
        "breakout_up",
        "breakout_down",
        "high_vol",
        "extreme_vol",
        "consolidation",
        "uptrend",
        "downtrend",
        "overbought",
        "oversold",
    ],
    "strategy_label": ["Gap_Type"],
    "output_audit": ["created_timestamp"],
    "close_model_time_features": [
        "month_sin_1",
        "month_cos_1",
        "day_sin_1",
        "day_cos_1",
        "day_of_year_sin_1",
        "day_of_year_cos_1",
    ],
    "daily_lookback_features": [
        f"daily_{column}_lag_{lag}"
        for lag in range(1, 5)
        for column in ["open", "high", "low", "volume"]
    ],
    "weekly_lookback_features": [
        f"weekly_{column}_lag_{lag}"
        for lag in range(1, 5)
        for column in ["open", "high", "low", "close", "volume"]
    ],
}


class StockCloseConfigResolver:

    @staticmethod
    def _ordered_unique(columns: list[str]) -> list[str]:
        return list(dict.fromkeys(columns))

    @staticmethod
    def configured_list(mapping: dict[str, Any], key: str) -> list[str]:
        return list(mapping.get(key, []))

    @staticmethod
    def resolve_column_config(columns: dict[str, Any]) -> dict[str, list[str]]:
        columns = {
            **DEFAULT_COLUMN_CONFIG,
            **(columns or {}),
        }
        entity_columns = StockCloseConfigResolver.configured_list(columns, "entity")
        price_columns = StockCloseConfigResolver.configured_list(columns, "price")
        analytics_calendar_columns = StockCloseConfigResolver.configured_list(columns, "analytics_calendar")
        target_columns = StockCloseConfigResolver.configured_list(columns, "target")
        tier_1_feature_columns = StockCloseConfigResolver.configured_list(columns, "tier_1_features")
        model_time_feature_columns = StockCloseConfigResolver.configured_list(columns, "model_time_features")
        indicator_columns = StockCloseConfigResolver.configured_list(columns, "indicator")
        condition_columns = StockCloseConfigResolver.configured_list(columns, "condition")
        strategy_label_columns = StockCloseConfigResolver.configured_list(columns, "strategy_label")
        output_audit_columns = StockCloseConfigResolver.configured_list(columns, "output_audit")
        close_model_time_feature_columns = StockCloseConfigResolver.configured_list(
            columns,
            "close_model_time_features",
        )
        daily_lookback_feature_columns = StockCloseConfigResolver.configured_list(
            columns,
            "daily_lookback_features",
        )
        weekly_lookback_feature_columns = StockCloseConfigResolver.configured_list(
            columns,
            "weekly_lookback_features",
        )
        tier_2_feature_columns = list(
            columns.get(
                "tier_2_features",
                [*tier_1_feature_columns, *model_time_feature_columns],
            )
        )
        tier_3_feature_columns = list(
            columns.get(
                "tier_3_features",
                tier_2_feature_columns,
            )
        )
        tier_4_feature_columns = list(
            columns.get(
                "tier_4_features",
                tier_3_feature_columns,
            )
        )
        tier_5_feature_columns = list(
            columns.get(
                "tier_5_features",
                [
                    *daily_lookback_feature_columns,
                    *weekly_lookback_feature_columns,
                ],
            )
        )
        tier_6_feature_columns = list(
            columns.get(
                "tier_6_features",
                [
                    "weekly_close_lag_1",
                    *model_time_feature_columns,
                ],
            )
        )
        model_feature_columns = StockCloseConfigResolver._ordered_unique(
            list(
                columns.get(
                    "model_features",
                    [
                        *tier_1_feature_columns,
                        *tier_2_feature_columns,
                        *tier_3_feature_columns,
                        *tier_4_feature_columns,
                        *tier_5_feature_columns,
                        *tier_6_feature_columns,
                    ],
                )
            )
        )
        indicator_feature_columns = list(
            columns.get(
                "indicator_features",
                [
                    *entity_columns,
                    *output_audit_columns,
                    *price_columns,
                    *target_columns,
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
            "tier_3_features": tier_3_feature_columns,
            "tier_4_features": tier_4_feature_columns,
            "tier_5_features": tier_5_feature_columns,
            "tier_6_features": tier_6_feature_columns,
            "model_features": model_feature_columns,
            "fourier_time_encoding": StockCloseConfigResolver.configured_list(columns, "fourier_time_encoding"),
            "daily_lookback_features": daily_lookback_feature_columns,
            "weekly_lookback_features": weekly_lookback_feature_columns,
            "model_time_features": model_time_feature_columns,
            "indicator": indicator_columns,
            "condition": condition_columns,
            "strategy_required_conditions": StockCloseConfigResolver.configured_list(
                columns,
                "strategy_required_conditions",
            ),
            "strategy_label": strategy_label_columns,
            "output_audit": output_audit_columns,
            "indicator_features": indicator_feature_columns,
            "conventional_gap_trading": conventional_gap_trading_columns,
            "close_model_time_features": close_model_time_feature_columns,
            "close_model_dataset": close_model_dataset_columns,
            "indicator_ready": list(
                columns.get(
                    "indicator_ready",
                    indicator_columns,
                )
            ),
            "model_ready": list(
                columns.get(
                    "model_ready",
                    model_feature_columns,
                )
            ),
        }
