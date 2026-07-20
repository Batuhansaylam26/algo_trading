from __future__ import annotations

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score


BASE_COLUMNS = {
    "unique_id",
    "ds",
    "y",
    "previous_actual_close",
    "actual_long",
}


class StockCloseModelMetrics:

    @staticmethod
    def model_prediction_columns(df: pd.DataFrame) -> list[str]:
        return [
            column
            for column in df.columns
            if column not in BASE_COLUMNS
            and "-lo-" not in column
            and "-hi-" not in column
        ]

    @staticmethod
    def add_previous_actual_close(
        joined_df: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        last_train_close = (
            train_df.sort_values(["unique_id", "ds"])
            .groupby("unique_id", observed=True)["y"]
            .last()
        )

        ordered = joined_df.sort_values(["unique_id", "ds"]).copy()
        previous_close = ordered.groupby("unique_id", observed=True)["y"].shift(1)
        ordered["previous_actual_close"] = previous_close.fillna(
            ordered["unique_id"].map(last_train_close)
        )
        return ordered

    @staticmethod
    def build_long_direction_frame(
        joined_df: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        direction_df = StockCloseModelMetrics.add_previous_actual_close(joined_df, train_df)
        models = StockCloseModelMetrics.model_prediction_columns(direction_df)
        direction_df["actual_long"] = (
            direction_df["y"] > direction_df["previous_actual_close"]
        )

        for model in models:
            direction_df[f"{model}_long"] = (
                direction_df[model] > direction_df["previous_actual_close"]
            )

        return direction_df[
            [
                "unique_id",
                "ds",
                "y",
                "previous_actual_close",
                "actual_long",
                *[f"{model}_long" for model in models],
            ]
        ]

    @staticmethod
    def long_only_directional_metrics(
        joined_df: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        direction_df = StockCloseModelMetrics.build_long_direction_frame(joined_df, train_df)
        rows = []

        for column in direction_df.columns:
            if not column.endswith("_long") or column == "actual_long":
                continue

            model = column.removesuffix("_long")
            valid_rows = direction_df[["actual_long", column]].dropna()

            if valid_rows.empty:
                rows.append(
                    {
                        "model": model,
                        "long_accuracy": None,
                        "long_precision": None,
                        "long_recall": None,
                        "long_signal_rate": None,
                        "rows": 0,
                    }
                )
                continue

            rows.append(
                {
                    "model": model,
                    "long_accuracy": accuracy_score(
                        valid_rows["actual_long"],
                        valid_rows[column],
                    ),
                    "long_precision": precision_score(
                        valid_rows["actual_long"],
                        valid_rows[column],
                        zero_division=0,
                    ),
                    "long_recall": recall_score(
                        valid_rows["actual_long"],
                        valid_rows[column],
                        zero_division=0,
                    ),
                    "long_signal_rate": float(valid_rows[column].mean()),
                    "rows": len(valid_rows),
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def long_only_directional_metrics_by_unique_id(
        joined_df: pd.DataFrame,
        train_df: pd.DataFrame,
    ) -> pd.DataFrame:
        frames = []
        for unique_id, ticker_df in joined_df.groupby("unique_id", observed=True):
            ticker_train_df = train_df[train_df["unique_id"].eq(unique_id)]
            metrics = StockCloseModelMetrics.long_only_directional_metrics(ticker_df, ticker_train_df)
            if metrics.empty:
                continue

            metrics.insert(0, "unique_id", unique_id)
            frames.append(metrics)

        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
