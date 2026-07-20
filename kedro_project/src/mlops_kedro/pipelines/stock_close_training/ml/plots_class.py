from __future__ import annotations

import re

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

from .local_artifacts import LightweightArtifactStore


class ForecastPlotter:

    @staticmethod
    def forecast_model_columns(df: pd.DataFrame) -> list[str]:
        return [
            column
            for column in df.columns
            if column not in {"unique_id", "ds", "y"}
            and "-lo-" not in column
            and "-hi-" not in column
        ]

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)

    @staticmethod
    def log_forecast_plots(
        *,
        train_df: pd.DataFrame,
        joined_df: pd.DataFrame,
        levels: list[int] | None = None,
        artifact_prefix: str = "plots",
    ) -> None:
        levels = levels or [80, 95]
        if joined_df.empty or "y" not in joined_df.columns:
            return

        train_history = train_df[["unique_id", "ds", "y"]].copy()
        plot_df = (
            joined_df.sort_values(["unique_id", "ds"])
            .drop_duplicates(subset=["unique_id", "ds"], keep="last")
            .copy()
        )

        last_train_values = (
            train_df.sort_values(["unique_id", "ds"])
            .groupby("unique_id", observed=True)["y"]
            .last()
        )

        for model_name in ForecastPlotter.forecast_model_columns(plot_df):
            safe_model_name = ForecastPlotter._safe_name(model_name)
            available_levels = [
                level
                for level in levels
                if {
                    f"{model_name}-lo-{level}",
                    f"{model_name}-hi-{level}",
                }.issubset(plot_df.columns)
            ]
            interval_columns = [
                f"{model_name}-{bound}-{level}"
                for level in available_levels
                for bound in ("lo", "hi")
            ]

            figure = ForecastPlotter._forecast_train_test_figure(
                train_history=train_history,
                test_forecasts=plot_df[["unique_id", "ds", "y", model_name, *interval_columns]],
                model_name=model_name,
                available_levels=available_levels,
            )
            mlflow.log_figure(
                figure=figure,
                artifact_file=f"{artifact_prefix}/forecasts/{safe_model_name}_forecast.png",
            )
            LightweightArtifactStore().save_plot(
                figure,
                f"{artifact_prefix}/forecasts/{safe_model_name}_forecast.png",
            )
            plt.close(figure)

            ForecastPlotter._log_directional_confusion_matrix(
                plot_df=plot_df,
                model_name=model_name,
                safe_model_name=safe_model_name,
                last_train_values=last_train_values,
                artifact_prefix=artifact_prefix,
            )

    @staticmethod
    def _forecast_train_test_figure(
        *,
        train_history: pd.DataFrame,
        test_forecasts: pd.DataFrame,
        model_name: str,
        available_levels: list[int],
        max_train_points: int = 120,
    ) -> plt.Figure:
        train_history = ForecastPlotter._normalized_forecast_frame(train_history)
        test_forecasts = ForecastPlotter._normalized_forecast_frame(test_forecasts)
        symbols = ForecastPlotter._ordered_symbols(train_history, test_forecasts)
        if not symbols:
            return plt.figure(figsize=(14, 5))

        figure, axes = plt.subplots(
            nrows=len(symbols),
            ncols=1,
            figsize=(16, max(5, 4.2 * len(symbols))),
            squeeze=False,
            constrained_layout=True,
        )
        interval_levels = sorted(available_levels, reverse=True)

        for axis, symbol in zip(axes.ravel(), symbols, strict=False):
            symbol_train = (
                train_history[train_history["_symbol_key"].eq(symbol)]
                .sort_values("ds")
                .tail(max_train_points)
            )
            symbol_test = test_forecasts[
                test_forecasts["_symbol_key"].eq(symbol)
            ].sort_values("ds")

            if not symbol_train.empty:
                axis.plot(
                    symbol_train["ds"],
                    symbol_train["y"],
                    color="#6b7280",
                    linewidth=1.8,
                    label=f"Train actual (last {len(symbol_train)})",
                )

            if not symbol_test.empty:
                axis.plot(
                    symbol_test["ds"],
                    symbol_test["y"],
                    color="#111827",
                    linewidth=2.2,
                    marker="o",
                    markersize=3.5,
                    label="Test actual",
                )
                ForecastPlotter._plot_prediction_intervals(
                    axis=axis,
                    symbol_test=symbol_test,
                    model_name=model_name,
                    levels=interval_levels,
                )
                axis.plot(
                    symbol_test["ds"],
                    symbol_test[model_name],
                    color="#f97316",
                    linewidth=2.2,
                    marker="x",
                    markersize=5,
                    label="Prediction",
                )
                axis.axvline(
                    symbol_test["ds"].min(),
                    color="#2563eb",
                    linestyle="--",
                    linewidth=1.2,
                    alpha=0.7,
                    label="Test start",
                )

            axis.set_title(f"{symbol} | {model_name}: train actual, test actual, prediction")
            axis.set_xlabel("Date")
            axis.set_ylabel("Close")
            axis.grid(alpha=0.25)
            axis.legend(loc="best")

        return figure

    @staticmethod
    def _normalized_forecast_frame(df: pd.DataFrame) -> pd.DataFrame:
        normalized = df.copy()
        if normalized.empty:
            normalized["_symbol_key"] = pd.Series(dtype=str)
            return normalized

        normalized["unique_id"] = normalized["unique_id"].astype(str)
        normalized["_symbol_key"] = normalized["unique_id"]
        normalized["ds"] = pd.to_datetime(normalized["ds"], errors="coerce")
        return normalized.dropna(subset=["ds"])

    @staticmethod
    def _ordered_symbols(*frames: pd.DataFrame) -> list[str]:
        symbols: list[str] = []
        for frame in frames:
            if frame.empty or "_symbol_key" not in frame.columns:
                continue
            for symbol in frame["_symbol_key"].dropna().astype(str):
                if symbol not in symbols:
                    symbols.append(symbol)
        return symbols

    @staticmethod
    def _plot_prediction_intervals(
        *,
        axis: plt.Axes,
        symbol_test: pd.DataFrame,
        model_name: str,
        levels: list[int],
    ) -> None:
        for index, level in enumerate(levels):
            lower_column = f"{model_name}-lo-{level}"
            upper_column = f"{model_name}-hi-{level}"
            if lower_column not in symbol_test.columns or upper_column not in symbol_test.columns:
                continue

            interval_df = symbol_test[["ds", lower_column, upper_column]].dropna()
            if interval_df.empty:
                continue

            axis.fill_between(
                interval_df["ds"],
                interval_df[lower_column].astype(float),
                interval_df[upper_column].astype(float),
                color="#60a5fa",
                alpha=max(0.10, 0.22 - index * 0.05),
                label=f"{level}% prediction interval",
            )

    @staticmethod
    def _log_directional_confusion_matrix(
        *,
        plot_df: pd.DataFrame,
        model_name: str,
        safe_model_name: str,
        last_train_values: pd.Series,
        artifact_prefix: str,
    ) -> None:
        direction_df = (
            plot_df[["unique_id", "ds", "y", model_name]]
            .dropna(subset=["y", model_name])
            .sort_values(["unique_id", "ds"])
            .copy()
        )
        if direction_df.empty:
            return

        previous_actual = direction_df.groupby("unique_id", observed=True)["y"].shift(1)
        train_baseline = direction_df["unique_id"].map(last_train_values)
        previous_actual = previous_actual.fillna(train_baseline)

        valid_mask = previous_actual.notna()
        if not valid_mask.any():
            return

        actual_long = direction_df.loc[valid_mask, "y"] > previous_actual.loc[valid_mask]
        predicted_long = (
            direction_df.loc[valid_mask, model_name] > previous_actual.loc[valid_mask]
        )

        long_accuracy = accuracy_score(actual_long, predicted_long)
        figure, axis = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_true=actual_long,
            y_pred=predicted_long,
            labels=[False, True],
            display_labels=["not_long", "long"],
            cmap="Blues",
            values_format="d",
            colorbar=False,
            ax=axis,
        )
        axis.set_title(
            f"{model_name} - Long Direction Confusion Matrix\n"
            f"Long Accuracy: {long_accuracy:.4f}"
        )
        figure.tight_layout()
        mlflow.log_figure(
            figure=figure,
            artifact_file=(
                f"{artifact_prefix}/confusion_matrices/"
                f"{safe_model_name}_long_confusion_matrix.png"
            ),
        )
        LightweightArtifactStore().save_plot(
            figure,
            (
                f"{artifact_prefix}/confusion_matrices/"
                f"{safe_model_name}_long_confusion_matrix.png"
            ),
        )
        plt.close(figure)

        report = classification_report(
            y_true=actual_long,
            y_pred=predicted_long,
            labels=[False, True],
            target_names=["not_long", "long"],
            zero_division=0,
            output_dict=True,
        )
        report["long_accuracy"] = float(long_accuracy)
        report["observation_count"] = int(len(actual_long))
        mlflow.log_dict(
            report,
            artifact_file=(
                f"{artifact_prefix}/reports/{safe_model_name}_long_report.json"
            ),
        )
        LightweightArtifactStore().save_params(
            report,
            ForecastPlotter._local_metrics_artifact_file(
                artifact_prefix,
                f"{safe_model_name}_long_report.json",
            ),
        )

    @staticmethod
    def _local_metrics_artifact_file(artifact_prefix: str, file_name: str) -> str:
        parts = [part for part in artifact_prefix.split("/") if part]
        if "plots" in parts:
            plots_index = parts.index("plots")
            parts = [*parts[:plots_index], "metrics", *parts[plots_index + 1 :]]
        else:
            parts = [*parts, "metrics"]
        return "/".join([*parts, file_name])
