from __future__ import annotations

import re

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from utilsforecast.plotting import plot_series


def forecast_model_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if column not in {"unique_id", "ds", "y"}
        and "-lo-" not in column
        and "-hi-" not in column
    ]


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)


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
    test_actuals = joined_df[["unique_id", "ds", "y"]].dropna(subset=["y"]).copy()
    history_df = (
        pd.concat([train_history, test_actuals], ignore_index=True)
        .sort_values(["unique_id", "ds"])
        .drop_duplicates(subset=["unique_id", "ds"], keep="last")
    )
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

    for model_name in forecast_model_columns(plot_df):
        safe_model_name = _safe_name(model_name)
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

        figure = plot_series(
            df=history_df,
            forecasts_df=plot_df[["unique_id", "ds", model_name, *interval_columns]],
            models=[model_name],
            level=available_levels or None,
            max_insample_length=120,
            plot_random=False,
            engine="matplotlib",
        )
        mlflow.log_figure(
            figure=figure,
            artifact_file=f"{artifact_prefix}/forecasts/{safe_model_name}_forecast.png",
        )
        plt.close(figure)

        _log_directional_confusion_matrix(
            plot_df=plot_df,
            model_name=model_name,
            safe_model_name=safe_model_name,
            last_train_values=last_train_values,
            artifact_prefix=artifact_prefix,
        )


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


def pecnet_prediction_figure(
    *,
    predictions: np.ndarray,
    actual: np.ndarray,
    dates: pd.Series,
    model_name: str,
) -> tuple[plt.Figure, pd.DataFrame]:
    predictions = np.asarray(predictions, dtype=float).reshape(-1)
    actual = np.asarray(actual, dtype=float).reshape(-1)
    dates = pd.Series(dates).reset_index(drop=True)

    usable_length = min(len(predictions), len(actual), len(dates))
    predictions = predictions[-usable_length:]
    actual = actual[-usable_length:]
    dates = dates.iloc[-usable_length:].reset_index(drop=True)

    valid_mask = np.isfinite(actual) & np.isfinite(predictions)
    predictions = predictions[valid_mask]
    actual = actual[valid_mask]
    dates = dates[valid_mask].reset_index(drop=True)

    residuals = actual - predictions
    mae = float(np.mean(np.abs(residuals))) if len(residuals) else np.nan
    rmse = float(np.sqrt(np.mean(residuals**2))) if len(residuals) else np.nan
    long_accuracy = (
        float(np.mean((np.diff(actual) > 0) == (np.diff(predictions) > 0)))
        if len(actual) > 1
        else np.nan
    )

    comparison_df = pd.DataFrame(
        {
            "ds": dates,
            "actual": actual,
            "prediction": predictions,
            "residual": residuals,
        }
    )

    figure, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(18, 11),
        constrained_layout=True,
    )
    axes[0, 0].plot(dates, actual, color="black", linewidth=2, label="Actual")
    axes[0, 0].plot(
        dates,
        predictions,
        color="darkorange",
        linewidth=2,
        label=model_name,
    )
    axes[0, 0].set_title(
        f"{model_name}: Actual vs Prediction\n"
        f"MAE={mae:,.4f} | RMSE={rmse:,.4f} | Long Accuracy={long_accuracy:.2%}"
    )
    axes[0, 0].legend()
    axes[0, 0].grid(alpha=0.25)

    axes[0, 1].scatter(actual, predictions, alpha=0.7, color="steelblue")
    if len(actual):
        value_min = min(actual.min(), predictions.min())
        value_max = max(actual.max(), predictions.max())
        axes[0, 1].plot(
            [value_min, value_max],
            [value_min, value_max],
            linestyle="--",
            color="red",
        )
    axes[0, 1].set_title("Actual vs Prediction Scatter")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].plot(dates, residuals, color="crimson")
    axes[1, 0].axhline(0, color="black", linestyle="--")
    axes[1, 0].fill_between(dates, residuals, 0, color="crimson", alpha=0.2)
    axes[1, 0].set_title("Residuals: Actual - Prediction")
    axes[1, 0].grid(alpha=0.25)

    axes[1, 1].hist(residuals, bins=min(30, max(5, len(residuals))), color="slateblue")
    axes[1, 1].set_title("Residual Distribution")
    axes[1, 1].grid(alpha=0.25)

    return figure, comparison_df
