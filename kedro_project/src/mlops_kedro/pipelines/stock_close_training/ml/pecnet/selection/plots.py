from __future__ import annotations

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd


def _log_feature_selection_heatmap(
    selection_df: pd.DataFrame,
    *,
    artifact_file: str,
    title: str,
    index_column: str = "ticker",
) -> None:
    if selection_df.empty or "correlation" not in selection_df.columns:
        return

    correlation_df = selection_df.copy()
    correlation_df["correlation"] = pd.to_numeric(
        correlation_df["correlation"],
        errors="coerce",
    )
    correlation_df = correlation_df[np.isfinite(correlation_df["correlation"])]
    if correlation_df.empty:
        return

    if index_column == "selection_order":
        correlation_df["_heatmap_index"] = (
            "#"
            + correlation_df["selection_order"].astype(int).astype(str)
            + " | "
            + correlation_df["reference_name"].astype(str)
        )
    else:
        correlation_df["_heatmap_index"] = correlation_df[index_column].astype(str)

    heatmap = correlation_df.pivot_table(
        index="_heatmap_index",
        columns="feature_name",
        values="correlation",
        aggfunc="mean",
    )
    if heatmap.empty:
        return

    width = max(8.0, min(24.0, 1.15 * len(heatmap.columns) + 4.0))
    height = max(4.5, min(18.0, 0.55 * len(heatmap.index) + 3.0))
    figure, axis = plt.subplots(figsize=(width, height))
    matrix = heatmap.to_numpy(dtype=float)
    image = axis.imshow(
        np.ma.masked_invalid(matrix),
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        aspect="auto",
    )

    axis.set_title(title)
    axis.set_xlabel("Feature")
    axis.set_ylabel("Selection" if index_column == "selection_order" else index_column)
    axis.set_xticks(np.arange(len(heatmap.columns)))
    axis.set_xticklabels(heatmap.columns, rotation=45, ha="right")
    axis.set_yticks(np.arange(len(heatmap.index)))
    axis.set_yticklabels(heatmap.index)

    if matrix.size <= 180:
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                if np.isfinite(value):
                    axis.text(
                        column_index,
                        row_index,
                        f"{value:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="black",
                    )

    figure.colorbar(image, ax=axis, label="Pearson correlation")
    figure.tight_layout()
    mlflow.log_figure(figure, artifact_file=artifact_file)
    plt.close(figure)
