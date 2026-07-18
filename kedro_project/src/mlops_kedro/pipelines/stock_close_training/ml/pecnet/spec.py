from __future__ import annotations

from typing import Any

import pandas as pd
import polars as pl

from ..common import non_feature_columns, split_train_test_by_horizon


def build_pecnet_spec(
    *,
    enabled: bool = True,
    test_horizon: int = 5,
    feature_columns: list[str] | None = None,
    preprocess_params: dict[str, Any] | None = None,
    hyperparams: dict[str, Any] | None = None,
    selection_params: dict[str, Any] | None = None,
    wandb_project: str = "stock-close-pecnet",
    wandb_mode: str = "offline",
    tier_name: str = "tier1",
) -> dict[str, Any]:
    return {
        "enabled": enabled,
        "tier_name": tier_name,
        "test_horizon": test_horizon,
        "feature_columns": feature_columns or [],
        "preprocess_params": preprocess_params or {},
        "hyperparams": hyperparams or {},
        "selection_params": selection_params or {},
        "wandb_project": wandb_project,
        "wandb_mode": wandb_mode,
    }

def to_pecnet_frame(df: pl.DataFrame) -> pd.DataFrame:
    available_feature_columns = [
        column for column in df.columns if column not in non_feature_columns()
    ]
    return (
        df.select(
            pl.col("unique_id").cast(pl.Utf8),
            pl.col("ds").cast(pl.Datetime("us"), strict=False),
            pl.col("y").cast(pl.Float64, strict=False),
            *[
                pl.col(column).cast(pl.Float64, strict=False)
                for column in available_feature_columns
            ],
        )
        .drop_nulls(["unique_id", "ds", "y", *available_feature_columns])
        .unique(subset=["unique_id", "ds"], keep="last", maintain_order=True)
        .sort(["unique_id", "ds"])
        .to_pandas()
    )

def make_pecnet_train_test_split(
    dataset: pl.DataFrame,
    *,
    test_horizon: int,
) -> dict[str, pd.DataFrame]:
    model_df = to_pecnet_frame(dataset)
    train_df, test_df = split_train_test_by_horizon(model_df, test_horizon)
    return {
        "full": model_df,
        "train": train_df,
        "test": test_df,
    }
