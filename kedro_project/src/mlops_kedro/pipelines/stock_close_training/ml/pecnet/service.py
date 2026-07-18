from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import polars as pl

from .spec import build_pecnet_spec, make_pecnet_train_test_split, to_pecnet_frame
from .training import PecnetTrainingWorkflow


@dataclass(slots=True)
class PecnetService:
    def build_spec(
        self,
        *,
        enabled: bool,
        test_horizon: int,
        feature_columns: list[str],
        preprocess_params: dict[str, Any],
        hyperparams: dict[str, Any],
        selection_params: dict[str, Any],
        tier_name: str,
    ) -> dict[str, Any]:
        return build_pecnet_spec(
            enabled=enabled,
            test_horizon=test_horizon,
            feature_columns=feature_columns,
            preprocess_params=preprocess_params,
            hyperparams=hyperparams,
            selection_params=selection_params,
            tier_name=tier_name,
        )

    def to_frame(self, df: pl.DataFrame) -> pd.DataFrame:
        return to_pecnet_frame(df)

    def make_train_test_split(
        self,
        dataset: pl.DataFrame,
        *,
        test_horizon: int,
    ) -> dict[str, pd.DataFrame]:
        return make_pecnet_train_test_split(dataset, test_horizon=test_horizon)

    def train_from_split(
        self,
        train_test_split: dict[str, pd.DataFrame],
        *,
        model_spec: dict[str, Any],
    ) -> dict[str, Any]:
        return PecnetTrainingWorkflow().train_from_split(
            train_test_split,
            model_spec=model_spec,
        )
