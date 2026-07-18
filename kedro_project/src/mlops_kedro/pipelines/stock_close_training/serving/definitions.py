from __future__ import annotations

import polars as pl
from feast import FeatureStore

from .connections import _ensure_feature_repo_on_path
from .constants import FEATURE_REPO_DIR
from .transforms import (
    _fill_close_model_dataset_daily_gaps,
    _to_pandas_for_close_model_dataset,
    _to_pandas_for_feature_store,
    _to_pandas_for_pecnet_preprocessed,
    _to_pandas_for_tier2_feature_dataset,
)


def _apply_model_feature_definitions() -> FeatureStore:
    _ensure_feature_repo_on_path()
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))

    from feature_repo.stock_features import (  # noqa: PLC0415
        stock_feature_row,
        stock_model_features_view,
        stock_model_tier2_dataset_service,
        stock_model_tier2_dataset_view,
        stock_model_tier_1_feature_service,
        stock_model_tier_2_feature_service,
        stock_model_tier_3_feature_service,
        stock_model_tier_5_feature_service,
        ticker,
    )

    store.apply(
        [
            ticker,
            stock_model_features_view,
            stock_model_tier_1_feature_service,
            stock_model_tier_2_feature_service,
            stock_model_tier_3_feature_service,
            stock_model_tier_5_feature_service,
            stock_feature_row,
            stock_model_tier2_dataset_view,
            stock_model_tier2_dataset_service,
        ]
    )
    return store

def _apply_pecnet_preprocessed_definition_and_push(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0

    _ensure_feature_repo_on_path()
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))

    from feature_repo.stock_features import (  # noqa: PLC0415
        pecnet_preprocessed_row,
        pecnet_preprocessed_training_service,
        pecnet_preprocessed_training_view,
    )

    store.apply(
        [
            pecnet_preprocessed_row,
            pecnet_preprocessed_training_view,
            pecnet_preprocessed_training_service,
        ]
    )
    store.write_to_online_store(
        "pecnet_preprocessed_training_data",
        _to_pandas_for_pecnet_preprocessed(df),
    )
    return len(df)

def _apply_feast_definitions_and_push(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0

    store = _apply_model_feature_definitions()
    store.write_to_online_store(
        "stock_model_features",
        _to_pandas_for_feature_store(df),
    )
    store.write_to_online_store(
        "stock_model_tier2_dataset",
        _to_pandas_for_tier2_feature_dataset(df),
    )
    return len(df)

def _apply_close_model_dataset_definition() -> None:
    _ensure_feature_repo_on_path()
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))

    from feature_repo.stock_features import (  # noqa: PLC0415
        stock_close_model_dataset_service,
        stock_close_model_dataset_view,
        stock_series,
    )

    store.apply(
        [
            stock_series,
            stock_close_model_dataset_view,
            stock_close_model_dataset_service,
        ]
    )

def _apply_close_model_dataset_definition_and_push(df: pl.DataFrame) -> int:
    if df.is_empty():
        return 0

    df = _fill_close_model_dataset_daily_gaps(df)
    _ensure_feature_repo_on_path()
    store = FeatureStore(repo_path=str(FEATURE_REPO_DIR))

    from feature_repo.stock_features import (  # noqa: PLC0415
        stock_close_model_dataset_service,
        stock_close_model_dataset_view,
        stock_series,
    )

    store.apply(
        [
            stock_series,
            stock_close_model_dataset_view,
            stock_close_model_dataset_service,
        ]
    )
    store.write_to_online_store(
        "stock_close_model_dataset",
        _to_pandas_for_close_model_dataset(df),
    )
    return len(df)
