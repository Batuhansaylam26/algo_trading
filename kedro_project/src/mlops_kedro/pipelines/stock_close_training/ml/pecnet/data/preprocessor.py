from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...common import log_mlflow_datasets
from ..runtime import _safe_name, _ticker_test_ratio


@dataclass(slots=True)
class PecnetDataPreprocessor:
    data_preprocessor_cls: Any

    def prepare_ticker_inputs(
        self,
        *,
        ticker_df: pd.DataFrame,
        ticker_train_df: pd.DataFrame,
        ticker_test_df: pd.DataFrame,
        ticker: str,
        tier_name: str,
        feature_columns: list[str],
        preprocess_params: dict[str, Any],
        test_horizon: int,
    ) -> tuple[dict[str, Any], dict[str, object]]:
        ticker_data = _preprocess_ticker(
            ticker_df=ticker_df,
            ticker=str(ticker),
            feature_columns=feature_columns,
            preprocess_params=preprocess_params,
            test_horizon=test_horizon,
            data_preprocessor_cls=self.data_preprocessor_cls,
        )
        preprocessed_df = _pecnet_preprocessed_training_frame(
            ticker_data=ticker_data,
            tier_name=tier_name,
        )
        _log_pecnet_preprocessed_inputs(
            preprocessed_df=preprocessed_df,
            tier_name=tier_name,
            ticker=str(ticker),
        )
        store_metadata = _publish_pecnet_preprocessed_inputs(preprocessed_df)
        return ticker_data, store_metadata


def _preprocess_ticker(
    *,
    ticker_df: pd.DataFrame,
    ticker: str,
    feature_columns: list[str],
    preprocess_params: dict[str, Any],
    test_horizon: int,
    data_preprocessor_cls,
) -> dict[str, Any]:
    dp = data_preprocessor_cls()
    dp.reset()

    ticker_df = ticker_df.sort_values("ds").copy()
    test_ratio = _ticker_test_ratio(len(ticker_df), test_horizon)
    params = {
        **preprocess_params,
        "test_ratio": test_ratio,
    }

    target_series = ticker_df["y"].to_numpy(dtype=float)
    X_train_target, X_test_target, y_train, y_test = dp.preprocess(
        data=target_series,
        **params,
    )

    feature_X_trains = []
    feature_X_tests = []
    available_feature_columns = [
        column for column in feature_columns if column in ticker_df.columns
    ]
    for column in available_feature_columns:
        X_train_feature, X_test_feature, _, _ = dp.preprocess(
            data=ticker_df[column].to_numpy(dtype=float),
            **params,
        )
        feature_X_trains.append(X_train_feature)
        feature_X_tests.append(X_test_feature)

    return {
        "ticker": ticker,
        "target_series": target_series,
        "dates": ticker_df["ds"].reset_index(drop=True),
        "X_train_target": X_train_target,
        "X_test_target": X_test_target,
        "y_train": y_train,
        "y_test": y_test,
        "feature_X_trains": feature_X_trains,
        "feature_X_tests": feature_X_tests,
        "feature_names": available_feature_columns,
        "preprocess_params": params,
        "test_ratio": test_ratio,
    }

def _as_2d_float_array(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return array.reshape(array.shape[0], -1)

def _preprocessed_dates(
    ticker_data: dict[str, Any],
    *,
    split_name: str,
    row_count: int,
) -> pd.Series:
    dates = pd.to_datetime(ticker_data["dates"], utc=True)
    test_count = len(ticker_data["y_test"])
    if split_name == "test":
        return dates.tail(test_count).tail(row_count).reset_index(drop=True)

    train_end = max(len(dates) - test_count, 0)
    return dates.iloc[:train_end].tail(row_count).reset_index(drop=True)

def _iter_preprocessed_variable_specs(
    ticker_data: dict[str, Any],
    *,
    split_name: str,
) -> list[tuple[int, str, Any]]:
    if split_name == "train":
        feature_arrays = ticker_data["feature_X_trains"]
        target_array = ticker_data["X_train_target"]
    else:
        feature_arrays = ticker_data["feature_X_tests"]
        target_array = ticker_data["X_test_target"]

    return [
        (0, "target", target_array),
        *[
            (index + 1, feature_name, feature_array)
            for index, (feature_name, feature_array) in enumerate(
                zip(ticker_data["feature_names"], feature_arrays, strict=False)
            )
        ],
    ]

def _pecnet_preprocessed_training_frame(
    *,
    ticker_data: dict[str, Any],
    tier_name: str,
) -> pd.DataFrame:
    rows = []
    ticker = str(ticker_data["ticker"])
    tier_safe = _safe_name(tier_name)
    ticker_safe = _safe_name(ticker)
    created_timestamp = pd.Timestamp.now(tz="UTC")

    for split_index, split_name in enumerate(("train", "test")):
        y_values = np.asarray(ticker_data[f"y_{split_name}"], dtype=float).reshape(-1)
        for variable_index, variable_name, values in _iter_preprocessed_variable_specs(
            ticker_data,
            split_name=split_name,
        ):
            matrix = _as_2d_float_array(values)
            sample_count = min(matrix.shape[0], len(y_values))
            dates = _preprocessed_dates(
                ticker_data,
                split_name=split_name,
                row_count=sample_count,
            )
            sample_count = min(sample_count, len(dates))
            if sample_count == 0:
                continue

            matrix = matrix[-sample_count:]
            targets = y_values[-sample_count:]
            variable_safe = _safe_name(str(variable_name))
            for sample_index, event_timestamp in enumerate(dates):
                timestamp = pd.Timestamp(event_timestamp).tz_convert("UTC")
                timestamp_key = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
                for step_index, value in enumerate(matrix[sample_index]):
                    row_key = (
                        f"{tier_safe}|{ticker_safe}|{split_name}|{variable_safe}|"
                        f"{sample_index}|{step_index}|{timestamp_key}"
                    )
                    rows.append(
                        {
                            "row_key": row_key,
                            "tier": tier_name,
                            "symbol": ticker,
                            "event_timestamp": timestamp,
                            "split": split_name,
                            "split_index": split_index,
                            "variable_name": str(variable_name),
                            "variable_index": variable_index,
                            "sample_index": sample_index,
                            "step_index": step_index,
                            "value": float(value),
                            "target_y": float(targets[sample_index]),
                            "created_timestamp": created_timestamp,
                        }
                    )

    return pd.DataFrame(rows)

def _log_pecnet_preprocessed_inputs(
    *,
    preprocessed_df: pd.DataFrame,
    tier_name: str,
    ticker: str,
) -> None:
    if preprocessed_df.empty:
        return

    artifact_prefix = (
        f"pecnet/{tier_name}/tickers/{_safe_name(str(ticker))}/preprocessed"
    )
    dataset_prefix = (
        f"stock_close_{tier_name}_{_safe_name(str(ticker))}_pecnet_preprocessed"
    )
    log_mlflow_datasets(
        train_df=preprocessed_df[preprocessed_df["split"] == "train"].copy(),
        test_df=preprocessed_df[preprocessed_df["split"] == "test"].copy(),
        dataset_prefix=dataset_prefix,
        artifact_prefix=artifact_prefix,
    )

def _publish_pecnet_preprocessed_inputs(
    preprocessed_df: pd.DataFrame,
) -> dict[str, object]:
    if preprocessed_df.empty:
        return {
            "timescale_rows": 0,
            "feast_online_rows": 0,
        }

    from ....serving.feast_store import (  # noqa: PLC0415
        publish_pecnet_preprocessed_training_data,
    )

    return publish_pecnet_preprocessed_training_data(
        pl.from_pandas(preprocessed_df)
    )
