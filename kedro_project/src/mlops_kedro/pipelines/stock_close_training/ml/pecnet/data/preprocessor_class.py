from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import polars as pl

from ...common import log_mlflow_datasets
from ..runtime import _safe_name, _ticker_test_row_ratio


LOGGER = logging.getLogger(__name__)


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
        test_row_count: int,
        publish_to_store: bool = True,
    ) -> tuple[dict[str, Any], dict[str, object], pd.DataFrame]:
        ticker_data = PecnetDataPreprocessor._preprocess_ticker(
            ticker_df=ticker_df,
            ticker_train_df=ticker_train_df,
            ticker_test_df=ticker_test_df,
            ticker=str(ticker),
            feature_columns=feature_columns,
            preprocess_params=preprocess_params,
            test_row_count=test_row_count,
            data_preprocessor_cls=self.data_preprocessor_cls,
            tier_name=tier_name,
        )
        preprocessed_df = PecnetDataPreprocessor._pecnet_preprocessed_training_frame(
            ticker_data=ticker_data,
            tier_name=tier_name,
        )
        PecnetDataPreprocessor._log_pecnet_preprocessed_inputs(
            preprocessed_df=preprocessed_df,
            tier_name=tier_name,
            ticker=str(ticker),
        )
        store_metadata = (
            PecnetDataPreprocessor._publish_pecnet_preprocessed_inputs(preprocessed_df)
            if publish_to_store
            else PecnetDataPreprocessor._deferred_pecnet_preprocessed_store_metadata(preprocessed_df)
        )
        return ticker_data, store_metadata, preprocessed_df











    @staticmethod
    def _preprocess_ticker(
        *,
        ticker_df: pd.DataFrame,
        ticker_train_df: pd.DataFrame,
        ticker_test_df: pd.DataFrame,
        ticker: str,
        feature_columns: list[str],
        preprocess_params: dict[str, Any],
        test_row_count: int,
        data_preprocessor_cls,
        tier_name: str,
    ) -> dict[str, Any]:
        dp = data_preprocessor_cls()
        dp.reset()

        if not ticker_train_df.empty and not ticker_test_df.empty:
            ticker_df = pd.concat(
                [ticker_train_df, ticker_test_df],
                ignore_index=True,
            )
        ticker_df = (
            ticker_df.sort_values("ds")
            .drop_duplicates(subset=["unique_id", "ds"], keep="last")
            .reset_index(drop=True)
            .copy()
        )
        test_ratio = PecnetDataPreprocessor._preprocessor_test_ratio(
            row_count=len(ticker_df),
            test_row_count=test_row_count,
            preprocess_params=preprocess_params,
        )
        params = {
            **preprocess_params,
            "test_ratio": test_ratio,
        }

        target_series = ticker_df["y"].to_numpy(dtype=float)
        X_train_target, X_test_target, y_train, y_test = dp.preprocess(
            data=target_series,
            profile="target",
            fit=True,
            **params,
        )
        preprocessor_artifacts = []
        target_artifact = PecnetDataPreprocessor._log_fitted_preprocessor_artifact(
            preprocessor=dp,
            tier_name=tier_name,
            ticker=ticker,
            variable_name="target",
            variable_kind="target",
            profile="target",
            row_count=len(target_series),
        )
        if target_artifact:
            preprocessor_artifacts.append(target_artifact)

        feature_X_trains = []
        feature_X_tests = []
        available_feature_columns = [
            column for column in feature_columns if column in ticker_df.columns
        ]
        for column in available_feature_columns:
            X_train_feature, X_test_feature, _, _ = dp.preprocess(
                data=ticker_df[column].to_numpy(dtype=float),
                profile=f"feature:{column}",
                fit=True,
                **params,
            )
            feature_X_trains.append(X_train_feature)
            feature_X_tests.append(X_test_feature)
            feature_artifact = PecnetDataPreprocessor._log_fitted_preprocessor_artifact(
                preprocessor=dp,
                tier_name=tier_name,
                ticker=ticker,
                variable_name=column,
                variable_kind="features",
                profile=f"feature:{column}",
                row_count=len(ticker_df[column]),
            )
            if feature_artifact:
                preprocessor_artifacts.append(feature_artifact)

        combined_artifact = PecnetDataPreprocessor._log_fitted_preprocessor_artifact(
            preprocessor=dp,
            tier_name=tier_name,
            ticker=ticker,
            variable_name="combined",
            variable_kind="combined",
            profile="all_profiles",
            row_count=len(ticker_df),
        )
        if combined_artifact:
            preprocessor_artifacts.append(combined_artifact)
        PecnetDataPreprocessor._log_preprocessor_manifest(
            tier_name=tier_name,
            ticker=ticker,
            preprocessor_artifacts=preprocessor_artifacts,
            feature_columns=available_feature_columns,
            preprocess_params=params,
        )

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
            "preprocessor_artifacts": preprocessor_artifacts,
        }

    @staticmethod
    def _log_fitted_preprocessor_artifact(
        *,
        preprocessor,
        tier_name: str,
        ticker: str,
        variable_name: str,
        variable_kind: str,
        profile: str,
        row_count: int,
    ) -> dict[str, Any] | None:
        try:
            import joblib  # noqa: PLC0415
            import mlflow  # noqa: PLC0415
        except ImportError as exc:
            LOGGER.warning("Skipping PECNet preprocessor logging: %s", exc)
            return None

        if mlflow.active_run() is None:
            return None

        tier_safe = _safe_name(str(tier_name))
        ticker_safe = _safe_name(str(ticker))
        variable_safe = _safe_name(str(variable_name))
        artifact_dir = (
            f"pecnet/{tier_safe}/tickers/{ticker_safe}/preprocessors/{variable_kind}"
        )
        filename = f"{variable_safe}.joblib"
        artifact_uri = f"{artifact_dir}/{filename}"
        metadata = {
            "tier": tier_name,
            "ticker": str(ticker),
            "variable_name": str(variable_name),
            "variable_kind": variable_kind,
            "profile": profile,
            "row_count": int(row_count),
            "preprocessor_class": type(preprocessor).__name__,
            "artifact_path": artifact_uri,
            "logged": False,
        }

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                path = Path(tmpdir) / filename
                joblib.dump(preprocessor, path, compress=3)
                mlflow.log_artifact(str(path), artifact_path=artifact_dir)
        except Exception as exc:  # noqa: BLE001
            metadata["error"] = str(exc)
            LOGGER.warning(
                "Failed to log PECNet preprocessor artifact | tier=%s ticker=%s "
                "variable=%s",
                tier_name,
                ticker,
                variable_name,
                exc_info=True,
            )
            return metadata

        metadata["logged"] = True
        return metadata

    @staticmethod
    def _log_preprocessor_manifest(
        *,
        tier_name: str,
        ticker: str,
        preprocessor_artifacts: list[dict[str, Any]],
        feature_columns: list[str],
        preprocess_params: dict[str, Any],
    ) -> None:
        if not preprocessor_artifacts:
            return

        try:
            import mlflow  # noqa: PLC0415
        except ImportError:
            return

        if mlflow.active_run() is None:
            return

        ticker_safe = _safe_name(str(ticker))
        mlflow.log_dict(
            {
                "tier": tier_name,
                "ticker": str(ticker),
                "feature_columns": feature_columns,
                "preprocess_params": preprocess_params,
                "preprocessors": preprocessor_artifacts,
            },
            f"pecnet/{tier_name}/tickers/{ticker_safe}/preprocessors/manifest.json",
        )

    @staticmethod
    def _preprocessor_test_ratio(
        *,
        row_count: int,
        test_row_count: int,
        preprocess_params: dict[str, Any],
    ) -> float:
        explicit_ratio = preprocess_params.get("test_ratio")
        if explicit_ratio is not None:
            return float(explicit_ratio)

        if test_row_count <= 0:
            return _ticker_test_row_ratio(row_count, test_row_count)

        sampling_periods = preprocess_params.get("sampling_periods") or [1, 4]
        sequence_size = int(preprocess_params.get("sequence_size", 4))
        conjoincy = bool(preprocess_params.get("conjoincy", False))
        stride = preprocess_params.get("stride")
        stride_val = int(stride) if stride and int(stride) > 1 else 1
        biggest_period = max(sampling_periods)
        required_timestamps = (
            biggest_period + sequence_size - 1
            if conjoincy
            else biggest_period * sequence_size
        )
        data_trimmed = max(row_count - required_timestamps + stride_val, 1)
        return min(max(test_row_count / data_trimmed, 0.01), 0.95)

    @staticmethod
    def _as_2d_float_array(values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 0:
            return array.reshape(1, 1)
        if array.ndim == 1:
            return array.reshape(-1, 1)
        return array.reshape(array.shape[0], -1)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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
            for variable_index, variable_name, values in PecnetDataPreprocessor._iter_preprocessed_variable_specs(
                ticker_data,
                split_name=split_name,
            ):
                matrix = PecnetDataPreprocessor._as_2d_float_array(values)
                sample_count = min(matrix.shape[0], len(y_values))
                dates = PecnetDataPreprocessor._preprocessed_dates(
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

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _deferred_pecnet_preprocessed_store_metadata(
        preprocessed_df: pd.DataFrame,
    ) -> dict[str, object]:
        return {
            "timescale_rows": 0,
            "feast_online_rows": 0,
            "deferred_rows": len(preprocessed_df),
            "deferred_to_parent_process": True,
        }
