from __future__ import annotations

import pickle
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.preprocessing import PowerTransformer


@dataclass(slots=True)
class ForecastYeoJohnsonPowerTransformer:
    method: str = "yeo-johnson"
    standardize: bool = True
    target_column: str = "y"
    target_transformer: PowerTransformer | None = None
    feature_transformer: PowerTransformer | None = None
    feature_columns: list[str] = field(default_factory=list)
    _target_before: np.ndarray = field(default_factory=lambda: np.asarray([]))
    _target_after: np.ndarray = field(default_factory=lambda: np.asarray([]))
    _features_before: np.ndarray = field(default_factory=lambda: np.asarray([]))
    _features_after: np.ndarray = field(default_factory=lambda: np.asarray([]))

    def fit_transform_frames(
        self,
        *,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_columns: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        transformed_train = train_df.copy()
        transformed_test = test_df.copy()
        self.target_transformer = PowerTransformer(
            method=self.method,
            standardize=self.standardize,
        )
        train_target = transformed_train[[self.target_column]].astype(float)
        self.target_transformer.fit(train_target)
        transformed_train[self.target_column] = self.target_transformer.transform(
            train_target,
        ).ravel()
        transformed_test[self.target_column] = self.target_transformer.transform(
            transformed_test[[self.target_column]].astype(float),
        ).ravel()
        self._target_before = self._finite_values(train_target.to_numpy().ravel())
        self._target_after = self._finite_values(
            transformed_train[self.target_column].to_numpy(),
        )
        self.feature_columns = self._numeric_feature_columns(train_df, feature_columns)
        if self.feature_columns:
            self.feature_transformer = PowerTransformer(
                method=self.method,
                standardize=self.standardize,
            )
            train_features = transformed_train[self.feature_columns].astype(float)
            self.feature_transformer.fit(train_features)
            transformed_train.loc[:, self.feature_columns] = self.feature_transformer.transform(
                train_features,
            )
            transformed_test.loc[:, self.feature_columns] = self.feature_transformer.transform(
                transformed_test[self.feature_columns].astype(float),
            )
            self._features_before = self._finite_values(train_features.to_numpy().ravel())
            self._features_after = self._finite_values(
                transformed_train[self.feature_columns].to_numpy().ravel(),
            )
        return transformed_train, transformed_test

    def inverse_transform_predictions(self, predictions: pd.DataFrame) -> pd.DataFrame:
        if self.target_transformer is None:
            return predictions

        restored = predictions.copy()
        for column in self._prediction_value_columns(restored):
            values = restored[column]
            valid_mask = values.notna()
            if not valid_mask.any():
                continue

            restored.loc[valid_mask, column] = self.target_transformer.inverse_transform(
                values.loc[valid_mask].astype(float).to_numpy().reshape(-1, 1),
            ).ravel()
        return restored

    def log_artifacts(self, *, model_family: str, tier_name: str) -> None:
        metadata = {
            "transformer": "sklearn.preprocessing.PowerTransformer",
            "method": self.method,
            "standardize": self.standardize,
            "target_column": self.target_column,
            "feature_columns": self.feature_columns,
        }
        mlflow.log_params(
            {
                "power_transformer.method": self.method,
                "power_transformer.standardize": self.standardize,
                "power_transformer.target_column": self.target_column,
                "power_transformer.feature_count": len(self.feature_columns),
            }
        )
        mlflow.log_dict(
            metadata,
            f"{model_family}/{tier_name}/transforms/power_transformer_metadata.json",
        )
        self._log_pickle(model_family=model_family, tier_name=tier_name)
        self._save_and_log_distribution_plots(
            model_family=model_family,
            tier_name=tier_name,
        )

    def _log_pickle(self, *, model_family: str, tier_name: str) -> None:
        with tempfile.TemporaryDirectory(prefix="stock_close_power_transformer_") as temp_dir:
            path = Path(temp_dir) / "power_transformers.pkl"
            with path.open("wb") as file_obj:
                pickle.dump(
                    {
                        "target_transformer": self.target_transformer,
                        "feature_transformer": self.feature_transformer,
                        "feature_columns": self.feature_columns,
                        "method": self.method,
                        "standardize": self.standardize,
                    },
                    file_obj,
                )
            mlflow.log_artifact(
                str(path),
                artifact_path=f"{model_family}/{tier_name}/transforms",
            )

    def _save_and_log_distribution_plots(
        self,
        *,
        model_family: str,
        tier_name: str,
    ) -> None:
        output_dir = self._transform_output_dir(model_family=model_family, tier_name=tier_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_distribution_plot(
            before=self._target_before,
            after=self._target_after,
            title=f"{model_family} {tier_name} target distribution",
            output_path=output_dir / "target_before_after_distribution.png",
            artifact_file=(
                f"{model_family}/{tier_name}/transforms/"
                "target_before_after_distribution.png"
            ),
        )
        if self._features_before.size and self._features_after.size:
            self._save_distribution_plot(
                before=self._features_before,
                after=self._features_after,
                title=f"{model_family} {tier_name} feature distribution",
                output_path=output_dir / "features_before_after_distribution.png",
                artifact_file=(
                    f"{model_family}/{tier_name}/transforms/"
                    "features_before_after_distribution.png"
                ),
            )

    @staticmethod
    def _save_distribution_plot(
        *,
        before: np.ndarray,
        after: np.ndarray,
        title: str,
        output_path: Path,
        artifact_file: str,
    ) -> None:
        if before.size == 0 or after.size == 0:
            return

        figure, axes = plt.subplots(ncols=2, figsize=(12, 4), constrained_layout=True)
        axes[0].hist(before, bins=60, color="#64748b", alpha=0.85)
        axes[0].set_title("Before transform")
        axes[0].set_ylabel("Frequency")
        axes[1].hist(after, bins=60, color="#2563eb", alpha=0.85)
        axes[1].set_title("After Yeo-Johnson + standardization")
        figure.suptitle(title)
        figure.savefig(output_path, dpi=140, bbox_inches="tight")
        mlflow.log_figure(figure, artifact_file=artifact_file)
        plt.close(figure)

    @staticmethod
    def _numeric_feature_columns(
        train_df: pd.DataFrame,
        feature_columns: list[str],
    ) -> list[str]:
        return [
            column
            for column in feature_columns
            if column in train_df.columns and is_numeric_dtype(train_df[column])
        ]

    @staticmethod
    def _prediction_value_columns(predictions: pd.DataFrame) -> list[str]:
        return [
            column
            for column in predictions.columns
            if column not in {"unique_id", "ds"} and is_numeric_dtype(predictions[column])
        ]

    @staticmethod
    def _finite_values(values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float).ravel()
        return array[np.isfinite(array)]

    @staticmethod
    def _transform_output_dir(*, model_family: str, tier_name: str) -> Path:
        root = ForecastYeoJohnsonPowerTransformer._project_root()
        return (
            root
            / "outputs"
            / "transforms"
            / ForecastYeoJohnsonPowerTransformer._safe_name(model_family)
            / ForecastYeoJohnsonPowerTransformer._safe_name(tier_name)
        )

    @staticmethod
    def _project_root() -> Path:
        cwd = Path.cwd().resolve()
        for candidate in [cwd, *cwd.parents]:
            if (candidate / "kedro_project").exists() and (candidate / "outputs").exists():
                return candidate
            if candidate.name == "kedro_project":
                return candidate.parent
        return cwd

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unknown"
