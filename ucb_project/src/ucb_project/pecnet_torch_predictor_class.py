from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .mlflow_forecast_repository_class import MlflowForecastRepository


@dataclass(slots=True)
class PecnetTorchPredictor:
    """Loads a logged PECNet .pt artifact and runs torch prediction when inputs exist."""

    repository: MlflowForecastRepository

    def predict_from_artifact(
        self,
        *,
        run_id: str,
        model_artifact_path: str,
        input_artifact_path: str | None = None,
        arrays: dict[str, Any] | None = None,
        device: str = "auto",
    ) -> dict[str, Any]:
        torch = self._torch_module()
        selected_device = self._device(torch, device)
        model_path = self._model_file(
            self.repository.download_artifact(run_id, model_artifact_path)
        )
        model = torch.load(model_path, map_location=selected_device, weights_only=False)
        if hasattr(model, "eval"):
            model.eval()

        x_arrays, y_test = self._prediction_inputs(
            run_id=run_id,
            input_artifact_path=input_artifact_path,
            arrays=arrays,
        )
        with torch.no_grad():
            if y_test is None:
                predictions = model.predict(*x_arrays)
            else:
                predictions = model.predict(*x_arrays, test_target=y_test)

        return {
            "model_path": str(model_path),
            "device": str(selected_device),
            "prediction_count": int(len(np.asarray(self._to_numpy(predictions)).reshape(-1))),
            "predictions": self._to_numpy(predictions).reshape(-1).tolist(),
        }

    def latest_logged_forecast(
        self,
        *,
        run_id: str,
        prediction_artifact_path: str,
        ticker: str,
    ) -> pd.DataFrame:
        import pandas as pd

        predictions = self.repository.load_artifact_table(run_id, prediction_artifact_path)
        if predictions.empty:
            return predictions
        return predictions[predictions["unique_id"].astype(str) == str(ticker)].reset_index(drop=True)

    def _prediction_inputs(
        self,
        *,
        run_id: str,
        input_artifact_path: str | None,
        arrays: dict[str, Any] | None,
    ) -> tuple[list[np.ndarray], np.ndarray | None]:
        if arrays:
            x_values = arrays.get("X") or arrays.get("x") or []
            x_arrays = [np.asarray(value, dtype=np.float32) for value in x_values]
            y_value = arrays.get("y_test")
            return x_arrays, None if y_value is None else np.asarray(y_value, dtype=np.float32)

        if not input_artifact_path:
            raise ValueError(
                "PECNet .pt inference needs preprocessed input arrays. "
                "Pass arrays={X:[...], y_test:[...]} or input_artifact_path."
            )

        input_dir = self.repository.download_artifact(run_id, input_artifact_path)
        x_files = sorted(input_dir.glob("*X_test*.npy")) or sorted(input_dir.glob("*x_test*.npy"))
        if not x_files:
            raise ValueError(f"No X_test .npy files found under {input_artifact_path}.")
        y_files = sorted(input_dir.glob("*y_test*.npy"))
        x_arrays = [np.load(path) for path in x_files]
        y_test = np.load(y_files[0]) if y_files else None
        return x_arrays, y_test

    @staticmethod
    def _model_file(path: Path) -> Path:
        if path.is_file() and path.suffix == ".pt":
            return path
        pt_files = sorted(path.rglob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No .pt model file found under {path}.")
        return pt_files[0]

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _torch_module():
        import torch

        return torch

    @staticmethod
    def _device(torch, requested: str):
        if requested != "auto":
            return torch.device(requested)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
