from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from ..runtime import cpu_count_from_env


LOGGER = logging.getLogger(__name__)


class PecnetRuntime:

    @staticmethod
    def _resolve_pecnetframework_path() -> Path:
        module_path = Path(__file__).resolve()
        candidates = [
            "/opt/pecnetframework",
            os.getenv("PECNETFRAMEWORK_PATH"),
            "/workspaces/yahooquery_lakehouse_revamp/pecnetframework",
            *(
                str(parent / "pecnetframework")
                for parent in module_path.parents
            ),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate).expanduser().resolve()
            if (path / "pecnet").exists():
                return path
        raise FileNotFoundError(
            "pecnetframework klasoru bulunamadi. PECNETFRAMEWORK_PATH env var ile "
            "klasoru goster veya repo root altina pecnetframework koy."
        )

    @staticmethod
    def _load_pecnet_runtime():
        pecnet_path = PecnetRuntime._resolve_pecnetframework_path()
        if str(pecnet_path) not in sys.path:
            sys.path.insert(0, str(pecnet_path))

        from pecnet.network import PecnetBuilder  # noqa: PLC0415
        from pecnet.models.BasicNN import BasicNN  # noqa: PLC0415
        from pecnet.preprocessing.DataPreprocessor import DataPreprocessor  # noqa: PLC0415
        from pecnet.utils import FeatureSelector, Utility  # noqa: PLC0415

        import torch  # noqa: PLC0415

        PecnetRuntime._patch_basic_nn_device_selection(BasicNN, torch)
        return Utility, PecnetBuilder, DataPreprocessor, BasicNN, FeatureSelector, torch

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)

    @staticmethod
    def _configure_torch_threads(torch_module) -> dict[str, Any]:
        requested_threads = cpu_count_from_env("MODEL_N_JOBS")
        torch_module.set_num_threads(requested_threads)

        interop_threads = min(requested_threads, 4)
        try:
            torch_module.set_num_interop_threads(interop_threads)
        except RuntimeError:
            interop_threads = (
                torch_module.get_num_interop_threads()
                if hasattr(torch_module, "get_num_interop_threads")
                else None
            )

        return {
            "torch_num_threads": int(torch_module.get_num_threads()),
            "torch_num_interop_threads": (
                int(interop_threads) if interop_threads is not None else None
            ),
            "torch_device": PecnetRuntime._resolve_torch_device_name(torch_module),
            "torch_device_requested": PecnetRuntime._requested_torch_device(),
            "torch_mps_available": PecnetRuntime._torch_mps_available(torch_module),
            "torch_cuda_available": bool(torch_module.cuda.is_available()),
        }

    @staticmethod
    def _patch_basic_nn_device_selection(basic_nn_cls, torch_module) -> None:
        if getattr(basic_nn_cls, "_dataops_device_patch_applied", False):
            return

        def init_devices(self):
            device_name = PecnetRuntime._resolve_torch_device_name(torch_module)
            self.device = torch_module.device(device_name)
            self.to(self.device)
            if not getattr(basic_nn_cls, "_dataops_device_logged", False):
                LOGGER.info("PECNet BasicNN torch device selected: %s", self.device)
                print(f"[PECNet] Torch device selected: {self.device}")
                setattr(basic_nn_cls, "_dataops_device_logged", True)

        basic_nn_cls.init_devices = init_devices
        setattr(basic_nn_cls, "_dataops_device_patch_applied", True)

    @staticmethod
    def _resolve_torch_device_name(torch_module) -> str:
        requested = PecnetRuntime._requested_torch_device()
        if requested in {"mps", "metal"}:
            if PecnetRuntime._torch_mps_available(torch_module):
                return "mps"
            LOGGER.warning("PECNET_TORCH_DEVICE=mps requested but MPS is unavailable.")
            return "cpu"
        if requested == "cuda":
            if torch_module.cuda.is_available():
                return "cuda"
            LOGGER.warning("PECNET_TORCH_DEVICE=cuda requested but CUDA is unavailable.")
            return "cpu"
        if requested == "cpu":
            return "cpu"
        if PecnetRuntime._torch_mps_available(torch_module):
            return "mps"
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _requested_torch_device() -> str:
        return os.getenv("PECNET_TORCH_DEVICE", "auto").strip().lower() or "auto"

    @staticmethod
    def _torch_mps_available(torch_module) -> bool:
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        return bool(mps_backend and mps_backend.is_available())

    @staticmethod
    def _ticker_test_ratio(row_count: int, test_horizon: int) -> float:
        if row_count <= test_horizon:
            raise ValueError(
                f"PECNet needs more rows than test_horizon. rows={row_count}, "
                f"test_horizon={test_horizon}"
            )
        return min(max(test_horizon / row_count, 0.01), 0.5)
