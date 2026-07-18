from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from ..runtime import cpu_count_from_env


def _resolve_pecnetframework_path() -> Path:
    candidates = [
        "/opt/pecnetframework",
        os.getenv("PECNETFRAMEWORK_PATH"),
        str(Path(__file__).resolve().parents[6] / "pecnetframework"),
        "/workspaces/yahooquery_lakehouse_revamp/pecnetframework",
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

def _load_pecnet_runtime():
    pecnet_path = _resolve_pecnetframework_path()
    if str(pecnet_path) not in sys.path:
        sys.path.insert(0, str(pecnet_path))

    from pecnet.network import PecnetBuilder  # noqa: PLC0415
    from pecnet.models.BasicNN import BasicNN  # noqa: PLC0415
    from pecnet.preprocessing.DataPreprocessor import DataPreprocessor  # noqa: PLC0415
    from pecnet.utils import FeatureSelector, Utility  # noqa: PLC0415

    import torch  # noqa: PLC0415

    return Utility, PecnetBuilder, DataPreprocessor, BasicNN, FeatureSelector, torch

def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)

def _configure_torch_threads(torch_module) -> dict[str, int | None]:
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
    }

def _ticker_test_ratio(row_count: int, test_horizon: int) -> float:
    if row_count <= test_horizon:
        raise ValueError(
            f"PECNet needs more rows than test_horizon. rows={row_count}, "
            f"test_horizon={test_horizon}"
        )
    return min(max(test_horizon / row_count, 0.01), 0.5)
