from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class LightweightArtifactStore:
    root_dir: Path | str | None = None
    enabled: bool | None = None

    def __post_init__(self) -> None:
        self.root_dir = self._resolve_root_dir(
            self.root_dir or self._default_root_dir()
        ).expanduser()
        if self.enabled is None:
            self.enabled = self._env_bool("LOCAL_ARTIFACTS_ENABLED", True)

    def save_params(self, payload: dict[str, Any], artifact_file: str) -> Path | None:
        return self._safe_write(
            artifact_file,
            lambda path: path.write_text(
                json.dumps(payload, indent=2, sort_keys=True, default=str),
                encoding="utf-8",
            ),
        )

    def save_metrics(self, frame: pd.DataFrame, artifact_file: str) -> Path | None:
        if frame is None or frame.empty:
            return None
        return self._safe_write(
            artifact_file,
            lambda path: frame.to_csv(path, index=False),
        )

    def save_plot(self, figure: Any, artifact_file: str) -> Path | None:
        return self._safe_write(
            artifact_file,
            lambda path: figure.savefig(path, dpi=120, bbox_inches="tight"),
        )

    def _safe_write(
        self,
        artifact_file: str,
        writer: Callable[[Path], Any],
    ) -> Path | None:
        if not self.enabled:
            return None

        path = self._artifact_path(artifact_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            writer(path)
            return path
        except Exception:
            LOGGER.warning(
                "Could not write lightweight local artifact: %s",
                path,
                exc_info=True,
            )
            return None

    def _artifact_path(self, artifact_file: str) -> Path:
        relative = Path(str(artifact_file).lstrip("/"))
        safe_parts = [
            self._safe_path_part(part)
            for part in relative.parts
            if part not in {"", "."}
        ]
        return Path(self.root_dir, *safe_parts)

    @staticmethod
    def _default_root_dir() -> Path:
        configured = os.getenv("LOCAL_ARTIFACT_DIR") or os.getenv(
            "PROJECT_LOCAL_ARTIFACT_DIR",
        )
        if configured:
            return LightweightArtifactStore._resolve_root_dir(configured)

        return (
            LightweightArtifactStore._project_root()
            / "outputs"
            / "artifacts"
            / "stock_close_training"
        )

    @staticmethod
    def _resolve_root_dir(path: Path | str) -> Path:
        root_dir = Path(path).expanduser()
        if root_dir.is_absolute():
            return root_dir
        return LightweightArtifactStore._project_root() / root_dir

    @staticmethod
    def _project_root() -> Path:
        cwd = Path.cwd().resolve()
        if cwd.name == "kedro_project":
            return cwd.parent
        for candidate in [cwd, *cwd.parents]:
            if (candidate / "kedro_project").exists():
                return candidate
        return cwd

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _safe_path_part(value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")
        return safe or "artifact"
