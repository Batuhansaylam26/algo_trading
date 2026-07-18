from __future__ import annotations

import os
import sys

from .constants import FEATURE_REPO_DIR


def _ensure_feature_repo_on_path() -> None:
    (FEATURE_REPO_DIR / "data").mkdir(parents=True, exist_ok=True)
    feature_repo_parent = str(FEATURE_REPO_DIR.parent)
    if feature_repo_parent not in sys.path:
        sys.path.insert(0, feature_repo_parent)

def _timescale_connection_kwargs() -> dict[str, str | int]:
    return {
        "host": os.getenv("TIMESCALE_HOST", "host.docker.internal"),
        "port": int(os.getenv("TIMESCALE_PORT", "5432")),
        "dbname": os.getenv("TIMESCALE_DB", "dataops"),
        "user": os.getenv("TIMESCALE_USER", "dataops"),
        "password": os.getenv("TIMESCALE_PASSWORD", "dataops"),
    }

def _schema_name(table_name: str) -> str:
    return table_name.split(".", maxsplit=1)[0]
