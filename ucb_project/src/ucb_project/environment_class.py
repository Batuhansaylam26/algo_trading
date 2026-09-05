from __future__ import annotations

import os
from pathlib import Path


class LocalMlflowEnvironment:
    """Applies local/container MLflow defaults used by this project."""

    @staticmethod
    def apply_defaults() -> None:
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "60")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_BACKOFF_FACTOR", "1")
        os.environ.setdefault(
            "MLFLOW_TRACKING_URI",
            LocalMlflowEnvironment.service_url(5001),
        )
        os.environ.setdefault(
            "MLFLOW_S3_ENDPOINT_URL",
            LocalMlflowEnvironment.service_url(9000),
        )
        os.environ.setdefault("AWS_ACCESS_KEY_ID", "admin")
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "admin1234")
        os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

    @staticmethod
    def service_url(port: int) -> str:
        host = "host.docker.internal" if LocalMlflowEnvironment.in_container() else "127.0.0.1"
        return f"http://{host}:{port}"

    @staticmethod
    def normalize_service_url(value: str | None, *, port: int) -> str:
        if not value or value.lower() == "auto":
            return LocalMlflowEnvironment.service_url(port)
        if LocalMlflowEnvironment.in_container():
            return value.replace("127.0.0.1", "host.docker.internal").replace(
                "localhost",
                "host.docker.internal",
            )
        return value.replace("host.docker.internal", "127.0.0.1")

    @staticmethod
    def in_container() -> bool:
        return Path("/.dockerenv").exists() or Path("/workspaces").exists()
