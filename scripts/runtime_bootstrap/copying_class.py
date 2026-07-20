from __future__ import annotations

import errno
import shutil
import time
from pathlib import Path

from .paths import (
    DEADLOCK_RETRY_COUNT,
    DEADLOCK_RETRY_SECONDS,
    EXCLUDED_DIRS,
    EXCLUDED_FILES,
    EXCLUDED_SUFFIXES,
    KEDRO_RUNTIME,
    KEDRO_RUNTIME_OVERLAY_FILES,
    KEDRO_SOURCE,
    PECNET_DEADLOCK_RETRY_COUNT,
    PECNET_RUNTIME,
    PECNET_RUNTIME_FILES,
    PECNET_SOURCE,
)













class RuntimeBootstrapCopier:

    @staticmethod
    def should_skip(path: Path) -> bool:
        return (
            any(part in EXCLUDED_DIRS for part in path.parts)
            or path.name in EXCLUDED_FILES
            or path.suffix in EXCLUDED_SUFFIXES
        )

    @staticmethod
    def reset_directory(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        for child in path.iterdir():
            try:
                if child.is_dir() and not child.is_symlink():
                    shutil.rmtree(child)
                else:
                    child.unlink()
            except OSError as error:
                if error.errno == errno.EBUSY:
                    print(f"Skipping busy runtime path during reset: {child}")
                    continue
                raise

    @staticmethod
    def copy_file_with_retry(
        source: Path,
        target: Path,
        *,
        retry_count: int = DEADLOCK_RETRY_COUNT,
    ) -> None:
        for attempt in range(1, retry_count + 1):
            try:
                shutil.copy2(source, target)
                return
            except OSError as error:
                if error.errno != errno.EDEADLK or attempt == retry_count:
                    raise
                time.sleep(DEADLOCK_RETRY_SECONDS * attempt)

    @staticmethod
    def copy_tree(source: Path, target: Path) -> None:
        for path in source.rglob("*"):
            relative_path = path.relative_to(source)
            if RuntimeBootstrapCopier.should_skip(relative_path):
                continue
            target_path = target / relative_path
            if path.is_dir():
                target_path.mkdir(parents=True, exist_ok=True)
                continue
            target_path.parent.mkdir(parents=True, exist_ok=True)
            RuntimeBootstrapCopier.copy_file_with_retry(path, target_path)

    @staticmethod
    def copy_selected_files(
        source: Path,
        target: Path,
        relative_paths: list[str],
        *,
        retry_count: int = DEADLOCK_RETRY_COUNT,
    ) -> None:
        for relative_path in relative_paths:
            source_path = source / relative_path
            target_path = target / relative_path
            target_path.parent.mkdir(parents=True, exist_ok=True)
            RuntimeBootstrapCopier.copy_file_with_retry(source_path, target_path, retry_count=retry_count)

    @staticmethod
    def has_runtime_contents(path: Path) -> bool:
        return path.exists() and any(path.iterdir())

    @staticmethod
    def refresh_tree_from_source(source: Path, target: Path, description: str) -> bool:
        staging = target.parent / f".{target.name}.next"
        RuntimeBootstrapCopier.reset_directory(staging)

        try:
            RuntimeBootstrapCopier.copy_tree(source, staging)
        except OSError as error:
            if error.errno != errno.EDEADLK:
                raise
            RuntimeBootstrapCopier.reset_directory(staging)
            if RuntimeBootstrapCopier.has_runtime_contents(target):
                print(f"Kept existing {description}: source copy hit EDEADLK")
                return False
            raise

        RuntimeBootstrapCopier.reset_directory(target)
        for child in staging.iterdir():
            shutil.move(str(child), target / child.name)
        staging.rmdir()
        return True

    @staticmethod
    def refresh_pecnet_runtime() -> bool:
        staging = PECNET_RUNTIME.parent / f".{PECNET_RUNTIME.name}.next"
        RuntimeBootstrapCopier.reset_directory(staging)

        try:
            RuntimeBootstrapCopier.copy_selected_files(
                PECNET_SOURCE,
                staging,
                PECNET_RUNTIME_FILES,
                retry_count=PECNET_DEADLOCK_RETRY_COUNT,
            )
        except OSError as error:
            if error.errno != errno.EDEADLK:
                raise
            RuntimeBootstrapCopier.reset_directory(staging)
            if RuntimeBootstrapCopier.has_runtime_contents(PECNET_RUNTIME):
                print("Kept existing PecNet framework runtime: source copy hit EDEADLK")
                return False
            raise

        RuntimeBootstrapCopier.reset_directory(PECNET_RUNTIME)
        for child in staging.iterdir():
            shutil.move(str(child), PECNET_RUNTIME / child.name)
        staging.rmdir()
        return True

    @staticmethod
    def try_runtime_step(description: str, function) -> bool:
        try:
            result = function()
        except OSError as error:
            if error.errno == errno.EDEADLK:
                print(f"Skipped {description}: resource deadlock avoided")
                return False
            raise
        return bool(result) if result is not None else True

    @staticmethod
    def sync_kedro_runtime_overlay() -> None:
        existing_files = [
            relative_path
            for relative_path in KEDRO_RUNTIME_OVERLAY_FILES
            if (KEDRO_SOURCE / relative_path).exists()
        ]
        RuntimeBootstrapCopier.copy_selected_files(KEDRO_SOURCE, KEDRO_RUNTIME, existing_files)
