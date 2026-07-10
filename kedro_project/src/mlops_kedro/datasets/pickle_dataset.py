from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from kedro.io import AbstractDataset


class PickleDataset(AbstractDataset):
    def __init__(self, filepath: str) -> None:
        self._filepath = Path(filepath)

    def _load(self) -> Any:
        with self._filepath.open("rb") as file:
            return pickle.load(file)

    def _save(self, data: Any) -> None:
        self._filepath.parent.mkdir(parents=True, exist_ok=True)
        with self._filepath.open("wb") as file:
            pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def _exists(self) -> bool:
        return self._filepath.exists()

    def _describe(self) -> dict[str, str]:
        return {"filepath": str(self._filepath)}
