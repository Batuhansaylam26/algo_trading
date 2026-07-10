from collections.abc import Sequence

from dagster._core.storage.db_io_manager import DbTypeHandler
from dagster_deltalake import DeltaLakeIOManager
from dagster_deltalake_polars import DeltaLakePolarsTypeHandler


class MyDeltaLakeIOManager(DeltaLakeIOManager):
    @staticmethod
    def type_handlers() -> Sequence[DbTypeHandler]:
        return [DeltaLakePolarsTypeHandler()]
