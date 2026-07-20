import os
import warnings


class TrainingRuntime:

    @staticmethod
    def cpu_count_from_env(env_name: str, default: int = 1) -> int:
        requested = int(os.getenv(env_name, str(default)))
        if requested <= 0:
            return os.cpu_count() or 1
        return requested

    @staticmethod
    def bool_from_env(env_name: str, default: bool = False) -> bool:
        value = os.getenv(env_name)
        if value is None:
            return default
        return value.lower() in {"1", "true", "yes"}

    @staticmethod
    def filter_sklearn_parallel_warnings() -> None:
        warnings.filterwarnings(
            "ignore",
            message=(
                "`sklearn.utils.parallel.delayed` should be used with "
                "`sklearn.utils.parallel.Parallel`.*"
            ),
            category=UserWarning,
            module="sklearn.utils.parallel",
        )
