import os


def cpu_count_from_env(env_name: str, default: int = 1) -> int:
    requested = int(os.getenv(env_name, str(default)))
    if requested <= 0:
        return os.cpu_count() or 1
    return requested
