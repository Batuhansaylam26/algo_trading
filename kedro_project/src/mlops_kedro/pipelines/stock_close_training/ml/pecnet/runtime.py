from __future__ import annotations

from .runtime_class import *  # noqa: F403
from .runtime_class import PecnetRuntime

pecnet_runtime = PecnetRuntime()
_resolve_pecnetframework_path = pecnet_runtime._resolve_pecnetframework_path
_load_pecnet_runtime = pecnet_runtime._load_pecnet_runtime
_safe_name = pecnet_runtime._safe_name
_configure_torch_threads = pecnet_runtime._configure_torch_threads
_patch_basic_nn_device_selection = pecnet_runtime._patch_basic_nn_device_selection
_resolve_torch_device_name = pecnet_runtime._resolve_torch_device_name
_requested_torch_device = pecnet_runtime._requested_torch_device
_torch_mps_available = pecnet_runtime._torch_mps_available
_ticker_test_ratio = pecnet_runtime._ticker_test_ratio
