from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from ..runtime import cpu_count_from_env


LOGGER = logging.getLogger(__name__)


class PecnetRuntime:

    @staticmethod
    def _resolve_pecnetframework_path() -> Path:
        module_path = Path(__file__).resolve()
        candidates = [
            "/opt/pecnetframework",
            os.getenv("PECNETFRAMEWORK_PATH"),
            "/workspaces/yahooquery_lakehouse_revamp/pecnetframework",
            *(
                str(parent / "pecnetframework")
                for parent in module_path.parents
            ),
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

    @staticmethod
    def _load_pecnet_runtime():
        pecnet_path = PecnetRuntime._resolve_pecnetframework_path()
        if str(pecnet_path) not in sys.path:
            sys.path.insert(0, str(pecnet_path))

        from pecnet.network import PecnetBuilder  # noqa: PLC0415
        from pecnet.models.BasicNN import BasicNN  # noqa: PLC0415
        from pecnet.preprocessing.DataPreprocessor import DataPreprocessor  # noqa: PLC0415
        from pecnet.utils import FeatureSelector, Utility  # noqa: PLC0415

        import torch  # noqa: PLC0415

        PecnetRuntime._patch_basic_nn_device_selection(BasicNN, torch)
        PecnetRuntime._patch_basic_nn_paper_training_options(BasicNN, Utility)
        PecnetRuntime._patch_data_preprocessor_paper_mode(DataPreprocessor)
        return Utility, PecnetBuilder, DataPreprocessor, BasicNN, FeatureSelector, torch

    @staticmethod
    def _safe_name(value: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value)

    @staticmethod
    def _configure_torch_threads(torch_module) -> dict[str, Any]:
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
            "torch_device": PecnetRuntime._resolve_torch_device_name(torch_module),
            "torch_device_requested": PecnetRuntime._requested_torch_device(),
            "torch_mps_available": PecnetRuntime._torch_mps_available(torch_module),
            "torch_cuda_available": bool(torch_module.cuda.is_available()),
        }

    @staticmethod
    def _patch_basic_nn_device_selection(basic_nn_cls, torch_module) -> None:
        if getattr(basic_nn_cls, "_dataops_device_patch_applied", False):
            return

        def init_devices(self):
            device_name = PecnetRuntime._resolve_torch_device_name(torch_module)
            self.device = torch_module.device(device_name)
            self.to(self.device)
            if not getattr(basic_nn_cls, "_dataops_device_logged", False):
                LOGGER.info("PECNet BasicNN torch device selected: %s", self.device)
                print(f"[PECNet] Torch device selected: {self.device}")
                setattr(basic_nn_cls, "_dataops_device_logged", True)

        basic_nn_cls.init_devices = init_devices
        setattr(basic_nn_cls, "_dataops_device_patch_applied", True)

    @staticmethod
    def _patch_basic_nn_paper_training_options(basic_nn_cls, utility_cls) -> None:
        if getattr(basic_nn_cls, "_dataops_paper_training_patch_applied", False):
            return

        import numpy as np  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        import torch.nn.init as initializers  # noqa: PLC0415
        from torch.optim import Adam, SGD  # noqa: PLC0415

        def set_hyperparameters(
            heuristic=False,
            learning_rate=0.001,
            use_scheduler=False,
            use_dropout=False,
            dropout_rate=0.1,
            epoch_size=1000,
            batch_size=32,
            hidden_units_sizes=None,
            hidden_units_strategy=None,
            optimizer_name="adam",
            momentum=0.0,
            activation="gelu",
            use_layer_norm=True,
            epoch_size_by_network_name=None,
        ):
            utility_cls.heuristic = heuristic
            utility_cls.learning_rate = learning_rate
            utility_cls.use_scheduler = use_scheduler
            utility_cls.use_dropout = use_dropout
            utility_cls.dropout_rate = dropout_rate
            utility_cls.epoch_size = epoch_size
            utility_cls.batch_size = batch_size
            utility_cls.hidden_units_sizes = hidden_units_sizes or [32, 16]
            utility_cls.hidden_units_strategy = hidden_units_strategy
            utility_cls.optimizer_name = optimizer_name
            utility_cls.momentum = momentum
            utility_cls.activation = activation
            utility_cls.use_layer_norm = use_layer_norm
            utility_cls.epoch_size_by_network_name = epoch_size_by_network_name or {}

        def epoch_size_for_network(self, default_epoch_size):
            overrides = getattr(utility_cls, "epoch_size_by_network_name", {}) or {}
            for network_prefix, epoch_size in overrides.items():
                if self.network_name == network_prefix or self.network_name.startswith(
                    network_prefix
                ):
                    return int(epoch_size)
            return default_epoch_size

        def hidden_units_for_network(self, sample_size, input_sequence_size, output_sequence_size):
            strategy = getattr(utility_cls, "hidden_units_strategy", None)
            if strategy == "patterson":
                denominator = 10 * max(input_sequence_size + output_sequence_size, 1)
                return [max(1, int(round(sample_size / denominator)))]
            return utility_cls.hidden_units_sizes

        def init_hyperparameters(self, sample_size, input_sequence_size, output_sequence_size):
            self.input_sequence_size = input_sequence_size
            self.output_sequence_size = output_sequence_size

            if utility_cls.heuristic:
                self.learning_rate = 0.01
                self.epoch_size = 300
                self.batch_size = int(np.sqrt(sample_size))
                self.use_scheduler = False
                self.use_dropout = False
                self.dropout_rate = 0
                self.optimizer_name = "adam"
                self.momentum = 0.0
                self.activation = "gelu"
                self.use_layer_norm = True
                h1 = int((2 * input_sequence_size / 3) + output_sequence_size)
                h2 = int(sample_size / (8 * (input_sequence_size + output_sequence_size))) - h1
                self.hidden_layer_units_sizes = [h1, h2]
                return

            self.learning_rate = utility_cls.learning_rate
            self.epoch_size = self._epoch_size_for_network(utility_cls.epoch_size)
            self.batch_size = utility_cls.batch_size
            self.hidden_layer_units_sizes = self._hidden_units_for_network(
                sample_size,
                input_sequence_size,
                output_sequence_size,
            )
            self.use_scheduler = utility_cls.use_scheduler
            self.use_dropout = utility_cls.use_dropout
            self.dropout_rate = utility_cls.dropout_rate
            self.optimizer_name = getattr(utility_cls, "optimizer_name", "adam")
            self.momentum = getattr(utility_cls, "momentum", 0.0)
            self.activation = getattr(utility_cls, "activation", "gelu")
            self.use_layer_norm = getattr(utility_cls, "use_layer_norm", True)

        def init_model(self):
            layers = [nn.Linear(self.input_sequence_size, self.hidden_layer_units_sizes[0])]
            initializers.kaiming_normal_(layers[0].weight)
            for index in range(1, len(self.hidden_layer_units_sizes)):
                layer = nn.Linear(
                    self.hidden_layer_units_sizes[index - 1],
                    self.hidden_layer_units_sizes[index],
                )
                initializers.kaiming_normal_(layer.weight)
                layers.append(layer)
            layers.append(nn.Linear(self.hidden_layer_units_sizes[-1], self.output_sequence_size))
            initializers.kaiming_normal_(layers[-1].weight)

            self.norm = nn.LayerNorm(self.hidden_layer_units_sizes[-1])
            self.dropout = nn.Dropout(self.dropout_rate) if self.use_dropout else nn.Identity()
            self.layers = nn.ModuleList(layers)
            if str(self.optimizer_name).lower() == "sgd":
                self.optimizer = SGD(
                    self.parameters(),
                    lr=self.learning_rate,
                    momentum=float(self.momentum),
                )
            else:
                self.optimizer = Adam(self.parameters(), self.learning_rate, weight_decay=1e-5)

        def activate(self, x):
            if str(self.activation).lower() == "relu":
                return F.relu(x)
            return F.gelu(x)

        def forward(self, x):
            if str(self.activation).lower() == "gelu" and self.use_layer_norm:
                for layer in self.layers[:-2]:
                    x = F.gelu(layer(x))
                    x = self.dropout(x)
                x = F.gelu(self.norm(self.layers[-2](x)))
                return self.layers[-1](x)

            for layer in self.layers[:-1]:
                x = self._activate(layer(x))
                x = self.dropout(x)
            if self.use_layer_norm:
                x = self.norm(x)
            return self.layers[-1](x)

        utility_cls.set_hyperparameters = staticmethod(set_hyperparameters)
        basic_nn_cls._epoch_size_for_network = epoch_size_for_network
        basic_nn_cls._hidden_units_for_network = hidden_units_for_network
        basic_nn_cls.init_hyperparameters = init_hyperparameters
        basic_nn_cls.init_model = init_model
        basic_nn_cls._activate = activate
        basic_nn_cls.forward = forward
        setattr(basic_nn_cls, "_dataops_paper_training_patch_applied", True)

    @staticmethod
    def _patch_data_preprocessor_paper_mode(data_preprocessor_cls) -> None:
        if getattr(data_preprocessor_cls, "_dataops_paper_mode_patch_applied", False):
            return

        original_preprocess = data_preprocessor_cls.preprocess

        def preprocess(
            self,
            data,
            sampling_periods=None,
            stride=None,
            sampling_statistics=None,
            sequence_size=4,
            error_sequence_size=4,
            wavelet_type="haar",
            wavelet_level=None,
            scale_factor=None,
            normalization_type=None,
            target_normalization_type="window_mean",
            conjoincy=False,
            test_ratio=0.2,
            paper_multi_timeframe=False,
            drop_first_wavelet_coefficient=True,
            *,
            profile="default",
            fit=False,
        ):
            setattr(
                self,
                "_DataPreprocessor__drop_first_wavelet_coefficient",
                drop_first_wavelet_coefficient,
            )
            if paper_multi_timeframe:
                return PecnetRuntime._paper_multi_timeframe_preprocess(
                    self,
                    data=data,
                    sampling_periods=sampling_periods,
                    stride=stride,
                    sampling_statistics=sampling_statistics,
                    sequence_size=sequence_size,
                    error_sequence_size=error_sequence_size,
                    wavelet_type=wavelet_type,
                    wavelet_level=wavelet_level,
                    scale_factor=scale_factor,
                    normalization_type=normalization_type,
                    target_normalization_type=target_normalization_type,
                    conjoincy=conjoincy,
                    test_ratio=test_ratio,
                    profile=profile,
                    fit=fit,
                )
            return original_preprocess(
                self,
                data=data,
                sampling_periods=sampling_periods,
                stride=stride,
                sampling_statistics=sampling_statistics,
                sequence_size=sequence_size,
                error_sequence_size=error_sequence_size,
                wavelet_type=wavelet_type,
                scale_factor=scale_factor,
                normalization_type=normalization_type,
                target_normalization_type=target_normalization_type,
                conjoincy=conjoincy,
                test_ratio=test_ratio,
                profile=profile,
                fit=fit,
            )

        def calculate_dwt(self, array, wavelet_type, level=None):
            import numpy as np  # noqa: PLC0415
            from pywt import wavedec  # noqa: PLC0415

            level = (
                level
                if level is not None
                else getattr(self, "_DataPreprocessor__wavelet_level", None)
            )
            coeffs = wavedec(array, wavelet_type, mode="zero", level=level)
            coefficients = np.concatenate(coeffs)
            if getattr(self, "_DataPreprocessor__drop_first_wavelet_coefficient", True):
                return coefficients[1:]
            return coefficients

        data_preprocessor_cls.preprocess = preprocess
        data_preprocessor_cls._calculate_dwt = calculate_dwt
        setattr(data_preprocessor_cls, "_dataops_paper_mode_patch_applied", True)

    @staticmethod
    def _paper_multi_timeframe_preprocess(
        data_preprocessor,
        *,
        data,
        sampling_periods,
        stride,
        sampling_statistics,
        sequence_size,
        error_sequence_size,
        wavelet_type,
        wavelet_level,
        scale_factor,
        normalization_type,
        target_normalization_type,
        conjoincy,
        test_ratio,
        profile,
        fit,
    ):
        import numpy as np  # noqa: PLC0415
        from copy import deepcopy  # noqa: PLC0415
        from pecnet.preprocessing.Normalizers import Normalizer, Scaler  # noqa: PLC0415

        if target_normalization_type != "window_mean":
            raise ValueError("paper_multi_timeframe requires target_normalization_type='window_mean'.")

        sampling_periods = sampling_periods or [1, 5]
        sampling_statistics = sampling_statistics or ["mean"]
        stride_val = stride if stride and stride > 1 else 1
        biggest_period = max(sampling_periods)
        required_timestamps = (
            biggest_period + sequence_size - 1
            if conjoincy
            else biggest_period * sequence_size
        )
        if len(data) < required_timestamps:
            raise ValueError(
                "Not enough data for processing. At least "
                f"{required_timestamps} timestamps are required."
            )
        if not (0 < test_ratio < 1):
            raise ValueError("Test ratio must be between 0 and 1.")

        setattr(data_preprocessor, "_DataPreprocessor__error_sequence_size", error_sequence_size)
        setattr(data_preprocessor, "_DataPreprocessor__sequence_size", sequence_size)
        setattr(data_preprocessor, "_DataPreprocessor__wavelet_type", wavelet_type)
        setattr(data_preprocessor, "_DataPreprocessor__wavelet_level", wavelet_level)
        setattr(data_preprocessor, "_DataPreprocessor__required_timestamps", required_timestamps)

        if data_preprocessor.target is None:
            data_trimmed = len(data) - required_timestamps + stride_val
            test_size_index = int(test_ratio * (data_trimmed / stride_val))
        else:
            test_size_index = int(test_ratio * len(data[: len(data_preprocessor.target)]))
        setattr(data_preprocessor, "_DataPreprocessor__test_size_index", test_size_index)

        split_idx = test_size_index * stride_val
        train_part = data[:-split_idx]
        test_part = data[-split_idx:]

        if scale_factor is not None:
            data_preprocessor.scaler = Scaler()
            train_part = data_preprocessor.scaler.fit_scale1D(train_part, scale_factor)
            test_part = data_preprocessor.scaler.scale1D(test_part)
            if data_preprocessor.target_scaler is None:
                data_preprocessor.target_scaler = deepcopy(data_preprocessor.scaler)

        if normalization_type is None:
            data_preprocessor.normalizer = None
        else:
            data_preprocessor.normalizer = Normalizer(normalization_type)
            if data_preprocessor.target_normalizer is None:
                data_preprocessor.target_normalizer = deepcopy(data_preprocessor.normalizer)
            train_part = data_preprocessor.normalizer.fit_transform(train_part.reshape(-1, 1))
            test_part = data_preprocessor.normalizer.transform(test_part.reshape(-1, 1))

        data_preprocessor._profiles[profile] = {
            "scaler": deepcopy(data_preprocessor.scaler) if scale_factor is not None else None,
            "normalizer": deepcopy(data_preprocessor.normalizer)
            if normalization_type is not None
            else None,
            "scale_factor": scale_factor,
            "normalization_type": normalization_type,
            "wavelet_type": wavelet_type,
            "wavelet_level": wavelet_level,
            "sequence_size": sequence_size,
            "error_sequence_size": error_sequence_size,
            "stride": stride_val,
            "required_timestamps": required_timestamps,
            "conjoincy": conjoincy,
            "target_normalization_type": target_normalization_type,
            "paper_multi_timeframe": True,
            "drop_first_wavelet_coefficient": False,
        }

        full_data = np.concatenate([train_part, test_part])
        windows = data_preprocessor.build_windows(full_data, window_length=required_timestamps)
        sequences = []
        denormalization_terms = []
        for window in windows:
            input_values = []
            for period in sorted(sampling_periods):
                groups = data_preprocessor._build_sampling_groups(window, period)
                for statistic in sampling_statistics:
                    input_values.extend(
                        [
                            data_preprocessor._calculate_statistics(group, statistic)
                            for group in groups
                        ][-sequence_size:]
                    )

            input_values = np.asarray(input_values, dtype=np.float32)
            window_mean = np.nanmean(input_values)
            wavelet_coeffs = data_preprocessor._calculate_dwt(
                input_values - window_mean,
                wavelet_type,
                level=wavelet_level,
            )
            sequences.append([[wavelet_coeffs]])
            denormalization_terms.append(window_mean)

        y = (
            np.asarray(full_data[required_timestamps:], dtype=np.float32)
            - np.asarray(denormalization_terms[:-1], dtype=np.float32)
        )
        final_y = np.append(y, 0).reshape(-1, 1)
        denorm = np.asarray(denormalization_terms, dtype=np.float32).reshape(-1, 1)

        setattr(data_preprocessor, "_DataPreprocessor__final_y_processed", final_y)
        setattr(data_preprocessor, "_DataPreprocessor__y_denormalization_term", denorm)
        if data_preprocessor.target is None:
            data_preprocessor.target = final_y.copy()
        if data_preprocessor.target_denormalization_term is None:
            data_preprocessor.target_denormalization_term = denorm.copy()

        X = np.asarray(sequences[: len(data_preprocessor.target)], dtype=np.float32)
        X_train, X_test = X[:-test_size_index], X[-test_size_index:]
        y_train, y_test = final_y[:-test_size_index], final_y[-test_size_index:]
        return X_train, X_test, y_train, y_test

    @staticmethod
    def _resolve_torch_device_name(torch_module) -> str:
        requested = PecnetRuntime._requested_torch_device()
        if requested in {"mps", "metal"}:
            if PecnetRuntime._torch_mps_available(torch_module):
                return "mps"
            LOGGER.warning("PECNET_TORCH_DEVICE=mps requested but MPS is unavailable.")
            return "cpu"
        if requested == "cuda":
            if torch_module.cuda.is_available():
                return "cuda"
            LOGGER.warning("PECNET_TORCH_DEVICE=cuda requested but CUDA is unavailable.")
            return "cpu"
        if requested == "cpu":
            return "cpu"
        if PecnetRuntime._torch_mps_available(torch_module):
            return "mps"
        if torch_module.cuda.is_available():
            return "cuda"
        return "cpu"

    @staticmethod
    def _requested_torch_device() -> str:
        return os.getenv("PECNET_TORCH_DEVICE", "auto").strip().lower() or "auto"

    @staticmethod
    def _torch_mps_available(torch_module) -> bool:
        mps_backend = getattr(getattr(torch_module, "backends", None), "mps", None)
        return bool(mps_backend and mps_backend.is_available())

    @staticmethod
    def _ticker_test_ratio(row_count: int, test_horizon: int) -> float:
        if row_count <= test_horizon:
            raise ValueError(
                f"PECNet needs more rows than test_horizon. rows={row_count}, "
                f"test_horizon={test_horizon}"
            )
        return min(max(test_horizon / row_count, 0.01), 0.5)
