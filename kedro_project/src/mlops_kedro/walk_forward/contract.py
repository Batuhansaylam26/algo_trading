"""
Pecnet Bar Requirement Contract Engine
======================================
Defines strict, fast-failing contracts for calculating and validating exact
raw bar counts required by Pecnet models for training and testing windows.
"""

from __future__ import annotations

from typing import Any


class PecnetRequirementsContract:
    """Contract enforcing exact bar count requirements for Pecnet models."""

    def __init__(
        self,
        sampling_periods: list[int],
        sequence_size: int,
        error_sequence_size: int,
        conjoincy: bool,
    ) -> None:
        if not sampling_periods:
            raise ValueError("sampling_periods must be a non-empty list of integers.")
        if sequence_size <= 0:
            raise ValueError(f"sequence_size must be positive, got {sequence_size}")
        if error_sequence_size < 0:
            raise ValueError(f"error_sequence_size cannot be negative, got {error_sequence_size}")

        self.sampling_periods = sampling_periods
        self.sequence_size = sequence_size
        self.error_sequence_size = error_sequence_size
        self.conjoincy = conjoincy

        self.biggest_period = max(sampling_periods)

        if self.conjoincy:
            self.required_timestamps = self.biggest_period + self.sequence_size - 1
        else:
            self.required_timestamps = self.biggest_period * self.sequence_size

        self.padding_length = self.required_timestamps + self.error_sequence_size

    @classmethod
    def from_preprocess_params(cls, preprocess_params: dict[str, Any]) -> PecnetRequirementsContract:
        """Instantiates contract from preprocess_params dictionary.

        Fails immediately if any required parameter is missing.
        """
        required_keys = ["sampling_periods", "sequence_size", "error_sequence_size", "conjoincy"]
        for key in required_keys:
            if key not in preprocess_params:
                raise KeyError(f"Missing mandatory preprocess parameter '{key}' in Pecnet configuration.")

        return cls(
            sampling_periods=list(preprocess_params["sampling_periods"]),
            sequence_size=int(preprocess_params["sequence_size"]),
            error_sequence_size=int(preprocess_params["error_sequence_size"]),
            conjoincy=bool(preprocess_params["conjoincy"]),
        )

    def calculate_required_train_bars(self, requested_train_samples: int) -> int:
        """Calculates exact required raw bars for requested training prediction samples."""
        if requested_train_samples <= 0:
            raise ValueError(f"requested_train_samples must be positive, got {requested_train_samples}")
        return requested_train_samples + self.padding_length

    def calculate_required_test_bars(self, requested_test_samples: int) -> int:
        """Calculates exact required raw bars for requested testing prediction samples."""
        if requested_test_samples <= 0:
            raise ValueError(f"requested_test_samples must be positive, got {requested_test_samples}")
        return requested_test_samples + self.padding_length

    def validate_bar_counts(
        self,
        provided_train_bars: int,
        provided_test_bars: int,
        requested_train_samples: int,
        requested_test_samples: int,
    ) -> None:
        """Validates provided train and test bar counts against contract expectations.

        Fails fast with ValueError if bar counts do not match requirements down to the exact bar.
        """
        required_train_bars = self.calculate_required_train_bars(requested_train_samples)
        required_test_bars = self.calculate_required_test_bars(requested_test_samples)

        errors = []
        if provided_train_bars != required_train_bars:
            errors.append(
                f"Train bar mismatch: provided {provided_train_bars} bars, "
                f"but exact contract requirement is {required_train_bars} bars "
                f"({requested_train_samples} samples + {self.padding_length} padding)."
            )

        if provided_test_bars != required_test_bars:
            errors.append(
                f"Test bar mismatch: provided {provided_test_bars} bars, "
                f"but exact contract requirement is {required_test_bars} bars "
                f"({requested_test_samples} samples + {self.padding_length} padding)."
            )

        if errors:
            raise ValueError("PecnetRequirementsContract validation failed:\n" + "\n".join(errors))

    def compute_concatenated_test_ratio(self, train_bars: int, test_bars: int) -> float:
        """Computes exact test_ratio for DataPreprocessor single-pass ingestion."""
        total_bars = train_bars + test_bars
        if total_bars <= 0:
            raise ValueError("Total combined bars must be positive.")
        test_ratio = float(test_bars / total_bars)
        if not (0 < test_ratio < 1):
            raise ValueError(f"Computed test_ratio {test_ratio} is invalid. Must be strictly between 0 and 1.")
        return test_ratio
