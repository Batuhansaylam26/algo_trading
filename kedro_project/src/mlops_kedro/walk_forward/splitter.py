"""
Decoupled Index-Based Walk-Forward Splitter
===========================================
Provides integer index-based partition boundaries for walk-forward
out-of-sample testing blocks. Operates strictly on sequence index ranges
without any time-unit, column name, or storage dependencies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class WalkForwardIndexBlock:
    """Represents pure integer index slice boundaries for a walk-forward block."""

    block_id: int
    padded_train_slice: slice
    train_slice: slice
    padded_test_slice: slice
    test_slice: slice
    padding_length: int


class WalkForwardSplitter:
    """Partitions a sequence index range into continuous out-of-sample blocks."""

    def __init__(
        self,
        training_prediction_count: int,
        testing_prediction_count: int,
    ) -> None:
        if training_prediction_count <= 0:
            raise ValueError(f"training_prediction_count must be positive, got {training_prediction_count}")
        if testing_prediction_count <= 0:
            raise ValueError(f"testing_prediction_count must be positive, got {testing_prediction_count}")

        self.training_prediction_count = training_prediction_count
        self.testing_prediction_count = testing_prediction_count

    def split_indices(
        self,
        total_sequence_length: int,
        padding_length: int,
    ) -> list[WalkForwardIndexBlock]:
        """Partitions sequence indices [0..total_sequence_length] into exact index block ranges."""
        if total_sequence_length <= 0:
            raise ValueError(f"total_sequence_length must be positive, got {total_sequence_length}")
        if padding_length <= 0:
            raise ValueError(f"padding_length must be positive, got {padding_length}")

        min_required_length = padding_length + self.training_prediction_count + self.testing_prediction_count
        if total_sequence_length < min_required_length:
            raise ValueError(
                f"Insufficient sequence length ({total_sequence_length}) for padding ({padding_length}) "
                f"+ training prediction count ({self.training_prediction_count}) "
                f"+ testing prediction count ({self.testing_prediction_count}). "
                f"Minimum required is {min_required_length} bars."
            )

        blocks: list[WalkForwardIndexBlock] = []
        start_test_idx = padding_length + self.training_prediction_count
        block_id = 0

        # Enforce exact block sizes without fallback truncation
        while start_test_idx + self.testing_prediction_count <= total_sequence_length:
            end_test_idx = start_test_idx + self.testing_prediction_count
            test_slice = slice(start_test_idx, end_test_idx)

            train_start_idx = start_test_idx - self.training_prediction_count
            train_slice = slice(train_start_idx, start_test_idx)

            padded_train_start_idx = train_start_idx - padding_length
            padded_train_slice = slice(padded_train_start_idx, start_test_idx)

            padded_test_start_idx = start_test_idx - padding_length
            padded_test_slice = slice(padded_test_start_idx, end_test_idx)

            blocks.append(
                WalkForwardIndexBlock(
                    block_id=block_id,
                    padded_train_slice=padded_train_slice,
                    train_slice=train_slice,
                    padded_test_slice=padded_test_slice,
                    test_slice=test_slice,
                    padding_length=padding_length,
                )
            )

            start_test_idx = end_test_idx
            block_id += 1

        if not blocks:
            raise ValueError(
                f"Unable to form any complete walk-forward block with exact testing_prediction_count ({self.testing_prediction_count}). "
                f"Total sequence length ({total_sequence_length}) is insufficient for padding ({padding_length}) "
                f"+ training prediction count ({self.training_prediction_count}) "
                f"+ testing prediction count ({self.testing_prediction_count})."
            )

        LOGGER.info(
            "Created %d exact walk-forward index blocks | padding_length=%d train_count=%d test_count=%d",
            len(blocks),
            padding_length,
            self.training_prediction_count,
            self.testing_prediction_count,
        )
        return blocks
