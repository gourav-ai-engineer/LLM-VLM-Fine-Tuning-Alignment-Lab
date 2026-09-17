"""Small utilities for experiment accounting."""

from __future__ import annotations

from typing import Iterable


def parameter_counts(parameters: Iterable[object]) -> tuple[int, int]:
    """Return (total, trainable) counts for PyTorch-like parameters."""
    total = 0
    trainable = 0
    for parameter in parameters:
        count = int(parameter.numel())
        total += count
        if bool(parameter.requires_grad):
            trainable += count
    return total, trainable


def trainable_percentage(total: int, trainable: int) -> float:
    """Return trainable parameters as a percentage of total parameters."""
    if total <= 0:
        raise ValueError("total must be positive")
    if trainable < 0 or trainable > total:
        raise ValueError("trainable must be between zero and total")
    return 100.0 * trainable / total
