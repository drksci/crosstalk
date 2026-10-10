"""Small, dependency-free metric helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def accuracy(predictions: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    if predictions.shape != labels.shape:
        raise ValueError(
            f"shape mismatch: predictions {predictions.shape} vs labels {labels.shape}"
        )
    return float(np.mean(predictions == labels))


def confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    n_classes: int,
) -> np.ndarray:
    """Counts with rows = true class and columns = predicted class."""
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    for name, values in (("labels", labels), ("predictions", predictions)):
        if values.size and (values.min() < 0 or values.max() >= n_classes):
            raise ValueError(
                f"{name} outside 0..{n_classes - 1}: min {values.min()}, max {values.max()}"
            )
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    return matrix


@dataclass(frozen=True)
class Spread:
    """Mean and sample standard deviation across replicates."""

    mean: float
    std: float
    n: int

    def format(self, digits: int = 3) -> str:
        if self.n > 1:
            return f"{self.mean:.{digits}f} ± {self.std:.{digits}f}"
        return f"{self.mean:.{digits}f}"


def spread(values: Sequence[float]) -> Spread:
    data = np.asarray(list(values), dtype=np.float64)
    if data.size == 0:
        raise ValueError("cannot summarise an empty sequence")
    std = float(data.std(ddof=1)) if data.size > 1 else 0.0
    return Spread(mean=float(data.mean()), std=std, n=int(data.size))
