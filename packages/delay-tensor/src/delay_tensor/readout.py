"""Linear readout heads.

Both experiment paths use this exact class with the same hyperparameters, so
any accuracy difference is attributable to the features, not the classifier.
Logistic regression replaces the archive's hand-rolled Adam loop: it is
deterministic, converges in one fit, and needs no autograd stack.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LinearReadout:
    """Standardised multinomial logistic regression over flattened features."""

    def __init__(self, *, C: float = 1.0, max_iter: int = 2000, seed: int = 0) -> None:
        if C <= 0:
            raise ValueError(f"C must be > 0, got {C}")
        self._pipeline = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=C,
                        max_iter=max_iter,
                        solver="lbfgs",
                        random_state=seed,
                    ),
                ),
            ]
        )

    def fit(self, features: np.ndarray, labels: np.ndarray) -> LinearReadout:
        x = self._check_features(features)
        y = np.asarray(labels)
        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"features and labels disagree on sample count: {x.shape[0]} vs {y.shape[0]}"
            )
        if len(np.unique(y)) < 2:
            raise ValueError("need at least two classes to fit a readout")
        self._pipeline.fit(x, y)
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self._pipeline.predict(self._check_features(features))

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        predictions = self.predict(features)
        return float(np.mean(predictions == np.asarray(labels)))

    @property
    def classes_(self) -> np.ndarray:
        return self._pipeline.named_steps["clf"].classes_

    @staticmethod
    def _check_features(features: np.ndarray) -> np.ndarray:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"features must be 2-D (samples, features), got shape {x.shape}")
        if x.shape[1] < 1:
            raise ValueError("features must contain at least one column")
        return x
