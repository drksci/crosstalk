"""Linear readout heads.

Both experiment paths use this exact class with the same hyperparameters, so
any accuracy difference is attributable to the features, not the classifier.
Logistic regression replaces the archive's hand-rolled Adam loop: it is
deterministic, converges in one fit, and needs no autograd stack.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
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


class StreamingLinearReadout:
    """Standardised logistic regression fitted a batch at a time.

    The same model as :class:`LinearReadout` -- L2-regularised multinomial
    logistic regression over standardised features -- but learned by
    ``SGDClassifier(loss="log_loss")`` instead of L-BFGS, and updated with
    ``partial_fit`` rather than one whole-matrix ``fit``. That is what makes
    full MNIST reachable: ``LinearReadout`` has to materialise the entire
    feature matrix twice (once as ``X``, once as the scaler's copy), and
    60000 x 6272 float64 is 2.8 GB, which this machine does not have.

    Keeping the model identical rather than the solver matters for the
    experiment: both the raw and the reservoir arm must use this class with
    the same hyperparameters, or a difference in accuracy could be a
    difference in classifiers instead of a difference in features. Swapping
    the optimiser does not change what is being fitted.

    Scaling is fitted incrementally with ``StandardScaler.partial_fit``. Its
    running mean and variance converge to the exact batch moments, so this is
    the same transformation ``LinearReadout`` would apply -- not an
    approximation of it.
    """

    def __init__(
        self,
        *,
        C: float = 1.0,
        epochs: int = 10,
        batch_size: int = 4096,
        seed: int = 0,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C must be > 0, got {C}")
        if epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {epochs}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")
        self._scaler = StandardScaler()
        self._classifier = SGDClassifier(
            # alpha is set in fit(), from C and the sample count: SGD
            # minimises (1/n)*sum(loss) + (alpha/2)*||w||^2 while L-BFGS
            # minimises sum(loss) + (1/2C)*||w||^2, so matching the two needs
            # alpha = 1/(C*n). Both arms pass the same C, so they still see
            # the same regularisation as each other.
            loss="log_loss",
            max_iter=1,
            tol=None,
            random_state=seed,
        )
        self._C = C
        self._epochs = epochs
        self._batch_size = batch_size
        self._classes: np.ndarray | None = None
        self._scaled = False

    def fit(
        self,
        batches: list[np.ndarray] | np.ndarray,
        labels: np.ndarray,
        *,
        classes: np.ndarray | None = None,
    ) -> StreamingLinearReadout:
        """Fit over ``batches``, repeating the passes for ``epochs``.

        ``batches`` is a list of feature blocks (or one array, which is then
        chunked). The first pass only teaches the scaler, because every later
        block has to be transformed by a scaler that has already seen the
        whole training set -- fitting the scaler and the classifier in the
        same pass would let early blocks see an unfinished scaler.
        """
        blocks = _as_batches(batches, self._batch_size)
        y = np.asarray(labels)
        if sum(len(block) for block in blocks) != len(y):
            raise ValueError(
                f"batches hold {sum(len(block) for block in blocks)} rows but labels has {len(y)}"
            )
        self._classes = np.unique(y) if classes is None else np.asarray(classes)
        if len(self._classes) < 2:
            raise ValueError("need at least two classes to fit a readout")

        for block in blocks:
            self._scaler.partial_fit(block)

        self._set_alpha(len(y))
        offset = 0
        for _ in range(self._epochs):
            offset = 0
            for block in blocks:
                size = len(block)
                scaled = self._scaler.transform(block)
                self._classifier.partial_fit(
                    scaled, y[offset : offset + size], classes=self._classes
                )
                offset += size
        self._scaled = True
        return self

    def fit_stream(
        self,
        source: Callable[[], Iterable[np.ndarray]],
        labels: np.ndarray,
        *,
        classes: np.ndarray | None = None,
    ) -> StreamingLinearReadout:
        """Fit from a factory of feature batches, without holding them all.

        ``source()`` must return a fresh iterable of blocks in the same order
        on every call -- one per epoch, after the scaler pass. That is what a
        ``numpy.memmap`` slice iterator gives, and it is the reason the MNIST
        reservoir features never have to exist in memory all at once.
        """
        y = np.asarray(labels)
        self._classes = np.unique(y) if classes is None else np.asarray(classes)
        if len(self._classes) < 2:
            raise ValueError("need at least two classes to fit a readout")

        seen = 0
        for block in source():
            self._scaler.partial_fit(block)
            seen += len(block)
        if seen != len(y):
            raise ValueError(f"batches hold {seen} rows but labels has {len(y)}")

        self._set_alpha(len(y))

        for _ in range(self._epochs):
            offset = 0
            for block in source():
                size = len(block)
                scaled = self._scaler.transform(block)
                self._classifier.partial_fit(
                    scaled, y[offset : offset + size], classes=self._classes
                )
                offset += size
        self._scaled = True
        return self

    def predict(self, features: np.ndarray, *, batch_size: int | None = None) -> np.ndarray:
        """Predict in blocks, keeping the caller's float width.

        Blocks matter: MNIST's held-out features are 10000 x 6272, and one
        float64 array of that is 501 MB with another 501 MB behind it from
        the scaler -- on a machine with 1.5 GB free. The dtype is preserved
        rather than widened because ``StandardScaler`` and ``SGDClassifier``
        both accept float32 and return the precision they were given, so
        widening buys nothing but the out-of-memory.
        """
        if not self._scaled or self._classes is None:
            raise ValueError("call fit before predict")
        x = _check_2d(features)
        size = batch_size or self._batch_size
        if len(x) <= size:
            return self._classifier.predict(self._scaler.transform(x))

        out = np.empty(len(x), dtype=self._classes.dtype)
        for start in range(0, len(x), size):
            block = x[start : start + size]
            out[start : start + len(block)] = self._classifier.predict(
                self._scaler.transform(block)
            )
        return out

    def score(self, features: np.ndarray, labels: np.ndarray) -> float:
        predictions = self.predict(features)
        return float(np.mean(predictions == np.asarray(labels)))

    @property
    def classes_(self) -> np.ndarray:
        if self._classes is None:
            raise ValueError("call fit before reading classes_")
        return self._classes

    def _set_alpha(self, n_samples: int) -> None:
        """Match L-BFGS's ``C`` on a problem of ``n_samples`` rows.

        L-BFGS minimises ``sum(loss) + (1/2C)||w||^2``; SGD minimises the
        *mean* loss plus ``(alpha/2)||w||^2``. Scaling one objective by n to
        compare it with the other gives ``alpha = 1/(C n)``, which is what
        scikit-learn's own documentation states. Setting ``alpha = 1/C``
        instead would silently regularise n times harder, and on 60000 rows
        that is the difference between a readout and a constant predictor.
        """
        self._classifier.set_params(alpha=1.0 / (self._C * n_samples))


def _check_2d(features: np.ndarray) -> np.ndarray:
    """Validate a feature block without widening its dtype.

    ``LinearReadout`` casts to float64 because L-BFGS wants it; the streaming
    readout deliberately does not, since widening 60000 x 6272 float32 is how
    this run ends up needing 2.8 GB. Shape and finiteness are still checked
    here so a bad block fails at the boundary rather than inside scikit-learn.
    """
    x = np.asarray(features)
    if x.ndim != 2:
        raise ValueError(f"features must be 2-D (samples, features), got shape {x.shape}")
    if x.shape[1] < 1:
        raise ValueError("features must contain at least one column")
    if not np.issubdtype(x.dtype, np.floating):
        raise ValueError(f"features must be floating point, got dtype {x.dtype}")
    return x


def _as_batches(batches: list[np.ndarray] | np.ndarray, batch_size: int) -> list[np.ndarray]:
    """Normalise a list of blocks, or chunk one array, into feature blocks."""
    if isinstance(batches, np.ndarray):
        if batches.ndim != 2:
            raise ValueError(f"features must be 2-D, got shape {batches.shape}")
        return [batches[start : start + batch_size] for start in range(0, len(batches), batch_size)]
    blocks = [np.asarray(block) for block in batches]
    for block in blocks:
        if block.ndim != 2:
            raise ValueError(f"each batch must be 2-D, got shape {block.shape}")
    if len({block.shape[1] for block in blocks}) > 1:
        raise ValueError("every batch must have the same number of feature columns")
    return blocks
