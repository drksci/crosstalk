"""Full MNIST, three ways: raw pixels, tensor features, and a small MLP.

The two linear arms are the experiment and the MLP is the yardstick:

    raw pixels    : 784 numbers  -> StreamingLinearReadout -> accuracy
    tensor        : 8 x 784      -> StreamingLinearReadout -> accuracy
    MLP           : 784 numbers  -> MLPClassifier           -> accuracy

Both linear arms use the same class, the same ``C``, the same epoch budget and
the same batch boundaries, so whatever separates them is the features and
nothing else. The MLP is deliberately *not* matched -- it reads raw pixels, is
nonlinear, and is the standard baseline. Its number is what turns a difference
between the two linear arms into a claim somebody could argue with, because
without it "0.91" and "0.92" are just two numbers.

The canonical 60000/10 split is used as published rather than reshuffled with
``stratified_split``: a benchmark's value is that it is the same one everyone
else reports on.
"""

from __future__ import annotations

import gc
import tempfile
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier

from delay_tensor.config import DelayTensorConfig
from delay_tensor.experiments.base import (
    DEFAULT_SEEDS,
    SeedRecord,
    Summary,
    TraceExample,
)
from delay_tensor.metrics import confusion_matrix
from delay_tensor.mnist import load_mnist, raster_scan
from delay_tensor.readout import StreamingLinearReadout
from delay_tensor.reservoir import DelayTensor

DEFAULT_EPOCHS = 10
DEFAULT_BATCH_SIZE = 4096
DEFAULT_RESERVOIR_BATCH = 2048
DEFAULT_HIDDEN_LAYER_SIZES = (128,)
DEFAULT_MLP_MAX_ITER = 30


def _choose_examples(labels: np.ndarray, max_examples: int) -> list[int]:
    """Take a few test rows spread across the label range, not just the head.

    The first rows of MNIST's test set are not a cross-section of it, and an
    illustration of three "9"s teaches a reader less than one of each digit
    would.
    """
    classes = np.unique(labels)
    step = max(1, len(classes) // max(1, min(max_examples, len(classes))))
    chosen = sorted(set(classes[::step]))[:max_examples]
    return sorted(int(np.flatnonzero(labels == class_id)[0]) for class_id in chosen)


def _batched(source: np.ndarray, batch: int) -> Iterator[np.ndarray]:
    """Yield ``source`` in ``batch``-sized views, starting at row 0.

    Starting at 0 in `batch` steps is exactly what ``_as_batches`` does for a
    dense array, so the tensor arm and the raw arm see identical batch
    boundaries -- and therefore identical alpha = 1/(C n) and identical
    optimiser trajectories.
    """
    for start in range(0, len(source), batch):
        yield np.asarray(source[start : start + batch])


def _feature_factory(source: np.ndarray, batch: int) -> Callable[[], Iterator[np.ndarray]]:
    """Build the factory ``fit_stream`` wants: a fresh block iterator per call.

    It is a closure on purpose -- ``fit_stream`` calls it once for the scaler
    pass and once per epoch, and a single generator would be exhausted on the
    first of those. Keeping it in its own function also keeps the memmap
    reference in its own frame: a closure written inline in ``run_mnist``'s
    loop would hold 1.5 GB alive right past the line that tries to free it
    before the MLP runs.
    """

    def factory() -> Iterator[np.ndarray]:
        return _batched(source, batch)

    return factory


def _encode(
    reservoir: DelayTensor,
    waveforms: np.ndarray,
    target: np.ndarray,
    *,
    batch: int,
    rng: np.random.Generator,
    capture: set[int],
) -> tuple[dict[str, float | str], dict[int, np.ndarray]]:
    """Stream the reservoir over ``waveforms``, writing features into ``target``.

    Only the rows in ``capture`` keep their traces. A whole test set's traces
    would be 10000 x 8 x 784 float64 -- 500 MB -- and the report shows three
    examples, so holding all of them would spend half the machine's memory on
    rows nobody reads.

    Features land in ``target`` as float32 as they are produced rather than
    being accumulated first: the float64 matrix would be 1.5 GB on its own,
    and this run has to leave room for the readout and the MLP on an 8 GB
    machine.

    Health is aggregated over the batches rather than reported from the first
    one, because a tensor that saturates on the tail of the data should say so.
    """
    state_total = 0.0
    saturation_total = 0.0
    cells = 0
    kept: dict[int, np.ndarray] = {}

    for start in range(0, len(waveforms), batch):
        stop = min(start + batch, len(waveforms))
        result = reservoir.run(waveforms[start:stop], rng)
        target[start:stop] = result.features

        rows = stop - start
        batch_cells = rows * result.traces.shape[1] * result.traces.shape[2]
        state_total += float(result.health["mean_abs_state"]) * batch_cells
        saturation_total += float(result.health["saturation_fraction"]) * batch_cells
        cells += batch_cells

        if capture:
            for offset, row in enumerate(range(start, stop)):
                if row in capture:
                    kept[row] = result.traces[offset]

    target.flush()
    saturation = saturation_total / cells
    health: dict[str, float | str] = {
        "status": "saturated" if saturation > 0.25 else "healthy",
        "mean_abs_state": state_total / cells,
        "saturation_fraction": saturation,
        "noise_std": reservoir.config.loss.noise_std,
        "decay": reservoir.config.loss.decay,
    }
    return health, kept


def run_mnist(
    config: DelayTensorConfig,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    epochs: int = DEFAULT_EPOCHS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    reservoir_batch: int = DEFAULT_RESERVOIR_BATCH,
    hidden_layer_sizes: tuple[int, ...] = DEFAULT_HIDDEN_LAYER_SIZES,
    mlp_max_iter: int = DEFAULT_MLP_MAX_ITER,
    max_examples: int = 3,
) -> Summary:
    """Compare raw pixels, tensor features and an MLP on all of MNIST."""
    if not seeds:
        raise ValueError("at least one seed is required")

    config.validate()
    data = load_mnist()
    class_names = data.class_names
    n_classes = len(class_names)

    # Raster order *is* the slot order: pixel (r, c) drives slot r*28 + c, so
    # the tensor reads each digit the way a scanning beam would and the slot
    # index still means "where on the screen".
    x_train = raster_scan(data.train_images).astype(np.float32)
    x_test = raster_scan(data.test_images).astype(np.float32)
    y_train, y_test = data.train_labels, data.test_labels
    del data

    example_rows = _choose_examples(y_test, max_examples)
    wanted = set(example_rows)
    reservoir = DelayTensor(config)

    confusion_raw = np.zeros((n_classes, n_classes), dtype=np.int64)
    confusion_reservoir = np.zeros((n_classes, n_classes), dtype=np.int64)
    confusion_mlp = np.zeros((n_classes, n_classes), dtype=np.int64)
    records: list[SeedRecord] = []
    examples: list[TraceExample] = []
    health: dict[str, float | str] = {}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        train_path = tmp_path / "train-features.f32"
        test_path = tmp_path / "test-features.f32"

        for seed in seeds:
            # Opened fresh each seed so the 1.5 GB mapping can be dropped
            # before the MLP runs -- adam wants headroom, and those pages are
            # no longer needed by then.
            train_features = np.memmap(
                train_path,
                dtype=np.float32,
                mode="w+",
                shape=(len(x_train), config.feature_count),
            )
            test_features = np.memmap(
                test_path,
                dtype=np.float32,
                mode="w+",
                shape=(len(x_test), config.feature_count),
            )

            # Train first, then test, with one rng carried across both: the
            # order run_replicates uses, so the two agree on what a seed means.
            #
            # noise_std in the config is what makes these features depend on
            # the seed at all. The mask is frozen, so without measurement
            # noise every replicate would see an identical tensor and the
            # spread would be measuring only shuffling -- which is why it
            # stays on.
            rng = np.random.default_rng([seed, 1])
            _encode(
                reservoir,
                x_train,
                train_features,
                batch=reservoir_batch,
                rng=rng,
                capture=set(),
            )
            seed_health, captured = _encode(
                reservoir,
                x_test,
                test_features,
                batch=reservoir_batch,
                rng=rng,
                capture=wanted if not records else set(),
            )

            # Same class, same C, same epochs, same batch boundaries on both
            # linear arms -- see _batched.
            raw_model = StreamingLinearReadout(
                C=1.0, epochs=epochs, batch_size=batch_size, seed=seed
            ).fit(x_train, y_train)
            raw_pred = raw_model.predict(x_test)

            # fit_stream calls the factory once for the scaler pass and once
            # per epoch, so it needs a fresh iterator each call.
            tensor_source = _feature_factory(train_features, batch_size)
            tensor_model = StreamingLinearReadout(
                C=1.0, epochs=epochs, batch_size=batch_size, seed=seed
            ).fit_stream(tensor_source, y_train)
            # predict() blocks internally and keeps float32, so the memmap can
            # be handed over as-is without a copy.
            tensor_pred = tensor_model.predict(test_features)

            # Free the feature mapping before the MLP: it is file-backed, so
            # the OS could evict it anyway, but evicting 1.5 GB under load is
            # what turns a 20-minute run into a swapping one. tensor_source
            # goes in the same statement because the factory's frame is what
            # holds the last reference to the memmap.
            del tensor_source, train_features, test_features, raw_model, tensor_model
            gc.collect()

            # The MLP reads raw pixels, not tensor features. It answers "what
            # does a conventional model reach on this split", and reading the
            # same input as the raw arm keeps that answer about the model
            # rather than about a second, different feature transform.
            mlp = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                solver="adam",
                early_stopping=True,
                max_iter=mlp_max_iter,
                random_state=seed,
            )
            mlp.fit(x_train, y_train)
            mlp_pred = mlp.predict(x_test)

            if not records:
                # Examples are taken from the first seed only, so a single
                # reservoir state -- and its traces -- matches the predictions
                # shown next to it.
                health = seed_health
                examples = [
                    TraceExample(
                        true_label=int(y_test[row]),
                        predicted_raw=int(raw_pred[row]),
                        predicted_reservoir=int(tensor_pred[row]),
                        predicted_mlp=int(mlp_pred[row]),
                        waveform=x_test[row],
                        traces=captured[row],
                    )
                    for row in example_rows
                ]

            records.append(
                SeedRecord(
                    seed=seed,
                    raw_accuracy=float(np.mean(raw_pred == y_test)),
                    reservoir_accuracy=float(np.mean(tensor_pred == y_test)),
                    mlp_accuracy=float(np.mean(mlp_pred == y_test)),
                    train_size=len(y_train),
                    test_size=len(y_test),
                )
            )
            confusion_raw += confusion_matrix(raw_pred, y_test, n_classes)
            confusion_reservoir += confusion_matrix(tensor_pred, y_test, n_classes)
            confusion_mlp += confusion_matrix(mlp_pred, y_test, n_classes)
            del mlp, raw_pred, tensor_pred, mlp_pred, captured
            gc.collect()

    return Summary(
        experiment="mnist",
        config=config,
        class_names=class_names,
        records=records,
        confusion_raw=confusion_raw,
        confusion_reservoir=confusion_reservoir,
        confusion_mlp=confusion_mlp,
        examples=examples,
        health=health,
        parameters={
            "epochs": epochs,
            "batch_size": batch_size,
            "reservoir_batch": reservoir_batch,
            "mlp_max_iter": mlp_max_iter,
            # The whole structure, not just its first layer: a caller may pass
            # (128, 64), and a metrics file that reports "128" for that is a
            # metrics file that lies about what was run.
            "mlp_hidden_layers": ",".join(str(units) for units in hidden_layer_sizes),
            "train_size": len(y_train),
            "test_size": len(y_test),
        },
    )
