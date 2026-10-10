"""Synthetic temporal datasets for reservoir sanity checks.

Everything here is seeded off a single ``numpy.random.Generator`` so a run is
reproducible from its seed alone.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

CLASS_NAMES = (
    "sine_burst",
    "square_burst",
    "chirp",
    "two_pulse",
    "delayed_pulse",
    "am_pulse",
)


@dataclass(frozen=True)
class Dataset:
    """A batch of normalised waveforms with integer class labels."""

    waveforms: np.ndarray  # (samples, time_samples), float64
    labels: np.ndarray  # (samples,), int64
    class_names: tuple[str, ...]

    @property
    def n_classes(self) -> int:
        return len(self.class_names)


def make_waveform_classification(
    samples_per_class: int = 64,
    time_samples: int = 64,
    seed: int = 1234,
    noise_std: float = 0.03,
) -> Dataset:
    """Six temporal classes with matched shape and amplitude.

    The classes share an amplitude range on purpose: the experiment should ask
    about temporal structure, not about an easy per-class scale cue.
    """
    if samples_per_class < 1:
        raise ValueError(f"samples_per_class must be >= 1, got {samples_per_class}")
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, time_samples)

    waveforms = []
    labels = []
    for class_id in range(len(CLASS_NAMES)):
        for _ in range(samples_per_class):
            wave = _make_class_waveform(class_id, t, rng)
            wave = wave + noise_std * rng.standard_normal(time_samples)
            waveforms.append(wave)
            labels.append(class_id)

    x = normalize(np.stack(waveforms))
    y = np.asarray(labels, dtype=np.int64)
    order = rng.permutation(len(y))
    return Dataset(waveforms=x[order], labels=y[order], class_names=CLASS_NAMES)


def make_temporal_xor(
    samples: int = 256,
    time_samples: int = 64,
    seed: int = 1234,
    noise_std: float = 0.02,
) -> Dataset:
    """Two pulses in separate time windows; the label is their XOR.

    Neither window alone carries the answer, so a readout has to mix events
    that arrive at different times.
    """
    if samples < 1:
        raise ValueError(f"samples must be >= 1, got {samples}")
    rng = np.random.default_rng(seed)
    index = np.arange(time_samples)

    waveforms = []
    labels = []
    for _ in range(samples):
        a = int(rng.integers(0, 2))
        b = int(rng.integers(0, 2))
        wave = np.zeros(time_samples)
        if a:
            wave += _gaussian_pulse(index, time_samples * 0.25, 2.5)
        if b:
            wave += _gaussian_pulse(index, time_samples * 0.75, 2.5)
        wave = wave + noise_std * rng.standard_normal(time_samples)
        waveforms.append(wave)
        labels.append(a ^ b)

    x = normalize(np.stack(waveforms))
    return Dataset(
        waveforms=x,
        labels=np.asarray(labels, dtype=np.int64),
        class_names=("false", "true"),
    )


def normalize(waveforms: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-waveform zero-mean, unit-variance normalisation."""
    centered = waveforms - waveforms.mean(axis=-1, keepdims=True)
    scale = centered.std(axis=-1, keepdims=True)
    return centered / np.maximum(scale, eps)


def stratified_split(
    waveforms: np.ndarray,
    labels: np.ndarray,
    test_fraction: float = 0.25,
    seed: int = 1234,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Shuffle within each class and hold out ``test_fraction`` of every class.

    A plain random split over a small class-balanced dataset can leave a class
    thinly represented in one side, which makes single-seed accuracies noisy
    for reasons that have nothing to do with the features.
    """
    if not 0.0 < test_fraction < 1.0:
        raise ValueError(f"test_fraction must be in (0, 1), got {test_fraction}")
    rng = np.random.default_rng(seed)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for class_id in np.unique(labels):
        idx = np.flatnonzero(labels == class_id)
        idx = idx[rng.permutation(len(idx))]
        n_test = max(1, round(test_fraction * len(idx)))
        n_test = min(n_test, len(idx) - 1)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train = np.asarray(sorted(train_idx))
    test = np.asarray(sorted(test_idx))
    train = train[rng.permutation(len(train))]
    test = test[rng.permutation(len(test))]
    return waveforms[train], labels[train], waveforms[test], labels[test]


DatasetFactory = Callable[[int], Dataset]


def _make_class_waveform(class_id: int, t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    phase = 0.05 * rng.standard_normal()
    amp = 0.8 + 0.4 * rng.random()
    freq = 4.0 + rng.random()
    gate = ((t > 0.15) & (t < 0.85)).astype(float)
    index = np.arange(t.size)

    if class_id == 0:
        return amp * np.sin(2.0 * np.pi * freq * (t + phase)) * gate
    if class_id == 1:
        return amp * np.sign(np.sin(2.0 * np.pi * freq * (t + phase))) * gate
    if class_id == 2:
        # Frequency sweeps within the burst, so the class is about time-varying
        # structure rather than a fixed carrier.
        chirp_phase = 2.0 * np.pi * (2.0 * t + 8.0 * t * t)
        return amp * np.sin(chirp_phase + phase) * gate
    if class_id == 3:
        return amp * (
            _gaussian_pulse(index, t.size * 0.30, 2.0) + _gaussian_pulse(index, t.size * 0.62, 2.5)
        )
    if class_id == 4:
        center = t.size * (0.55 + 0.15 * rng.random())
        return amp * _gaussian_pulse(index, center, 3.0)
    carrier = np.sin(2.0 * np.pi * 6.0 * t)
    envelope = 0.5 + 0.5 * np.sin(2.0 * np.pi * 1.5 * t + phase)
    return amp * carrier * envelope * gate


def _gaussian_pulse(index: np.ndarray, center: float, width: float) -> np.ndarray:
    return np.exp(-0.5 * ((index - center) / width) ** 2)
