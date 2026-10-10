"""Shared machinery for replicated experiments.

An experiment is a seeded dataset factory plus the same two-path comparison:

    raw path     : features  -> LinearReadout -> accuracy
    reservoir    : features' -> LinearReadout -> accuracy

Replicating over seeds and reporting a spread is the point: a single seed on a
576-sample synthetic dataset moves by several points, which is exactly how a
false result gets written down.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from delay_tensor.config import DelayTensorConfig
from delay_tensor.datasets import DatasetFactory, stratified_split
from delay_tensor.metrics import Spread, confusion_matrix, spread
from delay_tensor.readout import LinearReadout
from delay_tensor.reservoir import DelayTensor

DEFAULT_SEEDS = (1234, 1235, 1236)
DEFAULT_TEST_FRACTION = 0.25


@dataclass(frozen=True)
class SeedRecord:
    """One replicate of the two-path comparison."""

    seed: int
    raw_accuracy: float
    reservoir_accuracy: float
    train_size: int
    test_size: int

    @property
    def delta(self) -> float:
        return self.reservoir_accuracy - self.raw_accuracy


@dataclass(frozen=True)
class TraceExample:
    """A held-out waveform with the reservoir traces it produced."""

    true_label: int
    predicted_raw: int
    predicted_reservoir: int
    waveform: np.ndarray
    traces: np.ndarray  # (wires, slots)


@dataclass
class Summary:
    """Everything the report and the metrics file need."""

    experiment: str
    config: DelayTensorConfig
    class_names: tuple[str, ...]
    records: list[SeedRecord]
    confusion_raw: np.ndarray
    confusion_reservoir: np.ndarray
    examples: list[TraceExample]
    health: dict[str, float | str]
    parameters: dict[str, int | float] = field(default_factory=dict)

    @property
    def raw(self) -> Spread:
        return spread([r.raw_accuracy for r in self.records])

    @property
    def reservoir(self) -> Spread:
        return spread([r.reservoir_accuracy for r in self.records])

    @property
    def delta(self) -> Spread:
        return spread([r.delta for r in self.records])

    @property
    def chance_level(self) -> float:
        return 1.0 / len(self.class_names)

    @property
    def feature_count(self) -> int:
        return self.config.feature_count

    def to_metrics_dict(self) -> dict:
        """JSON-ready metrics: aggregate first, then one row per seed."""
        return {
            "experiment": self.experiment,
            "config_name": self.config.name,
            "class_names": list(self.class_names),
            "feature_count": self.feature_count,
            "chance_level": self.chance_level,
            "seeds": [r.seed for r in self.records],
            "parameters": self.parameters,
            "aggregate": {
                "raw_mean": self.raw.mean,
                "raw_std": self.raw.std,
                "reservoir_mean": self.reservoir.mean,
                "reservoir_std": self.reservoir.std,
                "delta_mean": self.delta.mean,
                "delta_std": self.delta.std,
                "n_seeds": self.raw.n,
            },
            "per_seed": [
                {
                    "seed": r.seed,
                    "raw_accuracy": r.raw_accuracy,
                    "reservoir_accuracy": r.reservoir_accuracy,
                    "delta": r.delta,
                    "train_size": r.train_size,
                    "test_size": r.test_size,
                }
                for r in self.records
            ],
            "confusion_raw": self.confusion_raw.tolist(),
            "confusion_reservoir": self.confusion_reservoir.tolist(),
            "reservoir_health": dict(self.health),
        }


def run_replicates(
    *,
    experiment: str,
    config: DelayTensorConfig,
    dataset_factory: DatasetFactory,
    seeds: Sequence[int],
    test_fraction: float = DEFAULT_TEST_FRACTION,
    parameters: dict[str, int | float] | None = None,
    max_examples: int = 3,
    max_example_wires: int | None = None,
) -> Summary:
    """Run the two-path comparison once per seed and aggregate the results."""
    if not seeds:
        raise ValueError("at least one seed is required")

    config.validate()
    reservoir = DelayTensor(config)
    records: list[SeedRecord] = []
    examples: list[TraceExample] = []
    health: dict[str, float | str] = {}
    n_classes = len(dataset_factory(seeds[0]).class_names)
    confusion_raw = np.zeros((n_classes, n_classes), dtype=np.int64)
    confusion_reservoir = np.zeros((n_classes, n_classes), dtype=np.int64)

    for seed in seeds:
        dataset = dataset_factory(seed)
        x_train, y_train, x_test, y_test = stratified_split(
            dataset.waveforms, dataset.labels, test_fraction=test_fraction, seed=seed
        )

        # Same readout class and hyperparameters on both sides: the comparison
        # is about feature generation, not classifier capacity.
        raw_model = LinearReadout(seed=seed).fit(x_train, y_train)
        raw_pred = raw_model.predict(x_test)

        # The reservoir is frozen. Only the readout trains.
        rng = np.random.default_rng([seed, 1])
        train_features = reservoir.run(x_train, rng).features
        result_test = reservoir.run(x_test, rng)
        reservoir_model = LinearReadout(seed=seed).fit(train_features, y_train)
        reservoir_pred = reservoir_model.predict(result_test.features)

        records.append(
            SeedRecord(
                seed=seed,
                raw_accuracy=float(np.mean(raw_pred == y_test)),
                reservoir_accuracy=float(np.mean(reservoir_pred == y_test)),
                train_size=len(y_train),
                test_size=len(y_test),
            )
        )
        confusion_raw += confusion_matrix(raw_pred, y_test, n_classes)
        confusion_reservoir += confusion_matrix(reservoir_pred, y_test, n_classes)
        if not health:
            health = result_test.health
            examples = _collect_examples(
                x_test,
                y_test,
                raw_pred,
                reservoir_pred,
                result_test.traces,
                max_examples=max_examples,
                max_wires=max_example_wires,
            )

    return Summary(
        experiment=experiment,
        config=config,
        class_names=dataset_factory(seeds[0]).class_names,
        records=records,
        confusion_raw=confusion_raw,
        confusion_reservoir=confusion_reservoir,
        examples=examples,
        health=health,
        parameters=dict(parameters or {}),
    )


def _collect_examples(
    waveforms: np.ndarray,
    labels: np.ndarray,
    raw_pred: np.ndarray,
    reservoir_pred: np.ndarray,
    traces: np.ndarray,
    *,
    max_examples: int,
    max_wires: int | None,
) -> list[TraceExample]:
    """Take a few test samples spread across the label range, not just the head."""
    classes = np.unique(labels)
    step = max(1, len(classes) // max(1, min(max_examples, len(classes))))
    chosen_classes = sorted(set(classes[::step]))[:max_examples]
    chosen = [int(np.flatnonzero(labels == class_id)[0]) for class_id in chosen_classes]

    # `:max_wires` with a None stop is just `:`, so the field view gets the
    # whole tensor unless a caller deliberately caps it for report size.
    kept = traces[:, :max_wires, :] if max_wires is not None else traces
    return [
        TraceExample(
            true_label=int(labels[idx]),
            predicted_raw=int(raw_pred[idx]),
            predicted_reservoir=int(reservoir_pred[idx]),
            waveform=waveforms[idx],
            traces=kept[idx],
        )
        for idx in chosen
    ]
