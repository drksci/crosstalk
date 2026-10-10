"""Waveform classification: the Stage-1 simulated experiment.

Six temporal classes, one frozen reservoir, one linear readout per path. The
question the experiment answers is narrow on purpose: do reservoir-transformed
features beat the raw waveform for a fixed, simple classifier?
"""

from __future__ import annotations

from collections.abc import Sequence

from delay_tensor.config import DelayTensorConfig
from delay_tensor.datasets import make_waveform_classification
from delay_tensor.experiments.base import (
    DEFAULT_SEEDS,
    DEFAULT_TEST_FRACTION,
    Summary,
    run_replicates,
)


def run_classify(
    config: DelayTensorConfig,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    samples_per_class: int = 64,
    time_samples: int = 64,
    noise_std: float = 0.03,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> Summary:
    """Compare raw and reservoir features on synthetic waveform classes."""

    def dataset_factory(seed: int):
        return make_waveform_classification(
            samples_per_class=samples_per_class,
            time_samples=time_samples,
            seed=seed,
            noise_std=noise_std,
        )

    return run_replicates(
        experiment="classify",
        config=config,
        dataset_factory=dataset_factory,
        seeds=seeds,
        test_fraction=test_fraction,
        parameters={
            "samples_per_class": samples_per_class,
            "time_samples": time_samples,
            "noise_std": noise_std,
            "test_fraction": test_fraction,
        },
    )
