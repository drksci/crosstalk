"""Temporal XOR: two events in separate windows, label = their XOR.

The two pulses arrive far enough apart that a linear readout over the raw
waveform cannot separate the classes; the reservoir has to carry the first
event forward until the second one lands. This is the archive's Stage-2 stub,
implemented rather than left as a placeholder.
"""

from __future__ import annotations

from collections.abc import Sequence

from delay_tensor.config import DelayTensorConfig
from delay_tensor.datasets import make_temporal_xor
from delay_tensor.experiments.base import (
    DEFAULT_SEEDS,
    DEFAULT_TEST_FRACTION,
    Summary,
    run_replicates,
)


def run_xor(
    config: DelayTensorConfig,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    samples: int = 256,
    time_samples: int = 64,
    noise_std: float = 0.02,
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> Summary:
    """Compare raw and reservoir features on temporal XOR."""

    def dataset_factory(seed: int):
        return make_temporal_xor(
            samples=samples,
            time_samples=time_samples,
            seed=seed,
            noise_std=noise_std,
        )

    return run_replicates(
        experiment="xor",
        config=config,
        dataset_factory=dataset_factory,
        seeds=seeds,
        test_fraction=test_fraction,
        parameters={
            "samples": samples,
            "time_samples": time_samples,
            "noise_std": noise_std,
            "test_fraction": test_fraction,
        },
    )
