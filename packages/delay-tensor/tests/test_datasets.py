import numpy as np
import pytest

from delay_tensor.datasets import (
    CLASS_NAMES,
    make_temporal_xor,
    make_waveform_classification,
    normalize,
    stratified_split,
)


def test_classification_shapes_labels_and_normalisation() -> None:
    dataset = make_waveform_classification(samples_per_class=8, time_samples=32, seed=3)
    assert dataset.waveforms.shape == (8 * len(CLASS_NAMES), 32)
    assert dataset.labels.shape == (8 * len(CLASS_NAMES),)
    assert dataset.n_classes == 6
    assert set(np.unique(dataset.labels)) == set(range(6))
    np.testing.assert_allclose(dataset.waveforms.mean(axis=1), 0.0, atol=1e-10)
    np.testing.assert_allclose(dataset.waveforms.std(axis=1), 1.0, atol=1e-6)


def test_seeded_generation_is_reproducible_and_seed_sensitive() -> None:
    a = make_waveform_classification(samples_per_class=4, time_samples=16, seed=11)
    b = make_waveform_classification(samples_per_class=4, time_samples=16, seed=11)
    c = make_waveform_classification(samples_per_class=4, time_samples=16, seed=12)
    np.testing.assert_allclose(a.waveforms, b.waveforms)
    assert not np.allclose(a.waveforms, c.waveforms)


def test_classes_are_not_distinguished_by_scale_only() -> None:
    # Every waveform is normalised per sample, so classes must not differ in
    # overall scale; a classifier has to use temporal structure.
    dataset = make_waveform_classification(samples_per_class=32, time_samples=64, seed=5)
    per_class_std = [dataset.waveforms[dataset.labels == k].std() for k in range(6)]
    assert max(per_class_std) / min(per_class_std) < 1.2


def test_temporal_xor_labels_follow_the_pulses() -> None:
    dataset = make_temporal_xor(samples=512, time_samples=64, seed=2, noise_std=0.0)
    assert dataset.class_names == ("false", "true")
    assert set(np.unique(dataset.labels)) == {0, 1}

    # With no noise, presence of each pulse is recoverable by correlating the
    # waveform with the two window templates, and it must XOR to the label.
    index = np.arange(64)
    early = np.exp(-0.5 * ((index - 16) / 2.5) ** 2)
    late = np.exp(-0.5 * ((index - 48) / 2.5) ** 2)
    corr_early = (dataset.waveforms * (early - early.mean())).sum(axis=1)
    corr_late = (dataset.waveforms * (late - late.mean())).sum(axis=1)
    recovered = ((corr_early > 0) ^ (corr_late > 0)).astype(int)
    assert np.mean(recovered == dataset.labels) > 0.95
    # Both classes are actually represented.
    assert 0.3 < dataset.labels.mean() < 0.7


def test_normalize_handles_constant_input() -> None:
    constant = np.ones((2, 8))
    result = normalize(constant)
    assert np.isfinite(result).all()
    assert np.allclose(result, 0.0)


def test_stratified_split_keeps_every_class_on_both_sides() -> None:
    dataset = make_waveform_classification(samples_per_class=12, time_samples=16, seed=9)
    x_train, y_train, x_test, y_test = stratified_split(
        dataset.waveforms, dataset.labels, test_fraction=0.25, seed=9
    )
    assert len(x_train) + len(x_test) == len(dataset.labels)
    assert set(np.unique(y_train)) == set(range(6))
    assert set(np.unique(y_test)) == set(range(6))
    # 25% of 12 rounded, at least one per class, never emptying a class.
    assert len(y_test) == 6 * 3
    # No sample appears twice.
    assert len({row.tobytes() for row in x_train} & {row.tobytes() for row in x_test}) == 0


def test_stratified_split_rejects_bad_fraction() -> None:
    with pytest.raises(ValueError, match="test_fraction"):
        stratified_split(np.zeros((4, 3)), np.zeros(4), test_fraction=0.0)
