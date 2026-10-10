import numpy as np
import pytest

from delay_tensor.config import (
    CouplingConfig,
    DelayTensorConfig,
    LossConfig,
    ReadoutConfig,
    TensorConfig,
)


def make_config(**overrides) -> DelayTensorConfig:
    return DelayTensorConfig(
        tensor=TensorConfig(wires=6, slots=24, slot_dt_ns=20.0, loop_period_ns=480.0),
        coupling=overrides.pop("coupling", CouplingConfig()),
        loss=overrides.pop("loss", LossConfig()),
        readout=overrides.pop("readout", ReadoutConfig()),
    )


def impulse(batch: int = 3, time: int = 24) -> np.ndarray:
    x = np.zeros((batch, time))
    x[:, 4] = 1.0
    return x


def test_shapes_and_feature_count() -> None:
    from delay_tensor.reservoir import DelayTensor

    reservoir = DelayTensor(make_config())
    result = reservoir.run(impulse())

    assert result.traces.shape == (3, 6, 24)
    assert result.features.shape == (3, 6 * 24)


def test_rightmost_wire_readout_exposes_only_slots() -> None:
    from delay_tensor.reservoir import DelayTensor

    reservoir = DelayTensor(make_config(readout=ReadoutConfig(traces="rightmost_wire")))
    result = reservoir.run(impulse())
    assert result.features.shape == (3, 24)
    # The edge nonlinearity still applies; only the wire selection changes.
    np.testing.assert_allclose(result.features, np.tanh(result.traces[:, -1, :]))


def test_deterministic_without_noise() -> None:
    from delay_tensor.reservoir import DelayTensor

    reservoir = DelayTensor(make_config())
    x = np.random.default_rng(0).standard_normal((4, 24))
    first = reservoir.run(x).features
    second = reservoir.run(x).features
    np.testing.assert_allclose(first, second)


def test_noise_requires_an_rng() -> None:
    from delay_tensor.reservoir import DelayTensor

    reservoir = DelayTensor(make_config(loss=LossConfig(noise_std=0.01)))
    with pytest.raises(ValueError, match="rng"):
        reservoir.run(impulse())


def test_noise_is_reproducible_from_the_rng_seed() -> None:
    from delay_tensor.reservoir import DelayTensor

    reservoir = DelayTensor(make_config(loss=LossConfig(noise_std=0.05)))
    x = np.random.default_rng(1).standard_normal((2, 24))
    a = reservoir.run(x, np.random.default_rng(7)).features
    b = reservoir.run(x, np.random.default_rng(7)).features
    c = reservoir.run(x, np.random.default_rng(8)).features
    np.testing.assert_allclose(a, b)
    assert not np.allclose(a, c)


def test_coupling_operator_only_reaches_neighbouring_wires() -> None:
    from delay_tensor.reservoir import DelayTensor

    # Excite wire 0 only and inspect the coupling term directly: with no
    # coupling nothing moves, with coupling only wire 1 is touched.
    state = np.zeros((1, 6))
    state[0, 0] = 1.0

    isolated = DelayTensor(make_config(coupling=CouplingConfig(neighbour_strength=0.0)))
    neighbours, mask_term = isolated._neighbour_term(state, isolated._masks[:, 0])
    assert np.allclose(neighbours, 0.0)
    assert mask_term is None  # modulated mode folds the mask into the term

    coupled = DelayTensor(make_config(coupling=CouplingConfig(neighbour_strength=0.2)))
    neighbours, _ = coupled._neighbour_term(state, coupled._masks[:, 0])
    assert neighbours[0, 1] > 0.0
    assert np.allclose(neighbours[0, [0, 2, 3, 4, 5]], 0.0)

    additive = DelayTensor(
        make_config(coupling=CouplingConfig(mask_mode="additive", mask_gain=0.05))
    )
    _, mask_term = additive._neighbour_term(state, additive._masks[:, 0])
    assert mask_term is not None
    np.testing.assert_allclose(mask_term, 0.05 * additive._masks[:, 0])


def test_coupling_changes_the_produced_traces() -> None:
    from delay_tensor.reservoir import DelayTensor

    x = np.zeros((1, 24))
    x[0, 3] = 1.0
    off = DelayTensor(make_config(coupling=CouplingConfig(neighbour_strength=0.0)))
    on = DelayTensor(make_config(coupling=CouplingConfig(neighbour_strength=0.2)))
    assert not np.allclose(off.run(x).traces, on.run(x).traces)


def test_second_neighbour_reaches_two_wires_away() -> None:
    from delay_tensor.reservoir import DelayTensor

    x = np.zeros((1, 24))
    x[0, 3] = 1.0
    config = make_config(
        coupling=CouplingConfig(neighbour_strength=0.0, second_neighbour_strength=0.2),
    )
    result = DelayTensor(config).run(x)
    assert np.abs(result.traces[0, 2]).max() > 0.0


def test_masks_are_stable_and_actually_modulate() -> None:
    from delay_tensor.reservoir import DelayTensor

    a = DelayTensor(make_config(coupling=CouplingConfig(mask_seed=0)))
    b = DelayTensor(make_config(coupling=CouplingConfig(mask_seed=0)))
    c = DelayTensor(make_config(coupling=CouplingConfig(mask_seed=1)))
    np.testing.assert_allclose(a._masks, b._masks)
    assert not np.allclose(a._masks, c._masks)
    # sigmoid of a standard-normal mask spans a usable range instead of
    # sitting flat at 0.5 the way the archive's 0.01-scale masks did.
    sigmoid = 1.0 / (1.0 + np.exp(-a._masks))
    assert sigmoid.min() < 0.3
    assert sigmoid.max() > 0.7


def test_states_stay_bounded_and_short_input_keeps_running() -> None:
    from delay_tensor.reservoir import DelayTensor

    reservoir = DelayTensor(make_config())
    short = np.ones((2, 5))
    result = reservoir.run(short)
    assert result.traces.shape == (2, 6, 24)
    assert np.abs(result.traces).max() <= 1.0
    # Slots past the input keep evolving on decay alone rather than being
    # zero-padded.
    assert np.abs(result.traces[:, :, -1]).max() > 0.0


def test_rejects_bad_input_shapes() -> None:
    from delay_tensor.reservoir import DelayTensor

    reservoir = DelayTensor(make_config())
    with pytest.raises(ValueError, match="2-D"):
        reservoir.run(np.zeros((2, 3, 4)))
