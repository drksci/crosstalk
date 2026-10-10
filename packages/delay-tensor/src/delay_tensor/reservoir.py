"""Nearest-neighbour coupled-line tensor: the simulated substrate.

The model is deliberately small and inspectable. Each slot advances one time
step of

    state[t+1] = decay * state[t] + drive[t] + coupling + mask_term

with neighbour terms taken from a spatial roll across the wire axis, and a
saturation nonlinearity standing in for later clipping electronics. It is a
feature generator, not a trained layer: nothing here ever sees a gradient.

Compared with the archive version:

* NumPy instead of PyTorch (the reservoir is frozen, autograd bought nothing).
* The whole slot window always runs, with zero drive once the input ends,
  instead of truncating at ``min(slots, time)`` and zero-padding the state.
* Second-neighbour coupling is implemented rather than silently ignored, and
  unsupported config keys are rejected at load time.
* Masks are drawn from a standard normal so ``mask_mode: modulated`` actually
  modulates; the archive drew them at 0.01 scale, where ``sigmoid`` is flat at
  0.5 and the mask did nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from delay_tensor.config import DelayTensorConfig


@dataclass(frozen=True)
class TensorRunResult:
    """Readout surface plus the full state history for one batch."""

    features: np.ndarray  # (batch, n_features)
    traces: np.ndarray  # (batch, wires, slots)
    health: dict[str, float | str]

    @property
    def batch(self) -> int:
        return self.features.shape[0]


class DelayTensor:
    """A frozen coupled-line reservoir over ``config.tensor`` geometry."""

    def __init__(self, config: DelayTensorConfig) -> None:
        config.validate()
        self.config = config
        self._masks = self._build_masks()

    def _build_masks(self) -> np.ndarray:
        coupling = self.config.coupling
        rng = np.random.default_rng(coupling.mask_seed)
        return rng.standard_normal((self.config.tensor.wires, self.config.tensor.slots))

    @property
    def masks(self) -> np.ndarray:
        """Per-(wire, slot) mask draws; identical for every waveform fed in.

        Exposed so benches (SPICE netlist generation) build the same tensor
        the NumPy model uses, instead of re-deriving mask generation.
        """
        return self._masks

    @property
    def drive_profile(self) -> np.ndarray:
        """Launch strongest at wire 0 and taper across the bundle."""
        return np.linspace(1.0, 0.2, self.config.tensor.wires)

    def run(
        self,
        waveforms: np.ndarray,
        rng: np.random.Generator | None = None,
    ) -> TensorRunResult:
        """Push ``waveforms`` (batch, time) through the tensor."""
        x = np.asarray(waveforms, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError(f"waveforms must be 2-D (batch, time), got shape {x.shape}")
        if x.shape[1] < 1:
            raise ValueError("waveforms must contain at least one time sample")
        if self.config.loss.noise_std > 0 and rng is None:
            raise ValueError("loss.noise_std > 0 requires an explicit rng for reproducibility")

        tensor = self.config.tensor
        loss = self.config.loss
        batch, time_samples = x.shape

        # Input longer than the slot window is truncated; a short input leaves
        # the remaining slots running with zero drive, which is what the tensor
        # physically does when the launch stops.
        drive = np.zeros((batch, tensor.slots), dtype=np.float64)
        used = min(time_samples, tensor.slots)
        drive[:, :used] = x[:, :used]

        state = np.zeros((batch, tensor.wires), dtype=np.float64)
        traces = np.zeros((batch, tensor.wires, tensor.slots), dtype=np.float64)
        profile = self.drive_profile
        mask = self._masks

        for slot in range(tensor.slots):
            neighbours, mask_term = self._neighbour_term(state, mask[:, slot])
            next_state = loss.decay * state + drive[:, slot, None] * profile + neighbours
            if mask_term is not None:
                next_state = next_state + mask_term
            if loss.noise_std > 0:
                next_state = next_state + loss.noise_std * rng.standard_normal(state.shape)
            state = np.tanh(next_state) if self.config.nonlinearity.type == "tanh" else next_state
            traces[:, :, slot] = state

        features = self._readout_features(traces)
        return TensorRunResult(features=features, traces=traces, health=self._health(traces))

    def _neighbour_term(
        self, state: np.ndarray, mask_column: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Nearest- (and second-) neighbour coupling across the wire axis.

        Returns the term to add to the state update, plus an additive mask term
        when ``mask_mode: additive`` is configured (``None`` for modulated mode,
        where the mask is already folded into the neighbour term).
        """
        coupling = self.config.coupling
        base = coupling.neighbour_strength
        second = coupling.second_neighbour_strength

        left = np.roll(state, 1, axis=1)
        right = np.roll(state, -1, axis=1)
        left[:, 0] = 0.0
        right[:, -1] = 0.0
        coupled = base * (left + right)

        if second > 0.0:
            left2 = np.roll(state, 2, axis=1)
            right2 = np.roll(state, -2, axis=1)
            left2[:, :2] = 0.0
            right2[:, -2:] = 0.0
            coupled = coupled + second * (left2 + right2)

        if coupling.mask_mode == "modulated":
            # The mask scales how strongly neighbours talk, which matches the
            # "programmable coupling window" idea better than injecting a
            # separate source per virtual node.
            return _sigmoid(mask_column) * coupled, None

        return coupled, coupling.mask_gain * mask_column

    def _readout_features(self, traces: np.ndarray) -> np.ndarray:
        source = (
            np.tanh(traces) if self.config.nonlinearity.location == "edge_and_state" else traces
        )
        if self.config.readout.traces == "rightmost_wire":
            return source[:, -1, :]
        return source.reshape(source.shape[0], -1)

    def _health(self, traces: np.ndarray) -> dict[str, float | str]:
        saturation = float(np.mean(np.abs(traces) > 0.99))
        return {
            "status": "saturated" if saturation > 0.25 else "healthy",
            "mean_abs_state": float(np.mean(np.abs(traces))),
            "saturation_fraction": saturation,
            "noise_std": float(self.config.loss.noise_std),
            "decay": float(self.config.loss.decay),
        }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))
