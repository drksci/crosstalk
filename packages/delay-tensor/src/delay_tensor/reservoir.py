"""Coupled-line tensor: the simulated substrate.

The model is deliberately small and inspectable. Each slot advances one time
step of

    state[t+1] = decay * state[t] + drive[t] + coupling + mask_term

with neighbour terms taken from a spatial roll across the wire axis, and a
saturation nonlinearity standing in for later clipping electronics. It is a
feature generator, not a trained layer: nothing here ever sees a gradient.

``coupling.topology`` selects what supplies the coupling term:

``nearest_neighbour``
    The arithmetic above -- coupling is an expression in ``neighbour_strength``
    and its siblings.

``extracted``
    A state-space model fitted to a simulated line, so the update becomes
    ``state <- A state + B[:, 0] u``. ``A`` replaces ``loss.decay`` and
    ``B[:, 0]`` replaces :attr:`DelayTensor.drive_profile`, because both are
    what the fit already encodes; the mask stays, gating the input path.
    Slot *k* is model step *k*, which is why the two topologies can share one
    loop.

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
from pathlib import Path

import numpy as np

from delay_tensor.config import ConfigError, DelayTensorConfig


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
        self._extracted = self._load_extracted()

    def _load_extracted(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Load ``(A, B)`` for ``topology: extracted``, or ``None``.

        A macromodel is only useful if it runs stably, so both conditions that
        make it useless are refused here rather than discovered as a saturated
        reservoir: the state dimension has to match ``tensor.wires``, and the
        discrete model of a passive line must have spectral radius at most one.
        """
        coupling = self.config.coupling
        if coupling.topology != "extracted":
            return None

        path = Path(coupling.model_path)
        if not path.is_file():
            raise ConfigError(f"coupling.model_path does not exist: {path}")
        with np.load(path) as data:
            missing = [key for key in ("A", "B") if key not in data]
            if missing:
                raise ConfigError(f"{path}: macromodel is missing {missing}; expected A and B")
            a_mat = np.asarray(data["A"], dtype=np.float64)
            b_mat = np.asarray(data["B"], dtype=np.float64)

        order = self.config.tensor.wires
        if a_mat.shape != (order, order):
            raise ConfigError(
                f"{path}: A is {a_mat.shape} but tensor.wires is {order} -- "
                f"fit the macromodel at order {order}, or set wires to {a_mat.shape[0]}"
            )
        if b_mat.ndim != 2 or b_mat.shape[0] != order or b_mat.shape[1] < 1:
            raise ConfigError(f"{path}: B is {b_mat.shape}, expected ({order}, m) with m >= 1")

        radius = float(np.abs(np.linalg.eigvals(a_mat)).max())
        if radius > 1.0 + 1e-6:
            raise ConfigError(
                f"{path}: spectral radius {radius:.6f} > 1, so the model grows without "
                "bound -- a passive line's macromodel cannot do that, check the fit"
            )
        return a_mat, b_mat

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
        extracted = self._extracted

        for slot in range(tensor.slots):
            if extracted is not None:
                candidate = self._extracted_update(state, extracted, drive[:, slot], mask[:, slot])
            else:
                neighbours, mask_term = self._neighbour_term(state, mask[:, slot])
                candidate = loss.decay * state + drive[:, slot, None] * profile + neighbours
                if mask_term is not None:
                    candidate = candidate + mask_term
            if loss.noise_std > 0:
                candidate = candidate + loss.noise_std * rng.standard_normal(state.shape)
            state = np.tanh(candidate) if self.config.nonlinearity.type == "tanh" else candidate
            traces[:, :, slot] = state

        features = self._readout_features(traces)
        return TensorRunResult(features=features, traces=traces, health=self._health(traces))

    def _extracted_update(
        self,
        state: np.ndarray,
        model: tuple[np.ndarray, np.ndarray],
        u: np.ndarray,
        mask_column: np.ndarray,
    ) -> np.ndarray:
        """One step of ``x <- A x + B[:, 0] u``, with the mask on the input path.

        Slot *k* is model step *k*, so ``A`` replaces ``loss.decay`` (its
        spectral radius is the decay) and ``B[:, 0]`` replaces
        :attr:`drive_profile` (the coupling shape is part of the fit).

        Column 0 is the input belonging to the one driven port, which is what
        the bench drives -- every other port is terminated, and linearity folds
        their effect into this column. The mask gates that input rather than
        ``A``: ``A`` *is* the line, so rescaling it would stop this being a
        simulation of one.
        """
        a_mat, b_mat = model
        term = np.outer(u, b_mat[:, 0])
        if self.config.coupling.mask_mode == "modulated":
            return state @ a_mat.T + _sigmoid(mask_column)[None, :] * term
        return state @ a_mat.T + term + self.config.coupling.mask_gain * mask_column

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
        coupling = self.config.coupling
        if self._extracted is not None:
            # the effective decay of an extracted run is A's own, not the
            # configured loss.decay, which does not apply in that mode
            decay = float(np.abs(np.linalg.eigvals(self._extracted[0])).max())
        else:
            decay = float(self.config.loss.decay)
        return {
            "status": "saturated" if saturation > 0.25 else "healthy",
            "mean_abs_state": float(np.mean(np.abs(traces))),
            "saturation_fraction": saturation,
            "noise_std": float(self.config.loss.noise_std),
            "decay": decay,
            "topology": coupling.topology,
        }


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))
