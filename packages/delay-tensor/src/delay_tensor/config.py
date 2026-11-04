"""Typed, validated configuration for a tensor run.

The archive version passed raw YAML dicts around and checked a couple of keys
at call sites, so a typo in a config surfaced as a ``KeyError`` deep inside the
backend. Here every section is a frozen dataclass, unknown keys are rejected
when the file is loaded, and validation runs once, up front.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import yaml

SUPPORTED_TOPOLOGIES = ("nearest_neighbour", "extracted")
SUPPORTED_MASK_MODES = ("modulated", "additive")
SUPPORTED_NONLINEARITIES = ("tanh", "identity")
SUPPORTED_LOCATIONS = ("state", "edge_and_state")
SUPPORTED_TRACE_MODES = ("all_wires", "rightmost_wire")


class ConfigError(ValueError):
    """A config file is structurally invalid."""


@dataclass(frozen=True)
class TensorConfig:
    """Geometry of the delay tensor.

    ``loop_period_ns`` must equal ``slots * slot_dt_ns``: a slot is one time
    bucket of the recirculating loop, so the two spell the same quantity.
    """

    wires: int
    slots: int
    slot_dt_ns: float
    loop_period_ns: float
    bands: int = 1

    def validate(self, prefix: str = "tensor") -> None:
        if self.wires < 2:
            raise ConfigError(f"{prefix}.wires must be >= 2, got {self.wires}")
        if self.slots < 2:
            raise ConfigError(f"{prefix}.slots must be >= 2, got {self.slots}")
        if self.slot_dt_ns <= 0:
            raise ConfigError(f"{prefix}.slot_dt_ns must be > 0, got {self.slot_dt_ns}")
        if self.loop_period_ns <= 0:
            raise ConfigError(f"{prefix}.loop_period_ns must be > 0, got {self.loop_period_ns}")
        if abs(self.loop_period_ns - self.slots * self.slot_dt_ns) > 1e-9:
            raise ConfigError(
                f"{prefix}.loop_period_ns ({self.loop_period_ns}) must equal "
                f"slots * slot_dt_ns ({self.slots} * {self.slot_dt_ns} = "
                f"{self.slots * self.slot_dt_ns})"
            )
        if self.bands != 1:
            raise ConfigError(
                f"{prefix}.bands: only a single band is modelled today, got {self.bands}"
            )


@dataclass(frozen=True)
class CouplingConfig:
    """Neighbour coupling and the mask that modulates it.

    ``mask_seed`` decouples mask generation from the experiment seed, so two
    runs that differ only in experiment seed see the same tensor.

    Two topologies. ``nearest_neighbour`` is the hand-written update: coupling
    is an arithmetic expression in ``neighbour_strength`` and its siblings.
    ``extracted`` replaces that expression with a state-space model fitted to a
    simulated line, so the coupling comes from ``model_path`` instead of from
    arithmetic -- which is why the neighbour strengths must then be zero rather
    than merely ignored. ``model_path`` is resolved against the config file's
    own directory by :func:`load_config`, so a config travels with its model.
    """

    topology: str = "nearest_neighbour"
    neighbour_strength: float = 0.08
    second_neighbour_strength: float = 0.0
    diagonal_strength: float = 0.0
    mask_mode: str = "modulated"
    mask_gain: float = 0.05
    mask_seed: int = 0
    model_path: str | None = None

    def validate(self, prefix: str = "coupling") -> None:
        if self.topology not in SUPPORTED_TOPOLOGIES:
            raise ConfigError(
                f"{prefix}.topology must be one of {SUPPORTED_TOPOLOGIES}, got {self.topology!r}"
            )
        if self.mask_mode not in SUPPORTED_MASK_MODES:
            raise ConfigError(
                f"{prefix}.mask_mode must be one of {SUPPORTED_MASK_MODES}, got {self.mask_mode!r}"
            )
        if self.topology == "extracted":
            if not self.model_path:
                raise ConfigError(f"{prefix}.topology 'extracted' requires {prefix}.model_path")
            stale = {
                name: getattr(self, name)
                for name in ("neighbour_strength", "second_neighbour_strength")
                if getattr(self, name) != 0.0
            }
            if stale:
                raise ConfigError(
                    f"{prefix}: topology 'extracted' takes its coupling from the macromodel, "
                    f"so these must be 0 and not merely ignored, got {stale}"
                )
        elif self.model_path is not None:
            raise ConfigError(
                f"{prefix}.model_path only applies to topology 'extracted', not {self.topology!r}"
            )
        for name in ("neighbour_strength", "second_neighbour_strength"):
            value = getattr(self, name)
            if value < 0:
                raise ConfigError(f"{prefix}.{name} must be >= 0, got {value}")
        if self.diagonal_strength != 0.0:
            raise ConfigError(
                f"{prefix}.diagonal_strength: cross-band coupling is not modelled yet, set it to 0"
            )
        if self.mask_gain < 0:
            raise ConfigError(f"{prefix}.mask_gain must be >= 0, got {self.mask_gain}")


@dataclass(frozen=True)
class LossConfig:
    """Per-slot decay, measurement noise, and timing jitter."""

    decay: float = 0.995
    noise_std: float = 0.0
    jitter_std_slots: float = 0.0

    def validate(self, prefix: str = "loss") -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ConfigError(f"{prefix}.decay must be in (0, 1], got {self.decay}")
        if self.noise_std < 0:
            raise ConfigError(f"{prefix}.noise_std must be >= 0, got {self.noise_std}")
        if self.jitter_std_slots < 0:
            raise ConfigError(
                f"{prefix}.jitter_std_slots must be >= 0, got {self.jitter_std_slots}"
            )
        if self.jitter_std_slots > 0:
            raise ConfigError(
                f"{prefix}.jitter_std_slots: timing jitter is not modelled yet, set it to 0"
            )


@dataclass(frozen=True)
class NonlinearityConfig:
    """Saturation stand-in for later clipping / limiter / comparator electronics."""

    type: str = "tanh"
    location: str = "edge_and_state"

    def validate(self, prefix: str = "nonlinearity") -> None:
        if self.type not in SUPPORTED_NONLINEARITIES:
            raise ConfigError(
                f"{prefix}.type must be one of {SUPPORTED_NONLINEARITIES}, got {self.type!r}"
            )
        if self.location not in SUPPORTED_LOCATIONS:
            raise ConfigError(
                f"{prefix}.location must be one of {SUPPORTED_LOCATIONS}, got {self.location!r}"
            )


@dataclass(frozen=True)
class ReadoutConfig:
    """Which traces become classifier features.

    ``all_wires`` exposes wires x slots virtual nodes; ``rightmost_wire`` is the
    stricter single-probe control.
    """

    traces: str = "all_wires"

    def validate(self, prefix: str = "readout") -> None:
        if self.traces not in SUPPORTED_TRACE_MODES:
            raise ConfigError(
                f"{prefix}.traces must be one of {SUPPORTED_TRACE_MODES}, got {self.traces!r}"
            )


@dataclass(frozen=True)
class DelayTensorConfig:
    """A complete, validated experiment config."""

    tensor: TensorConfig
    coupling: CouplingConfig = CouplingConfig()
    loss: LossConfig = LossConfig()
    nonlinearity: NonlinearityConfig = NonlinearityConfig()
    readout: ReadoutConfig = ReadoutConfig()
    name: str = "unnamed"

    @property
    def feature_count(self) -> int:
        """Number of virtual-node features the readout sees per waveform."""
        if self.readout.traces == "rightmost_wire":
            return self.tensor.slots
        return self.tensor.wires * self.tensor.slots

    def validate(self) -> None:
        self.tensor.validate()
        self.coupling.validate()
        self.loss.validate()
        self.nonlinearity.validate()
        self.readout.validate()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("name")
        return data


_SECTIONS: dict[str, type] = {
    "tensor": TensorConfig,
    "coupling": CouplingConfig,
    "loss": LossConfig,
    "nonlinearity": NonlinearityConfig,
    "readout": ReadoutConfig,
}


def load_config(path: str | Path) -> DelayTensorConfig:
    """Load and validate a YAML config file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path}: top level must be a mapping")

    unknown = sorted(set(raw) - set(_SECTIONS))
    if unknown:
        raise ConfigError(
            f"{config_path}: unknown section(s) {unknown}; expected one of {sorted(_SECTIONS)}"
        )
    if "tensor" not in raw:
        raise ConfigError(f"{config_path}: missing required section 'tensor'")

    sections: dict[str, Any] = {}
    for key, cls in _SECTIONS.items():
        if key not in raw:
            sections[key] = cls()
            continue
        value = raw[key]
        if not isinstance(value, dict):
            raise ConfigError(f"{config_path}: section '{key}' must be a mapping")
        allowed = {f.name for f in fields(cls)}
        unexpected = sorted(set(value) - allowed)
        if unexpected:
            raise ConfigError(
                f"{config_path}: unknown key(s) {unexpected} in section '{key}'; "
                f"expected {sorted(allowed)}"
            )
        try:
            sections[key] = cls(**value)
        except TypeError as exc:
            raise ConfigError(f"{config_path}: section '{key}' is invalid: {exc}") from exc

    config = DelayTensorConfig(name=config_path.stem, **sections)
    config.validate()
    if config.coupling.model_path:
        # resolve against the config file, not the working directory, so a
        # config and its macromodel can be moved together
        model = Path(config.coupling.model_path)
        if not model.is_absolute():
            model = (config_path.parent / model).resolve()
        config = replace(config, coupling=replace(config.coupling, model_path=str(model)))
    return config


def save_config(config: DelayTensorConfig, path: str | Path) -> None:
    """Write a config back to YAML, ready to be loaded again."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {"tensor": asdict(config.tensor)}
    payload.update(
        {
            "coupling": asdict(config.coupling),
            "loss": asdict(config.loss),
            "nonlinearity": asdict(config.nonlinearity),
            "readout": asdict(config.readout),
        }
    )
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
