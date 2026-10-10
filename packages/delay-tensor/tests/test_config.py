from pathlib import Path

import pytest

from delay_tensor.config import (
    ConfigError,
    CouplingConfig,
    DelayTensorConfig,
    LossConfig,
    TensorConfig,
    load_config,
    save_config,
)

CONFIGS = Path(__file__).parents[1] / "configs"


def make_config(**overrides) -> DelayTensorConfig:
    tensor = {
        "wires": 4,
        "slots": 16,
        "slot_dt_ns": 20.0,
        "loop_period_ns": 320.0,
    }
    tensor.update(overrides.pop("tensor", {}))
    return DelayTensorConfig(
        tensor=TensorConfig(**tensor),
        coupling=overrides.pop("coupling", CouplingConfig()),
        loss=overrides.pop("loss", LossConfig()),
        name=overrides.pop("name", "test"),
        **overrides,
    )


def test_shipped_configs_load_and_validate() -> None:
    for path in sorted(CONFIGS.glob("*.yaml")):
        config = load_config(path)
        assert config.name == path.stem
        assert config.feature_count == config.tensor.wires * config.tensor.slots


def test_round_trip_preserves_values(tmp_path: Path) -> None:
    original = load_config(CONFIGS / "tensor_8x64.yaml")
    target = tmp_path / "copy.yaml"
    save_config(original, target)
    reloaded = load_config(target)
    assert reloaded.to_dict() == original.to_dict()


def test_loop_period_must_match_geometry() -> None:
    config = make_config(tensor={"loop_period_ns": 999.0})
    with pytest.raises(ConfigError, match="loop_period_ns"):
        config.validate()


def test_unknown_section_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "bad.yaml"
    target.write_text(
        "tensor:\n  wires: 4\n  slots: 16\n  slot_dt_ns: 1.0\n  loop_period_ns: 16.0\nloss:\n  decay: 0.9\ntyop: {}\n"
    )
    with pytest.raises(ConfigError, match="unknown section"):
        load_config(target)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    target = tmp_path / "bad.yaml"
    target.write_text(
        "tensor:\n"
        "  wires: 4\n  slots: 16\n  slot_dt_ns: 1.0\n  loop_period_ns: 16.0\n"
        "readout:\n  traces: all_wires\n  edges: right\n"
    )
    with pytest.raises(ConfigError, match=r"unknown key.*edges"):
        load_config(target)


def test_unsupported_option_is_rejected_with_a_list() -> None:
    config = make_config(coupling=CouplingConfig(mask_mode="magic"))
    with pytest.raises(ConfigError, match=r"mask_mode.*modulated"):
        config.validate()


def test_unmodelled_fields_must_be_zero() -> None:
    with pytest.raises(ConfigError, match="diagonal_strength"):
        make_config(coupling=CouplingConfig(diagonal_strength=0.1)).validate()
    with pytest.raises(ConfigError, match="jitter_std_slots"):
        make_config(loss=LossConfig(jitter_std_slots=0.5)).validate()


def test_rightmost_wire_has_slot_count_features() -> None:
    from delay_tensor.config import ReadoutConfig

    config = make_config(readout=ReadoutConfig(traces="rightmost_wire"))
    assert config.feature_count == config.tensor.slots
