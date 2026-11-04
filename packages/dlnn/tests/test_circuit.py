"""Tier C (ngspice unit cell + extraction + calibrated behavioural model). Skipped without ngspice."""
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers_tdlnet import make_random_tdlnet  # noqa: E402

from dlnn import cascade, contract  # noqa: E402
from dlnn.data import modrec  # noqa: E402
from dlnn.tiers import circuit  # noqa: E402

needs_ngspice = pytest.mark.skipif(shutil.which("ngspice") is None,
                                   reason="ngspice not installed (apt install ngspice); C tier would only fall back")


@pytest.fixture(scope="module")
def small():
    ds = modrec.dataset(n_train=400, n_test=160)
    m = make_random_tdlnet(ds["x_train"], ds["y_train"], seed=0)
    return ds, m


@pytest.fixture(scope="module")
def spice_run(tmp_path_factory, small):
    ds, m = small
    d = tmp_path_factory.mktemp("c")
    npz = m.save(d / "model.npz")
    cfg = {"run_id": "t", "c_n_examples": 3, "c_outputs": 3, "c_workers": 2, "sweep_snrs": [10.0, 30.0]}
    res = circuit.run(npz, ds, cfg, d)
    return res, d, m, ds


def test_analytic_line_basics():
    f = np.array([1.0, 50e6, 100e6])
    ln = circuit.tap_lengths(8, 1)
    assert np.allclose(ln / circuit.VP, np.arange(8) * 1e-8)            # delay exactly k*Ts
    H = circuit.line_H(f, ln[7], 0.0)                                   # lossless matched line
    assert np.allclose(np.abs(H), 1.0, atol=1e-9)
    h100 = circuit.line_H(np.array([100e6]), 198.0, circuit.r_from_loss(32.0))
    assert abs(20 * np.log10(abs(h100[0])) + 32.0) < 0.5                # 32 dB/us over 1 us (198 m)


@needs_ngspice
def test_ac_and_dc_extraction(tmp_path):
    ac = circuit.extract_ac(tmp_path, 8, 1, circuit.r_from_loss(32.0), True)
    assert ac["H"].shape == (8, circuit.AC_NPTS + 1)
    assert ac["fit"]["fit_max_abs_err"] < 1e-3                          # SPICE LTRA == closed form
    assert abs(ac["fit"]["loss_db_per_us_100mhz"] - 32.0) < 0.5
    assert ac["taps"][7]["gain_100mhz_db"] < ac["taps"][1]["gain_100mhz_db"] < 0
    dc = circuit.extract_dc(tmp_path, True)["fit"]
    assert dc["r2"] >= 0.99
    assert 0.9 < dc["vsat_scale"] < 1.1 and 0.9 < dc["gain_scale"] < 1.1


@needs_ngspice
def test_tier_json_and_honesty(spice_run):
    res, d, m, ds = spice_run
    contract.validate(res)
    assert res.status == "ok" and res.tool["name"] == "ngspice" and res.tool["version"].startswith("4")
    txt = " ".join(res.fidelity["models"]).lower()
    assert "calibrated behavioural" in txt and "not spice results" in txt and "ngspice .tran" in txt
    assert any("layer 1" in o or "only layer 1" in o for o in res.fidelity["omits"])
    assert "params/circuit_extracted.json" in res.inputs_from
    ax = [s["axis"] for s in res.sweeps]
    assert "loss_db_per_us" in ax and "snr_db" in ax
    ls = next(s for s in res.sweeps if s["axis"] == "loss_db_per_us")
    assert ls["values"] == [0.0, 5.0, 14.0, 32.0] and len(ls["accuracy"]) == 4
    assert any(k.startswith("waveforms.") for k in res.arrays)
    contract.save(res, d)
    contract.load(d / "tiers" / "C.json")


@needs_ngspice
def test_unit_cell_nmse_and_extraction_file(spice_run):
    res, d, m, ds = spice_run
    ex = cascade.load_extracted(d / "params" / "circuit_extracted.json")
    t = ex["unit_cell_tran"]
    assert ex["source"] == "ngspice" and t["n_examples"] == 3
    assert t["nmse"] <= 0.05, t["nmse"]                                  # M2b acceptance
    assert t["nmse_calibrated_model_vs_spice"] < 0.02
    assert ex["diffpair"]["r2"] >= 0.99
    assert len(ex["netlist_sha256"]["unit_cell_example0"]) == 64
    cfg = cascade.apply_to_rf_cfg({"seed": 1}, ex)
    assert "seed" in cfg and cfg["inputs_from"] == ["params/circuit_extracted.json"]
    assert cfg["cascade_circuit"]["tap_H"]["H"].shape[0] == 8
    assert abs(cfg["loss_db_per_us"] - 32.0) < 0.5
    h = cascade.line_H(ex, ex["freq_hz"], ex["delay_s"][3])
    assert np.max(np.abs(h - ex["H"][3])) < 1e-3                         # fitted line reproduces the SPICE tap


def test_fallback_without_spice(tmp_path, small):
    ds, m = small
    npz = m.save(tmp_path / "model.npz")
    res = circuit.run(npz, ds, {"run_id": "t", "c_use_spice": False, "sweep_snrs": [10.0]}, tmp_path)
    contract.validate(res)
    assert res.status == "fallback" and res.tool["name"] == "numpy-analytic-rlgc"
    assert any("not a SPICE result" in o or "none is a SPICE result" in o for o in res.fidelity["omits"])
    ex = json.loads((tmp_path / "params" / "circuit_extracted.json").read_text())
    assert ex["source"] == "analytic-numpy-rlgc" and ex["unit_cell_tran"] is None
    assert cascade.load_extracted(tmp_path / "params" / "circuit_extracted.json")["source"] == "analytic-numpy-rlgc"


def test_calibrated_lossless_matches_virtual(small):
    """With R=0 and an ideal amp the behavioural model must reproduce V (sampling/ISI-free)."""
    ds, m = small
    from dlnn.tiers import virtual
    x = ds["x_test"][:20]
    cal = circuit.calibrated_forward(m, x, circuit.LineModel(0.0), circuit.AmpModel(), comp="none")
    ref = virtual.forward(m, x)
    for a, b in zip(cal["layers"], ref["layers"]):
        assert np.max(np.abs(a - b)) < 1e-3


def test_loss_compensation_modes_ordering(small):
    """More loss -> further from V; per-tap equalisation recovers most of it (behavioural model only)."""
    ds, m = small
    from dlnn.tiers import virtual
    x = ds["x_test"][:80]
    ref = virtual.forward(m, x)["out"]
    r32 = circuit.r_from_loss(32.0)
    err = {}
    for comp in circuit.COMP_MODES:
        out = circuit.calibrated_forward(m, x, circuit.LineModel(r32), circuit.AmpModel(), comp=comp)["out"]
        err[comp] = float(np.mean((out - ref) ** 2))
    assert err["tap"] < err["layer"] <= err["none"] * 1.5
    assert err["tap"] < 0.5 * err["none"]


@needs_ngspice
def test_reference_netlist_runs(tmp_path):
    p = circuit.write_reference_netlist(tmp_path / "unit_cell.cir")
    plots = circuit.run_ngspice(p)
    nm = plots[0]["names"]
    assert "v(y0)" in nm and "v(p0)" in nm and plots[0]["data"].shape[0] > 100
