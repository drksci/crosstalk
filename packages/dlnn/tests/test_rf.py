"""Tests of the RF tier (channel-level scikit-rf simulation). Uses a RANDOMLY INITIALISED TDL-Net-S, so agreement numbers
are agreement with that model's own V-tier outputs (not a statement about a trained network)."""
import sys
import types
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from dlnn import contract  # noqa: E402
from dlnn.data import modrec, narma10  # noqa: E402
from dlnn.model import reservoir  # noqa: E402
from dlnn.tiers import common, rf, virtual  # noqa: E402
from helpers_tdlnet_rf import random_model  # noqa: E402

IDEAL = dict(preset="ideal", snr_db=None)


@pytest.fixture(scope="module")
def ds():
    return modrec.dataset(n_train=2000, n_test=400)


@pytest.fixture(scope="module")
def model(ds):
    return random_model(ds, seed=0)


@pytest.fixture(scope="module")
def model_npz(model, tmp_path_factory):
    return model.save(tmp_path_factory.mktemp("rfm") / "model.npz")


@pytest.fixture(scope="module")
def vref(model, ds):
    return virtual.forward(model, ds["x_test"])


def agree(a, b):
    return float(np.mean(a.argmax(1) == b.argmax(1)))


def test_skrf_is_really_used():
    assert rf.HAVE_SKRF, "scikit-rf missing: the tier would be a fallback"
    import skrf
    assert rf.SKRF_VERSION == str(skrf.__version__)


def test_loss_calibration_matches_target_at_fref():
    for L in (5.0, 14.0, 32.0):
        rc = rf.resolve_cfg(dict(loss_db_per_us=L, gamma_mismatch=0.0))
        g = rf.make_grid(rc, 128)
        H, cw = rf.line_response(rc, g, 100, 8)          # 100 symbols = 1 us of delay
        i = int(np.argmin(abs(g.f - 100e6)))
        assert -20 * np.log10(abs(H[i]) / cw) == pytest.approx(L, abs=0.1)


def test_loss_scales_like_sqrt_f_and_is_dispersive():
    rc = rf.resolve_cfg(dict(loss_db_per_us=32.0, gamma_mismatch=0.0))
    g = rf.make_grid(rc, 128)
    H, cw = rf.line_response(rc, g, 100, 8)
    a = lambda f: -20 * np.log10(abs(H[int(np.argmin(abs(g.f - f)))]) / cw)  # noqa: E731
    assert a(100e6) / a(25e6) == pytest.approx(2.0, rel=0.1)       # sqrt(4)
    gs, _ = rf._fir_from_H(H, g)
    j = np.arange(-g.nsym // 2, g.nsym // 2)
    assert gs[j == 6][0] / cw > 0.01 and gs[j == -6][0] / cw == pytest.approx(0, abs=2e-3)   # one-sided ISI tail (smearing)


def test_ideal_channel_reproduces_virtual_exactly(model, ds, vref):
    r = rf.forward(model, ds["x_test"], IDEAL)
    assert agree(r["out"], vref["out"]) >= 0.98
    for a, b in zip(r["layers"], vref["layers"]):
        assert np.max(np.abs(a - b)) < 2e-3
    assert agree(r["out"], vref["out"]) == 1.0


def test_ideal_30db_stage_snr_agreement_documented(model, ds, vref):
    # Random model: V itself flips ~2 % of decisions under 30 dB input AWGN; per-stage 30 dB noise at 3 stages flips more.
    a30 = agree(rf.forward(model, ds["x_test"], dict(preset="ideal", snr_db=30.0))["out"], vref["out"])
    a40 = agree(rf.forward(model, ds["x_test"], dict(preset="ideal", snr_db=40.0))["out"], vref["out"])
    a50 = agree(rf.forward(model, ds["x_test"], dict(preset="ideal", snr_db=50.0))["out"], vref["out"])
    assert a30 >= 0.85 and a40 >= a30 - 0.01 and a50 >= a40 - 0.01 and a50 >= 0.97


def test_fine_grid_waveform_matches_symbol_rate_engine(model, ds):
    for cfg in (dict(preset="ideal"),
                dict(loss_db_per_us=14.0, gamma_mismatch=0.2, isolation_db=25.0, tap_amp_err_db=0.5,
                     tap_phase_err_deg=10.0, crosstalk_db=-30.0)):
        rc = rf.resolve_cfg(cfg)
        ch, g = rf.build_channels(model, rc)
        x = ds["x_test"][:3].astype(np.float32)
        lin = rf.fir_apply(x, ch[0])
        for e in range(2):
            yf = rf.fine_waveform(model.layers[0], ch[0], g, x[e])
            ys = yf[:, ::g.U][:, :x.shape[2]]
            assert np.linalg.norm(ys - lin[e]) / np.linalg.norm(lin[e]) < 5e-3


def test_numpy_fallback_matches_skrf_and_labels_fallback(model, ds, monkeypatch, tmp_path):
    rc = rf.resolve_cfg(dict(loss_db_per_us=14.0, gamma_mismatch=0.05))
    g = rf.make_grid(rc, 128)
    H1, _ = rf.line_response(rc, g, 50, 8)
    monkeypatch.setattr(rf, "HAVE_SKRF", False)
    rf._SIGMA_CACHE.clear()
    H2, _ = rf.line_response(rc, g, 50, 8)
    sel = g.f < 100e6
    assert np.max(np.abs(np.abs(H1[sel]) - np.abs(H2[sel])) / np.abs(H1[sel])) < 0.03
    small = {k: (v[:60] if isinstance(v, np.ndarray) else v) for k, v in ds.items()}
    small["sha256"] = "x"
    model.save(tmp_path / "m.npz")
    r = rf.run(tmp_path / "m.npz", small, {"run_id": "t", "sweep_snrs": [30.0]}, tmp_path)
    contract.validate(r)
    assert r.status == "fallback" and "FALLBACK" in " ".join(r.fidelity["omits"])
    monkeypatch.undo()
    rf._SIGMA_CACHE.clear()


def test_isolation_leak_scales_with_isolation(model, ds):
    x = ds["x_test"][:100].astype(np.float32)
    base = rf.fir_apply(x, rf.build_channels(model, rf.resolve_cfg(IDEAL))[0][0])
    err = {}
    for iso in (15.0, 25.0, 40.0):
        ch = rf.build_channels(model, rf.resolve_cfg({**IDEAL, "isolation_db": iso}))[0][0]
        err[iso] = np.linalg.norm(rf.fir_apply(x, ch) - base)
    assert err[15.0] > err[25.0] > err[40.0]
    assert err[15.0] / err[25.0] == pytest.approx(10 ** (10 / 20), rel=0.25)


def test_tap_amplitude_error_zero_is_exact_and_grows(model, ds):
    x = ds["x_test"][:100].astype(np.float32)
    base = rf.fir_apply(x, rf.build_channels(model, rf.resolve_cfg(IDEAL))[0][0])
    e = [np.linalg.norm(rf.fir_apply(x, rf.build_channels(model, rf.resolve_cfg({**IDEAL, "tap_amp_err_db": a}))[0][0]) - base)
         for a in (0.0, 0.5, 2.0)]
    assert e[0] < 1e-3 * np.linalg.norm(base) and e[1] < e[2]


def test_loss_causes_tap_droop_and_equalizer_restores_it(model):
    ly = model.layers[2]
    lo = rf.build_layer(ly, rf.resolve_cfg({**IDEAL, "loss_db_per_us": 14.0}), rf.make_grid(rf.resolve_cfg(IDEAL), 400), 2)
    g = lo.info["tap_main_gain_db"]
    assert g[0] > g[-1] + 5                      # 7*16 symbols = 1.12 us at 14 dB/us -> droop of tens of dB main-lobe gain
    eq = rf.build_layer(ly, rf.resolve_cfg({**IDEAL, "loss_db_per_us": 14.0, "tap_equalize": True}),
                        rf.make_grid(rf.resolve_cfg(IDEAL), 400), 2)
    assert np.allclose(eq.info["tap_main_gain_db"], 0.0, atol=1e-6)


def test_agreement_monotone_in_loss(model, ds, vref):
    a = [agree(rf.forward(model, ds["x_test"], {**IDEAL, "loss_db_per_us": L})["out"], vref["out"]) for L in (0.0, 5.0, 14.0, 32.0)]
    assert a[0] == 1.0 and a[0] > a[1] > a[3] and a[1] >= a[2] - 0.02


def test_rapp_amplifier_fit_and_limits():
    z = np.linspace(-6, 6, 4001)
    r = rf.amplifier(z, {"amp_model": "rapp", "rapp_p": "fit"})
    assert np.max(np.abs(r - np.tanh(z))) < 0.03
    assert abs(r[-1]) <= 1.0 + 1e-9 and np.isclose(rf.amplifier(np.array([1e-4]), {"amp_model": "rapp", "rapp_p": "fit"})[0], 1e-4, rtol=1e-3)
    assert np.allclose(rf.amplifier(z, {"amp_model": "tanh"}), np.tanh(z))


def test_adc_quantiser():
    f = np.random.default_rng(0).standard_normal((100, 8)).astype(np.float32)
    for b in (3, 6, 8):
        q = rf.quantize_adc(f, b)
        assert len(np.unique(q)) <= 2 ** b - 1
        assert np.max(np.abs(q - f)) <= np.abs(f).max() / (2 ** (b - 1) - 1) / 2 + 1e-6
    assert rf.quantize_adc(f, None) is f


def test_stage_noise_level_matches_snr_and_nf():
    from dlnn.model.npz import Layer, Model
    ly = Layer(W=np.zeros((2, 2, 3), np.int8), dilation=1, gain=np.ones(2, np.float32), bias=np.zeros(2, np.float32), vsat=1e3)
    m = Model(kind="tdlnet", layers=[ly], readout_W=np.zeros((2, 2), np.float32), readout_b=np.zeros(2, np.float32),
              readout_from=[0], readout_kind="pooled")
    rc = rf.resolve_cfg(dict(preset="ideal", snr_db=20.0, nf_db=3.0, adc_bits=None))
    ch, _ = rf.build_channels(m, rc)
    x = np.zeros((200, 2, 128), np.float32)
    y = rf.forward_with(m, x, rc, ch, [4.0])["layers"][0] * 1e3          # tanh(n/1e3)*1e3 ~ n
    assert np.std(y) == pytest.approx(2.0 * 10 ** (-(20 - 3) / 20), rel=0.03)   # sqrt(P_ref / SNR * NF)
    assert rf.stage_sigma(4.0, rc) == pytest.approx(2.0 * 10 ** (-17 / 20))


def test_run_contract_sweeps_and_arrays(model_npz, ds):
    r = rf.run(model_npz, ds, {"run_id": "t", "seed": 0}, ".")
    contract.validate(r)
    assert r.tier == "R" and r.status == "ok" and r.tool["name"] == "scikit-rf"
    axes = {s["axis"] for s in r.sweeps}
    assert {"snr_db", "loss_db_per_us", "isolation_db", "tap_amp_err_db", "adc_bits"} <= axes
    for s in r.sweeps:
        assert len(s["accuracy"]) == len(s["mean_kl"]) == len(s["values"])
        assert f"sweep.{s['axis']}.top1_agreement_vs_virtual" in r.arrays
    snr = next(s for s in r.sweeps if s["axis"] == "snr_db")
    assert snr["accuracy"][-1] >= snr["accuracy"][0] and snr["mean_kl"][-1] < snr["mean_kl"][0]
    assert any(k.startswith("waveforms.") for k in r.arrays)
    txt = " ".join(r.fidelity["omits"]).lower()
    assert "no em" in txt and "hardware" in txt
    assert all(m in r.metrics for m in ("accuracy", "top1_agreement_vs_virtual", "mean_kl_virtual_vs_tier"))


def test_run_regression_task(tmp_path):
    d = narma10.dataset()
    p = reservoir.fit(d).save(tmp_path / "m.npz")
    r = rf.run(p, d, {"run_id": "t"}, tmp_path)
    contract.validate(r)
    assert r.status == "ok" and "nmse" in r.metrics and r.metrics["nmse"] < 1.0
    contract.save(r, tmp_path)


def test_cascade_hook_missing_module_warns(model_npz, ds, tmp_path, monkeypatch):
    monkeypatch.setattr(rf, "_CASCADE_PY", tmp_path / "does_not_exist.py")
    small = {k: (v[:40] if isinstance(v, np.ndarray) else v) for k, v in ds.items()}
    pj = tmp_path / "circuit_extracted.json"
    pj.write_text("{}")
    r = rf.run(model_npz, small, {"run_id": "t", "cascade_params": str(pj), "sweep_snrs": [30.0]}, tmp_path)
    contract.validate(r)
    assert any("cascade" in w for w in r.warnings) and r.inputs_from == []


def _fake_extracted(loss=14.0):
    return {"schema": "dlnn-circuit-extracted/1", "source": "analytic-numpy-rlgc", "tool": {"name": "numpy", "version": "0"},
            "symbol_period_s": 1e-8,
            "line": {"fit": {"R_ohm_per_m": 1.0, "L_h_per_m": 250e-9, "C_f_per_m": 100e-12, "G_s_per_m": 0.0, "z0_ohm": 50.0,
                             "vp_m_per_s": 2e8, "termination_ohm": 50.0, "loss_db_per_us_100mhz": loss}},
            "taps": {"k": [0, 1], "freq_hz": [1.0, 1e8], "H_re": [[1.0, 1.0], [0.9, 0.5]], "H_im": [[0.0, 0.0], [0.0, -0.2]],
                     "delay_nominal_s": [0.0, 1e-8]},
            "diffpair": {"vsat_v": 0.05, "vsat_design_v": 0.0518, "A_v": 1.0, "r2": 0.999},
            "amp": {"vsat_scale": 1.1, "gain_scale": 0.95, "offset_rel": 0.01}}


def test_cascade_hook_real_cascade_module(model_npz, ds, tmp_path):
    import json
    from dlnn import cascade
    assert rf._CASCADE_PY.exists() and hasattr(cascade, "apply_to_rf_cfg")
    pj = tmp_path / "circuit_extracted.json"
    pj.write_text(json.dumps(_fake_extracted(14.0)))
    small = {k: (v[:40] if isinstance(v, np.ndarray) else v) for k, v in ds.items()}
    r = rf.run(model_npz, small, {"run_id": "t", "cascade_params": str(pj), "sweep_snrs": [30.0]}, tmp_path)
    contract.validate(r)
    assert r.inputs_from == [cascade.EXTRACT_REL_PATH]
    assert "14 dB/us" in " ".join(r.fidelity["models"])
    assert any("analytic-numpy-rlgc" in w for w in r.warnings)
    out = rf._apply_cascade({"cascade_params": str(pj)}, [], [])
    assert out["loss_db_per_us"] == 14.0 and out["amp_vsat_scale"] == 1.1 and out["amp_offset_rel"] == 0.01
    rc = rf.resolve_cfg(out)
    assert rc["loss_db_per_us"] == 14.0 and rc["amp_gain_scale"] == 0.95
