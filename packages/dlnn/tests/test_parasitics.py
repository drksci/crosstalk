"""Fast tests of tier P (lumped-element parasitics + tolerance). Random (untrained) models: numbers are agreement checks, not accuracy claims."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))

from dlnn import contract  # noqa: E402
from dlnn.data import modrec  # noqa: E402
from dlnn.tiers import common, parasitics as P, virtual  # noqa: E402
from helpers_tdlnet_rf import random_model  # noqa: E402


@pytest.fixture(scope="module")
def ds():
    return modrec.dataset(n_train=2000, n_test=200)


@pytest.fixture(scope="module")
def model(ds):
    m = random_model(ds, seed=0)
    for ly in m.layers:                 # the board has fixed tap spacing (d = 1)
        ly.dilation = 1
        ly.vsat = float(ly.vsat) * 3.0
    rng = np.random.default_rng(0)
    m.readout_W = rng.standard_normal(m.readout_W.shape).astype(np.float32)
    return m


@pytest.fixture(scope="module")
def pc():
    return P.resolve({})


def test_params_loaded(pc):
    assert pc["_have_params"]
    assert abs(pc["edge_ns"] - 726.0) < 1.0                       # 1 kohm + 330 pF, 10-90 %
    assert abs(pc["cp_f"] - 19.38e-12) < 0.1e-12                  # from the 3.3 MHz SRF
    assert pc["n_sections"] == 56 and pc["tap_every"] == 8


def test_skrf_abcd_matches_numpy_and_params_dc_gain(pc):
    ctx = P.Context(pc, use_skrf=True)
    f = ctx.f[:200]
    hs = P.ladder_taps_skrf(f, pc)
    hn = P.ladder_taps_numpy(f, ctx._Lnom, ctx._Cnom, pc)[0]
    assert np.max(np.abs(hs - hn)) < 1e-9
    bp = P.load_board_params()
    assert np.allclose(np.abs(hn[:, 0]), bp["line"]["tap_dc_gain"] if "tap_dc_gain" in bp["line"] else np.abs(hn[:, 0]), atol=1e-4)
    # low-pass: loss grows with tap index at 250 kHz and is several dB at tap 7
    i = np.argmin(abs(ctx.f - 244140))
    g = 20 * np.log10(np.abs(P.ladder_taps_numpy(ctx.f, ctx._Lnom, ctx._Cnom, pc)[0][:, i]))
    assert g[-1] < g[0] and 0.5 < g[0] - g[-1] < 5.0


def test_crosstalk_k_formula(pc):
    K = P.k_coupling({**pc, "crosstalk_layout": "parallel", "crosstalk_pitch_mm": 3.0, "crosstalk_k": None})
    assert abs(K[0, 1] - 0.02716) < 1e-4                          # params.json crosstalk.per_pitch 3mm
    Ka = P.k_coupling({**pc, "crosstalk_layout": "alternating", "crosstalk_k": None})
    assert abs(Ka[0, 1] - 0.1 * K[0, 1]) < 1e-9 and Ka[0, 2] > Ka[0, 1]      # distance 2 is parallel again
    Kk = P.k_coupling({**pc, "crosstalk_k": 0.01})
    assert abs(Kk[0, 1] - 0.01) < 1e-12 and np.allclose(Kk, Kk.T)


def test_coupled_modal_solution(pc):
    ctx = P.Context({**pc, "crosstalk_k": 0.0136}, use_skrf=False)
    T = ctx.T_coupled()
    assert T.shape[:2] == (8, 8)
    assert np.allclose(T, np.swapaxes(T, 0, 1), atol=1e-9)        # reciprocal
    i = 5
    assert np.abs(T[0, 1, 3, i]) > 1e-4 and np.abs(T[0, 1, 3, i]) < np.abs(T[0, 0, 3, i])
    z = P.Context({**pc, "crosstalk_k": 0.0}, use_skrf=False)
    assert z.T_coupled() is None


def test_nominal_forward_tracks_v_and_is_deterministic(model, ds):
    x = ds["x_test"]
    sim = P.Simulator(model, x)
    a = sim.forward(x)["out"]
    b = sim.forward(x)["out"]
    assert np.array_equal(a, b)
    v = virtual.forward(model, x)["out"]
    assert a.shape == v.shape and np.isfinite(a).all()
    assert np.mean(a.argmax(1) == v.argmax(1)) > 0.5               # trimmed nominal board follows V on a random model
    # the board as designed (no 6 dB trim) is a different operating point
    c = sim.forward(x, {"gain_trim_db": 0.0})["out"]
    assert not np.allclose(a, c)


def test_tolerance_draw_changes_output_and_seed_is_reproducible(model, ds):
    x = ds["x_test"][:100]
    sim = P.Simulator(model, x)
    base = sim.forward(x)["out"]
    u1, u2 = P.draw_units(model, 1), P.draw_units(model, 2)
    o1 = sim.forward(x, {"apply_tol": True, "l_tol_pct": 15, "vbe_mismatch_mv": 6}, u1)["out"]
    o1b = sim.forward(x, {"apply_tol": True, "l_tol_pct": 15, "vbe_mismatch_mv": 6}, P.draw_units(model, 1))["out"]
    o2 = sim.forward(x, {"apply_tol": True, "l_tol_pct": 15, "vbe_mismatch_mv": 6}, u2)["out"]
    assert np.array_equal(o1, o1b) and not np.allclose(o1, o2) and not np.allclose(o1, base)


def test_fir_window_covers_response(model, ds):
    sim = P.Simulator(model, ds["x_test"][:20])
    pc1 = sim.pc({})
    pc2 = sim.pc({"lag_hi": 22, "lag_lo": -4})
    u = P.zero_units(model)
    p1 = P.build_plant(model, P.Context(pc1), pc1, u)
    p2 = P.build_plant(model, P.Context(pc2), pc2, u)
    off = p1.mlo - p2.mlo
    for h1, h2 in zip(p1.h, p2.h):
        inner = h2[:, :, off:off + h1.shape[2]]
        out_e = np.sum(h2 ** 2) - np.sum(inner ** 2)
        assert out_e / np.sum(h2 ** 2) < 0.01                      # < 1 % of the kernel energy lies outside the used lag window


def test_monte_carlo_stats(model, ds):
    x, y = ds["x_test"][:80], ds["y_test"][:80]
    sim = P.Simulator(model, x)
    v = virtual.forward(model, x)["out"].argmax(1)
    mc = P.monte_carlo(sim, x, y, v, 4, seed=3)
    s = mc["stats"]
    assert s["n_trials"] == 4 and 0 <= s["yield"] <= 1 and s["accuracy_p5"] <= s["accuracy_mean"] + 1e-9
    assert len(mc["accuracy"]) == 4


def test_tier_json_validates_and_sweeps_registered(model, ds, tmp_path):
    p = model.save(tmp_path / "m.npz")
    r = P.run(p, ds, {"mc_trials": 3, "mc_subset": 100, "p_subset": 100, "p_sweep_axes": ["stage_snr_db", "crosstalk_k", "l_tol_pct"],
                      "p_sweep_boards": 2, "run_id": "t"}, tmp_path)
    d = contract.validate(r)
    assert d["tier"] == "P" and d["status"] in ("ok", "fallback") and d["n_examples"] == len(ds["y_test"])
    assert {s["axis"] for s in d["sweeps"]} == {"stage_snr_db", "crosstalk_k", "l_tol_pct"}
    assert "mc.accuracy" in r.arrays and "sweep.crosstalk_k.top1_agreement_vs_virtual" in r.arrays
    contract.save(r, tmp_path)
    assert contract.load(tmp_path / "tiers" / "P.json").tier == "P"
    assert any("Finding F1" in w for w in r.warnings)


def test_grid_export_schema_and_hash(model, ds, tmp_path):
    x, y = ds["x_test"][:40], ds["y_test"][:40]
    v = virtual.forward(model, x)["out"]
    sim = P.Simulator(model, x)
    axes = [("l_tol_pct", [0, 5], "%"), ("stage_snr_db", [60, 20], "dB")]
    g = P.run_grid(sim, x, y, v, axes=axes)
    assert g["accuracy"].shape == (2, 2)
    g["v_accuracy"] = 0.0
    out = P.export_grid(tmp_path / "g.json", {"m": g}, {"run_id": "t"})
    d = json.loads((tmp_path / "g.json").read_text())
    body = {k: v_ for k, v_ in d.items() if k not in ("sha256", "sha256_of")}
    assert hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest() == d["sha256"]
    assert d["models"]["m"]["n_points"] == 4 and d["schema"] == "dlnn-sweeps-grid/1" and out["run_id"] == "t"


def test_m1a_line_loss_and_q_corners():
    """M1a facts: line loss (tap 7 vs tap 0) about 2.3 dB at 250 kHz and 6.7 dB at 500 kHz at Q = 10; Q = 30 is much lower loss."""
    def loss(Q, fq):
        pc_ = P.resolve({"q_inductor": Q})
        c = P.Context(pc_, use_skrf=False)
        H = P.ladder_taps_numpy(c.f, c._Lnom, c._Cnom, pc_)[0]
        i = np.argmin(abs(c.f - fq))
        return -20 * np.log10(abs(H[7, i] / H[0, i]))
    assert abs(loss(10, 250e3) - 2.3) < 0.3 and abs(loss(10, 500e3) - 6.7) < 0.4
    assert loss(30, 500e3) < loss(10, 500e3) - 2.0
    pc_ = P.resolve({})
    assert abs(1 / (2 * np.pi * pc_["rc_tau_s"]) - 482e3) < 2e3 and pc_["q_inductor"] == 10.0 and pc_["C_end_f"] == 60e-12
    assert pc_["dcr_ohm"] == 3.3 and abs(pc_["L_h"] - 120e-6) < 1e-12
    # V_sat (2 V_T) drift +0.33 %/K: z scales as 1/T
    assert abs((273.15 + 27) / (273.15 + 28) - 1 + 0.0033) < 3e-5
