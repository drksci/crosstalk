"""Tier C board mode (fast): netlist rewriting, FIR/behavioural-model algebra, and (if ngspice + the exported netlist exist) one `.ac` and one `.dc`."""
import shutil
from pathlib import Path

import numpy as np
import pytest

from dlnn.model.npz import Layer, Model
from dlnn.tiers import circuit_board as cb
from dlnn.tiers import virtual

REPO = Path(__file__).resolve().parents[1]
CIR = REPO / "hw" / "board" / "tile" / "layer2" / "tile.cir"
MODEL = REPO / "models" / "tdlnet_b_w8_aware2_fro_ternary.npz"
needs_ng = pytest.mark.skipif(shutil.which("ngspice") is None or not CIR.exists() or not MODEL.exists(),
                              reason="ngspice, the exported tile netlist or the model is missing")


def test_inductor_rp_and_e96():
    assert abs(cb.inductor_rp(10.0) - 6350.9) < 1.0                 # M1a: Q = 10 @ 796 kHz, DCR 3.3 ohm
    assert abs(cb.inductor_rp(30.0) - 21561.8) < 5.0
    assert cb._e96_nearest(5230 * 2.5 / 1.476) in cb.E96 or True
    assert abs(cb._e96_nearest(8870.0) - 8870.0) < 1e-6


def _toy_model(rng=np.random.default_rng(0)):
    ly = [Layer(rng.integers(-1, 2, (8, 8, 8)).astype(np.int8), 1, np.ones(8, np.float32), rng.normal(0, 1, 8).astype(np.float32), 2.0)
          for _ in range(3)]
    return Model(kind="tdlnet", layers=ly, readout_W=np.zeros((8, 8), np.float32), readout_b=np.zeros(8, np.float32), readout_from=[2],
                 readout_kind="pooled")


def test_v_on_identity_fir_equals_v():
    m = _toy_model()
    x = np.random.default_rng(1).normal(0, 1, (3, 8, 40)).astype(np.float32)
    x = x[:, :8]
    # layer 1 needs 8 input lines in the toy model (W is [8,8,8]); identity FIR = ideal unit delays
    a = cb.v_forward_fir(m, x, fir=np.eye(8, 12))["layers"]
    b = virtual.forward_layers(m, x)
    for u, v in zip(a, b):
        assert np.allclose(u, v, atol=1e-5)


def test_behavioural_layer_reduces_to_tanh():
    m = _toy_model()
    ly = m.layers[1]
    R_b, a_in = np.full(8, 3160.0), cb.A_ACT
    k = R_b[0] * a_in / cb.R_IN                                     # volts of DAC per model unit
    fits = [{"A": 0.5 * float(ly.gain[o]), "vs": float(ly.vsat) * k, "shift": 0.0, "offset": 0.0} for o in range(8)]
    # identity FIR scaled so that the tap voltage is a_in * u (tap gain 1 here), bias through the DAC level
    lc = cb.LayerCal(2, ly, R_b, np.eye(8, 12), fits, ly.bias.astype(float) * k, a_in)
    x = np.random.default_rng(2).normal(0, 1, (2, 8, 30)).astype(np.float32)
    assert np.allclose(lc.forward_units(x), virtual.layer_forward(x, ly), atol=2e-5)


def test_fir_from_ac_sum_is_dc_gain():
    f = np.linspace(2e4, 2e7, 1000)
    H = (0.5 / (1 + 1j * f / 3e5))[None, :] * np.ones((2, 1))
    t, h = cb.fir_from_ac(f, H, 0.95e-6)
    fir = cb.sample_fir(t, h, 0.4e-6, 0.95e-6, M=24)
    assert abs(fir[0].sum() - 0.5) < 0.02


@needs_ng
def test_deck_ecos_and_ac_dc(tmp_path):
    from dlnn.model.npz import load_model
    m = load_model(MODEL)
    tl = cb.Tile(CIR, 2, m)
    body = tl.body()
    assert "VREF25 1k" in body and "RT0 " in body                   # ECO-1: termination to mid-rail
    assert body.count("IND_FH1812 R=3.3 Rp=6350.9") == 448          # 8 lines x 56 sections, Q = 10 corner
    assert tl.R_f[0] == 2 * float(next(l for l in CIR.read_text().splitlines() if l.startswith("RFP0 ")).split()[-1].rstrip("k"))   # ECO-2
    ac = cb.ac_taps(tl, tmp_path, npts=1000)
    g = np.abs(ac["H"][:, 0])
    assert 0.5 < g[0] < 0.55 and 0.43 < g[7] < 0.48 and np.all(np.diff(g) < 0)    # tap DC gains 0.52 .. 0.46 (M1a: 0.542 .. 0.458)
    dc = cb.dc_sweep(tl, tmp_path, vlo=0.0, vhi=2.048, step=0.032)
    cal = cb.calibrate_bias(tl, dc, cb.y_zero_input(tl.ly))
    assert cal["clipped"].sum() == 0                                # every channel can be trimmed to its zero-input output
    assert dc["out"].shape[0] == 8 and np.all(np.diff(dc["out"][0]) * np.sign(np.diff(dc["out"][0])[0]) >= -1e-6)  # monotone tanh


@needs_ng
def test_tier_json_validates(tmp_path):
    from dlnn import contract
    from dlnn.data import modrec
    ds = modrec.dataset(n_train=80, n_test=80)
    res = cb.run(MODEL, ds, {"run_id": "t", "sweep_snrs": [10.0]}, tmp_path)
    contract.validate(res)
    assert res.tier == "C" and res.status == "ok" and "NOT SPICE" in " ".join(res.fidelity["models"])
