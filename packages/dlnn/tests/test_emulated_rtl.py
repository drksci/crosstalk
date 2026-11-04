"""Tier E: numpy fixed-point golden model vs Verilog RTL (iverilog).  RTL tests skip without iverilog."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from helpers_tdlnet import make_random_tdlnet  # noqa: E402

from dlnn import contract  # noqa: E402
from dlnn.data import modrec  # noqa: E402
from dlnn.model.npz import Layer  # noqa: E402
from dlnn.tiers import emulated  # noqa: E402

HAVE_IV = emulated.rtl_info()["available"]
needs_iverilog = pytest.mark.skipif(not HAVE_IV, reason="iverilog/vvp not installed: RTL tests skipped "
                                                        "(numpy golden-model tests still run)")
TRAINED = Path(__file__).resolve().parents[1] / "models" / "tdlnet_modrec_ternary.npz"


@pytest.fixture(scope="module")
def ds():
    return modrec.dataset(n_train=2400, n_test=800)


@pytest.fixture(scope="module")
def rand_model(ds):
    return make_random_tdlnet(ds["x_train"], ds["y_train"], seed=3)


def _rand_layer(rng, O, I, K, d, gain, bias_scale=1.0, vsat=1.0):
    return Layer(W=rng.integers(-1, 2, (O, I, K)).astype(np.int8), dilation=d,
                 gain=np.asarray(gain, np.float32), bias=(bias_scale * rng.standard_normal(O)).astype(np.float32),
                 vsat=vsat)


# ---------------------------------------------------------------------------------- golden model only
def test_golden_matches_naive_loops():
    rng = np.random.default_rng(1)
    ly = _rand_layer(rng, 3, 2, 4, 3, [1.0, -0.5, 0.25])
    xq = rng.integers(-127, 128, (2, 2, 20)).astype(np.int32)
    s_in = 0.02
    y, s_out = emulated.layer_fixed(xq, s_in, ly)
    pr = emulated.layer_params(ly, s_in)
    for b in range(2):
        for o in range(3):
            for n in range(20):
                acc = int(pr["bq"][o]) + sum(int(ly.W[o, i, k]) * int(xq[b, i, n - k * 3])
                                             for i in range(2) for k in range(4) if n - k * 3 >= 0)
                z = min(127, max(-128, (acc * pr["M"] + (1 << 14)) >> 15))
                v = int(emulated.LUT[z + 128])
                assert y[b, o, n] == (v * int(pr["R"][o]) + (1 << 14)) >> 15
    assert s_out == pytest.approx(1.0 / 127)
    assert pr["R"][1] < 0 and np.any(y[:, 1] != 0)   # negative gain keeps its sign


def test_fallback_without_rtl(monkeypatch, rand_model, ds, tmp_path):
    monkeypatch.setattr(emulated, "rtl_info", lambda: {"available": False, "version": "", "reason": "no iverilog"})
    mp = rand_model.save(tmp_path / "m.npz")
    small = {**ds, "x_test": ds["x_test"][:64], "y_test": ds["y_test"][:64], "snr_test": ds["snr_test"][:64]}
    r = emulated.run(mp, small, {"run_id": "t", "sweep_snrs": [10.0]}, tmp_path)
    contract.validate(r)
    assert r.status == "fallback"
    assert any("did NOT run" in s for s in r.fidelity["omits"])
    assert "no iverilog" in " ".join(r.warnings)


# ---------------------------------------------------------------------------------- RTL vs golden
@needs_iverilog
@pytest.mark.parametrize("O,I,K,d,gain,bias_scale,vsat,in_w", [
    (8, 2, 8, 1, [1.0] * 8, 1.0, 1.0, 8),                                  # TDL-Net-S layer 1 geometry
    (5, 3, 4, 3, [1.0, 0.5, -1.0, 0.8, 0.3], 1.0, 0.7, 8),                 # odd sizes, mixed + negative gains
    (4, 4, 3, 5, [2.0] * 4, 30.0, 0.05, 8),                                # heavy saturation, huge bias/M
    (3, 2, 6, 2, [1.0, 1.0, 1.0], 0.0, 3.0, 5),                            # narrow (5-bit) input codes
])
def test_rtl_layer_bit_exact_random(tmp_path, O, I, K, d, gain, bias_scale, vsat, in_w):
    rng = np.random.default_rng(O * 100 + K)
    ly = _rand_layer(rng, O, I, K, d, gain, bias_scale, vsat)
    lim = 2 ** (in_w - 1) - 1
    xq = rng.integers(-lim, lim + 1, (6, I, 40)).astype(np.int32)
    xq[0] = lim                       # rails: DC at +full scale
    xq[1] = -lim
    s_in = 0.01
    pr = emulated.layer_params(ly, s_in)
    g, _ = emulated.layer_fixed(xq, s_in, ly)
    y, pool = emulated.run_layer_rtl(xq, ly, pr, in_w, tmp_path)
    assert y.shape == g.shape and np.array_equal(y, g)
    assert np.array_equal(pool, g.sum(axis=2))
    assert np.abs(g).max() > 0


@needs_iverilog
def test_rtl_comparison_is_not_vacuous(tmp_path):
    """Perturbing the golden model (one flipped tap) must show up as mismatches."""
    rng = np.random.default_rng(5)
    ly = _rand_layer(rng, 4, 2, 4, 2, [1.0] * 4)
    xq = rng.integers(-127, 128, (4, 2, 32)).astype(np.int32)
    pr = emulated.layer_params(ly, 0.02)
    y, _ = emulated.run_layer_rtl(xq, ly, pr, 8, tmp_path)
    W2 = ly.W.copy()
    W2[0, 0, 0] = 1 if W2[0, 0, 0] != 1 else -1
    g, _ = emulated.layer_fixed(xq, 0.02, Layer(W2, 2, ly.gain, ly.bias, ly.vsat))
    assert np.sum(y != g) > 0


@needs_iverilog
def test_rtl_chain_random_tdlnet(tmp_path, rand_model, ds):
    n = 8
    full = emulated.forward_int(rand_model, ds["x_test"], 8)
    st = emulated.verify_chain(rand_model, full["xq"][:n], full["s0"], tmp_path)
    assert st["mismatches"] == 0 and st["values_compared"] == 3 * (n * 8 * 128 + n * 8)
    gold = emulated.readout_from_int(rand_model, [q[:n] for q in full["yq"]], full["s"])
    rtl = emulated.readout_from_int(rand_model, st["rtl_yq"], st["scales"], st["rtl_pool"])
    assert np.array_equal(gold, rtl)


@needs_iverilog
def test_run_tier_with_rtl_and_adc_sweep(tmp_path, rand_model, ds):
    mp = rand_model.save(tmp_path / "m.npz")
    small = {**ds, "x_test": ds["x_test"][:300], "y_test": ds["y_test"][:300], "snr_test": ds["snr_test"][:300]}
    r = emulated.run(mp, small, {"run_id": "t", "rtl_examples": 8, "sweep_snrs": [0.0, 10.0, 20.0]}, tmp_path)
    contract.validate(r)
    contract.save(r, tmp_path)
    assert r.status == "ok" and r.tool["name"] == "numpy-fixedpoint"
    txt = " ".join(r.fidelity["models"])
    assert "Verilog RTL" in txt and "0 mismatches" in txt and "first 8 test examples" in txt
    assert any("not simulated on the full test set" in s for s in r.fidelity["omits"])
    ax = {s["axis"]: s for s in r.sweeps}
    assert ax["adc_bits"]["values"] == [4, 5, 6, 7, 8, 9, 10]
    assert len(ax["adc_bits"]["accuracy"]) == len(ax["adc_bits"]["mean_kl"]) == 7
    assert ax["adc_bits"]["mean_kl"][0] > ax["adc_bits"]["mean_kl"][-1]       # 4 bits is worse than 10
    side = json.loads((tmp_path / "tiers" / "E_rtl.json").read_text())
    assert side["ran"] and side["mismatches"] == 0 and side["adc_mismatches"] == 0
    assert side["examples"] == 8 and side["readout_identical"] is True


@needs_iverilog
@pytest.mark.skipif(not TRAINED.exists(), reason="trained model (M1) not present")
def test_trained_model_rtl(tmp_path, ds):  # pragma: no cover - only when M1 has delivered
    from dlnn.model.npz import load_model
    m = load_model(TRAINED)
    r = emulated.rtl_crosscheck(m, ds["x_test"], {"rtl_examples": 16})
    assert r["ran"] and r["mismatches"] == 0
