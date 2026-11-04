import json

import numpy as np
import pytest

from dlnn import compare, contract, pipeline
from dlnn.llm import backend, card
from dlnn.model import npz
from dlnn.tiers import circuit, emulated, rf, virtual


def test_reservoir_nmse_and_model_roundtrip(narma_model_npz, narma_ds):
    m = npz.load_model(narma_model_npz)
    assert all(l.W.dtype == np.int8 and set(np.unique(l.W)) <= {-1, 0, 1} for l in m.layers)
    out = virtual.forward(m, narma_ds["x_test"])["out"][0, 0, 100:]
    yt = narma_ds["y_test"][0, 0, 100:]
    assert np.mean((out - yt) ** 2) / np.var(yt) <= 0.40
    assert npz.load_model(narma_model_npz).sha256() == m.sha256()


def test_virtual_layer_matches_naive():
    rng = np.random.default_rng(0)
    ly = npz.Layer(W=rng.integers(-1, 2, (3, 2, 4)).astype(np.int8), dilation=3,
                   gain=np.array([1, 2, .5], np.float32), bias=np.zeros(3, np.float32), vsat=1.7)
    x = rng.standard_normal((2, 2, 30)).astype(np.float32)
    y = virtual.layer_forward(x, ly)
    for b in range(2):
        for o in range(3):
            for n in range(30):
                u = sum(ly.W[o, i, k] * x[b, i, n - k * 3] for i in range(2) for k in range(4) if n - k * 3 >= 0)
                assert np.isclose(y[b, o, n], ly.gain[o] * np.tanh(u / 1.7), atol=1e-5)


def test_emulated_lut_and_close_to_virtual(narma_model_npz, narma_ds):
    assert emulated.LUT.dtype == np.int8 and len(emulated.LUT) == 256 and emulated.LUT[128] == 0
    assert emulated.LUT[-1] == 127 and emulated.LUT[0] == -127
    m = npz.load_model(narma_model_npz)
    v = virtual.forward(m, narma_ds["x_test"])["out"]
    e = emulated.forward(m, narma_ds["x_test"])["out"]
    assert np.sqrt(np.mean((v - e) ** 2) / np.var(v)) < 0.2


def test_all_tiers_valid_and_compare(narma_model_npz, narma_ds, tmp_path):
    """V and E on NARMA10 through the contract + compare + card + NullBackend. C and R are real tiers since
    M2b/M2c (no stubs left); they have their own tests (test_circuit.py, test_rf.py) and run in the pipeline tests."""
    cfg = {"run_id": "t", "seed": 0}
    res = {}
    for t, mod in (("V", virtual), ("E", emulated)):
        r = mod.run(narma_model_npz, narma_ds, cfg, tmp_path)
        contract.validate(r)
        contract.save(r, tmp_path)
        res[t] = r
    assert [res[t].status for t in "VE"] == ["ok", "ok"]
    nv, ne = res["V"].metrics["nmse"], res["E"].metrics["nmse"]
    assert nv <= 0.40 and abs(ne - nv) / nv <= 0.10
    axes = {s["axis"] for s in res["E"].sweeps}
    assert axes == {"snr_db", "adc_bits"}
    p = compare.write(tmp_path)
    cmp = compare.load(p)
    assert cmp["metric_matrix"]["nmse"]["V"] == nv and cmp["deltas_vs_virtual"]["nmse"]["E"] == pytest.approx(ne - nv)
    c = card.render_card(cmp)
    assert card.approx_tokens(c) <= 1500
    com = backend.NullBackend().comment(cmp, c)
    assert com["source"] == "template" and contract.validate_commentary(com)


def test_c_and_r_are_not_stubs_by_contract():
    """The shared stub helper still exists for tests of the contract, but neither real tier module may call it."""
    import inspect
    for mod in (circuit, rf):
        assert "stub_run" not in inspect.getsource(mod)


def test_pipeline_end_to_end(tmp_path):
    run = pipeline.run_pipeline("narma10", tiers="V,E", out=tmp_path, run_id="e2e", quiet=True)
    for f in ("compare.json", "model.npz", "dlnn.sqlite", "llm/commentary.json", "timing.json",
              "figures/fig4_outputs_light.png", "figures/fig5_sweeps_dark.svg"):
        assert (run / f).exists(), f
    import sqlite3
    con = sqlite3.connect(run / "dlnn.sqlite")
    assert con.execute("select count(*) from tier_results").fetchone()[0] == 2
    assert con.execute("select status from tier_results where tier='E'").fetchone()[0] == "ok"
    assert con.execute("select count(*) from sweeps").fetchone()[0] > 0
    assert con.execute("select count(*) from predictions").fetchone()[0] == 2 * 2000


def test_missing_modrec_model_fails_clearly(tmp_path):
    with pytest.raises(pipeline.ModelUnavailable, match="not found"):
        pipeline.run_pipeline("modrec", out=tmp_path, model=tmp_path / "nope.npz")
    assert not list(tmp_path.iterdir())     # nothing written before the failure
