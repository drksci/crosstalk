"""modrec pipeline: small and fast (200 test examples, no RTL, no ngspice).

The heavy parts (iverilog RTL cross-check, ngspice unit cell) are covered by tests/test_emulated_rtl.py and
tests/test_circuit.py; here C runs through its documented analytic fallback and is checked to be LABELLED as a fallback.
"""
import json
import os
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from dlnn import compare, contract, pipeline

ROOT = Path(__file__).resolve().parents[1]
W32 = ROOT / "models" / "tdlnet_modrec_w32_ternary.npz"
W8 = ROOT / "models" / "tdlnet_modrec_ternary.npz"
needs_model = pytest.mark.skipif(not W32.exists(), reason="trained model models/tdlnet_modrec_w32_ternary.npz not present")


def test_stratified_subset_is_balanced_nested_and_deterministic():
    from dlnn.data import modrec
    g = modrec.generate(400, seed=7)
    idx = pipeline.stratified_subset(g["y"], g["snr"], 80)
    assert len(idx) == 80 and (np.diff(idx) > 0).all()
    assert (idx == pipeline.stratified_subset(g["y"], g["snr"], 80)).all()
    cells = {(int(c), float(s)) for c, s in zip(g["y"][idx], g["snr"][idx])}
    assert len(cells) == 40                                  # every (class, SNR) cell present
    for c in range(8):
        assert (g["y"][idx] == c).sum() == 10
    assert (idx[:8] == np.arange(8)).all()                   # leading examples coincide with the full set


def test_subset_dataset_hash_differs_and_slices():
    from dlnn.data import modrec
    ds = modrec.dataset(n_train=40, n_test=80)
    idx = pipeline.stratified_subset(ds["y_test"], ds["snr_test"], 40)
    sub = pipeline.subset_dataset(ds, idx)
    assert sub["sha256"] != ds["sha256"] and sub["parent_sha256"] == ds["sha256"]
    assert len(sub["y_test"]) == 40 and (sub["x_test"] == ds["x_test"][idx]).all()


def test_missing_or_float_model_is_refused(tmp_path):
    with pytest.raises(pipeline.ModelUnavailable):
        pipeline.run_pipeline("modrec", out=tmp_path, model=tmp_path / "missing.npz")
    fl = ROOT / "models" / "tdlnet_modrec_w32_float.npz"
    if fl.exists():
        with pytest.raises(pipeline.ModelUnavailable, match="not a ternary"):
            pipeline.run_pipeline("modrec", out=tmp_path, model=fl)


@pytest.fixture(scope="module")
def modrec_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("modrec_out")
    run = pipeline.run_pipeline(
        "modrec", tiers="V,E,C,R", cascade=True, llm="none", db="sqlite", out=out, run_id="t-modrec", quiet=True,
        model=W32, n_test=200, cr_subset=80,
        tier_cfg={"rtl": False, "c_use_spice": False, "sweep_snrs": [0.0, 10.0, 30.0]})
    return run


@needs_model
def test_modrec_run_files_and_contract(modrec_run):
    run = modrec_run
    for f in ("compare.json", "model.npz", "dlnn.sqlite", "llm/commentary.json", "llm/card.txt", "timing.json",
              "reference.json", "subsets.json", "params/circuit_extracted.json", "figures/fig4_outputs_light.png"):
        assert (run / f).exists(), f
    for t in "VECR":
        contract.load(run / "tiers" / f"{t}.json")                     # validates
    cmp = compare.load(run)
    assert cmp["tiers"] == ["V", "E", "C", "R"]
    ti = cmp["tier_info"]
    assert ti["V"]["status"] == "ok" and ti["R"]["status"] == "ok"
    assert ti["E"]["status"] == "fallback"                            # rtl disabled in this test: bit-true numpy model only
    assert any("RTL" in s for s in ti["E"]["fidelity"]["omits"])
    assert ti["C"]["status"] == "fallback"                            # ngspice disabled: must be labelled, never "ok"
    assert any("analytic" in s.lower() or "ngspice" in s.lower() for s in ti["C"]["fidelity"]["omits"])
    assert ti["V"]["n_examples"] == 200 and ti["C"]["n_examples"] == 80 and ti["R"]["n_examples"] == 80


@needs_model
def test_modrec_cascade_provenance(modrec_run):
    cmp = compare.load(modrec_run)
    assert cmp["tier_info"]["R"]["inputs_from"] == ["params/circuit_extracted.json"]
    ex = json.loads((modrec_run / "params" / "circuit_extracted.json").read_text())
    assert ex["source"] == "analytic-numpy-rlgc"                      # provenance travels with the numbers
    assert any("cascade" in w for w in cmp["tier_info"]["R"]["warnings"])


@needs_model
def test_modrec_reference_is_not_a_tier_and_paired_deltas(modrec_run):
    cmp = compare.load(modrec_run)
    ref = cmp["reference"]
    assert ref["is_tier"] is False and "REFERENCE" in ref["label"] and ref["n_examples"] == 200
    assert "R" in cmp["paired_vs_virtual"] and "C" in cmp["paired_vs_virtual"]
    p = cmp["paired_vs_virtual"]["R"]
    assert p["n_examples"] == 80 and p["delta"]["accuracy"] == pytest.approx(p["tier"]["accuracy"] - p["V_on_subset"]["accuracy"])
    assert cmp["deltas_vs_virtual"]["accuracy"]["R"] == pytest.approx(p["delta"]["accuracy"])
    for t in "ECR":
        a = cmp["metric_matrix"]["top1_agreement_vs_virtual"][t]
        assert 0.0 <= a <= 1.0
    assert cmp["timing"]["total_s"] > 0 and set("VECR") <= set(cmp["timing"]["stages_s"])
    assert cmp["knees"]


@needs_model
def test_modrec_subset_identical_across_c_and_r(modrec_run):
    sub = json.loads((modrec_run / "subsets.json").read_text())
    assert sub["tiers"]["C"]["indices"] == sub["tiers"]["R"]["indices"] and len(sub["tiers"]["C"]["indices"]) == 80
    r_pred = np.load(modrec_run / "tiers" / "R.npz")["pred"]
    c_pred = np.load(modrec_run / "tiers" / "C.npz")["pred"]
    assert r_pred.shape == c_pred.shape == (80,)


@needs_model
def test_modrec_sqlite(modrec_run):
    con = sqlite3.connect(modrec_run / "dlnn.sqlite")
    q = lambda s: con.execute(s).fetchone()[0]   # noqa: E731
    assert q("select count(*) from runs") == 1
    assert q("select count(*) from tier_results") == 4
    assert q("select count(*) from sweeps") > 20
    assert q("select count(*) from predictions where tier='V'") == 200
    assert q("select count(*) from predictions where tier='R'") == 80
    assert q("select count(*) from predictions where tier='R' and label is not null") == 80
    assert q("select count(*) from llm_outputs") == 1
    assert q("select source from llm_outputs") == "template"
    labels = np.load(modrec_run / "labels.npy")
    ids = [r[0] for r in con.execute("select example_id from predictions where tier='R' order by example_id")]
    assert ids == json.loads((modrec_run / "subsets.json").read_text())["tiers"]["R"]["indices"]
    lab = [r[0] for r in con.execute("select label from predictions where tier='R' order by example_id")]
    assert lab == [int(labels[i]) for i in ids]


@needs_model
def test_llm_backend_unavailable_fails_before_running(tmp_path, monkeypatch):
    from dlnn.llm import backend
    monkeypatch.setenv("DLNN_QWEN_GGUF", str(tmp_path / "nope.gguf"))
    with pytest.raises(backend.BackendUnavailable):
        pipeline.run_pipeline("modrec", llm="qwen", out=tmp_path, model=W32, n_test=40, quiet=True)
    assert not list(tmp_path.iterdir())


PG_DSN = os.environ.get("DLNN_PG_TEST_DSN")


@needs_model
@pytest.mark.skipif(not PG_DSN, reason="no PostgreSQL test server: set DLNN_PG_TEST_DSN (e.g. 'dbname=dlnn host=/var/run/postgresql')")
def test_postgres_load_roundtrip(modrec_run):
    from dlnn.db import load
    rid = load.load_run_pg(modrec_run, PG_DSN, {"test": True})
    s = load.pg_summary(rid, PG_DSN)
    assert s["runs"] == 1 and s["tier_results"] == 4 and s["predictions"] == 200 * 2 + 80 * 2 and s["llm_outputs"] == 1
    assert [t[0] for t in s["tiers"]] == ["C", "E", "R", "V"]


def test_postgres_without_dsn_is_a_clear_error(monkeypatch):
    from dlnn.db import load
    monkeypatch.delenv(load.PG_DSN_ENV, raising=False)
    with pytest.raises(load.DbUnavailable, match="pg-dsn"):
        load.connect_pg(None)
