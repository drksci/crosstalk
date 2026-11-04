import json

import numpy as np
import pytest

from dlnn import contract


def make(status="ok", **kw):
    d = dict(tier="V", status=status, run_id="r1", task="narma10", tool={"name": "t", "version": "1"},
             model_sha256="a", dataset_sha256="b", n_examples=3, metrics={"nmse": 0.1},
             fidelity={"models": ["m"], "omits": ["o"]}, arrays={"y_pred": np.zeros((1, 3), np.float32)})
    d.update(kw)
    return contract.TierResult(**d)


def test_roundtrip(tmp_path):
    r = make(sweeps=[{"axis": "snr_db", "values": [1.0, 2.0], "nmse": [0.2, 0.1]}])
    p = contract.save(r, tmp_path)
    assert p == tmp_path / "tiers" / "V.json" and (tmp_path / "tiers" / "V.npz").exists()
    j = json.loads(p.read_text())
    assert j["arrays"]["y_pred"]["shape"] == [1, 3]
    r2 = contract.load(p)
    assert r2.metrics == r.metrics and np.array_equal(r2.arrays["y_pred"], r.arrays["y_pred"])


def test_rejects_bad():
    with pytest.raises(contract.ContractError):
        contract.validate(make(status="bogus"))
    with pytest.raises(contract.ContractError):
        contract.validate(make(metrics={"not_a_metric": 1.0}))
    with pytest.raises(contract.ContractError):
        contract.validate(make(tier="X"))
    with pytest.raises(contract.ContractError):  # stub must say so
        contract.validate(make(status="stub", fidelity={"models": [], "omits": ["nothing"]}))
    with pytest.raises(contract.ContractError):  # sweep length mismatch
        contract.validate(make(sweeps=[{"axis": "snr_db", "values": [1.0], "nmse": [1, 2]}]))
    d = make().to_json_dict(); d["extra"] = 1
    with pytest.raises(contract.ContractError):
        contract.validate(d)


def test_stub_and_skipped_ok():
    contract.validate(make("stub", fidelity={"models": ["V+noise"], "omits": ["STUB placeholder, not SPICE"]}))
    contract.validate(make("skipped", metrics={}, warnings=["ngspice missing"]))
    with pytest.raises(contract.ContractError):
        contract.validate(make("skipped", metrics={}))


def test_nan_becomes_null(tmp_path):
    p = contract.save(make(metrics={"nmse": float("nan")}), tmp_path)
    assert json.loads(p.read_text())["metrics"]["nmse"] is None


def test_metric_ids_stable():
    assert {"accuracy", "nmse", "top1_agreement_vs_virtual", "mean_kl_virtual_vs_tier",
            "logit_nmse_vs_virtual", "macro_f1", "acc_per_snr"} <= set(contract.METRIC_IDS)
