"""Merge tier JSONs into compare.json (single input for figures, docs tables and the LLM card)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import contract

KNEE_RULES = {"accuracy": ("abs_drop", 0.05, "first sweep value where accuracy is >= 0.05 below its best-end value"),
              "nmse": ("rel_rise", 0.10, "first sweep value where NMSE is >= 10 % above its best-end value")}


def _knee(axis: str, values: list[float], series: list[float], metric: str) -> float | None:
    worse_high = contract.AXIS_HIGHER_IS_WORSE[axis]
    order = sorted(range(len(values)), key=lambda i: values[i], reverse=not worse_high)  # best -> worst
    ref = series[order[0]]
    kind, thr, _ = KNEE_RULES[metric]
    for i in order[1:]:
        bad = (series[i] <= ref - thr) if kind == "abs_drop" else (series[i] >= ref * (1 + thr))
        if bad:
            return float(values[i])
    return None


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text()) if p.exists() else None


def paired_blocks(run_dir: Path, results: dict[str, "contract.TierResult"]) -> dict[str, Any]:
    """For tiers that ran on a subset of the test set (``subsets.json``): V and the tier scored on the SAME examples.

    ``V_on_subset`` is V's own prediction array restricted to the tier's example indices, so ``delta`` (tier - V) is a
    like-for-like comparison, unlike ``metric_matrix`` where V's accuracy comes from the full test set.
    """
    sub = _read_json(run_dir / "subsets.json")
    if not sub or not sub.get("tiers") or not (run_dir / "labels.npy").exists() or not (run_dir / "tiers" / "V.npz").exists():
        return {}
    from .tiers.common import macro_f1
    y_all = np.load(run_dir / "labels.npy")
    snr_all = np.load(run_dir / "snr_test.npy") if (run_dir / "snr_test.npy").exists() else None
    pv_all = np.load(run_dir / "tiers" / "V.npz")["pred"]
    out: dict[str, Any] = {}
    for t, info in sub["tiers"].items():
        r = results.get(t)
        if r is None or r.status == "skipped" or "accuracy" not in r.metrics:
            continue
        idx = np.asarray(info["indices"], dtype=int)
        y, pv = y_all[idx], pv_all[idx]
        v: dict[str, Any] = {"accuracy": float(np.mean(pv == y)), "macro_f1": macro_f1(y, pv, int(y_all.max()) + 1)}
        if snr_all is not None:
            sn = snr_all[idx]
            v["acc_per_snr"] = {str(int(s)): float(np.mean(pv[sn == s] == y[sn == s])) for s in np.unique(sn)}
        tm = {k: r.metrics[k] for k in ("accuracy", "macro_f1", "acc_per_snr") if k in r.metrics}
        delta = {k: ({q: tm[k][q] - v[k][q] for q in tm[k]} if isinstance(tm[k], dict) else tm[k] - v[k])
                 for k in tm if k in v}
        out[t] = {"n_examples": int(len(idx)), "n_examples_full_test_set": int(sub.get("n_full", len(y_all))),
                  "indices_sha256": info.get("sha256"), "policy": sub.get("policy"),
                  "V_on_subset": v, "tier": tm, "delta": delta}
    return out


def build(run_dir: str | Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    results = {}
    for t in contract.TIERS:
        p = run_dir / "tiers" / f"{t}.json"
        if p.exists():
            results[t] = contract.load(p, with_arrays=False)
    if "V" not in results:
        raise contract.ContractError("virtual tier V is required to build compare.json")
    V = results["V"]
    cmp: dict[str, Any] = {
        "schema_version": contract.COMPARE_SCHEMA_VERSION, "run_id": V.run_id, "task": V.task,
        "model_sha256": V.model_sha256, "dataset_sha256": V.dataset_sha256, "tiers": list(results),
        "tier_info": {t: {"status": r.status, "tool": r.tool, "fidelity": r.fidelity, "runtime_s": r.runtime_s,
                          "warnings": r.warnings, "inputs_from": r.inputs_from, "n_examples": r.n_examples}
                      for t, r in results.items()},
        "metric_matrix": {}, "deltas_vs_virtual": {}, "sweeps": {}, "knees": []}
    for mid in contract.METRIC_IDS:
        row = {t: r.metrics[mid] for t, r in results.items() if mid in r.metrics}
        if not row:
            continue
        cmp["metric_matrix"][mid] = row
        if "V" in row and row["V"] is not None:
            d: dict[str, Any] = {}
            for t, v in row.items():
                if t == "V" or v is None:
                    continue
                d[t] = ({k: v[k] - row["V"][k] for k in v} if isinstance(v, dict) else v - row["V"])
            cmp["deltas_vs_virtual"][mid] = d
    for t, r in results.items():
        for sw in r.sweeps:
            cmp["sweeps"].setdefault(sw["axis"], {})[t] = {k: v for k, v in sw.items() if k != "axis"}
            for metric in ("accuracy", "nmse"):
                if metric in sw:
                    cmp["knees"].append({"axis": sw["axis"], "tier": t, "metric": metric,
                                         "knee_value": _knee(sw["axis"], sw["values"], sw[metric], metric),
                                         "rule": KNEE_RULES[metric][2]})
    paired = paired_blocks(run_dir, results)
    if paired:
        cmp["paired_vs_virtual"] = paired
        for t, blk in paired.items():     # like-for-like deltas for tiers that ran on a subset
            for mid, d in blk["delta"].items():
                cmp["deltas_vs_virtual"].setdefault(mid, {})[t] = d
    for key, fname in (("reference", "reference.json"), ("timing", "timing.json"), ("run_meta", "run_meta.json")):
        blk = _read_json(run_dir / fname)
        if blk is not None:
            cmp[key] = blk
    contract.validate_compare(cmp)
    return cmp


def write(run_dir: str | Path) -> Path:
    run_dir = Path(run_dir)
    p = run_dir / "compare.json"
    p.write_text(json.dumps(contract._jsonable(build(run_dir)), indent=2))
    return p


def load(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    p = p / "compare.json" if p.is_dir() else p
    return contract.validate_compare(json.loads(p.read_text()))
