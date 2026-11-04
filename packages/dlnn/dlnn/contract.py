"""Tier output contract (plan section 3). Stable interface shared by all tiers.

On disk, for run `out/<run_id>/`:
    tiers/<tier>.json   canonical record (this module's schema)
    tiers/<tier>.npz    arrays; the JSON's ``arrays`` index lists every key
See docs/contract.md for a field-by-field description.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = "dlnn-tier/1"
COMPARE_SCHEMA_VERSION = "dlnn-compare/1"
COMMENTARY_SCHEMA_VERSION = "dlnn-commentary/1"

TIERS = ("V", "E", "C", "R", "P")
STATUSES = ("ok", "skipped", "stub", "fallback")
SWEEP_AXES = ("snr_db", "adc_bits", "loss_db_per_us", "tap_amp_err_db", "isolation_db",
              # tier P (parasitics and tolerance)
              "l_tol_pct", "c_tol_pct", "clock_err_pct", "stage_snr_db", "edge_ns", "crosstalk_k", "vbe_mismatch_mv",
              "q_inductor", "temperature_c", "crosstalk_pitch_mm", "gain_trim_db")
# For each axis: does a LARGER value make things worse? (used for knee detection)
AXIS_HIGHER_IS_WORSE = {"snr_db": False, "adc_bits": False, "loss_db_per_us": True,
                        "tap_amp_err_db": True, "isolation_db": False,
                        "l_tol_pct": True, "c_tol_pct": True, "clock_err_pct": True, "stage_snr_db": False, "edge_ns": True,
                        "crosstalk_k": True, "vbe_mismatch_mv": True, "q_inductor": False, "temperature_c": True,
                        "crosstalk_pitch_mm": False, "gain_trim_db": False}

# Metric ids (the only keys allowed in TierResult.metrics, and the only ids the LLM may cite).
METRIC_IDS = (
    "accuracy",                 # classification, float in [0,1]
    "macro_f1",                 # classification
    "acc_per_snr",              # classification, {"0": acc, "5": acc, ...}
    "nmse",                     # regression, MSE/var(target)
    "top1_agreement_vs_virtual",  # classification: fraction of examples with same argmax as V
    "mean_kl_virtual_vs_tier",  # classification: mean KL(p_V || p_tier), nats
    "logit_nmse_vs_virtual",    # classification: ||logit_tier - logit_V||^2 / ||logit_V||^2
    "nmse_vs_virtual",          # regression: mean (y_tier - y_V)^2 / var(y_V)
)

# Standard array names inside <tier>.npz (all optional; see docs/contract.md)
ARRAY_NAMES = ("logits", "probs", "pred", "y_pred", "y_true", "trace_activations.", "waveforms.")


class ContractError(ValueError):
    """Raised by validate() with every violation listed in the message."""


_NUM = {"type": ["number", "null"]}
_NUMLIST = {"type": "array", "items": _NUM}

TIER_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DLNN tier result",
    "type": "object",
    "required": ["schema_version", "run_id", "tier", "status", "task", "tool", "model_sha256",
                 "dataset_sha256", "inputs_from", "n_examples", "arrays", "metrics", "sweeps",
                 "fidelity", "runtime_s", "warnings"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": SCHEMA_VERSION},
        "run_id": {"type": "string", "minLength": 1},
        "tier": {"enum": list(TIERS)},
        "status": {"enum": list(STATUSES)},
        "task": {"enum": ["narma10", "modrec"]},
        "tool": {"type": "object", "required": ["name", "version"], "additionalProperties": False,
                 "properties": {"name": {"type": "string"}, "version": {"type": "string"}}},
        "model_sha256": {"type": "string"},
        "dataset_sha256": {"type": "string"},
        "inputs_from": {"type": "array", "items": {"type": "string"}},
        "n_examples": {"type": "integer", "minimum": 0},
        "arrays": {
            "type": "object",
            "additionalProperties": {
                "type": "object", "required": ["file", "key", "shape", "dtype"],
                "additionalProperties": False,
                "properties": {"file": {"type": "string"}, "key": {"type": "string"},
                               "shape": {"type": "array", "items": {"type": "integer"}},
                               "dtype": {"type": "string"}}}},
        "metrics": {
            "type": "object",
            "propertyNames": {"enum": list(METRIC_IDS)},
            "additionalProperties": {"oneOf": [
                _NUM, _NUMLIST, {"type": "object", "additionalProperties": _NUM}]}},
        "sweeps": {
            "type": "array",
            "items": {
                "type": "object", "required": ["axis", "values"], "additionalProperties": False,
                "properties": {"axis": {"enum": list(SWEEP_AXES)},
                               "values": {"type": "array", "items": {"type": "number"}},
                               "accuracy": _NUMLIST, "mean_kl": _NUMLIST, "nmse": _NUMLIST}}},
        "fidelity": {
            "type": "object", "required": ["models", "omits"], "additionalProperties": False,
            "properties": {"models": {"type": "array", "items": {"type": "string"}},
                           "omits": {"type": "array", "items": {"type": "string"}}}},
        "runtime_s": {"type": "number", "minimum": 0},
        "warnings": {"type": "array", "items": {"type": "string"}},
    },
}

COMPARE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DLNN compare.json",
    "type": "object",
    "required": ["schema_version", "run_id", "task", "model_sha256", "dataset_sha256", "tiers",
                 "tier_info", "metric_matrix", "deltas_vs_virtual", "sweeps", "knees"],
    "additionalProperties": False,
    "properties": {
        "schema_version": {"const": COMPARE_SCHEMA_VERSION},
        "run_id": {"type": "string"},
        "task": {"enum": ["narma10", "modrec"]},
        "model_sha256": {"type": "string"},
        "dataset_sha256": {"type": "string"},
        "tiers": {"type": "array", "items": {"enum": list(TIERS)}, "minItems": 1},
        "tier_info": {"type": "object", "additionalProperties": {
            "type": "object", "required": ["status", "tool", "fidelity", "runtime_s", "warnings",
                                           "inputs_from", "n_examples"]}},
        "metric_matrix": {"type": "object", "propertyNames": {"enum": list(METRIC_IDS)},
                          "additionalProperties": {"type": "object"}},
        "deltas_vs_virtual": {"type": "object", "propertyNames": {"enum": list(METRIC_IDS)},
                              "additionalProperties": {"type": "object"}},
        "sweeps": {"type": "object", "propertyNames": {"enum": list(SWEEP_AXES)},
                   "additionalProperties": {"type": "object"}},
        "knees": {"type": "array", "items": {
            "type": "object", "required": ["axis", "tier", "metric", "knee_value", "rule"]}},
        # optional blocks written by the M4a pipeline (all absent in older compare.json files)
        "reference": {"type": "object", "required": ["label", "is_tier"],
                      "properties": {"is_tier": {"const": False}}},   # float baseline: a reference, NOT a fifth tier
        "paired_vs_virtual": {"type": "object"},   # per subset tier: V and tier metrics on the same examples
        "timing": {"type": "object"},              # wall seconds per pipeline stage
        "run_meta": {"type": "object"},            # model file, subset policy, machine load, ...
    },
}

COMMENTARY_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "DLNN llm/commentary.json",
    "type": "object",
    "required": ["schema_version", "source", "backend", "summary", "findings", "caveats", "questions"],
    "additionalProperties": True,
    "properties": {
        "schema_version": {"const": COMMENTARY_SCHEMA_VERSION},
        "source": {"enum": ["template", "model"]},
        "backend": {"type": "string"},
        "summary": {"type": "string"},
        "findings": {"type": "array", "items": {
            "type": "object", "required": ["metric_id", "tiers", "observation", "likely_cause"],
            "properties": {"metric_id": {"enum": list(METRIC_IDS)},
                           "tiers": {"type": "array", "items": {"enum": list(TIERS)}},
                           "observation": {"type": "string"}, "likely_cause": {"type": "string"}}}},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "questions": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass
class TierResult:
    tier: str
    status: str
    run_id: str
    task: str
    tool: dict[str, str]
    model_sha256: str
    dataset_sha256: str
    n_examples: int
    metrics: dict[str, Any] = field(default_factory=dict)
    sweeps: list[dict[str, Any]] = field(default_factory=list)
    fidelity: dict[str, list[str]] = field(default_factory=lambda: {"models": [], "omits": []})
    inputs_from: list[str] = field(default_factory=list)
    runtime_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    arrays: dict[str, np.ndarray] = field(default_factory=dict)  # in memory; index goes to JSON
    schema_version: str = SCHEMA_VERSION

    def to_json_dict(self, npz_name: str | None = None) -> dict[str, Any]:
        npz_name = npz_name or f"{self.tier}.npz"
        d = asdict(self)
        d.pop("arrays")
        d["arrays"] = {k: {"file": npz_name, "key": k, "shape": list(np.shape(v)),
                           "dtype": str(np.asarray(v).dtype)} for k, v in self.arrays.items()}
        return _jsonable(d)


def _jsonable(o: Any) -> Any:
    """Recursively convert numpy scalars/arrays to JSON types; NaN/inf -> None."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, np.generic):
        return _jsonable(o.item())
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return o


def _check(obj: Any, schema: dict[str, Any], extra: list[str]) -> None:
    import jsonschema
    v = jsonschema.Draft202012Validator(schema)
    errs = [f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}"
            for e in sorted(v.iter_errors(obj), key=lambda e: list(map(str, e.absolute_path)))]
    errs += extra
    if errs:
        raise ContractError("; ".join(errs))


def validate(obj: "TierResult | dict[str, Any]") -> dict[str, Any]:
    """Validate a TierResult or its JSON dict. Raises ContractError listing all problems."""
    d = obj.to_json_dict() if isinstance(obj, TierResult) else obj
    extra: list[str] = []
    if isinstance(d, dict):
        fid = d.get("fidelity", {})
        if d.get("status") in ("stub", "fallback") and not fid.get("omits"):
            extra.append("status stub/fallback requires non-empty fidelity.omits")
        if d.get("status") == "stub" and not any("stub" in s.lower() or "placeholder" in s.lower()
                                                 for s in fid.get("omits", [])):
            extra.append("status stub requires fidelity.omits to say it is a stub/placeholder")
        if d.get("status") in ("ok", "stub", "fallback") and not d.get("metrics"):
            extra.append("metrics must not be empty unless status is skipped")
        if d.get("status") == "skipped" and not d.get("warnings"):
            extra.append("status skipped requires a reason in warnings")
        for sw in d.get("sweeps", []) if isinstance(d.get("sweeps"), list) else []:
            n = len(sw.get("values", []))
            for k in ("accuracy", "mean_kl", "nmse"):
                if k in sw and len(sw[k]) != n:
                    extra.append(f"sweep {sw.get('axis')}.{k} length != values length")
    _check(d, TIER_SCHEMA, extra)
    return d


def validate_compare(obj: dict[str, Any]) -> dict[str, Any]:
    _check(obj, COMPARE_SCHEMA, [])
    return obj


def validate_commentary(obj: dict[str, Any]) -> dict[str, Any]:
    extra = []
    if obj.get("source") == "template" and obj.get("backend") != "null":
        extra.append("source 'template' is only allowed for the null backend")
    _check(obj, COMMENTARY_SCHEMA, extra)
    return obj


def save(result: TierResult, outdir: str | Path) -> Path:
    """Write ``<outdir>/tiers/<tier>.json`` (+ ``.npz`` if arrays). Returns the JSON path."""
    d = validate(result)  # refuse to write invalid results
    tdir = Path(outdir) / "tiers"
    tdir.mkdir(parents=True, exist_ok=True)
    if result.arrays:
        np.savez_compressed(tdir / f"{result.tier}.npz", **result.arrays)
    else:
        d["arrays"] = {}
    jp = tdir / f"{result.tier}.json"
    jp.write_text(json.dumps(d, indent=2))
    return jp


def load(path: str | Path, with_arrays: bool = True) -> TierResult:
    """Load ``<tier>.json`` (and its npz). Validates the JSON first."""
    path = Path(path)
    d = validate(json.loads(path.read_text()))
    arrays: dict[str, np.ndarray] = {}
    if with_arrays and d["arrays"]:
        with np.load(path.parent / next(iter(d["arrays"].values()))["file"]) as z:
            arrays = {k: z[k] for k in d["arrays"]}
    d = {k: v for k, v in d.items() if k != "arrays"}
    return TierResult(arrays=arrays, **d)
