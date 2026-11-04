"""Render the <=1.5k-token text card from compare.json: the ONLY thing the commentator LLM sees.

Everything numeric in the card is computed here in Python (deltas, materiality flags, sweep knees come from
``compare.py``). The LLM never sees raw arrays and is not asked to compute anything.

Sections: header, TIERS (status, tool, models, omits; stub/fallback tiers are shouted in capitals),
METRICS (value; delta vs V), ACC_PER_SNR, MATERIAL DIVERGENCES (Python-computed, optional), SWEEP KNEES, WARNINGS.
"""
from __future__ import annotations

from typing import Any

from .. import contract

MAX_TOKENS = 1500
MAX_CHARS = 3500          # char budget; digits tokenise one per token (Qwen), measured 2.5 chars/token on real cards
CHARS_PER_TOKEN = 2.4     # conservative estimate used by approx_tokens (real tokenizer is checked in tests)

# Materiality thresholds for the Python-computed "MATERIAL DIVERGENCES" section.
# (metric_id) -> (direction, threshold, text shown in the card).  "lt" = flagged when value < thr (agreement-like),
# "gt" = flagged when value > thr (distance-like), "drop" = flagged when delta vs V <= -thr.
THRESHOLDS: dict[str, tuple[str, float]] = {
    "top1_agreement_vs_virtual": ("lt", 0.97),
    "mean_kl_virtual_vs_tier": ("gt", 0.05),
    "logit_nmse_vs_virtual": ("gt", 0.05),
    "nmse_vs_virtual": ("gt", 0.02),
    "accuracy": ("drop", 0.03),
    "macro_f1": ("drop", 0.03),
    "nmse": ("rise", 0.02),
}
TIER_NAMES = {"V": "virtual, ideal math", "E": "emulated, fixed-point", "C": "circuit, SPICE-style",
              "R": "RF, S-parameter-style"}
DISPLAY_ORDER = ("accuracy", "macro_f1", "nmse", "top1_agreement_vs_virtual", "mean_kl_virtual_vs_tier",
                 "logit_nmse_vs_virtual", "nmse_vs_virtual")


def _f(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (int, float)):
        return f"{v:.3g}"
    return str(v)


def _clip(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def _scalar_rows(cmp: dict[str, Any]) -> dict[str, dict[str, float]]:
    return {mid: row for mid, row in cmp["metric_matrix"].items()
            if not any(isinstance(v, (dict, list)) for v in row.values())}


def divergences(cmp: dict[str, Any]) -> list[dict[str, Any]]:
    """Material divergences from V per (tier, metric), sorted by severity (value/threshold ratio). Pure function."""
    rows, dv = _scalar_rows(cmp), cmp["deltas_vs_virtual"]
    out: list[dict[str, Any]] = []
    for mid, (kind, thr) in THRESHOLDS.items():
        for t, v in rows.get(mid, {}).items():
            if t == "V" or v is None:
                continue
            d = dv.get(mid, {}).get(t)
            if kind == "lt":
                hit, sev = v < thr, (1 - v) / (1 - thr)
            elif kind == "gt":
                hit, sev = v > thr, v / thr
            elif kind == "drop":
                hit, sev = d is not None and d <= -thr, (-d / thr if d is not None else 0.0)
            else:  # rise
                hit, sev = d is not None and d >= thr, (d / thr if d is not None else 0.0)
            if hit:
                out.append({"metric_id": mid, "tier": t, "value": v, "delta": d, "severity": sev})
    out.sort(key=lambda r: -r["severity"])
    return out


def _threshold_text(present: set[str]) -> str:
    short = {"top1_agreement_vs_virtual": "top1_agreement_vs_virtual<{t:g}", "mean_kl_virtual_vs_tier": "mean_kl_virtual_vs_tier>{t:g}",
             "logit_nmse_vs_virtual": "logit_nmse_vs_virtual>{t:g}", "nmse_vs_virtual": "nmse_vs_virtual>{t:g}",
             "accuracy": "accuracy delta<=-{t:g}", "macro_f1": "macro_f1 delta<=-{t:g}", "nmse": "nmse delta>={t:g}"}
    return ", ".join(short[m].format(t=thr) for m, (kind, thr) in THRESHOLDS.items() if m in present)


def render_card(cmp: dict[str, Any], max_chars: int = MAX_CHARS, rank: bool = True) -> str:
    """Return the card text. ``rank=False`` omits the MATERIAL DIVERGENCES section (used for the eval ablation)."""
    task, tiers, ti = cmp["task"], cmp["tiers"], cmp["tier_info"]
    L: list[str] = [f"CARD run={cmp['run_id']} task={task} tiers={','.join(tiers)}",
                    "V is the reference (ideal math). Every metric 'vs_virtual' and every delta is relative to V.",
                    "", "TIERS"]
    for t in tiers:
        i = ti[t]
        st = i["status"]
        flag = ""
        if st in ("stub", "fallback"):
            flag = f"  <<< {st.upper()} / PLACEHOLDER: not the real simulation, say so"
        elif st == "skipped":
            flag = "  <<< SKIPPED: no results"
        L.append(f"{t} ({TIER_NAMES.get(t, t)}): status={st} tool={i['tool']['name']}{flag}")
        if i["fidelity"]["models"]:
            L.append("   models: " + _clip("; ".join(i["fidelity"]["models"]), 200))
        if i["fidelity"]["omits"]:
            L.append("   omits: " + _clip("; ".join(i["fidelity"]["omits"]), 140))
    L += ["", "METRICS (tier=value (delta vs V))"]
    rows, dv = _scalar_rows(cmp), cmp["deltas_vs_virtual"]
    for mid in DISPLAY_ORDER:
        if mid not in rows:
            continue
        d = dv.get(mid, {})
        L.append(f"{mid}: " + "; ".join(
            f"{t}={_f(v)}" + (f" ({d[t]:+.3g})" if t in d and isinstance(d[t], (int, float)) else "")
            for t, v in rows[mid].items()))
    if "acc_per_snr" in cmp["metric_matrix"]:
        aps = cmp["metric_matrix"]["acc_per_snr"]
        fmt = lambda r: ",".join(f"{k}:{v:.2f}" for k, v in sorted(r.items(), key=lambda kv: float(kv[0])))  # noqa: E731
        ref = aps.get("V")
        parts, close = [], []
        for t, r in aps.items():
            if t != "V" and ref and all(abs(r[k] - ref[k]) < 0.02 for k in r if k in ref):
                close.append(t)       # computed in Python: no need to print five numbers per close tier
            else:
                parts.append(f"{t}={fmt(r)}")
        L.append("acc_per_snr (SNR dB:accuracy): " + "; ".join(parts) +
                 (f"; {','.join(close)} within 0.02 of V at every SNR" if close else ""))
    if rank:
        L += ["", "MATERIAL DIVERGENCES FROM V (computed in Python; thresholds: " + _threshold_text(set(rows)) + ")"]
        dvs = divergences(cmp)
        if not dvs:
            L.append("none: no tier exceeds any threshold; all tiers agree with V within thresholds")
        else:
            by_tier: dict[str, list[dict[str, Any]]] = {}
            for r in dvs:
                by_tier.setdefault(r["tier"], []).append(r)
            for t, rs in by_tier.items():
                tag = " (tier is a stub/fallback)" if ti[t]["status"] in ("stub", "fallback") else ""
                L.append(f"{t} ({TIER_NAMES.get(t, t)}){tag}: " + "; ".join(
                    f"{r['metric_id']}={_f(r['value'])}" + (f" ({r['delta']:+.3g})" if r["delta"] is not None else "")
                    for r in rs))
                L.append(f"   {t} own models line: " + _clip("; ".join(ti[t]["fidelity"]["models"]), 160))
    knees = [k for k in cmp["knees"]]
    if knees:
        L += ["", "SWEEP KNEES (sweep value where metric first degrades past the rule, walking from the best end; none = never)"]
        by_axis: dict[str, list[str]] = {}
        for k in knees:
            by_axis.setdefault(k["axis"], []).append(
                f"{k['tier']}={_f(k['knee_value']) if k['knee_value'] is not None else 'none'}")
        for ax, items in by_axis.items():
            m = next(k["metric"] for k in knees if k["axis"] == ax)
            L.append(f"{ax} ({m}): " + "; ".join(items))
    warns = [f"{t}: {w}" for t in tiers for w in ti[t]["warnings"]]
    if warns:
        L += ["", "WARNINGS"] + [_clip(w, 110) for w in warns[:4]]
    # budget: drop trailing lines (warnings, then knees...) rather than cutting mid-line
    text = "\n".join(L)
    while len(text) > max_chars and L:
        L.pop()
        while L and (not L[-1].strip() or L[-1].isupper() or L[-1].startswith(("WARNINGS", "SWEEP KNEES"))):
            L.pop()       # never leave a dangling section header or blank line
        text = "\n".join(L)
    return text


def approx_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


def card_ids(cmp: dict[str, Any]) -> dict[str, list[str]]:
    """Ids the LLM may use for this card (metric ids actually present, tier ids actually present)."""
    return {"metric_ids": [m for m in contract.METRIC_IDS if m in cmp["metric_matrix"]],
            "tier_ids": list(cmp["tiers"])}
