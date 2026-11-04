"""Evaluate the commentator on 8 synthetic cards with injected divergences.   python -m dlnn.llm.eval

Scoring (fixed, not tuned):
  * injected card: CORRECT iff some finding has metric_id in the metrics the card flags as material for the injected
    tier AND lists the injected tier AND lists no tier other than {injected tier, V}.  STRICT additionally requires
    the metric to be the card's most severe one.
  * control card (no divergence): CORRECT iff no finding/summary claims a divergence for a non-V tier (keyword rule
    in ``claims_divergence``; raw outputs are saved in eval_report.json for audit).
  * numbers: every number in the output is matched to the card by verify.py.  raw = first model answer, final = after
    the single retry (remaining ones are flagged verified=false), exposed = numbers in text still marked verified.
The synthetic cards are invented numbers (dlnn/llm/synth.py), not simulation results.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from .. import contract
from . import backend as backend_mod, card as card_mod, synth, verify

ROOT = Path(__file__).resolve().parents[2]
BEGIN, END = "<!-- eval:begin -->", "<!-- eval:end -->"

_DEGRADE = re.compile(r"degrad|drop|worse|diverg|collaps|poor|fail|mismatch|significant|large|substantial|"
                      r"deviat|lower|higher|differ|loss of", re.I)
_NEG = re.compile(r"\b(no|not|none|without|negligible|little|minimal|nor|neither|insignificant|within)\b", re.I)


def claims_divergence(text: str) -> bool:
    """Keyword rule: a sentence claims divergence if it has a degradation word and no negation word."""
    for s in verify.split_sentences(text):
        if _DEGRADE.search(s) and not _NEG.search(s):
            return True
    return False


def _all_text(out: dict[str, Any]) -> str:
    parts = [out.get("summary", "")] + out.get("caveats", [])
    for f in out.get("findings", []):
        parts += [f.get("observation", ""), f.get("likely_cause", "")]
    return " ".join(parts)


def score_card(sc: synth.Scenario, exp: dict[str, Any], cmp: dict[str, Any], out: dict[str, Any], card: str) -> dict[str, Any]:
    findings = out.get("findings", [])
    r: dict[str, Any] = {"sid": sc.sid, "injected_tier": exp["tier"], "expected_metrics": exp["metrics"],
                         "primary": exp["primary"]}
    if exp["tier"] is not None:
        ok = strict = lenient_tier = False
        extra = 0
        for f in findings:
            tiers = set(f.get("tiers", []))
            tier_hit = exp["tier"] in tiers
            clean = tiers <= {exp["tier"], "V"}
            if f.get("metric_id") in exp["metrics"] and tier_hit and clean:
                ok = True
                strict = strict or f["metric_id"] == exp["primary"]
            lenient_tier = lenient_tier or (tier_hit and f.get("metric_id") in exp["metrics"])
            if (tiers - {exp["tier"], "V"}) and f.get("metric_id") in exp["metrics"]:
                extra += 1
        r.update(correct=ok, strict=strict, lenient_tier=lenient_tier, extra_tier_findings=extra)
    else:
        flagged = [f for f in findings if (set(f.get("tiers", [])) - {"V"}) and
                   claims_divergence(f.get("observation", "") + " " + f.get("likely_cause", ""))]
        summ = claims_divergence(out.get("summary", ""))
        r.update(correct=not flagged and not summ, strict=not flagged and not summ, false_alarm_findings=len(flagged),
                 false_alarm_summary=summ)
    stub_tiers = [t for t in cmp["tiers"] if cmp["tier_info"][t]["status"] in ("stub", "fallback")]
    if stub_tiers:
        txt = _all_text(out)
        r["stub_flagged"] = all(re.search(rf"\b{t}\b", txt) for t in stub_tiers) and \
            bool(re.search(r"stub|fallback|placeholder", txt, re.I))
    ver = out.get("verification", {})
    r["numbers"] = {"n": ver.get("n_numbers", 0), "raw_unverified": ver.get("n_unverified_raw", 0),
                    "final_unverified": ver.get("n_unverified_final", 0), "retried": ver.get("retried", False),
                    "id_failures_raw": ver.get("n_id_failures_raw", 0), "id_failures_final": ver.get("n_id_failures_final", 0)}
    # exposed: numbers inside text that is still marked verified (must be 0 by construction)
    exposed = 0
    ids = card_mod.card_ids(cmp)
    rep = verify.verify_output(out, card, ids["tier_ids"], ids["metric_ids"])
    bad_where = {f["where"] for f in rep["failures"] if f["kind"] == "number"}
    for w in bad_where:
        m = re.match(r"(summary|findings\[(\d+)\]|caveats\[(\d+)\]|questions\[(\d+)\])", w)
        if not m:
            continue
        if w == "summary" and out.get("summary_verified", True):
            exposed += 1
        elif m.group(2) is not None and out["findings"][int(m.group(2))].get("verified", True):
            exposed += 1
        elif m.group(3) is not None and out["caveats_verified"][int(m.group(3))]:
            exposed += 1
        elif m.group(4) is not None and out["questions_verified"][int(m.group(4))]:
            exposed += 1
    r["numbers"]["exposed_as_verified"] = exposed
    r["perf"] = out.get("llm", {}).get("perf")
    return r


def run_eval(be: backend_mod.Backend, rank: bool = True, only: list[str] | None = None) -> dict[str, Any]:
    cards, results = synth.all_cards(), []
    for sc, cmp in cards:
        if only and sc.sid not in only:
            continue
        card = card_mod.render_card(cmp, rank=rank)
        exp = synth.expected(sc, cmp)
        t0 = time.perf_counter()
        out = be.comment(cmp, card)
        wall = time.perf_counter() - t0
        contract.validate_commentary(out)
        res = score_card(sc, exp, cmp, out, card)
        res.update(title=sc.title, wall_s=round(wall, 1), card_tokens_est=card_mod.approx_tokens(card))
        res["output"] = out
        results.append(res)
        print(f"[{be.name}{'' if rank else ' no-rank'}] {sc.sid:6s} correct={res['correct']} strict={res['strict']} "
              f"nums={res['numbers']['n']} raw_bad={res['numbers']['raw_unverified']} "
              f"final_bad={res['numbers']['final_unverified']} wall={wall:.0f}s", flush=True)
    return summarise(be, results, rank)


def summarise(be: backend_mod.Backend, results: list[dict[str, Any]], rank: bool) -> dict[str, Any]:
    inj = [r for r in results if r["injected_tier"] is not None]
    ctl = [r for r in results if r["injected_tier"] is None]
    n = sum(r["numbers"]["n"] for r in results)
    toks = [r["perf"]["decode_tok_per_s"] for r in results if r.get("perf")]
    first = results[0]["output"].get("llm", {}) if results else {}
    return {
        "backend": be.name, "source": "model" if be.name != "null" else "template", "rank_section": rank,
        "model": first.get("model"), "model_sha256": first.get("model_sha256"),
        "n_cards": len(results),
        "attribution_correct": sum(r["correct"] for r in results), "attribution_total": len(results),
        "attribution_rate": round(sum(r["correct"] for r in results) / max(len(results), 1), 3),
        "injected_correct": sum(r["correct"] for r in inj), "injected_total": len(inj),
        "injected_strict": sum(r["strict"] for r in inj),
        "controls_correct": sum(r["correct"] for r in ctl), "controls_total": len(ctl),
        "stub_flagged": f"{sum(r.get('stub_flagged', False) for r in results)}/{sum('stub_flagged' in r for r in results)}",
        "numbers_total": n,
        "numbers_unverified_raw": sum(r["numbers"]["raw_unverified"] for r in results),
        "numbers_unverified_final_flagged": sum(r["numbers"]["final_unverified"] for r in results),
        "numbers_exposed_as_verified": sum(r["numbers"]["exposed_as_verified"] for r in results),
        "id_failures_raw": sum(r["numbers"]["id_failures_raw"] for r in results),
        "id_failures_final": sum(r["numbers"]["id_failures_final"] for r in results),
        "cards_retried": sum(r["numbers"]["retried"] for r in results),
        "decode_tok_per_s_mean": round(sum(toks) / len(toks), 2) if toks else None,
        "wall_s_total": round(sum(r["wall_s"] for r in results), 1),
        "results": results}


def md_table(summaries: list[dict[str, Any]]) -> str:
    lines = ["| backend | card section | attribution (all 8) | injected 5 (loose / strict) | controls 3 | stub flagged | "
             "numbers | unverified raw | unverified after retry (flagged) | exposed as verified | id/term failures raw / final | "
             "retried | decode tok/s |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(f"| {s['backend']} ({s['source']}) | {'with' if s['rank_section'] else 'without'} divergence list | "
                     f"{s['attribution_correct']}/{s['attribution_total']} = {100 * s['attribution_rate']:.0f} % | "
                     f"{s['injected_correct']}/{s['injected_total']} / {s['injected_strict']}/{s['injected_total']} | "
                     f"{s['controls_correct']}/{s['controls_total']} | {s['stub_flagged']} | {s['numbers_total']} | "
                     f"{s['numbers_unverified_raw']} | {s['numbers_unverified_final_flagged']} | "
                     f"{s['numbers_exposed_as_verified']} | {s['id_failures_raw']} / {s['id_failures_final']} | "
                     f"{s['cards_retried']} | {s['decode_tok_per_s_mean']} |")
    return "\n".join(lines)


def md_per_card(s: dict[str, Any]) -> str:
    lines = [f"**{s['backend']}, divergence list {'on' if s['rank_section'] else 'off'}**", "",
             "| card | injected | correct | strict | numbers (raw bad / final bad) | wall s |", "|---|---|---|---|---|---|"]
    for r in s["results"]:
        lines.append(f"| {r['sid']}: {r['title']} | {r['injected_tier'] or 'none'} | {r['correct']} | {r['strict']} | "
                     f"{r['numbers']['n']} ({r['numbers']['raw_unverified']} / {r['numbers']['final_unverified']}) | {r['wall_s']} |")
    return "\n".join(lines)


def write_outputs(summaries: list[dict[str, Any]], docs: Path, report: Path) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"generated": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                                  "scoring": __doc__, "summaries": summaries}, indent=1, default=str))
    block = "\n".join([BEGIN, "", "### Summary", "", md_table(summaries), "", "### Per card", "",
                       *[md_per_card(s) + "\n" for s in summaries], END])
    old = docs.read_text() if docs.exists() else "# LLM commentary evaluation\n\n" + BEGIN + "\n" + END + "\n"
    if BEGIN in old and END in old:
        a, b = old.index(BEGIN), old.index(END) + len(END)
        new = old[:a] + block + old[b:]
    else:
        new = old + "\n" + block + "\n"
    docs.write_text(new)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", default="qwen", help="qwen | bitnet | null | comma list")
    ap.add_argument("--ablate", action="store_true", help="also run without the Python-computed divergence list")
    ap.add_argument("--no-rank", action="store_true", help="run only without the divergence list")
    ap.add_argument("--only", default="", help="comma list of card ids")
    ap.add_argument("--docs", default=str(ROOT / "docs" / "llm_eval.md"))
    ap.add_argument("--report", default=str(ROOT / "models" / "llm" / "eval_report.json"))
    ap.add_argument("--merge", action="store_true", help="keep previous summaries in the report for other backends/modes")
    a = ap.parse_args(argv)
    only = [x for x in a.only.split(",") if x] or None
    summaries: list[dict[str, Any]] = []
    report = Path(a.report)
    if a.merge and report.exists():
        summaries = json.loads(report.read_text())["summaries"]
    for name in a.backend.split(","):
        be = backend_mod.get_backend(name)
        modes = [False] if a.no_rank else ([True, False] if a.ablate else [True])
        for rank in modes:
            s = run_eval(be, rank=rank, only=only)
            summaries = [x for x in summaries if not (x["backend"] == s["backend"] and x["rank_section"] == s["rank_section"])]
            summaries.append(s)
    write_outputs(summaries, Path(a.docs), report)
    print(md_table(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
