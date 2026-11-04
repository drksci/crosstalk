"""Number and id verification of LLM commentary against the card. Pure functions, no model needed.

Policy
  * Every number in the model's text must match some number printed in the card, within
    max(1 % relative, half a unit of the card's printed precision), or a correct rounding of the card value to >= 2 decimals.  Sign is ignored ("drop of 0.03" matches
    "(-0.03)"); a number followed by "%" also matches value/100.  The count of tiers is the only derived number allowed.
  * Scoped check: "metric_id = number" (also "of/is/at") must match a value the card prints for THAT metric
    (any tier, or a delta), else the failure kind is ``misattributed`` and counts as an unverified number.
  * Unit-bound numbers ("6-bit", "30 dB/us", "60 mV", "ns", "MHz") must occur with the same unit somewhere in the card:
    "32-bit" is flagged ``misattributed`` when the card only has "8-bit" and "32 dB/us".
  * Consistency (when the backend passes the card's structure): every tier on the Python-computed divergence list must
    have a finding (``missing_tier``); a sentence may not call an ``ok`` tier a stub/fallback/placeholder (``status_claim``).
  * Identifier-like tokens (metric ids, axis names) are stripped before number extraction so ``top1_agreement``
    does not yield "1".  Metric ids / tier ids in structured fields must be valid under the contract AND present in the card.
  * Snake_case tokens in free text that do not occur in the card (e.g. a metric id the card does not have) are
    reported as ``unknown_id``.
Limitation: outside the "metric_id = number" pattern a number is only checked for being present in the card, not for
being attached to the right tier (e.g. "E reaches 0.84" is not checked against E's column).
"""
from __future__ import annotations

import copy
import re
from dataclasses import dataclass
from typing import Any

from .. import contract

_NUM = re.compile(r"(?<![A-Za-z_0-9.])([-+]?)(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(\s*%)?")
_ID = re.compile(r"\b[A-Za-z]+(?:_[A-Za-z0-9]+)+\b")      # snake_case identifiers
_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")
# Tokens in prose that glue digits to letters (L1, int8, top1, Q.15, RG-58 ...) are not matched by _NUM's lookbehind.


@dataclass(frozen=True)
class Num:
    value: float          # absolute value as written
    decimals: int         # digits after the decimal point as written (0 for integers)
    text: str
    pct: bool = False


def _norm(text: str) -> str:
    return text.replace("\u2212", "-").replace("\u2013", "-")


def strip_ids(text: str) -> str:
    """Blank out identifier-like tokens (snake_case) so their digits are not read as numbers."""
    return _ID.sub(" ", text)


def extract_numbers(text: str) -> list[Num]:
    out: list[Num] = []
    t = strip_ids(_norm(text))
    for m in _NUM.finditer(t):
        digits = m.group(2)
        try:
            v = abs(float(digits))
        except ValueError:  # pragma: no cover
            continue
        dec = len(digits.split(".")[1]) if "." in digits and "e" not in digits.lower() else 0
        out.append(Num(v, dec, (m.group(1) + digits + (m.group(3) or "")).strip(), bool(m.group(3))))
    return out


def numbers_in_card(card: str) -> list[Num]:
    return extract_numbers(card)


def metric_values_in_card(card: str) -> dict[str, list[Num]]:
    """Numbers the card prints for each metric id: the whole 'metric_id: ...' line, plus 'metric_id=value (delta)'
    pairs on divergence lines. Used for the scoped check 'metric_id = number'."""
    out: dict[str, list[Num]] = {}
    for line in card.splitlines():
        head = line.split(":", 1)[0].strip()
        if head in contract.METRIC_IDS and ":" in line:
            out.setdefault(head, []).extend(extract_numbers(line.split(":", 1)[1]))
        for m in re.finditer(r"\b([a-z0-9_]+)=(-?\d[\d.]*(?:e[-+]?\d+)?)(?:\s*\(([-+]?\d[\d.]*(?:e[-+]?\d+)?)\))?", line):
            if m.group(1) in contract.METRIC_IDS:
                out.setdefault(m.group(1), []).extend(extract_numbers(" ".join(g for g in m.groups()[1:] if g)))
    return out


_UNIT = re.compile(r"(?<![A-Za-z_0-9.])(\d+(?:\.\d+)?)\s*-?\s*(bits?|dB/us|dB|mV|ns|us|MHz|GHz|%)(?![A-Za-z])", re.I)


def number_unit_pairs(text: str) -> set[tuple[float, str]]:
    """(value, unit) pairs such as (6, 'bit') for '6-bit ADC' or (30, 'db/us') for '30 dB/us'."""
    return {(float(m.group(1)), m.group(2).lower().rstrip("s") if m.group(2).lower().startswith("bit") else m.group(2).lower())
            for m in _UNIT.finditer(_norm(text))}


_SCOPED = re.compile(r"\b(" + "|".join(sorted(contract.METRIC_IDS, key=len, reverse=True)) +
                     r")\b\s*(?:=|:|of|is|at|was|reaches)?\s*\(?\s*([-+\u2212]?\d+(?:\.\d+)?)(\s*%)?")


def _close(a: Num, c: Num) -> bool:
    """a (model) vs c (card): within max(1 % relative, half a unit of the card's printed precision), or a correct
    rounding of the card value to >= 2 decimals (e.g. 0.84 for a printed 0.8421). A model integer never matches a
    fractional card value through rounding."""
    cands = [(a.value, a.decimals)]
    if a.pct:  # "91 %" may stand for a card value of 0.91
        cands.append((a.value / 100.0, a.decimals + 2))
    for val, adec in cands:
        half_unit = 0.5 * 10 ** (-c.decimals) if c.decimals > 0 else 0.0   # integers printed in the card are exact
        if abs(val - c.value) <= max(0.01 * abs(c.value), half_unit):
            return True
        if adec >= 2 and abs(round(c.value, adec) - val) < 1e-9:
            return True
    return False


def number_ok(a: Num, card_nums: list[Num], extra_allowed: tuple[float, ...] = ()) -> bool:
    if any(abs(a.value - e) < 1e-12 for e in extra_allowed):
        return True
    return any(_close(a, c) for c in card_nums)


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT.split(_norm(text).strip()) if p.strip()]
    return parts or ([text.strip()] if text.strip() else [])


@dataclass
class Failure:
    where: str            # e.g. "summary", "findings[1].observation", "caveats[0]"
    kind: str             # "number" | "misattributed" | "metric_id" | "tier_id" | "unknown_id"
    token: str
    sentence: str = ""

    def as_dict(self) -> dict[str, str]:
        return {"where": self.where, "kind": self.kind, "token": self.token, "sentence": self.sentence}


def _check_text(where: str, text: str, card_nums: list[Num], extra: tuple[float, ...],
                known_ids: set[str], scoped: dict[str, list[Num]] | None = None,
                card_units: set[tuple[float, str]] | None = None) -> tuple[list[Failure], int]:
    fails: list[Failure] = []
    n = 0
    card_units = card_units or set()
    for s in split_sentences(text):
        for val, unit in sorted(number_unit_pairs(s) - card_units):
            if unit != "%":
                fails.append(Failure(where, "misattributed", f"{val:g} {unit}", s))
        for m in _SCOPED.finditer(_norm(s)) if scoped else []:
            vals = scoped.get(m.group(1))
            a = extract_numbers(m.group(2) + (m.group(3) or ""))
            if vals and a and number_ok(a[0], card_nums, extra) and not number_ok(a[0], vals):
                fails.append(Failure(where, "misattributed", f"{m.group(1)}={a[0].text}", s))
        for a in extract_numbers(s):
            n += 1
            if not number_ok(a, card_nums, extra):
                fails.append(Failure(where, "number", a.text, s))
        for ident in _ID.findall(s):
            if ident not in known_ids:
                fails.append(Failure(where, "unknown_id", ident, s))
    return fails, n


def known_identifiers(card: str = "") -> set[str]:
    """snake_case identifiers the model may mention: those that appear in the card text itself."""
    return set(_ID.findall(card))


_TIER_TOKEN = re.compile(r"(?<![A-Za-z0-9_])([VECR])(?![A-Za-z0-9_])")
_PLACEHOLDER = re.compile(r"\b(stubs?|fallbacks?|placeholders?)\b", re.I)


def status_claims(text: str, tier_status: dict[str, str]) -> list[tuple[str, str]]:
    """(tier, sentence) pairs where a sentence calls a tier a stub/fallback/placeholder although the card says its
    status is ok. Ambiguous sentences (that also mention a genuinely stub/fallback tier) are not flagged."""
    out = []
    for sent in split_sentences(text):
        if not _PLACEHOLDER.search(sent):
            continue
        mentioned = (set(_TIER_TOKEN.findall(sent)) & set(tier_status)) - {"V"}   # V is the reference, never a stub
        if mentioned and all(tier_status[t] == "ok" for t in mentioned):
            out.extend((t, sent) for t in sorted(mentioned))
    return out


def verify_output(out: dict[str, Any], card: str, tiers: list[str] | None = None,
                  metric_ids: list[str] | None = None, expected_tiers: list[str] | None = None,
                  tier_status: dict[str, str] | None = None) -> dict[str, Any]:
    """Check one model output dict {summary, findings, caveats, questions}.

    Returns {"ok": bool, "n_numbers": int, "failures": [Failure.as_dict()], ...}. ``tiers`` / ``metric_ids`` default to
    the full contract sets; pass the card's own sets to also require ids to be present in the card.
    ``expected_tiers`` (tiers on the card's divergence list): each must appear in some finding, else ``missing_tier``.
    ``tier_status`` ({tier: status}): calling an ``ok`` tier a stub/fallback/placeholder is a ``status_claim``.
    """
    card_nums = numbers_in_card(card)
    tiers = tiers if tiers is not None else list(contract.TIERS)
    metric_ids = metric_ids if metric_ids is not None else list(contract.METRIC_IDS)
    extra = (float(len(tiers)),)
    known = known_identifiers(card)
    scoped = metric_values_in_card(card)
    card_units = number_unit_pairs(card)
    fails: list[Failure] = []
    n_total = 0

    def run(where: str, text: Any) -> None:
        nonlocal n_total
        if not isinstance(text, str):
            return
        f, n = _check_text(where, text, card_nums, extra, known, scoped, card_units)
        fails.extend(f)
        n_total += n

    run("summary", out.get("summary", ""))
    for i, f in enumerate(out.get("findings", [])):
        mid = f.get("metric_id")
        if mid not in contract.METRIC_IDS or mid not in metric_ids:
            fails.append(Failure(f"findings[{i}].metric_id", "metric_id", str(mid)))
        for t in f.get("tiers", []):
            if t not in contract.TIERS or t not in tiers:
                fails.append(Failure(f"findings[{i}].tiers", "tier_id", str(t)))
        run(f"findings[{i}].observation", f.get("observation", ""))
        run(f"findings[{i}].likely_cause", f.get("likely_cause", ""))
    for key in ("caveats", "questions"):
        for i, s in enumerate(out.get(key, [])):
            run(f"{key}[{i}]", s)
    if expected_tiers:
        covered = {t for f in out.get("findings", []) for t in f.get("tiers", [])}
        for t in expected_tiers:
            if t not in covered:
                fails.append(Failure("findings", "missing_tier", t))
    if tier_status:
        texts = [("summary", out.get("summary", ""))] + \
                [(f"findings[{i}].{k}", f.get(k, "")) for i, f in enumerate(out.get("findings", []))
                 for k in ("observation", "likely_cause")] + \
                [(f"caveats[{i}]", c) for i, c in enumerate(out.get("caveats", []))]
        for where, txt in texts:
            if isinstance(txt, str):
                for t, sent in status_claims(txt, tier_status):
                    fails.append(Failure(where, "status_claim", f"{t} is {tier_status[t]}", sent))
    return {"ok": not fails, "n_numbers": n_total, "n_failures": len(fails),
            "n_unverified_numbers": sum(1 for f in fails if f.kind in ("number", "misattributed")),
            "n_id_failures": sum(1 for f in fails if f.kind not in ("number", "misattributed")),
            "failures": [f.as_dict() for f in fails]}


def mark_unverified(out: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of ``out`` with ``verified`` flags: per finding, ``summary_verified``, ``caveats_verified`` /
    ``questions_verified`` (list of bool) and top-level ``numbers_verified`` (False if anything failed)."""
    o = copy.deepcopy(out)
    bad = {f["where"] for f in report["failures"]}
    bad_prefix = lambda p: any(w == p or w.startswith(p + ".") or w.startswith(p + "[") for w in bad)  # noqa: E731
    o["summary_verified"] = not bad_prefix("summary")
    for i, f in enumerate(o.get("findings", [])):
        f["verified"] = not bad_prefix(f"findings[{i}]")
    for key in ("caveats", "questions"):
        o[f"{key}_verified"] = [not bad_prefix(f"{key}[{i}]") for i in range(len(o.get(key, [])))]
    o["numbers_verified"] = report["n_unverified_numbers"] == 0
    o["ids_verified"] = report["n_id_failures"] == 0
    o["unverified"] = report["failures"]
    return o


def retry_message(report: dict[str, Any]) -> str:
    """Text appended to the conversation for the single retry."""
    lines = []
    for f in report["failures"]:
        if f["kind"] == "number":
            lines.append(f"- number '{f['token']}' in {f['where']} is not in the card (sentence: \"{f['sentence']}\")")
        elif f["kind"] == "misattributed":
            lines.append(f"- '{f['token']}' in {f['where']}: the card prints no such value for that metric (sentence: \"{f['sentence']}\")")
        elif f["kind"] == "missing_tier":
            lines.append(f"- tier {f['token']} is on the card's MATERIAL DIVERGENCES list but has no finding; add one")
        elif f["kind"] == "status_claim":
            lines.append(f"- {f['where']}: you called a tier a stub/fallback/placeholder but the card says {f['token']} "
                         f"(sentence: \"{f['sentence']}\")")
        elif f["kind"] in ("metric_id", "tier_id"):
            lines.append(f"- invalid {f['kind']} '{f['token']}' in {f['where']}")
        else:
            lines.append(f"- unknown identifier '{f['token']}' in {f['where']}")
    return ("Your previous answer failed verification against the card:\n" + "\n".join(lines[:12]) +
            "\nRewrite the whole JSON. Use ONLY numbers printed in the card, copied exactly as printed "
            "(including printed deltas); do not compute differences or percentages. If no printed number fits, "
            "state the point without a number. Use only metric ids and tier ids that appear in the card.")
