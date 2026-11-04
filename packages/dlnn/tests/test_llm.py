"""Tests for dlnn.llm: card, verify (pure functions), synthetic eval cards, scoring, optional model run."""
import json
from pathlib import Path

import pytest

from dlnn import contract
from dlnn.llm import backend, card, eval as llm_eval, prompt, synth, verify

GGUF = backend.MODELS_DIR / backend.QWEN_GGUF


def _llama_ok():
    try:
        import llama_cpp  # noqa: F401
        return True
    except Exception:
        return False


@pytest.fixture(scope="module")
def cards():
    return {sc.sid: (sc, cmp) for sc, cmp in synth.all_cards()}


# ---------------------------------------------------------------- verify: numbers
CARD = "accuracy: V=0.842; E=0.796 (-0.0461); R=0.7 | adc 8-bit | top1_agreement_vs_virtual: E=0.905 knee=10 loss 30 dB/us"


def test_extract_numbers_ignores_identifiers():
    nums = verify.extract_numbers("top1_agreement_vs_virtual is 0.905 for int8 at L1 and RG-58")
    assert [n.text for n in nums] == ["0.905", "58"]  # 1 in top1_agreement.. / int8 / L1 are not numbers


def test_extract_handles_signs_percent_and_unicode_minus():
    nums = verify.extract_numbers("drop of −0.05, i.e. 5 % and +0.2 and 1.2e-3")
    assert [(n.value, n.pct) for n in nums] == [(0.05, False), (5.0, True), (0.2, False), (0.0012, False)]


@pytest.mark.parametrize("text,ok", [
    ("accuracy is 0.842", True),
    ("accuracy is 0.84", True),          # correct 2-decimal rounding of 0.842
    ("accuracy is 0.8421", True),
    ("accuracy dropped by 0.046", True),  # 0.0461 printed, sign ignored
    ("E reaches 90.5 %", True),          # percent form of 0.905
    ("an 8-bit ADC at 30 dB/us", True),
    ("accuracy is 0.88", False),
    ("accuracy is 1", False),
    ("accuracy is 0.77", False),
    ("a 50 percent loss", False),
    ("accuracy is 0.832", False),
])
def test_number_matching(text, ok):
    c = verify.numbers_in_card(CARD)
    got = all(verify.number_ok(n, c) for n in verify.extract_numbers(text))
    assert got is ok


def test_relative_tolerance_one_percent():
    c = verify.numbers_in_card("v=123.0")
    assert verify.number_ok(verify.extract_numbers("124")[0], c)       # 0.8 %
    assert not verify.number_ok(verify.extract_numbers("126")[0], c)   # 2.4 %


def test_split_sentences_keeps_decimals():
    s = verify.split_sentences("E is 0.84 here. R is worse (0.7). Done")
    assert s == ["E is 0.84 here.", "R is worse (0.7).", "Done"] or len(s) == 3


# ---------------------------------------------------------------- verify: whole output
def _out(**kw):
    o = {"summary": "E differs from V.", "findings": [], "caveats": [], "questions": []}
    o.update(kw)
    return o


def test_verify_output_ok_and_flags():
    good = _out(findings=[{"metric_id": "accuracy", "tiers": ["E"], "observation": "E has accuracy 0.796.",
                           "likely_cause": "8-bit ADC."}])
    rep = verify.verify_output(good, CARD, ["V", "E", "R"], ["accuracy", "top1_agreement_vs_virtual"])
    assert rep["ok"] and rep["n_numbers"] == 2
    bad = _out(summary="E is at 0.93.", findings=[
        {"metric_id": "nmse", "tiers": ["E", "C"], "observation": "E is 0.796.", "likely_cause": "ADC of 4 bits."}])
    rep = verify.verify_output(bad, CARD, ["V", "E", "R"], ["accuracy", "top1_agreement_vs_virtual"])
    kinds = sorted((f["kind"], f["token"]) for f in rep["failures"])
    assert ("number", "0.93") in kinds and ("number", "4") in kinds
    assert ("metric_id", "nmse") in kinds and ("tier_id", "C") in kinds
    assert ("misattributed", "4 bit") in kinds                       # "4 bits" is also a unit-bound claim
    assert not rep["ok"] and rep["n_unverified_numbers"] == 3


def test_scoped_metric_number_check():
    card_txt = "accuracy: V=0.842; E=0.796 (-0.046)\nmean_kl_virtual_vs_tier: V=0; E=0.5\nother: 0.5 0.9"
    ok = _out(summary="E has accuracy=0.796 and accuracy of 0.842.")
    assert verify.verify_output(ok, card_txt)["ok"]
    # 0.5 is on the card, but not as an accuracy value -> misattributed
    bad = _out(summary="E has accuracy=0.5.")
    rep = verify.verify_output(bad, card_txt)
    assert [f["kind"] for f in rep["failures"]] == ["misattributed"] and rep["n_unverified_numbers"] == 1
    assert verify.mark_unverified(bad, rep)["numbers_verified"] is False


def test_unit_bound_numbers():
    c = "E: 8-bit absmax ADC; C: lines at 32 dB/us; bias +60 mV; delay 10 ns"
    assert verify.number_unit_pairs("a 8-bit ADC, 32 dB/us, 60 mV, 10 ns") == {(8.0, "bit"), (32.0, "db/us"), (60.0, "mv"), (10.0, "ns")}
    ok = _out(summary="E has an 8-bit ADC and C 32 dB/us of loss.")
    assert verify.verify_output(ok, c)["ok"]
    bad = _out(summary="C lists a 32-bit absmax ADC.")     # 32 and bit both occur on the card, never together
    rep = verify.verify_output(bad, c)
    assert [(f["kind"], f["token"]) for f in rep["failures"]] == [("misattributed", "32 bit")]
    assert not verify.verify_output(_out(summary="Loss is 8 dB/us."), c)["ok"]


def test_id_failure_does_not_flip_numbers_verified():
    o = _out(summary="Look at flux_capacity.")
    rep = verify.verify_output(o, CARD)
    m = verify.mark_unverified(o, rep)
    assert m["numbers_verified"] is True and m["ids_verified"] is False and m["summary_verified"] is False


def test_missing_tier_and_status_claim_checks():
    st = {"V": "ok", "E": "ok", "C": "fallback", "R": "ok"}
    o = _out(summary="R is a FALLBACK, not real.", findings=[
        {"metric_id": "accuracy", "tiers": ["E"], "observation": "E is 0.796.", "likely_cause": "x"}],
        caveats=["C is a fallback.", "C is a stub while R is fine."])
    rep = verify.verify_output(o, CARD, ["V", "E", "C", "R"], ["accuracy"], expected_tiers=["E", "R"], tier_status=st)
    kinds = sorted((f["kind"], f["token"]) for f in rep["failures"])
    assert kinds == [("missing_tier", "R"), ("status_claim", "R is ok")]
    assert rep["n_unverified_numbers"] == 0 and rep["n_id_failures"] == 2
    assert "R" in verify.retry_message(rep) and "stub/fallback/placeholder" in verify.retry_message(rep)
    assert verify.status_claims("C and R are STUBs.", {"C": "stub", "R": "stub"}) == []
    # the reference V being mentioned next to a real stub is not a status claim about V
    assert verify.status_claims("C is a FALLBACK; its agreement with V is not evidence.",
                                {"V": "ok", "C": "fallback"}) == []


def test_guard_retry_on_missing_tier(cards):
    cmp = cards["adc6"][1]                                  # divergence list has only E
    text = card.render_card(cmp)
    wrong = _ans("Agreement is 0.905.")
    wrong["findings"][0]["tiers"] = ["C"]                   # E has no finding
    be = _Scripted([wrong, _ans("Agreement is 0.905.")])
    out = be.comment(cmp, text)
    assert len(be.calls) == 2 and "no finding" in be.calls[1][-1]["content"]
    assert out["verification"]["n_id_failures_raw"] == 1 and out["verification"]["n_id_failures_final"] == 0
    assert out["ids_verified"] is True


def test_unknown_identifier_in_free_text():
    o = _out(summary="See flux_capacity and top1_agreement_vs_virtual.")
    rep = verify.verify_output(o, CARD, ["V", "E"], ["accuracy"])
    assert [f["token"] for f in rep["failures"] if f["kind"] == "unknown_id"] == ["flux_capacity"]


def test_mark_unverified_flags_only_bad_parts():
    o = _out(findings=[{"metric_id": "accuracy", "tiers": ["E"], "observation": "Accuracy 0.796.", "likely_cause": "x"},
                       {"metric_id": "accuracy", "tiers": ["R"], "observation": "Accuracy 0.11.", "likely_cause": "y"}],
             caveats=["fine", "bad 77"])
    rep = verify.verify_output(o, CARD, ["V", "E", "R"], ["accuracy"])
    m = verify.mark_unverified(o, rep)
    assert m["numbers_verified"] is False
    assert [f["verified"] for f in m["findings"]] == [True, False]
    assert m["caveats_verified"] == [True, False] and m["summary_verified"] is True
    assert len(m["unverified"]) == 2 and "7" in m["unverified"][0]["token"] + m["unverified"][1]["token"]
    assert "verified" not in o["findings"][0]  # input is not mutated


def test_retry_message_lists_mismatches():
    rep = verify.verify_output(_out(summary="Got 0.93 here."), CARD)
    msg = verify.retry_message(rep)
    assert "0.93" in msg and "summary" in msg and "ONLY numbers printed" in msg


# ---------------------------------------------------------------- card
def test_card_from_real_compare(tmp_path):
    runs = sorted(Path(__file__).resolve().parents[1].glob("out/*/compare.json"))
    if not runs:
        pytest.skip("no out/*/compare.json")
    cmp = json.loads(runs[-1].read_text())
    text = card.render_card(cmp)
    assert card.approx_tokens(text) <= card.MAX_TOKENS
    for t in cmp["tiers"]:
        assert f"\n{t} (" in text and f"): status=" in text
        if cmp["tier_info"][t]["status"] in ("stub", "fallback"):
            assert f"{cmp['tier_info'][t]['status'].upper()} / PLACEHOLDER" in text
    assert "SWEEP KNEES" in text and "MATERIAL DIVERGENCES" in text


@pytest.mark.skipif(not GGUF.exists() or not _llama_ok(), reason="needs the Qwen tokenizer")
def test_card_budget_with_real_qwen_tokenizer(cards):
    from llama_cpp import Llama
    tok = Llama(model_path=str(GGUF), vocab_only=True, verbose=False)
    texts = [card.render_card(cmp) for _, cmp in cards.values()]
    for p in sorted(Path(__file__).resolve().parents[1].glob("out/*/compare.json")):
        texts.append(card.render_card(json.loads(p.read_text())))
    for t in texts:
        assert len(tok.tokenize(t.encode())) <= card.MAX_TOKENS


def test_card_budget_and_sections(cards):
    for sid, (sc, cmp) in cards.items():
        text = card.render_card(cmp)
        assert card.approx_tokens(text) <= card.MAX_TOKENS, sid
        assert "METRICS" in text and "omits:" in text and "models:" in text
        assert "MATERIAL DIVERGENCES" not in card.render_card(cmp, rank=False)


def test_card_flags_stub_and_fallback(cards):
    assert "STUB / PLACEHOLDER" in card.render_card(cards["ctrl2"][1])
    assert "FALLBACK / PLACEHOLDER" in card.render_card(cards["iso15"][1])
    assert "PLACEHOLDER" not in card.render_card(cards["adc6"][1])


def test_card_truncation_drops_whole_lines(cards):
    text = card.render_card(cards["adc6"][1], max_chars=900)
    assert len(text) <= 900 and not text.endswith(("SWEEP KNEES", "WARNINGS", "\n"))  # no dangling section header
    full = card.render_card(cards["adc6"][1])
    assert full.startswith(text)


def test_divergences_pure_and_sorted(cards):
    d = card.divergences(cards["dil"][1])
    assert d and all(r["tier"] == "R" for r in d)
    sev = [r["severity"] for r in d]
    assert sev == sorted(sev, reverse=True)
    assert card.divergences(cards["ctrl1"][1]) == []


def test_all_numbers_in_card_verify_against_itself(cards):
    text = card.render_card(cards["vsat"][1])
    nums = verify.numbers_in_card(text)
    assert all(verify.number_ok(n, nums) for n in nums)


# ---------------------------------------------------------------- synthetic cards, schema, scoring
def test_synth_cards_validate_and_cover_injections(cards):
    assert len(cards) == 8
    tiers = {sid: synth.expected(sc, cmp)["tier"] for sid, (sc, cmp) in cards.items()}
    assert tiers == {"adc6": "E", "loss30": "C", "iso15": "R", "vsat": "C", "dil": "R",
                     "ctrl1": None, "ctrl2": None, "ctrl3": None}
    for sid, (sc, cmp) in cards.items():
        contract.validate_compare(cmp)
        exp = synth.expected(sc, cmp)
        assert (exp["metrics"] == []) == (sc.tier is None), sid


def test_output_schema_restricts_ids(cards):
    cmp = cards["adc6"][1]
    schema = prompt.schema_for_compare(cmp)
    item = schema["properties"]["findings"]["items"]["properties"]
    assert item["tiers"]["items"]["enum"] == ["V", "E", "C", "R"]
    assert set(item["metric_id"]["enum"]) <= set(contract.METRIC_IDS)
    assert "nmse" not in item["metric_id"]["enum"]  # regression-only metric absent on a modrec card


def test_system_prompt_contains_the_rules():
    s = prompt.SYSTEM_PROMPT
    assert "ONLY numbers" in s and "not modelled by" in s and "status=stub" in s and "never call it a stub" in s and "metric" in s


def _fake_out(findings, summary="x."):
    return {"summary": summary, "findings": findings, "caveats": [], "questions": [],
            "summary_verified": True, "caveats_verified": [], "questions_verified": [],
            "verification": {"n_numbers": 0, "n_unverified_raw": 0, "n_unverified_final": 0, "retried": False}}


def test_scoring_injected_and_controls(cards):
    sc, cmp = cards["adc6"]
    text = card.render_card(cmp)
    exp = synth.expected(sc, cmp)
    good = _fake_out([{"metric_id": "top1_agreement_vs_virtual", "tiers": ["E"], "observation": "o.", "likely_cause": "c."}])
    assert llm_eval.score_card(sc, exp, cmp, good, text)["correct"]
    wrong_tier = _fake_out([{"metric_id": "top1_agreement_vs_virtual", "tiers": ["R"], "observation": "o.", "likely_cause": "c."}])
    assert not llm_eval.score_card(sc, exp, cmp, wrong_tier, text)["correct"]
    extra = _fake_out([{"metric_id": "accuracy", "tiers": ["E", "C"], "observation": "o.", "likely_cause": "c."}])
    assert not llm_eval.score_card(sc, exp, cmp, extra, text)["correct"]
    wrong_metric = _fake_out([{"metric_id": "macro_f1", "tiers": ["E"], "observation": "o.", "likely_cause": "c."}])
    assert llm_eval.score_card(sc, exp, cmp, wrong_metric, text)["correct"]  # flagged as material for E
    csc, ccmp = cards["ctrl1"]
    ctext = card.render_card(ccmp)
    cexp = synth.expected(csc, ccmp)
    calm = _fake_out([{"metric_id": "accuracy", "tiers": ["E"], "observation": "E matches V.", "likely_cause": "none."}],
                     summary="All tiers agree with V; no material divergence.")
    assert llm_eval.score_card(csc, cexp, ccmp, calm, ctext)["correct"]
    alarm = _fake_out([{"metric_id": "accuracy", "tiers": ["E"], "observation": "E shows a large degradation.",
                        "likely_cause": "ADC."}])
    assert not llm_eval.score_card(csc, cexp, ccmp, alarm, ctext)["correct"]


def test_claims_divergence_negation():
    assert llm_eval.claims_divergence("E shows a significant drop.")
    assert not llm_eval.claims_divergence("There is no significant drop.")
    assert not llm_eval.claims_divergence("All tiers agree with V.")


def test_exposed_numbers_counted(cards):
    sc, cmp = cards["adc6"]
    text = card.render_card(cmp)
    exp = synth.expected(sc, cmp)
    o = _fake_out([{"metric_id": "accuracy", "tiers": ["E"], "observation": "Accuracy is 0.777.", "likely_cause": "c."}])
    o["findings"][0]["verified"] = True   # lying flag -> exposed
    assert llm_eval.score_card(sc, exp, cmp, o, text)["numbers"]["exposed_as_verified"] == 1
    o["findings"][0]["verified"] = False
    assert llm_eval.score_card(sc, exp, cmp, o, text)["numbers"]["exposed_as_verified"] == 0


# ---------------------------------------------------------------- backends
def test_null_backend_is_template(cards):
    cmp = cards["ctrl1"][1]
    out = backend.get_backend("none").comment(cmp, card.render_card(cmp))
    assert out["source"] == "template" and out["backend"] == "null"
    contract.validate_commentary(out)


def test_get_backend_unknown():
    with pytest.raises(ValueError):
        backend.get_backend("gpt")


class _Scripted(backend.GuardedBackend):
    """Guard logic with scripted 'model' answers (no weights needed)."""
    name = "scripted"
    model_name = "scripted"

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def _generate(self, messages, schema):
        self.calls.append(messages)
        return json.dumps(self.answers.pop(0)), {"completion_tokens": 1, "decode_tok_per_s": 1.0}


def _ans(obs):
    return {"summary": "E differs.", "findings": [{"metric_id": "top1_agreement_vs_virtual", "tiers": ["E"],
                                                   "observation": obs, "likely_cause": "ADC"}],
            "caveats": [], "questions": []}


def test_guard_retry_fixes_numbers(cards):
    cmp = cards["adc6"][1]
    text = card.render_card(cmp)
    be = _Scripted([_ans("Agreement is 0.612."), _ans("Agreement is 0.905.")])
    out = be.comment(cmp, text)
    assert out["source"] == "model" and out["numbers_verified"] is True
    assert out["verification"]["retried"] and out["verification"]["n_unverified_raw"] == 1
    assert out["verification"]["n_unverified_final"] == 0
    assert "0.612" in be.calls[1][-1]["content"]          # mismatch list appended to the retry
    assert out["raw_first_answer"] is not None
    contract.validate_commentary(out)


def test_guard_flags_after_failed_retry(cards):
    cmp = cards["adc6"][1]
    text = card.render_card(cmp)
    be = _Scripted([_ans("Agreement is 0.612."), _ans("Agreement is 0.333.")])
    out = be.comment(cmp, text)
    assert out["numbers_verified"] is False and out["findings"][0]["verified"] is False
    assert out["verification"]["n_unverified_final"] == 1 and len(be.calls) == 2   # exactly one retry


def test_guard_no_retry_when_clean(cards):
    cmp = cards["adc6"][1]
    be = _Scripted([_ans("Agreement is 0.905.")])
    out = be.comment(cmp, card.render_card(cmp))
    assert len(be.calls) == 1 and out["numbers_verified"] is True and not out["verification"]["retried"]


# ---------------------------------------------------------------- real model (skipped without GGUF / wheel)
@pytest.mark.skipif(not GGUF.exists() or not _llama_ok(), reason="Qwen GGUF or llama-cpp-python absent")
def test_qwen_backend_end_to_end(cards):
    cmp = cards["adc6"][1]
    text = card.render_card(cmp)
    be = backend.get_backend("qwen")
    out = be.comment(cmp, text)
    contract.validate_commentary(out)
    assert out["source"] == "model" and out["backend"] == "qwen"
    assert out["llm"]["model_sha256"] and out["llm"]["prompt_sha256"] and out["llm"]["perf"]["n_threads"] == 2
    ids = card.card_ids(cmp)
    for f in out["findings"]:
        assert f["metric_id"] in ids["metric_ids"] and set(f["tiers"]) <= set(ids["tier_ids"])
    # whatever the model wrote, flags must be consistent with an independent re-verification
    rep = verify.verify_output(out, text, ids["tier_ids"], ids["metric_ids"])
    assert out["numbers_verified"] == rep["ok"] or out["verification"]["n_failures_final"] == rep["n_failures"]
