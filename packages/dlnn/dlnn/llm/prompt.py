"""Prompt, output schema and message construction for the commentator LLM."""
from __future__ import annotations

import hashlib
import json
from typing import Any

from .. import contract
from . import card as card_mod

SYSTEM_PROMPT = """You write short commentary on a results CARD that compares four fidelity tiers of a delay-line neural network. You are only a commentator: you do not simulate and you are not the reference model. You never analyse anything beyond the card.
Tiers: V = virtual (ideal math, the reference), E = emulated (fixed-point arithmetic), C = circuit (SPICE-style), R = RF (S-parameter-style). Keep these names straight.

RULES
1. Cite ONLY numbers printed in the card, copied exactly as printed (including printed deltas such as (-0.03)). Never compute new numbers, differences, ratios or percentages. If no printed number fits, make the point without a number.
2. Refer to metrics by exact metric id (for example top1_agreement_vs_virtual) and to tiers by id (V, E, C, R). Use only ids that appear in the card.
3. Under "MATERIAL DIVERGENCES FROM V" each line names a tier and its diverging metrics. Write one finding for every tier on that list: "tiers" = that tier only, "metric_id" = the first metric on its line. Several tiers on the list means several findings. If the list says none, write no findings about divergence and say in the summary that all tiers agree with V.
4. "likely_cause" must be taken from that tier's OWN models/omits lines on the card (under the divergence list they are repeated as "<tier> own models line"; never use another tier's lines) (a bit width, a loss figure, an isolation figure, a bias or V_sat setting, a dilation setting, ...). If nothing there explains it, write "cause not determinable from the card". Never invent a mechanism: write "not modelled by <tier>" for effects the tier does not model.
5. Only a tier whose card line says status=stub or status=fallback is a placeholder: say so in caveats and do not read its numbers as physics. A tier with status=ok is a real simulation; never call it a stub or fallback.
6. Answer with JSON only: summary (at most 2 sentences), findings, caveats (each different; no repeated sentences), questions (follow-up checks, no numbers)."""


def output_schema(metric_ids: list[str] | None = None, tier_ids: list[str] | None = None) -> dict[str, Any]:
    """JSON schema used for grammar-constrained decoding. Ids are restricted to the card's own ids."""
    metric_ids = metric_ids or list(contract.METRIC_IDS)
    tier_ids = tier_ids or list(contract.TIERS)
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "findings": {"type": "array", "maxItems": 6, "items": {
                "type": "object",
                "properties": {"metric_id": {"enum": metric_ids},
                               "tiers": {"type": "array", "minItems": 1, "maxItems": 4,
                                         "items": {"enum": tier_ids}},
                               "observation": {"type": "string"},
                               "likely_cause": {"type": "string"}},
                "required": ["metric_id", "tiers", "observation", "likely_cause"]}},
            "caveats": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
            "questions": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
        },
        "required": ["summary", "findings", "caveats", "questions"],
    }


SYSTEM_PROMPT_NORANK = SYSTEM_PROMPT.replace(
    SYSTEM_PROMPT[SYSTEM_PROMPT.index("3. "):SYSTEM_PROMPT.index("4. ")],
    "3. Compare each tier with V yourself using the METRICS section. Write one finding for every tier whose values "
    "differ clearly from V: \"tiers\" = that tier only, \"metric_id\" = the metric that differs most. If no tier "
    "differs clearly, write no findings about divergence and say in the summary that all tiers agree with V.\n")


def _example_turns(rank: bool = True) -> list[dict[str, str]]:
    from . import synth  # local import: synth is only needed for the in-context example
    ex = card_mod.render_card(synth.example_regression(), rank=rank)
    return [{"role": "user", "content": "CARD\n" + ex + "\n\nWrite the commentary JSON now."},
            {"role": "assistant", "content": json.dumps(synth.EXAMPLE_ANSWER)}]


def build_messages(card: str, example: bool = True) -> list[dict[str, str]]:
    """system rules, one worked example (an invented card with a different task and defect), then the real card.
    The variant without the Python-computed divergence list is chosen from the card text itself."""
    rank = "MATERIAL DIVERGENCES" in card
    msgs = [{"role": "system", "content": SYSTEM_PROMPT if rank else SYSTEM_PROMPT_NORANK}]
    if example:
        msgs += _example_turns(rank)
    return msgs + [{"role": "user", "content": "CARD\n" + card + "\n\nWrite the commentary JSON now."}]


def schema_for_compare(cmp: dict[str, Any]) -> dict[str, Any]:
    ids = card_mod.card_ids(cmp)
    return output_schema(ids["metric_ids"], ids["tier_ids"])


def prompt_sha256(messages: list[dict[str, str]], schema: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(messages, sort_keys=True).encode())
    h.update(json.dumps(schema, sort_keys=True).encode())
    return h.hexdigest()
