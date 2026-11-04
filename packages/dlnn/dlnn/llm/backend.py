"""LLM commentary backends. The LLM only comments on compare.json; it is never a reference model.

`Backend.comment(compare, card)` must return a dict valid under contract.COMMENTARY_SCHEMA.
`source` is "template" for NullBackend (never presented as model output) and "model" for real LLMs.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from .. import contract
from . import card as card_mod, prompt as prompt_mod, verify as verify_mod

MODELS_DIR = Path(__file__).resolve().parents[2] / "models" / "llm"
QWEN_GGUF = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
BITNET_GGUF = "ggml-model-i2_s.gguf"


class BackendUnavailable(RuntimeError):
    """Raised by get_backend() when a model backend cannot run (package or weights missing)."""


class Backend:
    name = "base"

    def comment(self, compare: dict[str, Any], card: str) -> dict[str, Any]:  # pragma: no cover
        raise NotImplementedError

    def _wrap(self, source: str, body: dict[str, Any]) -> dict[str, Any]:
        out = {"schema_version": contract.COMMENTARY_SCHEMA_VERSION, "source": source,
               "backend": self.name, **body}
        return contract.validate_commentary(out)


class NullBackend(Backend):
    """Deterministic template commentary. Labelled source="template"; no model is involved."""
    name = "null"

    def comment(self, compare: dict[str, Any], card: str) -> dict[str, Any]:
        mm, ti, dv = compare["metric_matrix"], compare["tier_info"], compare["deltas_vs_virtual"]
        prim = "nmse" if compare["task"] == "narma10" else "accuracy"
        row = mm.get(prim, {})
        parts = ", ".join(f"{t}={row[t]:.4g}" for t in row)
        stubs = [t for t in compare["tiers"] if ti[t]["status"] in ("stub", "fallback")]
        findings = []
        for t, d in dv.get(prim, {}).items():
            findings.append({"metric_id": prim, "tiers": [t],
                             "observation": f"{prim} for tier {t} is {row[t]:.4g} versus {row['V']:.4g} for V (delta {d:+.3g}).",
                             "likely_cause": ("Placeholder stub (V plus input noise): the difference reflects only "
                                              "the injected noise, not circuit/RF physics." if t in stubs else
                                              "Fixed-point quantisation (ADC bits, LUT tanh, rounding) in the emulated tier."
                                              if t == "E" else "Not determined by the template.")})
        sv = "nmse_vs_virtual" if "nmse_vs_virtual" in mm else "mean_kl_virtual_vs_tier"
        if sv in mm:
            findings.append({"metric_id": sv, "tiers": [t for t in mm[sv] if t != "V"],
                             "observation": "Distance to V per tier: " + ", ".join(f"{t}={v:.4g}" for t, v in mm[sv].items() if t != "V") + ".",
                             "likely_cause": "Template statement only; no causal analysis performed."})
        body = {"summary": f"[TEMPLATE, not model output] Task {compare['task']}, {prim}: {parts}.",
                "findings": findings,
                "caveats": ([f"Tiers {', '.join(stubs)} are stubs (V plus noise), not SPICE/RF simulations."] if stubs else [])
                + ["This text was generated from a fixed template; no language model was run."],
                "questions": ["Does the gap between E and V shrink with more ADC bits (see the adc_bits sweep)?"]}
        return self._wrap("template", body)


def _file_sha256(path: Path) -> str:
    """sha256 of a (large) model file, cached in a sidecar ``<file>.sha256``."""
    side = path.with_name(path.name + ".sha256")
    if side.exists():
        return side.read_text().split()[0]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    try:
        side.write_text(f"{h.hexdigest()}  {path.name}\n")
    except OSError:
        pass
    return h.hexdigest()


class GuardedBackend(Backend):
    """Shared logic of real-model backends: generate -> parse -> verify numbers/ids -> ONE retry -> flag.

    Subclasses implement ``_generate(messages, schema) -> (text, perf)``.  The LLM only sees the card; the guard
    compares every number it writes with the card.  ``source`` is always "model" (never "template").
    """
    model_name = "?"
    model_file: Path | None = None
    seed = 1234
    temperature = 0.1

    def _generate(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError  # pragma: no cover

    @staticmethod
    def _parse(text: str) -> dict[str, Any] | None:
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            a, b = text.find("{"), text.rfind("}")
            if a < 0 or b <= a:
                return None
            try:
                obj = json.loads(text[a:b + 1])
            except json.JSONDecodeError:
                return None
        if not isinstance(obj, dict):
            return None
        obj.setdefault("summary", "")
        for k in ("findings", "caveats", "questions"):
            obj.setdefault(k, [])
        obj["findings"] = [f for f in obj["findings"] if isinstance(f, dict)]
        for f in obj["findings"]:
            f.setdefault("metric_id", "")
            f.setdefault("tiers", [])
            f.setdefault("observation", "")
            f.setdefault("likely_cause", "")
        return obj

    def comment(self, compare: dict[str, Any], card: str) -> dict[str, Any]:
        ids = {"metric_ids": [m for m in contract.METRIC_IDS if m in compare["metric_matrix"]],
               "tier_ids": list(compare["tiers"])}
        schema = prompt_mod.output_schema(ids["metric_ids"], ids["tier_ids"])
        messages = prompt_mod.build_messages(card)
        psha = prompt_mod.prompt_sha256(messages, schema)
        kw = {"expected_tiers": sorted({r["tier"] for r in card_mod.divergences(compare)}) if "MATERIAL DIVERGENCES" in card else None,
              "tier_status": {t: compare["tier_info"][t]["status"] for t in compare["tiers"]}}
        raw_text, perf = self._generate(messages, schema)
        first = self._parse(raw_text) or {"summary": "", "findings": [], "caveats": [], "questions": []}
        rep1 = verify_mod.verify_output(first, card, ids["tier_ids"], ids["metric_ids"], **kw)
        final, rep, retried, perf2 = first, rep1, False, None
        if not rep1["ok"]:
            retried = True
            msgs2 = messages + [{"role": "assistant", "content": raw_text},
                                {"role": "user", "content": verify_mod.retry_message(rep1)}]
            text2, perf2 = self._generate(msgs2, schema)
            second = self._parse(text2)
            if second is not None:
                rep2 = verify_mod.verify_output(second, card, ids["tier_ids"], ids["metric_ids"], **kw)
                # keep the retry unless it is worse than the first answer
                if rep2["n_failures"] <= rep1["n_failures"]:
                    final, rep = second, rep2
        marked = verify_mod.mark_unverified(final, rep)
        marked["verification"] = {
            "n_numbers": rep["n_numbers"], "retried": retried,
            "n_unverified_raw": rep1["n_unverified_numbers"], "n_failures_raw": rep1["n_failures"],
            "n_unverified_final": rep["n_unverified_numbers"], "n_failures_final": rep["n_failures"],
            "n_id_failures_raw": rep1["n_id_failures"], "n_id_failures_final": rep["n_id_failures"],
            "tolerance": "max(1 % relative, half a unit of the card's printed precision) or correct rounding to >=2 decimals; sign ignored; % accepted"}
        marked["llm"] = {
            "model": self.model_name, "model_file": self.model_file.name if self.model_file else None,
            "model_sha256": _file_sha256(self.model_file) if self.model_file and self.model_file.exists() else None,
            "prompt_sha256": psha, "seed": self.seed, "temperature": self.temperature,
            "decoding": "grammar-constrained JSON schema", "perf": perf,
            "perf_retry": perf2, "note": "commentary on the card only; the LLM ran no simulation"}
        marked["raw_first_answer"] = first if retried else None
        return self._wrap("model", marked)


class QwenBackend(GuardedBackend):
    """Qwen2.5-1.5B-Instruct Q4_K_M through llama-cpp-python (CPU, 2 threads, fixed seed)."""
    name = "qwen"
    model_name = "Qwen2.5-1.5B-Instruct Q4_K_M"

    def __init__(self, model_path: str | Path | None = None, n_threads: int = 2, n_ctx: int = 4096,
                 max_tokens: int = 900):
        self.model_file = Path(model_path or os.environ.get("DLNN_QWEN_GGUF") or MODELS_DIR / QWEN_GGUF)
        if not self.model_file.exists():
            raise BackendUnavailable(f"Qwen GGUF not found at {self.model_file} (see docs/llm_commentary.md)")
        try:
            from llama_cpp import Llama
        except Exception as e:  # ImportError or shared-library load failure
            raise BackendUnavailable(f"llama-cpp-python unusable: {e}") from e
        self.n_threads, self.n_ctx, self.max_tokens = n_threads, n_ctx, max_tokens
        self._llm = Llama(model_path=str(self.model_file), n_ctx=n_ctx, n_threads=n_threads,
                          n_threads_batch=n_threads, seed=self.seed, verbose=False)

    def _generate(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        t0 = time.perf_counter()
        t_first = None
        n_tok, parts = 0, []
        self._llm.reset()
        for chunk in self._llm.create_chat_completion(
                messages=messages, response_format={"type": "json_object", "schema": schema},
                temperature=self.temperature, seed=self.seed, max_tokens=self.max_tokens, stream=True):
            delta = chunk["choices"][0]["delta"].get("content")
            if delta:
                if t_first is None:
                    t_first = time.perf_counter()
                parts.append(delta)
                n_tok += 1
        t1 = time.perf_counter()
        t_first = t_first or t1
        return "".join(parts), {"completion_tokens": n_tok, "total_s": round(t1 - t0, 2),
                                "prefill_s": round(t_first - t0, 2),
                                "decode_tok_per_s": round(max(n_tok - 1, 0) / max(t1 - t_first, 1e-9), 2),
                                "n_threads": self.n_threads}


def get_backend(name: str) -> Backend:
    if name in ("none", "null"):
        return NullBackend()
    if name == "qwen":
        return QwenBackend()
    if name == "bitnet":
        return BitNetBackend()
    raise ValueError(f"unknown LLM backend '{name}' (none | qwen | bitnet)")


class BitNetBackend(GuardedBackend):
    """BitNet b1.58-2B-4T (i2_s GGUF) through bitnet.cpp's ``llama-completion`` binary (subprocess, 2 threads).

    Build (see docs/llm_eval.md): git clone --recursive https://github.com/microsoft/BitNet.git models/llm/bitnet.cpp;
    cmake -B build -DBITNET_X86_TL2=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLAMA_BUILD_TOOLS=ON
    -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_COMMON=ON; cmake --build build --target llama-completion.
    Weights: microsoft/bitnet-b1.58-2B-4T-gguf / ggml-model-i2_s.gguf -> models/llm/bitnet/.
    """
    name = "bitnet"
    model_name = "BitNet b1.58 2B 4T (i2_s)"

    def __init__(self, binary: str | Path | None = None, model_path: str | Path | None = None,
                 n_threads: int = 2, n_ctx: int = 4096, max_tokens: int = 900, timeout_s: int = 1800):
        self.binary = Path(binary or os.environ.get("DLNN_BITNET_BIN")
                           or MODELS_DIR / "bitnet.cpp" / "build" / "bin" / "llama-completion")
        self.model_file = Path(model_path or os.environ.get("DLNN_BITNET_GGUF") or MODELS_DIR / "bitnet" / BITNET_GGUF)
        for f in (self.binary, self.model_file):
            if not f.exists():
                raise BackendUnavailable(f"BitNet backend needs {f} (see docs/llm_eval.md)")
        self.n_threads, self.n_ctx, self.max_tokens, self.timeout_s = n_threads, n_ctx, max_tokens, timeout_s

    @staticmethod
    def _format(messages: list[dict[str, str]]) -> str:
        """BitNet-b1.58-2B-4T chat format: 'System: ..<|eot_id|>User: ..<|eot_id|>Assistant: ..<|eot_id|>'."""
        role = {"system": "System", "user": "User", "assistant": "Assistant"}
        out = "".join(f"{role[m['role']]}: {m['content']}<|eot_id|>" for m in messages)
        return out + "Assistant: "

    def _generate(self, messages: list[dict[str, str]], schema: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        import re
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory(prefix="dlnn_bitnet_") as td:
            (Path(td) / "schema.json").write_text(json.dumps(schema))
            (Path(td) / "prompt.txt").write_text(self._format(messages))
            cmd = [str(self.binary), "-m", str(self.model_file), "-t", str(self.n_threads), "-c", str(self.n_ctx),
                   "-n", str(self.max_tokens), "--temp", str(self.temperature), "--seed", str(self.seed),
                   "-no-cnv", "--no-display-prompt", "--no-warmup", "-jf", str(Path(td) / "schema.json"),
                   "-f", str(Path(td) / "prompt.txt")]
            t0 = time.perf_counter()
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=self.timeout_s, stdin=subprocess.DEVNULL)
            wall = time.perf_counter() - t0
        text = r.stdout.replace("[end of text]", "").strip()
        m = re.search(r"eval time =\s+([\d.]+) ms /\s+(\d+) runs.*?([\d.]+) tokens per second", r.stderr)
        n_tok = int(m.group(2)) if m else 0
        return text, {"completion_tokens": n_tok, "total_s": round(wall, 2),
                      "decode_tok_per_s": float(m.group(3)) if m else None, "n_threads": self.n_threads,
                      "returncode": r.returncode}
