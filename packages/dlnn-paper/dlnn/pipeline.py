"""End-to-end pipeline used by run.py and the notebooks.

Stages: dataset + model -> tiers V,E,C,R (C before R so that ``--cascade`` can hand params/circuit_extracted.json
to R) -> float-baseline reference (not a tier) -> compare.json -> figures -> LLM commentary -> database.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import importlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from . import compare, contract
from .llm import backend as llm_backend, card as llm_card
from .viz import figures

ROOT = Path(__file__).resolve().parents[1]
TIER_MODULES = {"V": "virtual", "E": "emulated", "C": "circuit", "R": "rf"}
TIER_ORDER = ("V", "E", "C", "R")
DEFAULT_MODEL = {"modrec": ROOT / "models" / "tdlnet_modrec_w32_ternary.npz"}
# Heavy tiers (C, R) run on a fixed stratified subset of the test set when this is smaller than the test set.
# None = the whole test set: measured on this box (width 32, 4000 examples, load ~3-4), V+E+C+R take about 10 min, well inside
# the 25 min budget, so no subset is needed by default. Use --cr-subset N if the box is slower or the model wider.
DEFAULT_CR_SUBSET: dict[str, int | None] = {"modrec": None}
TOTAL_BUDGET_S = 25 * 60


class ModelUnavailable(RuntimeError):
    pass


def _skipped(tier: str, run_id: str, task: str, msg: str, model_sha: str, ds_sha: str) -> contract.TierResult:
    return contract.TierResult(tier=tier, status="skipped", run_id=run_id, task=task,
                               tool={"name": "none", "version": "0"}, model_sha256=model_sha,
                               dataset_sha256=ds_sha, n_examples=0, warnings=[msg],
                               fidelity={"models": [], "omits": ["tier did not run"]})


def resolve_model(task: str, model: str | Path | None) -> Path:
    p = Path(model) if model else DEFAULT_MODEL.get(task)
    if p is None:
        return Path()
    p = p if p.is_absolute() else (Path.cwd() / p if (Path.cwd() / p).exists() else ROOT / p)
    if not p.exists():
        raise ModelUnavailable(f"{task}: model file {p} not found (train with `python -m dlnn.model.train --task modrec`)")
    return p


def prepare(task: str, run_dir: Path, model: str | Path | None = None, n_test: int | None = None) -> tuple[dict, Path]:
    """Dataset + model.npz for the task (the model is copied/written into run_dir)."""
    if task == "narma10":
        from .data import narma10
        from .model import reservoir
        ds = narma10.dataset()
        m = reservoir.fit(ds)
        return ds, m.save(run_dir / "model.npz")
    if task == "modrec":
        from .data import modrec
        from .model.npz import load_model
        src = resolve_model(task, model)
        try:
            load_model(src)
        except Exception as e:  # noqa: BLE001
            raise ModelUnavailable(f"{src} is not a ternary dlnn-model/1 file ({type(e).__name__}: {e}); "
                                   "the float baseline cannot be run through tiers E/C/R") from e
        ds = modrec.dataset() if n_test is None else modrec.dataset(n_train=40, n_test=int(n_test))
        dst = run_dir / "model.npz"
        shutil.copyfile(src, dst)
        return ds, dst
    raise ValueError(f"unknown task {task}")


def stratified_subset(y: np.ndarray, snr: np.ndarray, n: int) -> np.ndarray:
    """Fixed subset of ``n`` test indices, balanced over the (class, SNR) cells.

    From every cell take its first ``n // n_cells`` examples in test-set order (+1 for the first ``n % n_cells`` cells).
    The test set order is itself a seeded permutation (``dlnn/data/modrec.py``), so no extra randomness is needed and the
    choice is identical on every run and for every tier. Indices are returned sorted, so the leading examples (the ones the
    tiers store traces / the figures show) are the same examples as in the full set.
    """
    cells = [(int(c), float(s)) for c in np.unique(y) for s in np.unique(snr)]
    per, extra = divmod(int(n), len(cells))
    idx: list[int] = []
    for j, (c, s) in enumerate(cells):
        w = np.where((y == c) & (snr == s))[0]
        idx += w[: per + (1 if j < extra else 0)].tolist()
    return np.sort(np.array(idx, dtype=np.int64))


def subset_dataset(ds: dict, idx: np.ndarray) -> dict:
    sub = dict(ds)
    for k in ("x_test", "y_test", "snr_test"):
        if k in sub:
            sub[k] = ds[k][idx]
    sub["sha256"] = hashlib.sha256(f"{ds['sha256']}|subset|{idx.tobytes().hex()}".encode()).hexdigest()
    sub["parent_sha256"] = ds["sha256"]
    return sub


def float_reference(src_model: Path, float_model: str | Path | None, ds: dict, v_pred: np.ndarray) -> dict | None:
    """Accuracy of the float32 baseline (same topology, unconstrained taps) on the full test set.

    A REFERENCE, never a fifth tier: it is not ternary and cannot be pushed through tiers E, C or R.
    """
    from .model import export
    from .tiers import common, virtual
    src = Path(float_model) if float_model else None
    if src is None:
        cand = Path(src_model.name.replace("_ternary", "_float"))
        for base in (ROOT / "models",):
            if (base / cand).exists():
                src = base / cand
    if src is None or not Path(src).exists():
        return None
    fm = export.load_float_model(src)
    out = virtual.forward(fm, ds["x_test"])["out"]
    pred, y = out.argmax(1), ds["y_test"]
    ref = {"label": "float32 baseline (same topology, unconstrained float taps); a REFERENCE, not a fifth tier",
           "is_tier": False, "model_file": str(Path(src).relative_to(ROOT)) if Path(src).is_relative_to(ROOT) else str(src),
           "model_file_sha256": hashlib.sha256(Path(src).read_bytes()).hexdigest(),
           "tool": "dlnn.tiers.virtual.forward (numpy float32)", "n_examples": int(len(y)),
           "accuracy": float(np.mean(pred == y)), "macro_f1": common.macro_f1(y, pred, int(y.max()) + 1),
           "top1_agreement_with_ternary_V": float(np.mean(pred == v_pred)),
           "accuracy_ternary_V": float(np.mean(v_pred == y))}
    if "snr_test" in ds:
        ref["acc_per_snr"] = {str(int(s)): float(np.mean(pred[ds["snr_test"] == s] == y[ds["snr_test"] == s]))
                              for s in np.unique(ds["snr_test"])}
        ref["acc_per_snr_ternary_V"] = {str(int(s)): float(np.mean(v_pred[ds["snr_test"] == s] == y[ds["snr_test"] == s]))
                                        for s in np.unique(ds["snr_test"])}
        hi = ds["snr_test"] >= 10
        ref["accuracy_snr_ge_10db"] = float(np.mean(pred[hi] == y[hi]))
        ref["accuracy_snr_ge_10db_ternary_V"] = float(np.mean(v_pred[hi] == y[hi]))
    return ref


def _loadavg() -> list[float]:
    try:
        return [round(v, 2) for v in os.getloadavg()]
    except OSError:  # pragma: no cover
        return []


def run_pipeline(task: str = "narma10", tiers: str | list[str] = "V,E,C,R", cascade: bool = False,
                 llm: str = "none", db: str = "sqlite", out: str | Path = "out",
                 run_id: str | None = None, quiet: bool = False, *, model: str | Path | None = None,
                 float_model: str | Path | None = None, n_test: int | None = None,
                 cr_subset: int | None | str = "default", pg_dsn: str | None = None,
                 tier_cfg: dict | None = None) -> Path:
    """Run the pipeline and return the run directory.

    model / float_model / n_test / cr_subset apply to ``modrec``. ``cr_subset`` = number of test examples for tiers C
    and R (None or >= test-set size = the full test set; "default" = ``DEFAULT_CR_SUBSET``). ``tier_cfg`` is merged into
    the cfg handed to every tier (e.g. ``{"rtl": False}``).
    """
    t_start = time.time()
    log = (lambda *a: None) if quiet else print
    stage_s: dict[str, float] = {}
    if task == "modrec":
        resolve_model(task, model)      # raises ModelUnavailable before anything is written
    tiers = tiers.split(",") if isinstance(tiers, str) else list(tiers)
    tiers = [t for t in TIER_ORDER if t in tiers]
    if "V" not in tiers:
        tiers.insert(0, "V")  # V is the reference for every comparison
    be = None
    if llm not in ("none", "null"):
        be = llm_backend.get_backend(llm)   # fail before the expensive part if the LLM backend cannot run
    out = Path(out) if Path(out).is_absolute() else ROOT / out
    run_id = run_id or f"{task}-{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    run_dir = out / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    created = dt.datetime.now(dt.timezone.utc).isoformat()
    (run_dir / "created_at.txt").write_text(created)
    load0 = _loadavg()

    t0 = time.time()
    ds, model_npz = prepare(task, run_dir, model, n_test)
    n_full = int(len(ds["y_test"])) if ds["task"] == "classification" else int(ds["y_test"].shape[-1])
    if ds["task"] == "classification":
        np.save(run_dir / "labels.npy", ds["y_test"])
        np.save(run_dir / "snr_test.npy", ds["snr_test"])
    from .model.npz import load_model
    model_sha = load_model(model_npz).sha256()
    stage_s["dataset+model"] = round(time.time() - t0, 2)

    # fixed subset for the heavy tiers (identical for C and R, nested in the full test set that V and E use)
    if cr_subset == "default":
        cr_subset = DEFAULT_CR_SUBSET.get(task)
    subsets: dict = {}
    ds_heavy = ds
    if ds["task"] == "classification" and cr_subset and int(cr_subset) < n_full:
        idx = stratified_subset(ds["y_test"], ds["snr_test"], int(cr_subset))
        ds_heavy = subset_dataset(ds, idx)
        info = {"indices": idx.tolist(), "sha256": hashlib.sha256(idx.tobytes()).hexdigest()}
        subsets = {"n_full": n_full, "policy": (f"stratified: first {len(idx)} // 40 examples of every (class, SNR) cell "
                                                 "in test-set order; same indices for every subset tier"),
                   "tiers": {t: info for t in ("C", "R") if t in tiers}}
        (run_dir / "subsets.json").write_text(json.dumps(subsets))
        log(f"heavy tiers C,R run on a fixed stratified subset of {len(idx)} / {n_full} test examples")

    cfg_base = {"run_id": run_id, "seed": 0, **(tier_cfg or {})}
    warn: list[str] = []
    if cascade and not ({"C", "R"} <= set(tiers)):
        warn.append("--cascade needs both tiers C and R; running without cascade")
        log("warning:", warn[-1])
        cascade = False
    log(f"run {run_id}: model {model_sha[:12]} ({'reservoir fit' if task == 'narma10' else resolve_model(task, model).name}) "
        f"dataset {ds['sha256'][:12]}  load avg {load0}")

    statuses: dict[str, str] = {}
    for t in tiers:
        mod = importlib.import_module(f"dlnn.tiers.{TIER_MODULES[t]}")
        cfg = dict(cfg_base)
        tds = ds_heavy if t in ("C", "R") else ds
        if cascade and t == "R":
            p = run_dir / "params" / "circuit_extracted.json"
            if statuses.get("C") in ("ok", "fallback") and p.exists():
                cfg["cascade_params"] = str(p)
            else:
                warn.append(f"cascade: tier C did not produce {p.name} (status {statuses.get('C')}); R ran without it")
                log("warning:", warn[-1])
        t0 = time.time()
        try:
            res = mod.run(model_npz, tds, cfg, run_dir)
        except Exception as e:  # a failed tier must not stop the pipeline
            traceback.print_exc()
            res = _skipped(t, run_id, ds["name"], f"{type(e).__name__}: {e}", model_sha, ds["sha256"])
        contract.save(res, run_dir)
        statuses[t] = res.status
        stage_s[t] = round(time.time() - t0, 2)
        m = res.metrics.get("nmse", res.metrics.get("accuracy"))
        extra = (f" top1-agree-vs-V={res.metrics['top1_agreement_vs_virtual']:.4f}"
                 if "top1_agreement_vs_virtual" in res.metrics and t != "V" else "")
        log(f"  tier {t}: status={res.status} n={res.n_examples} metric={m if m is None else round(m, 4)}{extra} "
            f"({stage_s[t]:.1f}s){' inputs_from=' + str(res.inputs_from) if res.inputs_from else ''}")

    # float-baseline reference (modrec only)
    if ds["task"] == "classification":
        t0 = time.time()
        try:
            ref = float_reference(resolve_model(task, model), float_model, ds, np.load(run_dir / "tiers" / "V.npz")["pred"])
        except Exception as e:  # noqa: BLE001
            warn.append(f"float-baseline reference failed: {type(e).__name__}: {e}")
            ref = None
        if ref:
            (run_dir / "reference.json").write_text(json.dumps(ref, indent=2))
            log(f"  reference float baseline: accuracy {ref['accuracy']:.4f} (ternary V {ref['accuracy_ternary_V']:.4f})")
        stage_s["reference"] = round(time.time() - t0, 2)

    meta = {"model_file": (str(resolve_model(task, model).relative_to(ROOT)) if resolve_model(task, model).is_relative_to(ROOT)
                           else str(resolve_model(task, model))) if task == "modrec" else "reservoir (fitted per run)",
            "n_test_full": n_full, "cr_subset": (len(next(iter(subsets["tiers"].values()))["indices"])
                                                  if subsets.get("tiers") else None),
            "cascade": bool(cascade), "llm": llm, "db": db, "load_avg_start": load0, "total_budget_s": TOTAL_BUDGET_S}
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    def write_timing() -> None:
        (run_dir / "timing.json").write_text(json.dumps(
            {"stages_s": stage_s, "total_s": round(time.time() - t_start, 2),
             "note": "wall seconds per stage incl. saving; database load is not included"}, indent=2))

    t0 = time.time()
    write_timing()
    compare.write(run_dir)
    cmp = compare.load(run_dir)
    stage_s["compare"] = round(time.time() - t0, 2)
    t0 = time.time()
    try:
        figures.make_all(run_dir)
    except Exception as e:  # noqa: BLE001  figures are not allowed to lose a finished run
        traceback.print_exc()
        warn.append(f"figures failed: {type(e).__name__}: {e}")
    stage_s["figures"] = round(time.time() - t0, 2)

    t0 = time.time()
    (run_dir / "llm").mkdir(exist_ok=True)
    card = llm_card.render_card(cmp)
    (run_dir / "llm" / "card.txt").write_text(card)
    commentary = write_commentary(run_dir, cmp, card, be if be is not None else llm_backend.get_backend("none"))
    stage_s["llm"] = round(time.time() - t0, 2)
    log(f"  llm: backend={commentary['backend']} source={commentary['source']} "
        f"numbers_verified={commentary.get('numbers_verified', 'n/a (template)')} ({stage_s['llm']:.1f}s)")

    args = {"task": task, "tiers": tiers, "cascade": cascade, "llm": llm, "db": db, "warnings": warn,
            "model": meta["model_file"], "cr_subset": meta["cr_subset"], "n_test": n_test}
    (run_dir / "args.json").write_text(json.dumps(args, indent=2))
    stage_s["total_before_db"] = round(time.time() - t_start, 2)
    meta["load_avg_end"] = _loadavg()
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    write_timing()
    compare.write(run_dir)          # final compare.json carries the complete timing block

    t0 = time.time()
    if db == "sqlite":
        from .db import load as dbload
        dbload.load_run(run_dir, run_dir / "dlnn.sqlite", args)
        log(f"  db: sqlite {run_dir / 'dlnn.sqlite'} ({time.time() - t0:.1f}s)")
    elif db == "postgres":
        from .db import load as dbload
        rid = dbload.load_run_pg(run_dir, pg_dsn, args)
        s = dbload.pg_summary(rid, pg_dsn)
        log(f"  db: postgres loaded run {rid}; SELECT counts {({k: v for k, v in s.items() if k != 'tiers'})}; tiers {s['tiers']}")
    (out / "LATEST").write_text(run_id)
    total = time.time() - t_start
    log("timing (s): " + ", ".join(f"{k}={v}" for k, v in stage_s.items()))
    log(f"done in {total:.1f}s ({total / 60:.1f} min; budget {TOTAL_BUDGET_S / 60:.0f} min) -> {run_dir}")
    return run_dir


def write_commentary(run_dir: Path, cmp: dict, card: str, backend: "llm_backend.Backend") -> dict:
    """card -> backend -> (verify inside the backend) -> llm/commentary.json. Returns the commentary dict."""
    commentary = backend.comment(cmp, card)
    (Path(run_dir) / "llm" / "commentary.json").write_text(json.dumps(commentary, indent=2))
    return commentary


def latest_run(out: str | Path = "out") -> Path:
    out = Path(out) if Path(out).is_absolute() else ROOT / out
    return out / (out / "LATEST").read_text().strip()
