"""Web exporters: model.npz (schema dlnn-model/1) and dataset examples -> JSON for the browser widgets.

General by construction (any kind/layer count/readout kind); later workers extend it to every model.

Model JSON ("dlnn-web-model/1"): per layer ``W`` = int8 [O,I,K] base64 (row-major), ``dilation``, ``gain``, ``bias``
(float32 base64 [O]), ``vsat``; readout ``readout_W`` [n_out,n_feat] / ``readout_b`` (float32 base64), ``readout_from``,
``readout_kind``, ``input_scale``, ``input_fullscale``, ``meta``, ``sha256`` (Model.sha256()).

Examples JSON ("dlnn-web-examples/1"): ``x`` float32 base64 [N,C,T], ``y``, ``snr``, ``logits`` float32 base64 [N,n_out]
(per_step readout: [N,n_out,T]) and ``probs`` from the numpy virtual tier (tier V), optional per-layer outputs.

Arena ("dlnn-arena-models/1", ``--arena``): all models of the live arena (the DLNN plus the conventional competitors of
``models/arena/*.npz``, schemas documented in docs/model_export.md) in one JSON, with a shared 40-example set whose expected
logits were computed here in numpy (``reference_logits``, an independent re-implementation of each forward pass, itself
cross-checked against torch in tests/test_export_web.py) and the per-class feature statistics of the Python signal
generator (``examples_gen_stats.json``) used to check the JS signal generator statistically.

CLI:  python -m dlnn.viz.export_web --model models/x.npz --task modrec --n 40 --out docs/artifacts/data
      python -m dlnn.viz.export_web --p2 --out docs/artifacts/data        (p_tier.json, scale_story.json, board_cost.json)
      python -m dlnn.viz.export_web --arena --out docs/artifacts/data     (writes arena_models.json, examples_arena.json,
                                                                            examples_gen_stats.json, energy_refs.json copy)
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from ..model.npz import Model, load_model
from ..tiers import virtual

MODEL_SCHEMA = "dlnn-web-model/1"
EXAMPLES_SCHEMA = "dlnn-web-examples/1"


def b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode("ascii")


def model_to_json(model: Model, name: str = "") -> dict:
    layers = []
    for ly in model.layers:
        O, I, K = ly.W.shape
        layers.append({"O": int(O), "I": int(I), "K": int(K), "dilation": int(ly.dilation), "vsat": float(ly.vsat),
                       "W": b64(ly.W, np.int8), "gain": b64(ly.gain, np.float32), "bias": b64(ly.bias, np.float32)})
    return {"schema": MODEL_SCHEMA, "name": name, "kind": model.kind, "sha256": model.sha256(),
            "n_layers": len(layers), "layers": layers,
            "readout_W_shape": list(model.readout_W.shape), "readout_W": b64(model.readout_W, np.float32),
            "readout_b": b64(model.readout_b, np.float32), "readout_from": [int(i) for i in model.readout_from],
            "readout_kind": model.readout_kind, "input_scale": float(model.input_scale),
            "input_fullscale": None if model.input_fullscale is None else float(model.input_fullscale),
            "meta": model.meta}


def examples_to_json(model: Model, x: np.ndarray, y: np.ndarray, snr: np.ndarray | None = None,
                     classes: list[str] | None = None, with_layers: bool = False, name: str = "") -> dict:
    """Run tier V (numpy float32) on x [N,C,T] and package inputs + expected outputs."""
    x = np.asarray(x, dtype=np.float32)
    res = virtual.forward(model, x)
    out = np.asarray(res["out"], dtype=np.float32)
    ex = {"schema": EXAMPLES_SCHEMA, "name": name, "model_sha256": model.sha256(), "n": int(x.shape[0]),
          "shape": list(x.shape[1:]), "x": b64(x, np.float32), "y": [int(v) for v in y],
          "logits_shape": list(out.shape[1:]), "logits": b64(out, np.float32), "classes": classes,
          "pred": [int(v) for v in out.argmax(1)] if out.ndim == 2 else None}
    if snr is not None:
        ex["snr"] = [float(v) for v in snr]
    if with_layers:
        ex["layers"] = [{"shape": list(l.shape[1:]), "y": b64(l, np.float32)} for l in res["layers"]]
    ex["sha256"] = hashlib.sha256(json.dumps({k: ex[k] for k in ("x", "logits")}, sort_keys=True).encode()).hexdigest()
    return ex


def write_json(obj: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, separators=(",", ":")))
    return path


def modrec_examples(model: Model, n: int = 40, with_layers: bool = False, name: str = "") -> dict:
    """The stratified n-example subset of the modrec test set (same selection as the pipeline tiers)."""
    from ..data import modrec
    from ..pipeline import stratified_subset
    te = modrec.generate(4000, modrec.SEED + 1)         # identical to modrec.dataset()'s test half
    idx = stratified_subset(te["y"], te["snr"], n)
    return examples_to_json(model, te["x"][idx], te["y"][idx], te["snr"][idx], list(modrec.CLASSES), with_layers, name)


def export_modrec(model_npz: str | Path, out_dir: str | Path, n: int = 40, with_layers: bool = False) -> dict[str, Path]:
    model_npz, out_dir = Path(model_npz), Path(out_dir)
    stem = model_npz.stem
    model = load_model(model_npz)
    return {"model": write_json(model_to_json(model, stem), out_dir / f"model_{stem}.json"),
            "examples": write_json(modrec_examples(model, n, with_layers, stem), out_dir / f"examples_{stem}.json")}


# ----------------------------------------------------------------------------- arena export
ARENA_SCHEMA = "dlnn-arena-models/1"
ARENA_EXAMPLES_SCHEMA = "dlnn-arena-examples/1"
ARENA_ORDER = ("dlnn", "dlnn_board", "dlnn_fro", "mlp_pow_s", "mlp_s", "cnn", "reservoir", "cumulant", "mlp_l")      # fixed order = fixed colour slot in the page
ARENA_LABELS = {
    "dlnn": ("DLNN", "Ternary delay-line net (TDL-Net-B w8, d=1, ternary readout) on an ideal line"),
    "dlnn_board": ("DLNN-B", "Board-trained DLNN (TDL-Net-B w8, float readout), run on the simulated M1a board line: every tap sees the input through its measured line FIR (Q = 10 corner). Board-trained, on the simulated board line."),
    "dlnn_fro": ("DLNN-F", "Same DLNN as DLNN, but with a float digital readout (what the Nucleo would run), ideal line"),
    "mlp_pow_s": ("Feature MLP", "Fair conventional competitor: MLP (3 hidden units, matched size) on 319 phase-invariant hand-built features (|z|, instantaneous frequency, |FFT| of z, z^2 and z^4)"),
    "mlp_s": ("Raw-I/Q MLP", "MLP on flattened raw I/Q (256 inputs), 4 hidden units"),
    "cnn": ("1-D CNN", "3 causal conv layers 2-8-8-8, ReLU, float"),
    "reservoir": ("Reservoir", "Random ternary delay reservoir + ridge readout"),
    "cumulant": ("Cumulant", "Classical higher-order cumulant features + logistic regression"),
    "mlp_l": ("MLP-L", "MLP on flattened I/Q, 32 hidden units"),
}
ARENA_DLNN_NPZ = "tdlnet_b_w8_ternary.npz"
ARENA_DLNN_FRO_NPZ = "tdlnet_b_w8_fro_ternary.npz"
ARENA_DLNN_BOARD_NPZ = "tdlnet_b_w8_aware2_fro_ternary.npz"       # trained on the M1a line FIR (models/board_aware/line_fir_m1a.json)
ARENA_DLNN_BOARD_REPORT = "tdlnet_b_w8_aware2_fro_report.json"
PROVISIONAL: dict = {}
CUMULANT_FILL = dict(nan=0.0, posinf=1e3, neginf=-1e3)


def _f32(a) -> str:
    return b64(np.asarray(a), np.float32)


def _npz_scalar(z, k):
    v = z[k]
    return v.item() if v.ndim == 0 else v


def arena_entries(models_dir: str | Path) -> dict[str, dict]:
    """id -> {"data": json-able payload, "npz": path}. The payload is what the JS forward passes decode."""
    md = Path(models_dir)
    out: dict[str, dict] = {}
    out["dlnn"] = {"npz": md / ARENA_DLNN_NPZ, "data": model_to_json(load_model(md / ARENA_DLNN_NPZ), "tdlnet_b_w8_ternary")}
    out["dlnn_fro"] = {"npz": md / ARENA_DLNN_FRO_NPZ, "data": model_to_json(load_model(md / ARENA_DLNN_FRO_NPZ), "tdlnet_b_w8_fro_ternary")}
    bm = load_model(md / ARENA_DLNN_BOARD_NPZ)
    fir = bm.meta["arch"]["line_fir"]
    out["dlnn_board"] = {"npz": md / ARENA_DLNN_BOARD_NPZ, "fir": fir, "data": {**model_to_json(bm, "tdlnet_b_w8_aware2_fro_ternary"), "line_fir": fir}}
    for mid in ("mlp_s", "mlp_l", "mlp_pow_s"):
        z = np.load(md / "arena" / f"{mid}.npz", allow_pickle=False)
        n = len([k for k in z.files if k.endswith("_weight")])
        layers = [{"W": _f32(z[f"fc_{i}_weight"]), "shape": list(z[f"fc_{i}_weight"].shape), "b": _f32(z[f"fc_{i}_bias"])} for i in range(n)]
        data = {"schema": "dlnn-arena-mlp/1", "layers": layers}
        if "feature_set" in z.files:
            data = {"schema": "dlnn-arena-mlp-feat/1", "feature_set": str(z["feature_set"]), "in_dim": int(z["in_dim"]), "layers": layers}
        out[mid] = {"npz": md / "arena" / f"{mid}.npz", "data": data}
    z = np.load(md / "arena" / "cnn.npz", allow_pickle=False)
    cv = [{"W": _f32(z[f"cv_{i}_weight"]), "shape": list(z[f"cv_{i}_weight"].shape), "b": _f32(z[f"cv_{i}_bias"])} for i in range(3)]
    out["cnn"] = {"npz": md / "arena" / "cnn.npz", "data": {
        "schema": "dlnn-arena-cnn/1", "conv": cv, "fc_W": _f32(z["fc_weight"]), "fc_shape": list(z["fc_weight"].shape), "fc_b": _f32(z["fc_bias"])}}
    z = np.load(md / "arena" / "reservoir.npz", allow_pickle=False)
    L = int(z["n_layers"])
    layers = [{"O": int(z[f"L{i}_W"].shape[0]), "I": int(z[f"L{i}_W"].shape[1]), "K": int(z[f"L{i}_W"].shape[2]),
               "W": b64(z[f"L{i}_W"], np.int8), "dilation": int(z[f"L{i}_dilation"]), "bias": _f32(z[f"L{i}_bias"]),
               "vsat": float(z[f"L{i}_vsat"])} for i in range(L)]
    out["reservoir"] = {"npz": md / "arena" / "reservoir.npz", "data": {
        "schema": "dlnn-arena-reservoir/1", "layers": layers, "pool": str(z["pool"]), "feat_mean": _f32(z["feat_mean"]),
        "feat_std": _f32(z["feat_std"]), "readout_W": _f32(z["readout_W"]), "readout_shape": list(z["readout_W"].shape),
        "readout_b": _f32(z["readout_b"])}}
    z = np.load(md / "arena" / "cumulant.npz", allow_pickle=False)
    if str(z["classifier"]) != "logreg":
        raise NotImplementedError("cumulant tree classifier not exported to JS")
    out["cumulant"] = {"npz": md / "arena" / "cumulant.npz", "data": {
        "schema": "dlnn-arena-cumulant/1", "classifier": "logreg", "feature_names": [str(s) for s in z["feature_names"]],
        "feat_mean": _f32(z["feat_mean"]), "feat_std": _f32(z["feat_std"]), "W": _f32(z["W"]), "W_shape": list(z["W"].shape), "b": _f32(z["b"])}}
    return out


def reference_logits(mid: str, npz: str | Path, x: np.ndarray) -> np.ndarray:
    """Independent numpy forward pass of arena model ``mid`` on x [N,2,128] -> logits [N,8] float32 (docs/model_export.md)."""
    x = np.asarray(x, dtype=np.float32)
    if mid in ("dlnn", "dlnn_fro"):
        return np.asarray(virtual.forward(load_model(npz), x)["out"], dtype=np.float32)
    if mid == "dlnn_board":      # tier V on the M1a line FIR (circuit_board.v_forward_fir = TDLNet.forward with line_fir)
        from ..tiers.circuit_board import v_forward_fir
        m = load_model(npz)
        return np.asarray(v_forward_fir(m, x, fir=np.asarray(m.meta["arch"]["line_fir"], np.float32))["out"], dtype=np.float32)
    z = np.load(npz, allow_pickle=False)
    if mid == "mlp_pow_s":      # float64 feature extraction (phase-invariant 'pow_compact' set, see dlnn.model.arena.mlp_features)
        zz = (x[:, 0] + 1j * x[:, 1]).astype(np.complex128)
        zz = zz / np.sqrt(np.mean(np.abs(zz) ** 2, 1, keepdims=True))
        sq = np.sqrt(zz.shape[1])
        z2 = zz * zz
        streams = [np.abs(zz), np.angle(zz[:, 1:] * zz[:, :-1].conj()) / (2 * np.pi), np.abs(np.fft.fft(zz, axis=1)) / sq,
                   np.abs(np.fft.fft(z2, axis=1)) / sq, np.abs(np.fft.fft(z2 * z2, axis=1)) / sq]
        pool = lambda a: a[:, :a.shape[1] // 2 * 2].reshape(len(a), -1, 2).mean(2)
        h = np.concatenate([pool(a) for a in streams], 1)
        n = len([k for k in z.files if k.endswith("_weight")])
        for i in range(n):
            h = h @ z[f"fc_{i}_weight"].T + z[f"fc_{i}_bias"]
            if i < n - 1:
                h = np.maximum(h, 0)
        return h.astype(np.float32)
    if mid in ("mlp_s", "mlp_l"):          # float64 reference: removes float32 accumulation-order noise from the 1e-5 check
        x = x.astype(np.float64)
        h = x.reshape(len(x), -1)
        n = len([k for k in z.files if k.endswith("_weight")])
        for i in range(n):
            h = h @ z[f"fc_{i}_weight"].T + z[f"fc_{i}_bias"]
            if i < n - 1:
                h = np.maximum(h, 0)
        return h.astype(np.float32)
    if mid == "cnn":
        x = x.astype(np.float64)
        h = x
        for i in range(3):
            w, b = z[f"cv_{i}_weight"], z[f"cv_{i}_bias"]
            hp = np.pad(h, ((0, 0), (0, 0), (w.shape[2] - 1, 0)))
            T = h.shape[2]
            h = sum(np.einsum("oi,bit->bot", w[:, :, j], hp[:, :, j:j + T]) for j in range(w.shape[2])) + b[None, :, None]
            h = np.maximum(h, 0)
        return (h.mean(2) @ z["fc_weight"].T + z["fc_bias"]).astype(np.float32)
    if mid == "reservoir":
        from ..model.npz import Layer
        h, fs = x, []
        for i in range(int(z["n_layers"])):
            W = z[f"L{i}_W"]
            ly = Layer(W=W, dilation=int(z[f"L{i}_dilation"]), gain=np.ones(W.shape[0], np.float32), bias=z[f"L{i}_bias"], vsat=float(z[f"L{i}_vsat"]))
            h = virtual.layer_forward(h, ly)
            fs.append(h.mean(2))
            if str(z["pool"]) == "mean_sq":
                fs.append((h ** 2).mean(2))
        F_ = (np.concatenate(fs, 1) - z["feat_mean"]) / z["feat_std"]
        return (F_ @ z["readout_W"].T + z["readout_b"]).astype(np.float32)
    if mid == "cumulant":
        from ..model.arena import features
        F_ = np.nan_to_num(features(x), **CUMULANT_FILL)
        return (((F_ - z["feat_mean"]) / z["feat_std"]) @ z["W"] + z["b"]).astype(np.float32)
    raise KeyError(mid)


def gen_stats(n_per_class: int = 400, seed: int = 777) -> dict:
    """Per-class mean/std/n of the 25 cumulant-baseline features on Python-generated windows (SNR uniform over the
    training set, balanced): the reference for the statistical check of the JS signal generator."""
    from ..data import modrec
    from ..model.arena import FEATURE_NAMES, features
    d = modrec.generate(n_per_class * len(modrec.CLASSES), seed)          # balanced by construction
    F_ = features(d["x"])
    stats = {}
    for c, name in enumerate(modrec.CLASSES):
        f = F_[d["y"] == c]
        stats[name] = {"n": int(len(f)), "mean": [float(v) for v in f.mean(0)], "std": [float(v) for v in f.std(0)]}
    return {"schema": "dlnn-gen-stats/1", "seed": seed, "feature_names": list(FEATURE_NAMES), "snrs": list(modrec.SNRS),
            "dataset": "dlnn.data.modrec.generate(n, seed)", "per_class": stats}


def export_arena(models_dir: str | Path, out_dir: str | Path, n: int = 40) -> dict[str, Path]:
    import shutil
    from ..data import modrec
    from ..pipeline import stratified_subset
    models_dir, out_dir = Path(models_dir), Path(out_dir)
    ents = arena_entries(models_dir)
    rep = json.loads((models_dir / "arena" / "arena_report.json").read_text())
    te = modrec.generate(4000, modrec.SEED + 1)
    idx = stratified_subset(te["y"], te["snr"], n)
    x, y, snr = te["x"][idx], te["y"][idx], te["snr"][idx]
    models, logits = [], {}
    for mid in ARENA_ORDER:
        e = ents[mid]
        r = rep["models"][{"dlnn": "tdlnet_b_w8", "dlnn_fro": "tdlnet_b_w8_fro", "dlnn_board": "tdlnet_b_w8_fro"}.get(mid, mid)]
        if mid == "dlnn_board":      # same architecture and ops as DLNN-F; accuracy on the M1a line (Q10 corner), from its own report
            br = json.loads((models_dir / ARENA_DLNN_BOARD_REPORT).read_text())["cells"]["tdlnet_b_w8_aware2_fro|Q10"]
            r = {**r, "acc": br["acc"], "acc_ge10dB": br["acc_ge10dB"], "acc_per_class": br["acc_per_class"]}
        short, desc = ARENA_LABELS[mid]
        models.append({"id": mid, "label": short, "desc": desc, "source": str(e["npz"].relative_to(models_dir.parent)) if models_dir.parent in e["npz"].parents else e["npz"].name,
                       "stand_in": False, "provisional": PROVISIONAL.get(mid), "params": r["params"], "ops": r["ops"], "acc": r["acc"], "acc_ge10dB": r["acc_ge10dB"],
                       "acc_per_class": r.get("acc_per_class"), "data": e["data"]})
        logits[mid] = _f32(reference_logits(mid, e["npz"], x))
    am = {"schema": ARENA_SCHEMA, "classes": list(modrec.CLASSES), "window": modrec.NSAMP, "sps": modrec.SPS,
          "dataset_sha256": rep.get("dataset_sha256"), "ops_convention": rep.get("ops_convention"), "models": models}
    ex = {"schema": ARENA_EXAMPLES_SCHEMA, "n": int(n), "shape": [2, modrec.NSAMP], "classes": list(modrec.CLASSES),
          "x": b64(x, np.float32), "y": [int(v) for v in y], "snr": [float(v) for v in snr], "logits": logits}
    paths = {"models": write_json(am, out_dir / "arena_models.json"), "examples": write_json(ex, out_dir / "examples_arena.json"),
             "gen_stats": write_json(gen_stats(), out_dir / "examples_gen_stats.json")}
    er = models_dir / "arena" / "energy_refs.json"
    if er.exists():
        shutil.copyfile(er, out_dir / "energy_refs.json")
        paths["energy_refs"] = out_dir / "energy_refs.json"
    return paths


# ----------------------------------------------------------------------------- P2 data (W5, W6, W7, W9)
P2_RUN = Path("out/P2-M2b/tdlnet_b_w8_aware2_fro")


def export_p2(root: str | Path, out_dir: str | Path) -> dict[str, Path]:
    """p_tier.json (tier-P Monte Carlo, gain-trim sweep, nominal), scale_story.json (tiles/inductors/area/cost from hw/board/cost.json
    plus accuracy of w8 / w16 / v32 from the model reports) and board_cost.json (BOM/cost summary). All values are read from the source files."""
    root, out_dir = Path(root), Path(out_dir)
    rd = lambda p: json.loads((root / p).read_text())
    run = root / P2_RUN
    mc = json.loads((run / "tiers" / "P_montecarlo.json").read_text())
    gt = json.loads((run / "sweep_gain_trim_db.json").read_text())["entry"]
    nom = json.loads((run / "nominal.json").read_text())
    ptier = {"schema": "dlnn-p-tier/1", "label": "Simulation only (tier P, lumped-element parasitics + tolerance model). Designed, not built, nothing measured.",
             "source": str(P2_RUN), "model": "tdlnet_b_w8_aware2_fro",
             "monte_carlo": {k: mc[k] for k in ("n_trials", "accuracy_mean", "accuracy_std", "accuracy_p5", "accuracy_min", "accuracy_max", "yield", "yield_threshold",
                                                "yield_rule", "n_examples", "v_accuracy_same_subset")} |
                       {"accuracies": [a for f in sorted(run.glob("mc_*.json")) for a in json.loads(f.read_text())["accuracy"]]},
             "gain_trim": {"axis": "gain_trim_db", "values": gt["values"], "accuracy": gt["accuracy"], "subset": "stratified subset (sweep run)",
                           "nominal_auto_db": 8.98, "nominal_auto_source": "docs/parasitics_model.md finding F1 (M1a normalisation, mean on-lag tap gain)",
                           "nominal_accuracy_full_test": nom["Q10"]["accuracy"], "nominal_n_examples": nom["n_examples"],
                           "as_designed_0dB_full_test": nom["Q10_as_designed_trim0dB"]["accuracy"]}}
    cost = rd("hw/board/cost.json")
    reps = {"w8": ("models/tdlnet_b_w8_report.json", "W8_D3"), "w16": ("models/tdlnet_b_w16_report.json", "W16_D3"), "v32": ("models/v32_modrec_report.json", "W32_D4")}
    pts = []
    for k, (f, key) in reps.items():
        r = rd(f)
        pts.append({"id": k, "width": r["width"], "depth": r["recipe"]["depth"], "scaling_key": key, "acc": r["ternary"]["acc"], "acc_ge10dB": r["ternary"]["acc_ge10dB"],
                    "n_ternary_weights": r["ternary"]["n_ternary_weights"], "source": f})
    story = {"schema": "dlnn-scale-story/1", "label": cost["LABEL"], "scaling_formula": cost["scaling_formula"], "scaling": cost["scaling"],
             "inductor_only_cost_usd_at_2000_price": cost["inductor_only_cost_usd_at_2000_price"], "marginal_cost_per_tile_usd": cost["marginal_cost_per_tile_usd_at_volume_ex_fixed_fees"],
             "accuracy_points": pts, "source": ["hw/board/cost.json", *[v[0] for v in reps.values()]]}
    demo = cost["demonstrator_width8_3tiles"]
    bc = {"schema": "dlnn-board-cost/1", "label": cost["LABEL"], "tiles": [{k: t[k] for k in ("tag", "parts_usd", "joints", "unique_parts")} for t in demo["tiles"]],
          "tile_pcb_usd": demo["tile_pcb_usd_for_5_boards_(3_used)"], "tile_assembly_fees_usd": demo["tile_assembly_fees_usd"],
          "carrier": {k: v for k, v in demo["carrier"].items() if not isinstance(v, (list, dict))}, "nucleo_cables_supply_usd": demo["nucleo_cables_supply_usd"],
          "system_total_usd": demo["system_total_usd"], "system_total_usd_with_25pct_contingency": demo["system_total_usd_with_25pct_contingency"],
          "plan_claim_usd": demo["plan_claim_usd"], "pcb_assumptions": cost["pcb_assumptions"]["note"]}
    return {"p_tier": write_json(ptier, out_dir / "p_tier.json"), "scale_story": write_json(story, out_dir / "scale_story.json"), "board_cost": write_json(bc, out_dir / "board_cost.json")}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", help="model.npz (dlnn-model/1)")
    ap.add_argument("--arena", action="store_true", help="export the arena models + shared examples + generator stats")
    ap.add_argument("--p2", action="store_true", help="export p_tier.json, scale_story.json, board_cost.json (W5, W6, W7, W9)")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--task", default="modrec", choices=["modrec"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--layers", action="store_true", help="also export per-layer V outputs")
    ap.add_argument("--out", default="docs/artifacts/data")
    a = ap.parse_args(argv)
    if a.p2:
        for k, p in export_p2(".", a.out).items():
            print(f"{k}: {p} ({p.stat().st_size} bytes)")
        if not a.arena:
            return
    if a.arena:
        for k, p in export_arena(a.models_dir, a.out, a.n).items():
            print(f"{k}: {p} ({p.stat().st_size} bytes)")
        return
    if not a.model:
        ap.error("--model is required unless --arena")
    for k, p in export_modrec(a.model, a.out, a.n, a.layers).items():
        print(f"{k}: {p} ({p.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
