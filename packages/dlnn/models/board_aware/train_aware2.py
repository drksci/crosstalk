"""Train tdlnet_b_w8_aware2 on the M1a line FIR (Q corners + tap-delay jitter) and evaluate plain vs aware2 on 3 lines.
Selection (validation only = last 2000 train examples): per variant the seed with the best val macro accuracy (mean of Q10/Q30 lines)
among those whose MIN per-class val accuracy on both corners >= FLOOR; overall winner = best val macro among variants passing the floor.
The test set is reported only.  Usage: train_aware2.py [variant ...]   (default: all)  -> models/board_aware/aware2_search.json, scratch .pt
and, for --ship VARIANT, models/tdlnet_b_w8_aware2_{ternary.npz,ternary.pt,report.json}."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dlnn.data import modrec
from dlnn.model import export, train
from dlnn.model.tdlnet_torch import LineAug, TDLNet

torch.set_num_threads(2)
NAME, SEEDS, NVAL, FLOOR = "tdlnet_b_w8_aware2", (0, 1, 2), 2000, 0.35
J = json.loads((ROOT / "models/board_aware/line_fir_m1a.json").read_text())
FIR = {q: J["corners"][q]["fir"] for q in ("Q10", "Q30")}
LINES = {"ideal": None, "Q10": FIR["Q10"], "Q30": FIR["Q30"]}
CK = ROOT / "models/board_aware/candidates"
VARIANTS = {   # name -> (train_one kwargs, epochs, float_readout, warm start from plain w8)
    "base": (dict(), 60, False, False),
    "focal": (dict(focal_gamma=2.0), 60, False, False),
    "mix": (dict(p_ideal=0.3), 60, False, False),
    "ft": (dict(lr=1e-2, soft_frac=0.0), 20, False, True),
    "ft_mix": (dict(lr=1e-2, soft_frac=0.0, p_ideal=0.3), 20, False, True),
    "ft_focal": (dict(lr=1e-2, soft_frac=0.0, focal_gamma=2.0), 20, False, True),
    "fro": (dict(), 60, True, False),
    "fro_focal": (dict(focal_gamma=2.0), 60, True, False),
}


def evaluate(net, x, y, snr, classes, fir):
    net.set_line_fir(fir); net.eval()
    return train.summarize(train.predict(net, x), y, snr, classes)


def run_variant(ds, name):
    kw, epochs, fro, warm = VARIANTS[name]
    kw = dict(kw); p_ideal = kw.pop("p_ideal", 0.0)
    aug = LineAug([J["corners"]["Q10"]["dense"], J["corners"]["Q30"]["dense"]], J["P"], J["M"], jitter=0.14, p_ideal=p_ideal)
    classes = ds["classes"]
    xtr, ytr = ds["x_train"][:-NVAL], ds["y_train"][:-NVAL]
    xv, yv, sv = ds["x_train"][-NVAL:], ds["y_train"][-NVAL:], ds["snr_train"][-NVAL:]
    init = torch.load(ROOT / "models/tdlnet_b_w8_ternary.pt") if warm else None
    rows = []
    for sd in SEEDS:
        lr, sf = kw.get("lr", 3e-2), kw.get("soft_frac", 0.5)
        net, info = train.train_one(ds, False, epochs, 256, lr, sd, True, False, 1.0, x_tr=xtr, y_tr=ytr, width=8, depth=3,
                                    dilations=(1, 1, 1), line_fir=FIR["Q10"], line_aug=aug, init_state=init, float_readout=fro,
                                    soft_frac=sf, **{k: v for k, v in kw.items() if k not in ("lr", "soft_frac")})
        row = dict(variant=name, seed=sd, seconds=round(info["seconds"], 1))
        for q in ("Q10", "Q30"):
            rv = evaluate(net, xv, yv, sv, classes, FIR[q]); rt = evaluate(net, ds["x_test"], ds["y_test"], ds["snr_test"], classes, FIR[q])
            row[f"val_{q}"] = dict(acc=rv["acc"], macro=float(np.mean(list(rv["acc_per_class"].values()))), min_class=float(min(rv["acc_per_class"].values())),
                                   per_class=rv["acc_per_class"])
            row[f"test_{q}"] = dict(acc=rt["acc"], acc_ge10dB=rt["acc_ge10dB"], per_class=rt["acc_per_class"])   # reporting only
        row["val_macro"] = float(np.mean([row[f"val_{q}"]["macro"] for q in ("Q10", "Q30")]))
        row["val_min_class"] = float(min(row[f"val_{q}"]["min_class"] for q in ("Q10", "Q30")))
        row["passes_floor"] = row["val_min_class"] >= FLOOR
        CK.mkdir(exist_ok=True); torch.save(net.state_dict(), CK / f"{name}_s{sd}.pt")
        print(name, sd, "val macro %.4f min-class %.3f | test Q10 %.4f min %.3f Q30 %.4f" % (
            row["val_macro"], row["val_min_class"], row["test_Q10"]["acc"], min(row["test_Q10"]["per_class"].values()), row["test_Q30"]["acc"]), flush=True)
        rows.append(row)
    return rows


def ship(ds, name, sd, rows):
    kw, epochs, fro, warm = VARIANTS[name]
    classes = ds["classes"]
    net = TDLNet(widths=(2, 8, 8, 8), dilations=(1, 1, 1), float_readout=fro, input_fullscale=float(np.ceil(np.abs(ds["x_train"][:-NVAL]).max() * 2) / 2))
    net.load_state_dict(torch.load(CK / f"{name}_s{sd}.pt")); net.set_line_fir(FIR["Q10"]); net.eval()
    SN = NAME + ("_fro" if fro else "")
    path = ROOT / f"models/board_aware/candidates/{SN}_ship_ternary.npz"   # moved to models/ only if the ship bar is met
    m = export.save_ternary(net, path, dict(task="modrec", dataset_sha256=ds["sha256"], seed=sd, epochs=epochs, line_gain=False, variant=name,
                                            line_fir_source="models/board_aware/line_fir_m1a.json", line_fir_provenance=J["provenance"],
                                            line_aug=dict(corners=["Q10", "Q30"], tap_delay_jitter_Ts=0.14, p_ideal=kw.get("p_ideal", 0.0)),
                                            selection="validation: last 2000 train examples, macro acc (mean Q10/Q30) with min-class floor %.2f" % FLOOR))
    torch.save(net.state_dict(), path.with_suffix(".pt"))
    models = {}
    for nm, base in (("tdlnet_b_w8", ROOT / "models/tdlnet_b_w8_ternary"), (NAME + "_collapsed", ROOT / "models/tdlnet_b_w8_aware2_collapsed_ternary"), (SN, path.with_suffix(""))):
        cfg = json.loads(str(np.load(str(base) + ".npz")["meta_json"]))["arch"]
        n = TDLNet.from_config(dict(cfg, line_fir=None)); n.load_state_dict(torch.load(str(base) + ".pt")); models[nm] = n
    cells = {}
    for nm, n in models.items():
        for ln, fir in LINES.items():
            r = evaluate(n, ds["x_test"], ds["y_test"], ds["snr_test"], classes, fir)
            cells[f"{nm}|{ln}"] = dict(acc=r["acc"], acc_ge10dB=r["acc_ge10dB"], acc_per_class=r["acc_per_class"], acc_per_snr=r["acc_per_snr"])
            print(f"{nm:22s} {ln:6s} mixed {r['acc']:.4f}  >=10dB {r['acc_ge10dB']:.4f}  min-class {min(r['acc_per_class'].values()):.3f}", flush=True)
    aug = LineAug([J["corners"]["Q10"]["dense"], J["corners"]["Q30"]["dense"]], J["P"], J["M"])
    rng, jit = np.random.default_rng(0), {}
    for nm, n in models.items():
        accs = []
        for _ in range(10):
            delta = torch.from_numpy((rng.random(8) * 2 - 1) * 0.14).float()
            accs.append(evaluate(n, ds["x_test"], ds["y_test"], ds["snr_test"], classes, aug.at(0.0, delta).tolist())["acc"])
        jit[nm] = dict(mean=float(np.mean(accs)), min=float(np.min(accs)), max=float(np.max(accs)))
    c = cells[f"{SN}|Q10"]
    met = bool(min(c["acc_per_class"].values()) >= 0.35 and c["acc"] >= 0.79)
    final = ROOT / f"models/{SN}_ternary.npz"
    if met:
        import shutil
        shutil.copy(path, final); shutil.copy(path.with_suffix(".pt"), final.with_suffix(".pt")); path = final
    rep = dict(name=SN, variant=name, selected_seed=sd, dataset_sha256=ds["sha256"], model_sha256=m.sha256(), n_ternary_weights=m.n_ternary_weights,
               train_examples=len(ds["x_train"]) - NVAL, val_examples=NVAL, floor=FLOOR, search=rows, cells=cells, q10_plus_tap_jitter_14pct=jit,
               ship_bar=dict(min_class_test_Q10_ge=0.35, mixed_Q10_ge=0.79, min_class_test_Q10=min(c["acc_per_class"].values()), mixed_Q10=c["acc"],
                             met=met),
               line_fir_provenance=J["provenance"], ts_us=J["ts_us"], file=str(path),
               note="FIR covers line + input RC only (ideal buffer): no TLV9062, noise, crosstalk, S/H droop or mismatch.")
    (ROOT / f"models/{SN}_report.json").write_text(json.dumps(rep, indent=1))
    print("ship bar", rep["ship_bar"])


def main():
    args = sys.argv[1:]
    ds = modrec.dataset()
    if args and args[0] == "--ship":
        rows = json.loads((ROOT / "models/board_aware/aware2_search.json").read_text())
        ok = [r for r in rows if r["passes_floor"] and (len(args) < 2 or (VARIANTS[r["variant"]][2] == (args[1] == "fro")))]   # --ship [fro|ternary]
        b = max(ok, key=lambda r: r["val_macro"])
        print("winner", b["variant"], b["seed"], b["val_macro"]); ship(ds, b["variant"], b["seed"], rows); return
    sp = ROOT / "models/board_aware/aware2_search.json"
    rows = json.loads(sp.read_text()) if sp.exists() else []
    for v in (args or VARIANTS):
        rows = [r for r in rows if r["variant"] != v] + run_variant(ds, v)
        sp.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
