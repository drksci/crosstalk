"""Train + export the TDL-Net-S.   python -m dlnn.model.train --task modrec [--float] [--epochs N] [--out models/]

Without --float both the ternary model and the float baseline are trained (ternary first) and
`models/train_report.json` is written. With --float only the float baseline is (re)trained.
Train/test split: `dlnn.data.modrec.dataset()` (seed 20260101 / +1, disjoint, deterministic).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ..data import modrec
from . import export
from .tdlnet_torch import TDLNet

torch.set_num_threads(2)

DEFAULTS = dict(epochs=60, batch=256, lr=3e-2, seed=0, aug=True, line_gain=False, input_scale=1.0, restarts=3,
                soft_frac=0.5)


def rotate_iq(x: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random carrier-phase rotation of every example (the dataset phase is already uniform; this is augmentation)."""
    th = torch.rand(x.shape[0], 1, generator=gen) * (2 * math.pi)
    c, s = torch.cos(th), torch.sin(th)
    return torch.stack([c * x[:, 0] - s * x[:, 1], s * x[:, 0] + c * x[:, 1]], dim=1)


@torch.no_grad()
def predict(net: TDLNet, x: np.ndarray, quant: bool = False, bs: int = 1000) -> np.ndarray:
    net.eval()
    xt = torch.from_numpy(x)
    return torch.cat([net(xt[i:i + bs], quant=quant) for i in range(0, len(xt), bs)]).numpy()


def summarize(logits: np.ndarray, y: np.ndarray, snr: np.ndarray, classes) -> dict:
    pred = logits.argmax(1)
    per_snr = {str(int(s)): float(np.mean(pred[snr == s] == y[snr == s])) for s in np.unique(snr)}
    hi = snr >= 10
    return dict(acc=float(np.mean(pred == y)), acc_ge10dB=float(np.mean(pred[hi] == y[hi])), acc_per_snr=per_snr,
                acc_per_class={c: float(np.mean(pred[y == i] == i)) for i, c in enumerate(classes)})


def train_one(ds: dict, float_mode: bool, epochs: int, batch: int, lr: float, seed: int, aug: bool,
              line_gain: bool, input_scale: float, x_tr=None, y_tr=None, log=print,
              init_state: dict | None = None, float_readout: bool = False,
              teacher: TDLNet | None = None, kd_alpha: float = 0.0, kd_temp: float = 2.0,
              soft_frac: float = 0.0, width: int | None = None, depth: int = 3,
              dilations: tuple | None = None, line_fir=None, line_aug=None, focal_gamma: float = 0.0) -> tuple[TDLNet, dict]:
    torch.manual_seed(seed)
    x_tr = ds["x_train"] if x_tr is None else x_tr
    y_tr = ds["y_train"] if y_tr is None else y_tr
    fs = float(np.ceil(np.abs(x_tr).max() * 2) / 2)  # fixed ADC full scale (raw units) from the training set
    wd = {} if width is None else dict(widths=(2,) + (width,) * depth)
    if dilations is not None or depth != 3:
        wd["dilations"] = tuple(dilations) if dilations is not None else (1,) * depth
        wd.setdefault("widths", (2,) + (8,) * depth)
    if line_fir is not None:
        wd["line_fir"] = line_fir
    if line_aug is not None:
        wd["line_aug"] = line_aug
    net = TDLNet(**wd, float_mode=float_mode, line_gain=line_gain, input_scale=input_scale, input_fullscale=fs,
                 float_readout=float_readout)
    if init_state is not None:  # warm start (e.g. ternary QAT from a trained float net); shapes must match
        net.load_state_dict(init_state, strict=False)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    n = len(xt)
    steps = epochs * math.ceil(n / batch)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(seed)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        if not float_mode:  # progressive ternarisation: lam ramps 0 -> 1 over the first soft_frac of training
            net.set_lam(1.0 if soft_frac <= 0 else min(1.0, ep / max(1.0, soft_frac * epochs)))
        perm = torch.randperm(n, generator=gen)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = xt[idx]
            if aug:
                xb = rotate_iq(xb, gen)
            lg = net(xb, quant=True)
            if focal_gamma > 0:   # focal loss (additive option; default off = plain cross-entropy)
                ce = F.cross_entropy(lg, yt[idx], reduction="none")
                loss = (((1 - torch.exp(-ce)) ** focal_gamma) * ce).mean()
            else:
                loss = F.cross_entropy(lg, yt[idx])
            if teacher is not None and kd_alpha > 0:
                with torch.no_grad():
                    tl = teacher(xb)
                kd = F.kl_div(F.log_softmax(lg / kd_temp, 1), F.softmax(tl / kd_temp, 1), reduction="batchmean")
                loss = (1 - kd_alpha) * loss + kd_alpha * kd_temp ** 2 * kd
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss.detach()) * len(idx)
        if (ep + 1) % 10 == 0 or ep == 0:
            log(f"  [{'float' if float_mode else 'ternary'}] epoch {ep + 1:2d}/{epochs} loss {tot / n:.4f} "
                f"({time.time() - t0:.0f}s)")
    net.set_lam(1.0)
    return net, dict(seconds=time.time() - t0, input_fullscale_raw=fs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--task", default="modrec", choices=["modrec"])
    ap.add_argument("--float", action="store_true", help="train only the float baseline")
    ap.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    ap.add_argument("--batch", type=int, default=DEFAULTS["batch"])
    ap.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    ap.add_argument("--seed", type=int, default=DEFAULTS["seed"])
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--line-gain", action="store_true", help="per-output-line scale (amplifier gain per line)")
    ap.add_argument("--input-scale", type=float, default=DEFAULTS["input_scale"])
    ap.add_argument("--restarts", type=int, default=DEFAULTS["restarts"],
                    help="seeds tried per model; the one with the best TRAIN accuracy is kept (test never used)")
    ap.add_argument("--soft-frac", type=float, default=DEFAULTS["soft_frac"],
                    help="fraction of epochs over which ternarisation is ramped in (0 = ternary from step 0)")
    ap.add_argument("--width", type=int, default=None, help="hidden width (default 8 = TDL-Net-S); changes file names")
    ap.add_argument("--depth", type=int, default=3, help="number of TDL layers (default 3)")
    ap.add_argument("--dilations", default=None, help="comma list, one per layer, e.g. 1,1,1 (board: d = 1); default 1,4,16")
    ap.add_argument("--name", default=None, help="file stem, e.g. tdlnet_b_w8 -> models/tdlnet_b_w8_ternary.npz (+ _report.json)")
    ap.add_argument("--float-readout", action="store_true", help="digital float readout instead of ternary (ternary convs)")
    ap.add_argument("--ternary-only", action="store_true", help="skip the float baseline")
    ap.add_argument("--line-fir", default=None, help="json with the measured per-tap FIR (board-aware training)")
    ap.add_argument("--out", default="models/")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    ds = modrec.dataset()
    classes = ds["classes"]
    print(f"dataset sha256 {ds['sha256'][:16]}.. train {len(ds['y_train'])} test {len(ds['y_test'])}")
    dil = tuple(int(v) for v in a.dilations.split(",")) if a.dilations else None
    if dil is not None and len(dil) != a.depth:
        ap.error("--dilations must have --depth entries")
    fir = json.loads(Path(a.line_fir).read_text())["fir"] if a.line_fir else None
    kw = dict(epochs=a.epochs, batch=a.batch, lr=a.lr, aug=not a.no_aug, line_gain=a.line_gain,
              input_scale=a.input_scale, width=a.width, depth=a.depth, dilations=dil, line_fir=fir)
    tag = "modrec" if a.width is None else f"modrec_w{a.width}"
    report_path = out / ("train_report.json" if a.width is None else f"train_report_w{a.width}.json")
    if a.name:
        tag, report_path = a.name, out / f"{a.name}_report.json"
    report = json.loads(report_path.read_text()) if (a.float and report_path.exists()) else {}
    report.update(task=a.task, dataset_sha256=ds["sha256"], dataset_seed=ds["seed"],
                  n_train=int(len(ds["y_train"])), n_test=int(len(ds["y_test"])), width=a.width or 8,
                  recipe=dict(optimizer="Adam", lr=a.lr, schedule="cosine", batch=a.batch, epochs=a.epochs,
                              iq_rotation_aug=not a.no_aug, line_gain=a.line_gain, base_seed=a.seed,
                              restarts=a.restarts, soft_frac=a.soft_frac, depth=a.depth, dilations=a.dilations,
                              float_readout=a.float_readout, line_fir=bool(fir), torch=torch.__version__))
    for fm in ([True] if a.float else ([False] if a.ternary_only else [False, True])):
        name = "float" if fm else "ternary"
        tries = []
        for r in range(a.restarts):
            net_r, info_r = train_one(ds, fm, seed=a.seed + r, soft_frac=0.0 if fm else a.soft_frac,
                                      float_readout=a.float_readout, **kw)
            tr_acc = float(np.mean(predict(net_r, ds["x_train"]).argmax(1) == ds["y_train"]))
            # test numbers per seed are REPORTED only (never used for selection; selection = train accuracy)
            te_r = summarize(predict(net_r, ds["x_test"]), ds["y_test"], ds["snr_test"], classes)
            tries.append(dict(seed=a.seed + r, train_acc=round(tr_acc, 4), seconds=round(info_r["seconds"], 1),
                              test_acc=round(te_r["acc"], 4), test_acc_ge10dB=round(te_r["acc_ge10dB"], 4)))
            print(f"  [{name}] seed {a.seed + r}: train acc {tr_acc:.4f}")
            if r == 0 or tr_acc > max(t["train_acc"] for t in tries[:-1]):
                net, info = net_r, info_r
        best_seed = max(tries, key=lambda t: t["train_acc"])["seed"]
        info["seconds"] = sum(t["seconds"] for t in tries)
        # eval 1: exactly what the V tier computes (no activation quantisation); eval 2: training-time 8-bit quant
        lg = predict(net, ds["x_test"], quant=False)
        res = summarize(lg, ds["y_test"], ds["snr_test"], classes)
        res.update(seconds=round(info["seconds"], 1), epochs=a.epochs, seed=best_seed, restarts=tries)
        if not fm:
            res["acc_8bit_fakequant"] = summarize(predict(net, ds["x_test"], quant=True), ds["y_test"],
                                                  ds["snr_test"], classes)["acc"]
            path = out / (f"{tag}_ternary.npz" if a.name else f"tdlnet_{tag}_ternary.npz")
            m = export.save_ternary(net, path, dict(task=a.task, dataset_sha256=ds["sha256"], seed=best_seed,
                                                    epochs=a.epochs, line_gain=a.line_gain))
            res["model_sha256"] = m.sha256()
            res["n_ternary_weights"] = m.n_ternary_weights
            res["file"] = str(path)
            torch.save(net.state_dict(), path.with_suffix(".pt"))
        else:
            path = out / (f"{tag}_float.npz" if a.name else f"tdlnet_{tag}_float.npz")
            export.save_float(net, path, dict(task=a.task, dataset_sha256=ds["sha256"], seed=best_seed, epochs=a.epochs))
            res["file"] = str(path)
            torch.save(net.state_dict(), path.with_suffix(".pt"))
        report[name] = res
        print(f"{name}: acc {res['acc']:.4f}  acc>=10dB {res['acc_ge10dB']:.4f}  per-SNR {res['acc_per_snr']}  "
              f"({res['seconds']}s)")
    report["total_seconds"] = round(time.time() - t_all, 1)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"wrote {report_path}  total {report['total_seconds']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
