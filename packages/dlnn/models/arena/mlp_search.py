"""Fair MLP baseline search (see docs/build-log-P2-M1b.md, "MLP fairness").

  nohup .venv/bin/python models/arena/mlp_search.py --family raw --size s >> models/arena/candidates/progress.log 2>&1 &

Protocol: every hyper-parameter combination is trained on train[:14000] (seed 0) and scored on the validation split train[14000:]
(the last 2000 train examples). The best combination is refit on all 16000 train examples with seeds 0,1,2; the shipped seed is
the one with the highest TRAIN accuracy (same rule as the other arena models). Test accuracy is computed once per finished
candidate for the report and is never used for any choice. Each finished (family,size) candidate is saved to
models/arena/candidates/<family>_<size>.{json,npz}.
"""
from __future__ import annotations

import argparse, itertools, json, math, sys, time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dlnn.data import modrec
from dlnn.model import arena
from dlnn.model.train import rotate_iq, summarize

torch.set_num_threads(1)  # 4 searches side by side
OUT = Path(__file__).resolve().parent / "candidates"

# hidden-layer configs per (family, size class)
HIDDEN = {
    ("raw", "s"): [(4,)],
    ("raw", "l"): [(38,), (36, 16)],
    ("feat", "s"): [("full", (3,)), ("compact", (6,))],
    ("feat", "l"): [("full", (24,)), ("full", (20, 16)), ("compact", (48,)), ("compact", (40, 16))],
    ("featpow", "s"): [("pow", (2,)), ("pow_compact", (3,))],
    ("featpow", "l"): [("pow", (14,)), ("pow_compact", (28,)), ("pow_compact", (24, 12))],
}
EPOCHS = 60


def fit(net, xtr, ytr, seed, lr, wd, aug, epochs=EPOCHS, bs=128):
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(xtr) / bs)
    warm = 200
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(t, steps) / steps)))
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(xtr), generator=gen)
        for i in range(0, len(xtr), bs):
            idx = perm[i:i + bs]
            xb = rotate_iq(xtr[idx], gen) if aug else xtr[idx]
            loss = F.cross_entropy(net(xb), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=["raw", "feat", "featpow"])
    ap.add_argument("--size", required=True, choices=["s", "l"])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    log = lambda s: print(time.strftime("%H:%M:%S"), f"[{a.family}_{a.size}]", s, flush=True)
    ds = modrec.dataset()
    ytr = torch.from_numpy(ds["y_train"])
    nv = 2000
    fam = a.family
    fcache = {}

    def inputs(fset):
        if fam == "raw":
            return ds["x_train"].reshape(len(ytr), -1), ds["x_test"].reshape(len(ds["y_test"]), -1)
        if fset not in fcache:
            fcache[fset] = (arena.mlp_features(ds["x_train"], fset), arena.mlp_features(ds["x_test"], fset))
        return fcache[fset]

    grid = []
    for cfg in HIDDEN[(fam, a.size)]:
        fset, hid = cfg if fam != "raw" else (None, cfg)
        for lr, wd, dr in itertools.product((3e-3, 1e-2, 3e-2), (1e-2, 5e-2, 2e-1), (0.0, 0.2)):
            grid.append((fset, hid, lr, wd, dr))
    best = None
    t0 = time.time()
    for fset, hid, lr, wd, dr in grid:
        Xtr, _ = inputs(fset)
        mu, sd = Xtr[:-nv].mean(0), Xtr[:-nv].std(0) + 1e-6
        xt = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
        net = arena.MLP(hid, in_dim=Xtr.shape[1], drop=dr)
        fit(net, xt[:-nv], ytr[:-nv], 0, lr, wd, aug=False, epochs=a.epochs)
        va = arena.acc(arena.logits_of(net, xt[-nv:].numpy()), ds["y_train"][-nv:])
        log(f"val {va:.4f}  fset {fset} hidden {hid} lr {lr} wd {wd} drop {dr}  ({time.time() - t0:.0f}s)")
        if best is None or va > best[0]:
            best = (va, fset, hid, lr, wd, dr)
    va, fset, hid, lr, wd, dr = best
    log(f"BEST val {va:.4f} fset {fset} hidden {hid} lr {lr} wd {wd} drop {dr}")
    Xtr, Xte = inputs(fset)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6          # refit statistics on all train
    xt = torch.from_numpy(((Xtr - mu) / sd).astype(np.float32))
    xe = ((Xte - mu) / sd).astype(np.float32)
    seeds, nets = [], []
    for s in range(3):
        net = arena.MLP(hid, in_dim=Xtr.shape[1], drop=dr)
        fit(net, xt, ytr, s, lr, wd, aug=False, epochs=a.epochs)
        tr = arena.acc(arena.logits_of(net, xt.numpy()), ds["y_train"])
        r = summarize(arena.logits_of(net, xe), ds["y_test"], ds["snr_test"], ds["classes"])
        seeds.append(dict(seed=s, train_acc=round(tr, 4), test_acc=r["acc"], test_acc_ge10dB=r["acc_ge10dB"]))
        nets.append(net)
        log(f"seed {s}: train {tr:.4f} test {r['acc']:.4f} >=10dB {r['acc_ge10dB']:.4f}")
    k = int(np.argmax([d["train_acc"] for d in seeds]))
    net = nets[k]
    r = summarize(arena.logits_of(net, xe), ds["y_test"], ds["snr_test"], ds["classes"])
    net.fold_input_affine(mu, sd)                      # export takes the raw (unstandardised) input
    extra = dict(hidden=np.array(hid), in_dim=np.array(Xtr.shape[1]))
    if fam != "raw":
        extra["feature_set"] = np.array(fset)
    arena.save_torch_npz(OUT / f"{fam}_{a.size}.npz", net, "mlp" if fam == "raw" else "mlp-feat", extra)
    info = dict(family=fam, size=a.size, feature_set=fset, hidden=list(hid), lr=lr, wd=wd, dropout=dr, epochs=a.epochs,
                val_acc=round(va, 4), shipped_seed=k, seeds=seeds, params=arena.n_params(net),
                net_ops=net.count(), grid_size=len(grid), search_seconds=round(time.time() - t0, 1), **r)
    (OUT / f"{fam}_{a.size}.json").write_text(json.dumps(info, indent=2))
    log(f"DONE params {info['params']} test {r['acc']:.4f} >=10dB {r['acc_ge10dB']:.4f}")


if __name__ == "__main__":
    main()
