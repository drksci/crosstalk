"""Arena models (Amendment A1): conventional competitors for the small DLNN, same modrec data and splits.

  python -m dlnn.model.arena [--only mlp_s,cnn,...]  ->  models/arena/*.npz + models/arena/arena_report.json

Models: tdlnet_b_w8 (the DLNN, trained by dlnn.model.train; only evaluated and costed here), mlp_s (params-matched MLP on
flattened I/Q), mlp_l (larger MLP, extra), cnn (1-D CNN with ReLU, same topology as the DLNN but float), reservoir (random
ternary delay reservoir + ridge readout), cumulant (classical features + logistic regression / decision tree, numpy).
Protocol: hyper-parameters chosen on the last 2000 train examples (validation), then refit on all 16000 train examples; the
fixed 4000-example test set is only used for reporting. Ops are counted exactly per classification of one 128-sample window.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..data import modrec
from .npz import load_model
from .train import rotate_iq, summarize

torch.set_num_threads(2)
T = modrec.NSAMP
NC = len(modrec.CLASSES)


# ----------------------------------------------------------------------------- op counting
def ops(mults=0, adds=0, nonlin=0) -> dict:
    return dict(mults=int(mults), adds=int(adds), nonlin=int(nonlin))


def ops_sum(*o) -> dict:
    return {k: int(sum(d[k] for d in o)) for k in ("mults", "adds", "nonlin")}


def mac(n):  # a multiply-accumulate = 1 multiply + 1 add
    return ops(mults=n, adds=n)


def conv_ops(c_in, c_out, k, t_out, bias=True) -> dict:
    return ops(mults=c_in * c_out * k * t_out, adds=c_in * c_out * k * t_out + (c_out * t_out if bias else 0))


# ----------------------------------------------------------------------------- DLNN cost (ternary model file)
def dlnn_ops(m) -> dict:
    """Digital-equivalent op count of a dlnn-model/1 ternary net on one window (T = 128).

    Per layer: one add per non-zero tap per output sample (sign-select, no multiplier), 1 bias add, 1 gain multiply
    (the input-referred amplifier 1/vsat) and 1 tanh per output sample. Pool: O*(T-1) adds + O multiplies (mean).
    Readout: float 8x8 matrix (digital, on the Nucleo): 64 MACs + 8 bias adds; (a ternary readout would be adds only).
    """
    tot = ops()
    for ly in m.layers:
        O = ly.W.shape[0]
        nnz = int(np.count_nonzero(ly.W))
        tot = ops_sum(tot, ops(mults=O * T, adds=nnz * T + O * T, nonlin=O * T))
    O = m.layers[-1].W.shape[0]
    tot = ops_sum(tot, ops(mults=O, adds=O * (T - 1)))
    nr = m.readout_W.shape
    tot = ops_sum(tot, ops(mults=nr[0] * nr[1], adds=nr[0] * nr[1] + nr[0]))
    return tot


def dlnn_params(m) -> dict:
    nnz = sum(int(np.count_nonzero(l.W)) for l in m.layers)
    tot = sum(l.W.size for l in m.layers)
    other = sum(l.W.shape[0] * 2 + 1 for l in m.layers) + m.readout_W.size + m.readout_b.size
    return dict(params=int(tot + other), ternary_weights=int(tot), nonzero_ternary_weights=nnz, other_float_params=int(other))


# ----------------------------------------------------------------------------- torch models
class MLP(nn.Module):
    """Plain ReLU MLP. `in_dim` defaults to the 256 flattened I/Q samples; `drop` = dropout after each hidden ReLU (train only,
    absent from the exported net). `count()` is the op count of the network alone (input features are costed separately)."""

    def __init__(self, hidden=(4,), in_dim=2 * T, drop=0.0):
        super().__init__()
        dims = (in_dim,) + tuple(hidden) + (NC,)
        self.fc = nn.ModuleList([nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        h = x.flatten(1)
        for i, l in enumerate(self.fc):
            h = l(h)
            if i < len(self.fc) - 1:
                h = self.drop(torch.relu(h))
        return h

    def count(self):
        mu = ad = nl = 0
        for i, l in enumerate(self.fc):
            mu += l.in_features * l.out_features
            ad += l.in_features * l.out_features + l.out_features
            if i < len(self.fc) - 1:
                nl += l.out_features
        return ops(mu, ad, nl)

    def fold_input_affine(self, mean, std):
        """Fold input standardisation (x - mean)/std into layer 0 so the exported net takes the raw input."""
        with torch.no_grad():
            l = self.fc[0]
            m = torch.as_tensor(mean, dtype=torch.float32); s = torch.as_tensor(std, dtype=torch.float32)
            l.bias.sub_((l.weight * (m / s)[None]).sum(1))
            l.weight.div_(s[None])


# ----------------------------------------------------------------------------- phase-invariant MLP input features
FEAT_SETS = {   # name -> description
    "full": "|z| (128) + instantaneous frequency angle(z[n]z*[n-1])/2pi (127) + |FFT(z)| (128) = 383",
    "compact": "the three streams of `full`, stride-2 pooled (64 + 63 + 64 = 191)",
    "pow": "`full` + |FFT(z^2)| (128) + |FFT(z^4)| (128) = 639 (spectral lines of the squared/4th-power signal: phase-invariant, "
           "the classical PSK-order cue; a separate, more hand-engineered MLP row)",
    "pow_compact": "the five streams of `pow`, stride-2 pooled (64 + 63 + 64 + 64 + 64 = 319)",
}


def mlp_features(x, fset="full"):
    """x [B,2,T] raw I/Q -> [B,F] float32, invariant to carrier phase (and, via |FFT|, to timing/cyclic shift).
    z is first scaled to unit RMS. Pure numpy; op count in mlp_feature_ops()."""
    z = (x[:, 0] + 1j * x[:, 1]).astype(np.complex64)
    z = z / np.sqrt(np.mean(np.abs(z) ** 2, 1, keepdims=True))
    amp = np.abs(z)
    f1 = np.angle(z[:, 1:] * z[:, :-1].conj()) / (2 * np.pi)
    sp = np.abs(np.fft.fft(z, axis=1)) / np.sqrt(z.shape[1])
    streams = [amp, f1, sp]
    if fset.startswith("pow"):
        z2 = z * z
        streams += [np.abs(np.fft.fft(z2, axis=1)) / np.sqrt(z.shape[1]), np.abs(np.fft.fft(z2 * z2, axis=1)) / np.sqrt(z.shape[1])]
    if fset.endswith("compact"):
        pool = lambda a: a[:, :a.shape[1] // 2 * 2].reshape(len(a), -1, 2).mean(2)
        streams = [pool(a) for a in streams]
    return np.concatenate(streams, 1).astype(np.float32)


def mlp_feature_ops(fset="full", N=T) -> dict:
    """Exact per-window cost of mlp_features (the standardisation (x-mu)/sd is folded into the first layer, so it costs nothing):
    RMS normalise (2N mult + 2N-1 add for |z|^2 and its sum, 1 sqrt + 1 reciprocal, 2N mult scaling); |z| (2N mult, N add, N sqrt);
    inst. freq: complex mult (4 mult + 2 add) + 1 atan2 per sample, N-1 samples; FFT as in feature_ops (N/2 log2N butterflies of
    1 complex mult + 2 complex adds) + |X| (2N mult, N add, N sqrt); pooling (compact) = 1 add per input pair + 1 mult for the mean."""
    cm = lambda n: ops(4 * n, 2 * n)
    ca = lambda n: ops(0, 2 * n)
    L = int(math.log2(N))
    tot = ops(mults=2 * N + 2 * N, adds=2 * N - 1, nonlin=2)
    tot = ops_sum(tot, ops(mults=2 * N, adds=N, nonlin=N))
    tot = ops_sum(tot, cm(N - 1), ops(mults=N - 1, nonlin=N - 1))
    tot = ops_sum(tot, cm(N // 2 * L), ca(N * L), ops(mults=2 * N + N, adds=N, nonlin=N))
    if fset.startswith("pow"):          # z^2, z^4 (2 complex mults / sample), 2 more FFTs, 2 more |X|
        tot = ops_sum(tot, cm(2 * N), cm(2 * (N // 2 * L)), ca(2 * N * L), ops(mults=2 * (2 * N + N), adds=2 * N, nonlin=2 * N))
    if fset.endswith("compact"):
        nin = (N, N - 1, N) + ((N, N) if fset.startswith("pow") else ())
        tot = ops_sum(tot, ops(mults=sum(n // 2 for n in nin), adds=sum(n // 2 for n in nin)))
    return tot


class CNN(nn.Module):
    """3 x (causal conv k=8, ReLU) + global mean pool + linear. Same shape as TDL-Net-B but float weights, ReLU, no tanh."""

    def __init__(self, width=8, depth=3, k=8):
        super().__init__()
        ws = (2,) + (width,) * depth
        self.k = k
        self.cv = nn.ModuleList([nn.Conv1d(ws[i], ws[i + 1], k) for i in range(depth)])
        self.fc = nn.Linear(width, NC)

    def forward(self, x):
        h = x
        for c in self.cv:
            h = torch.relu(c(F.pad(h, (self.k - 1, 0))))
        return self.fc(h.mean(2))

    def count(self):
        tot = ops()
        for c in self.cv:
            tot = ops_sum(tot, conv_ops(c.in_channels, c.out_channels, self.k, T), ops(nonlin=c.out_channels * T))
        w = self.fc.in_features
        return ops_sum(tot, ops(mults=w, adds=w * (T - 1)), ops(mults=w * NC, adds=w * NC + NC))


def n_params(net) -> int:
    return int(sum(p.numel() for p in net.parameters()))


def fit_torch(make, ds, seed, epochs, lr, wd, n_use=None, aug=True, bs=256):
    torch.manual_seed(seed)
    net = make()
    x = torch.from_numpy(ds["x_train"][:n_use]); y = torch.from_numpy(ds["y_train"][:n_use])
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
    steps = epochs * math.ceil(len(x) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    gen = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(x), generator=gen)
        for i in range(0, len(x), bs):
            idx = perm[i:i + bs]
            xb = rotate_iq(x[idx], gen) if aug else x[idx]
            loss = F.cross_entropy(net(xb), y[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


@torch.no_grad()
def logits_of(net, x):
    net.eval()
    xt = torch.from_numpy(x)
    return torch.cat([net(xt[i:i + 1000]) for i in range(0, len(xt), 1000)]).numpy()


def acc(lg, y):
    return float(np.mean(lg.argmax(1) == y))


def run_torch_model(name, make, ds, epochs, lrs, wds, log):
    """hyper-parameters (lr, wd) on the validation split, then 3 seeds on all train data; seed by TRAIN accuracy."""
    nv = 2000
    best = None
    for lr in lrs:
        for wd in wds:
            net = fit_torch(make, ds, 0, epochs, lr, wd, n_use=len(ds["y_train"]) - nv)
            va = acc(logits_of(net, ds["x_train"][-nv:]), ds["y_train"][-nv:])
            log(f"  [{name}] lr {lr} wd {wd}: val {va:.4f}")
            if best is None or va > best[0]:
                best = (va, lr, wd)
    _, lr, wd = best
    tries, bestnet = [], None
    for s in range(3):
        net = fit_torch(make, ds, s, epochs, lr, wd)
        tr = acc(logits_of(net, ds["x_train"]), ds["y_train"])
        tries.append(dict(seed=s, train_acc=round(tr, 4)))
        if bestnet is None or tr > bestnet[0]:
            bestnet = (tr, net, s)
    return bestnet[1], dict(lr=lr, wd=wd, val_acc=round(best[0], 4), seed=bestnet[2], seeds=tries, epochs=epochs)


# ----------------------------------------------------------------------------- reservoir (numpy)
RES_CFG = dict(widths=(2, 32, 32, 32), taps=8, dilations=(1, 1, 1), density=0.6, drive=0.5, bias_range=0.5, seed=11)


def res_build(x_train, cfg):
    from . import reservoir
    from .npz import Layer
    from ..tiers import virtual
    rng = np.random.default_rng(cfg["seed"])
    w, layers = cfg["widths"], []
    h = x_train.astype(np.float32)
    for l, d in enumerate(cfg["dilations"]):
        W = reservoir.random_ternary((w[l + 1], w[l], cfg["taps"]), cfg["density"], rng)
        bias = rng.uniform(-cfg["bias_range"], cfg["bias_range"], w[l + 1]).astype(np.float32)
        ly = Layer(W=W, dilation=int(d), gain=np.ones(w[l + 1], np.float32), bias=bias, vsat=1.0)
        u = virtual.layer_preact(h, ly) + bias[None, :, None]
        ly.vsat = float(np.std(u) / cfg["drive"])
        layers.append(ly)
        h = virtual.layer_forward(h, ly)
    return layers


def res_feats(layers, x, mode):
    from ..tiers import virtual
    h = x.astype(np.float32)
    fs = []
    for ly in layers:
        h = virtual.layer_forward(h, ly)
        fs.append(h.mean(2))
        if mode == "mean_sq":
            fs.append((h ** 2).mean(2))
    return np.concatenate(fs, 1)


def run_reservoir(ds, log):
    from . import reservoir
    cfg = RES_CFG
    xtr, ytr = ds["x_train"], ds["y_train"]
    layers = res_build(xtr[:2000], cfg)
    nv = 2000
    out = {}
    best = None
    for mode in ("mean", "mean_sq"):
        Ftr = res_feats(layers, xtr, mode)
        mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
        Z = (Ftr - mu) / sd
        Zb = np.hstack([Z, np.ones((len(Z), 1))])
        Y = np.eye(NC)[ytr] * 2 - 1
        for lam in 10.0 ** np.arange(-3, 4):
            W = reservoir.ridge(Zb[:-nv], Y[:-nv], lam)
            va = acc(Zb[-nv:] @ W, ytr[-nv:])
            log(f"  [reservoir] pool {mode} lam {lam:g}: val {va:.4f}")
            if best is None or va > best[0]:
                best = (va, mode, lam)
    va, mode, lam = best
    Ftr = res_feats(layers, xtr, mode)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    Zb = np.hstack([(Ftr - mu) / sd, np.ones((len(Ftr), 1))])
    W = reservoir.ridge(Zb, np.eye(NC)[ytr] * 2 - 1, lam)
    nf = Ftr.shape[1]
    ro = W[:-1].T.astype(np.float32); rb = W[-1].astype(np.float32)

    def predict(x):
        Z = (res_feats(layers, x, mode) - mu) / sd
        return Z @ ro.T + rb

    nz = sum(int(np.count_nonzero(l.W)) for l in layers)
    mu_, ad = 0, 0
    c = ops()
    for ly in layers:
        O = ly.W.shape[0]
        c = ops_sum(c, ops(mults=O * T, adds=int(np.count_nonzero(ly.W)) * T + O * T, nonlin=O * T))
    pool_n = len(layers) * layers[0].W.shape[0] * (2 if mode == "mean_sq" else 1)
    c = ops_sum(c, ops(mults=pool_n + (len(layers) * layers[0].W.shape[0] * T if mode == "mean_sq" else 0),
                       adds=pool_n * (T - 1)),
                ops(mults=nf, adds=nf),                       # standardise
                mac(nf * NC), ops(adds=NC))                  # ridge readout
    info = dict(params=int(ro.size + rb.size), fixed_random_ternary_weights=int(sum(l.W.size for l in layers)),
                nonzero_random_weights=nz, pool=mode, ridge_lambda=lam, val_acc=round(va, 4), n_features=nf,
                reservoir_cfg={k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()})
    arrays = dict(schema=np.array("dlnn-arena-reservoir/1"), readout_W=ro, readout_b=rb, feat_mean=mu.astype(np.float32),
                  feat_std=sd.astype(np.float32), pool=np.array(mode), n_layers=np.array(len(layers)))
    for i, ly in enumerate(layers):
        arrays.update({f"L{i}_W": ly.W, f"L{i}_dilation": np.array(ly.dilation), f"L{i}_bias": ly.bias,
                       f"L{i}_vsat": np.array(ly.vsat, np.float32)})
    return predict, c, info, arrays


# ----------------------------------------------------------------------------- classical cumulant / feature baseline
FEATURE_NAMES = (
    "C20abs", "C40abs", "C41abs", "C42", "C60abs", "C63",                 # cumulants after CFO compensation (x^4-line estimate)
    "rawC40abs", "rawC41abs",                                             # without CFO compensation
    "line2", "line4", "line8",                                            # spectral-line strength of x^n (CFO/phase invariant)
    "amp_cv", "amp_kurt", "amp_skew",                                     # |x| coefficient of variation, kurtosis, skewness
    "f1_std", "f1_kurt", "f1_absmean", "f4_std", "f4_kurt", "f8_std",     # instantaneous-frequency stats (lag 1, 4, 8 products)
    "phd_std", "phd_kurt",                                                # |x| autocorrelation structure: std/kurt of Re(x[n+8]x*[n])/|..|
    "bw_std", "spec_flat", "spec_peak",                                   # spectrum width, flatness, peak concentration
)


def _kurt(v, axis=1):
    m = v - v.mean(axis, keepdims=True)
    return (m ** 4).mean(axis) / ((m ** 2).mean(axis) ** 2 + 1e-12)


def features(x):
    """x [B,2,T] -> [B,F] float64. Pure numpy; operation counts in feature_ops()."""
    z = (x[:, 0] + 1j * x[:, 1]).astype(np.complex128)
    B, N = z.shape
    z = z / np.sqrt(np.mean(np.abs(z) ** 2, 1, keepdims=True))          # unit power (C21 = 1)
    n = np.arange(N)

    def cum(zz):
        m20 = (zz ** 2).mean(1); m40 = (zz ** 4).mean(1); m41 = ((zz ** 3) * zz.conj()).mean(1)
        m42 = ((zz ** 2) * (zz.conj() ** 2)).mean(1); m21 = (np.abs(zz) ** 2).mean(1)
        m60 = (zz ** 6).mean(1); m63 = ((zz ** 3) * (zz.conj() ** 3)).mean(1)
        c40 = m40 - 3 * m20 ** 2
        c41 = m41 - 3 * m20 * m21
        c42 = m42 - np.abs(m20) ** 2 - 2 * m21 ** 2
        c60 = m60 - 15 * m40 * m20 + 30 * m20 ** 3
        c63 = m63 - 9 * m42 * m21 - 6 * np.abs(m20) ** 2 * m21 + 12 * m21 ** 3 + 0 * m41
        return np.abs(m20), np.abs(c40), np.abs(c41), np.real(c42), np.abs(c60), np.real(c63)

    def line(zz, p):
        F_ = np.fft.fft(zz ** p, axis=1)
        return np.abs(F_), np.argmax(np.abs(F_), 1)

    a4, i4 = line(z, 4)
    k = np.where(i4 > N // 2, i4 - N, i4)
    fhat = k / (4.0 * N)                                               # cycles / sample
    zc = z * np.exp(-2j * np.pi * fhat[:, None] * n[None])
    C = cum(zc)
    Craw = cum(z)
    lines = []
    for p in (2, 4, 8):
        a, _ = line(z, p)
        lines.append(a.max(1) / (np.mean(np.abs(z) ** p, 1) * N))
    amp = np.abs(z)
    am = amp.mean(1)
    cv = amp.std(1) / am
    ak = _kurt(amp)
    ask = (((amp - am[:, None]) ** 3).mean(1)) / (amp.std(1) ** 3 + 1e-12)

    def inst_f(lag):
        pr = z[:, lag:] * z[:, :-lag].conj()
        return np.angle(pr) / (2 * np.pi * lag)

    f1, f4, f8 = inst_f(1), inst_f(4), inst_f(8)
    ph8 = np.real(z[:, 8:] * z[:, :-8].conj())
    ph8 = ph8 / (np.abs(z[:, 8:] * z[:, :-8]) + 1e-9)
    S = np.abs(np.fft.fft(z, axis=1)) ** 2
    S = S / S.sum(1, keepdims=True)
    fr = np.fft.fftfreq(N)[None]
    mfreq = (S * fr).sum(1, keepdims=True)
    bw = np.sqrt((S * (fr - mfreq) ** 2).sum(1))
    flat = np.exp(np.mean(np.log(S + 1e-12), 1)) / S.mean(1)
    peak = np.sort(S, 1)[:, -4:].sum(1)
    cols = [C[0], C[1], C[2], C[3], C[4], C[5], Craw[1], Craw[2], *lines, cv, ak, ask,
            f1.std(1), _kurt(f1), np.abs(f1).mean(1), f4.std(1), _kurt(f4), f8.std(1),
            ph8.std(1), _kurt(ph8), bw, flat, peak]
    return np.stack(cols, 1)


def feature_ops(N=T) -> dict:
    """Exact real-arithmetic op count of `features` for one window (x^p via repeated complex multiplies;
    complex mult = 4 mult + 2 add; complex add = 2 add; FFT: N/2*log2N butterflies of 1 complex mult + 2 complex adds;
    atan2/sqrt/exp/log counted as 1 nonlinear op each)."""
    cm = lambda n: ops(4 * n, 2 * n)           # n complex multiplies
    ca = lambda n: ops(0, 2 * n)               # n complex adds
    fft = ops_sum(cm(N // 2 * int(math.log2(N))), ca(N * int(math.log2(N))))
    tot = ops(0, 0)
    tot = ops_sum(tot, ops(2 * N, 2 * N - 1 + 1), ops(nonlin=2))                     # |z|^2, mean, normalise (sqrt + div)
    tot = ops_sum(tot, ops(2 * N, 0))
    # powers of z: z2, z3(=z2*z), z4, z6, z8  -> 5 complex mults per sample; conj free
    tot = ops_sum(tot, cm(5 * N))
    # line strengths: 3 FFTs, |.|^2 (2 mult + 1 add) per bin + argmax compare (counted as adds), normalisation by mean|z|^p
    for _ in range(3):
        tot = ops_sum(tot, fft, ops(2 * N, N), ops(nonlin=N + 2))
    # CFO compensation: phasor exp (2 nonlin) per sample as recurrence: 1 complex mult per sample
    tot = ops_sum(tot, cm(N), ops(nonlin=2))
    # two cumulant sets: moments m20,m40,m41,m42,m60,m63,m21 over N samples (mults to form products 6/set, adds N each)
    one_set = ops_sum(cm(6 * N), ops(mults=3 * N, adds=7 * (2 * N)), ops(mults=30, adds=14, nonlin=6))
    tot = ops_sum(tot, one_set, one_set)
    # raw-set powers also needed for the raw z (z^p products reused above, count extra for derotated z)
    tot = ops_sum(tot, cm(5 * N))
    # amplitude stats: |z| (sqrt), mean, std, kurtosis, skew
    tot = ops_sum(tot, ops(mults=2 * N + 4 * N, adds=6 * N), ops(nonlin=N + 4))
    # instantaneous frequency at lags 1,4,8: complex mult + atan2 each, then std, kurt, abs-mean
    for lag in (1, 4, 8):
        n_ = N - lag
        tot = ops_sum(tot, cm(n_), ops(mults=5 * n_, adds=5 * n_), ops(nonlin=n_ + 3))
    # phd: product re part, abs (sqrt), divide, std, kurt
    tot = ops_sum(tot, cm(N - 8), ops(mults=2 * (N - 8) + 4 * N, adds=5 * N), ops(nonlin=N - 8 + 4))
    # spectrum of z (FFT), |.|^2, normalise, bandwidth, flatness (log), top-4 selection
    tot = ops_sum(tot, fft, ops(mults=2 * N + N + 3 * N, adds=N + 3 * N), ops(nonlin=N + 2),
                  ops(adds=4 * N))
    return tot


def softmax_np(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_logreg(Z, y, l2, iters=400, lr=0.5, Zv=None, yv=None, hidden=0, seed=0):
    """Multinomial logistic regression (hidden = 0) or 1-hidden-layer tanh MLP (hidden > 0) by full-batch gradient descent
    with Adam, numpy only."""
    rng = np.random.default_rng(seed)
    n, d = Z.shape
    Y = np.eye(NC)[y]
    if hidden:
        P = dict(W1=rng.standard_normal((d, hidden)) / math.sqrt(d), b1=np.zeros(hidden),
                 W2=rng.standard_normal((hidden, NC)) / math.sqrt(hidden), b2=np.zeros(NC))
    else:
        P = dict(W=np.zeros((d, NC)), b=np.zeros(NC))
    m = {k: np.zeros_like(v) for k, v in P.items()}; v2 = {k: np.zeros_like(v) for k, v in P.items()}
    for t in range(1, iters + 1):
        if hidden:
            H = np.tanh(Z @ P["W1"] + P["b1"]); G = (softmax_np(H @ P["W2"] + P["b2"]) - Y) / n
            g = dict(W2=H.T @ G + l2 * P["W2"], b2=G.sum(0))
            dH = (G @ P["W2"].T) * (1 - H ** 2)
            g.update(W1=Z.T @ dH + l2 * P["W1"], b1=dH.sum(0))
        else:
            G = (softmax_np(Z @ P["W"] + P["b"]) - Y) / n
            g = dict(W=Z.T @ G + l2 * P["W"], b=G.sum(0))
        for k in P:
            m[k] = 0.9 * m[k] + 0.1 * g[k]; v2[k] = 0.999 * v2[k] + 0.001 * g[k] ** 2
            P[k] -= lr * 0.05 * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v2[k] / (1 - 0.999 ** t)) + 1e-8)
    return P


def pred_logreg(P, Z):
    if "W1" in P:
        return np.tanh(Z @ P["W1"] + P["b1"]) @ P["W2"] + P["b2"]
    return Z @ P["W"] + P["b"]


def fit_tree(F_, y, max_depth, min_leaf=20, n_thr=24):
    """CART (gini, axis-aligned thresholds from quantiles). Returns arrays feat, thr, left, right, leaf_prob (node-indexed)."""
    nodes = []

    def gini_split(idx):
        yy = y[idx]
        cnt = np.bincount(yy, minlength=NC)
        best = (np.inf, None, None)
        n = len(idx)
        for f in range(F_.shape[1]):
            col = F_[idx, f]
            for th in np.unique(np.quantile(col, np.linspace(0.04, 0.96, n_thr))):
                L = col <= th
                nl = L.sum()
                if nl < min_leaf or n - nl < min_leaf:
                    continue
                cl = np.bincount(yy[L], minlength=NC); cr = cnt - cl
                g = nl * (1 - ((cl / nl) ** 2).sum()) + (n - nl) * (1 - ((cr / (n - nl)) ** 2).sum())
                if g < best[0]:
                    best = (g, f, th)
        return best

    def build(idx, depth):
        nid = len(nodes)
        nodes.append(None)
        p = np.bincount(y[idx], minlength=NC) / len(idx)
        if depth >= max_depth or len(idx) < 2 * min_leaf or p.max() > 0.999:
            nodes[nid] = (-1, 0.0, -1, -1, p)
            return nid
        g, f, th = gini_split(idx)
        if f is None:
            nodes[nid] = (-1, 0.0, -1, -1, p)
            return nid
        L = F_[idx, f] <= th
        l = build(idx[L], depth + 1); r = build(idx[~L], depth + 1)
        nodes[nid] = (f, th, l, r, p)
        return nid

    build(np.arange(len(y)), 0)
    return dict(feat=np.array([n[0] for n in nodes], np.int64), thr=np.array([n[1] for n in nodes], np.float32),
                left=np.array([n[2] for n in nodes], np.int64), right=np.array([n[3] for n in nodes], np.int64),
                prob=np.stack([n[4] for n in nodes]).astype(np.float32))


def pred_tree(Tr, F_):
    out = np.zeros((len(F_), NC), np.float32)
    for i, f in enumerate(F_):
        n = 0
        while Tr["feat"][n] >= 0:
            n = Tr["left"][n] if f[Tr["feat"][n]] <= Tr["thr"][n] else Tr["right"][n]
        out[i] = Tr["prob"][n]
    return out


def tree_depth_visited(Tr, F_):
    d = []
    for f in F_[:500]:
        n, k = 0, 0
        while Tr["feat"][n] >= 0:
            n = Tr["left"][n] if f[Tr["feat"][n]] <= Tr["thr"][n] else Tr["right"][n]
            k += 1
        d.append(k)
    return float(np.mean(d)), int(np.max(d))


def run_cumulant(ds, log):
    xtr, ytr = ds["x_train"], ds["y_train"]
    t0 = time.time()
    Ftr = features(xtr)
    log(f"  [cumulant] features {Ftr.shape} in {time.time() - t0:.1f}s")
    Ftr = np.nan_to_num(Ftr, nan=0.0, posinf=1e3, neginf=-1e3)
    nv = 2000
    mu, sd = Ftr[:-nv].mean(0), Ftr[:-nv].std(0) + 1e-9
    Z = (Ftr - mu) / sd
    cands = []
    for l2 in (1e-5, 1e-4, 1e-3, 1e-2):
        P = fit_logreg(Z[:-nv], ytr[:-nv], l2, Zv=None)
        cands.append(("logreg", dict(l2=l2), acc(pred_logreg(P, Z[-nv:]), ytr[-nv:])))
    for h in (16,):
        for l2 in (1e-4, 1e-3):
            P = fit_logreg(Z[:-nv], ytr[:-nv], l2, hidden=h, iters=600)
            cands.append(("mlp_h16", dict(l2=l2, hidden=h), acc(pred_logreg(P, Z[-nv:]), ytr[-nv:])))
    for d in (4, 6, 8):
        Tr = fit_tree(Ftr[:-nv], ytr[:-nv], d)
        cands.append(("tree", dict(max_depth=d), acc(pred_tree(Tr, Ftr[-nv:]), ytr[-nv:])))
    for c in cands:
        log(f"  [cumulant] {c[0]} {c[1]}: val {c[2]:.4f}")
    # report the two classifiers named in the task: best logistic regression and best decision tree; ship the better one
    best_lr = max([c for c in cands if c[0] == "logreg"], key=lambda c: c[2])
    best_tr = max([c for c in cands if c[0] == "tree"], key=lambda c: c[2])
    best_ml = max([c for c in cands if c[0] == "mlp_h16"], key=lambda c: c[2])
    kind, hp, va = max([best_lr, best_tr], key=lambda c: c[2])
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-9
    Z = (Ftr - mu) / sd
    fo = feature_ops()
    if kind == "logreg":
        P = fit_logreg(Z, ytr, hp["l2"])
        arrays = dict(W=P["W"].astype(np.float32), b=P["b"].astype(np.float32))
        predict = lambda x: pred_logreg(P, (np.nan_to_num(features(x), nan=0.0, posinf=1e3, neginf=-1e3) - mu) / sd)
        cls_ops = ops_sum(ops(mults=Z.shape[1], adds=Z.shape[1]), mac(Z.shape[1] * NC), ops(adds=NC))
        params = Z.shape[1] * NC + NC
    else:
        Tr = fit_tree(Ftr, ytr, hp["max_depth"])
        arrays = {f"tree_{k}": v for k, v in Tr.items()}
        predict = lambda x: pred_tree(Tr, np.nan_to_num(features(x), nan=0.0, posinf=1e3, neginf=-1e3))
        dmean, dmax = tree_depth_visited(Tr, Ftr)
        cls_ops = ops(adds=dmax)                                  # worst-case comparisons
        params = int(len(Tr["feat"]) * (2 + 2 + NC))
    tot = ops_sum(fo, cls_ops)
    other = {}
    if kind != "logreg":
        P = fit_logreg(Z, ytr, best_lr[1]["l2"])
        other["logreg_alternative_val_acc"] = round(best_lr[2], 4)
    else:
        other["tree_alternative_val_acc"] = round(best_tr[2], 4)
    other["tiny_mlp_h16_val_acc_not_shipped"] = round(best_ml[2], 4)
    info = dict(classifier=kind, hyper=hp, val_acc=round(va, 4), params=int(params), n_features=len(FEATURE_NAMES),
                feature_names=list(FEATURE_NAMES), feature_ops=fo, classifier_ops=cls_ops, **other)
    arrays.update(schema=np.array("dlnn-arena-cumulant/1"), classifier=np.array(kind), feat_mean=mu.astype(np.float32),
                  feat_std=sd.astype(np.float32), feature_names=np.array(FEATURE_NAMES))
    return predict, tot, info, arrays


# ----------------------------------------------------------------------------- driver
def evaluate(predict, ds, classes):
    t0 = time.time()
    lg = predict(ds["x_test"])
    r = summarize(lg, ds["y_test"], ds["snr_test"], classes)
    r["eval_seconds"] = round(time.time() - t0, 2)
    return r, lg


def save_torch_npz(path, net, kind, extra=None):
    a = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in net.state_dict().items()}
    a = {k.replace(".", "_"): v for k, v in a.items()}
    a["schema"] = np.array(f"dlnn-arena-{kind}/1")
    a["layer_names"] = np.array(list(net.state_dict().keys()))
    a.update(extra or {})
    np.savez(path, **a)


def predict_mlp_npz(path):
    """Numpy forward of an exported arena MLP (`dlnn-arena-mlp/1` raw I/Q, or `dlnn-arena-mlp-feat/1` with a `feature_set`)."""
    z = np.load(path, allow_pickle=False)
    names = [str(n) for n in z["layer_names"]]
    layers = [(z[n.replace(".", "_")], z[names[i + 1].replace(".", "_")]) for i, n in enumerate(names) if n.endswith("weight")]
    fset = str(z["feature_set"]) if "feature_set" in z.files else None

    def predict(x):
        h = mlp_features(x, fset) if fset else x.reshape(len(x), -1)
        for i, (W, b) in enumerate(layers):
            h = h @ W.T + b
            if i < len(layers) - 1:
                h = np.maximum(h, 0)
        return h
    return predict, fset


MLP_FAIR_ROWS = (   # report id, candidate stem, description
    ("mlp_s", "raw_s", "params-matched"), ("mlp_l", "raw_l", "~10k params"),
    ("mlp_feat_s", "feat_s", "params-matched"), ("mlp_feat_l", "feat_l", "~10k params"),
    ("mlp_pow_s", "featpow_s", "params-matched"), ("mlp_pow_l", "featpow_l", "~10k params"),
)


def assemble_mlp_fair(out: Path, ds, classes, models: dict, log) -> None:
    """Install the finished candidates of models/arena/mlp_search.py as report rows + npz files. The previous v1 raw-I/Q
    MLPs are kept as mlp_{s,l}_v1.npz / report rows mlp_{s,l}_v1. Every row is re-evaluated from the EXPORTED npz."""
    import shutil
    for n in ("mlp_s", "mlp_l"):
        v1 = out / f"{n}_v1.npz"
        if not v1.exists() and (out / f"{n}.npz").exists():
            shutil.copy(out / f"{n}.npz", v1)
        if f"{n}_v1" not in models and n in models and "fair" not in models[n]:
            models[f"{n}_v1"] = dict(models[n], file=f"{out}/{n}_v1.npz",
                                     desc=models[n]["desc"] + " [v1: un-normalised, 40 epochs, no dropout; kept for reference]")
    for rid, stem, size in MLP_FAIR_ROWS:
        cj = out / "candidates" / f"{stem}.json"
        if not cj.exists():
            log(f"  [mlp_fair] no candidate {stem}")
            continue
        c = json.loads(cj.read_text())
        shutil.copy(out / "candidates" / f"{stem}.npz", out / f"{rid}.npz")
        predict, fset = predict_mlp_npz(out / f"{rid}.npz")
        r, _ = evaluate(predict, ds, classes)
        assert abs(r["acc"] - c["acc"]) < 2e-3, (rid, r["acc"], c["acc"])      # exported (folded) net == trained net
        net_ops = c["net_ops"]
        fo = mlp_feature_ops(fset) if fset else ops()
        kind = "flattened raw I/Q (256), input standardisation folded into layer 0" if not fset else \
            f"phase-invariant features '{fset}': {FEAT_SETS[fset]}; standardisation folded into layer 0"
        models[rid] = dict(
            desc=f"MLP ({size}) on {kind}; ReLU, hidden {c['hidden']}", fair=True, file=f"{out}/{rid}.npz",
            params=c["params"], ops=ops_sum(net_ops, fo), net_ops=net_ops, feature_ops=fo,
            train=dict(lr=c["lr"], wd=c["wd"], dropout=c["dropout"], epochs=c["epochs"], val_acc=c["val_acc"],
                       shipped_seed=c["shipped_seed"], seeds=c["seeds"], grid_size=c["grid_size"], feature_set=c["feature_set"],
                       selection="grid on train[:14000] vs train[14000:] (seed 0); refit on all 16000, 3 seeds, ship highest train acc"),
            train_seconds=c["search_seconds"], **r)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default="models/arena")
    ap.add_argument("--dlnn", default="models/tdlnet_b_w8_ternary.npz")
    ap.add_argument("--dlnn-aware", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args(argv)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    ds = modrec.dataset()
    classes = ds["classes"]
    rp = out / "arena_report.json"
    report = json.loads(rp.read_text()) if rp.exists() else {}
    report.update(dataset_sha256=ds["sha256"], n_train=len(ds["y_train"]), n_test=len(ds["y_test"]), classes=classes,
                  window="128 complex samples, 8 samples/symbol", torch=torch.__version__,
                  ops_convention="mults/adds/nonlin per classification of one 128-sample window; a MAC = 1 mult + 1 add; "
                                 "ternary taps = 1 add per non-zero tap (no multiply)")
    models = report.setdefault("models", {})
    log = lambda s: print(s, flush=True)

    def want(n):
        return not only or n in only

    t_all = time.time()
    for dn, dfile, ddesc in (("tdlnet_b_w8", a.dlnn, "TDL-Net-B w8 d=1, ternary taps, ternary readout (the small DLNN)"),
                             ("tdlnet_b_w8_fro", a.dlnn.replace("w8_ternary", "w8_fro_ternary"),
                              "TDL-Net-B w8 d=1, ternary taps, float digital readout (readout runs on the MCU)")):
        if want(dn) and Path(dfile).exists():
            m = load_model(dfile)
            from ..tiers import virtual
            r, _ = evaluate(lambda x, m=m: virtual.forward(m, x)["out"], ds, classes)
            models[dn] = dict(desc=ddesc, file=dfile, **dlnn_params(m), ops=dlnn_ops(m), **r)
    if "mlp_fair" in only:
        assemble_mlp_fair(out, ds, classes, models, log)
    for name, hidden, lrs, wds in (("mlp_s", (4,), (3e-3, 1e-2), (1e-4, 1e-2)), ("mlp_l", (32,), (3e-3, 1e-2), (1e-4, 1e-2))):
        if only and name in only:       # v1 recipe; superseded by `--only mlp_fair` (models/arena/mlp_search.py)
            t0 = time.time()
            net, hp = run_torch_model(name, lambda h=hidden: MLP(h), ds, a.epochs, lrs, wds, log)
            r, _ = evaluate(lambda x, n=net: logits_of(n, x), ds, classes)
            save_torch_npz(out / f"{name}.npz", net, "mlp", dict(hidden=np.array(hidden)))
            models[name] = dict(desc=f"MLP on flattened I/Q (256 inputs), ReLU, hidden {list(hidden)}", file=f"{out}/{name}.npz",
                                params=n_params(net), ops=net.count(), train=hp, train_seconds=round(time.time() - t0, 1), **r)
    if want("cnn"):
        t0 = time.time()
        net, hp = run_torch_model("cnn", lambda: CNN(8, 3, 8), ds, a.epochs, (3e-3, 1e-2), (1e-4, 1e-2), log)
        r, _ = evaluate(lambda x, n=net: logits_of(n, x), ds, classes)
        save_torch_npz(out / "cnn.npz", net, "cnn")
        models["cnn"] = dict(desc="1-D CNN, 3 causal conv layers (2-8-8-8, k=8), ReLU, global mean pool, linear 8->8, float",
                             file=f"{out}/cnn.npz", params=n_params(net), ops=net.count(), train=hp,
                             train_seconds=round(time.time() - t0, 1), **r)
    if want("reservoir"):
        t0 = time.time()
        predict, c, info, arrays = run_reservoir(ds, log)
        r, _ = evaluate(predict, ds, classes)
        np.savez(out / "reservoir.npz", **arrays)
        models["reservoir"] = dict(desc="random ternary delay reservoir (2-32-32-32, K=8, d=1, tanh) + ridge readout on pooled layer outputs",
                                   file=f"{out}/reservoir.npz", ops=c, train_seconds=round(time.time() - t0, 1), **info, **r)
    if want("cumulant"):
        t0 = time.time()
        predict, c, info, arrays = run_cumulant(ds, log)
        r, _ = evaluate(predict, ds, classes)
        np.savez(out / "cumulant.npz", **arrays)
        models["cumulant"] = dict(desc="higher-order cumulants + amplitude/frequency/spectrum statistics -> logistic regression or tree",
                                  file=f"{out}/cumulant.npz", ops=c, train_seconds=round(time.time() - t0, 1), **info, **r)
    for k, v in models.items():
        o = v["ops"]
        print(f"{k:12s} params {v['params']:6d} mults {o['mults']:8d} adds {o['adds']:8d} nonlin {o['nonlin']:6d} "
              f"acc {v['acc']:.4f} >=10dB {v['acc_ge10dB']:.4f}")
    report["total_seconds_last_run"] = round(time.time() - t_all, 1)
    rp.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
