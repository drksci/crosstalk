"""Synthetic 8-class modulation recognition data (deterministic, numpy only).

Classes: OOK, BPSK, QPSK, 8PSK, 16QAM, 2FSK, 4FSK, GMSK. 128 complex samples at 8 samples/symbol,
random carrier phase, CFO U(-0.01, 0.01) cycles/sample (+-1 % of fs), random timing offset
U(0, 8) samples (fractional delay applied in the frequency domain), unit signal power, AWGN at
SNR drawn (balanced) from {0, 5, 10, 15, 20} dB. Output x has shape [N, 2, 128] (I, Q).
"""
from __future__ import annotations

import numpy as np

from . import sha256_arrays

CLASSES = ("OOK", "BPSK", "QPSK", "8PSK", "16QAM", "2FSK", "4FSK", "GMSK")
SNRS = (0, 5, 10, 15, 20)
SPS = 8
NSAMP = 128
SEED = 20260101
_NSYM = 24          # generate 24 symbols (192 samples); crop a 128-sample window after delay
_MARGIN = 8


def _rrc(beta: float = 0.35, span: int = 6, sps: int = SPS) -> np.ndarray:
    t = np.arange(-span * sps // 2, span * sps // 2 + 1) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if abs(ti) < 1e-9:
            h[i] = 1 - beta + 4 * beta / np.pi
        elif abs(abs(ti) - 1 / (4 * beta)) < 1e-9:
            h[i] = beta / np.sqrt(2) * ((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))
                                        + (1 - 2 / np.pi) * np.cos(np.pi / (4 * beta)))
        else:
            h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) \
                / (np.pi * ti * (1 - (4 * beta * ti) ** 2))
    return h / np.sqrt(np.sum(h ** 2))


_RRC = _rrc()


def _linear(sym: np.ndarray) -> np.ndarray:
    """sym [n, NSYM] complex -> RRC-shaped baseband [n, NSYM*SPS]."""
    up = np.zeros((sym.shape[0], sym.shape[1] * SPS), dtype=complex)
    up[:, ::SPS] = sym
    return np.stack([np.convolve(r, _RRC, mode="same") for r in up])


def _symbols(cls: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if cls == "OOK":
        return rng.integers(0, 2, (n, _NSYM)).astype(complex)
    if cls == "BPSK":
        return (2 * rng.integers(0, 2, (n, _NSYM)) - 1).astype(complex)
    if cls == "QPSK":
        return np.exp(1j * (np.pi / 4 + np.pi / 2 * rng.integers(0, 4, (n, _NSYM))))
    if cls == "8PSK":
        return np.exp(1j * (np.pi / 4 * rng.integers(0, 8, (n, _NSYM))))
    if cls == "16QAM":
        lv = np.array([-3, -1, 1, 3])
        return lv[rng.integers(0, 4, (n, _NSYM))] + 1j * lv[rng.integers(0, 4, (n, _NSYM))]
    raise ValueError(cls)


def _cpfsk(levels: np.ndarray, rng: np.random.Generator, n: int) -> np.ndarray:
    a = rng.choice(levels, (n, _NSYM))
    f = np.repeat(a, SPS, axis=1) * (1.0 / (2 * SPS))  # modulation index h = 1, cycles/sample
    return np.exp(2j * np.pi * np.cumsum(f, axis=1))


def _gmsk(rng: np.random.Generator, n: int, bt: float = 0.3, span: int = 4) -> np.ndarray:
    bits = 2 * rng.integers(0, 2, (n, _NSYM)) - 1
    t = np.arange(-span * SPS // 2, span * SPS // 2 + 1) / SPS
    sigma = np.sqrt(np.log(2)) / (2 * np.pi * bt)
    g = np.exp(-t ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    a = np.repeat(bits, SPS, axis=1).astype(float)
    sm = np.stack([np.convolve(r, g, mode="same") for r in a])
    return np.exp(1j * np.pi * 0.5 * np.cumsum(sm, axis=1) / SPS)


def _waveform(cls: str, n: int, rng: np.random.Generator) -> np.ndarray:
    if cls in ("OOK", "BPSK", "QPSK", "8PSK", "16QAM"):
        return _linear(_symbols(cls, n, rng))
    if cls == "2FSK":
        return _cpfsk(np.array([-1.0, 1.0]), rng, n)
    if cls == "4FSK":
        return _cpfsk(np.array([-3.0, -1.0, 1.0, 3.0]), rng, n)
    if cls == "GMSK":
        return _gmsk(rng, n)
    raise ValueError(cls)


def generate(n: int, seed: int = SEED, snrs: tuple = SNRS) -> dict:
    """n examples; class and SNR are exactly balanced when n is a multiple of 8*len(snrs)."""
    rng = np.random.default_rng(seed)
    ncls, nsnr = len(CLASSES), len(snrs)
    idx = np.arange(n)
    y = (idx % ncls).astype(np.int64)
    snr = np.array(snrs)[(idx // ncls) % nsnr].astype(np.float32)
    perm = rng.permutation(n)
    y, snr = y[perm], snr[perm]

    x = np.zeros((n, 2, NSAMP), dtype=np.float32)
    freqs = np.fft.fftfreq(_NSYM * SPS)
    for c, name in enumerate(CLASSES):
        sel = np.where(y == c)[0]
        m = len(sel)
        w = _waveform(name, m, rng)
        # fractional-sample delay in frequency domain, integer offset by crop start
        delay = rng.uniform(0, SPS, m)
        frac = delay - np.floor(delay)
        w = np.fft.ifft(np.fft.fft(w, axis=1) * np.exp(-2j * np.pi * freqs[None] * frac[:, None]), axis=1)
        start = _MARGIN + np.floor(delay).astype(int)
        w = np.stack([w[i, s:s + NSAMP] for i, s in enumerate(start)])
        w /= np.sqrt(np.mean(np.abs(w) ** 2, axis=1, keepdims=True))        # unit power
        cfo = rng.uniform(-0.01, 0.01, (m, 1))
        phase = rng.uniform(0, 2 * np.pi, (m, 1))
        w = w * np.exp(1j * (phase + 2 * np.pi * cfo * np.arange(NSAMP)[None]))
        sigma = np.sqrt(10 ** (-snr[sel] / 10) / 2)[:, None]
        w = w + sigma * (rng.standard_normal((m, NSAMP)) + 1j * rng.standard_normal((m, NSAMP)))
        x[sel, 0], x[sel, 1] = w.real, w.imag
    return {"x": x, "y": y, "snr": snr}


def dataset(n_train: int = 16000, n_test: int = 4000, seed: int = SEED) -> dict:
    """Train/test dict with sha256 (train and test use different derived seeds)."""
    tr, te = generate(n_train, seed), generate(n_test, seed + 1)
    ds = {"name": "modrec", "task": "classification", "classes": list(CLASSES), "seed": seed,
          "x_train": tr["x"], "y_train": tr["y"], "snr_train": tr["snr"],
          "x_test": te["x"], "y_test": te["y"], "snr_test": te["snr"]}
    ds["sha256"] = sha256_arrays({k: ds[k] for k in
                                  ("x_train", "y_train", "snr_train", "x_test", "y_test", "snr_test")})
    return ds
