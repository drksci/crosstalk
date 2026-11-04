"""TDL-Net-S in torch: 3 causal dilated ternary conv layers (+tanh), RC average pool, ternary linear readout.

Layer l:  u[o,n] = sum_{i,k} W_t[o,i,k] x[i, n-k*d]   (zeros before n=0; tap k = delay k*d samples)
          pre    = gamma_l * u + b            (BitNet b1.58: effective weight gamma*W_t)
          y      = g[o] * tanh(pre)           (g = 1 unless the per-line-gain fix is enabled)
This is exactly the virtual-tier semantics  y = gain * tanh((u + bias_e)/vsat)  with vsat = 1/gamma,
bias_e = b/gamma, gain = g (see dlnn/model/export.py, docs/model_export.md).
`float_mode=True` gives the identical topology with unconstrained float weights (baseline).
`quant=True` adds the training-time 8-bit absmax ADC fake-quant on the input and on every layer output.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import fake_quant, ternary_ste

DEFAULT_ARCH = dict(widths=(2, 8, 8, 8), taps=8, dilations=(1, 4, 16), n_classes=8)


class LineAug:
    """Board-aware tolerance augmentation (M1a line): random Q corner + per-tap delay jitter, resampled from a dense FIR grid.

    dense [Q, K, (M+1)*P+1]: dense[q, k, j] = h_k(c0 + (j/P - 1) T_s) (see models/board_aware/line_fir_m1a.json).
    sample() -> FIR [K, M]: corner weight a ~ U(0,1) mixes corners 0 and 1; tap k is read at lag m + delta_k, delta_k ~ U(-jitter, jitter) (T_s units).
    """

    def __init__(self, dense, P: int, M: int, jitter: float = 0.14, p_ideal: float = 0.0):
        self.dense = torch.as_tensor(dense, dtype=torch.float32)
        self.P, self.M, self.jitter, self.p_ideal = int(P), int(M), float(jitter), float(p_ideal)

    def at(self, a: float, delta: torch.Tensor | None = None) -> torch.Tensor:
        d = self.dense
        h = d[0] if d.shape[0] == 1 else (1 - a) * d[0] + a * d[1]
        K = h.shape[0]
        delta = torch.zeros(K) if delta is None else delta
        pos = self.P + (torch.arange(self.M)[None, :] + delta[:, None]) * self.P            # [K, M] fractional grid index
        i0 = pos.floor().clamp(0, h.shape[1] - 2).long()
        fr = pos - i0
        return torch.gather(h, 1, i0) * (1 - fr) + torch.gather(h, 1, i0 + 1) * fr

    def sample(self) -> torch.Tensor:
        K = self.dense.shape[1]
        if self.p_ideal > 0 and float(torch.rand(())) < self.p_ideal:   # mix in the ideal unit-delay line
            return torch.eye(K, self.M)
        return self.at(float(torch.rand(())), (torch.rand(K) * 2 - 1) * self.jitter)


class TDLConv(nn.Module):
    def __init__(self, c_in: int, c_out: int, taps: int, dilation: int, float_mode: bool, line_gain: bool):
        super().__init__()
        self.c_in, self.c_out, self.K, self.d = c_in, c_out, taps, dilation
        self.float_mode = float_mode
        self.lam = 1.0  # ternarisation strength (training schedule only): W_used = lam*gamma*W_t + (1-lam)*W
        # latent weights indexed by tap k (k = delay index): W[o, i, k]
        std = 1.0 / math.sqrt(c_in * taps)
        self.W = nn.Parameter(torch.randn(c_out, c_in, taps) * std)
        self.b = nn.Parameter(torch.zeros(c_out))
        self.g = nn.Parameter(torch.ones(c_out)) if line_gain else None
        # per-layer input-referred amplifier gain alpha (>0), decoupled from gamma; ternary only. vsat = 1/(alpha*gamma)
        self.log_alpha = None if float_mode else nn.Parameter(torch.zeros(()))
        self.aug: LineAug | None = None  # optional training-time FIR sampler (replaces self.fir in train mode)
        self.fir: torch.Tensor | None = None  # optional measured line FIR [K, M] (board-aware training, d = 1 only)

    def alpha(self) -> torch.Tensor:
        return torch.exp(self.log_alpha) if self.log_alpha is not None else torch.ones(())

    def weights(self) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """(effective weight used in forward, W_t or None, gamma or None)."""
        if self.float_mode:
            return self.W, None, None
        W_eff, Wt, gamma = ternary_ste(self.W)
        if self.lam < 1.0:
            W_eff = self.lam * W_eff + (1.0 - self.lam) * self.W
        return W_eff, Wt, gamma

    def forward(self, x: torch.Tensor, fq: float | None = None) -> torch.Tensor:
        W_eff, _, _ = self.weights()
        if self.aug is not None and self.training:
            fir = self.aug.sample()
            Wl = torch.einsum("oik,km->oim", W_eff, fir)
            pre = F.conv1d(F.pad(x, (fir.shape[1] - 1, 0)), Wl.flip(-1) * self.alpha(), self.b)
            return self._post(torch.tanh(pre), fq)
        if self.fir is not None:  # tap k sees x through its measured FIR h_k[m]: W_long[o,i,m] = sum_k W[o,i,k] h_k[m]
            assert self.d == 1
            Wl = torch.einsum("oik,km->oim", W_eff, self.fir)
            pre = F.conv1d(F.pad(x, (self.fir.shape[1] - 1, 0)), Wl.flip(-1) * self.alpha(), self.b)
            return self._post(torch.tanh(pre), fq)
        pad = (self.K - 1) * self.d
        # conv1d is cross-correlation: out[n] = sum_j w[j] xp[n + j d], xp = x left-padded by pad -> tap k = K-1-j
        pre = F.conv1d(F.pad(x, (pad, 0)), W_eff.flip(-1) * self.alpha(), self.b, dilation=self.d)
        return self._post(torch.tanh(pre), fq)

    def _post(self, y: torch.Tensor, fq: float | None) -> torch.Tensor:
        if self.g is not None:
            y = y * self.g[None, :, None]
        if fq is not None:  # ADC / int8 line quantiser; |y| <= max|g|
            gmax = self.g.detach().abs().max() if self.g is not None else 1.0
            y = fake_quant(y, gmax, 8)
        return y


class TDLNet(nn.Module):
    def __init__(self, float_mode: bool = False, line_gain: bool = False, input_scale: float = 1.0,
                 input_fullscale: float = 4.0, float_readout: bool = False, widths=None, taps=None, dilations=None, n_classes=None,
                 line_fir=None, line_aug: "LineAug | None" = None):
        super().__init__()
        a = DEFAULT_ARCH
        self.widths = tuple(widths or a["widths"])
        self.taps = taps or a["taps"]
        self.dilations = tuple(dilations or a["dilations"])
        self.n_classes = n_classes or a["n_classes"]
        self.float_mode, self.line_gain = float_mode, line_gain
        self.lam = 1.0
        self.float_readout = float_readout or float_mode  # experiment switch: unconstrained digital readout
        self.input_scale, self.input_fullscale = float(input_scale), float(input_fullscale)
        self.convs = nn.ModuleList([
            TDLConv(self.widths[l], self.widths[l + 1], self.taps, self.dilations[l], float_mode, line_gain)
            for l in range(len(self.dilations))])
        self.line_fir = None if line_fir is None else [list(r) for r in line_fir]
        if self.line_fir is not None:
            fir = torch.tensor(self.line_fir, dtype=torch.float32)
            for c in self.convs:
                c.fir = fir
        if line_aug is not None:   # not part of config(): training-time only; eval uses line_fir
            for c in self.convs:
                c.aug = line_aug      # plain attribute (LineAug is not an nn.Module): state_dict unchanged
        f = self.widths[-1]
        self.R = nn.Parameter(torch.randn(self.n_classes, f) / math.sqrt(f))
        self.rb = nn.Parameter(torch.zeros(self.n_classes))
        # per-output-line readout scale (digital post-gain), only with the line-gain fix
        self.rs = nn.Parameter(torch.ones(self.n_classes)) if line_gain else None
        self.log_ra = None if float_mode else nn.Parameter(torch.zeros(()))  # digital logit scale (ternary readout)

    def readout_weights(self) -> torch.Tensor:
        if self.float_readout:
            W = self.R
        else:
            W = ternary_ste(self.R)[0]
            W = (self.lam * W + (1.0 - self.lam) * self.R) * torch.exp(self.log_ra)
        return W * self.rs[:, None] if self.rs is not None else W

    def forward(self, x: torch.Tensor, quant: bool = False, return_layers: bool = False):
        """x [B,2,T] raw I/Q -> logits [B,n_classes]. quant=True: training-time 8-bit fake-quant."""
        h = x * self.input_scale
        if quant and not self.float_mode:
            h = fake_quant(h, self.input_fullscale * self.input_scale, 8)
        outs = []
        for c in self.convs:
            h = c(h, fq=1.0 if (quant and not self.float_mode) else None)
            outs.append(h)
        feat = h.mean(dim=2)  # RC average pool
        logits = feat @ self.readout_weights().T + self.rb
        return (logits, outs) if return_layers else logits

    @classmethod
    def from_config(cls, cfg: dict) -> "TDLNet":
        return cls(**cfg)

    def set_line_fir(self, fir) -> None:
        """Fix the (eval-time) per-tap FIR [K][M]; None = ideal unit delay."""
        self.line_fir = None if fir is None else [list(r) for r in fir]
        t = None if fir is None else torch.tensor(self.line_fir, dtype=torch.float32)
        for c in self.convs:
            c.fir = t

    def set_lam(self, lam: float) -> None:
        self.lam = lam
        for c in self.convs:
            c.lam = lam

    def config(self) -> dict:
        return dict(float_mode=self.float_mode, line_gain=self.line_gain, float_readout=self.float_readout, input_scale=self.input_scale,
                    input_fullscale=self.input_fullscale, widths=list(self.widths), taps=self.taps,
                    dilations=list(self.dilations), n_classes=self.n_classes, line_fir=self.line_fir)
