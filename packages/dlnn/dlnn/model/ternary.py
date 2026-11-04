"""BitNet b1.58 ternary quantisation + 8-bit fake-quant with straight-through estimators (torch)."""
from __future__ import annotations

import torch


def ternarize(W: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """W_t = clip(round(W / gamma), -1, 1), gamma = mean|W| (per tensor = per layer). Returns (W_t, gamma)."""
    gamma = W.abs().mean().clamp_min(eps)
    return torch.clamp(torch.round(W / gamma), -1.0, 1.0), gamma


def ternary_ste(W: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Straight-through ternarisation. Returns (W_eff, W_t, gamma).

    W_eff = gamma * W_t in the forward pass and identity-gradient to the latent W in the backward pass.
    """
    Wt, gamma = ternarize(W)
    W_eff = W + (gamma * Wt - W).detach()
    return W_eff, Wt.detach(), gamma.detach()


def round_ste(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def fake_quant(x: torch.Tensor, fullscale: float | torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Symmetric absmax fake-quant on a fixed grid: levels = 2^(bits-1)-1, step = fullscale/levels, clip + STE.

    Matches the E tier ADC (`dlnn.tiers.emulated.quantize_input`) up to the rounding mode of exact .5 ties.
    """
    levels = 2 ** (bits - 1) - 1
    step = fullscale / levels
    q = torch.clamp(round_ste(x / step), -levels, levels)
    return q * step
