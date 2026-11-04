"""scikit-rf as an independent referee for the line models.

Two jobs, deliberately kept apart.

:func:`characterize_line` builds one microstrip line with scikit-rf's ``MLine``
media -- Hammerstad-Jensen with Kirschning-Jansen dispersion, a
Djordjevic-Svensson dielectric and a finite conductor thickness -- so that the
closed form in :func:`topologies.microstrip_z0_eps_eff` has something to be
measured against. That closed form is explicitly quasi-static and has no
thickness term; scikit-rf has neither restriction, so the two are *expected*
to differ and the bench's job is to say by how much and why, not to make them
agree. Setting ``t_m=None`` is the attribution experiment: it isolates the
thickness correction from dispersion and from the dielectric model.

:func:`coupled_pair_network` assembles a 4-port for an edge-coupled pair from
its even- and odd-mode impedance matrices. The pair is symmetric about its
midline, so a symmetric (even) excitation sees an open there and an
antisymmetric (odd) one a short; each mode therefore travels as its own
independent 2-port, and the 4-port is recovered by transforming back. That is
why the modal matrices can be stacked rather than solved for.

Port order is scikit-rf's own convention for a coupled section:
``[1, 2, 3, 4] = [near-A, far-A, near-B, far-B]``.

scikit-rf is BSD, so unlike ``spice_runner`` this module carries no licence
obligation into the package. It still only imports under the bench group.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skrf import Frequency, Network
from skrf.media import MLine

from topologies import C_LIGHT, skin_resistance_per_m, z_to_s

# scikit-rf spells these ep_r / tand, not eps_r / tan_delta
_COPPER_RHO = 1.68e-8


@dataclass(frozen=True)
class LineReference:
    """scikit-rf's answer for one microstrip line, kept in its own units.

    ``z0`` is complex -- a lossy line's characteristic impedance is complex --
    so the comparison against a real closed form has to say which part it
    means. ``eps_eff`` is real by construction: it is ``(beta c / omega)^2``,
    the phase-velocity definition, which is what a quasi-static ``eps_eff``
    means. scikit-rf also exposes ``ep_reff`` as ``(gamma c / omega)^2``, a
    complex quantity that carries the loss; reporting that against a real
    number would compare two different things.
    """

    frequency_hz: np.ndarray
    z0: np.ndarray
    eps_eff: np.ndarray
    alpha_np_per_m: np.ndarray
    beta_rad_per_m: np.ndarray
    network: Network
    z0_ref: float
    thickness_m: float | None


def characterize_line(
    *,
    w_m: float,
    h_m: float,
    freq_hz: np.ndarray,
    length_m: float,
    t_m: float | None = None,
    eps_r: float = 4.3,
    tan_delta: float = 0.0,
    rho: float = _COPPER_RHO,
    z0_ref: float = 50.0,
) -> LineReference:
    """Build scikit-rf's single-line model over ``freq_hz``.

    Pass ``t_m=None`` for an infinitely thin conductor -- the setting that
    makes scikit-rf comparable to a closed form without a thickness term.
    """
    freq_hz = _validate_sweep(freq_hz)

    media = MLine(
        frequency=Frequency.from_f(freq_hz, unit="hz"),
        w=w_m,
        h=h_m,
        t=t_m,
        ep_r=eps_r,
        tand=tan_delta,
        rho=rho,
    )

    gamma = np.asarray(media.gamma)
    beta = np.asarray(gamma.imag)
    eps_eff = (beta * C_LIGHT / (2.0 * np.pi * freq_hz)) ** 2

    network = media.line(length_m, unit="m")
    network.renormalize(z0_ref)

    return LineReference(
        frequency_hz=freq_hz,
        z0=np.asarray(media.z0),
        eps_eff=eps_eff,
        alpha_np_per_m=np.asarray(gamma.real),
        beta_rad_per_m=beta,
        network=network,
        z0_ref=float(z0_ref),
        thickness_m=t_m,
    )


def dielectric_alpha_np_per_m(freq_hz: np.ndarray, eps_eff: float, tan_delta: float) -> np.ndarray:
    """Dielectric attenuation ``k0 sqrt(eps_eff) tan(delta) / 2``, in Np/m."""
    freq_hz = np.asarray(freq_hz, dtype=float)
    return (2.0 * np.pi * freq_hz / C_LIGHT) * np.sqrt(eps_eff) * tan_delta / 2.0


def conductor_alpha_np_per_m(freq_hz: np.ndarray, width_m: float, z_mode_ohm: float) -> np.ndarray:
    """Conductor attenuation ``R / 2Zc``, in Np/m.

    ``R`` is the skin-effect series resistance per metre for a strip of the
    full width -- the same for both modes, which is an approximation: even and
    odd modes put current in different places. It is a small correction next
    to the modal impedance difference, and the bench reports it rather than
    pretending the modal losses are exact.
    """
    freq_hz = np.asarray(freq_hz, dtype=float)
    resistance = np.array([skin_resistance_per_m(float(f), width_m) for f in freq_hz])
    return resistance / (2.0 * float(z_mode_ohm))


def mode_z_matrix(
    z_mode_ohm: float,
    eps_mode: float,
    freq_hz: np.ndarray,
    length_m: float,
    alpha_np_per_m: float | np.ndarray = 0.0,
) -> np.ndarray:
    """2-port impedance matrix of one propagation mode, shape ``(F, 2, 2)``.

    ``Zc coth(g l)`` on the diagonal and ``Zc / sinh(g l)`` off it -- the same
    pair :func:`topologies.chain_to_z` derives for a scalar line, reached here
    through ``gamma`` so a lossy mode needs no separate branch.
    """
    freq_hz = _validate_sweep(freq_hz)
    beta = 2.0 * np.pi * freq_hz / C_LIGHT * np.sqrt(eps_mode)
    gamma_l = (np.asarray(alpha_np_per_m, dtype=float) + 1j * beta) * length_m

    z11 = float(z_mode_ohm) / np.tanh(gamma_l)
    z12 = float(z_mode_ohm) / np.sinh(gamma_l)

    out = np.zeros((freq_hz.size, 2, 2), dtype=np.complex128)
    out[:, 0, 0] = out[:, 1, 1] = z11
    out[:, 0, 1] = out[:, 1, 0] = z12
    return out


def coupled_pair_network(
    *,
    z_even_ohm: float,
    eps_even: float,
    z_odd_ohm: float,
    eps_odd: float,
    freq_hz: np.ndarray,
    length_m: float,
    z0_ref: float = 50.0,
    alpha_even_np_per_m: float | np.ndarray = 0.0,
    alpha_odd_np_per_m: float | np.ndarray = 0.0,
) -> Network:
    """4-port network of an edge-coupled pair from its modal Z-matrices.

    With ``[1, 3, 2, 4]`` as the intermediate ordering (both near ends, then
    both far ends), let ``T = H / sqrt(2)`` be the Hadamard transform and ``P``
    swap the two middle indices so the vector reads mode-by-mode instead of
    port-by-port. Each mode is then an independent 2-port, and

    .. code-block:: text

        Z_pair = (I x T) P blkdiag(Z_even, Z_odd) P (I x T)

    ``T`` is an involution and ``P`` a transposition, so both are their own
    inverse. Setting ``z_even == z_odd`` collapses the cross terms to zero,
    which is the check :func:`test_rf_char.test_uncoupled_pair_decouples`
    makes -- a construction this indirect earns its keep only if that limit
    comes out right.
    """
    freq_hz = _validate_sweep(freq_hz)
    z_even = mode_z_matrix(z_even_ohm, eps_even, freq_hz, length_m, alpha_even_np_per_m)
    z_odd = mode_z_matrix(z_odd_ohm, eps_odd, freq_hz, length_m, alpha_odd_np_per_m)

    hadamard = np.array([[1.0, 1.0], [1.0, -1.0]]) / np.sqrt(2.0)
    pair_transform = np.kron(np.eye(2), hadamard)
    mode_first = [0, 2, 1, 3]  # [ae, ao, ae', ao'] -> [ae, ae', ao, ao']

    z_pair = np.zeros((freq_hz.size, 4, 4), dtype=np.complex128)
    for i in range(freq_hz.size):
        modal = np.zeros((4, 4), dtype=np.complex128)
        modal[:2, :2] = z_even[i]
        modal[2:, 2:] = z_odd[i]
        reordered = modal[np.ix_(mode_first, mode_first)]
        z_pair[i] = pair_transform @ reordered @ pair_transform

    # intermediate order [1, 3, 2, 4] -> [1, 2, 3, 4]
    port_order = [0, 2, 1, 3]
    s_mat = z_to_s(z_pair, z0_ref)[:, port_order, :][:, :, port_order]

    return Network(
        frequency=Frequency.from_f(freq_hz, unit="hz"),
        s=s_mat,
        z0=np.full((freq_hz.size, 4), float(z0_ref)),
        name="coupled_pair",
    )


def export_touchstone(network: Network, path) -> str:
    """Write ``network`` to a Touchstone file and return the path written.

    scikit-rf refuses to write a network whose port impedances are complex
    unless it is told what to renormalize to, so ``r_ref`` is always passed
    rather than left to depend on whether the caller remembered to
    :meth:`~skrf.network.Network.renormalize` first.
    """
    from pathlib import Path

    destination = Path(path)
    network.write_touchstone(
        filename=destination.name,
        dir=str(destination.parent),
        r_ref=float(np.real(network.z0.flat[0])),
    )
    return str(destination)


def _validate_sweep(freq_hz) -> np.ndarray:
    """A sweep needs at least two points: scikit-rf builds a ``Frequency`` grid."""
    freq_hz = np.asarray(freq_hz, dtype=float)
    if freq_hz.ndim != 1:
        raise ValueError(f"freq_hz must be 1-D, got shape {freq_hz.shape}")
    if freq_hz.size < 2:
        raise ValueError(f"a frequency sweep needs at least 2 points, got {freq_hz.size}")
    if np.any(freq_hz <= 0):
        raise ValueError("frequencies must be > 0 Hz")
    if np.any(np.diff(freq_hz) <= 0):
        raise ValueError("frequencies must be strictly increasing")
    return freq_hz
