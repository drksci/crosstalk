"""Three real circuit topologies, reduced to per-unit-length RLCG matrices.

The delay-tensor used to say "coupling" and mean ``sigmoid(mask) *
neighbour_strength``. This module is the replacement vocabulary: a topology is
a physical object with dimensions, material and a loss budget, and everything
downstream -- the ngspice deck, the scikit-rf network, the extracted macromodel
-- reads the same four matrices.

Every topology is described by per-unit-length

.. code-block:: text

    Z(w) = R + j w L        series impedance      [ohm/m]
    Y(w) = G + j w C        shunt admittance      [S/m]

``R`` and ``G`` are evaluated once at a stated ``loss_freq_hz`` rather than
swept, because a compact macromodel needs one time-invariant system. That is a
narrowband linearisation and it is called out in every result written from
here; skin effect and dielectric loss are both frequency dependent, so the
model is honest only in a band around the linearisation point.

Three topologies, deliberately different in all three axes:

========================  ========  ==========  ==============================
name                      N         params from sections
========================  ========  ==========  ==============================
``mcl``                   4         electrical  64   distributed bus
``microstrip_pair``       2         geometric   48   coupled microstrip
``lc_ladder``             2         lumped       4   coarse synthetic delay line
========================  ========  ==========  ==============================

``mcl`` is a four-conductor PCB bus whose coupling is *tridiagonal*: conductor
``w`` only talks to ``w-1`` and ``w+1``. That is the same nearest-neighbour
assumption the recurrence made, except here it falls out of physical adjacency
instead of being asserted.

Two solutions of the same matrices, because they cross-check each other:

* :func:`chain_distributed` -- exact chain matrix of the uniform line, from
  matrix functions of ``Z Y``. This is the ground truth for the *continuous*
  line.
* :func:`chain_ladder` -- product of ``sections`` lumped pi-sections. This is
  literally the circuit that gets written to SPICE, so it must match ngspice to
  round-off, and it must converge onto ``chain_distributed`` as ``sections``
  grows. That convergence is the bench's evidence that the discretisation is
  fine enough.

Pure NumPy: no scikit-rf, no PySpice, so ``benches/tests`` runs in the plain uv
venv without the bench group installed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# SI constants. CODATA 2018; enough digits that geometry round-trips cleanly.
C_LIGHT = 299_792_458.0
MU_0 = 4.0e-7 * np.pi
EPS_0 = 8.8541878128e-12
SIGMA_CU = 5.80e7  # annealed copper, S/m

SUPPORTED_TOPOLOGIES = ("mcl", "microstrip_pair", "lc_ladder")


@dataclass(frozen=True)
class Topology:
    """One physical line, as per-unit-length matrices plus how to section it.

    ``R``/``L``/``G``/``C`` are all ``(conductors, conductors)``. ``L`` and ``C``
    are the Maxwell inductance and capacitance matrices: ``C`` has a positive
    diagonal and non-positive off-diagonal (charge on one conductor falls when
    a neighbour is raised), while ``L`` has a *positive* off-diagonal because
    parallel currents reinforce.
    """

    name: str
    kind: str
    conductors: int
    length_m: float
    R: np.ndarray
    L: np.ndarray
    G: np.ndarray
    C: np.ndarray
    sections: int
    loss_freq_hz: float
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.conductors
        for label, matrix in (("R", self.R), ("L", self.L), ("G", self.G), ("C", self.C)):
            if np.shape(matrix) != (n, n):
                raise ValueError(f"{self.name}.{label} must be {(n, n)}, got {np.shape(matrix)}")
        if self.length_m <= 0:
            raise ValueError(f"{self.name}.length_m must be > 0, got {self.length_m}")
        if self.sections < 1:
            raise ValueError(f"{self.name}.sections must be >= 1, got {self.sections}")
        if self.loss_freq_hz <= 0:
            raise ValueError(f"{self.name}.loss_freq_hz must be > 0, got {self.loss_freq_hz}")

    @property
    def port_count(self) -> int:
        """Two ports per conductor: one at each end."""
        return 2 * self.conductors

    @property
    def section_length_m(self) -> float:
        return self.length_m / self.sections

    @property
    def delay_s(self) -> float:
        """One-way delay from the lossless phase velocity, seconds."""
        return float(self.length_m * np.sqrt(np.mean(self.L * self.C)))


# ---------------------------------------------------------------------------
# geometry -> scalar microstrip parameters
# ---------------------------------------------------------------------------


def microstrip_z0_eps_eff(w: float, h: float, eps_r: float) -> tuple[float, float]:
    """Single microstrip ``Z0`` and ``eps_eff`` (Pozar, *Microwave Eng.* 6.3).

    The quasi-static Hammerstad form: the field straddles substrate and air, so
    ``eps_eff`` sits between ``1`` and ``eps_r``. The ``0.04 (1-u)^2`` term is
    the narrow-line correction, needed below ``w/h = 1`` where fringing reaches
    both sides of the trace.

    Validity: quasi-static, so it carries no dispersion. scikit-rf's ``MLine``
    *does* model dispersion (Kirschning-Jansen), which is exactly why the bench
    compares the two rather than trusting either.
    """
    if w <= 0 or h <= 0:
        raise ValueError(f"microstrip w and h must be > 0, got w={w}, h={h}")
    if eps_r < 1:
        raise ValueError(f"eps_r must be >= 1, got {eps_r}")

    u = w / h
    narrow = (1.0 + 12.0 / u) ** -0.5 + (0.04 * (1.0 - u) ** 2 if u < 1.0 else 0.0)
    eps_eff = 0.5 * (eps_r + 1.0) + 0.5 * (eps_r - 1.0) * narrow

    if u <= 1.0:
        z0 = (60.0 / np.sqrt(eps_eff)) * np.log(8.0 * h / w + w / (4.0 * h))
    else:
        z0 = (120.0 * np.pi) / (
            np.sqrt(eps_eff) * (u + 1.393 + 0.667 * np.log(u + 1.444))
        )
    return float(z0), float(eps_eff)


def coupling_from_gap(gap_m: float, h: float) -> float:
    """Coupling coefficient ``k`` of an edge-coupled pair, from the gap.

    ``k = sech(pi gap / 2h)`` is the conformal-mapping result for two coplanar
    strips over a ground plane: the odd-mode field is squeezed into the gap, so
    coupling decays with gap scaled by substrate height rather than by gap
    alone. Limits are the ones that matter physically --

    * ``gap -> 0``: traces touch, ``k -> 1``, odd mode becomes a short.
    * ``gap -> inf``: ``k -> 0``, the pair decouples into two isolated lines.

    ``k`` is then used for *both* inductance and capacitance (the
    ``k-equivalence`` approximation). It is exact only for a uniform dielectric;
    microstrip is inhomogeneous, so this is the one engineering assumption in
    topology (b) and the bench measures how much it costs.
    """
    if gap_m < 0:
        raise ValueError(f"gap must be >= 0, got {gap_m}")
    if h <= 0:
        raise ValueError(f"h must be > 0, got {h}")
    x = 0.5 * np.pi * gap_m / h
    if x > 700.0:
        # cosh overflows past ~710, but sech has long since underflowed to
        # something no float can tell from zero -- return the limit rather
        # than a RuntimeWarning and an inf
        return 0.0
    return float(1.0 / np.cosh(x))


def skin_resistance_per_m(freq_hz: float, width_m: float) -> float:
    """AC series resistance per metre from the skin effect.

    ``Rs = sqrt(pi f mu / sigma)`` is the surface resistance; dividing by the
    trace width gives ohm/m. This ignores current crowding at the trace edges,
    which is the dominant correction in reality -- a hand-rolled formula would
    claim more accuracy than it has, so the bench leans on the ladder's own
    agreement with ngspice instead.
    """
    if freq_hz <= 0 or width_m <= 0:
        raise ValueError(f"freq and width must be > 0, got {freq_hz}, {width_m}")
    rs = np.sqrt(np.pi * freq_hz * MU_0 / SIGMA_CU)
    return float(rs / width_m)


# ---------------------------------------------------------------------------
# topology builders
# ---------------------------------------------------------------------------


def multiconductor_bus(
    conductors: int = 4,
    *,
    length_m: float = 0.10,
    l_self: float = 320e-9,
    k_cap: float = 0.35,
    eps_eff: float = 3.0,
    width_m: float = 0.3e-3,
    loss_freq_hz: float = 1e9,
    tan_delta: float = 0.02,
    sections: int = 64,
    name: str = "mcl",
) -> Topology:
    """A multi-conductor bus with adjacent-line crosstalk.

    The nearest-neighbour structure lives in the **capacitance**. ``k_cap`` is
    the fraction of the isolated per-metre capacitance that couples into each
    neighbour, so

    .. code-block:: text

        C = c_self (I - k_cap A)        A: adjacency, tridiagonal

    has an off-diagonal only between neighbours -- conductor 0 never reaches
    conductor 2 through the electric field -- and every one of them is ``<= 0``.
    That sign is the whole point: a Maxwell capacitance matrix with negative
    off-diagonals is exactly the thing :func:`spice_shunt_and_bridge` can turn
    into two-terminal SPICE capacitors. ``l_self`` fixes the scale, by setting
    the mean of the derived ``L``'s diagonal.

    Why the capacitance rather than the inductance? Because it has to be. The
    inverse of a tridiagonal matrix with positive off-diagonals has a
    **checkerboard sign pattern**: entries run ``(+, -, +, -)`` with distance.
    So deriving ``C`` from a tridiagonal ``L`` puts a *positive* off-diagonal on
    ``C`` at distance 2 for any bus of three or more conductors -- and a positive
    off-diagonal is a negative capacitance, which no deck of passive elements
    can contain. A pair is the exception, which is why the two formulations
    agree exactly when ``conductors == 2`` and only diverge from three up.

    ``L`` is then derived as ``_modal_partner(C)``, so ``L C = (eps_eff/c^2) I``
    and every mode travels at ``c/sqrt(eps_eff)`` whatever the coupling. The
    magnetic side does reach conductor 2 -- ``L`` is dense, its entries falling
    off like ``k_cap^distance`` -- so the honest claim is nearest-neighbour
    *exact* capacitance and nearest-neighbour *dominant* inductance, at
    ``k_cap^2`` for the next-but-one conductor.
    """
    if conductors < 2:
        raise ValueError(f"conductors must be >= 2, got {conductors}")
    if not 0.0 <= k_cap < 0.5:
        raise ValueError(
            f"k_cap must be in [0, 0.5) so C stays positive definite with "
            f"non-negative row sums, got {k_cap}"
        )
    if eps_eff < 1:
        raise ValueError(f"eps_eff must be >= 1, got {eps_eff}")

    adjacency = _adjacency(conductors)
    c_shape = np.eye(conductors) - k_cap * adjacency

    # Scale c_self so the derived L comes out with mean diagonal l_self.
    # Probing at c_self = 1 first is what makes that one division, not a
    # root-find: L is linear in 1/c_self.
    l_probe = _modal_partner(c_shape, [eps_eff] * conductors)
    c_self = float(np.mean(np.diag(l_probe))) / l_self
    c_mat = c_self * c_shape
    l_mat = _modal_partner(c_mat, [eps_eff] * conductors)

    omega = 2.0 * np.pi * loss_freq_hz
    r_mat = skin_resistance_per_m(loss_freq_hz, width_m) * np.eye(conductors)
    g_mat = omega * tan_delta * c_mat

    return Topology(
        name=name,
        kind="mcl",
        conductors=conductors,
        length_m=length_m,
        R=r_mat,
        L=l_mat,
        G=g_mat,
        C=c_mat,
        sections=sections,
        loss_freq_hz=loss_freq_hz,
        meta={
            "c_self_F_per_m": c_self,
            "k_cap": k_cap,
            "l_self_H_per_m": float(np.mean(np.diag(l_mat))),
            "eps_eff": eps_eff,
            "width_m": width_m,
            "tan_delta": tan_delta,
            "coupling_reach": "exact nearest-neighbour in C; L dense, decays as k_cap^distance",
        },
    )


def microstrip_pair(
    *,
    w: float = 1.5e-3,
    gap: float = 0.4e-3,
    h: float = 0.8e-3,
    t: float = 35e-6,
    eps_r: float = 4.3,
    tan_delta: float = 0.02,
    length_m: float = 0.08,
    loss_freq_hz: float = 1e9,
    sections: int = 48,
    name: str = "microstrip_pair",
) -> Topology:
    """An edge-coupled microstrip pair: ``w, gap, h, t, eps_r, tan_delta`` in.

    Geometry goes in; electromagnetic parameters come out. ``microstrip_z0_eps_eff``
    gives the isolated line's ``Z0`` and ``eps_eff``, which set the *scale* of
    the line,

    .. code-block:: text

        Ls = Z0 sqrt(eps_eff) / c        Cg = sqrt(eps_eff) / (c Z0)
        Ls Cg = eps_eff / c^2            Ls / Cg = Z0^2

    and ``coupling_from_gap`` gives ``k``, which sets how the pair splits that
    scale into even and odd modes:

    .. code-block:: text

        Zeven = Z0 sqrt((1+k)/(1-k))     Zodd = Z0 sqrt((1-k)/(1+k))

    The two are reciprocal, so their geometric mean is exactly the isolated
    ``Z0`` and ``(Zeven - Zodd)/(Zeven + Zodd)`` is exactly ``k`` -- the
    coupling coefficient is self-consistent rather than an independent knob.

    Both limits are checked in tests: ``k -> 0`` recovers two isolated ``Z0``
    lines with the modal split closed. ``k -> 1`` -- the traces in contact --
    shorts the odd mode and drives ``Zeven`` to infinity, so it is refused
    rather than silently divided through.

    ``t`` is recorded but not used: the quasi-static ``Z0`` above ignores trace
    thickness, and so does ``Rs / w``. Carrying an unused field would imply a
    thickness correction that is not implemented.
    """
    z0, eps_eff = microstrip_z0_eps_eff(w, h, eps_r)
    k = coupling_from_gap(gap, h)
    if k >= 1.0:
        raise ValueError(
            f"gap={gap} m puts the traces in contact: k=1 shorts the odd mode and "
            "drives Zeven to infinity, so there is no realizable pair. Use gap > 0."
        )

    # Mode impedances. p = sqrt((1+k)/(1-k)) so Zeven = Z0 p and Zodd = Z0/p,
    # which makes the geometric mean exactly the isolated Z0 and the ratio
    # (Zeven - Zodd)/(Zeven + Zodd) exactly k -- the coupling coefficient is
    # therefore self-consistent rather than an independent knob.
    p = np.sqrt((1.0 + k) / (1.0 - k))
    z_even = z0 * p
    z_odd = z0 / p

    # Modal birefringence. Microstrip is inhomogeneous: the even mode's field
    # sits mostly in the substrate and the odd mode's straddles the gap in air,
    # so they must travel at different speeds or far-end crosstalk would never
    # accumulate. Splitting in log space keeps the geometric mean at eps_eff
    # while letting the pair open up as coupling tightens, and the k -> 1
    # limits land where they should: eps_even -> eps_r, eps_odd -> eps_eff^2/eps_r.
    budget = np.log(eps_r / eps_eff)
    eps_even = eps_eff * np.exp(+k * budget)
    eps_odd = eps_eff * np.exp(-k * budget)

    c_light = C_LIGHT
    l_even = z_even * np.sqrt(eps_even) / c_light
    l_odd = z_odd * np.sqrt(eps_odd) / c_light
    c_even = np.sqrt(eps_even) / (c_light * z_even)
    c_odd = np.sqrt(eps_odd) / (c_light * z_odd)

    # eigh() sorts ascending, and for a pair that is [odd, even] because
    # L_odd = Ls(1-k) < L_even = Ls(1+k).
    l_mat = np.array(
        [
            [0.5 * (l_even + l_odd), 0.5 * (l_even - l_odd)],
            [0.5 * (l_even - l_odd), 0.5 * (l_even + l_odd)],
        ]
    )
    c_mat = np.array(
        [
            [0.5 * (c_even + c_odd), 0.5 * (c_even - c_odd)],
            [0.5 * (c_even - c_odd), 0.5 * (c_even + c_odd)],
        ]
    )

    omega = 2.0 * np.pi * loss_freq_hz
    r_mat = skin_resistance_per_m(loss_freq_hz, w) * np.eye(2)
    g_mat = omega * tan_delta * c_mat

    return Topology(
        name=name,
        kind="microstrip_pair",
        conductors=2,
        length_m=length_m,
        R=r_mat,
        L=l_mat,
        G=g_mat,
        C=c_mat,
        sections=sections,
        loss_freq_hz=loss_freq_hz,
        meta={
            "w_m": w,
            "gap_m": gap,
            "h_m": h,
            "t_m": t,
            "eps_r": eps_r,
            "tan_delta": tan_delta,
            "z0_single_ohm": z0,
            "eps_eff": eps_eff,
            "k_coupling": k,
            "z_even_ohm": z_even,
            "z_odd_ohm": z_odd,
            "eps_even": eps_even,
            "eps_odd": eps_odd,
            "velocity_even_m_per_s": c_light / np.sqrt(eps_even),
            "velocity_odd_m_per_s": c_light / np.sqrt(eps_odd),
        },
    )


def lc_ladder(
    *,
    conductors: int = 2,
    sections: int = 4,
    l_self: float = 280e-9,
    k_cap: float = 0.30,
    length_m: float = 0.010,
    loss_freq_hz: float = 1e9,
    r_series: float = 0.02,
    tan_delta: float = 0.0,
    eps_eff: float = 3.0,
    name: str = "lc_ladder",
) -> Topology:
    """A coarse lumped LC delay line -- the Kron-reduced end of the spectrum.

    ``l_self = 280 nH/m`` with ``eps_eff = 3`` lands on ``Z0 = 48.5 ohm``, so
    this is a real 50-ohm line rather than an arbitrary ladder. Four sections
    over a centimetre is *not* a faithful model of a 1 cm line: one section is
    about a seventieth of a guided wavelength at 1 GHz, which puts the ladder's
    own cutoff near 20 GHz. That is the point of including it -- it is the
    coarsest discretisation the pipeline has to survive, and it is what a
    macromodel fitted on the fine topologies still has to describe.

    Losses are deliberately near zero (``r_series`` only, no dielectric), so
    the topology isolates *dispersion from discretisation* rather than mixing
    it with attenuation.
    """
    if sections < 1:
        raise ValueError(f"sections must be >= 1, got {sections}")
    if r_series < 0:
        raise ValueError(f"r_series must be >= 0, got {r_series}")

    ladder = multiconductor_bus(
        conductors,
        length_m=length_m,
        l_self=l_self,
        k_cap=k_cap,
        eps_eff=eps_eff,
        loss_freq_hz=loss_freq_hz,
        tan_delta=tan_delta,
        sections=sections,
        name=name,
    )
    r_mat = r_series * np.eye(conductors)
    return Topology(
        name=ladder.name,
        kind=ladder.kind,
        conductors=ladder.conductors,
        length_m=ladder.length_m,
        R=r_mat,
        L=ladder.L,
        G=ladder.G,
        C=ladder.C,
        sections=sections,
        loss_freq_hz=loss_freq_hz,
        meta={
            **ladder.meta,
            "r_series_ohm": r_series,
            "sections": sections,
            "kind_note": "lumped ladder; dispersion is discretisation, not physics",
        },
    )


def default_topology(kind: str) -> Topology:
    """Build the bench's canonical instance of each topology."""
    if kind == "mcl":
        return multiconductor_bus()
    if kind == "microstrip_pair":
        return microstrip_pair()
    if kind == "lc_ladder":
        return lc_ladder()
    raise ValueError(f"unknown topology {kind!r}; expected one of {SUPPORTED_TOPOLOGIES}")


def spice_shunt_and_bridge(
    topology: Topology,
    *,
    matrix: np.ndarray | None = None,
    label: str = "C",
) -> tuple[np.ndarray, np.ndarray]:
    """Split a Maxwell matrix into the two-terminal elements a SPICE deck can contain.

    ``matrix`` defaults to ``topology.C``; pass ``topology.G`` to realize the
    dielectric loss the same way, since ``G`` is a conductance matrix with the
    identical structure and needs resistors rather than capacitors. The
    arguments below say ``C`` throughout because that is the default.

    ``C`` is a Maxwell matrix -- charge on conductor ``c`` is
    ``sum_d C[c, d] v_d`` -- but SPICE only has two-terminal capacitors. A
    bridge ``B`` between ``c`` and ``d`` contributes ``B (v_c - v_d)``, so

    .. code-block:: text

        B[c, d] = -C[c, d]          (>= 0, Maxwell off-diagonals are negative)
        S[c]    = sum_d C[c, d]      (the row sum -- see below)

    The row sum is the part worth spelling out: the shunt is *not* the diagonal.
    Writing ``S = C[c, c]`` would double-count the bridge and raise the line's
    capacitance by the whole mutual term. Because a Maxwell matrix over a ground
    reference has non-negative row sums, ``S >= 0`` holds for any physical line;
    :func:`multiconductor_bus` and :func:`microstrip_pair` both assert it.

    Off-diagonal signs are checked before anything is clipped, because ``clip``
    is exactly how a non-Maxwell ``C`` would fail *silently*: a positive
    ``C[c, d]`` wants ``B = -C[c, d] < 0``, i.e. a negative capacitor, so the
    only values ``clip`` can produce are wrong ones. The charge it drops shows
    up as a diagonal error in the realized matrix and an analytic-versus-deck
    disagreement with no visible cause -- which is how ``mcl`` ran at 0.57
    before this check existed.
    """
    source = topology.C if matrix is None else matrix

    scale = max(float(np.max(np.abs(source))), 1e-30)
    off_diagonal = source - np.diag(np.diag(source))
    worst = float(np.max(off_diagonal))
    if worst > 1e-12 * scale:
        raise ValueError(
            f"{topology.name}: {label} has a positive off-diagonal ({worst:.3e}), which needs "
            f"a negative element and so no set of SPICE two-terminal parts realizes it; a "
            f"Maxwell matrix must have off-diagonals <= 0"
        )

    bridge = np.clip(-source, 0.0, None)
    np.fill_diagonal(bridge, 0.0)
    shunt = source.sum(axis=1)
    if np.any(shunt < -1e-18 * scale):
        raise ValueError(
            f"{topology.name}: {label} has a negative row sum ({shunt.min():.3e}), so it is "
            "not a Maxwell matrix and no set of SPICE two-terminal parts realizes it"
        )
    return shunt, bridge


def _adjacency(conductors: int) -> np.ndarray:
    """Tridiagonal adjacency: 1 on the first off-diagonals, 0 elsewhere."""
    adjacency = np.zeros((conductors, conductors))
    idx = np.arange(conductors - 1)
    adjacency[idx, idx + 1] = 1.0
    adjacency[idx + 1, idx] = 1.0
    return adjacency


def _modal_partner(given: np.ndarray, eps_modes: list[float]) -> np.ndarray:
    """The matrix ``Y`` with ``given @ Y = (eps_eff/c^2) I``, in ``given``'s own eigenbasis.

    The relation is symmetric -- ``L`` and ``C`` of a homogeneous line share one
    eigenbasis -- so the same routine pairs either one with the other: pass
    ``C`` to get ``L``, or ``L`` to get ``C``. Modal permittivity ``eps_p``
    fixes each pair through ``v = 1 / sqrt(x_p y_p)``, giving
    ``y_p = eps_p / (c^2 x_p)``, and ``Y = U diag(...) U^T`` in ``X``'s own
    eigenvectors ``U``.

    This is the whole reason the partner is derived rather than picked. The
    tempting spelling -- ``c_ground (I - k T)``, mirroring how the known matrix
    is written -- gives

    .. code-block:: text

        L C = Ls Cg (I - k^2 T^2)

    which for a pair is a scalar times the identity (``T^2 = I`` when there are
    only two conductors) but for a bus is *not*, so mode ``p`` comes out at
    speed ``1/sqrt(1 - k^2 t_p^2)`` and the line acquires 2-20% of invented
    dispersion. ``L C = (eps_eff/c^2) I`` is what a homogeneous line really
    does: same speed in every mode, regardless of coupling.
    """
    l_val, u = np.linalg.eigh(given)
    if np.any(l_val <= 0):
        raise ValueError(f"matrix is not positive definite (min eigenvalue {l_val.min():.3e})")

    eps = np.asarray(eps_modes, dtype=np.float64)
    if eps.shape != l_val.shape:
        raise ValueError(f"eps_modes must have {l_val.size} entries, got {eps.shape}")
    if np.any(eps < 1.0):
        raise ValueError(f"mode permittivities must be >= 1, got min {eps.min():.4f}")

    partner_mode = eps / (C_LIGHT**2 * l_val)
    partner = (u * partner_mode) @ u.T
    return 0.5 * (partner + partner.T)


# ---------------------------------------------------------------------------
# chain matrices
# ---------------------------------------------------------------------------


def chain_distributed(topology: Topology, freq_hz: float) -> np.ndarray:
    """Exact ``(2N, 2N)`` chain matrix of the *continuous* uniform line.

    Solves the telegrapher equations in matrix form. With ``Z = R + jwL`` and
    ``Y = G + jwC``,

    .. code-block:: text

        Gamma^2 = Z Y,        Zc = Gamma^-1 Z

        [V(0)]   [ cosh(Gl)    Zc sinh(Gl)  ] [V(l) ]
        [    ] = [                          ] [     ]
        [I(0)]   [ sinh(Gl)/Zc cosh(Gl)     ] [-I(l)]

    which for ``N = 1`` collapses to the textbook ``Z0 coth(g l)`` set of
    Z-parameters. Matrix functions come from an eigendecomposition of ``Z Y``,
    which is fine here because ``L`` and ``C`` are both polynomials of the same
    tridiagonal adjacency and therefore share eigenvectors.

    Eigenvalue signs are normalised to ``Re(gamma) >= 0``. Branch choice does
    not affect the result -- ``A`` and ``D`` are even in ``gamma`` while ``Zc``
    is odd, so ``Zc sinh`` and ``sinh / Zc`` are both even -- but normalising
    keeps the exponential from blowing up on round-off.
    """
    w = 2.0 * np.pi * freq_hz
    z_mat = topology.R + 1j * w * topology.L
    y_mat = topology.G + 1j * w * topology.C

    mu, vecs = np.linalg.eig(z_mat @ y_mat)
    gamma = _principal_sqrt(mu)
    inv_vecs = np.linalg.inv(vecs)

    cosh_gl = vecs @ np.diag(np.cosh(gamma * topology.length_m)) @ inv_vecs
    sinh_gl = vecs @ np.diag(np.sinh(gamma * topology.length_m)) @ inv_vecs

    gamma_mat = vecs @ np.diag(gamma) @ inv_vecs
    z_char = np.linalg.solve(gamma_mat, z_mat)

    a_mat = cosh_gl
    b_mat = z_char @ sinh_gl
    c_mat = sinh_gl @ np.linalg.inv(z_char)
    d_mat = cosh_gl
    return np.block([[a_mat, b_mat], [c_mat, d_mat]])


def pi_section(topology: Topology, freq_hz: float) -> np.ndarray:
    """Chain matrix of one lumped pi-section spanning ``length/sections``.

    Series impedance ``Zs`` is split across the conductors as the full
    ``(R + jwL) dl``; the shunt ``(G + jwC) dl`` is split in half and hung at
    each end of the section. Mutual inductance between conductors lands inside
    ``Zs`` as an off-diagonal term, so no ``K``-element bookkeeping is needed
    for the analytic form -- SPICE needs it, and ``rf_netlist.py`` adds it.

    Block order is shunt, series, shunt. Multiplying the three block matrices
    gives ``C = Y + Y Z Y / 4``, *not* the ``Y (1 + Z Y / 2)`` some references
    quote -- that second form is for a different shunt arrangement, and the
    difference is checked against a hand-built pi in the tests.
    """
    w = 2.0 * np.pi * freq_hz
    n = topology.conductors
    dl = topology.section_length_m

    z_series = (topology.R + 1j * w * topology.L) * dl
    y_shunt = (topology.G + 1j * w * topology.C) * dl

    eye = np.eye(n, dtype=complex)
    zero = np.zeros((n, n), dtype=complex)

    def shunt(admittance: np.ndarray) -> np.ndarray:
        return np.block([[eye, zero], [admittance, eye]])

    series = np.block([[eye, z_series], [zero, eye]])
    return shunt(0.5 * y_shunt) @ series @ shunt(0.5 * y_shunt)


def chain_ladder(topology: Topology, freq_hz: float) -> np.ndarray:
    """Chain matrix of ``sections`` lumped pi-sections in cascade.

    This is the circuit ``rf_netlist.py`` writes, so it -- not
    :func:`chain_distributed` -- is what ngspice must reproduce. Comparing the
    two is how the bench decides whether ``sections`` is large enough.
    """
    section = pi_section(topology, freq_hz)
    return np.linalg.matrix_power(section, topology.sections)


# ---------------------------------------------------------------------------
# chain -> impedance -> scattering
# ---------------------------------------------------------------------------


def chain_to_z(chain: np.ndarray, conductors: int) -> np.ndarray:
    """Convert a chain matrix to the ``(2N, 2N)`` port impedance matrix.

    Port order is **all conductor starts, then all conductor ends**:
    ``[c0_start, ..., cN_start, c0_end, ..., cN_end]``. Every port current
    flows *into* the network.

    Splitting ``V1 = A V2 - B I2`` and ``I1 = C V2 - D I2`` on ``V2`` gives

    .. code-block:: text

        Z11 = A C^-1        Z12 = A C^-1 D - B
        Z21 = C^-1          Z22 = C^-1 D

    For a scalar lossless line this yields ``Z0 coth(g l)`` on the diagonal and
    ``Z0 / sinh(g l)`` off it, which is the textbook Z-parameter pair.
    """
    n = conductors
    a_mat = chain[:n, :n]
    b_mat = chain[:n, n:]
    c_mat = chain[n:, :n]
    d_mat = chain[n:, n:]

    c_inv = np.linalg.inv(c_mat)
    z11 = a_mat @ c_inv
    z21 = c_inv
    z12 = a_mat @ c_inv @ d_mat - b_mat
    z22 = c_inv @ d_mat
    return np.block([[z11, z12], [z21, z22]])


def z_to_s(z_mat: np.ndarray, z0: float = 50.0) -> np.ndarray:
    """Renormalise a port impedance matrix to a real ``z0`` reference.

    ``S = (Z - z0 I)(Z + z0 I)^-1``. Both factors are polynomials in ``Z``, so
    they commute whenever ``Z`` is symmetric (any reciprocal network) and the
    two orders agree -- but the right-multiplication is written out rather than
    relied on, so a future non-reciprocal topology cannot silently invert it.

    Accepts a single ``(N, N)`` matrix or a stack of ``(F, N, N)``; the identity
    is sized off the *last* axis so a frequency sweep broadcasts instead of
    quietly using ``F`` as the port count.
    """
    eye = np.eye(z_mat.shape[-1])
    return (z_mat - z0 * eye) @ np.linalg.inv(z_mat + z0 * eye)


def scattering(
    topology: Topology,
    freq_hz: float | np.ndarray,
    *,
    z0: float = 50.0,
    discrete: bool = False,
) -> np.ndarray:
    """Scattering matrix over ``freq_hz``: ``(..., 2N, 2N)`` complex.

    ``discrete=False`` uses the exact distributed line; ``True`` uses the
    sectioned ladder, which is the one ngspice actually simulates. Passing a
    scalar frequency returns ``(2N, 2N)``; passing an array of ``F`` returns
    ``(F, 2N, 2N)``.
    """
    freqs = np.atleast_1d(np.asarray(freq_hz, dtype=np.float64))
    out = np.empty((freqs.size, 2 * topology.conductors, 2 * topology.conductors), dtype=complex)
    for i, freq in enumerate(freqs):
        chain = chain_ladder(topology, freq) if discrete else chain_distributed(topology, freq)
        out[i] = z_to_s(chain_to_z(chain, topology.conductors), z0=z0)
    if np.isscalar(freq_hz) or np.ndim(freq_hz) == 0:
        return out[0]
    return out


def mode_parameters(topology: Topology, freq_hz: float) -> dict[str, np.ndarray]:
    """Even/odd (general: modal) split of a line, for inspection and plots.

    Diagonalising ``L`` and ``C`` with the same eigenvectors is legitimate
    because both are polynomials of the same adjacency matrix. The check that
    makes it legitimate -- that ``U^T C U`` really is diagonal -- is reported
    as ``simultaneous`` rather than assumed, so a topology introduced later
    with non-commuting ``L`` and ``C`` shows up here instead of producing a
    quietly wrong answer.
    """
    l_eigval, u = np.linalg.eigh(topology.L)
    c_mode = u.T @ topology.C @ u
    off_diagonal = c_mode - np.diag(np.diag(c_mode))
    simultaneous = bool(np.max(np.abs(off_diagonal)) <= 1e-9 * np.max(np.abs(c_mode)))

    c_diag = np.diag(c_mode).real
    z_mode = np.sqrt(np.maximum(l_eigval, 0.0) / np.maximum(c_diag, 1e-30))

    w = 2.0 * np.pi * freq_hz
    delay = np.sqrt(l_eigval * c_diag)
    velocity = np.where(delay > 0, 1.0 / np.maximum(delay, 1e-30), 0.0)

    order = np.argsort(-z_mode)  # largest impedance first: even mode leads
    return {
        "mode_index": order,
        "z_mode_ohm": z_mode[order],
        "velocity_m_per_s": velocity[order],
        "phase_delay_s": (topology.length_m / np.maximum(velocity[order], 1e-30)),
        "epsilon_eff_mode": np.where(
            velocity[order] > 0, (C_LIGHT / np.maximum(velocity[order], 1e-30)) ** 2, np.inf
        ),
        "angular_freq_rad_per_s": np.array([w]),
        "simultaneous": np.array([simultaneous]),
    }


def _principal_sqrt(values: np.ndarray) -> np.ndarray:
    """Square root with ``Re >= 0`` (and ``Im >= 0`` when ``Re`` vanishes).

    Physical decaying solutions need a non-negative real part. On the branch
    cut -- a lossless line, where ``mu`` is real and negative -- the imaginary
    sign is arbitrary, so it is pinned positive to make runs reproducible.
    """
    root = np.sqrt(np.asarray(values, dtype=complex))
    scale = np.maximum(np.abs(root), 1.0)
    degenerate = np.abs(root.real) <= 1e-12 * scale
    flip = (root.real < -1e-15 * scale) | (degenerate & (root.imag < 0))
    root[flip] = -root[flip]
    return root
