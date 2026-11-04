"""Hand-written netlist builders + ngspice runner shared by the M1a checks (no KiCad yet).

Everything here is a LUMPED model at 1 MHz scale: not RF.  Inductor model (datasheet-derived):
    n_in --[DCR]--[L]-- n_out, with Cp and Rp across the whole inductor (n_in..n_out).
"""
import math, os, subprocess, tempfile, shutil
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORK_PLAIN = os.path.join(HERE, "_work", "plain")
WORK_PSA = os.path.join(HERE, "_work", "psa")      # contains .spiceinit 'set ngbehavior=psa' for PSpice-dialect TI macro
for d, init in ((WORK_PLAIN, None), (WORK_PSA, "set ngbehavior=psa\n")):
    os.makedirs(d, exist_ok=True)
    if init:
        with open(os.path.join(d, ".spiceinit"), "w") as f:
            f.write(init)


def inductor_rp(L, dcr, Q, f_q):
    """R_p (parallel, frequency independent) that reproduces datasheet Q at f_q given series DCR.
    1/Q = DCR/(wL) + wL/Rp  ->  Rp = wL / (1/Q - DCR/(wL)).   (parallel-RLC Q definition; standard
    inductor equivalent circuit, e.g. Coilcraft 'Modeling' note / TDK 'Inductor model' app note)."""
    w = 2 * math.pi * f_q
    x = 1.0 / Q - dcr / (w * L)
    if x <= 0:
        return 1e12
    return w * L / x


def inductor_cp(L, srf):
    """C_p from SRF: f_srf = 1/(2 pi sqrt(L Cp))."""
    return 1.0 / ((2 * math.pi * srf) ** 2 * L)


def line_netlist(L=120e-6, C=120e-12, n=56, tap_every=8, dcr=3.3, srf=3.3e6, Q=10.0, f_q=796e3,
                 rs=1e3, rl=1e3, prefilter=True, f_rc=1.5e6, rc_r=1e3, c_tap=6e-12, cp_override=None,
                 rp_override=None, cend=60e-12, Lvals=None, Cvals=None, end_cap_extra=0.0, src="AC 1", extra=""):
    """Return (netlist_body, taps) for the full ladder.  Node names: n0..n{n}; taps tNk = n{k*tap_every}.
    Lvals/Cvals: optional per-section arrays (Monte Carlo)."""
    cp = inductor_cp(L, srf) if cp_override is None else cp_override
    rp = inductor_rp(L, dcr, Q, f_q) if rp_override is None else rp_override
    lines = [f"* LC ladder: {n} sections L={L:.4g} C={C:.4g} DCR={dcr} Cp={cp:.4g} Rp={rp:.4g} Q={Q}@{f_q:.4g}"]
    lines.append(f"Vsrc vs 0 {src}")
    if prefilter:
        cf = 1.0 / (2 * math.pi * f_rc * rc_r)
        lines += [f"Rpre vs vf {rc_r}", f"Cpre vf 0 {cf:.5g}", "Ebuf vb 0 vf 0 1", f"Rs vb n0 {rs}"]
    else:
        lines += [f"Rs vs n0 {rs}"]
    lines.append(f"C0 n0 0 {(cend if Cvals is None else Cvals[0]):.6g}")
    for k in range(1, n + 1):
        Lk = L if Lvals is None else Lvals[k - 1]
        Ck = (cend if k == n else C) if Cvals is None else Cvals[k]
        lines += [f"Rd{k} n{k-1} m{k} {dcr}", f"L{k} m{k} n{k} {Lk:.6g}",
                  f"Cp{k} n{k-1} n{k} {cp:.5g}", f"Rp{k} n{k-1} n{k} {rp:.5g}", f"C{k} n{k} 0 {Ck:.6g}"]
    lines.append(f"Rl n{n} 0 {rl}")
    taps = []
    for t in range(0, n // tap_every + 1):
        node = f"n{t*tap_every}"
        taps.append(node)
        if c_tap:
            lines.append(f"Ctap{t} {node} 0 {c_tap}")
    if end_cap_extra:
        lines.append(f"Cx n{n} 0 {end_cap_extra}")
    lines.append(extra)
    return "\n".join(lines), taps


def run_ngspice(deck, outfile_names, psa=False, timeout=600, tag="run"):
    """Run a deck (string) in batch mode in the plain or psa work dir; returns dict name->np.array of the
    wrdata files the deck writes (deck must write them with 'wrdata <name> ...' in a .control block)."""
    wd = WORK_PSA if psa else WORK_PLAIN
    path = os.path.join(wd, tag + ".cir")
    with open(path, "w") as f:
        f.write(deck)
    for nm in outfile_names:
        p = os.path.join(wd, nm)
        if os.path.exists(p):
            os.remove(p)
    r = subprocess.run(["ngspice", "-b", path], cwd=wd, capture_output=True, text=True, timeout=timeout)
    out = {}
    for nm in outfile_names:
        p = os.path.join(wd, nm)
        if not os.path.exists(p):
            raise RuntimeError(f"ngspice produced no {nm}\n" + r.stdout[-2000:] + r.stderr[-2000:])
        out[nm] = np.loadtxt(p)
    return out, r.stdout
