"""Placement helpers shared by gen_tile.py / gen_carrier.py (thin layer over kisch.Sheet.add)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

LIBDIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIBDIR))
import dlnn_parts as P  # noqa: E402
import kisch  # noqa: E402

kisch.add_symdir(LIBDIR)
kisch.SYMDIRS.append(Path("/usr/share/kicad/symbols"))

G = kisch.GRID


def snap(v):
    return kisch.snap(v)


def sim_props(kind, outdir):
    """Sim.* properties; the library path is relative to the project dir ($KIPRJMOD)."""
    if kind is None:
        return {}
    rel = os.path.relpath(LIBDIR, Path(outdir).resolve())
    d = dict(P.SIM[kind])
    if "Sim.Library" in d:
        d["Sim.Library"] = "${KIPRJMOD}/" + rel + "/dlnn_sim.lib"
    return d


class Placer:
    """Binds an output dir (for Sim paths), tracks multi-unit package usage across sheets."""

    def __init__(self, outdir):
        self.outdir = Path(outdir)
        self.pkgs = {}      # ref -> dict(key, used:set, n:int)
        self.lcsc = {}      # LCSC -> set(ref) for BOM bookkeeping

    # ------------------------------------------------------------ generic
    def put(self, sh, key, ref, x, y, rot=0, unit=1, nets=None, value=None, dnp=False, part=None, sim=None,
            extra=None, **kw):
        t = part or P.PART[key]
        lib, val, fp, lcsc, mpn, mfr = t[:6]
        props = {"LCSC": lcsc, "MPN": mpn, "Manufacturer": mfr}
        props.update(sim_props(sim, self.outdir))
        if extra:
            props.update(extra)
        return sh.add(lib, ref, value or val, x, y, rot, unit, nets, fp, props, dnp=dnp, **kw)

    def gen(self, sh, lib, ref, value, fp, x, y, rot=0, unit=1, nets=None, lcsc="TBD", mpn="", mfr="", sim="NOSIM",
            dnp=False, extra=None, **kw):
        return self.put(sh, None, ref, x, y, rot, unit, nets, value=value, dnp=dnp,
                        part=(lib, value, fp, lcsc, mpn, mfr), sim=sim, extra=extra, **kw)

    # ------------------------------------------------------------ passives
    def res(self, sh, ref, ohm, x, y, rot, n1, n2, dnp=False, **kw):
        """0603 1 % resistor of any value (E96/E24); 1k and 10k have verified C-numbers."""
        part = P.res_e96(ohm)
        return self.put(sh, None, ref, x, y, rot, 1, {"1": n1, "2": n2}, part=part, dnp=dnp, **kw)

    # ------------------------------------------------------------ multi-unit packages
    def unit(self, sh, key, ref, unit, x, y, nets, rot=0, sim=None, **kw):
        pk = self.pkgs.setdefault(ref, dict(key=key, used=set()))
        pk["used"].add(unit)
        return self.put(sh, key, ref, x, y, rot, unit, nets, sim=sim, **kw)

    TLV_PINS = {1: ("3", "2", "1"), 2: ("5", "6", "7"), 3: ("10", "9", "8"), 4: ("12", "13", "14")}

    def opamp(self, sh, ref, unit, x, y, inp, inn, out, **kw):
        p, n, o = self.TLV_PINS[unit]
        kw.setdefault("show_val", False)
        kw.setdefault("fsize", 1.0)
        return self.unit(sh, "TLV9064", ref, unit, x, y, {p: inp, n: inn, o: out}, sim="TLV9064", **kw)

    def opamp_power(self, sh, ref, x, y, **kw):
        kw.setdefault("show_val", False)
        kw.setdefault("fsize", 1.0)
        return self.unit(sh, "TLV9064", ref, 5, x, y, {"4": "+5VA", "11": "GND"}, sim="TLV9064", **kw)

    def park_all(self, sh, x0, y0, pitch=30.48, per_row=8):
        """Place every unit not yet used of every package that has been touched (pins tied off safely)."""
        j = 0
        for ref, pk in sorted(self.pkgs.items()):
            key = pk["key"]
            if key not in ("TLV9064", "BC847BS", "HCT4066"):
                continue
            if key == "TLV9064":
                for u in range(1, 5):
                    if u not in pk["used"]:
                        x = x0 + (j % per_row) * pitch
                        y = y0 + (j // per_row) * 25.4
                        net = ("L", f"PK_{ref}_{u}")
                        self.opamp(sh, ref, u, x, y, "GND", net, net)
                        j += 1
                if 5 not in pk["used"]:
                    x = x0 + (j % per_row) * pitch
                    y = y0 + (j // per_row) * 25.4
                    self.opamp_power(sh, ref, x, y)
                    j += 1
            elif key == "BC847BS":
                for u in range(1, 3):
                    if u not in pk["used"]:
                        x = x0 + (j % per_row) * pitch
                        y = y0 + (j // per_row) * 25.4
                        self.unit(sh, "BC847BS", ref, u, x, y, {"1" if u == 1 else "4": None, "2" if u == 1 else "5": None,
                                                              "6" if u == 1 else "3": None}, sim="DUAL_NPN",
                                  show_val=False, fsize=1.0)
                        j += 1
            elif key == "HCT4066":
                for u in range(1, 5):
                    if u not in pk["used"]:
                        x = x0 + (j % per_row) * pitch
                        y = y0 + (j // per_row) * 25.4
                        pins = {1: ("1", "2", "13"), 2: ("4", "3", "5"), 3: ("8", "9", "6"), 4: ("11", "10", "12")}[u]
                        self.unit(sh, "HCT4066", ref, u, x, y, {pins[0]: None, pins[1]: None, pins[2]: "GND"},
                                  sim="HCT4066", show_val=False, fsize=1.0)
                        j += 1
        return j


def write_sim_lib():
    p = LIBDIR / "dlnn_sim.lib"
    p.write_text(P.sim_lib_text())
    return p
