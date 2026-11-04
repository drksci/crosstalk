"""Minimal KiCad 9 schematic emitter (hierarchical, deterministic UUIDs, no wires needed).

Why not kicad-sch-api: 0.5.6 cannot write global labels, DNP / exclude_from_sim flags, multi-sheet
instance paths or reference strings with underscores.  This module reads stock / custom
.kicad_sym libraries, flattens `extends`, computes pin positions and writes .kicad_sch text
that KiCad 9.0.x loads (ERC, netlist, SVG, PDF verified in hw/board/README_schematic.md).
"""
from __future__ import annotations

import math
import uuid
from pathlib import Path

SYMDIRS = [Path("/usr/share/kicad/symbols")]
NS = uuid.UUID("6f1c3a52-0d1e-4b57-9a5f-d1a4b0a8c001")
GRID = 1.27


# ---------------------------------------------------------------- s-expression parser / writer
class Q(str):
    """quoted string"""


def tokenize(t):
    i, n = 0, len(t)
    while i < n:
        c = t[i]
        if c in " \t\r\n":
            i += 1
        elif c in "()":
            yield c
            i += 1
        elif c == '"':
            j = i + 1
            buf = []
            while t[j] != '"':
                if t[j] == "\\":
                    j += 1
                buf.append(t[j])
                j += 1
            yield Q("".join(buf))
            i = j + 1
        else:
            j = i
            while j < n and t[j] not in ' \t\r\n()"':
                j += 1
            yield t[i:j]
            i = j


def parse(t):
    stack = [[]]
    for tok in tokenize(t):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            x = stack.pop()
            stack[-1].append(x)
        else:
            stack[-1].append(tok)
    return stack[0][0]


def esc(s):
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def ser(node, ind=0):
    if not isinstance(node, list):
        return esc(node) if isinstance(node, Q) else str(node)
    if not any(isinstance(c, list) for c in node):
        return "(" + " ".join(ser(c) for c in node) + ")"
    pad = "\t" * (ind + 1)
    head = []
    rest = []
    for c in node:
        if isinstance(c, list) or rest:
            rest.append(c)
        else:
            head.append(c)
    s = "(" + " ".join(ser(c) for c in head)
    for c in rest:
        s += "\n" + pad + ser(c, ind + 1)
    return s + "\n" + "\t" * ind + ")"


def find(node, head):
    return [c for c in node if isinstance(c, list) and c and c[0] == head]


# ---------------------------------------------------------------- symbol library
class Sym:
    def __init__(self, lib_id, tree):
        self.lib_id = lib_id
        self.tree = tree
        self.name = lib_id.split(":")[1]
        self.pins = {}  # unit -> list of dict
        self.units = set()
        for sub in find(tree, "symbol"):
            sn = str(sub[1])
            parts = sn.rsplit("_", 2)
            u = int(parts[1])
            self.units.add(u)
            for p in find(sub, "pin"):
                at = find(p, "at")[0]
                hide = any(c == "hide" or (isinstance(c, list) and c[:2] == ["hide", "yes"]) for c in p)
                self.pins.setdefault(u, []).append(dict(
                    type=p[1], x=float(at[1]), y=float(at[2]), a=float(at[3]),
                    length=float(find(p, "length")[0][1]),
                    name=str(find(p, "name")[0][1]), num=str(find(p, "number")[0][1]), hide=hide))
        self.n_units = max([u for u in self.units if u > 0] or [1])
        self.props = {str(p[1]): str(p[2]) for p in find(tree, "property")}

    def unit_pins(self, unit):
        return self.pins.get(0, []) + self.pins.get(unit, [])


_cache = {}
_libtrees = {}


def _load_lib(lib):
    if lib not in _libtrees:
        cands = [d / f"{lib}.kicad_sym" for d in SYMDIRS]
        for c in cands:
            if c.exists():
                _libtrees[lib] = parse(c.read_text())
                break
        else:
            raise FileNotFoundError(lib)
    return _libtrees[lib]


def add_symdir(d):
    SYMDIRS.insert(0, Path(d))


def _raw(lib, name):
    for s in find(_load_lib(lib), "symbol"):
        if s[1] == name:
            return s
    raise KeyError(f"{lib}:{name}")


def _flatten(lib, name):
    s = _raw(lib, name)
    ext = find(s, "extends")
    if not ext:
        return [x for x in s]
    parent = _flatten(lib, str(ext[0][1]))
    pname = str(ext[0][1])
    out = [parent[0], Q(name)]
    cprops = {str(p[1]): p for p in find(s, "property")}
    seen = set()
    for c in parent[2:]:
        if isinstance(c, list) and c and c[0] == "property":
            key = str(c[1])
            out.append(cprops.get(key, c))
            seen.add(key)
        elif isinstance(c, list) and c and c[0] == "symbol":
            c = list(c)
            c[1] = Q(str(c[1]).replace(pname + "_", name + "_", 1))
            out.append(c)
        else:
            out.append(c)
    for k, p in cprops.items():
        if k not in seen:
            out.append(p)
    return out


def get_sym(lib_id):
    if lib_id not in _cache:
        lib, name = lib_id.split(":")
        t = _flatten(lib, name)
        t[1] = Q(lib_id)
        # sub-symbol names keep short name (KiCad convention)
        _cache[lib_id] = Sym(lib_id, t)
    return _cache[lib_id]


# ---------------------------------------------------------------- geometry
def snap(v):
    return round(round(v / GRID) * GRID, 4)


def fmt(v):
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def place_pt(x, y, rot, px, py, mirror_x=False):
    """lib point -> sheet point for a symbol at (x,y) rotated rot degrees CCW."""
    if mirror_x:
        py = -py
    th = math.radians(rot)
    rx = px * math.cos(th) - py * math.sin(th)
    ry = px * math.sin(th) + py * math.cos(th)
    return (round(x + rx, 4), round(y - ry, 4))


def outward(pin, rot):
    """outward direction (screen angle 0=E,90=N,180=W,270=S) of a pin after rotation"""
    return (pin["a"] + 180 + rot) % 360


POWER_SYM = {"GND": "power:GND", "+5VA": "power:+5VA", "+3V3": "power:+3V3", "+5V": "power:+5V",
             "VDD": "power:VDD"}


class Sheet:
    def __init__(self, name, project, root_uuid, sheet_uuid, paper="A3", title="", comment=""):
        self.name, self.project = name, project
        self.root_uuid, self.sheet_uuid = root_uuid, sheet_uuid
        self.paper, self.title, self.comment = paper, title, comment
        self.uuid = str(uuid.uuid5(NS, f"{project}/{name}/file"))
        self.items = []
        self.libs = {}
        self.refs = {}   # ref -> dict(lib, value, unit set, dnp, bom, props)
        self.n = 0
        self.pwr_n = 0
        self.nc_pins = 0

    def _uid(self, kind=""):
        self.n += 1
        return str(uuid.uuid5(NS, f"{self.project}/{self.name}/{kind}/{self.n}"))

    # ----- primitives
    def wire(self, x1, y1, x2, y2):
        if (x1, y1) == (x2, y2):
            return
        self.items.append(["wire", ["pts", ["xy", fmt(x1), fmt(y1)], ["xy", fmt(x2), fmt(y2)]],
                           ["stroke", ["width", 0], ["type", "default"]], ["uuid", Q(self._uid("w"))]])

    def junction(self, x, y):
        self.items.append(["junction", ["at", fmt(x), fmt(y)], ["diameter", 0], ["color", 0, 0, 0, 0],
                           ["uuid", Q(self._uid("j"))]])

    def no_connect(self, x, y):
        self.nc_pins += 1
        self.items.append(["no_connect", ["at", fmt(x), fmt(y)], ["uuid", Q(self._uid("nc"))]])

    def text(self, s, x, y, size=1.27, bold=False):
        eff = ["effects", ["font", ["size", size, size]] + ([["bold", "yes"]] if bold else []),
               ["justify", "left", "bottom"]]
        self.items.append(["text", Q(s), ["exclude_from_sim", "no"], ["at", fmt(x), fmt(y), 0], eff,
                           ["uuid", Q(self._uid("t"))]])

    def label(self, name, x, y, angle=0, size=1.27):
        just = ["justify", "left", "bottom"] if angle in (0, 90) else ["justify", "right", "bottom"]
        self.items.append(["label", Q(name), ["at", fmt(x), fmt(y), angle],
                           ["effects", ["font", ["size", size, size]], just], ["uuid", Q(self._uid("l"))]])

    def glabel(self, name, x, y, angle=0, size=1.27):
        just = "left" if angle in (0, 90) else "right"
        self.items.append(["global_label", Q(name), ["shape", "passive"], ["at", fmt(x), fmt(y), angle],
                           ["fields_autoplaced", "yes"],
                           ["effects", ["font", ["size", size, size]], ["justify", just]],
                           ["uuid", Q(self._uid("g"))],
                           ["property", Q("Intersheetrefs"), Q("${INTERSHEET_REFS}"), ["at", fmt(x), fmt(y), 0],
                            ["effects", ["font", ["size", 1.27, 1.27]], ["hide", "yes"]]]])

    def power(self, kind, x, y, rot=0):
        self.pwr_n += 1
        lib_id = POWER_SYM[kind]
        self._use(lib_id)
        ref = f"#PWR_{self.name}_{self.pwr_n}"
        self.items.append(["symbol", ["lib_id", Q(lib_id)], ["at", fmt(x), fmt(y), rot], ["unit", 1],
                           ["exclude_from_sim", "no"], ["in_bom", "no"], ["on_board", "no"], ["dnp", "no"],
                           ["uuid", Q(self._uid("p"))],
                           self._prop("Reference", ref, x, y + 3, hide=True),
                           self._prop("Value", kind, x, y + 2, hide=True),
                           self._prop("Footprint", "", x, y, hide=True),
                           self._prop("Datasheet", "", x, y, hide=True),
                           ["pin", Q("1"), ["uuid", Q(self._uid("pp"))]],
                           self._inst(ref, 1)])

    def pwr_flag(self, x, y, rot=0):
        self.pwr_n += 1
        lib_id = "power:PWR_FLAG"
        self._use(lib_id)
        ref = f"#FLG_{self.name}_{self.pwr_n}"
        self.items.append(["symbol", ["lib_id", Q(lib_id)], ["at", fmt(x), fmt(y), rot], ["unit", 1],
                           ["exclude_from_sim", "no"], ["in_bom", "no"], ["on_board", "no"], ["dnp", "no"],
                           ["uuid", Q(self._uid("p"))],
                           self._prop("Reference", ref, x, y + 3, hide=True),
                           self._prop("Value", "PWR_FLAG", x, y + 2, hide=True),
                           self._prop("Footprint", "", x, y, hide=True),
                           self._prop("Datasheet", "", x, y, hide=True),
                           ["pin", Q("1"), ["uuid", Q(self._uid("pp"))]],
                           self._inst(ref, 1)])

    def _use(self, lib_id):
        if lib_id not in self.libs:
            self.libs[lib_id] = get_sym(lib_id)

    def _prop(self, k, v, x, y, hide=False, size=1.27, rot=0, justify=None):
        eff = ["effects", ["font", ["size", size, size]]]
        if justify:
            eff.append(["justify", justify])
        if hide:
            eff.append(["hide", "yes"])
        return ["property", Q(k), Q(v), ["at", fmt(x), fmt(y), rot], eff]

    def _inst(self, ref, unit):
        return ["instances", ["project", Q(self.project),
                              ["path", Q(f"/{self.root_uuid}/{self.sheet_uuid}" if self.sheet_uuid else f"/{self.root_uuid}"),
                               ["reference", Q(ref)], ["unit", unit]]]]

    # ----- net attachment to a pin end
    def attach(self, net, x, y, out_dir, size=1.27):
        if net is None:
            self.no_connect(x, y)
            return
        if isinstance(net, tuple):      # ("L", name) local label; ("G", name) global; ("P", name) power
            kind, name = net
        else:
            kind, name = ("P", net) if net in POWER_SYM else ("G", net)
        if kind == "P":
            if out_dir == 270 and name == "GND":
                self.power("GND", x, y, 0)
                return
            if out_dir == 90 and name != "GND":
                self.power(name, x, y, 0)
                return
            kind = "G"
        if kind == "W":     # explicit wire-less no-op marker (net made by overlap)
            return
        a = out_dir
        if kind == "L":
            self.label(name, x, y, a, size)
        else:
            self.glabel(name, x, y, a, size)

    # ----- components
    def add(self, lib_id, ref, value, x, y, rot=0, unit=1, nets=None, fp=None, props=None,
            dnp=False, bom=True, show_ref=True, show_val=True, fsize=1.27, nc_unlisted=False,
            pin_nets_ok_missing=False, ref_pos=None, val_pos=None):
        """Place one unit of a symbol.  nets: {pin number or name: net}.  Unlisted pins raise unless
        nc_unlisted (then no-connect flags)."""
        sym = get_sym(lib_id)
        self._use(lib_id)
        x, y = snap(x), snap(y)
        nets = dict(nets or {})
        props = dict(props or {})
        pins = sym.unit_pins(unit)
        seen = set()
        pin_uuids = []
        for p in pins:
            if p["num"] in seen:
                continue
            seen.add(p["num"])
            pin_uuids.append(p["num"])
            key = p["num"] if p["num"] in nets else (p["name"] if p["name"] in nets else None)
            if (p["hide"] or p["type"] == "no_connect") and key is None:
                continue
            px, py = place_pt(x, y, rot, p["x"], p["y"])
            if key is None:
                if nc_unlisted:
                    self.no_connect(px, py)
                    continue
                raise ValueError(f"{self.name}:{ref} unit {unit} pin {p['num']} ({p['name']}) has no net")
            self.attach(nets.pop(key), px, py, outward(p, rot), fsize)
        if nets:
            raise ValueError(f"{self.name}:{ref} nets for unknown pins {list(nets)}")
        # properties
        base = sym.props
        ox, oy = 2.54, -(sym.props and 0) - 2.54
        lib_fp = base.get("Footprint", "")
        items = ["symbol", ["lib_id", Q(lib_id)], ["at", fmt(x), fmt(y), rot], ["unit", unit],
                 ["exclude_from_sim", "yes" if dnp else "no"], ["in_bom", "yes" if bom and not dnp else "no"],
                 ["on_board", "yes"], ["dnp", "yes" if dnp else "no"], ["fields_autoplaced", "yes"],
                 ["uuid", Q(self._uid("s"))]]
        rp = ref_pos or (2.54, -3.81)
        vp = val_pos or (2.54, 3.81)
        items.append(self._prop("Reference", ref, x + rp[0], y + rp[1], hide=not show_ref, size=fsize, justify="left"))
        items.append(self._prop("Value", value, x + vp[0], y + vp[1], hide=not show_val, size=fsize, justify="left"))
        items.append(self._prop("Footprint", fp if fp is not None else lib_fp, x, y, hide=True))
        items.append(self._prop("Datasheet", props.pop("Datasheet", "~"), x, y, hide=True))
        for k, v in props.items():
            items.append(self._prop(k, v, x, y, hide=True))
        for pn in pin_uuids:
            items.append(["pin", Q(pn), ["uuid", Q(self._uid("sp"))]])
        items.append(self._inst(ref, unit))
        self.items.append(items)
        r = self.refs.setdefault(ref, dict(lib_id=lib_id, value=value, units=set(), dnp=dnp, bom=bom, fp=fp))
        r["units"].add(unit)
        return {p["num"]: place_pt(x, y, rot, p["x"], p["y"]) for p in pins}

    def pin_pos(self, lib_id, x, y, rot, num, unit=1):
        sym = get_sym(lib_id)
        for p in sym.unit_pins(unit):
            if p["num"] == num:
                return place_pt(snap(x), snap(y), rot, p["x"], p["y"])
        raise KeyError(num)

    # ----- output
    def to_text(self, sheets=None):
        body = []
        for lib_id, s in sorted(self.libs.items()):
            body.append(s.tree)
        out = ["kicad_sch", ["version", 20250114], ["generator", Q("eeschema")], ["generator_version", Q("9.0")],
               ["uuid", Q(self.uuid)], ["paper", Q(self.paper)],
               ["title_block", ["title", Q(self.title or self.name)], ["date", Q("2026-11-03")], ["rev", Q("M3a")],
                ["company", Q("dlnn phase 2: designed, ERC-checked, not built, not measured")]]
               + ([["comment", 1, Q(self.comment)]] if self.comment else []),
               ["lib_symbols"] + body]
        out += self.items
        if sheets:
            out += sheets
        out.append(["sheet_instances", ["path", Q("/"), ["page", Q("1")]]])
        out.append(["embedded_fonts", "no"])
        return ser(out) + "\n"

    def save(self, path):
        Path(path).write_text(self.to_text())


class Project:
    """root sheet + child sheets (one file per child, each used once)."""

    def __init__(self, name, paper="A3", title="", comment=""):
        self.name = name
        self.root_uuid = str(uuid.uuid5(NS, f"{name}/root"))
        self.root = Sheet("root", name, self.root_uuid, None, paper, title, comment)
        self.root.uuid = self.root_uuid
        self.children = []   # (Sheet, filename, desc)

    def sheet(self, name, paper="A3", title="", desc="", comment=""):
        suuid = str(uuid.uuid5(NS, f"{self.name}/sheet/{name}"))
        sh = Sheet(name, self.name, self.root_uuid, suuid, paper, title or name, comment)
        self.children.append((sh, f"{name}.kicad_sch", desc))
        return sh

    def save(self, outdir, libdir=None):
        import os
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        blocks = []
        x0, y0, w, h, dx, dy, cols = 25.4, 40.64, 50.8, 12.7, 76.2, 25.4, 4
        for i, (sh, fn, desc) in enumerate(self.children):
            sh.save(outdir / fn)
            x = x0 + (i % cols) * dx
            y = y0 + (i // cols) * dy
            blocks.append(["sheet", ["at", fmt(x), fmt(y)], ["size", fmt(w), fmt(h)], ["exclude_from_sim", "no"],
                           ["in_bom", "yes"], ["on_board", "yes"], ["dnp", "no"], ["fields_autoplaced", "yes"],
                           ["stroke", ["width", 0.1524], ["type", "solid"]], ["fill", ["color", 0, 0, 0, 0.0]],
                           ["uuid", Q(sh.sheet_uuid)],
                           ["property", Q("Sheetname"), Q(sh.name), ["at", fmt(x), fmt(y - 0.7), 0],
                            ["effects", ["font", ["size", 1.27, 1.27]], ["justify", "left", "bottom"]]],
                           ["property", Q("Sheetfile"), Q(fn), ["at", fmt(x), fmt(y + h + 0.7), 0],
                            ["effects", ["font", ["size", 1.27, 1.27]], ["justify", "left", "top"]]],
                           ["instances", ["project", Q(self.name), ["path", Q(f"/{self.root_uuid}"), ["page", Q(str(i + 2))]]]]])
            if desc:
                self.root.text(desc, x, y + h + 4.5, 1.0)
        (outdir / f"{self.name}.kicad_sch").write_text(self.root.to_text(blocks))
        (outdir / f"{self.name}.kicad_pro").write_text('{\n  "meta": {"filename": "%s.kicad_pro", "version": 1}\n}\n' % self.name)
        libdir = Path(libdir) if libdir else Path(__file__).resolve().parent
        rel = os.path.relpath(libdir.resolve(), outdir.resolve())
        (outdir / "sym-lib-table").write_text(
            '(sym_lib_table\n  (version 7)\n  (lib (name "dlnn")(type "KiCad")(uri "${KIPRJMOD}/%s/dlnn.kicad_sym")(options "")(descr "dlnn custom symbols"))\n)\n' % rel)
