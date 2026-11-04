#!/usr/bin/env python3
"""TILE-8 schematic generator (M3a).  Writes a hierarchical KiCad 9 project: tile.kicad_sch + 20ish sheets.

    hw/.venv-eda/bin/python hw/board/gen_tile.py --layer {1,2,3} [--model PATH] [--out DIR] [--tapmap PATH]

Sheets: power, line0..7 (delay line: input RC + buffer, 56-section LC ladder, terminations, 8 tap buffers),
crossbar (8 lines x 8 taps x 8 outputs x {+,-} 0402 10k), ch0..7 (summer, bias, BJT diff pair + mirror tail,
diff-amp, 74HCT4066 track-and-hold + buffer), connectors (J_IN, J_OUT, J_PS, MCP4728 x2, tile ID, monitor).
Values and parts: hw/board/params.json, hw/board/parts.csv (M1a).  R_f / R_b are recomputed with
derive_params.layer_params() for the chosen model (identical to params.json for its own model).
Net-name convention (global labels): SIG{i} line input, T{i}_{k} tap k of line i (after its buffer), SN{o}/SP{o}
summing nodes (W=+1 resistor RXP goes to SN, W=-1 resistor RXN to SP), OUT{o} tile output, BIAS{o}, VREF25, MIRB.
Everything is "designed, ERC-checked, not built, not measured".
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))
import kisch  # noqa: E402
import place  # noqa: E402
import dlnn_parts as P  # noqa: E402

N_SEC, TAP_EVERY, N_TAPS, N_OUT = 56, 8, 8, 8
DEFAULT_MODEL = REPO / "models" / "tdlnet_b_w8_aware2_fro_ternary.npz"
DEFAULT_TAPMAP = REPO / "docs" / "artifacts" / "data" / "board" / "tapmap.json"


def sx(v):
    return kisch.snap(v)


# ====================================================================================================
class Tile:
    def __init__(self, layer, model_path, outdir):
        self.layer = layer
        self.outdir = Path(outdir)
        self.params = json.load(open(HERE / "params.json"))
        self.model_path = Path(model_path)
        z = dict(np.load(self.model_path, allow_pickle=True))
        self.z = z
        self.model_sha = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
        import derive_params as dp
        self.lp = dp.layer_params(dp.DESIGN, z)[layer - 1]
        self.W = z[f"L{layer-1}_W"].astype(int)           # [O, I, K]
        self.bias = z[f"L{layer-1}_bias"].astype(float)
        assert self.W.shape[0] == N_OUT and self.W.shape[2] == N_TAPS, self.W.shape
        self.n_lines = 2 if layer == 1 else 8
        assert self.W.shape[1] == self.n_lines, (self.W.shape, self.n_lines)
        self.rf = float(self.lp["Rf_E96_ohm"])
        self.rb_model = float(self.lp["Rb_E96_ohm"])
        self.rda = [float(v) for v in self.lp["diffamp_Rf_E96_ohm"]]
        self.rin_da = float(self.lp["diffamp_Rin_ohm"])
        self.a_in = float(self.lp["A_in_v_per_unit"])
        # M3b: the bias DAC is a 3.3 V MCP4728 with its 2.048 V reference (+-1.024 V about the centre code), so R_b is lowered from
        # the derive_params value until the largest |bias| needs <= 0.8 V of DAC swing (same bias current, lower resistance).
        self.rb = P.e96_nearest(min(self.rb_model, 0.8e4 / (self.a_in * float(np.abs(self.bias).max()))))
        self.pl = place.Placer(self.outdir)
        self.prj = kisch.Project("tile", paper="A3", title=f"TILE-8 layer-{layer} variant (M3a)",
                                 comment=f"model {self.model_path.name}  R_f={self.rf:g}  R_b={self.rb:g} (derived {self.rb_model:g})  designed, not built")
        self.sheets = {}
        # sheet creation order = page order
        self.sheets["power"] = self.prj.sheet("power", "A3", "Power, reference, tail-mirror reference",
                                              desc="5 V analog / 3.3 V rails, 2.5 V reference, V_cm buffer, current-mirror reference")
        for i in range(self.n_lines):
            self.sheets[f"line{i}"] = self.prj.sheet(f"line{i}", "A2", f"Delay line {i}",
                                                     desc=f"line {i}: RC + buffer, 56-section LC ladder, 8 taps")
        self.sheets["crossbar"] = self.prj.sheet("crossbar", "A1", "Crossbar: 0402 10k footprints",
                                                 desc=f"{self.n_lines}x8x8x2 footprints; W=+1 -> RXP, W=-1 -> RXN, W=0 -> both DNP")
        for o in range(N_OUT):
            self.sheets[f"ch{o}"] = self.prj.sheet(f"ch{o}", "A3", f"Channel {o}",
                                                   desc=f"output {o}: summers, bias, BJT tanh, diff-amp, S/H")
        self.sheets["connectors"] = self.prj.sheet("connectors", "A3", "Connectors, bias DACs, ID, monitor",
                                                   desc="J_IN, J_OUT 2x13, J_PS 2x8, 2 x MCP4728, tile ID, monitor")
        # spare amp in the first input-buffer quad hosts the V_cm buffer on <=3-line tiles
        self.vref_in_uib = self.n_lines <= 3
        self.counts_extra = {}

    # ------------------------------------------------------------------------------------------------
    def build(self):
        self.build_power()
        for i in range(self.n_lines):
            self.build_line(i)
        self.build_crossbar()
        for o in range(N_OUT):
            self.build_channel(o)
        self.build_connectors()
        # park unused units of shared packages (power sheet, bottom)
        sh = self.sheets["power"]
        self.pl.park_all(sh, 40.64, 200.66, pitch=30.48, per_row=10)
        sh.text("Unused op-amp / switch / transistor units of shared packages, tied off (+ to GND, - to output)",
                40.64, 195.58, 1.27)
        self.prj.save(self.outdir)

    # ------------------------------------------------------------------------------------------------
    def build_power(self):
        sh, pl = self.sheets["power"], self.pl
        sh.text("TILE-8 power: +5VA (analog, from ribbon), +3V3 (logic reference, pass-through), 2.5 V V_cm reference, "
                "current-mirror reference for the 8 diff-pair tails", 25.4, 20.32, 1.5, True)
        # rails: flags
        sh.text("Rails enter on the ribbon (J_IN pins 17,19 = +5VA, 21 = +3V3, GND) and pass to J_OUT", 25.4, 27.94)
        for k, (net, x) in enumerate([("+5VA", 38.1), ("+3V3", 63.5), ("GND", 88.9)]):
            sh.pwr_flag(x, 45.72)
        # PWR_FLAG pins: flag symbol has a single pin at the symbol origin; attach the rail by symbol placed at the same point
        sh.power("+5VA", 38.1, 45.72)
        sh.power("+3V3", 63.5, 45.72)
        sh.power("GND", 88.9, 45.72)
        # bulk caps
        pl.put(sh, "C10u", "CB1", 38.1, 71.12, 0, nets={"1": "+5VA", "2": "GND"}, show_val=False, fsize=1.0)
        pl.put(sh, "C10u", "CB2", 53.34, 71.12, 0, nets={"1": "+5VA", "2": "GND"}, show_val=False, fsize=1.0)
        pl.put(sh, "C10u", "CB3", 76.2, 71.12, 0, nets={"1": "+3V3", "2": "GND"}, show_val=False, fsize=1.0)
        pl.put(sh, "C10u", "CB4", 101.6, 71.12, 0, nets={"1": "VREF25", "2": "GND"}, show_val=False, fsize=1.0)
        sh.text("CB1,CB2: 10u +5VA   CB3: 10u +3V3   CB4: 10u V_cm (VREF25)", 38.1, 88.9)
        # 2.5 V reference: +5VA - R - V2R5 (LM4040 cathode) ; buffer -> VREF25
        pl.res(sh, "RREF1", 2490, 152.4, 50.8, 0, "+5VA", ("L", "V2R5"), show_val=True, fsize=1.0)
        pl.put(sh, "LM4040", "UREF1", 152.4, 76.2, 90, nets={"1": ("L", "V2R5"), "2": "GND"}, sim="LM4040",
               show_val=True, fsize=1.0)
        pl.put(sh, "C100n", "CREF1", 177.8, 76.2, 0, nets={"1": ("L", "V2R5"), "2": "GND"}, show_val=False, fsize=1.0)
        if self.vref_in_uib:
            pl.opamp(sh, "UIB0", 3, 215.9, 66.04, ("L", "V2R5"), "VREF25", "VREF25")
            vref_note = "V_cm buffer = spare unit 3 of input-buffer quad UIB0"
        else:
            pl.opamp(sh, "UVR1", 1, 215.9, 66.04, ("L", "V2R5"), "VREF25", "VREF25")
            pl.opamp_power(sh, "UVR1", 254.0, 66.04)
            pl.put(sh, "C100n", "CDVR1", 279.4, 66.04, 0, nets={"1": "+5VA", "2": "GND"}, show_val=False, fsize=1.0)
            vref_note = "V_cm buffer = unit 1 of UVR"
        sh.text(f"LM4040B25 2.5 V shunt reference (1 mA via 2.49k) -> TLV9064 follower -> VREF25.  {vref_note}", 139.7,
                35.56)
        # input buffer quad power
        nq = (self.n_lines + 3) // 4
        for q in range(nq):
            x = 38.1 + q * 50.8
            pl.opamp_power(sh, f"UIB{q}", x, 124.46)
            pl.put(sh, "C100n", f"CDIB{q}", x + 20.32, 124.46, 0, nets={"1": "+5VA", "2": "GND"}, show_val=False, fsize=1.0)
        sh.text("Input-buffer quad power (the 8 line input buffers share two quads; units placed on the line sheets)",
                38.1, 106.68)
        # current-mirror reference (shared diode-connected half of UTR + R_ref) for the 8 diff-pair tails
        pl.res(sh, "RMIR1", 4320, 241.3, 124.46, 0, "+5VA", "MIRB", show_val=True, fsize=1.0)
        pl.unit(sh, "BC847BS", "UTR1", 1, 266.7, 139.7, {"1": "GND", "2": "MIRB", "6": "MIRB"}, sim="DUAL_NPN",
                show_val=False, fsize=1.0)
        sh.text("Mirror: UTR unit1 diode-connected + R_ref 4.32k -> 1.008 mA reference; tails are the 8 units UT0..UT3 on ch sheets",
                190.5, 106.68)
        # clocks pull-down
        pl.res(sh, "RPD1", 10000, 330.2, 50.8, 0, "SH_CLK", "GND", show_val=True, fsize=1.0)
        sh.text("10k pull-down on SH_CLK", 317.5, 40.64)

    # ------------------------------------------------------------------------------------------------
    def build_line(self, i):
        sh, pl = self.sheets[f"line{i}"], self.pl
        sh.text(f"Delay line {i}: 56 x (L 120 uH FHW1812IF121JST + C 120 pF C0G), Z0 = 1 kohm, T = 120 ns/section, "
                f"taps k = 0..7 at sections 0, 8, ..., 56 (T_s = 8 sections = 0.96 us)", 25.4, 20.32, 1.5, True)
        sh.text("All L{i}_s: 120 uH 5 %, 1812;  C{i}_n: 120 pF C0G 0603;  end caps 2 x 30 pF (60 pF) at nodes 0 and 56;  "
                "Rs = Rt = 1k".replace("{i}", str(i)), 25.4, 27.94)
        # ---- input chain: SIG -> Rpre -> (CP) -> buffer -> Rs -> N0
        y = 63.5
        pl.res(sh, f"RP{i}", 1000, 63.5, y, 90, f"SIG{i}", ("L", "RC"), show_val=False, fsize=1.0, ref_pos=(-2.5, -2.4))
        pl.put(sh, "C330p", f"CP{i}", 83.82, y + 3.81, 0, nets={"1": ("L", "RC"), "2": "GND"}, show_val=False, fsize=1.0,
               ref_pos=(1.6, 0))
        pl.opamp(sh, f"UIB{i // 4}", i % 4 + 1, 114.3, y, ("L", "RC"), ("L", "IB"), ("L", "IB"))
        pl.res(sh, f"RS{i}", 1000, 144.78, y, 90, ("L", "IB"), ("L", "N0"), show_val=False, fsize=1.0, ref_pos=(-2.5, -2.4))
        sh.text("RP+CP: 1k + 330 pF = 482 kHz input RC;  RS = source termination 1k", 50.8, 45.72)
        # ---- ladder
        X0, DX = 50.8, 17.78
        YA, YB = 139.7, 215.9
        node_x = {}
        rows = [(YA, 0, 28), (YB, 28, 56)]

        def hang_cap(x, yw, ref, key="C120p", n=None):
            pl.put(sh, key, ref, x, yw + 3.81, 0, nets={"1": ("W", ""), "2": "GND"}, show_val=False, fsize=0.9,
                   ref_pos=(1.4, 0.4))

        for (yw, n_a, n_b) in rows:
            for n in range(n_a, n_b + 1):
                if n != 28 or yw == YA:
                    node_x[n] = X0 + (n - n_a) * DX
        for (yw, n_a, n_b) in rows:
            # sections n_a+1 .. n_b
            for s in range(n_a + 1, n_b + 1):
                xl = X0 + (s - 1 - n_a) * DX
                xr = X0 + (s - n_a) * DX
                cx = (xl + xr) / 2
                pl.put(sh, "L120u", f"L{i}_{s}", cx, yw, 90, nets={"1": ("W", ""), "2": ("W", "")}, sim="L", show_val=False,
                       fsize=0.9, ref_pos=(-2.2, -2.8))
                sh.wire(xl, yw, cx - 3.81, yw)
                sh.wire(cx + 3.81, yw, xr, yw)
            # interior shunt capacitors and junctions
            for n in range(n_a, n_b + 1):
                x = X0 + (n - n_a) * DX
                interior = 0 < n < 56
                if interior and not (n == 28 and yw == YB):
                    hang_cap(x, yw, f"C{i}_{n}")
                if interior or n == 0:
                    pass
                # junctions where wire meets cap/stub
                if (interior and not (n == 28 and yw == YB)) or n in (0, 56):
                    sh.junction(x, yw)
        # row A start: node 0 with two 30 pF caps and the N0 label
        x0 = node_x[0]
        hang_cap(x0, YA, f"C{i}_0_1", "C30p")
        hang_cap(x0 - 6.35, YA, f"C{i}_0_2", "C30p")
        sh.wire(x0 - 10.16, YA, x0 - 6.35, YA)
        sh.wire(x0 - 6.35, YA, x0, YA)
        sh.junction(x0 - 6.35, YA)
        sh.label("N0", x0 - 10.16, YA, 180, 1.0)
        # node 28: row A end stub -> label, row B start stub <- label
        sh.wire(node_x[28] if False else X0 + 28 * DX, YA, X0 + 28 * DX + 5.08, YA)
        sh.label("N28", X0 + 28 * DX + 5.08, YA, 0, 1.0)
        sh.wire(X0 - 5.08, YB, X0, YB)
        sh.label("N28", X0 - 5.08, YB, 180, 1.0)
        # node 56: two 30 pF caps and Rt
        x56 = node_x[56]
        hang_cap(x56, YB, f"C{i}_56_1", "C30p")
        hang_cap(x56 + 6.35, YB, f"C{i}_56_2", "C30p")
        sh.wire(x56, YB, x56 + 6.35, YB)
        sh.wire(x56 + 6.35, YB, x56 + 12.7, YB)
        sh.junction(x56 + 6.35, YB)
        sh.junction(x56 + 12.7, YB)
        pl.res(sh, f"RT{i}", 1000, x56 + 12.7, YB + 3.81, 0, ("W", ""), "GND", show_val=False, fsize=0.9, ref_pos=(1.4, 0.4))
        # tap stubs + labels
        for k in range(N_TAPS):
            n = k * TAP_EVERY
            if n == 0:
                continue
            yw = YA if n <= 28 else YB
            x = node_x[n]
            sh.wire(x, yw, x, yw - 5.08)
            sh.label(f"N{n}", x, yw - 5.08, 90, 1.0)
        # ---- tap buffers (two quads)
        yb = 300.0
        yb = sx(yb)
        for half, name in enumerate(("A", "B")):
            ref = f"UT{name}{i}"
            xs = 63.5 + half * 241.3
            for j in range(4):
                k = 2 * j + half      # M3b: quad A = even taps (west ladder gutter), quad B = odd taps (east gutter)
                n = k * TAP_EVERY
                x = xs + j * 40.64
                pl.opamp(sh, ref, j + 1, x, yb, ("L", f"N{n}"), f"T{i}_{k}", f"T{i}_{k}")
                sh.text(f"tap {k} (node {n})", x - 5.08, yb - 8.89, 1.0)
            pl.opamp_power(sh, ref, xs + 4 * 40.64, yb)
            pl.put(sh, "C100n", f"CDT{name}{i}", xs + 4 * 40.64 + 20.32, yb, 0, nets={"1": "+5VA", "2": "GND"},
                   show_val=False, fsize=0.9, ref_pos=(1.4, 0))
        sh.text("Tap buffers: TLV9064 followers, outputs T{i}_k feed the crossbar (global labels)".replace("{i}", str(i)),
                63.5, 265.0)

    # ------------------------------------------------------------------------------------------------
    def build_crossbar(self):
        sh, pl = self.sheets["crossbar"], self.pl
        W = self.W
        n_in = self.n_lines
        sh.text(f"Crossbar, layer {self.layer}: weight W[o,i,k] in {{-1,0,+1}}: W=+1 populate RXP_o_i_k (tap -> SN_o), "
                f"W=-1 populate RXN_o_i_k (tap -> SP_o), W=0 both DNP.  All 10k 1 % 0402 (C25744).  "
                f"The pick-and-place file is the trained model.", 25.4, 20.32, 1.5, True)
        stats = dict(pos=0, neg=0, zero=0)
        COLW, ROWH, RB = 50.8, 48.26, 5.08
        x_base, y_base = 38.1, 45.72
        for o in range(N_OUT):
            sh.text(f"output {o}:  +  -> SN{o}   - -> SP{o}", x_base + o * COLW, y_base - 7.62, 1.27, True)
        for i in range(n_in):
            sh.text(f"line {i}", 12.7, y_base + i * ROWH + 2.54, 1.5, True)
            for o in range(N_OUT):
                for k in range(N_TAPS):
                    w = int(W[o, i, k])
                    y = sx(y_base + i * ROWH + k * RB)
                    xp = sx(x_base + o * COLW + 12.7)
                    xn = sx(xp + 25.4)
                    tap = f"T{i}_{k}"
                    # '+': tap -> SN ; '-': tap -> SP
                    pl.put(sh, "R10k_X", f"RXP_{o}_{i}_{k}", xp, y, 90, nets={"1": tap, "2": f"SN{o}"}, dnp=(w != +1),
                           show_val=False, fsize=0.8, ref_pos=(-4.0, -1.9))
                    pl.put(sh, "R10k_X", f"RXN_{o}_{i}_{k}", xn, y, 90, nets={"1": tap, "2": f"SP{o}"}, dnp=(w != -1),
                           show_val=False, fsize=0.8, ref_pos=(-4.0, -1.9))
                    stats["pos" if w == 1 else "neg" if w == -1 else "zero"] += 1
        self.xbar_stats = stats

    # ------------------------------------------------------------------------------------------------
    def build_channel(self, o):
        sh, pl = self.sheets[f"ch{o}"], self.pl
        q = o // 2
        ua = (o % 2) * 2           # base unit index in a shared quad
        rf, rb = self.rf, self.rb
        # bias path (M3b): MCP4728 on 3.3 V, internal 2.048 V ref (0..2.048 V, 0.5 mV LSB).  I(SN) = (V_dac-2.5)/R_b + 2.5/R_o:
        # R_o (from +5VA) centres the DAC range (V_dac = 1.024 V = code 2048 -> zero bias current).
        ro = P.e96_nearest(rb * 2.5 / (2.5 - 1.024))
        vdac = 2.5 + rb * self.a_in * float(self.bias[o]) / 10000.0 - rb * 2.5 / ro
        assert 0.05 < vdac < 2.0, (o, vdac)
        code = int(round(vdac / 2.048 * 4096))
        self.ro = getattr(self, "ro", {}); self.ro[o] = ro
        sh.text(f"Channel {o}: differential summer (SP/SN) + bias DAC, BJT diff pair (tanh), difference amp x0.525, "
                f"74HCT4066 track-and-hold + buffer -> OUT{o}", 25.4, 20.32, 1.5, True)
        sh.text(f"layer {self.layer}: R_f = {rf:g} ohm, R_b = {rb:g} ohm, diff-amp R_f = {self.rda[o]:g} ohm; "
                f"bias {self.bias[o]:+.3f} model units -> DAC {vdac:.3f} V (code {code}, 2.048 V ref); RO = {ro:g} ohm", 25.4, 27.94)
        # --- summer P: SP{o} -> OP{o}
        y = 63.5
        pl.opamp(sh, f"UCS{q}", ua + 1, 63.5, y, "VREF25", f"SP{o}", ("L", "OP"))
        pl.res(sh, f"RFP{o}", rf, 101.6, y - 8.89, 0, f"SP{o}", ("L", "OP"), show_val=True, fsize=1.0, ref_pos=(2.0, -1.0),
               val_pos=(2.0, 1.5))
        pl.put(sh, "C15p", f"CFP{o}", 116.84, y - 8.89, 0, nets={"1": f"SP{o}", "2": ("L", "OP")}, show_val=False,
               fsize=1.0, ref_pos=(2.0, 0))
        # --- summer N: SN{o} -> ON{o}, bias R_b
        y2 = 114.3
        pl.opamp(sh, f"UCS{q}", ua + 2, 63.5, y2, "VREF25", f"SN{o}", ("L", "ON"))
        pl.res(sh, f"RFN{o}", rf, 101.6, y2 - 8.89, 0, f"SN{o}", ("L", "ON"), show_val=True, fsize=1.0, ref_pos=(2.0, -1.0),
               val_pos=(2.0, 1.5))
        pl.put(sh, "C15p", f"CFN{o}", 116.84, y2 - 8.89, 0, nets={"1": f"SN{o}", "2": ("L", "ON")}, show_val=False,
               fsize=1.0, ref_pos=(2.0, 0))
        pl.res(sh, f"RB{o}", rb, 142.24, y2 - 8.89, 0, f"BIAS{o}", f"SN{o}", show_val=True, fsize=1.0, ref_pos=(2.0, -1.0),
               val_pos=(2.0, 1.5))
        pl.put(sh, "C100n", f"CBF{o}", 162.56, y2 - 8.89, 0, nets={"1": f"BIAS{o}", "2": "GND"}, show_val=False, fsize=1.0,
               ref_pos=(2.0, 0))
        pl.res(sh, f"RO{o}", ro, 182.88, y2 - 8.89, 0, "+5VA", f"SN{o}", show_val=True, fsize=1.0, ref_pos=(2.0, -1.0),
               val_pos=(2.0, 1.5))
        sh.text("RO: bias offset current from +5VA so the 3.3 V-powered DAC (0..2.048 V) can trim around the 2.5 V virtual ground",
                142.24, y2 + 10.16, 1.0)
        sh.text("SP/SN: crossbar summing nodes (virtual ground at 2.5 V)", 38.1, 88.9, 1.0)
        # --- diff pair BCM847BS: Q1 base = ON, Q2 base = OP, common tail
        xq1, xq2, yq = 205.74, 233.68, 88.9
        pl.unit(sh, "BCM847BS", f"QD{o}", 1, xq1, yq, {"2": ("L", "ON"), "6": ("L", "DC1"), "1": ("L", "TAIL")}, sim="DUAL_NPN",
                show_val=True, fsize=1.0)
        pl.unit(sh, "BCM847BS", f"QD{o}", 2, xq2, yq, {"5": ("L", "OP"), "3": ("L", "DC2"), "4": ("L", "TAIL")}, sim="DUAL_NPN",
                show_val=False, fsize=1.0)
        pl.res(sh, f"RLD1_{o}", 1000, xq1 + 2.54, yq - 17.78, 0, "+5VA", ("L", "DC1"), show_val=False, fsize=1.0, ref_pos=(2.0, 0))
        pl.res(sh, f"RLD2_{o}", 1000, xq2 + 2.54, yq - 17.78, 0, "+5VA", ("L", "DC2"), show_val=False, fsize=1.0, ref_pos=(2.0, 0))
        # tail: unit of a shared BC847BS package: mirror transistor
        pl.unit(sh, "BC847BS", f"UT{o // 2}", o % 2 + 1, 220.98, 139.7,
                {"6" if o % 2 == 0 else "3": ("L", "TAIL"), "2" if o % 2 == 0 else "5": "MIRB",
                 "1" if o % 2 == 0 else "4": "GND"}, sim="DUAL_NPN", show_val=False, fsize=1.0)
        sh.text("diff pair: Q1 base = ON, Q2 base = OP; tail = mirror output (MIRB from power sheet), ~1 mA", 190.5, 160.02, 1.0)
        # --- diff-amp: (DC1 - DC2) * 10.5k/20k about VREF25
        xd, yd = 300.0, 101.6
        xd = sx(xd)
        rda = self.rda[o]
        rin = self.rin_da
        pl.opamp(sh, f"UDA{q}", ua + 1, xd, yd, ("L", "DAIP"), ("L", "DAIN"), ("L", "DAO"))
        yr = 63.5
        pl.res(sh, f"RDA1_{o}", rin, 262.0, yr, 0, ("L", "DC1"), ("L", "DAIP"), show_val=True, fsize=1.0, ref_pos=(1.6, -1.0),
               val_pos=(1.6, 1.5))
        pl.res(sh, f"RDA2_{o}", rda, 281.0, yr, 0, ("L", "DAIP"), "VREF25", show_val=True, fsize=1.0, ref_pos=(1.6, -1.0),
               val_pos=(1.6, 1.5))
        pl.res(sh, f"RDA3_{o}", rin, 300.0, yr, 0, ("L", "DC2"), ("L", "DAIN"), show_val=True, fsize=1.0, ref_pos=(1.6, -1.0),
               val_pos=(1.6, 1.5))
        pl.res(sh, f"RDA4_{o}", rda, 319.0, yr, 0, ("L", "DAIN"), ("L", "DAO"), show_val=True, fsize=1.0, ref_pos=(1.6, -1.0),
               val_pos=(1.6, 1.5))
        pl.put(sh, "C8p2", f"CDA{o}", 338.0, yr, 0, nets={"1": ("L", "DAIN"), "2": ("L", "DAO")}, show_val=False, fsize=1.0,
               ref_pos=(1.6, 0))
        # --- S/H: 4066 unit (o%4) + hold cap + buffer
        sw_ref = f"USH{o // 4}"
        sw_pins = {0: ("1", "2", "13"), 1: ("4", "3", "5"), 2: ("8", "9", "6"), 3: ("11", "10", "12")}[o % 4]
        xs_, ys_ = 63.5, 215.9
        pl.unit(sh, "HCT4066", sw_ref, o % 4 + 1, xs_, ys_, {sw_pins[0]: ("L", "DAO"), sw_pins[1]: ("L", "HOLD"),
                                                         sw_pins[2]: "SH_CLK"}, sim="HCT4066", show_val=False, fsize=1.0)
        pl.put(sh, "C560p", f"CH{o}", 101.6, ys_ + 3.81, 0, nets={"1": ("L", "HOLD"), "2": "GND"}, show_val=False, fsize=1.0,
               ref_pos=(1.6, 0))
        pl.opamp(sh, f"UDA{q}", ua + 2, 139.7, ys_, ("L", "HOLD"), f"OUT{o}", f"OUT{o}")
        sh.text("track-and-hold: 74HCT4066 + 560 pF C0G, buffer -> OUT (ribbon SIG, 2.5 V +- 0.5 V PAM into the next tile)",
                38.1, 190.5, 1.0)
        # --- power units / decoupling
        if o % 2 == 0:
            pl.opamp_power(sh, f"UCS{q}", 101.6, 254.0)
            pl.put(sh, "C100n", f"CDCS{q}", 121.92, 254.0, 0, nets={"1": "+5VA", "2": "GND"}, show_val=False, fsize=1.0,
                   ref_pos=(1.6, 0))
            pl.opamp_power(sh, f"UDA{q}", 165.1, 254.0)
            pl.put(sh, "C100n", f"CDDA{q}", 185.42, 254.0, 0, nets={"1": "+5VA", "2": "GND"}, show_val=False, fsize=1.0,
                   ref_pos=(1.6, 0))
        if o % 4 == 0:
            pl.unit(sh, "HCT4066", sw_ref, 5, 228.6, 254.0, {"14": "+5VA", "7": "GND"}, sim="HCT4066", show_val=False, fsize=1.0)
            pl.put(sh, "C100n", f"CDSH{o // 4}", 254.0, 254.0, 0, nets={"1": "+5VA", "2": "GND"}, show_val=False, fsize=1.0,
                   ref_pos=(1.6, 0))
        self.dac_code = getattr(self, "dac_code", {})
        self.dac_code[o] = (vdac, code)

    # ------------------------------------------------------------------------------------------------
    def build_connectors(self):
        sh, pl = self.sheets["connectors"], self.pl
        sh.text("Connectors: J_IN / J_OUT 2x13 IDC ribbon (<= 10 cm), J_PS 2x8 partial-sum header (DNP at width 8), "
                "2 x MCP4728 bias DACs, tile-ID jumpers, monitor tap", 25.4, 20.32, 1.5, True)
        conn = self.params["connector"]["J_SIG_2x13"]

        def jnets(prefix_net):
            nets = {}
            for pin in range(1, 27):
                nm = conn[str(pin)]
                if nm.startswith("SIG"):
                    idx = int(nm[3:])
                    ok = idx < self.n_lines if prefix_net == "SIG" else True
                    nets[str(pin)] = f"{prefix_net}{idx}" if ok else None
                elif nm == "GND":
                    nets[str(pin)] = "GND"
                else:
                    nets[str(pin)] = nm
            return nets

        pl.put(sh, "J2x13", "JIN1", 63.5, 109.22, 0, nets=jnets("SIG"), sim="NOSIM", show_val=True, fsize=1.27,
               extra={"Description": "ribbon IN: SIG0..7 = previous-tile outputs (or carrier DAC I/Q on lines 0,1)"})
        # J_OUT: odd pins SIG* map to OUT{o}
        pl.put(sh, "J2x13", "JOUT1", 152.4, 109.22, 0, nets=jnets("OUT"), sim="NOSIM", show_val=True, fsize=1.27,
               extra={"Description": "ribbon OUT: OUT0..7 = this tile's held PAM outputs"})
        sh.text("J_IN", 55.88, 76.2, 1.27, True)
        sh.text("J_OUT", 144.78, 76.2, 1.27, True)
        # J_PS 2x8 (DNP at width 8): PSP/PSN tie to summing nodes SP/SN
        ps = self.params["connector"]["J_PS_2x8_stitching_only"]
        nets = {}
        for pin in range(1, 17):
            nm = ps[str(pin)]          # PSP{o} -> SP{o} ; PSN{o} -> SN{o}
            nets[str(pin)] = ("G", ("SP" if nm.startswith("PSP") else "SN") + nm[3:])
        pl.gen(sh, "Connector_Generic:Conn_02x08_Odd_Even", "JPS1", "J_PS 2x8", P.FP["PH2x8"], 241.3, 109.22, 0, 1, nets,
               lcsc="n/a (DNP)", mpn="2.54 mm 2x8 pin header", mfr="generic", dnp=True, show_val=True, fsize=1.27,
               extra={"Description": "stitching ports: PSP/PSN pins wired to summing nodes SP/SN; DNP in the width-8 build"})
        sh.text("J_PS (DNP, stitching only; pins = SP0/SN0 ... SP7/SN7 summing nodes)", 228.6, 76.2, 1.27, True)
        # monitor header + test point on line 0 tap 7
        pl.gen(sh, "Connector_Generic:Conn_01x02", "JMON1", "MON", P.FP["PH1x2"], 304.8, 101.6, 0, 1,
               {"1": "T0_7", "2": "GND"}, lcsc="n/a (not assembled)", mpn="1x2 pin header", mfr="generic", show_val=True)
        pl.gen(sh, "Connector:TestPoint", "TPMON1", "MON", P.FP["TP"], 330.2, 101.6, 0, 1, {"1": "T0_7"},
               lcsc="n/a (test pad)", mpn="", mfr="", show_val=False, extra={})
        sh.text("monitor: line 0, tap 7 (T0_7) for clock calibration", 292.1, 88.9)
        # tile-ID jumpers: closed jumper = address bit 0, open = 1 (10k pull-up to +3V3).  TID0..2 are the TCA9548A A0..A2 pins.
        for n in range(3):
            x = 304.8 + n * 33.02
            pl.gen(sh, "Jumper:SolderJumper_2_Open", f"JPID{n}", "ID", P.FP["SJ"], x, 134.62, 0, 1,
                   {"1": ("L", f"TID{n}"), "2": "GND"}, lcsc="n/a (PCB pads)", mpn="solder jumper", mfr="", show_val=False)
            pl.gen(sh, "Connector:TestPoint", f"TPID{n}", f"TID{n}", P.FP["TP"], x, 147.32, 0, 1, {"1": ("L", f"TID{n}")},
                   lcsc="n/a (test pad)", show_val=False)
            pl.res(sh, f"RID{n}", 10000, x + 12.7, 134.62, 0, "+3V3", ("L", f"TID{n}"), show_val=False, fsize=1.0)
        sh.text("tile ID = TCA9548A address: 3 solder jumpers to GND (closed = 0), 10k pull-ups to +3V3 (open = 1); address 0x70 + ID. "
                "Test pads only (all 26 ribbon pins are used)", 292.1, 124.46)
        # ---- I2C topology (fix of M3a issue 1): per-tile TCA9548A (0x70..0x77 by tile ID); the ribbon carries one shared host
        # bus SDA/SCL through every tile (J_IN -> J_OUT), each tile's two MCP4728 sit alone on mux channels 0 and 1.
        xm, ym = 330.2, 215.9
        mnets = {"22": "SCL", "23": "SDA", "3": "+3V3", "21": ("L", "TID2"), "2": ("L", "TID1"), "1": ("L", "TID0"),
                 "24": "+3V3", "12": "GND", "5": ("L", "SCL0"), "4": ("L", "SDA0"), "7": ("L", "SCL1"), "6": ("L", "SDA1")}
        for ch in range(2, 8):
            mnets[str({2: 9, 3: 11, 4: 14, 5: 16, 6: 18, 7: 20}[ch])] = None
            mnets[str({2: 8, 3: 10, 4: 13, 5: 15, 6: 17, 7: 19}[ch])] = None
        pl.put(sh, None, "UMUX1", xm, ym, 0, part=("Interface_Expansion:TCA9548APWR", "TCA9548APWR", P.FP["TCA"], "C130026",
               "TCA9548APWR", "Texas Instruments"), nets=mnets, sim="NOSIM", show_val=True, fsize=1.27)
        pl.put(sh, "C100n", "CDMUX1", xm + 30.48, ym - 20.32, 0, nets={"1": "+3V3", "2": "GND"}, show_val=False, fsize=1.0,
               ref_pos=(1.6, 0))
        for n, net in enumerate(("SDA0", "SCL0", "SDA1", "SCL1")):
            pl.res(sh, f"RPU{n+1}", 10000, xm + 30.48 + n * 12.7, ym + 20.32, 0, "+3V3", ("L", net), show_val=False, fsize=1.0)
        sh.text("TCA9548A per tile: host SDA/SCL (ribbon, pulled up on the carrier) -> ch0 -> UDAC0, ch1 -> UDAC1; ch2..7 unused. "
                "RESET tied to +3V3. Channel pull-ups RPU1..4 10k to +3V3", xm - 30.48, ym - 33.02, 1.0)
        # MCP4728 x2 on 3.3 V (fix of issue 2): internal 2.048 V reference, gain x1 (config bits VREF=1, G=0, set by firmware).
        for d in range(2):
            x = 91.44 + d * 101.6
            y = 215.9
            nets = {"1": "+3V3", "2": ("L", f"SCL{d}"), "3": ("L", f"SDA{d}"), "4": "GND", "5": None, "10": "GND"}
            for ch in range(4):
                nets[str(6 + ch)] = f"BIAS{d * 4 + ch}"
            pl.put(sh, "MCP4728", f"UDAC{d}", x, y, 0, nets=nets, sim="MCP4728", show_val=True, fsize=1.27)
            pl.put(sh, "C100n", f"CDDAC{d}", x + 30.48, y, 0, nets={"1": "+3V3", "2": "GND"}, show_val=False, fsize=1.0,
                   ref_pos=(1.6, 0))
            sh.text(f"UDAC{d}: bias outputs BIAS{d*4}..BIAS{d*4+3}; VDD = 3.3 V, internal 2.048 V reference, gain x1, centre code 2048 "
                    f"= 1.024 V; LDAC tied low; alone on mux channel {d} so the shared default address 0x60 is harmless",
                    x - 20.32, y - 25.4, 1.0)
        sh.text("Bias DAC range (fix of issue 2): see channel sheets, RO from +5VA offsets the 0..2.048 V DAC range to the summer virtual ground",
                63.5, 254.0)

    # ------------------------------------------------------------------------------------------------
    def tapmap_entries(self, tap_spacing_ns, ideal_ns):
        out = []
        W = self.W
        for o in range(N_OUT):
            for i in range(self.n_lines):
                for k in range(N_TAPS):
                    w = int(W[o, i, k])
                    out.append(dict(layer=self.layer, o=o, i=i, k=k, w=w,
                                    ref_p=f"RXP_{o}_{i}_{k}", ref_n=f"RXN_{o}_{i}_{k}",
                                    populated=(f"RXP_{o}_{i}_{k}" if w == 1 else f"RXN_{o}_{i}_{k}" if w == -1 else None),
                                    polarity=("+" if w == 1 else "-" if w == -1 else "0"),
                                    tap_net=f"T{i}_{k}", sum_net=(f"SN{o}" if w == 1 else f"SP{o}" if w == -1 else None),
                                    tap_node=k * TAP_EVERY, tap_buffer=f"UT{'A' if k % 2 == 0 else 'B'}{i}.u{k // 2 + 1}",
                                    delay_ns=round(k * tap_spacing_ns, 3), delay_ideal_ns=k * ideal_ns))
        return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--layer", type=int, choices=(1, 2, 3), default=2, help="layer variant: lines populated and R_f")
    ap.add_argument("--model", default=str(DEFAULT_MODEL), help="model .npz (crossbar population)")
    ap.add_argument("--out", default=str(HERE / "tile"), help="output project dir (default hw/board/tile)")
    ap.add_argument("--tapmap", default=str(DEFAULT_TAPMAP), help="tapmap.json to merge this layer into ('' = skip)")
    a = ap.parse_args()
    place.write_sim_lib()
    t = Tile(a.layer, a.model, a.out)
    t.build()
    ln = t.params["line"]
    spacing = ln["tap_spacing_nominal_s"] * 1e9
    ideal = ln["tap_spacing_ideal_s"] * 1e9
    if a.tapmap:
        tp = Path(a.tapmap)
        tp.parent.mkdir(parents=True, exist_ok=True)
        data = json.load(open(tp)) if tp.exists() else {}
        data.setdefault("schema", "dlnn-tapmap/1")
        data["note"] = ("maps (layer, o, i, k) to crossbar refdes, polarity and the physical delay of tap k; delay = k x "
                        "tap_spacing_nominal (params.json line.tap_spacing_nominal_s, 8 sections, exact-ladder mean at 125 kHz), "
                        "delay_ideal = k x 960 ns; designed, not measured")
        data["tap_spacing_nominal_ns"] = round(spacing, 3)
        data["tap_spacing_ideal_ns"] = ideal
        data["polarity"] = "W=+1 -> RXP populated (tap -> SN_o); W=-1 -> RXN populated (tap -> SP_o); W=0 -> both DNP"
        data.setdefault("layers", {})[str(a.layer)] = dict(
            model=str(Path(a.model).relative_to(REPO)) if Path(a.model).is_absolute() and REPO in Path(a.model).parents else str(a.model),
            model_file_sha256=t.model_sha, n_lines=t.n_lines, R_f_ohm=t.rf, R_b_ohm=t.rb,
            populated=int((t.W != 0).sum()), footprints=int(t.W.size * 2),
            entries=t.tapmap_entries(spacing, ideal))
        json.dump(data, open(tp, "w"), indent=1)
    s = t.xbar_stats
    print(f"layer {a.layer}: lines {t.n_lines}, crossbar +{s['pos']} -{s['neg']} 0:{s['zero']} populated {s['pos']+s['neg']} "
          f"of {2*t.W.size} footprints; R_f {t.rf:g} R_b {t.rb:g}; out {a.out}")


if __name__ == "__main__":
    main()
