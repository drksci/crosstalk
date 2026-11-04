#!/usr/bin/env python3
"""cost_model.py : BOM + PCB + assembly ESTIMATE for TILE-8 -> hw/board/cost.json  (and hw/board/parts.csv)

ESTIMATE, NEEDS A LIVE QUOTE.  Nothing here is a quote.  Unit prices are the JLCPCB part-page price tiers read on 2026-11-03
(USD, one reading; they change daily and stock is not checked); fees are from https://jlcpcb.com/help/article/pcb-assembly-price
(read 2026-11-03); the bare-PCB price is an ASSUMPTION built from an older JLCPCB blog rate; the Nucleo price is an ASSUMPTION (ST site blocked).

Usage:  python hw/board/cost_model.py [--params hw/board/params.json] [--out hw/board/cost.json] [--parts hw/board/parts.csv]
        python hw/board/cost_model.py --refresh-prices   (re-reads the JLC part pages, needs the network; rewrites the PRICE tiers in memory only)

Scaling formula (width W, depth D, K = 8 taps, S = 56 sections, tile = 8 lines x 8 outputs):
    layer 1 has 2 input lines -> W/8 tiles with only 2 lines populated (112 inductors each)
    layers 2..D: (W/8)^2 full tiles (448 inductors each), because the line of an input must be replicated on every tile that uses it
    inductors = 448 (D-1) (W/8)^2 + 112 W/8        area = tiles * 160 x 100 mm        (W=8, D=3 -> 3 tiles, 1008 inductors)
"""
import argparse
import csv
import json
import math
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------- fees (JLCPCB, read 2026-11-03)
FEES = {
    "source": "https://jlcpcb.com/help/article/pcb-assembly-price (read 2026-11-03)",
    "economic": {"setup": 8.18, "stencil": 1.53, "per_joint": 0.0016, "feeder_per_unique_extended": 3.07, "sides": 1},
    "standard_double_sided": {"setup": 51.12, "stencil": 16.42, "per_joint": 0.0016, "feeder_per_unique_part": 1.53, "sides": 2, "handling_min_diff": 14.93},
    "hand_soldering_per_order": 3.58, "manual_per_joint": 0.0164, "confirm_parts_placement": 0.45,
}
PCB = {
    "note": "ASSUMPTION: area rate 70.6 USD/m^2 for small-batch 4-layer (older JLCPCB blog 'Special Discount on Quality 4 Layers PCBs') + 40 USD fixed engineering/flying-probe fee; "
            "2-layer carrier: 25 USD/m^2... all of this is to be replaced by a live quote (impedance-control surcharge not included).",
    "usd_per_m2_4layer": 70.6, "fixed_4layer_usd": 40.0, "usd_per_m2_2layer": 25.0, "fixed_2layer_usd": 5.0, "min_boards": 5,
    "tile_size_mm": (160.0, 100.0), "carrier_size_mm": (100.0, 80.0),
}
OTHER = {"nucleo_g474re_usd": 18.0, "nucleo_note": "ASSUMPTION (ST price page blocked); ST list price is of this order", "ribbon_cables_usd": 3 * 2.0,
         "dc_supply_usd": 10.0, "usb_cable_usd": 0.0, "enclosure_usd": 0.0}

# ----------------------------------------------------------------------------- BOM.  qty: per full tile / per layer-1 tile / per carrier
# price tiers: [(min_qty, usd)] from the JLC part page (2026-11-03).  status: 'verified-JLC-page' = MPN, package, params and price read on the
# jlcpcb.com part page this session; 'datasheet-checked' = the manufacturer datasheet was also read.
def bom(n_lines_full, pop_by_tile, n_sections=56):
    """Return list of BOM items.  pop_by_tile = [populated 0402 count for tile 1, 2, 3...]."""
    S = n_sections
    items = []

    def add(**kw):
        items.append(kw)
    # qty functions get (lines, populated) -> number per tile
    add(key="L", function="delay-line inductor", value="120 uH +-5 %", package="1812", mpn="FHW1812IF121JST", mfr="FH (Guangdong Fenghua)", c="C394134", lib="Extended",
        tiers=[(1, .1406), (50, .113), (150, .0993), (500, .0889), (2000, .0717)], q=lambda n, p: n * S, status="verified-JLC-page",
        ds="https://jlcpcb.com/partdetail/C394134", params="L 120uH +-5%; DCR 3.3 ohm; SRF 3.3 MHz; Q>=10 @796 kHz; Irated 220 mA; unshielded wirewound ferrite, 4.5x3.2 mm. "
        "MISSES spec: SRF<5 MHz, not shielded (see board_design.md 3.1)")
    add(key="C120", function="delay-line shunt capacitor", value="120 pF C0G +-5 %", package="0603", mpn="GCM1885C1H121JA16D", mfr="Murata", c="C126547", lib="Extended",
        tiers=[(1, .023), (200, .0181), (600, .0153), (4000, .0136)], q=lambda n, p: n * (S - 1), status="verified-JLC-page",
        ds="https://jlcpcb.com/partdetail/C126547", params="120pF 50V C0G +-5% 0603 (alt KEMET C0603X121J5GACTU C2178446)")
    add(key="C30", function="ladder end half-capacitors (2 x 30 pF = 60 pF per end)", value="30 pF C0G +-5 %", package="0603", mpn="0603CG300J500NT", mfr="FH", c="C1658", lib="Basic",
        tiers=[(1, .0051), (1000, .0041), (4000, .0033)], q=lambda n, p: n * 4, status="verified-JLC-page", ds="https://jlcpcb.com/partdetail/C1658", params="30pF 50V C0G +-5% 0603")
    add(key="C330", function="input RC capacitor (482 kHz with 1k)", value="330 pF C0G +-5 %", package="0603", mpn="GRM1885C1H331JA01D", mfr="Murata", c="C85978", lib="Extended",
        tiers=[(1, .0093), (1000, .0076), (4000, .0067)], q=lambda n, p: n, status="verified-JLC-page", ds="https://jlcpcb.com/partdetail/C85978", params="330pF 50V C0G +-5% 0603")
    add(key="R10k", function="crossbar weight resistor (populated = nonzero weight)", value="10 kohm 1 %", package="0402", mpn="0402WGF1002TCE", mfr="UNI-ROYAL", c="C25744", lib="Basic",
        tiers=[(1, .0028), (1000, .0021), (3000, .0018), (10000, .0016)], q=lambda n, p: p, status="verified-JLC-page", ds="https://jlcpcb.com/partdetail/C25744",
        params="10k +-1% 100ppm 62.5mW; count = nonzero weights of the model (CPL = model)")
    add(key="TLV9064", function="tap buffers, input buffers, summers, diff-amps, hold buffers, Vref buffer", value="quad op-amp", package="TSSOP-14", mpn="TLV9064IPWR", mfr="Texas Instruments", c="C779410",
        lib="Extended", tiers=[(1, .4058), (50, .303), (150, .2588), (500, .1908), (2000, .1663)],
        q=lambda n, p: math.ceil((9 * n + 34) / 4.0), status="datasheet-checked",
        ds="https://www.ti.com/lit/ds/symlink/tlv9064.pdf", params="GBW 10 MHz; en 16 nV/rtHz @1k, 10 nV @10k; Iq 538 uA/ch; Cic 4 pF, Cid 2 pF; Ib 0.5 pA; SR 6.5 V/us; RRIO 1.8-5.5 V. "
        "PSpice macro TLV9062 (same die) used in ngspice")
    add(key="BCM847BS", function="diff-pair (tanh)", value="matched dual NPN", package="SOT-363", mpn="BCM847BS,135", mfr="Nexperia", c="C549556", lib="Extended",
        tiers=[(1, .3185), (50, .2511), (150, .2223), (500, .1862), (2500, .1702)], q=lambda n, p: 8, status="datasheet-checked (Diodes BCM847BS DS41216 read; Nexperia blocked)",
        ds="https://www.diodes.com/assets/Datasheets/BCM847BS.pdf", params="|dVBE|<=2 mV @2 mA; hFE 200-450, hFE1/hFE2 0.9-1.1; VBE(on) 610-710 mV; CCBO 2 pF; fT>=100 MHz (listing: 250 MHz)")
    add(key="BC847BS", function="tail current mirror (2 tails per package) + diode reference", value="dual NPN", package="SOT-363", mpn="BC847BS-7-F", mfr="Diodes Inc.", c="C156292", lib="Extended",
        tiers=[(1, .0732), (100, .0586), (300, .0512), (3000, .0442)], q=lambda n, p: 5, status="datasheet-checked", ds="https://www.diodes.com/assets/Datasheets/BC847BS.pdf",
        params="hFE 200-450 @2 mA; VBE(on) 665 mV typ; CCBO 2 pF; CEBO 11 pF; fT>=100 MHz; (alt BC847BS,115 Nexperia C8653 Extended $0.0942)")
    add(key="MCP4728", function="bias / V_BE-trim DAC (4 ch each, VDD ref)", value="12-bit 4-ch I2C DAC", package="MSOP-10", mpn="MCP4728-E/UN", mfr="Microchip", c="C108207", lib="Extended",
        tiers=[(1, 2.3305), (10, 2.0501), (30, 1.8533), (100, 1.5745)], q=lambda n, p: 2, status="datasheet-checked", ds="https://ww1.microchip.com/downloads/en/DeviceDoc/22187E.pdf",
        params="12 bit; I_DD 800 uA typ; settling 6 us; I2C address bits EEPROM-programmable (2 per tile need distinct addresses: one-time re-address, risk R6)")
    add(key="4066", function="track-and-hold switches (4 per package)", value="quad analog switch", package="SOIC-14", mpn="74HCT4066D,118", mfr="Nexperia", c="C5998", lib="Extended",
        tiers=[(1, .3499), (50, .2685), (150, .2336), (500, .1901), (2500, .1708)], q=lambda n, p: 2, status="JLC-page; R_on from TI CD74HC4066 datasheet (HCT assumed equal)",
        ds="https://www.ti.com/lit/ds/symlink/cd74hc4066.pdf", params="R_on 35 typ / 95 max ohm @25C, 118 @85C (TI HC4066, VCC 4.5 V); JLC lists 54 ohm for the Nexperia HCT part; HCT = 3.3 V logic compatible")
    add(key="C560", function="hold capacitor", value="560 pF C0G +-5 %", package="0603", mpn="CL10C561JB81PNC", mfr="Samsung", c="C472798", lib="Extended",
        tiers=[(1, .0104), (500, .0081)], q=lambda n, p: 8, status="verified-JLC-page", ds="https://jlcpcb.com/partdetail/C472798", params="560pF 50V C0G +-5% 0603")
    add(key="C15", function="summer feedback capacitor", value="15 pF C0G", package="0402", mpn="0402CG150J500NT", mfr="FH", c="C1548", lib="Basic",
        tiers=[(1, .0026), (1000, .0019)], q=lambda n, p: 16, status="verified-JLC-page", ds="https://jlcpcb.com/partdetail/C1548", params="15pF 50V C0G +-5% 0402")
    add(key="C8p2", function="diff-amp feedback capacitor", value="8.2 pF C0G", package="0402", mpn="GCM1555C1H8R2CA16D", mfr="Murata", c="C393675", lib="Extended",
        tiers=[(1, .009), (500, .007)], q=lambda n, p: 8, status="verified-JLC-page (tolerance not shown on page)", ds="https://jlcpcb.com/partdetail/C393675", params="8.2pF 50V C0G 0402")
    add(key="C100n", function="decoupling", value="100 nF X7R", package="0402", mpn="CL05B104KO5NNNC", mfr="Samsung", c="C1525", lib="Basic",
        tiers=[(1, .0031), (1000, .0023)], q=lambda n, p: 40, status="verified-JLC-page", ds="https://jlcpcb.com/partdetail/C1525", params="100nF 16V X7R +-10% 0402")
    add(key="C10u", function="bulk", value="10 uF X5R", package="0805", mpn="CL21A106KAYNNNE", mfr="Samsung", c="C15850", lib="Basic",
        tiers=[(1, .0422), (200, .0338)], q=lambda n, p: 4, status="verified-JLC-page", ds="https://jlcpcb.com/partdetail/C15850", params="10uF 25V X5R +-10% 0805")
    add(key="R_misc_1k", function="Rs, Rl, Rpre (3 per line), R_L (16), others at 1 kohm", value="1 kohm 1 %", package="0603", mpn="0603WAF1001T5E", mfr="UNI-ROYAL", c="C21190", lib="Basic",
        tiers=[(1, .0017), (1000, .0014)], q=lambda n, p: 3 * n + 16, status="verified-JLC-page", ds="https://jlcpcb.com/partdetail/C21190", params="1k +-1% 100mW 0603")
    add(key="R_misc_E96", function="R_f (16), R_b (8), diff-amp 4 x 8, R_ref, bias, V_ref: 1 % E96 values (many distinct = many extended parts)", value="E96 1 %", package="0603", mpn="0603WAF####T5E",
        mfr="UNI-ROYAL", c="(per value; 1k C21190, 10k C25804, 330R C23138, 30k C22984 verified)", lib="Basic or Extended per value", tiers=[(1, .0021), (1000, .0016)], q=lambda n, p: 16 + 8 + 32 + 4,
        status="partly verified", ds="https://jlcpcb.com/partdetail/C25804", params="unique values per tile ~ 12 -> counted in the unique-part fee")
    add(key="J_SIG", function="inter-tile 2x13 ribbon header in + out (through-hole)", value="2x13 0.1 in IDC header", package="THT", mpn="2.54-2*13P Straight Header", mfr="BOOMELE", c="C9040", lib="Extended",
        tiers=[(1, .4445), (10, .3494), (30, .31), (100, .2592)], q=lambda n, p: 2, status="verified-JLC-page", ds="https://jlcpcb.com/partdetail/C9040", params="2x13P 2.54 mm, 3 A, gold, IDC box header; hand/wave solder")
    add(key="LM4040", function="2.5 V reference (Vref buffer input)", value="2.5 V shunt +-0.2 %", package="SOT-23", mpn="LM4040B25FTA", mfr="Diodes Inc.", c="C460725", lib="Extended",
        tiers=[(1, 1.6417), (10, 1.3613), (30, 1.2087), (100, 1.0349)], q=lambda n, p: 1, status="verified-JLC-page", ds="https://jlcpcb.com/partdetail/C460725", params="2.5V +-0.2%, 100 ppm/K, 35 uVrms noise")
    return items


def carrier_bom():
    return [
        {"key": "LDO5", "function": "+5VA low-noise LDO", "mpn": "MP20051DQ-LF-Z", "c": "C134275", "lib": "Extended", "qty": 1, "usd": 1.14, "value": "5.0 V 1 A, 13 uVrms, 140 mV dropout",
         "status": "verified-JLC-page; input must be >= 5.5 V so supply ~6 V (fallback AMS1117-5.0 C6187 Basic needs >= 6.3 V, 100 uVrms)"},
        {"key": "I2CMUX", "function": "per-tile I2C mux (identical tiles)", "mpn": "TCA9548APWR", "c": "C130026", "lib": "Extended", "qty": 1, "usd": 0.8233, "value": "1:8 I2C switch", "status": "verified-JLC-page"},
        {"key": "OPAMP", "function": "DAC I/Q conditioners, ADC filters, ref buffer", "mpn": "TLV9064IPWR", "c": "C779410", "lib": "Extended", "qty": 2, "usd": 0.4058, "value": "quad", "status": "datasheet-checked"},
        {"key": "J_RIBBON", "function": "tile ribbon headers (3 tiles)", "mpn": "2.54-2*13P Straight Header", "c": "C9040", "lib": "Extended", "qty": 4, "usd": 0.4445, "value": "2x13", "status": "verified-JLC-page"},
        {"key": "J_NUCLEO", "function": "ST morpho female headers (2 x 2x19)", "mpn": "(generic 2.54 mm 19x2 female header x2)", "c": "not verified", "lib": "?", "qty": 2, "usd": 1.2, "value": "", "status": "ASSUMPTION price"},
        {"key": "PASSIVES", "function": "carrier resistors/caps (about 60 parts)", "mpn": "0402/0603 basic", "c": "various", "lib": "Basic", "qty": 60, "usd": 0.004, "value": "", "status": "ASSUMPTION"},
    ]


def price_at(tiers, qty):
    p = tiers[0][1]
    for mq, usd in tiers:
        if qty >= mq:
            p = usd
    return p


def pcb_cost(w_mm, h_mm, layers, n_boards):
    area = w_mm * h_mm * 1e-6
    n = max(PCB["min_boards"], n_boards)
    rate, fixed = (PCB["usd_per_m2_4layer"], PCB["fixed_4layer_usd"]) if layers == 4 else (PCB["usd_per_m2_2layer"], PCB["fixed_2layer_usd"])
    return n * area * rate + fixed, n


def tile_cost(items, n_lines, populated, n_tiles_ordered, tag):
    """Cost of ONE tile at the given order quantity (tile count of this kind per order)."""
    rows, parts_total, joints, unique = [], 0.0, 0, 0
    for it in items:
        q = it["q"](n_lines, populated)
        if q == 0:
            continue
        up = price_at(it["tiers"], q * n_tiles_ordered)
        rows.append({"key": it["key"], "qty": q, "unit_usd": up, "line_usd": q * up, "jlc": it["c"], "lib": it["lib"]})
        parts_total += q * up
        pads = {"L": 2, "TLV9064": 14, "BCM847BS": 6, "BC847BS": 6, "MCP4728": 10, "4066": 14, "LM4040": 3, "J_SIG": 26}
        joints += q * pads.get(it["key"], 2)
        unique += 1 if it["key"] != "R_misc_E96" else 12
    return {"tag": tag, "parts_usd": parts_total, "joints": joints, "unique_parts": unique, "rows": rows}


def assembly(tile_rows, boards, n_unique_ext):
    """Economic (1 side) and standard (2 sides) fee model for one design/order."""
    e, s = FEES["economic"], FEES["standard_double_sided"]
    joints = tile_rows["joints"] * boards
    out = {}
    out["economic_single_sided"] = e["setup"] + e["stencil"] + joints * e["per_joint"] + n_unique_ext * e["feeder_per_unique_extended"] + FEES["hand_soldering_per_order"] + FEES["confirm_parts_placement"]
    out["standard_double_sided"] = s["setup"] + s["stencil"] + joints * s["per_joint"] + tile_rows["unique_parts"] * s["feeder_per_unique_part"] + FEES["hand_soldering_per_order"] + FEES["confirm_parts_placement"] + s["handling_min_diff"] * 0
    return out


def scaling(width, depth, full_tile_usd, l1_tile_usd, tile_mm=(160.0, 100.0)):
    g = width // 8
    full = (depth - 1) * g * g
    l1 = g
    ind = 448 * full + 112 * l1
    area_cm2 = (full + l1) * tile_mm[0] * tile_mm[1] / 100.0
    return {"width": width, "depth": depth, "tiles_full": full, "tiles_layer1": l1, "tiles_total": full + l1, "inductors": ind, "board_area_cm2": area_cm2,
            "cost_usd_estimate": full * full_tile_usd + l1 * l1_tile_usd + 50.0, "weights_ternary": None}


def refresh_prices(items):
    for it in items:
        c = it["c"]
        if not re.match(r"^C\d+$", c):
            continue
        h = subprocess.run(["curl", "-sS", "-m", "40", f"https://jlcpcb.com/partdetail/{c}"], capture_output=True, text=True).stdout.replace('\\"', '"')
        pr = sorted({(int(a), float(b)) for a, b in re.findall(r'"componentCode":"' + c + r'","startNumber":(\d+),"endNumber":\d+,"productPrice":([0-9.]+)', h)})
        if pr:
            it["tiers"] = pr[:6]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=str(HERE / "params.json"))
    ap.add_argument("--out", default=str(HERE / "cost.json"))
    ap.add_argument("--parts", default=str(HERE / "parts.csv"))
    ap.add_argument("--refresh-prices", action="store_true")
    a = ap.parse_args()
    P = json.loads(Path(a.params).read_text())
    pops = [l["populated_0402"] for l in P["layers"]]
    n_lines_full = P["design_inputs"]["n_lines"]
    items = bom(n_lines_full, pops)
    if a.refresh_prices:
        refresh_prices(items)
    # demonstrator: tile 1 (2 lines populated), tiles 2, 3 (8 lines)
    t1 = tile_cost(items, 2, pops[0], 1, "layer-1 tile (2 of 8 lines populated)")
    t2 = tile_cost(items, n_lines_full, pops[1], 1, "layer-2 tile")
    t3 = tile_cost(items, n_lines_full, pops[2], 1, "layer-3 tile")
    # a single order of 3 tiles: price tier uses the combined quantity of equal parts: approximate by ordering qty = 3 tiles of the larger type
    t_all = [t1, t2, t3]
    tiles_total_parts = sum(t["parts_usd"] for t in t_all)
    boards = 3
    pcb_tile, nb = pcb_cost(*PCB["tile_size_mm"], 4, boards)
    pcb_carrier, nbc = pcb_cost(*PCB["carrier_size_mm"], 2, 1)
    unique_ext = len({it["c"] for it in items if it["lib"].startswith("Ext") or "Basic or" in it["lib"]}) + 10      # + E96 values
    asm = {}
    for k, v in (("economic_single_sided", None), ("standard_double_sided", None)):
        tot = {"joints": sum(t["joints"] for t in t_all), "unique_parts": t2["unique_parts"]}
        asm[k] = assembly(tot, 1, unique_ext)[k]
    cb = carrier_bom()
    carrier_parts = sum(c["qty"] * c["usd"] for c in cb)
    carrier_asm = FEES["economic"]["setup"] + FEES["economic"]["stencil"] + 150 * 2 * FEES["economic"]["per_joint"] + 6 * FEES["economic"]["feeder_per_unique_extended"] + FEES["hand_soldering_per_order"]
    other = OTHER["nucleo_g474re_usd"] + OTHER["ribbon_cables_usd"] + OTHER["dc_supply_usd"]
    # headline: standard double-sided assembly for tiles, economic for the carrier
    total_econ = tiles_total_parts + pcb_tile + asm["economic_single_sided"] + pcb_carrier + carrier_parts + carrier_asm + other
    total_std = tiles_total_parts + pcb_tile + asm["standard_double_sided"] + pcb_carrier + carrier_parts + carrier_asm + other
    per_tile_econ = (tiles_total_parts + pcb_tile + asm["economic_single_sided"]) / 3
    per_tile_std = (tiles_total_parts + pcb_tile + asm["standard_double_sided"]) / 3
    # per-tile cost with fixed fees amortised away (volume): parts + per-joint only
    l1_asm = t1["joints"] * FEES["economic"]["per_joint"]
    full_parts = (t2["parts_usd"] + t3["parts_usd"]) / 2
    full_tile_marginal = full_parts + (t2["joints"] + t3["joints"]) / 2 * FEES["economic"]["per_joint"] + PCB["usd_per_m2_4layer"] * 0.016
    l1_tile_marginal = t1["parts_usd"] + l1_asm + PCB["usd_per_m2_4layer"] * 0.016
    sc = {f"W{w}_D{d}": scaling(w, d, full_tile_marginal, l1_tile_marginal) for w in (8, 16, 32, 64) for d in (3, 4, 6)}
    ind_cost = {k: v["inductors"] * 0.0717 for k, v in sc.items()}
    out = {
        "LABEL": "ESTIMATE, needs a live JLCPCB quote; prices are single readings of JLC part pages on 2026-11-03, not quotes; not a commitment",
        "fees": FEES, "pcb_assumptions": PCB, "other": OTHER,
        "demonstrator_width8_3tiles": {
            "tiles": [{"tag": t["tag"], "parts_usd": round(t["parts_usd"], 2), "joints": t["joints"], "unique_parts": t["unique_parts"],
                       "top_lines": sorted(t["rows"], key=lambda r: -r["line_usd"])[:6]} for t in t_all],
            "tile_parts_total_usd": round(tiles_total_parts, 2),
            "tile_pcb_usd_for_5_boards_(3_used)": round(pcb_tile, 2),
            "tile_assembly_fees_usd": {k: round(v, 2) for k, v in asm.items()},
            "carrier": {"parts_usd": round(carrier_parts, 2), "pcb_usd": round(pcb_carrier, 2), "assembly_usd": round(carrier_asm, 2), "bom": cb},
            "nucleo_cables_supply_usd": round(other, 2),
            "per_tile_average_usd": {"economic_single_sided_fees": round(per_tile_econ, 2), "standard_double_sided_fees": round(per_tile_std, 2)},
            "system_total_usd": {"economic_single_sided_tiles": round(total_econ, 2), "standard_double_sided_tiles": round(total_std, 2)},
            "system_total_usd_with_25pct_contingency": {"low": round(total_econ * 1.25, 2), "high": round(total_std * 1.25, 2),
                                                       "contingency_covers": "shipping, import duty/VAT, one respin of a tile, stock-outs/alternates, hand-soldered THT, cables, test jigs (NOT modelled; ASSUMPTION 25 %)"},
            "plan_claim_usd": "600-800 (tile 150-250, layer-1 100, carrier+Nucleo 50)",
        },
        "marginal_cost_per_tile_usd_at_volume_ex_fixed_fees": {"full_tile": round(full_tile_marginal, 2), "layer1_tile": round(l1_tile_marginal, 2)},
        "scaling_formula": "inductors = 448 (D-1) (W/8)^2 + 112 (W/8); tiles = (D-1)(W/8)^2 + W/8; area = tiles x 160 x 100 mm; cost ~ tiles x marginal tile cost + 50",
        "scaling": sc,
        "inductor_only_cost_usd_at_2000_price": {k: round(v, 0) for k, v in ind_cost.items()},
        "inductor_count_vs_width_depth": {k: v["inductors"] for k, v in sc.items()},
        "weights_of_the_demonstrator": {"nonzero_0402_resistors_by_layer": pops, "total": int(sum(pops))},
    }
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    # parts.csv
    with open(a.parts, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["function", "value", "package", "mpn", "manufacturer", "jlc_lcsc_c_number", "jlc_library", "price_usd_qty1", "price_usd_highest_tier_listed", "price_tiers_jlc_2026-11-03",
                    "qty_full_tile", "qty_layer1_tile", "datasheet_or_part_url", "key_parameters", "verification_status"])
        for it in items:
            w.writerow([it["function"], it["value"], it["package"], it["mpn"], it["mfr"], it["c"], it["lib"], it["tiers"][0][1], it["tiers"][-1][1],
                        ";".join(f"{q}+:{p}" for q, p in it["tiers"]), it["q"](n_lines_full, 512), it["q"](2, pops[0]), it["ds"], it["params"], it["status"]])
        for c in cb:
            w.writerow([c["function"], c["value"], "", c["mpn"], "", c["c"], c["lib"], c["usd"], c["usd"], "", "", "", "", "", c["status"]])
        for fn, mp, c, par in (("MCU board (ST NUCLEO-G474RE)", "NUCLEO-G474RE", "n/a (ST, not on JLC)", "STM32G474RE, Cortex-M4F 170 MHz, 12-bit ADC x5 up to 4 MS/s, 12-bit DAC x2 (DAC1 ch1/ch2 = I/Q), I2C x4"),
                               ("alt inductor, 1210 (smaller)", "FHW1210IF121JST", "C394101", "120uH +-5% DCR 7.9 ohm SRF 4 MHz Q>=8@796k: fails loss criterion at Qmin (3.5 dB @250 kHz)"),
                               ("alt inductor, shielded drum (big)", "SDR0805-121KL / SRR7045-121M", "C2041705 / C2046680", "120uH 480mOhm/380mOhm 7.8x7.8/7x7 mm: shielded-ish, SRF not on page; 49-61 mm2 each x 448 does not fit a 160x100 tile")):
            w.writerow([fn, "", "", mp, "", c, "", "", "", "", "", "", "", par, "listing only" if c.startswith("C") else "n/a"])
    print("wrote", a.out, a.parts)
    print(json.dumps(out["demonstrator_width8_3tiles"]["system_total_usd"]), out["marginal_cost_per_tile_usd_at_volume_ex_fixed_fees"])


if __name__ == "__main__":
    main()
