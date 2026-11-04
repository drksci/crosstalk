#!/usr/bin/env python3
"""Carrier schematic generator (M3a): NUCLEO-G474RE Morpho headers, DAC I/Q conditioner, ADC protection/anti-alias,
7-9 V jack -> +5VA LDO, TCA9548A I2C mux, tile ribbon headers.   Writes hw/board/carrier/carrier.kicad_sch (+ sheets).

    hw/.venv-eda/bin/python hw/board/gen_carrier.py [--out DIR]

Nucleo morpho pin assignments VERIFIED (M3b) against ST UM2505 Rev 4 (STM32G4 Nucleo-64 boards MB1367), Table 16 "Pin assignment of
the ST morpho connectors" and Table 17, read from the mirror https://cdn.components.ru/media/PDF/1370437.pdf (www.st.com is blocked by
the proxy), and the STM32G474RE pin functions (analog channels, DAC outputs, TIM1_CH1 AF6, I2C1 AF4) against the ST-CubeMX-derived
Zephyr table https://raw.githubusercontent.com/zephyrproject-rtos/hal_stm32/main/dts/st/g4/stm32g474r(b-c-e)tx-pinctrl.dtsi
(the ST DS12712 datasheet itself could not be fetched).  PA5 drives LD2 (SB6), PA2/PA3 are the ST-LINK VCP: not used.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import kisch  # noqa: E402
import place  # noqa: E402
import dlnn_parts as P  # noqa: E402

CN7 = {8: "GND", 19: "GND", 20: "GND", 22: "GND", 16: "+3V3", 28: "ADC0", 30: "ADC1", 32: "DAC_I", 34: "ADC2", 35: "ADC3",
       37: "ADC4"}
CN10 = {15: "ADC5", 24: "ADC6", 22: "ADC7", 13: "DAC_Q", 3: "SCL", 5: "SDA", 23: "SH_CLK", 17: "SYNC", 9: "GND", 20: "GND"}
CN7_NAMES = {28: "PA0 ADC1_IN1", 30: "PA1 ADC1_IN2", 32: "PA4 DAC1_OUT1", 34: "PB0 ADC1_IN15", 35: "PC2 ADC2_IN8",
             37: "PC3 ADC2_IN9", 16: "3V3"}
CN10_NAMES = {15: "PA7 ADC2_IN4", 24: "PB1 ADC1_IN12", 22: "PB2 ADC2_IN12", 13: "PA6 DAC2_OUT1", 3: "PB8 I2C1_SCL (AF4)",
              5: "PB9 I2C1_SDA (AF4)", 23: "PA8 TIM1_CH1 (AF6)", 17: "PB6 GPIO (SYNC)"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "carrier"))
    a = ap.parse_args()
    out = Path(a.out)
    place.write_sim_lib()
    pl = place.Placer(out)
    prj = kisch.Project("carrier", "A3", "Carrier board for NUCLEO-G474RE + tiles (M3a)",
                        "designed, ERC-checked, not built, not measured")
    s_mcu = prj.sheet("mcu", "A3", "Nucleo Morpho headers", desc="CN7 / CN10: DAC, ADC, I2C, clocks")
    s_dac = prj.sheet("dac", "A3", "DAC I/Q conditioner", desc="level shift 1.65 V -> 2.5 V mid, V_cm reference")
    s_adc = prj.sheet("adc", "A3", "ADC front end", desc="8 x series R, BAT54S clamp, anti-alias C")
    s_pwr = prj.sheet("power", "A3", "Power and I2C mux", desc="jack, LDO +5VA, TCA9548A")
    s_til = prj.sheet("tiles", "A3", "Tile ribbon headers", desc="JT1 -> tile 1, JADC <- last tile")

    # ---------------- mcu
    sh = s_mcu
    sh.text("NUCLEO-G474RE Morpho headers (female 2x19, CN7 left / CN10 right).  Unused pins no-connect.  "
            "Pin map verified against UM2505 Rev 4 Table 16 (see gen_carrier.py header).", 25.4, 20.32, 1.5, True)
    for ref, d, names, x in (("CN7", CN7, CN7_NAMES, 80.01), ("CN10", CN10, CN10_NAMES, 240.03)):
        nets = {}
        for p in range(1, 39):
            nets[str(p)] = d.get(p)
            if p in d and d[p] in ("GND",):
                nets[str(p)] = "GND"
        pl.gen(sh, "Connector_Generic:Conn_02x19_Odd_Even", ref + "1" if False else ref, ref, P.FP["SOCK19"], x, 148.59, 0, 1,
               nets, lcsc="C22373924", mpn="HC-PM254-8.5H-2x19PZ", mfr="Hong Cheng", show_val=True,
               extra={"Description": "; ".join(f"{p}:{n}" for p, n in sorted(names.items()))})
        sh.text(ref + ": " + ", ".join(f"{p}={n}" for p, n in sorted(names.items())), x - 50.8, 58.42 if ref == "CN7" else 66.04, 1.0)
    # PWR_FLAGs
    sh.pwr_flag(25.4, 40.64); sh.power("+3V3", 25.4, 40.64)
    sh.pwr_flag(38.1, 40.64); sh.power("GND", 38.1, 40.64)

    # ---------------- dac conditioner
    sh = s_dac
    sh.text("DAC I/Q conditioner: V_out = V_dac + 2.5 V - 1.65 V (unity difference amp, 4 x 10k), I/Q centred on 2.5 V for the "
            "tile input RC (1k + 330 pF, on the layer-1 tile).  Unit 3: 3V3/2 follower, unit 4: V_cm buffer.",
            25.4, 20.32, 1.4, True)
    pl.res(sh, "RVA1", 10000, 38.1, 55.88, 0, "+3V3", ("L", "VAD"))
    pl.res(sh, "RVA2", 10000, 38.1, 76.2, 0, ("L", "VAD"), "GND")
    pl.opamp(sh, "UC1", 3, 76.2, 66.04, ("L", "VAD"), ("L", "VAB"), ("L", "VAB"))
    pl.res(sh, "RRF1", 2490, 38.1, 101.6, 0, "+5VA", ("L", "V2R5"))
    pl.put(sh, "LM4040", "UREF1", 38.1, 124.46, 90, nets={"1": ("L", "V2R5"), "2": "GND"}, sim="LM4040", fsize=1.0)
    pl.opamp(sh, "UC1", 4, 76.2, 111.76, ("L", "V2R5"), ("L", "V25"), ("L", "V25"))
    pl.opamp_power(sh, "UC1", 134.62, 66.04)
    pl.put(sh, "C100n", "CD1", 154.94, 66.04, 0, nets={"1": "+5VA", "2": "GND"}, show_val=False, fsize=1.0)
    for c, dac in enumerate(("DAC_I", "DAC_Q")):
        x = 203.2 + c * 101.6
        pl.res(sh, f"RDI{c}1", 10000, x, 55.88, 0, dac, ("L", f"DP{c}"))
        pl.res(sh, f"RDI{c}2", 10000, x + 15.24, 55.88, 0, ("L", f"DP{c}"), ("L", "V25"))
        pl.res(sh, f"RDI{c}3", 10000, x, 88.9, 0, ("L", "VAB"), ("L", f"DN{c}"))
        pl.res(sh, f"RDI{c}4", 10000, x + 15.24, 88.9, 0, ("L", f"DN{c}"), ("L", f"CO{c}"))
        pl.opamp(sh, "UC1", 1 + c, x + 40.64, 71.12, ("L", f"DP{c}"), ("L", f"DN{c}"), ("L", f"CO{c}"))
        pl.put(sh, "C100n", f"CDV{c}", x + 40.64, 109.22, 0, nets={"1": ("L", "V25"), "2": "GND"}, show_val=False, fsize=1.0)
        # global net name for tile
        pl.res(sh, f"RDO{c}1", 100, x + 66.04, 71.12, 90, ("L", f"CO{c}"), f"CSIG{c}")
    sh.text("RDO: 100 ohm series build-out into the ribbon", 254.0, 40.64, 1.0)

    # ---------------- adc
    sh = s_adc
    sh.text("ADC front end: last tile OUT0..7 (ribbon JADC) -> 1k series, BAT54S clamp to +3V3/GND, 330 pF anti-alias "
            "(482 kHz) -> Nucleo ADC1/2 inputs ADC0..7", 25.4, 20.32, 1.4, True)
    for i in range(8):
        x = 50.8 + (i % 4) * 88.9
        y = 63.5 + (i // 4) * 88.9
        pl.res(sh, f"RAI{i}", 1000, x, y, 90, f"RO{i}", f"ADC{i}")
        pl.gen(sh, "Diode:BAT54S", f"DAI{i}", "BAT54S", "Package_TO_SOT_SMD:SOT-23", x + 25.4, y + 10.16, 0, 1,
               {"1": "GND", "2": "+3V3", "3": f"ADC{i}"}, lcsc="C47546", mpn="BAT54S,215", mfr="Nexperia", sim="NOSIM")
        pl.put(sh, "C330p", f"CAI{i}", x + 12.7, y + 3.81, 0, nets={"1": ("W", ""), "2": "GND"}, show_val=False, fsize=1.0)
        sh.wire(x + 3.81, y, x + 12.7, y)
        sh.junction(x + 12.7, y)
        sh.wire(x + 12.7, y, x + 25.4, y)
        sh.glabel(f"ADC{i}", x + 25.4, y, 0, 1.0)
        sh.label(f"ADC{i}", x + 25.4, y + 5.08, 0, 1.0) if False else None

    # ---------------- power
    sh = s_pwr
    sh.text("Power: 7-9 V barrel jack -> series Schottky -> AMS1117-5.0 (JLC C6187, Basic, SOT-223, Vin max 15 V, dropout 1.3 V) -> +5VA.  "
            "MP20051DQ (C134275) dropped: JLC page lists 5.5 V max supply and an adjustable 0.8-5 V output, unusable from a 7-9 V jack.  "
            "A 7 V supply is preferred (dissipation (Vin-5 V) x 0.22 A).", 25.4, 20.32, 1.4, True)
    pl.gen(sh, "Connector:Barrel_Jack", "J1", "7-9 V DC", P.FP["JACK"], 38.1, 63.5, 0, 1, {"1": ("L", "VJ0"), "2": "GND"},
           lcsc="TBD", mpn="DC barrel jack 5.5/2.1", mfr="generic", show_val=True)
    pl.gen(sh, "Device:D_Schottky", "D1", "SS14", P.FP["SMA"], 76.2, 60.96, 0, 1, {"1": ("L", "VJ1"), "2": ("L", "VJ0")},
           lcsc="C2480", mpn="SS14", mfr="MDD", sim="NOSIM", show_val=True)
    sh.pwr_flag(96.52, 45.72); sh.label("VJ1", 96.52, 45.72, 0, 1.0)
    pl.put(sh, "C10u", "CJ1", 109.22, 71.12, 0, nets={"1": ("L", "VJ1"), "2": "GND"}, show_val=False, fsize=1.0)
    pl.put(sh, "C10u", "CJ2", 168.91, 71.12, 0, nets={"1": "+5VA", "2": "GND"}, show_val=False, fsize=1.0)
    pl.put(sh, "C10u", "CJ3", 184.15, 71.12, 0, nets={"1": ("L", "ESR"), "2": "GND"}, show_val=False, fsize=1.0)
    pl.put(sh, None, "U1", 142.24, 66.04, 0, part=("Regulator_Linear:AMS1117-5.0", "AMS1117-5.0", "Package_TO_SOT_SMD:SOT-223-3_TabPin2",
           "C6187", "AMS1117-5.0", "Advanced Monolithic Systems"), nets={"3": ("L", "VJ1"), "1": "GND", "2": "+5VA"}, sim="NOSIM",
           show_val=True)
    pl.res(sh, "RESR1", 1, 168.91, 55.88, 0, "+5VA", ("L", "ESR"))
    sh.text("RESR1 1R in series with CJ3: AMS1117 wants 0.1-10 ohm ESR on the output capacitor", 160.0, 50.0, 1.0)
    # host I2C pull-ups (the per-tile TCA9548A muxes live on the tiles; the ribbon carries one shared bus)
    pl.res(sh, "RPU1", 4700, 76.2, 175.26, 0, "+3V3", "SCL")
    pl.res(sh, "RPU2", 4700, 91.44, 175.26, 0, "+3V3", "SDA")
    sh.text("I2C host pull-ups 4.7k (SCL, SDA, 3.3 V) for the shared bus; per-tile TCA9548A (address 0x70+ID) sit on the tiles", 63.5,
            160.02, 1.0)

    # ---------------- tile headers
    sh = s_til
    sh.text("Ribbon headers: JT1 -> tile 1 J_IN (I/Q on SIG0/SIG1, power, shared I2C, S/H clock, sync); tile 1 J_OUT -> tile 2 J_IN -> "
            "tile 3 J_OUT -> JADC (8 signals).  One shared I2C bus; each tile has its own TCA9548A (ID jumpers).", 25.4, 20.32, 1.4, True)
    conn = P.__dict__.get("CONN")
    import json
    cj = json.load(open(HERE / "params.json"))["connector"]["J_SIG_2x13"]
    for t in range(1):
        nets = {}
        for pin in range(1, 27):
            nm = cj[str(pin)]
            if nm == "GND":
                nets[str(pin)] = "GND"
            elif nm.startswith("SIG"):
                nets[str(pin)] = f"CSIG{int(nm[3:])}" if (t == 0 and int(nm[3:]) < 2) else None
            else:
                nets[str(pin)] = nm
        pl.put(sh, "J2x13", f"JT{t+1}", 63.5 + t * 88.9, 120.65, 0, nets=nets, sim="NOSIM", show_val=True, fsize=1.27)
    nets = {}
    for pin in range(1, 27):
        nm = cj[str(pin)]
        nets[str(pin)] = f"RO{int(nm[3:])}" if nm.startswith("SIG") else ("GND" if nm == "GND" else None)
    pl.put(sh, "J2x13", "JADC1", 152.4, 120.65, 0, nets=nets, sim="NOSIM", show_val=True, fsize=1.27)
    prj.save(out)
    print("carrier written to", out)


if __name__ == "__main__":
    main()
