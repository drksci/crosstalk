"""P0 demo: 2-section LC ladder + 2 buffered taps + inverting summer, written with kicad-sch-api.
Run with hw/.venv-eda/bin/python. Output: hw/p0/demo.kicad_sch (single source of truth)."""
import sys
from pathlib import Path
import kicad_sch_api as ksa

OUT = Path(__file__).parent
sch = ksa.create_schematic("p0_demo")
sch.set_paper_size("A3")
sch.set_title_block(title="DLNN P0 demo: 2-section LC ladder + taps + summer", date="2026-11-03", rev="P0",
                    company="dlnn phase 2 feasibility spike")
_pwr = [0]
_flg = [0]

def add(lib, ref, val, pos, fp=None, props=None):
    c = sch.components.add(lib, reference=ref, value=val, position=pos)
    if fp:
        c.footprint = fp
    for k, v in (props or {}).items():
        c.properties[k] = v
    return c

def pin(ref, n):
    p = sch.get_component_pin_position(ref, n)
    return (p.x, p.y)

def label(ref, n, name):
    if name in ("VPOS", "VNEG"):   # carried by the +5V / -5V power symbols
        return
    x, y = pin(ref, n)
    sch.add_label(name, position=(x, y))

def power(kind, ref, n):
    """Power symbol (and optional PWR_FLAG) with its pin on top of component pin ref.n."""
    _pwr[0] += 1
    x, y = pin(ref, n)
    add(f"power:{kind}", f"#PWR{_pwr[0]:02d}", kind, (x, y))

def flag(ref, n):
    _flg[0] += 1
    x, y = pin(ref, n)
    add("power:PWR_FLAG", f"#FLG{_flg[0]:02d}", "PWR_FLAG", (x, y))

R = "Resistor_SMD:R_0603_1608Metric"
C = "Capacitor_SMD:C_0603_1608Metric"
L = "Inductor_SMD:L_1210_3225Metric"
PH2 = "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"
PH3 = "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical"
SOT = "Package_TO_SOT_SMD:SOT-23-5"
LIB = "models/lc_parts.lib"
ind = lambda: {"Sim.Library": LIB, "Sim.Name": "IND_DCR", "Sim.Pins": "1=a 2=b", "Sim.Params": "L=120u R=3"}
opa = lambda: {"Sim.Library": LIB, "Sim.Name": "OPA_IDEAL", "Sim.Pins": "1=inp 3=inn 4=out 5=vcc 2=vee"}

# columns laid out on a 25.4 mm pitch; every net joins by label at the pin end
x0, y0 = 30.48, 50.8
# connectors
NOSIM = {"Sim.Enable": "0"}
add("Connector_Generic:Conn_01x02", "J1", "IN", (x0, y0), PH2, NOSIM)
label("J1", "1", "VDRV"); label("J1", "2", "GND"); power("GND", "J1", "2")
add("Connector_Generic:Conn_01x03", "J2", "PWR", (x0, y0 + 50.8), PH3, NOSIM)
label("J2", "1", "VPOS"); label("J2", "2", "GND"); label("J2", "3", "VNEG")
power("+5V", "J2", "1"); power("GND", "J2", "2"); power("-5V", "J2", "3")
flag("J2", "1"); flag("J2", "2"); flag("J2", "3")
add("Connector_Generic:Conn_01x02", "J3", "OUT", (x0, y0 + 101.6), PH2, NOSIM)
label("J3", "1", "OUT"); label("J3", "2", "GND"); power("GND", "J3", "2")

# ladder: Rs - L1 - N1(C1) - L2 - N2(C2, Rt)
add("Device:R", "R1", "1k", (x0 + 40.64, y0), R)
label("R1", "1", "VDRV"); label("R1", "2", "N0")
add("Device:L", "L1", "120u", (x0 + 66.04, y0), L, ind())
label("L1", "1", "N0"); label("L1", "2", "N1")
add("Device:C", "C1", "120p", (x0 + 91.44, y0), C)
label("C1", "1", "N1"); label("C1", "2", "GND"); power("GND", "C1", "2")
add("Device:L", "L2", "120u", (x0 + 116.84, y0), L, ind())
label("L2", "1", "N1"); label("L2", "2", "N2")
add("Device:C", "C2", "120p", (x0 + 142.24, y0), C)
label("C2", "1", "N2"); label("C2", "2", "GND"); power("GND", "C2", "2")
add("Device:R", "R2", "1k", (x0 + 167.64, y0), R)
label("R2", "1", "N2"); label("R2", "2", "GND"); power("GND", "R2", "2")

# tap buffers (voltage followers) U1, U2 and inverting summer U3
for i, (ref, tap) in enumerate((("U1", "N1"), ("U2", "N2"))):
    xu = x0 + 50.8 + i * 50.8
    yu = y0 + 50.8
    add("Amplifier_Operational:LMV321", ref, "LMV321", (xu, yu), SOT, opa())
    label(ref, "1", tap); label(ref, "3", f"T{i+1}"); label(ref, "4", f"T{i+1}")
    label(ref, "5", "VPOS"); label(ref, "2", "VNEG")
    power("+5V", ref, "5"); power("-5V", ref, "2")
add("Amplifier_Operational:LMV321", "U3", "LMV321", (x0 + 152.4, y0 + 50.8), SOT, opa())
label("U3", "1", "GND"); power("GND", "U3", "1")
label("U3", "3", "SUM"); label("U3", "4", "OUT")
label("U3", "5", "VPOS"); label("U3", "2", "VNEG"); power("+5V", "U3", "5"); power("-5V", "U3", "2")
for j, (ref, tap) in enumerate((("R3", "T1"), ("R4", "T2"))):
    add("Device:R", ref, "10k", (x0 + 50.8 + j * 25.4, y0 + 101.6), R)
    label(ref, "1", tap); label(ref, "2", "SUM")
add("Device:R", "R5", "10k", (x0 + 101.6, y0 + 101.6), R)
label("R5", "1", "SUM"); label("R5", "2", "OUT")
# decoupling
add("Device:C", "C3", "100n", (x0 + 127, y0 + 101.6), C)
label("C3", "1", "VPOS"); label("C3", "2", "GND"); power("+5V", "C3", "1"); power("GND", "C3", "2")
add("Device:C", "C4", "100n", (x0 + 152.4, y0 + 101.6), C)
label("C4", "1", "GND"); label("C4", "2", "VNEG"); power("GND", "C4", "1"); power("-5V", "C4", "2")

sch.save(OUT / "demo.kicad_sch")
print("saved", OUT / "demo.kicad_sch")
