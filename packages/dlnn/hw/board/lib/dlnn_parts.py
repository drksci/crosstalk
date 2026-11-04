"""Shared part table (values, footprints, LCSC C-numbers) for gen_tile.py / gen_carrier.py.

Source of truth for values and C-numbers: hw/board/parts.csv and hw/board/params.json (M1a).
"""
import math

KPRJ_LIB = "${KIPRJMOD}/../lib/dlnn_sim.lib"
FP = dict(
    R0402="Resistor_SMD:R_0402_1005Metric", R0603="Resistor_SMD:R_0603_1608Metric",
    C0402="Capacitor_SMD:C_0402_1005Metric", C0603="Capacitor_SMD:C_0603_1608Metric",
    C0805="Capacitor_SMD:C_0805_2012Metric", L1812="Inductor_SMD:L_1812_4532Metric",
    TSSOP14="Package_SO:TSSOP-14_4.4x5mm_P0.65mm", SOIC14="Package_SO:SOIC-14_3.9x8.7mm_P1.27mm",
    MSOP10="Package_SO:MSOP-10_3x3mm_P0.5mm", SOT363="Package_TO_SOT_SMD:SOT-363_SC-70-6",
    SOT23="Package_TO_SOT_SMD:SOT-23", IDC13="Connector_IDC:IDC-Header_2x13_P2.54mm_Vertical",
    PH2x8="Connector_PinHeader_2.54mm:PinHeader_2x08_P2.54mm_Vertical",
    PH1x2="Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical",
    SJ="Jumper:SolderJumper-2_P1.3mm_Open_RoundedPad1.0x1.5mm", TP="TestPoint:TestPoint_Pad_D1.0mm",
    SOCK19="Connector_PinSocket_2.54mm:PinSocket_2x19_P2.54mm_Vertical",
    JACK="Connector_BarrelJack:BarrelJack_Horizontal", SMA="Diode_SMD:D_SMA",
    TCA="Package_SO:TSSOP-24_4.4x7.8mm_P0.65mm", LDO="Package_DFN_QFN:QFN-12-1EP_3x3mm_P0.5mm_EP1.65x1.65mm",
)

# key -> (lib_id, value, footprint, LCSC, MPN, manufacturer, params.json/parts.csv type name)
PART = {
    "L120u": ("Device:L", "120u", FP["L1812"], "C394134", "FHW1812IF121JST", "FH", "inductor"),
    "C120p": ("Device:C", "120p", FP["C0603"], "C126547", "GCM1885C1H121JA16D", "Murata", "c120"),
    "C30p": ("Device:C", "30p", FP["C0603"], "C1658", "0603CG300J500NT", "FH", "c30"),
    "C330p": ("Device:C", "330p", FP["C0603"], "C85978", "GRM1885C1H331JA01D", "Murata", "c330"),
    "R10k_X": ("Device:R", "10k", FP["R0402"], "C25744", "0402WGF1002TCE", "UNI-ROYAL", "xbar"),
    "C560p": ("Device:C", "560p", FP["C0603"], "C472798", "CL10C561JB81PNC", "Samsung", "c560"),
    "C15p": ("Device:C", "15p", FP["C0402"], "C1548", "0402CG150J500NT", "FH", "c15"),
    "C8p2": ("Device:C", "8.2p", FP["C0402"], "C393675", "GCM1555C1H8R2CA16D", "Murata", "c8p2"),
    "C100n": ("Device:C", "100n", FP["C0402"], "C1525", "CL05B104KO5NNNC", "Samsung", "c100n"),
    "C10u": ("Device:C", "10u", FP["C0805"], "C15850", "CL21A106KAYNNNE", "Samsung", "c10u"),
    "R1k": ("Device:R", "1k", FP["R0603"], "C21190", "0603WAF1001T5E", "UNI-ROYAL", "r1k"),
    "TLV9064": ("dlnn:TLV9064IPW", "TLV9064IPW", FP["TSSOP14"], "C779410", "TLV9064IPWR", "Texas Instruments", "tlv"),
    "BCM847BS": ("Transistor_BJT:BC847BS", "BCM847BS", FP["SOT363"], "C549556", "BCM847BS,135", "Nexperia", "bcm"),
    "BC847BS": ("Transistor_BJT:BC847BS", "BC847BS", FP["SOT363"], "C156292", "BC847BS-7-F", "Diodes Inc.", "bc"),
    "MCP4728": ("Analog_DAC:MCP4728", "MCP4728", FP["MSOP10"], "C108207", "MCP4728-E/UN", "Microchip", "mcp"),
    "HCT4066": ("4xxx:4066", "74HCT4066", FP["SOIC14"], "C5998", "74HCT4066D,118", "Nexperia", "sw"),
    "LM4040": ("Reference_Voltage:LM4040DBZ-2.5", "LM4040B25FTA", FP["SOT23"], "C460725", "LM4040B25FTA", "Diodes Inc.", "ref"),
    "J2x13": ("Connector_Generic:Conn_02x13_Odd_Even", "2x13 IDC", FP["IDC13"], "C9040", "2.54-2*13P Straight Header", "BOOMELE", "jsig"),
}

E96_LCSC = {"1k": "C21190", "10k": "C25804", "330": "C23138", "30k": "C22984"}
E96 = [1.00, 1.02, 1.05, 1.07, 1.10, 1.13, 1.15, 1.18, 1.21, 1.24, 1.27, 1.30, 1.33, 1.37, 1.40, 1.43, 1.47,
       1.50, 1.54, 1.58, 1.62, 1.65, 1.69, 1.74, 1.78, 1.82, 1.87, 1.91, 1.96, 2.00, 2.05, 2.10, 2.15, 2.21,
       2.26, 2.32, 2.37, 2.43, 2.49, 2.55, 2.61, 2.67, 2.74, 2.80, 2.87, 2.94, 3.01, 3.09, 3.16, 3.24, 3.32,
       3.40, 3.48, 3.57, 3.65, 3.74, 3.83, 3.92, 4.02, 4.12, 4.22, 4.32, 4.42, 4.53, 4.64, 4.75, 4.87, 4.99,
       5.11, 5.23, 5.36, 5.49, 5.62, 5.76, 5.90, 6.04, 6.19, 6.34, 6.49, 6.65, 6.81, 6.98, 7.15, 7.32, 7.50,
       7.68, 7.87, 8.06, 8.25, 8.45, 8.66, 8.87, 9.09, 9.31, 9.53, 9.76]


def ohm_str(r):
    r = float(r)
    if r >= 1e6:
        s, u = r / 1e6, "M"
    elif r >= 1e3:
        s, u = r / 1e3, "k"
    else:
        s, u = r, ""
    t = f"{s:.4g}"
    return t + u


_LC_CACHE = None


def lcsc_cache():
    """lib/lcsc_cache.json: MPN -> verified JLC row (see jlc_lookup.py); a C-number is used only if it is in here."""
    global _LC_CACHE
    if _LC_CACHE is None:
        import json
        from pathlib import Path
        f = Path(__file__).with_name("lcsc_cache.json")
        _LC_CACHE = json.loads(f.read_text()) if f.exists() else {}
    return _LC_CACHE


def e96_mpn(r):
    """UNI-ROYAL 0603WAF####T5E code: 3 significant digits + multiplier exponent (R = decimal point below 100 ohm)"""
    r = float(r)
    if r == 1.0:
        return "0603WAF100KT5E"       # UNI-ROYAL's own code for 1 ohm (C22936, JLC Basic)
    e = int(math.floor(math.log10(r))) - 2
    m = int(round(r / 10 ** e))
    if r < 100:
        s = f"{m:03d}"
        if r < 10:
            return f"0603WAF{s[0]}R{s[1:]}T5E"
        return f"0603WAF{s[:2]}R{s[2]}T5E"
    return f"0603WAF{m:03d}{e}T5E"


def e96_nearest(r):
    """nearest E96 value (ohm) to r"""
    e = int(math.floor(math.log10(r)))
    cands = [m * 10.0 ** (e + k) for k in (-1, 0, 1) for m in E96]
    return float(f"{min(cands, key=lambda c: abs(math.log(c / r))):.4g}")


def res_e96(r):
    """(lib, value, fp, LCSC, MPN) for an 0603 1 % resistor of value r ohm; LCSC from the verified JLC cache else TBD"""
    v = ohm_str(r)
    mpn = e96_mpn(r)
    row = lcsc_cache().get(mpn)
    lcsc = row["lcsc"] if row else E96_LCSC.get(v, "TBD")
    return ("Device:R", v, FP["R0603"], lcsc, mpn, "UNI-ROYAL")


def props(part):
    """LCSC / MPN / Manufacturer fields from a PART-like tuple (lib, value, fp, lcsc, mpn, mfr, ...)"""
    return {"LCSC": part[3], "MPN": part[4], "Manufacturer": part[5]}


SIM = dict(
    L={"Sim.Device": "SUBCKT", "Sim.Library": KPRJ_LIB, "Sim.Name": "IND_FH1812", "Sim.Pins": "1=a 2=b"},
    TLV9064={"Sim.Device": "SUBCKT", "Sim.Library": KPRJ_LIB, "Sim.Name": "OPA_QUAD",
             "Sim.Pins": "3=in1p 2=in1n 1=out1 5=in2p 6=in2n 7=out2 10=in3p 9=in3n 8=out3 12=in4p 13=in4n 14=out4 4=vcc 11=vee"},
    DUAL_NPN={"Sim.Device": "SUBCKT", "Sim.Library": KPRJ_LIB, "Sim.Name": "DUAL_NPN",
              "Sim.Pins": "1=e1 2=b1 3=c2 4=e2 5=b2 6=c1"},
    HCT4066={"Sim.Device": "SUBCKT", "Sim.Library": KPRJ_LIB, "Sim.Name": "SW4066_QUAD",
             "Sim.Pins": "1=a1 2=b1 13=e1 4=a2 3=b2 5=e2 8=a3 9=b3 6=e3 11=a4 10=b4 12=e4 14=vdd 7=vss"},
    MCP4728={"Sim.Device": "SUBCKT", "Sim.Library": KPRJ_LIB, "Sim.Name": "DAC4_IDEAL",
             "Sim.Pins": "1=vdd 2=scl 3=sda 4=ldac 5=rdy 6=oa 7=ob 8=oc 9=od 10=vss"},
    LM4040={"Sim.Device": "SUBCKT", "Sim.Library": KPRJ_LIB, "Sim.Name": "LM4040_25", "Sim.Pins": "1=k 2=a"},
    NOSIM={"Sim.Enable": "0"},
)


def sim_lib_text():
    vt, vbe_on, bf = 0.025852, 0.665, 300.0
    ic = 2e-3 / (bf / (bf + 1))
    is_ = ic / math.exp(vbe_on / vt)
    cp = 1.0 / (2 * math.pi * 1e3 * (10e6 / 1e5))
    return f"""* dlnn_sim.lib: baseline behavioural models for the TILE-8 / carrier netlists (hw/board/lib).
* GENERIC and IDEALISED: not vendor models, nothing measured.  The C tier (M2a) substitutes the TI TLV906x macro,
* the datasheet-derived inductor model and the fitted NPN; these stand-ins only make the exported netlist elaborate.
.subckt IND_FH1812 a b L=120u R=3.3 Cp=19.4p Rp=6350
L1 a m {{L}}
R1 m b {{R}}
Cp1 a b {{Cp}}
Rp1 a b {{Rp}}
.ends IND_FH1812
* quad op-amp: VCVS 1e5 with a single pole for GBW 10 MHz (TLV9064), ideal output, rails only loaded by 1 Gohm
.subckt OPA_QUAD in1p in1n out1 vcc in2p in2n out2 in3p in3n out3 in4p in4n out4 vee
E1 x1 0 in1p in1n 1e5
R1 x1 y1 1k
C1 y1 0 {cp:.4e}
Eo1 out1 0 y1 0 1
E2 x2 0 in2p in2n 1e5
R2 x2 y2 1k
C2 y2 0 {cp:.4e}
Eo2 out2 0 y2 0 1
E3 x3 0 in3p in3n 1e5
R3 x3 y3 1k
C3 y3 0 {cp:.4e}
Eo3 out3 0 y3 0 1
E4 x4 0 in4p in4n 1e5
R4 x4 y4 1k
C4 y4 0 {cp:.4e}
Eo4 out4 0 y4 0 1
Rvcc vcc 0 1e9
Rvee vee 0 1e9
.ends OPA_QUAD
* dual NPN (BCM847BS / BC847BS), pins in symbol order e1 b1 c2 e2 b2 c1; generic model fitted as in hw/board/spice/run_checks.py
.model QNPN_G npn (IS={is_:.4e} BF=300 VAF=100 RB=10 TF=0.6n CJE=13p CJC=4.8p MJE=0.33 MJC=0.33 VJE=0.75 VJC=0.75)
.subckt DUAL_NPN e1 b1 c2 e2 b2 c1
Q1 c1 b1 e1 QNPN_G
Q2 c2 b2 e2 QNPN_G
.ends DUAL_NPN
* 74HCT4066 quad switch: Ron 118 ohm (TI HC4066 85 C max, HCT assumed equal), threshold 2.5 V on the enable
.model SW4066 SW (Ron=118 Roff=1e9 Vt=1.5 Vh=0.2)
.subckt SW4066_QUAD a1 b1 e1 a2 b2 e2 a3 b3 e3 a4 b4 e4 vdd vss
S1 a1 b1 e1 vss SW4066
S2 a2 b2 e2 vss SW4066
S3 a3 b3 e3 vss SW4066
S4 a4 b4 e4 vss SW4066
Rvdd vdd 0 1e9
.ends SW4066_QUAD
* MCP4728 as four DC sources (VA..VD default 1.024 V = code 2048 with the internal 2.048 V reference; the bias offset resistor RO
* on each channel sheet centres this on the 2.5 V virtual ground; set per channel from the model bias)
.subckt DAC4_IDEAL vdd scl sda ldac rdy oa ob oc od vss VA=1.024 VB=1.024 VC=1.024 VD=1.024
Va oa vss {{VA}}
Vb ob vss {{VB}}
Vc oc vss {{VC}}
Vd od vss {{VD}}
Rvdd vdd 0 1e9
Rscl scl 0 1e9
Rsda sda 0 1e9
Rldac ldac 0 1e9
Rrdy rdy 0 1e9
.ends DAC4_IDEAL
* 2.5 V shunt reference as an ideal source
.subckt LM4040_25 k a
V1 k a DC 2.5
.ends LM4040_25
"""
