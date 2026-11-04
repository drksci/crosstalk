# TILE-8 board design (M1a): derived, simulated, and what failed

Status: **designed, simulated in ngspice with datasheet-derived models, not built, nothing measured.** Every number below is produced by
`hw/board/derive_params.py` (-> `hw/board/params.json`), `hw/board/spice/run_checks.py` (-> `hw/board/spice/results.json`) or
`hw/board/cost_model.py` (-> `hw/board/cost.json`, `hw/board/parts.csv`). Costs are **estimates, they need a live JLCPCB quote**.
Vendor-part numbers are datasheet or JLCPCB-listing values; the BJT is a generic model fitted to datasheet table points; the TI op-amp
model is TI's own PSpice macro (TLV9062; TI staff say TLV9064 is the same die). The inductor model is L + DCR + C_p (from SRF) + R_p (from Q),
not a vendor model.

Reproduce (about 15 s on one core, ngspice 42, the phase-1 venv):

```
.venv/bin/python hw/board/spice/run_checks.py      # downloads the TI TLV9062 macro from ti.com into spice/_work/ (not committed)
.venv/bin/python hw/board/derive_params.py [--model models/<any model>.npz]
.venv/bin/python hw/board/cost_model.py
.venv/bin/python hw/board/spice/plot_results.py    # figures in hw/board/spice/figures/
```

## 1. Block diagram

![TILE-8 signal path](figures/board_block.svg)

```
Nucleo DAC1 (I,Q) -> level shift (carrier) -> [per line: 1k + 330 pF RC (482 kHz) -> buffer -> Rs 1k]
   -> LC ladder: 56 x (L 120 uH, C 120 pF C0G), half-C (2 x 30 pF) at both ends, Rl 1k
   -> taps at sections 0,8,..,56 -> 64 tap buffers (TLV9064, gain 1)
   -> crossbar 8 lines x 8 taps x 8 outputs x {+,-} 0402 10k footprints  (populated = weight; CPL file = model)
   -> per output: two inverting summers (node P, node N), R_f per layer, + DAC bias via R_b
   -> BJT diff pair (BCM847BS, current-mirror tail, 1 mA, 1k loads)  = tanh(v_d / 52 mV)
   -> difference amp (20k / 10.5k, 8.2 pF) -> 74HCT4066 track-and-hold (560 pF) -> buffer -> 2.5 V +- 0.5 V PAM -> ribbon -> next tile
Layer 1 populates 2 of the 8 lines. The last tile's 8 outputs go to the Nucleo ADCs.
```

## 2. Derived numbers and where they come from

Design inputs are tagged in `derive_params.py` as `[plan]`, `[JLC]` (JLCPCB listing), `[TI]`/`[DIO]` (datasheet), `[CHOICE]` (made in M1a) or
`[ASSUME]` (not a datasheet value). The model used here is `models/tdlnet_modrec_ternary.npz`; pass `--model` for the M1b retrain.

### 2.1 Delay line (formulas in `line_basic`, `inductor_model`, `ladder_transfer`)

| Quantity | Value | Source |
|---|---|---|
| Z0 = sqrt(L/C) | 1000 ohm | constant-k ladder (Pozar 4e periodic structures; Brillouin 1953) |
| per-section delay sqrt(LC) | 120 ns | same |
| ideal cut-off 1/(pi sqrt(LC)) | 2.653 MHz | same |
| cut-off with the inductor's C_p (19.4 pF from SRF 3.3 MHz) | 2.07 MHz | cos(theta) = -1 with series arm jwL/(1 - w^2 L Cp): f_c' = f_c / sqrt(1 + 4 Cp/C) |
| ideal group delay per 8 sections | 0.960 us at DC, 0.961 / 0.964 / 0.978 / 1.036 us at 125 / 250 / 500 / 1000 kHz | tau_g = 8 tau0 / sqrt(1 - (f/f_c)^2); +8 % at 1 MHz as in the plan |
| tap spacing, exact ladder with parasitics, 125 kHz | mean of taps 1..7 = 0.962 us; **per-interval 0.87-1.10 us (frequency dependent, see 4.3)**; step 50 % crossings: 0.956-0.984 us | `ladder_transfer` (agrees with ngspice to 0.016 dB / 2 mrad) |
| clock calibration | f_s = 1/0.9625 us = 1.039 MS/s nominal | monitor pin, MCU-calibrated as in the plan |
| DC gain per tap (resistive divider, DCR 3.3 ohm x 8 per tap) | 0.542 (tap 0) ... 0.458 (tap 7); lossless ideal 0.5 | `tap_dc_gain`; +0.7 dB / -0.77 dB vs ideal |
| tap loading | 6 pF per tap (TLV906x Cic 4 pF + 2 pF assumed) -> +0.7 % C, +0.35 % delay | `tap_buffer_loading` |
| delay tolerance (L +-5 %, C +-5 % uniform) | systematic worst case +-5.0 %; random sigma 2.0 % per section -> **18 ns at tap 7 (1.9 % of a tap spacing)**; 7 ns between adjacent taps. The plan's 27 ns was for L +-10 %. | tau = N sqrt(LC), first-order error propagation; Monte Carlo of the exact ladder (2000 trials): 18.5 ns |
| input RC | 1 kohm + 330 pF C0G = 482 kHz (plan: 1.5 MHz) | see 4.2 |
| ladder end capacitors | 2 x 30 pF = 60 pF at node 0 and node 56 (plan: full 120 pF) | see 4.3 |

### 2.2 Crossbar, summer, bias (`layer_params`)

Chain, for layer l with input scale A_in (volts per model unit): summer `v_d = R_f sum(W a_phys)/R_in`, pair
`V_out = alpha I_EE R_L tanh(v_d / 2 V_T)`, model `y = gain tanh(u / vsat)`. Matching the tanh arguments gives
**R_f = R_in * 2 V_T / (A_in * vsat)** (2 V_T = 51.73 mV at 27 C; alpha = beta/(beta+1) = 0.9967). Activations are 2.5 V +- 0.5 V per +-1 model unit (plan);
layer-1 input is 0.3 V per model unit (4.5 model-unit full scale -> +-1.35 V, [CHOICE]).

| Layer | vsat (model) | R_f exact -> E96 | R_b (bias) | diff-amp R_f (R_in 20k) | nonzero weights (0402 populated / DNP) | worst busiest-node swing | max loads on one tap buffer |
|---|---|---|---|---|---|---|---|
| 1 | 1.297 | 1329 -> **1330 ohm** | 28.0 k | 10.5 k (0.525) | 83 / 173 | 1.98 V (limit 2.3) | 7 |
| 2 | 1.203 | 860 -> **866 ohm** | 20.5 k | 10.5 k | 271 / 753 | 1.08 V | 7 |
| 3 | 2.077 | 498 -> **499 ohm** | 22.1 k | 10.5 k | 213 / 811 | 0.50 V | 8 |

Realised V_sat is within 0.7 % of design for all three. **The plan quoted R_f of 1.6-3.6 kohm; with R_in = 10 kohm and 0.5 V activations the derived values
are 0.5-1.3 kohm** (the plan's range would correspond to activations of about 0.15-0.2 V, not 0.5 V). Gain error of the diff-amp is set by the 30 kohm load it puts on the 1 kohm collectors
(R_L,eff = 0.968 kohm), corrected in the 10.5 k choice; bias DAC LSB = 0.0011-0.0015 model units; summer white noise 25-38 uV rms in the 482 kHz band
(0.05-0.07 % of 2 V_T). The plan's S/H charge injection is not in any datasheet I could read: 2 pC is **assumed** (3.6 mV on 560 pF).

### 2.3 Diff pair, S/H, power

| Item | Value | Source |
|---|---|---|
| tail | **current mirror** (shared diode-connected BC847BS half + R_ref 4.32 kohm -> 1.008 mA), replaces the plan's resistor tail | see 4.4 |
| V_sat temperature coefficient | +0.33 %/K (2 V_T = 47.1 / 52.1 / 56.1 mV at 0 / 27 / 50 C in ngspice) | k T / q; all layers drift together |
| V_BE mismatch | BCM847BS |dV_BE| <= 2 mV = 0.039 of V_sat, trimmed by the bias DAC | Diodes BCM847BS DS41216 Rev 2 |
| S/H | 74HCT4066 (R_on 35 typ / 95 max @25 C / 118 ohm @85 C at 4.5 V, TI CD74HC4066 datasheet; HCT assumed equal), track 450 ns, **C_hold 560 pF** (plan: 1 nF) -> tau 66 ns, acquisition error 0.11 % at 85 C, kT/C 2.7 uV rms, droop ~0 | `sample_hold_params`; 1 nF would leave 2.2 % at 85 C (tau 118 ns) |
| power, full tile | 106 op-amps x 538 uA + 8 mA diff pairs + 5 mA crossbar (duty-factor assumption) + 1.6 mA DACs = **72 mA at 5 V (0.36 W)**; layer-1 tile 42 mA; system 216 mA incl. carrier (1.1 W) | `power_budget` |
| supply | +5VA from the MP20051DQ (13 uVrms, input >= 5.5 V: use a ~6 V DC supply). The Nucleo's USB 5 V cannot feed a 5.0 V LDO | JLC page; dropout 140 mV @1 A |

### 2.4 Connectors (`connector_pinout`)

The plan's 2x13 header cannot carry 8 signals + 8 GND + 8 partial-sum pairs + power + I2C + clocks (16 + 16 + ... > 26 pins). Resolved as:

* **J_SIG, 2x13 IDC ribbon (<= 10 cm), tile-to-tile and carrier-to-tile:** odd pins 1-15 = SIG0..SIG7, each with a GND on the next even pin; 17,19 = +5VA; 18,20,26 = GND; 21 = +3V3; 22 = SDA; 23 = SCL; 24 = SH_CLK; 25 = SYNC.
* **J_PS, 2x8 header, fitted only for W > 8 stitching** (PSP0..7 / PSN0..7): DNP in the width-8 demonstrator.
* Tile ID = 3 solder jumpers; I2C is muxed per tile on the carrier (TCA9548A) so tiles are identical. Monitor = test point on line 0, tap 7.

## 3. Parts (`hw/board/parts.csv`)

### 3.1 The inductor, the critical part

Requirement from the plan: shielded 120 uH, 1008/1210 or similar, SRF >= 5 MHz, DCR <= 8 ohm. **No JLC part meets all of it.** What exists on JLC (read on the part pages):

| Part | C-number | Package | L | DCR | SRF | Q | Shielded | Verdict |
|---|---|---|---|---|---|---|---|---|
| FH FHW1812IF121JST | **C394134** (Extended, $0.14 -> $0.07) | 1812 (4.5 x 3.2 mm) | 120 uH +-5 % | 3.3 ohm | 3.3 MHz | >= 10 @ 796 kHz | no (ferrite drum, wirewound) | **chosen**: passes the three acceptance criteria in simulation; misses SRF >= 5 MHz and shielding |
| FH FHW1210IF121JST | C394101 | 1210 (3.2 x 2.5 mm) | 120 uH +-5 % | 7.9 ohm | 4 MHz | >= 8 | no | smaller (fits the 3 mm pitch), but loss at Q-min is 3.5 dB @250 kHz (fails), DC loss alone 3.2 dB |
| FH FHW1008IF121JST | C394078 | 1008 | 120 uH | 13 ohm | 3 MHz | >= 8 | no | rejected (DCR 13 ohm, 30 mA) |
| Bourns SDR0805-121KL / SRR7045-121M | C2041705 / C2046680 | 7.8 x 7.8 / 7 x 7 mm | 120 uH | 0.48 / 0.38 ohm | not listed | not listed | semi-shielded power inductors | electrically fine but 49-61 mm2 x 448 does not fit a 160 x 100 mm tile |
| Murata LQH43 series | C980xx | 1812 | no 120 uH value on JLC (scanned C98000-C98045) | | | | | n/a |

Q is the weak point. The listing gives the **minimum** Q (10 @ 796 kHz); I could not download the FH datasheet (JLC links are signed and expire), so typical Q is unknown.
Simulations use Q-min (R_p = 6.35 kohm) and an **assumed** typical Q of 30 (3x min, not a datasheet value), plus a lossless bound for ringing.
Because the line is a 56-section cascade, tap-7 loss is set by Q: 2.26 dB at 250 kHz for Q-min, 6.7 dB at 500 kHz, 29 dB at 1 MHz. So the line is **a low-pass at about 280 kHz (-3 dB, Q-min) to
370 kHz (Q 30)**, which is adequate for 125 kBd but means the 1 MS/s ZOH samples are strongly smoothed (a one-symbol pulse of 1 V at tap 7 peaks at 0.32 V against a 0.458 V step plateau, with pre-/post-cursors of +25.7 % / +17.4 % of the peak).
Unshielded chips also raise the crosstalk risk (section 6). If shielded parts become mandatory the alternatives are Coilcraft shielded 1008/1210 parts that are not on JLC (not verified), or a hand-assembled build.

### 3.2 Other parts (all C-numbers read on jlcpcb.com on 2026-11-03; see the CSV for price tiers)

| Function | MPN | C-number | Library | Note |
|---|---|---|---|---|
| quad op-amp | TLV9064IPWR (TSSOP-14) | C779410 | Ext | GBW 10 MHz, en 16 nV/rtHz @1 kHz, Iq 538 uA, Cic 4 pF (datasheet read) |
| matched dual NPN | BCM847BS,135 (Nexperia) | C549556 | Ext | dV_BE <= 2 mV; datasheet read from Diodes' BCM847BS (same spec) |
| dual NPN (mirror) | BC847BS-7-F (Diodes) | C156292 | Ext | datasheet read; alt Nexperia BC847BS,115 C8653 |
| bias DAC | MCP4728-E/UN | C108207 | Ext | 12 bit, 4 ch, 800 uA typ; address bits EEPROM-programmable |
| T/H switch | 74HCT4066D,118 | C5998 | Ext | R_on from the TI HC4066 datasheet (HCT assumed equal) |
| I2C mux | TCA9548APWR | C130026 | Ext | |
| LDO 5.0 V | MP20051DQ-LF-Z (fallback AMS1117-5.0 C6187, Basic) | C134275 | Ext | 13 uVrms |
| 2.5 V reference | LM4040B25FTA | C460725 | Ext | |
| C0G caps | 120 pF GCM1885C1H121JA16D C126547; 30 pF 0603CG300J500NT C1658 (Basic); 330 pF GRM1885C1H331JA01D C85978; 560 pF CL10C561JB81PNC C472798; 15 pF 0402CG150J500NT C1548 (Basic); 8.2 pF GCM1555C1H8R2CA16D C393675 | | | 62 pF and 620 pF C0G were not found, so the ends use 2 x 30 pF and the hold cap is 560 pF |
| crossbar R | 0402WGF1002TCE 10 k 1 % | C25744 | Basic | |
| ribbon header | 2.54-2*13P (BOOMELE) | C9040 | Ext | through-hole |
| MCU | NUCLEO-G474RE | not on JLC | | price assumed $18 (ST site blocked) |

JLC does not show stock reliably in the page data I could parse, so **stock is unchecked**; the inductor listing's stock figures were inconsistent between the two reads.

## 4. ngspice acceptance (`hw/board/spice/run_checks.py`)

Definitions (my choices; the plan did not define them): **ringing** = step overshoot at tap 7, (peak - final)/(final - initial), 50 ns edge at the line input, **including** the input RC;
**loss** = |H(tap 7)/H(tap 0)| in dB at 250 kHz (twice the 125 kBd symbol rate; tap 0 is the same node without a line), i.e. DCR + Q loss, relative to the tap with no line;
**isolation** = output of the summer with the footprint DNP (pad capacitance only) relative to the same path populated, at 500 kHz (Nyquist of the 1 MS/s rate) and 1 MHz.

| Criterion | Result | Verdict |
|---|---|---|
| Ringing at tap 7 <= 10 % | Q-min: **0.0 %**; Q 30 (assumed): **2.8 %**; lossless bound (DCR 0, Q 1e5): **8.8 %**; 20 tolerance Monte Carlo trials (Q 30): 2.7-2.8 % | **pass**, with 1.2 points margin in the worst case |
| Loss <= 3 dB (250 kHz) | Q-min: **2.26 dB**; Q 30: **1.29 dB**. At 125 kHz 1.07 / 0.85 dB; at 500 kHz 6.7 / 2.8 dB; DC -0.77 dB vs a lossless matched line | **pass** at 250 kHz; **fails at 500 kHz for Q-min** (by design the band is < 300 kHz) |
| Crossbar "0" isolation <= -45 dB | **-56.1 dB at 500 kHz, -50.1 dB at 1 MHz**, -44.1 dB at 2 MHz (C_dnp 0.05 pF, TLV9062 macro summer, 8 populated inputs, 10 pF node capacitance). Margin: C_dnp may rise to 0.089 pF at 1 MHz | **pass** (0.05 pF is a design value; a coplanar-strip estimate of the 0402 gap gives 0.019 pF, `pad_gap_capacitance`) |
| Diff pair tanh fit | generic NPN, mirror tail: **R^2 = 1.0000** (residual 0.30 mV rms of a 1.0 V swing), fitted V_sat 52.08 mV (ideal 51.73), swing 1.011 V; +-1.1 % swing and +0.33 %/K V_sat over 0-50 C; 10-90 % time 8 ns | pass |
| Summer, TLV9062 macro, 8 inputs | gain -0.0866 (= -R_f/R_in), f_3dB 7.0 MHz with C_f = 15 pF, no peaking, 0.1 % settling 182 ns | pass |
| Slice (summer -> pair -> diff amp -> T/H -> buffer, 24 random symbols, 6 inputs) | error vs ideal tanh **0.99 % of full scale rms, 3.5 % max** (17 mV); gain 1.01 | informational |

![line response](../hw/board/spice/figures/line_response.png)
![diff pair](../hw/board/spice/figures/diffpair_tanh.png)

### 4.1 What was changed to pass, and what was found (iteration log)

1. **Ringing with the plan's 1.5 MHz RC was 13.7 % (Q 30) and 23 % (lossless)**, matching the plan's "20 %". No input filter: 15.7 %. 1.0 MHz: 11.6 %. **0.5 MHz (482 kHz: 1 kohm + 330 pF) gives 2.8 % / 8.8 %.** The ringing is the sharp constant-k cut-off, not the inductor loss, so the filter has to hold even for a lossless line.
2. **Full-value capacitors at the ends of the ladder are a 60 pF shunt discontinuity.** With C = 120 pF at both end nodes, the tap-to-tap delay swings by 93 % peak-to-peak over 20-250 kHz; with C/2 (2 x 30 pF) it is 29 %. This is the largest unplanned finding of this milestone.
3. **Remaining spacing ripple (about +-14 %, 29-32 % peak-to-peak over 20-250 kHz)**: the DCR (3.3 ohm x 56) makes Z0 complex at low frequency (Delta Z/Z = R/(2 w L) ~ 1.7 % at 125 kHz), the 1 kohm terminations reflect about 0.9 %, and the round trip from each tap to the far end produces a sinusoidal group-delay ripple versus frequency (period 1/(2 T_remaining)). It is deterministic and can be trained on or calibrated, but it is a real ISI source. Not fixed.
4. **Resistor tail gave a distorted tanh** (fit R^2 0.99966, residual 18 mV rms, swing 1.057 V, V_sat 58 mV, because I_EE rises ~10 % when the larger base swings +0.2 V). A V_b/R_E tail fixed that but drifted +-8 % in swing over 0-50 C (V_BE tempco). A **current-mirror tail** (V_BE cancels) gives R^2 1.0000 and +-1.1 %.
5. **Diff-amp ringing**: 0 pF across its 10 k feedback rings (11.6 % FS error), 22 pF is too slow; 8.2 pF is best in the slice (1.0-1.4 % FS error). SPICE-tuned, not a formula.
6. S/H hold capacitor 1 nF -> 560 pF (R_on 118 ohm at 85 C).

## 5. Design changes against the plan

| Plan | Now | Why |
|---|---|---|
| input RC 1.5 MHz | 482 kHz | ringing 14-23 % -> 3-9 % |
| 120 uH shielded, SRF >= 5 MHz, DCR <= 8 ohm | FHW1812IF121JST (unshielded, SRF 3.3 MHz, DCR 3.3 ohm) | nothing on JLC meets all; this one meets the three electrical criteria in simulation |
| full C at ladder ends (implicit) | 60 pF at both ends | tap-spacing ripple 93 % -> 29 % |
| resistor tail | current-mirror tail | tanh shape, temperature |
| 2 x MCP4728 per tile with a shared bus | same, plus a TCA9548A on the carrier; the second MCP4728 on each tile needs a one-time address re-write | identical tiles |
| 2x13 header carrying everything | 2x13 for signals/power/I2C, separate 2x8 for partial sums | pin count (26 < 40) |
| diff-amp x0.1 | x0.525 (20 k / 10.5 k) | 2.5 V +- 0.5 V per +-1 requires 0.5 / (alpha I_EE R_L,eff) |
| R_f 1.6-3.6 kohm | 0.5-1.3 kohm | matching V_sat for R_in 10 k, 0.5 V activations |
| 1 nF hold cap | 560 pF | acquisition error |
| tile $150-250, system $600-800 | tile ~$110-120, system ~$410-440 modelled (~$510-550 with 25 % contingency) | section 7 |

## 6. Crosstalk and layout guidance

Not a field solve. Two neighbouring unshielded wirewound chips are modelled as magnetic point dipoles (`mutual_inductance_estimate`):
for parallel axes side by side at pitch p, **k = M/L ~ V_eff / (2 pi p^3)** with V_eff = 0.1 of the body volume (46 mm3 for a 1812; the 0.1 is **assumed**), i.e. k = 0.027 at 3 mm,
0.0059 at 5 mm, 0.0014 at 8 mm. With axes orthogonal the first-order coupling of an ideal dipole is zero; 10 % of the parallel value is **assumed** to remain.
Forward crosstalk of two LC lines coupled only by inductance is |V_f/V_in| ~ (k/2) w T_d (Matthaei-Young-Jones; coherent over the whole line), evaluated at 250 kHz with T_d = 6.7 us:

| pitch | parallel axes | orthogonal (alternating) |
|---|---|---|
| 3 mm | **-16.9 dB** | -36.9 dB |
| 5 mm | -30.2 dB | -50.2 dB |
| 8 mm | -42.4 dB | -62.4 dB |

So **the plan's crosstalk target (-40 dB) is not demonstrated at 3 mm pitch**: it needs alternating orientation **and** about 4 mm or more between the rows, or shielded inductors. A 1812 body is 3.2 mm
wide, so a 3 mm pitch is not even geometrically possible with the chosen part; 5 mm is the minimum, which also fixes the layout area: 448 x (4.5 x 3.2 mm body at 5 mm row pitch) is about 7,000 mm2 of the 16,000 mm2 tile, so inductors go on **both sides** (standard double-sided assembly).

Guidance:

* Rows of 56 inductors per line, all on one board side per line, row pitch >= 5 mm, **adjacent rows rotated 90 degrees** (long axis alternating), never two parallel-axis chips side by side across lines.
* Solid GND on L2 directly under the inductor rows (JLC04161H-7628, prepreg about 0.18 mm); no signal routing on L2; keep slots out of the plane under the rows. Eddy currents in the plane reduce the field coupling; this is a qualitative benefit that is **not** in the estimate.
* Tap buffers and the crossbar resistors sit on the far side from the inductors; the crossbar nets are low-impedance (summing nodes at 2.5 V virtual ground, 10 k sources).
* Each unpopulated footprint pair must keep its two pads' gap >= 0.5 mm and avoid running another net parallel to it (C_dnp budget 0.09 pF for -45 dB at 1 MHz).
* **Measure** k on two real parts at 3 / 5 mm before committing a layout; the number above is a guide to where to look.

### 6.1 Four-layer stack-up

JLC04161H-7628, 1.6 mm: L1 signal (inductor rows, tap buffers) / L2 GND / L3 +5VA (and 2.5 V) / L4 signal (crossbar, summers, diff pairs). Reasons: continuous GND next to the
inductor layer; crossbar routing needs a second signal layer; the +5VA plane doubles as the local reference. I did not find layer thicknesses on JLC's stack-up page (the images are not machine readable); the 7.1 mil (about 0.18 mm) L1-L2 prepreg for 7628
comes from JLC's impedance-calculator note and is a **secondary-source value**. No controlled-impedance traces are needed at 1 MHz.

## 7. Cost (estimate, needs a live quote)

Fees: JLC economic PCBA setup $8.18, stencil $1.53, $0.0016 per joint, $3.07 per unique extended part; standard double-sided $51.12 + $16.42 + $1.53 per unique part
(https://jlcpcb.com/help/article/pcb-assembly-price, read 2026-11-03). Part prices are JLC page tiers read once. Bare PCB price is **assumed** ($70.6/m2 + $40 for 4 layers, from an older JLC blog), the Nucleo ($18) is assumed.

| Width-8 demonstrator (3 tiles + carrier + Nucleo) | USD |
|---|---|
| tile parts (all three tiles) | 184 |
| tile PCBs (5 x 4-layer, 3 used) | 46 |
| tile assembly fees (economic 1-side / standard 2-side) | 97 / 130 |
| carrier (parts 7, PCB 6, assembly 32), Nucleo, cables, supply | 45 + 34 |
| **system total, modelled** | **406 (economic) - 439 (standard)** |
| with a 25 % contingency (shipping, duty, respin; assumed) | **~510-550** |

The inductors are 44 of the 76 dollars of parts on a full tile (448 x $0.099). The weights themselves cost 83 / 271 / 213 resistors = 567 x $0.002.

Scaling (`cost.json`): tiles = (D-1)(W/8)^2 + W/8, inductors = 448 (D-1)(W/8)^2 + 112 W/8, area = tiles x 160 x 100 mm.

| W, D | tiles | inductors | board area | tiles only, volume price, no fixed fees | inductors alone |
|---|---|---|---|---|---|
| 8, 3 | 3 | 1,008 | 480 cm2 | $250 | $72 |
| 16, 4 | 14 | 5,600 | 2,240 cm2 | $1.1 k | $0.4 k |
| 32, 4 | 52 | 21,952 | 8,320 cm2 | $4.1 k | $1.6 k |
| 64, 6 | 328 | 144,256 | 52,480 cm2 | $26.6 k | $10.3 k |

(The plan's "width 32 = 48 tiles, about 21,500 inductors, about $8-10k" counts only the 48 full tiles; four more layer-1 tiles make 52 / 21,952. The cost here is $4.1 k at volume pricing without contingency, below the plan's $8-10 k.)

## 8. Open risks

1. **Inductor.** Unshielded, SRF 3.3 MHz below the requirement, Q only known as a minimum, listing stock unclear (the 1812 part had very low stock in one search result). Loss and ringing pass in simulation; crosstalk and the real Q are open. First action: buy 20 parts, measure L, DCR, SRF, Q(f) and k at 3 / 5 mm.
2. **Crosstalk.** The -40 dB target is not met by the dipole estimate at 3 mm even with alternating orientation (-37 dB); a 3.2 mm body does not fit a 3 mm pitch. Layout needs 5 mm pitch (or shielded parts), so the two-sided assembly is mandatory.
3. **Group-delay ripple of +-14 % of a tap spacing** (frequency dependent) and one-symbol ISI of 17-26 %: the network must be retrained on the real kernel (M1b) and the clock calibration only fixes the mean spacing.
4. **Model gaps.** The BJT is generic (VAF, R_B assumed; datasheet curves not digitised); S/H charge injection and the 74HCT4066 R_on are not from a datasheet I read; the TLV9062 macro stands in for the TLV9064 (same die per a TI e2e answer, not confirmed in the datasheet).
5. **Temperature.** V_sat moves +0.33 %/K for all layers together; a 50 K range changes the tanh width by 17 %. Needs either thermal control, calibration of the gain, or a retrain that tolerates it.
6. **MCP4728 addressing.** Two per tile with identical default addresses; the one-time address re-write uses the LDAC pin and EEPROM.
7. **Supply.** The 5 V rail needs a ~6 V DC input; the Nucleo's USB 5 V is too low for a 5.0 V LDO.
8. **Cost.** Estimate only; JLC stock for 448 inductors x several boards of one extended part, the extended-part fee, minimum order quantities and shipping are not verified.
9. **Plan consistency.** Several plan numbers (R_f range, header pin count, 27 ns delay sigma, tile cost) do not reproduce from the plan's own parts; the values above supersede them.
