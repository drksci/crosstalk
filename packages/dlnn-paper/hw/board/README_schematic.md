# M3a: schematic generators (TILE-8 and carrier)

Status: designed, ERC-clean, not built, nothing measured.

Run everything (about 1 min, ERC + netlists + counts + SVG/PDF export):

    hw/board/lib/build_m3a.sh            # add --no-export to skip SVG/PDF
    hw/.venv-eda/bin/python hw/board/gen_tile.py --layer {1,2,3} [--model PATH] [--out DIR] [--tapmap PATH]
    hw/.venv-eda/bin/python hw/board/gen_carrier.py [--out DIR]
    hw/.venv-eda/bin/python hw/board/lib/check_schematic.py hw/board/tile/layer2 --layer 2

Outputs: `tile/tile.kicad_sch` (layer 2), `tile/layer{1,2,3}/`, `carrier/carrier.kicad_sch`; each with `erc.rpt`, `.net`
(kicadsexpr) and `.cir` (spice); `docs/artifacts/board/schematic/` (SVG, PDF, BOM); `docs/artifacts/data/board/tapmap.json`.

## Method
`lib/kisch.py` is a small deterministic KiCad 9 emitter (hierarchical, global labels, `dnp`/`exclude_from_bom`). kicad-sch-api 0.5.6
was not used because it cannot write global labels, DNP flags, multi-sheet instance paths or refs with underscores. Symbols come from
the stock KiCad libraries plus `lib/dlnn.kicad_sym` (TLV9064 quad, MP20051DQ placeholder pin-out). `lib/dlnn_parts.py` holds
values/footprints/LCSC; `lib/dlnn_sim.lib` is a baseline (idealised) SPICE library the `Sim.*` fields point at.
Every sheet connects by labels (global labels across sheets); the ladder is drawn with wires.
KiCad refdes must end in a digit (else the netlist reports annotation errors), hence `C0_0_1` end caps.

## Sheets (tile)
power, line0..7 (only 0,1 on layer 1), crossbar, ch0..7, connectors.
Refdes: `L{line}_{s}` s=1..56; `C{line}_{n}` n=1..55, end caps `C{line}_0_1/_2`, `C{line}_56_1/_2`; `RP/CP/RS/RT{line}`;
`UTA/UTB{line}` tap-buffer quads; `UIB{q}` input buffers (unit = line%4+1); `RXP_{o}_{i}_{k}` / `RXN_{o}_{i}_{k}`;
channel: `UCS{o/2}` summers, `RFP/RFN/CFP/CFN/RB/CBF{o}`, `QD{o}` BCM847BS, `RLD1_/RLD2_{o}`, `UT{o/2}` mirror tails, `UTR1` mirror
reference, `RDA1_..4_{o}`, `CDA{o}`, `UDA{o/2}` diff-amp + hold buffer, `USH{o/4}` 74HCT4066, `CH{o}`.
Nets: `SIG{i}` input, `T{i}_{k}` tap k after its buffer, `SN{o}` (W=+1 resistor RXP) and `SP{o}` (W=-1, RXN) summing nodes,
`OUT{o}`, `BIAS{o}`, `VREF25`, `MIRB`. Circuit follows the validated slice in `spice/run_checks.py` check_F.
R_f, R_b, diff-amp R are recomputed per model with `derive_params.layer_params()` (default model: 931/475/523 ohm, 20.5k/10k/7.87k;
params.json holds 1330/866/499 for the older modrec model). `tapmap.json` gives (layer,o,i,k) -> refdes, polarity, delay.

## Results (default model models/tdlnet_b_w8_ternary.npz)
ERC: 0 errors, 0 warnings on tile (all variants) and carrier. Netlists export cleanly. Populated crossbar resistors = non-zero
weights: layer 1: 86 of 256 footprints, layer 2: 306 of 1024, layer 3: 305 of 1024.
Counts (`tile/layer*/counts.txt`) match parts.csv for every type except: E96 0603 resistors 59 vs 60; layer-1 100 nF 26 vs 40
(csv copies the full-tile number). params.json's own populated counts (83/271/213) belong to the older model.

## M3b changes (schematic fixes; ERC still 0 errors, 0 warnings on all variants)
* **I2C topology.** One shared host bus (SDA/SCL pass through every tile, J_IN -> J_OUT) with a **TCA9548A on every tile** (`UMUX1`, address 0x70 + tile ID
  from the three ID solder jumpers, 10k pull-ups `RID0..2`); each tile's two MCP4728 sit alone on mux channels 0 and 1 (`RPU1..4` channel pull-ups), so the
  shared default MCP4728 address 0x60 is harmless (this also closes risk R6). The carrier mux was removed; the carrier only holds the host pull-ups.
  Alternatives rejected: one mux on the carrier with N downstream buses needs 3 more ribbon pins (all 26 are used); MCP4728 address re-programming needs an LDAC handshake per device.
* **MCP4728 at 3.3 V** (VIH 0.7 x 3.3 V), internal 2.048 V reference, gain x1 (0..2.048 V, 0.5 mV LSB, centre code 2048 = 1.024 V). The bias network
  needed +-2 V of swing about the 2.5 V virtual ground at the model's biases, so per layer `R_b` is lowered until the largest |bias| needs <= 0.8 V of DAC swing
  (layer 1/2/3: 5.23k/3.16k/7.87k for the aware2_fro model; derive_params values are 13k/10k/7.87k) and a per-channel offset resistor `RO{o}` from +5VA to the
  summing node centres the DAC range (`RO = R_b x 2.5/(2.5-1.024)`, E96). Same bias current, finer LSB in volts, no level shifter. Not simulated in ngspice beyond ERC/netlist.
* **Nucleo pin map verified** against ST UM2505 Rev 4 (STM32G4 Nucleo-64, MB1367) Table 16/17, read from the mirror
  https://cdn.components.ru/media/PDF/1370437.pdf (st.com is blocked by the proxy), and the STM32G474RE pin functions against ST's CubeMX data as
  published in Zephyr hal_stm32 (https://raw.githubusercontent.com/zephyrproject-rtos/hal_stm32/main/dts/st/g4/stm32g474r(b-c-e)tx-pinctrl.dtsi);
  the ST datasheet DS12712 itself was not reachable. Result: DAC_I PA4 = DAC1_OUT1 (CN7-32), **DAC_Q moved from PA5 (LD2) to PA6 = DAC2_OUT1 (CN10-13)**,
  ADC0..7 = PA0, PA1, PB0 (CN7-28/30/34), PC2, PC3 (CN7-35/37), PA7, PB1, PB2 (CN10-15/24/22) (PA2/PA3 are the ST-LINK VCP, PC0/PC1 depend on SB34-37),
  SH_CLK PA8 = TIM1_CH1 AF6 (CN10-23), SYNC PB6 (CN10-17), I2C1 PB8/PB9 AF4 (CN10-3/5; needs option byte nSWBOOT0 = 0 so a PB8 pull-up does not select the bootloader).
  Earlier pins for PA2/PA3/PC0/PC1 as ADC and PA5 as DAC were replaced.
* **LDO**: MP20051DQ dropped (JLC page C134275: 5.5 V max supply, adjustable 0.8-5 V output: unusable from a 7-9 V jack). Replaced by AMS1117-5.0 (C6187, JLC Basic,
  SOT-223, Vin max 15 V, dropout 1.3 V, 100 uVrms noise instead of 13): 7 V supply preferred (dissipation), 1R series resistor with the output capacitor for ESR.
* **LCSC numbers**: all tile and carrier E96 resistors, BAT54S (C47546), SS14 (C2480), the Morpho sockets (C22373924, Hong Cheng 2x19 8.5 mm; footprint match to be checked)
  were resolved by exact-MPN match through the JLCPCB parts search (`lib/jlc_lookup.py`, evidence in `lib/lcsc_cache.json`). Still TBD: the DC barrel jack (footprint not matched to a listed part).
  Low stock at the 2026-11-03 reading: 0603WAF9310T5E (12), 0603WAF8871T5E (1), BCM847BS (4): re-check or substitute before ordering.
* Tap-buffer quads are regrouped for layout: quad A = even taps (0,2,4,6 = ladder nodes 0,16,32,48), quad B = odd taps (nodes 8,24,40,56).
* Tile ribbon pins, J_PS (DNP), monitor header, mounting and the unused-unit parking are unchanged.

## Known issues, deviations, open items
* The 482 kHz input RC is on the tile (M1a), not the carrier; the carrier conditioner is a unity level-shift (1.65 V -> 2.5 V mid).
* Tile ID jumpers are on test pads and the TCA9548A address pins only: all 26 ribbon pins are used. J_PS pins are wired to the summing nodes SP/SN (assumption).
* parts.csv / cost.json were not regenerated for the M3b part changes (not owned here); counts.txt DIFF lines against parts.csv are the new parts (TCA9548A, RO, RID, RPU, offset E96 values).
* Unused amp/switch units of shared packages are parked on the power sheet.
