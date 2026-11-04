# Parasitics and tolerance tier (P)

`dlnn/tiers/parasitics.py`, tier id `P`. **At about 1 MHz this is not RF.** The board is a lumped-element circuit (the wavelength
is hundreds of metres, the board is 0.16 m), so there is no transmission-line physics, no S-parameter measurement and no EM
solving. scikit-rf is used only as an ABCD two-port cascade engine for the LC ladders. **This is a simulation of a board that is
designed, not built; nothing here is a measurement.** The coax/RF tier (R) is a different, earlier study.

Parameters come from `hw/board/params.json` (M1a, derived by `hw/board/derive_params.py`) and `parts.csv`; where that file has no
value an assumption is used and labelled **[A]** below. `[P]` = from params.json / datasheet reading.

## Signal chain (one layer = one tile)

```
x[n] (model units, 12-bit DAC) -> zero-order hold, period T -> input RC -> driver buffer (GBW) -> [ribbon, NEXT (tiles 2, 3)]
  -> Rs -> 56-section LC ladder (Cp, Rp, DCR, mutual inductance) -> 8 taps (sections 0, 8, ..., 56) -> tap buffer (GBW)
  -> crossbar (10 kohm, +-1 %, DNP-pad leakage) -> inverting summer (R_f error, GBW pole) -> + bias DAC + noise
  -> BJT diff pair (tanh, V_BE offset, temperature) -> diff amp (gain error) -> track-and-hold (acquisition error, pedestal, phase)
  -> next tile ... last tile: 12-bit ADC -> digital mean + readout
```

## Equations

Notation: `w = 2 pi f`, `s = j w`.

* **Section.** Series branch (between node j-1 and j): `Y_ser = 1/(DCR + s L_j) + s Cp + 1/Rp_j`; shunt at node j: `Y_sh = s (C_j + c_tap [tap nodes]) (+ 1/Rl at the last node)`.
  `Cp = 1/((2 pi f_srf)^2 L)` (SRF 3.3 MHz -> 19.4 pF, [P]); `Rp` from the datasheet Q at `f_Q = 796 kHz`:
  `1/Q = DCR/(w L) + w L/Rp` (the same definition as `hw/board/spice/linelib.py`), scaled with L (`Rp ~ Q w L`). Q = 10 is the
  datasheet minimum (default); Q = 30 is an **assumed** typical (the datasheet gives only a minimum). Ends: half capacitors 60 pF,
  `Rs = Rl = 1 kohm`, `c_tap = 6 pF` per tap node.
* **ABCD cascade.** Nominal ladder: skrf `Network.a` per section, cascaded with `**`. Prefix = `Rs, C0, (series, shunt) x j`; the rest of the
  ladder to the open end gives `Y_rest = C_s/A_s` (suffix ABCD, `Rl` inside the last shunt branch); tap transfer
  `H_j(f) = V_j/V_driver = 1/(A_p + B_p Y_rest)`. Toleranced lines (per-section L, C) use the equivalent vectorised backward recursion
  (`V_{j-1} = V_j + J_j/Y_ser,j`, `J_j = I_j + Y_sh,j V_j`); the tests assert both agree to 1e-9 on the nominal ladder
  (the DC tap gains reproduce `params.json` `tap_dc_gain`, 0.542 ... 0.458).
* **Mutual inductance.** Dipole estimate for two 1812 bodies (`V_body` = 46.08 mm^3), [P] formula from `params.json`:
  `k_par(p) = 0.1 V_body / (2 pi p^3)` (0.0272 at p = 3 mm, 0.0059 at 5 mm); orthogonal axes `k_orth = 0.1 k_par` [A]. The layout
  `alternating` makes distance-1 neighbours orthogonal and distance-2 lines parallel again (at 2p); `parallel` makes all lines
  parallel. Section j of line i couples to section j of line i' with `M = k L`. For identical lines the coupled ladders decouple into
  modes of the coupling matrix `K` (`K = U diag(lambda) U^T`, mode m = a single ladder with `L (1 + lambda_m)`), so the nominal
  coupled transfers are exact: `T_ij = sum_m U_im H_m U_jm`. Tolerances are added to first order: the diagonal becomes the toleranced
  line `H_i^tol`, plus the coupling correction `T^coupled_nom - diag(H_nom)`. `crosstalk_k` (sweep axis) is the nearest-neighbour
  coefficient; other entries keep the dipole scaling.
* **Input stage.** `H_rc = 1/(1 + s R C)`, `R = 1 kohm, C = 330 pF` (482 kHz, 10-90 % edge `2.2 R C` = 726 ns, [P]; sweep axis `edge_ns`);
  the buffers are single pole `1/(1 + s/(2 pi GBW))` with GBW = 10 MHz [P] (driver and tap buffer); summer pole `GBW/noise_gain` [P].
* **Ribbon (tiles 2, 3).** Driver output resistance 50 ohm [A], `30 pF/m x 0.1 m` to ground ([P], task) giving a pole at about 1 GHz, plus NEXT
  from the adjacent wire through `C_m = 5 pF/m x 0.1 m` [A] (the 2x13 header interleaves GND): `X(f) = s R_o C_m/(1 + s R_o (C_m + C_g))`,
  about -80 dB at 1 MHz. It is negligible; it is included so the claim can be checked rather than assumed.
* **Pulse and sampling.** Each sample is held for T (`P(f) = T sinc(f T) e^{-j pi f T}`). The linear path is
  `R_{o,j}(f) = P(f) H_sum(f) sum_{i,k} (A_oik + s B_oik) sum_m T_{im,k}(f) M_{mj}(f)`, evaluated at the S/H instants
  `t0 + m T` (lags -2..13) by direct inverse DFT on a 40 ns, 1024-point grid -> symbol-rate MIMO FIR `h[o,j,m]`, applied by
  `rf.fir_apply` (the FFT-convolution engine of tier R). Tests check that less than 1 % of the kernel energy is outside the lag window.
* **Clock calibration.** `T = (centroid(tap 7) - centroid(tap 0))/7` of tile 1, line 0 pulse response (t < 30 us), the definition of
  `models/board_aware/line_fir_m1a.py` (0.889 us vs 0.890 us there). The earlier group-delay-at-122-kHz definition (`cal_method="group_delay"`) gives 0.972 us, a 9 % clock error that wrecks the board-aware models. The S/H phase `t0` of every tile maximises the on-lag fraction of that tile's line 0 tap responses. Residual errors: `T (1 + eps_clk)`,
  `eps_clk ~ U(+-clock_err_pct)` [A: 0.5 %], phase error `U(+-6 ns)` per tile [A: about one 170 MHz timer tick].
* **Crossbar.** Per footprint: populated `+-R_in/(R_in (1 + e))`, `e ~ U(+-1 %)` [P]; DNP footprint: admittance `s C_dnp` with
  `C_dnp = 50 fF (1 + 30 % u)` (0402 pad, [P] design value; the 30 % spread is [A]) so a DNP pad leaks `s C_dnp R_in` (-50 dB at 1 MHz)
  of the opposite polarity; `W = 0` has two such pads whose difference cancels except for the spread.
* **Summer / bias / diff pair.** `z = (S + b_q) (1 + e_Rf) T_fac / (vsat r)` with `S` the FIR output, `b_q` the bias rounded to the bias-DAC LSB,
  `e_Rf ~ U(+-1 %)` [P], `r = vsat_realised_over_design` (E96 R_f, params.json), `T_fac = T_300/T` (2 V_T is proportional to absolute
  temperature; +0.33 %/K, [P]); `z += dV_BE/(2 V_T)` with `dV_BE ~ U(+-vbe_mismatch_mv)` (BCM847BS, 2 mV, **not trimmed away**, conservative);
  `y = tanh(z)`. Stage noise `sigma_z = ratio_to_vsat` from `params.json summer_noise` (-65 dB), or with `stage_snr_db`
  `sigma_z^2 = P_ref 10^(-SNR/10)` (`P_ref` = mean z^2 of the ideal V forward, the convention of tier R).
* **Diff amp, T/H.** `y *= (diffamp_gain_realised/diffamp_gain_exact) (1 + 2 tol_R u)`; track-and-hold keeps `acq_error = 0.11 %` [P] of the previous held value
  and adds the charge-injection pedestal (3.57 mV, assumed 2 pC, [A]).
* **Readout.** 12-bit ADC over 3.3 V on the 2.5 V +- 0.5 V PAM, then mean + model readout in float.
* **Tolerances (Monte Carlo).** Per part uniform in `+-tol` (a bounded distribution; the parts are binned): L 5 % (FHW1812IF121JST, [P];
  the plan said 10 %, sweep axis goes to 15 %), C0G 5 %, R 1 %. Half of the L/C budget (`lot_fraction` = 0.5 [A]) is common to a whole tile
  (same reel), the other half is independent per part. DCR, Cp and GBW have no tolerance (not in the datasheet reading).
  The sweeps use one fixed set of unit draws scaled by the tolerance (common random numbers), tolerance axes average 4 boards.

## Finding F1: gain trim

`derive_params.py` sizes `R_f = R_in 2 V_T/(A_in vsat)` as if the tap amplitude were `A_in x`, but a matched ladder (`Rs = Rl = 1 kohm`)
delivers about half of the driven amplitude (tap DC gain 0.54 ... 0.46 in `params.json`). Left as designed, the summer is 6 dB under-driven
relative to the fixed bias and the three-layer cascade collapses. Tier P therefore has `gain_trim_db`: default `"auto"` = the M1a normalisation (mean on-lag tap gain of tile 1 line 0 = 1 over the Q = 10/30 corners, 8.98 dB; the level the board-aware models were trained on; equivalent to larger `R_f` and `R_b` or a driver gain); 6.02 dB is the older fixed level fix; `gain_trim_db = 0` is the board as designed. Both numbers are in the tier JSON
(`warnings`, `diag.*` arrays) and in the build log. This is a design bug report for the hardware owner, not a tuned metric.

## What is not modelled

No EM or layout (the crosstalk is a dipole estimate, no capacitive or ground-return coupling, no non-neighbour modes beyond the 1/d^3 matrix);
no inductor tempco, saturation or Rp(f) (Q is frequency independent beyond the datasheet point); no op-amp slew, offset, output impedance (except the
50 ohm ribbon driver), PSRR; no supply or ground noise; no clock jitter (only static period and phase error); no beta or Early effect in the diff pair, no
tail-current variation; no 74HC4066 charge-injection dependence on signal; no inductor DCR or Cp tolerance; no temperature dependence except V_T.
Dilation must be 1 (the tile has fixed tap spacing). **No hardware measurement exists; the numbers are predictions of a model.**

## Outputs

`out/<run>/<model>/tiers/P.json|npz` (contract-valid); the nominal metrics use nominal parts on the full 4000-example test set;
sweeps use a fixed stratified subset (1000 examples); the Monte Carlo uses a fixed stratified 800-example subset (indices in `mc.subset_indices`),
and its statistics are in `tiers/P_montecarlo.json`. `docs/artifacts/data/sweeps_grid.json` (`dlnn-sweeps-grid/1`) holds the 5 x 5 x 5 x 5 x 5 grid
for the article slider (see the build log for the subset size).

## Results (model `tdlnet_b_w8_aware2_fro`, the board default; run `P2-M2b`)

**At ~1 MHz this is not RF.** Simulation of a designed, unbuilt board; nothing measured. Model file sha256 `3405923d2c64bb0c2d028a251701cf0e1397baafa0d9233e8e87fe8a238f8706`.
Chunks and logs: `out/P2-M2b/` (`progress.log`, `summary_all.json`, tier JSON in `tdlnet_b_w8_aware2_fro/tiers/P.json`, validated).

| full 4000-example test set | P accuracy | V on ideal line | V on M1a line FIR | P - V(M1a FIR) | top-1 agreement P vs V(M1a FIR) |
|---|---|---|---|---|---|
| Q = 10 | 0.8705 | 0.655 | 0.878 | -0.75 pt | 0.948 |
| Q = 30 | 0.8722 | 0.655 | 0.878 | -0.55 pt | 0.946 |

The board-aware model is trained for the real line, so the right reference is V with the M1a ngspice FIR per tap; against ideal V (0.655) P is +21 pt, which is not a hardware gain. KL vs ideal V is 0.76 (not meaningful for this model).
The board as designed (`gain_trim_db = 0`, finding F1) gives 0.168: the 6 dB under-drive collapses it. Sweep of the trim: 0 dB 0.18, 6 dB 0.46, 8 dB 0.84, 10 dB 0.83. This model needs the gain level within about 2 dB.

**Monte Carlo** (1000 boards, fixed stratified 800-example subset, 337 s compute; L/C 5 %, R 1 %, clock 0.5 %, V_BE 2 mV, phase 6 ns): mean 0.8564, sigma 0.0056, p5 0.8475, min 0.8413; yield (accuracy >= V on the subset - 5 pt) 1.000. Tolerances cost essentially nothing at the datasheet widths. Attribution (one tolerance at a time, 100 boards each): nothing is distinguishable from the nominal 0.850 on that subset (all within +-0.7 pt); the largest spreads come from gain tolerance and V_BE mismatch (sigma 0.4 pt).

**Sweeps** (1000-example subset, accuracy; nominal 0.850): l_tol 0-15 %: 0.858-0.863; c_tol 0-15 %: 0.858-0.861; clock error 0-8 %: 0.860-0.856; stage SNR 60/30/20/10 dB: 0.850/0.857/0.860/0.776 (noise collapses only at 10 dB); input edge 100-1500 ns: 0.845-0.863; crosstalk parallel 3 mm 0.859, orthogonal 3 mm 0.850, parallel 5 mm 0.851, orthogonal 5 mm 0.849 (the crosstalk is harmless at the dipole estimate k = 0.027); Q 10-100: 0.850-0.857; temperature -20..85 C: 0.857-0.869 (+0.33 %/K drift); V_BE mismatch up to 8 mV flat, 16 mV 0.846. Values vary by under 1 pt and are mostly inside subset noise (1000 examples: about 1.1 pt standard error), so no tolerance knee exists within the swept ranges except gain trim and 10 dB noise. All arrays are in the tier npz.

**Grid** `docs/artifacts/data/sweeps_grid.json` (`dlnn-sweeps-grid/1`, sha256 `717c05750591c7099e39d1fcd1d8f1c5963b8061ef53026211464848f68886d9`): 5 axes (l_tol_pct, clock_err_pct, stage_snr_db, crosstalk_k, vbe_mismatch_mv) x 5 points = 3125 points, 800 examples each, one golden board, 305 s; accuracy range 0.849-0.873.

**Scope.** Only `aware2_fro` was run to completion with the current model and calibration (scope cut). Plain and `aware` partial chunks are in `out/P2-M2b/partial_not_assembled/` (not validated, may mix code versions). A first complete run with the old group-delay clock and fixed 6.02 dB trim is in `out/P2-M2b/superseded_v1_group_delay_cal_trim6dB/`. In quick nominal checks with the final code (full test set, Q = 10): plain 0.7655 vs V ideal 0.8055 (-4.0 pt, above the 3 pt target) and V on M1a FIR 0.7722; the gap is line ISI and dispersion that ideal V does not include (P is within 0.7 pt of V on the M1a FIR), not a tuned metric. With the older 6.02 dB trim plain was 0.7985; the result depends on the gain level by about 3 pt.

## Notebook API (`dlnn.tiers.parasitics`)

No hidden global state; `model` is an npz path or a loaded model, `dataset` is `dlnn.data.modrec.dataset()`, `outdir` (optional) receives JSON.

| function | returns |
|---|---|
| `build_line(params=None, corner="Q10"/"Q30")` | `{"fir": [8, 12] tap FIRs, "T_us", "t0_us"}` of tile 1 line 0 |
| `run_nominal(model, dataset, cfg, outdir)` | accuracy / agreement / KL vs V at Q10, Q30, board-as-designed |
| `mc_study(model, dataset, n_trials, subset, seed, cfg, outdir)` | MC stats (mean, std, p5, yield) + per-trial arrays |
| `sweep(model, dataset, axis, values, cfg, subset, boards, outdir)` | one axis: accuracy, mean KL, agreement |
| `make_grid(model, dataset, axes, subset, cfg, outdir)` | dict in the `dlnn-sweeps-grid/1` schema (with sha256) |
| `export_web(outdir, model, dataset, n_trials=50, ..., grid_points=3)` | `p_web.json` + `sweeps_grid.json`; 50 trials and a 3^5 grid take about 70 s |

The checkpointed, resumable batch CLI is `python -m dlnn.tiers.parasitics [--models ...] [--assemble]`.
