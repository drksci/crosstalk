# Build log P2-M1b (retrain for the board + arena models)

Files: `dlnn/model/{train,tdlnet_torch}.py` (extended), `dlnn/model/arena.py` (new), `models/tdlnet_b_*`, `models/v32_modrec_*`,
`models/board_aware/{line_fir.py,line_fir.json,line.cir,eval_aware.py,eval_aware.json}`, `models/arena/*`, `tests/test_model_b.py`,
`docs/model_export.md` (appended). Dataset: `modrec.dataset()` (sha256 `e1b8e744...`), 16000 train / 4000 test, the same for every model.
Protocol: 3 seeds per model, selected by **train accuracy** (the test set is never used for selection; per-seed test numbers are reported only).

## Gate (M1b): w8-d1 ternary >= 0.70 mixed and >= 0.75 at >= 10 dB: PASSED (large margin)
`python -m dlnn.model.train --name tdlnet_b_w8 --width 8 --dilations 1,1,1 --ternary-only`  (K = 8, d = 1, 2->8->8->8, ternary taps AND ternary readout)

| model | seed (train acc) | test mixed | test >= 10 dB |
|---|---|---|---|
| w8-d1 ternary (shipped = seed 1) | 0 (0.7708) | 0.7788 | 0.7983 |
| | **1 (0.7997)** | **0.8055** | **0.8375** |
| | 2 (0.7953) | 0.8080 | 0.8483 |
| w8-d1, ternary taps + float digital readout (`tdlnet_b_w8_fro`) | selected | 0.8512 | 0.8883 |
| w8-d1 board-aware (`tdlnet_b_w8_aware`, ideal line / measured line) | selected | 0.7997 / 0.8065 | 0.8254 / 0.8333 |
| w16-d1 ternary | 0 (0.8934) / 1 / 2 | 0.899 / 0.880 / 0.895 | 0.938 / 0.922 / 0.933 |
| v32 (width 32, 4 layers, d = 1) ternary | 0 (0.9319) / 1 / 2 | 0.9347 / 0.9290 / 0.9313 | 0.976 / 0.970 / 0.972 |
| v32 float baseline | 0 / 1 / 2 | 0.9285 / 0.9257 / 0.9275 | 0.974 / 0.969 / 0.970 |

Shipped values: w16 0.8990 / 0.9383 (seed 0), v32 ternary 0.9313 / 0.9721 (seed 2), v32 float 0.9285 / 0.9738. Per-SNR and per-class numbers are in
`models/*_report.json`. Surprise worth stating: d = 1 w8 (0.8055) beats the phase-1 dilated w8 (0.7348 mixed, 0.7629 at >= 10 dB); the dense 22-sample
receptive field with fine time resolution suits this synthetic set better than the 1/4/16 dilation. This is on a synthetic set (not comparable to
RadioML). 8PSK stays weak (0.33 with ternary readout, 0.38 with float readout). Readout used for the gate: **ternary** (stored as float32 gamma*{-1,0,1});
the float digital readout is a further +4.6 points and physically honest (readout is on the Nucleo) but the gate was met without it.
Fallbacks A/B were not needed and were not trained. v32's float baseline is not better than its ternary net (same recipe, not tuned for float).

## Board-aware variant (`tdlnet_b_w8_aware`)
`models/board_aware/line_fir.py` writes `line.cir` (56 sections, L 120 uH + 3 ohm, C 120 pF, Rs = Rt = 1 kohm, C_in = 106 pF = 1.5 MHz RC with the 1 kohm
source resistor, 10 ns edges), runs ngspice (1.2 s), takes each tap's response (nodes 0, 8, ..., 56) to a one-T_s pulse and samples it on the
symbol clock (clock = (centroid tap 7 - centroid tap 0)/7 = 0.893 us, i.e. MCU-calibrated; one common S/H phase chosen to maximise on-lag energy;
mean on-lag gain normalised to 1, the per-layer amplifier gain is learnt). Result `line_fir.json`: per-tap on-lag gain 0.8-1.15, neighbour-lag
leakage up to -0.31, off-nominal energy fraction 0.04-8 %. The torch layer uses `W_long[o,i,m] = sum_k W[o,i,k] h_k[m]` (ternary taps unchanged).
Effect is small: the non-aware w8 loses 0.7 point on the measured line (0.8055 -> 0.7983); aware training recovers it (0.8065 on the measured
line; but 0.7997 on an ideal line). Caveats: the ngspice line is the behavioural tap response only (no buffers, no summer, no noise, no mismatch,
no S/H droop), and the ladder here shows no net tap loss (tap-7 peak +0.5 dB vs tap 0, termination-dependent), unlike the planner's -2.4 dB figure.
The aware model's export is a normal `dlnn-model/1` (FIR in `meta_json.arch.line_fir`); the V/E tiers ignore the FIR.

## Arena (A1)  `python -m dlnn.model.arena`  -> `models/arena/*.npz`, `arena_report.json`
Hyper-parameters on the last 2000 train examples, refit on all 16000; ops for one 128-sample window (a MAC = mult + add; ternary tap = one add).

| model | params | mults | adds | nonlin | mixed | >= 10 dB |
|---|---|---|---|---|---|---|
| (a) TDL-Net-B w8 (ternary readout) | 1275 (1152 ternary, 697 non-zero) | 3144 | 93376 | 3072 | 0.8055 | 0.8375 |
| (a') same with float digital readout | 1275 | 3144 | 88000 | 3072 | 0.8512 | 0.8883 |
| (b) MLP s: 256-4-8 | 1068 | 1056 | 1068 | 4 | 0.2850 | 0.2963 |
| (b') MLP l: 256-32-8 | 8488 | 8448 | 8488 | 32 | 0.5465 | 0.5679 |
| (c) 1-D CNN, ReLU, 2-8-8-8 k=8, float | 1248 | 147528 | 151616 | 3072 | 0.9093 | 0.9546 |
| (d) random ternary reservoir 2-32-32-32 + ridge (pool mean and mean of y^2) | 1544 trained (+ 16896 fixed random taps) | 26496 | 1341064 | 12288 | 0.7183 | 0.7438 |
| (e) cumulant/statistics features + softmax regression (25 features) | 208 | 27384 | 26394 | 1172 | 0.8745 | 0.9583 |

Honest reading: on this synthetic set the conventional float CNN with the same topology (0.909) and the classical feature classifier (0.875, 0.958 at >= 10 dB)
both beat the ternary DLNN w8 at similar parameter counts; the raw-I/Q MLP is poor (no phase/CFO invariance, 128 x 2 inputs). The DLNN's argument
is cost (ternary adds, no multipliers; analog delay line), not accuracy at w8. The decision tree (depth 8, validation 0.789) lost to logistic
regression (0.87) and is not shipped. A tiny 16-unit tanh MLP on the same features reached 0.886 validation (not shipped, not in the file set).
The reservoir's `mean_sq` pooling needs a digital square (the DLNN pool is a plain mean); plain-mean pooling was worse on validation. The reservoir's
random layers are generated, not trained, and the 1.34 M adds figure is dominated by the 32-wide dense random taps (60 % non-zero).
Features (`dlnn.model.arena.FEATURE_NAMES`): |C20|, |C40|, |C41|, C42, |C60|, C63 after CFO compensation (4th-power spectral line / 4), raw |C40|, |C41|,
line strength of x^2, x^4, x^8, amplitude CV/kurtosis/skewness, instantaneous-frequency std/kurtosis (lags 1, 4, 8), |abs|-normalised lag-8 product, spectrum
width/flatness/peak. Op counts for (e) are an analytic count of the numpy code (FFT as radix-2 butterflies; atan2/sqrt/log/exp = 1 nonlinear op each); the
other counts are exact from the layer shapes.

## Energy refs (A4 inputs): `models/arena/energy_refs.json`
Verified from the primary source: Horowitz ISSCC 2014 Fig. 1.1.9 (PDF read; 8b add 0.03 pJ, 32b add 0.1, 8b mult 0.2, 32b mult 3.1, fp16 add 0.4, fp32 add 0.9,
fp16 mult 1.1, fp32 mult 3.7 pJ, 8 KB cache 10 pJ, DRAM 1.3-2.6 nJ, 70 pJ/instruction overhead; 45 nm, 0.9 V). The tile figure (about 100 mA at 5 V) is the plan's own design
estimate (no external source). **STM32G4 uA/MHz is not recorded**: st.com and the Mouser copy of the flyer could not be fetched (503 / reset); a search snippet
quoted about 163 uA/MHz at 150 MHz and 173 uA/MHz at 170 MHz boost from a third-party copy, unverified.

## Training time and determinism
w8 ternary 3 seeds 97 s (train-loop seconds; 105 s wall incl. data), w8-aware 167 s, w16 304 s, v32 ternary 530 s (+ float 311 s), w8 float-readout 93 s; arena: MLP/CNN/reservoir/cumulant
4.5 min total (2 threads, box shared). A full re-run of the w8 command into a second directory gave a byte-identical `.npz` (same sha256) and identical report numbers;
the arena cumulant model also re-ran byte-identically.

## Deviations
- `train.py` selection by train accuracy (phase-1 protocol), not a held-out validation split; the arena baselines use a real validation split for hyper-parameters.
- Per-seed test accuracy is now stored in `restarts[]` of the train reports (reporting only).
- The aware-training line response is a pulse-sampled FIR, not a full co-simulation of buffers and summers.
- Fallback A/B not trained (gate passed).

## Board-aware v2 (`tdlnet_b_w8_aware2`): retrained on the REAL M1a line
Files: `models/board_aware/{line_fir_m1a.py,line_fir_m1a.json,train_aware2.py}`, `models/tdlnet_b_w8_aware2_{ternary.npz,ternary.pt,report.json}`,
`LineAug` + `TDLNet.set_line_fir` in `tdlnet_torch.py`, `line_aug=` kwarg of `train.train_one` (additive; phase-1 CLI unchanged), tests in `tests/test_model_b.py`.

**Line FIR.** `hw/board/spice/linelib.line_netlist` with the M1a design values (`hw/board/params.json`): 56 x (120 uH, DCR 3.3 ohm), 120 pF with 60 pF end caps, SRF 3.3 MHz
(Cp), 6 pF tap loads, 1k/1k terminations, input RC 1 kohm + 330 pF (482 kHz) behind an ideal unity buffer; inductor Rp set so Q = 10 / 30 at 796 kHz. Input = one zero-order-hold
symbol (pulse of width T_s), taps = nodes 0, 8, ..., 56, sampled on the symbol clock. T_s = 0.890 us (mean over both corners of (centroid tap 7 - centroid tap 0)/7, MCU-calibrated),
one common S/H phase, common gain normalisation (mean on-lag gain over both corners = 1, so the Q = 10 vs 30 loss difference survives). ngspice-42, netlist sha256 `2bb8e976...`
(in the json `provenance`, with linelib sha). The TLV9062 macro is NOT used: linelib's line deck has an ideal buffer after the RC and the macro only appears in the summer/slice decks.
Result (fir[k][m], 12 lags): on-lag gain 0.78-1.09 (Q10) / 0.80-1.22 (Q30), previous-lag leakage 0.11-0.42 of the main lag on taps 1-7 growing with tap number (dispersion + RC), off-nominal energy 1-24 % (Q10).
The json also holds a dense (T_s/8) grid per corner for the jitter augmentation.

**Training.** Same recipe as `tdlnet_b_w8` (TDL-Net-B d = 1, K = 8, 2-8-8-8, ternary taps and readout, Adam 3e-2 cosine, 60 epochs, soft-ternary ramp 0.5, IQ-rotation aug) except:
every layer-forward in train mode draws a corner mix a ~ U(0,1) between the Q10 and Q30 FIRs and an independent per-tap delay error delta_k ~ U(-0.14, +0.14) T_s (tap-spacing
ripple, linear interpolation of the dense grid; per tap, not accumulated along the line, a deliberately simple reading of the frequency-dependent +-14 % spacing ripple).
Selection by **validation only**: last 2000 train examples held out (training on the other 14000), seeds 0/1/2, score = mean val accuracy on the Q10 and Q30 lines (0.7837 / 0.7905 / 0.7837 -> seed 1).
Test is reported, never used for selection. No refit on the full 16000 (so aware2 saw 12 % less data than `tdlnet_b_w8`). 3 seeds took ~115 s.
Eval uses the fixed FIR (no jitter). Export: ordinary `dlnn-model/1` (loads via `load_model` and `virtual.forward` unchanged); `arch.line_fir` = Q10 FIR, augmentation + provenance in `meta`.

**Numbers (test set, 4000 examples; `models/tdlnet_b_w8_aware2_report.json`)**

| model \ line | ideal mixed / >=10 dB | M1a Q=10 | M1a Q=30 |
|---|---|---|---|
| `tdlnet_b_w8` (plain) | 0.8055 / 0.8375 | 0.7722 / 0.7983 | 0.7692 / 0.7937 |
| `tdlnet_b_w8_aware2` | 0.7083 / 0.7592 | 0.7970 / 0.8192 | 0.7995 / 0.8204 |

Per-class on Q = 10 (plain -> aware2): OOK 0.950 -> 0.982, BPSK 0.954 -> 0.940, QPSK 0.494 -> **0.000**, 8PSK 0.250 -> 0.768, 16QAM 0.612 -> 0.768, 2FSK 0.966 -> 0.964, 4FSK 0.964 -> 0.992, GMSK 0.988 -> 0.962.
Q10 line plus a random +-14 % per-tap delay error (10 draws): plain 0.7685 (0.7585-0.7738), aware2 0.7967 (0.7950-0.7983), i.e. aware2 is insensitive to it.

Reading: the real M1a line costs the plain net 3.3-3.6 points (the earlier 0.7 point loss was for the milder M1b line). Aware2 recovers to ~0.80 on both Q corners (0.9 point under the plain net on an ideal line) and is flat over Q and tap-delay error,
but it is a board-only net: on an ideal line it falls to 0.708. The class split changed: **QPSK collapses to 0 % on the selected seed** (true QPSK is labelled 8PSK 365/500, 16QAM 61, GMSK 51, BPSK 19, 2FSK 4; QPSK and 8PSK are merged into the 8PSK output), and the
two PSK/QAM confusions remain the weak spots. A collapsed class with a near-flat total is a fragile solution; the 3 seeds were all within 0.5 point of each other on validation but per-class behaviour of the other seeds was not inspected.
Unchanged limits: this is still a 3-seed, single-dataset result with a test difference of ~2.5 points that is larger than seed spread but not independently replicated.

**Limits.** The FIR covers the LC line and the input RC only (ideal unity buffer after the RC, no TLV9062, no tap buffers, no summer, no S/H droop/charge injection, no thermal/ADC noise, no crosstalk, no component
mismatch beyond the delay jitter, no Q spread between sections, one common clock/phase, lines treated as identical for all 8 input channels and layers). Those belong to the P tier. The Q = 10 and 30 corners are the
datasheet minimum and the assumed typical (Q typical is unknown). The +-14 % jitter is applied as independent per-tap time shifts, not as the real frequency-dependent group-delay ripple (which is already in the FIR as ISI).

## Board-aware v2 fix: class collapse (QPSK = 0) removed
Files: `models/board_aware/train_aware2.py` (variants + selection + ship), `models/board_aware/aware2_search.json` (all candidates, 3 seeds each, val + test per class), `models/board_aware/candidates/*.pt` (every trained candidate, `progress.log`),
`models/tdlnet_b_w8_aware2_{ternary,report}`, `models/tdlnet_b_w8_aware2_fro_{ternary,report}`, `models/tdlnet_b_w8_aware2_collapsed_ternary.{npz,pt}` (old model, kept for the record; its numbers are in both reports as `..._collapsed`), tests in `tests/test_model_b.py`.
Cause: on seeds 0/1 (and seed 1 under focal) the QPSK/8PSK pair merges into one output; plain CE with a flat total hides it. The previous selection (mean val accuracy) could not see it.

**Selection (validation only = last 2000 train examples).** Per variant and seed: val macro accuracy = mean over the Q10 and Q30 lines of mean per-class accuracy; a candidate is eligible only if its minimum per-class val accuracy on both lines is >= 0.35. Winner = best val macro among eligible. Test set is reported, never selected on.
Variants (all 60 epochs from scratch unless noted, same recipe as before): base CE; focal (gamma 2); mix (30 % of layer-forwards on the ideal line); ft / ft_mix / ft_focal (20 epochs, lr 1e-2, from `tdlnet_b_w8_ternary.pt`, no soft ramp); fro (float digital readout); fro_focal.

| variant | val macro by seed (0/1/2) | val min-class (0/1/2) | test Q10 mixed (0/1/2) | seeds passing the 0.35 floor |
|---|---|---|---|---|
| base CE | 0.781 / 0.790 / 0.789 | 0 / 0 / 0.18 | 0.789 / 0.797 / 0.796 | 0 of 3 (two collapse to QPSK = 0) |
| focal | 0.813 / 0.778 / 0.793 | 0.51 / 0 / 0.46 | 0.812 / 0.778 / 0.815 | 2 |
| mix | 0.734 / 0.748 / 0.740 | 0 / 0 / 0 | 0.743 / 0.758 / 0.749 | 0 |
| ft | 0.809 / 0.781 / 0.790 | 0.33 / 0.27 / 0.26 | 0.813 / 0.783 / 0.808 | 0 |
| ft_mix | 0.756 / 0.756 / 0.774 | 0.28 / 0 / 0.22 | 0.769 / 0.760 / 0.776 | 0 |
| ft_focal | 0.807 / 0.792 / 0.794 | 0.45 / 0.41 / 0.39 | 0.815 / 0.792 / 0.812 | 3 |
| **fro** | 0.885 / 0.877 / 0.857 | 0.67 / 0.62 / 0.55 | 0.878 / 0.881 / 0.864 | 3 |
| fro_focal | 0.855 / 0.865 / 0.822 | 0.42 / 0.57 / 0.35 | 0.844 / 0.870 / 0.830 | 3 |

Reading: focal loss is what stops the merge for the ternary readout; fine-tuning alone helps but stays under the floor; ideal-line mixing hurts (the board line and the ideal line want different solutions) and does not fix the collapse. The float digital readout is clearly best and never collapses; adding focal to it lowers accuracy.
Per-seed per-class test accuracy on both lines for every candidate is in `aware2_search.json` (`test_Q10.per_class`, `test_Q30.per_class`).

**Shipped (both clear the bar: min per-class test Q10 >= 0.35 and mixed Q10 >= 0.79).**
- `tdlnet_b_w8_aware2_ternary.npz` = ternary readout (the board-default ternary net), variant focal, seed 0 (best eligible ternary-readout candidate on validation).
- `tdlnet_b_w8_aware2_fro_ternary.npz` = float digital readout, variant fro, seed 0 (best candidate overall on validation). Physically honest since the readout is digital on the MCU; taps stay ternary. Use this one when the digital float readout is accepted.

Test set, mixed / >= 10 dB, and per-class at Q = 10 (OOK, BPSK, QPSK, 8PSK, 16QAM, 2FSK, 4FSK, GMSK):

| model | ideal | Q = 10 | Q = 30 | per-class at Q = 10 |
|---|---|---|---|---|
| plain `tdlnet_b_w8` | 0.806 / 0.838 | 0.772 / 0.798 | 0.769 / 0.794 | .950 .954 .494 .250 .612 .966 .964 .988 |
| old aware2 (collapsed) | 0.708 / 0.759 | 0.797 / 0.819 | 0.800 / 0.820 | .982 .940 **.000** .768 .768 .964 .992 .962 |
| **aware2 (focal, ternary readout)** | 0.675 / 0.754 | 0.812 / 0.851 | 0.809 / 0.850 | .956 .918 .492 .548 .696 .966 .976 .946 |
| **aware2_fro (float readout)** | 0.655 / 0.720 | 0.878 / 0.918 | 0.878 / 0.917 | .992 .956 .690 .664 .790 .978 .988 .966 |

Q10 line plus random +-14 % per-tap delay error (10 draws, mean): plain 0.7715, aware2 0.8087, aware2_fro 0.8769. Both are board-only nets (ideal-line accuracy is 0.66-0.68, below the plain net); the weakest classes remain QPSK and 8PSK (0.49-0.69).
Caveats: the focal ternary model gains only 1.5 points of mixed accuracy over the collapsed one and gives up 5 points of 8PSK to get QPSK back; per-class seed spread is large (QPSK 0 to 0.60 across focal seeds), so the floor rule matters more than the seed mean. Still one dataset, 3 seeds per variant, and the FIR limits listed above apply. The two `tests/test_model_b.py` tests re-evaluate both shipped `.pt` files on the Q10 line (min class >= 0.35, mixed >= 0.79, equal to the report). `test_arena_report_mlp_rows_match_npz` currently fails because of the in-progress arena work (not touched here).

## MLP fairness (arena `mlp_s` / `mlp_l` were a strawman; retuned)
Files: `models/arena/mlp_search.py` (search driver), `models/arena/candidates/{raw,feat,featpow}_{s,l}.{json,npz}` + `progress.log` (round-2 results; `candidates/round1/` keeps the first, narrower grid),
`dlnn/model/arena.py` (`MLP(drop=, in_dim=)`, `fold_input_affine`, `mlp_features`, `mlp_feature_ops`, `predict_mlp_npz`, `assemble_mlp_fair`; `python -m dlnn.model.arena --only mlp_fair`),
`models/arena/mlp_{s,l}.npz` (same filenames and schema, still a raw-I/Q ReLU MLP), `mlp_{s,l}_v1.npz` (the old nets), new `mlp_feat_{s,l}.npz`, `mlp_pow_{s,l}.npz`; test `test_arena_report_mlp_rows_match_npz`.

**Protocol.** Grid per row on `train[:14000]` (seed 0), scored on the validation split `train[14000:]`; the best combination is refit on all 16000 train examples with seeds 0/1/2 and the shipped seed is the one with the highest
train accuracy (as for the other arena models). Test accuracy is computed for the report only. AdamW, 60 epochs, batch 128, 200-step warm-up + cosine lr (v1: 40 epochs, no warm-up, no dropout, no input scaling).
Grid: lr {3e-3, 1e-2, 3e-2} x wd {1e-2, 5e-2, 0.2} x dropout {0, 0.2} x hidden config (round 1 had lr {3e-3,1e-2}, wd {1e-3,1e-2,5e-2}; its best points sat on the grid edge, so round 2 widened it; the round-2 optima for
wd (0.2) and sometimes lr (0.03) are again on the upper edge, so a still wider search might add a point or two; not run).
Input standardisation (per-input mean/std from train) is folded into layer 0 after training, so the exported nets take the raw input and the web arena's loader/forward is unchanged. Dropout is train-only. Rotation augmentation
(used by v1 and the CNN) was dropped in the search (untested whether it changes the raw rows; the data already has uniform random phase).

**Rows (test, 4000 ex.; mixed / >=10 dB per seed 0,1,2; ops = network + feature extraction, exact, per 128-sample window).**
| row | input | hidden | params | mixed per seed | >=10 dB per seed | shipped |
|---|---|---|---|---|---|---|
| `mlp_s_v1` | raw I/Q 256 | 4 | 1068 | (train-acc seeds only) | | 0.285 / 0.296 |
| `mlp_s` | raw I/Q 256, standardised | 4 | 1068 | 0.250 / 0.261 / 0.247 | 0.267 / 0.281 / 0.263 | 0.261 / 0.281 |
| `mlp_l_v1` | raw I/Q | 32 | 8488 | | | 0.547 / 0.568 |
| `mlp_l` | raw I/Q, standardised | 38 | 10078 | 0.548 / 0.546 / 0.542 | 0.561 / 0.560 / 0.558 | 0.542 / 0.558 |
| `mlp_feat_s` | 191 phase-inv. features (`compact`) | 6 | 1208 | 0.713 / 0.704 / 0.691 | 0.766 / 0.758 / 0.743 | 0.713 / 0.766 |
| `mlp_feat_l` | same | 40-16 | 8472 | 0.773 / 0.769 / 0.771 | 0.853 / 0.848 / 0.852 | 0.773 / 0.853 |
| `mlp_pow_s` | 319 features incl. |FFT(z^2)|, |FFT(z^4)| (`pow_compact`) | 3 | 992 | 0.781 / 0.783 / 0.781 | 0.848 / 0.835 / 0.840 | 0.783 / 0.835 |
| `mlp_pow_l` | same | 28 | 9192 | 0.845 / 0.848 / 0.848 | 0.933 / 0.925 / 0.928 | 0.845 / 0.933 |
(For reference: DLNN 0.806 / 0.838 at 1275 params, CNN 0.909 / 0.955, cumulant 0.875 / 0.958.) Features: `full` = |z| (128) + instantaneous frequency angle(z[n]z*[n-1])/2pi (127) + |FFT(z)| (128), z scaled to unit RMS;
`compact` = stride-2 mean pooling of each stream; `pow` adds |FFT(z^2)| and |FFT(z^4)| (the classical PSK-order cue; this is hand-engineered and sits between a plain MLP and the cumulant baseline, so it is a separate row, not a replacement).
Op counts (mults/adds/nonlin): mlp_feat_s 4964/4852/391, mlp_feat_l 12178/12116/441, mlp_pow_s 10255/10908/644, mlp_pow_l 18430/19108/669 (network alone: 1:1 MACs; features ~3.8k mult for `full`). Per-seed values above are from the JSONs.

**Findings.** (1) Raw-I/Q MLPs cannot be rescued by tuning: normalisation, dropout, weight decay, longer cosine schedule and wider search moved the ~1k net from 0.285 to 0.26 (within noise, slightly worse) and the ~10k net 0.547 -> 0.542;
validation accuracy is the same as test (0.29 / 0.54), so this is underfitting of a phase-dependent input, not a selection artefact. (2) What fixes it is the input representation:
phase-invariant magnitude/frequency/spectrum features take the matched-size MLP to 0.713 and the ~10k one to 0.773; adding the z^2/z^4 line spectra gives 0.78 (matched, still below the CNN's 0.909 and cumulant 0.875) and 0.845 (~10k).
(3) Honest reading: the matched-size feature MLP (0.71) is below the ternary DLNN (0.806), the matched pow-MLP (0.78) is below it too at ~8x the ops of feature extraction alone (10.3k mult vs 3.1k); the 10k pow-MLP (0.845) beats the DLNN but at 7x the parameters.
(4) Failures/notes: first launch of 4 searches with 2 threads each on a shared 4-core box ran ~12x slower (one fit 378 s); restarted with 1 thread each (about 15 s per fit). Two kill attempts hit my own shell (pkill pattern matched the command line). Round-1 optima on grid edges, hence round 2.
Not tried: batch norm, residual/skip MLPs, label smoothing, mixup, larger searches (3 seeds per row, one data set). 3-seed spread is <=2 points everywhere. The matched feature rows have 992-1208 params; the raw rows 1068. The raw ~10k row is 10078 params (v1: 8488).
The web arena (`ARENA_ORDER`) still shows only `mlp_s`/`mlp_l`; the feature rows exist in the report/npz but `export_web.py` was not touched (not mine).

**STM32G4 energy reference (verified).** `energy_refs.json` now has `stm32g474_run_mode_current`: datasheet DS12288 Rev 2, Table 21 (IDD(Run), code with data processing from Flash single bank, ART on), Range 1 Boost, 170 MHz,
all peripherals off: 29.50 mA typ at 25 C (30.0 mA max; 31.0 mA typ at 85 C), i.e. 173.5 uA/MHz; 150 MHz Range 1: 24.5 mA (163 uA/MHz). Read from the RS mirror `https://docs.rs-online.com/2131/A700000006432374.pdf`
(the st.com PDFs, mouser, farnell, alldatasheet, stmcu.cn all failed: 503/403/timeout). Note the G474 datasheet is DS12288, not DS12712; Rev 2 was read, later revisions not checked; the table gives no explicit VDD.
