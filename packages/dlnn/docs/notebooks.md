# Notebooks and figures (M4b)

Simulation only, synthetic data, no hardware measured. The notebooks are the modelling/simulation surface: each one runs real
code (numpy, iverilog, ngspice, scikit-rf) live on a small subset and reads the heavy results from a finished pipeline run.
They never rerun the 10-minute pipeline.

## The seven notebooks (`notebooks/00..06_*.ipynb`, saved **with executed outputs**)

| notebook | what it shows | live compute | measured runtime* |
|---|---|---|---|
| `00_overview` | the concept (layer equations, tap-to-hardware table, the four tiers), headline results table read from the run and from `out/LATEST`, width 8 vs 32, figure 1 (network from the real weights), figure 7, the honesty box | none | 4 s |
| `01_task_and_model` | the eight signal classes (waveforms, constellations, spectra), dataset sha256 regenerated and compared, training report (float vs ternary, width 8 vs 32, per-SNR, per-class incl. the **8PSK collapse of the width-8 net**), confusion matrices, ternary weight histograms vs the float baseline | dataset generation | 6 s |
| `02_virtual` | the ideal forward pass step by step for one example (tap sum, tanh, three layers, pool, softmax) with equations, checked against `virtual.forward` and the stored V tier; hardware cost (additions only) | V on 500 examples | 4 s |
| `03_emulated` | fixed-point spec, ADC and LUT, layer 1 in integers, live ADC-bits sweep (500 examples) next to the stored 4000-example sweep, **live RTL vs numpy bit-exact check** (8 examples through iverilog; skips with a message if iverilog is missing) next to the stored 64-example result, mismatch accounting (E vs V flips are near-ties) | iverilog run, 7 ADC widths | 25 s |
| `04_circuit` | the SPICE netlists (`.dc`, `.ac`, unit cell), **live ngspice**: diff-pair `.dc` + tanh fit R^2, tap-line `.ac` H_k(f) + closed-form fit, `.tran` of the unit cell on 3 examples vs V and vs the calibrated model; an explicit SPICE / calibrated / extrapolated table; the loss sweep and loss-compensation modes; figures 2 and 3 | 3 ngspice analyses | 10 s |
| `05_rf` | a scikit-rf `Coaxial` line and a full tap-path cascade built by hand, loss dependence of |H| and of the impulse response (layer-1 vs layer-3 taps), **live sweeps of all five axes on 500 examples** against the stored 4000-example sweeps, the per-stage-SNR convention vs input AWGN; figures 3 and 5 | ~16 channel builds + forwards | 34 s |
| `06_compare_and_llm` | the side-by-side matrix, per-SNR table, per-tier `omits`, figures 4-7, the card text, the verified commentary (figure 8) with unverified findings dimmed and an independent audit against `compare.json` | none (reads outputs) | 5 s |

\* wall time of `jupyter nbconvert --execute` on the 4-core shared box (other jobs running), 2026-11-03; each notebook is far below the 6-minute limit.

## Run them

```bash
# one notebook (outputs are saved into the .ipynb)
.venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dlnn notebooks/03_emulated.ipynb
# all seven
for n in notebooks/0*.ipynb; do .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=dlnn --ExecutePreprocessor.timeout=900 $n; done
# regenerate the (unexecuted) sources first if you edited notebooks/_build.py
.venv/bin/python notebooks/_build.py
```

The kernel `dlnn` is registered by `./setup.sh`. Everything common lives in `dlnn/viz/nb.py` (helpers) and `dlnn/viz/figures.py` (figures), so the notebooks stay readable.

## Environment variables

| variable | meaning | default |
|---|---|---|
| `DLNN_RUN` | run directory (a name under `out/` or a path); must contain `compare.json` | `modrec-w32-full` (the width-32 headline run) |
| `DLNN_LLM_RUN` | run whose `llm/commentary.json` notebook 06 shows | newest run under `out/` with the same model and a real-model commentary (`modrec-w32-qwen-pg`), else `DLNN_RUN` |
| `DLNN_NB_N` | subset size of the live computations (notebooks 02, 03, 05) | 500 |
| `RUN_LLM` | `1` additionally re-runs Qwen live in notebook 06, only if `models/llm/qwen2.5-1.5b-instruct-q4_k_m.gguf` exists | unset (the saved commentary is shown) |

Try the width-8 secondary model with `DLNN_RUN=modrec-w8-full` (the notebooks that quote width-32 numbers in prose do so from the run's own files where possible; the markdown commentary still describes the headline run).

## Figures (`docs/figures/`, PNG and SVG, light and dark, opaque background in the exported files)

Produced by `dlnn/viz/figures.py`; written to `out/<run>/figures/` by the pipeline (`figures.make_all`) and exported to `docs/figures/` with
`.venv/bin/python -m dlnn.viz.export` (about 40 s; width-32 headline run -> `docs/figures/`, width-8 secondary -> `docs/figures/w8/`).
`fig8_llm_*` in `docs/figures/` is the real Qwen commentary of the final run (`final-w32`); `docs/figures/w8/fig8_llm_*` is the width-8 run's template commentary (labelled TEMPLATE). (M4c: the earlier `docs/figures/template/` directory was removed because the final run's own commentary is model output.)

| file | figure | notes |
|---|---|---|
| `fig1_network_*` | network / tap diagram | lines as bars, polarity markers = the actual ternary weights of output unit 0, tap length in metres (RG-58, 19.8 cm/ns, 10 ns), loss of the longest tap at 32 dB/us (36 dB in layer 3) |
| `fig2_traces_*` | one example's line voltages after each layer | V/E/C/R overlaid + residual vs V + NMSE; hollow circles = ngspice samples of the layer-1 unit cell; the footnote says layers 2-3 of C are not SPICE |
| `fig3_channel_*` | tap-path responses, |H(f)|, folded combiner waveforms | C = ngspice `.ac`, R = scikit-rf; folded combiner waveforms at layer 1, formerly captioned "eye" (C: ngspice summing node for 1 example, cached in `figures/.eye_cache_*.npz`; R: stored fine-grid waveform). The two channels use different pulse shapes (stated on the figure) |
| `fig4_outputs_*` | class probabilities of 4 examples per tier | **title shows the TRUE label from the dataset** (earlier version printed V's prediction), KL(V\|\|tier) and top-1 per tier |
| `fig5_sweeps_*` | accuracy and mean KL vs the five sweep axes | one block per axis, tiers as series; the 32 dB/us operating point of a `--cascade` run is marked on the loss axis; R's `snr_db` is per-stage (annotated) |
| `fig6_confusion_*` | V's confusion matrix and (tier - V) for E, C, R | shared diverging scale, E ~ V is blank by design |
| `fig7_agreement_*` | top-1 agreement tier x tier | |
| `fig8_llm_*` | LLM commentary panel | citation chips (metric id, tier in its colour, verification state), unverified findings dimmed and labelled, TEMPLATE vs MODEL OUTPUT badge, `ids_verified` badge next to `numbers_verified`, and an independent-audit block (same rules as `results.md`: ok tiers called fallback/"not the real tool", duplicated metric values, unsupported bit widths) |

Design rules followed (dataviz skill): tiers use the validated categorical slots 1-4 (V blue, E orange, C aqua, R yellow; light and dark steps)
with distinct markers *and* line styles (aqua/yellow are below 3:1 on the light surface, so every series is also direct-labelled or legended);
text always uses ink tokens; one y-axis per panel; mean KL on a symlog axis; sequential ramp = one hue, diverging = blue/red around a neutral grey;
transparent backgrounds in the notebooks' source figures, but **opaque** in the exported `docs/figures/` PNG and SVG files (white for the light variants, the dark surface colour for the dark ones; M4d); 7 in wide (phone-readable).

## How the figures make the 32 dB/us operating point visible

With `--cascade` R inherits the line loss that C fitted (32 dB/us, equal to the value put into the netlist) with no repeaters and C runs at the same loss, so their headline accuracy
is about 0.66 / 0.70 against V's 0.863. Two causes, kept apart in the figures and captions: the no-repeater operating point, and C's default per-layer gain trim, which over-drives layer 3. This is shown, not hidden: figure 1 prints the loss of the longest taps, figure 2's footnote and residual
rows show the damage growing with depth, figure 3 shows the loss and dispersion of the lines, figure 5 marks 32 dB/us on the loss axis (both
curves return to about 0.86 at 0 dB/us), and notebook 04 additionally shows that the choice of loss compensation in C matters (0.663 with the
tier's single gain trim per layer vs 0.804 with none and 0.805 with an ideal per-tap equaliser at the same loss, so most of C's gap is that modelling choice and only about 6 of the 20 points are line loss).

## Tests

`.venv/bin/python -m pytest -q tests/test_viz.py` (about 25 s; skips if `out/modrec-w32-full` is absent): every figure writes PNG + SVG, dark variant and
transparent corner by default, fig 4 titles carry the true label, fig 5 has all five axes and the 32 dB/us annotation, fig 7 values equal `compare.json`,
fig 8 labels TEMPLATE vs MODEL OUTPUT and flags UNVERIFIED, `make_all` never raises, and a small NARMA10 pipeline run still gets its figures.
