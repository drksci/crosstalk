# Build log M4b: figures 1-8 and executed notebooks

Owner of: `dlnn/viz/{figures,common,nb,export}.py`, `notebooks/*`, `docs/figures/*`, `docs/notebooks.md`, `tests/test_viz.py`.
Not touched: tiers, llm, model, pipeline, run.py, contract/compare, hw, plan, build-log. Only read-only calls into tier/model/rf/circuit modules.
Simulation only; nothing here is a hardware claim.

## What was built

- `dlnn/viz/figures.py`: figures 1-8 (see `docs/notebooks.md` for the table), light and dark, PNG + SVG, transparent, 7 in wide.
  `make_all(run_dir)` is what `dlnn/pipeline.py` already calls; it now renders every applicable figure, never raises (failures are collected in
  `make_all.errors`) and skips classification-only figures for NARMA10. Names `fig4_outputs_*` and `fig5_sweeps_*` are unchanged.
- `dlnn/viz/common.py` (palette, theme, `Run` loader, paired-subset helpers), `dlnn/viz/nb.py` (notebook helpers, env vars), `dlnn/viz/export.py`
  (`python -m dlnn.viz.export` -> `docs/figures/`, 72 files: headline run, `w8/` secondary, `template/` commentary).
- `notebooks/_build.py` builds the seven notebooks; all seven were **executed** with `jupyter nbconvert --execute --inplace` (outputs saved).
- `tests/test_viz.py` (17 tests, about 1 min because one test runs a small modrec pipeline with C/R on a subset).
- Colours: dataviz reference slots 1-4 for the tiers (V blue, E orange, C aqua, R yellow). `validate_palette.js` on the four light hexes: all hard gates pass
  (worst adjacent CVD dE 9.1, normal-vision 22.9), contrast WARN on aqua (2.74) and yellow (2.11) -> relief applied: markers + line styles + legends/direct labels
  on every series. Dark: all gates pass (CVD 8.4, contrast >= 3:1).

## Bug fixed

`fig4_outputs`: panel titles printed `label N` = **V's predicted class**, not the label. Now the title is `example e: true <class> @ <snr> dB` from `labels.npy`,
the true row is shaded and bold, and each tier gets its own top-1 right/wrong. The default examples are chosen deterministically from the first 16 so that one
example where V itself is wrong is always shown (test `test_fig4_titles_use_the_true_label_not_vs_prediction`).

## Decisions and how each requirement was met

- **Figure 2** uses the per-layer activations stored in each tier's npz (`trace_activations.L*`, 16 examples x all lines) - no regeneration needed. The channel shown per layer is the
  line with the largest V variance (layer 1 restricted to the 8 SPICE outputs so the ngspice samples can be overlaid; the stored unit-cell waveforms `waveforms.ex*` cover examples 0-3).
- **Figure 3**: C's H_k(f) = the SPICE `.ac` samples stored in `params/circuit_extracted.json`; R's = `rf.line_response` (skrf cascade) at the run's operating point (cascade -> 32 dB/us).
  Impulse responses are shown as the response to one 10 ns raised-cosine symbol (a rectangular symbol rings from band limiting). The eye at the layer-1 combiner: R from the stored fine-grid
  waveform; C needs the summing node, which the pipeline does not save, so the figure runs **one ngspice `.tran` (1 example, 1 output, about 1 s)** and caches it in
  `<run>/figures/.eye_cache_ex*_o*.npz`. If ngspice or the SPICE extraction is missing (C fell back to the analytic line) the panel says so and the C curves are labelled
  "analytic RLC (FALLBACK, not SPICE)". The two eyes use different pulse shapes (C trapezoid with 1 ns edges, R raised cosine); the caption says so.
- **Subset runs** (C/R on a stratified subset): `Run.index_of` / `paired_preds` read `subsets.json`; figures 6 and 7 compare tiers on the examples both ran on (found and fixed by running the
  pipeline with `--cr-subset 80`; covered by a test).
- **Honesty of the C/R gap** is visible in figures 1, 2, 3, 5 (the 32 dB/us operating point is drawn on the loss axis and in the footnotes) and in notebook 04/05/06, not just in prose.
- **Figure 8** distinguishes `source: template` (badge "TEMPLATE (fixed text, not model output)") from model output, dims unverified findings, and appends an *independent audit*
  (`figures.audit_commentary`) that compares caveat text with `compare.json` statuses.
- Notebooks read heavy results from `DLNN_RUN` (default `out/modrec-w32-full`); `out/LATEST` currently names `narma10-m4a` (a sanity run), so notebook 00 shows both, labelled.

## Acceptance run

```
.venv/bin/python -m pytest -q tests/test_viz.py          -> 17 passed (about 60 s)
.venv/bin/python -m pytest -q tests/test_tiers.py tests/test_pipeline_modrec.py   -> 17 passed, 1 skipped (Postgres), 61 s
for n in notebooks/0*.ipynb; do .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=dlnn $n; done
```

| notebook | cells (code) | error outputs | images | runtime |
|---|---|---|---|---|
| 00_overview | 12 (7) | 0 | 2 | 4 s |
| 01_task_and_model | 13 (8) | 0 | 6 | 7 s |
| 02_virtual | 14 (7) | 0 | 6 | 5 s |
| 03_emulated | 13 (6) | 0 | 3 | 26 s (live iverilog: 8 examples, 52,992 values, 0 mismatches) |
| 04_circuit | 17 (9) | 0 | 6 | 11 s (live ngspice `.dc`, `.ac`, `.tran` x 3 examples) |
| 05_rf | 14 (8) | 0 | 5 | 33 s (live sweeps, 500 examples, 5 axes) |
| 06_compare_and_llm | 19 (10) | 0 | 6 | 5 s |

Live numbers that agree with the stored run: diff-pair V_sat 52.02 mV, R^2 0.99999997; `.ac` fit max |dH| 1.3e-13, 32.00 dB/us; unit-cell layer-1 NMSE vs V 0.0238 (3 examples; stored 16: 0.0220),
calibrated vs SPICE 2.6e-7; RTL vs numpy 0 mismatches; ADC-bits sweep on 500 examples within about 1 point of the stored sweep; R live sweeps within a few points of the stored ones (subset noise).

## Findings worth knowing (not caused by the viz code)

1. **C's default loss compensation is not the best option at 32 dB/us.** `params/circuit_diagnostics.json` (calibrated model, full set) gives accuracy 0.804 with no compensation and 0.805 with an ideal
   per-tap equaliser, against 0.663 for the tier's default (one gain trim per layer). So part of the headline C gap (0.663) is a modelling choice, not only the line loss. Shown in notebook 04; belongs to the
   circuit-tier owner to revisit. At 0 dB/us all modes give about 0.863.
2. **The saved `llm/card.txt` of `modrec-w32-full` is 4494 characters (about 1870 tokens by `card.approx_tokens`)**, above `card.MAX_CHARS` = 3500 / the 1.5k-token budget; notebook 06 prints this instead of hiding it.
3. The saved Qwen commentary (`modrec-w32-qwen-pg`) has `numbers_verified: false` (the "6-bit ADC" claim) and a caveat "R is a FALLBACK" although R is `ok`; figure 8 and notebook 06 show both (the second via the independent audit).
   The commentary does not mention the 32 dB/us operating point, which the card also omits.
4. `out/LATEST` points at a NARMA10 run, so "latest run" is not the modrec headline.

## Not done / limits

- Figure 3's eye compares different pulse shapes and tap models (stated on the figure); it is a visualisation of two channel models, not a measurement.
- The 4 examples of figure 4 and the example of figure 2 are fixed choices (documented), not averages; use `figures.render(..., examples=[...])` / `example=` to look at others.
- No Postgres or LLM run was repeated here; notebook 06 re-runs Qwen only with `RUN_LLM=1`.
- The notebooks' Markdown commentary describes the headline run; with `DLNN_RUN=modrec-w8-full` the numbers shown are the width-8 ones but some prose still quotes width 32.
