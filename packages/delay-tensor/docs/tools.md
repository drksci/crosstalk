# Tooling notes

Researched 2026-10-10 from
[github.com/topics/electronics](https://github.com/topics/electronics?o=desc&s=stars),
[github.com/topics/circuit-simulation](https://github.com/topics/circuit-simulation?o=desc&s=stars),
and the linked repositories. Stars and "last updated" are as GitHub showed on
that date.

Selection criteria, in order: **light weight** (no heavyweight local install or
runtime for what the job actually needs), **actively developed** (commits in
the last few months), **popular** (a community that answers issues), and
**browser-native** where a reader might look at the result.

## What this package uses

| Need | Choice | Why |
| --- | --- | --- |
| Reservoir dynamics | NumPy | Vectorised loops, no autograd for a frozen system |
| Linear readout | scikit-learn (~64k★, very active) | Deterministic fit, one dependency instead of PyTorch |
| Config | PyYAML + dataclasses | Typed and validated without a schema framework |
| Charts in reports | Hand-written inline SVG | A report should be a static file with no network requests |

Dev-only: `pytest`, `ruff`. Total runtime dependency set is four packages.

## SPICE validation — built, not hypothetical

The section below was a recommendation. It has since been implemented; the
table stays as the research trail for what was considered.

| Piece | Where | Note |
| --- | --- | --- |
| Operating-point equivalence bench | `benches/01-equivalence.ipynb` | The netlist unrolls the slot axis into stages; max **\|error\| 5.274e-16** across 2,048 stages against NumPy |
| Netlist + runner | `benches/netlist.py`, `benches/spice_runner.py` | `SpiceParser(...).build_circuit()` with `simulator="ngspice-shared"` |
| Static rendering | `benches/render.py` | nbconvert with `--HTMLExporter.mathjax_url=` and `require_js_url=` emptied, then every `<script>` stripped and asserted absent |
| Bench environment | micromamba `$HOME/micromamba/envs/delay-bench` | `ngspice-41`, PySpice 1.5, jupyterlab, ipykernel, nbconvert, matplotlib |

**The licence boundary is the important part.** PySpice is GPL-3.0, so it is
never imported by `delay_tensor` — the package stays NumPy/scikit-learn and
distributable. Every PySpice import lives in `benches/spice_runner.py`, which
is a development tool and is not part of the wheel. The bench caught a real
bug while it was being written: the netlist's `tanh` gating was wrong, and the
equivalence run found it.

ngspice's v41 banner is silenced with a `logging.Filter` on logger
`PySpice.Spice.NgSpice.Shared.NgSpiceShared` rather than by redirecting
stdout, which would also swallow simulator errors.

## vjax: not runnable here

[jax/vjax](https://github.com/google/jax)-style autodiff was considered for
the readout. It **cannot be run on this machine**: the host is an Intel
x86_64 Mac (i3-1000NG4) and jaxlib publishes no wheel for that platform, so
`pip install jaxlib` fails rather than falling back to a CPU build. Any vjax
work has to happen on a remote or CUDA machine; do not add it to the local
dependency set expecting it to resolve.

This is also why the readout is scikit-learn rather than anything compiled:
`pip install` has to succeed on the machine that runs the tests.

Research trail, as GitHub showed on 2026-10-10:

| Tool | Stars | Activity | Use here |
| --- | --- | --- | --- |
| [PySpice](https://github.com/PySpice-org/PySpice) | 871 | Sep 2026 | Python bindings for ngspice/Xyce — drive simulations in-process instead of `subprocess` + log scraping |
| [scikit-rf](https://github.com/scikit-rf/scikit-rf) | 946 | Oct 2026 | S-parameters, cascaded networks, Touchstone; the natural home for a coupled-line microcell's frequency-domain check |
| [Qucs-S](https://github.com/ra3xdh/qucs_s) | 1.4k | Oct 2026 | GUI front-end over ngspice/Xyce for looking at a netlist interactively |
| ngspice itself | — | active | Not GitHub-primary (SourceForge/GitLab); install with `brew install ngspice` |

Avoid for new work: [ahkab](https://github.com/ahkab/ahkab) (380★, last
updated Apr 2024), [circuitsim](https://github.com/circuitsim/circuit-simulator)
(289★, Mar 2023). Both are quiet enough that you would inherit their bugs.

## If schematics / PCB come back

The archive hand-wrote KiCad S-expressions in Python and then needed
`kicad-cli` to render them — a bespoke generator for a format KiCad already
owns.

| Tool | Stars | Activity | Use here |
| --- | --- | --- | --- |
| [tscircuit](https://github.com/tscircuit/tscircuit) | 2.8k | Oct 10 2026, 5.5k commits | Circuit as TypeScript/React; `tsci dev` gives a live browser preview of schematic **and** PCB, and exports Gerbers. MIT. Best fit for "show the tensor to someone in a browser" |
| [atopile](https://github.com/atopile/atopile) | 4k | Jun 2026 | Code-defined boards with reuse/validation; Python |
| [skidl](https://github.com/devbisme/skidl) | 1.7k | Sep 2026 | Describe netlists in Python and emit KiCad — lightest way back into the KiCad ecosystem |
| [LibrePCB](https://github.com/LibrePCB/LibrePCB) | 3k | Oct 2026 | Full EDA suite if a real layout is needed |
| KiCad ([source mirror](https://github.com/KiCad/kicad-source-mirror)) | 3k | Oct 2026 | The reference EDA; `kicad-cli` only as an *optional* local tool, never a build dependency |

Caution: [tracespace](https://github.com/tracespace/tracespace) (953★, Gerber
→ SVG) was last updated Jan 2025.

## Web visualisation options

Ranked from "already done" to "more machinery".

1. **Inline SVG (what the reports do now).** Zero dependencies, ~56 KB per
   complete report, prints and archives perfectly. Right answer for static
   artifacts.

2. **[uPlot](https://github.com/leeoniya/uPlot)** — 10.6k★, MIT, ~50 KB
   minified, canvas time-series chart with zoom/legend/synced cursors. Use it
   when traces need panning or live streaming. The archive's alternative was
   Plotly, whose own benchmark table puts Plotly.js at 3.6 MB against uPlot's
   48 KB — and it was inlined into every report.

3. **[tscircuit](https://tscircuit.com/playground)** — in-browser schematic
   and PCB rendering (React Fiber), shareable URLs, local preview via
   `tsci dev`, embeddable. This is the option for showing the tensor coupon
   as a real board rather than as a diagram.

4. **[CircuitJS / Falstad](https://github.com/pfalstad/circuitjs1)** — 2.4k★,
   GPL-2.0, a full circuit simulator that runs in the browser (hosted at
   [falstad.com/circuit](https://www.falstad.com/circuit/)). Embeddable with an
   iframe and URL parameters (`?cct=...`, `?running=false`), so a report can
   hand the reader an LC ladder cell to poke at. Good fit for explaining the
   microcell; GPL means don't vendor it into a permissive-licensed package.

5. **[Wokwi](https://wokwi.com)** — browser simulation of real firmware
   targets (ESP32, AVR, RP2040), free for open-source projects. Relevant to
   `archive/firmware/esp32-controller` rather than to this simulation package.

## Summary of what changed and why

| Archive used | Replaced with | Reason |
| --- | --- | --- |
| `torch>=2.2,<2.3` | NumPy + scikit-learn | Frozen reservoir needs no autograd; ~2 GB saved |
| `minify-html` + `rjsmin` + inlined Plotly | Static HTML + inline SVG | Reports drop from MB to KB and render offline |
| Hand-rolled KiCad S-expression generator | dropped | Simulation-only scope; see tscircuit/skidl above |
| `ngspice` subprocess with silent skip | dropped | Same; PySpice when it returns |
| Single-seed accuracy | multi-seed mean ± std + CSV | A single seed on this dataset is not a result |
