# Docs

| Read | When you want |
|---|---|
| [story.md](story.md) | how the repository got here: the mainline and the waypoints that changed it |
| [architecture.md](architecture.md) | the system on one page: both halves, the benches, data flow, status |
| [design/delay-tensor.md](design/delay-tensor.md) | the reservoir: model, modules, invariants, results |
| [design/benches.md](design/benches.md) | the ngspice and scikit-rf benches, and line extraction |
| [design/dlnn.md](design/dlnn.md) | the ternary delay-line network, its fidelity tiers and the board pipeline |
| [timeline.json](timeline.json) | the waypoints as data |
| [../AGENTS.md](../AGENTS.md) | working in the repository as an agent: commands, invariants, where truth lives |
| [../llms.txt](../llms.txt) | a compact index for language models |

Package-level detail stays with each package: [`packages/delay-tensor/docs`](../packages/delay-tensor/docs), [`packages/dlnn/docs`](../packages/dlnn/docs), [`packages/dlnn-paper/docs`](../packages/dlnn-paper/docs).

The figures in [assets/](assets) are drawn from the repository's own run outputs and benches, in the same plotter style as the published write-up ("Crosstalk", d/rksci research, November 2026). Nothing in this repository has been fabricated or measured on hardware; every page says which of simulated, emulated, designed or pending applies.
