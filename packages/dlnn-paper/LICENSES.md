# Licences

| what | licence | file |
|---|---|---|
| code (`dlnn/`, `notebooks/`, `hw/board/*.py`, scripts, JS in `docs/artifacts/assets/js/`) | MIT | `LICENSE` |
| hardware (`hw/board/` generators' outputs: schematics/netlists in `hw/board/tile/`, `hw/spice/` models, `docs/artifacts/board/`) | CERN-OHL-P-2.0 (permissive) | `LICENSE-HARDWARE` |
| text (`docs/`, the article in `docs/artifacts/article/`, figures, notebook prose) | CC BY 4.0 | `LICENSE-TEXT` |

Vendored third-party files under `docs/artifacts/vendor/` (d3, three.js, KiCanvas, EEcircuit/ngspice-wasm) keep their own licences (see the files in each folder).
Datasheet-derived part values are facts about third-party parts; no vendor model files are redistributed (the TI TLV9062 macro is not included).
