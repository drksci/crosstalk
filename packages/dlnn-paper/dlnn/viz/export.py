"""Export the final figures (1-8, PNG + SVG, light + dark) of finished runs into docs/figures/.

    .venv/bin/python -m dlnn.viz.export                 # headline run -> docs/figures/, width-8 secondary -> docs/figures/w8/
    .venv/bin/python -m dlnn.viz.export --run modrec-w32-full --llm-run modrec-w32-qwen-pg

Figures are also written to ``<run>/figures/`` (transparent) unless ``--no-run-dir``; the docs copies are opaque (white / dark surface). Figure 8 in docs/figures is the real Qwen commentary (``--llm-run``);
``fig8_llm_template_*`` shows the template commentary of the headline run so both labels are visible.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import figures
from .common import Run

ROOT = Path(__file__).resolve().parents[2]


def export(run_name: str = "modrec-w32-full", llm_run: str | None = "modrec-w32-qwen-pg", docs: Path | None = None,
           secondary: str | None = "modrec-w8-full", write_run_dir: bool = True) -> list[Path]:
    docs = Path(docs or ROOT / "docs" / "figures")
    docs.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    for name, dest in ((run_name, docs), (secondary, docs / "w8")):
        if not name or not (ROOT / "out" / name / "compare.json").exists():
            continue
        run = Run(ROOT / "out" / name)
        if write_run_dir:
            figures.make_all(run.dir)                   # -> out/<run>/figures (transparent, as the pipeline writes them)
        out += figures.make_all(run.dir, outdir=dest, opaque=True)   # -> docs: opaque white / dark-surface background
        for k, v in figures.make_all.errors.items():
            print("ERROR", name, k, v)
        com_main = json.loads((run.dir / "llm" / "commentary.json").read_text())
        if dest == docs:
            lr = ROOT / "out" / llm_run / "llm" / "commentary.json" if llm_run else None
            if lr is not None and lr.exists():
                com = json.loads(lr.read_text())
                if com_main.get("source") == "template":      # keep both visible, clearly named
                    for th in ("light", "dark"):
                        out += figures.render("fig8_llm", run, th, outdir=dest, commentary=com, opaque=True)
                        out += figures.render("fig8_llm", run, th, outdir=dest / "template", commentary=com_main, opaque=True)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="modrec-w32-full")
    ap.add_argument("--llm-run", default="modrec-w32-qwen-pg")
    ap.add_argument("--secondary", default="modrec-w8-full")
    ap.add_argument("--no-run-dir", action="store_true", help="do not (re)write out/<run>/figures; only docs/figures")
    a = ap.parse_args()
    paths = export(a.run, a.llm_run, secondary=a.secondary, write_run_dir=not a.no_run_dir)
    print(f"{len(paths)} files")
