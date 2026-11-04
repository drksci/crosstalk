"""Tiny helpers shared by the dlnn-paper notebooks (paths, tool versions, reference comparison, runtime log).

Everything here is bookkeeping; the science is in dlnn.tiers.*, dlnn.model.*, dlnn.viz.export_web.
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "artifacts" / "data"          # the files the article reads (the notebooks write here)
EXPECTED = ROOT / "expected"                          # reference snapshot of those files + model hashes
WORK = ROOT / "work"                                  # scratch (netlists, ngspice raw files, lite models); safe to delete
LICENSE_NOTE = "Simulation of a design that is not built. Nothing here is a measurement."


def sha256_file(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_json(path, obj, indent=None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=indent, separators=None if indent else (",", ":")))
    return path


def read_json(path):
    return json.loads(Path(path).read_text())


def tool_version(name: str) -> str | None:
    """First line of `<tool> --version` (None when the tool is not on PATH)."""
    exe = shutil.which(name)
    if exe is None:
        return None
    try:
        if name == "ngspice":      # ngspice prints its banner on stdout of a batch run
            r = subprocess.run([exe, "-v"], capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
            txt = (r.stdout + r.stderr).strip().splitlines()
            return next((l.strip("* ").strip() for l in txt if "ngspice-" in l), txt[0] if txt else "ngspice (version unknown)")
        flag = "-V" if name == "iverilog" else "--version"
        r = subprocess.run([exe, flag], capture_output=True, text=True, timeout=20, stdin=subprocess.DEVNULL)
        return (r.stdout or r.stderr).strip().splitlines()[0]
    except Exception as e:  # noqa: BLE001
        return f"{name} (version query failed: {type(e).__name__})"


def banner(title: str) -> dict:
    """Print the environment (so a reader of the executed notebook sees what produced the numbers)."""
    import numpy
    env = {"python": platform.python_version(), "numpy": numpy.__version__, "platform": platform.platform(),
           "ngspice": tool_version("ngspice"), "iverilog": tool_version("iverilog"), "node": tool_version("node"),
           "colab": "google.colab" in sys.modules}
    print(f"{title}\n  " + "\n  ".join(f"{k}: {v}" for k, v in env.items()))
    print(f"  {LICENSE_NOTE}")
    return env


def check(rows: list, name: str, got: float, want: float, tol: float, unit: str = "", note: str = "") -> bool:
    """Append a comparison row (|got - want| <= tol) and return pass/fail."""
    ok = abs(float(got) - float(want)) <= tol
    rows.append({"check": name, "got": float(got), "reference": float(want), "tol": tol, "unit": unit, "result": "PASS" if ok else "FAIL", "note": note})
    return ok


def show_rows(rows: list):
    import pandas as pd
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_colwidth", 70, "display.width", 200, "display.float_format", "{:.6g}".format):
        print(df.to_string(index=False))
    return df


class Timer:
    def __init__(self):
        self.t0 = time.time()

    def lap(self, label: str = "") -> float:
        dt = time.time() - self.t0
        print(f"[{label}] {dt:.1f} s since notebook start")
        return dt


def record_runtime(notebook: str, seconds: float, extra: dict | None = None) -> None:
    """Append this run to docs/artifacts/data/runtimes.json (what the README quotes)."""
    p = DATA / "runtimes.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur[notebook] = {"seconds": round(seconds, 1), "ngspice": tool_version("ngspice"), "python": platform.python_version(),
                     "cpus": __import__("os").cpu_count(), "colab": "google.colab" in sys.modules,
                     "finished": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}
    write_json(p, cur, indent=1)
