"""Shared pieces of the notebook sources (used by the _build_*.py scripts; not needed to RUN the notebooks)."""
import nbformat as nbf

REPO_URL = "https://github.com/<owner>/dlnn-paper"      # placeholder, set when the repo is published

COLAB = '''# Colab only: install the system tools, clone the repo, install the pinned Python requirements. Skipped everywhere else.
import sys
if 'google.colab' in sys.modules:
    import os, subprocess
    REPO_URL = "%s"      # TODO: replace with the real URL once the repo is public
    subprocess.run("apt-get -qq install -y ngspice iverilog", shell=True, check=True)
    if not os.path.isdir("/content/dlnn-paper"):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/dlnn-paper"], check=True)
    os.chdir("/content/dlnn-paper")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], check=True)''' % REPO_URL

SETUP = '''import os, sys, time, json
from pathlib import Path
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "requirements.txt").exists() and (p / "dlnn").is_dir())
os.chdir(ROOT); sys.path.insert(0, str(ROOT))          # run from the repo root so relative paths in the library resolve
import numpy as np
from dlnn import paper
from dlnn.paper import DATA, EXPECTED, WORK
T0 = time.time()
ENV = paper.banner("%s")'''

END = '''paper.record_runtime("%s", time.time() - T0)
print(f"{time.time() - T0:.0f} s wall")'''


def build(path, name, cells):
    n = nbf.v4.new_notebook()
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.metadata["language_info"] = {"name": "python"}
    out = []
    for kind, src in cells:
        src = src.strip("\n")
        out.append(nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src))
    n.cells = out
    nbf.write(n, path)
