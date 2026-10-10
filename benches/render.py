"""Render a bench notebook to a truly static, self-contained HTML page.

nbconvert's HTML is *nearly* what we want, but not quite:

* it loads MathJax and require.js from a CDN, so the page phones home;
* blanking those URLs leaves ``<script src="">``, which browsers resolve to
  the document URL and fetch again;
* it inlines a toolbar script that a read-only page never needs.

So this drops every ``<script>`` and every external reference, then asserts
the result. The notebooks carry no math and no JS-driven widgets, so nothing
is lost — the same standard ``delay_tensor.report`` holds itself to::

    "<script" not in report

Usage::

    $HOME/micromamba/envs/delay-bench/bin/python benches/render.py benches/01-equivalence.ipynb
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Any URL a browser would go to the network for.
_EXTERNAL = re.compile(r'(?:src|href)\s*=\s*["\'](?:https?:)?//[^"\']*["\']', re.IGNORECASE)
_SCRIPT = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
# Self-referential or empty src still causes a request.
_EMPTY_SRC = re.compile(r"<script\b[^>]*\bsrc\s*=\s*[\"']\s*[\"'][^>]*>\s*</script>", re.IGNORECASE)


def render(notebook: Path) -> Path:
    """Convert ``notebook`` to HTML beside it, stripped of script and CDN."""
    if not notebook.is_file():
        raise FileNotFoundError(notebook)

    jupyter = _jupyter_binary()
    # Both exporters take the URL as a trait; empty string emits src="".
    subprocess.run(
        [
            jupyter,
            "nbconvert",
            "--to",
            "html",
            str(notebook),
            "--output-dir",
            str(notebook.parent),
            "--HTMLExporter.mathjax_url=",
            "--HTMLExporter.require_js_url=",
        ],
        check=True,
    )

    output = notebook.with_suffix(".html")
    html = output.read_text(encoding="utf-8")
    html = _SCRIPT.sub("", html)
    html = _EMPTY_SRC.sub("", html)

    _assert_static(html, output)
    output.write_text(html, encoding="utf-8")
    print(f"{output}  {len(html):,} bytes  (no script, no external refs)")
    return output


def _jupyter_binary() -> str:
    """Resolve the jupyter next to the running interpreter, not whatever PATH says."""
    candidate = Path(sys.executable).with_name("jupyter")
    if candidate.exists():
        return str(candidate)
    return "jupyter"


def _assert_static(html: str, output: Path) -> None:
    """Fail loudly rather than ship a page that reaches the network."""
    problems = []
    if "<script" in html.lower():
        problems.append("contains <script>")
    external = _EXTERNAL.findall(html)
    if external:
        problems.append(f"{len(external)} external ref(s), e.g. {external[0][:80]}")
    if problems:
        raise RuntimeError(f"{output}: " + "; ".join(problems))


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        render(Path(arg))
