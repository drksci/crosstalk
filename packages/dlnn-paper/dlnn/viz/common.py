"""Shared plumbing for the figures and notebooks: palette, matplotlib theme, run loader, saving helpers.

Palette: dataviz reference categorical slots 1-4 for the tiers (V blue, E orange, C aqua, R yellow), validated with
`validate_palette.js` (light: all hard gates pass, contrast WARN on aqua/yellow -> relief = direct labels, markers and line
styles on every series; dark: all gates pass). Text always wears the ink tokens, never the series colour.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .. import contract  # noqa: E402

STYLES = Path(__file__).parent / "styles"
TIERS = ("V", "E", "C", "R")
TIER_COL = {"light": {"V": "#2a78d6", "E": "#eb6834", "C": "#1baf7a", "R": "#eda100"},
            "dark": {"V": "#3987e5", "E": "#d95926", "C": "#199e70", "R": "#c98500"}}
MARKERS = {"V": "o", "E": "s", "C": "^", "R": "D"}
LINESTYLES = {"V": "-", "E": (0, (1.5, 1.5)), "C": "--", "R": "-."}
TIER_SHORT = {"V": "V ideal math", "E": "E fixed-point", "C": "C circuit", "R": "R RF channel"}
TIER_LONG = {"V": "V  virtual (ideal numpy)", "E": "E  emulated (fixed-point; RTL on a subset)",
             "C": "C  circuit (ngspice cell + calibrated model)", "R": "R  RF (scikit-rf channel model)"}
INK = {"light": dict(text="#0b0b0b", sec="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7", mid="#f0efec",
                     surface="#fcfcfb", pos="#2a78d6", neg="#e34948", warn="#a35f00"),
       "dark": dict(text="#f0efec", sec="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835", mid="#383835",
                    surface="#1a1a19", pos="#3987e5", neg="#e66767", warn="#e0a030")}
SEQ = {"light": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
       "dark": ["#22262b", "#184f95", "#256abf", "#6da7ec", "#cde2fb"]}
AXIS_LABEL = {"snr_db": "SNR (dB)", "adc_bits": "ADC bits", "loss_db_per_us": "line loss (dB / us)",
              "tap_amp_err_db": "tap amplitude error, sigma (dB)", "isolation_db": "coupler isolation (dB)"}
V_P_CM_NS = 19.8     # RG-58 phase velocity
T_S_NS = 10.0        # symbol period
LOSS_RG58 = 32.0     # dB per microsecond at 100 MHz


def ctx(theme: str):
    """matplotlib style context for a theme (transparent background, ink tokens, thin grid)."""
    k = INK[theme]
    over = {"text.color": k["text"], "axes.labelcolor": k["text"], "axes.edgecolor": k["axis"],
            "xtick.color": k["sec"], "ytick.color": k["sec"], "axes.titlecolor": k["text"],
            "grid.color": k["grid"], "legend.labelcolor": k["text"], "font.size": 8.5, "axes.titlesize": 9.5,
            "axes.labelsize": 8.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
            "lines.linewidth": 1.6, "lines.markersize": 5, "axes.titleweight": "bold", "axes.titlelocation": "left",
            "font.family": "DejaVu Sans"}
    return plt.style.context([str(STYLES / f"{theme}.mplstyle"), over])


def tier_label(tier: str, status: str) -> str:
    return TIER_SHORT[tier] if status == "ok" else f"{TIER_SHORT[tier]} ({status.upper()}: not the real tool)"


def save_fig(fig, outdir: Path, name: str, theme: str, opaque: bool = False) -> list[Path]:
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "svg"):
        p = outdir / f"{name}_{theme}.{ext}"
        if opaque:      # docs export: draw the page colour into the file (white / dark surface) so it reads on any page
            fig.savefig(p, transparent=False, facecolor="#ffffff" if theme == "light" else INK["dark"]["surface"], edgecolor="none",
                        bbox_inches="tight", pad_inches=0.12)
        else:
            fig.savefig(p, transparent=True, bbox_inches="tight", pad_inches=0.12)
        paths.append(p)
    plt.close(fig)
    return paths


def end_labels(ax, items, min_gap_pt: float = 8.5, dx: float = 5.0, fontsize: float = 7.5):
    """Direct labels at the right end of series without overlap. items: [(x, y, text)] in data coordinates.
    Labels are ink-coloured text (series identity stays in the mark)."""
    fig = ax.figure
    fig.canvas.draw()
    inv = fig.dpi / 72.0
    pts = []
    for x, y, t in items:
        try:
            _, py = ax.transData.transform((x, y))
        except Exception:  # noqa: BLE001
            continue
        pts.append([py / inv, x, y, t])
    pts.sort(key=lambda r: r[0])
    for i in range(1, len(pts)):
        if pts[i][0] - pts[i - 1][0] < min_gap_pt:
            pts[i][0] = pts[i - 1][0] + min_gap_pt
    for py, x, y, t in pts:
        _, py0 = ax.transData.transform((x, y))
        ax.annotate(t, (x, y), xytext=(dx, py - py0 / inv), textcoords="offset points", fontsize=fontsize,
                    color=matplotlib.rcParams["text.color"], va="center", annotation_clip=False)


def wrap(text: str, width: int) -> list[str]:
    import textwrap
    return textwrap.wrap(text, width=width, break_long_words=True) or [""]


class Run:
    """Lazy view of one run directory (compare.json, tier results, labels, model, circuit extraction)."""

    def __init__(self, run_dir: str | Path):
        self.dir = Path(run_dir)
        self.cmp = json.loads((self.dir / "compare.json").read_text())
        self.task = self.cmp["task"]
        self.tiers = [t for t in TIERS if t in self.cmp["tiers"]]
        self.res: dict[str, contract.TierResult] = {t: contract.load(self.dir / "tiers" / f"{t}.json")
                                                    for t in self.tiers}
        self._cache: dict = {}

    # ------------------------------------------------------------------------------------------ small accessors
    @property
    def status(self) -> dict[str, str]:
        return {t: self.cmp["tier_info"][t]["status"] for t in self.tiers}

    @property
    def cascade(self) -> bool:
        return bool(self.cmp.get("run_meta", {}).get("cascade", False))

    @property
    def labels(self) -> np.ndarray:
        return np.load(self.dir / "labels.npy")

    @property
    def snr(self) -> np.ndarray:
        return np.load(self.dir / "snr_test.npy")

    @property
    def classes(self) -> list[str]:
        from ..data import modrec
        return list(modrec.CLASSES)

    @property
    def model(self):
        if "model" not in self._cache:
            from ..model.npz import load_model
            self._cache["model"] = load_model(self.dir / "model.npz")
        return self._cache["model"]

    @property
    def params(self) -> dict | None:
        p = self.dir / "params" / "circuit_extracted.json"
        if "params" not in self._cache:
            self._cache["params"] = json.loads(p.read_text()) if p.exists() else None
        return self._cache["params"]

    @property
    def x_test(self) -> np.ndarray:
        """The test inputs (regenerated deterministically from the generator, not stored in the run)."""
        if "x_test" not in self._cache:
            if self.task == "modrec":
                from ..data import modrec
                n = self.res["V"].n_examples
                self._cache["x_test"] = modrec.generate(n, modrec.SEED + 1)["x"]
            else:
                raise NotImplementedError("x_test is only regenerated for modrec")
        return self._cache["x_test"]

    def index_of(self, tier: str) -> np.ndarray:
        """Test-set indices of the examples tier `tier` ran on (all of them unless C/R ran on a stratified subset)."""
        n = len(self.res[tier].arrays["pred"]) if "pred" in self.res[tier].arrays else self.res[tier].n_examples
        sp = self.dir / "subsets.json"
        if sp.exists():
            t = json.loads(sp.read_text()).get("tiers", {}).get(tier)
            if t and len(t["indices"]) == n:
                return np.asarray(t["indices"], dtype=np.int64)
        return np.arange(n)

    def paired_preds(self, a: str, b: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predictions of tiers a and b on the examples both ran on: (pred_a, pred_b, test-set indices)."""
        ia, ib = self.index_of(a), self.index_of(b)
        common = np.intersect1d(ia, ib)
        pa, pb = self.res[a].arrays["pred"], self.res[b].arrays["pred"]
        return pa[np.searchsorted(ia, common)], pb[np.searchsorted(ib, common)], common

    def sweeps_by_axis(self, axis: str) -> dict:
        return self.cmp["sweeps"].get(axis, {})

    def preds(self) -> dict[str, np.ndarray]:
        return {t: self.res[t].arrays["pred"] for t in self.tiers if "pred" in self.res[t].arrays}

    def trace(self, tier: str, layer: int) -> np.ndarray | None:
        return self.res[tier].arrays.get(f"trace_activations.L{layer}")

    def pick_channel(self, layer: int, example: int, limit: int | None = None) -> int:
        """Output line with the largest V variance for this example (optionally among the first `limit` lines)."""
        v = self.trace("V", layer)[example]
        if limit is not None:
            v = v[:limit]
        return int(np.argmax(v.var(axis=-1)))


def conf_matrix(y: np.ndarray, p: np.ndarray, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (y, p), 1)
    return cm


def lum(hexcol: str) -> float:
    h = hexcol.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    return 0.2126 * r + 0.7152 * g + 0.0722 * b
