"""Smoke tests for dlnn.viz (figures 1-8). They use the finished headline run out/modrec-w32-full and skip if it is absent."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dlnn.viz import figures
from dlnn.viz.common import Run

RUN = Path(__file__).resolve().parents[1] / "out" / "modrec-w32-full"
QWEN_RUN = RUN.parent / "modrec-w32-qwen-pg"
need_run = pytest.mark.skipif(not (RUN / "compare.json").exists(), reason="out/modrec-w32-full not present")


@pytest.fixture(scope="module")
def run():
    return Run(RUN)


@need_run
@pytest.mark.parametrize("name", list(figures.BUILDERS))
def test_each_figure_writes_png_and_svg(name, tmp_path, run):
    kw = {"allow_spice": False} if name == "fig3_channel" else {}      # no ngspice call in the smoke test
    paths = figures.render(name, run, "light", outdir=tmp_path, **kw)
    assert {p.suffix for p in paths} == {".png", ".svg"}
    for p in paths:
        assert p.exists() and p.stat().st_size > 2000, p
        assert p.name == f"{name}_light{p.suffix}"


@need_run
def test_dark_variant_and_transparent_background(tmp_path, run):
    from PIL import Image
    p = [q for q in figures.render("fig7_agreement", run, "dark", outdir=tmp_path) if q.suffix == ".png"][0]
    im = Image.open(p).convert("RGBA")
    assert im.getpixel((0, 0))[3] == 0          # transparent corner
    assert "dark" in p.name


@need_run
def test_fig4_titles_use_the_true_label_not_vs_prediction(run):
    fig = figures.build("fig4_outputs", run, "light")
    labels, classes = run.labels, run.classes
    titles = [a.get_title(loc="left") or a.get_title() for a in fig.axes if a.get_title(loc="left") or a.get_title()]
    ex = figures.pick_examples(run, 4)
    assert len(ex) == 4
    for e in ex:
        assert any(f"example {e}: true {classes[int(labels[e])]}" in t for t in titles), (e, titles)
    # the figure must include at least one example where V itself is wrong (so "label" != V's prediction is visible)
    assert any(int(run.res["V"].arrays["pred"][e]) != int(labels[e]) for e in ex)
    import matplotlib.pyplot as plt
    plt.close(fig)


@need_run
def test_fig5_has_every_sweep_axis_and_marks_the_headline_loss(run):
    fig = figures.build("fig5_sweeps", run, "light")
    titles = {a.get_title(loc="left") for a in fig.axes}
    assert {"snr_db", "adc_bits", "loss_db_per_us", "isolation_db", "tap_amp_err_db"} <= titles
    txt = " ".join(t.get_text() for a in fig.axes for t in a.texts)
    assert "32 dB/us" in txt          # the cascade operating point is annotated, not hidden
    import matplotlib.pyplot as plt
    plt.close(fig)


@need_run
def test_agreement_matrix_matches_compare_json(run):
    fig = figures.build("fig7_agreement", run, "light")
    cmp = run.cmp
    vals = {t.get_text() for a in fig.axes for t in a.texts}
    for t in ("E", "C", "R"):
        assert f"{cmp['metric_matrix']['top1_agreement_vs_virtual'][t]:.3f}" in vals
    import matplotlib.pyplot as plt
    plt.close(fig)


@need_run
def test_fig8_labels_template_and_flags_unverified(run):
    import matplotlib.pyplot as plt
    tmpl = json.loads((RUN / "llm" / "commentary.json").read_text())
    if tmpl["source"] == "template":
        fig = figures.build("fig8_llm", run, "light", commentary=tmpl)
        txt = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "TEMPLATE" in txt and "MODEL OUTPUT" not in txt
        plt.close(fig)
    if QWEN_RUN.joinpath("llm", "commentary.json").exists():
        com = json.loads((QWEN_RUN / "llm" / "commentary.json").read_text())
        fig = figures.build("fig8_llm", run, "light", commentary=com)
        txt = " ".join(t.get_text() for t in fig.axes[0].texts)
        assert "MODEL OUTPUT" in txt and "TEMPLATE (fixed" not in txt
        if not com.get("numbers_verified", True):
            assert "UNVERIFIED" in txt
        plt.close(fig)


def test_audit_catches_a_contradicting_caveat():
    com = {"summary": "x", "caveats": ["R is a FALLBACK (scikit-rf), not the real RF simulation"], "findings": []}
    assert figures.audit_commentary(com, {"R": "ok", "C": "ok"})
    assert not figures.audit_commentary(com, {"R": "fallback"})


@need_run
def test_make_all_never_raises_and_reports_errors(tmp_path):
    out = figures.make_all(RUN, themes=("light",), outdir=tmp_path, names=["fig1_network", "fig6_confusion"])
    assert len(out) == 4 and figures.make_all.errors == {}
    bad = figures.make_all(tmp_path / "does-not-exist")
    assert bad == [] and "run" in figures.make_all.errors


def test_make_all_on_a_small_regression_run(tmp_path):
    """The pipeline calls make_all for narma10 too: the classification-only figures are skipped, the rest render."""
    from dlnn import pipeline
    run_dir = pipeline.run_pipeline("narma10", tiers="V,E", out=tmp_path, run_id="viz", quiet=True)
    names = {p.name for p in (run_dir / "figures").glob("*.png")}
    assert {"fig4_outputs_light.png", "fig5_sweeps_dark.png", "fig1_network_light.png"} <= names
    assert not any(n.startswith(("fig6", "fig7", "fig2")) for n in names)
    assert np.all([(run_dir / "figures" / n).stat().st_size > 1000 for n in names])


def test_make_all_on_a_modrec_run_with_subset_tiers(tmp_path):
    """C and R on a stratified subset (fewer examples than V/E): agreement and confusion figures use the paired examples."""
    from dlnn import pipeline
    if not (Path(__file__).resolve().parents[1] / "models" / "tdlnet_modrec_w32_ternary.npz").exists():
        pytest.skip("headline model not present")
    run_dir = pipeline.run_pipeline("modrec", tiers="V,E,C,R", cascade=True, out=tmp_path, run_id="vizsub", quiet=True,
                                    n_test=80, cr_subset=40, tier_cfg={"rtl": False, "c_use_spice": False})
    names = {p.name for p in (run_dir / "figures").glob("*_light.png")}
    figures.make_all.errors = {}
    figures.make_all(run_dir, themes=("light",), outdir=tmp_path / "again")
    assert figures.make_all.errors == {}, figures.make_all.errors
    assert {"fig6_confusion_light.png", "fig7_agreement_light.png", "fig3_channel_light.png"} <= names
