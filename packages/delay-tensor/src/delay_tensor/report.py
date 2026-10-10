"""Run outputs: one self-contained directory per experiment run.

Everything a run needs to be understood later lives in its own folder:

    config.yaml       the exact config used
    metrics.json      aggregate + per-seed metrics, machine readable
    runs.csv          one row per seed, for a spreadsheet or pandas
    observations.txt  plain-text verdict and next move
    report.html       charts, confusion matrices, traces, config echo
"""

from __future__ import annotations

import csv
import html
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from delay_tensor.config import DelayTensorConfig, save_config
from delay_tensor.experiments.base import Summary
from delay_tensor.svg import (
    bar_chart,
    field_view,
    heatmap,
    line_chart,
    line_chart_legend,
    tensor_diagram,
)

MAX_EXAMPLE_WIRES = 4


@dataclass(frozen=True)
class RunPaths:
    directory: Path
    report: Path
    metrics: Path
    csv: Path
    observations: Path
    config: Path


def write_run(run_dir: str | Path, summary: Summary, *, command: str | None = None) -> RunPaths:
    """Write every run artifact into ``run_dir`` and return their paths."""
    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)

    save_config(summary.config, path / "config.yaml")
    metrics_path = path / "metrics.json"
    metrics_path.write_text(
        json.dumps(summary.to_metrics_dict(), indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    csv_path = _write_csv(path / "runs.csv", summary)
    observations = render_observations(summary)
    observations_path = path / "observations.txt"
    observations_path.write_text(observations + "\n", encoding="utf-8")
    report_path = path / "report.html"
    report_path.write_text(
        render_report(summary, observations=observations, command=command),
        encoding="utf-8",
    )
    return RunPaths(
        directory=path,
        report=report_path,
        metrics=metrics_path,
        csv=csv_path,
        observations=observations_path,
        config=path / "config.yaml",
    )


def render_observations(summary: Summary) -> str:
    """Plain-text verdict: what happened, how certain it is, what to do next."""
    raw, reservoir, delta = summary.raw, summary.reservoir, summary.delta
    verdict, next_move = _verdict(delta.mean, delta.std, summary.chance_level)
    tensor = summary.config.tensor

    lines = [
        f"Experiment: {summary.experiment} across {raw.n} seed(s), config {summary.config.name}.",
        f"Chance level: {summary.chance_level:.3f} ({len(summary.class_names)} classes).",
        f"Raw baseline accuracy: {raw.format()}.",
        f"Reservoir accuracy: {reservoir.format()}.",
        f"Accuracy delta: {delta.mean:+.3f} ± {delta.std:.3f}.",
        f"Verdict: {verdict}",
        f"Next move: {next_move}",
        f"Feature surface: {tensor.wires} wires x {tensor.slots} slots = {summary.feature_count} "
        f"virtual nodes (readout.traces={summary.config.readout.traces}).",
        f"Reservoir health: {summary.health.get('status', 'unknown')} "
        f"(mean |state| {summary.health.get('mean_abs_state', 0.0):.3f}, "
        f"saturation {summary.health.get('saturation_fraction', 0.0):.1%}).",
        "Reproduce with the command line shown in report.html; every seed is listed in runs.csv.",
    ]
    return "\n".join(lines)


def render_report(summary: Summary, *, observations: str, command: str | None = None) -> str:
    """Render the full static HTML report."""
    config = summary.config
    generated = datetime.now().astimezone().isoformat(timespec="seconds")

    sections = [
        _header(summary, generated, command),
        _outcomes(summary),
        _chart_section(summary),
        _seed_table(summary),
        _confusion_section(summary),
        _examples_section(summary),
        _diagram_section(config),
        _observations_section(observations),
        _config_section(config),
    ]
    return _PAGE.format(
        title=f"Delay Tensor · {summary.experiment} · {config.name}",
        body="".join(sections),
        generated=html.escape(generated),
    )


def _header(summary: Summary, generated: str, command: str | None) -> str:
    meta = [
        f"<span><b>experiment</b> {html.escape(summary.experiment)}</span>",
        f"<span><b>config</b> {html.escape(summary.config.name)}</span>",
        f"<span><b>seeds</b> {len(summary.records)}</span>",
        f"<span><b>generated</b> {html.escape(generated)}</span>",
    ]
    if command:
        meta.append(f"<span><b>command</b> <code>{html.escape(command)}</code></span>")
    return (
        "<header>"
        "<h1>Delay Tensor — simulated experiment</h1>"
        '<p class="lede">A frozen coupled-line reservoir and a raw-waveform baseline, '
        "scored by the same linear readout.</p>"
        f'<p class="meta">{"".join(meta)}</p>'
        "</header>"
    )


def _outcomes(summary: Summary) -> str:
    cards = [
        ("Raw baseline", summary.raw.format(), "waveforms straight into the readout"),
        ("Reservoir", summary.reservoir.format(), "waveforms through the tensor first"),
        (
            "Delta",
            f"{summary.delta.mean:+.3f}",
            f"± {summary.delta.std:.3f} across {summary.raw.n} seed(s)",
        ),
        ("Chance", f"{summary.chance_level:.3f}", f"{summary.feature_count} virtual nodes"),
    ]
    body = "".join(
        f'<div class="card"><div class="card-label">{html.escape(label)}</div>'
        f'<div class="card-value">{html.escape(value)}</div>'
        f'<div class="card-note">{html.escape(note)}</div></div>'
        for label, value, note in cards
    )
    return f'<section class="cards">{body}</section>'


def _chart_section(summary: Summary) -> str:
    chart = bar_chart(
        labels=["raw baseline", "reservoir"],
        values=[summary.raw.mean, summary.reservoir.mean],
        errors=[summary.raw.std, summary.reservoir.std],
        ymax=1.0,
        aria_label="mean accuracy with standard deviation whiskers",
    )
    return (
        "<section><h2>Accuracy</h2>"
        f"{chart}"
        '<p class="caption">Bars show the mean over seeds; whiskers show ±1 standard '
        "deviation. Both paths use the same readout class and hyperparameters.</p>"
        "</section>"
    )


def _seed_table(summary: Summary) -> str:
    rows = "".join(
        f"<tr><td>{record.seed}</td><td>{record.raw_accuracy:.3f}</td>"
        f"<td>{record.reservoir_accuracy:.3f}</td><td>{record.delta:+.3f}</td>"
        f"<td>{record.train_size}</td><td>{record.test_size}</td></tr>"
        for record in summary.records
    )
    return (
        "<section><h2>Per-seed results</h2>"
        "<table><thead><tr><th>seed</th><th>raw</th><th>reservoir</th><th>delta</th>"
        "<th>train</th><th>test</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
        '<p class="caption">Full metrics, including row-normalised confusion matrices, '
        "are in <code>metrics.json</code>.</p>"
        "</section>"
    )


def _confusion_section(summary: Summary) -> str:
    names = [html.escape(name) for name in summary.class_names]
    panels = [
        ("Raw baseline (count over seeds)", summary.confusion_raw),
        ("Reservoir (count over seeds)", summary.confusion_reservoir),
    ]
    body = "".join(
        f"<figure><figcaption>{title}</figcaption>{heatmap(matrix, names, names)}</figure>"
        for title, matrix in panels
    )
    return (
        "<section><h2>Confusion matrices</h2>"
        '<div class="split">' + body + "</div>"
        '<p class="caption">Rows are the true class, columns the predicted class; '
        "cell shading is the share of each row.</p>"
        "</section>"
    )


def _examples_section(summary: Summary) -> str:
    if not summary.examples:
        return ""
    tensors = []
    for example in summary.examples:
        class_name = summary.class_names[example.true_label]
        raw_ok = example.predicted_raw == example.true_label
        res_ok = example.predicted_reservoir == example.true_label
        raw_chart = line_chart(
            [("input", example.waveform)],
            y_label="amplitude",
            aria_label=f"input waveform for class {class_name}",
        )
        reservoir_series = [
            (f"W{wire}", example.traces[wire]) for wire in range(example.traces.shape[0])
        ]
        res_chart = line_chart(
            reservoir_series,
            y_label="state",
            aria_label=f"reservoir traces for class {class_name}",
        )
        tensors.append(
            '<div class="example">'
            f"<h3>{html.escape(class_name)}</h3>"
            f'<p class="meta">raw readout: {_verdict_chip(raw_ok)} · '
            f"reservoir readout: {_verdict_chip(res_ok)}</p>"
            '<p class="caption">input waveform</p>'
            + raw_chart
            + line_chart_legend(reservoir_series)
            + '<p class="caption">reservoir traces</p>'
            + res_chart
            + '<p class="caption">EM field — all wires at once, wavefront sweeping</p>'
            + field_view(
                example.traces,
                aria_label=f"electromagnetic field view for class {class_name}",
            )
            + "</div>"
        )
    return (
        "<section><h2>Held-out examples</h2>"
        '<div class="examples">' + "".join(tensors) + "</div>"
        '<p class="caption">One test waveform per class from the first seed, with the '
        "reservoir state it produced. The field panel shades every wire and slot at once — "
        "teal positive, orange negative — and a wavefront sweeps it on a loop. The loop is "
        "SMIL, not JavaScript; it stops under <code>prefers-reduced-motion</code>.</p>"
        "</section>"
    )


def _verdict_chip(ok: bool) -> str:
    text = "correct" if ok else "wrong"
    css = "chip chip-ok" if ok else "chip chip-bad"
    return f'<span class="{css}">{text}</span>'


def _diagram_section(config: DelayTensorConfig) -> str:
    return (
        "<section><h2>Tensor</h2>"
        + tensor_diagram(config.tensor.wires, config.tensor.slots)
        + '<p class="caption">Teal rails are wires, orange ticks are the nearest-neighbour '
        "coupling windows, circles are edge probes.</p>"
        "</section>"
    )


def _observations_section(observations: str) -> str:
    items = "".join(f"<li>{html.escape(line)}</li>" for line in observations.splitlines())
    return f'<section><h2>Observations</h2><ul class="observations">{items}</ul></section>'


def _config_section(config: DelayTensorConfig) -> str:
    body = yaml.safe_dump(config.to_dict(), sort_keys=False)
    return f"<section><h2>Config</h2><pre>{html.escape(body)}</pre></section>"


def _write_csv(path: Path, summary: Summary) -> Path:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["seed", "raw_accuracy", "reservoir_accuracy", "delta", "train_size", "test_size"]
        )
        for record in summary.records:
            writer.writerow(
                [
                    record.seed,
                    f"{record.raw_accuracy:.6f}",
                    f"{record.reservoir_accuracy:.6f}",
                    f"{record.delta:.6f}",
                    record.train_size,
                    record.test_size,
                ]
            )
    return path


def _verdict(delta_mean: float, delta_std: float, chance: float) -> tuple[str, str]:
    """Interpret the delta in words, taking the seed spread seriously."""
    if delta_mean > 0 and delta_mean <= delta_std:
        return (
            "The reservoir edge is inside run-to-run noise; nothing is established yet.",
            "Increase seeds or tighten the split before changing the tensor.",
        )
    if delta_mean >= 0.05:
        return (
            "Reservoir features cleared the initial lift threshold.",
            "Repeat across configs and task variants, then test robustness to drift and noise.",
        )
    if delta_mean > 0:
        return (
            "Reservoir features helped, but the margin is modest.",
            "Compare more seeds and tensor variants before treating the lift as real.",
        )
    if abs(delta_mean) < 0.01:
        return (
            "Reservoir and raw readouts are effectively tied.",
            f"Inspect separability (chance is {chance:.3f}) and try a harder task such as "
            "temporal XOR, or widen coupling.",
        )
    return (
        "Raw waveform features beat the current reservoir path.",
        "Tune coupling strength, readout taps, and task difficulty before drawing "
        "hardware conclusions.",
    )


_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --paper: #ffffff;
    --ink: #111827;
    --muted: #6b7280;
    --rule: #111827;
    --hair: #d1d5db;
    --quiet: #f9fafb;
    --ok: #0f766e;
    --bad: #b91c1c;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    margin: 0;
    background: var(--paper);
    color: var(--ink);
    font: 15px/1.5 "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  }}
  .shell {{ width: min(980px, calc(100vw - 32px)); margin: 0 auto; padding: 36px 0 64px; }}
  header {{ border-top: 3px double var(--rule); padding-top: 14px; margin-bottom: 26px; }}
  h1 {{ font-size: 26px; margin: 6px 0 4px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 18px; margin: 34px 0 10px; border-bottom: 1px solid var(--hair); padding-bottom: 4px; }}
  h3 {{ font-size: 14px; margin: 0 0 4px; text-transform: uppercase; letter-spacing: 0.06em; }}
  .lede {{ margin: 0 0 8px; color: var(--muted); font-style: italic; }}
  .meta {{ color: var(--muted); font-size: 12.5px; display: flex; flex-wrap: wrap; gap: 14px; margin: 6px 0 0; }}
  .meta b {{ color: var(--ink); font-weight: 600; }}
  code, pre {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12.5px; }}
  pre {{ background: var(--quiet); border: 1px solid var(--hair); padding: 12px 14px; overflow-x: auto; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }}
  .card {{ border: 1px solid var(--hair); border-radius: 6px; padding: 12px 14px; background: var(--quiet); }}
  .card-label {{ font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; color: var(--muted); }}
  .card-value {{ font-size: 26px; margin: 4px 0 2px; }}
  .card-note {{ font-size: 12.5px; color: var(--muted); }}
  table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; }}
  th, td {{ border-bottom: 1px solid var(--hair); padding: 7px 10px; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ background: var(--quiet); font-weight: 600; }}
  .split {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }}
  figure {{ margin: 0; }}
  figcaption {{ font-size: 12.5px; color: var(--muted); margin-bottom: 6px; }}
  .examples {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }}
  .example {{ border: 1px solid var(--hair); border-radius: 6px; padding: 12px; }}
  .chart {{ width: 100%; height: auto; border: 1px solid var(--hair); border-radius: 4px; }}
  .caption {{ font-size: 12.5px; color: var(--muted); margin: 6px 0 0; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 10px; font-size: 11.5px; color: var(--muted); margin-top: 6px; }}
  .key {{ display: inline-flex; align-items: center; gap: 5px; }}
  .swatch {{ width: 10px; height: 10px; border-radius: 2px; display: inline-block; }}
  .observations {{ margin: 0; padding-left: 18px; }}
  .observations li {{ margin-bottom: 5px; }}
  .chip {{ font-size: 11px; padding: 1px 7px; border-radius: 999px; border: 1px solid currentColor; }}
  .chip-ok {{ color: var(--ok); }}
  .chip-bad {{ color: var(--bad); }}
  footer {{ margin-top: 40px; border-top: 1px solid var(--hair); padding-top: 10px; color: var(--muted); font-size: 12.5px; }}
</style>
</head>
<body>
<div class="shell">
{body}
<footer>Delay Tensor simulated experiment · generated {generated} · static HTML, no external assets.</footer>
</div>
</body>
</html>
"""
