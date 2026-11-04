"""Minimal inline-SVG charts for the run report.

No plotting library and no bundled JavaScript: a run report should stay a
small static file that opens anywhere, including from a USB stick or an
archived CI artifact. The archive version inlined a minified Plotly build and
produced a multi-megabyte page for four charts.
"""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from xml.sax.saxutils import escape

# Distinct suffix per rendered gradient, so several field views can share a page.
_UID = itertools.count()

PALETTE = ("#0f766e", "#c2410c", "#2563eb", "#7c2d12", "#6d28d9", "#a16207")
INK = "#111827"
MUTED = "#6b7280"
HAIRLINE = "#d1d5db"


def line_chart(
    series: Sequence[tuple[str, Sequence[float]]],
    *,
    width: int = 620,
    height: int = 220,
    aria_label: str = "line chart",
    y_label: str = "",
) -> str:
    """Plot one or more equal-length series against their sample index."""
    if not series:
        raise ValueError("line_chart needs at least one series")
    length = len(series[0][1])
    if any(len(values) != length for _, values in series):
        raise ValueError("all series must have the same length")
    if length == 0:
        raise ValueError("series must not be empty")

    left, right, top, bottom = 46, 12, 14, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [v for _, series_values in series for v in series_values]
    vmin, vmax = min(values), max(values)
    if vmax - vmin < 1e-12:
        vmax = vmin + 1.0
    pad = 0.05 * (vmax - vmin)
    vmin, vmax = vmin - pad, vmax + pad

    def x_of(i: int) -> float:
        return left + (plot_w * i / max(length - 1, 1))

    def y_of(v: float) -> float:
        return top + plot_h * (1.0 - (v - vmin) / (vmax - vmin))

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(aria_label)}" '
        'class="chart">'
    ]
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>')

    for tick in _ticks(vmin, vmax, 4):
        y = y_of(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="{HAIRLINE}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
            f'fill="{MUTED}">{tick:.2g}</text>'
        )

    for index, (_, series_values) in enumerate(series):
        color = PALETTE[index % len(PALETTE)]
        points = " ".join(f"{x_of(i):.1f},{y_of(v):.1f}" for i, v in enumerate(series_values))
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="1.8" stroke-linejoin="round"/>'
        )

    for i in _index_ticks(length, 4):
        x = x_of(i)
        parts.append(
            f'<text x="{x:.1f}" y="{height - 10}" text-anchor="middle" font-size="10" '
            f'fill="{MUTED}">{i}</text>'
        )

    parts.append(
        f'<text x="6" y="{top + 10}" font-size="10" fill="{MUTED}">{escape(y_label)}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def line_chart_legend(series: Sequence[tuple[str, Sequence[float]]]) -> str:
    """Colour key for a line chart rendered with the same palette."""
    items = []
    for index, (label, _) in enumerate(series):
        color = PALETTE[index % len(PALETTE)]
        items.append(
            f'<span class="key"><span class="swatch" style="background:{color}"></span>'
            f"{escape(label)}</span>"
        )
    return f'<div class="legend">{"".join(items)}</div>'


def bar_chart(
    labels: Sequence[str],
    values: Sequence[float],
    errors: Sequence[float] | None = None,
    *,
    width: int = 620,
    height: int = 240,
    ymax: float | None = None,
    aria_label: str = "bar chart",
) -> str:
    """Bars with optional error whiskers, drawn on a 0..ymax scale."""
    if len(labels) != len(values):
        raise ValueError("labels and values must have the same length")
    if errors is not None and len(errors) != len(values):
        raise ValueError("errors must match values")
    if not labels:
        raise ValueError("bar_chart needs at least one bar")

    left, right, top, bottom = 46, 12, 26, 34
    plot_w, plot_h = width - left - right, height - top - bottom
    if ymax is None:
        ymax = max(
            v + (e or 0.0) for v, e in zip(values, errors or [0.0] * len(values), strict=True)
        )
        ymax = max(ymax * 1.1, 1e-6)
    if ymax <= 0:
        raise ValueError("ymax must be positive")

    def y_of(v: float) -> float:
        return top + plot_h * (1.0 - min(v, ymax) / ymax)

    slot = plot_w / len(values)
    bar_w = min(72.0, slot * 0.5)

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(aria_label)}" '
        'class="chart">'
    ]
    for tick in _ticks(0.0, ymax, 4):
        y = y_of(tick)
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w}" y2="{y:.1f}" '
            f'stroke="{HAIRLINE}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left - 6}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
            f'fill="{MUTED}">{tick:.2g}</text>'
        )

    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        color = PALETTE[index % len(PALETTE)]
        cx = left + slot * (index + 0.5)
        y = y_of(value)
        parts.append(
            f'<rect x="{cx - bar_w / 2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" '
            f'height="{max(top + plot_h - y, 0):.1f}" fill="{color}" rx="3"/>'
        )
        if errors is not None:
            err = errors[index]
            parts.append(_whisker(cx, y_of(value - err), y_of(value + err), color))
        parts.append(
            f'<text x="{cx:.1f}" y="{y - 6:.1f}" text-anchor="middle" font-size="11" '
            f'fill="{INK}">{value:.3f}</text>'
        )
        parts.append(
            f'<text x="{cx:.1f}" y="{height - 18}" text-anchor="middle" font-size="11" '
            f'fill="{INK}">{escape(label)}</text>'
        )

    parts.append("</svg>")
    return "".join(parts)


def heatmap(
    matrix: Sequence[Sequence[float]],
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    *,
    cell: int = 40,
    aria_label: str = "confusion matrix",
) -> str:
    """Row-normalised confusion matrix on a single-hue sequential ramp."""
    if len(matrix) != len(row_labels):
        raise ValueError("matrix rows and row_labels must match")
    if any(len(row) != len(col_labels) for row in matrix):
        raise ValueError("matrix columns and col_labels must match")

    label_w, top = 96, 26
    width = label_w + cell * len(col_labels) + 8
    height = top + cell * len(row_labels) + 34
    row_totals = [max(sum(row), 1) for row in matrix]
    peak = max(
        (value / total for row, total in zip(matrix, row_totals, strict=True) for value in row),
        default=0.0,
    )

    parts = [
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{escape(aria_label)}" '
        'class="chart">'
    ]
    for j, label in enumerate(col_labels):
        parts.append(
            f'<text x="{label_w + cell * (j + 0.5):.1f}" y="{top - 8}" text-anchor="middle" '
            f'font-size="10" fill="{MUTED}">{escape(label)}</text>'
        )
    for i, row in enumerate(matrix):
        parts.append(
            f'<text x="{label_w - 8}" y="{top + cell * (i + 0.6):.1f}" text-anchor="end" '
            f'font-size="10" fill="{MUTED}">{escape(row_labels[i])}</text>'
        )
        for j, value in enumerate(row):
            fraction = (value / row_totals[i]) / peak if peak > 0 else 0.0
            fill = _ramp(fraction)
            text_fill = "#ffffff" if fraction > 0.55 else INK
            x = label_w + cell * j
            y = top + cell * i
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" '
                f'fill="{fill}" rx="2"/>'
            )
            parts.append(
                f'<text x="{x + (cell - 2) / 2:.1f}" y="{y + cell / 2 + 4:.1f}" '
                f'text-anchor="middle" font-size="11" fill="{text_fill}">{value}</text>'
            )
    parts.append("</svg>")
    return "".join(parts)


def tensor_diagram(wires: int, slots: int, *, shown_slots: int = 16, width: int = 620) -> str:
    """Schematic of the tensor: wires, slot marks, coupling windows, launch, probe."""
    shown = min(slots, shown_slots)
    row_gap = 22.0
    left, right = 74.0, 74.0
    top = 46.0
    height = top + wires * row_gap + 40
    plot_w = width - left - right
    segment = plot_w / shown

    parts = [
        f'<svg viewBox="0 0 {width} {height:.0f}" role="img" aria-label="delay tensor schematic" '
        'class="chart">'
    ]
    parts.append(f'<rect x="0" y="0" width="{width}" height="{height:.0f}" fill="#ffffff"/>')
    parts.append(
        f'<text x="{left}" y="20" font-size="11" fill="{INK}">launch (wire 0 strongest)</text>'
    )
    parts.append(
        f'<text x="{width - right}" y="20" font-size="11" fill="{MUTED}" '
        f'text-anchor="end">edge probes</text>'
    )

    for wire in range(wires):
        y = top + wire * row_gap
        color = PALETTE[wire % len(PALETTE)]
        opacity = 0.95 - wire * 0.05
        parts.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_w:.1f}" y2="{y:.1f}" '
            f'stroke="{color}" stroke-width="4" stroke-linecap="round" opacity="{max(opacity, 0.4):.2f}"/>'
        )
        parts.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" font-size="10" '
            f'fill="{MUTED}">W{wire}</text>'
        )
        parts.append(f'<circle cx="{left + plot_w:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        if wire < wires - 1:
            for slot in range(0, shown, 2):
                x = left + (slot + 0.5) * segment
                parts.append(
                    f'<line x1="{x:.1f}" y1="{y + 6:.1f}" x2="{x:.1f}" y2="{y + row_gap - 6:.1f}" '
                    f'stroke="#c2410c" stroke-width="1.6" opacity="0.65"/>'
                )

    for slot in range(shown + 1):
        x = left + slot * segment
        parts.append(
            f'<line x1="{x:.1f}" y1="{top - 12}" x2="{x:.1f}" y2="{top + wires * row_gap - 8:.1f}" '
            f'stroke="{HAIRLINE}" stroke-width="0.8"/>'
        )
    parts.append(
        f'<text x="{left}" y="{height - 14:.0f}" font-size="10" fill="{MUTED}">'
        f"{wires} wires x {slots} slots; first {shown} coupling windows shown</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def field_view(
    traces: Sequence[Sequence[float]],
    *,
    width: int = 620,
    row_height: int = 18,
    min_cell: float = 1.6,
    duration_s: float = 3.0,
    aria_label: str = "electromagnetic field view",
) -> str:
    """Signed field strength on every wire and slot, with a looping wavefront.

    ``traces`` is the ``(wires, slots)`` state matrix. Each cell is shaded by
    its signed value -- teal for a positive field, orange for a negative one,
    white through zero -- so the coupling pattern reads as a picture instead
    of a stack of line charts. A soft wavefront sweeps left to right and
    repeats, which is what the eye needs in order to see propagation along the
    lines rather than a frozen matrix.

    The sweep is SMIL, not JavaScript: the report keeps its promise of being a
    static file that opens anywhere, from a USB stick or an archived artifact.
    ``prefers-reduced-motion`` turns the sweep off and leaves the still field,
    which is the same picture without the movement.

    ``min_cell`` is the floor in rendered pixels for one slot's column, so a
    long delay line stays separable rather than collapsing into a smear. The
    caller must let the SVG overflow horizontally when it exceeds its column;
    ``min-width`` alone only states the requirement.
    """
    # len() only: the report hands this an ndarray, whose truth value is
    # ambiguous, and a bare `if not traces` would raise instead of reporting.
    wires = len(traces)
    if wires == 0 or len(traces[0]) == 0:
        raise ValueError("field_view needs at least one wire with at least one slot")
    slots = len(traces[0])
    if any(len(row) != slots for row in traces):
        raise ValueError("every wire must have the same number of slots")

    peak = max((abs(value) for row in traces for value in row), default=0.0)
    scale = 1.0 / peak if peak > 0 else 0.0

    left, right, top = 46.0, 12.0, 26.0
    bottom = 30.0
    # `min_cell` has to be a floor on *rendered* pixels, and a viewBox cannot
    # be that: the report styles this as `width:100%`, so a wider viewBox is
    # divided by a wider element and every unit comes out the same size --
    # growing it changes nothing about what the eye sees. The floor is a CSS
    # min-width instead, which `width:100%` cannot cancel: the element is at
    # least this many pixels wide, and the viewBox then maps one unit to one
    # pixel at that size. A 784-slot raster needs 58 + 784*1.6 = 1312 px and
    # scrolls; a 64-slot one needs 160 px, fits any column, and scales up.
    floor_w = left + right + slots * min_cell
    width = max(float(width), floor_w)
    plot_w = width - left - right
    cell_w = plot_w / slots
    height = top + wires * row_height + bottom
    wave_w = max(cell_w * 3.0, 10.0)
    gradient_id = f"fieldwave{next(_UID)}"

    parts = [
        f'<svg viewBox="0 0 {width:g} {height:.0f}" role="img" '
        f'style="min-width:{floor_w:g}px" '
        f'aria-label="{escape(aria_label)}" class="chart field">',
        "<style>@media (prefers-reduced-motion: reduce){.field-wave{display:none}}</style>",
        f'<defs><linearGradient id="{gradient_id}" x1="0" y1="0" x2="1" y2="0">'
        '<stop offset="0" stop-color="#ffffff" stop-opacity="0"/>'
        '<stop offset="0.5" stop-color="#ffffff" stop-opacity="0.55"/>'
        '<stop offset="1" stop-color="#ffffff" stop-opacity="0"/>'
        "</linearGradient></defs>",
        f'<rect x="0" y="0" width="{width}" height="{height:.0f}" fill="#ffffff"/>',
        # Legend: the two poles of the diverging ramp.
        f'<rect x="{width - right - 92}" y="9" width="10" height="10" rx="2" '
        f'fill="{_diverging(-1.0)}"/>',
        f'<text x="{width - right - 78}" y="18" font-size="10" fill="{MUTED}">negative</text>',
        f'<rect x="{width - right - 30}" y="9" width="10" height="10" rx="2" '
        f'fill="{_diverging(1.0)}"/>',
        f'<text x="{width - right - 16}" y="18" font-size="10" fill="{MUTED}">+</text>',
    ]

    for wire in range(wires):
        y = top + wire * row_height
        parts.append(
            f'<text x="{left - 8}" y="{y + row_height / 2 + 3:.1f}" text-anchor="end" '
            f'font-size="10" fill="{MUTED}">W{wire}</text>'
        )
        for slot in range(slots):
            x = left + slot * cell_w
            fill = _diverging(traces[wire][slot] * scale)
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.1f}" width="{cell_w + 0.4:.2f}" '
                f'height="{row_height - 2}" fill="{fill}"/>'
            )

    # The wavefront: a translucent band that crosses the whole window and loops.
    parts.append(
        f'<rect class="field-wave" x="{left - wave_w:.1f}" y="{top - 2:.1f}" '
        f'width="{wave_w:.1f}" height="{wires * row_height + 4:.0f}" '
        f'fill="url(#{gradient_id})">'
        f'<animate attributeName="x" from="{left - wave_w:.1f}" to="{left + plot_w:.1f}" '
        f'dur="{duration_s:g}s" repeatCount="indefinite"/>'
        "</rect>"
    )

    parts.append(
        f'<text x="{left}" y="{height - 10:.0f}" font-size="10" fill="{MUTED}">slot 0</text>'
    )
    parts.append(
        f'<text x="{left + plot_w:.0f}" y="{height - 10:.0f}" font-size="10" '
        f'text-anchor="end" fill="{MUTED}">slot {slots - 1}</text>'
    )
    parts.append(
        f'<text x="{left + plot_w / 2:.0f}" y="{height - 10:.0f}" font-size="10" '
        f'text-anchor="middle" fill="{MUTED}">'
        f"{wires} wires x {slots} slots, shading = field strength</text>"
    )
    parts.append("</svg>")
    return "".join(parts)


def _whisker(x: float, y_low: float, y_high: float, color: str) -> str:
    cap = 10.0
    return (
        f'<line x1="{x:.1f}" y1="{y_high:.1f}" x2="{x:.1f}" y2="{y_low:.1f}" '
        f'stroke="{INK}" stroke-width="1.4"/>'
        f'<line x1="{x - cap / 2:.1f}" y1="{y_high:.1f}" x2="{x + cap / 2:.1f}" '
        f'y2="{y_high:.1f}" stroke="{INK}" stroke-width="1.4"/>'
        f'<line x1="{x - cap / 2:.1f}" y1="{y_low:.1f}" x2="{x + cap / 2:.1f}" '
        f'y2="{y_low:.1f}" stroke="{INK}" stroke-width="1.4"/>'
        f'<circle cx="{x:.1f}" cy="{y_high:.1f}" r="3" fill="{color}"/>'
    )


def _ticks(vmin: float, vmax: float, count: int) -> list[float]:
    step = (vmax - vmin) / count
    return [vmin + step * i for i in range(count + 1)]


def _index_ticks(length: int, count: int) -> list[int]:
    if length <= count:
        return list(range(length))
    step = (length - 1) / count
    return sorted({round(step * i) for i in range(count + 1)})


def _ramp(t: float) -> str:
    """Light teal to deep teal, where t is in [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    start = (240, 253, 250)
    end = (15, 118, 110)
    channels = [round(s + (e - s) * t) for s, e in zip(start, end, strict=True)]
    return f"rgb({channels[0]},{channels[1]},{channels[2]})"


def _diverging(value: float) -> str:
    """Orange through white to teal, where value is in [-1, 1].

    A signed field needs a midpoint that reads as "nothing here"; the
    single-hue ramp cannot show polarity, and polarity is the whole point of
    an EM field view.
    """
    value = min(max(value, -1.0), 1.0)
    white = (255, 255, 255)
    if value >= 0.0:
        end = (15, 118, 110)  # teal
    else:
        end = (194, 65, 12)  # orange
    t = abs(value)
    channels = [round(s + (e - s) * t) for s, e in zip(white, end, strict=True)]
    return f"rgb({channels[0]},{channels[1]},{channels[2]})"
