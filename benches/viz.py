"""Static SVG output for the SPICE benches.

Same house rules as ``delay_tensor.svg``: hand-written SVG, no plotting
library, no JavaScript, nothing that reaches the network. The notebook HTML
therefore stays self-contained and diff-able.
"""

from __future__ import annotations

import html
import math

from delay_tensor.svg import _ramp, line_chart, line_chart_legend


def log_error_heatmap(
    errors: list[list[float]],
    *,
    width: int = 620,
    height: int = 220,
    title: str = "",
) -> str:
    """Abs error on a log10 colour scale, wires down and slots across.

    Linear colour would be useless here: at machine precision every cell is
    equally "zero". The log scale instead exposes *where* rounding
    concentrates and whether it grows along the slot axis.
    """
    if not errors or not errors[0]:
        raise ValueError("errors must be a non-empty 2-D grid")

    rows, cols = len(errors), len(errors[0])
    finite = [math.log10(e) for row in errors for e in row if e > 0.0]
    floor = min(finite) if finite else -18.0
    ceil = max(max(finite), floor + 1.0) if finite else -16.0

    pad_left, pad_right, pad_top, pad_bottom = 54, 16, 34 if title else 12, 30
    plot_w = width - pad_left - pad_right
    plot_h = height - pad_top - pad_bottom
    cell_w, cell_h = plot_w / cols, plot_h / rows

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" role="img" '
        f'aria-label="{html.escape(title or "error heatmap")}">',
        f'<rect width="{width}" height="{height}" fill="none"/>',
    ]
    if title:
        parts.append(
            f'<text x="{pad_left}" y="{title_y(pad_top)}" font-size="12" '
            f'fill="#c9d6e8" font-family="sans-serif">{html.escape(title)}</text>'
        )

    for r, row in enumerate(errors):
        for c, value in enumerate(row):
            exponent = math.log10(value) if value > 0.0 else floor
            t = 0.0 if ceil == floor else (exponent - floor) / (ceil - floor)
            t = min(max(t, 0.0), 1.0)
            x = pad_left + c * cell_w
            y = pad_top + r * cell_h
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{cell_w + 0.4:.2f}" '
                f'height="{cell_h + 0.4:.2f}" fill="{_ramp(1.0 - t)}"/>'
            )

    # wire labels down the left edge, slot ticks along the bottom
    for r in range(rows):
        y = pad_top + (r + 0.5) * cell_h
        parts.append(
            f'<text x="{pad_left - 6}" y="{y + 3:.1f}" font-size="9" text-anchor="end" '
            f'fill="#8fa3bd" font-family="sans-serif">w{r}</text>'
        )
    tick_every = max(1, cols // 8)
    for c in range(0, cols, tick_every):
        x = pad_left + (c + 0.5) * cell_w
        parts.append(
            f'<text x="{x:.1f}" y="{pad_top + plot_h + 13:.1f}" font-size="9" '
            f'text-anchor="middle" fill="#8fa3bd" font-family="sans-serif">{c}</text>'
        )
    parts.append(
        f'<text x="{pad_left + plot_w / 2:.1f}" y="{height - 4}" font-size="10" '
        f'text-anchor="middle" fill="#8fa3bd" font-family="sans-serif">slot</text>'
    )

    # colour key: 10^floor (deep) to 10^ceil (light)
    key_w, key_h = 140, 9
    key_x, key_y = width - pad_right - key_w, 12 if not title else 20
    steps = 24
    for i in range(steps):
        t = i / (steps - 1)
        x = key_x + t * key_w
        parts.append(
            f'<rect x="{x:.2f}" y="{key_y}" width="{key_w / steps + 0.4:.2f}" '
            f'height="{key_h}" fill="{_ramp(1.0 - t)}"/>'
        )
    parts.append(
        f'<text x="{key_x}" y="{key_y + key_h + 10}" font-size="9" fill="#8fa3bd" '
        f'font-family="sans-serif">1e{floor:.0f}</text>'
    )
    parts.append(
        f'<text x="{key_x + key_w}" y="{key_y + key_h + 10}" font-size="9" '
        f'text-anchor="end" fill="#8fa3bd" font-family="sans-serif">1e{ceil:.0f}</text>'
    )
    parts.append("</svg>")
    return "".join(parts)


def title_y(pad_top: int) -> int:
    return pad_top - 12


def error_by_slot(errors: list[list[float]]) -> str:
    """Worst error seen by the end of each slot — does rounding compound?"""
    worst = [max(row[c] for row in errors) for c in range(len(errors[0]))]
    return line_chart(
        [("max |error|", worst)], width=620, height=180, y_label="abs error", aria_label="error by slot"
    )


def error_legend() -> str:
    return line_chart_legend([("max |error|", [])])
