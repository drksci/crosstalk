import re
import xml.etree.ElementTree as ET

import pytest

from delay_tensor.svg import (
    bar_chart,
    field_view,
    heatmap,
    line_chart,
    line_chart_legend,
    tensor_diagram,
)


def parse(svg: str) -> ET.Element:
    return ET.fromstring(svg)


def test_line_chart_is_well_formed_and_labels_every_axis() -> None:
    chart = parse(line_chart([("a", [0.0, 1.0, 0.5]), ("b", [1.0, 0.0, 0.7])], aria_label="demo"))
    assert chart.tag == "svg"
    assert chart.attrib["aria-label"] == "demo"
    polylines = chart.findall("polyline")
    assert len(polylines) == 2
    assert len(polylines[0].attrib["points"].split()) == 3


def test_line_chart_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one"):
        line_chart([])
    with pytest.raises(ValueError, match="same length"):
        line_chart([("a", [1.0, 2.0]), ("b", [1.0])])
    with pytest.raises(ValueError, match="empty"):
        line_chart([("a", [])])


def test_line_chart_escapes_text() -> None:
    chart = line_chart([("a & b", [1.0, 2.0])], y_label="V & I")
    assert "&amp;" in chart
    parse(chart)


def test_bar_chart_draws_whiskers_and_bars() -> None:
    chart = parse(bar_chart(["raw", "reservoir"], [0.6, 0.8], errors=[0.01, 0.02], ymax=1.0))
    assert len(chart.findall("rect")) == 2
    # whiskers are drawn in the ink colour; gridlines are hairline grey.
    whiskers = [line for line in chart.findall("line") if line.get("stroke") == "#111827"]
    assert len(whiskers) == 2 * 3  # one vertical plus two caps per bar
    with pytest.raises(ValueError, match="labels and values"):
        bar_chart(["a"], [0.1, 0.2])
    with pytest.raises(ValueError, match="errors"):
        bar_chart(["a"], [0.1], errors=[0.1, 0.2])


def test_heatmap_shapes_and_counts() -> None:
    matrix = [[3, 1], [0, 4]]
    chart = parse(heatmap(matrix, ["yes", "no"], ["p-yes", "p-no"]))
    rects = chart.findall("rect")
    assert len(rects) == 4
    texts = [node.text for node in chart.findall("text")]
    assert "3" in texts and "4" in texts
    with pytest.raises(ValueError, match="row_labels"):
        heatmap(matrix, ["only-one"], ["a", "b"])


def test_tensor_diagram_mentions_geometry() -> None:
    chart = parse(tensor_diagram(wires=8, slots=64))
    texts = [node.text for node in chart.findall("text")]
    assert any(text and "8 wires x 64 slots" in text for text in texts)
    assert any(text and text.startswith("W") for text in texts)


def test_legend_matches_palette_order() -> None:
    legend = line_chart_legend([("first", [0.0]), ("second", [0.0])])
    assert legend.count("swatch") == 2
    assert "first" in legend and "second" in legend


def test_field_view_draws_one_cell_per_wire_and_slot() -> None:
    traces = [[1.0, -1.0, 0.0], [0.5, 0.0, -0.5]]
    chart = parse(field_view(traces, aria_label="field demo"))
    assert chart.attrib["aria-label"] == "field demo"
    # background + two legend swatches + one cell per (wire, slot) + the band.
    rects = chart.findall("rect")
    assert len(rects) == 1 + 2 + 2 * 3 + 1
    texts = [node.text for node in chart.findall("text")]
    assert any(text and "2 wires x 3 slots" in text for text in texts)


def test_field_view_loops_without_javascript() -> None:
    """The sweep must repeat forever and must not be JavaScript."""
    svg = field_view([[1.0, -1.0]])
    assert "<script" not in svg.lower()
    chart = parse(svg)
    animate = chart.findall(".//animate")
    assert len(animate) == 1
    assert animate[0].attrib["repeatCount"] == "indefinite"
    assert animate[0].attrib["attributeName"] == "x"


def test_field_view_stops_under_reduced_motion() -> None:
    svg = field_view([[1.0, -1.0]])
    assert "@media (prefers-reduced-motion: reduce)" in svg
    assert ".field-wave{display:none}" in svg


def test_field_view_shows_polarity_through_colour() -> None:
    """Positive and negative cells must not share a fill, or polarity is lost."""
    svg = field_view([[1.0, -1.0, 0.0]])
    cells = [rect for rect in parse(svg).findall("rect") if rect.get("height") == "16"]
    fills = {rect.get("fill") for rect in cells}
    assert len(fills) == 3, "positive, negative and zero must read as three colours"
    assert "rgb(255,255,255)" in fills, "zero field is white, the ramp's midpoint"


def test_field_view_scales_by_peak_not_absolute() -> None:
    """A quiet tensor still has to fill the ramp, or every cell is near-white."""
    quiet = field_view([[0.01, -0.01]])
    loud = field_view([[1.0, -1.0]])
    assert "rgb(15,118,110)" in quiet and "rgb(15,118,110)" in loud


def test_field_view_keeps_long_delay_lines_resolvable() -> None:
    """A 784-slot raster must not render cells thinner than a device pixel.

    The floor is a CSS ``min-width``, not a bigger viewBox: the report styles
    these as ``width:100%``, so a wider viewBox divided by a wider element
    leaves every unit exactly the same rendered size. Asserting on the viewBox
    would pass while the picture stayed a smear, which is what the first
    version of this did.
    """
    row = [float(i % 3 - 1) for i in range(784)]
    chart = parse(field_view([row] * 8))

    floor_w = float(re.search(r"min-width:([\d.]+)px", chart.attrib["style"]).group(1))
    # Enough room for the 58 units of gutter plus one min_cell per slot.
    assert floor_w >= 46.0 + 12.0 + 784 * 1.6

    # At min-width the element is floor_w px wide and the viewBox is floor_w
    # units, so one unit is one pixel and each slot's column is min_cell px.
    viewBox = [float(part) for part in chart.attrib["viewBox"].split()]
    assert viewBox[2] == floor_w, "at min-width the two must coincide, cell = min_cell"

    # A short delay line asks for far less than any column, so it keeps
    # scaling to fit instead of overflowing its box.
    short = parse(field_view([[1.0, -1.0, 0.0]]))
    short_floor = float(re.search(r"min-width:([\d.]+)px", short.attrib["style"]).group(1))
    assert short_floor < 620
    assert float(short.attrib["viewBox"].split()[2]) == 620


def test_field_view_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="at least one wire"):
        field_view([])
    with pytest.raises(ValueError, match="at least one wire"):
        field_view([[]])
    with pytest.raises(ValueError, match="same number of slots"):
        field_view([[1.0, 2.0], [1.0]])
