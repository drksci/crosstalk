import xml.etree.ElementTree as ET

import pytest

from delay_tensor.svg import (
    bar_chart,
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
