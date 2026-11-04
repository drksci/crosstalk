import json
from pathlib import Path

import pytest

from delay_tensor.config import DelayTensorConfig, ReadoutConfig, TensorConfig
from delay_tensor.experiments import run_classify, run_xor
from delay_tensor.report import write_run

CONFIG = DelayTensorConfig(
    tensor=TensorConfig(wires=4, slots=16, slot_dt_ns=20.0, loop_period_ns=320.0)
)


def test_classify_aggregates_seeds_and_counts_confusions() -> None:
    summary = run_classify(
        CONFIG,
        seeds=(7, 8),
        samples_per_class=10,
        time_samples=16,
        noise_std=0.0,
        test_fraction=0.25,
    )

    assert summary.experiment == "classify"
    assert len(summary.records) == 2
    assert summary.raw.n == summary.reservoir.n == 2
    assert summary.feature_count == 64
    assert summary.chance_level == pytest.approx(1 / 6)
    for record in summary.records:
        assert 0.0 <= record.raw_accuracy <= 1.0
        assert 0.0 <= record.reservoir_accuracy <= 1.0
        assert record.delta == pytest.approx(record.reservoir_accuracy - record.raw_accuracy)

    # Confusion matrices are summed over seeds, so their totals are the total
    # number of held-out predictions.
    expected = sum(record.test_size for record in summary.records)
    assert summary.confusion_raw.sum() == expected
    assert summary.confusion_reservoir.sum() == expected
    assert summary.confusion_raw.shape == (6, 6)

    # One example per class from the first seed, spread across the label range.
    labels_shown = [example.true_label for example in summary.examples]
    assert len(labels_shown) == 3
    assert len(set(labels_shown)) == 3
    assert min(labels_shown) == 0
    assert summary.examples[0].traces.shape[0] <= 4
    assert summary.health["status"] in {"healthy", "saturated"}


def test_metrics_dict_is_json_serialisable_and_complete() -> None:
    summary = run_classify(CONFIG, seeds=(7,), samples_per_class=8, time_samples=16)
    payload = json.loads(json.dumps(summary.to_metrics_dict()))

    assert payload["experiment"] == "classify"
    assert payload["seeds"] == [7]
    assert payload["aggregate"]["n_seeds"] == 1
    assert payload["feature_count"] == 64
    assert len(payload["per_seed"]) == 1
    assert payload["parameters"]["samples_per_class"] == 8
    assert payload["confusion_reservoir"][0] and isinstance(
        payload["confusion_reservoir"][0][0], int
    )


def test_xor_runs_on_the_same_harness() -> None:
    summary = run_xor(CONFIG, seeds=(3, 4), samples=48, time_samples=16, noise_std=0.0)
    assert summary.experiment == "xor"
    assert summary.class_names == ("false", "true")
    assert summary.chance_level == pytest.approx(0.5)
    assert summary.confusion_reservoir.shape == (2, 2)
    assert 0.0 <= summary.reservoir.mean <= 1.0


def test_readout_mode_changes_the_feature_surface() -> None:
    config = DelayTensorConfig(
        tensor=CONFIG.tensor,
        readout=ReadoutConfig(traces="rightmost_wire"),
        name="rightmost",
    )
    summary = run_classify(config, seeds=(7,), samples_per_class=8, time_samples=16)
    assert summary.feature_count == 16


def test_requires_at_least_one_seed() -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        run_classify(CONFIG, seeds=())


def test_write_run_produces_every_artifact(tmp_path: Path) -> None:
    summary = run_classify(CONFIG, seeds=(7,), samples_per_class=8, time_samples=16)
    paths = write_run(tmp_path / "run", summary, command="delay-tensor classify --config x")

    for path in (paths.report, paths.metrics, paths.csv, paths.observations, paths.config):
        assert path.exists(), path
        assert path.stat().st_size > 0

    metrics = json.loads(paths.metrics.read_text())
    assert metrics["aggregate"]["n_seeds"] == 1

    csv_lines = paths.csv.read_text().strip().splitlines()
    assert csv_lines[0].startswith("seed,raw_accuracy")
    assert len(csv_lines) == 2

    report = paths.report.read_text()
    for marker in (
        "<!doctype html>",
        "Confusion matrices",
        "Per-seed results",
        "Held-out examples",
        "delay-tensor classify --config x",
        "static HTML, no external assets",
    ):
        assert marker in report
    # No external assets: nothing to fetch, nothing to break later.
    assert "http://" not in report
    assert "<script" not in report
    assert report.count("<svg") == report.count("</svg>")
