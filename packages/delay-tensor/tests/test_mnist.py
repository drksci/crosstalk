"""The three-way MNIST comparison and its third arm in the report."""

from __future__ import annotations

import gc
import json
import weakref
from pathlib import Path

import numpy as np
import pytest

from delay_tensor.config import DelayTensorConfig, TensorConfig
from delay_tensor.experiments import run_classify, run_mnist
from delay_tensor.experiments.mnist import _batched, _choose_examples, _feature_factory
from delay_tensor.mnist import Mnist
from delay_tensor.report import write_run

# Small enough to finish in well under a second: 16 slots stand in for the
# real 784, and a truncated raster still exercises the whole composition --
# memmap encode, both linear arms, the MLP, and the Summary that carries them.
CONFIG = DelayTensorConfig(
    tensor=TensorConfig(wires=2, slots=16, slot_dt_ns=20.0, loop_period_ns=320.0),
    name="mnist-small",
)


def _fake_mnist() -> Mnist:
    """A tiny stand-in with the real dtype, shape and label range."""
    rng = np.random.default_rng(0)
    train_images = rng.integers(0, 256, size=(60, 28, 28), dtype=np.uint8)
    test_images = rng.integers(0, 256, size=(40, 28, 28), dtype=np.uint8)
    return Mnist(
        train_images=train_images,
        train_labels=np.arange(60) % 10,
        test_images=test_images,
        test_labels=np.arange(40) % 10,
    )


@pytest.fixture()
def tiny_mnist(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("delay_tensor.experiments.mnist.load_mnist", _fake_mnist)


def test_mnist_runs_the_three_way_comparison(tiny_mnist: None) -> None:
    summary = run_mnist(
        CONFIG,
        seeds=(11,),
        epochs=1,
        batch_size=16,
        reservoir_batch=16,
        mlp_max_iter=3,
        max_examples=3,
    )

    assert summary.experiment == "mnist"
    assert summary.class_names == tuple(str(d) for d in range(10))
    assert summary.feature_count == 32
    # The canonical split, not a reshuffle of it.
    assert summary.records[0].train_size == 60
    assert summary.records[0].test_size == 40

    # Every arm reports, and none of them is a stub.
    record = summary.records[0]
    for accuracy in (record.raw_accuracy, record.reservoir_accuracy, record.mlp_accuracy):
        assert accuracy is not None
        assert 0.0 <= accuracy <= 1.0

    # The third arm carries its own confusion matrix, summed over seeds.
    assert summary.confusion_mlp is not None
    assert summary.confusion_mlp.shape == (10, 10)
    assert summary.confusion_mlp.sum() == 40
    assert summary.mlp is not None
    assert summary.has_mlp

    # Examples come from the first seed with all three predictions attached.
    assert len(summary.examples) == 3
    assert len({example.true_label for example in summary.examples}) == 3
    for example in summary.examples:
        assert example.predicted_mlp is not None
        assert example.waveform.shape == (784,)
        assert example.traces.shape == (2, 16)

    assert summary.health["status"] in {"healthy", "saturated"}


def test_mnist_needs_at_least_one_seed(tiny_mnist: None) -> None:
    with pytest.raises(ValueError, match="at least one seed"):
        run_mnist(CONFIG, seeds=())


def test_mnist_report_renders_the_third_arm(tmp_path: Path, tiny_mnist: None) -> None:
    summary = run_mnist(
        CONFIG, seeds=(11,), epochs=1, batch_size=16, reservoir_batch=16, mlp_max_iter=3
    )
    paths = write_run(tmp_path / "run", summary, command="delay-tensor mnist --config x")

    report = paths.report.read_text()
    # A third card, a third bar, a third seed column, a third confusion panel.
    assert 'card-label">MLP</div>' in report
    assert ">MLP</text>" in report  # the bar chart's own axis label
    assert "<th>MLP</th>" in report
    assert "MLP (count over seeds)" in report
    assert "MLP: " in report  # the verdict chip on a held-out example

    metrics = json.loads(paths.metrics.read_text())
    assert "mlp_mean" in metrics["aggregate"]
    assert "mlp_accuracy" in metrics["per_seed"][0]
    assert "confusion_mlp" in metrics
    # The whole layer structure, not just its first layer.
    assert metrics["parameters"]["mlp_hidden_layers"] == "128"
    assert metrics["parameters"]["epochs"] == 1

    header = paths.csv.read_text().splitlines()[0]
    # The column sits with the arms, not tacked on the end, so a spreadsheet
    # that ignores it still lines raw and reservoir up correctly.
    assert header == (
        "seed,raw_accuracy,reservoir_accuracy,mlp_accuracy,delta,train_size,test_size"
    )

    # The field view's min-width only helps if the column may overflow it;
    # without the scroll wrapper, width:100% would scale it back into the
    # sub-pixel smear the min-width exists to prevent.
    assert '<div class="field-scroll">' in report
    assert "overflow-x: auto" in report
    assert "min-width:" in report

    # The static-file promise is unchanged by the third arm.
    assert "<script" not in report
    assert "http://" not in report and "https://" not in report
    assert report.count("<svg") == report.count("</svg>")


def test_two_arm_runs_omit_the_third_arm(tmp_path: Path) -> None:
    """classify and xor must render exactly as they did before the third arm."""
    summary = run_classify(CONFIG, seeds=(7,), samples_per_class=8, time_samples=16)
    assert not summary.has_mlp
    assert summary.mlp is None

    paths = write_run(tmp_path / "run", summary)
    report = paths.report.read_text()
    assert "MLP" not in report
    assert "<th>MLP</th>" not in report
    assert "confusion_mlp" not in paths.metrics.read_text()
    assert "mlp_" not in paths.metrics.read_text()
    assert paths.csv.read_text().splitlines()[0] == (
        "seed,raw_accuracy,reservoir_accuracy,delta,train_size,test_size"
    )
    assert "<script" not in report


def test_observations_name_the_third_arm_when_present(tmp_path: Path, tiny_mnist: None) -> None:
    summary = run_mnist(
        CONFIG, seeds=(11,), epochs=1, batch_size=16, reservoir_batch=16, mlp_max_iter=3
    )
    paths = write_run(tmp_path / "run", summary)
    text = paths.observations.read_text()
    assert "MLP baseline:" in text
    # It is a reference, not a contestant, and the wording says so.
    assert "for reference rather than as a contestant" in text


def test_choose_examples_spreads_labels_rather_than_taking_the_head() -> None:
    labels = np.arange(50) % 10
    rows = _choose_examples(labels, 3)
    assert len(rows) == 3
    # Distinct rows, and their labels are not all the same digit.
    assert len(set(rows)) == 3
    assert len({int(labels[row]) for row in rows}) == 3


def test_choose_examples_never_returns_fewer_than_asked_for_one_class() -> None:
    labels = np.zeros(20, dtype=np.int64)
    assert _choose_examples(labels, 3) == [0]


def test_batched_covers_every_row_without_overlap() -> None:
    source = np.arange(100, dtype=np.float32).reshape(50, 2)
    blocks = list(_batched(source, 16))
    # Starts at row 0 in `batch` steps, which is what _as_batches does for a
    # dense array: the two linear arms must see identical boundaries.
    assert [len(block) for block in blocks] == [16, 16, 16, 2]
    assert np.array_equal(np.concatenate(blocks), source)
    # Views, not copies: a copy here would double the peak memory.
    assert np.shares_memory(blocks[0], source)


def test_feature_factory_releases_its_source(tmp_path: Path) -> None:
    """run_mnist frees 1.5 GB of memmap before the MLP runs; this is why.

    The factory must hold the mapping only for as long as the factory itself
    does. If it leaked the reference, `del tensor_source` would free nothing
    and the MLP would be fitted while the whole feature matrix was still
    resident -- which on an 8 GB machine is the difference between finishing
    and swapping.
    """
    mapping = np.memmap(tmp_path / "features.f32", dtype=np.float32, mode="w+", shape=(4, 3))
    reference = weakref.ref(mapping)
    factory = _feature_factory(mapping, 2)

    # Consumed once, exactly as fit_stream does for its scaler pass.
    assert [len(block) for block in factory()] == [2, 2]

    del mapping, factory
    gc.collect()
    assert reference() is None, "the factory kept the memmap alive past its use"
