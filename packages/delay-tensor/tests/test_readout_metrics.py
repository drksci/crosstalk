import numpy as np
import pytest

from delay_tensor.metrics import accuracy, confusion_matrix, spread
from delay_tensor.readout import LinearReadout


def test_accuracy_and_confusion_matrix_agree() -> None:
    labels = np.array([0, 1, 1, 2, 2, 2])
    predictions = np.array([0, 1, 0, 2, 2, 1])
    assert accuracy(predictions, labels) == pytest.approx(4 / 6)

    matrix = confusion_matrix(predictions, labels, n_classes=3)
    assert matrix.sum() == len(labels)
    np.testing.assert_array_equal(matrix.sum(axis=1), [1, 2, 3])
    assert matrix[1, 0] == 1  # true class 1, predicted 0
    assert matrix[2, 2] == 2  # true class 2, predicted 2 twice


def test_confusion_matrix_rejects_out_of_range_labels() -> None:
    with pytest.raises(ValueError, match="labels"):
        confusion_matrix(np.array([0]), np.array([5]), n_classes=2)
    with pytest.raises(ValueError, match="predictions"):
        confusion_matrix(np.array([7]), np.array([0]), n_classes=2)


def test_spread_reports_sample_std() -> None:
    result = spread([0.5, 0.7])
    assert result.n == 2
    assert result.mean == pytest.approx(0.6)
    assert result.std == pytest.approx(np.std([0.5, 0.7], ddof=1))
    assert spread([0.5]).std == 0.0
    assert spread([0.5, 0.7]).format() == "0.600 ± 0.141"
    with pytest.raises(ValueError, match="empty"):
        spread([])


def test_readout_separates_a_linearly_separable_problem() -> None:
    rng = np.random.default_rng(0)
    positives = rng.normal(loc=2.0, scale=0.3, size=(40, 5))
    negatives = rng.normal(loc=-2.0, scale=0.3, size=(40, 5))
    x = np.vstack([positives, negatives])
    y = np.array([1] * 40 + [0] * 40)

    model = LinearReadout(seed=0).fit(x, y)
    assert model.score(x, y) == 1.0
    assert set(model.classes_) == {0, 1}


def test_readout_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        LinearReadout().fit(np.zeros(3), np.zeros(3))
    with pytest.raises(ValueError, match="sample count"):
        LinearReadout().fit(np.zeros((4, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="two classes"):
        LinearReadout().fit(np.zeros((4, 3)), np.zeros(4))
    with pytest.raises(ValueError, match="C"):
        LinearReadout(C=0.0)
