import numpy as np

from dlnn.data import modrec, narma10


def test_narma10_deterministic_and_causal():
    a, b = narma10.dataset(), narma10.dataset()
    assert a["sha256"] == b["sha256"]
    assert a["x_train"].shape == (1, 1, 4000) and a["y_test"].shape == (1, 1, 2000)
    assert np.all(np.isfinite(a["y_train"])) and a["x_train"].min() >= 0 and a["x_train"].max() <= 0.5
    u, y = narma10.narma10_series(50)
    # target[t] = y[t+1] depends only on u[t], u[t-9]: recompute one step by hand
    t = 30
    assert np.isclose(y[t], 0.3 * y[t - 1] + 0.05 * y[t - 1] * y[t - 10:t].sum() + 1.5 * u[t - 9] * u[t] + 0.1)


def test_narma10_seed_changes_hash():
    assert narma10.dataset(seed=1)["sha256"] != narma10.dataset()["sha256"]


def test_modrec_shapes_balance_determinism():
    d1, d2 = modrec.generate(400, seed=3), modrec.generate(400, seed=3)
    assert d1["x"].shape == (400, 2, 128) and d1["x"].dtype == np.float32
    assert np.array_equal(d1["x"], d2["x"]) and np.array_equal(d1["y"], d2["y"])
    assert np.all(np.bincount(d1["y"]) == 50)                       # 8 classes balanced
    vals, counts = np.unique(d1["snr"], return_counts=True)
    assert list(vals) == [0, 5, 10, 15, 20] and np.all(counts == 80)
    for c in range(8):                                              # SNR balanced inside each class
        assert len(set(np.bincount(np.searchsorted(vals, d1["snr"][d1["y"] == c])))) == 1
    assert np.isfinite(d1["x"]).all()
    assert not np.array_equal(modrec.generate(400, seed=4)["x"], d1["x"])


def test_modrec_signal_properties():
    d = modrec.generate(800, seed=5)
    z = d["x"][:, 0] + 1j * d["x"][:, 1]
    hi = d["snr"] == 20
    for c, name in enumerate(modrec.CLASSES):
        s = z[hi & (d["y"] == c)]
        p = np.mean(np.abs(s) ** 2, axis=1)
        assert np.all((p > 0.9) & (p < 1.2)), name                  # unit signal power + 1 % noise at 20 dB
    env = np.abs(z[hi & (d["y"] == modrec.CLASSES.index("2FSK"))])
    ook = np.abs(z[hi & (d["y"] == modrec.CLASSES.index("OOK"))])
    assert env.std(axis=1).mean() < 0.2 * ook.std(axis=1).mean() + 0.1  # FSK ~constant envelope, OOK not


def test_modrec_dataset_hash():
    ds = modrec.dataset(n_train=80, n_test=40)
    assert ds["x_train"].shape == (80, 2, 128) and len(ds["sha256"]) == 64
    assert ds["sha256"] == modrec.dataset(n_train=80, n_test=40)["sha256"]
