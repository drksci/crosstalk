"""M1b tests: board-matched TDL-Net-B (d = 1), line-FIR aware layers, arena op counting, shipped-model gate."""
# ruff: noqa: E402  (torch is importorskip-ed before the dlnn.model imports that need it)
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed (dlnn[train] extra)")

from dlnn.data import modrec
from dlnn.model import arena, export, train
from dlnn.model.npz import load_model
from dlnn.model.tdlnet_torch import TDLNet
from dlnn.tiers import virtual

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


def test_cli_dilations_depth_options_exist():
    with pytest.raises(SystemExit):
        train.main(["--depth", "3", "--dilations", "1,1"])      # length mismatch is rejected by the parser


def test_identity_fir_equals_plain_conv():
    torch.manual_seed(0)
    K = 8
    plain = TDLNet(widths=(2, 8, 8, 8), dilations=(1, 1, 1))
    fir = np.eye(K, K + 2).tolist()                              # tap k -> lag k exactly (+2 unused lags)
    aware = TDLNet(widths=(2, 8, 8, 8), dilations=(1, 1, 1), line_fir=fir)
    aware.load_state_dict(plain.state_dict())
    x = torch.randn(3, 2, 128)
    assert torch.allclose(plain(x), aware(x), atol=1e-5)


def test_aware_layer_is_causal():
    fir = json.loads((ROOT / "models/board_aware/line_fir.json").read_text())["fir"]
    net = TDLNet(widths=(2, 8, 8, 8), dilations=(1, 1, 1), line_fir=fir)
    x = torch.randn(2, 2, 128)
    y1 = net(x, return_layers=True)[1][-1]
    x2 = x.clone(); x2[:, :, 90:] += 1.0
    y2 = net(x2, return_layers=True)[1][-1]
    assert torch.allclose(y1[:, :, :90], y2[:, :, :90], atol=1e-6)


def test_line_fir_is_near_identity():
    j = json.loads((ROOT / "models/board_aware/line_fir.json").read_text())
    f = np.array(j["fir"])
    assert f.shape[0] == 8 and list(j["main_lag"]) == list(range(8))
    assert max(j["off_nominal_energy_frac"]) < 0.15


def test_dlnn_op_count_exact_small():
    from dlnn.model.npz import Layer, Model
    W = np.zeros((2, 2, 8), np.int8); W[0, 0, 0] = 1; W[1, 1, 3] = -1; W[1, 0, 7] = 1     # 3 non-zero taps
    ly = Layer(W=W, dilation=1, gain=np.ones(2, np.float32), bias=np.zeros(2, np.float32), vsat=1.0)
    m = Model(kind="tdlnet", layers=[ly], readout_W=np.zeros((8, 2), np.float32), readout_b=np.zeros(8, np.float32),
              readout_from=[0], readout_kind="pooled")
    T = 128
    o = arena.dlnn_ops(m)
    assert o["adds"] == 3 * T + 2 * T + 2 * (T - 1) + 16 + 8
    assert o["mults"] == 2 * T + 2 + 16
    assert o["nonlin"] == 2 * T


def test_mlp_cnn_counts_match_parameter_formula():
    mlp = arena.MLP((4,))
    assert arena.n_params(mlp) == 256 * 4 + 4 + 4 * 8 + 8
    assert mlp.count()["mults"] == 256 * 4 + 4 * 8
    cnn = arena.CNN()
    assert cnn.count()["nonlin"] == 3 * 8 * 128


def test_cumulant_features_shape_and_finite():
    ds = modrec.generate(64, seed=5)
    f = arena.features(ds["x"])
    assert f.shape == (64, len(arena.FEATURE_NAMES)) and np.isfinite(f).all()
    assert arena.feature_ops()["mults"] > 0


def test_shipped_board_models_load_and_gate():
    p = MODELS / "tdlnet_b_w8_ternary.npz"
    if not p.exists():
        pytest.skip("model not trained")
    m = load_model(p)
    assert all(l.dilation == 1 for l in m.layers) and len(m.layers) == 3 and m.layers[0].W.shape == (8, 2, 8)
    rep = json.loads((MODELS / "tdlnet_b_w8_report.json").read_text())["ternary"]
    assert rep["acc"] >= 0.70 and rep["acc_ge10dB"] >= 0.75       # M1b gate (reported on the fixed test set)
    ds = modrec.dataset(n_train=800, n_test=800)
    out = virtual.forward(m, ds["x_test"][:64])["out"]
    net = TDLNet.from_config(m.meta["arch"])
    net.load_state_dict(torch.load(MODELS / "tdlnet_b_w8_ternary.pt"))
    net.eval()
    with torch.no_grad():
        ref = net(torch.from_numpy(ds["x_test"][:64])).numpy()
    assert np.abs(out - ref).max() < 1e-3


def test_train_deterministic_tiny():
    ds = modrec.dataset(n_train=800, n_test=200)
    kw = dict(width=8, depth=3, dilations=(1, 1, 1), log=lambda s: None)
    a, _ = train.train_one(ds, False, 2, 256, 3e-2, 0, True, False, 1.0, **kw)
    b, _ = train.train_one(ds, False, 2, 256, 3e-2, 0, True, False, 1.0, **kw)
    assert np.array_equal(train.predict(a, ds["x_test"]), train.predict(b, ds["x_test"]))


def test_aware2_model_loads_in_virtual_tier_and_line_aug_sane():
    p = MODELS / "tdlnet_b_w8_aware2_ternary.npz"
    if not p.exists():
        pytest.skip("model not trained")
    m = load_model(p)
    assert len(m.layers) == 3 and m.layers[0].W.shape == (8, 2, 8) and all(l.dilation == 1 for l in m.layers)
    assert m.meta["line_fir_provenance"]["netlist_sha256"] and m.meta["line_aug"]["tap_delay_jitter_Ts"] == 0.14
    rep = json.loads((MODELS / "tdlnet_b_w8_aware2_report.json").read_text())
    assert rep["cells"]["tdlnet_b_w8_aware2|Q10"]["acc"] >= 0.75 and rep["cells"]["tdlnet_b_w8_aware2|Q30"]["acc"] >= 0.75
    ds = modrec.dataset(n_train=800, n_test=800)
    out = virtual.forward(m, ds["x_test"][:32])["out"]
    net = TDLNet.from_config(dict(m.meta["arch"], line_fir=None))      # V tier = ideal line
    net.load_state_dict(torch.load(MODELS / "tdlnet_b_w8_aware2_ternary.pt"))
    net.eval()
    with torch.no_grad():
        ref = net(torch.from_numpy(ds["x_test"][:32])).numpy()
    assert np.abs(out - ref).max() < 1e-3


def test_line_aug_identity_and_jitter():
    from dlnn.model.tdlnet_torch import LineAug
    P, M, K = 8, 12, 8
    dense = np.zeros((2, K, (M + 1) * P + 1))
    for k in range(K):
        dense[:, k, P + k * P] = 1.0                              # tap k = ideal delay k on the dense grid
    a = LineAug(dense, P, M, jitter=0.14)
    assert torch.allclose(a.at(0.3), torch.eye(K, M), atol=1e-6)
    f = a.sample()
    assert f.shape == (K, M) and torch.isfinite(f).all()
    j = json.loads((ROOT / "models/board_aware/line_fir_m1a.json").read_text())
    assert set(j["corners"]) == {"Q10", "Q30"} and j["provenance"]["ngspice"].startswith("ngspice")


def test_arena_report_mlp_rows_match_npz():
    """Every MLP row of arena_report.json describes its npz: parameter count, exact op count and (for the fast raw rows)
    the architecture; the folded-input npz reproduces the stored accuracy on a subset of the fixed test set."""
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    rep = json.loads((root / "models/arena/arena_report.json").read_text())["models"]
    rows = [k for k in rep if k.startswith("mlp_")]
    assert {"mlp_s", "mlp_l", "mlp_s_v1", "mlp_l_v1"} <= set(rows)
    for k in rows:
        z = np.load(root / rep[k]["file"], allow_pickle=False)
        names = [str(n) for n in z["layer_names"]]
        ws = [z[n.replace(".", "_")] for n in names if n.endswith("weight")]
        bs = [z[n.replace(".", "_")] for n in names if n.endswith("bias")]
        assert sum(w.size for w in ws) + sum(b.size for b in bs) == rep[k]["params"], k
        net = dict(mults=sum(w.size for w in ws), adds=sum(w.size for w in ws) + sum(b.size for b in bs),
                   nonlin=sum(w.shape[0] for w in ws[:-1]))
        r = rep[k].get("net_ops", rep[k]["ops"])
        assert r == net, k
        if "feature_ops" in rep[k]:
            assert rep[k]["ops"] == arena.ops_sum(net, rep[k]["feature_ops"]), k
        assert ws[-1].shape[0] == 8 and ws[0].shape[1] == (256 if "feature_set" not in z.files else ws[0].shape[1])
    ds = modrec.generate(800, seed=modrec.SEED + 1)           # same generator/seed as the fixed test set (first 800 + balance differs)
    for k in rows:
        pred, _ = arena.predict_mlp_npz(root / rep[k]["file"])
        acc = float(np.mean(pred(ds["x"]).argmax(1) == ds["y"]))
        assert abs(acc - rep[k]["acc"]) < 0.06, (k, acc, rep[k]["acc"])      # 800-sample subset: loose binomial tolerance


@pytest.mark.parametrize("stem", ["tdlnet_b_w8_aware2", "tdlnet_b_w8_aware2_fro"])
def test_aware2_shipped_no_class_collapse_on_q10_line(stem):
    """Shipped board-aware nets must not drop a class: re-evaluates the .pt on the M1a Q=10 line (full test set); min per-class >= 0.35, mixed >= 0.79."""
    p = MODELS / f"{stem}_ternary.npz"
    if not p.exists():
        pytest.skip("model not shipped")
    m = load_model(p)
    j = json.loads((ROOT / "models/board_aware/line_fir_m1a.json").read_text())
    net = TDLNet.from_config(dict(m.meta["arch"], line_fir=None))
    net.load_state_dict(torch.load(MODELS / f"{stem}_ternary.pt"))
    net.set_line_fir(j["corners"]["Q10"]["fir"]); net.eval()
    ds = modrec.dataset()
    r = train.summarize(train.predict(net, ds["x_test"]), ds["y_test"], ds["snr_test"], ds["classes"])
    assert min(r["acc_per_class"].values()) >= 0.35, r["acc_per_class"]
    assert r["acc"] >= 0.79
    rep = json.loads((MODELS / f"{stem}_report.json").read_text())
    assert abs(rep["cells"][f"{stem}|Q10"]["acc"] - r["acc"]) < 1e-9


def test_aware2_collapsed_kept_for_the_record():
    p = MODELS / "tdlnet_b_w8_aware2_collapsed_ternary.npz"
    if not p.exists():
        pytest.skip("not present")
    rep = json.loads((MODELS / "tdlnet_b_w8_aware2_report.json").read_text())
    assert min(rep["cells"]["tdlnet_b_w8_aware2_collapsed|Q10"]["acc_per_class"].values()) == 0.0
