"""M1 model tests: ternary quantiser/STE, torch <-> virtual-tier equivalence, export schema, float baseline."""
# ruff: noqa: E402  (torch is importorskip-ed before the dlnn.model imports that need it)
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed (dlnn[train] extra)")

from dlnn.data import modrec
from dlnn.model import export, train
from dlnn.model.npz import SCHEMA, load_model
from dlnn.model.tdlnet_torch import TDLNet
from dlnn.model.ternary import fake_quant, ternarize, ternary_ste
from dlnn.tiers import emulated, virtual

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"


@pytest.fixture(scope="module")
def small_ds():
    return modrec.dataset(n_train=2000, n_test=400)


@pytest.fixture(scope="module")
def tiny_nets(small_ds):
    """Quickly trained (2 epochs) ternary (with per-line gain, to cover that path) and float nets."""
    tern, _ = train.train_one(small_ds, False, 2, 256, 3e-3, 0, True, True, 1.0, log=lambda s: None)
    flt, _ = train.train_one(small_ds, True, 2, 256, 3e-3, 0, True, False, 1.0, log=lambda s: None)
    return tern, flt


def test_ternarize_rule():
    W = torch.randn(8, 8, 8)
    Wt, gamma = ternarize(W)
    assert set(Wt.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert torch.allclose(gamma, W.abs().mean())
    assert torch.equal(Wt, torch.clamp(torch.round(W / gamma), -1, 1))


def test_ste_gradient_is_identity():
    W = torch.randn(4, 3, 8, requires_grad=True)
    W_eff, Wt, gamma = ternary_ste(W)
    assert torch.allclose(W_eff, gamma * Wt)
    (W_eff * 2.0).sum().backward()
    assert torch.allclose(W.grad, torch.full_like(W, 2.0))


def test_fake_quant_grid():
    q = fake_quant(torch.linspace(-2, 2, 1001), 1.0, 8)
    assert q.abs().max() <= 1.0 + 1e-6
    assert len(torch.unique(q)) == 255


def test_causality():
    net = TDLNet()
    x = torch.randn(2, 2, 128)
    y1 = net(x, return_layers=True)[1][-1]
    x2 = x.clone()
    x2[:, :, 100:] += 1.0                     # change the future only
    y2 = net(x2, return_layers=True)[1][-1]
    assert torch.allclose(y1[:, :, :100], y2[:, :, :100])


def test_export_schema_and_virtual_reproduces_torch(tiny_nets, small_ds, tmp_path):
    net, _ = tiny_nets
    p = tmp_path / "t.npz"
    export.save_ternary(net, p, {"t": 1})
    with np.load(p, allow_pickle=False) as z:
        assert str(z["schema"]) == SCHEMA and str(z["kind"]) == "tdlnet"
        assert str(z["readout_kind"]) == "pooled" and list(z["readout_from"]) == [2]
        for l, (O, I, d) in enumerate([(8, 2, 1), (8, 8, 4), (8, 8, 16)]):
            assert z[f"L{l}_W"].dtype == np.int8 and z[f"L{l}_W"].shape == (O, I, 8)
            assert set(np.unique(z[f"L{l}_W"])) <= {-1, 0, 1}
            assert int(z[f"L{l}_dilation"]) == d
    m = load_model(p)
    assert m.n_ternary_weights == 128 + 512 + 512
    x = small_ds["x_test"][:256]
    ref = train.predict(net, x, quant=False)
    out = virtual.forward(m, x)["out"]
    assert np.abs(out - ref).max() <= 1e-4


def test_emulated_tier_loads_and_runs(tiny_nets, small_ds, tmp_path):
    net, _ = tiny_nets
    m = export.save_ternary(net, tmp_path / "t.npz")
    o = emulated.forward(m, small_ds["x_test"][:64])["out"]
    assert o.shape == (64, 8) and np.isfinite(o).all()


def test_float_baseline_evaluates_in_virtual_tier(tiny_nets, small_ds, tmp_path):
    _, net = tiny_nets
    p = tmp_path / "f.npz"
    export.save_float(net, p)
    with pytest.raises(ValueError):
        load_model(p)                         # a float file is never a valid dlnn-model/1
    m = export.load_float_model(p)
    x = small_ds["x_test"][:256]
    assert np.abs(virtual.forward(m, x)["out"] - train.predict(net, x)).max() <= 1e-4


@pytest.mark.skipif(not (MODELS / "tdlnet_modrec_ternary.pt").exists(), reason="run dlnn.model.train first")
def test_shipped_models_reproduce_torch_logits():
    ds = modrec.dataset()
    x = ds["x_test"][:256]
    for kind, loader in (("ternary", load_model), ("float", export.load_float_model)):
        npz = MODELS / f"tdlnet_modrec_{kind}.npz"
        net = TDLNet.from_config(loader(npz).meta["arch"])
        net.load_state_dict(torch.load(MODELS / f"tdlnet_modrec_{kind}.pt"))
        ref = train.predict(net, x, quant=False)
        out = virtual.forward(loader(npz), x)["out"]
        assert np.abs(out - ref).max() <= 1e-4, kind
