"""JS forward pass (docs/artifacts/assets/js/tdlnet.js, run in node) must match the numpy virtual tier on 40 examples."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dlnn.model.npz import load_model
from dlnn.viz import export_web

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models" / "tdlnet_modrec_ternary.npz"
NODE = shutil.which("node")


def _run_node(model_json: Path, ex_json: Path) -> dict:
    r = subprocess.run([NODE, str(ROOT / "tests" / "js" / "check_tdlnet.mjs"), str(model_json), str(ex_json)],
                       capture_output=True, text=True, timeout=120, check=True)
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.skipif(NODE is None, reason="node not installed")
@pytest.mark.skipif(not MODEL.exists(), reason="models/tdlnet_modrec_ternary.npz not present")
def test_js_forward_matches_numpy(tmp_path):
    paths = export_web.export_modrec(MODEL, tmp_path, n=40)
    res = _run_node(paths["model"], paths["examples"])
    assert res["n"] == 40
    assert res["max_abs_err"] <= 1e-5, res
    assert res["top1_agree"] == 40, res


def test_model_json_roundtrip(tmp_path):
    m = load_model(MODEL)
    j = export_web.model_to_json(m, "t")
    assert j["sha256"] == m.sha256() and j["n_layers"] == len(m.layers)
    assert np.frombuffer(__import__("base64").b64decode(j["layers"][0]["W"]), np.int8).size == m.layers[0].W.size


# ----------------------------------------------------------------------------- arena (M4a)
ARENA_DIR = ROOT / "models" / "arena"


@pytest.mark.skipif(not ARENA_DIR.exists(), reason="models/arena not present")
def test_arena_export_and_js_forward(tmp_path):
    """Every arena model's JS forward pass matches the numpy logits within 1e-5 and the JS signal generator matches the
    Python generator's per-class feature statistics (tests/js/check_arena.mjs)."""
    if NODE is None:
        pytest.skip("node not installed")
    paths = export_web.export_arena(ROOT / "models", tmp_path, n=40)
    assert {"models", "examples", "gen_stats"} <= set(paths)
    r = subprocess.run([NODE, str(ROOT / "tests" / "js" / "check_arena.mjs"), str(tmp_path)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-1000:]
    res = json.loads(r.stdout)
    assert res["ok"] and all(m["max_abs_err"] <= 1e-5 for m in res["models"].values())


@pytest.mark.skipif(not ARENA_DIR.exists(), reason="models/arena not present")
def test_arena_numpy_reference_matches_torch():
    """The numpy reference forward passes used to produce the expected logits agree with the torch models they re-implement."""
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F
    from dlnn.data import modrec
    x = modrec.generate(40, 5)["x"]
    for mid in ("mlp_s", "mlp_l", "cnn"):
        z = np.load(ARENA_DIR / f"{mid}.npz")
        h = torch.from_numpy(x)
        if mid == "cnn":
            for i in range(3):
                h = torch.relu(F.conv1d(F.pad(h, (7, 0)), torch.from_numpy(z[f"cv_{i}_weight"]), torch.from_numpy(z[f"cv_{i}_bias"])))
            ref = F.linear(h.mean(2), torch.from_numpy(z["fc_weight"]), torch.from_numpy(z["fc_bias"])).numpy()
        else:
            h = h.flatten(1)
            n = len([k for k in z.files if k.endswith("_weight")])
            for i in range(n):
                h = F.linear(h, torch.from_numpy(z[f"fc_{i}_weight"]), torch.from_numpy(z[f"fc_{i}_bias"]))
                h = torch.relu(h) if i < n - 1 else h
            ref = h.numpy()
        got = export_web.reference_logits(mid, ARENA_DIR / f"{mid}.npz", x)
        assert np.max(np.abs(got - ref)) < 1e-3, mid
