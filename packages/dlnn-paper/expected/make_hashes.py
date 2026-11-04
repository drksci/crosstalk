"""Write expected/hashes.json: sha256 of every shipped model file (+ the model's canonical dlnn hash) and of the exported tile netlists.
Run from the repo root:  python expected/make_hashes.py   (00_verify.ipynb checks the files against this list)"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dlnn.model.npz import load_model

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
out = {"schema": "dlnn-hashes/1", "note": "sha256 of the files shipped in this repo; 'model_sha256' is dlnn.model.npz.Model.sha256() (what the run JSONs quote).", "models": {}, "netlists": {}, "data_reference": {}}
for p in sorted((ROOT / "models").rglob("*.npz")):
    rel = str(p.relative_to(ROOT))
    try:
        ms = load_model(p).sha256()
    except Exception:
        ms = None                       # arena npz files are not dlnn-model/1
    out["models"][rel] = {"sha256": sha(p), "model_sha256": ms}
for p in sorted((ROOT / "hw" / "board" / "tile").glob("layer*/tile.cir")):
    out["netlists"][str(p.relative_to(ROOT))] = sha(p)
out["netlists"]["hw/board/params.json"] = sha(ROOT / "hw/board/params.json")
for p in sorted((ROOT / "expected" / "data").glob("*.json")):
    out["data_reference"][str(p.relative_to(ROOT))] = sha(p)
(ROOT / "expected" / "hashes.json").write_text(json.dumps(out, indent=1))
print(len(out["models"]), "models,", len(out["netlists"]), "netlist/param files hashed")
