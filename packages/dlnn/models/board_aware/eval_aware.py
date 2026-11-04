"""Evaluate w8 models on the ideal line and on the measured LC line (FIR from ngspice). python models/board_aware/eval_aware.py"""
import json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from dlnn.data import modrec
from dlnn.model import train
from dlnn.model.tdlnet_torch import TDLNet

ds = modrec.dataset()
fir = json.loads((ROOT / "models/board_aware/line_fir.json").read_text())["fir"]
out = {}
for name in ("tdlnet_b_w8", "tdlnet_b_w8_aware"):
    npz = np.load(ROOT / f"models/{name}_ternary.npz")
    cfg = json.loads(str(npz["meta_json"]))["arch"]
    sd = torch.load(ROOT / f"models/{name}_ternary.pt")
    for line in ("ideal", "measured"):
        c = dict(cfg); c["line_fir"] = fir if line == "measured" else None
        net = TDLNet.from_config(c); net.load_state_dict(sd); net.eval()
        r = train.summarize(train.predict(net, ds["x_test"]), ds["y_test"], ds["snr_test"], ds["classes"])
        out[f"{name}|{line}_line"] = dict(acc=r["acc"], acc_ge10dB=r["acc_ge10dB"], acc_per_snr=r["acc_per_snr"])
        print(f"{name:20s} on {line:8s} line: acc {r['acc']:.4f}  >=10dB {r['acc_ge10dB']:.4f}")
(ROOT / "models/board_aware/eval_aware.json").write_text(json.dumps(out, indent=1))
