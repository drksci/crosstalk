"""M2c criterion re-measured on the TRAINED nets: R tier, preset=ideal (0 dB/us, ideal couplers, no mismatch/crosstalk, NF 0, ideal ADC),
per-stage SNR 30 dB, top-1 agreement vs V, 4000 test examples, 3 noise seeds. Also noiseless and 40 dB, and 30 dB input-AWGN V(noisy) vs V(clean)."""
import sys, json, time
import numpy as np
from dlnn.data import modrec
from dlnn.model.npz import load_model
from dlnn.tiers import rf, virtual, emulated
from dlnn import sweeps
ds = modrec.dataset(); x = ds["x_test"]; y = ds["y_test"]
res = {}
for name in sys.argv[1:]:
    m = load_model(name); pv = virtual.forward(m, x)["out"].argmax(1)
    r = {"V_acc": float(np.mean(pv==y))}
    for snr in (None, 40.0, 30.0):
        ag, ac = [], []
        for seed in (0,1,2):
            o = rf.forward(m, x, {"preset":"ideal","snr_db":snr,"seed":seed})["out"].argmax(1)
            ag.append(float(np.mean(o==pv))); ac.append(float(np.mean(o==y)))
            if snr is None: break
        r[f"stage_snr_{snr}"] = {"agree_per_seed": ag, "agree_mean": float(np.mean(ag)), "acc_mean": float(np.mean(ac))}
    xn = sweeps.add_awgn(x, 30.0, np.random.default_rng(1))
    r["V_input_awgn30_vs_Vclean"] = float(np.mean(virtual.forward(m, xn)["out"].argmax(1)==pv))
    res[name] = r; print(name, json.dumps(r), flush=True)
