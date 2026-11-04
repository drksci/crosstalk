"""Compare eecircuit-engine (browser) vs ngspice batch waveforms. usage: compare_spice.py tran_ngspice.txt eecircuit_tran.json"""
import json, sys
import numpy as np
ng = np.loadtxt(sys.argv[1]); ee = json.load(open(sys.argv[2]))
names = ["v(/n1)", "v(/n2)", "v(/t1)", "v(/t2)", "v(/out)"]
t_ng = ng[:, 0]; t_ee = np.array(ee["data"]["time"])
out = {"points_ngspice": len(t_ng), "points_eecircuit": len(t_ee), "max_dt_s": float(np.max(np.abs(t_ng - t_ee))) if len(t_ng) == len(t_ee) else None,
       "init_ms": ee["init_ms"], "run_ms": ee["run_ms"], "wall_ms": ee["wall_ms"], "signals": {}}
def cross(t, v, lvl=0.5):
    i = np.where((v[:-1] < lvl) & (v[1:] >= lvl))[0][0]
    return t[i] + (lvl - v[i]) / (v[i + 1] - v[i]) * (t[i + 1] - t[i])
for k, n in enumerate(names):
    a = ng[:, 2 * k + 1]; b = np.interp(t_ng, t_ee, np.array(ee["data"][n]))
    d = {"max_abs_err_V": float(np.max(np.abs(a - b))), "peak_V": float(np.max(np.abs(a)))}
    if n != "v(/out)":
        d["t50_ngspice_ns"] = cross(t_ng, a) * 1e9; d["t50_eecircuit_ns"] = cross(t_ng, b) * 1e9
        d["t50_err_ps"] = (d["t50_ngspice_ns"] - d["t50_eecircuit_ns"]) * 1e3
    out["signals"][n] = d
out["tap_spacing_t50_ns"] = out["signals"]["v(/n2)"]["t50_ngspice_ns"] - out["signals"]["v(/n1)"]["t50_ngspice_ns"]
print(json.dumps(out, indent=1)); json.dump(out, open("compare_spice.json", "w"), indent=1)
