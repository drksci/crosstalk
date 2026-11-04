"""Read a KiCad kicadsexpr netlist (from kicad-cli sch export netlist) into plain dicts."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import kisch  # noqa: E402


def read_net(path):
    t = kisch.parse(Path(path).read_text())
    comps = {}
    for c in kisch.find(kisch.find(t, "components")[0], "comp"):
        ref = str(kisch.find(c, "ref")[0][1])
        d = dict(ref=ref, value=str(kisch.find(c, "value")[0][1]), fp=str(kisch.find(c, "footprint")[0][1]), dnp=False,
                 fields={}, exclude_bom=False)
        for p in kisch.find(c, "property"):
            pname = str(p[1][1]) if isinstance(p[1], list) else str(p[1])
            if pname == "dnp":
                d["dnp"] = True
            elif pname == "exclude_from_bom":
                d["exclude_bom"] = True
        fl = kisch.find(c, "fields")
        if fl:
            for f in kisch.find(fl[0], "field"):
                d["fields"][str(kisch.find(f, "name")[0][1]) if False else str(f[1][1])] = str(f[2]) if len(f) > 2 else ""
        comps[ref] = d
    nets = {}
    for n in kisch.find(kisch.find(t, "nets")[0], "net"):
        name = str(kisch.find(n, "name")[0][1])
        nets[name] = [(str(kisch.find(nd, "ref")[0][1]), str(kisch.find(nd, "pin")[0][1])) for nd in kisch.find(n, "node")]
    return comps, nets


if __name__ == "__main__":
    c, n = read_net(sys.argv[1])
    print(len(c), len(n))
    print(c["RXP_0_0_0"], list(n)[:5])
