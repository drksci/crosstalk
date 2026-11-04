"""KiCad 9 kicadsexpr netlist -> unplaced .kicad_pcb via kinet2pcb 1.1.4.
Two small workarounds are needed with KiCad 9.0.9 + kinet2pcb 1.1.4:
 1. kinet2pcb's parser trips on the (property (name ..) (value ..)) children of each (comp) -> strip them first;
 2. headless pcbnew.FootprintLoad() has no plugin -> route it through PCB_IO_MGR.
Usage: netlist_to_pcb.py in.net out.kicad_pcb  (run with hw/.venv-eda/bin/python)"""
import re, sys, tempfile, os
import pcbnew
import importlib
k2p = importlib.import_module('kinet2pcb.kinet2pcb')

def _fp_load(libdir, name):
    return pcbnew.PCB_IO_MGR.PluginFind(pcbnew.PCB_IO_MGR.KICAD_SEXP).FootprintLoad(libdir, name)

def main(net_in, pcb_out):
    t = open(net_in).read()
    t = re.sub(r'\n\s*\(property \(name "[^"]*"\) \(value "[^"]*"\)\)', '', t)
    with tempfile.NamedTemporaryFile("w", suffix=".net", delete=False) as f:
        f.write(t)
    pcbnew.FootprintLoad = _fp_load
    k2p.pcbnew.FootprintLoad = _fp_load
    k2p.kinet2pcb(f.name, pcb_out, None)
    os.unlink(f.name)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
