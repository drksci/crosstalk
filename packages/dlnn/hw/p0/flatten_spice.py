"""Inline the .include of the KiCad spice export, add the P0 stimulus and a .tran -> demo_flat.cir
(the identical netlist is run by ngspice batch and by eecircuit-engine in the browser).
usage: flatten_spice.py demo.cir models/lc_parts.lib demo_flat.cir"""
import re, sys
net, lib, out = sys.argv[1:4]
t = open(net).read()
t = re.sub(r'^\.include.*$', open(lib).read().rstrip(), t, flags=re.M)
t = re.sub(r'^\.end\s*$', '', t, flags=re.M)
t += """* stimulus + analysis (P0): 0->1 V step, 5 ns edges, delayed 100 ns
Vstim /VDRV 0 PULSE(0 1 100n 5n 5n 4u 8u)
.tran 2n 3u
.end
"""
open(out, "w").write(t)
