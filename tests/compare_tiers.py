#!/usr/bin/env python3
"""Compare the dumps written by kernel_tiers for different CPU tiers."""
import sys, struct
def load(p):
    d = open(p,'rb').read(); return struct.unpack('<%df' % (len(d)//4), d)
a = load(sys.argv[1]); worst = 0
for p in sys.argv[2:]:
    b = load(p); assert len(a)==len(b), "different output sizes"
    m = max(abs(x) for x in a) or 1
    e = max(abs(x-y) for x,y in zip(a,b))/m
    print("%s vs %s: max rel diff %.2e" % (sys.argv[1], p, e)); worst = max(worst, e)
sys.exit(0 if worst < 5e-2 else 1)
