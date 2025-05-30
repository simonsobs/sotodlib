# This code has been ported over from enlib: 
# https://github.com/amaurea/enlib
# Ported over on 10/31/2023
# The original code is public domain. Modifications and additions 
# are subject to the main license in this repository.

# -*- coding: utf-8 -*-
"""This module provides a simple class for benchmarking selected parts of the
code during regular execution of a program. Usage:
    with mark("category"):
        statements
will record the time and memory usage of statement each time it is
hit in the normal course of execution of the program. For each
category, the mean and standard deviation of time and memory
are computed. The statistics are available as the module-global
stats object. str(stats) will output something like this:

      n        time          cpu           mem           leak
            mean    std   mean    std   mean    std   mean    std
 foo  200   0.03  0.020   0.03  0.021   0.76   0.00   0.00  0.000
 bar 1010   0.00  0.015   0.00  0.016   0.76   0.00   0.00  0.000

The individual entries are also avilable by indexing stats like
a dictionary: stats["foo"]["time"].mean, .std, .n, .last
return the mean, standard deviation, number of hits and
last value for the execution time for category "foo", for example.
"""
import time
import numpy as np
from collections import defaultdict
from pixell import memory
try:
    clockfun = time.clock
except ImportError:
    clockfun = time.process_time

class Value:
    def __init__(self, n=0, v=0, vv=0):
        self.n  = n
        self.v  = v
        self.vv = vv
        self.last = v
    def add(self,v):
        self.n  += 1
        self.v  += v
        self.vv += v**2
        self.last = v
    @property
    def mean(self): return self.v/self.n if self.n > 0 else np.nan
    @property
    def std(self):
        if self.n == 0: return np.nan
        var = self.vv/self.n - (self.v/self.n)**2
        if var < 0: return 0.0
        else: return var**0.5
    def __repr__(self): return "Value(mean=%f, std=%f, n=%d)" % (self.mean, self.std, self.n)

class Entry(defaultdict):
    def __init__(self):
        defaultdict.__init__(self, Value)
    def __repr__(self):
        names = sorted(self)
        return "Entry("+", ".join(["%s=%s" % (name,str(self[name])) for name in names]) + ")"

class Register(defaultdict):
    def __init__(self, fmt=[("time","%6.2f","%6.3f",1),("cpu","%6.2f","%6.3f",1),("mem","%6.2f","%6.2f",2.0**30),("leak","%6.2f","%6.3f",2.0**30)]):
        defaultdict.__init__(self, Entry)
        self.info = fmt
    def add(self, name, *args):
        entry = self[name]
        for info,v in zip(self.info,args):
            entry[info[0]].add(v)
    def get(self, name):
        return self[name]["time"].mean
    def __repr__(self):
        # Sort by first ordering criterion
        if len(self) == 0: return ""
        nhit = np.array([self[name][self.info[0][0]].n    for name in self])
        vals = np.array([self[name][self.info[0][0]].mean for name in self])
        inds = np.argsort(nhit*vals)[::-1]
        names = list(self)
        names = [names[i] for i in inds]
        nhit  = nhit[inds]
        # For pretty output, determine ranges of each column
        name_dig = max([len(name) for name in names])
        nhit_dig = max([len("%d"%i) for i in nhit])
        lines = []
        # Header
        pre = " "*name_dig
        line1,line2 = pre + " %s" % "n".center(nhit_dig), pre + " "*(1+nhit_dig)
        for info in self.info:
            lens = [len(i%0) for i in info[1:3]]
            line1 += " " + info[0].center(sum(lens)+1)
            line2 += " " + " ".join("%*s" % (l,v) for l,v in zip(lens,["mean","std"]))
        lines.append(line1)
        lines.append(line2)
        for name,nh in zip(names,nhit):
            entry = self[name]
            line = "%-*s %*d" % (name_dig,name,nhit_dig,nh)
            for info in self.info:
                val = entry[info[0]]
                unit= float(info[3] if len(info) > 3 else 1)
                line += (" " + info[1] + " " + info[2]) % (val.mean/unit, val.std/unit)
            lines.append(line)
        return "\n".join(lines)
    def write(self, fname):
        with open(fname,"w") as f:
            f.write(str(self)+"\n")

stats = Register()

class mark:
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        self.time1  = time.time()
        self.clock1 = clockfun()
        self.mem1   = memory.current()
    def __exit__(self, type, value, traceback):
        self.time2  = time.time()
        self.clock2 = clockfun()
        self.mem2   = memory.current()
        stats.add(self.name, self.time2 -self.time1, self.clock2-self.clock1, self.mem1, self.mem2-self.mem1)

class show:
    def __init__(self, name, display=True):
        self.name = name
        self.display = display
    def __enter__(self):
        self.time1  = time.time()
        self.clock1 = clockfun()
        self.mem1   = memory.current()
    def __exit__(self, type, value, traceback):
        self.time2  = time.time()
        self.clock2 = clockfun()
        self.mem2   = memory.current()
        self.memmax = memory.max()
        if self.display:
            print("%5.2f %5.2f %5.2f %s" % (self.time2-self.time1, self.mem2*1e-9, self.memmax*1e-9, self.name))

class dummy:
    def __init__(self, name): pass
    def __enter__(self): pass
    def __exit__(self, type, value, traceback): pass
