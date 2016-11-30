"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

import sys
from operator import itemgetter

import first
import thinkstats2
import nsfg


def Mode(hist):
    """Returns the value with the highest frequency.

    hist: Hist object

    returns: value from Hist
    """
    max_val = 0
    max_count = 0
    for val, count in hist.Items():
        if count > max_count:
            max_val = val
            max_count = count
    return max_val


def AllModes(hist):
    """Returns value-freq pairs in decreasing order of frequency.

    hist: Hist object

    returns: iterator of value-freq pairs
    """
    s = sorted(hist.Items(), key=lambda pair: pair[1])
    s.reverse()
    return s

def WeightComparison():
    preg = nsfg.ReadFemPreg()

    live = preg[preg.outcome == 1]
    firsts = live[live.birthord == 1]
    others = live[live.birthord != 1]

    d = thinkstats2.CohenEffectSize(firsts.totalwgt_lb, others.totalwgt_lb)
    print(d)
    print("Still a really small effect size, though not as small as difference in pregnancy length")



def main(script):
    """Tests the functions in this module.

    script: string script name
    """
    live, firsts, others = first.MakeFrames()
    hist = thinkstats2.Hist(live.prglngth)

    # test Mode    
    mode = Mode(hist)
    print('Mode of preg length', mode)
    assert(mode == 39)

    # test AllModes
    modes = AllModes(hist)
    assert(modes[0][1] == 4693)

    for value, freq in modes[:5]:
        print(value, freq)

    print('%s: All tests passed.' % script)

    WeightComparison()

if __name__ == '__main__':
    main(*sys.argv)
