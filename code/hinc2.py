"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

import numpy as np

import hinc
import thinkplot
import thinkstats2


def InterpolateSample(df, log_upper=6.0):
    """Makes a sample of log10 household income.

    Assumes that log10 income is uniform in each range.

    df: DataFrame with columns income and freq
    log_upper: log10 of the assumed upper bound for the highest range

    returns: NumPy array of log10 household income
    """
    # compute the log10 of the upper bound for each range
    df['log_upper'] = np.log10(df.income)

    # get the lower bounds by shifting the upper bound and filling in
    # the first element
    df['log_lower'] = df.log_upper.shift(1)
    df.log_lower[0] = 3.0

    # plug in a value for the unknown upper bound of the highest range
    df.log_upper[41] = log_upper

    # use the freq column to generate the right number of values in
    # each range
    arrays = []
    for _, row in df.iterrows():
        vals = np.linspace(row.log_lower, row.log_upper, row.freq)
        arrays.append(vals)

    # collect the arrays into a single sample
    log_sample = np.concatenate(arrays)
    return log_sample


def main():
    df = hinc.ReadData()
    log_sample = InterpolateSample(df, log_upper=6.0)

    log_cdf = thinkstats2.Cdf(log_sample)

    print("median", thinkstats2.Median(log_sample))
    print("pearson's median skewness", thinkstats2.PearsonMedianSkewness(log_sample))
    print("skewness", thinkstats2.Skewness(log_sample))
    print("mean", log_cdf.Mean())

    print("the higher our log_upper, the more right-skewed (according to g_1) or at least less left-skewed (according to g_p) things get")
    print("the mean moves to the right a bit, too.")

    print("proportion of the population with income < mean", log_cdf.Prob(log_cdf.Mean()))
    print("the higher the upper bound, the greater the proprtion below the mean.")

    thinkplot.Cdf(log_cdf)
    thinkplot.Show(xlabel='household income',
                   ylabel='CDF')


if __name__ == "__main__":
    main()
