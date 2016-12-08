"""This file contains code used in "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""

from __future__ import print_function

import thinkstats2
import thinkplot

import math
import random
import numpy as np


def MeanError(estimates, actual):
    """Computes the mean error of a sequence of estimates.

    estimate: sequence of numbers
    actual: actual value

    returns: float mean error
    """
    errors = [estimate-actual for estimate in estimates]
    return np.mean(errors)


def RMSE(estimates, actual):
    """Computes the root mean squared error of a sequence of estimates.

    estimate: sequence of numbers
    actual: actual value

    returns: float RMSE
    """
    e2 = [(estimate-actual)**2 for estimate in estimates]
    mse = np.mean(e2)
    return math.sqrt(mse)


def Estimate1(n=7, m=1000):
    """Evaluates RMSE of sample mean and median as estimators.

    n: sample size
    m: number of iterations
    """
    mu = 0
    sigma = 1

    means = []
    medians = []
    for _ in range(m):
        xs = [random.gauss(mu, sigma) for _ in range(n)]
        xbar = np.mean(xs)
        median = np.median(xs)
        means.append(xbar)
        medians.append(median)

    print('Experiment 1')
    print('rmse xbar', RMSE(means, mu))
    print('rmse median', RMSE(medians, mu))


def Estimate2(n=7, m=1000):
    """Evaluates S and Sn-1 as estimators of sample variance.

    n: sample size
    m: number of iterations
    """
    mu = 0
    sigma = 1

    estimates1 = []
    estimates2 = []
    for _ in range(m):
        xs = [random.gauss(mu, sigma) for _ in range(n)]
        biased = np.var(xs)
        unbiased = np.var(xs, ddof=1)
        estimates1.append(biased)
        estimates2.append(unbiased)

    print('Experiment 2')
    print('mean error biased', MeanError(estimates1, sigma**2))
    print('mean error unbiased', MeanError(estimates2, sigma**2))

def ex1(n=7, m=1000):
    """
    Calculate the MSE for the sample mean and median as estimators of the true mean
    and the RMSE of the biased and unbiased sample variance as estimators of the true variance.

    Note that the MSE for the sample mean converges to 0 as m approaches positive infinity,
    whereas the RMSE of both biased and unbiased sample variance do not converge to 0 as
    m increases.
    """

    mu = 0
    sigma = 1

    means = []
    medians = []
    var_biased = []
    var_unbiased = []

    for _ in range(m):
        xs = [random.gauss(mu, sigma) for _ in range(n)]
        xbar = np.mean(xs)
        median = np.median(xs)
        means.append(xbar)
        medians.append(median)
        var_biased.append(np.var(xs))
        var_unbiased.append(np.var(xs, ddof=1))

    print('Exercise 8.1')
    print('mean error of the mean', MeanError(means, mu))
    print('mean error of the median', MeanError(medians, mu))
    print('RMSE biased S', RMSE(var_biased, sigma))
    print('RMSE unbiased S', RMSE(var_unbiased, sigma))

def Estimate3(n=7, m=1000):
    """Evaluates L and Lm as estimators of the exponential parameter.

    n: sample size
    m: number of iterations
    """
    lam = 2

    means = []
    medians = []
    for _ in range(m):
        xs = np.random.exponential(1.0/lam, n)
        L = 1 / np.mean(xs)
        Lm = math.log(2) / np.median(xs)
        means.append(L)
        medians.append(Lm)

    print('Experiment 3')
    print('rmse L', RMSE(means, lam))
    print('rmse Lm', RMSE(medians, lam))
    print('mean error L', MeanError(means, lam))
    print('mean error Lm', MeanError(medians, lam))

def ex2_helper(n=10, m=1000):
    lam = 2
    estimates = []
    for _ in range(m):
        xs = np.random.exponential(1.0/lam, n)
        L = 1 / np.mean(xs)
        estimates.append(L)
    return estimates

def ex2(n=10, m=1000):
    def VertLine(x, y=1):
        thinkplot.Plot([x, x], [0, y], color='0.8', linewidth=3)

    print("Exericse 8.2")
    lam = 2
    estimates = ex2_helper()
    cdf = thinkstats2.Cdf(estimates)
    print('90% confidence interval is', cdf.Percentile(5), cdf.Percentile(95))
    print('RMSE', RMSE(estimates, lam))
    thinkplot.Cdf(cdf)
    VertLine(cdf.Percentile(5))
    VertLine(cdf.Percentile(95))
    thinkplot.show(xlabel='lambda', ylabel='probability')

    errors = []
    for n in np.arange(5, 55, 5):
        estimates = ex2_helper(n=n)
        errors.append(RMSE(estimates, lam))
    thinkplot.Plot(np.arange(5, 55, 5), errors)
    thinkplot.show(xlabel='n', ylabel='RMSE')

def SimulateSample(mu=90, sigma=7.5, n=9, m=1000):
    """Plots the sampling distribution of the sample mean.

    mu: hypothetical population mean
    sigma: hypothetical population standard deviation
    n: sample size
    m: number of iterations
    """
    def VertLine(x, y=1):
        thinkplot.Plot([x, x], [0, y], color='0.8', linewidth=3)

    means = []
    for _ in range(m):
        xs = np.random.normal(mu, sigma, n)
        xbar = np.mean(xs)
        means.append(xbar)

    stderr = RMSE(means, mu)
    print('standard error', stderr)

    cdf = thinkstats2.Cdf(means)
    ci = cdf.Percentile(5), cdf.Percentile(95)
    print('confidence interval', ci)
    VertLine(ci[0])
    VertLine(ci[1])

    # plot the CDF
    thinkplot.Cdf(cdf)
    thinkplot.Save(root='estimation1',
                   xlabel='sample mean',
                   ylabel='CDF',
                   title='Sampling distribution')

def SimulateGame(lam=4):
    goals = 0
    time = 0.0
    while time < 1.0:
        time_to_next_goal = np.random.exponential(1.0/lam)
        goals += 1
        time += time_to_next_goal
    return goals

def ex3():
    def VertLine(x, y=1):
        thinkplot.Plot([x, x], [0, y], color='0.8', linewidth=3)
    
    lam = 4
    goal_totals = [SimulateGame(lam=lam) for _ in range(1000)]
    print('RMSE', RMSE(goal_totals, lam))
    hist = thinkstats2.Hist(goal_totals)
    cdf = thinkstats2.Cdf(goal_totals)
    thinkplot.PrePlot(rows=2, cols=2)
    thinkplot.SubPlot(1)
    thinkplot.Hist(hist)
    thinkplot.SubPlot(2)
    thinkplot.Cdf(cdf)
    VertLine(cdf.Percentile(5))
    VertLine(cdf.Percentile(95))
    thinkplot.SubPlot(3)

    # lambda vs. rmse
    # rmse goes up as lambda goes up
    lams = range(1, 15)
    rmses = [RMSE([SimulateGame(lam=l) for _ in range(1000)], l) for l in lams]
    thinkplot.Plot(lams, rmses)
    thinkplot.SubPlot(4)

    # m vs. rmse
    # maybe rmse very slowly goes down as m goes up?
    # not at all clear that's really the case...
    ms = np.arange(10, 1000, 10)
    rmses = [RMSE([SimulateGame() for _ in range(m)], 4) for m in ms]
    thinkplot.Plot(ms, rmses)

    thinkplot.show()


def main():
    thinkstats2.RandomSeed(17)

    # Estimate1()
    # Estimate2()
    # Estimate3(m=1000)
    # SimulateSample()
    # ex1()
    # ex2()
    ex3()


if __name__ == '__main__':
    main()
