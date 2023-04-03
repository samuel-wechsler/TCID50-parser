"""
control.py

This module l
"""

import numpy as np
from scipy.stats import binom


def prob_cpe(N_0, Q, D, d):
    """
    This function calculates the probability of CPE occuring at parameters:
    N_0: count of virus particles in intial stock
    Q: particle to PFU ratio
    D: serial dilution factor
    d: dilution number in series
    """
    return 1 - np.exp(- N_0 / (Q * np.power(D, d)))


def get_outlier_rows(plate, titer, D, D_0, Q):
    """
    This function returns all rows that are considered to be outliers (i.e., p<0.05).
    D: serial dilution factor
    D_0: initial dilution factor
    Q: particle to pfu ratio
    """
    outlier_rows = []
    d = 1
    Q *= 1.5

    for row in plate:
        # get probability of CPE
        p = prob_cpe(titer*0.7 * D_0, Q, D, d)

        # get probability of current row distribution
        r = binom.pmf(k=sum(row), n=len(row), p=p)

        # if outlier, append row index
        if r < 0.05:
            outlier_rows.append(d-1)
        d += 1

    return outlier_rows
