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
    """
    outlier_rows = []
    d = 1

    for row in plate:
        # get probability of CPE
        p = prob_cpe(titer*0.7, Q, D, d + np.log10(D_0))
        # get probability of current row distribution
        r = binom.pmf(k=sum(row), n=len(row), p=p)

        # if outlier, append row index
        if r < 0.10:
            outlier_rows.append(d-1)
        d += 1

    return outlier_rows
