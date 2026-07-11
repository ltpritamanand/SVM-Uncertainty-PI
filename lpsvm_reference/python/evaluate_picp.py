"""
Python port of evaluate_PICP.m
"""

import numpy as np


def evaluate_picp(y, lo, hi):
    """PICP and MPIW — identical to evaluate_PICP.m."""
    picp = float(np.mean((y >= lo) & (y <= hi)))
    mpiw = float(np.mean(hi - lo))
    return picp, mpiw
