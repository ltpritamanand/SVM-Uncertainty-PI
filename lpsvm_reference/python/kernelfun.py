"""
Python port of kernelfun used inside quantileLPONENORMTSVR12.m
K(x,y) = exp(-gamma * ||x - y||^2)
"""

import numpy as np
from scipy.spatial.distance import cdist


def rbf_kernel(X, Y=None, gamma=1.0):
    """RBF kernel matrix.
    K_ij = exp(-gamma * ||X_i - Y_j||^2)

    Matches the inlined kernelfun in quantileLPONENORMTSVR12.m:
        sqdist = pdist2(X, Y, 'euclidean').^2;
        K = exp(-gamma * sqdist);
    """
    if Y is None:
        Y = X
    sqdist = cdist(X, Y, 'sqeuclidean')
    return np.exp(-gamma * sqdist)
