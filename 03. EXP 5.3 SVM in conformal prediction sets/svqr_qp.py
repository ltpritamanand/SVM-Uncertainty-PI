"""
SVQR-QP: Quantile SVR via Quadratic Programming.

Direct port of epsilon_quantilesvr2.m + kernelfun.m — same implementation
as probab_forecast_svr.py (the approved reference). Uses cvxopt QP solver.

Kernel : K(x,y) = exp(-gamma * ||x-y||^2)   [kernelfun.m with pars=gamma]
eps1   : 0  (no insensitive tube)
bias   : 0  (nobias(rbf) = 0)

QP dual:
  min  0.5 v^T [H -H; -H H] v + [-y; y]^T v
  s.t. 0 <= alpha1 <= tau*C
       0 <= beta1  <= (1-tau)*C
  beta = alpha1 - beta1
  f(x) = K(x, X_train) @ beta

Public API
----------
rbf_kernel(X1, X2, gamma)                -> K matrix
solve_qp(X_tr_s, y_tr_s, gamma, C, tau)  -> (beta, sparsity)
predict_svqr(beta, X_tr_s, X_q_s, gamma) -> y_pred
"""

import numpy as np
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers

solvers.options['show_progress'] = False


def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    """K(xi, xj) = exp(-gamma * ||xi-xj||^2).  Matches kernelfun.m."""
    sqdist = cdist(X1, X2, 'sqeuclidean')
    return np.exp(-gamma * sqdist)


def solve_qp(X_tr_s: np.ndarray, y_tr_s: np.ndarray,
             gamma: float, C: float, tau: float):
    """Solve SVQR dual QP with cvxopt. Returns (beta, sparsity).

    Parameters
    ----------
    X_tr_s : (n, d) training features (scaled)
    y_tr_s : (n,)   training targets  (scaled)
    gamma  : RBF kernel parameter  K = exp(-gamma * ||x-y||^2)
    C      : regularisation / box constraint
    tau    : quantile level (e.g. 0.05 lower, 0.95 upper)
    """
    n  = X_tr_s.shape[0]
    H  = rbf_kernel(X_tr_s, X_tr_s, gamma)

    Hb = np.vstack([np.hstack([H, -H]),
                    np.hstack([-H,  H])])

    # linear term: eps1 = 0  ->  c = [-y; y]
    c_vec = np.concatenate([-y_tr_s, y_tr_s])

    # box constraints: 0 <= alpha1 <= tau*C,  0 <= beta1 <= (1-tau)*C
    vub = np.concatenate([tau * C * np.ones(n),
                          (1.0 - tau) * C * np.ones(n)])

    I = np.eye(2 * n)
    P = matrix(Hb)
    q = matrix(c_vec)
    G = matrix(np.vstack([-I, I]))
    h = matrix(np.concatenate([np.zeros(2 * n), vub]))

    sol   = solvers.qp(P, q, G, h)
    alpha = np.array(sol['x']).flatten()
    beta  = alpha[:n] - alpha[n:]

    sparsity = 1.0 - np.count_nonzero(np.abs(beta) > 1e-5) / n
    return beta, sparsity


def predict_svqr(beta: np.ndarray,
                 X_tr_s: np.ndarray,
                 X_query_s: np.ndarray,
                 gamma: float) -> np.ndarray:
    """K(X_query, X_train) @ beta.  Returns predictions in scaled y-space."""
    return rbf_kernel(X_query_s, X_tr_s, gamma) @ beta
