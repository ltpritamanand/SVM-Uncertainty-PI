"""
Python port of quantileLPONENORMTSVR12.m
Sparse SVQR via LP (L1-norm regularisation).

Paper Equation 9:
  min  (1/2) sum(r+p) + C * sum(q*xi + (1-q)*xi*)
  s.t. y - H*(r-p) <= xi
       H*(r-p) - y <= xi*
       r, p, xi, xi* >= 0

Where H = [K(train,train), ones] and u = r - p are the kernel weights + bias.
"""

import numpy as np
from scipy.optimize import linprog
from kernelfun import rbf_kernel


def quantile_lp_onenorm(X_train, y_train, X_test, gamma, c3, c1, tau):
    """
    Solve sparse SVQR as a Linear Programme.

    Parameters
    ----------
    X_train : (n, d)  training features
    y_train : (n,)    training targets
    X_test  : (m, d)  test features
    gamma   : float   RBF kernel parameter (K = exp(-gamma * ||x-y||^2))
    c3      : float   L1 regularisation coefficient
    c1      : float   Pinball loss coefficient (C in the paper)
    tau     : float   Quantile level (e.g. 0.05 or 0.95)

    Returns
    -------
    train_pred : (n,)  predictions on training data
    test_pred  : (m,)  predictions on test data
    sparsity   : float fraction of non-zero weights
    """
    n = X_train.shape[0]

    # Kernel matrix + bias column
    K = rbf_kernel(X_train, gamma=gamma)         # (n, n)
    H = np.hstack([K, np.ones((n, 1))])           # (n, n+1)

    # Decision variables: [r(n+1), p(n+1), xi(n), xi_star(n)]
    n_rp = n + 1

    # Objective: min c3*sum(r+p) + c1*[tau*sum(xi) + (1-tau)*sum(xi*)]
    f = np.concatenate([
        c3 * np.ones(n_rp),             # r
        c3 * np.ones(n_rp),             # p
        c1 * tau * np.ones(n),           # xi
        c1 * (1 - tau) * np.ones(n),     # xi*
    ])

    # Inequality constraints: A_ub @ x <= b_ub
    I_n = np.eye(n)
    Z_n = np.zeros((n, n))
    A_ub = np.vstack([
        np.hstack([-H,  H, -I_n, Z_n]),   # y - H*(r-p) <= xi
        np.hstack([ H, -H, Z_n, -I_n]),   # H*(r-p) - y <= xi*
    ])
    b_ub = np.concatenate([-y_train, y_train])

    # Bounds: all variables >= 0
    bounds = [(0, None)] * len(f)

    # Solve
    result = linprog(f, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                     method='highs', options={'presolve': True, 'disp': False})

    if not result.success:
        x = np.zeros(len(f))
    else:
        x = result.x

    # Extract weights: u = r - p
    u = x[:n_rp] - x[n_rp:2 * n_rp]

    # Training predictions
    train_pred = H @ u

    # Test predictions
    K_test = rbf_kernel(X_test, X_train, gamma=gamma)
    H_test = np.hstack([K_test, np.ones((X_test.shape[0], 1))])
    test_pred = H_test @ u

    # Sparsity: fraction of non-zero kernel weights (excluding bias)
    sparsity = float(np.sum(np.abs(u[:-1]) > 1e-8) / n)

    return train_pred, test_pred, sparsity
