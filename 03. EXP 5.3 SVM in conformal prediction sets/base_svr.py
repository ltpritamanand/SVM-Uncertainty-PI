#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
import itertools

def kernelfun(X, kerfPara, Y=None):
    """
    RBF kernel evaluation.
    """
    if Y is None:
        Y = X
    if kerfPara['type'] == 'rbf':
        gamma = kerfPara['pars']
        sqdist = cdist(X, Y, 'sqeuclidean')
        return np.exp(-gamma * sqdist)
    else:
        raise ValueError("Unknown kernel function")

def quantileLPONENORMTSVR12(train, ytrain, test, s, c3, c1, tau1):
    """
    Solves the LP problem to get predictions for a given quantile.
    """
    n1 = train.shape[0]
    e = np.ones((n1, 1))

    # Build the kernel matrix and the H matrix.
    A = kernelfun(train, {'type': 'rbf', 'pars': s})
    H = np.hstack([A, e])

    # Build the objective function vector.
    f = np.concatenate([
        c3 * np.ones(n1 + 1),
        c3 * np.ones(n1 + 1),
        c1 * tau1 * np.ones(n1),
        c1 * (1 - tau1) * np.ones(n1)
    ])

    # Construct constraints.
    I_n = np.eye(n1)
    zeros_n = np.zeros((n1, n1))
    A1_top = np.hstack([-H, H, -I_n, zeros_n])
    A1_bottom = np.hstack([H, -H, zeros_n, -I_n])
    A1 = np.vstack([A1_top, A1_bottom])
    B1 = np.concatenate([-ytrain.flatten(), ytrain.flatten()])

    # Define bounds.
    lb1 = np.zeros_like(f)
    ub1 = np.full_like(f, np.inf)
    bounds = [(lb1[i], ub1[i]) for i in range(len(f))]

    # Debug print before calling the LP solver.
    global verbose
    if verbose:
        print(f"  -- Solving LP for tau={tau1:.3f} with s={s:.4f}, c3={c3}, c1={c1}")
    result = linprog(f, A_ub=A1, b_ub=B1, bounds=bounds, method='highs')
    if not result.success:
        print("Warning: linprog did not converge. Using fallback solution.")
        x = lb1
    else:
        x = result.x

    # Compute coefficient vector and predictions.
    u1 = x[0:(n1 + 1)] - x[(n1 + 1):2*(n1 + 1)]
    trainf1 = H.dot(u1)
    Htest = np.hstack([
        kernelfun(test, {'type': 'rbf', 'pars': s}, train),
        np.ones((test.shape[0], 1))
    ])
    f1 = Htest.dot(u1)
    sparsity = np.count_nonzero(u1) / len(u1)

    if verbose:
        print(f"  -- LP solved. Sparsity: {sparsity:.4f}")
    return trainf1, f1, sparsity

def evaluate_PICP(y, Low_Q, Up_Q):
    """
    Evaluate Prediction Interval Coverage Probability (PICP) and
    Mean Prediction Interval Width (MPIW).
    """
    y = y.flatten()
    Low_Q = Low_Q.flatten()
    Up_Q = Up_Q.flatten()
    PICP = np.mean((y >= Low_Q) & (y <= Up_Q))
    MPIW = np.mean(Up_Q - Low_Q)
    return PICP, MPIW

def build_dataset(y, win):
    """
    Build the sliding-window dataset.
    Each row of X_all is created from a sliding window of length 'win' on y,
    and y_all is the subsequent value.
    """
    X_all = []
    y_all = []
    for i in range(len(y) - win):
        X_all.append(y[i:i+win])
        y_all.append(y[i+win])
    return np.array(X_all), np.array(y_all).reshape(-1, 1)

def main():
    global verbose
    verbose = True  # Enable verbose printing
    start_time = time.time()
    print("Starting program execution.")

    # --------------------------------------------------------------------
    # ---- Replace CSV reading with whitespace‐delimited NO2.txt input ----
    print("Reading data from NO2.txt ...")
    from pathlib import Path
    data_path = Path(__file__).resolve().parents[2] / "datasets" / "NO2.txt"
    df = pd.read_csv(data_path, delim_whitespace=True, header=None)
    # Use the first column as the univariate series
    y = pd.to_numeric(df.iloc[:, 0], errors='coerce').values
    y = y[~np.isnan(y)]  # Remove any NaN values
    print("Data read successfully. Total data points:", len(y))
    # --------------------------------------------------------------------

    # --- HYPERPARAMETER GRID SEARCH ---------------------------------------
    # tune gamma = 2**eg, C1 = 2**ec1, c3 = 2**ec3 for eg,ec1,ec3 in [-5..10]
    exponents  = list(range(-5, 11))
    best_picp  = -np.inf
    best_params = None

    q_lower, q_upper = 0.025, 0.975

    for eg, ec1, ec3 in itertools.product(exponents, repeat=3):
        # fixed window size = 2
        X_all, y_all = build_dataset(y, 2)
        n = len(X_all)
        t_end = int(0.6 * n)
        c_end = int(0.8 * n)
        X_tr, y_tr   = X_all[:t_end],       y_all[:t_end]
        X_cal, y_cal = X_all[t_end:c_end],  y_all[t_end:c_end]
        # compute hyper‐parameters
        gamma = 2.0 ** eg
        C1    = 2.0 ** ec1
        c3    = 2.0 ** ec3
        # evaluate only PICP on calibration
        _, low, _ = quantileLPONENORMTSVR12(X_tr, y_tr, X_cal, gamma, c3, C1, q_lower)
        _, up,  _ = quantileLPONENORMTSVR12(X_tr, y_tr, X_cal, gamma, c3, C1, q_upper)
        PICP, _ = evaluate_PICP(y_cal, low, up)
        if PICP > best_picp:
            best_picp   = PICP
            best_params = dict(eg=eg, ec1=ec1, ec3=ec3)

    print("Best params:", best_params, "PICP:", best_picp)

    # --- RUN BEST PARAMS 10 TIMES -----------------------------------------
    runs = []
    for i in range(10):
        eg, ec1, ec3 = (best_params[k] for k in ('eg','ec1','ec3'))
        X_all, y_all = build_dataset(y, 2)
        n = len(X_all)
        t_end = int(0.6 * n)
        c_end = int(0.8 * n)
        X_tr, y_tr   = X_all[:t_end],       y_all[:t_end]
        X_cal, y_cal = X_all[t_end:c_end],  y_all[t_end:c_end]
        X_te, y_te   = X_all[c_end:],       y_all[c_end:]
        gamma = 2.0 ** eg
        C1    = 2.0 ** ec1
        c3    = 2.0 ** ec3
        # evaluate on test set
        _, low, _ = quantileLPONENORMTSVR12(X_tr, y_tr, X_te, gamma, c3, C1, q_lower)
        _, up,  _ = quantileLPONENORMTSVR12(X_tr, y_tr, X_te, gamma, c3, C1, q_upper)
        PICP, MPIW = evaluate_PICP(y_te, low, up)

        print(f"Run {i+1}: PICP={PICP:.4f}, MPIW={MPIW:.4f}")
        runs.append((PICP, MPIW))

    runs = np.array(runs)
    print(f"Avg over 10 runs → PICP={runs[:,0].mean():.4f}, MPIW={runs[:,1].mean():.4f}")
    print(f" Std over 10 runs → PICP={runs[:,0].std():.4f}, MPIW={runs[:,1].std():.4f}")

if __name__ == '__main__':
    main()
