#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
from cvxopt import matrix, solvers

# Turn off cvxopt progress printing.
solvers.options['show_progress'] = False

def kernelfun(X, kerfPara, Y=None):
    """
    Evaluate the RBF kernel.
    """
    if Y is None:
        Y = X
    if kerfPara['type'] == 'rbf':
        gamma = kerfPara['pars']
        sqdist = cdist(X, Y, 'sqeuclidean')
        return np.exp(-gamma * sqdist)
    else:
        raise ValueError("Unknown kernel function")

# A helper function to set a tolerance based on C.
def svtol(C):
    return 1e-5

# For this demonstration we simply disable the bias-computation by returning 0.
def nobias(kerfType):
    # For our 'rbf' kernel we do not impose extra equality constraints.
    return 0

def epsilon_quantilesvr2(X, Y, test, kerfPara, C, tau, eps1):
    """
    Converted from epsilon_quantilesvr2.m.
    
    Inputs:
      X, Y    : Training data (Y should be a column vector)
      test    : Test data (each row is an example)
      kerfPara: Kernel parameter structure, e.g. {'type':'rbf', 'pars': s}
      C       : Regularisation parameter
      tau     : Quantile level (0.025 for lower quantile, 0.975 for upper quantile)
      eps1    : A parameter (set to 0 in our experiments)
    
    Returns:
      PredictY : Predictions on the test set
      f1       : (Intermediate) predictions on training data
      nsv      : Number of support vectors
      sparsity : Fraction of zero coefficients in the model
    """
    epsilon = svtol(C)
    n = X.shape[0]
    # Compute the kernel matrix on training set.
    H = kernelfun(X, kerfPara)  # shape: (n,n)
    
    # Build the Hessian for the QP: Hb = [H -H; -H H]
    Hb_top = np.hstack([H, -H])
    Hb_bottom = np.hstack([-H, H])
    Hb = np.vstack([Hb_top, Hb_bottom])
    
    # Build the linear term vector.
    c_part1 = ((1 - tau) * eps1 * np.ones((n, 1)) - Y)
    c_part2 = (tau * eps1 * np.ones((n, 1)) + Y)
    c_vec = np.vstack([c_part1, c_part2]).flatten()
    
    # Define variable bounds.
    vlb = np.zeros(2 * n)
    vub = np.concatenate([tau * C * np.ones(n), (1 - tau) * C * np.ones(n)])
    
    # Set equality constraints if required (via nobias).
    if nobias(kerfPara['type']) != 0:
        A = np.hstack([np.ones(n), -np.ones(n)]).reshape(1, 2 * n)
        b_eq = np.array([0])
    else:
        A = None
        b_eq = None

    # Convert matrices into cvxopt format.
    P = matrix(Hb)
    q = matrix(c_vec)
    # Set up inequality constraints to impose lower and upper bounds.
    I = np.eye(2 * n)
    G1 = -I  # for x >= 0  (i.e. -I * x <= 0)
    h1 = np.zeros(2 * n)
    G2 = I   # for x <= vub (i.e. I * x <= vub)
    h2 = vub
    G = matrix(np.vstack([G1, G2]))
    h = matrix(np.hstack([h1, h2]))
    
    if A is not None:
        A_qp = matrix(A)
        b_qp = matrix(b_eq)
        sol = solvers.qp(P, q, G, h, A_qp, b_qp)
    else:
        sol = solvers.qp(P, q, G, h)
    
    alpha = np.array(sol['x']).flatten()
    alpha1 = alpha[:n]
    beta1 = alpha[n:2*n]
    beta = alpha1 - beta1
    sparsity = 1 - (np.count_nonzero(beta) / len(beta))
    
    # Compute bias if needed. Here we simply set bias = 0.
    if nobias(kerfPara['type']) != 0:
        if tau > 0.5:
            svii = np.where((np.abs(beta) > epsilon) & (np.abs(beta) < (tau * C - epsilon)))[0]
        else:
            svii = np.where((np.abs(beta) > epsilon) & (np.abs(beta) < ((1 - tau) * C - epsilon)))[0]
        if len(svii) > 0:
            bias = np.mean(Y[svii] - np.sign(beta[svii]))
        else:
            print('No support vectors with interpolation error e - cannot compute bias.')
            bias = (np.max(Y) + np.min(Y)) / 2.0
    else:
        bias = 0
    
    f1 = H.dot(beta) + bias
    # Compute the kernel matrix between test set and training set.
    Htest = kernelfun(test, kerfPara, X)
    PredictY = Htest.dot(beta) + bias
    nsv = np.sum(np.abs(beta) > epsilon)
    return PredictY, f1, nsv, sparsity

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
    Given the full signal y and a window size 'win', construct
    a dataset where each row is a sliding window and the target is
    the value immediately after the window.
    """
    X_all = []
    y_all = []
    for i in range(len(y) - win):
        X_all.append(y[i:i+win])
        y_all.append(y[i+win])
    return np.array(X_all), np.array(y_all).reshape(-1, 1)

def main():
    start_time = time.time()
    # Read the beer production data from beer.csv.
    from pathlib import Path
    data_path = Path(__file__).resolve().parents[2] / "datasets" / "beer.csv"
    df = pd.read_csv(data_path)
    # Assume column "Monthly beer production" holds the numeric data.
    production = df["Monthly beer production"].values.astype(float)
    print("Read beer production data; total data points:", len(production))
    
    # Build the dataset using a chosen sliding window size.
    win_chosen = 12
    X, Y = build_dataset(production, win_chosen)
    n_total = X.shape[0]
    # Split the dataset into 70% training and 30% test.
    train_size = int(np.floor(0.7 * n_total))
    X_train = X[:train_size, :]
    Y_train = Y[:train_size, :]
    X_test = X[train_size:, :]
    Y_test = Y[train_size:, :]
    print(f"Dataset built with window size = {win_chosen}.")
    print(f"Training examples: {X_train.shape[0]}, Test examples: {X_test.shape[0]}")
    
    # Set chosen parameters.
    q_lower = 0.025
    q_upper = 0.975
    # Here we choose a simple RBF kernel with parameter s = 1 (gamma = 1).
    kerfPara1 = {'type': 'rbf', 'pars': 2**13}
    kerfPara2 = {'type': 'rbf', 'pars':2**13}
    # Choose the regularisation parameter C.
    C_value = 2**6
    eps1 = 0.0
    
    # Run epsilon_quantilesvr2 to obtain quantile predictions on test set.
    print("Computing lower quantile predictions ...")
    Pred_lower, f_lower, nsv_lower, sparsity_lower = epsilon_quantilesvr2(X_train, Y_train, X_test, kerfPara1, C_value, q_lower, eps1)
    print("Computing upper quantile predictions ...")
    Pred_upper, f_upper, nsv_upper, sparsity_upper = epsilon_quantilesvr2(X_train, Y_train, X_test, kerfPara2, C_value, q_upper, eps1)
    
    # Evaluate the prediction interval: calculate PICP and MPIW.
    PICP_val, MPIW_val = evaluate_PICP(Y_test, Pred_lower, Pred_upper)
    print(f"Prediction Interval Coverage Probability (PICP): {PICP_val:.4f}")
    print(f"Mean Prediction Interval Width (MPIW): {MPIW_val:.4f}")
    
    # Plot true test values and predicted lower/upper quantiles.
    plt.figure()
    plt.plot(Y_test, 'b-', label='True Values')
    plt.plot(Pred_lower, 'r-', label='Predicted Lower Quantile')
    plt.plot(Pred_upper, 'k-', label='Predicted Upper Quantile')
    plt.legend()
    plt.title('Prediction Intervals on Test Data')
    plt.xlabel('Test Sample Index')
    plt.ylabel('Monthly Beer Production')
    plt.show()
    
    total_time = time.time() - start_time
    print(f"Execution completed in {total_time:.2f} seconds.")

if __name__ == '__main__':
    main() 
