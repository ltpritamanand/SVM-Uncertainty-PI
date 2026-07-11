#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist
from scipy.optimize import linprog
from pathlib import Path

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

def svtol(C):
    """
    Compute tolerance for support vector detection.
    """
    return 1e-4

def nobias(kerfType):
    """
    For an RBF kernel we assume that a bias term is used.
    Return 1 to indicate that equality constraint will be imposed.
    """
    if kerfType == 'rbf':
        return 1
    else:
        return 0

def epsilon_quantilesvr2(X, Y, test, kerfPara, C, tau, eps1):
    """
    Returns:
      PredictY: predicted values on the test set,
      f1: predictions on the training set,
      nsv: number of support vectors,
      sparsity: 1 - (number of nonzero coefficients / length(coefficients)).
    """
    ytrain = np.asarray(Y).reshape(-1, 1)
    n1 = X.shape[0]
    e = np.ones((n1, 1))

    K = kernelfun(X, kerfPara)
    H = np.hstack([K, e])

    c3 = 1.0
    c1 = float(C)

    f = np.concatenate(
        [
            c3 * np.ones(n1 + 1),
            c3 * np.ones(n1 + 1),
            c1 * float(tau) * np.ones(n1),
            c1 * (1.0 - float(tau)) * np.ones(n1),
        ]
    )

    I_n = np.eye(n1)
    zeros_n = np.zeros((n1, n1))
    A1_top = np.hstack([-H, H, -I_n, zeros_n])
    A1_bottom = np.hstack([H, -H, zeros_n, -I_n])
    A_ub = np.vstack([A1_top, A1_bottom])
    b_ub = np.concatenate([-ytrain.flatten(), ytrain.flatten()])

    bounds = [(0, None)] * len(f)
    res = linprog(f, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:
        return np.zeros(test.shape[0]), np.zeros(X.shape[0]), 0, 0

    x = res.x
    u = x[: (n1 + 1)] - x[(n1 + 1) : 2 * (n1 + 1)]
    f1 = H.dot(u)

    K_test = kernelfun(test, kerfPara, X)
    H_test = np.hstack([K_test, np.ones((test.shape[0], 1))])
    PredictY = H_test.dot(u)

    beta = u[:-1]
    nsv = int(np.sum(np.abs(beta) > 1e-6))
    sparsity = 1.0 - (nsv / float(len(beta))) if len(beta) else 0.0
    return PredictY, f1, nsv, sparsity

def evaluate_PICP(y, Low_Q, Up_Q):
    """
    Evaluate Prediction Interval Coverage Probability (PICP)
    and Mean Prediction Interval Width (MPIW).
    """
    y = y.flatten()
    Low_Q = Low_Q.flatten()
    Up_Q = Up_Q.flatten()
    PICP = np.mean((y >= Low_Q) & (y <= Up_Q))
    MPIW = np.mean(Up_Q - Low_Q)
    return PICP, MPIW

def build_dataset(y, win):
    """
    Build a sliding-window dataset.
    For each index i, use y[i:i+win] as features and y[i+win] as target.
    """
    X_all = []
    y_all = []
    for i in range(len(y) - win):
        X_all.append(y[i:i+win])
        y_all.append(y[i+win])
    return np.array(X_all), np.array(y_all).reshape(-1, 1)

def grid_search_tuning_fixed_win(y, win=3, q_lower=0.025, q_upper=0.975, desired_PICP=0.95):
    """
    Perform a simple grid search over s1 and C hyperparameters with fixed window size.
    Returns the best parameters (s1, C) that minimize a score
    defined as: score = |PICP - desired_PICP| + MPIW.
    """
    best_score = float('inf')
    best_params = None
    results = []
    
    # Define reasonable parameter grids
    # Use 2**x syntax for powers of 2
    s1_list = [0.0146]  # 2^-5 to 2^5
    C_list = [8]   # 2^0 to 2^10
    
    # Build dataset with fixed window size
    X_all, Y_all = build_dataset(y, win)
    n_total = X_all.shape[0]
    
    # Split: 60% training and 20% validation (reserve last 20% for test later)
    train_end = int(np.floor(n_total * 0.6))
    val_end = int(np.floor(n_total * 0.8))
    X_train = X_all[:train_end, :]
    y_train = Y_all[:train_end, :]
    X_val = X_all[train_end:val_end, :]
    y_val = Y_all[train_end:val_end, :]
    
    for s1 in s1_list:
        for C in C_list:
            try:
                kerfPara = {'type': 'rbf', 'pars': s1}
                # Run quantile SVR for lower and upper quantiles on the validation set.
                Pred_val_lower, _, _, _ = epsilon_quantilesvr2(X_train, y_train, X_val, kerfPara, C, q_lower, 0)
                Pred_val_upper, _, _, _ = epsilon_quantilesvr2(X_train, y_train, X_val, kerfPara, C, q_upper, 0)
                
                # Check if QP solver failed (returned zeros)
                if np.all(Pred_val_lower == 0) or np.all(Pred_val_upper == 0):
                    print(f"Skipping s1={s1}, C={C} due to solver failure")
                    continue
                    
                PICP_val, MPIW_val = evaluate_PICP(y_val, Pred_val_lower, Pred_val_upper)
                # Simple score: we want coverage close to desired and low width.
                score = abs(PICP_val - desired_PICP) + MPIW_val
                results.append((s1, C, PICP_val, MPIW_val, score))
                print(f"s1={s1}, C={C} => PICP={PICP_val:.4f}, MPIW={MPIW_val:.4f}, score={score:.4f}")
                
                if score < best_score:
                    best_score = score
                    best_params = (s1, C)
            except Exception as e:
                print(f"Error with s1={s1}, C={C}: {str(e)}")
                continue
    
    if best_params is None:
        print("\nNo valid parameters found. Using defaults.")
        best_params = (1.0, 32.0)  # Default values
    else:
        print("\nBest parameters found:")
        print(f"s1={best_params[0]}, C={best_params[1]} with score {best_score:.4f}")
        
    return best_params, results

def main():
    start_time = time.time()
    print("Starting program execution.")
    
    # Read data from NO2.txt
    print("Reading data from NO2.txt ...")
    base_dir = Path(__file__).resolve().parents[2]
    data_path = base_dir / "datasets" / "NO2.txt"
    data = np.loadtxt(data_path)
    print(f"Data read successfully. Total data points: {data.shape[0]}, dimensions: {data.shape[1]}")
    
    # We use the first column as the target variable (NO2 concentration).
    y = data[:, 0]
    print(f"Using first column as target variable. Range: {y.min():.2f} to {y.max():.2f}")
    
    # Fixed parameters for the model
    win_chosen = 3  # Window size
    q_lower = 0.025  # Lower quantile
    q_upper = 0.975  # Upper quantile
    desired_PICP = 0.95  # Desired prediction interval coverage probability
    
    # Fixed parameters (no more grid search)
    # ----- Grid Search Tuning with Fixed Window Size -----
    # print(f"\nStarting grid search tuning with fixed window size={win_chosen} ...")
    # best_params, grid_results = grid_search_tuning_fixed_win(y, win_chosen, q_lower, q_upper, desired_PICP)
    # best_s1, best_C = best_params
    # Use best parameters directly
    best_s1   = 2**(-5)   # γ exponent eg = -5
    C_lower   = 2**(-5)   # C exponent ec1 = -5 for lower quantile
    C_upper   = 2**(-2)   # C exponent ec3 = -2 for upper quantile
    print(f"\nUsing fixed parameters: gamma=2**(-5), C_lower={C_lower}, C_upper={C_upper}")
    
    # ----- Build Dataset with Fixed Window Size and Final Split -----
    X_all, Y_all = build_dataset(y, win_chosen)
    n_total = X_all.shape[0]
    # Split into 60% training, 20% validation, and 20% test.
    train_end = int(np.floor(n_total * 0.6))
    val_end   = int(np.floor(n_total * 0.8))
    X_train   = X_all[:train_end, :]
    y_train   = Y_all[:train_end, :]
    X_val     = X_all[train_end:val_end, :]
    y_val     = Y_all[train_end:val_end, :]
    X_test    = X_all[val_end:, :]
    y_test    = Y_all[val_end:, :]
    print(f"\nFinal dataset split with win={win_chosen}: "
          f"{X_train.shape[0]} training, {X_val.shape[0]} validation, {X_test.shape[0]} test samples.")
    
    # Create kernel parameter structure with best_s1.
    kerfPara = {'type': 'rbf', 'pars': best_s1}
    
    # Run epsilon_quantilesvr2 on the test set using fixed Cs
    # print("\nRunning epsilon_quantilesvr2 on test set with best parameters ...")
    # Pred_test_lower, f_test_lower, nsv_test_lower, sparsity_test_lower = epsilon_quantilesvr2(
    #     X_train, y_train, X_test, kerfPara, best_C, q_lower, 0)
    # Pred_test_upper, f_test_upper, nsv_test_upper, sparsity_test_upper = epsilon_quantilesvr2(
    #     X_train, y_train, X_test, kerfPara, best_C, q_upper, 0)
    print("\nRunning epsilon_quantilesvr2 on test set with fixed parameters ...")
    Pred_test_lower, f_test_lower, nsv_test_lower, sparsity_test_lower = epsilon_quantilesvr2(
        X_train, y_train, X_test, kerfPara, C_lower, q_lower, 0
    )
    Pred_test_upper, f_test_upper, nsv_test_upper, sparsity_test_upper = epsilon_quantilesvr2(
        X_train, y_train, X_test, kerfPara, C_upper, q_upper, 0
    )
    PICP_test, MPIW_test = evaluate_PICP(y_test, Pred_test_lower, Pred_test_upper)
    print(f"Test set evaluation: PICP = {PICP_test:.4f}, MPIW = {MPIW_test:.4f}")
    
    # Plot the prediction intervals for the test set.
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, 'b-', label='True Values')
    plt.plot(Pred_test_lower, 'r-', label='Predicted Lower Quantile')
    plt.plot(Pred_test_upper, 'k-', label='Predicted Upper Quantile')
    plt.fill_between(range(len(y_test)), Pred_test_lower.flatten(), Pred_test_upper.flatten(), 
                     color='gray', alpha=0.3, label='Prediction Interval')
    plt.legend()
    plt.title('Prediction Intervals for NO2 Data (Test Set)')
    plt.xlabel('Test Sample Index')
    plt.ylabel('NO2 Concentration')
    plt.savefig('NO2_prediction_intervals.png')
    plt.show()
    
    total_time = time.time() - start_time
    print(f"\nProgram execution complete in {total_time:.2f} seconds.")

if __name__ == '__main__':
    main()
