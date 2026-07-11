#!/usr/bin/env python3
# This shebang tells the system to run the script with Python3

import numpy as np                      # Import NumPy for numerical operations
import pandas as pd                     # Import pandas for reading and processing CSV data
import matplotlib.pyplot as plt         # Import matplotlib for plotting graphs
import time                             # Import time module to measure execution times
from scipy.spatial.distance import cdist  # Import cdist to compute pairwise distances
from scipy.stats import norm            # Import norm for normal distribution functions

def kernelfun(X, kerfPara, Y=None):
    """
    RBF kernel evaluation.
    This function computes the Radial Basis Function (RBF) kernel
    between two sets of input vectors.
    
    Parameters:
      X: An array of data points.
      kerfPara: A dictionary with keys 'type' and 'pars'. For RBF, 
                'pars' corresponds to the gamma parameter.
      Y: (Optional) A second array of data points. If not given, Y = X.
                
    Returns:
      The RBF kernel matrix computed using the formula: exp(-gamma * ||x-y||^2).
    """
    if Y is None:
        Y = X                   # If Y is not provided, use X for both arguments.
    if kerfPara['type'] == 'rbf':
        gamma = kerfPara['pars']  # Extract the gamma parameter from kerfPara.
        # Compute the squared Euclidean distance between each pair of points in X and Y.
        sqdist = cdist(X, Y, 'sqeuclidean')
        return np.exp(-gamma * sqdist)  # Compute and return the RBF kernel matrix.
    else:
        raise ValueError("Unknown kernel function")  # Raise an error for unsupported kernel types.

def build_dataset(y, win):
    """
    Build the sliding-window dataset.
    For each index i in the input array, a window of length 'win' is taken as the feature,
    and the element at position i+win is the target.
    
    Example:
      If y = [3, 5, 7, 9] and win = 2, then:
      X_all = [[3,5], [5,7]]
      y_all = [7,9]
    
    Parameters:
      y: 1-D array of values.
      win: An integer that defines the window size.
    
    Returns:
      X_all: A 2-D NumPy array where each row is a window segment.
      y_all: A column vector of target values.
    """
    # Create a list for the input segments (windows)
    X_all = [y[i:i+win] for i in range(len(y) - win)]
    # The target is always the element immediately following each window
    y_all = [y[i+win] for i in range(len(y) - win)]
    return np.array(X_all), np.array(y_all).reshape(-1, 1)  # Convert lists to NumPy arrays.

def leastsquaresvr(train, ytrain, test, kerfPara, C):
    """
    Solves a least-squares kernel regression (ridge regression) problem.
    This function computes the solution 'u' for the ridge regression problem:
      u = (H^T H + (1/C)I)^(-1) H^T ytrain,
    where H is the augmented kernel matrix [K(train, train), ones].
    The function then computes predictions for both training and test data.
    
    Parameters:
      train: 2-D array of training features.
      ytrain: 2-D column vector of training targets.
      test: 2-D array of test features.
      kerfPara: Dictionary specifying the kernel type ('rbf') and parameter (gamma).
      C: Regularization parameter (used in the ridge regression formulation).
    
    Returns:
      PredictY: Predictions for the training data.
      f1: Predictions for the test data.
      sparsity: Ratio of nonzero coefficients in the solution 'u'.
    """
    # Build the training kernel matrix A using the RBF kernel for training data.
    A = kernelfun(train, kerfPara)
    n_train = train.shape[0]  # Number of training samples.
    # Augment the kernel matrix with a column of ones to incorporate a bias term.
    H = np.hstack([A, np.ones((n_train, 1))])
    
    # Form the regularization matrix: (1/C)*I.
    reg_matrix = np.eye(H.shape[1]) / C
    # Solve the ridge regression equation: u = (H^T H + (1/C)*I)^(-1) H^T ytrain.
    u = np.linalg.solve(H.T @ H + reg_matrix, H.T @ ytrain)
    
    # Compute predictions for the training set using H * u.
    PredictY = H @ u
    
    # Build the test kernel matrix to compute test predictions.
    # This calculates the RBF kernel between test data and training data.
    A_test = kernelfun(test, kerfPara, train)
    # Augment with a bias term (ones).
    H_test = np.hstack([A_test, np.ones((test.shape[0], 1))])
    # Compute test predictions.
    f1 = H_test @ u
    
    # Calculate the sparsity: fraction of nonzero entries in the coefficient vector.
    sparsity = np.count_nonzero(u) / len(u)
    return PredictY, f1, sparsity

def main():
    start_time = time.time()  # Record the start time for execution.
    print("Starting program execution.")
    
    # Read data from beer.csv using the "Monthly beer production" column.
    print("Reading data from beer.csv ...")
    from pathlib import Path
    data_path = Path(__file__).resolve().parents[2] / "datasets" / "beer.csv"
    df = pd.read_csv(data_path)
    col_name = "Monthly beer production"  # Specify the target column name.
    # Convert the column to numeric, coercing errors to NaN, drop NaN, and extract values as a NumPy array.
    y = pd.to_numeric(df[col_name], errors='coerce').dropna().values
    print("Data read successfully. Total data points:", len(y))
    
    # --- Demonstration using leastsquaresvr ---
    print("\n--- Running example demonstration with leastsquaresvr ---")
    win_chosen = 1                    # Chosen sliding-window length.
    s1_chosen = 0.0158 * 2             # Chosen scaling for the kernel's gamma parameter.
    # Create the kernel parameter dictionary for the RBF kernel.
    kerfPara = {'type': 'rbf', 'pars': s1_chosen}
    # Choose a regularization exponent (for example, 8) and compute C as 2^(exponent).
    c1_chosen = 7
    C = 2.0 ** (c1_chosen)
    print(f"Chosen parameters: win_chosen = {win_chosen}, s1_chosen = {s1_chosen}, C = {C}")
    
    # Build the dataset using the chosen window size.
    X_all, y_all = build_dataset(y, win_chosen)
    n_total = X_all.shape[0]  # Total samples after windowing.
    idx_split = int(np.floor(n_total * 0.7))  # Use 70% for training.
    print(f"Rebuilding dataset: {n_total} samples, training on first {idx_split} samples.")
    # Split dataset into training and test sets.
    X_train, y_train = X_all[:idx_split, :], y_all[:idx_split, :]
    X_test, y_test = X_all[idx_split:, :], y_all[idx_split:, :]
    
    # Run the least squares kernel regression.
    PredictY, f1, sparsity = leastsquaresvr(X_train, y_train, X_test, kerfPara, C)
    print(f"leastsquaresvr sparsity: {sparsity:.4f}")
    
    # Compute the standard deviation of the residuals on the test set.
    sigma = np.std(y_test - f1)
    # Compute the quantile values (using normal distribution) based on sigma.
    quantile_values = norm.ppf([0.025, 0.975], loc=0, scale=sigma)
    # Compute lower and upper prediction intervals for the test data.
    Low_Q = f1 + quantile_values[0]
    Up_Q = f1 + quantile_values[1]
    print(f"Final Evaluation: sigma = {sigma:.4f}")
    print(f"Predicted Lower Quantile (first 5 values): {Low_Q[:5]}")
    print(f"Predicted Upper Quantile (first 5 values): {Up_Q[:5]}")
    
    # Compute Prediction Interval Coverage Probability (PICP)
    # and Mean Prediction Interval Width (MPIW) for the test data.
    picp = np.mean((y_test >= Low_Q) & (y_test <= Up_Q))
    mpiw = np.mean(Up_Q - Low_Q)
    total_time_elapsed = time.time() - start_time   # Total execution time.
    print(f"Training time: {total_time_elapsed:.2f} seconds")
    print(f"PICP: {picp:.4f}")
    print(f"MPIW: {mpiw:.4f}")
    
    # Plot the true test values and the prediction intervals.
    plt.figure()
    plt.plot(y_test, 'b-', label='True Values')
    plt.plot(Low_Q, 'r-', label='Predicted Lower Quantile')
    plt.plot(Up_Q, 'k-', label='Predicted Upper Quantile')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.09), ncol=3, fontsize=8)  # Display legend above the plot in a horizontal line with smaller font
    plt.subplots_adjust(top=0.83)  # Adjust top margin to make room for the legend
    plt.show()

    print("Program execution complete.")

if __name__ == '__main__':
    main()
