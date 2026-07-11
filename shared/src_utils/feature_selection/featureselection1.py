# feature_selection.py
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.optimize import linprog
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# Datasets are bundled inside final_github/shared/datasets
# featureselection1.py lives at final_github/shared/src_utils/feature_selection/
_DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"

# Function to run the linear quantile SVR using linear programming
def linear_quantile_lponenorm_tsvr(train, ytrain, test, s, c3, c1, tau1):
    """
    Solves the quantile regression problem using linear programming.
    Returns predictions for training and test sets, sparsity percentage, and the weight vector u1.
    """
    n1, n2 = train.shape
    # Create design matrix H with an appended column of ones (bias)
    H = np.hstack((train, np.ones((n1, 1))))
    # Objective function: four blocks concatenated
    f = np.concatenate([
        c3 * np.ones(n2 + 1),
        c3 * np.ones(n2 + 1),
        c1 * tau1 * np.ones(n1),
        c1 * (1 - tau1) * np.ones(n1)
    ])
    # Construct inequality constraints
    # First block: -H, H, -I, 0
    A1 = np.hstack(( -H,  H, -np.eye(n1), np.zeros((n1, n1)) ))
    # Second block: H, -H, 0, -I
    A2 = np.hstack(( H, -H, np.zeros((n1, n1)), -np.eye(n1) ))
    A = np.vstack((A1, A2))
    # Right-hand side for constraints
    b = np.concatenate((-ytrain, ytrain))
    # Define bounds (all variables >= 0 with no upper bound)
    num_vars = len(f)
    bounds = [(0, None)] * num_vars

    # Solve the LP problem using the HiGHS solver (recommended over interior-point)
    res = linprog(f, A_ub=A, b_ub=b, bounds=bounds, method='highs', options={'disp': False})
    if not res.success:
        raise Exception(f"linprog did not converge. Exit flag: {res.status}. Check the problem setup.")
    
    x = res.x
    # Retrieve model weights: difference between first and second block coefficients
    u1 = x[:n2 + 1] - x[n2 + 1: 2 * (n2 + 1)]
    # Predictions for the training data
    train_pred = np.dot(H, u1)
    # Construct design matrix for test data
    n_test = test.shape[0]
    Htest = np.hstack((test, np.ones((n_test, 1))))
    test_pred = np.dot(Htest, u1)
    # Calculate sparsity: percentage of small entries in u1
    sparsity = (np.sum(np.abs(u1) < 1e-4) * 100.0) / len(u1)
    return train_pred, test_pred, sparsity, u1

# Kernel function (if needed; currently only linear kernel is used)
def kernelfun(X, kerfPara, Y=None):
    if Y is None:
        Y = X
    if kerfPara.get('type', None) == 'rbf':
        gamma = kerfPara['pars']
        # Compute squared Euclidean distances
        sqdist = np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * np.dot(X, Y.T)
        K = np.exp(-gamma * sqdist)
    else:
        K = np.dot(X, Y.T)
    return K

def main():
    # Close any existing figures
    plt.close('all')
    
    # %% Step 1: Load the dataset and preprocess
    # Use the student-mat.csv dataset (semicolon-delimited)
    data = pd.read_csv(_DATASETS_DIR / "uci-secom.csv", delimiter=",", header=0, low_memory=False)
    
    # Convert remaining columns to numeric (non-convertible entries become NaN)
    data = data.apply(pd.to_numeric, errors='coerce')
    # Drop any columns that are entirely NaN (e.g., "call" columns)
    data = data.dropna(axis=1, how='all')
    
    print(f"Data shape after preprocessing: {data.shape}")
    data = data.to_numpy()
    print(f'Loaded dataset with {data.shape[0]} records and {data.shape[1]} numeric features.')
    
    # Impute NaN/inf values to prevent linprog errors
    imputer = SimpleImputer(strategy="mean")
    data = imputer.fit_transform(data)
    # Replace any remaining inf values with 0
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Separate predictor variables (X) and response variable (Y) 
    # (Assuming the last column is the response)
    X = data[:, :-1]
    Y = data[:, -1]
    
    # %% Step 2: Provide results before feature selection
    s = 2**0         # Kernel parameter
    c1 = 2**(-10)  # Regularization parameter
    c3 = 0.1         # Second regularization parameter specific to one-norm
    lower_quantiles = [0.025]
    
    # Split data into training (80%) and testing (20%) sets
    trainX, testX, ytrain, ytest = train_test_split(X, Y, test_size=0.2, random_state=42)
    ytrain = ytrain.flatten()
    ytest = ytest.flatten()
    
    results_before = []
    headers = ['Tau', 'RMSE (Lower, Upper)', 'Time', 'Sparsity (Lower, Upper)', 
               'Coverage Probability (Lower, Upper)', 'PICP', 'MPIW']
    results_before.append(headers)
    
    for tau_lower in lower_quantiles:
        tau_upper = 0.95 + tau_lower  # Complement quantile
        start_time = time.time()
        # Predict lower quantile
        _, Low_Q, sparsity_lower, _ = linear_quantile_lponenorm_tsvr(trainX, ytrain, testX, s, c3, c1, tau_lower)
        # Predict upper quantile
        _, Up_Q, sparsity_upper, _ = linear_quantile_lponenorm_tsvr(trainX, ytrain, testX, s, c3, c1, tau_upper)
        elapsed_time = time.time() - start_time
        
        CP_lower = np.mean(ytest <= Low_Q)
        CP_upper = np.mean(ytest <= Up_Q)
        PICP = np.mean((ytest >= Low_Q) & (ytest <= Up_Q))
        MPIW = np.mean(Up_Q - Low_Q)
        RMSE_lower = np.sqrt(np.mean((ytest - Low_Q) ** 2))
        RMSE_upper = np.sqrt(np.mean((ytest - Up_Q) ** 2))
        
        tau_str = f"({tau_lower}, {tau_upper})"
        rmse_str = f"{RMSE_lower:.4f}, {RMSE_upper:.4f}"
        time_str = f"{elapsed_time:.4f}"
        sparsity_str = f"{sparsity_lower:.4f}, {sparsity_upper:.4f}"
        cp_str = f"{CP_lower:.4f}, {CP_upper:.4f}"
        picp_str = f"{PICP:.4f}"
        mpiw_str = f"{MPIW:.4f}"
        results_before.append([tau_str, rmse_str, time_str, sparsity_str, cp_str, picp_str, mpiw_str])
    
    print("Detailed Results Before Feature Selection:")
    for i, row in enumerate(results_before[1:], start=1):
        print(f"--- Result {i} ---")
        for h, v in zip(headers, row):
            print(f"{h}: {v}")
        print("")
    
    # %% Step 3: Feature selection based on regression weights
    # Extract feature weights for lower and upper quantiles
    _, _, _, l_weights_full = linear_quantile_lponenorm_tsvr(trainX, ytrain, testX, s, c3, c1, lower_quantiles[0])
    _, _, _, u_weights_full = linear_quantile_lponenorm_tsvr(trainX, ytrain, testX, s, c3, c1, 0.95 + lower_quantiles[0])
    n_features = trainX.shape[1]
    l_feature_importance = np.abs(l_weights_full[:n_features])
    u_feature_importance = np.abs(u_weights_full[:n_features])
    threshold = 0.0001
    
    # Taking intersection: invalid if weight is less than threshold in both quantile estimates
    l_invalid_indices = np.where(l_feature_importance < threshold)[0]
    u_invalid_indices = np.where(u_feature_importance < threshold)[0]
    invalid_indices = np.intersect1d(l_invalid_indices, u_invalid_indices)
    valid_indices = np.setdiff1d(np.arange(n_features), invalid_indices)
    
    print("Total number of features:", n_features)
    print("Number of features dropped:", len(invalid_indices))
    print("Number of features retained:", len(valid_indices))
    if len(invalid_indices) > 0:
        print("Dropped feature indices (0-indexed):", invalid_indices)
        print("These features were dropped because their absolute weights in both quantile ranges were below the threshold of", threshold)
    print("Selected feature indices (0-indexed):", valid_indices)
    
    # %% Step 4: Provide results after feature selection
    trainX_selected = trainX[:, valid_indices]
    testX_selected = testX[:, valid_indices]
    
    results_after = []
    results_after.append(headers)
    l_selected_weights_at = None
    u_selected_weights_at = None
    
    for tau_lower in lower_quantiles:
        tau_upper = 0.90 + tau_lower
        start_time = time.time()
        _, Low_Q, sparsity_lower, l_selected_weights_at = linear_quantile_lponenorm_tsvr(
            trainX_selected, ytrain, testX_selected, s, c3, c1, tau_lower)
        _, Up_Q, sparsity_upper, u_selected_weights_at = linear_quantile_lponenorm_tsvr(
            trainX_selected, ytrain, testX_selected, s, c3, c1, tau_upper)
        elapsed_time = time.time() - start_time
        
        CP_lower = np.mean(ytest <= Low_Q)
        CP_upper = np.mean(ytest <= Up_Q)
        PICP = np.mean((ytest >= Low_Q) & (ytest <= Up_Q))
        MPIW = np.mean(Up_Q - Low_Q)
        RMSE_lower = np.sqrt(np.mean((ytest - Low_Q) ** 2))
        RMSE_upper = np.sqrt(np.mean((ytest - Up_Q) ** 2))
        
        tau_str = f"({tau_lower}, {tau_upper})"
        rmse_str = f"{RMSE_lower:.4f}, {RMSE_upper:.4f}"
        time_str = f"{elapsed_time:.4f}"
        sparsity_str = f"{sparsity_lower:.4f}, {sparsity_upper:.4f}"
        cp_str = f"{CP_lower:.4f}, {CP_upper:.4f}"
        picp_str = f"{PICP:.4f}"
        mpiw_str = f"{MPIW:.4f}"
        results_after.append([tau_str, rmse_str, time_str, sparsity_str, cp_str, picp_str, mpiw_str])
    
    print("Detailed Results After Feature Selection:")
    for i, row in enumerate(results_after[1:], start=1):
        print(f"--- Result {i} ---")
        for h, v in zip(headers, row):
            print(f"{h}: {v}")
        print("")
    
    # %% Step 5: Plotting Feature Weights
    plt.figure()
    plt.bar(np.arange(1, n_features + 1), u_weights_full[:n_features])
    plt.title('Original Upper Features and Their Weights (Bar Plot)')
    plt.xlabel('Feature Index')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.xticks(np.arange(1, n_features + 1))
    
    plt.figure()
    plt.plot(np.arange(1, n_features + 1), l_weights_full[:n_features], marker='o')
    plt.title('Original Lower Features and Their Weights (Line Plot)')
    plt.xlabel('Feature Index')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.xticks(np.arange(1, n_features + 1))
    
    plt.figure()
    plt.bar(np.arange(1, len(valid_indices) + 1), u_selected_weights_at[:len(valid_indices)])
    plt.title('Selected Upper Features and Their Weights (Bar Plot)')
    plt.xlabel('Feature Index (Selected)')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.xticks(np.arange(1, len(valid_indices) + 1), valid_indices + 1)
    
    plt.figure()
    plt.plot(np.arange(1, len(valid_indices) + 1), l_selected_weights_at[:len(valid_indices)], marker='o')
    plt.title('Selected Lower Features and Their Weights (Line Plot)')
    plt.xlabel('Feature Index (Selected)')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.xticks(np.arange(1, len(valid_indices) + 1), valid_indices + 1)
    
    plt.show()

if __name__ == "__main__":
    main()