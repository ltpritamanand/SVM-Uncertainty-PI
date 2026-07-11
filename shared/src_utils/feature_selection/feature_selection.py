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
# feature_selection.py lives at final_github/shared/src_utils/feature_selection/
_DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"

# Function to run the linear quantile SVR using linear programming
def linear_quantile_lponenorm_tsvr(train, ytrain, test, s, c3, c1, tau1):
    """
    Solves the quantile regression problem using linear programming.
    Returns predictions for training and test sets, sparsity percentage, and the weight vector u1.
    """
    n1, n2 = train.shape
    # Create design matrix H with appended ones for bias
    H = np.hstack((train, np.ones((n1, 1))))
    # Objective function: concatenation of four blocks 
    f = np.concatenate([
        c3 * np.ones(n2 + 1),
        c3 * np.ones(n2 + 1),
        c1 * tau1 * np.ones(n1),
        c1 * (1 - tau1) * np.ones(n1)
    ])
    # Construct inequality constraints
    A1 = np.hstack(( -H,  H, -np.eye(n1), np.zeros((n1, n1)) ))
    A2 = np.hstack(( H, -H, np.zeros((n1, n1)), -np.eye(n1) ))
    A = np.vstack((A1, A2))
    b = np.concatenate((-ytrain, ytrain))
    num_vars = len(f)
    bounds = [(0, None)] * num_vars

    # Solve the LP using the HiGHS solver
    res = linprog(f, A_ub=A, b_ub=b, bounds=bounds, method='highs', options={'disp': False})
    if not res.success:
        raise Exception(f"linprog did not converge. Exit flag: {res.status}. Check the problem setup.")
    
    x = res.x
    # Compute weight vector as difference between the two blocks
    u1 = x[:n2 + 1] - x[n2 + 1:2 * (n2 + 1)]
    train_pred = np.dot(H, u1)
    n_test = test.shape[0]
    Htest = np.hstack((test, np.ones((n_test, 1))))
    test_pred = np.dot(Htest, u1)
    sparsity = (np.sum(np.abs(u1) < 1e-4) * 100.0) / len(u1)
    return train_pred, test_pred, sparsity, u1

def main():
    plt.close('all')
    
    # %% Step 1: Load the Boston Housing dataset
    data_path = _DATASETS_DIR / "bostonhousingdata.xlsx"
    data = pd.read_excel(data_path, header=0)
    
    # Convert to numeric and impute missing values (if any)
    data = data.apply(pd.to_numeric, errors='coerce')
    imputer = SimpleImputer(strategy="mean")
    data_imputed = pd.DataFrame(imputer.fit_transform(data))
    print(f"Data shape after imputation: {data_imputed.shape}")
    
    # Convert to numpy array
    data = data_imputed.to_numpy()
    print(f"Dataset loaded with {data.shape[0]} records and {data.shape[1]} features!")
    
    # %% Step 2: Split data into 60% train, 20% validation, 20% test
    # 1) Hold out the test set (20% of original)
    X_temp, testX, y_temp, ytest = train_test_split(data[:, :-1], data[:, -1], test_size=0.2, random_state=42)

    # 2) Split the remaining 80% into train (60%) and validation (20%)
    trainX, valX, ytrain, yval = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

    # Flatten the targets
    ytrain = ytrain.flatten()
    yval   = yval.flatten()
    ytest  = ytest.flatten()
    
    # %% Step 3: Before feature selection
    # Parameters for quantile SVR
    s = 2**0      # Kernel parameter
    c1 = 2**(-6.80) # Regularization parameter
    c3 = 0.1      # Second reg. parameter (one-norm)
    lower_quantiles = [0.025]
    
    results_before = []
    headers = ['Tau', 'RMSE (Lower, Upper)', 'Time', 'Sparsity (Lower, Upper)', 
               'Coverage Probability (Lower, Upper)', 'PICP', 'MPIW']
    results_before.append(headers)
    
    for tau_lower in lower_quantiles:
        tau_upper = 0.95 + tau_lower
        start_time = time.time()
        # Predict at lower and upper quantiles
        _, Low_Q, sparsity_lower, _ = linear_quantile_lponenorm_tsvr(trainX, ytrain, testX, s, c3, c1, tau_lower)
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
        print(f"{headers[0]}: {row[0]}")
        print(f"{headers[1]}: {row[1]}")
        print(f"{headers[2]}: {row[2]}")
        print(f"{headers[3]}: {row[3]}")
        print(f"{headers[4]}: {row[4]}")
        print(f"{headers[5]}: {row[5]}")
        print(f"{headers[6]}: {row[6]}")
        print("")
    
    # %% Step 4: Feature Selection
    # Here, we select features based on the regression weights (drop those where both quantile weights are below threshold)
    _, _, _, l_weights_full = linear_quantile_lponenorm_tsvr(trainX, ytrain, testX, s, c3, c1, lower_quantiles[0])
    _, _, _, u_weights_full = linear_quantile_lponenorm_tsvr(trainX, ytrain, testX, s, c3, c1, 0.95 + lower_quantiles[0])
    n_features = trainX.shape[1]
    l_feature_importance = np.abs(l_weights_full[:n_features])
    u_feature_importance = np.abs(u_weights_full[:n_features])
    threshold = 0.0001
    l_invalid_indices = np.where(l_feature_importance < threshold)[0]
    u_invalid_indices = np.where(u_feature_importance < threshold)[0]
    invalid_indices = np.intersect1d(l_invalid_indices, u_invalid_indices)
    valid_indices = np.setdiff1d(np.arange(n_features), invalid_indices)
    
    print("Total number of features:", n_features)
    print("Number of features dropped:", len(invalid_indices))
    print("Number of features retained:", len(valid_indices))
    if len(invalid_indices) > 0:
        print("Dropped feature indices (0-indexed):", invalid_indices)
    print("Selected feature indices (0-indexed):", valid_indices)
    
    # %% Step 5: After feature selection: evaluating performance on selected features
    trainX_selected = trainX[:, valid_indices]
    testX_selected = testX[:, valid_indices]
    
    results_after = []
    results_after.append(headers)
    
    for tau_lower in lower_quantiles:
        tau_upper = 0.95 + tau_lower
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
        print(f"{headers[0]}: {row[0]}")
        print(f"{headers[1]}: {row[1]}")
        print(f"{headers[2]}: {row[2]}")
        print(f"{headers[3]}: {row[3]}")
        print(f"{headers[4]}: {row[4]}")
        print(f"{headers[5]}: {row[5]}")
        print(f"{headers[6]}: {row[6]}")
        print("")
    
    # %% Step 6: Plotting Feature Weights
    plt.figure()
    plt.bar(np.arange(1, n_features + 1), u_weights_full[:n_features])
    plt.title('Original Upper Features and Their Weights (Bar Plot)')
    plt.xlabel('Feature Index')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.xticks(np.arange(1, n_features + 1))
    
    plt.figure()
    plt.bar(np.arange(1, n_features + 1), l_weights_full[:n_features])
    plt.title('Original Lower Features and Their Weights (Bar Plot)')
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
    plt.xticks(np.arange(1, len(valid_indices) + 1), valid_indices + 1)  # converting to 1-indexed
    
    plt.figure()
    plt.bar(np.arange(1, len(valid_indices) + 1), l_selected_weights_at[:len(valid_indices)])
    plt.title('Selected Lower Features and Their Weights (Bar Plot)')
    plt.xlabel('Feature Index (Selected)')
    plt.ylabel('Weight')
    plt.grid(True)
    plt.xticks(np.arange(1, len(valid_indices) + 1), valid_indices + 1)
    
    plt.show()

if __name__ == "__main__":
    main()
