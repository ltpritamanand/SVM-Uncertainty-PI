# feature_selection.py

import os
import sys
# make sure the workspace root is on PYTHONPATH so "cqr.py" can be found
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# import the CQR routine
from cqr import train_two_model_cqr_conformal


def main():
    plt.close('all')

    # %% Step 1: Load & preprocess the Madelon dataset
    # madelon_train.data and madelon_train.labels should be in MADELON/ subfolder
    data_df = pd.read_csv("MADELON/madelon_train.data",
                          header=None, delim_whitespace=True)
    labels_df = pd.read_csv("MADELON/madelon_train.labels",
                            header=None, delim_whitespace=True)

    # Ensure numeric and impute any missing entries
    data_df = data_df.apply(pd.to_numeric, errors='coerce')
    labels_df = labels_df.apply(pd.to_numeric, errors='coerce')
    imputer = SimpleImputer(strategy="mean")
    X_full = imputer.fit_transform(data_df)
    y_full = imputer.fit_transform(labels_df).flatten()

    print(f"Data shape after imputation: {X_full.shape}")

    # %% Step 2: Split into Train / Calibration / Test (60% / 20% / 20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X_full, y_full, test_size=0.20, random_state=42
    )
    X_train, X_cal, y_train, y_cal = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42
    )

    # reshape targets for CQR compatibility
    y_train = y_train.reshape(-1, 1)
    y_cal   = y_cal.reshape(-1, 1)
    y_test  = y_test.reshape(-1, 1)

    # %% Step 3: Before Feature Selection using CQR
    q_low, q_high, alpha = 0.025, 0.975, 0.05
    print("\n=== Before Feature Selection (CQR) ===")
    time_before, picp_before, mpiw_before, low_cov_before, high_cov_before, \
        model_lo, model_hi, lb_before, ub_before = train_two_model_cqr_conformal(
            X_train, y_train, X_cal, y_cal, X_test, y_test,
            q_low=q_low, q_high=q_high, alpha=alpha, verbose=False
        )
    print(f"Time: {time_before:.2f}s  PICP: {picp_before:.4f}  MPIW: {mpiw_before:.4f}")

    # %% Step 4: Manual Feature Drop (hard-coded)
    # From previous runs we drop all but these 8 features:
    keep_idx = np.array([90, 228, 276, 332, 402, 404, 423, 445], dtype=int)
    n_features = X_train.shape[1]
    drop_idx = np.setdiff1d(np.arange(n_features), keep_idx)

    print("\nUsing manual feature drop:")
    print(f"  Total features            : {n_features}")
    print(f"  Number dropped            : {drop_idx.size}")
    print(f"  Dropped indices           : {drop_idx.tolist()}")
    print(f"  Number retained           : {keep_idx.size}")
    print(f"  Retained indices          : {keep_idx.tolist()}")

    # %% Step 5: After Feature Selection using CQR
    print("\n=== After Feature Selection (CQR) ===")
    X_train_sel = X_train[:, keep_idx]
    X_cal_sel   = X_cal[:, keep_idx]
    X_test_sel  = X_test[:, keep_idx]

    time_after, picp_after, mpiw_after, low_cov_after, high_cov_after, \
        model_lo_sel, model_hi_sel, lb_after, ub_after = train_two_model_cqr_conformal(
            X_train_sel, y_train, X_cal_sel, y_cal, X_test_sel, y_test,
            q_low=q_low, q_high=q_high, alpha=alpha, verbose=False
        )
    print(f"Time: {time_after:.2f}s  PICP: {picp_after:.4f}  MPIW: {mpiw_after:.4f}")

    # %% Step 6: Plotting Feature Importances Before and After FS
    # Full-model input‐layer importances
    W_lo_full = model_lo.layers[0].get_weights()[0]  # shape (500, hidden_units)
    W_hi_full = model_hi.layers[0].get_weights()[0]
    imp_lo_full = np.sum(np.abs(W_lo_full), axis=1)
    imp_hi_full = np.sum(np.abs(W_hi_full), axis=1)

    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(1, n_features+1), imp_hi_full, color='tab:blue')
    plt.title("Upper Quantile Feature Importances Before FS")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.grid(True)

    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(1, n_features+1), imp_lo_full, color='tab:orange')
    plt.title("Lower Quantile Feature Importances Before FS")
    plt.xlabel("Feature Index")
    plt.ylabel("Importance")
    plt.grid(True)

    # Selected-feature importances after retraining
    W_lo_sel = model_lo_sel.layers[0].get_weights()[0]
    W_hi_sel = model_hi_sel.layers[0].get_weights()[0]
    imp_lo_sel = np.sum(np.abs(W_lo_sel), axis=1)
    imp_hi_sel = np.sum(np.abs(W_hi_sel), axis=1)

    plt.figure(figsize=(6, 3))
    plt.bar(np.arange(1, len(keep_idx)+1), imp_hi_sel, color='tab:blue')
    plt.title("Upper Quantile Importances After FS")
    plt.xlabel("Selected Feature #")
    plt.ylabel("Importance")
    plt.xticks(np.arange(1, len(keep_idx)+1), (keep_idx+1).tolist(), rotation=90)
    plt.grid(True)

    plt.figure(figsize=(6, 3))
    plt.bar(np.arange(1, len(keep_idx)+1), imp_lo_sel, color='tab:orange')
    plt.title("Lower Quantile Importances After FS")
    plt.xlabel("Selected Feature #")
    plt.ylabel("Importance")
    plt.xticks(np.arange(1, len(keep_idx)+1), (keep_idx+1).tolist(), rotation=90)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()