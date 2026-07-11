"""
Python port of probab_forecast_LPSvm.m
Sparse SVQR (L1-norm) probabilistic forecast on beer.csv.

Grid search over window_size, kernel gamma (s), and regularisation C.
Evaluate on validation; re-run best config on test set.

Fixes applied vs original MATLAB:
  1. Consistent kernel: exp(-gamma * ||x-y||^2) everywhere
  2. Plot uses BEST params, not last iteration
  3. Test set is evaluated after finding best params
  4. Windowed data built once per window size (not every inner iteration)
  5. Grid unchanged (25 x 51 x 51 = 65,025 combos)

Usage:
  python probab_forecast_lpsvm.py
  python probab_forecast_lpsvm.py --quick   # reduced grid for testing
"""

import argparse
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from quantile_lp_onenorm import quantile_lp_onenorm
from evaluate_picp import evaluate_picp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ─────────────────────────── CONFIG (matches MATLAB) ──────────────────────
Q_LOWER    = 0.025
Q_UPPER    = 0.975
TARGET_COV = 0.95
C3         = 1.0          # L1 norm coefficient (fixed)

# Grid (same as MATLAB: win 1..25, s and C from 2^-25 to 2^25)
WIN_SIZES  = list(range(1, 26))                            # 25
S_EXPS     = list(range(-25, 26))                          # 51
C_EXPS     = list(range(-25, 26))                          # 51


def load_beer():
    csv_path = os.path.join(SCRIPT_DIR, "beer.csv")
    df = pd.read_csv(csv_path, header=None)
    y = df.values.flatten().astype(np.float64)
    return y


def build_windows(y, win):
    """Sliding window features + targets (same as MATLAB lines 40-44)."""
    n = len(y) - win
    X = np.empty((n, win), dtype=np.float64)
    tgt = np.empty(n, dtype=np.float64)
    for i in range(n):
        X[i] = y[i:i + win]
        tgt[i] = y[i + win]
    return X, tgt


def split_train_val_test(X, y):
    """Same split as MATLAB:
       test  = last 30%
       train = first 70%, then split 90/10 -> train/val
    """
    n = len(y)
    n_trainval = int(np.floor(n * 0.7))

    X_test  = X[n_trainval:]
    y_test  = y[n_trainval:]

    X_tv = X[:n_trainval]
    y_tv = y[:n_trainval]

    n_train = int(np.floor(len(y_tv) * 0.9))
    X_train = X_tv[:n_train]
    y_train = y_tv[:n_train]
    X_val   = X_tv[n_train:]
    y_val   = y_tv[n_train:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def run_grid_search(y, win_sizes, s_exps, c_exps, q_lower, q_upper, c3):
    """Grid search: for each (win, s, C), solve LP-SVQR and evaluate on val."""
    total = len(win_sizes) * len(s_exps) * len(c_exps)
    print(f"  Grid: {len(win_sizes)} x {len(s_exps)} x {len(c_exps)} = {total} combos")

    # Store PICP and MPIW for each combo
    shape = (len(win_sizes), len(s_exps), len(c_exps))
    picp_grid = np.full(shape, np.nan)
    mpiw_grid = np.full(shape, np.nan)

    t0 = time.time()
    count = 0

    for ii, win in enumerate(win_sizes):
        # FIX 5: build windowed data ONCE per window size
        X_all, y_all = build_windows(y, win)
        X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X_all, y_all)

        for jj, s_exp in enumerate(s_exps):
            s = 2.0 ** s_exp

            for kk, c_exp in enumerate(c_exps):
                C1 = 2.0 ** c_exp
                count += 1

                try:
                    _, pred_lo, _ = quantile_lp_onenorm(
                        X_train, y_train, X_val, gamma=s, c3=c3, c1=C1, tau=q_lower)
                    _, pred_hi, _ = quantile_lp_onenorm(
                        X_train, y_train, X_val, gamma=s, c3=c3, c1=C1, tau=q_upper)

                    picp, mpiw = evaluate_picp(y_val, pred_lo, pred_hi)
                    picp_grid[ii, jj, kk] = picp
                    mpiw_grid[ii, jj, kk] = mpiw
                except Exception:
                    pass

                if count % 500 == 0 or count == total:
                    elapsed = time.time() - t0
                    print(f"    [{count:>6}/{total}]  "
                          f"win={win}  s=2^{s_exp}  C=2^{c_exp}  "
                          f"({elapsed:.0f}s elapsed)")

    return picp_grid, mpiw_grid


def find_best(picp_grid, mpiw_grid, win_sizes, s_exps, c_exps, threshold):
    """Find config with PICP > threshold and minimum MPIW (same as MATLAB)."""
    valid = picp_grid > threshold
    if not np.any(valid):
        print(f"  No configs with PICP > {threshold}. Using max PICP instead.")
        flat = np.nanargmax(picp_grid)
        ii, jj, kk = np.unravel_index(flat, picp_grid.shape)
    else:
        # Among valid configs, find minimum MPIW
        masked = np.where(valid, mpiw_grid, np.inf)
        flat = np.argmin(masked)
        ii, jj, kk = np.unravel_index(flat, picp_grid.shape)

    best_win = win_sizes[ii]
    best_s   = 2.0 ** s_exps[jj]
    best_C   = 2.0 ** c_exps[kk]
    best_picp = picp_grid[ii, jj, kk]
    best_mpiw = mpiw_grid[ii, jj, kk]

    return {
        "win": best_win,
        "s_exp": s_exps[jj], "s": best_s,
        "c_exp": c_exps[kk], "C": best_C,
        "val_picp": best_picp, "val_mpiw": best_mpiw,
        "idx": (ii, jj, kk),
    }


def evaluate_best_on_test(y, best, q_lower, q_upper, c3):
    """FIX 3: re-run with best params and evaluate on TEST set."""
    X_all, y_all = build_windows(y, best["win"])
    X_train, y_train, X_val, y_val, X_test, y_test = split_train_val_test(X_all, y_all)

    # Validation predictions (for plotting)
    _, val_lo, sp_lo = quantile_lp_onenorm(
        X_train, y_train, X_val, gamma=best["s"], c3=c3, c1=best["C"], tau=q_lower)
    _, val_hi, sp_hi = quantile_lp_onenorm(
        X_train, y_train, X_val, gamma=best["s"], c3=c3, c1=best["C"], tau=q_upper)
    val_picp, val_mpiw = evaluate_picp(y_val, val_lo, val_hi)

    # Test predictions (FIX 3)
    _, test_lo, _ = quantile_lp_onenorm(
        X_train, y_train, X_test, gamma=best["s"], c3=c3, c1=best["C"], tau=q_lower)
    _, test_hi, _ = quantile_lp_onenorm(
        X_train, y_train, X_test, gamma=best["s"], c3=c3, c1=best["C"], tau=q_upper)
    test_picp, test_mpiw = evaluate_picp(y_test, test_lo, test_hi)

    return {
        "val_lo": val_lo, "val_hi": val_hi, "y_val": y_val,
        "val_picp": val_picp, "val_mpiw": val_mpiw,
        "test_lo": test_lo, "test_hi": test_hi, "y_test": y_test,
        "test_picp": test_picp, "test_mpiw": test_mpiw,
        "sparsity_lo": sp_lo, "sparsity_hi": sp_hi,
        "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
    }


# ─────────────────────────── PLOTTING (FIX 2: plots best, not last) ──────
def plot_results(results, best, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Validation PI plot
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.arange(len(results["y_val"]))
    ax.plot(idx, results["y_val"], "b-", lw=0.8, label="Actual")
    ax.plot(idx, results["val_lo"], "r-", lw=1, label="Lower PI")
    ax.plot(idx, results["val_hi"], "k-", lw=1, label="Upper PI")
    ax.set_title(f"LP-SVM Validation | win={best['win']}  s=2^{best['s_exp']}  "
                 f"C=2^{best['c_exp']}\n"
                 f"PICP={results['val_picp']:.4f}  MPIW={results['val_mpiw']:.4f}")
    ax.set_xlabel("Index"); ax.set_ylabel("Value")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lpsvm_val_pi.png"), dpi=120)
    plt.close(fig)

    # Test PI plot (FIX 3)
    fig, ax = plt.subplots(figsize=(10, 4))
    idx = np.arange(len(results["y_test"]))
    ax.plot(idx, results["y_test"], "b-", lw=0.8, label="Actual")
    ax.plot(idx, results["test_lo"], "r-", lw=1, label="Lower PI")
    ax.plot(idx, results["test_hi"], "k-", lw=1, label="Upper PI")
    ax.set_title(f"LP-SVM Test | win={best['win']}  s=2^{best['s_exp']}  "
                 f"C=2^{best['c_exp']}\n"
                 f"PICP={results['test_picp']:.4f}  MPIW={results['test_mpiw']:.4f}")
    ax.set_xlabel("Index"); ax.set_ylabel("Value")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "lpsvm_test_pi.png"), dpi=120)
    plt.close(fig)

    print(f"  Plots -> {out_dir}/")


def write_results(results, best, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = os.path.join(out_dir, f"lpsvm_results_{ts}.txt")

    with open(path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  LP-SVM (Sparse SVQR) Probabilistic Forecast\n")
        f.write("=" * 60 + "\n\n")
        f.write("Best Hyperparameters (selected on validation)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Window size  : {best['win']}\n")
        f.write(f"  Kernel gamma : 2^{best['s_exp']} = {best['s']}\n")
        f.write(f"  C (pinball)  : 2^{best['c_exp']} = {best['C']}\n")
        f.write(f"  c3 (L1 norm) : {C3}\n")
        f.write(f"  q_lower      : {Q_LOWER}\n")
        f.write(f"  q_upper      : {Q_UPPER}\n\n")
        f.write("Data Split\n")
        f.write("-" * 40 + "\n")
        f.write(f"  n_train : {results['n_train']}\n")
        f.write(f"  n_val   : {results['n_val']}\n")
        f.write(f"  n_test  : {results['n_test']}\n\n")
        f.write("Results\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Validation PICP : {results['val_picp']:.4f}\n")
        f.write(f"  Validation MPIW : {results['val_mpiw']:.4f}\n")
        f.write(f"  Test PICP       : {results['test_picp']:.4f}\n")
        f.write(f"  Test MPIW       : {results['test_mpiw']:.4f}\n\n")
        f.write(f"  Sparsity (lower): {results['sparsity_lo']:.4f}\n")
        f.write(f"  Sparsity (upper): {results['sparsity_hi']:.4f}\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")

    print(f"  Results -> {path}")


# ─────────────────────────── MAIN ────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Reduced grid for fast testing")
    args = ap.parse_args()

    # Reduced grid for quick testing
    if args.quick:
        win_sizes = [1, 5, 10, 15, 20]
        s_exps    = list(range(-10, 11, 2))
        c_exps    = list(range(-10, 11, 2))
    else:
        win_sizes = WIN_SIZES
        s_exps    = S_EXPS
        c_exps    = C_EXPS

    print(f"\nLoading beer.csv...")
    y = load_beer()
    print(f"  Data length: {len(y)}")

    print(f"\nRunning LP-SVM grid search...")
    t0 = time.time()
    picp_grid, mpiw_grid = run_grid_search(
        y, win_sizes, s_exps, c_exps, Q_LOWER, Q_UPPER, C3)
    elapsed = time.time() - t0
    print(f"  Grid search done in {elapsed:.1f}s")

    best = find_best(picp_grid, mpiw_grid, win_sizes, s_exps, c_exps, TARGET_COV)
    print(f"\n  Best config:")
    print(f"    win={best['win']}  s=2^{best['s_exp']}  C=2^{best['c_exp']}")
    print(f"    Val PICP={best['val_picp']:.4f}  Val MPIW={best['val_mpiw']:.4f}")

    print(f"\nRe-running best config on val + test...")
    results = evaluate_best_on_test(y, best, Q_LOWER, Q_UPPER, C3)
    print(f"    Val  PICP={results['val_picp']:.4f}  MPIW={results['val_mpiw']:.4f}")
    print(f"    Test PICP={results['test_picp']:.4f}  MPIW={results['test_mpiw']:.4f}")
    print(f"    Sparsity: lo={results['sparsity_lo']:.4f}  hi={results['sparsity_hi']:.4f}")

    out_dir = os.path.join(SCRIPT_DIR, "results")
    plot_results(results, best, out_dir)
    write_results(results, best, out_dir)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
