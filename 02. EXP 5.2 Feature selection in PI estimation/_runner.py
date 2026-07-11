# _runner.py
#
# Shared runner for the per-dataset scripts (run_boston.py, run_student.py,
# run_spambase.py, run_secom.py, run_madelon.py, run_all.py).
#
# Each dataset script hardcodes its best-known (s, c1, c3, threshold) from
# the tuning grid and calls run_dataset() below to produce the Before/After
# FS row in the same format as the paper's Table 2.

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent

# Dataset registry — path resolved against this script's data/ folder.
DATASETS = {
    "boston":   "bostonhousingdata.xlsx",
    "student":  "student-mat.csv",
    "spambase": "spambase.data",
    "secom":    "uci-secom.csv",
    "madelon":  "MADELON/madelon_train.data",
}


def load_dataset(data_path):
    """Auto-dispatch on file extension. Returns array with target in last col."""
    ext = data_path.suffix.lower()
    if data_path.name == "madelon_train.data":
        X = np.loadtxt(data_path)
        y = np.loadtxt(data_path.parent / "madelon_train.labels").astype(float)
        return np.hstack([X, y.reshape(-1, 1)])
    if ext in {".xlsx", ".xls"}:
        data = pd.read_excel(data_path, header=0)
    else:
        sep = ";" if data_path.name == "student-mat.csv" else ","
        data = pd.read_csv(data_path, delimiter=sep, header=0, low_memory=False)
    data = data.apply(pd.to_numeric, errors="coerce")
    data = data.dropna(axis=1, how="all")
    imputer = SimpleImputer(strategy="mean")
    data = pd.DataFrame(imputer.fit_transform(data))
    arr = data.to_numpy()
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def linear_quantile_lponenorm_tsvr(train, ytrain, test, s, c3, c1, tau1):
    """LP quantile SVR via HiGHS. Returns (train_pred, test_pred, sparsity, u1)."""
    n1, n2 = train.shape
    H = np.hstack((train, np.ones((n1, 1))))
    f = np.concatenate([
        c3 * np.ones(n2 + 1), c3 * np.ones(n2 + 1),
        c1 * tau1 * np.ones(n1), c1 * (1 - tau1) * np.ones(n1),
    ])
    A1 = np.hstack((-H,  H, -np.eye(n1), np.zeros((n1, n1))))
    A2 = np.hstack(( H, -H,  np.zeros((n1, n1)), -np.eye(n1)))
    A = np.vstack((A1, A2))
    b = np.concatenate((-ytrain, ytrain))
    bounds = [(0, None)] * len(f)
    res = linprog(f, A_ub=A, b_ub=b, bounds=bounds, method="highs", options={"disp": False})
    if not res.success:
        raise RuntimeError(f"linprog failed: {res.status}")
    x = res.x
    u1 = x[:n2 + 1] - x[n2 + 1: 2 * (n2 + 1)]
    train_pred = H @ u1
    Htest = np.hstack((test, np.ones((test.shape[0], 1))))
    test_pred = Htest @ u1
    sparsity = (np.sum(np.abs(u1) < 1e-4) * 100.0) / len(u1)
    return train_pred, test_pred, sparsity, u1


def _eval(trainX, ytrain, testX, ytest, s, c1, c3, tau_l=0.025, tau_u=0.975):
    """Fit LP quantile SVR at lower/upper tau. Returns PICP, MPIW, time, weights."""
    t0 = time.time()
    _, Low_Q, _, l_w = linear_quantile_lponenorm_tsvr(trainX, ytrain, testX, s, c3, c1, tau_l)
    _, Up_Q,  _, u_w = linear_quantile_lponenorm_tsvr(trainX, ytrain, testX, s, c3, c1, tau_u)
    elapsed = time.time() - t0
    PICP = float(np.mean((ytest >= Low_Q) & (ytest <= Up_Q)))
    MPIW = float(np.mean(Up_Q - Low_Q))
    return PICP, MPIW, elapsed, l_w, u_w


def run_dataset(name, s, c1, c3, threshold, verbose=True):
    """Run Before-FS + After-FS at fixed params. Returns a result dict."""
    if verbose:
        print(f"\n{'='*72}\n{name.upper()}  (s={s:g}, c1={c1:g}, c3={c3:g}, thr={threshold:g})\n{'='*72}")
    data_path = SCRIPT_DIR / "data" / DATASETS[name]
    data = load_dataset(data_path)
    n_rows, n_cols = data.shape
    d = n_cols - 1

    X, y = data[:, :-1], data[:, -1]
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    trainX, testX, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=42)
    ytrain, ytest = ytrain.flatten(), ytest.flatten()

    # Before FS
    PICP_b, MPIW_b, T_b, l_w, u_w = _eval(trainX, ytrain, testX, ytest, s, c1, c3)

    # Feature drop
    li = np.where(np.abs(l_w[:d]) < threshold)[0]
    ui = np.where(np.abs(u_w[:d]) < threshold)[0]
    drop = np.intersect1d(li, ui)
    keep = np.setdiff1d(np.arange(d), drop)
    red_pct = 100.0 * len(drop) / d

    # After FS
    if len(keep) == 0:
        PICP_a, MPIW_a, T_a = float("nan"), float("nan"), 0.0
    else:
        PICP_a, MPIW_a, T_a, _, _ = _eval(trainX[:, keep], ytrain, testX[:, keep], ytest, s, c1, c3)

    if verbose:
        print(f"| {name.capitalize():<14} | ({n_rows},{d}) | "
              f"{PICP_b:.3f} | {MPIW_b:.3f} | {T_b:.2f} | "
              f"{PICP_a:.3f} | {MPIW_a:.3f} | {T_a:.2f} | {red_pct:.1f}% |")

    return {
        "dataset": name, "shape": [n_rows, d],
        "params": {"s": s, "c1": c1, "c3": c3, "threshold": threshold},
        "before": {"PICP": PICP_b, "MPIW": MPIW_b, "time": T_b},
        "after":  {"PICP": PICP_a, "MPIW": MPIW_a, "time": T_a,
                    "features_kept": int(len(keep)), "reduction_pct": red_pct},
    }


def print_table_header():
    print("| Dataset        | Dim.        | PICP (before) | MPIW (before) | Time (before) | "
          "PICP (after) | MPIW (after) | Time (after) | % Red. Feature |")
    print("|---|---|---|---|---|---|---|---|---|")
