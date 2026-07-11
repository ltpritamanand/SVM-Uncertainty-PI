#!/usr/bin/env python3
"""
Exp 15 — Hyperparameter tuning + benchmark for CQR-NN, SVQR+CP, QRF+CP, XGB+CP.

Strategy per dataset:
  80% train+val  →  used for tuning + benchmark (re-split each data seed)
  20% test        (held out at random_state=42, shared across all D x I runs)

Tuning (single split, fixed seed=0):
  grid selects config with:
    primary  : PICP >= TARGET_COVERAGE (0.90)  → prefer coverage
    secondary: smallest MPIW

Benchmark (nested D=10 data seeds x I=10 init seeds):
  outer test split fixed; each data seed re-splits (train_proper, cal); each
  init seed re-inits the model.  SVQR is deterministic → I=1, σ_opt=0.
  σ_opt = sqrt(E_d[Var_i[f]])   — model-init uncertainty
  σ_in  = sqrt(Var_d[E_i[f]])   — data-split uncertainty

After tuning:
  - Saves best params to  outputs/hparams/best_params_DATASET.json
  - Runs D x I benchmark with best params
  - Saves results to  outputs/results/tuned_TIMESTAMP.{csv,txt}
"""

import itertools
import json
import time
import warnings
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import Bunch

# --- local helpers ---
sys.path.insert(0, str(Path(__file__).parent))
from svqr_qp import solve_qp, predict_svqr
from cqr import train_two_model_cqr_conformal

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ALPHA            = 0.1
TARGET_COVERAGE  = 1 - ALPHA          # 0.90
Q_LOW, Q_HIGH    = 0.05, 0.95

DATASETS_PATH = str(
    Path(__file__).resolve().parents[1] / "shared" / "datasets"
) + "/"

SCRIPT_DIR = Path(__file__).parent
HPARAM_DIR = SCRIPT_DIR / "outputs" / "hparams"
RESULT_DIR = SCRIPT_DIR / "outputs" / "results"
HPARAM_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Hyperparameter grids
# ---------------------------------------------------------------------------

CQR_NN_GRID = list(itertools.product(
    [32, 64,128, 256, 512],          # hidden_dim
    [100, 200, 300],         # epochs
    [1e-3, 1e-2, 5e-2],      # learning_rate
    [32, 64,128,256],            # batch_size
))  # 3×3×3×3 = 81 configs

# Paper convention (probab_forecast_Svm.m: s1val = -25:1:25, s1 = 2^s1val).
# gamma = 2^gamma_exp directly — no /n_features scaling.
# Three grids:
#   'default' : step 3, integer exponents (17×17 = 289 configs — fast)
#   'fine'    : step 1, integer exponents (matches paper's grid)
#   'xfine'   : step 0.5, half-integer exponents (2× denser, catches non-integer optima)
SVQR_GAMMA_EXPS_DEFAULT = list(range(-15, 15, 3))                  # 17 values (int)
SVQR_C_EXPS_DEFAULT     = list(range(-15, 15, 3))                  # 17 values (int)
SVQR_GAMMA_EXPS_FINE    = list(range(-20, 20))                     # 40 values (int step 1)
SVQR_C_EXPS_FINE        = list(range(-20, 20))                     # 40 values (int step 1)
SVQR_GAMMA_EXPS_XFINE   = list(np.arange(-15.0, 15.01, 0.5))       # 81 values (step 0.5)
SVQR_C_EXPS_XFINE       = list(np.arange(-15.0, 15.01, 0.5))       # 81 values (step 0.5)

# Module-level defaults — main() may swap these when --svqr-grid is passed.
SVQR_GAMMA_EXPS = SVQR_GAMMA_EXPS_DEFAULT
SVQR_C_EXPS     = SVQR_C_EXPS_DEFAULT
SVQR_GRID       = list(itertools.product(SVQR_GAMMA_EXPS, SVQR_C_EXPS))

QRF_GRID = list(itertools.product(
    [50, 100, 150, 200,300],          # n_estimators
    [1, 5, 10, 15],          # min_samples_leaf
))  # 3×4 = 12 configs

XGB_SUBSAMPLE = 1.0     # fixed → deterministic XGB (no row subsampling)
XGB_GRID = list(itertools.product(
    [100, 200, 500],            # n_estimators
    [3, 5, 8],                  # max_depth
    [0.01, 0.05, 0.1],          # learning_rate
))  # 3×3×3 = 27 configs

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score(picp: float, mpiw: float) -> float:
    """Primary: coverage >= target. Secondary: smallest MPIW."""
    if picp >= TARGET_COVERAGE:
        return 1000.0 - mpiw          # covered → minimise MPIW
    return (picp - TARGET_COVERAGE) * 1000.0  # heavy penalty for under-coverage


# ---------------------------------------------------------------------------
# Dataset loaders  (same as exp15_cqr.py)
# ---------------------------------------------------------------------------

def load_boston_housing():
    try:
        df = pd.read_excel(DATASETS_PATH + 'bostonhousingdata.xlsx', header=None)
        X, y = df.iloc[:, :-1].values, df.iloc[:, -1].values
        return Bunch(data=X, target=y,
                     feature_names=[f'feat_{i}' for i in range(X.shape[1])])
    except Exception:
        from sklearn.datasets import fetch_california_housing
        data = fetch_california_housing()
        np.random.seed(42)
        idx = np.random.choice(data.data.shape[0], 506, replace=False)
        return Bunch(data=data.data[idx], target=data.target[idx],
                     feature_names=data.feature_names)

def load_energy_efficiency():
    df = pd.read_csv(DATASETS_PATH + 'energy_efficiency.csv')
    return Bunch(data=df.iloc[:, :-2].values, target=df.iloc[:, -2].values,
                 feature_names=df.columns[:-2].tolist())

def load_concrete():
    df = pd.read_csv(DATASETS_PATH + 'Concrete_Data.csv')
    return Bunch(data=df.iloc[:, :-1].values, target=df.iloc[:, -1].values,
                 feature_names=df.columns[:-1].tolist())

def load_yacht():
    df = pd.read_csv(DATASETS_PATH + 'yacht_hydrodynamics.data', sep=r'\s+', header=None)
    return Bunch(data=df.iloc[:, :-1].values, target=df.iloc[:, -1].values,
                 feature_names=[f'X{i}' for i in range(df.shape[1]-1)])

def load_servo():
    df = pd.read_csv(DATASETS_PATH + 'servo.data', header=None)
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    X = np.zeros((len(df), 4))
    for i in range(4):
        X[:, i] = le.fit_transform(df.iloc[:, i].astype(str))
    return Bunch(data=X, target=df.iloc[:, 4].values,
                 feature_names=['motor', 'screw', 'pgain', 'vgain'])

def load_auto_mpg():
    """UCI Auto MPG. n=398 -> 392 after dropping 6 rows with '?' horsepower.
    Target = mpg; 7 continuous+ordinal features."""
    import re
    rows = []
    with open(DATASETS_PATH + 'auto-mpg.data') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # maxsplit=8: keep quoted car name intact at end so we don't over-split.
            parts = re.split(r'\s+', line, maxsplit=8)
            rows.append(parts[:8])
    cols = ['mpg', 'cylinders', 'displacement', 'horsepower',
            'weight', 'acceleration', 'model_year', 'origin']
    df = pd.DataFrame(rows, columns=cols).replace('?', np.nan).apply(pd.to_numeric)
    df = df.dropna()
    X = df.iloc[:, 1:].values.astype(float)
    y = df.iloc[:, 0].values.astype(float)
    return Bunch(data=X, target=y, feature_names=cols[1:])

def load_real_estate():
    """UCI Real Estate Valuation (Sindian, Taiwan). n=414, d=6, all continuous.
    Target = Y house price of unit area."""
    df = pd.read_excel(DATASETS_PATH + 'real_estate_valuation.xlsx')
    # First col 'No' is a row index — drop; last col 'Y ...' is target.
    X = df.iloc[:, 1:-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)
    return Bunch(data=X, target=y, feature_names=df.columns[1:-1].tolist())

DATASETS = [
    ('Boston',     load_boston_housing),
    ('Energy',     load_energy_efficiency),
    ('Concrete',   load_concrete),
    ('Yacht',      load_yacht),
    ('Servo',      load_servo),
    ('AutoMPG',    load_auto_mpg),
    ('RealEstate', load_real_estate),
]

# ---------------------------------------------------------------------------
# QRF  (needed for both tuning and final benchmark)
# ---------------------------------------------------------------------------

class QuantileRandomForest:
    def __init__(self, n_estimators=100, min_samples_leaf=5, random_state=42):
        self.n_estimators     = n_estimators
        self.min_samples_leaf = min_samples_leaf
        self.random_state     = random_state

    def fit(self, X, y):
        self.y_train = y.ravel()
        self.rf = RandomForestRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state, n_jobs=-1)
        self.rf.fit(X, self.y_train)
        self._build_leaf_cache(X)
        return self

    def _build_leaf_cache(self, X):
        self.per_tree_leaf_to_y = []
        for tree in self.rf.estimators_:
            leaf_ids = tree.apply(X)
            leaf_to_y = {}
            for lid in np.unique(leaf_ids):
                leaf_to_y[lid] = self.y_train[leaf_ids == lid]
            self.per_tree_leaf_to_y.append(leaf_to_y)

    def predict_quantiles(self, X, q_lo=0.05, q_hi=0.95):
        n = X.shape[0]
        lo, hi = np.empty(n), np.empty(n)
        all_leaves = np.array([t.apply(X) for t in self.rf.estimators_])
        for i in range(n):
            ys = []
            for t, tl in enumerate(all_leaves):
                yv = self.per_tree_leaf_to_y[t].get(tl[i])
                if yv is not None and len(yv):
                    ys.append(yv)
            if ys:
                y_all = np.concatenate(ys)
                lo[i], hi[i] = np.quantile(y_all, q_lo), np.quantile(y_all, q_hi)
            else:
                lo[i] = hi[i] = self.rf.predict(X[i:i+1])[0]
        return lo, hi


def _qrf_eval(X_tr, y_tr, X_cal, y_cal, X_test, y_test,
              n_estimators, min_samples_leaf, seed=42):
    qrf = QuantileRandomForest(n_estimators=n_estimators,
                               min_samples_leaf=min_samples_leaf, random_state=seed)
    qrf.fit(X_tr, y_tr)
    lo_cal, hi_cal = qrf.predict_quantiles(X_cal, Q_LOW, Q_HIGH)
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n_cal  = len(scores)
    k      = int(np.ceil((1 - ALPHA) * (n_cal + 1)))
    Q      = np.sort(scores)[min(k - 1, n_cal - 1)]
    lo_t, hi_t = qrf.predict_quantiles(X_test, Q_LOW, Q_HIGH)
    lower, upper = lo_t - Q, hi_t + Q
    picp = float(np.mean((y_test >= lower) & (y_test <= upper)))
    mpiw = float(np.mean(upper - lower))
    return picp, mpiw, lower, upper


# ---------------------------------------------------------------------------
# Per-model tuning functions
# ---------------------------------------------------------------------------

def tune_cqr_nn(X_tune, y_tune, dataset_name):
    """Grid search over CQR-NN hyperparameters."""
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tune, y_tune, test_size=0.25, random_state=0)
    # use cal as proxy test for scoring
    X_val, X_cal2, y_val, y_cal2 = train_test_split(
        X_cal, y_cal, test_size=0.5, random_state=0)

    total = len(CQR_NN_GRID)
    best_score = -np.inf
    best_params = dict(hidden_dim=200, epochs=200,
                       learning_rate=1e-2, batch_size=40)  # safe default

    for i, (hd, ep, lr, bs) in enumerate(CQR_NN_GRID, 1):
        try:
            _, picp, mpiw, *_ = train_two_model_cqr_conformal(
                X_tr, y_tr, X_cal2, y_cal2, X_val, y_val,
                q_low=Q_LOW, q_high=Q_HIGH, alpha=ALPHA,
                hidden_dim=hd, epochs=ep, batch_size=bs,
                learning_rate=lr, seed=0, verbose=False)
            sc = _score(picp, mpiw)
        except Exception:
            sc, picp, mpiw = -np.inf, 0.0, np.inf

        status = f"  [{i:>3}/{total}]  hd={hd:<4} ep={ep:<4} lr={lr:.0e} bs={bs:<3} "
        status += f"→ PICP={picp:.4f}  MPIW={mpiw:.4f}"
        if sc > best_score:
            best_score = sc
            best_params = dict(hidden_dim=hd, epochs=ep,
                               learning_rate=lr, batch_size=bs)
            status += "  ★ best"
        print(status)

    print(f"  Best CQR-NN params: {best_params}  (score={best_score:.2f})")
    return best_params


def tune_svqr_cp(X_tune, y_tune, dataset_name):
    """Grid search over SVQR-QP hyperparameters (gamma, C).

    Paper convention: gamma = 2^gamma_exp;  C = 2^C_exp.
    Kernel: exp(-gamma * ||x-y||^2).
    """
    from sklearn.preprocessing import StandardScaler

    # split: train_proper | conformal_cal | score_val
    X_tr, X_rest, y_tr, y_rest = train_test_split(X_tune, y_tune, test_size=0.4, random_state=0)
    X_cal, X_val, y_cal, y_val = train_test_split(X_rest, y_rest, test_size=0.5, random_state=0)

    # scale once on X_tr, y_tr
    sc_X = StandardScaler().fit(X_tr)
    sc_y = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_s  = sc_X.transform(X_tr)
    X_cal_s = sc_X.transform(X_cal)
    X_val_s = sc_X.transform(X_val)
    y_tr_s  = sc_y.transform(y_tr.reshape(-1, 1)).ravel()

    total = len(SVQR_GRID)
    best_score = -np.inf
    best_params = dict(gamma_exp=0, C_exp=4,
                       gamma=1.0, C=16.0)  # safe default

    for i, (ge, ce) in enumerate(SVQR_GRID, 1):
        gamma = 2.0 ** ge          # paper convention (no /n_features)
        C     = 2.0 ** ce
        try:
            beta_lo, _ = solve_qp(X_tr_s, y_tr_s, gamma, C, Q_LOW)
            beta_hi, _ = solve_qp(X_tr_s, y_tr_s, gamma, C, Q_HIGH)

            # conformal on X_cal
            lo_cal_s = predict_svqr(beta_lo, X_tr_s, X_cal_s, gamma)
            hi_cal_s = predict_svqr(beta_hi, X_tr_s, X_cal_s, gamma)
            lo_cal   = sc_y.inverse_transform(lo_cal_s.reshape(-1, 1)).ravel()
            hi_cal   = sc_y.inverse_transform(hi_cal_s.reshape(-1, 1)).ravel()

            scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
            n_cal  = len(scores)
            k      = int(np.ceil((1 - ALPHA) * (n_cal + 1)))
            q_hat  = float(np.sort(scores)[min(k - 1, n_cal - 1)])

            # evaluate on X_val
            lo_v_s = predict_svqr(beta_lo, X_tr_s, X_val_s, gamma)
            hi_v_s = predict_svqr(beta_hi, X_tr_s, X_val_s, gamma)
            lo_v   = sc_y.inverse_transform(lo_v_s.reshape(-1, 1)).ravel() - q_hat
            hi_v   = sc_y.inverse_transform(hi_v_s.reshape(-1, 1)).ravel() + q_hat

            picp = float(np.mean((y_val >= lo_v) & (y_val <= hi_v)))
            mpiw = float(np.mean(hi_v - lo_v))
            sc   = _score(picp, mpiw)
        except Exception as e:
            sc, picp, mpiw = -np.inf, 0.0, np.inf

        status = (f"  [{i:>4}/{total}]  gamma_exp={ge:+.1f}  C_exp={ce:+.1f}  "
                  f"(gamma={gamma:.5f}, C={C:.2f})  → PICP={picp:.4f}  MPIW={mpiw:.4f}")
        if sc > best_score:
            best_score  = sc
            best_params = dict(gamma_exp=ge, C_exp=ce, gamma=gamma, C=C)
            status += "  ★ best"
        print(status)

    print(f"  Best SVQR+CP params: gamma_exp={best_params['gamma_exp']}  "
          f"C_exp={best_params['C_exp']}  "
          f"(gamma={best_params['gamma']:.4f}, C={best_params['C']:.0f})  "
          f"score={best_score:.2f}")
    return best_params


def tune_qrf_cp(X_tune, y_tune, dataset_name):
    """Grid search over QRF+CP hyperparameters."""
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tune, y_tune, test_size=0.25, random_state=0)
    X_val, X_cal2, y_val, y_cal2 = train_test_split(
        X_cal, y_cal, test_size=0.5, random_state=0)

    total = len(QRF_GRID)
    best_score = -np.inf
    best_params = dict(n_estimators=100, min_samples_leaf=5)  # safe default

    for i, (n_est, msl) in enumerate(QRF_GRID, 1):
        try:
            picp, mpiw, _, _ = _qrf_eval(X_tr, y_tr, X_val, y_val, X_cal2, y_cal2,
                                          n_est, msl)
            sc = _score(picp, mpiw)
        except Exception:
            sc, picp, mpiw = -np.inf, 0.0, np.inf

        status = (f"  [{i:>3}/{total}]  n_est={n_est:<4}  msl={msl:<3} "
                  f"→ PICP={picp:.4f}  MPIW={mpiw:.4f}")
        if sc > best_score:
            best_score  = sc
            best_params = dict(n_estimators=n_est, min_samples_leaf=msl)
            status += "  ★ best"
        print(status)

    print(f"  Best QRF+CP params: {best_params}  (score={best_score:.2f})")
    return best_params


def tune_xgb_cp(X_tune, y_tune, dataset_name):
    """Grid search over XGB+CP hyperparameters (n_est, max_depth, lr, subsample)."""
    from xgboost import XGBRegressor

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tune, y_tune, test_size=0.25, random_state=0)
    X_val, X_cal2, y_val, y_cal2 = train_test_split(
        X_cal, y_cal, test_size=0.5, random_state=0)

    total = len(XGB_GRID)
    best_score = -np.inf
    best_params = dict(n_estimators=200, max_depth=3,
                       learning_rate=0.05, subsample=XGB_SUBSAMPLE)  # safe default

    for i, (ne, md, lr) in enumerate(XGB_GRID, 1):
        try:
            shared = dict(objective="reg:quantileerror",
                          n_estimators=ne, max_depth=md,
                          learning_rate=lr, subsample=XGB_SUBSAMPLE,
                          verbosity=0, n_jobs=-1)
            xgb_lo = XGBRegressor(quantile_alpha=Q_LOW,  random_state=0,     **shared)
            xgb_hi = XGBRegressor(quantile_alpha=Q_HIGH, random_state=10000, **shared)
            xgb_lo.fit(X_tr, y_tr)
            xgb_hi.fit(X_tr, y_tr)

            lo_cal = xgb_lo.predict(X_cal2)
            hi_cal = xgb_hi.predict(X_cal2)
            scores = np.maximum(lo_cal - y_cal2, y_cal2 - hi_cal)
            n_cal  = len(scores)
            k      = int(np.ceil((1 - ALPHA) * (n_cal + 1)))
            q_hat  = float(np.sort(scores)[min(k - 1, n_cal - 1)])

            lo_v = xgb_lo.predict(X_val) - q_hat
            hi_v = xgb_hi.predict(X_val) + q_hat
            picp = float(np.mean((y_val >= lo_v) & (y_val <= hi_v)))
            mpiw = float(np.mean(hi_v - lo_v))
            sc   = _score(picp, mpiw)
        except Exception:
            sc, picp, mpiw = -np.inf, 0.0, np.inf

        status = (f"  [{i:>3}/{total}]  n_est={ne:<4} md={md:<2} lr={lr:.2f} "
                  f"→ PICP={picp:.4f}  MPIW={mpiw:.4f}")
        if sc > best_score:
            best_score  = sc
            best_params = dict(n_estimators=ne, max_depth=md,
                               learning_rate=lr, subsample=XGB_SUBSAMPLE)
            status += "  ★ best"
        print(status)

    print(f"  Best XGB+CP params: {best_params}  (score={best_score:.2f})")
    return best_params


# ---------------------------------------------------------------------------
# Uncertainty decomposition  (exp9 formula)
# ---------------------------------------------------------------------------

def decompose_raw(lo_arr, hi_arr):
    """v_opt = E_d[Var_i[f]];  v_in = Var_d[E_i[f]];  summed lo + hi.

    lo_arr, hi_arr have shape (D, I, N_test).  When I<2 → v_opt=0.
    """
    mu_lo_d = lo_arr.mean(axis=1)
    mu_hi_d = hi_arr.mean(axis=1)

    I = lo_arr.shape[1]
    if I < 2:
        v_opt = 0.0
    else:
        v_opt = (lo_arr.var(axis=1, ddof=1).mean()
                 + hi_arr.var(axis=1, ddof=1).mean())

    D = lo_arr.shape[0]
    if D < 2:
        v_in = 0.0
    else:
        v_in = (mu_lo_d.var(axis=0, ddof=1).mean()
                + mu_hi_d.var(axis=0, ddof=1).mean())
    return float(v_opt), float(v_in)


# ---------------------------------------------------------------------------
# Final benchmark with tuned params
# ---------------------------------------------------------------------------

def _svqr_run(X_train, y_train, X_test, y_test, params, data_seed=42, init_seed=0):
    """Final benchmark run for SVQR+CP. Deterministic — σ_opt = 0 across init seeds."""
    from sklearn.preprocessing import StandardScaler

    gamma = params['gamma']
    C     = params['C']

    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=data_seed)

    sc_X = StandardScaler().fit(X_tr)
    sc_y = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_s  = sc_X.transform(X_tr)
    X_cal_s = sc_X.transform(X_cal)
    X_te_s  = sc_X.transform(X_test)
    y_tr_s  = sc_y.transform(y_tr.reshape(-1, 1)).ravel()

    beta_lo, _ = solve_qp(X_tr_s, y_tr_s, gamma, C, Q_LOW)
    beta_hi, _ = solve_qp(X_tr_s, y_tr_s, gamma, C, Q_HIGH)

    lo_cal_s = predict_svqr(beta_lo, X_tr_s, X_cal_s, gamma)
    hi_cal_s = predict_svqr(beta_hi, X_tr_s, X_cal_s, gamma)
    lo_cal   = sc_y.inverse_transform(lo_cal_s.reshape(-1, 1)).ravel()
    hi_cal   = sc_y.inverse_transform(hi_cal_s.reshape(-1, 1)).ravel()

    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n_cal  = len(scores)
    k      = int(np.ceil((1 - ALPHA) * (n_cal + 1)))
    q_hat  = float(np.sort(scores)[min(k - 1, n_cal - 1)])

    lo_te_s = predict_svqr(beta_lo, X_tr_s, X_te_s, gamma)
    hi_te_s = predict_svqr(beta_hi, X_tr_s, X_te_s, gamma)
    lo_te   = sc_y.inverse_transform(lo_te_s.reshape(-1, 1)).ravel()
    hi_te   = sc_y.inverse_transform(hi_te_s.reshape(-1, 1)).ravel()

    lower, upper = lo_te - q_hat, hi_te + q_hat
    return (float(np.mean((y_test >= lower) & (y_test <= upper))),
            float(np.mean(upper - lower)), lower, upper)


def _cqr_run(X_train, y_train, X_test, y_test, params, data_seed=42, init_seed=42):
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=data_seed)
    _, cov, mpiw, *_, lo, hi = train_two_model_cqr_conformal(
        X_tr, y_tr, X_cal, y_cal, X_test, y_test,
        q_low=Q_LOW, q_high=Q_HIGH, alpha=ALPHA,
        hidden_dim=params['hidden_dim'], epochs=params['epochs'],
        batch_size=params['batch_size'], learning_rate=params['learning_rate'],
        seed=init_seed, verbose=False)
    return float(cov), float(mpiw), lo, hi


def _qrf_run(X_train, y_train, X_test, y_test, params, data_seed=42, init_seed=42):
    if params is None:
        params = dict(n_estimators=100, min_samples_leaf=5)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=data_seed)
    return _qrf_eval(X_tr, y_tr, X_cal, y_cal, X_test, y_test,
                     params['n_estimators'], params['min_samples_leaf'], seed=init_seed)


def _xgb_run(X_train, y_train, X_test, y_test, params, data_seed=42, init_seed=42):
    """XGB quantile regression + conformal calibration."""
    from xgboost import XGBRegressor
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=data_seed)

    shared = dict(objective="reg:quantileerror",
                  n_estimators=params['n_estimators'],
                  max_depth=params['max_depth'],
                  learning_rate=params['learning_rate'],
                  subsample=XGB_SUBSAMPLE,   # fixed → deterministic
                  verbosity=0, n_jobs=-1)
    xgb_lo = XGBRegressor(quantile_alpha=Q_LOW,  random_state=init_seed,          **shared)
    xgb_hi = XGBRegressor(quantile_alpha=Q_HIGH, random_state=init_seed + 10000,  **shared)
    xgb_lo.fit(X_tr, y_tr)
    xgb_hi.fit(X_tr, y_tr)

    lo_cal = xgb_lo.predict(X_cal)
    hi_cal = xgb_hi.predict(X_cal)
    scores = np.maximum(lo_cal - y_cal, y_cal - hi_cal)
    n_cal  = len(scores)
    k      = int(np.ceil((1 - ALPHA) * (n_cal + 1)))
    q_hat  = float(np.sort(scores)[min(k - 1, n_cal - 1)])

    lower = xgb_lo.predict(X_test) - q_hat
    upper = xgb_hi.predict(X_test) + q_hat
    picp  = float(np.mean((y_test >= lower) & (y_test <= upper)))
    mpiw  = float(np.mean(upper - lower))
    return picp, mpiw, lower, upper


METHODS = ['CQR-NN', 'SVQR+CP', 'QRF+CP', 'XGB+CP']
DETERMINISTIC_METHODS = {'SVQR+CP', 'XGB+CP'}   # σ_opt = 0 → I=1


def _run_one(method_name, X_tr, y_tr, X_te, y_te, params, data_seed, init_seed):
    if method_name == 'CQR-NN':
        return _cqr_run(X_tr, y_tr, X_te, y_te, params, data_seed=data_seed, init_seed=init_seed)
    if method_name == 'SVQR+CP':
        return _svqr_run(X_tr, y_tr, X_te, y_te, params, data_seed=data_seed, init_seed=init_seed)
    if method_name == 'QRF+CP':
        return _qrf_run(X_tr, y_tr, X_te, y_te, params, data_seed=data_seed, init_seed=init_seed)
    if method_name == 'XGB+CP':
        return _xgb_run(X_tr, y_tr, X_te, y_te, params, data_seed=data_seed, init_seed=init_seed)
    raise ValueError(f"unknown method: {method_name}")


def run_final_benchmark(all_best_params, n_data=10, n_init=10, methods=None):
    """Nested D data seeds x I init seeds benchmark.

    Outer (train_val, test) split fixed at random_state=42.
    Per data seed d ∈ [0..D): re-split (train_proper, cal) with random_state=d.
    Per init seed i ∈ [0..I): re-init the model with that seed.
    SVQR is deterministic → I=1 for it (σ_opt=0), σ_in still varies with d.
    """
    results = []

    for dataset_name, load_fn in DATASETS:
        print(f"\n{'='*60}")
        print(f"FINAL BENCHMARK — {dataset_name}  (D={n_data} x I={n_init})")
        print('='*60)

        try:
            data = load_fn()
            X, y = data.data, data.target
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        ds_params = all_best_params.get(dataset_name, {})

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        N_test = len(y_test)

        methods_iter = methods if methods is not None else METHODS
        for method_name in methods_iter:
            params = ds_params.get(method_name)
            if params is None:
                print(f"\n  {method_name}  SKIPPED — no tuned params")
                continue

            I = 1 if method_name in DETERMINISTIC_METHODS else n_init
            print(f"\n  {method_name}  params={params}  (D={n_data} x I={I})")

            preds_lo = np.zeros((n_data, I, N_test), dtype=np.float32)
            preds_hi = np.zeros((n_data, I, N_test), dtype=np.float32)
            time_list       = []
            per_d_picp      = []
            per_d_mpiw      = []
            any_failed      = False

            for d in range(n_data):
                for i in range(I):
                    t0 = time.time()
                    try:
                        picp, mpiw, lo, hi = _run_one(
                            method_name, X_train, y_train, X_test, y_test,
                            params, data_seed=d, init_seed=i)
                        elapsed = time.time() - t0
                        preds_lo[d, i] = lo.astype(np.float32)
                        preds_hi[d, i] = hi.astype(np.float32)
                        time_list.append(elapsed)
                        print(f"    d={d+1:2d}/i={i+1:2d}  "
                              f"PICP={picp*100:.2f}%  MPIW={mpiw:.4f}  t={elapsed:.1f}s")
                    except Exception as e:
                        any_failed = True
                        print(f"    d={d+1:2d}/i={i+1:2d}  ERROR — {e}")
                        import traceback; traceback.print_exc()

                # per-d init-averaged interval
                mean_lo_d = preds_lo[d].mean(axis=0)
                mean_hi_d = preds_hi[d].mean(axis=0)
                pd_picp = float(np.mean((y_test >= mean_lo_d) & (y_test <= mean_hi_d)))
                pd_mpiw = float(np.mean(mean_hi_d - mean_lo_d))
                per_d_picp.append(pd_picp)
                per_d_mpiw.append(pd_mpiw)
                print(f"    [d={d+1:2d} init-mean]  PICP={pd_picp*100:.2f}%  MPIW={pd_mpiw:.4f}")

            v_opt, v_in = decompose_raw(preds_lo, preds_hi)
            sigma_opt   = float(np.sqrt(v_opt))
            sigma_in    = float(np.sqrt(v_in))

            results.append({
                'Dataset':     dataset_name,
                'Method':      method_name,
                'PICP (%)':    float(np.mean(per_d_picp) * 100),
                'MPIW':        float(np.mean(per_d_mpiw)),
                'σ_opt':       sigma_opt,
                'σ_in':        sigma_in,
                'Time (sec.)': float(np.mean(time_list)) if time_list else float('nan'),
                'Std. PICP':   float(np.std(per_d_picp) * 100),
                'Std. MPIW':   float(np.std(per_d_mpiw)),
                'Best Params': str(params),
            })
            print(f"    → σ_opt={sigma_opt:.4f}  σ_in={sigma_in:.4f}  "
                  f"PICP={np.mean(per_d_picp)*100:.2f}%  MPIW={np.mean(per_d_mpiw):.4f}")

    return results


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _print_results(results):
    print("\n" + "="*140)
    print("TUNED BENCHMARK RESULTS  (D x I nested)")
    print("="*140)
    print(f"{'Dataset':<12} {'Method':<10} {'PICP(%)':>8} {'MPIW':>10} "
          f"{'σ_opt':>10} {'σ_in':>10} {'Time(s)':>8} {'±PICP':>7}  Best Params")
    print("-"*140)
    cur = None
    for r in results:
        ds  = r['Dataset'] if r['Dataset'] != cur else ""
        cur = r['Dataset']
        print(f"{ds:<12} {r['Method']:<10} {r['PICP (%)']:>8.2f} {r['MPIW']:>10.4f} "
              f"{r['σ_opt']:>10.4f} {r['σ_in']:>10.4f} {r['Time (sec.)']:>8.2f} "
              f"{r['Std. PICP']:>7.2f}  {r['Best Params']}")
    print("="*140)


def _latex_table(results):
    print("\n% --- LaTeX table (tuned results) ---")
    print("\\begin{table}[h]\\centering")
    print("\\caption{Tuned Conformal Prediction Benchmark (D=10 data seeds x I=10 init seeds)}")
    print("\\begin{tabular}{llcccccc}\\hline")
    print("Dataset & Method & PICP (\\%) & MPIW & $\\sigma_{\\mathrm{opt}}$ "
          "& $\\sigma_{\\mathrm{in}}$ & Time (s) & $\\sigma_{PICP}$ \\\\\\hline")
    cur = None
    for r in results:
        ds  = r['Dataset'] if r['Dataset'] != cur else ""
        cur = r['Dataset']
        print(f"{ds} & {r['Method']} & {r['PICP (%)']:.2f} & {r['MPIW']:.4f} & "
              f"{r['σ_opt']:.4f} & {r['σ_in']:.4f} & {r['Time (sec.)']:.2f} & "
              f"{r['Std. PICP']:.2f} \\\\")
    print("\\hline\\end{tabular}\\end{table}")


def _save(results, ts):
    df = pd.DataFrame(results)
    csv_p = RESULT_DIR / f"tuned_{ts}.csv"
    txt_p = RESULT_DIR / f"tuned_{ts}.txt"
    df.to_csv(csv_p, index=False)
    with open(txt_p, 'w', encoding='utf-8') as f:
        f.write(f"Exp15 tuned benchmark — {ts}\n\n")
        f.write(df.to_string(index=False))
        f.write("\n")
    print(f"\nSaved CSV  : {csv_p}")
    print(f"Saved text : {txt_p}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Exp15 hyperparameter tuning + benchmark")
    parser.add_argument("--n-data", type=int, default=10,
                        help="D — number of data-split seeds in benchmark (default 10)")
    parser.add_argument("--n-init", type=int, default=10,
                        help="I — number of model-init seeds in benchmark (default 10)")
    parser.add_argument("--force-retune", action="store_true",
                        help="Re-run ALL tuning even if best_params JSON already exists")
    parser.add_argument("--datasets", nargs="+",
                        default=[d for d, _ in DATASETS],
                        help="Which datasets to process (default: all 5)")
    parser.add_argument("--methods", nargs="+",
                        default=['CQR-NN', 'SVQR+CP', 'QRF+CP', 'XGB+CP'],
                        choices=['CQR-NN', 'SVQR+CP', 'QRF+CP', 'XGB+CP'],
                        help="Which methods to tune AND benchmark (default: all 4)")
    parser.add_argument("--skip-tune", action="store_true",
                        help="Skip tuning entirely (benchmark-only mode using existing JSONs)")
    parser.add_argument("--svqr-grid", choices=['default', 'fine', 'xfine'], default='default',
                        help="'default' = int step-3 grid (289 configs); "
                             "'fine' = int step-1 grid (~1600 configs, paper); "
                             "'xfine' = half-step (6561 configs, catches non-integer optima)")
    args = parser.parse_args()

    # Apply --svqr-grid selection before any tuning
    global SVQR_GAMMA_EXPS, SVQR_C_EXPS, SVQR_GRID
    if args.svqr_grid == 'fine':
        SVQR_GAMMA_EXPS = SVQR_GAMMA_EXPS_FINE
        SVQR_C_EXPS     = SVQR_C_EXPS_FINE
    elif args.svqr_grid == 'xfine':
        SVQR_GAMMA_EXPS = SVQR_GAMMA_EXPS_XFINE
        SVQR_C_EXPS     = SVQR_C_EXPS_XFINE
    if args.svqr_grid != 'default':
        SVQR_GRID = list(itertools.product(SVQR_GAMMA_EXPS, SVQR_C_EXPS))

    tuner_fns = {
        'CQR-NN':  tune_cqr_nn,
        'SVQR+CP': tune_svqr_cp,
        'QRF+CP':  tune_qrf_cp,
        'XGB+CP':  tune_xgb_cp,
    }
    grid_sizes = {
        'CQR-NN':  len(CQR_NN_GRID),
        'SVQR+CP': len(SVQR_GRID),
        'QRF+CP':  len(QRF_GRID),
        'XGB+CP':  len(XGB_GRID),
    }

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    print(f"\n{'='*60}")
    print(f"Exp 15 — Hyperparameter Tuning + Benchmark  [{ts}]")
    print(f"  Target coverage : {TARGET_COVERAGE:.0%}")
    print(f"  Benchmark       : D={args.n_data} data seeds x I={args.n_init} init seeds")
    for m in METHODS:
        print(f"  {m:<8} configs : {grid_sizes[m]}")
    print('='*60)

    all_best_params = {}

    for dataset_name, load_fn in DATASETS:
        if dataset_name not in args.datasets:
            continue

        json_path = HPARAM_DIR / f"best_params_{dataset_name}.json"

        # Always load the existing JSON if present — never wipe other methods' entries.
        ds_best = {}
        if json_path.exists():
            with open(json_path) as f:
                ds_best = json.load(f)

        # Decide which of the requested methods actually need tuning right now.
        if args.force_retune:
            methods_to_tune = list(args.methods)   # redo all requested methods
        else:
            methods_to_tune = [m for m in args.methods if m not in ds_best]

        if not methods_to_tune:
            all_best_params[dataset_name] = ds_best
            print(f"\n[{dataset_name}] Loaded existing params from {json_path}  "
                  f"(requested methods {args.methods} already present; "
                  f"use --force-retune to redo)")
            continue

        if ds_best:
            print(f"\n[{dataset_name}] Have {list(ds_best.keys())}; "
                  f"tuning (from --methods): {methods_to_tune}"
                  + ("   [--force-retune]" if args.force_retune else ""))

        if args.skip_tune:
            all_best_params[dataset_name] = ds_best
            print(f"\n[{dataset_name}] --skip-tune set; will benchmark existing JSON only "
                  f"(would have tuned: {methods_to_tune})")
            continue

        try:
            data = load_fn()
            X, y = data.data, data.target
        except Exception as e:
            print(f"  Error loading {dataset_name}: {e}")
            continue

        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X_tv, X_test, y_tv, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        print(f"\n{'='*60}")
        print(f"TUNING — {dataset_name}   ({methods_to_tune})")
        print('='*60)

        for m in methods_to_tune:
            print(f"\n  --- {m} ({grid_sizes[m]} configs) ---")
            t0 = time.time()
            ds_best[m] = tuner_fns[m](X_tv, y_tv, dataset_name)
            print(f"  {m} tuning done in {time.time()-t0:.1f}s")

        all_best_params[dataset_name] = ds_best

        with open(json_path, 'w') as f:
            json.dump(ds_best, f, indent=2)
        print(f"\n  Saved best params → {json_path}")

    # --- Summary of best params ---
    print(f"\n{'='*60}")
    print("BEST PARAMETERS SUMMARY")
    print('='*60)
    for ds, params in all_best_params.items():
        print(f"\n  {ds}:")
        for m, p in params.items():
            print(f"    {m:<10}: {p}")

    # --- Final benchmark ---
    print(f"\n{'='*60}")
    print(f"RUNNING FINAL BENCHMARK  (D={args.n_data} x I={args.n_init})")
    print('='*60)
    results = run_final_benchmark(all_best_params,
                                  n_data=args.n_data, n_init=args.n_init,
                                  methods=args.methods)

    _print_results(results)
    _latex_table(results)
    _save(results, ts)


if __name__ == "__main__":
    main()
