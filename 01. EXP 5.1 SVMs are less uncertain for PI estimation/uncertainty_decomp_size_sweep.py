"""
Uncertainty Decomposition — Multi-Model Training-Size Sweep
=============================================================
Nested 10 (data seeds) x 10 (init seeds) experiment for five models —
ANN, QRF, SVM (SVQR), XGB, GPR — across training sizes [50, 100, 200, 300, 400, 800].

Model backends and hyperparameters are copied from
  src/runnable_experiments/(NN)vs(QRF)vs(SVM.py
and stay FIXED across all D x I x N runs. Across the 10 init seeds we only
vary what each model's own randomness controls (NN weight init, QRF internal bagging,
XGB stochastic boosting). SVM (SVQR) and GPR are deterministic -> I=1, sigma_opt=0.

Decomposition (per test point, summed lo + hi)
-----------------------------------------------
  sigma2_opt   = E_d [ Var_i [f] ]          init / optimization noise
  sigma2_in    = Var_d [ E_i [f] ]          data-subsample noise
  sigma2_model = sigma2_opt + sigma2_in     total predictive variance
  sigma_opt    = sqrt(sigma2_opt)
  sigma_in     = sqrt(sigma2_in)
  sigma_model  = sqrt(sigma2_model)

Output
------
uncertainty_decomp_results/
  sweep_<MODEL>.csv             one row per N (per model) — simple columns only
  sweep_<MODEL>_runs.csv        one row per (N, data_seed, init_seed)
  sweep_<MODEL>_legend.txt      column legend
  sweep_<MODEL>.png             sigma_opt / sigma_in vs N per model
  sweep_comparison.png          all models, 2-panel comparison
  sweep_comparison.csv          combined summary across models
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from xgboost import XGBRegressor
from scipy.optimize import minimize as sp_min
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")  # non-GUI backend; avoids Tk thread crashes with n_jobs=-1
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODELS          = ["ANN", "QRF", "SVM", "XGB", "GPR", "NGB"]

SIZES           = [50, 100, 200, 300, 400, 800]
N_TEST          = 600
TEST_SEED       = 9999
X_LOW, X_HIGH   = -6.0, 6.0
NOISE_LOW, NOISE_HIGH = -1.0, 1.0

Q_LOWER         = 0.05
Q_UPPER         = 0.95
N_DATA_SEEDS    = 10
N_INIT_SEEDS    = 10

# Hyperparameters selected by val-set sweep
#   (exp2_stability_nn_qrf_svm/results/hyperparam_tuning/best_hyperparams_2026_06_19_16_15_22.json,
#    n_train=50, n_val=50, M=10, seed=2; metric = sum_rmse on val X points).

# ANN  (best: hidden=128, lr=3e-3)
# Hyperparameter values below come from the exp8 stage-1 tuning best
# (exp8_ablation_model_size/results/hyperparam_tuning/best_hyperparams_2026_06_22_11_58_38.json).
ANN_HIDDEN      = 512
ANN_EPOCHS      = 100
ANN_BATCH       = 32
ANN_LR          = 3e-3

# SVM / SVQR (best: gamma=1.0 -> s=0.5, C=10)
SVM_GAMMA       = 0.7071067811865476   # 2^-0.5
SVM_C           = 11.313708498984761   # 2^3.5

# QRF (best: n_estimators=30, min_samples_leaf=5, max_features='sqrt')
QRF_N_ESTIMATORS     = 50
QRF_MIN_SAMPLES_LEAF = 5
QRF_MAX_FEATURES     = "sqrt"

# XGBoost (best: max_depth=2, n_estimators=200, lr=0.05, subsample=0.8)
XGB_N_ESTIMATORS     = 200
XGB_MAX_DEPTH        = 2
XGB_LR               = 0.05
XGB_SUBSAMPLE        = 0.8
XGB_COLSAMPLE        = 0.8
XGB_MIN_CHILD_WEIGHT = 3
XGB_REG_LAMBDA       = 2.0

# GPR (best: length_scale=0.707, alpha=0.5)
GPR_LENGTH_SCALE = 0.707
GPR_ALPHA        = 0.5

# NGB (best from exp8: n_estimators=200, lr=0.01, max_depth=2, min_samples_leaf=10)
# minibatch_frac=1.0 -> fully deterministic given fixed data -> I=1, sigma_opt=0
NGB_N_ESTIMATORS    = 300
NGB_LEARNING_RATE   = 0.01
NGB_MAX_DEPTH       = 2
NGB_MIN_SAMPLES_LEAF = 10

# Per-model init seeds (SVM, GPR, NGB are deterministic -> I=1)
N_INIT_SEEDS_PER_MODEL = {"ANN": 10, "QRF": 10, "SVM": 1, "XGB": 10, "GPR": 1, "NGB": 1}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)


# ─── DATA ─────────────────────────────────────────────────────────────────────
def generate_data(seed: int, n: int):
    """Draw n samples of y = sin(x)/x + Uniform(-1,1), x ~ Uniform(-1,1)."""
    rng = np.random.default_rng(seed)
    x   = rng.uniform(X_LOW, X_HIGH, size=n).astype(np.float32)
    eps = rng.uniform(NOISE_LOW, NOISE_HIGH, size=n).astype(np.float32)
    y   = (np.sinc(x / np.pi) + eps).astype(np.float32)
    return x.reshape(-1, 1), y


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL BACKENDS
#  Signature: train_predict_X(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed) -> (lo, hi, dt)
#  Inputs are already standardised externally; outputs are inverse-transformed.
# ═══════════════════════════════════════════════════════════════════════════════

# ─── ANN: single MLP with two pinball heads (lo, hi) ──────────────────────────
class _MLP(nn.Module):
    def __init__(self, in_dim=1, hidden=ANN_HIDDEN):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, x):
        return self.net(x)


def _pinball(pred, target, q):
    diff = target - pred
    return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))


def train_predict_ann(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    """Mirrors _train_nn_fixed_data: same data, different torch seed per run."""
    t0 = time.time()
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    model = _MLP(in_dim=X_tr_s.shape[1]).to(DEVICE)

    X_t = torch.tensor(X_tr_s, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_tr_s.reshape(-1, 1), dtype=torch.float32).to(DEVICE)
    dl  = DataLoader(TensorDataset(X_t, y_t), batch_size=ANN_BATCH, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=ANN_LR, weight_decay=0.0)

    model.train()
    for _ in range(ANN_EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            out  = model(xb)
            loss = (_pinball(out[:, 0], yb.squeeze(1), Q_LOWER) +
                    _pinball(out[:, 1], yb.squeeze(1), Q_UPPER))
            loss.backward()
            opt.step()

    model.eval()
    Xe = torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(Xe).cpu().numpy()
    lo_s = np.minimum(out[:, 0], out[:, 1])
    hi_s = np.maximum(out[:, 0], out[:, 1])
    dt = time.time() - t0

    lo = scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
    return lo, hi, dt


# ─── QRF: RF + leaf-distribution quantile prediction (no outer bootstrap) ─────
def _qrf_leaf_predict(rf, X_tr_s, y_tr_s, X_te_s, quantile):
    train_leaves = rf.apply(X_tr_s)
    test_leaves  = rf.apply(X_te_s)
    n_test, n_trees = test_leaves.shape
    preds = np.empty(n_test, dtype=np.float64)
    for i in range(n_test):
        ys = []
        for t in range(n_trees):
            mask = train_leaves[:, t] == test_leaves[i, t]
            if mask.any():
                ys.append(y_tr_s[mask])
        if ys:
            preds[i] = np.quantile(np.concatenate(ys), quantile)
        else:
            preds[i] = float(rf.predict(X_te_s[i:i+1])[0])
    return preds


def train_predict_qrf(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    """QRF on fixed training data; variance across init_seeds comes from
    RandomForest's internal bagging / feature randomness only (no outer
    bootstrap), matching the NN/XGB fixed-data design."""
    t0  = time.time()

    rf = RandomForestRegressor(
        n_estimators=QRF_N_ESTIMATORS,
        min_samples_leaf=QRF_MIN_SAMPLES_LEAF,
        max_features=QRF_MAX_FEATURES,
        random_state=init_seed, n_jobs=-1)
    rf.fit(X_tr_s, y_tr_s)

    lo_s = _qrf_leaf_predict(rf, X_tr_s, y_tr_s, X_te_s, Q_LOWER)
    hi_s = _qrf_leaf_predict(rf, X_tr_s, y_tr_s, X_te_s, Q_UPPER)
    dt = time.time() - t0

    lo = scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
    return lo, hi, dt


# ─── SVM: SVQR QP port of epsilon_quantilesvr2.m (RBF, eps=0, bias=0) ─────────
def _rbf_kernel_mat(X1, X2, s):
    """exp(-||x-y||^2 / (2*s)). s = 1/(2*gamma) — matches kernelfun.m."""
    X1sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2sq = np.sum(X2 ** 2, axis=1, keepdims=True)
    D2 = np.maximum(X1sq + X2sq.T - 2.0 * (X1 @ X2.T), 0.0)
    return np.exp(-D2 / (2.0 * s))


def _solve_svqr_qp(X_tr_s, y_tr_s, s, C, tau):
    n  = X_tr_s.shape[0]
    H  = _rbf_kernel_mat(X_tr_s, X_tr_s, s)
    Hb = np.block([[H, -H], [-H, H]]) + 1e-10 * np.eye(2 * n)
    c  = np.concatenate([-y_tr_s, y_tr_s])
    bounds = [(0.0, tau * C)] * n + [(0.0, (1.0 - tau) * C)] * n
    v0 = np.zeros(2 * n)
    res = sp_min(
        fun=lambda v: 0.5 * float(v @ Hb @ v) + float(c @ v),
        jac=lambda v: Hb @ v + c,
        x0=v0, method='SLSQP', bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-10},
    )
    return res.x[:n] - res.x[n:]


def train_predict_svm(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    """SVQR QP — deterministic, init_seed unused."""
    t0 = time.time()
    s = 1.0 / (2.0 * SVM_GAMMA)
    beta_lo = _solve_svqr_qp(X_tr_s, y_tr_s, s, SVM_C, Q_LOWER)
    beta_hi = _solve_svqr_qp(X_tr_s, y_tr_s, s, SVM_C, Q_UPPER)

    K_te = _rbf_kernel_mat(X_te_s, X_tr_s, s)
    lo_s = K_te @ beta_lo
    hi_s = K_te @ beta_hi
    dt = time.time() - t0

    lo = scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
    return np.minimum(lo, hi), np.maximum(lo, hi), dt


# ─── XGBoost: quantile regression, fixed-data + stochastic boosting noise ────
def train_predict_xgb(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    """Mirrors _train_xgb_fixed_data: same data, different random_state per run."""
    t0 = time.time()
    shared = dict(
        objective="reg:quantileerror",
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LR, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT, reg_lambda=XGB_REG_LAMBDA,
        verbosity=0, n_jobs=-1,
    )
    xgb_lo = XGBRegressor(quantile_alpha=Q_LOWER,
                          random_state=init_seed, **shared)
    xgb_hi = XGBRegressor(quantile_alpha=Q_UPPER,
                          random_state=init_seed + 10000, **shared)
    xgb_lo.fit(X_tr_s, y_tr_s)
    xgb_hi.fit(X_tr_s, y_tr_s)
    dt = time.time() - t0

    lo = scaler_y.inverse_transform(xgb_lo.predict(X_te_s).reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(xgb_hi.predict(X_te_s).reshape(-1, 1)).flatten()
    return np.minimum(lo, hi), np.maximum(lo, hi), dt


# ─── GPR: fixed RBF kernel, alpha as noise floor, deterministic ───────────────
def train_predict_gpr(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    """Mirrors _train_gpr_fixed_data. PI = mu ± z*sqrt(std_f^2 + alpha) in scaled-y units."""
    t0 = time.time()
    kernel = RBF(length_scale=GPR_LENGTH_SCALE)
    gpr = GaussianProcessRegressor(
        kernel=kernel, optimizer=None,
        alpha=GPR_ALPHA, normalize_y=False, random_state=init_seed)
    gpr.fit(X_tr_s, y_tr_s)
    mu_s, std_f = gpr.predict(X_te_s, return_std=True)
    std_pred = np.sqrt(std_f ** 2 + GPR_ALPHA)
    z_lo = norm.ppf(Q_LOWER)
    z_hi = norm.ppf(Q_UPPER)
    lo_s = mu_s + z_lo * std_pred
    hi_s = mu_s + z_hi * std_pred
    dt = time.time() - t0

    lo = scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
    return np.minimum(lo, hi), np.maximum(lo, hi), dt


# ─── NGB: NGBoost with Normal dist, deterministic (minibatch_frac=1.0) ────────
def train_predict_ngb(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    """NGBoost Normal distribution — deterministic with minibatch_frac=1.0.
    init_seed unused (I=1). PI from Normal.ppf at Q_LOWER / Q_UPPER."""
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor
    from scipy.stats import norm

    t0 = time.time()
    ngb = NGBRegressor(
        Dist=Normal,
        Base=DecisionTreeRegressor(
            max_depth=NGB_MAX_DEPTH,
            min_samples_leaf=NGB_MIN_SAMPLES_LEAF,
            random_state=0,
        ),
        n_estimators=NGB_N_ESTIMATORS,
        learning_rate=NGB_LEARNING_RATE,
        minibatch_frac=1.0,
        natural_gradient=True,
        verbose=False,
        random_state=0,
    )
    ngb.fit(X_tr_s, y_tr_s)
    dist = ngb.pred_dist(X_te_s)
    lo_s = dist.ppf(Q_LOWER)
    hi_s = dist.ppf(Q_UPPER)
    dt = time.time() - t0

    lo = scaler_y.inverse_transform(np.minimum(lo_s, hi_s).reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(np.maximum(lo_s, hi_s).reshape(-1, 1)).flatten()
    return lo, hi, dt


MODEL_FUNCS = {
    "ANN": train_predict_ann,
    "QRF": train_predict_qrf,
    "SVM": train_predict_svm,
    "XGB": train_predict_xgb,
    "GPR": train_predict_gpr,
    "NGB": train_predict_ngb,
}


# ─── METRICS ──────────────────────────────────────────────────────────────────
def picp_mpiw(y_true, lo, hi):
    return (float(np.mean((y_true >= lo) & (y_true <= hi))),
            float(np.mean(hi - lo)))


def interval_score(y_true, lo, hi, alpha=0.10):
    """Interval Score (IS) for a (1-alpha)*100% prediction interval.

    IS = (u - l) + (2/alpha) * max(l - y, 0) + (2/alpha) * max(y - u, 0)

    Lower is better. Rewards narrow intervals but penalises missed coverage.
    Proper scoring rule — jointly evaluates calibration and sharpness.
    """
    width = hi - lo
    below = np.maximum(lo - y_true, 0.0)
    above = np.maximum(y_true - hi, 0.0)
    penalty = (2.0 / alpha) * (below + above)
    return float(np.mean(width + penalty))


def compute_true_bounds(X_te_raw):
    """True 90% PI bounds for sinc + Uniform(-1,1) noise.
    For Uniform(-1,1): q_0.05 = -0.9, q_0.95 = 0.9  ->  sinc(x) ± 0.9."""
    y_clean = np.sinc(X_te_raw.flatten() / np.pi)
    return y_clean - 0.9, y_clean + 0.9


def sum_rmse(true_lo, true_hi, lo, hi):
    """RMSE of mean PI vs true quantile bounds (not vs noisy y_test).
    Matches runnable_experiments: (RMSE_lo + RMSE_hi) / 2."""
    rmse_lo = float(np.sqrt(np.mean((lo - true_lo) ** 2)))
    rmse_hi = float(np.sqrt(np.mean((hi - true_hi) ** 2)))
    return (rmse_lo + rmse_hi) / 2.0


def decompose_raw(lo_arr, hi_arr):
    """sigma2_opt = E_d[Var_i[f]];  sigma2_in = Var_d[E_i[f]];  summed lo + hi.

    When I=1 (SVM, GPR), Var_i is degenerate -> sigma2_opt = 0.
    """
    mu_lo_d = lo_arr.mean(axis=1)   # (D, N_test)
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


# ─── ONE SIZE RUN (one model) ────────────────────────────────────────────────
def run_for_size(N, X_te_raw, y_te, true_lo, true_hi, model_name):
    train_fn = MODEL_FUNCS[model_name]
    I        = N_INIT_SEEDS_PER_MODEL[model_name]

    print(f"\n  === {model_name}  N_TRAIN={N}  (D x I = {N_DATA_SEEDS} x {I}) ===")

    preds_lo = np.zeros((N_DATA_SEEDS, I, N_TEST), dtype=np.float32)
    preds_hi = np.zeros((N_DATA_SEEDS, I, N_TEST), dtype=np.float32)
    run_rows = []
    t_start  = time.time()

    for d in range(N_DATA_SEEDS):
        X_tr_raw, y_tr_raw = generate_data(seed=d * 10000 + N, n=N)
        sc_X = StandardScaler().fit(X_tr_raw)
        sc_y = StandardScaler().fit(y_tr_raw.reshape(-1, 1))
        X_tr_s = sc_X.transform(X_tr_raw).astype(np.float32)
        X_te_s = sc_X.transform(X_te_raw).astype(np.float32)
        y_tr_s = sc_y.transform(y_tr_raw.reshape(-1, 1)).flatten().astype(np.float32)

        for i in range(I):
            lo, hi, dt = train_fn(X_tr_s, y_tr_s, X_te_s, sc_y, i)
            preds_lo[d, i] = lo
            preds_hi[d, i] = hi
            p, m = picp_mpiw(y_te, lo, hi)
            sr   = sum_rmse(true_lo, true_hi, lo, hi)
            is_val = interval_score(y_te, lo, hi, alpha=1.0 - (Q_UPPER - Q_LOWER))
            run_rows.append({
                "model": model_name, "n_train": N,
                "data_seed": d, "init_seed": i,
                "picp": round(p, 6), "mpiw": round(m, 4),
                "sum_rmse": round(sr, 4), "interval_score": round(is_val, 4),
                "time_s": round(dt, 2),
            })
            print(f"     {model_name}  N={N:4d}  d={d+1:2d}/i={i+1:2d}  "
                  f"PICP={p:.4f}  MPIW={m:.3f}  sumRMSE={sr:.3f}  IS={is_val:.3f}  t={dt:.1f}s")

        # Per-data-seed summary: init-averaged PI metrics for this data draw
        mean_lo_d = preds_lo[d].mean(axis=0)
        mean_hi_d = preds_hi[d].mean(axis=0)
        pd_picp, pd_mpiw = picp_mpiw(y_te, mean_lo_d, mean_hi_d)
        pd_sr = sum_rmse(true_lo, true_hi, mean_lo_d, mean_hi_d)
        pd_is = interval_score(y_te, mean_lo_d, mean_hi_d, alpha=1.0 - (Q_UPPER - Q_LOWER))
        run_rows.append({
            "model": model_name, "n_train": N,
            "data_seed": d, "init_seed": -1,
            "picp": round(pd_picp, 6), "mpiw": round(pd_mpiw, 4),
            "sum_rmse": round(pd_sr, 4), "interval_score": round(pd_is, 4),
            "time_s": 0.0,
        })
        print(f"     {model_name}  N={N:4d}  d={d+1:2d}  [init-mean]  "
              f"PICP={pd_picp:.4f}  MPIW={pd_mpiw:.3f}  sumRMSE={pd_sr:.3f}  IS={pd_is:.3f}")

    elapsed_min = (time.time() - t_start) / 60.0

    v_opt, v_in  = decompose_raw(preds_lo, preds_hi)
    sigma2_model = v_opt + v_in
    sigma_opt    = float(np.sqrt(v_opt))
    sigma_in     = float(np.sqrt(v_in))
    sigma_model  = float(np.sqrt(sigma2_model))

    # Headline metrics: per-data-seed mean PI (init-averaged), then averaged across data seeds.
    # For each data seed d: mean over I init seeds -> one PI per d.
    # RMSE/PICP/MPIW of that per-d mean PI, then average over D data seeds.
    per_d_rmse = []
    per_d_picp = []
    per_d_mpiw = []
    per_d_is   = []
    for d in range(N_DATA_SEEDS):
        mean_lo_d = preds_lo[d].mean(axis=0)    # mean over I init seeds -> (N_test,)
        mean_hi_d = preds_hi[d].mean(axis=0)
        per_d_rmse.append(sum_rmse(true_lo, true_hi, mean_lo_d, mean_hi_d))
        p, m = picp_mpiw(y_te, mean_lo_d, mean_hi_d)
        per_d_picp.append(p)
        per_d_mpiw.append(m)
        per_d_is.append(interval_score(y_te, mean_lo_d, mean_hi_d,
                                        alpha=1.0 - (Q_UPPER - Q_LOWER)))
    sr   = float(np.mean(per_d_rmse))
    picp = float(np.mean(per_d_picp))
    mpiw = float(np.mean(per_d_mpiw))
    is_mean = float(np.mean(per_d_is))

    summary = {
        "model"         : model_name,
        "n_train"       : N,
        "n_test"        : N_TEST,
        "sum_rmse"      : round(sr, 4),
        "picp"          : round(picp, 4),
        "mpiw"          : round(mpiw, 4),
        "interval_score": round(is_mean, 4),
        "sigma_opt"     : round(sigma_opt, 4),
        "sigma_in"      : round(sigma_in, 4),
        "sigma2_model"  : round(sigma2_model, 6),
        "sigma_model"   : round(sigma_model, 4),
        "elapsed_min"   : round(elapsed_min, 2),
    }
    print(f"     -> sigma_opt={sigma_opt:.4f}  sigma_in={sigma_in:.4f}  "
          f"sigma2_model={sigma2_model:.4f}  sigma_model={sigma_model:.4f}  "
          f"IS={is_mean:.3f}  [{elapsed_min:.1f} min]")
    return summary, run_rows


# ─── PER-MODEL PLOT ───────────────────────────────────────────────────────────
def plot_model_sweep(sum_df: pd.DataFrame, model_name: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(sum_df["n_train"], sum_df["sigma_opt"], marker="o", label="sigma opt", lw=2)
    ax.plot(sum_df["n_train"], sum_df["sigma_in"],  marker="s", label="sigma input",  lw=2)
    ax.set_xlabel("Total Training Points", fontsize=14)
    ax.set_ylabel("Sigma  (y units)", fontsize=14)
    ax.set_title(f"{model_name}_uncertainty", fontsize=16)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ─── COMBINED COMPARISON PLOT (sigma opt + sigma input) ──────────────────────
def plot_comparison(all_sum: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    titles  = ["sigma opt (init noise)", "sigma input (data noise)"]
    columns = ["sigma_opt", "sigma_in"]
    markers = {"ANN": "o", "QRF": "s", "SVM": "D", "XGB": "^", "GPR": "P", "NGB": "X"}
    colors  = {"ANN": "C0", "QRF": "C1", "SVM": "C2", "XGB": "C3", "GPR": "C4", "NGB": "C5"}
    # SVM plotted last in sigma_opt panel so it appears on top of other curves.
    draw_orders = {
        "sigma_opt": ["ANN", "QRF", "XGB", "GPR", "NGB", "SVM"],
        "sigma_in":  ["ANN", "QRF", "XGB", "GPR", "SVM", "NGB"],
    }

    ylabels = [r"$\sigma^2$ opt", r"$\sigma^2$ in"]
    for ax, col, title, ylabel in zip(axes, columns, titles, ylabels):
        for m in draw_orders[col]:
            sub = all_sum[all_sum["model"] == m]
            ax.plot(sub["n_train"], sub[col],
                    marker=markers[m], color=colors[m], label=m, lw=2)
        ax.set_xlabel("Total Training Points", fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=15)
        ax.tick_params(labelsize=13)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=13)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


# ─── SUM-OF-RMSE COMPARISON PLOT ────────────────────────────────────────────
def plot_sumrmse_comparison(all_sum: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"ANN": "o", "QRF": "s", "SVM": "D", "XGB": "^", "GPR": "P", "NGB": "X"}
    colors  = {"ANN": "C0", "QRF": "C1", "SVM": "C2", "XGB": "C3", "GPR": "C4", "NGB": "C5"}
    draw_order = ["ANN", "QRF", "XGB", "GPR", "SVM", "NGB"]
    for m in draw_order:
        sub = all_sum[all_sum["model"] == m]
        ax.plot(sub["n_train"], sub["sum_rmse"],
                marker=markers[m], color=colors[m], label=m, lw=2)
    ax.set_xlabel("Total Training Points", fontsize=14)
    ax.set_ylabel("Mean of RMSE", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ─── SIGMA2 MODEL PLOT ────────────────────────────────────────────────────────
def plot_sigma2_model_comparison(all_sum: pd.DataFrame, out_path: Path):
    """sigma2_model = sigma2_opt + sigma2_in for all models vs training size."""
    fig, ax = plt.subplots(figsize=(8, 5))
    markers = {"ANN": "o", "QRF": "s", "SVM": "D", "XGB": "^", "GPR": "P", "NGB": "X"}
    colors  = {"ANN": "C0", "QRF": "C1", "SVM": "C2", "XGB": "C3", "GPR": "C4", "NGB": "C5"}
    draw_order = ["ANN", "QRF", "XGB", "GPR", "SVM", "NGB"]
    for m in draw_order:
        sub = all_sum[all_sum["model"] == m]
        ax.plot(sub["n_train"], sub["sigma2_model"],
                marker=markers[m], color=colors[m], label=m, lw=2)
    ax.set_xlabel("Total Training Points", fontsize=14)
    ax.set_ylabel(r"$\sigma^2$ model", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ─── LEGEND ───────────────────────────────────────────────────────────────────
LEGEND_TEXT = """\
Column legend for sweep_<MODEL>.csv
====================================
One row per training size N. The test set is the SAME 500 points across
every row, so sigma_* values are directly comparable across N and across models.

Per-N summary columns
  model          Model name (ANN / QRF / SVM / XGB / GPR).
  n_train        Training points per data seed.
  n_test         Fixed test-set size.
  sum_rmse       (RMSE(mean_lo, true_lo) + RMSE(mean_hi, true_hi)) / 2,
                 where true_lo/true_hi are the known optimal PI bounds
                 for sinc + Uniform(-1,1).  Matches the runnable_experiments
                 definition.
  picp           Coverage of the mean PI on test set (target = Q_UPPER - Q_LOWER = 0.90).
  mpiw           Mean width of the mean PI on the test set.
  interval_score Interval Score — proper scoring rule for PIs.
                 IS = (u-l) + (2/alpha)*max(l-y,0) + (2/alpha)*max(y-u,0)
                 where alpha = 1 - (Q_UPPER - Q_LOWER) = 0.10.
                 Lower is better. Jointly evaluates calibration and sharpness.
  sigma_opt      sqrt( E_d [ Var_i [ f ] ] ) — init noise (sd-scale).
                 SVM and GPR are deterministic -> sigma_opt = 0.
  sigma_in       sqrt( Var_d [ E_i [ f ] ] ) — data noise (sd-scale).
  sigma2_model   sigma_opt^2 + sigma_in^2 — total predictive variance.
  sigma_model    sqrt(sigma2_model) — total predictive sd.
  elapsed_min    Wall-clock minutes for this N's D x I grid.

Per-run CSV columns (sweep_<MODEL>_runs.csv)
  model, n_train, data_seed, init_seed, picp, mpiw, sum_rmse, interval_score, time_s

Model-specific notes (hyperparameters fixed across all D x I x N runs).
All values come from exp8 stage-1 val-set tuning (best_hyperparams_2026_06_22_11_58_38.json).
----------------------------------------------------------------------
  ANN  : 1 hidden layer (512 ReLU), two pinball outputs (lo, hi), AdamW,
         100 epochs, batch=32, lr=3e-3.  Init seed = torch.manual_seed().
  QRF  : RandomForestRegressor, n_estimators=30, min_samples_leaf=5,
         max_features='sqrt'.  Init seed = random_state (no outer bootstrap;
         variance comes from RF's internal bagging + feature randomness).
         Quantiles from leaf distributions.
  SVM  : SVQR QP (port of epsilon_quantilesvr2.m), RBF kernel gamma=2^-0.5,
         C=2^3.5, eps=0, bias=0. Deterministic -> I=1, sigma_opt=0.
  XGB  : XGBRegressor reg:quantileerror, n_estimators=200, max_depth=2,
         lr=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
         reg_lambda=2.0. Init seed = random_state (stochastic boosting only —
         same training data across runs).
  GPR  : RBF length_scale=0.707, alpha=0.5 (noise variance in
         scaled-y units), no kernel optimisation. PI = mu ± z*sqrt(std_f^2 + alpha).
         Deterministic -> I=1, sigma_opt=0.
  NGB  : NGBoost with Normal distribution, n_estimators=200, lr=0.01,
         max_depth=2, min_samples_leaf=10. minibatch_frac=1.0 -> fully
         deterministic given fixed data -> I=1, sigma_opt=0.
         PI from Normal.ppf at Q_LOWER / Q_UPPER.
"""


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help=f"Models to run (default: {MODELS})")
    args = ap.parse_args()
    run_models = [m for m in args.models if m in MODELS]
    if not run_models:
        print(f"No valid models selected. Available: {MODELS}")
        return

    print(f"Device      : {DEVICE}")
    print(f"Models      : {run_models}")
    print(f"Sizes (N)   : {SIZES}")
    print(f"Test size   : {N_TEST}  (seed={TEST_SEED}, shared across all)")
    print(f"Quantiles   : {Q_LOWER} / {Q_UPPER}")
    print(f"Data seeds  : {N_DATA_SEEDS}")
    for m in run_models:
        print(f"  {m:4s} init seeds: {N_INIT_SEEDS_PER_MODEL[m]}")
    total = sum(len(SIZES) * N_DATA_SEEDS * N_INIT_SEEDS_PER_MODEL[m] for m in run_models)
    print(f"Total fits  : {total}")

    X_te_raw, y_te = generate_data(seed=TEST_SEED, n=N_TEST)
    true_lo, true_hi = compute_true_bounds(X_te_raw)

    all_sum   = []
    all_runs  = []
    grand_t0  = time.time()

    for model_name in run_models:
        print(f"\n{'='*70}\n  MODEL: {model_name}\n{'='*70}")
        model_sums = []
        model_runs = []

        for N in SIZES:
            s, runs = run_for_size(N, X_te_raw, y_te, true_lo, true_hi, model_name)
            model_sums.append(s)
            model_runs.extend(runs)

        mdf = pd.DataFrame(model_sums)
        rdf = pd.DataFrame(model_runs)
        mdf.to_csv(OUT_DIR / f"sweep_{model_name}.csv", index=False)
        rdf.to_csv(OUT_DIR / f"sweep_{model_name}_runs.csv", index=False)
        (OUT_DIR / f"sweep_{model_name}_legend.txt").write_text(
            LEGEND_TEXT, encoding="utf-8")
        plot_model_sweep(mdf, model_name,
                         OUT_DIR / f"sweep_{model_name}.png")

        all_sum.append(mdf)
        all_runs.append(rdf)

    grand_min = (time.time() - grand_t0) / 60.0

    combined_sum  = pd.concat(all_sum, ignore_index=True)
    combined_runs = pd.concat(all_runs, ignore_index=True)
    combined_sum.to_csv(OUT_DIR  / "sweep_comparison.csv", index=False)
    combined_runs.to_csv(OUT_DIR / "sweep_comparison_runs.csv", index=False)
    plot_comparison(combined_sum, OUT_DIR / "sweep_comparison.png")
    plot_sumrmse_comparison(combined_sum, OUT_DIR / "sweep_sumrmse.png")
    plot_sigma2_model_comparison(combined_sum, OUT_DIR / "sweep_sigma2_model.png")
    (OUT_DIR / "sweep_comparison_legend.txt").write_text(
        LEGEND_TEXT, encoding="utf-8")

    # ── Headline tables ──────────────────────────────────────────────────────
    print("\n" + "=" * 110)
    print(f"  ALL MODELS — sinc UQ sweep  (D={N_DATA_SEEDS}, n_test={N_TEST})")
    print("=" * 110)
    for model_name in run_models:
        sub = combined_sum[combined_sum["model"] == model_name]
        print(f"\n  --- {model_name} ---")
        print(f"  {'N':>6}  {'sum_rmse':>9}  {'PICP':>6}  {'MPIW':>7}  {'IS':>9}  "
              f"{'sigma_opt':>10}  {'sigma_in':>10}  {'sigma2_model':>13}  {'sigma_model':>12}")
        print("  " + "-" * 108)
        for _, r in sub.iterrows():
            print(f"  {int(r['n_train']):6d}  {r['sum_rmse']:9.3f}  {r['picp']:.4f}  "
                  f"{r['mpiw']:7.3f}  {r['interval_score']:9.3f}  "
                  f"{r['sigma_opt']:10.4f}  {r['sigma_in']:10.4f}  "
                  f"{r['sigma2_model']:13.4f}  {r['sigma_model']:12.4f}")

    print(f"\n  Total runtime: {grand_min:.1f} min")
    print(f"  Per-model CSVs:   {OUT_DIR}/sweep_<MODEL>.csv")
    print(f"  Per-run  CSVs:    {OUT_DIR}/sweep_<MODEL>_runs.csv")
    print(f"  Comparison CSV:   {OUT_DIR}/sweep_comparison.csv")
    print(f"  Per-model plots:  {OUT_DIR}/sweep_<MODEL>.png")
    print(f"  Comparison plot:  {OUT_DIR}/sweep_comparison.png")


if __name__ == "__main__":
    main()
