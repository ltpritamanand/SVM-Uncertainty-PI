"""
exp14: Test-set size sweep — fixed N_TRAIN=100, varying test sizes.
=====================================================================
All six models (ANN, QRF, SVM, XGB, GPR, NGB) are trained on a fixed
N_TRAIN=100 sinc+U(-1,1) dataset and evaluated on five test sets of
increasing size: [600, 1200, 1800, 2400, 3000].

Protocol (same DxI nested design as exp9):
  - D=10 independent training draws (data seeds), N_TRAIN=100 each.
  - I=10 init seeds for stochastic models (ANN, QRF, XGB); I=1 for
    deterministic models (SVM, GPR, NGB).
  - For each (model, test_size): DxI fits, predict on the fixed test
    set of that size, then decompose.

Metrics per (model, test_size):
  sum_rmse     — mean over D seeds of RMSE(mean_I PI, true bounds) / 2
  sigma2_opt   = E_d[Var_i[f]]   (init / optimisation noise, squared)
  sigma2_in    = Var_d[E_i[f]]   (data-subsample noise, squared)
  sigma2_model = sigma2_opt + sigma2_in

Output (results/):
  sweep_comparison.csv          combined summary (model x test_size)
  sweep_<MODEL>.csv             per-model summary
  sweep_comparison.png          sigma2_opt + sigma2_in  2-panel plot
  sweep_sumrmse.png             sum_rmse vs test_size
  sweep_sigma2_model.png        sigma2_model vs test_size

Usage:
  python test_size_sweep.py
  python test_size_sweep.py --models ANN SVM      # subset of models
"""

import argparse
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─── CONFIG ───────────────────────────────────────────────────────────────────
MODELS       = ["ANN", "QRF", "SVM", "XGB", "GPR", "NGB"]
N_TRAIN      = 100
TEST_SIZES   = [600, 1200, 1800, 2400, 3000]

# Each test size gets its own fixed seed so test sets are independent.
TEST_SEEDS   = {600: 9999, 1200: 8888, 1800: 7777, 2400: 6666, 3000: 5555}

N_DATA_SEEDS = 10
N_INIT_SEEDS = 10

X_LOW, X_HIGH         = -6.0, 6.0
NOISE_LOW, NOISE_HIGH = -1.0, 1.0
Q_LOWER, Q_UPPER      = 0.05, 0.95

N_INIT_SEEDS_PER_MODEL = {
    "ANN": 10, "QRF": 10, "SVM": 1, "XGB": 10, "GPR": 1, "NGB": 1
}

# Fixed XGB params not covered by the tuning grid
XGB_COLSAMPLE        = 0.8
XGB_MIN_CHILD_WEIGHT = 3
XGB_REG_LAMBDA       = 2.0

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)

# ── Load hyperparameters from exp8 stage-1 tuning ────────────────────────────
import json as _json
_EXP8_TUNE_DIR = (SCRIPT_DIR.parent
                  / "appendixA_nn_ablation" / "results" / "hyperparam_tuning")
_tune_files = sorted(_EXP8_TUNE_DIR.glob("best_hyperparams_*.json"))
if not _tune_files:
    raise FileNotFoundError(
        f"No best_hyperparams_*.json found in {_EXP8_TUNE_DIR}. "
        "Run exp8 stage-1 tuning first.")
_BEST = _json.loads(_tune_files[-1].read_text())["best"]
print(f"[exp14] Loaded hyperparams from: {_tune_files[-1].name}")

_nn  = _BEST["NN"]["config"]
ANN_HIDDEN = _nn["hidden_sizes"][0]   # single hidden layer width
ANN_EPOCHS = _nn["epochs"]
ANN_BATCH  = _nn["batch_size"]
ANN_LR     = _nn["lr"]

_svm = _BEST["SVM"]["config"]
SVM_GAMMA = _svm["gamma"]
SVM_C     = _svm["C"]

_qrf = _BEST["QRF"]["config"]
QRF_N_ESTIMATORS     = _qrf["n_estimators"]
QRF_MIN_SAMPLES_LEAF = _qrf["min_samples_leaf"]
QRF_MAX_FEATURES     = _qrf["max_features"]

_xgb = _BEST["XGB"]["config"]
XGB_N_ESTIMATORS = _xgb["n_estimators"]
XGB_MAX_DEPTH    = _xgb["max_depth"]
XGB_LR           = _xgb["learning_rate"]
XGB_SUBSAMPLE    = _xgb["subsample"]

_gpr = _BEST["GPR"]["config"]
GPR_LENGTH_SCALE = _gpr["length_scale"]
GPR_ALPHA        = _gpr["alpha"]

_ngb = _BEST["NGB"]["config"]
NGB_N_ESTIMATORS     = _ngb["n_estimators"]
NGB_LEARNING_RATE    = _ngb["learning_rate"]
NGB_MAX_DEPTH        = _ngb["max_depth"]
NGB_MIN_SAMPLES_LEAF = _ngb["min_samples_leaf"]


# ─── DATA ─────────────────────────────────────────────────────────────────────
def generate_data(seed: int, n: int):
    rng = np.random.default_rng(seed)
    x   = rng.uniform(X_LOW, X_HIGH, size=n).astype(np.float32)
    eps = rng.uniform(NOISE_LOW, NOISE_HIGH, size=n).astype(np.float32)
    y   = (np.sinc(x / np.pi) + eps).astype(np.float32)
    return x.reshape(-1, 1), y


def compute_true_bounds(X_raw):
    """True 90% PI: sinc(x) ± 0.9  (Uniform(-1,1) quantiles at 0.05/0.95)."""
    y_clean = np.sinc(X_raw.flatten() / np.pi)
    return y_clean - 0.9, y_clean + 0.9


# ─── METRICS ──────────────────────────────────────────────────────────────────
def sum_rmse(true_lo, true_hi, lo, hi):
    rmse_lo = float(np.sqrt(np.mean((lo - true_lo) ** 2)))
    rmse_hi = float(np.sqrt(np.mean((hi - true_hi) ** 2)))
    return (rmse_lo + rmse_hi) / 2.0


def decompose_raw(lo_arr, hi_arr):
    """sigma2_opt = E_d[Var_i[f]];  sigma2_in = Var_d[E_i[f]];  summed lo+hi."""
    mu_lo_d = lo_arr.mean(axis=1)
    mu_hi_d = hi_arr.mean(axis=1)
    I = lo_arr.shape[1]
    v_opt = 0.0 if I < 2 else (lo_arr.var(axis=1, ddof=1).mean()
                                + hi_arr.var(axis=1, ddof=1).mean())
    D = lo_arr.shape[0]
    v_in = 0.0 if D < 2 else (mu_lo_d.var(axis=0, ddof=1).mean()
                               + mu_hi_d.var(axis=0, ddof=1).mean())
    return float(v_opt), float(v_in)


# ═══════════════════════════════════════════════════════════════════════════════
#  MODEL BACKENDS  (identical to exp9)
#  Signature: fn(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed) -> (lo, hi, dt)
# ═══════════════════════════════════════════════════════════════════════════════

# ─── ANN ──────────────────────────────────────────────────────────────────────
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
            loss.backward(); opt.step()
    model.eval()
    Xe = torch.tensor(X_te_s, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(Xe).cpu().numpy()
    lo_s = np.minimum(out[:, 0], out[:, 1])
    hi_s = np.maximum(out[:, 0], out[:, 1])
    lo = scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
    return lo, hi, time.time() - t0


# ─── QRF ──────────────────────────────────────────────────────────────────────
def _qrf_leaf_predict(rf, X_tr_s, y_tr_s, X_te_s, quantile):
    train_leaves = rf.apply(X_tr_s)
    test_leaves  = rf.apply(X_te_s)
    n_test = test_leaves.shape[0]
    preds  = np.empty(n_test, dtype=np.float64)
    for i in range(n_test):
        ys = []
        for t in range(train_leaves.shape[1]):
            mask = train_leaves[:, t] == test_leaves[i, t]
            if mask.any():
                ys.append(y_tr_s[mask])
        preds[i] = (np.quantile(np.concatenate(ys), quantile) if ys
                    else float(rf.predict(X_te_s[i:i+1])[0]))
    return preds


def train_predict_qrf(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    t0 = time.time()
    rf = RandomForestRegressor(
        n_estimators=QRF_N_ESTIMATORS,
        min_samples_leaf=QRF_MIN_SAMPLES_LEAF,
        max_features=QRF_MAX_FEATURES,
        random_state=init_seed, n_jobs=-1)
    rf.fit(X_tr_s, y_tr_s)
    lo_s = _qrf_leaf_predict(rf, X_tr_s, y_tr_s, X_te_s, Q_LOWER)
    hi_s = _qrf_leaf_predict(rf, X_tr_s, y_tr_s, X_te_s, Q_UPPER)
    lo = scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
    return lo, hi, time.time() - t0


# ─── SVM ──────────────────────────────────────────────────────────────────────
def _rbf_kernel_mat(X1, X2, s):
    X1sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2sq = np.sum(X2 ** 2, axis=1, keepdims=True)
    D2   = np.maximum(X1sq + X2sq.T - 2.0 * (X1 @ X2.T), 0.0)
    return np.exp(-D2 / (2.0 * s))


def _solve_svqr_qp(X_tr_s, y_tr_s, s, C, tau):
    n  = X_tr_s.shape[0]
    H  = _rbf_kernel_mat(X_tr_s, X_tr_s, s)
    Hb = np.block([[H, -H], [-H, H]]) + 1e-10 * np.eye(2 * n)
    c  = np.concatenate([-y_tr_s, y_tr_s])
    bounds = [(0.0, tau * C)] * n + [(0.0, (1.0 - tau) * C)] * n
    res = sp_min(
        fun=lambda v: 0.5 * float(v @ Hb @ v) + float(c @ v),
        jac=lambda v: Hb @ v + c,
        x0=np.zeros(2 * n), method='SLSQP', bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-10},
    )
    return res.x[:n] - res.x[n:]


def train_predict_svm(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    t0 = time.time()
    s  = 1.0 / (2.0 * SVM_GAMMA)
    beta_lo = _solve_svqr_qp(X_tr_s, y_tr_s, s, SVM_C, Q_LOWER)
    beta_hi = _solve_svqr_qp(X_tr_s, y_tr_s, s, SVM_C, Q_UPPER)
    K_te = _rbf_kernel_mat(X_te_s, X_tr_s, s)
    lo_s = K_te @ beta_lo; hi_s = K_te @ beta_hi
    lo = scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
    return np.minimum(lo, hi), np.maximum(lo, hi), time.time() - t0


# ─── XGB ──────────────────────────────────────────────────────────────────────
def train_predict_xgb(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    t0 = time.time()
    shared = dict(
        objective="reg:quantileerror",
        n_estimators=XGB_N_ESTIMATORS, max_depth=XGB_MAX_DEPTH,
        learning_rate=XGB_LR, subsample=XGB_SUBSAMPLE,
        colsample_bytree=XGB_COLSAMPLE,
        min_child_weight=XGB_MIN_CHILD_WEIGHT, reg_lambda=XGB_REG_LAMBDA,
        verbosity=0, n_jobs=-1,
    )
    xgb_lo = XGBRegressor(quantile_alpha=Q_LOWER, random_state=init_seed,        **shared)
    xgb_hi = XGBRegressor(quantile_alpha=Q_UPPER, random_state=init_seed + 10000, **shared)
    xgb_lo.fit(X_tr_s, y_tr_s); xgb_hi.fit(X_tr_s, y_tr_s)
    lo = scaler_y.inverse_transform(xgb_lo.predict(X_te_s).reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(xgb_hi.predict(X_te_s).reshape(-1, 1)).flatten()
    return np.minimum(lo, hi), np.maximum(lo, hi), time.time() - t0


# ─── GPR ──────────────────────────────────────────────────────────────────────
def train_predict_gpr(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    t0 = time.time()
    kernel = RBF(length_scale=GPR_LENGTH_SCALE)
    gpr = GaussianProcessRegressor(
        kernel=kernel, optimizer=None,
        alpha=GPR_ALPHA, normalize_y=False, random_state=init_seed)
    gpr.fit(X_tr_s, y_tr_s)
    mu_s, std_f = gpr.predict(X_te_s, return_std=True)
    std_pred = np.sqrt(std_f ** 2 + GPR_ALPHA)
    lo_s = mu_s + norm.ppf(Q_LOWER) * std_pred
    hi_s = mu_s + norm.ppf(Q_UPPER) * std_pred
    lo = scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
    return np.minimum(lo, hi), np.maximum(lo, hi), time.time() - t0


# ─── NGB ──────────────────────────────────────────────────────────────────────
def train_predict_ngb(X_tr_s, y_tr_s, X_te_s, scaler_y, init_seed):
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.tree import DecisionTreeRegressor
    t0 = time.time()
    ngb = NGBRegressor(
        Dist=Normal,
        Base=DecisionTreeRegressor(
            max_depth=NGB_MAX_DEPTH,
            min_samples_leaf=NGB_MIN_SAMPLES_LEAF,
            random_state=0),
        n_estimators=NGB_N_ESTIMATORS,
        learning_rate=NGB_LEARNING_RATE,
        minibatch_frac=1.0, natural_gradient=True,
        verbose=False, random_state=0,
    )
    ngb.fit(X_tr_s, y_tr_s)
    dist = ngb.pred_dist(X_te_s)
    lo_s = dist.ppf(Q_LOWER); hi_s = dist.ppf(Q_UPPER)
    lo = scaler_y.inverse_transform(np.minimum(lo_s, hi_s).reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(np.maximum(lo_s, hi_s).reshape(-1, 1)).flatten()
    return lo, hi, time.time() - t0


MODEL_FUNCS = {
    "ANN": train_predict_ann,
    "QRF": train_predict_qrf,
    "SVM": train_predict_svm,
    "XGB": train_predict_xgb,
    "GPR": train_predict_gpr,
    "NGB": train_predict_ngb,
}


# ─── CORE RUN — one (model, test_size) cell ───────────────────────────────────
def run_cell(model_name, test_size, X_te_raw, y_te, true_lo, true_hi):
    """D x I fits on N_TRAIN=100; predict on fixed test set of `test_size` pts."""
    train_fn = MODEL_FUNCS[model_name]
    I        = N_INIT_SEEDS_PER_MODEL[model_name]

    preds_lo = np.zeros((N_DATA_SEEDS, I, test_size), dtype=np.float32)
    preds_hi = np.zeros((N_DATA_SEEDS, I, test_size), dtype=np.float32)

    t_start = time.time()
    for d in range(N_DATA_SEEDS):
        X_tr_raw, y_tr_raw = generate_data(seed=d * 10000 + N_TRAIN, n=N_TRAIN)
        sc_X = StandardScaler().fit(X_tr_raw)
        sc_y = StandardScaler().fit(y_tr_raw.reshape(-1, 1))
        X_tr_s = sc_X.transform(X_tr_raw).astype(np.float32)
        X_te_s = sc_X.transform(X_te_raw).astype(np.float32)
        y_tr_s = sc_y.transform(y_tr_raw.reshape(-1, 1)).flatten().astype(np.float32)

        for i in range(I):
            lo, hi, dt = train_fn(X_tr_s, y_tr_s, X_te_s, sc_y, i)
            preds_lo[d, i] = lo
            preds_hi[d, i] = hi
            sr = sum_rmse(true_lo, true_hi, lo, hi)
            print(f"    {model_name}  T={test_size}  d={d+1:2d}/i={i+1:2d}"
                  f"  sum_rmse={sr:.4f}  t={dt:.1f}s")

    elapsed_min = (time.time() - t_start) / 60.0

    v_opt, v_in  = decompose_raw(preds_lo, preds_hi)
    sigma2_model = v_opt + v_in

    # sum_rmse: per data-seed (init-averaged PI) then mean over D seeds
    per_d_sr = []
    for d in range(N_DATA_SEEDS):
        mean_lo_d = preds_lo[d].mean(axis=0)
        mean_hi_d = preds_hi[d].mean(axis=0)
        per_d_sr.append(sum_rmse(true_lo, true_hi, mean_lo_d, mean_hi_d))
    sr_mean = float(np.mean(per_d_sr))

    print(f"    -> {model_name} T={test_size}:  sum_rmse={sr_mean:.4f}"
          f"  σ²opt={v_opt:.5f}  σ²in={v_in:.5f}"
          f"  σ²model={sigma2_model:.5f}  [{elapsed_min:.1f} min]")

    return {
        "model":        model_name,
        "n_train":      N_TRAIN,
        "test_size":    test_size,
        "sum_rmse":     round(sr_mean, 4),
        "sigma2_opt":   round(v_opt,         6),
        "sigma2_in":    round(v_in,          6),
        "sigma2_model": round(sigma2_model,  6),
        "sigma_opt":    round(float(np.sqrt(v_opt)),         4),
        "sigma_in":     round(float(np.sqrt(v_in)),          4),
        "sigma_model":  round(float(np.sqrt(sigma2_model)),  4),
        "elapsed_min":  round(elapsed_min, 2),
    }


# ─── PLOTS ────────────────────────────────────────────────────────────────────
MARKERS = {"ANN": "o", "QRF": "s", "SVM": "D", "XGB": "^", "GPR": "P", "NGB": "X"}
COLORS  = {"ANN": "C0", "QRF": "C1", "SVM": "C2", "XGB": "C3", "GPR": "C4", "NGB": "C5"}
DRAW_ORDER = ["ANN", "QRF", "XGB", "GPR", "SVM", "NGB"]


def _draw_lines(ax, df, col, draw_order):
    for m in draw_order:
        sub = df[df["model"] == m]
        if sub.empty:
            continue
        ax.plot(sub["test_size"], sub[col],
                marker=MARKERS[m], color=COLORS[m], label=m, lw=2)
    ax.set_xlabel("Test-set size", fontsize=14)
    ax.tick_params(labelsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=13)


def plot_comparison(df, out_path):
    """2-panel: sigma2_opt (left) + sigma2_in (right) vs test size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    specs = [
        ("sigma2_opt", r"$\sigma^2$ opt", "sigma2 opt (init noise)",
         ["ANN", "QRF", "XGB", "GPR", "NGB", "SVM"]),
        ("sigma2_in",  r"$\sigma^2$ in",  "sigma2 in (data noise)",
         ["ANN", "QRF", "XGB", "GPR", "SVM", "NGB"]),
    ]
    for ax, (col, ylabel, title, order) in zip(axes, specs):
        _draw_lines(ax, df, col, order)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.set_title(title, fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_sumrmse(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_lines(ax, df, "sum_rmse", DRAW_ORDER)
    ax.set_ylabel("Mean of RMSE", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_sigma2_model(df, out_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    _draw_lines(ax, df, "sigma2_model", DRAW_ORDER)
    ax.set_ylabel(r"$\sigma^2$ model", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=MODELS,
                    help=f"Models to run (default: all {MODELS})")
    args = ap.parse_args()
    run_models = [m for m in args.models if m in MODELS]
    if not run_models:
        print(f"No valid models. Available: {MODELS}"); return

    print(f"Device      : {DEVICE}")
    print(f"N_TRAIN     : {N_TRAIN}  (fixed)")
    print(f"Test sizes  : {TEST_SIZES}")
    print(f"Models      : {run_models}")
    print(f"Data seeds  : {N_DATA_SEEDS}")
    for m in run_models:
        print(f"  {m:4s}  init seeds: {N_INIT_SEEDS_PER_MODEL[m]}")

    all_rows  = []
    grand_t0  = time.time()

    for test_size in TEST_SIZES:
        seed = TEST_SEEDS[test_size]
        X_te_raw, y_te = generate_data(seed=seed, n=test_size)
        true_lo, true_hi = compute_true_bounds(X_te_raw)
        print(f"\n{'='*70}\n  TEST SIZE: {test_size}  (seed={seed})\n{'='*70}")

        for model_name in run_models:
            print(f"\n  --- {model_name}  (D={N_DATA_SEEDS},"
                  f" I={N_INIT_SEEDS_PER_MODEL[model_name]}) ---")
            row = run_cell(model_name, test_size, X_te_raw, y_te,
                           true_lo, true_hi)
            all_rows.append(row)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_DIR / "sweep_comparison.csv", index=False)
    print(f"\nSaved: {OUT_DIR}/sweep_comparison.csv")

    for m in run_models:
        sub = df[df["model"] == m]
        sub.to_csv(OUT_DIR / f"sweep_{m}.csv", index=False)

    plot_comparison(df, OUT_DIR / "sweep_comparison.png")
    plot_sumrmse(df,    OUT_DIR / "sweep_sumrmse.png")
    plot_sigma2_model(df, OUT_DIR / "sweep_sigma2_model.png")

    print(f"\nPlots saved to {OUT_DIR}/")
    print(f"Total runtime: {(time.time()-grand_t0)/60:.1f} min")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"  exp14 summary — N_TRAIN={N_TRAIN}, all models")
    print("=" * 90)
    for m in run_models:
        sub = df[df["model"] == m]
        print(f"\n  --- {m} ---")
        print(f"  {'test_size':>10}  {'sum_rmse':>9}  {'σ²opt':>10}"
              f"  {'σ²in':>10}  {'σ²model':>10}")
        print("  " + "-" * 60)
        for _, r in sub.iterrows():
            print(f"  {int(r['test_size']):10d}  {r['sum_rmse']:9.4f}"
                  f"  {r['sigma2_opt']:10.5f}  {r['sigma2_in']:10.5f}"
                  f"  {r['sigma2_model']:10.5f}")


if __name__ == "__main__":
    main()
