#!/usr/bin/env python3
"""
Replicate Table B1: Stability analysis comparing NN, QRF, SVM, XGB on synthetic sinc data.

Paper setup: n_train=50, M=10, seed=58, 1-alpha=0.90 (q=0.05/0.95), 600 test points.
Expected (NN/QRF/SVM, from paper):
  Method       SumofRMSEs    PICP    sigma2_model
  NN             0.353       0.587      0.132
  QRFQuantile    0.264       0.755      0.123
  SVMQuantile    0.266       0.848      0.00
XGB is reported as an additional baseline.

Key decisions:
- SVM  : SVQR QP (port of epsilon_quantilesvr2.m), RBF kernel, eps=0, direct quantile
         loss; all M members identical -> sigma2=0 exactly
- QRF  : M runs on same fixed data, different random_state (internal RF bagging only)
- NN   : M runs on same fixed data with different random seeds (init variance only)
- XGB  : M runs on same fixed data, different random seeds; subsample=0.8/colsample=0.8
         provides internal stochastic-boosting noise (analogous to NN weight-init noise)
- SumofRMSEs = average RMSE of lower+upper bounds vs true quantiles on grid = (RMSE_lo+RMSE_hi)/2
- sigma2_model = mean((f_q^j - mean_q)^2) over runs and test points, summed over lo+hi heads
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

current_file = Path(__file__).resolve()
# Point to shared/src_utils/ for imported modules
shared_dir = current_file.parent.parent / "shared" / "src_utils"
if str(shared_dir) not in sys.path:
    sys.path.insert(0, str(shared_dir))

import run_synthetic_tube_quantile_ensembles as experiment
from evaluate_nn_ensemble import _bounds_stats, picp_mpiw, interval_score


# ---------------------------------------------------------------------------
# Training functions
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# SVQR: Python port of epsilon_quantilesvr2.m + kernelfun.m (RBF, eps1=0)
# ---------------------------------------------------------------------------

def _rbf_kernel_mat(X1, X2, s):
    """K(xi,xj) = exp(-||xi-xj||^2 / (2*s)).
    Matches kernelfun.m: omega = exp(-omega/(2*kernel_pars(1))).
    s=1.0 is equivalent to sklearn gamma=0.5."""
    # squared distance matrix
    X1sq = np.sum(X1 ** 2, axis=1, keepdims=True)
    X2sq = np.sum(X2 ** 2, axis=1, keepdims=True)
    D2 = X1sq + X2sq.T - 2.0 * (X1 @ X2.T)
    D2 = np.maximum(D2, 0.0)
    return np.exp(-D2 / (2.0 * s))


def _epsilon_quantilesvr2_py(X_tr_s, y_tr_s, s, C, tau):
    """Python port of epsilon_quantilesvr2.m with eps1=0, RBF kernel.
    Accepts ONLY scaled training data — no test data to prevent any leakage.
    Test-set prediction is handled entirely by SVQRMember.predict_quantiles.

    QP (nobias=0 for RBF -> no equality constraint, bias=0):
      min  0.5 * v^T * Hb * v  +  c^T * v
      s.t. 0 <= alpha1 <= tau*C          (first n variables)
           0 <= beta1  <= (1-tau)*C      (last n variables)
    where:
      Hb = [H -H; -H H] + 1e-10*I      H = RBF kernel on training data
      c  = [-y; y]                       (eps1=0)
      beta = alpha1 - beta1              (dual coefficients)
      bias = 0                           (nobias('rbf') = 0)
    """
    from scipy.optimize import minimize as sp_min

    n = X_tr_s.shape[0]
    svtol = 1e-7 * max(C, 1.0)          # svtol.m: eps = 1e-7 * max(C,1)

    H  = _rbf_kernel_mat(X_tr_s, X_tr_s, s)   # train kernel only
    Hb = np.block([[H, -H], [-H, H]]) + 1e-10 * np.eye(2 * n)
    c  = np.concatenate([-y_tr_s, y_tr_s])     # eps1 = 0

    bounds = [(0.0, tau * C)] * n + [(0.0, (1.0 - tau) * C)] * n
    v0     = np.zeros(2 * n)

    res = sp_min(
        fun=lambda v: 0.5 * float(v @ Hb @ v) + float(c @ v),
        jac=lambda v: Hb @ v + c,
        x0=v0, method='SLSQP', bounds=bounds,
        options={'maxiter': 2000, 'ftol': 1e-12},
    )

    alpha1, beta1 = res.x[:n], res.x[n:]
    beta     = alpha1 - beta1                  # dual coefficients
    bias     = 0.0                             # nobias('rbf') = 0
    sparsity = 1.0 - np.count_nonzero(np.abs(beta) > svtol) / n
    return beta, bias, sparsity


class SVQRMember:
    """SVQR member: two independently solved quantile QPs (lo and hi)."""
    def __init__(self, beta_lo, beta_hi, bias_lo, bias_hi,
                 X_tr_s, s, scaler_X, scaler_y):
        self.beta_lo  = beta_lo
        self.beta_hi  = beta_hi
        self.bias_lo  = bias_lo
        self.bias_hi  = bias_hi
        self.X_tr_s   = X_tr_s
        self.s        = s
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

    def predict_quantiles(self, X):
        X_s  = self.scaler_X.transform(X)
        K    = _rbf_kernel_mat(X_s, self.X_tr_s, self.s)
        lo_s = K @ self.beta_lo + self.bias_lo
        hi_s = K @ self.beta_hi + self.bias_hi
        lo = self.scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
        hi = self.scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
        return np.minimum(lo, hi), np.maximum(lo, hi)


def _train_svqr(*, X_train, y_train, seed, n_ens, q_lower, q_upper,
                gamma=0.5, C=10.0):
    """Solve SVQR QP (epsilon_quantilesvr2.m) for q_lower and q_upper.
    s = 1/(2*gamma) converts sklearn gamma convention to MATLAB kerfPara.pars.
    Deterministic given fixed data -> sigma2=0 exactly (like SVM).
    All n_ens members are identical copies."""
    from sklearn.preprocessing import StandardScaler

    s = 1.0 / (2.0 * gamma)          # gamma=0.5 -> s=1.0

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_s = scaler_X.fit_transform(X_train)
    y_s = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()

    # QP sees only scaled training data — test prediction is in SVQRMember
    beta_lo, bias_lo, sp_lo = _epsilon_quantilesvr2_py(X_s, y_s, s, C, q_lower)
    beta_hi, bias_hi, sp_hi = _epsilon_quantilesvr2_py(X_s, y_s, s, C, q_upper)

    print(f"    [SVQR] sparsity lo={sp_lo:.3f}  hi={sp_hi:.3f}")

    member = SVQRMember(beta_lo, beta_hi, bias_lo, bias_hi,
                        X_s, s, scaler_X, scaler_y)
    return [member] * n_ens           # identical copies -> sigma2 = 0


class FastQRFMember:
    def __init__(self, rf, X_train, y_train):
        self.rf = rf
        self.X_train = X_train
        self.y_train = y_train
        self._build_cache()

    def _build_cache(self):
        self._leaf_cache = []
        for tree in self.rf.estimators_:
            leaf_ids = tree.apply(self.X_train)
            ltoy = {}
            for lid in np.unique(leaf_ids):
                ltoy[lid] = self.y_train[leaf_ids == lid]
            self._leaf_cache.append(ltoy)

    def predict_quantiles(self, X, *, q_lower, q_upper):
        n = X.shape[0]
        lo, hi = np.empty(n), np.empty(n)
        all_leaves = np.array([t.apply(X) for t in self.rf.estimators_])
        for i in range(n):
            ys = []
            for t, leaves in enumerate(all_leaves):
                y_leaf = self._leaf_cache[t].get(leaves[i])
                if y_leaf is not None and len(y_leaf):
                    ys.append(y_leaf)
            if ys:
                y_all = np.concatenate(ys)
                lo[i], hi[i] = np.quantile(y_all, q_lower), np.quantile(y_all, q_upper)
            else:
                lo[i] = hi[i] = self.rf.predict(X[i:i+1])[0]
        return np.minimum(lo, hi), np.maximum(lo, hi)


def _train_qrf_bootstrap(*, X_train, y_train, seed, n_ens, n_estimators,
                         min_samples_leaf, max_features):
    """Train n_ens QRF members on the SAME fixed data with different random seeds.
    Variance comes from RandomForest's internal bagging / feature randomness only —
    no outer bootstrap, matching the NN/XGB fixed-data design."""
    from sklearn.ensemble import RandomForestRegressor
    members = []
    for m in range(n_ens):
        rf = RandomForestRegressor(n_estimators=n_estimators, random_state=seed + m,
                                   min_samples_leaf=min_samples_leaf,
                                   max_features=max_features, n_jobs=-1)
        rf.fit(X_train, y_train)
        members.append(FastQRFMember(rf, X_train, y_train))
    return members


class XGBMember:
    """XGBoost quantile-regression member (separate lo / hi boosters)."""
    def __init__(self, model_lo, model_hi):
        self.model_lo = model_lo
        self.model_hi = model_hi

    def predict_quantiles(self, X):
        lo = self.model_lo.predict(X)
        hi = self.model_hi.predict(X)
        return np.minimum(lo, hi), np.maximum(lo, hi)


def _train_xgb_fixed_data(*, X_train, y_train, seed, n_ens, q_lower, q_upper,
                          n_estimators=200, max_depth=2, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          min_child_weight=3, reg_lambda=2.0):
    """Train n_ens XGBoost members on the SAME fixed data with different random seeds.
    Variance comes from internal stochastic boosting (subsample/colsample_bytree),
    matching the NN design: fixed data, only the random seed varies across runs.

    Hyperparameter rationale for n_train=50:
      max_depth=2      : shallow stumps prevent overfitting; n=50 cannot support depth-5 splits
      n_estimators=500 : more trees compensate for small depth and low learning rate
      learning_rate=0.02: slow shrinkage improves generalisation at small n
      min_child_weight=3: each leaf needs >= 3 samples, prevents 1-sample leaf quantiles
      reg_lambda=2.0   : stronger L2 regularisation than default (1.0)
      subsample=0.8    : stochastic boosting; also the source of sigma2 with fixed data
    """
    import xgboost as xgb
    members = []
    for m in range(n_ens):
        shared = dict(
            objective="reg:quantileerror",
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight, reg_lambda=reg_lambda,
            verbosity=0, n_jobs=-1,
        )
        m_lo = xgb.XGBRegressor(quantile_alpha=q_lower,
                                 random_state=seed + m, **shared)
        m_lo.fit(X_train, y_train)

        m_hi = xgb.XGBRegressor(quantile_alpha=q_upper,
                                 random_state=seed + m + 10000, **shared)
        m_hi.fit(X_train, y_train)

        members.append(XGBMember(m_lo, m_hi))
    return members


class GPRMember:
    """GPR member: PI from full predictive distribution (f + noise).

    gpr.predict(return_std=True) returns sigma_f(x) — posterior std of the
    latent function in original y units. For a new observation y = f(x) + noise,
    the correct predictive std is:
        sigma_pred = sqrt(sigma_f² + alpha * y_std²)
    where alpha is the noise variance in normalised y space (as passed to GPR)
    and y_std = gpr._y_train_std undoes the normalize_y scaling."""
    def __init__(self, gpr, scaler_X, q_lower, q_upper):
        self.gpr           = gpr
        self.scaler_X      = scaler_X
        self.q_lower       = q_lower
        self.q_upper       = q_upper
        # noise variance in original y units: alpha (normalised) × y_std²
        self._noise_var    = gpr.alpha * float(gpr._y_train_std ** 2)

    def predict_quantiles(self, X):
        from scipy.stats import norm
        X_s      = self.scaler_X.transform(X)
        mu, std_f = self.gpr.predict(X_s, return_std=True)  # std of f(x), original y units
        std_pred = np.sqrt(std_f ** 2 + self._noise_var)    # full predictive std
        z_lo = norm.ppf(self.q_lower)   # ≈ -1.645
        z_hi = norm.ppf(self.q_upper)   # ≈  1.645
        lo = mu + z_lo * std_pred
        hi = mu + z_hi * std_pred
        return np.minimum(lo, hi), np.maximum(lo, hi)


def _train_gpr_fixed_data(*, X_train, y_train, seed, n_ens, q_lower, q_upper):
    """Train one GPR on the fixed data and replicate n_ens times -> sigma2=0 exactly.

    Params copied from the old LP-based SVM in src/experiments/(NN)vs(QRF)vs(SVM.py
    (_train_svm / _solve_lp_quantile), which the supervisor identified as structurally
    equivalent to GPR (both are kernel mean regressors, not true SVQR):
      gamma=1.0 in sklearn convention -> length_scale = 1/sqrt(2*gamma) = 1/sqrt(2) ≈ 0.707
      c3 = C = 1.0 (LP regularisation on kernel weights) -> alpha = 1.0 in GPR

    Kernel : RBF(length_scale=0.707), fixed — no optimisation.
    Alpha  : 1.0 — matches LP regularisation strength c3=1.0.
    y scale: normalize_y=True handles y centering/scaling internally;
             gpr.predict() returns values in original y units directly.
    Deterministic given fixed data -> sigma2=0."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF
    from sklearn.preprocessing import StandardScaler

    scaler_X = StandardScaler()
    X_s = scaler_X.fit_transform(X_train)

    # gamma=1.0 (old SVM default) -> length_scale = 1/sqrt(2*1.0)
    length_scale = 1.0 / np.sqrt(2.0 * 1.0)
    kernel = RBF(length_scale=length_scale)
    gpr = GaussianProcessRegressor(
        kernel=kernel,
        optimizer=None,       # kernel fixed — matching old SVM's fixed gamma
        alpha=0.5,            # tuned (val-set sweep)
        normalize_y=True,
        random_state=seed)
    gpr.fit(X_s, y_train)

    member = GPRMember(gpr, scaler_X, q_lower, q_upper)
    return [member] * n_ens   # identical copies -> sigma2 = 0


class NGBMember:
    """NGBoost member: PI from predicted Normal distribution (loc, scale).
    Deterministic given fixed data and fixed random_state -> sigma2=0."""
    def __init__(self, ngb, scaler_X, q_lower, q_upper):
        self.ngb      = ngb
        self.scaler_X = scaler_X
        self.q_lower  = q_lower
        self.q_upper  = q_upper

    def predict_quantiles(self, X):
        X_s  = self.scaler_X.transform(X)
        dist = self.ngb.pred_dist(X_s)
        lo   = dist.ppf(self.q_lower)
        hi   = dist.ppf(self.q_upper)
        return np.minimum(lo, hi), np.maximum(lo, hi)


def _train_ngb_fixed_data(*, X_train, y_train, seed, n_ens, q_lower, q_upper,
                          n_estimators=200, learning_rate=0.01, max_depth=2,
                          min_samples_leaf=10):
    """Train one NGBoost on the fixed data and replicate n_ens times -> sigma2=0.
    NGBoost with Normal distribution outputs loc (mean) and scale (std) per point.
    PI = Normal.ppf(tau). Deterministic with fixed seed -> sigma2=0 (same as SVM/GPR)."""
    from ngboost import NGBRegressor
    from ngboost.distns import Normal
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor

    scaler_X = StandardScaler()
    X_s = scaler_X.fit_transform(X_train)

    members = []
    for m in range(n_ens):
        ngb = NGBRegressor(
            Dist=Normal,
            Base=DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=seed + m),
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            minibatch_frac=1,
            natural_gradient=True,
            verbose=False,
            random_state=seed + m,
        )
        ngb.fit(X_s, y_train)
        members.append(NGBMember(ngb, scaler_X, q_lower, q_upper))
    return members


def _train_nn_fixed_data(*, X_train, y_train, seed, n_ens, q_lower, q_upper,
                         hidden_sizes, dropout, epochs, batch_size, lr,
                         weight_decay, standardize):
    """Train n_ens NN members on the SAME data with different random seeds."""
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    def pinball_loss(preds, target, q):
        diff = target - preds
        return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))

    scaler = StandardScaler() if standardize else experiment._IdentityScaler()
    X_s = scaler.fit_transform(X_train)

    members = []
    for m in range(n_ens):
        print(f"  [NN] Training member {m+1}/{n_ens}...")
        torch.manual_seed(seed + m)
        np.random.seed(seed + m)

        layers = []
        d = X_s.shape[1]
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            d = h
        layers.append(nn.Linear(d, 2))
        model = nn.Sequential(*layers)

        X_t = torch.tensor(X_s, dtype=torch.float32)
        y_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
        dl = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        model.train()
        for _ in range(epochs):
            for xb, yb in dl:
                opt.zero_grad()
                out = model(xb)
                loss = (pinball_loss(out[:, 0], yb.squeeze(1), q_lower) +
                        pinball_loss(out[:, 1], yb.squeeze(1), q_upper))
                loss.backward()
                opt.step()

        members.append(experiment.NNMember(scaler=scaler, model=model))
    return members


# ---------------------------------------------------------------------------
# Predict + metrics
# ---------------------------------------------------------------------------

def _predict_bounds(*, method, members, X_eval, q_lower, q_upper):
    if method == "nn_qnt":
        pred = experiment.predict_nn_ensemble(members, X_eval)
        return pred.lower_members, pred.upper_members
    lower, upper = [], []
    for mem in members:
        if method == "qrf_qnt":
            lo, hi = mem.predict_quantiles(X_eval, q_lower=q_lower, q_upper=q_upper)
        else:  # svm_qnt or xgb_qnt
            lo, hi = mem.predict_quantiles(X_eval)
        lower.append(lo)
        upper.append(hi)
    return np.stack(lower), np.stack(upper)


def compute_true_bounds(y_clean, noise_type, q_lower, q_upper):
    if noise_type == "uniform":
        return y_clean + (-1.0 + 2.0 * q_lower), y_clean + (-1.0 + 2.0 * q_upper)
    from scipy.stats import norm
    return y_clean + norm.ppf(q_lower), y_clean + norm.ppf(q_upper)


def calculate_sigma2_model(lo_members, hi_members, k=None, M=None):
    """
    sigma2_model = mean_{i,j} (f_0.05^j(xi) - mean_0.05(xi))^2
                 + mean_{i,j} (f_0.95^j(xi) - mean_0.95(xi))^2
    (mean over runs and test points; lo+hi summed)
    """
    mean_lo = np.mean(lo_members, axis=0)
    mean_hi = np.mean(hi_members, axis=0)
    return float(((lo_members - mean_lo) ** 2).mean()
                 + ((hi_members - mean_hi) ** 2).mean())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_single_method(ax, x_grid, X_test, y_test, stats_grid, method_name,
                       n_train, n_ens, true_lower, true_upper,
                       lower_members, upper_members, show_legend=True):
    ax.scatter(X_test, y_test, alpha=0.3, s=10, color="gray", zorder=0.5)
    if lower_members is not None:
        for i in range(lower_members.shape[0]):
            lbl = "Estimated PI in 10 runs" if i == 0 and show_legend else "_nolegend_"
            ax.plot(x_grid, lower_members[i], color="red", alpha=0.4, lw=1.5, zorder=1.5, label=lbl)
            ax.plot(x_grid, upper_members[i], color="red", alpha=0.4, lw=1.5, zorder=1.5, label="_nolegend_")
    ax.plot(x_grid, true_lower, "k--", lw=1.8,
            label="True PI" if show_legend else "_nolegend_", zorder=1.0)
    ax.plot(x_grid, true_upper, "k--", lw=1.8, zorder=1.0)
    ax.plot(x_grid, stats_grid["mean_lower"], color="blue", lw=3,
            label="Mean PI" if show_legend else "_nolegend_", zorder=2.0)
    ax.plot(x_grid, stats_grid["mean_upper"], color="blue", lw=3, zorder=2.0)
    ax.set_title(f"{method_name}")
    ax.set_xlabel("x", fontsize=20)
    ax.set_ylabel("y", fontsize=20)
    ax.tick_params(labelsize=18)
    if show_legend:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3, fontsize=19, frameon=False)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-6, 6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=50)
    ap.add_argument("--m", type=int, default=10)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out-dir", type=str, default=str(current_file.parent / "results" / "stability_plots"))
    # Defaults from exp8 stage-1 tuning (best_hyperparams_2026_06_22_11_58_38.json):
    ap.add_argument("--svm-gamma", type=float, default=0.7071067811865476)
    ap.add_argument("--svm-C", type=float, default=11.313708498984761)
    ap.add_argument("--rf-n-estimators", type=int, default=30)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=5)
    ap.add_argument("--rf-max-features", type=str, default="sqrt")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y_noisy, _ = experiment.generate_sinc_dataset(seed=args.seed, n_samples=3000, noise='uniform')
    (X_train_full, y_train_full), (_, _), (X_test, y_test), _ = experiment.split_train_cal_test(
        X, y_noisy, seed=args.seed, train_frac=0.6, cal_frac=0.2
    )
    # Draw 100 indices (50 train + 50 val) to match exp8 stage-1 rng sequence,
    # then use first 50 as training data.
    rng = np.random.default_rng(args.seed)
    n_train = int(min(args.n_train, X_train_full.shape[0]))
    idx = rng.choice(X_train_full.shape[0], size=n_train + 50, replace=False)
    tr_idx = idx[:n_train]
    X_train, y_train = X_train_full[tr_idx], y_train_full[tr_idx]

    k = X_test.shape[0]  # 600 test points
    print(f"Dataset: n_train={n_train}, n_test={k}")

    x_grid = np.linspace(-6.0, 6.0, 400)
    X_grid = x_grid.reshape(-1, 1)
    y_grid_clean = np.sinc(x_grid / np.pi)
    true_lo_grid, true_hi_grid = compute_true_bounds(y_grid_clean, 'uniform', 0.05, 0.95)

    # True bounds on the 600 test points (for SumRMSE evaluation)
    y_test_clean = np.sinc(X_test.flatten() / np.pi)
    true_lo_test, true_hi_test = compute_true_bounds(y_test_clean, 'uniform', 0.05, 0.95)

    methods = [("nn_qnt", "NN"), ("qrf_qnt", "QRF"),
               ("svm_qnt", "SVM"), ("xgb_qnt", "XGB"), ("gpr_qnt", "GPR"),
               ("ngb_qnt", "NGB")]
    all_metrics = {}

    for method, label in methods:
        print(f"\n{'='*40}")
        print(f"Training {label}...")

        if method == "nn_qnt":
            # Fixed data, 10 different random seeds -> measures init variance
            # Best tuned (exp8 stage-1): hidden=[512], lr=3e-3, ep=100, bs=32.
            members = _train_nn_fixed_data(
                X_train=X_train, y_train=y_train, seed=args.seed, n_ens=args.m,
                q_lower=0.05, q_upper=0.95, hidden_sizes=[512], dropout=0.0,
                epochs=args.epochs, batch_size=32, lr=3e-3, weight_decay=0.0,
                standardize=True,
            )
        elif method == "qrf_qnt":
            # Fixed data, different random_state -> internal RF bagging variance only
            members = _train_qrf_bootstrap(
                X_train=X_train, y_train=y_train, seed=args.seed, n_ens=args.m,
                n_estimators=args.rf_n_estimators, min_samples_leaf=args.rf_min_samples_leaf,
                max_features=args.rf_max_features,
            )
        elif method == "xgb_qnt":
            # Fixed data, different random seeds -> internal stochastic boosting variance
            members = _train_xgb_fixed_data(
                X_train=X_train, y_train=y_train, seed=args.seed, n_ens=args.m,
                q_lower=0.05, q_upper=0.95,
            )
        elif method == "gpr_qnt":
            # Single GPR replicated n_ens times -> sigma2=0 (deterministic like SVM)
            members = _train_gpr_fixed_data(
                X_train=X_train, y_train=y_train, seed=args.seed, n_ens=args.m,
                q_lower=0.05, q_upper=0.95,
            )
        elif method == "ngb_qnt":
            # Single NGBoost replicated n_ens times -> sigma2=0 (deterministic like SVM/GPR)
            members = _train_ngb_fixed_data(
                X_train=X_train, y_train=y_train, seed=args.seed, n_ens=args.m,
                q_lower=0.05, q_upper=0.95,
            )
        else:
            # SVQR QP (port of epsilon_quantilesvr2.m) -> sigma2=0 exactly
            members = _train_svqr(
                X_train=X_train, y_train=y_train, seed=args.seed, n_ens=args.m,
                q_lower=0.05, q_upper=0.95, gamma=args.svm_gamma, C=args.svm_C,
            )

        # Grid predictions for plotting
        lo_grid, hi_grid = _predict_bounds(
            method=method, members=members, X_eval=X_grid, q_lower=0.05, q_upper=0.95
        )
        stats_grid = _bounds_stats(lo_grid, hi_grid)

        # Test set predictions (600 points): PICP, MPIW, sigma2, SumRMSE
        lo_test, hi_test = _predict_bounds(
            method=method, members=members, X_eval=X_test, q_lower=0.05, q_upper=0.95
        )
        mean_lo_test = np.mean(lo_test, axis=0)
        mean_hi_test = np.mean(hi_test, axis=0)
        picp, _ = picp_mpiw(y_test, mean_lo_test, mean_hi_test)
        sigma2 = calculate_sigma2_model(lo_test, hi_test, k=k, M=args.m)

        # SumofRMSEs on 600 test points (matches exp8 protocol)
        rmse_lo = float(np.sqrt(np.mean((mean_lo_test - true_lo_test) ** 2)))
        rmse_hi = float(np.sqrt(np.mean((mean_hi_test - true_hi_test) ** 2)))
        sum_rmse = (rmse_lo + rmse_hi) / 2.0

        mpiw = float(np.mean(mean_hi_test - mean_lo_test))
        sigma_model = float(np.sqrt(sigma2))
        is_val = interval_score(y_test, mean_lo_test, mean_hi_test, alpha=0.10)
        all_metrics[label] = {
            "method": label,
            "n_train": n_train,
            "n_test": k,
            "SumRMSE": round(sum_rmse, 4),
            "PICP": round(picp, 4),
            "MPIW": round(mpiw, 4),
            "interval_score": round(is_val, 4),
            "sigma2_model": round(sigma2, 6),
            "sigma_model": round(sigma_model, 6),
        }
        print(f"  {label}: SumofRMSEs={sum_rmse:.3f}  PICP={picp:.3f}  "
              f"MPIW={mpiw:.3f}  IS={is_val:.3f}  sigma2_model={sigma2:.4f}  sigma_model={sigma_model:.4f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        plot_single_method(ax, x_grid, X_test, y_test, stats_grid,
                           label, n_train, args.m, true_lo_grid, true_hi_grid, lo_grid, hi_grid)
        fig.tight_layout()
        fig.savefig(out_dir / f"{method}_stability.png", dpi=160)
        plt.close(fig)
        print(f"  Plot saved: {out_dir / f'{method}_stability.png'}")

    # ── Console table ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  TABLE B1 (n_train={n_train}, M={args.m}, seed={args.seed})")
    print(f"{'='*70}")
    print(f"{'Method':<16}  {'SumofRMSEs':>12}  {'PICP':>8}  {'MPIW':>8}  {'IS':>9}  {'sigma2_model':>14}  {'sigma_model':>12}")
    print(f"{'-'*88}")
    for label in ["NN", "QRF", "SVM", "XGB", "GPR", "NGB"]:
        r = all_metrics[label]
        print(f"{label:<16}  {r['SumRMSE']:>12.3f}  {r['PICP']:>8.3f}  "
              f"{r['MPIW']:>8.3f}  {r['interval_score']:>9.3f}  "
              f"{r['sigma2_model']:>14.4f}  {r['sigma_model']:>12.4f}")
    print(f"{'='*88}")
    print(f"\nExpected (paper): NN(0.353,0.587,0.132) QRF(0.264,0.755,0.123) SVM(0.266,0.848,0.00)")
    print(f"XGB, GPR, and NGB are additional baselines (not in paper's Table B1).")

    # ── Save results to disk ─────────────────────────────────────────────────
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # CSV
    results_df = pd.DataFrame([all_metrics[l] for l in ["NN", "QRF", "SVM", "XGB", "GPR", "NGB"]])
    csv_path = out_dir / f"stability_results_{ts}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nResults CSV:  {csv_path}")

    # Text summary
    txt_path = out_dir / f"stability_results_{ts}.txt"
    with open(txt_path, "w") as f:
        f.write(f"Stability results — sinc + Uniform(-1,1) noise\n")
        f.write(f"n_train={n_train}  M={args.m}  seed={args.seed}  n_test={k}\n")
        f.write(f"{'='*88}\n")
        f.write(f"{'Method':<16}  {'SumofRMSEs':>12}  {'PICP':>8}  "
                f"{'MPIW':>8}  {'IS':>9}  {'sigma2_model':>14}  {'sigma_model':>12}\n")
        f.write(f"{'-'*88}\n")
        for label in ["NN", "QRF", "SVM", "XGB", "GPR", "NGB"]:
            r = all_metrics[label]
            f.write(f"{label:<16}  {r['SumRMSE']:>12.3f}  {r['PICP']:>8.3f}  "
                    f"{r['MPIW']:>8.3f}  {r['interval_score']:>9.3f}  "
                    f"{r['sigma2_model']:>14.4f}  {r['sigma_model']:>12.4f}\n")
        f.write(f"{'='*88}\n\n")
        f.write("sigma2_model: mean((f_q^j - mean_q)^2) over runs x test points, "
                "summed over lo+hi heads.\n")
        f.write("sigma_model:  sqrt(sigma2_model) — in original y units.\n")
        f.write("SumofRMSEs:  (RMSE(mean_lo, true_lo) + RMSE(mean_hi, true_hi)) / 2 "
                "on the 600 test points (matches exp8 protocol).\n")
        f.write("PICP: fraction of test points covered by the mean PI.\n")
        f.write("MPIW: mean width of the mean PI on the test set.\n")
        f.write("IS: Interval Score — proper scoring rule (lower is better). "
                "IS = (u-l) + (2/alpha)*max(l-y,0) + (2/alpha)*max(y-u,0).\n")
    print(f"Results TXT:  {txt_path}")
    print(f"Plots saved:  {out_dir.resolve()}/")


if __name__ == "__main__":
    main()