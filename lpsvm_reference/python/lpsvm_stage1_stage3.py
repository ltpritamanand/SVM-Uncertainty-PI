#!/usr/bin/env python3
"""
exp13: LP-SVM (Sparse SVQR, L1-norm) — Stage 1 + Stage 3
=========================================================
Mirrors exp8 pipeline but for LP-SVM only.

Stage 1 — TUNING:
  - sinc + Uniform(-1,1) data, seed=2, 50 train + 50 val.
  - Grid search over gamma (RBF kernel) and C (pinball loss).
  - Score by sum_rmse on the 50 val points.
  - Best params -> best_hyperparams.json.

Stage 3 — EVALUATION:
  - D=10 data seeds x I=1 init seed (LP is deterministic).
  - Fixed val (50) and test (600) sets.
  - Predict on val AND test.  Decompose sigma_opt / sigma_in.
  - sigma_opt = 0 (deterministic LP solver).
  - sigma_in = variance across data seeds.

Usage:
  python lpsvm_stage1_stage3.py               # full pipeline
  python lpsvm_stage1_stage3.py --skip-tune   # reuse saved best_hyperparams
  python lpsvm_stage1_stage3.py --stage 1     # tune only
  python lpsvm_stage1_stage3.py --stage 3     # evaluate only
"""

import argparse
import itertools
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ════════════════════════════════════════════════════════════════════════════
#  CONFIG (same as exp8)
# ════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "results"
OUT_DIR.mkdir(exist_ok=True)
TUNE_DIR   = OUT_DIR / "hyperparam_tuning"
TUNE_DIR.mkdir(exist_ok=True)

X_LOW, X_HIGH         = -6.0, 6.0
NOISE_LOW, NOISE_HIGH = -1.0, 1.0

Q_LOWER, Q_UPPER = 0.05, 0.95

N_TRAIN = 50
N_VAL   = 50
N_TEST  = 600

TUNE_SEED    = 2
VAL_SEED     = 7777
TEST_SEED    = 9999
N_DATA_SEEDS = 10
N_INIT_SEEDS = 1       # LP-SVM is deterministic -> I=1

# Tuning grid: gamma, C (pinball), c3 (L1 penalty)
LPSVM_GRID = {
    "gamma": [2.0 ** (k / 2.0) for k in range(-20, 21)],   # 41 values: 2^-10 .. 2^10
    "C":     [2.0 ** (k / 2.0) for k in range(-20, 21)],   # 41 values
    "c3":    [1],   # 41 values
}

# ════════════════════════════════════════════════════════════════════════════
#  LP-SVM SOLVER (port of quantileLPONENORMTSVR12.m — paper Equation 9)
# ════════════════════════════════════════════════════════════════════════════

def rbf_kernel(X, Y=None, gamma=1.0):
    if Y is None:
        Y = X
    return np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))


def solve_lpsvm_quantile(X_train, y_train, X_eval, gamma, c1, tau, c3=1.0):
    """Sparse SVQR via LP.  Returns predictions on eval set."""
    n = X_train.shape[0]

    K = rbf_kernel(X_train, gamma=gamma)
    H = np.hstack([K, np.ones((n, 1))])
    n_rp = n + 1

    f = np.concatenate([
        c3 * np.ones(n_rp),
        c3 * np.ones(n_rp),
        c1 * tau * np.ones(n),
        c1 * (1 - tau) * np.ones(n),
    ])

    I_n = np.eye(n)
    Z_n = np.zeros((n, n))
    A_ub = np.vstack([
        np.hstack([-H,  H, -I_n, Z_n]),
        np.hstack([ H, -H, Z_n, -I_n]),
    ])
    b_ub = np.concatenate([-y_train, y_train])

    bounds = [(0, None)] * len(f)
    result = linprog(f, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                     method='highs', options={'presolve': True, 'disp': False})

    if not result.success:
        return np.full(X_eval.shape[0], np.nan)

    x = result.x
    u = x[:n_rp] - x[n_rp:2 * n_rp]

    K_eval = rbf_kernel(X_eval, X_train, gamma=gamma)
    H_eval = np.hstack([K_eval, np.ones((X_eval.shape[0], 1))])
    return H_eval @ u


def train_lpsvm(X_tr_s, y_tr_s, X_eval_s, scaler_y, init_seed, gamma, C, c3=1.0):
    """Train LP-SVM (both quantiles), return lo/hi in original scale."""
    lo_s = solve_lpsvm_quantile(X_tr_s, y_tr_s, X_eval_s, gamma, C, Q_LOWER, c3)
    hi_s = solve_lpsvm_quantile(X_tr_s, y_tr_s, X_eval_s, gamma, C, Q_UPPER, c3)
    lo = scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
    return np.minimum(lo, hi), np.maximum(lo, hi)


# ════════════════════════════════════════════════════════════════════════════
#  DATA + METRICS (same as exp8)
# ════════════════════════════════════════════════════════════════════════════

def generate_sinc_dataset(seed, n_samples):
    """Matches shared/src_utils generate_sinc_dataset (noise='uniform')."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(X_LOW, X_HIGH, size=(n_samples, 1))
    y_clean = np.sinc(X[:, 0] / np.pi)
    eps = rng.uniform(NOISE_LOW, NOISE_HIGH, size=n_samples)
    y_noisy = y_clean + eps
    return X, y_noisy, y_clean


def split_pool(X, y, seed, train_frac=0.6, cal_frac=0.2):
    """Matches shared/src_utils split_train_cal_test, returns train pool only."""
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(np.floor(train_frac * n))
    return X[perm[:n_train]], y[perm[:n_train]]


def generate_data(seed, n):
    """Stage 3 local data (same as exp8 generate_data)."""
    rng = np.random.default_rng(seed)
    x   = rng.uniform(X_LOW, X_HIGH, size=n).astype(np.float32)
    eps = rng.uniform(NOISE_LOW, NOISE_HIGH, size=n).astype(np.float32)
    y   = (np.sinc(x / np.pi) + eps).astype(np.float32)
    return x.reshape(-1, 1), y


def compute_true_bounds(X_raw):
    y_clean = np.sinc(X_raw.flatten() / np.pi)
    return y_clean - 0.9, y_clean + 0.9


def picp_mpiw(y_true, lo, hi):
    return (float(np.mean((y_true >= lo) & (y_true <= hi))),
            float(np.mean(hi - lo)))


def interval_score(y_true, lo, hi, alpha=0.10):
    """Interval Score (IS) for a (1-alpha)*100% prediction interval.

    IS = (u - l) + (2/alpha) * max(l - y, 0) + (2/alpha) * max(y - u, 0)

    Lower is better. Proper scoring rule — jointly evaluates calibration and sharpness.
    """
    width = hi - lo
    below = np.maximum(lo - y_true, 0.0)
    above = np.maximum(y_true - hi, 0.0)
    penalty = (2.0 / alpha) * (below + above)
    return float(np.mean(width + penalty))


def sum_rmse(true_lo, true_hi, lo, hi):
    rmse_lo = float(np.sqrt(np.mean((lo - true_lo) ** 2)))
    rmse_hi = float(np.sqrt(np.mean((hi - true_hi) ** 2)))
    return (rmse_lo + rmse_hi) / 2.0


def decompose_raw(lo_arr, hi_arr):
    """sigma2_opt = E_d[Var_i[f]]; sigma2_in = Var_d[E_i[f]]; summed lo+hi."""
    mu_lo_d = lo_arr.mean(axis=1)
    mu_hi_d = hi_arr.mean(axis=1)
    I = lo_arr.shape[1]
    v_opt = 0.0 if I < 2 else (lo_arr.var(axis=1, ddof=1).mean()
                               + hi_arr.var(axis=1, ddof=1).mean())
    D = lo_arr.shape[0]
    v_in  = 0.0 if D < 2 else (mu_lo_d.var(axis=0, ddof=1).mean()
                               + mu_hi_d.var(axis=0, ddof=1).mean())
    return float(v_opt), float(v_in)


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 1: Hyperparameter Tuning
# ════════════════════════════════════════════════════════════════════════════

def stage1_tune():
    print("\n" + "=" * 70)
    print("  STAGE 1: LP-SVM Hyperparameter Tuning (50 train + 50 val)")
    print("=" * 70)

    # Same data pipeline as exp8 stage 1
    X, y_noisy, _ = generate_sinc_dataset(seed=TUNE_SEED, n_samples=3000)
    X_pool, y_pool = split_pool(X, y_noisy, seed=TUNE_SEED)

    rng = np.random.default_rng(TUNE_SEED)
    idx = rng.choice(X_pool.shape[0], size=N_TRAIN + N_VAL, replace=False)
    X_tr, y_tr = X_pool[idx[:N_TRAIN]], y_pool[idx[:N_TRAIN]]
    X_va, y_va = X_pool[idx[N_TRAIN:]], y_pool[idx[N_TRAIN:]]

    # True quantile bounds on val
    y_clean_val = np.sinc(X_va.flatten() / np.pi)
    true_lo_val = y_clean_val + (-1.0 + 2.0 * Q_LOWER)   # y_clean - 0.9
    true_hi_val = y_clean_val + (-1.0 + 2.0 * Q_UPPER)    # y_clean + 0.9

    # Scale
    sc_X = StandardScaler().fit(X_tr)
    sc_y = StandardScaler().fit(y_tr.reshape(-1, 1))
    X_tr_s = sc_X.transform(X_tr).astype(np.float64)
    X_va_s = sc_X.transform(X_va).astype(np.float64)
    y_tr_s = sc_y.transform(y_tr.reshape(-1, 1)).flatten().astype(np.float64)

    configs = list(itertools.product(
        LPSVM_GRID["gamma"], LPSVM_GRID["C"], LPSVM_GRID["c3"]))
    ng, nc, nc3 = len(LPSVM_GRID["gamma"]), len(LPSVM_GRID["C"]), len(LPSVM_GRID["c3"])
    print(f"  Grid: {ng} x {nc} x {nc3} = {len(configs)} configs")

    rows = []
    best_sr = float("inf")
    best_cfg = None
    t0 = time.time()

    for i, (gamma, C, c3) in enumerate(configs, 1):
        try:
            lo, hi = train_lpsvm(X_tr_s, y_tr_s, X_va_s, sc_y, 0, gamma, C, c3)
            if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
                sr = picp_val = score = float("nan")
                status = "nan_pred"
            else:
                rmse_lo   = float(np.sqrt(np.mean((lo - true_lo_val) ** 2)))
                rmse_hi   = float(np.sqrt(np.mean((hi - true_hi_val) ** 2)))
                sr        = (rmse_lo + rmse_hi) / 2.0
                picp_val  = float(np.mean((y_va >= lo) & (y_va <= hi)))
                # penalise deviation from target coverage + sum_rmse
                score     = abs(picp_val - 0.90) * 10.0 + sr
                status    = "ok"
        except Exception as e:
            sr = picp_val = score = float("nan")
            status = f"err: {type(e).__name__}"

        rows.append({"gamma": gamma, "C": C, "c3": c3,
                     "sum_rmse": sr, "picp_val": picp_val,
                     "score": score, "status": status})

        if score == score and score < best_sr:
            best_sr  = score
            best_cfg = {"gamma": gamma, "C": C, "c3": c3}

        if i % 5000 == 0 or i == len(configs):
            elapsed = time.time() - t0
            print(f"    [{i:>6}/{len(configs)}]  best_score={best_sr:.4f}  ({elapsed:.0f}s)")

    print(f"\n  BEST LP-SVM: score={best_sr:.4f}  "
          f"gamma={best_cfg['gamma']:.6g}  C={best_cfg['C']:.6g}  c3={best_cfg['c3']:.6g}")

    # Save
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    pd.DataFrame(rows).to_csv(TUNE_DIR / f"lpsvm_tuning_full_{ts}.csv", index=False)

    best = {"LPSVM": {"config": best_cfg, "sum_rmse": round(best_sr, 5)}}
    best_path = TUNE_DIR / f"best_hyperparams_{ts}.json"
    with open(best_path, "w") as f:
        json.dump({"n_train": N_TRAIN, "n_val": N_VAL,
                   "seed": TUNE_SEED, "best": best}, f, indent=2)
    print(f"  -> {best_path}")
    return best


def load_latest_best():
    files = sorted(TUNE_DIR.glob("best_hyperparams_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f).get("best")


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 3: D x I Evaluation with Decomposition
# ════════════════════════════════════════════════════════════════════════════

def run_stage3(best):
    print("\n" + "=" * 70)
    print(f"  STAGE 3: LP-SVM Evaluation (D={N_DATA_SEEDS}, I={N_INIT_SEEDS})")
    print("=" * 70)

    cfg = best["LPSVM"]["config"]
    gamma, C, c3 = cfg["gamma"], cfg["C"], cfg.get("c3", 1.0)
    print(f"  Best params: gamma={gamma:.6g}  C={C:.6g}  c3={c3:.6g}")

    X_val_raw,  y_val  = generate_data(seed=VAL_SEED,  n=N_VAL)
    X_test_raw, y_test = generate_data(seed=TEST_SEED, n=N_TEST)
    true_lo_v, true_hi_v = compute_true_bounds(X_val_raw)
    true_lo_t, true_hi_t = compute_true_bounds(X_test_raw)

    preds_lo_v = np.zeros((N_DATA_SEEDS, N_INIT_SEEDS, N_VAL),  dtype=np.float64)
    preds_hi_v = np.zeros((N_DATA_SEEDS, N_INIT_SEEDS, N_VAL),  dtype=np.float64)
    preds_lo_t = np.zeros((N_DATA_SEEDS, N_INIT_SEEDS, N_TEST), dtype=np.float64)
    preds_hi_t = np.zeros((N_DATA_SEEDS, N_INIT_SEEDS, N_TEST), dtype=np.float64)

    t0 = time.time()
    for d in range(N_DATA_SEEDS):
        X_tr_raw, y_tr_raw = generate_data(seed=d * 10000 + N_TRAIN, n=N_TRAIN)
        sc_X = StandardScaler().fit(X_tr_raw)
        sc_y = StandardScaler().fit(y_tr_raw.reshape(-1, 1))
        X_tr_s = sc_X.transform(X_tr_raw).astype(np.float64)
        X_va_s = sc_X.transform(X_val_raw).astype(np.float64)
        X_te_s = sc_X.transform(X_test_raw).astype(np.float64)
        y_tr_s = sc_y.transform(y_tr_raw.reshape(-1, 1)).flatten().astype(np.float64)

        for i in range(N_INIT_SEEDS):
            lo_v, hi_v = train_lpsvm(X_tr_s, y_tr_s, X_va_s, sc_y, i, gamma, C, c3)
            lo_t, hi_t = train_lpsvm(X_tr_s, y_tr_s, X_te_s, sc_y, i, gamma, C, c3)
            preds_lo_v[d, i] = lo_v; preds_hi_v[d, i] = hi_v
            preds_lo_t[d, i] = lo_t; preds_hi_t[d, i] = hi_t

        elapsed = time.time() - t0
        print(f"    data_seed {d+1}/{N_DATA_SEEDS} done  ({elapsed:.1f}s)")

    # Decompose
    v_opt_v, v_in_v = decompose_raw(preds_lo_v, preds_hi_v)
    v_opt_t, v_in_t = decompose_raw(preds_lo_t, preds_hi_t)

    # sum_rmse: per-data-seed, init-averaged PI, mean over D seeds
    per_d_sr_v, per_d_sr_t = [], []
    for d in range(N_DATA_SEEDS):
        mean_lo_v_d = preds_lo_v[d].mean(axis=0)
        mean_hi_v_d = preds_hi_v[d].mean(axis=0)
        mean_lo_t_d = preds_lo_t[d].mean(axis=0)
        mean_hi_t_d = preds_hi_t[d].mean(axis=0)
        per_d_sr_v.append(sum_rmse(true_lo_v, true_hi_v, mean_lo_v_d, mean_hi_v_d))
        per_d_sr_t.append(sum_rmse(true_lo_t, true_hi_t, mean_lo_t_d, mean_hi_t_d))
    sr_v = float(np.mean(per_d_sr_v))
    sr_t = float(np.mean(per_d_sr_t))

    # PICP / MPIW on test (grand-mean PI)
    mean_lo_t = preds_lo_t.mean(axis=(0, 1))
    mean_hi_t = preds_hi_t.mean(axis=(0, 1))
    p_t, m_t = picp_mpiw(y_test, mean_lo_t, mean_hi_t)

    # Same for val
    mean_lo_v = preds_lo_v.mean(axis=(0, 1))
    mean_hi_v = preds_hi_v.mean(axis=(0, 1))
    p_v, m_v = picp_mpiw(y_val, mean_lo_v, mean_hi_v)

    alpha = 1.0 - (Q_UPPER - Q_LOWER)
    is_v = interval_score(y_val, mean_lo_v, mean_hi_v, alpha=alpha)
    is_t = interval_score(y_test, mean_lo_t, mean_hi_t, alpha=alpha)

    results = {
        "model": "LPSVM",
        "config": json.dumps(cfg),
        "sum_rmse_val":       round(sr_v, 4),
        "sum_rmse_test":      round(sr_t, 4),
        "interval_score_val": round(is_v, 4),
        "interval_score_test":round(is_t, 4),
        "sigma_opt_val":      round(np.sqrt(v_opt_v), 4),
        "sigma_opt_test":     round(np.sqrt(v_opt_t), 4),
        "sigma_in_val":       round(np.sqrt(v_in_v), 4),
        "sigma_in_test":      round(np.sqrt(v_in_t), 4),
        "picp_val":           round(p_v, 4),
        "picp_test":          round(p_t, 4),
        "mpiw_val":           round(m_v, 4),
        "mpiw_test":          round(m_t, 4),
    }

    print(f"\n  LP-SVM Results:")
    print(f"    sum_rmse  : val={sr_v:.4f}  test={sr_t:.4f}")
    print(f"    IS        : val={is_v:.3f}  test={is_t:.3f}")
    print(f"    sigma_opt : val={np.sqrt(v_opt_v):.4f}  test={np.sqrt(v_opt_t):.4f}")
    print(f"    sigma_in  : val={np.sqrt(v_in_v):.4f}  test={np.sqrt(v_in_t):.4f}")
    print(f"    PICP      : val={p_v:.4f}  test={p_t:.4f}")
    print(f"    MPIW      : val={m_v:.4f}  test={m_t:.4f}")

    return results


# ════════════════════════════════════════════════════════════════════════════
#  Plots + Save
# ════════════════════════════════════════════════════════════════════════════

def plot_results(results, ts):
    plot_dir = OUT_DIR / "plots"
    plot_dir.mkdir(exist_ok=True)

    metrics = ["sum_rmse", "interval_score", "sigma_opt", "sigma_in"]
    labels  = {"sum_rmse": "sum RMSE", "interval_score": "Interval Score",
               "sigma_opt": r"$\sigma_{opt}$", "sigma_in": r"$\sigma_{in}$"}
    val_vals  = [results[f"{m}_val"]  for m in metrics]
    test_vals = [results[f"{m}_test"] for m in metrics]

    # Bar chart: val vs test for each metric
    fig, ax = plt.subplots(figsize=(8, 4.5))
    xpos = np.arange(len(metrics))
    width = 0.35
    bars_v = ax.bar(xpos - width/2, val_vals,  width, label="val",  color="C0")
    bars_t = ax.bar(xpos + width/2, test_vals, width, label="test", color="C3")
    ax.set_xticks(xpos)
    ax.set_xticklabels([labels[m] for m in metrics], fontsize=12)
    ax.set_ylabel("Value", fontsize=13)
    ax.set_title("LP-SVM (Sparse SVQR): val vs test", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=11)
    for bars in [bars_v, bars_t]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h,
                        f"{h:.4f}", ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    fig.savefig(plot_dir / f"lpsvm_val_vs_test_{ts}.png", dpi=160)
    plt.close(fig)

    # PICP + MPIW + IS bar chart
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric, title in zip(axes,
                                  ["picp", "mpiw", "interval_score"],
                                  ["PICP", "MPIW", "Interval Score"]):
        v = results[f"{metric}_val"]
        t = results[f"{metric}_test"]
        bars = ax.bar(["val", "test"], [v, t], color=["C0", "C3"], width=0.5)
        ax.set_title(f"LP-SVM: {title}", fontsize=13)
        ax.set_ylabel(title, fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, [v, t]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{val:.4f}", ha='center', va='bottom', fontsize=10)
    fig.tight_layout()
    fig.savefig(plot_dir / f"lpsvm_picp_mpiw_is_{ts}.png", dpi=160)
    plt.close(fig)

    print(f"  Plots -> {plot_dir}/")


def save_results(results, ts):
    csv_path = OUT_DIR / f"lpsvm_stage3_results_{ts}.csv"
    pd.DataFrame([results]).to_csv(csv_path, index=False)
    print(f"  CSV -> {csv_path}")

    cfg = json.loads(results["config"])
    txt_path = OUT_DIR / f"lpsvm_stage3_results_{ts}.txt"
    with open(txt_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("  LP-SVM (Sparse SVQR) — Stage 3 Results\n")
        f.write("=" * 60 + "\n\n")
        f.write("Config\n" + "-" * 40 + "\n")
        f.write(f"  gamma         : {cfg['gamma']}\n")
        f.write(f"  C             : {cfg['C']}\n")
        f.write(f"  c3 (L1 coeff) : {cfg.get('c3', 1.0)}\n")
        f.write(f"  q_lower/upper : {Q_LOWER} / {Q_UPPER}\n")
        f.write(f"  D (data seeds): {N_DATA_SEEDS}\n")
        f.write(f"  I (init seeds): {N_INIT_SEEDS}\n\n")
        f.write("Metrics\n" + "-" * 40 + "\n")
        f.write(f"  {'Metric':<16}  {'Val':>10}  {'Test':>10}\n")
        f.write(f"  {'-'*40}\n")
        for m in ["sum_rmse", "interval_score", "sigma_opt", "sigma_in", "picp", "mpiw"]:
            f.write(f"  {m:<16}  {results[f'{m}_val']:>10.4f}  "
                    f"{results[f'{m}_test']:>10.4f}\n")
        f.write(f"\n  sigma_opt = 0 because LP solver is deterministic.\n")
        f.write(f"  sigma_in  = variance across {N_DATA_SEEDS} data seeds.\n\n")
        f.write(f"Timestamp: {datetime.now()}\n")
    print(f"  TXT -> {txt_path}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tune", action="store_true")
    ap.add_argument("--stage", type=int, default=0,
                    help="0=all, 1=tune only, 3=evaluate only")
    args = ap.parse_args()

    if args.skip_tune or args.stage == 3:
        best = load_latest_best()
        if best is None:
            print("No best_hyperparams found; running stage 1 first.")
            best = stage1_tune()
        else:
            print(f"\nLoaded best_hyperparams from {TUNE_DIR}")
            print(f"  LPSVM: {best['LPSVM']['config']}")
    else:
        best = stage1_tune()

    if args.stage == 1:
        return

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results = run_stage3(best)
    plot_results(results, ts)
    save_results(results, ts)

    print(f"\n{'='*60}")
    print(f"  Done!  Outputs -> {OUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
