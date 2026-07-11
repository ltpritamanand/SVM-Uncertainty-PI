#!/usr/bin/env python3
"""
exp8 NN-only, N=100: ablation study for NN with 100 training points.
================================================================================
Identical pipeline to ablation_study_model_size.py but:
  - N_TRAIN = 100  (was 50)
  - Only NN model throughout (stage 1 tune, stage 2 ablation, stage 3 compare)
  - Output goes to results_nn_n100/

Usage:
  python nn_n_100.py                 # full pipeline
  python nn_n_100.py --skip-tune     # reuse latest best_hyperparams.json
  python nn_n_100.py --stage 2       # only ablation (needs --skip-tune)
"""

import argparse
import importlib.util
import json
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


# ════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ════════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DIR    = SCRIPT_DIR / "results_nn_n100"
OUT_DIR.mkdir(exist_ok=True)
TUNE_DIR   = OUT_DIR / "hyperparam_tuning"
TUNE_DIR.mkdir(exist_ok=True)

X_LOW, X_HIGH         = -6.0, 6.0
NOISE_LOW, NOISE_HIGH = -1.0, 1.0

Q_LOWER, Q_UPPER = 0.05, 0.95

N_TRAIN = 100
N_VAL   = 50
N_TEST  = 600

TUNE_SEED  = 2       # fixed seed for stage 1 train/val draw
VAL_SEED   = 7777    # fixed seed for stage 2 val set
TEST_SEED  = 9999    # fixed seed for shared test set
N_DATA_SEEDS = 10
N_INIT_SEEDS = 10

# ── Ablation axes (NN only) ─────────────────────────────────────────────────
ABLATION_SPECS = [
    ("NN", "hidden_sizes_w", [16, 32, 64, 128, 256, 512, 1024],
        "Hidden nodes (neurons)", "nn"),
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ════════════════════════════════════════════════════════════════════════════
#  Stage 1: tuning — NN only
# ════════════════════════════════════════════════════════════════════════════
import itertools

_STAB_PATH = (SCRIPT_DIR.parent
              / "figure1_stability_intro" / "(NN)vs(QRF)vs(SVM.py")
_spec = importlib.util.spec_from_file_location("exp2_stability_main", _STAB_PATH)
stab = importlib.util.module_from_spec(_spec)
_SHARED_UTILS = SCRIPT_DIR.parent / "shared" / "src_utils"
if str(_SHARED_UTILS) not in sys.path:
    sys.path.insert(0, str(_SHARED_UTILS))
_spec.loader.exec_module(stab)

import run_synthetic_tube_quantile_ensembles as experiment


NN_GRID = {
    "hidden_sizes": [[32], [64], [128], [256], [512], [1024]],
    "lr":           [1e-3, 3e-3],
    "epochs":       [100, 200, 300],
    "batch_size":   [32, 64, 128, 256],
}


def _grid(d):
    keys = list(d.keys())
    for vals in itertools.product(*[d[k] for k in keys]):
        yield dict(zip(keys, vals))


def _stab_train_nn(X_tr, y_tr, seed, M, cfg):
    return stab._train_nn_fixed_data(
        X_train=X_tr, y_train=y_tr, seed=seed, n_ens=M,
        q_lower=Q_LOWER, q_upper=Q_UPPER,
        hidden_sizes=cfg["hidden_sizes"], dropout=0.0,
        epochs=cfg["epochs"], batch_size=cfg["batch_size"], lr=cfg["lr"],
        weight_decay=0.0, standardize=True,
    )


TUNE_MODELS = [
    ("nn_qnt", "NN", NN_GRID, _stab_train_nn),
]


def _tune_sum_rmse(true_lo, true_hi, lo, hi):
    rmse_lo = float(np.sqrt(np.mean((lo - true_lo) ** 2)))
    rmse_hi = float(np.sqrt(np.mean((hi - true_hi) ** 2)))
    return (rmse_lo + rmse_hi) / 2.0, rmse_lo, rmse_hi


def _tune_eval(X_val, true_lo_val, true_hi_val, method, members):
    lo_mem, hi_mem = stab._predict_bounds(
        method=method, members=members, X_eval=X_val,
        q_lower=Q_LOWER, q_upper=Q_UPPER,
    )
    mean_lo = np.mean(lo_mem, axis=0)
    mean_hi = np.mean(hi_mem, axis=0)
    return _tune_sum_rmse(true_lo_val, true_hi_val, mean_lo, mean_hi)


def stage1_tune(verbose=True):
    print("\n" + "=" * 70)
    print(f"  STAGE 1: hyperparameter tuning  ({N_TRAIN} train + {N_VAL} val, M=10)")
    print("=" * 70)

    X, y_noisy, _ = experiment.generate_sinc_dataset(
        seed=TUNE_SEED, n_samples=3000, noise='uniform')
    (X_pool, y_pool), _, _, _ = experiment.split_train_cal_test(
        X, y_noisy, seed=TUNE_SEED, train_frac=0.6, cal_frac=0.2)
    rng = np.random.default_rng(TUNE_SEED)
    need = N_TRAIN + N_VAL
    idx = rng.choice(X_pool.shape[0], size=need, replace=False)
    tr_idx, val_idx = idx[:N_TRAIN], idx[N_TRAIN:]
    X_tr, y_tr = X_pool[tr_idx], y_pool[tr_idx]
    X_va, y_va = X_pool[val_idx], y_pool[val_idx]

    y_clean_val = np.sinc(X_va.flatten() / np.pi)
    true_lo_val, true_hi_val = stab.compute_true_bounds(
        y_clean_val, 'uniform', Q_LOWER, Q_UPPER)

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    all_rows, best = [], {}

    for method, label, grid, trainer in TUNE_MODELS:
        configs = list(_grid(grid))
        print(f"\n  Tuning {label}: {len(configs)} configs")
        rows = []
        for i, cfg in enumerate(configs, 1):
            t0 = time.time()
            try:
                members = trainer(X_tr, y_tr, TUNE_SEED, N_INIT_SEEDS, cfg)
                sr, rlo, rhi = _tune_eval(
                    X_va, true_lo_val, true_hi_val, method, members)
                status = "ok"
            except Exception as e:
                sr = rlo = rhi = float("nan")
                status = f"err: {type(e).__name__}"
            dt = time.time() - t0
            row = {"model": label, "config": json.dumps(cfg, default=str),
                   "sum_rmse": round(sr, 5) if sr == sr else None,
                   "rmse_lo": round(rlo, 5) if rlo == rlo else None,
                   "rmse_hi": round(rhi, 5) if rhi == rhi else None,
                   "time_s": round(dt, 2), "status": status}
            rows.append(row); all_rows.append(row)
            if verbose:
                print(f"    [{i:>3}/{len(configs)}] sr={row['sum_rmse']}  "
                      f"({dt:.1f}s)  {cfg}")
        ok = [r for r in rows if r["sum_rmse"] is not None]
        if ok:
            br = min(ok, key=lambda r: r["sum_rmse"])
            best[label] = {"config":   json.loads(br["config"]),
                           "sum_rmse": br["sum_rmse"],
                           "rmse_lo":  br["rmse_lo"],
                           "rmse_hi":  br["rmse_hi"]}
            print(f"    BEST {label}: sr={br['sum_rmse']}  cfg={br['config']}")

    pd.DataFrame(all_rows).to_csv(TUNE_DIR / f"tuning_full_{ts}.csv", index=False)
    best_path = TUNE_DIR / f"best_hyperparams_{ts}.json"
    with open(best_path, "w") as f:
        json.dump({"n_train": N_TRAIN, "n_val": N_VAL, "M": N_INIT_SEEDS,
                   "seed": TUNE_SEED, "best": best}, f, indent=2)
    print(f"\n  -> best_hyperparams written: {best_path}")
    return best


def load_latest_best():
    files = sorted(TUNE_DIR.glob("best_hyperparams_*.json"))
    if not files:
        return None
    with open(files[-1]) as f:
        return json.load(f).get("best")


# ════════════════════════════════════════════════════════════════════════════
#  Stage 2 + 3 helpers
# ════════════════════════════════════════════════════════════════════════════

def generate_data(seed: int, n: int):
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
    """sigma2_opt = E_d[Var_i[f]]; summed lo+hi. Shape: (D, I, N)."""
    I = lo_arr.shape[1]
    v_opt = 0.0 if I < 2 else (lo_arr.var(axis=1, ddof=1).mean()
                               + hi_arr.var(axis=1, ddof=1).mean())
    return float(v_opt)


def _pinball(pred, target, q):
    diff = target - pred
    return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))


# ─── NN ─────────────────────────────────────────────────────────────────────
class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_sizes):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(d, h), nn.ReLU()]; d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)


def train_nn(X_tr_s, y_tr_s, X_eval_s, scaler_y, init_seed,
             hidden_sizes, lr, epochs, batch_size):
    torch.manual_seed(init_seed); np.random.seed(init_seed)
    model = _MLP(in_dim=X_tr_s.shape[1], hidden_sizes=hidden_sizes).to(DEVICE)
    X_t = torch.tensor(X_tr_s, dtype=torch.float32).to(DEVICE)
    y_t = torch.tensor(y_tr_s.reshape(-1, 1), dtype=torch.float32).to(DEVICE)
    dl  = DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size, shuffle=True)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            out  = model(xb)
            loss = (_pinball(out[:, 0], yb.squeeze(1), Q_LOWER) +
                    _pinball(out[:, 1], yb.squeeze(1), Q_UPPER))
            loss.backward(); opt.step()
    model.eval()
    Xe = torch.tensor(X_eval_s, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(Xe).cpu().numpy()
    lo_s = np.minimum(out[:, 0], out[:, 1])
    hi_s = np.maximum(out[:, 0], out[:, 1])
    lo = scaler_y.inverse_transform(lo_s.reshape(-1, 1)).flatten()
    hi = scaler_y.inverse_transform(hi_s.reshape(-1, 1)).flatten()
    return lo, hi


# ════════════════════════════════════════════════════════════════════════════
#  Stages 2 + 3: fixed train/val (stage-1 seed), I init seeds, no D loop
# ════════════════════════════════════════════════════════════════════════════


def _make_predict_fn(family, axis_key, axis_value, base_cfg):
    if family == "NN":
        def pf(X_tr_s, y_tr_s, X_eval_s, sc_y, init,
               _w=axis_value, _lr=base_cfg["lr"],
               _ep=base_cfg["epochs"], _bs=base_cfg["batch_size"]):
            return train_nn(X_tr_s, y_tr_s, X_eval_s, sc_y, init,
                            hidden_sizes=[_w], lr=_lr,
                            epochs=_ep, batch_size=_bs)
        return pf
    raise ValueError(f"unknown family: {family}")


def stage2_ablate(best):
    print("\n" + "=" * 70)
    print(f"  STAGE 2: ablation  (N={N_TRAIN}, I={N_INIT_SEEDS}, stage-1 train/val)")
    print("=" * 70)

    # Same train/val split as Stage 1.
    X, y_noisy, _ = experiment.generate_sinc_dataset(
        seed=TUNE_SEED, n_samples=3000, noise='uniform')
    (X_pool, y_pool), _, _, _ = experiment.split_train_cal_test(
        X, y_noisy, seed=TUNE_SEED, train_frac=0.6, cal_frac=0.2)
    rng = np.random.default_rng(TUNE_SEED)
    idx = rng.choice(X_pool.shape[0], size=N_TRAIN + N_VAL, replace=False)
    X_tr_raw = X_pool[idx[:N_TRAIN]];  y_tr_raw = y_pool[idx[:N_TRAIN]]
    X_va_raw = X_pool[idx[N_TRAIN:]];  y_va_raw = y_pool[idx[N_TRAIN:]]

    y_clean_val = np.sinc(X_va_raw.flatten() / np.pi)
    true_lo_v, true_hi_v = stab.compute_true_bounds(y_clean_val, 'uniform', Q_LOWER, Q_UPPER)

    X_test_raw, y_test = generate_data(seed=TEST_SEED, n=N_TEST)
    true_lo_t, true_hi_t = compute_true_bounds(X_test_raw)

    rows = []
    for family, axis_key, axis_values, axis_label, folder in ABLATION_SPECS:
        base_cfg = best[family]["config"]
        print(f"\n  --- {family}: sweep {axis_key} over {axis_values} "
              f"(other params fixed from stage-1 best) ---")
        for v in axis_values:
            label = f"{family}_{axis_key}={v}"
            print(f"\n  >> {label}")
            pf = _make_predict_fn(family, axis_key, v, base_cfg)
            r = run_config_stage3(label, pf, N_INIT_SEEDS,
                                  X_tr_raw, y_tr_raw,
                                  X_va_raw, y_va_raw, true_lo_v, true_hi_v,
                                  X_test_raw, y_test, true_lo_t, true_hi_t)
            r.update({"family": family, "axis_key": axis_key,
                      "axis_value": v, "axis_label": axis_label,
                      "folder": folder, "config_label": label})
            rows.append(r)
    return rows


def run_config_stage3(label, predict_fn, n_init_seeds,
                      X_tr_raw, y_tr_raw,
                      X_val_raw, y_val, true_lo_val, true_hi_val,
                      X_test_raw, y_test, true_lo_test, true_hi_test):
    """Stage 3 version: fixed single training set, I init seeds, no D loop."""
    sc_X = StandardScaler().fit(X_tr_raw)
    sc_y = StandardScaler().fit(y_tr_raw.reshape(-1, 1))
    X_tr_s  = sc_X.transform(X_tr_raw).astype(np.float32)
    X_va_s  = sc_X.transform(X_val_raw).astype(np.float32)
    X_te_s  = sc_X.transform(X_test_raw).astype(np.float32)
    y_tr_s  = sc_y.transform(y_tr_raw.reshape(-1, 1)).flatten().astype(np.float32)

    preds_lo_v = np.zeros((n_init_seeds, len(y_val)),  dtype=np.float32)
    preds_hi_v = np.zeros((n_init_seeds, len(y_val)),  dtype=np.float32)
    preds_lo_t = np.zeros((n_init_seeds, N_TEST), dtype=np.float32)
    preds_hi_t = np.zeros((n_init_seeds, N_TEST), dtype=np.float32)

    t0 = time.time()
    for i in range(n_init_seeds):
        lo_v, hi_v = predict_fn(X_tr_s, y_tr_s, X_va_s, sc_y, i)
        lo_t, hi_t = predict_fn(X_tr_s, y_tr_s, X_te_s, sc_y, i)
        preds_lo_v[i] = lo_v; preds_hi_v[i] = hi_v
        preds_lo_t[i] = lo_t; preds_hi_t[i] = hi_t

    v_opt_v = decompose_raw(preds_lo_v[np.newaxis], preds_hi_v[np.newaxis])
    v_opt_t = decompose_raw(preds_lo_t[np.newaxis], preds_hi_t[np.newaxis])

    mean_lo_v = preds_lo_v.mean(axis=0); mean_hi_v = preds_hi_v.mean(axis=0)
    mean_lo_t = preds_lo_t.mean(axis=0); mean_hi_t = preds_hi_t.mean(axis=0)
    sr_v = sum_rmse(true_lo_val,  true_hi_val,  mean_lo_v, mean_hi_v)
    sr_t = sum_rmse(true_lo_test, true_hi_test, mean_lo_t, mean_hi_t)

    p_t, m_t = picp_mpiw(y_test, mean_lo_t, mean_hi_t)

    alpha = 1.0 - (Q_UPPER - Q_LOWER)
    is_v = interval_score(y_val,  mean_lo_v, mean_hi_v, alpha=alpha)
    is_t = interval_score(y_test, mean_lo_t, mean_hi_t, alpha=alpha)

    elapsed = (time.time() - t0) / 60.0
    print(f"     -> {label}: sr_val={sr_v:.4f}  sr_test={sr_t:.4f}  "
          f"IS_val={is_v:.3f}  IS_test={is_t:.3f}  "
          f"σopt_v={np.sqrt(v_opt_v):.4f}  σopt_t={np.sqrt(v_opt_t):.4f}  "
          f"[{elapsed:.1f} min]")
    return {
        "sum_rmse_val": round(sr_v, 4),  "sum_rmse_test": round(sr_t, 4),
        "interval_score_val": round(is_v, 4), "interval_score_test": round(is_t, 4),
        "sigma_opt_val": round(np.sqrt(v_opt_v), 4),
        "sigma_opt_test": round(np.sqrt(v_opt_t), 4),
        "picp_test":     round(p_t, 4),
        "mpiw_test":     round(m_t, 4),
        "elapsed_min":   round(elapsed, 2),
    }


def stage3_compare(best):
    print("\n" + "=" * 70)
    print(f"  STAGE 3: model comparison at stage-1 best params  (stage-1 train/val)")
    print("=" * 70)

    # Reconstruct the exact same train/val split used in Stage 1.
    X, y_noisy, _ = experiment.generate_sinc_dataset(
        seed=TUNE_SEED, n_samples=3000, noise='uniform')
    (X_pool, y_pool), _, _, _ = experiment.split_train_cal_test(
        X, y_noisy, seed=TUNE_SEED, train_frac=0.6, cal_frac=0.2)
    rng = np.random.default_rng(TUNE_SEED)
    idx = rng.choice(X_pool.shape[0], size=N_TRAIN + N_VAL, replace=False)
    X_tr_raw = X_pool[idx[:N_TRAIN]];  y_tr_raw = y_pool[idx[:N_TRAIN]]
    X_va_raw = X_pool[idx[N_TRAIN:]];  y_va_raw = y_pool[idx[N_TRAIN:]]

    y_clean_val = np.sinc(X_va_raw.flatten() / np.pi)
    true_lo_v, true_hi_v = stab.compute_true_bounds(y_clean_val, 'uniform', Q_LOWER, Q_UPPER)

    X_test_raw, y_test = generate_data(seed=TEST_SEED, n=N_TEST)
    true_lo_t, true_hi_t = compute_true_bounds(X_test_raw)

    rows = []

    nn_b = best["NN"]["config"]
    def pf_nn(X_tr_s, y_tr_s, X_eval_s, sc_y, init):
        return train_nn(X_tr_s, y_tr_s, X_eval_s, sc_y, init,
                        hidden_sizes=nn_b["hidden_sizes"], lr=nn_b["lr"],
                        epochs=nn_b["epochs"], batch_size=nn_b["batch_size"])
    print(f"\n  --- NN @ {nn_b} ---")
    r = run_config_stage3("NN", pf_nn, N_INIT_SEEDS,
                          X_tr_raw, y_tr_raw,
                          X_va_raw, y_va_raw, true_lo_v, true_hi_v,
                          X_test_raw, y_test, true_lo_t, true_hi_t)
    r.update({"model": "NN", "config": json.dumps(nn_b, default=str)})
    rows.append(r)

    return rows


# ════════════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════════════

METRIC_LABELS = {
    "sum_rmse":       "sum RMSE",
    "interval_score": "Interval Score",
    "sigma_opt":      "sigma_opt",
}


def _safe_slug(s: str) -> str:
    return str(s).replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")


def plot_ablation_axis(rows, family, axis_key, axis_label, folder, ts):
    sub = [r for r in rows
           if r["family"] == family and r["axis_key"] == axis_key]
    if not sub:
        return
    xs_raw = [r["axis_value"] for r in sub]
    categorical = any(isinstance(x, str) for x in xs_raw)
    if categorical:
        order = list(dict.fromkeys(xs_raw))
        xs_idx = [order.index(x) for x in xs_raw]
        order_idx = sorted(range(len(sub)), key=lambda i: xs_idx[i])
    else:
        order_idx = sorted(range(len(sub)), key=lambda i: xs_raw[i])

    out_folder = OUT_DIR / folder
    out_folder.mkdir(parents=True, exist_ok=True)

    for metric in ("sum_rmse", "sigma_opt"):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        vy = [sub[i][f"{metric}_val"]  for i in order_idx]
        ty = [sub[i][f"{metric}_test"] for i in order_idx]
        if categorical:
            xs_plot = list(range(len(order_idx)))
            ax.set_xticks(xs_plot)
            ax.set_xticklabels([str(xs_raw[i]) for i in order_idx])
        else:
            xs_plot = [xs_raw[i] for i in order_idx]
        ax.plot(xs_plot, vy, marker="o", lw=2, color="C0", label="val")
        ax.plot(xs_plot, ty, marker="s", lw=2, color="C3", label="test")
        ax.set_xlabel(axis_label, fontsize=13)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=13)
        ax.set_title(f"{family} (N={N_TRAIN}): {METRIC_LABELS[metric]} vs {axis_label}  "
                     f"(val vs test)", fontsize=13)
        ax.grid(True, alpha=0.3); ax.legend(fontsize=11)
        fig.tight_layout()
        slug = _safe_slug(axis_key)
        out = out_folder / f"{family.lower()}_{slug}_{metric}_{ts}.png"
        fig.savefig(out, dpi=160); plt.close(fig)
        print(f"  plot: {folder}/{out.name}")


def plot_model_comparison(rows, ts):
    out_folder = OUT_DIR / "compare"
    out_folder.mkdir(parents=True, exist_ok=True)
    models = [r["model"] for r in rows]
    xpos = np.arange(len(models))
    width = 0.35
    for metric in ("sum_rmse", "sigma_opt"):
        fig, ax = plt.subplots(figsize=(9, 4.5))
        v = [r[f"{metric}_val"]  for r in rows]
        t = [r[f"{metric}_test"] for r in rows]
        ax.bar(xpos - width/2, v, width, label="val",  color="C0")
        ax.bar(xpos + width/2, t, width, label="test", color="C3")
        ax.set_xticks(xpos); ax.set_xticklabels(models)
        ax.set_ylabel(METRIC_LABELS[metric], fontsize=13)
        ax.set_title(f"NN (N={N_TRAIN}) @ stage-1 best: "
                     f"{METRIC_LABELS[metric]}  (val vs test)", fontsize=13)
        ax.grid(True, alpha=0.3, axis="y"); ax.legend(fontsize=11)
        fig.tight_layout()
        out = out_folder / f"compare_{metric}_val_vs_test_{ts}.png"
        fig.savefig(out, dpi=160); plt.close(fig)
        print(f"  plot: compare/{out.name}")


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-tune", action="store_true",
                    help="Reuse latest best_hyperparams.json instead of retuning.")
    ap.add_argument("--stage", type=int, default=0,
                    help="0 = all, 1 = tune only, 2 = ablation only, "
                         "3 = compare only.")
    args = ap.parse_args()

    if args.skip_tune or args.stage in (2, 3):
        best = load_latest_best()
        if best is None:
            print("No best_hyperparams.json found; running stage 1.")
            best = stage1_tune()
        else:
            print(f"\nLoaded best_hyperparams from {TUNE_DIR}\n"
                  f"  best models: {list(best.keys())}")
    else:
        best = stage1_tune()

    if args.stage == 1:
        return

    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_rows_2, out_rows_3 = [], []

    if args.stage in (0, 2):
        out_rows_2 = stage2_ablate(best)
        df2 = pd.DataFrame(out_rows_2)

        df2.to_csv(OUT_DIR / f"ablation_val_test_{ts}.csv", index=False)
        print(f"\n  ablation CSV (combined): ablation_val_test_{ts}.csv")
        for folder in sorted(df2["folder"].unique()):
            sub_folder = OUT_DIR / folder
            sub_folder.mkdir(parents=True, exist_ok=True)
            df2[df2["folder"] == folder].to_csv(
                sub_folder / f"ablation_{folder}_{ts}.csv", index=False)
            print(f"  ablation CSV ({folder}): {folder}/ablation_{folder}_{ts}.csv")

        for family, axis_key, _, axis_label, folder in ABLATION_SPECS:
            plot_ablation_axis(out_rows_2, family, axis_key,
                               axis_label, folder, ts)

    if args.stage in (0, 3):
        out_rows_3 = stage3_compare(best)
        df3 = pd.DataFrame(out_rows_3)
        (OUT_DIR / "compare").mkdir(parents=True, exist_ok=True)
        df3.to_csv(OUT_DIR / "compare" / f"compare_val_test_{ts}.csv",
                   index=False)
        print(f"\n  comparison CSV: compare/compare_val_test_{ts}.csv")
        plot_model_comparison(out_rows_3, ts)


if __name__ == "__main__":
    main()
