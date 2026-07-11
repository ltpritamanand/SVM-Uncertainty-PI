"""
Amprion Quantile Regression — LSTM / GRU / TCN (PyTorch)
=========================================================
Pipeline
--------
1. Load Amprion.csv → flatten, fillNaN, downsample ×2
2. Window size = 24, target = next point
3. Stage 1: Grid search on 1 000-point subset (80/20 train/val)
   Grid: hidden_size × lr × batch_size
4. Stage 2: 10 seeds × 5 data sizes (4k, 6k, 8k, 10k, 12k)
   For each size: take N samples, last 1k = test, rest = train+val (90/10)
   No held-out test set — test set varies with data size (sliding window).
   Metrics: PICP, MPIW, σ²_model, training time
5. Plots: per-model (6) + cross-model (3)
6. Results .txt files per model + cross-model summary

Coverage: 90%  →  τ_lower = 0.05, τ_upper = 0.95
"""

import os
import sys
import time
import warnings
import itertools
import random
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use('default')
matplotlib.rcParams.update({
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.edgecolor': '#444444',
    'axes.labelcolor': '#222222',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.color': '#222222',
    'ytick.color': '#222222',
    'text.color': '#222222',
    'grid.color': '#dddddd',
    'grid.linestyle': '-',
    'grid.alpha': 1.0,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': '#cccccc',
    'figure.dpi': 100,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
})
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────── CONFIG ────────────────────────────────────────
WINDOW       = 24
Q_LOWER      = 0.05
Q_UPPER      = 0.95
TARGET_COV   = 0.90
DATA_SIZES   = [4000, 6000, 8000, 10000, 12000, 16000]   # total sample sizes
TUNE_SIZE    = 1000            # subset for Stage-1 grid search (1k train points)
N_RUNS       = 10              # seeds for Stage-2
N_EPOCHS     = 50              # training epochs per model fit
THRESHOLD    = 0.90            # PICP threshold for hyperparameter selection

# Sliding test set protocol: for each data size N, last 1k = test, rest = train+val.
# Example: 4k → 3k train+val + 1k test; 6k → 5k train+val + 1k test, etc.
TEST_SIZE    = 1000            # last N points of each data size used as test
VAL_FRAC     = 0.10            # fraction of train+val pool used as validation

HIDDEN_SIZES = [
    # 1-layer
    32, 64, 128, 256, 512, 1024,
    # 2-layer (uniform width per layer)
    (32, 64), (64, 128), (128, 128), (256, 512),
]
LR_LIST     = [1e-4, 1e-3, 1e-2]
BATCH_SIZES = [32, 64, 128]

assert torch.cuda.is_available(), "CUDA not available! Install PyTorch with CUDA support: pip install torch --index-url https://download.pytorch.org/whl/cu118"
DEVICE = torch.device("cuda")
print(f"Device: {DEVICE}  |  CUDA: {torch.version.cuda}  |  GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────── OUTPUT DIRS ───────────────────────────────────
BASE_OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

def make_dirs():
    for model_name in ["LSTM", "GRU", "TCN"]:
        for sub in ["plots", "results"]:
            os.makedirs(f"{BASE_OUT}/{model_name}/{sub}", exist_ok=True)
    os.makedirs(f"{BASE_OUT}/cross_model/plots", exist_ok=True)
    os.makedirs(f"{BASE_OUT}/cross_model/results", exist_ok=True)

make_dirs()

# ─────────────────────────── DATA PIPELINE ─────────────────────────────────
def load_50hertz(csv_path: str) -> np.ndarray:
    """Mirrors the MATLAB pipeline exactly (readmatrix auto-skips text headers)."""
    # Read with first row as header, then drop the Date column (non-numeric)
    df = pd.read_csv(csv_path)
    # Drop the Date column (or any non-numeric columns) - keep only numeric data
    numeric_df = df.select_dtypes(include=[np.number])
    raw = numeric_df.values                                   # rows=days, cols=intervals
    y = raw.flatten().astype(float)                           # chronological row-major flatten
    # linear interpolation of NaN
    series = pd.Series(y)
    series = series.interpolate(method="linear", limit_direction="both")
    y = series.dropna().values
    print(f"  Total before downsample: {len(y)}")
    y = y[::2]                                                # every 2nd point (downsample ×2)
    print(f"  Total after  downsample: {len(y)}")
    return y


def build_windows(y: np.ndarray, win: int):
    """Build (X, target) sliding-window pairs."""
    X, tgt = [], []
    for i in range(len(y) - win - 1):
        X.append(y[i : i + win])
        tgt.append(y[i + win])
    return np.array(X, dtype=np.float32), np.array(tgt, dtype=np.float32)


def split_data(X, y, splits=(0.6, 0.7)):
    n = len(y)
    i1 = int(n * splits[0])
    i2 = int(n * splits[1])
    return (X[:i1], y[:i1],
            X[i1:i2], y[i1:i2],
            X[i2:], y[i2:])


# ─────────────────────────── SCALING (fit on train only — zero leakage) ──
def scale_splits(X_tr, y_tr, X_va, y_va, X_te, y_te):
    """
    Fit StandardScaler on TRAIN split only, then transform ALL splits.
    Guarantees no data leakage from validation or test into the scaler.
    Returns scaled arrays plus the fitted scaler_y for inverse-transform later.
    """
    # Fit on training data ONLY
    scaler_X = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))

    # Transform all splits using the training-fitted scalers
    X_tr_s = scaler_X.transform(X_tr)
    X_va_s = scaler_X.transform(X_va)
    X_te_s = scaler_X.transform(X_te)

    y_tr_s = scaler_y.transform(y_tr.reshape(-1, 1)).flatten()
    y_va_s = scaler_y.transform(y_va.reshape(-1, 1)).flatten()
    y_te_s = scaler_y.transform(y_te.reshape(-1, 1)).flatten()

    return X_tr_s, X_va_s, X_te_s, y_tr_s, y_va_s, y_te_s, scaler_y


def to_loader(X, y, batch_size, shuffle=True):
    Xt = torch.tensor(X).unsqueeze(-1)   # (N, win, 1)
    yt = torch.tensor(y).unsqueeze(-1)   # (N, 1)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

# ─────────────────────────── METRICS ───────────────────────────────────────
def evaluate_picp_mpiw(y_true, y_lower, y_upper):
    picp = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))
    mpiw = float(np.mean(y_upper - y_lower))
    return picp, mpiw

# ─────────────────────────── PINBALL LOSS ──────────────────────────────────
class PinballLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, pred, target):
        err = target - pred
        loss = torch.where(err >= 0, self.tau * err, (self.tau - 1) * err)
        return loss.mean()

# ─────────────────────────── MODELS ────────────────────────────────────────
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])


class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, dilation=dilation, padding=pad)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None

    def forward(self, x):
        # Causal: trim right padding
        out = self.relu(self.conv1(x)[..., :x.shape[-1]])
        out = self.drop(out)
        out = self.relu(self.conv2(out)[..., :x.shape[-1]])
        out = self.drop(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_levels=3,
                 kernel_size=3, dropout=0.1):
        super().__init__()
        channels = [hidden_size] * num_levels
        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, 2**i, dropout))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (N, T, 1) → (N, 1, T) for Conv1d
        x = x.permute(0, 2, 1)
        out = self.network(x)          # (N, C, T)
        return self.fc(out[:, :, -1])  # last time step


MODEL_CLASSES = {"LSTM": LSTMModel, "GRU": GRUModel, "TCN": TCNModel}

# ─────────────────────────── TRAINING ──────────────────────────────────────
def build_model(model_name, hidden_size, seed):
    torch.manual_seed(seed)
    cls = MODEL_CLASSES[model_name]
    if isinstance(hidden_size, (list, tuple)):
        # tuple → 2-layer, e.g. (128, 128): hidden=128, num_layers=2
        hs, nl = hidden_size[0], len(hidden_size)
    else:
        # int → 1-layer, e.g. 128: hidden=128, num_layers=1
        hs, nl = hidden_size, 1
    model = cls(hidden_size=hs, num_levels=nl) if model_name == "TCN" \
            else cls(hidden_size=hs, num_layers=nl)
    return model.to(DEVICE)


def train_one_quantile(model, loader_tr, tau, lr, n_epochs):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = PinballLoss(tau)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0.0
        for Xb, yb in loader_tr:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
        scheduler.step(total_loss)
    return model


def predict(model, X_np):
    model.eval()
    Xt = torch.tensor(X_np).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        preds = []
        bs = 512
        for i in range(0, len(Xt), bs):
            preds.append(model(Xt[i:i+bs]).cpu().numpy().flatten())
    return np.concatenate(preds)


def train_and_eval(model_name, hidden_size, lr, batch_size,
                   X_tr, y_tr, X_va, y_va, X_te, y_te, seed,
                   scaler_y, n_epochs=N_EPOCHS):
    """
    Train lower and upper quantile models on scaled data, then inverse-transform
    predictions and ground truth back to original scale for metric computation.

    scaler_y : StandardScaler fitted on training targets only
               (pass None if data is already in original scale, for backwards compat)
    hidden_size may be an int (1-layer) or a tuple of equal ints (e.g. (128,128) = 2-layer).
    """
    loader_tr = to_loader(X_tr, y_tr, batch_size)

    t0 = time.time()
    model_lo = build_model(model_name, hidden_size, seed)
    model_lo = train_one_quantile(model_lo, loader_tr, Q_LOWER, lr, n_epochs)

    model_hi = build_model(model_name, hidden_size, seed + 10000)
    model_hi = train_one_quantile(model_hi, loader_tr, Q_UPPER, lr, n_epochs)
    elapsed = time.time() - t0

    # Predictions are in SCALED space
    pred_va_lo_s = predict(model_lo, X_va)
    pred_va_hi_s = predict(model_hi, X_va)
    pred_te_lo_s = predict(model_lo, X_te)
    pred_te_hi_s = predict(model_hi, X_te)

    if scaler_y is not None:
        # Inverse-transform predictions back to original scale
        pred_va_lo = scaler_y.inverse_transform(pred_va_lo_s.reshape(-1, 1)).flatten()
        pred_va_hi = scaler_y.inverse_transform(pred_va_hi_s.reshape(-1, 1)).flatten()
        pred_te_lo = scaler_y.inverse_transform(pred_te_lo_s.reshape(-1, 1)).flatten()
        pred_te_hi = scaler_y.inverse_transform(pred_te_hi_s.reshape(-1, 1)).flatten()
        # Inverse-transform ground truth back to original scale
        y_va_orig = scaler_y.inverse_transform(y_va.reshape(-1, 1)).flatten()
        y_te_orig = scaler_y.inverse_transform(y_te.reshape(-1, 1)).flatten()
    else:
        pred_va_lo, pred_va_hi = pred_va_lo_s, pred_va_hi_s
        pred_te_lo, pred_te_hi = pred_te_lo_s, pred_te_hi_s
        y_va_orig, y_te_orig = y_va, y_te

    picp_va, mpiw_va = evaluate_picp_mpiw(y_va_orig, pred_va_lo, pred_va_hi)
    picp_te, mpiw_te = evaluate_picp_mpiw(y_te_orig, pred_te_lo, pred_te_hi)

    return (picp_va, mpiw_va, picp_te, mpiw_te, elapsed,
            pred_te_lo, pred_te_hi, model_lo, model_hi)

# ─────────────────────────── STAGE 1 ───────────────────────────────────────
def stage1_grid_search(model_name, y_full):
    print(f"\n{'='*60}")
    print(f"  STAGE 1 — Grid Search: {model_name}")
    print(f"{'='*60}")

    y_tune = y_full[:TUNE_SIZE]
    X_all, y_all = build_windows(y_tune, WINDOW)
    n = len(y_all)
    i_split = int(n * 0.80)
    X_tr, y_tr = X_all[:i_split], y_all[:i_split]
    X_va, y_va = X_all[i_split:], y_all[i_split:]

    # Scale data — fit scaler on TRAIN split only (zero leakage)
    # X_va passed as dummy test set; test results are unused in Stage 1
    X_tr_s, X_va_s, X_te_s, y_tr_s, y_va_s, y_te_s, scaler_y = scale_splits(
        X_tr, y_tr, X_va, y_va, X_va, y_va)

    grid = list(itertools.product(HIDDEN_SIZES, LR_LIST, BATCH_SIZES))
    print(f"  Grid size: {len(grid)} combinations (with StandardScaler)")

    candidates = []
    for hs, lr, bs in grid:
        try:
            picp_va, mpiw_va, _, _, _, _, _, _, _ = train_and_eval(
                model_name, hs, lr, bs,
                X_tr_s, y_tr_s, X_va_s, y_va_s, X_te_s, y_te_s,
                seed=42, scaler_y=scaler_y, n_epochs=30)
            candidates.append((hs, lr, bs, picp_va, mpiw_va))
            print(f"    hs={str(hs):12s} lr={lr:.0e} bs={bs:3d} "
                  f"→ PICP={picp_va:.4f}  MPIW={mpiw_va:.4f}")
        except Exception as e:
            print(f"    hs={hs} lr={lr} bs={bs} → FAILED: {e}")

    # Selection: PICP ≥ threshold → min MPIW; else max PICP (tiebreak min MPIW)
    valid = [(c, m, hs, lr, bs) for hs, lr, bs, c, m in candidates if c >= THRESHOLD]
    if valid:
        best = min(valid, key=lambda x: x[1])
        best_hs, best_lr, best_bs = best[2], best[3], best[4]
        best_picp, best_mpiw = best[0], best[1]
    else:
        # fallback: max PICP, tiebreak min MPIW
        all_ = [(c, m, hs, lr, bs) for hs, lr, bs, c, m in candidates]
        best = sorted(all_, key=lambda x: (-x[0], x[1]))[0]
        best_picp, best_mpiw = best[0], best[1]
        best_hs, best_lr, best_bs = best[2], best[3], best[4]

    print(f"\n  BEST → hidden={best_hs} lr={best_lr:.0e} bs={best_bs} "
          f"PICP={best_picp:.4f} MPIW={best_mpiw:.4f}")
    return {"hidden_size": best_hs, "lr": best_lr, "batch_size": best_bs,
            "tune_picp": best_picp, "tune_mpiw": best_mpiw}

# ─────────────────────────── STAGE 2 ───────────────────────────────────────
def stage2_multi_size(model_name, y_full, best_params):
    """
    Sliding test set protocol.
      For each data size N (4k, 6k, 8k, 10k, 12k):
        - Take first N samples of y_full
        - Last TEST_SIZE = 1k samples → test set (different for each size)
        - Remaining (N - 1k) samples → train + val pool
        - Train/val split: 90/10 chronological split inside the pool
      Example: 4k → ~3k train+val + ~1k test; 6k → ~5k train+val + ~1k test.
    """
    print(f"\n{'='*60}")
    print(f"  STAGE 2 — Sliding Test Set Evaluation: {model_name}")
    print(f"{'='*60}")

    hs  = best_params["hidden_size"]
    lr  = best_params["lr"]
    bs  = best_params["batch_size"]

    n_full = len(y_full)
    if max(DATA_SIZES) > n_full:
        raise RuntimeError(
            f"Need at least {max(DATA_SIZES)} samples; have {n_full}.")

    results = {}

    for curr_size in DATA_SIZES:
        # Take first N samples
        y_chunk = y_full[:curr_size]

        # Split: last TEST_SIZE = test, rest = train+val pool (at sample level)
        y_test_raw = y_chunk[curr_size - TEST_SIZE:]
        y_pool = y_chunk[:curr_size - TEST_SIZE]

        # Build windows
        X_te, y_te = build_windows(y_test_raw, WINDOW)
        X_pool, y_pool_t = build_windows(y_pool, WINDOW)

        # Chronological train/val split inside the pool
        n_pool = len(y_pool_t)
        i_val = int(n_pool * (1.0 - VAL_FRAC))
        X_tr, y_tr = X_pool[:i_val], y_pool_t[:i_val]
        X_va, y_va = X_pool[i_val:], y_pool_t[i_val:]

        print(f"\n  ── Size {curr_size}: "
              f"train={len(X_tr):>5}  val={len(X_va):>5}  test={len(X_te):>5} ──")

        # Scale: fit on TRAIN only, then transform train/val/test
        scaler_X = StandardScaler().fit(X_tr)
        scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))
        X_tr_s = scaler_X.transform(X_tr)
        X_va_s = scaler_X.transform(X_va)
        X_te_s = scaler_X.transform(X_te)
        y_tr_s = scaler_y.transform(y_tr.reshape(-1, 1)).flatten()
        y_va_s = scaler_y.transform(y_va.reshape(-1, 1)).flatten()
        y_te_s = scaler_y.transform(y_te.reshape(-1, 1)).flatten()

        picps_va, mpiws_va = [], []
        picps_te, mpiws_te = [], []
        times = []
        preds_lo_runs = []
        preds_hi_runs = []

        seeds = list(range(N_RUNS))

        for run_idx, seed in enumerate(seeds):
            (pv, mv, pt, mt, elapsed,
             p_lo, p_hi, _, _) = train_and_eval(
                model_name, hs, lr, bs,
                X_tr_s, y_tr_s, X_va_s, y_va_s, X_te_s, y_te_s,
                seed=seed, scaler_y=scaler_y, n_epochs=N_EPOCHS)

            picps_va.append(pv); mpiws_va.append(mv)
            picps_te.append(pt); mpiws_te.append(mt)
            times.append(elapsed)
            preds_lo_runs.append(p_lo)
            preds_hi_runs.append(p_hi)

            print(f"    Run {run_idx+1:2d}/{N_RUNS}  "
                  f"Val PICP={pv:.4f} MPIW={mv:.4f}  "
                  f"Test PICP={pt:.4f} MPIW={mt:.4f}  "
                  f"Time={elapsed:.1f}s")

        # ── σ² model uncertainty ─────────────────────────────────────────
        preds_lo_arr = np.array(preds_lo_runs)
        preds_hi_arr = np.array(preds_hi_runs)

        mean_lo = preds_lo_arr.mean(axis=0)
        mean_hi = preds_hi_arr.mean(axis=0)
        sigma2_lo_raw = ((preds_lo_arr - mean_lo)**2).mean()
        sigma2_hi_raw = ((preds_hi_arr - mean_hi)**2).mean()
        sigma2_raw    = sigma2_lo_raw + sigma2_hi_raw

        # Normalise by test-set statistics (fair across sliding test sets)
        # y_te is already in original scale (from build_windows), use it directly
        var_test   = np.var(y_te)
        range_test = (np.max(y_te) - np.min(y_te)) ** 2

        sigma2_var_norm   = sigma2_raw / (var_test + 1e-12)
        sigma2_range_norm = sigma2_raw / (range_test + 1e-12)

        results[curr_size] = {
            "picp_val"      : picps_va,
            "mpiw_val"      : mpiws_va,
            "picp_test"     : picps_te,
            "mpiw_test"     : mpiws_te,
            "time"          : times,
            "sigma2"        : sigma2_raw,
            "sigma2_raw"    : sigma2_raw,
            "sigma2_var_norm"   : sigma2_var_norm,
            "sigma2_range_norm" : sigma2_range_norm,
            "preds_lo"      : preds_lo_arr,
            "preds_hi"      : preds_hi_arr,
            "y_te"          : y_te,
        }

    return results

# ─────────────────────────── PLOTTING (per model) ──────────────────────────
def plot_per_model(model_name, results):
    sizes = sorted(results.keys())

    def _extract(key):
        vals = [results[s][key] for s in sizes]
        means = [np.mean(v) for v in vals]
        stds  = [np.std(v)  for v in vals]
        return np.array(means), np.array(stds)

    picp_te_m, picp_te_s  = _extract("picp_test")
    mpiw_te_m, mpiw_te_s  = _extract("mpiw_test")
    picp_va_m, picp_va_s  = _extract("picp_val")
    mpiw_va_m, mpiw_va_s  = _extract("mpiw_val")
    time_m, time_s        = _extract("time")

    ratio_te_m = mpiw_te_m / np.clip(picp_te_m, 1e-6, None)
    ratio_va_m = mpiw_va_m / np.clip(picp_va_m, 1e-6, None)
    ratio_te_s = ratio_te_m * np.sqrt(
        (mpiw_te_s / np.clip(mpiw_te_m, 1e-8, None))**2 +
        (picp_te_s / np.clip(picp_te_m, 1e-8, None))**2)

    sigma2s = [results[s]["sigma2"] for s in sizes]
    # std of PICP / MPIW across runs
    picp_std  = [np.std(results[s]["picp_test"]) for s in sizes]
    mpiw_std  = [np.std(results[s]["mpiw_test"]) for s in sizes]

    out = f"{BASE_OUT}/{model_name}/plots"
    colors = {"val": "#2196F3", "test": "#E53935"}
    sizes_arr = np.array(sizes)

    def savefig(fig, name):
        fig.tight_layout()
        fig.savefig(f"{out}/{name}", dpi=120)
        plt.close(fig)

    # ── 1. PICP vs size ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes_arr, picp_va_m, "-s", color=colors["val"], label="Validation", linewidth=1.5, markersize=6)
    ax.plot(sizes_arr, picp_te_m, "-o", color=colors["test"], label="Test", linewidth=1.5, markersize=6)
    ax.axhline(TARGET_COV, color="red", linestyle="--", linewidth=1, label=f"Target: {TARGET_COV}")
    ax.set_xlabel("Dataset Size (samples)"); ax.set_ylabel("PICP (Coverage)")
    ax.set_title(f"{model_name} - PICP vs Data Size"); ax.legend(); ax.grid(True)
    savefig(fig, "1_picp_vs_size.png")

    # ── 2. MPIW vs size ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes_arr, mpiw_va_m, "-s", color=colors["val"], label="Validation", linewidth=1.5, markersize=6)
    ax.plot(sizes_arr, mpiw_te_m, "-o", color=colors["test"], label="Test", linewidth=1.5, markersize=6)
    ax.set_xlabel("Dataset Size (samples)"); ax.set_ylabel("MPIW")
    ax.set_title(f"{model_name} - MPIW vs Data Size"); ax.legend(); ax.grid(True)
    savefig(fig, "2_mpiw_vs_size.png")

    # ── 3. MPIW/PICP ratio ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(sizes_arr - 300, ratio_va_m, width=500, color=colors["val"], label="Validation")
    ax.bar(sizes_arr + 300, ratio_te_m, width=500, color=colors["test"], label="Test")
    ax.set_xlabel("Dataset Size (samples)"); ax.set_ylabel("MPIW / PICP Ratio")
    ax.set_title(f"{model_name} - MPIW/PICP vs Data Size"); ax.legend(); ax.grid(True, axis='y')
    savefig(fig, "3_ratio_vs_size.png")

    # ── 4. Training time vs size ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes_arr, time_m, "-o", color="#2196F3", linewidth=1.5, markersize=6)
    ax.set_xlabel("Dataset Size (samples)"); ax.set_ylabel("Training Time (s)")
    ax.set_title(f"{model_name} - Training Time vs Data Size"); ax.grid(True)
    savefig(fig, "4_time_vs_size.png")

    # ── 5. Std of PICP vs size ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes_arr, picp_std, "-o", color="#7B1FA2", linewidth=1.5, markersize=6)
    ax.set_xlabel("Dataset Size (samples)"); ax.set_ylabel("Std of PICP")
    ax.set_title(f"{model_name} - Std(PICP) vs Data Size"); ax.grid(True)
    savefig(fig, "5_picp_std_vs_size.png")

    # ── 6. Std of MPIW vs size ──
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(sizes_arr, mpiw_std, "-o", color="#F57C00", linewidth=1.5, markersize=6)
    ax.set_xlabel("Dataset Size (samples)"); ax.set_ylabel("Std of MPIW")
    ax.set_title(f"{model_name} - Std(MPIW) vs Data Size"); ax.grid(True)
    savefig(fig, "6_mpiw_std_vs_size.png")

    print(f"  Plots saved → {out}/")
    return {
        "sizes"      : sizes_arr,
        "picp_te_m"  : picp_te_m,
        "picp_te_s"  : picp_te_s,
        "mpiw_te_m"  : mpiw_te_m,
        "mpiw_te_s"  : mpiw_te_s,
        "ratio_te_m" : ratio_te_m,
        "ratio_te_s" : ratio_te_s,
    }

# ─────────────────────────── RESULTS TXT ───────────────────────────────────
def write_results_txt(model_name, results, best_params):
    ts  = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = f"{BASE_OUT}/{model_name}/results/results_{ts}.txt"
    sizes = sorted(results.keys())

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{'='*65}\n")
        f.write(f"  Amprion Quantile {model_name} - Sliding Test Set Results\n")
        f.write(f"{'='*65}\n\n")

        f.write("Experiment Configuration\n")
        f.write("-"*45 + "\n")
        f.write(f"Model             : {model_name}\n")
        f.write(f"Window Size       : {WINDOW}\n")
        f.write(f"Quantile Lower    : {Q_LOWER}\n")
        f.write(f"Quantile Upper    : {Q_UPPER}\n")
        f.write(f"Target Coverage   : {TARGET_COV}\n")
        f.write(f"Training Epochs   : {N_EPOCHS}\n")
        f.write(f"Runs per Size     : {N_RUNS}\n")
        f.write(f"Tuning Subset     : {TUNE_SIZE} points\n")
        f.write(f"Device            : {DEVICE}\n\n")

        f.write("Best Hyperparameters (from Stage-1)\n")
        f.write("-"*45 + "\n")
        f.write(f"Hidden Size : {best_params['hidden_size']}\n")
        f.write(f"LR          : {best_params['lr']:.0e}\n")
        f.write(f"Batch Size  : {best_params['batch_size']}\n")
        f.write(f"Tune PICP   : {best_params['tune_picp']:.6f}\n")
        f.write(f"Tune MPIW   : {best_params['tune_mpiw']:.6f}\n\n")

        f.write("Performance Summary (mean ± std over 10 runs)\n")
        f.write("-"*140 + "\n")
        hdr = (f"{'Size':>8}  {'Val PICP':>12}  {'Val MPIW':>12}  "
               f"{'Test PICP':>12}  {'Test MPIW':>12}  "
               f"{'MPIW/PICP':>12}  {'Time (s)':>10}  "
               f"{'sigma2_raw':>12}  {'sigma2_var_norm':>16}  {'sigma2_range_norm':>18}\n")
        f.write(hdr)
        f.write("-"*140 + "\n")
        for s in sizes:
            r = results[s]
            pv   = f"{np.mean(r['picp_val']):.4f}±{np.std(r['picp_val']):.4f}"
            mv   = f"{np.mean(r['mpiw_val']):.4f}±{np.std(r['mpiw_val']):.4f}"
            pt   = f"{np.mean(r['picp_test']):.4f}±{np.std(r['picp_test']):.4f}"
            mt   = f"{np.mean(r['mpiw_test']):.4f}±{np.std(r['mpiw_test']):.4f}"
            rat  = f"{np.mean(r['mpiw_test'])/max(np.mean(r['picp_test']),1e-6):.4f}"
            tm   = f"{np.mean(r['time']):.2f}±{np.std(r['time']):.2f}"
            s2r  = f"{r['sigma2_raw']:.4f}"
            s2v  = f"{r['sigma2_var_norm']:.6f}"
            s2rn = f"{r['sigma2_range_norm']:.6f}"
            f.write(f"{s:>8}  {pv:>12}  {mv:>12}  {pt:>12}  {mt:>12}  "
                    f"{rat:>12}  {tm:>10}  {s2r:>12}  {s2v:>16}  {s2rn:>18}\n")

        f.write("\nProtocol: SLIDING test set — for each size N, last 1k samples = test, rest = train+val.\n")
        f.write("Test set changes with data size (e.g., 4k→3k+1k, 6k→5k+1k, etc.).\n")
        f.write("Notes on sigma2 variants:\n")
        f.write("  sigma2_raw          : mean per-point variance across 10 runs (original y-units²).\n")
        f.write("  sigma2_var_norm     : sigma2_raw / var(y_test). Model instability relative\n")
        f.write("                        to how much the test data itself varies. < 1 → model\n")
        f.write("                        variance is smaller than data variance — good.\n")
        f.write("  sigma2_range_norm   : sigma2_raw / range(y_test)². Similar but uses full\n")
        f.write("                        spread; more sensitive to outliers in the test set.\n")
        f.write("  Both normalised variants use the specific test-set statistics for each\n")
        f.write("  data size, making them fair for the sliding test-set protocol.\n")

        f.write("\nExecution completed.\n")
        f.write(f"Timestamp: {datetime.now()}\n")

    print(f"  Results → {path}")
    return path

# ─────────────────────────── CROSS-MODEL PLOTS ─────────────────────────────
def plot_cross_model(summaries: dict):
    """summaries: {model_name: {sizes, picp_te_m, ...}}"""
    colors = {"LSTM": "#1565C0", "GRU": "#2E7D32", "TCN": "#C62828"}
    markers = {"LSTM": "o", "GRU": "s", "TCN": "^"}
    out = f"{BASE_OUT}/cross_model/plots"

    def savefig(fig, name):
        fig.tight_layout()
        fig.savefig(f"{out}/{name}", dpi=120)
        plt.close(fig)

    # ── 7. PICP vs size — all models ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for mn, d in summaries.items():
        ax.plot(d["sizes"], d["picp_te_m"], f"-{markers[mn]}", color=colors[mn],
                label=mn, linewidth=1.5, markersize=6)
    ax.axhline(TARGET_COV, color="red", linestyle="--", linewidth=1, label=f"Target: {TARGET_COV}")
    ax.set_xlabel("Dataset Size (samples)"); ax.set_ylabel("PICP (Coverage)")
    ax.set_title("Cross-Model: PICP vs Data Size"); ax.legend(); ax.grid(True)
    savefig(fig, "7_cross_picp_vs_size.png")

    # ── 8. MPIW vs size ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for mn, d in summaries.items():
        ax.plot(d["sizes"], d["mpiw_te_m"], f"-{markers[mn]}", color=colors[mn],
                label=mn, linewidth=1.5, markersize=6)
    ax.set_xlabel("Dataset Size (samples)"); ax.set_ylabel("MPIW")
    ax.set_title("Cross-Model: MPIW vs Data Size"); ax.legend(); ax.grid(True)
    savefig(fig, "8_cross_mpiw_vs_size.png")

    # ── 9. MPIW/PICP ratio ──
    n_models = len(summaries)
    width = 400
    offsets = np.linspace(-(n_models-1)*width/2, (n_models-1)*width/2, n_models)
    fig, ax = plt.subplots(figsize=(8, 5))
    for (mn, d), offset in zip(summaries.items(), offsets):
        ax.bar(np.array(d["sizes"]) + offset, d["ratio_te_m"], width=width,
               color=colors[mn], label=mn)
    ax.set_xlabel("Dataset Size (samples)"); ax.set_ylabel("MPIW / PICP Ratio")
    ax.set_title("Cross-Model: MPIW/PICP vs Data Size"); ax.legend(); ax.grid(True, axis='y')
    savefig(fig, "9_cross_ratio_vs_size.png")

    print(f"  Cross-model plots → {out}/")

# ─────────────────────────── CROSS-MODEL SUMMARY TXT ──────────────────────
def write_cross_model_summary(summaries, all_results, all_params):
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = f"{BASE_OUT}/cross_model/results/cross_model_summary_{ts}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("="*75 + "\n")
        f.write("  Amprion - Cross-Model Comparison Summary (LSTM / GRU / TCN)\n")
        f.write("="*75 + "\n\n")
        f.write(f"Coverage Target : {TARGET_COV} (τ={Q_LOWER}/{Q_UPPER})\n")
        f.write(f"Window Size     : {WINDOW}\n")
        f.write(f"Runs per Size   : {N_RUNS}\n")
        f.write(f"Data Sizes      : {DATA_SIZES}\n\n")

        f.write("Best Hyperparameters per Model\n")
        f.write("-"*45 + "\n")
        for mn, bp in all_params.items():
            f.write(f"  {mn:5s}: hidden={str(bp['hidden_size']):12s}  "
                    f"lr={bp['lr']:.0e}  bs={bp['batch_size']:3d}  "
                    f"TunePICP={bp['tune_picp']:.4f}  TuneMPIW={bp['tune_mpiw']:.4f}\n")
        f.write("\n")

        f.write("Test Set Results (mean ± std, 10 runs)\n")
        f.write("-"*120 + "\n")
        f.write(f"{'Model':6s}  {'Size':>6}  {'PICP':>13}  {'MPIW':>13}  "
                f"{'Ratio':>10}  {'Time(s)':>10}  {'sigma2_var_norm':>16}  {'sigma2_range_norm':>18}\n")
        f.write("-"*120 + "\n")

        for mn in ["LSTM", "GRU", "TCN"]:
            res = all_results[mn]
            for s in DATA_SIZES:
                r = res[s]
                pt  = f"{np.mean(r['picp_test']):.4f}±{np.std(r['picp_test']):.4f}"
                mt  = f"{np.mean(r['mpiw_test']):.4f}±{np.std(r['mpiw_test']):.4f}"
                rat = f"{np.mean(r['mpiw_test'])/max(np.mean(r['picp_test']),1e-6):.4f}"
                tm  = f"{np.mean(r['time']):.2f}±{np.std(r['time']):.2f}"
                s2v  = f"{r['sigma2_var_norm']:.6f}"
                s2rn = f"{r['sigma2_range_norm']:.6f}"
                f.write(f"{mn:6s}  {s:>6}  {pt:>13}  {mt:>13}  "
                        f"{rat:>10}  {tm:>10}  {s2v:>16}  {s2rn:>18}\n")
            f.write("\n")

        f.write(f"\nTimestamp: {datetime.now()}\n")

    print(f"  Cross-model summary → {path}")
    return path

# ─────────────────────────── MAIN ──────────────────────────────────────────
def main():
    # ── CHANGE THIS to your dataset path ──
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "shared", "datasets", "Amprion.csv")

    if not os.path.exists(csv_path):
        print(f"\nAmprion.csv not found at: {csv_path}")
        sys.exit(1)

    print(f"\nLoading data from: {csv_path}")
    y_full = load_50hertz(csv_path)
    print(f"Final dataset: {len(y_full)} samples\n")

    all_params  = {}
    all_results = {}
    summaries   = {}

    for model_name in ["LSTM", "GRU", "TCN"]:
        # Stage 1
        best_params = stage1_grid_search(model_name, y_full)
        all_params[model_name] = best_params

        # Stage 2
        results = stage2_multi_size(model_name, y_full, best_params)
        all_results[model_name] = results

        # Per-model plots
        summary = plot_per_model(model_name, results)
        summaries[model_name] = summary

        # Per-model results txt
        write_results_txt(model_name, results, best_params)

    # Cross-model plots
    plot_cross_model(summaries)

    # Cross-model summary
    write_cross_model_summary(summaries, all_results, all_params)

    print("\n" + "="*60)
    print("  All done!")
    print(f"  Outputs → {BASE_OUT}/")
    print("="*60)


if __name__ == "__main__":
    main()