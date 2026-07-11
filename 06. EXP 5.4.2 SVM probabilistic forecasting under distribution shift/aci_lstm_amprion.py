"""
Amprion LSTM + ACI (Adaptive Conformal Inference)
==================================================
Reviewer Response: Distribution Shift Experiment

Train on first 5000 points, test on next 10000 points.
ACI with rolling calibration window = 500.
alpha = 0.1, gamma = 0.001.
No fine-tuning during test — plain model + ACI only.

Best hyperparams (from prior tuning, no re-tuning):
  LSTM: hidden=128, lr=1e-2, bs=32, window=24
"""

import os
import sys
import time
import warnings
from datetime import datetime

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

# ─────────────────────────── DEVICE ────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"  CUDA: {torch.version.cuda}  |  GPU: {torch.cuda.get_device_name(0)}")

# ─────────────────────────── CONFIG ────────────────────────────────────────
WINDOW      = 24
Q_LOWER     = 0.05
Q_UPPER     = 0.95
TARGET_COV  = 0.90
N_EPOCHS    = 50

TRAIN_END   = 5000     # first 5k for training
TEST_SIZE   = 10000    # next 10k for test
CAL_WINDOW  = 500      # ACI rolling calibration window
ALPHA       = 0.1
GAMMA       = 0.001

# Best hyperparams from prior tuning
HIDDEN_SIZE = 128
LR          = 1e-2
BATCH_SIZE  = 32
SEED        = 42

TOTAL_NEEDED = TRAIN_END + TEST_SIZE  # = 15000

# Output directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR    = os.path.join(SCRIPT_DIR, "aci_lstm_results")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────── DATA PIPELINE ─────────────────────────────────
# Reused from v4_amprion_big_test.py

def load_amprion(csv_path: str) -> np.ndarray:
    """Load Amprion dataset — same logic as v4_amprion_big_test.py."""
    df = pd.read_csv(csv_path)
    numeric_df = df.select_dtypes(include=[np.number])
    raw = numeric_df.values
    y = raw.flatten().astype(float)
    series = pd.Series(y)
    series = series.interpolate(method="linear", limit_direction="both")
    y = series.dropna().values
    print(f"  Total before downsample: {len(y)}")
    y = y[::2]
    print(f"  Total after  downsample: {len(y)}")
    return y


def build_windows(y: np.ndarray, win: int):
    """Build sliding windows — same logic as v4_amprion_big_test.py."""
    X, tgt = [], []
    for i in range(len(y) - win - 1):
        X.append(y[i : i + win])
        tgt.append(y[i + win])
    return np.array(X, dtype=np.float32), np.array(tgt, dtype=np.float32)


def to_loader(X, y, batch_size, shuffle=True):
    """Create DataLoader — same logic as v4_amprion_big_test.py."""
    Xt = torch.tensor(X).unsqueeze(-1)
    yt = torch.tensor(y).unsqueeze(-1)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# ─────────────────────────── MODEL ─────────────────────────────────────────
# Reused from v4_amprion_big_test.py

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# ─────────────────────────── PINBALL LOSS ──────────────────────────────────
# Reused from v4_amprion_big_test.py

class PinballLoss(nn.Module):
    def __init__(self, tau):
        super().__init__()
        self.tau = tau

    def forward(self, pred, target):
        err = target - pred
        loss = torch.where(err >= 0, self.tau * err, (self.tau - 1) * err)
        return loss.mean()


# ─────────────────────────── TRAINING ──────────────────────────────────────
# Reused from v4_amprion_big_test.py

def train_one_quantile(model, loader_tr, tau, lr, n_epochs):
    """Train a single quantile model — same logic as v4_amprion_big_test.py."""
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
    """Predict — same logic as v4_amprion_big_test.py."""
    model.eval()
    Xt = torch.tensor(X_np).unsqueeze(-1).to(DEVICE)
    with torch.no_grad():
        preds = []
        for i in range(0, len(Xt), 512):
            preds.append(model(Xt[i:i+512]).cpu().numpy().flatten())
    return np.concatenate(preds)


# ─────────────────────────── METRICS ───────────────────────────────────────

def evaluate_picp_mpiw(y_true, y_lower, y_upper):
    picp = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))
    mpiw = float(np.mean(y_upper - y_lower))
    return picp, mpiw


def find_local_picp(lower, upper, predictions, win=24):
    """Local PICP — matches MATLAB edge handling (padded, no skipping)."""
    window = win
    n = len(lower)
    local_picp = []
    half_left  = (window - 1) // 2
    half_right = window // 2

    for i in range(n):
        start = max(0, i - half_left)
        end   = min(n, i + half_right + 1)
        cur = 0
        for j in range(start, end):
            cur += (lower[j] <= predictions[j] and predictions[j] <= upper[j])
        local_picp.append(cur / (end - start))
    return local_picp


def find_lpice(local_picp, req_cov):
    """LPICE — same logic as calculate_metrics.py."""
    sum_sq = 0
    for lp in local_picp:
        shortfall = max(req_cov - lp, 0)
        sum_sq += shortfall ** 2
    return float(np.sqrt(sum_sq / len(local_picp)))


def find_accuracy(local_picp, req_cov):
    """Accuracy — same logic as calculate_metrics.py."""
    acc = sum(1 for lp in local_picp if lp >= req_cov) / len(local_picp)
    return acc


# ─────────────────────────── ACI ───────────────────────────────────────────
# Adapted from rolling.ipynb

def run_lstm_aci(X_tr_s, y_tr_s, X_te_s, y_te_s, scaler_y,
                 hidden_size, lr, batch_size, n_epochs, seed,
                 alpha, gamma, cal_window):
    """Run LSTM with ACI on test set. No fine-tuning during test."""

    torch.manual_seed(seed)

    # Build models and train on initial data
    model_lo = LSTMModel(hidden_size=hidden_size).to(DEVICE)
    model_hi = LSTMModel(hidden_size=hidden_size).to(DEVICE)

    loader_tr = to_loader(X_tr_s, y_tr_s, batch_size)

    print("  Training lower quantile model...")
    t0_train = time.time()
    model_lo = train_one_quantile(model_lo, loader_tr, Q_LOWER, lr, n_epochs)
    print("  Training upper quantile model...")
    model_hi = train_one_quantile(model_hi, loader_tr, Q_UPPER, lr, n_epochs)
    t_train = time.time() - t0_train
    print(f"  Training time: {t_train:.2f}s")

    # Get predictions on training set for initial calibration
    pred_lo_tr = predict(model_lo, X_tr_s)
    pred_hi_tr = predict(model_hi, X_tr_s)

    # Conformity scores from training set (in scaled space)
    scores_tr = np.maximum(pred_lo_tr - y_tr_s, y_tr_s - pred_hi_tr)

    # Initialize calibration window with last cal_window training scores
    cal_scores = scores_tr[-cal_window:].copy()

    # Test predictions (batch, no ACI)
    t0_predict = time.time()
    pred_lo_te = predict(model_lo, X_te_s)
    pred_hi_te = predict(model_hi, X_te_s)
    t_predict = time.time() - t0_predict
    print(f"  Batch prediction time: {t_predict:.2f}s")

    # ACI: iterate through test points one by one
    n_test = len(y_te_s)
    pred_lo_aci = np.zeros(n_test, dtype=np.float32)
    pred_hi_aci = np.zeros(n_test, dtype=np.float32)
    alpha_trajectory = np.zeros(n_test, dtype=np.float32)
    adapt_err_seq = np.zeros(n_test, dtype=np.float32)

    alphat = alpha

    t0_aci = time.time()
    for t in range(n_test):
        q_lo = pred_lo_te[t]
        q_hi = pred_hi_te[t]
        y_t = y_te_s[t]

        # ACI: compute adaptive conformal quantile
        alphat = np.clip(alphat, 1e-6, 1 - 1e-6)
        level_adapt = (1.0 - alphat) * (1.0 + 1.0 / len(cal_scores))
        level_adapt = np.clip(level_adapt, 0.0, 1.0)
        confQuant = float(np.quantile(cal_scores, level_adapt))

        # Apply conformal correction (in scaled space)
        pred_lo_aci[t] = q_lo - confQuant
        pred_hi_aci[t] = q_hi + confQuant

        # Compute conformity score
        newScore = max(q_lo - y_t, y_t - q_hi)
        adaptErr = float(newScore > confQuant)

        # Update alpha_t
        alphat_new = alphat + gamma * (alpha - adaptErr)
        alphat = np.clip(alphat_new, 1e-6, 1 - 1e-6)

        # Update rolling calibration window
        cal_scores = np.append(cal_scores[1:], newScore)

        alpha_trajectory[t] = alphat
        adapt_err_seq[t] = adaptErr
    t_aci = time.time() - t0_aci
    print(f"  ACI loop time: {t_aci:.2f}s")

    # Inverse transform to original scale
    pred_lo_raw = scaler_y.inverse_transform(pred_lo_te.reshape(-1, 1)).flatten()
    pred_hi_raw = scaler_y.inverse_transform(pred_hi_te.reshape(-1, 1)).flatten()
    pred_lo_aci_orig = scaler_y.inverse_transform(pred_lo_aci.reshape(-1, 1)).flatten()
    pred_hi_aci_orig = scaler_y.inverse_transform(pred_hi_aci.reshape(-1, 1)).flatten()
    y_te_orig = scaler_y.inverse_transform(y_te_s.reshape(-1, 1)).flatten()

    # Safety clipping (from rolling.ipynb): clip ACI intervals to ±5× training range
    y_tr_orig = scaler_y.inverse_transform(y_tr_s.reshape(-1, 1)).flatten()
    y_range = np.max(y_tr_orig) - np.min(y_tr_orig)
    safety_margin = 5 * y_range
    y_min_safe = np.min(y_tr_orig) - safety_margin
    y_max_safe = np.max(y_tr_orig) + safety_margin
    pred_lo_aci_orig = np.clip(pred_lo_aci_orig, y_min_safe, y_max_safe)
    pred_hi_aci_orig = np.clip(pred_hi_aci_orig, y_min_safe, y_max_safe)

    timing = {"train": t_train, "predict": t_predict, "aci": t_aci}
    return (pred_lo_raw, pred_hi_raw, pred_lo_aci_orig, pred_hi_aci_orig,
            y_te_orig, alpha_trajectory, adapt_err_seq, timing)


# ─────────────────────────── MAIN ──────────────────────────────────────────
def main():
    # CSV path
    csv_path = os.path.join(SCRIPT_DIR, "..", "shared", "datasets", "Amprion.csv")
    csv_path = os.path.normpath(csv_path)

    if not os.path.exists(csv_path):
        print(f"\nAmprion.csv not found at: {csv_path}")
        sys.exit(1)

    # ── Load data ──────────────────────────────────────────────────────────
    print(f"\nLoading data from: {csv_path}")
    y_full = load_amprion(csv_path)
    print(f"Final dataset: {len(y_full)} samples")

    if len(y_full) < TOTAL_NEEDED:
        print(f"ERROR: Need at least {TOTAL_NEEDED} samples; got {len(y_full)}.")
        sys.exit(1)

    # ── Split ──────────────────────────────────────────────────────────────
    y_train_raw = y_full[:TRAIN_END]
    y_test_raw  = y_full[TRAIN_END : TRAIN_END + TEST_SIZE]

    X_tr, y_tr = build_windows(y_train_raw, WINDOW)
    X_te, y_te = build_windows(y_test_raw, WINDOW)

    print(f"  train={len(X_tr)}  test={len(X_te)}")

    # ── Scale ──────────────────────────────────────────────────────────────
    scaler_X = StandardScaler().fit(X_tr)
    scaler_y = StandardScaler().fit(y_tr.reshape(-1, 1))

    X_tr_s = scaler_X.transform(X_tr)
    X_te_s = scaler_X.transform(X_te)
    y_tr_s = scaler_y.transform(y_tr.reshape(-1, 1)).flatten()
    y_te_s = scaler_y.transform(y_te.reshape(-1, 1)).flatten()

    # ── Run LSTM + ACI ─────────────────────────────────────────────────────
    print(f"\nRunning LSTM + ACI (hidden={HIDDEN_SIZE}, lr={LR}, bs={BATCH_SIZE})...")
    print(f"  ACI: cal_window={CAL_WINDOW}, alpha={ALPHA}, gamma={GAMMA}")

    t0 = time.time()
    (pred_lo_raw, pred_hi_raw, pred_lo_aci, pred_hi_aci,
     y_te_orig, alpha_traj, adapt_errs, timing) = run_lstm_aci(
        X_tr_s, y_tr_s, X_te_s, y_te_s, scaler_y,
        hidden_size=HIDDEN_SIZE, lr=LR, batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS, seed=SEED,
        alpha=ALPHA, gamma=GAMMA, cal_window=CAL_WINDOW)
    total_time = time.time() - t0

    # ── Evaluate ───────────────────────────────────────────────────────────
    picp_raw, mpiw_raw = evaluate_picp_mpiw(y_te_orig, pred_lo_raw, pred_hi_raw)
    picp_aci, mpiw_aci = evaluate_picp_mpiw(y_te_orig, pred_lo_aci, pred_hi_aci)

    local_picp_raw = find_local_picp(pred_lo_raw, pred_hi_raw, y_te_orig, 24)
    local_picp_aci = find_local_picp(pred_lo_aci, pred_hi_aci, y_te_orig, 24)

    mean_lpicp_raw = float(np.mean(local_picp_raw))
    mean_lpicp_aci = float(np.mean(local_picp_aci))

    lpice_raw = find_lpice(local_picp_raw, TARGET_COV)
    lpice_aci = find_lpice(local_picp_aci, TARGET_COV)

    acc_raw = find_accuracy(local_picp_raw, TARGET_COV)
    acc_aci = find_accuracy(local_picp_aci, TARGET_COV)

    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"{'Method':14s}  {'PICP':>8s}  {'MPIW':>10s}  {'LPICE':>8s}  {'Accuracy':>8s}")
    print(f"{'-'*54}")
    print(f"{'LSTM (raw)':14s}  {picp_raw:8.4f}  {mpiw_raw:10.4f}  {lpice_raw:8.4f}  {acc_raw:8.4f}")
    print(f"{'LSTM+ACI':14s}  {picp_aci:8.4f}  {mpiw_aci:10.4f}  {lpice_aci:8.4f}  {acc_aci:8.4f}")
    print(f"\n  Timing:")
    print(f"    Training (2 models) : {timing['train']:.2f}s")
    print(f"    Batch prediction    : {timing['predict']:.2f}s")
    print(f"    ACI loop            : {timing['aci']:.2f}s")
    print(f"    Total               : {total_time:.2f}s")

    # ── Plots ──────────────────────────────────────────────────────────────
    # 1. Test predictions: raw vs ACI
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y_te_orig,    'b-', linewidth=1.0, alpha=0.8, label='Actual')
    ax.plot(pred_lo_raw,  'r--', linewidth=0.8, alpha=0.6, label='Lower (raw)')
    ax.plot(pred_hi_raw,  'r--', linewidth=0.8, alpha=0.6, label='Upper (raw)')
    ax.plot(pred_lo_aci,  'g-', linewidth=1.0, alpha=0.8, label='Lower (ACI)')
    ax.plot(pred_hi_aci,  'g-', linewidth=1.0, alpha=0.8, label='Upper (ACI)')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Value')
    ax.set_title(f'LSTM: Raw vs ACI | PICP_raw={picp_raw:.3f} PICP_aci={picp_aci:.3f}')
    ax.legend(loc='best')
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lstm_raw_vs_aci.png"), dpi=150)
    plt.close(fig)

    # 2. Alpha trajectory
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(alpha_traj, 'b-', linewidth=1.0)
    ax.axhline(ALPHA, color='r', linestyle='--', linewidth=1.5,
               label=f'Target alpha={ALPHA}')
    ax.set_xlabel('Test Step')
    ax.set_ylabel('alpha_t')
    ax.set_title('LSTM+ACI: Alpha Trajectory')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lstm_alpha_trajectory.png"), dpi=150)
    plt.close(fig)

    # 3. Local PICP comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(local_picp_raw, 'r-', linewidth=0.8, alpha=0.7, label='LSTM (raw)')
    ax.plot(local_picp_aci, 'b-', linewidth=1.0, alpha=0.8, label='LSTM+ACI')
    ax.axhline(TARGET_COV, color='k', linestyle='--', linewidth=1.5,
               label=f'Target {TARGET_COV}')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Local PICP')
    ax.set_title(f'Local PICP | LPICE_raw={lpice_raw:.4f} LPICE_aci={lpice_aci:.4f}')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lstm_local_picp_comparison.png"), dpi=150)
    plt.close(fig)

    # 4. PICP/MPIW bar chart
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(['LSTM (raw)', 'LSTM+ACI'], [picp_raw, picp_aci],
                color=['#E05C5C', '#3A86FF'], alpha=0.8)
    axes[0].axhline(TARGET_COV, color='r', linestyle='--', linewidth=1.5)
    axes[0].set_ylabel('PICP')
    axes[0].set_title('PICP Comparison')
    axes[0].grid(True, axis='y')

    axes[1].bar(['LSTM (raw)', 'LSTM+ACI'], [mpiw_raw, mpiw_aci],
                color=['#E05C5C', '#3A86FF'], alpha=0.8)
    axes[1].set_ylabel('MPIW')
    axes[1].set_title('MPIW Comparison')
    axes[1].grid(True, axis='y')

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "lstm_picp_mpiw_bars.png"), dpi=150)
    plt.close(fig)

    print(f"\n  Plots saved to: {OUT_DIR}/")

    # ── Save results to text file ──────────────────────────────────────────
    ts = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    results_path = os.path.join(OUT_DIR, f"results_lstm_aci_{ts}.txt")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  Amprion LSTM + ACI Results\n")
        f.write("=" * 60 + "\n\n")

        f.write("Experiment Protocol: Distribution Shift (ACI)\n")
        f.write("-" * 45 + "\n")
        f.write(f"  Training set  : first {TRAIN_END} samples (indices 0–{TRAIN_END-1})\n")
        f.write(f"  Test set      : next  {TEST_SIZE} samples (indices {TRAIN_END}–{TRAIN_END+TEST_SIZE-1})\n")
        f.write(f"  ACI cal window: {CAL_WINDOW}\n")
        f.write(f"  Alpha         : {ALPHA}\n")
        f.write(f"  Gamma         : {GAMMA}\n\n")

        f.write("Hyperparameters (from prior tuning, no re-tuning)\n")
        f.write("-" * 45 + "\n")
        f.write(f"  Hidden Size : {HIDDEN_SIZE}\n")
        f.write(f"  LR          : {LR:.0e}\n")
        f.write(f"  Batch Size  : {BATCH_SIZE}\n")
        f.write(f"  Window      : {WINDOW}\n")
        f.write(f"  Epochs      : {N_EPOCHS}\n")
        f.write(f"  Seed        : {SEED}\n\n")

        f.write("Performance Summary\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Method':14s}  {'PICP':>10s}  {'MPIW':>10s}  {'LPICE':>10s}  {'Accuracy':>10s}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'LSTM (raw)':14s}  {picp_raw:10.6f}  {mpiw_raw:10.4f}  {lpice_raw:10.6f}  {acc_raw:10.4f}\n")
        f.write(f"{'LSTM+ACI':14s}  {picp_aci:10.6f}  {mpiw_aci:10.4f}  {lpice_aci:10.6f}  {acc_aci:10.4f}\n")
        f.write("-" * 70 + "\n\n")

        f.write("Timing\n")
        f.write("-" * 45 + "\n")
        f.write(f"  Training (2 models) : {timing['train']:.2f}s\n")
        f.write(f"  Batch prediction    : {timing['predict']:.2f}s\n")
        f.write(f"  ACI loop            : {timing['aci']:.2f}s\n")
        f.write(f"  Total               : {total_time:.2f}s\n\n")

        f.write("Execution completed successfully.\n")
        f.write(f"Timestamp: {datetime.now()}\n")

    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
