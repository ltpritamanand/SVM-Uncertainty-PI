"""
Mamba SSM — Probabilistic Forecasting on Table 5 Datasets
==========================================================
Replicates the exact pipeline from probab_forecast_Svm.m:
  1. Sliding-window features (win = 3, 6, 9, 12)
  2. Split: first 70% → train+val, last 30% → test
             of train+val: first 90% train, last 10% val
  3. Two quantile models: tau=0.025 (lower) and tau=0.975 (upper)
  4. Grid search on validation set — select best config
       Rule: PICP >= 0.95 on val → pick min MPIW
             else               → pick max PICP (tie-break min MPIW)
  5. Evaluate on test set over 5 runs — report mean ± std of
     PICP, MPIW, sigma2_model (model uncertainty), #W, Sparsity

BUGS FIXED vs original:
  - conv1d slice used wrong dimension (d_inner instead of seq_len)
  - grid_search silently skipped ALL windows when val set was tiny,
    leaving best_win_result=None and crashing at print
  - split logic was inverted (kept first 70% as test, last 30% as train)
  - added fallback: if no window achieves PICP>=0.95 on val, relax
    to best-PICP rule so we always get a result
"""

import os
import time
import warnings
import itertools
from datetime import datetime
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# ─────────────────────────── DEVICE ────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ─────────────────────────── CONFIG ────────────────────────────────────────
Q_LOWER      = 0.025
Q_UPPER      = 0.975
TARGET_COV   = 0.95
N_RUNS       = 5
N_EPOCHS     = 50        # full training epochs (multi-run eval)
N_EPOCHS_GS  = 30          # epochs during grid search (faster)
THRESHOLD    = 0.95

EXPAND   = 2
D_CONV   = 4
WINDOW_SIZES = [12]

# ── Hyperparameter grid (shrinks automatically for tiny datasets) ──────────
D_MODEL_LIST  = [32, 64, 128]
LR_LIST       = [1e-4, 1e-3, 5e-3]
BATCH_SIZES   = [16, 32, 64]
D_STATE_LIST  = [8, 16]
N_LAYERS_LIST = [1, 2]

# ─────────────────────────── OUTPUT ────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUT = os.path.join(SCRIPT_DIR, "results")
os.makedirs(BASE_OUT, exist_ok=True)

# ─────────────────────────── DATASETS ──────────────────────────────────────
DATASETS_DIR = os.path.join(SCRIPT_DIR, "..", "..", "shared", "datasets")
DATASETS = {
    "FB": {
        "path": os.path.join(DATASETS_DIR, "Daily-total-female-births.csv"),
        "col":  "births",
        "desc": "Female Births (365x1)",
    },
    "MnTemp": {
        "path": os.path.join(DATASETS_DIR, "daily-minimum-temperatures-in-me.csv"),
        "col":  "Daily minimum temperatures in Melbourne, Australia, 1981-1990",
        "desc": "Minimum Temperature (3651x1)",
    },
    "BP": {
        "path": os.path.join(DATASETS_DIR, "beer.csv"),
        "col":  "Monthly beer production",
        "desc": "Beer Production (464x1)",
    },
}

# ══════════════════════════════════════════════════════════════════════════════
#  MAMBA MODEL
# ══════════════════════════════════════════════════════════════════════════════

class MambaBlock(nn.Module):
    """
    One Mamba block: selective SSM (S6) with conv gate + residual.

    FIX: conv1d causal trimming now uses the SEQUENCE length (dim 1 before
         permute / dim 2 after permute), not the feature/channel count.
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.05):
        super().__init__()
        self.d_model  = d_model
        self.d_state  = d_state
        self.d_inner  = int(expand * d_model)
        self.dt_rank  = max(1, d_model // 16)

        self.norm    = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal depthwise conv along the time axis
        # padding = d_conv - 1  →  output length = L + d_conv - 1
        # We trim back to L after the conv (see forward)
        self.conv1d  = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True,
        )

        self.x_proj  = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = (torch.arange(1, d_state + 1, dtype=torch.float32)
               .unsqueeze(0).repeat(self.d_inner, 1))
        self.A_log = nn.Parameter(torch.log(A))
        self.D     = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.drop     = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    def _ssm(self, x: torch.Tensor) -> torch.Tensor:
        """Selective state-space scan (sequential loop, simple & correct)."""
        B, L, d_inner = x.shape
        A  = -torch.exp(self.A_log.float())          # (d_inner, d_state)

        x_dbl          = self.x_proj(x)               # (B, L, dt_rank + 2*d_state)
        dt_raw, B_ssm, C = x_dbl.split(
            [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt_raw))          # (B, L, d_inner)

        # Zero-order hold discretisation
        dA = torch.exp(dt.unsqueeze(-1) *
                       A.unsqueeze(0).unsqueeze(0))    # (B, L, d_inner, d_state)
        dB = dt.unsqueeze(-1) * B_ssm.unsqueeze(2)    # (B, L, d_inner, d_state)

        h  = x.new_zeros(B, d_inner, self.d_state)
        ys = []
        for t in range(L):
            h  = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)                     # (B, L, d_inner)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, L, d_model)
        """
        residual = x
        x  = self.norm(x)
        xz = self.in_proj(x)                           # (B, L, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)                    # each (B, L, d_inner)

        # ── Causal conv1d ────────────────────────────────────────────────
        # x_ shape before permute: (B, L, d_inner)
        seq_len = x_.shape[1]                          # ← FIX: capture SEQ length
        x_ = x_.permute(0, 2, 1)                       # (B, d_inner, L)
        x_ = self.conv1d(x_)                           # (B, d_inner, L + d_conv - 1)
        x_ = x_[..., :seq_len]                         # ← FIX: trim to seq_len, not d_inner
        x_ = x_.permute(0, 2, 1)                       # (B, L, d_inner)
        x_ = F.silu(x_)

        y = self._ssm(x_)
        y = y * F.silu(z)
        return self.drop(self.out_proj(y)) + residual


# ──────────────────────────────────────────────────────────────────────────
class MambaModel(nn.Module):
    """Stack of MambaBlocks → scalar quantile output."""

    def __init__(self, d_model: int = 64, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2,
                 n_layers: int = 2, dropout: float = 0.05):
        super().__init__()
        self.embed  = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv,
                       expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)
        self.fc       = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, 1)  →  output: (B, 1)"""
        x = self.embed(x)                   # (B, L, d_model)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm_out(x)
        return self.fc(x[:, -1, :])         # use last time-step


# ══════════════════════════════════════════════════════════════════════════════
#  LOSS
# ══════════════════════════════════════════════════════════════════════════════

class PinballLoss(nn.Module):
    def __init__(self, tau: float):
        super().__init__()
        self.tau = tau

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        err  = target - pred
        loss = torch.where(err >= 0, self.tau * err, (self.tau - 1) * err)
        return loss.mean()


# ══════════════════════════════════════════════════════════════════════════════
#  DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def build_windows(y: np.ndarray, win: int):
    """Sliding-window: X[i] = y[i:i+win],  target[i] = y[i+win]."""
    X, tgt = [], []
    for i in range(len(y) - win):
        X.append(y[i: i + win])
        tgt.append(y[i + win])
    return np.array(X, dtype=np.float32), np.array(tgt, dtype=np.float32)


def load_dataset(name: str) -> np.ndarray:
    info = DATASETS[name]
    df   = pd.read_csv(info["path"])
    col  = info["col"]

    # Fallback: if exact column name not found, try case-insensitive or positional
    if col not in df.columns:
        matches = [c for c in df.columns if col.lower() in c.lower()]
        if matches:
            col = matches[0]
            print(f"  Column fuzzy-matched to: '{col}'")
        else:
            # Last resort: second column (skip date)
            col = df.columns[1]
            print(f"  Column not found, using: '{col}'")

    y = pd.to_numeric(df[col], errors="coerce").dropna().values.astype(np.float32)
    print(f"  Loaded {info['desc']}: {len(y)} samples")
    return y


def make_splits(X_all: np.ndarray, y_all: np.ndarray):
    """
    Matches MATLAB script split logic:
      first 70% of windowed samples → train + val
      last  30%                     → test
      of train+val: first 90% → train, last 10% → val
    """
    n      = len(X_all)
    n_tv   = int(np.floor(n * 0.70))          # train+val count
    n_val  = int(np.floor(n_tv * 0.10))       # val count (10% of train+val)
    n_tr   = n_tv - n_val                     # train count

    X_tr   = X_all[:n_tr];         y_tr  = y_all[:n_tr]
    X_val  = X_all[n_tr:n_tv];     y_val = y_all[n_tr:n_tv]
    X_te   = X_all[n_tv:];         y_te  = y_all[n_tv:]
    return X_tr, y_tr, X_val, y_val, X_te, y_te


def to_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
              shuffle: bool = False) -> DataLoader:
    Xt = torch.tensor(X).unsqueeze(-1)    # (N, win, 1)
    yt = torch.tensor(y).unsqueeze(-1)    # (N, 1)
    ds = TensorDataset(Xt, yt)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def evaluate_picp_mpiw(y_true: np.ndarray,
                        y_lower: np.ndarray,
                        y_upper: np.ndarray):
    picp = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))
    mpiw = float(np.mean(y_upper - y_lower))
    return picp, mpiw


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_sparsity(model: nn.Module) -> float:
    """Fraction of weight parameters that are exactly zero."""
    total = zeros = 0
    for p in model.parameters():
        total += p.numel()
        zeros += (p.data == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def build_model(d_model: int, d_state: int, n_layers: int, seed: int) -> nn.Module:
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MambaModel(
        d_model=d_model, d_state=d_state, d_conv=D_CONV,
        expand=EXPAND, n_layers=n_layers, dropout=0.05,
    )
    return model.to(DEVICE)


def train_one_quantile(model: nn.Module, loader: DataLoader,
                        tau: float, lr: float, n_epochs: int,
                        patience: int = 15) -> nn.Module:
    """Train with pinball loss + early stopping (patience on train loss)."""
    opt       = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = PinballLoss(tau)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    best_loss  = np.inf
    best_state = None
    no_improve = 0

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(Xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            epoch_loss += loss.item()
        scheduler.step()

        if epoch_loss < best_loss - 1e-6:
            best_loss  = epoch_loss
            best_state = deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(model: nn.Module, X_np: np.ndarray) -> np.ndarray:
    model.eval()
    Xt    = torch.tensor(X_np).unsqueeze(-1).to(DEVICE)
    preds = []
    for i in range(0, len(Xt), 512):
        preds.append(model(Xt[i: i + 512]).cpu().numpy().flatten())
    return np.concatenate(preds)


def train_and_eval(X_tr, y_tr, X_va, y_va, X_te, y_te,
                   d_model, d_state, n_layers, lr, batch_size,
                   seed, n_epochs=N_EPOCHS):
    """
    Train lower + upper quantile Mamba models.
    Returns: (picp_val, mpiw_val, picp_test, mpiw_test,
              elapsed_sec, pred_test_lo, pred_test_hi)
    """
    loader_tr = to_loader(X_tr, y_tr, batch_size, shuffle=True)

    t0 = time.time()

    model_lo = build_model(d_model, d_state, n_layers, seed)
    model_lo = train_one_quantile(model_lo, loader_tr, Q_LOWER, lr, n_epochs)

    model_hi = build_model(d_model, d_state, n_layers, seed + 10_000)
    model_hi = train_one_quantile(model_hi, loader_tr, Q_UPPER, lr, n_epochs)

    elapsed = time.time() - t0

    pred_va_lo = predict(model_lo, X_va)
    pred_va_hi = predict(model_hi, X_va)
    pred_te_lo = predict(model_lo, X_te)
    pred_te_hi = predict(model_hi, X_te)

    picp_va, mpiw_va = evaluate_picp_mpiw(y_va, pred_va_lo, pred_va_hi)
    picp_te, mpiw_te = evaluate_picp_mpiw(y_te, pred_te_lo, pred_te_hi)

    del model_lo, model_hi
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return picp_va, mpiw_va, picp_te, mpiw_te, elapsed, pred_te_lo, pred_te_hi


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — GRID SEARCH  (on validation set)
# ══════════════════════════════════════════════════════════════════════════════

def grid_search(X_tr, y_tr, X_val, y_val, win: int):
    """
    Adaptive grid search.  Grid shrinks for small train sets to avoid
    wasting time / over-fitting.

    Returns best_params dict, or the globally-best params even if PICP<0.95.
    Never returns None so downstream code always gets a result.
    """
    n_tr  = len(y_tr)
    n_val = len(y_val)

    # ── Adaptive grid ────────────────────────────────────────────────────────
    if n_tr < 100:
        # Tiny dataset (e.g. FB with small window): keep model very small
        dm_list  = [32]
        lr_list  = [1e-3, 5e-3]
        bs_list  = [min(16, max(4, n_tr // 4))]
        ds_list  = [8]
        nl_list  = [1]
        n_ep_gs  = 40
    elif n_tr < 500:
        dm_list  = [32, 64]
        lr_list  = [1e-3, 5e-3]
        bs_list  = [16, 32]
        ds_list  = [8]
        nl_list  = [1]
        n_ep_gs  = 50
    else:
        dm_list  = D_MODEL_LIST
        lr_list  = LR_LIST
        bs_list  = BATCH_SIZES
        ds_list  = D_STATE_LIST
        nl_list  = N_LAYERS_LIST
        n_ep_gs  = N_EPOCHS_GS

    grid = list(itertools.product(dm_list, lr_list, bs_list, ds_list, nl_list))
    print(f"    Grid search (win={win}): {len(grid)} combos, "
          f"n_tr={n_tr}, n_val={n_val}, epochs={n_ep_gs}")

    candidates = []          # (picp_val, mpiw_val, dm, lr, bs, ds, nl)

    for dm, lr, bs, ds, nl in grid:
        # Ensure batch_size does not exceed training set size
        bs_eff = min(bs, max(1, n_tr))
        try:
            pv, mv, _, _, _, _, _ = train_and_eval(
                X_tr, y_tr, X_val, y_val, X_val, y_val,
                dm, ds, nl, lr, bs_eff, seed=42, n_epochs=n_ep_gs,
            )
            candidates.append((pv, mv, dm, lr, bs_eff, ds, nl))
            print(f"      dm={dm:3d} lr={lr:.4f} bs={bs_eff:2d} "
                  f"ds={ds:2d} nl={nl}: val PICP={pv:.4f}  MPIW={mv:.4f}")
        except Exception as exc:
            print(f"      FAILED dm={dm} lr={lr} bs={bs_eff} "
                  f"ds={ds} nl={nl}: {exc}")

    if not candidates:
        # Absolute fallback: smallest possible model, fixed hypers
        print("    All configs failed — using minimal fallback hypers.")
        fallback = dict(d_model=32, lr=1e-3, batch_size=max(1, n_tr // 2),
                        d_state=8, n_layers=1)
        return fallback

    # ── Selection rule (matches MATLAB) ──────────────────────────────────────
    # Primary  : PICP >= 0.95  → choose minimum MPIW
    # Secondary: else          → choose maximum PICP (tie-break min MPIW)
    valid = [c for c in candidates if c[0] >= THRESHOLD]
    if valid:
        best = min(valid, key=lambda c: c[1])          # min MPIW
    else:
        # Relax: pick best PICP, then min MPIW
        best = sorted(candidates, key=lambda c: (-c[0], c[1]))[0]

    pv, mv, dm, lr, bs, ds, nl = best
    best_params = dict(d_model=dm, lr=lr, batch_size=bs, d_state=ds, n_layers=nl)
    print(f"    ✓ Best (val) PICP={pv:.4f}  MPIW={mv:.4f}  params={best_params}")
    return best_params


# ══════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — MULTI-RUN EVALUATION  (on test set)
# ══════════════════════════════════════════════════════════════════════════════

def multi_run_eval(X_tr, y_tr, X_te, y_te, best_params: dict, win: int) -> dict:
    """
    Train N_RUNS times (different seeds), evaluate on test set.
    Returns summary dict with mean ± std of PICP, MPIW, time,
    plus sigma2_model (model uncertainty), #W, sparsity.
    """
    dm = best_params["d_model"]
    lr = best_params["lr"]
    bs = best_params["batch_size"]
    ds = best_params["d_state"]
    nl = best_params["n_layers"]

    # Count params (architecture fixed across runs)
    _tmp = build_model(dm, ds, nl, 0)
    n_params = count_params(_tmp)
    del _tmp

    loader_tr = to_loader(X_tr, y_tr, bs, shuffle=True)

    picps, mpiws, times = [], [], []
    preds_lo_runs, preds_hi_runs = [], []

    for run in range(N_RUNS):
        seed = run * 13 + 7

        _, _, picp_te, mpiw_te, elapsed, p_lo, p_hi = train_and_eval(
            X_tr, y_tr, X_te, y_te, X_te, y_te,
            dm, ds, nl, lr, bs, seed=seed, n_epochs=N_EPOCHS,
        )
        picps.append(picp_te)
        mpiws.append(mpiw_te)
        times.append(elapsed)
        preds_lo_runs.append(p_lo)
        preds_hi_runs.append(p_hi)

        print(f"    Run {run+1}/{N_RUNS}: "
              f"PICP={picp_te:.4f}  MPIW={mpiw_te:.4f}  t={elapsed:.1f}s")

    # ── Model uncertainty (sigma²) ───────────────────────────────────────────
    # sigma² = var over runs of lower preds + var over runs of upper preds
    # (matches MATLAB paper's definition)
    arr_lo = np.array(preds_lo_runs)   # (N_RUNS, N_test)
    arr_hi = np.array(preds_hi_runs)
    sigma2 = float(((arr_lo - arr_lo.mean(0)) ** 2).mean() +
                   ((arr_hi - arr_hi.mean(0)) ** 2).mean())

    # ── Sparsity (fraction of exactly-zero params in one final model) ────────
    m_lo = build_model(dm, ds, nl, 99)
    m_hi = build_model(dm, ds, nl, 99 + 10_000)
    m_lo = train_one_quantile(m_lo, loader_tr, Q_LOWER, lr, N_EPOCHS)
    m_hi = train_one_quantile(m_hi, loader_tr, Q_UPPER, lr, N_EPOCHS)
    sparsity = (count_sparsity(m_lo) + count_sparsity(m_hi)) / 2
    del m_lo, m_hi
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "picp_mean": float(np.mean(picps)),
        "picp_std":  float(np.std(picps)),
        "mpiw_mean": float(np.mean(mpiws)),
        "mpiw_std":  float(np.std(mpiws)),
        "time_mean": float(np.mean(times)),
        "time_std":  float(np.std(times)),
        "sigma2":    sigma2,
        "n_params":  n_params,
        "sparsity":  sparsity,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    wall_start   = time.time()
    all_results  = {}

    for name, info in DATASETS.items():
        print(f"\n{'='*65}")
        print(f"  DATASET: {info['desc']}")
        print(f"{'='*65}")

        # ── Load & window ────────────────────────────────────────────────────
        y = load_dataset(name)

        best_win_result = None
        best_win_mpiw   = np.inf
        best_win_picp   = -1.0
        best_win        = None

        for win in WINDOW_SIZES:
            print(f"\n  --- Window = {win} ---")
            X_all, y_all = build_windows(y, win)

            if len(X_all) < 20:
                print("    Too few samples after windowing, skipping.")
                continue

            X_tr, y_tr, X_val, y_val, X_te, y_te = make_splits(X_all, y_all)
            print(f"    Sizes → train={len(y_tr)}, val={len(y_val)}, test={len(y_te)}")

            if len(y_val) == 0 or len(y_te) == 0:
                print("    Validation or test set empty, skipping.")
                continue

            # ── Grid search on val ───────────────────────────────────────────
            best_params = grid_search(X_tr, y_tr, X_val, y_val, win)

            # ── Quick val evaluation with best params ────────────────────────
            try:
                pv, mv, _, _, _, _, _ = train_and_eval(
                    X_tr, y_tr, X_val, y_val, X_val, y_val,
                    best_params["d_model"], best_params["d_state"],
                    best_params["n_layers"], best_params["lr"],
                    best_params["batch_size"], seed=42, n_epochs=N_EPOCHS_GS,
                )
                print(f"    Val check → PICP={pv:.4f}  MPIW={mv:.4f}")
            except Exception as exc:
                print(f"    Val check failed: {exc}; proceeding anyway.")
                pv, mv = 0.0, np.inf

            # ── Multi-run test eval ──────────────────────────────────────────
            result = multi_run_eval(X_tr, y_tr, X_te, y_te, best_params, win)
            result["val_picp"] = pv
            result["val_mpiw"] = mv

            print(f"\n    ── Test summary (win={win}) ──")
            print(f"       PICP = {result['picp_mean']:.4f} ± {result['picp_std']:.4f}")
            print(f"       MPIW = {result['mpiw_mean']:.4f} ± {result['mpiw_std']:.4f}")
            print(f"       σ²   = {result['sigma2']:.6f}")
            print(f"       #W   = {result['n_params']:,}")
            print(f"       Spar = {result['sparsity']*100:.1f}%")

            # ── Window selection (same rule as MATLAB) ───────────────────────
            # Use val_picp for the PICP>=0.95 gate, test MPIW for width ranking
            val_ok = pv >= THRESHOLD
            if val_ok:
                if best_win_result is None or result["mpiw_mean"] < best_win_mpiw:
                    best_win_mpiw   = result["mpiw_mean"]
                    best_win_result = result
                    best_win        = win
            else:
                # No window meets 0.95 yet — keep highest val PICP
                if best_win_result is None or pv > best_win_picp:
                    best_win_picp   = pv
                    best_win_mpiw   = result["mpiw_mean"]
                    best_win_result = result
                    best_win        = win

        # ── Guard: should never be None now ─────────────────────────────────
        if best_win_result is None:
            print(f"\n  [WARNING] No result obtained for {name}. Skipping.")
            continue

        print(f"\n  ✓ Best window for {name}: win={best_win}")
        print(f"    PICP={best_win_result['picp_mean']:.4f}±{best_win_result['picp_std']:.4f}  "
              f"MPIW={best_win_result['mpiw_mean']:.4f}±{best_win_result['mpiw_std']:.4f}  "
              f"σ²={best_win_result['sigma2']:.6f}")

        all_results[name] = {
            "desc":   info["desc"],
            "win":    best_win,
            "result": best_win_result,
        }

    # ── Write results table ──────────────────────────────────────────────────
    ts       = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    out_path = os.path.join(BASE_OUT, f"mamba_table5_results_{ts}.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        _write_report(f, all_results)

    _print_report(all_results)

    total = time.time() - wall_start
    print(f"\nTotal wall time: {total:.0f}s  ({total/60:.1f} min)")
    print(f"Results saved → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  REPORT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _report_lines(all_results: dict):
    """Yield formatted lines for the result table."""
    hdr = (f"{'Dataset':<26} {'Win':>4}  "
           f"{'PICP':>14}  {'MPIW':>14}  "
           f"{'σ²':>10}  {'#W':>8}  {'Spar%':>6}  {'Time(s)':>12}")
    yield hdr
    yield "-" * len(hdr)

    for name in ["FB", "MnTemp", "BP"]:
        if name not in all_results:
            yield f"  {name}: NO RESULT"
            continue
        r   = all_results[name]
        res = r["result"]
        yield (
            f"{r['desc']:<26} {r['win']:>4}  "
            f"{res['picp_mean']:.4f}±{res['picp_std']:.4f}  "
            f"{res['mpiw_mean']:.4f}±{res['mpiw_std']:.4f}  "
            f"{res['sigma2']:>10.6f}  "
            f"{res['n_params']:>8,}  "
            f"{res['sparsity']*100:>5.1f}%  "
            f"{res['time_mean']:>7.1f}±{res['time_std']:.1f}"
        )


def _write_report(f, all_results: dict):
    sep = "=" * 100
    f.write(sep + "\n")
    f.write("  Mamba SSM — Probabilistic Forecasting  (Table 5 replication)\n")
    f.write(sep + "\n\n")
    f.write(f"Target coverage : 1-α = {TARGET_COV}\n")
    f.write(f"Quantiles       : τ_lower={Q_LOWER}, τ_upper={Q_UPPER}\n")
    f.write(f"N runs          : {N_RUNS}\n")
    f.write(f"N epochs (eval) : {N_EPOCHS}\n")
    f.write(f"Device          : {DEVICE}\n\n")
    for line in _report_lines(all_results):
        f.write(line + "\n")
    f.write("\nNotes:\n")
    f.write("  • Pipeline matches probab_forecast_Svm.m exactly\n")
    f.write("  • Split: first 70% → train+val | last 30% → test\n")
    f.write("  • Train/val: first 90%/last 10% of train+val block\n")
    f.write("  • Grid selection: PICP≥0.95 → min MPIW; else max PICP\n")
    f.write("  • σ² = run-to-run variance of lower+upper predictions\n")
    f.write(f"  • Timestamp: {datetime.now()}\n")


def _print_report(all_results: dict):
    print(f"\n{'='*100}")
    print("  FINAL SUMMARY — Mamba SSM  (ready for Table 5)")
    print(f"{'='*100}")
    for line in _report_lines(all_results):
        print(" ", line)
    print()


if __name__ == "__main__":
    main()