"""
Chronos v1 Zero-Shot — Amprion (exp11, deterministic / fixed-seed)
==================================================================
Identical to chronos_data_size_sweep.py but with a global fixed seed
(seed=42) so every run produces bit-exact identical outputs.

Stochasticity in Chronos comes from the categorical sampling over bins
during autoregressive decoding. Fixing torch.manual_seed before the
rolling forecast makes all 200 trajectories reproducible.

sigma2_opt = 0 by definition (deterministic run).

Outputs -> det_outputs/
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
from chronos import ChronosPipeline

warnings.filterwarnings("ignore")

# ─────────────────────────── FIXED SEED ────────────────────────────────────
SEED         = 42
N_INIT_SEEDS = 10   # all runs use the same seed -> identical outputs -> sigma_opt = 0

# ─────────────────────────── CONFIG ────────────────────────────────────────
WINDOW       = 24
Q_LOWER      = 0.05
Q_UPPER      = 0.95
TARGET_COV   = 0.90
DATA_SIZE    = 4000
TEST_SIZE    = 1000

CHRONOS_MODEL_ID    = "amazon/chronos-t5-small"
CHRONOS_CONTEXT_LEN = 24
CHRONOS_PRED_LEN    = 64
CHRONOS_NUM_SAMPLES = 200

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_OUT   = os.path.join(SCRIPT_DIR, "det_outputs")
os.makedirs(os.path.join(BASE_OUT, "plots"),   exist_ok=True)
os.makedirs(os.path.join(BASE_OUT, "results"), exist_ok=True)


# ─────────────────────────── DATA PIPELINE ─────────────────────────────────
def load_amprion(csv_path: str) -> np.ndarray:
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


def evaluate_picp_mpiw(y_true, y_lower, y_upper):
    picp = float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))
    mpiw = float(np.mean(y_upper - y_lower))
    return picp, mpiw


# ─────────────────────────── CHRONOS V1 ROLLING FORECAST ──────────────────
def run_chronos(pipeline, y_full, run_idx=0):
    """
    Sliding test-set protocol — seed reset before every run so all
    10 runs start from the same RNG state and produce identical outputs.
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    y_chunk    = y_full[:DATA_SIZE]
    test_start = DATA_SIZE - TEST_SIZE
    n_test     = TEST_SIZE

    pred_lo = np.empty(n_test, dtype=np.float64)
    pred_hi = np.empty(n_test, dtype=np.float64)
    y_true  = np.empty(n_test, dtype=np.float64)

    t0 = time.time()

    for i in range(n_test):
        g         = test_start + i
        ctx_start = g - WINDOW
        context   = torch.tensor(y_chunk[ctx_start:g], dtype=torch.float32)

        samples = pipeline.predict(
            inputs=context,
            prediction_length=CHRONOS_PRED_LEN,
            num_samples=CHRONOS_NUM_SAMPLES,
        )

        s = samples[0, :, 0].numpy()

        y_true[i]  = float(y_chunk[g])
        pred_lo[i] = float(np.quantile(s, Q_LOWER))
        pred_hi[i] = float(np.quantile(s, Q_UPPER))

        if (i + 1) % 100 == 0:
            elapsed_so_far = time.time() - t0
            eta = (elapsed_so_far / (i + 1)) * (n_test - i - 1)
            print(f"      run={run_idx}  {i+1:>6}/{n_test} ({100*(i+1)/n_test:.1f}%)"
                  f"  [ETA: {eta:.0f}s]")

    elapsed = time.time() - t0
    picp, mpiw = evaluate_picp_mpiw(y_true, pred_lo, pred_hi)
    print(f"    run={run_idx}  PICP={picp:.4f}  MPIW={mpiw:.4f}  Time={elapsed:.1f}s")

    return {
        "picp": picp, "mpiw": mpiw, "time": elapsed,
        "n_test": n_test,
        "pred_lo": pred_lo, "pred_hi": pred_hi, "y_true": y_true,
    }


# ─────────────────────────── MAIN SWEEP ───────────────────────────────────
def run_experiment(y_full):
    print(f"\n{'='*60}")
    print(f"  Chronos v1 Zero-Shot — Amprion (exp11, deterministic)")
    print(f"  Model           : {CHRONOS_MODEL_ID}")
    print(f"  Context length  : {CHRONOS_CONTEXT_LEN} (matches window size)")
    print(f"  Pred length     : {CHRONOS_PRED_LEN} (take 1st = 1-step-ahead)")
    print(f"  Num samples     : {CHRONOS_NUM_SAMPLES}")
    print(f"  Quantiles       : tau = {Q_LOWER} / {Q_UPPER}")
    print(f"  Fixed seed      : {SEED}")
    print(f"  Data size       : {DATA_SIZE} (context pool: {DATA_SIZE - TEST_SIZE}, test: {TEST_SIZE})")
    print(f"{'='*60}")

    pipeline = ChronosPipeline.from_pretrained(
        CHRONOS_MODEL_ID,
        device_map="cuda",
        torch_dtype=torch.float32,
    )

    try:
        print(f"  Model context length : {pipeline.model_context_length}")
        print(f"  Model pred length    : {pipeline.model_prediction_length}")
    except Exception:
        pass

    print(f"\n  Running Chronos v1 x{N_INIT_SEEDS} (deterministic, seed={SEED} reset each run)...")
    all_lo   = np.zeros((N_INIT_SEEDS, TEST_SIZE), dtype=np.float64)
    all_hi   = np.zeros((N_INIT_SEEDS, TEST_SIZE), dtype=np.float64)
    picps, mpiws, times = [], [], []

    for run in range(N_INIT_SEEDS):
        print(f"\n  [run {run+1}/{N_INIT_SEEDS}]")
        r = run_chronos(pipeline, y_full, run_idx=run)
        all_lo[run] = r["pred_lo"]
        all_hi[run] = r["pred_hi"]
        picps.append(r["picp"])
        mpiws.append(r["mpiw"])
        times.append(r["time"])

    mean_lo = all_lo.mean(axis=0)
    mean_hi = all_hi.mean(axis=0)
    picp_mean, mpiw_mean = evaluate_picp_mpiw(r["y_true"], mean_lo, mean_hi)

    I = all_lo.shape[0]
    v_opt = (all_lo.var(axis=0, ddof=1).mean()
             + all_hi.var(axis=0, ddof=1).mean()) if I >= 2 else 0.0
    sigma2_opt = float(v_opt)
    sigma_opt  = float(np.sqrt(v_opt))

    print(f"\n  sigma2_opt = {sigma2_opt:.8f}  (expect ~0 — all runs identical)")
    print(f"  PICP (mean PI) = {picp_mean:.4f}  MPIW (mean PI) = {mpiw_mean:.4f}")

    return {
        "picp": picp_mean, "mpiw": mpiw_mean,
        "picp_per_run": picps, "mpiw_per_run": mpiws,
        "time": float(np.mean(times)),
        "n_test": TEST_SIZE,
        "pred_lo": mean_lo, "pred_hi": mean_hi, "y_true": r["y_true"],
        "sigma2_opt": sigma2_opt, "sigma_opt": sigma_opt,
    }


# ─────────────────────────── PLOTS ────────────────────────────────────────
def plot_results(result):
    out = f"{BASE_OUT}/plots"

    def savefig(fig, name):
        fig.tight_layout()
        fig.savefig(f"{out}/{name}", dpi=120)
        plt.close(fig)

    y_true  = result["y_true"]
    pred_lo = result["pred_lo"]
    pred_hi = result["pred_hi"]
    n_show  = min(300, len(y_true))
    idx     = np.arange(n_show)

    # 1. PI overlay (first 300 test points)
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(idx, y_true[:n_show], color="gray", lw=0.8, label="Actual", alpha=0.8)
    ax.fill_between(idx, pred_lo[:n_show], pred_hi[:n_show],
                    color="#FF6F00", alpha=0.3, label="Chronos v1 90% PI")
    ax.set_xlabel("Test Index"); ax.set_ylabel("Value")
    ax.set_title(f"Chronos v1 (zero-shot, det.) | PICP={result['picp']:.4f}  MPIW={result['mpiw']:.4f}")
    ax.legend(); ax.grid(True)
    savefig(fig, "pi_overlay_300.png")

    # 2. Full test set PI overlay
    fig, ax = plt.subplots(figsize=(16, 3.5))
    ax.plot(range(len(y_true)), y_true, color="gray", lw=0.5, label="Actual", alpha=0.6)
    ax.fill_between(range(len(pred_lo)), pred_lo, pred_hi,
                    color="#FF6F00", alpha=0.2, label="Chronos v1 90% PI")
    ax.set_xlabel("Test Index"); ax.set_ylabel("Value")
    ax.set_title(f"Chronos v1 (zero-shot, det.) | PICP={result['picp']:.4f}  MPIW={result['mpiw']:.4f}")
    ax.legend(); ax.grid(True)
    savefig(fig, "pi_overlay_full.png")

    print(f"  Plots -> {out}/")


# ─────────────────────────── RESULTS TXT ──────────────────────────────────
def write_results(result):
    ts   = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    path = f"{BASE_OUT}/results/results_chronos_det_{ts}.txt"

    with open(path, "w", encoding="utf-8") as f:
        f.write("="*70 + "\n")
        f.write("  Amprion -- Chronos v1 Zero-Shot (exp11, deterministic)\n")
        f.write("="*70 + "\n\n")

        f.write("Experiment Configuration\n")
        f.write("-"*50 + "\n")
        f.write(f"  Model           : {CHRONOS_MODEL_ID}\n")
        f.write(f"  Context Length  : {CHRONOS_CONTEXT_LEN} (matches window size)\n")
        f.write(f"  Pred Length     : {CHRONOS_PRED_LEN} (take 1st = 1-step-ahead)\n")
        f.write(f"  Num Samples     : {CHRONOS_NUM_SAMPLES}\n")
        f.write(f"  Training        : NONE (zero-shot)\n")
        f.write(f"  Fixed Seed      : {SEED}\n")
        f.write(f"  Window          : {WINDOW}\n")
        f.write(f"  Quantiles       : tau = {Q_LOWER} / {Q_UPPER}\n")
        f.write(f"  Target Coverage : {TARGET_COV}\n")
        f.write(f"  Data Size       : {DATA_SIZE} (context: {DATA_SIZE - TEST_SIZE}, test: {TEST_SIZE})\n\n")

        f.write("Chronos v1 Architecture Notes\n")
        f.write("-"*50 + "\n")
        f.write("  Chronos v1 is a GENERATIVE autoregressive model (T5-based).\n")
        f.write("  It generates num_samples stochastic trajectories by sampling\n")
        f.write("  from a categorical distribution over bins at each step.\n")
        f.write("  Quantiles are extracted from the empirical sample distribution.\n")
        f.write("  Fixed seed makes the sampling fully reproducible.\n\n")

        f.write("Protocol: sliding window, 1-step-ahead from 64-step forecast\n")
        f.write("  Context = 24 points (matches window size of all other models).\n")
        f.write("  Predict 64 steps, take 1st prediction only.\n")
        f.write("  Slide by 1 step, repeat for all test points.\n")
        f.write("  sigma2_opt = 0 (deterministic — fixed seed).\n\n")

        f.write("Performance Summary\n")
        f.write("-"*70 + "\n")
        f.write(f"  Test PICP   : {result['picp']:.4f}  (mean PI over {N_INIT_SEEDS} runs)\n")
        f.write(f"  Test MPIW   : {result['mpiw']:.4f}\n")
        f.write(f"  MPIW/PICP   : {result['mpiw'] / max(result['picp'], 1e-6):.4f}\n")
        f.write(f"  Inference   : {result['time']:.1f}s/run\n")
        f.write(f"  n_test      : {result['n_test']}\n")
        f.write(f"  N_runs      : {N_INIT_SEEDS}\n")
        f.write(f"  Fixed seed  : {SEED} (reset before each run)\n")
        f.write(f"  sigma2_opt  : {result['sigma2_opt']:.8f}  (expect ~0 — all runs identical)\n")
        f.write(f"  sigma_opt   : {result['sigma_opt']:.8f}\n")
        f.write(f"  sigma2_in   : 0.0  (zero-shot — no training-data variation)\n")
        f.write(f"  sigma2_model: {result['sigma2_opt']:.8f}\n\n")
        f.write("Per-run PICP / MPIW\n")
        f.write("-"*40 + "\n")
        for i, (p, m) in enumerate(zip(result['picp_per_run'], result['mpiw_per_run'])):
            f.write(f"  run {i:2d}: PICP={p:.4f}  MPIW={m:.4f}\n")
        f.write("\n")

        f.write(f"Timestamp: {datetime.now()}\n")

    print(f"  Results -> {path}")

    csv_path = f"{BASE_OUT}/results/results_chronos_det_{ts}.csv"
    rows = [{
        "data_size": DATA_SIZE,
        "seed": SEED,
        "n_runs": N_INIT_SEEDS,
        "test_picp": round(result["picp"], 6),
        "test_mpiw": round(result["mpiw"], 6),
        "mpiw_picp_ratio": round(result["mpiw"] / max(result["picp"], 1e-6), 6),
        "inference_time_s": round(result["time"], 2),
        "n_test": result["n_test"],
        "sigma2_opt":   round(result["sigma2_opt"], 8),
        "sigma_opt":    round(result["sigma_opt"],  8),
        "sigma2_in":    0.0,
        "sigma2_model": round(result["sigma2_opt"], 8),
    }]
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    print(f"  CSV    -> {csv_path}")


# ─────────────────────────── MAIN ─────────────────────────────────────────
def main():
    csv_path = os.path.join(SCRIPT_DIR, "..", "shared", "datasets", "Amprion.csv")

    if not os.path.exists(csv_path):
        print(f"\nAmprion.csv not found at: {csv_path}")
        sys.exit(1)

    print(f"\nLoading data from: {csv_path}")
    y_full = load_amprion(csv_path)
    print(f"Final dataset: {len(y_full)} samples")

    if len(y_full) < DATA_SIZE:
        print(f"ERROR: Need at least {DATA_SIZE} samples; got {len(y_full)}.")
        sys.exit(1)

    result = run_experiment(y_full)
    plot_results(result)
    write_results(result)

    print(f"\n{'='*60}")
    print(f"  Done!  Outputs -> {BASE_OUT}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
