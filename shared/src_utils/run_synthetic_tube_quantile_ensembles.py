#!/usr/bin/env python3
"""
Synthetic tube experiment (seed=58, n=3000):
  y = sin(x)/x + noise, x ~ U[-2π, 2π], noise ~ U[-1, 1] (or Normal)

Runs:
  - Ensemble "SVQR" (QuantileRegressor on RBF random features) for lower/upper quantiles
  - Ensemble NN (MLP) for lower/upper quantiles with pinball loss

Produces:
  - results_new/synthetic_tube/{x_grid,y_grid_clean,svr_lower_grid,svr_upper_grid,svr_width_grid,nn_lower_grid,nn_upper_grid,nn_width_grid}.npy
  - intervals_on_grid.png, width_on_grid.png
  - summary.txt (coverage/width + simple uncertainty stats)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _repo_root() -> Path:
    # final_github/shared/src_utils/this_file.py -> parents[2] = final_github
    return Path(__file__).resolve().parents[2]


def generate_sinc_dataset(
    *,
    seed: int,
    n_samples: int,
    noise: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns: X (n,1), y_noisy (n,), y_clean (n,)
    """
    rng = np.random.default_rng(seed)
    # x in [-6, 6] matches paper Figure 1 and exp8 / exp9 local generators.
    X = rng.uniform(low=-6.0, high=6.0, size=(n_samples, 1))
    # sin(x)/x with safe handling at x=0:
    # np.sinc(z) = sin(pi z)/(pi z) => sin(x)/x = np.sinc(x/pi)
    y_clean = np.sinc(X[:, 0] / np.pi)
    if noise == "uniform":
        eps = rng.uniform(low=-1.0, high=1.0, size=n_samples)
    elif noise == "normal":
        eps = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    else:
        raise ValueError(f"noise must be 'uniform' or 'normal', got: {noise!r}")
    y_noisy = y_clean + eps
    return X, y_noisy, y_clean


def split_train_cal_test(
    X: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    train_frac: float = 0.6,
    cal_frac: float = 0.2,
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray], np.ndarray]:
    """
    Random split into train/cal/test (train_frac, cal_frac, rest test).
    Returns (X_train,y_train),(X_cal,y_cal),(X_test,y_test), permutation indices used.
    """
    assert X.shape[0] == y.shape[0]
    n = X.shape[0]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(np.floor(train_frac * n))
    n_cal = int(np.floor(cal_frac * n))
    train_idx = perm[:n_train]
    cal_idx = perm[n_train : n_train + n_cal]
    test_idx = perm[n_train + n_cal :]
    return (X[train_idx], y[train_idx]), (X[cal_idx], y[cal_idx]), (X[test_idx], y[test_idx]), perm


def picp_mpiw(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    y = y.reshape(-1)
    lo = lo.reshape(-1)
    hi = hi.reshape(-1)
    inside = (y >= lo) & (y <= hi)
    return float(np.mean(inside)), float(np.mean(hi - lo))


def calibrate_scale_for_target_picp(
    y_cal: np.ndarray,
    lo_cal: np.ndarray,
    hi_cal: np.ndarray,
    *,
    target_picp: float,
    max_scale: float = 10.0,
    iters: int = 60,
) -> float:
    """
    Finds scale s such that intervals [c - s*w, c + s*w] achieve PICP ~= target on calibration set,
    where c=(lo+hi)/2 and w=(hi-lo)/2.
    """
    y_cal = y_cal.reshape(-1)
    c = 0.5 * (lo_cal.reshape(-1) + hi_cal.reshape(-1))
    w = 0.5 * (hi_cal.reshape(-1) - lo_cal.reshape(-1))
    w = np.maximum(w, 1e-12)

    def coverage(s: float) -> float:
        lo_s = c - s * w
        hi_s = c + s * w
        inside = (y_cal >= lo_s) & (y_cal <= hi_s)
        return float(np.mean(inside))

    # Monotone in s. Binary search.
    lo_s, hi_s = 0.0, max_scale
    if coverage(hi_s) < target_picp:
        return hi_s
    for _ in range(iters):
        mid = 0.5 * (lo_s + hi_s)
        if coverage(mid) >= target_picp:
            hi_s = mid
        else:
            lo_s = mid
    return hi_s


@dataclass(frozen=True)
class EnsemblePred:
    lower_members: np.ndarray  # (E, N)
    upper_members: np.ndarray  # (E, N)

    @property
    def center_members(self) -> np.ndarray:
        return 0.5 * (self.lower_members + self.upper_members)

    def aggregate_mean_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.mean(self.lower_members, axis=0)
        hi = np.mean(self.upper_members, axis=0)
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        return lo2, hi2

    def mean_pred_var(self) -> float:
        # Epistemic proxy: mean over x of Var(center across ensemble)
        return float(np.mean(np.var(self.center_members, axis=0)))


class _IdentityScaler:
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return X

    def transform(self, X: np.ndarray) -> np.ndarray:
        return X


@dataclass
class SVQRMember:
    scaler: object
    rff: object
    qr_l: object
    qr_u: object

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        Xe_s = self.scaler.transform(X)
        Xe_phi = self.rff.transform(Xe_s)
        lo = self.qr_l.predict(Xe_phi)
        hi = self.qr_u.predict(Xe_phi)
        return np.minimum(lo, hi), np.maximum(lo, hi)


def train_svqr_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    n_ens: int,
    q_lower: float,
    q_upper: float,
    n_rff: int,
    rff_gamma: float,
    alpha: float,
    bootstrap: bool,
    standardize: bool,
) -> list[SVQRMember]:
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.linear_model import QuantileRegressor
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    members: list[SVQRMember] = []

    for m in range(n_ens):
        boot_idx = rng.integers(low=0, high=n, size=n) if bootstrap else np.arange(n)
        Xb = X_train[boot_idx]
        yb = y_train[boot_idx]

        scaler = StandardScaler() if standardize else _IdentityScaler()
        Xb_s = scaler.fit_transform(Xb)

        # If bootstrap=False and n_ens>1, keep random_state fixed so repeats are identical.
        rff_state = (seed + m) if bootstrap else seed
        rff = RBFSampler(gamma=rff_gamma, n_components=n_rff, random_state=rff_state)
        Xb_phi = rff.fit_transform(Xb_s)

        qr_l = QuantileRegressor(quantile=q_lower, alpha=alpha, solver="highs")
        qr_u = QuantileRegressor(quantile=q_upper, alpha=alpha, solver="highs")
        qr_l.fit(Xb_phi, yb)
        qr_u.fit(Xb_phi, yb)

        members.append(SVQRMember(scaler=scaler, rff=rff, qr_l=qr_l, qr_u=qr_u))

    return members


def predict_svqr_ensemble(members: list[SVQRMember], X_eval: np.ndarray) -> EnsemblePred:
    lower_members = []
    upper_members = []
    for mem in members:
        lo, hi = mem.predict(X_eval)
        lower_members.append(lo)
        upper_members.append(hi)
    return EnsemblePred(lower_members=np.stack(lower_members, axis=0), upper_members=np.stack(upper_members, axis=0))


@dataclass
class NNMember:
    scaler: object
    model: object

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import torch

        Xe_s = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            Xe_t = torch.tensor(Xe_s, dtype=torch.float32)
            out = self.model(Xe_t).cpu().numpy()
        lo = np.minimum(out[:, 0], out[:, 1])
        hi = np.maximum(out[:, 0], out[:, 1])
        return lo, hi


def train_nn_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    n_ens: int,
    q_lower: float,
    q_upper: float,
    hidden_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    standardize: bool,
) -> list[NNMember]:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    class QuantileNet(nn.Module):
        def __init__(self, input_dim: int, hidden_dim_: int):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, hidden_dim_)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_dim_, 2)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    def pinball_loss(preds, target, q: float):
        diff = target - preds
        return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))

    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    members: list[NNMember] = []

    for m in range(n_ens):
        boot_idx = rng.integers(low=0, high=n, size=n)
        Xb = X_train[boot_idx]
        yb = y_train[boot_idx]

        scaler = StandardScaler() if standardize else _IdentityScaler()
        Xb_s = scaler.fit_transform(Xb)

        torch.manual_seed(seed + m)
        np.random.seed(seed + m)

        Xb_t = torch.tensor(Xb_s, dtype=torch.float32)
        yb_t = torch.tensor(yb.reshape(-1, 1), dtype=torch.float32)
        ds = TensorDataset(Xb_t, yb_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = QuantileNet(input_dim=Xb_s.shape[1], hidden_dim_=hidden_dim)
        opt = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for _ in range(epochs):
            for xb, ytrue in dl:
                opt.zero_grad()
                out = model(xb)
                lo = out[:, 0]
                hi = out[:, 1]
                loss = pinball_loss(lo, ytrue.squeeze(1), q_lower) + pinball_loss(hi, ytrue.squeeze(1), q_upper)
                loss.backward()
                opt.step()

        members.append(NNMember(scaler=scaler, model=model))

    return members


def predict_nn_ensemble(members: list[NNMember], X_eval: np.ndarray) -> EnsemblePred:
    lower_members = []
    upper_members = []
    for mem in members:
        lo, hi = mem.predict(X_eval)
        lower_members.append(lo)
        upper_members.append(hi)
    return EnsemblePred(lower_members=np.stack(lower_members, axis=0), upper_members=np.stack(upper_members, axis=0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=58)
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--noise", type=str, default="uniform", choices=["uniform", "normal"])
    ap.add_argument("--n-ens", type=int, default=10, help="Ensemble size for NN. SVQR defaults to single deterministic fit.")
    ap.add_argument("--target-picp", type=float, default=0.90)
    ap.add_argument("--q-lower", type=float, default=0.05)
    ap.add_argument("--q-upper", type=float, default=0.95)
    ap.add_argument("--xgrid-min", type=float, default=-6.0)
    ap.add_argument("--xgrid-max", type=float, default=6.0)
    ap.add_argument("--xgrid-n", type=int, default=3000)
    # SVQR/RFF params
    ap.add_argument("--svr-mode", type=str, default="single", choices=["single", "ensemble"])
    ap.add_argument("--svr-n-rff", type=int, default=800)
    ap.add_argument("--svr-rff-gamma", type=float, default=2.0)
    ap.add_argument("--svr-alpha", type=float, default=1e-4)
    ap.add_argument("--svr-standardize", action="store_true", help="Standardize X for SVQR (off by default).")
    # NN params
    ap.add_argument("--nn-hidden", type=int, default=64)
    ap.add_argument("--nn-epochs", type=int, default=300)
    ap.add_argument("--nn-batch", type=int, default=256)
    ap.add_argument("--nn-lr", type=float, default=1e-3)
    ap.add_argument("--nn-no-standardize", action="store_true", help="Disable StandardScaler for NN inputs.")
    args = ap.parse_args()

    np.random.seed(args.seed)

    X, y_noisy, y_clean = generate_sinc_dataset(seed=args.seed, n_samples=args.n_samples, noise=args.noise)

    x_grid = np.linspace(args.xgrid_min, args.xgrid_max, args.xgrid_n)
    X_grid = x_grid.reshape(-1, 1)
    y_grid_clean = np.sinc(x_grid / np.pi)

    (X_train, y_train), (X_cal, y_cal), (X_test, y_test), _ = split_train_cal_test(
        X, y_noisy, seed=args.seed, train_frac=0.6, cal_frac=0.2
    )

    # Keep clean targets for reporting
    y_test_clean = np.sinc(X_test[:, 0] / np.pi)

    out_dir = _repo_root() / "results_new" / "synthetic_tube"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- SVQR (default: single deterministic; optional ensemble) ---
    svr_n = 1 if args.svr_mode == "single" else args.n_ens
    svr_members = train_svqr_ensemble(
        X_train,
        y_train,
        seed=args.seed,
        n_ens=svr_n,
        q_lower=args.q_lower,
        q_upper=args.q_upper,
        n_rff=args.svr_n_rff,
        rff_gamma=args.svr_rff_gamma,
        alpha=args.svr_alpha,
        bootstrap=(args.svr_mode == "ensemble"),
        standardize=args.svr_standardize,
    )
    svr_grid = predict_svqr_ensemble(svr_members, X_grid)
    svr_cal = predict_svqr_ensemble(svr_members, X_cal)
    svr_test = predict_svqr_ensemble(svr_members, X_test)

    svr_lo_grid, svr_hi_grid = svr_grid.aggregate_mean_bounds()
    svr_lo_cal, svr_hi_cal = svr_cal.aggregate_mean_bounds()
    svr_lo_test, svr_hi_test = svr_test.aggregate_mean_bounds()

    svr_scale = calibrate_scale_for_target_picp(y_cal, svr_lo_cal, svr_hi_cal, target_picp=args.target_picp)
    svr_c_grid = 0.5 * (svr_lo_grid + svr_hi_grid)
    svr_w_grid = 0.5 * (svr_hi_grid - svr_lo_grid)
    svr_lower_grid = svr_c_grid - svr_scale * svr_w_grid
    svr_upper_grid = svr_c_grid + svr_scale * svr_w_grid

    svr_c_test = 0.5 * (svr_lo_test + svr_hi_test)
    svr_w_test = 0.5 * (svr_hi_test - svr_lo_test)
    svr_lower_test = svr_c_test - svr_scale * svr_w_test
    svr_upper_test = svr_c_test + svr_scale * svr_w_test

    svr_picp_cal, svr_mpiw_cal = picp_mpiw(
        y_cal,
        0.5 * (svr_lo_cal + svr_hi_cal) - svr_scale * (0.5 * (svr_hi_cal - svr_lo_cal)),
        0.5 * (svr_lo_cal + svr_hi_cal) + svr_scale * (0.5 * (svr_hi_cal - svr_lo_cal)),
    )
    svr_picp_test, svr_mpiw_test = picp_mpiw(y_test, svr_lower_test, svr_upper_test)

    # --- Ensemble NN (pinball loss) ---
    nn_members = train_nn_ensemble(
        X_train,
        y_train,
        seed=args.seed,
        n_ens=args.n_ens,
        q_lower=args.q_lower,
        q_upper=args.q_upper,
        hidden_dim=args.nn_hidden,
        epochs=args.nn_epochs,
        batch_size=args.nn_batch,
        lr=args.nn_lr,
        standardize=not args.nn_no_standardize,
    )
    nn_grid = predict_nn_ensemble(nn_members, X_grid)
    nn_cal = predict_nn_ensemble(nn_members, X_cal)
    nn_test = predict_nn_ensemble(nn_members, X_test)

    nn_lo_grid, nn_hi_grid = nn_grid.aggregate_mean_bounds()
    nn_lo_cal, nn_hi_cal = nn_cal.aggregate_mean_bounds()
    nn_lo_test, nn_hi_test = nn_test.aggregate_mean_bounds()

    nn_scale = calibrate_scale_for_target_picp(y_cal, nn_lo_cal, nn_hi_cal, target_picp=args.target_picp)
    nn_c_grid = 0.5 * (nn_lo_grid + nn_hi_grid)
    nn_w_grid = 0.5 * (nn_hi_grid - nn_lo_grid)
    nn_lower_grid = nn_c_grid - nn_scale * nn_w_grid
    nn_upper_grid = nn_c_grid + nn_scale * nn_w_grid

    nn_c_test = 0.5 * (nn_lo_test + nn_hi_test)
    nn_w_test = 0.5 * (nn_hi_test - nn_lo_test)
    nn_lower_test = nn_c_test - nn_scale * nn_w_test
    nn_upper_test = nn_c_test + nn_scale * nn_w_test

    nn_picp_cal, nn_mpiw_cal = picp_mpiw(
        y_cal,
        0.5 * (nn_lo_cal + nn_hi_cal) - nn_scale * (0.5 * (nn_hi_cal - nn_lo_cal)),
        0.5 * (nn_lo_cal + nn_hi_cal) + nn_scale * (0.5 * (nn_hi_cal - nn_lo_cal)),
    )
    nn_picp_test, nn_mpiw_test = picp_mpiw(y_test, nn_lower_test, nn_upper_test)

    # Save arrays
    np.save(out_dir / "x_grid.npy", x_grid)
    np.save(out_dir / "y_grid_clean.npy", y_grid_clean)

    np.save(out_dir / "svr_lower_grid.npy", svr_lower_grid)
    np.save(out_dir / "svr_upper_grid.npy", svr_upper_grid)
    np.save(out_dir / "svr_width_grid.npy", svr_upper_grid - svr_lower_grid)

    np.save(out_dir / "nn_lower_grid.npy", nn_lower_grid)
    np.save(out_dir / "nn_upper_grid.npy", nn_upper_grid)
    np.save(out_dir / "nn_width_grid.npy", nn_upper_grid - nn_lower_grid)

    # Plots
    plt.figure(figsize=(10, 4))
    plt.plot(x_grid, y_grid_clean, "k-", linewidth=1.5, label="clean y = sin(x)/x")
    svr_label = "SVQR (single)" if svr_n <= 1 else f"SVQR ensemble (n={svr_n})"
    plt.fill_between(x_grid, svr_lower_grid, svr_upper_grid, alpha=0.25, label=svr_label)
    plt.fill_between(x_grid, nn_lower_grid, nn_upper_grid, alpha=0.25, label=f"NN ensemble (n={args.n_ens})")
    plt.xlim([x_grid.min(), x_grid.max()])
    plt.legend(loc="best")
    plt.title("Prediction intervals on grid")
    plt.tight_layout()
    plt.savefig(out_dir / "intervals_on_grid.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(x_grid, svr_upper_grid - svr_lower_grid, label="SVQR width")
    plt.plot(x_grid, nn_upper_grid - nn_lower_grid, label="NN width")
    plt.xlim([x_grid.min(), x_grid.max()])
    plt.legend(loc="best")
    plt.title("Interval width on grid (final uncertainty)")
    plt.tight_layout()
    plt.savefig(out_dir / "width_on_grid.png", dpi=160)
    plt.close()

    # Summary
    def stats_lines(
        name: str,
        *,
        n_models: int,
        picp_cal: float,
        mpiw_cal: float,
        scale: float,
        picp_test: float,
        mpiw_test: float,
        y_test_ref: np.ndarray,
        mid_test: np.ndarray,
        pred_var: float,
    ) -> str:
        err = mid_test - y_test_ref.reshape(-1)
        bias = float(np.mean(err))
        err_std = float(np.std(err))
        return (
            f"{name} (n={n_models})\n"
            f"  cal:  PICP={picp_cal:.4f}, MPIW={mpiw_cal:.4f}, scale={scale:.4f}\n"
            f"  test: PICP={picp_test:.4f}, MPIW={mpiw_test:.4f}\n"
            f"  test stats vs {'clean' if y_test_ref is y_test_clean else 'noisy'} y: bias={bias:.4f}, err_std={err_std:.4f}, pred_var={pred_var:.4f}\n"
        )

    svr_mid_test = 0.5 * (svr_lower_test + svr_upper_test)
    nn_mid_test = 0.5 * (nn_lower_test + nn_upper_test)

    # Use ensemble-member centers for epistemic proxy
    svr_pred_var = 0.0 if svr_n <= 1 else svr_test.mean_pred_var()
    nn_pred_var = nn_test.mean_pred_var()

    summary = []
    summary.append(f"Target PICP: {args.target_picp:.2f}\n\n")
    summary.append(
        stats_lines(
            "Quantile-loss SVQR",
            n_models=svr_n,
            picp_cal=svr_picp_cal,
            mpiw_cal=svr_mpiw_cal,
            scale=svr_scale,
            picp_test=svr_picp_test,
            mpiw_test=svr_mpiw_test,
            y_test_ref=y_test,
            mid_test=svr_mid_test,
            pred_var=svr_pred_var,
        )
    )
    summary.append(
        stats_lines(
            "Quantile-loss SVQR",
            n_models=svr_n,
            picp_cal=svr_picp_cal,
            mpiw_cal=svr_mpiw_cal,
            scale=svr_scale,
            picp_test=svr_picp_test,
            mpiw_test=svr_mpiw_test,
            y_test_ref=y_test_clean,
            mid_test=svr_mid_test,
            pred_var=svr_pred_var,
        )
    )
    summary.append("\n")
    summary.append(
        stats_lines(
            "Quantile-loss NN ensemble",
            n_models=args.n_ens,
            picp_cal=nn_picp_cal,
            mpiw_cal=nn_mpiw_cal,
            scale=nn_scale,
            picp_test=nn_picp_test,
            mpiw_test=nn_mpiw_test,
            y_test_ref=y_test,
            mid_test=nn_mid_test,
            pred_var=nn_pred_var,
        )
    )
    summary.append(
        stats_lines(
            "Quantile-loss NN ensemble",
            n_models=args.n_ens,
            picp_cal=nn_picp_cal,
            mpiw_cal=nn_mpiw_cal,
            scale=nn_scale,
            picp_test=nn_picp_test,
            mpiw_test=nn_mpiw_test,
            y_test_ref=y_test_clean,
            mid_test=nn_mid_test,
            pred_var=nn_pred_var,
        )
    )
    summary.append(f"\nSaved arrays to: {out_dir}\n")

    (out_dir / "summary.txt").write_text("".join(summary))

    print("".join(summary))


if __name__ == "__main__":
    main()

