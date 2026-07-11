import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Datasets are bundled inside final_github/shared/datasets
DATASETS_DIR = Path(__file__).resolve().parents[1] / "shared" / "datasets"


class _QuantileMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 200):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _pinball_loss(pred: torch.Tensor, y: torch.Tensor, q: float) -> torch.Tensor:
    diff = y - pred
    return torch.mean(torch.maximum(q * diff, (q - 1) * diff))


def _np_quantile_higher(x: np.ndarray, q: float) -> float:
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    try:
        return float(np.quantile(x, q, method="higher"))
    except TypeError:
        return float(np.quantile(x, q, interpolation="higher"))


def train_two_model_cqr_conformal(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    q_low: float = 0.1,
    q_high: float = 0.9,
    alpha: float = 0.1,
    hidden_dim: int = 200,
    epochs: int = 400,
    batch_size: int = 40,
    learning_rate: float = 1e-2,
    seed: int | None = None,
    verbose: bool = False,
):
    y_train = np.asarray(y_train).reshape(-1).astype(np.float32)
    y_cal = np.asarray(y_cal).reshape(-1).astype(np.float32)
    y_test = np.asarray(y_test).reshape(-1).astype(np.float32)

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_cal_s = scaler.transform(X_cal).astype(np.float32)
    X_test_s = scaler.transform(X_test).astype(np.float32)

    device = torch.device("cpu")

    X_train_t = torch.from_numpy(X_train_s).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=batch_size,
        shuffle=True,
    )

    model_lo = _QuantileMLP(input_dim=X_train_s.shape[1], hidden_dim=hidden_dim).to(device)
    model_hi = _QuantileMLP(input_dim=X_train_s.shape[1], hidden_dim=hidden_dim).to(device)
    opt_lo = optim.Adam(model_lo.parameters(), lr=learning_rate)
    opt_hi = optim.Adam(model_hi.parameters(), lr=learning_rate)

    start_time = time.time()
    model_lo.train()
    model_hi.train()
    for _ in range(epochs):
        for xb, yb in train_loader:
            opt_lo.zero_grad()
            pred_lo = model_lo(xb)
            loss_lo = _pinball_loss(pred_lo, yb, q_low)
            loss_lo.backward()
            opt_lo.step()

            opt_hi.zero_grad()
            pred_hi = model_hi(xb)
            loss_hi = _pinball_loss(pred_hi, yb, q_high)
            loss_hi.backward()
            opt_hi.step()
    training_time = time.time() - start_time

    model_lo.eval()
    model_hi.eval()
    with torch.no_grad():
        X_cal_t = torch.from_numpy(X_cal_s).to(device)
        X_test_t = torch.from_numpy(X_test_s).to(device)
        q_lo_cal = model_lo(X_cal_t).cpu().numpy()
        q_hi_cal = model_hi(X_cal_t).cpu().numpy()
        q_lo_test = model_lo(X_test_t).cpu().numpy()
        q_hi_test = model_hi(X_test_t).cpu().numpy()

    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    q_hat = _np_quantile_higher(scores, 1 - alpha)

    lower_bound = q_lo_test - q_hat
    upper_bound = q_hi_test + q_hat

    low_cov = float(np.mean(y_test <= lower_bound))
    up_cov = float(np.mean(upper_bound >= y_test))
    coverage = float(np.mean((y_test >= lower_bound) & (y_test <= upper_bound)))
    mpiw = float(np.mean(upper_bound - lower_bound))

    if verbose:
        print(f"Low Coverage: {low_cov:.4f}")
        print(f"High Coverage: {up_cov:.4f}")
        print(f"PICP (conformal): {coverage:.4f}")
        print(f"MPIW (conformal): {mpiw:.4f}")
        print(f"Training time: {training_time:.2f} seconds\n")

    return training_time, coverage, mpiw, low_cov, up_cov, model_lo, model_hi, lower_bound, upper_bound


def _make_lag_features(y: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1)
    if y.size <= win:
        raise ValueError("Not enough data to build lag features.")
    X = np.stack([y[i : i + win] for i in range(y.size - win)], axis=0)
    t = y[win:]
    return X, t


def _load_no2() -> tuple[np.ndarray, np.ndarray]:
    p = DATASETS_DIR / "NO2.txt"
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found: {p}")
    data = np.loadtxt(p)
    if data.ndim == 1:
        X, y = _make_lag_features(data, win=3)
        return X, y
    if data.shape[1] == 1:
        X, y = _make_lag_features(data[:, 0], win=3)
        return X, y
    X = data[:, 1:]
    y = data[:, 0]
    return X, y


if __name__ == "__main__":
    X, y = _load_no2()

    q_low = 0.025
    q_high = 0.975
    alpha = 0.05

    coverages: list[float] = []
    mpiws: list[float] = []
    training_times: list[float] = []
    low_covs: list[float] = []
    up_covs: list[float] = []

    print("Running CQR model on NO2 dataset 10 times...")

    for i in range(10):
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42 + i)
        X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42 + i)

        t, coverage, mpiw, low_cov, up_cov, _, _, lower_bound, upper_bound = train_two_model_cqr_conformal(
            X_train,
            y_train,
            X_cal,
            y_cal,
            X_test,
            y_test,
            q_low=q_low,
            q_high=q_high,
            alpha=alpha,
            seed=42 + i,
            verbose=False,
        )

        training_times.append(t)
        coverages.append(coverage)
        mpiws.append(mpiw)
        low_covs.append(low_cov)
        up_covs.append(up_cov)

    mean_coverage = float(np.mean(coverages))
    std_coverage = float(np.std(coverages))
    mean_mpiw = float(np.mean(mpiws))
    std_mpiw = float(np.std(mpiws))
    mean_training_time = float(np.mean(training_times))
    mean_low_cov = float(np.mean(low_covs))
    mean_up_cov = float(np.mean(up_covs))

    print("\n=== Summary Statistics over 10 Runs ===")
    print(f"Mean PICP: {mean_coverage:.4f} ± {std_coverage:.4f}")
    print(f"Mean MPIW: {mean_mpiw:.4f} ± {std_mpiw:.4f}")
    print(f"Mean Lower Bound Coverage: {mean_low_cov:.4f}")
    print(f"Mean Upper Bound Coverage: {mean_up_cov:.4f}")
    print(f"Mean Training Time: {mean_training_time:.2f} seconds")

    y_test_1d = np.asarray(y_test).reshape(-1)
    idx = np.argsort(y_test_1d)
    sorted_y_test = y_test_1d[idx]
    sorted_lower_bound = lower_bound[idx]
    sorted_upper_bound = upper_bound[idx]

    plt.figure(figsize=(10, 6))
    x_values = np.arange(len(sorted_y_test))
    plt.scatter(x_values, sorted_y_test, color="blue", label="Actual Values", alpha=0.6)
    plt.fill_between(
        x_values,
        sorted_lower_bound,
        sorted_upper_bound,
        color="gray",
        alpha=0.3,
        label=f"Prediction Interval ({1-alpha:.0%})",
    )
    plt.title(
        f"NO2 Dataset with Conformal Prediction Intervals\n"
        f"Mean PICP: {mean_coverage:.4f} ± {std_coverage:.4f}, Mean MPIW: {mean_mpiw:.4f} ± {std_mpiw:.4f}"
    )
    plt.xlabel("Sample Index (Sorted by Target Value)")
    plt.ylabel("Target Value")
    plt.legend()
    plt.tight_layout()
    plt.show()
