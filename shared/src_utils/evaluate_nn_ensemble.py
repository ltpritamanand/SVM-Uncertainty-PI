
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

current_file = Path(__file__).resolve()
src_dir = current_file.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import run_synthetic_tube_quantile_ensembles as experiment

def picp_mpiw(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[float, float]:
    y = y.reshape(-1)
    lo = lo.reshape(-1)
    hi = hi.reshape(-1)
    inside = (y >= lo) & (y <= hi)
    return float(np.mean(inside)), float(np.mean(hi - lo))


def interval_score(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = 0.10) -> float:
    """Interval Score (IS) for a (1-alpha)*100% prediction interval.

    IS = (u - l) + (2/alpha) * max(l - y, 0) + (2/alpha) * max(y - u, 0)

    Lower is better. Proper scoring rule — jointly evaluates calibration and sharpness.
    """
    y = y.reshape(-1)
    lo = lo.reshape(-1)
    hi = hi.reshape(-1)
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    penalty = (2.0 / alpha) * (below + above)
    return float(np.mean(width + penalty))


def _sample_var(x: np.ndarray, axis: int) -> np.ndarray:
    ddof = 1 if x.shape[axis] > 1 else 0
    return np.var(x, axis=axis, ddof=ddof)


def _bounds_stats(lower_members: np.ndarray, upper_members: np.ndarray, *, variance_inflation: float = 1.96) -> dict:
    mean_lower = np.mean(lower_members, axis=0)
    mean_upper = np.mean(upper_members, axis=0)
    var_lower = _sample_var(lower_members, axis=0)
    std_lower = np.std(lower_members, axis=0, ddof=1 if lower_members.shape[0] > 1 else 0)
    var_upper = _sample_var(upper_members, axis=0)
    std_upper = np.std(upper_members, axis=0, ddof=1 if upper_members.shape[0] > 1 else 0)

    agg_lower = mean_lower - variance_inflation * var_lower
    agg_upper = mean_upper + variance_inflation * var_upper

    center_members = 0.5 * (lower_members + upper_members)
    var_center = _sample_var(center_members, axis=0)
    std_center = np.std(center_members, axis=0, ddof=1 if center_members.shape[0] > 1 else 0)

    width_members = upper_members - lower_members
    mean_width = np.mean(width_members, axis=0)
    std_width = np.std(width_members, axis=0, ddof=1 if width_members.shape[0] > 1 else 0)
    var_width = _sample_var(width_members, axis=0)

    return {
        "mean_lower": mean_lower,
        "mean_upper": mean_upper,
        "agg_lower": agg_lower,
        "agg_upper": agg_upper,
        "var_lower": var_lower,
        "std_lower": std_lower,
        "var_upper": var_upper,
        "std_upper": std_upper,
        "var_center": var_center,
        "std_center": std_center,
        "mean_width": mean_width,
        "std_width": std_width,
        "var_width": var_width,
    }


def _train_nn(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_ens: int,
    q_lower: float,
    q_upper: float,
    hidden_sizes: list[int],
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    standardize: bool,
) -> list:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.preprocessing import StandardScaler
    from torch.utils.data import DataLoader, TensorDataset

    def pinball_loss(preds, target, q: float):
        diff = target - preds
        return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))

    class QuantileNet(nn.Module):
        def __init__(self, input_dim: int, hidden_sizes_: list[int], dropout_: float):
            super().__init__()
            layers: list[nn.Module] = []
            d = input_dim
            for h in hidden_sizes_:
                layers.append(nn.Linear(d, h))
                layers.append(nn.ReLU())
                if dropout_ > 0:
                    layers.append(nn.Dropout(p=dropout_))
                d = h
            layers.append(nn.Linear(d, 2))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    members: list[experiment.NNMember] = []

    for m in range(n_ens):
        print(f"  [NN] Training member {m+1}/{n_ens}...")
        boot_idx = rng.integers(low=0, high=n, size=n)
        Xb = X_train[boot_idx]
        yb = y_train[boot_idx]

        scaler = StandardScaler() if standardize else experiment._IdentityScaler()
        Xb_s = scaler.fit_transform(Xb)

        torch.manual_seed(seed + m)
        np.random.seed(seed + m)

        Xb_t = torch.tensor(Xb_s, dtype=torch.float32)
        yb_t = torch.tensor(yb.reshape(-1, 1), dtype=torch.float32)
        ds = TensorDataset(Xb_t, yb_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        model = QuantileNet(input_dim=Xb_s.shape[1], hidden_sizes_=hidden_sizes, dropout_=dropout)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

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

        members.append(experiment.NNMember(scaler=scaler, model=model))

    return members


class QRFMember:
    def __init__(self, rf, *, per_tree_leaf_to_y: list[dict[int, np.ndarray]]):
        self.rf = rf
        self.per_tree_leaf_to_y = per_tree_leaf_to_y

    def predict_quantiles(self, X: np.ndarray, *, q_lower: float, q_upper: float) -> tuple[np.ndarray, np.ndarray]:
        lo = np.empty((X.shape[0],), dtype=float)
        hi = np.empty((X.shape[0],), dtype=float)

        for i in range(X.shape[0]):
            ys = []
            x1 = X[i : i + 1]
            for tree, leaf_to_y in zip(self.rf.estimators_, self.per_tree_leaf_to_y):
                leaf = int(tree.apply(x1)[0])
                y_leaf = leaf_to_y.get(leaf)
                if y_leaf is not None and y_leaf.size:
                    ys.append(y_leaf)
            if ys:
                y_all = np.concatenate(ys, axis=0)
                lo[i] = float(np.quantile(y_all, q_lower))
                hi[i] = float(np.quantile(y_all, q_upper))
            else:
                pred = float(self.rf.predict(x1)[0])
                lo[i] = pred
                hi[i] = pred

        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        return lo2, hi2


def _train_qrf(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_ens: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str,
) -> list[QRFMember]:
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    members: list[QRFMember] = []
    for m in range(n_ens):
        print(f"  [QRF] Training member {m+1}/{n_ens}...")
        boot_idx = rng.integers(low=0, high=n, size=n)
        Xb = X_train[boot_idx]
        yb = y_train[boot_idx]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed + m,
            bootstrap=True,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
        )
        rf.fit(Xb, yb)
        per_tree_leaf_to_y: list[dict[int, np.ndarray]] = []
        for tree in rf.estimators_:
            leaf_ids = tree.apply(Xb)
            order = np.argsort(leaf_ids, kind="mergesort")
            leaf_sorted = leaf_ids[order]
            y_sorted = yb[order]
            uniq, starts = np.unique(leaf_sorted, return_index=True)
            leaf_to_y: dict[int, np.ndarray] = {}
            for k, leaf in enumerate(uniq):
                s = int(starts[k])
                e = int(starts[k + 1]) if k + 1 < starts.size else int(leaf_sorted.size)
                leaf_to_y[int(leaf)] = y_sorted[s:e]
            per_tree_leaf_to_y.append(leaf_to_y)

        members.append(QRFMember(rf, per_tree_leaf_to_y=per_tree_leaf_to_y))
    return members


# SVM Implementation adapted from src/models/base_svr.py

def _kernelfun(X, Y, gamma):
    from scipy.spatial.distance import cdist
    sqdist = cdist(X, Y, 'sqeuclidean')
    return np.exp(-gamma * sqdist)

def _solve_lp_quantile(X_train, y_train, tau, gamma, c3, c1):
    from scipy.optimize import linprog
    
    n1 = X_train.shape[0]
    e = np.ones((n1, 1))
    
    # Kernel matrix
    A = _kernelfun(X_train, X_train, gamma)
    H = np.hstack([A, e])
    
    # Objective function
    # f = [c3*ones(n1+1), c3*ones(n1+1), c1*tau*ones(n1), c1*(1-tau)*ones(n1)]
    f = np.concatenate([
        c3 * np.ones(n1 + 1),
        c3 * np.ones(n1 + 1),
        c1 * tau * np.ones(n1),
        c1 * (1 - tau) * np.ones(n1)
    ])
    
    # Constraints
    # A1 * x <= B1
    # A1 constructed from blocks
    I_n = np.eye(n1)
    zeros_n = np.zeros((n1, n1))
    
    # Rows 1..n: -H*u - v + s_plus - s_minus <= -y  => -H(u_p - u_m) ... 
    # Actually the paper formulation is needed to be exact, but following base_svr.py:
    # A1_top = [-H, H, -I_n, zeros_n]
    # A1_bottom = [H, -H, zeros_n, -I_n]
    # B1 = [-y, y]
    
    A1_top = np.hstack([-H, H, -I_n, zeros_n])
    A1_bottom = np.hstack([H, -H, zeros_n, -I_n])
    A1 = np.vstack([A1_top, A1_bottom])
    B1 = np.concatenate([-y_train.flatten(), y_train.flatten()])
    
    # Bounds: x >= 0
    bounds = [(0, None) for _ in range(len(f))]
    
    # Solve LP
    # Using 'highs' method as in base_svr.py
    res = linprog(f, A_ub=A1, b_ub=B1, bounds=bounds, method='highs')
    
    if not res.success:
        print(f"Warning: LP failed for tau={tau}")
        return np.zeros(n1 + 1) # Fallback
        
    x = res.x
    # u = u_plus - u_minus
    # x structure: [u_plus(n1+1), u_minus(n1+1), s_plus(n1), s_minus(n1)] based on f?
    # Wait, let's check base_svr.py again.
    # f = [c3(n1+1), c3(n1+1), c1*tau(n1), c1(1-tau)(n1)]
    # So first (n1+1) are u_plus, next (n1+1) are u_minus.
    
    u_plus = x[0 : n1 + 1]
    u_minus = x[n1 + 1 : 2 * (n1 + 1)]
    u = u_plus - u_minus
    return u

class SVMMember:
    def __init__(self, u_lower, u_upper, X_train, gamma, scaler):
        self.u_lower = u_lower
        self.u_upper = u_upper
        self.X_train = X_train
        self.gamma = gamma
        self.scaler = scaler

    def predict_quantiles(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X_s = self.scaler.transform(X)
        
        # Kernel between test (X_s) and train (X_train)
        # We need to compute Htest
        # Htest = [K(X_test, X_train), 1]
        K_test = _kernelfun(X_s, self.X_train, self.gamma)
        H_test = np.hstack([K_test, np.ones((X_s.shape[0], 1))])
        
        lo = H_test.dot(self.u_lower)
        hi = H_test.dot(self.u_upper)
        
        # Ensure lo <= hi? No, let the raw predictions be, but we can swap if needed.
        # But for quantile regression crossing is possible.
        # We'll stick to raw for now, or sort them? 
        # The QRF implementation does min/max. Let's do that.
        
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        return lo2, hi2

def _train_svm(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_ens: int,
    q_lower: float,
    q_upper: float,
    gamma: float = 1.0,
    C: float = 1.0,
) -> list[SVMMember]:
    from sklearn.preprocessing import StandardScaler
    
    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    members: list[SVMMember] = []
    
    # Params from base_svr.py logic
    c3 = C
    c1 = 1.0 # Heuristic
    
    for m in range(n_ens):
        print(f"  [SVM] Training member {m+1}/{n_ens}...")
        boot_idx = rng.integers(low=0, high=n, size=n)
        Xb = X_train[boot_idx]
        yb = y_train[boot_idx]
        
        scaler = StandardScaler()
        Xb_s = scaler.fit_transform(Xb)
        
        # Train Lower
        u_lower = _solve_lp_quantile(Xb_s, yb, q_lower, gamma, c3, c1)
        
        # Train Upper
        u_upper = _solve_lp_quantile(Xb_s, yb, q_upper, gamma, c3, c1)
        
        members.append(SVMMember(u_lower, u_upper, Xb_s, gamma, scaler))
        
    return members




def _train_qrf(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    n_ens: int,
    n_estimators: int,
    min_samples_leaf: int,
    max_features: str,
) -> list[QRFMember]:
    from sklearn.ensemble import RandomForestRegressor

    rng = np.random.default_rng(seed)
    n = X_train.shape[0]
    members: list[QRFMember] = []
    for m in range(n_ens):
        boot_idx = rng.integers(low=0, high=n, size=n)
        Xb = X_train[boot_idx]
        yb = y_train[boot_idx]

        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=seed + m,
            bootstrap=True,
            min_samples_leaf=min_samples_leaf,
            max_features=max_features,
            n_jobs=-1,
        )
        rf.fit(Xb, yb)
        per_tree_leaf_to_y: list[dict[int, np.ndarray]] = []
        for tree in rf.estimators_:
            leaf_ids = tree.apply(Xb)
            order = np.argsort(leaf_ids, kind="mergesort")
            leaf_sorted = leaf_ids[order]
            y_sorted = yb[order]
            uniq, starts = np.unique(leaf_sorted, return_index=True)
            leaf_to_y: dict[int, np.ndarray] = {}
            for k, leaf in enumerate(uniq):
                s = int(starts[k])
                e = int(starts[k + 1]) if k + 1 < starts.size else int(leaf_sorted.size)
                leaf_to_y[int(leaf)] = y_sorted[s:e]
            per_tree_leaf_to_y.append(leaf_to_y)

        members.append(QRFMember(rf, per_tree_leaf_to_y=per_tree_leaf_to_y))
    return members


def _predict_bounds(
    *,
    method: str,
    members: list,
    X_eval: np.ndarray,
    q_lower: float,
    q_upper: float,
) -> tuple[np.ndarray, np.ndarray]:
    if method == "nn_qnt":
        pred = experiment.predict_nn_ensemble(members, X_eval)
        return pred.lower_members, pred.upper_members
    if method == "qrf_qnt":
        lower = []
        upper = []
        for mem in members:
            lo, hi = mem.predict_quantiles(X_eval, q_lower=q_lower, q_upper=q_upper)
            lower.append(lo)
            upper.append(hi)
        return np.stack(lower, axis=0), np.stack(upper, axis=0)
    if method == "svm_qnt":
        lower = []
        upper = []
        for mem in members:
            lo, hi = mem.predict_quantiles(X_eval)
            lower.append(lo)
            upper.append(hi)
        return np.stack(lower, axis=0), np.stack(upper, axis=0)
    raise ValueError(f"Unknown method: {method!r}")


def _evaluate_once(
    *,
    method: str,
    seed: int,
    n_samples: int,
    noise: str,
    n_ens: int,
    q_lower: float,
    q_upper: float,
    hidden_sizes: list[int],
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    standardize: bool,
    rf_n_estimators: int,
    rf_min_samples_leaf: int,
    rf_max_features: str,
    xgrid_n: int,
    variance_inflation: float,
    print_run_bounds: bool,
    run_label: str,
    svm_gamma: float = 1.0,
    svm_C: float = 1.0,
) -> dict:
    X, y_noisy, _ = experiment.generate_sinc_dataset(seed=seed, n_samples=n_samples, noise=noise)
    (X_train, y_train), (_, _), (X_test, y_test), _ = experiment.split_train_cal_test(
        X, y_noisy, seed=seed, train_frac=0.6, cal_frac=0.2
    )

    if method == "nn_qnt":
        members = _train_nn(
            X_train=X_train,
            y_train=y_train,
            seed=seed,
            n_ens=n_ens,
            q_lower=q_lower,
            q_upper=q_upper,
            hidden_sizes=hidden_sizes,
            dropout=dropout,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            standardize=standardize,
        )
    elif method == "qrf_qnt":
        members = _train_qrf(
            X_train=X_train,
            y_train=y_train,
            seed=seed,
            n_ens=n_ens,
            n_estimators=rf_n_estimators,
            min_samples_leaf=rf_min_samples_leaf,
            max_features=rf_max_features,
        )
    elif method == "svm_qnt":
        members = _train_svm(
            X_train=X_train,
            y_train=y_train,
            seed=seed,
            n_ens=n_ens,
            q_lower=q_lower,
            q_upper=q_upper,
            gamma=svm_gamma,
            C=svm_C,
        )
    else:
        raise ValueError(f"Unknown method: {method!r}")

    lower_members_test, upper_members_test = _predict_bounds(
        method=method, members=members, X_eval=X_test, q_lower=q_lower, q_upper=q_upper
    )


    if print_run_bounds:
        print(f"\n--- Sample Bounds for First Test Point ({run_label}) ---")
        for i in range(n_ens):
            l = lower_members_test[i, 0]
            u = upper_members_test[i, 0]
            print(f"Run {i+1}: Lower={l:.4f}, Upper={u:.4f}, Width={u-l:.4f}")

    stats_test = _bounds_stats(lower_members_test, upper_members_test, variance_inflation=variance_inflation)
    picp_mean, mpiw_mean = picp_mpiw(y_test, stats_test["mean_lower"], stats_test["mean_upper"])
    picp_agg, mpiw_agg = picp_mpiw(y_test, stats_test["agg_lower"], stats_test["agg_upper"])

    x_grid = np.linspace(-6.0, 6.0, xgrid_n)
    X_grid = x_grid.reshape(-1, 1)
    lower_members_grid, upper_members_grid = _predict_bounds(
        method=method, members=members, X_eval=X_grid, q_lower=q_lower, q_upper=q_upper
    )
    stats_grid = _bounds_stats(lower_members_grid, upper_members_grid, variance_inflation=variance_inflation)

    return {
        "method": method,
        "n_train": int(X_train.shape[0]),
        "n_samples": int(n_samples),
        "picp_test_mean": float(picp_mean),
        "mpiw_test_mean": float(mpiw_mean),
        "picp_test_agg": float(picp_agg),
        "mpiw_test_agg": float(mpiw_agg),
        "avg_std_lower_test": float(np.mean(stats_test["std_lower"])),
        "avg_std_upper_test": float(np.mean(stats_test["std_upper"])),
        "avg_std_center_test": float(np.mean(stats_test["std_center"])),
        "avg_std_width_test": float(np.mean(stats_test["std_width"])),
        "avg_std_center_grid": float(np.mean(stats_grid["std_center"])),
        "avg_std_width_grid": float(np.mean(stats_grid["std_width"])),
        "avg_width_grid": float(np.mean(stats_grid["mean_width"])),
        "stats_test": stats_test,
        "stats_grid": stats_grid,
        "x_grid": x_grid,
        "y_test": y_test,
        "X_test": X_test,
    }


def _evaluate_sweep_train_sizes(
    *,
    method: str,
    seed: int,
    n_samples: int,
    noise: str,
    n_ens: int,
    q_lower: float,
    q_upper: float,
    hidden_sizes: list[int],
    dropout: float,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    standardize: bool,
    rf_n_estimators: int,
    rf_min_samples_leaf: int,
    rf_max_features: str,
    xgrid_n: int,
    variance_inflation: float,
    train_sizes: list[int],
    svm_gamma: float = 1.0,
    svm_C: float = 1.0,
) -> list[dict]:
    X, y_noisy, _ = experiment.generate_sinc_dataset(seed=seed, n_samples=n_samples, noise=noise)
    (X_train_full, y_train_full), (_, _), (X_test, y_test), _ = experiment.split_train_cal_test(
        X, y_noisy, seed=seed, train_frac=0.6, cal_frac=0.2
    )

    rng = np.random.default_rng(seed)
    x_grid = np.linspace(-6.0, 6.0, xgrid_n)
    X_grid = x_grid.reshape(-1, 1)

    results: list[dict] = []
    for n_train in train_sizes:
        n_train = int(min(n_train, X_train_full.shape[0]))
        idx = rng.choice(X_train_full.shape[0], size=n_train, replace=False)
        X_train = X_train_full[idx]
        y_train = y_train_full[idx]

        if method == "nn_qnt":
            members = _train_nn(
                X_train=X_train,
                y_train=y_train,
                seed=seed,
                n_ens=n_ens,
                q_lower=q_lower,
                q_upper=q_upper,
                hidden_sizes=hidden_sizes,
                dropout=dropout,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                weight_decay=weight_decay,
                standardize=standardize,
            )
        elif method == "qrf_qnt":
            members = _train_qrf(
                X_train=X_train,
                y_train=y_train,
                seed=seed,
                n_ens=n_ens,
                n_estimators=rf_n_estimators,
                min_samples_leaf=rf_min_samples_leaf,
                max_features=rf_max_features,
            )
        elif method == "svm_qnt":
            members = _train_svm(
                X_train=X_train,
                y_train=y_train,
                seed=seed,
                n_ens=n_ens,
                q_lower=q_lower,
                q_upper=q_upper,
                gamma=svm_gamma,
                C=svm_C,
            )
        else:
            raise ValueError(f"Unknown method: {method!r}")

        lower_members_test, upper_members_test = _predict_bounds(
            method=method, members=members, X_eval=X_test, q_lower=q_lower, q_upper=q_upper
        )

        stats_test = _bounds_stats(lower_members_test, upper_members_test, variance_inflation=variance_inflation)
        picp_mean, mpiw_mean = picp_mpiw(y_test, stats_test["mean_lower"], stats_test["mean_upper"])
        picp_agg, mpiw_agg = picp_mpiw(y_test, stats_test["agg_lower"], stats_test["agg_upper"])

        lower_members_grid, upper_members_grid = _predict_bounds(
            method=method, members=members, X_eval=X_grid, q_lower=q_lower, q_upper=q_upper
        )
        stats_grid = _bounds_stats(lower_members_grid, upper_members_grid, variance_inflation=variance_inflation)

        results.append(
            {
                "method": method,
                "n_train": int(n_train),
                "picp_test_mean": float(picp_mean),
                "mpiw_test_mean": float(mpiw_mean),
                "picp_test_agg": float(picp_agg),
                "mpiw_test_agg": float(mpiw_agg),
                "avg_std_lower_test": float(np.mean(stats_test["std_lower"])),
                "avg_std_upper_test": float(np.mean(stats_test["std_upper"])),
                "avg_std_center_test": float(np.mean(stats_test["std_center"])),
                "avg_std_width_test": float(np.mean(stats_test["std_width"])),
                "avg_std_center_grid": float(np.mean(stats_grid["std_center"])),
                "avg_std_width_grid": float(np.mean(stats_grid["std_width"])),
                "avg_width_grid": float(np.mean(stats_grid["mean_width"])),
            }
        )

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, default="nn_qnt", choices=["nn_qnt", "qrf_qnt", "svm_qnt"])
    ap.add_argument("--seed", type=int, default=58)
    ap.add_argument("--n-samples", type=int, default=3000)
    ap.add_argument("--noise", type=str, default="uniform", choices=["uniform", "normal"])
    ap.add_argument("--n-ens", type=int, default=10)
    ap.add_argument("--q-lower", type=float, default=0.05)
    ap.add_argument("--q-upper", type=float, default=0.95)
    ap.add_argument("--var-inflation", type=float, default=1.96)
    ap.add_argument("--hidden-sizes", type=str, default="64")
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--no-standardize", action="store_true")
    ap.add_argument("--rf-n-estimators", type=int, default=200)
    ap.add_argument("--rf-min-samples-leaf", type=int, default=5)
    ap.add_argument("--rf-max-features", type=str, default="sqrt")
    ap.add_argument("--svm-gamma", type=float, default=1.0)
    ap.add_argument("--svm-C", type=float, default=1.0)
    ap.add_argument("--xgrid-n", type=int, default=1200)
    ap.add_argument("--compare-plot", action="store_true")
    ap.add_argument("--compare-plot-all", action="store_true")
    ap.add_argument("--n-train", type=int, default=300)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--sweep-epochs", type=int, default=100)
    ap.add_argument(
        "--train-sizes",
        type=str,
        default="100,300,600,900,1200,1500,1800",
    )
    args = ap.parse_args()

    hidden_sizes = [int(x.strip()) for x in args.hidden_sizes.split(",") if x.strip()]

    if args.compare_plot_all:
        X, y_noisy, _ = experiment.generate_sinc_dataset(seed=args.seed, n_samples=args.n_samples, noise=args.noise)
        (X_train_full, y_train_full), (_, _), (X_test, y_test), _ = experiment.split_train_cal_test(
            X, y_noisy, seed=args.seed, train_frac=0.6, cal_frac=0.2
        )

        rng = np.random.default_rng(args.seed)
        n_train = int(min(args.n_train, X_train_full.shape[0]))
        idx = rng.choice(X_train_full.shape[0], size=n_train, replace=False)
        X_train = X_train_full[idx]
        y_train = y_train_full[idx]

        x_grid = np.linspace(-6.0, 6.0, args.xgrid_n)
        X_grid = x_grid.reshape(-1, 1)
        y_grid_clean = np.sinc(x_grid / np.pi)

        methods = ["svm_qnt", "qrf_qnt", "nn_qnt"]
        panels: list[dict] = []
        for method in methods:
            if method == "nn_qnt":
                members = _train_nn(
                    X_train=X_train,
                    y_train=y_train,
                    seed=args.seed,
                    n_ens=args.n_ens,
                    q_lower=args.q_lower,
                    q_upper=args.q_upper,
                    hidden_sizes=hidden_sizes,
                    dropout=args.dropout,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    standardize=not args.no_standardize,
                )
            elif method == "qrf_qnt":
                members = _train_qrf(
                    X_train=X_train,
                    y_train=y_train,
                    seed=args.seed,
                    n_ens=args.n_ens,
                    n_estimators=args.rf_n_estimators,
                    min_samples_leaf=args.rf_min_samples_leaf,
                    max_features=args.rf_max_features,
                )
            elif method == "svm_qnt":
                members = _train_svm(
                    X_train=X_train,
                    y_train=y_train,
                    seed=args.seed,
                    n_ens=args.n_ens,
                    q_lower=args.q_lower,
                    q_upper=args.q_upper,
                    gamma=args.svm_gamma,
                    C=args.svm_C,
                )
            else:
                raise ValueError(f"Unknown method for compare-plot-all: {method}")

            lower_members_grid, upper_members_grid = _predict_bounds(
                method=method, members=members, X_eval=X_grid, q_lower=args.q_lower, q_upper=args.q_upper
            )
            stats_grid = _bounds_stats(lower_members_grid, upper_members_grid, variance_inflation=args.var_inflation)

            rmse_lower = float(np.sqrt(np.mean((stats_grid["mean_lower"] - stats_grid["agg_lower"]) ** 2)))
            rmse_upper = float(np.sqrt(np.mean((stats_grid["mean_upper"] - stats_grid["agg_upper"]) ** 2)))

            panels.append(
                {
                    "method": method,
                    "stats_grid": stats_grid,
                    "rmse_lower": rmse_lower,
                    "rmse_upper": rmse_upper,
                }
            )

        fig, axes = plt.subplots(1, len(panels), figsize=(18, 5), sharey=True)
        if len(panels) == 1:
            axes = [axes]

        for ax, p in zip(axes, panels):
            method = p["method"]
            stats_grid = p["stats_grid"]
            rmse_lower = p["rmse_lower"]
            rmse_upper = p["rmse_upper"]

            ax.scatter(X_test, y_test, alpha=0.25, s=8, color="gray", label="Test Data")
            ax.plot(x_grid, y_grid_clean, "k--", linewidth=1.5, label="True Signal")

            ax.plot(x_grid, stats_grid["mean_lower"], color="blue", linestyle="--", linewidth=1.5, label="Mean bounds")
            ax.plot(x_grid, stats_grid["mean_upper"], color="blue", linestyle="--", linewidth=1.5)

            ax.plot(x_grid, stats_grid["agg_lower"], color="red", linestyle="-", linewidth=2, label="Agg bounds")
            ax.plot(x_grid, stats_grid["agg_upper"], color="red", linestyle="-", linewidth=2)

            ax.set_title(f"{method} (n_train={n_train}, m={args.n_ens})")
            ax.set_xlabel("x")
            ax.grid(True, alpha=0.3)

            ax.text(
                0.02,
                0.98,
                f"RMSE lo: {rmse_lower:.4f}\nRMSE hi: {rmse_upper:.4f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )

        axes[0].set_ylabel("y")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncols=4, frameon=False)
        fig.suptitle("Mean vs Aggregated bounds (with RMSE) + True curve")
        fig.tight_layout(rect=[0, 0, 1, 0.90])
        out_name = f"all_methods_ntrain{n_train}_mean_vs_agg.png"
        fig.savefig(out_name, dpi=160)
        print(f"Plot saved to {out_name}")
        return

    if args.compare_plot:
        X, y_noisy, _ = experiment.generate_sinc_dataset(seed=args.seed, n_samples=args.n_samples, noise=args.noise)
        (X_train_full, y_train_full), (_, _), (X_test, y_test), _ = experiment.split_train_cal_test(
            X, y_noisy, seed=args.seed, train_frac=0.6, cal_frac=0.2
        )

        rng = np.random.default_rng(args.seed)
        n_train = int(min(args.n_train, X_train_full.shape[0]))
        idx = rng.choice(X_train_full.shape[0], size=n_train, replace=False)
        X_train = X_train_full[idx]
        y_train = y_train_full[idx]

        if args.method == "nn_qnt":
            members = _train_nn(
                X_train=X_train,
                y_train=y_train,
                seed=args.seed,
                n_ens=args.n_ens,
                q_lower=args.q_lower,
                q_upper=args.q_upper,
                hidden_sizes=hidden_sizes,
                dropout=args.dropout,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                standardize=not args.no_standardize,
            )
        elif args.method == "qrf_qnt":
            members = _train_qrf(
                X_train=X_train,
                y_train=y_train,
                seed=args.seed,
                n_ens=args.n_ens,
                n_estimators=args.rf_n_estimators,
                min_samples_leaf=args.rf_min_samples_leaf,
                max_features=args.rf_max_features,
            )
        elif args.method == "svm_qnt":
            members = _train_svm(
                X_train=X_train,
                y_train=y_train,
                seed=args.seed,
                n_ens=args.n_ens,
                q_lower=args.q_lower,
                q_upper=args.q_upper,
                gamma=args.svm_gamma,
                C=args.svm_C,
            )
        else:
            raise ValueError(f"Unknown method for compare-plot: {args.method}")

        x_grid = np.linspace(-6.0, 6.0, args.xgrid_n)

        X_grid = x_grid.reshape(-1, 1)
        y_grid_clean = np.sinc(x_grid / np.pi)

        lower_members_test, upper_members_test = _predict_bounds(
            method=args.method, members=members, X_eval=X_test, q_lower=args.q_lower, q_upper=args.q_upper
        )
        stats_test = _bounds_stats(
            lower_members_test, upper_members_test, variance_inflation=args.var_inflation
        )
        picp_mean, mpiw_mean = picp_mpiw(y_test, stats_test["mean_lower"], stats_test["mean_upper"])
        picp_agg, mpiw_agg = picp_mpiw(y_test, stats_test["agg_lower"], stats_test["agg_upper"])

        lower_members_grid, upper_members_grid = _predict_bounds(
            method=args.method, members=members, X_eval=X_grid, q_lower=args.q_lower, q_upper=args.q_upper
        )
        stats_grid = _bounds_stats(
            lower_members_grid, upper_members_grid, variance_inflation=args.var_inflation
        )

        # Print detailed values for a few grid points
        print("\n" + "=" * 80)
        print("Detailed Data Values for Selected Grid Points (Ensemble vs Non-Ensemble)")
        print("=" * 80)
        
        # Select 5 equidistant points from the grid
        grid_indices = np.linspace(0, args.xgrid_n - 1, 5, dtype=int)
        
        for idx in grid_indices:
            x_val = x_grid[idx]
            print(f"\nGrid Point: x = {x_val:.4f}")
            print("-" * 40)
            
            # Print individual run bounds
            print(f"{'Run':<5} | {'Lower Bound':<12} | {'Upper Bound':<12}")
            print("-" * 35)
            for m in range(args.n_ens):
                l_val = lower_members_grid[m, idx]
                u_val = upper_members_grid[m, idx]
                print(f"{m+1:<5} | {l_val:12.4f} | {u_val:12.4f}")
            print("-" * 35)
            
            # Print Non-Ensemble (Mean) bounds
            mean_l = stats_grid['mean_lower'][idx]
            mean_u = stats_grid['mean_upper'][idx]
            print(f"{'Mean (Non-Ens)':<15} | {mean_l:12.4f} | {mean_u:12.4f}")
            
            # Print Ensemble (Aggregated) bounds
            agg_l = stats_grid['agg_lower'][idx]
            agg_u = stats_grid['agg_upper'][idx]
            print(f"{'Agg (Ensemble)':<15} | {agg_l:12.4f} | {agg_u:12.4f}")
            
            # Print Variance
            var_l = stats_grid['var_lower'][idx]
            var_u = stats_grid['var_upper'][idx]
            print(f"{'Variance':<15} | {var_l:12.4f} | {var_u:12.4f}")
            
        print("\n" + "=" * 80 + "\n")

        # Calculate RMSE between Mean and Aggregated curves (on grid)
        rmse_lower = np.sqrt(np.mean((stats_grid["mean_lower"] - stats_grid["agg_lower"])**2))
        rmse_upper = np.sqrt(np.mean((stats_grid["mean_upper"] - stats_grid["agg_upper"])**2))
        
        print("-" * 40)
        print("RMSE between Mean vs Agg Curves (on Grid)")
        print(f"RMSE Lower Curves: {rmse_lower:.6f}")
        print(f"RMSE Upper Curves: {rmse_upper:.6f}")
        print("-" * 40 + "\n")

        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, alpha=0.3, s=10, color="gray", label="Test Data")
        plt.plot(x_grid, y_grid_clean, "k--", label="True Signal")

        # Plot Mean bounds (Non-ensemble) as dashed blue lines
        plt.plot(
            x_grid,
            stats_grid["mean_lower"],
            color="blue",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean bounds (PICP={picp_mean:.3f}, MPIW={mpiw_mean:.3f})",
        )
        plt.plot(
            x_grid,
            stats_grid["mean_upper"],
            color="blue",
            linestyle="--",
            linewidth=1.5,
        )

        # Plot Aggregated bounds (Ensemble) as solid red lines
        plt.plot(
            x_grid,
            stats_grid["agg_lower"],
            color="red",
            linestyle="-",
            linewidth=2,
            label=f"Agg bounds (PICP={picp_agg:.3f}, MPIW={mpiw_agg:.3f})",
        )
        plt.plot(
            x_grid,
            stats_grid["agg_upper"],
            color="red",
            linestyle="-",
            linewidth=2,
        )

        plt.title(
            f"{args.method}: mean vs ensemble-aggregated bounds (n_train={n_train}, m={args.n_ens})"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_name = f"{args.method}_ntrain{n_train}_mean_vs_agg.png"
        plt.savefig(out_name, dpi=160)
        print(f"Plot saved to {out_name}")
        return

    if not args.sweep:
        print("Generating data...")
        out = _evaluate_once(
            method=args.method,
            seed=args.seed,
            n_samples=args.n_samples,
            noise=args.noise,
            n_ens=args.n_ens,
            q_lower=args.q_lower,
            q_upper=args.q_upper,
            hidden_sizes=hidden_sizes,
            dropout=args.dropout,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            standardize=not args.no_standardize,
            rf_n_estimators=args.rf_n_estimators,
            rf_min_samples_leaf=args.rf_min_samples_leaf,
            rf_max_features=args.rf_max_features,
            xgrid_n=args.xgrid_n,
            variance_inflation=args.var_inflation,
            print_run_bounds=True,
            run_label=f"method={args.method}, n_samples={args.n_samples}, n_ens={args.n_ens}",
            svm_gamma=args.svm_gamma,
            svm_C=args.svm_C,
        )

        stats_test = out["stats_test"]
        mean_lower = stats_test["mean_lower"]
        mean_upper = stats_test["mean_upper"]
        agg_lower = stats_test["agg_lower"]
        agg_upper = stats_test["agg_upper"]

        print("\n--- Aggregated Statistics (First 5 Test Points) ---")
        print(
            f"{'Idx':<4} | {'Mean Lower':<10} {'Std Lower':<10} {'Var Lower':<10} | {'Mean Upper':<10} {'Std Upper':<10} {'Var Upper':<10}"
        )
        print("-" * 85)
        for i in range(5):
            print(
                f"{i:<4} | {mean_lower[i]:<10.4f} {stats_test['std_lower'][i]:<10.4f} {stats_test['var_lower'][i]:<10.4f} | {mean_upper[i]:<10.4f} {stats_test['std_upper'][i]:<10.4f} {stats_test['var_upper'][i]:<10.4f}"
            )

        print("\n" + "=" * 40)
        print("Global Statistics across Test Set")
        print("=" * 40)
        print(f"Avg Variance (Lower Bound): {float(np.mean(stats_test['var_lower'])):.6f}")
        print(f"Avg Std Dev  (Lower Bound): {float(np.mean(stats_test['std_lower'])):.6f}")
        print(f"Avg Variance (Upper Bound): {float(np.mean(stats_test['var_upper'])):.6f}")
        print(f"Avg Std Dev  (Upper Bound): {float(np.mean(stats_test['std_upper'])):.6f}")
        print(f"Avg Variance (Center):      {float(np.mean(stats_test['var_center'])):.6f}")
        print(f"Avg Std Dev  (Center):      {float(np.mean(stats_test['std_center'])):.6f}")
        print("=" * 40)

        print("\n" + "=" * 40)
        print("Final Results on Test Set (Uncalibrated)")
        print("=" * 40)
        print(f"PICP mean-bounds: {out['picp_test_mean']:.4f}")
        print(f"MPIW mean-bounds: {out['mpiw_test_mean']:.4f}")
        print(f"PICP agg-bounds:  {out['picp_test_agg']:.4f}")
        print(f"MPIW agg-bounds:  {out['mpiw_test_agg']:.4f}")
        
        # Calculate RMSE between Mean and Aggregated curves (on grid)
        # We use the grid stats for "between the curves" visualization metric
        grid_mean_lower = out["stats_grid"]["mean_lower"]
        grid_agg_lower = out["stats_grid"]["agg_lower"]
        grid_mean_upper = out["stats_grid"]["mean_upper"]
        grid_agg_upper = out["stats_grid"]["agg_upper"]
        
        rmse_lower = np.sqrt(np.mean((grid_mean_lower - grid_agg_lower)**2))
        rmse_upper = np.sqrt(np.mean((grid_mean_upper - grid_agg_upper)**2))
        
        print("-" * 40)
        print("RMSE between Mean vs Agg Curves (on Grid)")
        print(f"RMSE Lower Curves: {rmse_lower:.6f}")
        print(f"RMSE Upper Curves: {rmse_upper:.6f}")
        print("=" * 40)

        x_grid = out["x_grid"]
        y_grid_clean = np.sinc(x_grid / np.pi)
        grid_mean_lower = out["stats_grid"]["agg_lower"]
        grid_mean_upper = out["stats_grid"]["agg_upper"]
        X_test = out["X_test"]
        y_test = out["y_test"]

        plt.figure(figsize=(10, 6))
        plt.scatter(X_test, y_test, alpha=0.3, s=10, color="gray", label="Test Data")
        plt.plot(x_grid, y_grid_clean, "k--", label="True Signal")
        plt.fill_between(
            x_grid,
            grid_mean_lower,
            grid_mean_upper,
            color="blue",
            alpha=0.2,
            label="Mean Prediction Interval",
        )
        plt.plot(x_grid, grid_mean_lower, color="blue", linewidth=1, linestyle=":")
        plt.plot(x_grid, grid_mean_upper, color="blue", linewidth=1, linestyle=":")

        plt.title(
            f"{args.method} (m={args.n_ens}) Aggregated Bounds\nPICP={out['picp_test_agg']:.3f}, MPIW={out['mpiw_test_agg']:.3f}"
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plot_path = f"{args.method}_ensemble_bounds_plot.png"
        plt.savefig(plot_path, dpi=160)
        print(f"Plot saved to {plot_path}")
        return

    train_sizes = [int(x.strip()) for x in args.train_sizes.split(",") if x.strip()]
    print(
        f"Sweeping train sizes: {train_sizes} (method={args.method}, epochs={args.sweep_epochs}, n_ens={args.n_ens}, seed={args.seed})"
    )
    results = _evaluate_sweep_train_sizes(
        method=args.method,
        seed=args.seed,
        n_samples=args.n_samples,
        noise=args.noise,
        n_ens=args.n_ens,
        q_lower=args.q_lower,
        q_upper=args.q_upper,
        hidden_sizes=hidden_sizes,
        dropout=args.dropout,
        epochs=args.sweep_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        standardize=not args.no_standardize,
        rf_n_estimators=args.rf_n_estimators,
        rf_min_samples_leaf=args.rf_min_samples_leaf,
        rf_max_features=args.rf_max_features,
        xgrid_n=args.xgrid_n,
        variance_inflation=args.var_inflation,
        train_sizes=train_sizes,
        svm_gamma=args.svm_gamma,
        svm_C=args.svm_C,
    )

    print("\n--- Uncertainty vs Training Points ---")
    header = (
        f"{'n_train':>7} | {'PICP(mean)':>10} {'MPIW(mean)':>10} | {'PICP(agg)':>9} {'MPIW(agg)':>9} | {'std_lo(test)':>12} {'std_hi(test)':>12} {'std_ctr(test)':>12} | "
        f"{'std_ctr(grid)':>12} {'width(grid)':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        print(
            f"{r['n_train']:7d} | {r['picp_test_mean']:10.3f} {r['mpiw_test_mean']:10.4f} | {r['picp_test_agg']:9.3f} {r['mpiw_test_agg']:9.4f} | {r['avg_std_lower_test']:12.6f} {r['avg_std_upper_test']:12.6f} {r['avg_std_center_test']:12.6f} | {r['avg_std_center_grid']:12.6f} {r['avg_width_grid']:12.6f}"
        )

    n_tr = np.array([r["n_train"] for r in results], dtype=float)
    std_ctr_grid = np.array([r["avg_std_center_grid"] for r in results], dtype=float)
    std_lo_test = np.array([r["avg_std_lower_test"] for r in results], dtype=float)
    std_hi_test = np.array([r["avg_std_upper_test"] for r in results], dtype=float)
    mpiw_mean = np.array([r["mpiw_test_mean"] for r in results], dtype=float)
    picp_mean = np.array([r["picp_test_mean"] for r in results], dtype=float)
    mpiw_agg = np.array([r["mpiw_test_agg"] for r in results], dtype=float)
    picp_agg = np.array([r["picp_test_agg"] for r in results], dtype=float)

    plt.figure(figsize=(10, 5))
    plt.plot(n_tr, std_ctr_grid, marker="o", label="Avg Std Center (grid)")
    plt.plot(n_tr, std_lo_test, marker="o", label="Avg Std Lower (test)")
    plt.plot(n_tr, std_hi_test, marker="o", label="Avg Std Upper (test)")
    plt.xlabel("Number of training points")
    plt.ylabel("Std across ensemble (mean over x)")
    plt.title(f"Model uncertainty vs training set size ({args.method})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    unc_path = f"{args.method}_uncertainty_vs_train_size.png"
    plt.savefig(unc_path, dpi=160)
    print(f"Plot saved to {unc_path}")

    plt.figure(figsize=(10, 5))
    plt.plot(n_tr, mpiw_mean, marker="o", label="MPIW mean-bounds (test)")
    plt.plot(n_tr, picp_mean, marker="o", label="PICP mean-bounds (test)")
    plt.plot(n_tr, mpiw_agg, marker="o", label="MPIW agg-bounds (test)")
    plt.plot(n_tr, picp_agg, marker="o", label="PICP agg-bounds (test)")
    plt.xlabel("Number of training points")
    plt.title(f"Coverage/width vs training set size ({args.method}, uncalibrated)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    met_path = f"{args.method}_metrics_vs_train_size.png"
    plt.savefig(met_path, dpi=160)
    print(f"Plot saved to {met_path}")

if __name__ == "__main__":
    main()
