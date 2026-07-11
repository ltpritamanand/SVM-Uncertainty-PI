import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
import time
import os
from sklearn.utils import Bunch

# Path for saving figures
# Use absolute path relative to this file to avoid permission errors
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(BASE_DIR, 'results_new', 'figures')
os.makedirs(RESULTS_DIR, exist_ok=True)

# For reproducibility
np.random.seed(42)
torch.manual_seed(42)

class QuantileNet(nn.Module):
    """Neural network for quantile regression with two outputs."""
    def __init__(self, input_dim, hidden_dim=20):
        super(QuantileNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 2)  # 2 outputs for lower and upper quantiles
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
def pinball_loss(preds, target, quantile):
    """Compute the pinball loss for a specific quantile."""
    diff = target - preds
    return torch.mean(torch.max(quantile * diff, (quantile - 1) * diff))

def load_boston_housing():
    """Load Boston Housing dataset (or use California Housing as replacement)."""
    try:
        # Try to load Boston Housing (deprecated, may not work)
        from sklearn.datasets import load_boston
        data = load_boston()
        print("Using Boston Housing dataset")
        return data
    except:
        # Use California Housing as a replacement
        print("Boston Housing dataset is deprecated. Using California Housing as replacement.")
        data = fetch_california_housing()
        
        # Create a smaller subset to match Boston Housing size
        n_samples = 506  # Original Boston Housing size
        indices = np.random.choice(data.data.shape[0], n_samples, replace=False)
        
        # Create a Bunch object to match the Boston Housing format
        boston = Bunch(
            data=data.data[indices],
            target=data.target[indices],
            feature_names=data.feature_names,
            DESCR="California Housing dataset (subset to match Boston Housing size)"
        )
        return boston

def load_energy_efficiency():
    """Load Energy Efficiency dataset."""
    # Check if the dataset is available locally in the datasets folder
    dataset_path = os.path.join(BASE_DIR, 'datasets', 'energy_efficiency.csv')
    try:
        if os.path.exists(dataset_path):
            df = pd.read_csv(dataset_path)
            print(f"Using Energy Efficiency dataset from {dataset_path}")
        else:
            # Try to load from current directory as fallback
            df = pd.read_csv('energy_efficiency.csv')
            print("Using Energy Efficiency dataset from local file")
    except:
        # If not available, create a synthetic dataset with similar properties
        print("Energy Efficiency dataset not found. Creating synthetic data.")
        n_samples = 768
        n_features = 8
        X = np.random.randn(n_samples, n_features)
        y = 0.5 * X[:, 0] + 0.2 * X[:, 1] - 0.7 * X[:, 2] + 0.1 * X[:, 3] + 0.3 * np.random.randn(n_samples)
        
        # Create a Bunch object
        energy = Bunch(
            data=X,
            target=y,
            feature_names=[f'feature_{i}' for i in range(n_features)],
            DESCR="Synthetic Energy Efficiency dataset"
        )
        return energy
    
    # If the file exists, convert to Bunch format
    X = df.iloc[:, :-1].values  # Assuming the target is the last column
    y = df.iloc[:, -1].values
    
    energy = Bunch(
        data=X,
        target=y,
        feature_names=df.columns[:-1].tolist(),
        DESCR="Energy Efficiency dataset"
    )
    return energy

def svqr_cp(X_train, y_train, X_test, y_test, alpha=0.1, quantile_lower=0.05, quantile_upper=0.95):
    """Implement SVQR+CP method."""
    # Ensure y is 1D
    if len(y_train.shape) > 1 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
    if len(y_test.shape) > 1 and y_test.shape[1] == 1:
        y_test = y_test.ravel()
        
    # Split training data into training and calibration sets
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Train SVQR models (using QuantileRegressor as a proxy for SVQR)
    qr_lower = QuantileRegressor(quantile=quantile_lower, alpha=0.001, solver='highs')
    qr_upper = QuantileRegressor(quantile=quantile_upper, alpha=0.001, solver='highs')
    
    qr_lower.fit(X_train_proper, y_train_proper)
    qr_upper.fit(X_train_proper, y_train_proper)
    
    # Compute nonconformity scores on calibration set
    errors = []
    for i in range(len(X_cal)):
        y_pred_l = qr_lower.predict(X_cal[i].reshape(1, -1))[0]
        y_pred_u = qr_upper.predict(X_cal[i].reshape(1, -1))[0]
        
        # Nonconformity score
        error = max(y_pred_l - y_cal[i], y_cal[i] - y_pred_u, 0)
        errors.append(error)
    
    # Determine calibration quantile
    n_cal = len(errors)
    errors_sorted = np.sort(errors)
    k = int(np.ceil((1 - alpha) * (n_cal + 1)))
    Q = errors_sorted[min(k-1, n_cal-1)]  # Ensure index doesn't exceed array bounds
    
    # Apply conformal prediction on test set
    y_pred_l_test = qr_lower.predict(X_test)
    y_pred_u_test = qr_upper.predict(X_test)
    
    # Adjust with conformal quantile
    y_pred_l_adjusted = y_pred_l_test - Q
    y_pred_u_adjusted = y_pred_u_test + Q
    
    # Compute metrics
    inside = np.logical_and(y_test >= y_pred_l_adjusted, y_test <= y_pred_u_adjusted)
    picp = np.mean(inside)
    mpiw = np.mean(y_pred_u_adjusted - y_pred_l_adjusted)
    
    return {
        'picp': picp,
        'mpiw': mpiw,
        'lower_bounds': y_pred_l_adjusted,
        'upper_bounds': y_pred_u_adjusted
    }

def ssvqr_cp(X_train, y_train, X_test, y_test, alpha=0.1, quantile_lower=0.05, quantile_upper=0.95):
    """Implement Sparse SVQR+CP method with L1 regularization."""
    # Ensure y is 1D
    if len(y_train.shape) > 1 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
    if len(y_test.shape) > 1 and y_test.shape[1] == 1:
        y_test = y_test.ravel()
        
    # Split training data into training and calibration sets
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Train SSVQR models with L1 regularization (stronger alpha value)
    # QuantileRegressor uses L1 penalty by default
    qr_lower = QuantileRegressor(quantile=quantile_lower, alpha=0.05, solver='highs')
    qr_upper = QuantileRegressor(quantile=quantile_upper, alpha=0.05, solver='highs')
    
    qr_lower.fit(X_train_proper, y_train_proper)
    qr_upper.fit(X_train_proper, y_train_proper)
    
    # Compute nonconformity scores on calibration set
    errors = []
    for i in range(len(X_cal)):
        y_pred_l = qr_lower.predict(X_cal[i].reshape(1, -1))[0]
        y_pred_u = qr_upper.predict(X_cal[i].reshape(1, -1))[0]
        
        # Nonconformity score
        error = max(y_pred_l - y_cal[i], y_cal[i] - y_pred_u, 0)
        errors.append(error)
    
    # Determine calibration quantile
    n_cal = len(errors)
    errors_sorted = np.sort(errors)
    k = int(np.ceil((1 - alpha) * (n_cal + 1)))
    Q = errors_sorted[min(k-1, n_cal-1)]  # Ensure index doesn't exceed array bounds
    
    # Apply conformal prediction on test set
    y_pred_l_test = qr_lower.predict(X_test)
    y_pred_u_test = qr_upper.predict(X_test)
    
    # Adjust with conformal quantile
    y_pred_l_adjusted = y_pred_l_test - Q
    y_pred_u_adjusted = y_pred_u_test + Q
    
    # Compute metrics
    inside = np.logical_and(y_test >= y_pred_l_adjusted, y_test <= y_pred_u_adjusted)
    picp = np.mean(inside)
    mpiw = np.mean(y_pred_u_adjusted - y_pred_l_adjusted)
    
    # Calculate sparsity (percentage of zero coefficients)
    n_features = X_train.shape[1]
    n_zeros_lower = np.sum(np.abs(qr_lower.coef_) < 1e-6)
    n_zeros_upper = np.sum(np.abs(qr_upper.coef_) < 1e-6)
    sparsity = ((n_zeros_lower + n_zeros_upper) / (2 * n_features)) * 100
    
    return {
        'picp': picp,
        'mpiw': mpiw,
        'lower_bounds': y_pred_l_adjusted,
        'upper_bounds': y_pred_u_adjusted,
        'sparsity': sparsity
    }

def cqr_nn(X_train, y_train, X_test, y_test, alpha=0.1, quantile_lower=0.05, quantile_upper=0.95, 
           hidden_dim=20, epochs=500, batch_size=32, learning_rate=0.001):
    """Implement neural network-based CQR method."""
    # Ensure y is 1D
    if len(y_train.shape) > 1 and y_train.shape[1] == 1:
        y_train = y_train.ravel()
    if len(y_test.shape) > 1 and y_test.shape[1] == 1:
        y_test = y_test.ravel()
        
    # Split training data into training and calibration sets
    X_train_proper, X_cal, y_train_proper, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=42
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_proper)
    X_cal_scaled = scaler.transform(X_cal)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.FloatTensor(y_train_proper.reshape(-1, 1))
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X_train_scaled.shape[1]
    model = QuantileNet(input_dim, hidden_dim)
    
    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train model
    model.train()
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Lower and upper quantile predictions
            lower_pred = outputs[:, 0]
            upper_pred = outputs[:, 1]
            
            # Compute losses for both quantiles
            loss_lower = pinball_loss(lower_pred, targets.squeeze(), quantile_lower)
            loss_upper = pinball_loss(upper_pred, targets.squeeze(), quantile_upper)
            
            # Combined loss
            loss = loss_lower + loss_upper
            
            loss.backward()
            optimizer.step()
        
        # Print progress every 100 epochs
        if (epoch + 1) % 100 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')
    
    # Calibration phase
    model.eval()
    with torch.no_grad():
        # Get predictions on calibration set
        X_cal_tensor = torch.FloatTensor(X_cal_scaled)
        cal_preds = model(X_cal_tensor).numpy()
        
        # Compute nonconformity scores
        errors = []
        for i in range(len(X_cal)):
            y_pred_l = cal_preds[i, 0]
            y_pred_u = cal_preds[i, 1]
            
            # Nonconformity score
            error = max(y_pred_l - y_cal[i], y_cal[i] - y_pred_u, 0)
            errors.append(error)
        
        # Determine calibration quantile
        n_cal = len(errors)
        errors_sorted = np.sort(errors)
        k = int(np.ceil((1 - alpha) * (n_cal + 1)))
        Q = errors_sorted[min(k-1, n_cal-1)]  # Ensure index doesn't exceed array bounds
        
        # Apply conformal prediction on test set
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        test_preds = model(X_test_tensor).numpy()
        
        # Adjust with conformal quantile
        y_pred_l_adjusted = test_preds[:, 0] - Q
        y_pred_u_adjusted = test_preds[:, 1] + Q
        
        # Compute metrics
        inside = np.logical_and(y_test >= y_pred_l_adjusted, y_test <= y_pred_u_adjusted)
        picp = np.mean(inside)
        mpiw = np.mean(y_pred_u_adjusted - y_pred_l_adjusted)
    
    return {
        'picp': picp,
        'mpiw': mpiw,
        'lower_bounds': y_pred_l_adjusted,
        'upper_bounds': y_pred_u_adjusted
    }

# The following functions will be moved to src/experiments/run_all_benchmarks.py:
# def run_all_experiments(n_runs=5): ...
# def create_visualization(dataset_name, results, y_test, svqr_results, cqr_results): ...
# def print_summary(results): ...

# The if __name__ == "__main__": block will also be moved. 