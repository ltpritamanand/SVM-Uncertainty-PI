#!/usr/bin/env python3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from scipy.spatial.distance import cdist

def kernelfun(X, kerfPara, Y=None):
    if Y is None:
        Y = X
    if kerfPara['type'] == 'rbf':
        gamma = kerfPara['pars']
        return np.exp(-gamma * cdist(X, Y, 'sqeuclidean'))
    else:
        raise ValueError("Unknown kernel function")

def build_dataset(y, win):
    X_all = [y[i:i+win] for i in range(len(y)-win)]
    y_all = [y[i+win] for i in range(len(y)-win)]
    return np.array(X_all), np.array(y_all).reshape(-1, 1)

def tube_loss_regression(train, ytrain, test, ytest, tau, para):
    n = train.shape[0]
    p1 = para['p1']
    r = para['r']
    eta = para['eta']
    eps1 = para['eps1']
    max_iter = para['iter']
    lambd = para['lambda']
    delta = para['delta']
    np.random.seed(1)
    X_mean = np.mean(ytrain)
    X_std = np.std(ytrain)
    w1 = np.random.rand(n, 1) * 0.001
    w2 = np.random.rand(n, 1) * 0.001
    b1 = np.array([[X_mean + 0.05 * X_std]])
    b2 = np.array([[X_mean - 0.05 * X_std]])
    H = kernelfun(train, {'type':'rbf', 'pars': p1})
    for i in range(max_iter):
        pred_up = H @ w1 + b1
        pred_low = H @ w2 + b2
        mid_pred = H @ (r*w1 + (1-r)*w2) + (r*b1 + (1-r)*b2)
        m1 = np.where((ytrain.flatten() < pred_up.flatten()) & (ytrain.flatten() > pred_low.flatten()) & (ytrain.flatten() > mid_pred.flatten()))[0]
        m2 = np.where((ytrain.flatten() < pred_up.flatten()) & (ytrain.flatten() > pred_low.flatten()) & (ytrain.flatten() < mid_pred.flatten()))[0]
        m3 = np.where(ytrain.flatten() < pred_low.flatten())[0]
        m4 = np.where(ytrain.flatten() > pred_up.flatten())[0]
        total = len(np.concatenate((m1, m2, m3, m4)))
        if total:
            loss1 = np.sum((1-tau) * pred_up.flatten()[m1] - ytrain.flatten()[m1])
            loss2 = np.sum((1-tau) * (ytrain.flatten()[m2] - pred_low.flatten()[m2]))
            loss3 = np.sum(tau * (pred_low.flatten()[m3] - ytrain.flatten()[m3]))
            loss4 = np.sum(tau * (ytrain.flatten()[m4] - pred_up.flatten()[m4]))
            current_loss = (loss1 + loss2 + loss3 + loss4) / total
        else:
            current_loss = 0
        grad_w1 = np.zeros_like(w1)
        grad_b1 = 0
        grad_w2 = np.zeros_like(w2)
        grad_b2 = 0
        if len(m1)>0:
            grad_w1 += np.mean(H[m1,:].T * (1-tau), axis=1, keepdims=True)
            grad_b1 += (1-tau)
        if len(m4)>0:
            grad_w1 -= np.mean(H[m4,:].T * tau, axis=1, keepdims=True)
            grad_b1 -= tau
        if len(m2)>0:
            grad_w2 += np.mean(H[m2,:].T * (tau-1), axis=1, keepdims=True)
            grad_b2 += (tau-1)
        if len(m3)>0:
            grad_w2 += np.mean(H[m3,:].T * tau, axis=1, keepdims=True)
            grad_b2 += tau
        grad_w1 += (lambd/n) * w1
        grad_w2 += (lambd/n) * w2
        flg = 1 if np.min(pred_up - pred_low) <= 0 else 0
        grad_w1 += (delta/n) * flg * np.mean(H, axis=0).reshape(-1,1)
        grad_w2 += (delta/n) * flg * np.mean(H, axis=0).reshape(-1,1)
        grad_b1 += (delta/n) * flg
        grad_b2 += (delta/n) * flg
        w1 -= eta * grad_w1
        b1 -= eta * grad_b1
        w2 -= eta * grad_w2
        b2 -= eta * grad_b2
        eta /= (1 + eps1)
        if eta < 1e-5:
            break
    Htest = kernelfun(test, {'type':'rbf', 'pars': p1}, train)
    pred_upper = Htest @ w1 + b1
    pred_lower = Htest @ w2 + b2
    cov = np.mean((ytest >= pred_lower) & (ytest <= pred_upper))
    CI = np.mean(pred_upper - pred_lower)
    return pred_upper, pred_lower, current_loss, cov, CI

def main():
    start = time.time()
    # Load real dataset from beer.csv
    from pathlib import Path
    data_path = Path(__file__).resolve().parents[2] / "datasets" / "beer.csv"
    df = pd.read_csv(data_path)
    # Print the complete dataset
    print("Complete dataset:")
    print(df)
    
    # Convert the "Monthly beer production" column to a numeric array
    y = pd.to_numeric(df["Monthly beer production"], errors='coerce').dropna().values

    # Build dataset using a sliding window over the time series (window length = 1)
    win = 1
    X_all, y_all = build_dataset(y, win)
    # Print the built dataset arrays X_all and y_all
    print("\nX_all:")
    print(X_all)
    print("\ny_all:")
    print(y_all)
    
    # Split data into 70% training and 30% testing
    idx = int(0.7 * X_all.shape[0])
    X_train, y_train = X_all[:idx, :], y_all[:idx, :]
    X_test, y_test   = X_all[idx:, :], y_all[idx:, :]

    # Model Parameters tuned for the beer dataset
    para = {
        'kernel': 2,
        'p1': 0.35,
        'r': 0.5,
        'eta': 0.015,
        'eps1': 2e-4,
        'iter': 150000,
        'lambda': 350.0,
        'delta': 0.0
    }
    tau = 0.95

    pred_upper, pred_lower, loss, cov, CI = tube_loss_regression(X_train, y_train, X_test, y_test, tau, para)
    print(f"Loss: {loss:.4f}")
    print(f"Coverage: {cov:.4f}")
    print(f"Tube Width: {CI:.4f}")
    print(f"Time: {time.time() - start:.2f} s")

    # Plot the true production along with the predicted lower and upper bounds
    plt.figure()
    plt.plot(y_test, 'b-', label="True")
    plt.plot(pred_lower, 'r-', label="Lower")
    plt.plot(pred_upper, 'k-', label="Upper")
    plt.title("Tube Loss Regression on Beer Production Data")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
