# Sec 5.4.2 — SVM Probabilistic Forecasting Under Distribution Shift (ACI: SVM + LSTM)

Applies Adaptive Conformal Inference (Gibbs & Candès 2021) to SVQR and LSTM quantile models on the Amprion load dataset. ACI online-recalibrates α using a 500-score rolling window (γ=0.001).

## Files

- `aci_svm_amprion.m` — SVM+ACI experiment (MATLAB)
- `aci_lstm_amprion.py` — LSTM+ACI experiment (Python)

## Run

SVM (MATLAB):
```matlab
aci_svm_amprion
```

LSTM (Python):
```bash
python aci_lstm_amprion.py
```

## Key results (from paper)

| Method | PICP | MPIW | Train time | Inference |
|---|---|---|---|---|
| SVM+ACI | 0.899 | 24.57 | 115.75 s | 0.34 s |
| LSTM+ACI | 0.901 | — | — | — |

Outputs written to `outputs/`.

## Figures (from paper)

### SVM + ACI

**Raw quantile intervals vs. ACI-recalibrated intervals.**

![SVM raw vs ACI](outputs/aci_svm_results/svm_raw_vs_aci.png)

**α trajectory over the test stream** — online adaptation.

![SVM alpha trajectory](outputs/aci_svm_results/svm_alpha_trajectory.png)

**Local PICP comparison (rolling window).**

![SVM local PICP](outputs/aci_svm_results/svm_local_picp_comparison.png)

**PICP / MPIW bar summary.**

![SVM PICP MPIW bars](outputs/aci_svm_results/svm_picp_mpiw_bars.png)

### LSTM + ACI

**Raw vs. ACI intervals.**

![LSTM raw vs ACI](outputs/aci_lstm_results/lstm_raw_vs_aci.png)

**α trajectory.**

![LSTM alpha trajectory](outputs/aci_lstm_results/lstm_alpha_trajectory.png)

**Local PICP comparison.**

![LSTM local PICP](outputs/aci_lstm_results/lstm_local_picp_comparison.png)

**PICP / MPIW bar summary.**

![LSTM PICP MPIW bars](outputs/aci_lstm_results/lstm_picp_mpiw_bars.png)
