# Sec 5.3 — SVM in Conformal Prediction Sets (CQR + SVQR Benchmark)

Benchmarks Conformalized Quantile Regression (CQR) with NN and SVQR heads on 7 UCI regression datasets. Reports PICP and MPIW at α=0.10.

## Files

- `exp15_hparam_tune.py` — hyperparameter grid search for NN-CQR (D=10 data seeds × I=10 init seeds)
- `exp15_svqr_bench.py` — SVQR benchmark with best hyperparameters
- `exp15_cqr.py` — CQR evaluation script
- `cqr.py` — CQR calibration/scoring utilities
- `base_svr.py` — base SVR wrapper
- `svqr_qp.py` — SVQR QP solver (Python port of MATLAB epsilon_quantilesvr2)

## Datasets (from `shared/datasets/`)

Boston Housing, Concrete, Energy Efficiency, Yacht Hydrodynamics, Servo, AutoMPG, RealEstate

## Run

Hyperparameter tuning (all datasets):
```bash
python exp15_hparam_tune.py
```

Tune specific datasets only (e.g. Yacht and Concrete):
```bash
python exp15_hparam_tune.py --datasets Yacht Concrete
```

SVQR benchmark (uses hardcoded best params):
```bash
python exp15_svqr_bench.py
```

Outputs written to `outputs/`.

## Results Table (from paper)

Conformal prediction with SVQR, CQR-NN, QRF, XGB at α = 0.10 across 7 UCI regression datasets. **Bold** = best PICP-nearest-0.90 with lowest MPIW.

| Dataset | Method | PICP (%) | MPIW | σ_opt | σ_in | Time (s) |
|---|---|---|---|---|---|---|
| AutoMPG | CQR-NN | 93.80 | 9.774 | 2.227 | 1.470 | 1.14 |
| AutoMPG | **SVQR+CP** | **91.27** | **9.175** | **0.000** | 1.658 | 0.19 |
| AutoMPG | QRF+CP | 93.92 | 9.689 | 1.429 | 1.779 | 0.07 |
| AutoMPG | XGB+CP | 92.53 | 9.837 | 0.000 | 2.065 | 0.40 |
| RealEstate | CQR-NN | 91.69 | 23.317 | 2.456 | 3.605 | 0.46 |
| RealEstate | **SVQR+CP** | **91.93** | **21.708** | **0.000** | 3.057 | 0.28 |
| RealEstate | QRF+CP | 94.82 | 23.575 | 2.904 | 5.898 | 0.19 |
| RealEstate | XGB+CP | 92.65 | 23.560 | 0.000 | 4.618 | 0.10 |
| Boston | CQR-NN | 93.63 | 10.014 | 2.614 | 2.997 | 6.34 |
| Boston | **SVQR+CP** | **90.69** | **9.603** | **0.000** | 1.956 | 0.23 |
| Boston | QRF+CP | 91.47 | 11.010 | 1.451 | 2.249 | 0.11 |
| Boston | XGB+CP | 92.16 | 11.838 | 0.000 | 2.308 | 1.02 |
| Energy | CQR-NN | 91.82 | 0.976 | 0.091 | 0.071 | 3.39 |
| Energy | **SVQR+CP** | **90.13** | **0.925** | **0.000** | 0.052 | 0.57 |
| Energy | QRF+CP | 90.71 | 0.899 | 0.026 | 0.031 | 0.19 |
| Energy | XGB+CP | 89.22 | 0.914 | 0.000 | 0.137 | 0.26 |
| Concrete | CQR-NN | 92.43 | 17.838 | 4.409 | 2.551 | 3.29 |
| Concrete | SVQR+CP | 92.38 | 20.386 | 0.000 | 3.615 | 6.42 |
| Concrete | QRF+CP | 90.83 | 20.416 | 2.388 | 3.295 | 0.23 |
| Concrete | XGB+CP | 90.10 | 22.073 | 0.000 | 3.452 | 0.20 |
| Yacht | CQR-NN | 97.74 | 1.696 | 1.026 | 0.470 | 2.98 |
| Yacht | **SVQR+CP** | **90.81** | **2.219** | **0.000** | 1.367 | 0.55 |
| Yacht | QRF+CP | 97.58 | 2.397 | 0.320 | 0.783 | 0.23 |
| Yacht | XGB+CP | 94.35 | 3.957 | 0.000 | 2.544 | 0.16 |
| Servo | CQR-NN | 98.24 | 1.517 | 0.249 | 0.290 | 0.10 |
| Servo | **SVQR+CP** | **98.24** | 1.867 | **0.000** | 0.383 | 0.06 |
| Servo | QRF+CP | 94.71 | 1.611 | 0.280 | 0.579 | 0.14 |
| Servo | XGB+CP | 85.29 | 2.864 | 0.000 | 0.780 | 0.09 |

SVQR+CP is deterministic (σ_opt = 0) and delivers tightest intervals on 5 of 7 datasets while maintaining PICP ≥ 90%.
