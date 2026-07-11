# shared/

Resources used by multiple sections. Scripts resolve paths relative to this directory automatically — do not move it.

## datasets/

All datasets required by any experiment in this repo. No internet access needed at runtime.

| File | Used by | Description |
|---|---|---|
| `Amprion.csv` | sec5.5, sec5.6, sec5.7 | Hourly electricity load (Amprion TSO) |
| `Daily-total-female-births.csv` | sec5.4 | Daily births 1959 (UCI) |
| `bostonhousingdata.xlsx` | sec5.3 | Boston Housing (UCI) |
| `Concrete_Data.csv` | sec5.3 | Concrete compressive strength (UCI) |
| `energy_efficiency.csv` | sec5.3 | Building energy efficiency (UCI) |
| `yacht_hydrodynamics.data` | sec5.3 | Yacht hydrodynamics (UCI) |
| `servo.data` | sec5.3 | Servo motor (UCI) |
| `auto-mpg.data` | sec5.3 | Auto MPG (UCI, 392 rows) |
| `real_estate_valuation.xlsx` | sec5.3 | Real estate prices, Taiwan (UCI) |
| `beer.csv` | lpsvm_reference | Monthly beer production (Australia) |
| `MADELON/` | sec5.2 | MADELON feature selection benchmark (UCI) |

## matlab_utils/

Shared MATLAB helper functions.

| File | Purpose |
|---|---|
| `train_qsvr.m` | Solve SVQR QP once, return frozen model struct |
| `predict_qsvr.m` | Pure inference using frozen model struct |
| `kernelfun.m` | RBF / polynomial kernel matrix |

## src_utils/

Shared Python utilities imported by `08. Exp 1 Figure 1` and `09. Appendix A 1 Ablation study`.

| File | Purpose |
|---|---|
| `run_synthetic_tube_quantile_ensembles.py` | NN / QRF / SVM / XGB model wrappers for sinc experiments |
| `evaluate_nn_ensemble.py` | Ensemble evaluation (σ_opt / σ_in decomposition) |
| `feature_selection/selectors.py` | LPSVM feature selector (loads MADELON from `datasets/`) |
| `utils/datasets.py` | Dataset loader utilities |
