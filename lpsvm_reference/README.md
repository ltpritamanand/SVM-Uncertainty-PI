# LPSVM Reference Implementation

Reference LPSVM (LP-norm SVM) implementations used in Sec 5.2 (feature selection) and as a methodological baseline. Provided in both MATLAB and Python.

## Files

```
matlab/
  probab_forecast_LPSvm.m      — probabilistic forecasting with LPSVM
  quantileLPONENORMTSVR12.m   — LP 1-norm TSVR quantile solver
  kernelfun.m                  — kernel matrix computation
  evaluate_PICP.m              — PICP/MPIW evaluation
  nobias.m, svtol.m            — helpers

python/
  lpsvm_stage1_stage3.py       — stage 1 (training) + stage 3 (evaluation) pipeline
  quantile_lp_onenorm.py       — Python port of LP 1-norm solver
  kernelfun.py                 — kernel computation
  evaluate_picp.py             — evaluation utilities
  probab_forecast_lpsvm.py     — forecasting wrapper
```

## Run

MATLAB:
```matlab
cd matlab
probab_forecast_LPSvm
```

Python:
```bash
python python/lpsvm_stage1_stage3.py
```

## Dataset

`shared/datasets/beer.csv` — monthly beer production (Australia, 1956–1995).
