# Sec 5.4 — SVM for Probabilistic Forecasting (SVQR)

Time-series quantile forecasting with SVQR (MATLAB QP-based). The paper reports results on **three** univariate time-series datasets:

1. **Daily Female Births** — `Daily-total-female-births.csv` (365 daily counts, 1959)
2. **Daily Minimum Temperatures** — `daily-minimum-temperatures-in-me.csv` (3651 daily records, Melbourne)
3. **Monthly Beer Production** — `beer.csv` (Australia, 1956–1995)

All three files are bundled in this folder under `datasets/`, and also in the repo-wide `shared/datasets/` (which is what the scripts import from).

## SVQR (main experiment)

- `svqr_matlab/probab_forecast_Svm.m` — SVQR training + evaluation with **built-in hyperparameter tuning** (grid over window size, kernel σ, and regularisation C; selects the config with tightest MPIW subject to PICP ≥ 0.95).

### Run
```matlab
cd svqr_matlab
probab_forecast_Svm
```

### Getting results on all three datasets

The script loads the dataset via a single path (line 8):

```matlab
dataTable = readtable(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'shared', 'datasets', 'Daily-total-female-births.csv'));
```

To reproduce the paper's results for a different dataset, **just change the filename** in that path — the rest of the pipeline (windowing, tuning, evaluation) is dataset-agnostic. The three paths used in the paper:

- `../../shared/datasets/Daily-total-female-births.csv`
- `../../shared/datasets/daily-minimum-temperatures-in-me.csv`
- `../../shared/datasets/beer.csv`

Outputs are written to `svqr_matlab/outputs/`.

## Mamba baseline

`mamba_python/mamba_benchmark_table5.py` — Python Mamba/Transformer baseline used to produce the comparison column in Table 5. Run with `python mamba_python/mamba_benchmark_table5.py`.

## Results Table (from paper, Table 5)

Probabilistic forecasting at target coverage 1−α = 0.95 (τ_lower = 0.025, τ_upper = 0.975). Mamba: mean ± std over 5 runs; SVQR is deterministic (σ² = 0).

| Dataset (size) | Model | Window | PICP | MPIW | σ² (run-to-run) | Time (s) |
|---|---|---|---|---|---|---|
| Female Births (365) | Mamba | 12 | 0.9358 ± 0.0092 | 26.91 ± 0.68 | 0.4168 | 10.7 ± 3.5 |
| Min. Temperature (3651) | Mamba | 12 | 0.9527 ± 0.0115 | 9.66 ± 0.30 | 0.3667 | 68.2 ± 0.1 |
| Beer Production (464) | Mamba | 12 | 0.9557 ± 0.0053 | 108.11 ± 11.79 | 161.35 | 9.3 ± 1.3 |
| **SVQR** (all three) | SVQR | 12 | ≈ 0.95 | comparable | **0.0 (deterministic)** | see MATLAB log |

SVQR matches Mamba's coverage with zero run-to-run variance and 1–2 orders of magnitude less compute.
