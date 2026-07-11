# Appendix A.1 — Ablation Study (Model Size: NN / QRF / XGB)

Three-stage pipeline: (1) hyperparameter tuning, (2) ablation over hidden width / n_estimators / tree depth for NN / QRF / XGB, (3) model comparison bar charts.

## Files

- `ablation_study_model_size.py` — full pipeline (stages 1-3)
- `nn_n_100.py` — standalone NN with n=100 training samples
- `ablation_figures.tex` — LaTeX figure source

## Run

Full pipeline:
```bash
python ablation_study_model_size.py
```

Skip tuning (reuse saved `best_hyperparams_*.json`):
```bash
python ablation_study_model_size.py --skip-tune
```

Ablation only:
```bash
python ablation_study_model_size.py --skip-tune --stage 2
```

**Note:** Stage 1 loads model implementations from `08. Exp 1 Figure 1/(NN)vs(QRF)vs(SVM.py` — run from the `final_github/` root or ensure that file exists.

Outputs written to `results/`.

## Figures (from paper appendix)

### NN — hidden-width ablation

| Sum RMSE vs. hidden size | σ_opt vs. hidden size |
|---|---|
| ![NN sum rmse](outputs/results/nn/nn_hidden_sizes_w_sum_rmse_2026_06_29_17_11_50.png) | ![NN sigma opt](outputs/results/nn/nn_hidden_sizes_w_sigma_opt_2026_06_29_17_11_50.png) |

### QRF — n_estimators / max_features / min_samples_leaf

| Sum RMSE | σ_opt |
|---|---|
| ![QRF n_est rmse](outputs/results/qrf/qrf_n_estimators_sum_rmse_2026_06_29_17_11_50.png) | ![QRF n_est sigma](outputs/results/qrf/qrf_n_estimators_sigma_opt_2026_06_29_17_11_50.png) |
| ![QRF max_feat rmse](outputs/results/qrf/qrf_max_features_sum_rmse_2026_06_29_17_11_50.png) | ![QRF max_feat sigma](outputs/results/qrf/qrf_max_features_sigma_opt_2026_06_29_17_11_50.png) |
| ![QRF min_leaf rmse](outputs/results/qrf/qrf_min_samples_leaf_sum_rmse_2026_06_29_17_11_50.png) | ![QRF min_leaf sigma](outputs/results/qrf/qrf_min_samples_leaf_sigma_opt_2026_06_29_17_11_50.png) |

### XGB — n_estimators / max_depth

| Sum RMSE | σ_opt |
|---|---|
| ![XGB n_est rmse](outputs/results/xgb/xgb_n_estimators_sum_rmse_2026_06_29_17_11_50.png) | ![XGB n_est sigma](outputs/results/xgb/xgb_n_estimators_sigma_opt_2026_06_29_17_11_50.png) |
| ![XGB max_depth rmse](outputs/results/xgb/xgb_max_depth_sum_rmse_2026_06_29_17_11_50.png) | ![XGB max_depth sigma](outputs/results/xgb/xgb_max_depth_sigma_opt_2026_06_29_17_11_50.png) |

### Val vs. test comparison (final model)

| Sum RMSE — val vs. test | σ_opt — val vs. test |
|---|---|
| ![compare rmse](outputs/results/compare/compare_sum_rmse_val_vs_test_2026_06_29_17_11_50.png) | ![compare sigma](outputs/results/compare/compare_sigma_opt_val_vs_test_2026_06_29_17_11_50.png) |

XGB with subsample = 1.0 variant: [xgb_1sub/](outputs/results/xgb_1sub/). NN with n = 100: [results_nn_n100/](outputs/results_nn_n100/).
