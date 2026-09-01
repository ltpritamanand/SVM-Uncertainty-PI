# Support Vector Machines (SVMs): A More Certain Estimate of Uncertainty

Companion code for the paper submitted to TMLR.

## Overview
The main contributions of this paper center on
establishing distinctive and compelling properties of SVM models that strongly support their promotion for
UQ regression tasks, especially under small-sample setting


The repository is organized as follows : 


## Structure

```
final_github/
├── shared/                                                       # Datasets + MATLAB utilities (shared across sections)
│   ├── datasets/                                                 # All UCI/benchmark datasets
│   └── matlab_utils/                                             # train_qsvr.m, predict_qsvr.m, kernelfun.m, etc.
│
├── 01. EXP 5.1 SVMs are less uncertain for PI estimation/        # Sec 5.1 — σ_opt / σ_in decomposition sweep
├── 02. EXP 5.2 Feature selection in PI estimation/               # Sec 5.2 — LPSVM feature selection
├── 03. EXP 5.3 SVM in conformal prediction sets/                 # Sec 5.3 — CQR + SVQR benchmark (7 datasets)
├── 04. EXP 5.4 SVM for probabilistic forecasting/                # Sec 5.4 — SVQR / Mamba forecasting (3 time series)
├── 05. EXP 5.4.1 Scalability Analysis - .../                     # Sec 5.4.1 — SVM vs deep learning at scale (Amprion)
├── 06. EXP 5.4.2 SVM probabilistic forecasting .../              # Sec 5.4.2 — ACI online calibration (SVM + LSTM)
├── 07. EXP 5.4.3 Pretrained zero-shot time-series model/         # Sec 5.4.3 — Chronos zero-shot forecasting
├── 08. Exp 1 Figure 1/                                           # Figure 1 — stability comparison (NN / QRF / SVM / XGB / GPR / NGB)
├── 09. Appendix A 1 Ablation study/                              # Appendix A.1 — NN/QRF/XGB model-size ablation
├── 10. Appendix A 2 Test Size study/                             # Appendix A.2 — test-size sensitivity sweep
└── lpsvm_reference/                                              # Reference LPSVM implementation (MATLAB + Python)
```

## What each experiment shows

| Folder | Section | What it demonstrates |
|---|---|---|
| `01.` | 5.1 | Introduces the σ_opt / σ_samp decomposition and sweeps it across models — the paper's core diagnostic |
| `02.` | 5.2 | LPSVM feature selection (Boston, Student, Spambase, SECOM, Madelon) — SVM-weight-ranked feature selection improves PICP/MPIW/runtime without a black-box retrain |
| `03.` | 5.3 | Conformalized SVQR (SVQR+CP) vs. CQR-NN, QRF+CP, XGB+CP on 7 UCI datasets — SVQR+CP stays at σ_opt = 0 and wins tightest MPIW on 5/7 datasets while holding PICP ≥ 90% |
| `04.` | 5.4 | SVQR vs. Mamba on 3 univariate series (female births, min. temperature, beer production) — SVQR matches Mamba's ~0.95 coverage with zero run-to-run variance and far less compute |
| `05.` | 5.4.1 | SVQR vs. NN/LSTM/GRU/TCN on the Amprion electricity load dataset (up to ~16k samples) — scalability of PICP/MPIW/training time as data grows |
| `06.` | 5.4.2 | Adaptive Conformal Inference (Gibbs & Candès 2021) applied to SVM and LSTM under distribution shift — online-recalibrated SVQR (PICP 0.899) is competitive with LSTM (PICP 0.901) at much lower train/inference cost |
| `07.` | 5.4.3 | Amazon Chronos (zero-shot foundation model) evaluated via repeated unseeded forward passes — shows that even pretrained foundation models carry nonzero σ_opt |
| `08.` | Fig. 1 / Table B1 | Direct visual comparison: NN vs. QRF vs. SVM vs. XGB vs. GPR vs. NGB, M = 10 independently trained ensembles on a synthetic sinc function — SVM/GPR envelopes overlap tightly (σ_opt = 0); NN/QRF/XGB/NGB visibly jitter |
| `09.` | Appendix A.1 | Ablation over model capacity (NN hidden width, QRF n_estimators/max_features/min_samples_leaf, XGB n_estimators/max_depth) — shows instability isn't resolved by tuning capacity alone |
| `10.` | Appendix A.2 | Test-set-size sensitivity sweep (100 → 600 samples) — confirms the σ_opt/σ_samp decomposition and RMSE results aren't an artifact of test-split size |


## Dependencies

```bash
pip install -r requirements.txt
```

MATLAB experiments require MATLAB R2021b+ with the Optimization Toolbox (`quadprog`).

## Reproducibility

- Each subfolder contains its own README with detailed run instructions, key hyperparameters, and figure/table references back to the paper.
- All datasets are bundled in `shared/datasets/` (and, where relevant, duplicated locally within a section's folder).

