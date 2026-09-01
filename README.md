# Support Vector Machines (SVMs) : A More Certain Estimate of Uncertainty 

Companion code for the paper submitted to TMLR.

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
├── 04. EXP 5.4 SVM for probabilistic forecasting/                # Sec 5.4 — SVQR / Mamba forecasting (births dataset)
├── 05. EXP 5.4.1 Scalability Analysis - .../                     # Sec 5.4.1 — SVM vs deep learning at scale (Amprion)
├── 06. EXP 5.4.2 SVM probabilistic forecasting .../              # Sec 5.4.2 — ACI online calibration (SVM + LSTM)
├── 07. EXP 5.4.3 Pretrained zero-shot time-series model/         # Sec 5.4.3 — Chronos zero-shot forecasting
├── 08. Exp 1 Figure 1/                                           # Figure 1 — stability comparison (NN / QRF / SVM / XGB)
├── 09. Appendix A 1 Ablation study/                              # Appendix A.1 — NN/QRF/XGB ablation study
├── 10. Appendix A 2 Test Size study/                             # Appendix A.2 — test-size sensitivity sweep
└── lpsvm_reference/                                              # Reference LPSVM implementation (MATLAB + Python)
```

## Dependencies

```
pip install -r requirements.txt
```

MATLAB experiments require MATLAB R2021b+ with the Optimization Toolbox (`quadprog`).

## Reproducibility

Each subfolder contains its own README with run instructions. All datasets are bundled in `shared/datasets/`. Seeds are fixed as documented in each script.
