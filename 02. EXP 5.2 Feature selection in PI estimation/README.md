# Sec 5.2 — Feature Selection in PI Estimation (LPSVM)

Demonstrates LPSVM-based feature selection across five UCI datasets. Ranks features by SVM weight magnitude and compares PICP / MPIW / runtime before and after selection.

## Files

- `boston_svm_feature_selection.py`, `student_svm_feature_selection.py`, `spambase_svm_feature_selection.py`, `secom_svm_feature_selection.py`, `madelon_svm_feature_selection.py` — **per-dataset entry-point scripts**. Each hardcodes the best `(s, c1, c3, threshold)` for its dataset and prints one Before-FS / After-FS row in the paper's table format. All five use tuned params from a grid search.
- `all_svm_feature_selection.py` — runs all five datasets and prints the full comparison table.
- `_runner.py` — shared logic (dataset loader, LP quantile SVR fit, feature drop, metrics) imported by every per-dataset script.
- `data/` — bundled datasets used by this experiment:
  - `bostonhousingdata.xlsx` — Boston Housing (UCI)
  - `MADELON/` — MADELON feature-selection benchmark (UCI): `madelon_train.data`, `madelon_train.labels`, `madelon_valid.data`, `madelon_test.data`, `madelon.param`
  - `spambase.data` — Spambase (UCI, 4601 × 58)
  - `student-mat.csv` — Student Performance (UCI, 395 × 16)
  - `uci-secom.csv` — SECOM semiconductor manufacturing (UCI)

## Run

**One dataset at a time — pick the file for the dataset you want:**

```bash
python boston_svm_feature_selection.py
python student_svm_feature_selection.py
python spambase_svm_feature_selection.py
python secom_svm_feature_selection.py
python madelon_svm_feature_selection.py
```

**All datasets at once (paper-style comparison table):**

```bash
python all_svm_feature_selection.py
```

## Notes

- Features are standardized (LP quantile SVR is scale-sensitive on mixed-scale data like Spambase / SECOM).
- The dataset loader in `_runner.py` auto-dispatches on file extension — `.xlsx` uses `read_excel`; `.csv` / `.data` uses `read_csv` with NaN/±inf sanitization. MADELON has a dedicated branch that joins its separate features + labels files.
- Spambase and MADELON have classification-style targets (0/1 and ±1). PICP/MPIW are computed the same way as regression, matching the paper's Table 2 convention.
- Timings depend on machine (paper numbers are indicative, not exact).
