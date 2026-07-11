# Runs all five datasets and prints one paper-style summary table.
from _runner import run_dataset, print_table_header

# (name, s, c1, c3, threshold) — all tuned from the grid search.
DATASET_PARAMS = [
    ("boston",   2**0, 2**-6, 0.1, 1e-5),
    ("student",  2**0, 2**-6, 0.1, 1e-5),
    ("spambase", 2**0, 2** 8, 0.1, 5e-3),
    ("secom",    2**0, 2**-6, 0.1, 1e-5),
    ("madelon",  2**0, 2** 0, 0.1, 5e-2),
]

if __name__ == "__main__":
    print_table_header()
    for name, s, c1, c3, thr in DATASET_PARAMS:
        run_dataset(name, s=s, c1=c1, c3=c3, threshold=thr, verbose=True)
