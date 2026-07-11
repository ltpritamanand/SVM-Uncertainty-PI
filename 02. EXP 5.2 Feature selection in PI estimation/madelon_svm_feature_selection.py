# MADELON — best params from tune grid.
from _runner import run_dataset

if __name__ == "__main__":
    run_dataset("madelon", s=2**0, c1=2**0, c3=0.1, threshold=5e-2)
