# Best params from tune grid — reproduces the paper's Spambase row.
from _runner import run_dataset

if __name__ == "__main__":
    run_dataset("spambase", s=2**0, c1=2**8, c3=0.1, threshold=5e-3)
