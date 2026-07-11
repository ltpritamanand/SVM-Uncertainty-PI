# Best params from tune grid — reproduces the paper's Student Performance row.
from _runner import run_dataset

if __name__ == "__main__":
    run_dataset("student", s=2**0, c1=2**-6, c3=0.1, threshold=1e-5)
