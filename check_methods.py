"""Smoke tests for discrete sampling, MLE, KS, and cutoff selection."""

from pathlib import Path
import sys

# Resolve project-local imports independently of the working directory.
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
from main import estimate_alpha, ks_distance, sample_powerlaw, fit_powerlaw


def main():
    rng = np.random.default_rng(12345)
    alpha_true = 2.7
    xmin = 5
    data = sample_powerlaw(alpha_true, xmin, 20000, rng)

    assert data.min() >= xmin
    estimated = estimate_alpha(data, xmin)
    distance = ks_distance(data, xmin, estimated)
    assert abs(estimated - alpha_true) < 0.08, (estimated, alpha_true)
    assert distance < 0.03, distance

    fitted = fit_powerlaw(data)
    assert fitted["n_tail"] >= 100
    assert fitted["xmin"] >= xmin
    assert abs(fitted["alpha"] - alpha_true) < 0.2

    print("PASS: discrete sampling, MLE, KS, and x_min estimation")
    print(f"Known alpha={alpha_true:.2f}; fixed-xmin MLE={estimated:.3f}; KS={distance:.4f}")


if __name__ == "__main__":
    main()
