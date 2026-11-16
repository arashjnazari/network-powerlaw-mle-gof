"""Discrete power-law inference on BA and ER network degree distributions.

Implements exact discrete MLE, KS-based x_min selection and a
semiparametric bootstrap goodness-of-fit test.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import zeta


# Reproducible experiment configuration.
SEED = 2026
N_NODES = 4000
BA_M = 3
BOOTSTRAPS = 1000
MIN_TAIL = 100
OUT = Path(__file__).resolve().parent / "output"


def estimate_alpha(tail, xmin):
    """Exact MLE for the discrete power law P(k) ~ k^(-alpha), k>=xmin."""
    sum_logs = np.log(tail).sum()
    n = len(tail)

    def negative_log_likelihood(alpha):
        return alpha * sum_logs + n * np.log(zeta(alpha, xmin))

    result = minimize_scalar(
        negative_log_likelihood,
        method="bounded",
        bounds=(1.01, 20.0),
        options={"xatol": 1e-5},
    )
    if not result.success:
        raise RuntimeError("MLE optimization did not converge")
    return float(result.x)


def ks_distance(tail, xmin, alpha):
    """KS distance between empirical and fitted discrete tail CDFs."""
    sorted_tail = np.sort(tail)
    k = np.arange(xmin, int(sorted_tail[-1]) + 1)
    empirical_cdf = np.searchsorted(sorted_tail, k, side="right") / len(tail)
    model_cdf = 1.0 - zeta(alpha, k + 1) / zeta(alpha, xmin)
    return float(np.max(np.abs(empirical_cdf - model_cdf)))


def fit_powerlaw(degrees, min_tail=MIN_TAIL):
    """Find xmin minimizing KS distance, re-estimating alpha for each xmin."""
    degrees = np.asarray(degrees, dtype=np.int64)
    candidates = np.unique(degrees)
    best = None

    for xmin in candidates:
        if xmin < 1:
            continue  # Degree zero cannot follow this positive-integer model.
        tail = degrees[degrees >= xmin]
        if len(tail) < min_tail:
            break  # Later candidates have even fewer observations.
        alpha = estimate_alpha(tail, int(xmin))
        distance = ks_distance(tail, int(xmin), alpha)
        candidate = {
            "alpha": alpha,
            "xmin": int(xmin),
            "ks": distance,
            "n_tail": len(tail),
        }
        if best is None or candidate["ks"] < best["ks"]:
            best = candidate

    if best is None:
        raise ValueError(f"Not enough positive degrees for a tail of {min_tail}")
    return best


def sample_powerlaw(alpha, xmin, size, rng):
    """Exact discrete power-law sampling by continuous-envelope rejection.

    Round Pareto draws into integer bins. The acceptance probability corrects
    for each bin's integrated density, yielding mass proportional to k^-alpha.
    """
    result = np.empty(size, dtype=np.int64)
    filled = 0
    while filled < size:
        remaining = size - filled
        u = np.maximum(rng.random(remaining + 32), np.finfo(float).tiny)
        continuous = (xmin - 0.5) * u ** (-1.0 / (alpha - 1.0))
        k = np.floor(continuous + 0.5).astype(np.int64)
        bin_integral = (
            (k - 0.5) ** (1.0 - alpha)
            - (k + 0.5) ** (1.0 - alpha)
        ) / (alpha - 1.0)
        accept_probability = k.astype(float) ** (-alpha) / bin_integral
        accepted = k[rng.random(len(k)) < accept_probability]
        take = min(remaining, len(accepted))
        result[filled : filled + take] = accepted[:take]
        filled += take
    return result


def goodness_of_fit(degrees, fitted, bootstraps, rng):
    """Semiparametric bootstrap KS p-value with xmin and alpha refitted."""
    n = len(degrees)
    xmin = fitted["xmin"]
    below = degrees[degrees < xmin]
    fraction_tail = fitted["n_tail"] / n
    exceedances = 0

    for replication in range(bootstraps):
        n_tail = rng.binomial(n, fraction_tail)
        n_body = n - n_tail
        body = rng.choice(below, n_body, replace=True) if n_body else np.array([], dtype=int)
        tail = sample_powerlaw(fitted["alpha"], xmin, n_tail, rng)
        synthetic = np.concatenate((body, tail))
        refitted = fit_powerlaw(synthetic)
        exceedances += refitted["ks"] >= fitted["ks"]
        if (replication + 1) % 100 == 0 or replication + 1 == bootstraps:
            print(f"  Bootstrap: {replication + 1}/{bootstraps}", flush=True)

    # Add-one correction avoids reporting p=0 from a finite simulation.
    return (exceedances + 1) / (bootstraps + 1)


def save_degree_data_and_plot(degrees, fitted, name, output_dir):
    counts = np.bincount(degrees)
    ks = np.flatnonzero(counts)
    empirical_ccdf = np.cumsum(counts[::-1])[::-1][ks] / len(degrees)
    pd.DataFrame({
        "degree": ks,
        "count": counts[ks],
        "fraction": counts[ks] / len(degrees),
        "ccdf": empirical_ccdf,
    }).to_csv(output_dir / f"{name.lower()}_degrees.csv", index=False)

    positive = ks > 0
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(ks[positive], empirical_ccdf[positive], "o", markersize=4,
              label=f"{name} empirical CCDF")
    xmin = fitted["xmin"]
    fit_ks = np.arange(xmin, int(degrees.max()) + 1)
    p_tail = fitted["n_tail"] / len(degrees)
    model_ccdf = p_tail * zeta(fitted["alpha"], fit_ks) / zeta(fitted["alpha"], xmin)
    ax.loglog(fit_ks, model_ccdf, "-", linewidth=2,
              label=f"Discrete power law: alpha={fitted['alpha']:.2f}, xmin={xmin}")
    ax.axvline(xmin, linestyle=":", linewidth=1, color="gray")
    ax.set(xlabel="Degree k", ylabel="P(K >= k)", title=f"{name} degree distribution")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / f"{name.lower()}_ccdf.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    er_p = (2 * BA_M) / (N_NODES - 1)
    graphs = {
        "BA": nx.barabasi_albert_graph(N_NODES, BA_M, seed=SEED),
        "ER": nx.erdos_renyi_graph(N_NODES, er_p, seed=SEED),
    }
    rows = []

    for name, graph in graphs.items():
        degrees = np.fromiter((d for _, d in graph.degree()), dtype=np.int64)
        fit = fit_powerlaw(degrees)
        print(f"\n{name}: fitting and running {BOOTSTRAPS} bootstrap replications...", flush=True)
        p_value = goodness_of_fit(degrees, fit, BOOTSTRAPS, rng)
        save_degree_data_and_plot(degrees, fit, name, OUT)
        row = {
            "network": name,
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "mean_degree": float(degrees.mean()),
            "alpha_mle": fit["alpha"],
            "xmin": fit["xmin"],
            "n_tail": fit["n_tail"],
            "tail_fraction": fit["n_tail"] / len(degrees),
            "ks": fit["ks"],
            "p_bootstrap": p_value,
            "bootstraps": BOOTSTRAPS,
        }
        rows.append(row)
        print(f"{name}: alpha={fit['alpha']:.3f}, xmin={fit['xmin']}, "
              f"n_tail={fit['n_tail']}, KS={fit['ks']:.4f}, p={p_value:.3f}")

    pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
    print(f"Results saved in: {OUT}")


if __name__ == "__main__":
    main()
