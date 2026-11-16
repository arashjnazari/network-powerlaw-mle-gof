"""Analyze BA power-law tail sensitivity to the lower cutoff xmin."""

from pathlib import Path
import sys

# Resolve project-local imports independently of the working directory.
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from main import (
    BA_M,
    MIN_TAIL,
    N_NODES,
    OUT,
    SEED,
    estimate_alpha,
    fit_powerlaw,
    ks_distance,
)


def analyse_cutoffs(degrees, min_tail=MIN_TAIL):
    """Refit exact discrete MLE and KS at each eligible observed x_min."""
    rows = []
    degrees = np.asarray(degrees, dtype=np.int64)

    for xmin in np.unique(degrees):
        if xmin < 1:
            continue
        tail = degrees[degrees >= xmin]
        if len(tail) < min_tail:
            break

        alpha = estimate_alpha(tail, int(xmin))
        ks = ks_distance(tail, int(xmin), alpha)
        rows.append({
            "xmin": int(xmin),
            "alpha_mle": float(alpha),
            "ks": float(ks),
            "n_tail": int(len(tail)),
            "tail_fraction": float(len(tail) / len(degrees)),
        })

    if not rows:
        raise ValueError("No cutoff has enough observations in the tail.")
    return pd.DataFrame(rows)


def draw_plots(df, selected_xmin, output_dir):
    """Save exponent, KS-distance, and tail-size sensitivity plots."""
    x = df["xmin"].to_numpy()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, df["alpha_mle"], "o-", label="MLE estimate")
    ax.axhline(3.0, color="gray", linestyle="--", label="BA asymptotic exponent = 3")
    ax.axvline(selected_xmin, color="black", linestyle=":", label="Minimum-KS cutoff")
    ax.set(xlabel="Lower cutoff (x_min)", ylabel="Estimated exponent (alpha)",
           title="BA network: sensitivity of the power-law exponent")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ba_alpha_vs_xmin.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, df["ks"], "o-")
    ax.axvline(selected_xmin, color="black", linestyle=":", label="Minimum-KS cutoff")
    ax.set(xlabel="Lower cutoff (x_min)", ylabel="KS distance",
           title="BA network: KS distance at each fitted cutoff")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ba_ks_vs_xmin.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, df["n_tail"], "o-")
    ax.axvline(selected_xmin, color="black", linestyle=":", label="Minimum-KS cutoff")
    ax.set(xlabel="Lower cutoff (x_min)", ylabel="Number of nodes in tail",
           title="BA network: sample size decreases as cutoff rises")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "ba_tail_size_vs_xmin.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    # Use the same graph parameters and seed as the main analysis.
    graph = nx.barabasi_albert_graph(N_NODES, BA_M, seed=SEED)
    degrees = np.fromiter((degree for _, degree in graph.degree()), dtype=np.int64)
    df = analyse_cutoffs(degrees)

    selected = fit_powerlaw(degrees, min_tail=MIN_TAIL)
    # Both routines search the same cutoffs, fitting alpha and then computing KS.
    best_row = df.loc[df["ks"].idxmin()]
    assert int(best_row["xmin"]) == selected["xmin"]
    assert np.isclose(best_row["alpha_mle"], selected["alpha"])

    csv_file = OUT / "ba_xmin_sensitivity.csv"
    df.to_csv(csv_file, index=False)
    draw_plots(df, selected["xmin"], OUT)

    print("BA x_min sensitivity (exact discrete MLE):")
    print(df.to_string(index=False, formatters={
        "alpha_mle": "{:.3f}".format,
        "ks": "{:.4f}".format,
        "tail_fraction": "{:.3f}".format,
    }))
    print(f"\nBest KS: x_min={selected['xmin']}, "
          f"alpha={selected['alpha']:.3f}, KS={selected['ks']:.4f}, "
          f"n_tail={selected['n_tail']}")
    print("An exponent close to 3 does not establish goodness of fit.")
    print("Higher cutoffs reduce the tail sample size and can increase estimation uncertainty.")
    print(f"Saved CSV and 3 plots to: {OUT}")


if __name__ == "__main__":
    main()
