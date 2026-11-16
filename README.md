# Discrete Power-Law MLE and Goodness-of-Fit in Networks

A reproducible Python experiment on **degree distributions** in two synthetic networks:

- **BA**: Barabási–Albert preferential-attachment network.
- **ER**: Erdős–Rényi random network with a comparable expected mean degree.

The analysis evaluates fitted discrete power-law tails using bootstrap goodness-of-fit diagnostics, lower-cutoff sensitivity, and comparison with alternative tail distributions.

**Scope:** exploratory analysis of marginal degree distributions. The bootstrap and product likelihood treat degree observations as independent, whereas degrees within a network are dependent. Consequently, the reported p-values and AIC scores are **not calibrated tests of the BA or ER graph-generating mechanisms**.

## Research workflow

1. Generate BA (`n=4000, m=3`) and ER (`n=4000, p=6/3999`) networks with `SEED = 2026`.
2. Compute the complete integer degree sequence of each network.
3. For each eligible observed `xmin`, fit the discrete power-law exponent by numerical maximum likelihood:

   `P(X=x | X>=xmin) = x^(-alpha) / zeta(alpha, xmin)`

   `log L(alpha) = -alpha * sum(log(x_i)) - n_tail * log(zeta(alpha, xmin))`

   Here `zeta(alpha, xmin)` is the Hurwitz zeta function. The likelihood is the discrete model, not the continuous Pareto approximation or ordinary least squares on a log–log plot.
4. Select `xmin` by minimizing the discrete Kolmogorov–Smirnov (KS) distance, requiring at least `MIN_TAIL = 100` tail observations.
5. Calculate a semiparametric bootstrap KS p-value. Resample observations below `xmin` from the empirical body and generate the tail from the fitted discrete power law. Refit both `xmin` and `alpha` on every synthetic dataset. The documented run uses **1,000 replications**.
6. Plot empirical complementary cumulative distribution functions (CCDFs) and the fitted power-law tails; export CSV and PNG results.
7. Examine BA cutoff sensitivity: estimate `alpha`, KS, and `n_tail` at each eligible `xmin`.
8. Compare four discrete models on exactly the same selected tail within each network: power law, discrete exponential, binned lognormal, and lower-truncated Poisson.

## Installation and execution

Install the dependencies and execute the workflow from the project directory:

```bash
python -m pip install -r requirements.txt
python run_all.py
```

The workflow uses the active Python interpreter and runs the following scripts in order:

| Step | File | Purpose |
| --- | --- | --- |
| 1 | `check_methods.py` | Smoke tests of discrete sampling, MLE, KS, and cutoff search. Not exhaustive validation. |
| 2 | `main.py` | Generate the networks, fit the power-law tails, run the bootstrap, and save the base results. |
| 3 | `xmin_sensitivity.py` | Recreate the seeded BA graph and save cutoff-sensitivity results. |
| 4 | `compare_models.py` | Read the saved counts and summary, then compare all four tail models. |

The workflow stops if a script fails. A successful run generates five CSV files and
seven PNG figures in `output/`, prints `ALL STEPS COMPLETED SUCCESSFULLY`, and
replaces existing outputs with the same filenames. No external data download is
required.

### Configuration

The experiment parameters are defined in `main.py`:

```python
SEED = 2026
N_NODES = 4000
BA_M = 3
BOOTSTRAPS = 1000
MIN_TAIL = 100
```

### Repository contents

```text
network-powerlaw-mle-gof/
├── README.md
├── requirements.txt
├── .gitignore
├── run_all.py
├── main.py
├── check_methods.py
├── xmin_sensitivity.py
├── compare_models.py
└── output/
```

The `output/` directory is created automatically. Analysis outputs are not excluded
by `.gitignore`; Python caches, local environments, and editor metadata are excluded.

## Output

| Files in `output/` | Contents |
| --- | --- |
| `summary.csv` | Network summaries, `alpha_mle`, `xmin`, `n_tail`, `tail_fraction`, KS, `p_bootstrap`, and the replication count. |
| `ba_degrees.csv`, `er_degrees.csv` | Empirical degree counts, frequencies, and CCDF values. |
| `ba_ccdf.png`, `er_ccdf.png` | Empirical CCDFs with the selected power-law tail, scaled by its fraction of the full sample. |
| `ba_xmin_sensitivity.csv` | BA estimates and sample sizes at the eligible cutoffs. |
| `ba_alpha_vs_xmin.png`, `ba_ks_vs_xmin.png`, `ba_tail_size_vs_xmin.png` | Cutoff-sensitivity plots. |
| `model_comparison.csv` | Model parameters, log-likelihoods, AIC, and within-network `delta_AIC`. |
| `ba_model_comparison.png`, `er_model_comparison.png` | Conditional empirical CCDFs and all four fitted models on a common tail. |

All network data are **synthetically generated**; no empirical biological or collaboration datasets are used.

## Results (seed 2026, 1,000 bootstrap replications)

Values below reproduce the rounded console output from the documented run; `summary.csv` retains the numerical values saved by the program.

| Network | MLE alpha | xmin | Tail nodes | Tail fraction | KS | Bootstrap p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BA | 2.643 | 3 | 4000 | 100% | 0.0130 | 0.016 |
| ER | 9.099 | 10 | 346 | 8.65% | 0.0498 | 0.001 |

Both p-values fall below the project's nominal rejection threshold of `0.1` (and also below `0.05`). **Under the bootstrap's independent-observation reference model, neither selected pure power-law tail passes the goodness-of-fit test.** Because within-network dependence is not reproduced by this bootstrap, this is an exploratory diagnostic rather than a calibrated rejection of a network-generating model.

The code uses an add-one correction:

`p_bootstrap = (exceedances + 1) / (BOOTSTRAPS + 1)`

An exceedance is a synthetic dataset whose refitted KS distance is at least as large as the observed distance. With 1,000 replications, the rounded results above correspond to 15 exceedances for BA and none for ER. The ER value is therefore at the simulation's reporting floor, `1/1001 ≈ 0.001`, **not an exact probability of 0.001**. A p-value is not the probability that the fitted model is true.

### Relation to the BA asymptotic degree distribution

The standard stationary, large-network BA degree distribution is [2]:

`P(X=x) = 2*m*(m+1) / [x*(x+1)*(x+2)], x >= m`

It approaches `x^(-3)` for large degree, but it is not a pure power law at small degree. In this run, `xmin=3` includes all 4,000 nodes, including the small-degree region. The distinction between an asymptotic power-law tail and a pure power law over the entire fitted range can therefore help explain the discrepancy; it does not establish the cause of the observed p-value by itself. Finite network size and within-network dependence also require caution.

A fitted `alpha=2.643` is not a redefinition of the BA asymptotic exponent. Likewise, an estimate closer to 3 at a higher cutoff is not, by itself, evidence of a better or acceptable fit. The sensitivity plots report the accompanying KS distances and shrinking tail sample sizes. Cutoff selection is based on KS minimization rather than proximity to a target exponent or p-value.

## Model comparison on the same tail

All four likelihoods are normalized over integer degrees `x >= xmin`:

| Model | Tail probability model | Fitted parameters |
| --- | --- | --- |
| Power law | `x^(-alpha) / zeta(alpha, xmin)` | `alpha` (1) |
| Exponential | `(1-q) * q^(x-xmin)`; a shifted geometric distribution | `q` (1) |
| Lognormal | `P(x-0.5 <= Y < x+0.5 | Y >= xmin-0.5)`, with lognormal `Y` | `mu`, `sigma` (2) |
| Poisson | `P(Y=x) / P(Y>=xmin)`, with Poisson `Y` | `lambda_poisson` (1) |

The Poisson rate is fitted by maximizing the **conditional** likelihood; it is not set equal to the observed tail mean. The lognormal is a binned continuous lognormal, not a separately defined integer-valued lognormal family.

`AIC = 2*n_parameters - 2*log_likelihood`

`delta_AIC = AIC - minimum AIC within the same network`

`xmin` is held fixed at the power-law-selected value for all four models. These are conditional comparisons; the parameter counts do not account for the preceding data-driven cutoff search. The usual independent-observation AIC interpretation is additionally limited by network dependence.

### BA: xmin = 3, n_tail = 4000

| Model | Log-likelihood | AIC | delta_AIC |
| --- | ---: | ---: | ---: |
| Power law | -8282.610 | 16567.219 | 7.769 |
| Exponential | -8992.181 | 17986.361 | 1426.911 |
| Lognormal | -8277.725 | 16559.450 | 0.000 |
| Poisson | -14731.433 | 29464.866 | 12905.416 |

The binned lognormal has the lowest exploratory AIC. This does not show that BA networks have a lognormal generating mechanism, nor does it establish an adequate absolute fit for the lognormal.

### ER: xmin = 10, n_tail = 346

| Model | Log-likelihood | AIC | delta_AIC |
| --- | ---: | ---: | ---: |
| Power law | -456.642 | 915.283 | 11.790 |
| Exponential | -452.214 | 906.427 | 2.934 |
| Lognormal | -450.694 | 905.387 | 1.894 |
| Poisson | -450.747 | 903.493 | 0.000 |

Poisson has the lowest exploratory AIC; lognormal is close in this comparison. Although lognormal achieves a slightly higher likelihood, its additional fitted parameter increases its AIC penalty. These results concern only the selected 346-node tail, not the full ER degree distribution.

Absolute BA and ER AIC values are not comparable because they refer to different datasets and sample sizes. AIC ranks the candidate models on a common sample; it is not a goodness-of-fit p-value or a probability that a model is true.

## Interpretation and limits

**Within-network dependence.** The bootstrap creates independent observations, not complete graph realizations. The product likelihood likewise ignores dependence among node degrees. Increasing `BOOTSTRAPS` reduces simulation noise but does not repair this modeling limitation. Graph-level inference would require a null model and simulation procedure that preserve the relevant network structure.

**One realization per generator.** Results describe the documented seeded networks. Robustness across independent graph seeds and sizes has not been established here.

**Tail selection and support.** The selected cutoff can discard most observations. Model comparisons use the same cutoff and observations within each network. Fitted distributions have unbounded upper support; finite simple graphs do not.

**Different questions.** Bootstrap goodness-of-fit is implemented only for the fitted power law. The lognormal, exponential, and Poisson have not undergone corresponding absolute-fit bootstrap tests in this project. Lowest AIC is not acceptance of a model, and failure to reject would not prove one.

**Numerical constraints.** Power-law optimization is restricted to `1.01 < alpha < 20`. Lognormal optimization uses bounded `mu` and `sigma`. The saved `lognormal_near_boundary` diagnostic checks selected boundaries rather than every possible numerical issue. Small differences can depend on optimizer or library versions. `requirements.txt` specifies minimum versions, not a fully pinned environment.

**Plot limits.** Model-comparison plots display conditional CCDF values down to `0.5/n_tail`; lower theoretical probabilities are outside the axes. This visualization choice does not modify likelihoods, AIC, or the fitted models.

## References

[1] Clauset, A., Shalizi, C. R., & Newman, M. E. J. (2009). *Power-law distributions in empirical data*. SIAM Review, 51(4), 661–703. https://doi.org/10.1137/070710111

Author-maintained methods and implementations: https://aaronclauset.github.io/powerlaws/

[2] Barabási, A.-L. *Network Science*, Chapter 5, especially Section 5.13 and Eq. (5.41), for the stationary BA degree distribution and its large-degree limit: https://networksciencebook.com/translations/en/partials/ch-5.html
