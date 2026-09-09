"""
parallel_analysis_efa.py

Generic pipeline: parallel analysis -> factor count estimate -> EFA
(maximum likelihood extraction, oblique rotation) with loadings.

Assumes two objects already exist in the namespace:
  unified_df     -- DataFrame containing all pooled waves
  numerical_cols -- list of numeric column names to factor-analyze

Everything runs off a single pairwise-complete correlation matrix
(pandas .corr(), pairwise-complete by default), so no respondent row is
ever dropped for having a missing item and nothing is imputed. This
matters when waves differ in which items were missing.

Run this after `unified_df` and `numerical_cols` are defined, e.g.:
    exec(open("parallel_analysis_efa.py").read())
or import the functions and call run_pipeline(unified_df, numerical_cols).
"""

import inspect

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from factor_analyzer import FactorAnalyzer
from factor_analyzer.rotator import Rotator

# --- compatibility shim -----------------------------------------------------
# Older factor_analyzer releases call sklearn's check_array(force_all_finite=...),
# a kwarg newer scikit-learn versions renamed to ensure_all_finite. factor_analyzer
# imports check_array into its own module namespace, so patch it there directly.
import factor_analyzer.factor_analyzer as _fa_mod
import sklearn.utils.validation as _skv

if "force_all_finite" not in inspect.signature(_skv.check_array).parameters:
    _orig_check_array = _skv.check_array

    def _check_array_compat(*args, force_all_finite=None, **kwargs):
        if force_all_finite is not None:
            kwargs.setdefault("ensure_all_finite", force_all_finite)
        return _orig_check_array(*args, **kwargs)

    _skv.check_array = _check_array_compat
    _fa_mod.check_array = _check_array_compat
# -----------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
N_PARALLEL_ITER = 1000     # random datasets simulated for parallel analysis
PARALLEL_PERCENTILE = 95   # percentile of random eigenvalues used as cutoff
RANDOM_SEED = 42
ROTATION = "promax"        # oblique rotation; alt: "oblimin"
FA_METHOD = "ml"           # maximum likelihood extraction
LOADING_THRESHOLD = 0.40   # |loading| below this = too weak to assign
CROSS_LOAD_GAP = 0.15      # flag if top-2 loadings are closer than this
SCREE_PLOT_PATH = "parallel_analysis_scree.png"


def pairwise_complete_corr(df: pd.DataFrame, cols: list) -> tuple[pd.DataFrame, np.ndarray]:
    """Pairwise-complete correlation matrix + per-pair N. No rows dropped."""
    items = df[cols]
    R = items.corr(method="pearson")
    notna_int = items.notna().astype(int)
    pairwise_n = notna_int.T.dot(notna_int).values
    return R, pairwise_n


def kmo_overall(R_vals: np.ndarray) -> float:
    """Overall Kaiser-Meyer-Olkin sampling adequacy, computed from R directly."""
    p = R_vals.shape[0]
    R_inv = np.linalg.pinv(R_vals)
    D = np.diag(1 / np.sqrt(np.diag(R_inv)))
    partial_corr = -D @ R_inv @ D
    np.fill_diagonal(partial_corr, 0)
    r_sq_sum = np.sum(R_vals ** 2) - p
    partial_sq_sum = np.sum(partial_corr ** 2)
    return r_sq_sum / (r_sq_sum + partial_sq_sum)


def _eigs_desc(mat: np.ndarray) -> np.ndarray:
    return np.sort(np.linalg.eigvalsh(mat))[::-1]


def parallel_analysis(
    R_vals: np.ndarray,
    n_obs: int,
    n_iter: int = N_PARALLEL_ITER,
    percentile: float = PARALLEL_PERCENTILE,
    seed: int = RANDOM_SEED,
    plot_path: str | None = SCREE_PLOT_PATH,
) -> int:
    """Horn's (1965) parallel analysis. Returns the suggested factor count."""
    p = R_vals.shape[0]
    observed_eigs = _eigs_desc(R_vals)

    rng = np.random.default_rng(seed)
    sim_eigs = np.empty((n_iter, p))
    for i in range(n_iter):
        sim_data = rng.standard_normal(size=(n_obs, p))
        sim_eigs[i] = _eigs_desc(np.corrcoef(sim_data, rowvar=False))

    sim_percentile = np.percentile(sim_eigs, percentile, axis=0)
    sim_mean = sim_eigs.mean(axis=0)
    n_factors = int(np.sum(observed_eigs > sim_percentile))

    print(f"Parallel analysis (n_sim={n_obs}, {n_iter} iterations): "
          f"retain {n_factors} factor(s) -- observed eigenvalue exceeds the "
          f"{percentile}th-percentile random eigenvalue.\n")

    if plot_path:
        comp_numbers = np.arange(1, p + 1)
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.plot(comp_numbers, observed_eigs, "o-", color="steelblue", label="Observed data")
        ax.plot(comp_numbers, sim_percentile, "s--", color="firebrick",
                label=f"Random data ({percentile}th pct, {n_iter} sims)")
        ax.plot(comp_numbers, sim_mean, ":", color="firebrick", alpha=0.5, label="Random data (mean)")
        ax.axhline(1, color="gray", linestyle=":", label="Kaiser criterion (eigenvalue = 1)")
        ax.axvline(n_factors + 0.5, color="green", linestyle=":", alpha=0.6,
                   label=f"Parallel-analysis cutoff ({n_factors} factors)")
        ax.set_xlabel("Component number")
        ax.set_ylabel("Eigenvalue")
        ax.set_title("Parallel Analysis Scree Plot")
        ax.set_xticks(comp_numbers)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"Scree/elbow plot saved to {plot_path}\n")

    return n_factors


def run_efa(
    R_vals: np.ndarray,
    cols: list,
    n_factors: int,
    rotation: str = ROTATION,
    method: str = FA_METHOD,
):
    """ML-extraction EFA with oblique rotation. Returns (loadings_df, variance_df, phi_df|None)."""
    fa = FactorAnalyzer(
        n_factors=n_factors,
        rotation=rotation if n_factors > 1 else None,
        method=method,
        is_corr_matrix=True,
    )
    fa.fit(R_vals)

    factor_names = [f"Factor{i+1}" for i in range(n_factors)]
    loadings_df = pd.DataFrame(fa.loadings_, index=cols, columns=factor_names)

    variance_df = pd.DataFrame(
        fa.get_factor_variance(),
        index=["SS Loadings", "Proportion Var", "Cumulative Var"],
        columns=factor_names,
    )

    phi_df = None
    if n_factors > 1 and rotation in ("promax", "oblimin"):
        phi_df = pd.DataFrame(fa.phi_, index=factor_names, columns=factor_names)

    return loadings_df, variance_df, phi_df


def assign_items(loadings_df: pd.DataFrame, threshold: float = LOADING_THRESHOLD,
                  gap: float = CROSS_LOAD_GAP) -> pd.DataFrame:
    """Per-item top-factor assignment with weak-loading / cross-loading flags."""
    abs_loadings = loadings_df.abs()
    rows = []
    for item, row in abs_loadings.iterrows():
        sorted_row = row.sort_values(ascending=False)
        top_factor, top_val = sorted_row.index[0], sorted_row.iloc[0]
        second_val = sorted_row.iloc[1] if len(sorted_row) > 1 else 0.0
        second_factor = sorted_row.index[1] if len(sorted_row) > 1 else None
        if top_val < threshold:
            flag = "WEAK (<|0.40| on every factor -- consider dropping)"
            assigned = None
        elif (top_val - second_val) < gap:
            flag = f"CROSS-LOADS with {second_factor} (gap={top_val - second_val:.2f})"
            assigned = top_factor
        else:
            flag = ""
            assigned = top_factor
        rows.append({
            "item": item,
            "assigned_factor": assigned,
            "loading": loadings_df.loc[item, top_factor],
            "flag": flag,
        })
    return pd.DataFrame(rows).set_index("item")


def run_pipeline(df: pd.DataFrame, cols: list, n_factors_override: int | None = None):
    """Full pipeline: pairwise corr -> KMO -> parallel analysis -> ML/oblique EFA -> assignment."""
    R, pairwise_n = pairwise_complete_corr(df, cols)
    R_vals = R.values
    p = R_vals.shape[0]

    n_missing = int(df[cols].isna().sum().sum())
    print(f"{len(cols)} numeric columns, {df.shape[0]} rows. "
          f"Missing cells: {n_missing} ({n_missing / df[cols].size:.2%}) -- "
          f"handled via pairwise-complete correlations, no rows dropped.\n")

    triu = np.triu_indices(p, k=1)
    print(f"Pairwise sample sizes range from {pairwise_n[triu].min()} to "
          f"{pairwise_n[triu].max()} across column pairs.\n")

    kmo = kmo_overall(R_vals)
    kmo_label = ("meritorious+" if kmo >= 0.8 else
                 "middling/mediocre" if kmo >= 0.6 else
                 "unacceptable -- reconsider column set")
    print(f"KMO overall sampling adequacy: {kmo:.3f} ({kmo_label})\n")

    n_obs = int(np.median(pairwise_n[triu]))
    n_factors_pa = parallel_analysis(R_vals, n_obs=n_obs)

    n_factors = n_factors_override or n_factors_pa
    print(f"Using N_FACTORS = {n_factors} "
          f"(pass n_factors_override to run_pipeline() to change this after "
          f"eyeballing the scree plot).\n")

    loadings_df, variance_df, phi_df = run_efa(R_vals, cols, n_factors)

    print(f"=== EFA loadings (ML extraction, "
          f"{ROTATION if n_factors > 1 else 'no'} rotation) ===")
    print(loadings_df.round(3))

    print("\nVariance accounted for:")
    print(variance_df.round(3))

    if phi_df is not None:
        print("\nFactor correlation matrix (oblique rotation permits correlated "
              "factors -- check this before treating factors as independent scales):")
        print(phi_df.round(3))
    print()

    assignment_df = assign_items(loadings_df)
    print("=== Suggested column -> factor assignment ===")
    print(assignment_df)
    print("\nColumn counts per assigned factor:")
    print(assignment_df["assigned_factor"].value_counts(dropna=False))

    return {
        "R": R,
        "n_factors_parallel_analysis": n_factors_pa,
        "n_factors_used": n_factors,
        "loadings": loadings_df,
        "variance": variance_df,
        "phi": phi_df,
        "assignment": assignment_df,
    }


if __name__ == "__main__":
    results = run_pipeline(unified_df, numerical_cols)
