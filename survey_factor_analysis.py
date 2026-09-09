"""
survey_factor_analysis.py

Determines latent factor structure for the pulse-survey Likert items
(Q1-Q4 blocks + WB block) using:

  1. Parallel analysis (Horn, 1965) on a pairwise-complete correlation
     matrix -> scree/elbow plot to ESTIMATE the number of latent factors.
  2. Literal PCA (eigendecomposition of the correlation matrix) with
     loadings, for comparison against #3.
  3. Exploratory factor analysis with maximum likelihood extraction and
     an oblique rotation (promax by default) -> the actual factor
     loadings to use for scale construction.

NA handling: every step operates on a single pairwise-complete correlation
matrix (pandas .corr(), which uses all available pairs per item-pair by
default). No respondent row is ever dropped for having a missing item, and
no value is imputed -- this matters if/when you pool multiple wave files
where item-level missingness differs across waves.

NOTE ON TERMINOLOGY: "PCA with maximum likelihood estimation" is not a
single procedure -- PCA is eigendecomposition (no likelihood involved);
ML extraction belongs to common factor analysis. Section 3 below is the
ML + oblique factor solution; Section 2 is literal PCA. Use Section 3's
loadings to build your scales; Section 2 is there because it was asked
for explicitly and it's a useful sanity check against Section 3.
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
# One path per wave. Files are concatenated row-wise before analysis. Only
# one file exists in this repo today (no wave column, no separate wave
# files) -- add more paths here as additional waves become available.
DATA_PATHS = [
    "synthetic_survey_responses.csv",
]

N_PARALLEL_ITER = 1000     # random datasets simulated for parallel analysis
PARALLEL_PERCENTILE = 95   # percentile of random eigenvalues used as cutoff
RANDOM_SEED = 42
ROTATION = "promax"        # oblique rotation; alt: "oblimin"
FA_METHOD = "ml"           # maximum likelihood extraction
LOADING_THRESHOLD = 0.40   # |loading| below this = too weak to assign
CROSS_LOAD_GAP = 0.15      # flag if top-2 loadings are closer than this

# Set this after inspecting the parallel-analysis scree plot printed below.
# Left as None, it defaults to the parallel-analysis suggestion.
N_FACTORS_OVERRIDE = None

pd.set_option("display.width", 120)

# ---------------------------------------------------------------------------
# LOAD + STACK WAVES
# ---------------------------------------------------------------------------
frames = [pd.read_csv(p) for p in DATA_PATHS]
df = pd.concat(frames, ignore_index=True)
print(f"Pooled data: {df.shape[0]} respondent-rows x {df.shape[1]} columns "
      f"from {len(DATA_PATHS)} wave file(s).")

item_cols = [c for c in df.columns if c.startswith("Q") or c.startswith("WB_")]
items = df[item_cols]
print(f"{len(item_cols)} Likert items selected for factor analysis: "
      f"{item_cols[0]}...{item_cols[-1]}")

n_missing = int(items.isna().sum().sum())
print(f"Total missing cells: {n_missing} ({n_missing / items.size:.2%}) -- "
      f"handled via pairwise-complete correlations, no rows dropped.\n")

# ---------------------------------------------------------------------------
# PAIRWISE-COMPLETE CORRELATION MATRIX (everything below uses this matrix)
# ---------------------------------------------------------------------------
R = items.corr(method="pearson")  # pandas uses pairwise-complete obs by default
R_vals = R.values
p = R_vals.shape[0]

notna_int = items.notna().astype(int)
pairwise_n = notna_int.T.dot(notna_int).values  # n used per item pair
print(f"Pairwise sample sizes range from {pairwise_n[np.triu_indices(p, 1)].min()} "
      f"to {pairwise_n[np.triu_indices(p, 1)].max()} across item pairs.\n")

# ---------------------------------------------------------------------------
# KMO sampling adequacy (computed directly from R, no raw-data dependency)
# ---------------------------------------------------------------------------
R_inv = np.linalg.pinv(R_vals)
D = np.diag(1 / np.sqrt(np.diag(R_inv)))
partial_corr = -D @ R_inv @ D
np.fill_diagonal(partial_corr, 0)
r_sq_sum = np.sum(R_vals ** 2) - p
partial_sq_sum = np.sum(partial_corr ** 2)
kmo_overall = r_sq_sum / (r_sq_sum + partial_sq_sum)
kmo_label = ("meritorious+" if kmo_overall >= 0.8 else
             "middling/mediocre" if kmo_overall >= 0.6 else
             "unacceptable -- reconsider item set")
print(f"KMO overall sampling adequacy: {kmo_overall:.3f} ({kmo_label})\n")

# ---------------------------------------------------------------------------
# 1. PARALLEL ANALYSIS (Horn, 1965)
# ---------------------------------------------------------------------------
def eigs_desc(mat):
    return np.sort(np.linalg.eigvalsh(mat))[::-1]

observed_eigs = eigs_desc(R_vals)

# Representative n for the simulation: median pairwise n across item pairs.
n_obs = int(np.median(pairwise_n[np.triu_indices(p, k=1)]))
rng = np.random.default_rng(RANDOM_SEED)

sim_eigs = np.empty((N_PARALLEL_ITER, p))
for i in range(N_PARALLEL_ITER):
    sim_data = rng.standard_normal(size=(n_obs, p))
    sim_eigs[i] = eigs_desc(np.corrcoef(sim_data, rowvar=False))

sim_percentile = np.percentile(sim_eigs, PARALLEL_PERCENTILE, axis=0)
sim_mean = sim_eigs.mean(axis=0)

n_factors_pa = int(np.sum(observed_eigs > sim_percentile))
print(f"Parallel analysis (n_sim={n_obs}, {N_PARALLEL_ITER} iterations): "
      f"retain {n_factors_pa} factor(s) -- observed eigenvalue exceeds the "
      f"{PARALLEL_PERCENTILE}th-percentile random eigenvalue.\n")

comp_numbers = np.arange(1, p + 1)
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(comp_numbers, observed_eigs, "o-", color="steelblue", label="Observed data")
ax.plot(comp_numbers, sim_percentile, "s--", color="firebrick",
        label=f"Random data ({PARALLEL_PERCENTILE}th pct, {N_PARALLEL_ITER} sims)")
ax.plot(comp_numbers, sim_mean, ":", color="firebrick", alpha=0.5, label="Random data (mean)")
ax.axhline(1, color="gray", linestyle=":", label="Kaiser criterion (eigenvalue = 1)")
ax.axvline(n_factors_pa + 0.5, color="green", linestyle=":", alpha=0.6,
           label=f"Parallel-analysis cutoff ({n_factors_pa} factors)")
ax.set_xlabel("Component number")
ax.set_ylabel("Eigenvalue")
ax.set_title("Parallel Analysis Scree Plot -- All Survey Items")
ax.set_xticks(comp_numbers)
ax.tick_params(axis="x", labelrotation=90, labelsize=7)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("parallel_analysis_scree.png", dpi=150)
plt.close(fig)
print("Scree/elbow plot saved to parallel_analysis_scree.png\n")

N_FACTORS = N_FACTORS_OVERRIDE or n_factors_pa
print(f"Using N_FACTORS = {N_FACTORS} for the extractions below "
      f"(edit N_FACTORS_OVERRIDE at the top of the script to change this "
      f"after eyeballing the elbow).\n")

# ---------------------------------------------------------------------------
# 2. LITERAL PCA (eigendecomposition of R) -- unrotated + oblique-rotated
# ---------------------------------------------------------------------------
eigvals, eigvecs = np.linalg.eigh(R_vals)
order = np.argsort(eigvals)[::-1]
eigvals, eigvecs = eigvals[order], eigvecs[:, order]

pca_loadings = eigvecs[:, :N_FACTORS] * np.sqrt(eigvals[:N_FACTORS])
pca_loadings_df = pd.DataFrame(
    pca_loadings, index=item_cols, columns=[f"PC{i+1}" for i in range(N_FACTORS)]
)

print("=== 2a. PCA loadings (unrotated) ===")
print(pca_loadings_df.round(3))
var_explained = eigvals[:N_FACTORS] / p
print(f"\nProportion of variance explained per component: {np.round(var_explained, 3)}")
print(f"Cumulative variance explained: {var_explained.sum():.3f}\n")

if N_FACTORS > 1:
    pca_rotated, _ = Rotator(method=ROTATION).fit_transform(pca_loadings), None
    pca_rotated_df = pd.DataFrame(
        pca_rotated, index=item_cols, columns=[f"PC{i+1}" for i in range(N_FACTORS)]
    )
    print(f"=== 2b. PCA loadings ({ROTATION} oblique rotation, for comparison to Section 3) ===")
    print(pca_rotated_df.round(3))
    print()
else:
    pca_rotated_df = pca_loadings_df
    print("Only 1 component retained -- rotation is not meaningful with a single factor.\n")

# ---------------------------------------------------------------------------
# 3. EXPLORATORY FACTOR ANALYSIS -- ML extraction, oblique rotation
# ---------------------------------------------------------------------------
fa = FactorAnalyzer(n_factors=N_FACTORS, rotation=ROTATION if N_FACTORS > 1 else None,
                     method=FA_METHOD, is_corr_matrix=True)
fa.fit(R_vals)

fa_loadings_df = pd.DataFrame(
    fa.loadings_, index=item_cols, columns=[f"Factor{i+1}" for i in range(N_FACTORS)]
)

print(f"=== 3a. EFA loadings (ML extraction, {ROTATION if N_FACTORS > 1 else 'no'} rotation) ===")
print(fa_loadings_df.round(3))

variance_df = pd.DataFrame(
    fa.get_factor_variance(),
    index=["SS Loadings", "Proportion Var", "Cumulative Var"],
    columns=[f"Factor{i+1}" for i in range(N_FACTORS)],
)
print("\nVariance accounted for:")
print(variance_df.round(3))

if N_FACTORS > 1 and ROTATION in ("promax", "oblimin"):
    phi = fa.phi_
    print("\nFactor correlation matrix (oblique rotation permits correlated factors --"
          " check this before treating factors as independent scales):")
    print(pd.DataFrame(phi, index=fa_loadings_df.columns, columns=fa_loadings_df.columns).round(3))
print()

# ---------------------------------------------------------------------------
# 3b. ITEM -> FACTOR ASSIGNMENT HELPER
# ---------------------------------------------------------------------------
def assign_items(loadings_df, threshold=LOADING_THRESHOLD, gap=CROSS_LOAD_GAP):
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

assignment_df = assign_items(fa_loadings_df)
print("=== 3c. Suggested item -> factor assignment (ML EFA, oblique rotation) ===")
print(assignment_df)

print("\nItem counts per assigned factor:")
print(assignment_df["assigned_factor"].value_counts(dropna=False))

print("\n--- Interpretation guide ---")
print("|loading| >= 0.70 : strong")
print("|loading| >= 0.40 : moderate, usually keep on the scale")
print("|loading| <  0.40 : weak, candidate for dropping")
print("Cross-loading items are ambiguous -- decide by theory, not just magnitude.")
