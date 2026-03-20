###############################################################################
#  survey_psychometrics.R
#
#  PURPOSE
#  -------
#  Demonstrates common psychometric techniques applied to Likert-scale pulse
#  survey data.  Topics covered:
#
#    A. Composite (scale) score creation
#    B. Internal-consistency reliability (Cronbach's alpha)
#    C. Exploratory factor analysis / PCA with parallel analysis
#    D. Pairwise correlation analysis & construct-validity assessment
#    E. OLS regression with model comparison
#    F. Bayesian regression (optional; requires the brms package)
#
#  The script is intentionally verbose with annotations so it can serve as a
#  learning resource for analysts new to psychometrics in R.
###############################################################################


# ── CONFIG ──────────────────────────────────────────────────────────────────
DATA_PATH <- "synthetic_survey_responses.csv"
# ────────────────────────────────────────────────────────────────────────────


# ── LIBRARIES ───────────────────────────────────────────────────────────────
# Core data manipulation
library(dplyr)
library(tidyr)

# Psychometrics & statistics
library(psych)       # alpha(), principal(), fa.parallel()
library(stats)       # prcomp(), lm(), shapiro.test()

# Visualisation
library(ggplot2)
library(ggpubr)      # ggarrange() for multi-panel layouts
library(corrplot)    # corrplot() for visual correlation matrices

# Correlation helpers
library(correlation) # correlation() returns tidy pairwise tables

# Note: brms (Bayesian regression) is optional.
# Section F wraps all brms code in tryCatch() so the rest of the script
# runs cleanly even when brms is not installed.
# ────────────────────────────────────────────────────────────────────────────


# ── LOAD DATA ───────────────────────────────────────────────────────────────
# Expect a CSV with columns:
#   respondent_id, Q1_1..Q1_4  (Engagement items)
#   Q2_1..Q2_4                 (Manager items)
#   Q3_1..Q3_6                 (Inclusion & Diversity items)
#   Q4_1..Q4_4                 (Additional workplace items)
#   WB_1..WB_20                (Well-being items)
#   plus demographic columns (department, gender, etc.)
# ────────────────────────────────────────────────────────────────────────────
df <- read.csv(DATA_PATH, stringsAsFactors = FALSE)

cat("Dataset dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
cat("First few column names:", paste(head(names(df), 15), collapse = ", "), "\n\n")


###############################################################################
#
#  SECTION A: COMPOSITE SCORE CREATION
#
#  Composite scores (a.k.a. scale scores) average the Likert ratings across
#  items that are designed to measure the same underlying construct.  Taking
#  the mean rather than the sum keeps the metric on the original response
#  scale, which makes interpretation easier.
#
#  We create three composites:
#    1. Engagement          (Q1_1 through Q1_4)
#    2. Manager Effectiveness (Q2_1, Q2_4, Q4_3)
#    3. Inclusion & Diversity (Q3_1 through Q3_6)
#
#  na.rm = TRUE allows a score to be computed even when one item is missing,
#  which is common in real survey data.  In practice you might also want to
#  set a minimum item-response threshold (e.g., require >= 50 % non-missing).
#
###############################################################################

# -- Engagement composite --
# Q1_1 through Q1_4 are four items tapping overall engagement.
engagement_items <- c("Q1_1", "Q1_2", "Q1_3", "Q1_4")

df$engagement_score <- rowMeans(df[, engagement_items], na.rm = TRUE)

cat("Engagement score summary:\n")
print(summary(df$engagement_score))
cat("\n")


# -- Manager Effectiveness composite --
# Q2_1 and Q2_4 capture direct-manager behaviours; Q4_3 captures a related
# supportiveness item.  Combining items from different blocks is fine as long
# as theory and reliability evidence support it.
manager_items <- c("Q2_1", "Q2_4", "Q4_3")

df$manager_effectiveness_score <- rowMeans(df[, manager_items], na.rm = TRUE)

cat("Manager Effectiveness score summary:\n")
print(summary(df$manager_effectiveness_score))
cat("\n")


# -- Inclusion & Diversity composite --
# Q3_1 through Q3_6 capture perceived inclusion and diversity climate.
id_items <- c("Q3_1", "Q3_2", "Q3_3", "Q3_4", "Q3_5", "Q3_6")

df$id_score <- rowMeans(df[, id_items], na.rm = TRUE)

cat("Inclusion & Diversity score summary:\n")
print(summary(df$id_score))
cat("\n")


# -- Histograms of each composite --
# Visualising the distribution helps spot ceiling / floor effects and skew.
p_eng <- ggplot(df, aes(x = engagement_score)) +
  geom_histogram(binwidth = 0.25, fill = "steelblue", colour = "white") +
  labs(title = "Engagement Score Distribution",
       x = "Engagement (mean of 4 items)", y = "Count") +
  theme_minimal()

p_mgr <- ggplot(df, aes(x = manager_effectiveness_score)) +
  geom_histogram(binwidth = 0.25, fill = "darkorange", colour = "white") +
  labs(title = "Manager Effectiveness Distribution",
       x = "Manager Effectiveness (mean of 3 items)", y = "Count") +
  theme_minimal()

p_id <- ggplot(df, aes(x = id_score)) +
  geom_histogram(binwidth = 0.25, fill = "forestgreen", colour = "white") +
  labs(title = "Inclusion & Diversity Distribution",
       x = "I&D (mean of 6 items)", y = "Count") +
  theme_minimal()

# Arrange the three histograms in one panel
composite_panel <- ggarrange(p_eng, p_mgr, p_id, ncol = 3, nrow = 1)
print(composite_panel)


###############################################################################
#
#  SECTION B: RELIABILITY ANALYSIS (Cronbach's Alpha)
#
#  Cronbach's alpha estimates internal-consistency reliability -- the degree
#  to which items in a scale "hang together."
#
#  Interpretation rules of thumb (Nunnally & Bernstein, 1994):
#    alpha > 0.9  -->  Excellent reliability
#    alpha > 0.8  -->  Good reliability
#    alpha > 0.7  -->  Acceptable reliability
#    alpha < 0.7  -->  Questionable -- consider revising the scale
#
#  psych::alpha() also reports:
#    - Item-total correlations (how each item relates to the total)
#    - "Alpha if item dropped" (whether removing an item improves alpha)
#
###############################################################################

interpret_alpha <- function(a) {
  # Helper that maps a numeric alpha value to a label.
  if (is.na(a)) return("Cannot compute (too few items or all NA)")
  if (a >= 0.9) return("Excellent")
  if (a >= 0.8) return("Good")
  if (a >= 0.7) return("Acceptable")
  return("Questionable (< 0.70)")
}


# -- Engagement reliability --
cat("=== Engagement Scale (Q1_1 to Q1_4) ===\n")
alpha_eng <- psych::alpha(df[, engagement_items], check.keys = TRUE)
print(alpha_eng)
cat("Interpretation:", interpret_alpha(alpha_eng$total$raw_alpha), "\n\n")


# -- Manager Effectiveness reliability --
cat("=== Manager Effectiveness Scale (Q2_1, Q2_4, Q4_3) ===\n")
alpha_mgr <- psych::alpha(df[, manager_items], check.keys = TRUE)
print(alpha_mgr)
cat("Interpretation:", interpret_alpha(alpha_mgr$total$raw_alpha), "\n\n")


# -- Inclusion & Diversity reliability (full scale) --
cat("=== Inclusion & Diversity Scale (Q3_1 to Q3_6) ===\n")
alpha_id <- psych::alpha(df[, id_items], check.keys = TRUE)
print(alpha_id)
cat("Interpretation:", interpret_alpha(alpha_id$total$raw_alpha), "\n\n")


# -- Subscale reliabilities for I&D --
# Sometimes a broad scale has conceptual subscales.  Here we illustrate
# splitting the 6 I&D items into two hypothetical subscales:
#   Organisational I&D climate : Q3_1, Q3_2, Q3_3
#   Individual I&D experience  : Q3_4, Q3_5, Q3_6

org_id_items  <- c("Q3_1", "Q3_2", "Q3_3")
ind_id_items  <- c("Q3_4", "Q3_5", "Q3_6")

cat("=== Organisational I&D Subscale ===\n")
alpha_org_id <- psych::alpha(df[, org_id_items], check.keys = TRUE)
print(alpha_org_id)
cat("Interpretation:", interpret_alpha(alpha_org_id$total$raw_alpha), "\n\n")

cat("=== Individual I&D Subscale ===\n")
alpha_ind_id <- psych::alpha(df[, ind_id_items], check.keys = TRUE)
print(alpha_ind_id)
cat("Interpretation:", interpret_alpha(alpha_ind_id$total$raw_alpha), "\n\n")


###############################################################################
#
#  SECTION C: FACTOR ANALYSIS (PCA + Parallel Analysis)
#
#  Factor analysis (and its close cousin PCA) asks: can the variation in many
#  items be explained by a smaller number of latent dimensions?
#
#  Steps:
#    1. Run PCA on the I&D items to extract components.
#    2. Inspect the scree plot -- look for the "elbow" where eigenvalues
#       level off.
#    3. Run parallel analysis (Horn, 1965) to statistically determine
#       how many factors to retain.
#    4. Inspect factor loadings -- items that load > |0.40| on a factor
#       are usually considered meaningfully associated with that factor.
#
#  Loadings interpretation guide:
#    |loading| >= 0.70  Strong association with the factor
#    |loading| >= 0.40  Moderate -- item likely belongs on the factor
#    |loading| <  0.40  Weak -- item may be a candidate for removal
#    Cross-loadings (high on 2+ factors) suggest the item is ambiguous
#
###############################################################################

# Subset to inclusion items for the factor analysis
id_matrix <- df[, id_items]

# Remove any rows that are entirely NA (can't contribute to PCA)
id_matrix <- id_matrix[complete.cases(id_matrix), ]

cat("Running PCA on", nrow(id_matrix), "complete cases across",
    ncol(id_matrix), "I&D items.\n\n")

# -- PCA via stats::prcomp() --
pca_result <- prcomp(id_matrix, center = TRUE, scale. = TRUE)

# Variance explained by each component
var_explained <- summary(pca_result)$importance
cat("Variance explained by each principal component:\n")
print(round(var_explained, 3))
cat("\n")

# -- Scree plot --
# The scree plot shows eigenvalues (variance explained) by component number.
# Retain components above the "elbow."
eigenvalues <- pca_result$sdev^2

scree_df <- data.frame(
  Component = seq_along(eigenvalues),
  Eigenvalue = eigenvalues
)

p_scree <- ggplot(scree_df, aes(x = Component, y = Eigenvalue)) +
  geom_line(colour = "steelblue", linewidth = 1) +
  geom_point(colour = "steelblue", size = 3) +
  geom_hline(yintercept = 1, linetype = "dashed", colour = "red") +
  scale_x_continuous(breaks = scree_df$Component) +
  labs(title = "Scree Plot -- I&D Items",
       subtitle = "Red dashed line = Kaiser criterion (eigenvalue = 1)",
       x = "Component Number", y = "Eigenvalue") +
  theme_minimal()

print(p_scree)

# -- Parallel analysis --
# Generates random data of the same dimensions many times and compares the
# observed eigenvalues to the random ones.  Factors whose observed eigenvalue
# exceeds the 95th-percentile random eigenvalue are retained.
cat("\n--- Parallel Analysis (I&D items) ---\n")
pa_result <- fa.parallel(id_matrix, fa = "both", n.iter = 100,
                         main = "Parallel Analysis -- I&D Items")
cat("Suggested number of factors (FA):", pa_result$nfact, "\n")
cat("Suggested number of components (PCA):", pa_result$ncomp, "\n\n")

# -- Factor loadings (using psych::principal for PCA approach) --
# nfactors is set to the number suggested by parallel analysis for PCA.
n_factors <- max(1, pa_result$ncomp)  # at least 1

cat("Extracting", n_factors, "component(s) via PCA with varimax rotation.\n\n")
pca_loadings <- psych::principal(id_matrix, nfactors = n_factors,
                                  rotate = "varimax")

cat("Component loadings:\n")
print(pca_loadings$loadings, cutoff = 0.30)
cat("\nVariance accounted for by each component:\n")
print(round(pca_loadings$Vaccounted, 3))
cat("\n")

cat("--- Interpretation Guide ---\n")
cat("|loading| >= 0.70 : Strong association with the factor\n")
cat("|loading| >= 0.40 : Moderate -- item likely belongs on the factor\n")
cat("|loading| <  0.40 : Weak -- consider dropping or reassigning\n")
cat("Cross-loadings (high on 2+ factors) suggest the item is ambiguous.\n\n")


###############################################################################
#
#  SECTION D: CORRELATION ANALYSIS
#
#  Pairwise correlations reveal how survey items relate to each other.
#  Key uses in psychometrics:
#
#    - Construct validity: items within the same scale should correlate more
#      strongly with each other than with items from other scales.
#    - Discriminant validity: scales measuring different constructs should
#      show moderate-to-low inter-scale correlations.
#    - Multicollinearity flags: correlations > 0.80 between predictors
#      can destabilise regression estimates.
#
#  We use the {correlation} package for a tidy, comprehensive output.
#
###############################################################################

# Gather all Likert-scale item columns (Q and WB items)
likert_cols <- grep("^(Q[0-9]|WB_)", names(df), value = TRUE)

# Build the item-level data frame
item_df <- df[, likert_cols]

# Remove any columns that are entirely NA (can happen with optional items)
all_na_cols <- sapply(item_df, function(x) all(is.na(x)))
if (any(all_na_cols)) {
  cat("Removing columns with all NAs:",
      paste(names(which(all_na_cols)), collapse = ", "), "\n")
  item_df <- item_df[, !all_na_cols]
}

cat("Computing pairwise correlations across", ncol(item_df), "items...\n\n")

# -- Correlation matrix using the {correlation} package --
# method = "pearson" is standard for Likert data treated as continuous.
# p_adjust = "holm" controls for multiple comparisons.
cor_results <- correlation::correlation(item_df, method = "pearson",
                                         p_adjust = "holm")
print(summary(cor_results))

# -- Export correlation results to CSV --
cor_output_path <- "correlation_results.csv"
write.csv(as.data.frame(cor_results), cor_output_path, row.names = FALSE)
cat("\nFull correlation table exported to:", cor_output_path, "\n")

# -- Visual correlation matrix --
# corrplot works with a base-R cor() matrix.
cor_matrix <- cor(item_df, use = "pairwise.complete.obs")

corrplot(cor_matrix,
         method   = "color",
         type     = "lower",
         tl.cex   = 0.6,
         tl.col   = "black",
         addCoef.col = "black",
         number.cex  = 0.4,
         title    = "Pairwise Correlations -- All Survey Items",
         mar      = c(0, 0, 2, 0))

cat("\n--- Interpretation ---\n")
cat("Strong within-scale correlations (r > 0.50) support convergent validity.\n")
cat("Low cross-scale correlations (r < 0.30) support discriminant validity.\n")
cat("Very high correlations (r > 0.80) may indicate item redundancy.\n\n")


###############################################################################
#
#  SECTION E: OLS REGRESSION
#
#  We model Engagement as a function of Inclusion & Diversity score,
#  Manager Effectiveness, and a selection of additional items.
#
#  Strategy:
#    Model 1 (bivariate):   engagement ~ id_score
#    Model 2 (multivariate): engagement ~ id_score + manager_effectiveness
#                                        + Q4_1 + Q4_2
#
#  AIC (Akaike Information Criterion) is used for model comparison:
#    - Lower AIC = better trade-off between fit and complexity.
#
#  We also check residual normality with:
#    - Shapiro-Wilk test  (null: residuals are normal)
#    - Anderson-Darling test via nortest (if available)
#
###############################################################################

# -- Model 1: Bivariate --
cat("=== Model 1: Engagement ~ I&D Score ===\n")
model1 <- lm(engagement_score ~ id_score, data = df)
print(summary(model1))
cat("AIC:", AIC(model1), "\n\n")


# -- Model 2: Multivariate --
cat("=== Model 2: Engagement ~ I&D + Manager Effectiveness + Q4_1 + Q4_2 ===\n")
model2 <- lm(engagement_score ~ id_score + manager_effectiveness_score +
                Q4_1 + Q4_2, data = df)
print(summary(model2))
cat("AIC:", AIC(model2), "\n\n")


# -- AIC comparison --
cat("--- AIC Comparison ---\n")
aic_table <- AIC(model1, model2)
print(aic_table)
cat("Lower AIC indicates a better balance of fit and parsimony.\n\n")


# -- Scatter plot with regression line (bivariate model) --
p_scatter <- ggplot(df, aes(x = id_score, y = engagement_score)) +
  geom_jitter(alpha = 0.4, width = 0.05, height = 0.05,
              colour = "steelblue") +
  geom_smooth(method = "lm", se = TRUE, colour = "red") +
  labs(title = "Engagement vs. Inclusion & Diversity",
       subtitle = paste0("R-squared = ",
                         round(summary(model1)$r.squared, 3)),
       x = "I&D Score", y = "Engagement Score") +
  theme_minimal()

print(p_scatter)


# -- Normality of residuals --

# Shapiro-Wilk (built into base R; limited to n <= 5000)
resids <- residuals(model2)

if (length(resids) <= 5000) {
  cat("--- Shapiro-Wilk Test on Model 2 Residuals ---\n")
  sw_test <- shapiro.test(resids)
  print(sw_test)
  cat("If p < 0.05, residuals deviate significantly from normality.\n\n")
} else {
  # For large samples, take a random subset
  cat("--- Shapiro-Wilk Test (random subset of 5000 residuals) ---\n")
  sw_test <- shapiro.test(sample(resids, 5000))
  print(sw_test)
  cat("Note: Subset used because Shapiro-Wilk requires n <= 5000.\n\n")
}

# Anderson-Darling (requires the nortest package)
if (requireNamespace("nortest", quietly = TRUE)) {
  cat("--- Anderson-Darling Normality Test ---\n")
  ad_test <- nortest::ad.test(resids)
  print(ad_test)
  cat("If p < 0.05, residuals deviate significantly from normality.\n\n")
} else {
  cat("Note: Install the 'nortest' package for the Anderson-Darling test.\n\n")
}


###############################################################################
#
#  SECTION F: BAYESIAN REGRESSION (Optional -- requires brms)
#
#  This section re-estimates the multivariate regression from Section E
#  using a Bayesian framework via the brms package.
#
#  Why Bayesian?
#    - Provides full posterior distributions for every parameter, giving
#      richer uncertainty information than a single p-value.
#    - Allows incorporation of prior knowledge (here we use weakly
#      informative priors so the data dominate).
#    - LOO (Leave-One-Out) cross-validation offers a principled way to
#      compare models.
#
#  Priors used:
#    - Intercept : Normal(0, 10)   -- weakly informative
#    - Slopes    : Normal(0, 5)    -- weakly informative
#    - Sigma     : Half-Cauchy(0, 2) -- weakly informative for residual SD
#
#  The entire section is wrapped in tryCatch() so the script does not fail
#  if brms (or its dependency rstan / cmdstanr) is not installed.
#
###############################################################################

tryCatch({

  if (!requireNamespace("brms", quietly = TRUE)) {
    stop("brms is not installed. Skipping Bayesian regression section.")
  }

  library(brms)
  cat("\n========================================\n")
  cat("  SECTION F: Bayesian Regression (brms)\n")
  cat("========================================\n\n")

  # -- Define weakly informative priors --
  # These are generic and not tuned to a specific dataset.
  # Normal(0, 10) for the intercept keeps it broad.
  # Normal(0, 5) for slopes is wide enough to let data speak.
  # Half-Cauchy(0, 2) for sigma is a common default for residual SD.
  bayes_priors <- c(
    prior(normal(0, 10), class = "Intercept"),
    prior(normal(0, 5),  class = "b"),
    prior(cauchy(0, 2),  class = "sigma")
  )

  # -- Fit the Bayesian model --
  # Using the same formula as Model 2 in Section E.
  cat("Fitting Bayesian model (this may take a few minutes)...\n")
  bayes_model <- brm(
    formula = engagement_score ~ id_score + manager_effectiveness_score +
                Q4_1 + Q4_2,
    data    = df,
    prior   = bayes_priors,
    family  = gaussian(),
    chains  = 4,
    iter    = 2000,
    warmup  = 1000,
    seed    = 42,
    silent  = 2       # suppress Stan messages
  )

  cat("\n--- Bayesian Model Summary ---\n")
  print(summary(bayes_model))

  # -- Posterior distributions --
  # plot() from brms produces trace plots and density plots for each parameter.
  cat("\nPlotting posterior distributions...\n")
  print(plot(bayes_model))

  # -- LOO cross-validation --
  # LOO-IC (Leave-One-Out Information Criterion) is analogous to AIC but
  # uses the full posterior.  Lower LOOIC = better predictive accuracy.
  cat("\n--- LOO Cross-Validation ---\n")
  loo_result <- loo(bayes_model)
  print(loo_result)
  cat("LOOIC:", loo_result$estimates["looic", "Estimate"], "\n")

  # -- (Optional) Fit a simpler Bayesian model for comparison --
  cat("\nFitting simpler Bayesian model for LOO comparison...\n")
  bayes_model_simple <- brm(
    formula = engagement_score ~ id_score,
    data    = df,
    prior   = c(
      prior(normal(0, 10), class = "Intercept"),
      prior(normal(0, 5),  class = "b"),
      prior(cauchy(0, 2),  class = "sigma")
    ),
    family  = gaussian(),
    chains  = 4,
    iter    = 2000,
    warmup  = 1000,
    seed    = 42,
    silent  = 2
  )

  loo_simple <- loo(bayes_model_simple)

  cat("\n--- LOO Model Comparison ---\n")
  loo_comparison <- loo_compare(loo_simple, loo_result)
  print(loo_comparison)
  cat("The model listed first has the better (higher) expected log\n")
  cat("predictive density (elpd).  A difference > 4 is often considered\n")
  cat("practically meaningful.\n\n")

}, error = function(e) {
  cat("\n[Section F skipped]", conditionMessage(e), "\n")
  cat("To enable Bayesian regression, install brms:\n")
  cat('  install.packages("brms")\n\n')
})


###############################################################################
cat("Done. All sections complete.\n")
###############################################################################
