# =============================================================================
# Latent Profile Analysis (LPA) with Gaussian Mixture Models
# =============================================================================
#
# PURPOSE:
#   Identify latent subgroups (profiles) within a population based on
#   continuous response patterns. LPA is a person-centered approach that
#   classifies individuals into unobserved subgroups based on their
#   pattern of scores across multiple indicators. This is useful for
#   workforce segmentation, identifying distinct employee "types," or
#   discovering hidden clusters in survey data.
#
# ANALYTICAL TECHNIQUES:
#   - Careless responding detection (longstring, Mahalanobis distance)
#   - Gaussian Mixture Model clustering (mclust)
#   - BIC/ICL model selection criteria
#   - Bootstrap Likelihood Ratio Test (BLRT)
#   - Profile visualization (line plots of standardized means)
#
# STEPS:
#   1. Load data and select indicators for profiling
#   2. Screen for careless responding (longstring analysis, outlier detection)
#   3. Standardize indicators (z-score scaling)
#   4. Fit mixture models across different numbers of profiles and covariance structures
#   5. Select optimal model using BIC, ICL, and BLRT
#   6. Extract and visualize profile means
#   7. Label and interpret profiles
#
# INTERPRETATION GUIDE:
#   - BIC: lower = better fit; plot the "elbow" to find optimal number of profiles
#   - ICL: like BIC but penalizes classification uncertainty more heavily
#   - BLRT: tests whether K profiles fit significantly better than K-1
#   - Covariance structure (e.g., VEE, EEE): controls how flexible the clusters are
#     VEE = variable volume, equal shape, equal orientation (recommended starting point)
#   - Profile means near 0 = average; > 0 = above average; < 0 = below average
#   - Profiles should be theoretically interpretable and practically useful
#   - Very small profiles (< 5% of sample) may be unreliable
#
# DATA SOURCE:
#   Uses the public Young People Survey dataset from GitHub.
#   Original: https://raw.githubusercontent.com/whipson/tidytuesday/master/young_people.csv
#   This dataset contains interest ratings across multiple domains.
#
# =============================================================================

# Clear workspace
rm(list = ls(all = TRUE))
graphics.off()
options(scipen = 999)

library(dplyr)
library(careless)      # Careless responding detection
library(psych)         # Psychometric analysis
library(mclust)        # Gaussian mixture models
library(reshape2)      # Data reshaping
library(tidyverse)     # Data manipulation and visualization

# ---- CONFIGURATION ----
# Number of standard deviations for longstring cutoff
LONGSTRING_CUTOFF <- 10

# Chi-square significance threshold for Mahalanobis distance outlier detection
OUTLIER_ALPHA <- 0.001

# =============================================================================
# STEP 1: Load Data
# =============================================================================
cat("Step 1: Loading Young People Survey data...\n")

survey <- read.csv(
  "https://raw.githubusercontent.com/whipson/tidytuesday/master/young_people.csv"
) %>%
  select(History:Pets)  # Select interest rating columns only

cat("  Survey dimensions:", nrow(survey), "respondents x", ncol(survey), "items\n")
cat("  Items:", paste(colnames(survey), collapse = ", "), "\n")

# =============================================================================
# STEP 2: Careless Responding Detection
# =============================================================================
cat("\nStep 2: Screening for careless responding...\n")

# Add careless responding indicators
interests <- survey %>%
  mutate(
    # Longstring: maximum number of consecutive identical responses
    string = longstring(.),
    # Mahalanobis distance: multivariate outlier detection
    md = outlier(., plot = FALSE)
  )

# Chi-square cutoff for Mahalanobis distance
cutoff <- qchisq(p = 1 - OUTLIER_ALPHA, df = ncol(survey))
cat("  Mahalanobis distance cutoff (p <", OUTLIER_ALPHA, "):", round(cutoff, 2), "\n")

# Filter out careless responders and outliers
interests_clean <- interests %>%
  filter(
    string <= LONGSTRING_CUTOFF,  # Remove long identical response strings
    md < cutoff                    # Remove multivariate outliers
  ) %>%
  select(-string, -md)

cat("  Removed", nrow(survey) - nrow(interests_clean), "cases\n")
cat("  Remaining:", nrow(interests_clean), "respondents\n")

# =============================================================================
# STEP 3: Standardize Indicators
# =============================================================================
cat("\nStep 3: Standardizing indicators (z-score scaling)...\n")

interests_clustering <- interests_clean %>%
  na.omit() %>%
  mutate_all(list(scale))

cat("  Final clustering dataset:", nrow(interests_clustering), "rows x",
    ncol(interests_clustering), "columns\n")

# =============================================================================
# STEP 4: Fit Gaussian Mixture Models
# =============================================================================
cat("\nStep 4: Fitting mixture models and computing BIC...\n")
cat("  This tests multiple covariance structures and profile counts.\n")
cat("  (May take a few minutes...)\n\n")

BIC_results <- mclustBIC(interests_clustering)

cat("BIC Summary (top models):\n")
print(summary(BIC_results))

# Plot BIC across models
plot(BIC_results, main = "BIC by Number of Profiles and Covariance Structure")

# =============================================================================
# STEP 5: Select Optimal Model
# =============================================================================
cat("\nStep 5: Evaluating model selection criteria...\n")

# Fit the top-ranked model
# Using VEE with 3 components as a starting point (adjust based on BIC)
mod1 <- Mclust(interests_clustering, modelNames = "VEE", G = 3, x = BIC_results)

cat("\nSelected Model Summary:\n")
print(summary(mod1))
cat("  Model:", mod1$modelName, "with", mod1$G, "profiles\n")
cat("  Log-likelihood:", mod1$loglik, "\n")
cat("  BIC:", mod1$bic, "\n")

# ICL (Integrated Complete-data Likelihood)
cat("\nComputing ICL...\n")
ICL_results <- mclustICL(interests_clustering)
plot(ICL_results, main = "ICL by Number of Profiles")
cat("ICL Summary:\n")
print(summary(ICL_results))

# Bootstrap Likelihood Ratio Test (BLRT)
cat("\nRunning Bootstrap LRT (tests K vs K-1 profiles)...\n")
cat("  (This may take several minutes...)\n")
tryCatch({
  blrt <- mclustBootstrapLRT(interests_clustering, modelName = "VEE")
  print(blrt)
}, error = function(e) {
  cat("  BLRT computation failed:", e$message, "\n")
  cat("  This can happen with small samples or specific covariance structures.\n")
})

# =============================================================================
# STEP 6: Extract and Visualize Profile Means
# =============================================================================
cat("\nStep 6: Visualizing latent profiles...\n")

# Extract profile means (standardized)
means <- data.frame(mod1$parameters$mean, stringsAsFactors = FALSE) %>%
  rownames_to_column() %>%
  rename(Interest = rowname) %>%
  melt(id.vars = "Interest", variable.name = "Profile", value.name = "Mean") %>%
  mutate(
    Mean = round(Mean, 2),
    Mean = ifelse(Mean > 1, 1, Mean)  # Cap extreme values for visualization
  )

# Profile line plot (standardized means across indicators)
p <- means %>%
  ggplot(aes(Interest, Mean, group = Profile, color = Profile)) +
  geom_point(size = 2.25) +
  geom_line(size = 1.25) +
  scale_x_discrete(limits = c(
    "Active sport", "Adrenaline sports", "Passive sport",
    "Countryside, outdoors", "Gardening", "Cars",
    "Art exhibitions", "Dancing", "Musical instruments", "Theatre", "Writing", "Reading",
    "Geography", "History", "Law", "Politics", "Psychology", "Religion", "Foreign languages",
    "Biology", "Chemistry", "Mathematics", "Medicine", "Physics", "Science and technology",
    "Internet", "PC",
    "Celebrities", "Economy Management", "Fun with friends", "Shopping", "Pets"
  )) +
  labs(x = NULL, y = "Standardized Mean Interest") +
  theme_bw(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  ) +
  ggtitle("Latent Profile Analysis: Interest Profiles")

print(p)

# =============================================================================
# STEP 7: Label Profiles
# =============================================================================
cat("\nStep 7: Profile labeling...\n")

# Profile sizes
profile_sizes <- table(mod1$classification)
profile_pct <- round(prop.table(profile_sizes) * 100, 1)

cat("\nProfile Sizes:\n")
for (i in seq_along(profile_sizes)) {
  cat("  Profile", i, ":", profile_sizes[i], "members (",
      profile_pct[i], "%)\n")
}

# Create labeled profile plot
# Labels should be derived from examining the profile means
# Example labels (replace with data-driven interpretations):
p_labeled <- means %>%
  mutate(Profile = recode(Profile,
    X1 = paste0("Profile 1: ", profile_pct[1], "%"),
    X2 = paste0("Profile 2: ", profile_pct[2], "%"),
    X3 = paste0("Profile 3: ", profile_pct[3], "%")
  )) %>%
  ggplot(aes(Interest, Mean, group = Profile, color = Profile)) +
  geom_point(size = 2.25) +
  geom_line(size = 1.25) +
  scale_x_discrete(limits = c(
    "Active sport", "Adrenaline sports", "Passive sport",
    "Countryside, outdoors", "Gardening", "Cars",
    "Art exhibitions", "Dancing", "Musical instruments", "Theatre", "Writing", "Reading",
    "Geography", "History", "Law", "Politics", "Psychology", "Religion", "Foreign languages",
    "Biology", "Chemistry", "Mathematics", "Medicine", "Physics", "Science and technology",
    "Internet", "PC",
    "Celebrities", "Economy Management", "Fun with friends", "Shopping", "Pets"
  )) +
  labs(x = NULL, y = "Standardized Mean Interest") +
  theme_bw(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    legend.position = "top"
  ) +
  ggtitle("Latent Profiles with Membership Percentages")

print(p_labeled)

cat("\n=== Latent Profile Analysis Complete ===\n")
cat("Review the profile plot to identify meaningful subgroups. Each line\n")
cat("represents a profile's average score across all indicators. Profiles\n")
cat("that differ meaningfully on theoretically important indicators are\n")
cat("most useful for practical applications like targeted interventions,\n")
cat("segmented communication, or tailored development programs.\n")
