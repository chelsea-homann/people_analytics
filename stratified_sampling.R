# =============================================================================
# Stratified Random Sampling for Focus Groups
# =============================================================================
#
# PURPOSE:
#   Generate representative random samples for qualitative research (e.g.,
#   focus groups, interviews) using stratified random sampling. Ensures each
#   demographic stratum (e.g., work arrangement x gender) is proportionally
#   represented in the final sample.
#
# ANALYTICAL TECHNIQUE:
#   - Stratified random sampling: divide the population into homogeneous
#     subgroups (strata), then randomly sample from each stratum
#   - This ensures minority subgroups are represented even when simple
#     random sampling might miss them
#
# STEPS:
#   1. Load workforce data and define sampling variables
#   2. Create stratification groups (combination of demographic variables)
#   3. Review population distribution across strata
#   4. Define sample sizes per stratum (proportional or fixed)
#   5. Draw random samples from each stratum
#   6. Combine and export final sample
#
# INTERPRETATION GUIDE:
#   - Proportional sampling: each stratum's sample size is proportional to
#     its share of the population (mirrors real composition)
#   - Fixed/quota sampling: set specific targets per stratum to ensure
#     minimum representation (may overrepresent small groups)
#   - The final sample should be checked against population proportions
#     to assess representativeness
#
# DATA REQUIREMENTS:
#   Employee-level data with demographic variables for stratification
#
# =============================================================================

library(readr)
library(dplyr)
library(summarytools)

# ---- CONFIGURATION ----
DATA_PATH <- "synthetic_workforce.csv"

# Define the stratification variables
# These will be combined to create sampling strata
STRATA_VAR1 <- "department"    # Primary stratification variable
STRATA_VAR2 <- "gender"        # Secondary stratification variable

# Total target sample size
TARGET_SAMPLE_SIZE <- 100

# Sampling method: "proportional" or "fixed"
# "proportional" = sample size per stratum mirrors population proportion
# "fixed" = specify exact counts per stratum (set below)
SAMPLING_METHOD <- "proportional"

# Minimum sample per stratum (for proportional method)
MIN_PER_STRATUM <- 2

# =============================================================================
# STEP 1: Load Data
# =============================================================================
cat("Step 1: Loading workforce data...\n")

master_data <- read_csv(DATA_PATH, show_col_types = FALSE)
cat("  Total population:", nrow(master_data), "employees\n")

# =============================================================================
# STEP 2: Create Stratification Groups
# =============================================================================
cat("\nStep 2: Creating stratification groups...\n")

# Select relevant columns for sampling
sampling_data <- master_data %>%
  select(
    employee_id,
    !!sym(STRATA_VAR1),
    !!sym(STRATA_VAR2),
    name
  )

# Create combined strata label
sampling_data$sample_group <- paste(
  sampling_data[[STRATA_VAR1]],
  sampling_data[[STRATA_VAR2]]
)

cat("  Stratification:", STRATA_VAR1, "x", STRATA_VAR2, "\n")
cat("  Number of unique strata:", length(unique(sampling_data$sample_group)), "\n")

# =============================================================================
# STEP 3: Review Population Distribution
# =============================================================================
cat("\nStep 3: Population distribution across strata...\n\n")

strata_counts <- sampling_data %>%
  group_by(sample_group) %>%
  summarise(n = n(), .groups = "drop") %>%
  arrange(desc(n))

strata_counts$proportion <- round(strata_counts$n / sum(strata_counts$n), 3)

cat("Strata Population Counts:\n")
print(as.data.frame(strata_counts), row.names = FALSE)

# =============================================================================
# STEP 4: Calculate Sample Sizes Per Stratum
# =============================================================================
cat("\nStep 4: Calculating sample sizes per stratum...\n")

if (SAMPLING_METHOD == "proportional") {
  cat("  Method: Proportional sampling (target N =", TARGET_SAMPLE_SIZE, ")\n")

  strata_counts$sample_n <- pmax(
    MIN_PER_STRATUM,
    round(strata_counts$proportion * TARGET_SAMPLE_SIZE)
  )

  # Ensure we don't try to sample more than available
  strata_counts$sample_n <- pmin(strata_counts$sample_n, strata_counts$n)

  cat("  Adjusted total sample:", sum(strata_counts$sample_n), "\n\n")
} else {
  cat("  Method: Fixed quota sampling\n")
  cat("  Define FIXED_QUOTAS list in configuration for custom sizes.\n\n")
  # Example: set all strata to same size
  strata_counts$sample_n <- pmin(5, strata_counts$n)
}

cat("Sample Allocation Plan:\n")
print(as.data.frame(strata_counts), row.names = FALSE)

# =============================================================================
# STEP 5: Draw Stratified Random Samples
# =============================================================================
cat("\nStep 5: Drawing random samples from each stratum...\n")

set.seed(42)  # For reproducibility

sample_list <- list()

for (i in seq_len(nrow(strata_counts))) {
  stratum_name <- strata_counts$sample_group[i]
  sample_size <- strata_counts$sample_n[i]

  stratum_data <- sampling_data %>%
    filter(sample_group == stratum_name)

  if (nrow(stratum_data) >= sample_size) {
    sampled <- stratum_data %>% sample_n(sample_size)
  } else {
    sampled <- stratum_data  # Take all if less than target
  }

  sample_list[[i]] <- sampled
  cat("  ", stratum_name, ": sampled", nrow(sampled), "of", nrow(stratum_data), "\n")
}

# =============================================================================
# STEP 6: Combine and Export
# =============================================================================
cat("\nStep 6: Combining and exporting final sample...\n")

final_sample <- bind_rows(sample_list)

cat("  Final sample size:", nrow(final_sample), "\n")
cat("  Strata represented:", length(unique(final_sample$sample_group)), "\n")

# Quick check: compare sample vs population proportions
cat("\nSample vs Population Representation:\n")
sample_dist <- final_sample %>%
  group_by(sample_group) %>%
  summarise(sample_n = n(), .groups = "drop")

comparison <- merge(strata_counts[, c("sample_group", "n", "proportion")],
                    sample_dist, by = "sample_group", all.x = TRUE)
comparison$sample_proportion <- round(comparison$sample_n / sum(comparison$sample_n, na.rm = TRUE), 3)
comparison$diff <- comparison$sample_proportion - comparison$proportion
print(as.data.frame(comparison), row.names = FALSE)

# Export
OUTPUT_FILE <- "focus_group_sample.csv"
write.csv(final_sample, file = OUTPUT_FILE, row.names = FALSE)
cat("\nFinal sample exported to:", OUTPUT_FILE, "\n")

cat("\n=== Stratified Sampling Complete ===\n")
cat("Review the sample vs population proportions above to verify\n")
cat("adequate representation. Small strata may be over-represented\n")
cat("due to the minimum sample size floor.\n")
