# =============================================================================
# Turnover Prediction Using Logistic Regression
# =============================================================================
#
# PURPOSE:
#   Predict voluntary employee turnover using engagement survey responses
#   as predictors. Logistic regression models the probability of termination
#   as a function of survey item scores, enabling identification of which
#   engagement dimensions most strongly predict attrition risk.
#
# ANALYTICAL TECHNIQUE:
#   - Binary logistic regression (GLM with binomial family)
#   - Odds ratios for effect size interpretation
#   - Variable recoding for categorical predictors
#
# STEPS:
#   1. Load and prepare survey data with termination outcomes
#   2. Recode categorical variables to numeric
#   3. Fit logistic regression with survey items as predictors
#   4. Interpret coefficients as log-odds and convert to odds ratios
#   5. Evaluate model significance and identify key predictors
#
# INTERPRETATION GUIDE:
#   - Positive coefficient: higher scores on that item are associated with
#     HIGHER probability of turnover (counterintuitive for engagement items)
#   - Negative coefficient: higher scores REDUCE turnover probability
#   - Odds ratio > 1: each 1-unit increase in the predictor multiplies the
#     odds of turnover by that factor
#   - Odds ratio < 1: each 1-unit increase REDUCES odds of turnover
#   - Significance (p < 0.05): the predictor reliably distinguishes between
#     employees who leave vs. stay
#   - For engagement items, we expect negative coefficients (higher engagement
#     = lower turnover), so any positive coefficients warrant investigation
#
# DATA REQUIREMENTS:
#   Survey response data with:
#   - Likert-scale survey items (1-5)
#   - Binary termination outcome (0/1)
#   - Optional: demographic and job-level covariates
#
# =============================================================================

library(readr)
library(dplyr)
library(tidyr)
library(caret)

# ---- CONFIGURATION ----
DATA_PATH <- "synthetic_survey_responses.csv"

# =============================================================================
# STEP 1: Load and Prepare Data
# =============================================================================
cat("Step 1: Loading survey data with termination outcomes...\n")

df <- read_csv(DATA_PATH, show_col_types = FALSE)
cat("  Dataset dimensions:", nrow(df), "rows x", ncol(df), "columns\n")
cat("  Termination rate:", round(mean(df$terminated, na.rm = TRUE) * 100, 1), "%\n")

# =============================================================================
# STEP 2: Recode Categorical Variables
# =============================================================================
cat("\nStep 2: Recoding categorical variables...\n")

# Recode management level to numeric ordinal scale
# This captures the hierarchical nature of management levels
df <- df %>%
  mutate(management_level_num = recode(management_level,
    "Individual Contributor" = 1,
    "Manager" = 2,
    "Senior Leader" = 3
  ))

# Recode gender to binary (for use as covariate if needed)
df <- df %>%
  mutate(gender_num = ifelse(gender == "Female", 1, 0))

# Recode ethnicity to binary
df <- df %>%
  mutate(minority_num = ifelse(ethnicity == "Minority", 1, 0))

cat("  Management level distribution:\n")
print(table(df$management_level))
cat("  Termination by management level:\n")
print(table(df$management_level, df$terminated))

# =============================================================================
# STEP 3: Fit Logistic Regression Model
# =============================================================================
cat("\nStep 3: Fitting logistic regression model...\n")
cat("  Predicting: terminated (0/1)\n")
cat("  Predictors: engagement, manager, inclusion, and organizational survey items\n\n")

# Model using survey items as predictors
# Each Q variable represents a Likert-scale survey item (1-5)
# Q1_x = Engagement items
# Q2_x = Manager effectiveness items
# Q3_x = Inclusion & diversity items
# Q4_x = Organizational effectiveness items

fit <- glm(terminated ~
             Q1_1 +  # Recommend as great place to work
             Q1_2 +  # Overall satisfaction
             Q1_3 +  # Rarely think about leaving
             Q1_4 +  # Proud to work here
             Q2_1 +  # Career goals can be met
             Q2_2 +  # Learning and development opportunities
             Q2_3 +  # Work gives personal accomplishment
             Q2_4 +  # Manager helps career development
             Q3_1 +  # Can share opinions without fear
             Q3_2 +  # Committed to equal opportunity
             Q3_3 +  # Leadership committed to diversity
             Q3_4 +  # Manager treats people fairly
             Q3_5 +  # Can be myself at work
             Q3_6 +  # Feel part of a team
             Q4_1 +  # Open and honest communication
             Q4_2 +  # Manager communicates change reasons
             Q4_3 +  # Trust senior leadership
             Q4_4,   # Clear link between work and strategy
           data = df, family = binomial)

cat("Model Summary:\n")
cat("=" * 60, "\n")
print(summary(fit))

# =============================================================================
# STEP 4: Odds Ratios
# =============================================================================
cat("\nStep 4: Computing odds ratios...\n")
cat("  Odds ratios convert log-odds to multiplicative effects.\n")
cat("  OR > 1 = increased odds of turnover per unit increase\n")
cat("  OR < 1 = decreased odds of turnover per unit increase\n\n")

# Convert log-odds coefficients to odds ratios
odds_ratios <- exp(coef(fit))
cat("Odds Ratios:\n")
print(round(odds_ratios, 4))

# Confidence intervals for odds ratios
cat("\n95% Confidence Intervals for Odds Ratios:\n")
or_ci <- exp(confint(fit))
print(round(or_ci, 4))

# =============================================================================
# STEP 5: Convert Log-Odds to Probabilities
# =============================================================================
cat("\nStep 5: Converting to predicted probabilities...\n")

# Function to convert logits to probabilities
logit2prob <- function(model) {
  odds <- exp(coef(model))
  prob <- odds / (1 + odds)
  return(prob)
}

probs <- logit2prob(fit)
cat("Predicted Probabilities (at coefficient values):\n")
print(round(probs, 4))

# =============================================================================
# STEP 6: Model Diagnostics
# =============================================================================
cat("\nStep 6: Model diagnostics...\n")

# Null deviance vs Residual deviance
# A large drop indicates the predictors explain substantial variance
cat("  Null deviance:    ", fit$null.deviance, "on", fit$df.null, "df\n")
cat("  Residual deviance:", fit$deviance, "on", fit$df.residual, "df\n")
cat("  Deviance reduction:", round(fit$null.deviance - fit$deviance, 2), "\n")
cat("  AIC:", fit$aic, "\n")

# Pseudo R-squared (McFadden)
pseudo_r2 <- 1 - (fit$deviance / fit$null.deviance)
cat("  McFadden's Pseudo R-squared:", round(pseudo_r2, 4), "\n")

# Identify significant predictors
coef_summary <- summary(fit)$coefficients
sig_predictors <- coef_summary[coef_summary[, 4] < 0.05, ]
cat("\nSignificant Predictors (p < 0.05):\n")
if (nrow(sig_predictors) > 0) {
  print(round(sig_predictors, 4))
} else {
  cat("  No individually significant predictors at p < 0.05\n")
  cat("  (This can occur with multicollinearity among correlated survey items)\n")
}

cat("\n=== Analysis Complete ===\n")
cat("Key takeaway: Examine the odds ratios and significance levels to\n")
cat("identify which engagement dimensions most strongly predict turnover.\n")
cat("Items with significant negative coefficients are protective factors;\n")
cat("addressing low scores on these items may reduce attrition risk.\n")
