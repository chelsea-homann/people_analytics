###############################################################################
# PAY EQUITY REGRESSION ANALYSIS
###############################################################################
#
# PURPOSE
# -------
# This script detects pay disparities across demographic groups using stepwise
# ordinary least squares (OLS) regression with log-transformed salary as the
# dependent variable. Log transformation allows regression coefficients to be
# interpreted as approximate percentage differences in pay.
#
# The stepwise approach starts with an "unadjusted" model containing only the
# minority indicator, then progressively adds legitimate pay-determining factors
# (controls) one at a time. This reveals:
#
#   1. The RAW (unadjusted) pay gap between minority and non-minority employees.
#   2. How much of that gap is explained by each successive control variable.
#   3. The ADJUSTED pay gap that remains after all legitimate factors are
#      accounted for -- the portion that may reflect inequitable pay practices.
#
# INTERPRETING LOG-SALARY COEFFICIENTS
# -------------------------------------
# When the dependent variable is log(salary), a coefficient on a dummy variable
# (like minority_flag) approximates a percentage difference:
#
#   coefficient = -0.05  -->  minority employees earn roughly 5% less
#   coefficient = -0.12  -->  minority employees earn roughly 12% less
#
# More precisely, the percent difference is (exp(coef) - 1) * 100, but for
# small coefficients (|coef| < 0.15) the raw coefficient is a close
# approximation.
#
# READING THE STEPWISE RESULTS
# ----------------------------
# - Model 1 (minority only): the UNADJUSTED gap. This is the simple average
#   pay difference without controlling for anything.
# - Models 2-8 (adding controls one at a time): watch how the minority
#   coefficient changes. If it shrinks toward zero, the added variable
#   "explains" part of the raw gap (e.g., differences in education or grade
#   account for some of the disparity).
# - Final model (all controls): the ADJUSTED gap. A statistically significant
#   negative coefficient here suggests that minority employees are paid less
#   even after accounting for all measured legitimate factors.
#
# A large unadjusted gap that narrows substantially in the full model indicates
# that much of the raw difference is driven by structural factors (grade mix,
# tenure, etc.) rather than direct pay inequity. Conversely, a gap that
# persists in the full model warrants further investigation.
#
###############################################################################


###############################################################################
# CONFIGURATION
###############################################################################

# Path to the input CSV file.
DATA_PATH <- "synthetic_workforce.csv"

# Which employee population to analyze.
# Options: "exempt", "non_exempt", "all"
POPULATION_FILTER <- "all"


###############################################################################
# LIBRARIES
###############################################################################

library(readr)        # Fast CSV reading
library(dplyr)        # Data manipulation
library(broom)        # Tidy model output
library(stargazer)    # Side-by-side regression tables
library(summarytools) # Descriptive statistics


###############################################################################
# STEP 1: LOAD AND FILTER DATA
###############################################################################
# Read the workforce data from CSV. No API calls -- all data is sourced from
# a flat file. After loading, we apply population filters so the analysis is
# scoped to the desired employee group.

raw_data <- read_csv(DATA_PATH, show_col_types = FALSE)

cat("Rows loaded from CSV:", nrow(raw_data), "\n")

# Apply exempt-status filter based on POPULATION_FILTER.
if (POPULATION_FILTER == "exempt") {
  filtered_data <- raw_data %>% filter(exempt == TRUE)
} else if (POPULATION_FILTER == "non_exempt") {
  filtered_data <- raw_data %>% filter(exempt == FALSE)
} else {
  filtered_data <- raw_data
}

# Keep only active regular employees (exclude temps, contractors, etc.).
# Adjust the value to match your employee_type coding.
filtered_data <- filtered_data %>%
  filter(employee_type == "Regular")

# Drop rows with missing salary or missing minority status -- these are
# required for every model.
filtered_data <- filtered_data %>%
  filter(!is.na(annual_salary), !is.na(minority_status))

cat("Rows after filtering:", nrow(filtered_data), "\n")
cat("Population filter applied:", POPULATION_FILTER, "\n\n")


###############################################################################
# STEP 2: CREATE GROUPINGS / BINNING
###############################################################################
# Bin continuous variables into categories so they behave as factors in the
# regression. Binning prevents the model from assuming a strictly linear
# relationship and better captures nonlinear pay structures (e.g., pay bands
# by grade).

# --- Grade grouping ---
# Use the grade_profile column directly as a categorical. If your grades are
# numeric and numerous, you could bin them into ranges here.
filtered_data <- filtered_data %>%
  mutate(grade_group = as.character(grade_profile))

# --- Education grouping ---
# Collapse highest_degree into broad tiers. Adjust labels to match your data.
filtered_data <- filtered_data %>%
  mutate(education_group = case_when(
    highest_degree %in% c("High School", "GED")         ~ "High School",
    highest_degree %in% c("Associate")                   ~ "Associate",
    highest_degree %in% c("Bachelor")                    ~ "Bachelor",
    highest_degree %in% c("Master")                      ~ "Master",
    highest_degree %in% c("Doctorate", "PhD", "JD", "MD") ~ "Doctorate",
    TRUE                                                  ~ "Other"
  ))

# --- Time-in-job binning ---
# Convert continuous time_in_job_years into ordinal bands. These bands reflect
# typical experience tiers and help the model capture diminishing returns to
# time in role.
filtered_data <- filtered_data %>%
  mutate(time_in_job_bin = case_when(
    time_in_job_years < 1                          ~ "< 1 year",
    time_in_job_years >= 1  & time_in_job_years < 3  ~ "1-3 years",
    time_in_job_years >= 3  & time_in_job_years < 5  ~ "3-5 years",
    time_in_job_years >= 5  & time_in_job_years < 10 ~ "5-10 years",
    time_in_job_years >= 10                         ~ "10+ years",
    TRUE                                            ~ "Unknown"
  ))


###############################################################################
# STEP 3: LOG-TRANSFORM SALARY
###############################################################################
# Taking the natural log of salary is standard in pay-equity analysis because:
#   - Pay distributions are right-skewed; log transformation normalizes them.
#   - Coefficients on dummy variables approximate percentage differences,
#     which are more meaningful than dollar differences when comparing across
#     pay levels.
#   - Multiplicative pay effects (e.g., a 5% raise) become additive in log
#     space, satisfying OLS linearity assumptions.

filtered_data <- filtered_data %>%
  mutate(log_salary = log(annual_salary))


###############################################################################
# STEP 4: CREATE MINORITY DUMMY VARIABLE
###############################################################################
# Create a binary 0/1 indicator for minority status. This is the key
# independent variable of interest. A value of 1 indicates the employee
# belongs to a minority group; 0 indicates non-minority.
#
# The coefficient on this variable represents the estimated pay gap between
# minority and non-minority employees (negative = minority earns less).

filtered_data <- filtered_data %>%
  mutate(minority_flag = ifelse(minority_status == TRUE, 1L, 0L))

cat("Minority distribution:\n")
print(table(filtered_data$minority_flag, useNA = "ifany"))
cat("\n")


###############################################################################
# STEP 5: CAST CATEGORICALS AS FACTORS
###############################################################################
# R's lm() function automatically creates dummy variables for factors. Setting
# the reference level (first level) ensures consistent interpretation: each
# coefficient represents the difference relative to the reference category.

filtered_data <- filtered_data %>%
  mutate(
    education_group  = factor(education_group,
                              levels = c("High School", "Associate",
                                         "Bachelor", "Master",
                                         "Doctorate", "Other")),
    location         = as.factor(location),
    time_in_job_bin  = factor(time_in_job_bin,
                              levels = c("< 1 year", "1-3 years",
                                         "3-5 years", "5-10 years",
                                         "10+ years", "Unknown")),
    grade_group      = as.factor(grade_group),
    job_level        = as.factor(job_level),
    is_manager       = factor(is_manager, levels = c(FALSE, TRUE),
                              labels = c("Non-Manager", "Manager"))
  )


###############################################################################
# STEP 6: SUMMARY STATISTICS BY MINORITY STATUS
###############################################################################
# Before running regressions, inspect descriptive statistics to understand the
# composition of each group. Large differences in education, tenure, or grade
# mix suggest that much of the raw gap may be structural.

cat("=== Summary Statistics: Non-Minority (minority_flag == 0) ===\n")
filtered_data %>%
  filter(minority_flag == 0) %>%
  select(annual_salary, tenure_years, time_in_job_years) %>%
  descr(stats = c("mean", "sd", "min", "med", "max", "n.valid"),
        transpose = TRUE) %>%
  print()

cat("\n=== Summary Statistics: Minority (minority_flag == 1) ===\n")
filtered_data %>%
  filter(minority_flag == 1) %>%
  select(annual_salary, tenure_years, time_in_job_years) %>%
  descr(stats = c("mean", "sd", "min", "med", "max", "n.valid"),
        transpose = TRUE) %>%
  print()

cat("\n=== Education Distribution by Minority Status ===\n")
print(table(filtered_data$minority_flag, filtered_data$education_group))

cat("\n=== Grade Distribution by Minority Status ===\n")
print(table(filtered_data$minority_flag, filtered_data$grade_group))
cat("\n")


###############################################################################
# STEP 7: STEPWISE OLS REGRESSION
###############################################################################
# We build the regression in stages. Each model adds one control variable to
# the previous model. This lets us see exactly which factors account for the
# raw pay gap and how much of the gap each factor explains.
#
# Model 1 -- UNADJUSTED: minority status only.
#   Interpretation: the raw average log-pay difference between groups, with
#   no controls. This is the "headline" gap figure.
#
# Models 2-8 -- PARTIALLY ADJUSTED: adding controls one at a time.
#   Watch the minority_flag coefficient: if it shrinks, the newly added
#   variable explains part of the gap. If it stays stable or grows, the
#   new variable does not explain the disparity.
#
# Model 8 -- FULLY ADJUSTED: all controls included.
#   The minority_flag coefficient here is the adjusted gap -- the pay
#   difference that remains after accounting for all measured legitimate
#   factors. A statistically significant adjusted gap is the strongest
#   evidence of potential pay inequity.

# Model 1: Minority only (unadjusted gap)
model_1 <- lm(log_salary ~ minority_flag,
              data = filtered_data)

# Model 2: Add education
model_2 <- lm(log_salary ~ minority_flag + education_group,
              data = filtered_data)

# Model 3: Add location
model_3 <- lm(log_salary ~ minority_flag + education_group + location,
              data = filtered_data)

# Model 4: Add time-in-job
model_4 <- lm(log_salary ~ minority_flag + education_group + location +
                 time_in_job_bin,
              data = filtered_data)

# Model 5: Add tenure
model_5 <- lm(log_salary ~ minority_flag + education_group + location +
                 time_in_job_bin + tenure_years,
              data = filtered_data)

# Model 6: Add grade
model_6 <- lm(log_salary ~ minority_flag + education_group + location +
                 time_in_job_bin + tenure_years + grade_group,
              data = filtered_data)

# Model 7: Add job level
model_7 <- lm(log_salary ~ minority_flag + education_group + location +
                 time_in_job_bin + tenure_years + grade_group + job_level,
              data = filtered_data)

# Model 8: Add manager status (full model)
model_8 <- lm(log_salary ~ minority_flag + education_group + location +
                 time_in_job_bin + tenure_years + grade_group + job_level +
                 is_manager,
              data = filtered_data)


###############################################################################
# STEP 8: DISPLAY ALL MODEL SUMMARIES
###############################################################################

# --- 8a. Individual model summaries ---
# Print each model so you can see full coefficient tables, R-squared, and
# significance for every term -- not just minority_flag.

models <- list(model_1, model_2, model_3, model_4,
               model_5, model_6, model_7, model_8)

model_labels <- c(
  "Model 1: Minority Only (Unadjusted)",
  "Model 2: + Education",
  "Model 3: + Location",
  "Model 4: + Time-in-Job",
  "Model 5: + Tenure",
  "Model 6: + Grade",
  "Model 7: + Job Level",
  "Model 8: + Manager (Full Model)"
)

for (i in seq_along(models)) {
  cat("\n")
  cat(strrep("=", 70), "\n")
  cat(model_labels[i], "\n")
  cat(strrep("=", 70), "\n")
  print(summary(models[[i]]))
}

# --- 8b. Side-by-side comparison table (stargazer) ---
# This table shows all eight models in columns so you can track how the
# minority_flag coefficient evolves as controls are added. Look at the row
# for minority_flag and read left to right.

cat("\n")
cat(strrep("=", 70), "\n")
cat("SIDE-BY-SIDE MODEL COMPARISON\n")
cat(strrep("=", 70), "\n\n")

stargazer(model_1, model_2, model_3, model_4,
          model_5, model_6, model_7, model_8,
          type            = "text",
          title           = "Stepwise Pay Equity Regression (Dep. Var: log(salary))",
          column.labels   = c("Unadj.", "+Educ", "+Loc", "+TIJ",
                               "+Tenure", "+Grade", "+JobLvl", "Full"),
          dep.var.labels  = "log(Annual Salary)",
          covariate.labels = NULL,
          keep.stat       = c("n", "rsq", "adj.rsq", "f"),
          digits          = 4,
          star.cutoffs    = c(0.05, 0.01, 0.001))

# --- 8c. Tidy summary of the minority coefficient across all models ---
# Extract just the minority_flag row from each model for a concise comparison.

cat("\n")
cat(strrep("=", 70), "\n")
cat("MINORITY COEFFICIENT PROGRESSION\n")
cat(strrep("=", 70), "\n\n")
cat("This table tracks the minority_flag coefficient (approximate % pay gap)\n")
cat("as controls are added. A negative value means minority employees earn\n")
cat("less. Watch for the coefficient to narrow toward zero as legitimate\n")
cat("factors are accounted for.\n\n")

minority_progression <- bind_rows(
  lapply(seq_along(models), function(i) {
    tidy(models[[i]]) %>%
      filter(term == "minority_flag") %>%
      mutate(
        model       = model_labels[i],
        approx_pct  = round((exp(estimate) - 1) * 100, 2)
      ) %>%
      select(model, estimate, std.error, statistic, p.value, approx_pct)
  })
)

print(as.data.frame(minority_progression), row.names = FALSE)

cat("\n")
cat("'approx_pct' = (exp(coefficient) - 1) * 100, the exact % pay\n")
cat("difference attributable to minority status in each model.\n")


###############################################################################
# INTERPRETATION GUIDE (PRINTED AT END FOR REFERENCE)
###############################################################################

cat("\n")
cat(strrep("#", 70), "\n")
cat("# INTERPRETATION GUIDE\n")
cat(strrep("#", 70), "\n\n")

cat("1. UNADJUSTED GAP (Model 1):\n")
cat("   The minority_flag coefficient with no controls. This is the raw\n")
cat("   average difference in log(salary). Multiply by ~100 for an\n")
cat("   approximate percentage gap, or use (exp(coef)-1)*100 for the\n")
cat("   exact figure.\n\n")

cat("2. ADJUSTED GAP (Model 8 / Full Model):\n")
cat("   The minority_flag coefficient after all legitimate pay factors\n")
cat("   are held constant. This isolates the pay difference that cannot\n")
cat("   be explained by education, location, tenure, grade, job level,\n")
cat("   time in job, or manager status.\n\n")

cat("3. STEPWISE NARROWING:\n")
cat("   Compare the minority_flag coefficient across models 1 through 8.\n")
cat("   - A large drop when adding a control means that factor explains\n")
cat("     a substantial portion of the raw gap.\n")
cat("   - If the coefficient barely changes, that control does not\n")
cat("     account for the disparity.\n")
cat("   - A coefficient that GROWS when a control is added (suppression)\n")
cat("     means the control was masking an even larger underlying gap.\n\n")

cat("4. STATISTICAL SIGNIFICANCE:\n")
cat("   Look at the p-value for minority_flag in the full model.\n")
cat("   - p < 0.05: the adjusted gap is statistically significant at\n")
cat("     the 95% confidence level.\n")
cat("   - p >= 0.05: the adjusted gap is not distinguishable from zero\n")
cat("     at conventional significance levels.\n")
cat("   Note: statistical significance depends on sample size. A small\n")
cat("   but real gap may not reach significance with few observations.\n\n")

cat("5. R-SQUARED PROGRESSION:\n")
cat("   Watch how R-squared increases across models. Higher R-squared\n")
cat("   means the model explains more variance in pay. The full model's\n")
cat("   R-squared indicates how well the measured factors collectively\n")
cat("   predict salary.\n\n")

cat("6. PRACTICAL SIGNIFICANCE:\n")
cat("   Even if statistically significant, consider whether the adjusted\n")
cat("   gap is practically meaningful. A 0.5% gap in a 500-person firm\n")
cat("   may warrant monitoring; a 5% gap almost certainly warrants\n")
cat("   remediation.\n\n")

cat(strrep("#", 70), "\n")
cat("# END OF ANALYSIS\n")
cat(strrep("#", 70), "\n")
