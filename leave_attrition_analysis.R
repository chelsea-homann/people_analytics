# =============================================================================
# Leave of Absence & Attrition Analysis
# =============================================================================
#
# PURPOSE:
#   Investigate the association between employee leave of absence history
#   and subsequent voluntary termination. Tests whether employees who take
#   leave are more (or less) likely to terminate compared to those who
#   do not take leave.
#
# ANALYTICAL TECHNIQUE:
#   - Chi-square test of independence (leave status x termination status)
#   - Contingency table analysis
#   - Bar chart visualization of leave x termination relationship
#   - Rehire detection and filtering
#
# STEPS:
#   1. Load workforce data (active workers, leave history, terminations)
#   2. Create distinct employee records and merge datasets
#   3. Create binary indicators: leave (yes/no), terminated (yes/no), rehire
#   4. Filter to relevant population (exclude rehires, filter by hire date)
#   5. Build contingency table and run chi-square test
#   6. Visualize the association
#
# INTERPRETATION GUIDE:
#   - Chi-square p < 0.05: significant association between leave and termination
#   - If employees who take leave are MORE likely to terminate, this may indicate
#     leave is a leading indicator of disengagement or burnout
#   - If employees who take leave are LESS likely to terminate, leave policies
#     may be serving as a retention mechanism
#   - Effect size: compare termination rates between leave/no-leave groups
#   - Cramér's V can quantify the strength of association for chi-square
#
# DATA REQUIREMENTS:
#   Workforce data with: employee_id, hire_date, leave_type, term_date
#   (Using synthetic_workforce.csv which contains all these fields)
#
# =============================================================================

library(readr)
library(dplyr)
library(ggplot2)

# ---- CONFIGURATION ----
DATA_PATH <- "synthetic_workforce.csv"

# Filter to employees hired on or after this date
HIRE_DATE_CUTOFF <- "2015-01-01"

# =============================================================================
# STEP 1: Load Data
# =============================================================================
cat("Step 1: Loading workforce data...\n")

workforce <- read_csv(DATA_PATH, show_col_types = FALSE)
cat("  Total records:", nrow(workforce), "\n")

# =============================================================================
# STEP 2: Parse Dates and Create Indicators
# =============================================================================
cat("\nStep 2: Parsing dates and creating analysis variables...\n")

workforce$hire_date <- as.Date(workforce$hire_date, "%m/%d/%Y")
workforce$term_date <- as.Date(workforce$term_date, "%m/%d/%Y")

# Create distinct employee records
employee_data <- workforce %>%
  distinct(employee_id, .keep_all = TRUE)

# Create binary leave indicator
employee_data$leave <- ifelse(
  !is.na(employee_data$leave_type) & employee_data$leave_type != "None",
  "leave", "no_leave"
)

# Create binary termination indicator
employee_data$term <- ifelse(
  !is.na(employee_data$term_date), "term", "no_term"
)

# Detect rehires (term date before most recent hire date)
employee_data$rehire <- ifelse(
  !is.na(employee_data$term_date) & employee_data$term_date < employee_data$hire_date,
  "rehire", "not_rehire"
)
employee_data$rehire[is.na(employee_data$rehire)] <- "not_rehire"

cat("  Leave distribution:\n")
print(table(employee_data$leave))
cat("  Termination distribution:\n")
print(table(employee_data$term))
cat("  Rehire count:", sum(employee_data$rehire == "rehire"), "\n")

# =============================================================================
# STEP 3: Filter Population
# =============================================================================
cat("\nStep 3: Filtering to analysis population...\n")

# Remove rehires (their tenure calculation is complicated)
model_dataset <- subset(employee_data, rehire == "not_rehire")
cat("  After removing rehires:", nrow(model_dataset), "\n")

# Filter to employees hired after cutoff
model_dataset <- subset(model_dataset, hire_date >= as.Date(HIRE_DATE_CUTOFF))
cat("  After hire date filter (>=", HIRE_DATE_CUTOFF, "):", nrow(model_dataset), "\n")

# Select key columns for analysis
model_dataset <- model_dataset %>%
  select(employee_id, leave, term)

# =============================================================================
# STEP 4: Contingency Table
# =============================================================================
cat("\nStep 4: Building contingency table...\n\n")

contingency <- table(model_dataset$leave, model_dataset$term)
cat("Contingency Table (Leave x Termination):\n")
print(contingency)

cat("\nRow Proportions (termination rate by leave status):\n")
print(round(prop.table(contingency, 1), 3))

cat("\nColumn Proportions (leave rate by termination status):\n")
print(round(prop.table(contingency, 2), 3))

# =============================================================================
# STEP 5: Chi-Square Test
# =============================================================================
cat("\nStep 5: Chi-square test of independence...\n\n")

test <- chisq.test(contingency)
print(test)

# Effect size: Cramér's V
n <- sum(contingency)
cramers_v <- sqrt(test$statistic / (n * (min(nrow(contingency), ncol(contingency)) - 1)))
cat("\nCramér's V:", round(cramers_v, 4), "\n")
cat("  Small effect: V ~ 0.1, Medium: V ~ 0.3, Large: V ~ 0.5\n")

# Expected frequencies (should all be > 5 for chi-square validity)
cat("\nExpected Frequencies:\n")
print(round(test$expected, 1))

if (any(test$expected < 5)) {
  cat("  WARNING: Some expected frequencies < 5. Consider Fisher's exact test.\n")
  cat("\nFisher's Exact Test:\n")
  print(fisher.test(contingency))
}

# =============================================================================
# STEP 6: Visualization
# =============================================================================
cat("\nStep 6: Creating visualization...\n")

# Bar chart of leave x termination
plot_data <- as.data.frame(contingency)
colnames(plot_data) <- c("leave_status", "term_status", "count")

p <- ggplot(model_dataset, aes(x = leave, fill = term)) +
  geom_bar(position = "dodge") +
  scale_fill_manual(values = c("no_term" = "steelblue", "term" = "coral"),
                    name = "Termination Status",
                    labels = c("Active", "Terminated")) +
  labs(
    title = "Leave of Absence and Termination Association",
    x = "Leave Status",
    y = "Count"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(p)

# Proportional bar chart
p2 <- ggplot(model_dataset, aes(x = leave, fill = term)) +
  geom_bar(position = "fill") +
  scale_fill_manual(values = c("no_term" = "steelblue", "term" = "coral"),
                    name = "Termination Status",
                    labels = c("Active", "Terminated")) +
  geom_hline(yintercept = mean(model_dataset$term == "term"),
             linetype = "dashed", color = "black") +
  labs(
    title = "Termination Rate by Leave Status",
    subtitle = "Dashed line = overall termination rate",
    x = "Leave Status",
    y = "Proportion"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(p2)

cat("\n=== Leave & Attrition Analysis Complete ===\n")
cat("If the chi-square test is significant, there is evidence that leave\n")
cat("of absence is associated with termination outcomes. Examine the row\n")
cat("proportions to determine the direction of the effect (whether leave\n")
cat("increases or decreases termination likelihood).\n")
