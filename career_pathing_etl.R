# =============================================================================
# Career Pathing ETL Pipeline
# =============================================================================
#
# PURPOSE:
#   Transform employee hire records, internal job changes, and termination
#   data into a wide-format career trajectory dataset. Each row represents
#   one employee, with columns for each successive job held (job1, job2, ...
#   jobN). This enables analysis of career progression, time-to-promotion,
#   and lateral/vertical movement patterns.
#
# ANALYTICAL TECHNIQUES:
#   - SQL window functions (ROW_NUMBER with PARTITION BY) for ranking
#   - Wide-format pivoting of longitudinal job change data
#   - Multi-table joins (hire + career changes + terminations)
#
# STEPS:
#   1. Load configuration and data
#   2. Parse dates from character to Date format
#   3. Filter job changes to relevant movement types (promotions, laterals, demotions)
#   4. Rank each employee's job changes chronologically
#   5. Dynamically pivot job changes into wide format (job1, job2, ... jobN)
#   6. Join hire data, career changes, and termination data
#   7. Export final dataset
#
# INTERPRETATION GUIDE:
#   - job1_profile through jobN_profile: shows the sequence of roles held
#   - job1_level through jobN_level: track upward/lateral/downward movement
#   - job1_reason through jobN_reason: categorizes each move as promotion/lateral/demotion
#   - base_pay columns: track compensation progression over career
#   - Use the 'rank' column to identify most recent position from hire data
#
# DATA REQUIREMENTS:
#   - hire_report: employee hire records with demographics
#   - career_changes: internal job movement history
#   - term_details: termination records (optional, left-joined)
#
# =============================================================================

library(readr)
library(dplyr)
library(sqldf)
library(tidyr)

# ---- CONFIGURATION ----
# Update these paths to point to your data files
DATA_DIR <- "."  # Directory containing the CSV files
HIRE_FILE <- file.path(DATA_DIR, "synthetic_workforce.csv")
CAREER_CHANGES_FILE <- file.path(DATA_DIR, "synthetic_career_changes.csv")

# Maximum number of job ranks to pivot into wide format
# Set this based on the maximum number of job changes any employee has
MAX_JOB_RANKS <- 15

# Job change reasons to include in the analysis
# Adjust these to match your organization's reason codes
VALID_CHANGE_REASONS <- c(
  "Promotion > New Job > Higher Level",
  "Lateral Move > New Job > Same Level",
  "Demotion > New Job > Lower Level"
)

# =============================================================================
# STEP 1: Import Data
# =============================================================================
cat("Step 1: Loading data files...\n")

hire_report <- read_csv(HIRE_FILE, show_col_types = FALSE)
career_changes <- read_csv(CAREER_CHANGES_FILE, show_col_types = FALSE)

cat("  Hire records loaded:", nrow(hire_report), "rows\n")
cat("  Career changes loaded:", nrow(career_changes), "rows\n")

# =============================================================================
# STEP 2: Parse Dates
# =============================================================================
cat("Step 2: Parsing date fields...\n")

hire_report$hire_date <- as.Date(hire_report$hire_date, "%m/%d/%Y")
career_changes$hire_date <- as.Date(career_changes$hire_date, "%m/%d/%Y")
career_changes$effective_date <- as.Date(career_changes$effective_date, "%m/%d/%Y")

# Parse term dates if present in workforce data
if ("term_date" %in% colnames(hire_report)) {
  hire_report$term_date <- as.Date(hire_report$term_date, "%m/%d/%Y")
}

# =============================================================================
# STEP 3: Create Hire Dataset with Most Recent Position
# =============================================================================
cat("Step 3: Identifying most recent hire position per employee...\n")

# Use window function to rank positions, keeping the most recent
hire_report_subset <- sqldf("
  SELECT
    employee_id,
    name,
    department,
    job_profile,
    job_level,
    grade,
    annual_salary,
    gender,
    minority_status,
    veteran,
    disability,
    age_group,
    location,
    hire_date,
    term_date,
    term_reason,
    term_category,
    ROW_NUMBER() OVER (
      PARTITION BY employee_id
      ORDER BY hire_date DESC
    ) AS rank
  FROM hire_report
")

# Keep only the most recent position per employee
hire_report_subset <- subset(hire_report_subset, rank == 1)
hire_report_subset$rank <- NULL

cat("  Unique employees in hire data:", nrow(hire_report_subset), "\n")

# =============================================================================
# STEP 4: Filter and Rank Career Changes
# =============================================================================
cat("Step 4: Filtering career changes to relevant movement types...\n")

# Filter to valid change reasons
career_changes_filtered <- career_changes[
  career_changes$reason %in% VALID_CHANGE_REASONS,
]

cat("  Career changes after filtering:", nrow(career_changes_filtered), "\n")

# Assign chronological rank per employee
career_changes_ranked <- sqldf("
  SELECT *,
    ROW_NUMBER() OVER (
      PARTITION BY employee_id
      ORDER BY effective_date ASC
    ) AS job_rank
  FROM career_changes_filtered
")

# Check the distribution of job ranks
max_rank <- max(career_changes_ranked$job_rank, na.rm = TRUE)
cat("  Maximum job changes for a single employee:", max_rank, "\n")

# =============================================================================
# STEP 5: Dynamic Wide-Format Pivot
# =============================================================================
cat("Step 5: Pivoting career changes to wide format...\n")

# This replaces the original approach of manually creating job1, job2, ..., jobN
# using copy-pasted SQL blocks. Instead, we use a loop to dynamically build
# the wide-format dataset for any number of job changes.

# Start with job1 (the "from" position at first change)
job1 <- sqldf("
  SELECT
    employee_id,
    worker_name,
    hire_date AS job1_hiredate,
    job_profile_from AS job1_profile,
    base_pay_from AS job1_base_pay,
    department_from AS job1_department,
    job_level_from AS job1_level,
    grade_from AS job1_grade,
    reason AS job1_reason,
    is_manager_from AS job1_ismgr_from,
    is_manager_to AS job1_ismgr_to,
    gender,
    minority_status,
    age
  FROM career_changes_ranked
  WHERE job_rank = 1
")

# Build subsequent jobs (job2, job3, ..., jobN) using "To" fields
# Each subsequent job comes from the "To" columns of the corresponding rank
job_list <- list(job1)

effective_ranks <- min(max_rank, MAX_JOB_RANKS)

for (i in 1:effective_ranks) {
  job_num <- i + 1
  job_rank_val <- i  # job2 comes from job_rank=1's "To" fields, job3 from rank=2, etc.

  query <- sprintf("
    SELECT
      employee_id,
      effective_date AS job%d_date,
      job_profile_to AS job%d_profile,
      base_pay_to AS job%d_base_pay,
      department_to AS job%d_department,
      job_level_to AS job%d_level,
      grade_to AS job%d_grade,
      reason AS job%d_reason,
      is_manager_from AS job%d_ismgr_from,
      is_manager_to AS job%d_ismgr_to
    FROM career_changes_ranked
    WHERE job_rank = %d
  ", job_num, job_num, job_num, job_num, job_num, job_num, job_num, job_num, job_num, job_rank_val)

  job_df <- sqldf(query)

  if (nrow(job_df) > 0) {
    job_list[[length(job_list) + 1]] <- job_df
  } else {
    break  # No more job changes at this rank
  }
}

cat("  Number of job-level tables created:", length(job_list), "\n")

# =============================================================================
# STEP 6: Join All Job Tables
# =============================================================================
cat("Step 6: Joining all career change tables...\n")

# Start with job1 and progressively left join each subsequent job table
final_career <- job_list[[1]]

if (length(job_list) > 1) {
  for (i in 2:length(job_list)) {
    final_career <- merge(final_career, job_list[[i]],
                          by = "employee_id", all.x = TRUE)
  }
}

cat("  Employees with career change data:", nrow(final_career), "\n")

# =============================================================================
# STEP 7: Final Join - Hire + Career + Termination
# =============================================================================
cat("Step 7: Creating final combined dataset...\n")

# Left join hire data with career changes
final_dataset <- merge(hire_report_subset, final_career,
                        by = "employee_id", all.x = TRUE)

cat("  Final dataset dimensions:", nrow(final_dataset), "rows x",
    ncol(final_dataset), "columns\n")

# =============================================================================
# STEP 8: Export
# =============================================================================
OUTPUT_FILE <- file.path(DATA_DIR, "career_pathing_output.csv")
write.csv(final_dataset, file = OUTPUT_FILE, row.names = FALSE, na = "")

cat("\nComplete! Output saved to:", OUTPUT_FILE, "\n")
cat("Columns in final dataset:\n")
cat(paste(" ", colnames(final_dataset), collapse = "\n"), "\n")
