# =============================================================================
# Recruitment Fairness Analysis
# =============================================================================
#
# PURPOSE:
#   Evaluate whether recruitment and selection processes show evidence of
#   bias by demographic group (gender, ethnicity). Uses chi-square tests,
#   odds ratios, mosaic plots, loglinear analysis, and decision trees to
#   assess adverse impact at each stage of the hiring funnel.
#
# ANALYTICAL TECHNIQUES:
#   - Chi-square test of independence (association between demographics and outcomes)
#   - Odds ratios (effect size for 2x2 tables)
#   - Standardized residuals (identify which cells drive significance)
#   - Mosaic plots (visual representation of contingency tables)
#   - Loglinear analysis (modeling three-way interactions: gender x ethnicity x outcome)
#   - Decision trees (classification of shortlisting by demographics)
#
# STEPS:
#   1. Load and code recruitment data
#   2. Compute frequency tables and proportions for each stage
#   3. Test gender bias in shortlisting (chi-square + odds ratio)
#   4. Test ethnicity bias in shortlisting (chi-square + odds ratio)
#   5. Combined analysis: loglinear model (gender x ethnicity x shortlisting)
#   6. Decision tree visualization
#   7. Stratified analysis by gender
#
# INTERPRETATION GUIDE:
#   - Chi-square p < 0.05: significant association between demographic and outcome
#   - Odds ratio = 1: no bias; OR < 1: group disadvantaged; OR > 1: group advantaged
#   - Standardized residuals > |1.96|: that cell significantly contributes to the
#     overall chi-square at p < 0.05 level
#   - Loglinear: tests whether the three-way interaction (gender x ethnicity x
#     shortlisting) is needed, or if two-way effects are sufficient
#   - The "four-fifths rule" (EEOC): selection rate for a protected group should
#     be at least 80% of the rate for the highest-scoring group
#
# DATA SOURCE:
#   This analysis uses a published textbook dataset for recruitment fairness.
#   Download from: search for "Chapter 8 RECRUITMENT APPLICANTS.csv"
#   from the People Analytics textbook.
#   Alternatively, use any recruitment dataset with columns for:
#     Gender (1=Male, 2=Female), BAMEyn (1=Yes, 2=No),
#     ShortlistedNY (0/1), Interviewed (0/1), OfferNY (0/1), etc.
#
# =============================================================================

library(tidyverse)
library(psych)
library(questionr)    # odds.ratio
library(gmodels)      # CrossTable
library(MASS)         # loglm
library(vcd)          # mosaic plots
library(rpart)        # decision trees
library(rpart.plot)   # tree visualization

# ---- CONFIGURATION ----
# Update this path to your recruitment data file
DATA_PATH <- "Chapter 8 RECRUITMENT APPLICANTS.csv"

# =============================================================================
# STEP 1: Load and Code Data
# =============================================================================
cat("Step 1: Loading and coding recruitment data...\n")

applicants_file <- read.csv(DATA_PATH, header = TRUE, sep = ";")
attach(applicants_file)

cat("  Total applicants:", nrow(applicants_file), "\n")
cat("  Variables:", paste(names(applicants_file), collapse = ", "), "\n")

# Code dummy variables as labeled factors for readable output
gender_mf <- factor(Gender, levels = 1:2, labels = c("Male", "Female"))
bame_yn <- factor(BAMEyn, levels = 1:2, labels = c("Yes", "No"))
shortlisted_ny <- factor(ShortlistedNY, levels = 0:1, labels = c("Rejected", "Shortlisted"))
interviewed_ny <- factor(Interviewed, levels = 0:1, labels = c("Not", "Yes"))
female_on_panel <- factor(FemaleONpanel, levels = 1:2, labels = c("No", "Yes"))
offer_ny <- factor(OfferNY, levels = 0:1, labels = c("Not offered", "Offered"))
accept_yn <- factor(AcceptNY, levels = 0:1, labels = c("Declined", "Accepted"))
join_yn <- factor(JoinYN, levels = 0:1, labels = c("Not joined", "Joined"))

applicants_df <- data.frame(gender_mf, bame_yn, shortlisted_ny, interviewed_ny,
                            female_on_panel, offer_ny, accept_yn, join_yn)

# =============================================================================
# STEP 2: Descriptive Statistics - Funnel Proportions
# =============================================================================
cat("\nStep 2: Computing funnel proportions at each stage...\n\n")

# Helper function to compute and print frequency table
print_freq <- function(var, label) {
  freq_table <- table(var)
  perc_table <- prop.table(freq_table) %>% round(2)
  results <- cbind(Total = freq_table, Proportion = perc_table)
  cat(label, ":\n")
  print(results)
  cat("\n")
}

print_freq(gender_mf, "Gender Distribution")
print_freq(bame_yn, "BAME Distribution")
print_freq(shortlisted_ny, "Shortlisting Outcomes")
print_freq(interviewed_ny, "Interview Outcomes")
print_freq(offer_ny, "Offer Outcomes")
print_freq(join_yn, "Joining Outcomes")

# =============================================================================
# STEP 3: Gender Bias in Shortlisting
# =============================================================================
cat("Step 3: Testing gender bias in shortlisting...\n\n")

shortl_by_gender <- table(gender_mf, shortlisted_ny)
cat("Contingency Table (Gender x Shortlisting):\n")
print(shortl_by_gender)

cat("\nRow proportions (selection rate by gender):\n")
print(prop.table(shortl_by_gender, 1) %>% round(2))

# Chi-square test
chisq_gender <- chisq.test(gender_mf, shortlisted_ny, correct = FALSE)
cat("\nChi-Square Test:\n")
print(chisq_gender)

# Standardized residuals
# Values outside +/- 1.96 are significant at p < 0.05
# Values outside +/- 2.58 are significant at p < 0.01
cat("\nStandardized Residuals (z-scores):\n")
cat("  |z| > 1.96 = p < .05, |z| > 2.58 = p < .01, |z| > 3.29 = p < .001\n")
print(chisq_gender$residuals %>% round(2))

# Odds ratio
cat("\nOdds Ratio:\n")
gender_or <- odds.ratio(shortl_by_gender)
cat("  OR =", round(gender_or$OR, 2), "\n")
cat("  Interpretation: odds of shortlisting for males vs females\n")
cat("  1/OR =", round(1 / gender_or$OR, 2),
    "(inverse: how many times more/less likely)\n")

# Mosaic plot
mosaic(shortl_by_gender, shade = TRUE, abbreviate_labs = 1,
       main = "Gender x Shortlisting (Shading = Residuals)")

# Visualization: Frequency plot
ggplot(applicants_df, aes(shortlisted_ny, fill = gender_mf)) +
  scale_fill_discrete(name = "Gender") +
  geom_bar(position = "dodge") +
  labs(title = "Shortlisting by Gender", x = "", y = "Count") +
  theme_minimal()

# =============================================================================
# STEP 4: Ethnicity (BAME) Bias in Shortlisting
# =============================================================================
cat("\n\nStep 4: Testing ethnicity (BAME) bias in shortlisting...\n\n")

shortl_by_bame <- table(bame_yn, shortlisted_ny)
cat("Contingency Table (BAME x Shortlisting):\n")
print(shortl_by_bame)

cat("\nRow proportions (selection rate by BAME status):\n")
print(prop.table(shortl_by_bame, 1) %>% round(2))

# Chi-square test
chisq_bame <- chisq.test(bame_yn, shortlisted_ny, correct = FALSE)
cat("\nChi-Square Test:\n")
print(chisq_bame)

cat("\nStandardized Residuals:\n")
print(chisq_bame$residuals %>% round(2))

# Odds ratio
cat("\nOdds Ratio:\n")
bame_or <- odds.ratio(shortl_by_bame)
cat("  OR =", round(bame_or$OR, 2), "\n")

# Mosaic plot
mosaic(shortl_by_bame, shade = TRUE, abbreviate_labs = 1,
       labeling_args = list(set_varnames = c(bame_yn = "BAME",
                                              shortlisted_ny = "Shortlisted")),
       main = "BAME x Shortlisting (Shading = Residuals)")

# Fourfold display
fourfold(shortl_by_bame)

# =============================================================================
# STEP 5: Loglinear Analysis (Three-Way: Gender x BAME x Shortlisting)
# =============================================================================
cat("\n\nStep 5: Loglinear analysis - three-way interaction...\n\n")
cat("  Testing whether the Gender x BAME x Shortlisting interaction is needed,\n")
cat("  or whether simpler two-way effects are sufficient.\n\n")

gender <- gender_mf
bame <- bame_yn
shortlisted <- shortlisted_ny
log_df <- data.frame(gender, bame, shortlisted)

my_table <- xtabs(~ gender + bame + shortlisted, data = log_df)
cat("Three-Way Frequency Table:\n")
print(ftable(my_table))

# Saturated model (fits data perfectly)
saturated_model <- loglm(~ gender * bame * shortlisted, data = my_table)
cat("\nSaturated Model (perfect fit):\n")
print(saturated_model)

# Remove three-way interaction
three_way_removed <- update(saturated_model, .~. - gender:bame:shortlisted)
cat("\nModel without three-way interaction:\n")
print(summary(three_way_removed))

cat("\nCompare: Saturated vs Three-Way Removed:\n")
print(anova(saturated_model, three_way_removed))

# Test each two-way interaction
bame_shortlisted <- update(three_way_removed, .~. - bame:shortlisted)
gender_shortlisted <- update(three_way_removed, .~. - gender:shortlisted)
gender_bame <- update(three_way_removed, .~. - gender:bame)

cat("\nDropping BAME:Shortlisted:\n")
print(anova(three_way_removed, bame_shortlisted))

cat("\nDropping Gender:Shortlisted:\n")
print(anova(three_way_removed, gender_shortlisted))

cat("\nDropping Gender:BAME:\n")
print(anova(three_way_removed, gender_bame))

# Final model (keep significant two-way interactions)
cat("\nFinal Model: gender + bame + shortlisted + gender:shortlisted + bame:shortlisted\n")
final_model <- loglm(~ gender + bame + shortlisted +
                        gender:shortlisted + bame:shortlisted, data = my_table)
print(final_model)
mosaic(final_model, main = "Final Loglinear Model Fit")

# =============================================================================
# STEP 6: Decision Tree
# =============================================================================
cat("\n\nStep 6: Decision tree for shortlisting prediction...\n")
cat("  Visualizes which demographic splits best predict shortlisting outcome.\n\n")

# Recode for tree
lut_gender <- c("1" = "Male", "2" = "Female")
lut_bame <- c("1" = "BAME", "2" = "non-BAME")
Gender_label <- lut_gender[as.character(Gender)]
BAME_label <- lut_bame[as.character(BAMEyn)]

fit_tree <- rpart(ShortlistedNY ~ Gender_label + BAME_label, method = "class")
rpart.plot(fit_tree, main = "Decision Tree: Predicting Shortlisting")

# =============================================================================
# STEP 7: Stratified Analysis
# =============================================================================
cat("\nStep 7: Stratified analysis - BAME effect within each gender...\n\n")

male_only <- subset(log_df, gender == "Male")
female_only <- subset(log_df, gender == "Female")

cat("Males only - BAME x Shortlisting:\n")
CrossTable(male_only$bame, male_only$shortlisted, chisq = TRUE,
           sresid = TRUE, format = "SPSS")

cat("\nFemales only - BAME x Shortlisting:\n")
CrossTable(female_only$bame, female_only$shortlisted, chisq = TRUE,
           sresid = TRUE, format = "SPSS")

cat("\n=== Recruitment Fairness Analysis Complete ===\n")
cat("Summary: Review chi-square results, odds ratios, and loglinear model\n")
cat("to determine whether demographic group membership significantly predicts\n")
cat("selection outcomes. Significant effects may indicate adverse impact\n")
cat("requiring further investigation of selection criteria and procedures.\n")
