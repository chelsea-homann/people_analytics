# =============================================================================
# PURPOSE
# =============================================================================
# Network Analysis of Survey Favorability Across Organizational Hierarchy
#
# This script creates network visualizations of survey favorability scores
# mapped onto an organizational reporting structure using igraph. Each node
# represents an employee, and edges represent reporting relationships
# (employee -> manager). Nodes are colored by favorability:
#
#   - RED    = Unfavorable (response < 3)
#   - YELLOW = Neutral     (response == 3)
#   - GREEN  = Favorable   (response > 3)
#
# The resulting network visualization reveals:
#   - Clusters of favorable or unfavorable sentiment within the org hierarchy
#   - Whether sentiment patterns follow reporting lines (e.g., entire teams
#     trending unfavorable may indicate a management or team-level issue)
#   - Isolated pockets of low favorability that might otherwise go unnoticed
#     in aggregate summaries
#   - How connected groups of employees share similar or divergent attitudes
#
# How to interpret the plot:
#   - Node color indicates individual favorability on the selected question
#   - Connected clusters sharing the same color suggest team-level trends
#   - Mixed-color clusters suggest variation within a reporting chain
#   - Hub nodes (managers with many direct reports) can highlight leadership
#     influence on sentiment
#
# Based on the enhanced V4 version of the network analysis approach.
# =============================================================================

# =============================================================================
# CONFIG
# =============================================================================

# Path to survey data file (CSV with employee IDs, manager IDs, responses)
DATA_PATH <- "synthetic_survey_responses.csv"

# Which survey question column to visualize (see question reference list below)
QUESTION_COLUMN <- "Q1_1"

# Optional: filter to a specific organizational unit
# Set FILTER_COLUMN to a demographic/org column and FILTER_VALUE to the
# desired value. Set FILTER_COLUMN to NA to skip filtering.
FILTER_COLUMN <- NA        # e.g., "department"
FILTER_VALUE  <- NA        # e.g., "Engineering"

# Favorability thresholds (assumes a 1-5 Likert scale)
FAVORABILITY_THRESHOLDS <- list(
  unfavorable_max = 2,     # response < 3 is unfavorable
  neutral_value   = 3,     # response == 3 is neutral
  favorable_min   = 4      # response > 3 is favorable
)

# Export settings
EXPORT_TO_FILE <- FALSE                  # Set TRUE to save plot as image
EXPORT_PATH    <- "network_plot.png"     # Output file path
EXPORT_WIDTH   <- 2400                   # Image width in pixels
EXPORT_HEIGHT  <- 2000                   # Image height in pixels

# =============================================================================
# QUESTION REFERENCE LIST
# =============================================================================
# The survey data contains the following question columns (Q1_1 through Q4_4).
# Set QUESTION_COLUMN above to the desired column name.
#
#   Q1_1 : I would recommend this organization as a great place to work.
#   Q1_2 : I am proud to work for this organization.
#   Q1_3 : I rarely think about looking for a job at another company.
#   Q1_4 : I see myself still working here in two years.
#
#   Q2_1 : My manager treats me with respect.
#   Q2_2 : My manager gives me useful feedback on my performance.
#   Q2_3 : My manager supports my professional development.
#   Q2_4 : My manager communicates expectations clearly.
#
#   Q3_1 : I have the resources I need to do my job well.
#   Q3_2 : I understand how my work contributes to organizational goals.
#   Q3_3 : Processes and procedures allow me to be effective in my role.
#   Q3_4 : I am able to maintain a healthy work-life balance.
#
#   Q4_1 : This organization values diverse perspectives.
#   Q4_2 : I feel a sense of belonging at this organization.
#   Q4_3 : Decisions here are made fairly and transparently.
#   Q4_4 : I trust senior leadership to act in employees' best interest.
# =============================================================================

# =============================================================================
# LIBRARIES
# =============================================================================
library(dplyr)
library(igraph)
library(stringr)
library(readr)

# =============================================================================
# STEP 1: Load survey data
# =============================================================================
# Expected columns: respondent_id, manager_id, manager_name, department,
#                   gender, ethnicity, is_manager, job_level,
#                   performance_rating, Q1_1 through Q4_4

cat("Loading survey data from:", DATA_PATH, "\n")
survey_data <- read_csv(DATA_PATH, show_col_types = FALSE)

cat("Loaded", nrow(survey_data), "rows and", ncol(survey_data), "columns.\n")
cat("Using question column:", QUESTION_COLUMN, "\n")

# Validate that the question column exists
if (!QUESTION_COLUMN %in% colnames(survey_data)) {
  stop(paste0(
    "Question column '", QUESTION_COLUMN, "' not found in data. ",
    "Available Q columns: ",
    paste(grep("^Q\\d", colnames(survey_data), value = TRUE), collapse = ", ")
  ))
}

# =============================================================================
# STEP 2: Select columns for edge list and compute favorability
# =============================================================================
# Build a working data frame with the columns needed for the network:
#   - respondent_id and manager_id define the edges
#   - the selected question column determines node color
#   - demographic columns are retained for optional filtering

edge_data <- survey_data %>%
  select(
    respondent_id,
    manager_id,
    manager_name,
    department,
    gender,
    ethnicity,
    is_manager,
    job_level,
    performance_rating,
    response = all_of(QUESTION_COLUMN)
  ) %>%
  filter(!is.na(respondent_id), !is.na(manager_id))

cat("Edge list contains", nrow(edge_data), "employee-manager pairs.\n")

# =============================================================================
# STEP 3: Compute favorability color for each node
# =============================================================================
# Assign a color to each respondent based on their survey response:
#   Red    = Unfavorable (below neutral threshold)
#   Yellow = Neutral     (equal to neutral threshold)
#   Green  = Favorable   (above neutral threshold)
#   White  = Missing / NA response

compute_favorability_color <- function(response_value) {
  case_when(
    is.na(response_value)                                          ~ "white",
    response_value < FAVORABILITY_THRESHOLDS$neutral_value         ~ "red",
    response_value == FAVORABILITY_THRESHOLDS$neutral_value        ~ "yellow",
    response_value > FAVORABILITY_THRESHOLDS$neutral_value         ~ "green",
    TRUE                                                           ~ "white"
  )
}

edge_data <- edge_data %>%
  mutate(favorability_color = compute_favorability_color(response))

# Summarize favorability distribution
cat("\nFavorability distribution:\n")
edge_data %>%
  count(favorability_color) %>%
  mutate(pct = round(n / sum(n) * 100, 1)) %>%
  print()

# =============================================================================
# STEP 4: Optional filter to a specific organizational unit
# =============================================================================
if (!is.na(FILTER_COLUMN) && !is.na(FILTER_VALUE)) {
  if (!FILTER_COLUMN %in% colnames(edge_data)) {
    stop(paste0("Filter column '", FILTER_COLUMN, "' not found in data."))
  }
  cat("\nFiltering to", FILTER_COLUMN, "==", FILTER_VALUE, "\n")
  edge_data <- edge_data %>%
    filter(.data[[FILTER_COLUMN]] == FILTER_VALUE)
  cat("After filtering:", nrow(edge_data), "rows remain.\n")
}

# =============================================================================
# STEP 5: Build igraph network from edge list
# =============================================================================
# Create a directed graph where edges go from employee to manager.
# This represents the reporting structure (who reports to whom).

edges <- edge_data %>%
  select(from = respondent_id, to = manager_id)

# Build a node attribute table keyed on respondent_id.
# For manager nodes that do not appear as respondents, we assign white (unknown).
node_attrs <- edge_data %>%
  select(id = respondent_id, favorability_color, department, job_level) %>%
  distinct(id, .keep_all = TRUE)

# Create the graph
g <- graph_from_data_frame(edges, directed = TRUE, vertices = NULL)

cat("\nNetwork summary:\n")
cat("  Nodes:", vcount(g), "\n")
cat("  Edges:", ecount(g), "\n")

# =============================================================================
# STEP 6: Set node colors based on favorability
# =============================================================================
# Map each vertex to its favorability color. Vertices not in the node_attrs
# table (e.g., managers with no survey response) get white.

vertex_names <- V(g)$name
color_lookup <- setNames(node_attrs$favorability_color, as.character(node_attrs$id))

V(g)$color <- ifelse(
  vertex_names %in% names(color_lookup),
  color_lookup[vertex_names],
  "white"
)

# Set additional visual attributes
V(g)$size <- ifelse(degree(g, mode = "in") > 3, 8, 5)   # Managers appear larger
V(g)$label <- NA                                          # Suppress labels for clarity
V(g)$frame.color <- "gray50"

E(g)$arrow.size <- 0.3
E(g)$color <- "gray80"

# =============================================================================
# STEP 7: Plot the network
# =============================================================================
# Use a force-directed layout (Fruchterman-Reingold) for organic clustering.
# The layout positions connected nodes near each other, making reporting
# structure clusters visually apparent.

cat("\nGenerating network plot...\n")

layout_fr <- layout_with_fr(g)

# Define plot function for reuse (screen and export)
plot_network <- function() {
  plot(
    g,
    layout       = layout_fr,
    vertex.size  = V(g)$size,
    vertex.color = V(g)$color,
    vertex.label = V(g)$label,
    vertex.frame.color = V(g)$frame.color,
    edge.arrow.size    = E(g)$arrow.size,
    edge.color         = E(g)$color,
    main = paste0("Survey Favorability Network: ", QUESTION_COLUMN),
    sub  = paste0(
      "Nodes: ", vcount(g), " | Edges: ", ecount(g),
      if (!is.na(FILTER_COLUMN)) paste0(" | Filter: ", FILTER_COLUMN, " = ", FILTER_VALUE) else ""
    )
  )

  # Add legend
  legend(
    "bottomleft",
    legend = c("Favorable (> 3)", "Neutral (= 3)", "Unfavorable (< 3)", "No response"),
    fill   = c("green", "yellow", "red", "white"),
    border = "gray50",
    bty    = "n",
    cex    = 0.8,
    title  = "Favorability"
  )
}

# Plot to screen
plot_network()

# =============================================================================
# STEP 8: Optional export to image file
# =============================================================================
if (EXPORT_TO_FILE) {
  cat("Exporting plot to:", EXPORT_PATH, "\n")
  png(EXPORT_PATH, width = EXPORT_WIDTH, height = EXPORT_HEIGHT, res = 200)
  plot_network()
  dev.off()
  cat("Export complete.\n")
}

cat("\nDone.\n")
