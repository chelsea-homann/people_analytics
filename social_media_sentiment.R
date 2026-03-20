# =============================================================================
# Social Media Sentiment Analysis
# =============================================================================
#
# PURPOSE:
#   Analyze sentiment in social media posts using the NRC Emotion Lexicon.
#   Compares sentiment distributions across different topics or hashtags,
#   generates word clouds, and produces sentiment bar charts. Useful for
#   understanding public perception, employer brand monitoring, or
#   competitive benchmarking.
#
# ANALYTICAL TECHNIQUES:
#   - NRC sentiment lexicon scoring (8 emotions + positive/negative)
#   - Word cloud generation from text corpus
#   - Text preprocessing (lowercasing, removing URLs, mentions, punctuation)
#   - Comparative sentiment visualization
#
# STEPS:
#   1. Load social media post data (from CSV)
#   2. Clean and preprocess text
#   3. Generate word clouds per topic
#   4. Score each post using the NRC sentiment dictionary
#   5. Aggregate and visualize sentiment scores across topics
#
# INTERPRETATION GUIDE:
#   - NRC Lexicon classifies words into 8 emotions: anger, anticipation,
#     disgust, fear, joy, sadness, surprise, trust
#   - Also provides overall positive/negative classification
#   - Higher scores = more words in the text matching that emotion category
#   - Compare across topics to identify which topics generate more
#     positive/negative sentiment or specific emotional reactions
#   - Word clouds show the most frequent terms after stop word removal
#
# NOTE:
#   The original version used Twitter API with twitteR package. Since API
#   access has changed significantly, this version reads from a CSV file.
#   You can populate the CSV from any social media data source or API.
#
# DATA REQUIREMENTS:
#   CSV with columns: post_id, text, topic (or hashtag)
#
# =============================================================================

library(NLP)
library(syuzhet)       # NRC sentiment scoring
library(tm)            # Text mining
library(SnowballC)     # Stemming
library(wordcloud)     # Word cloud visualization
library(ggplot2)
library(dplyr)
library(RColorBrewer)

# ---- CONFIGURATION ----
DATA_PATH <- "synthetic_social_posts.csv"

# Topics/hashtags to compare (set to NULL to use all topics in data)
TOPICS_TO_COMPARE <- NULL  # e.g., c("workplace", "technology", "leadership")

# Minimum word frequency for word cloud display
WORDCLOUD_MIN_FREQ <- 2
WORDCLOUD_MAX_WORDS <- 200

# =============================================================================
# STEP 1: Load Data
# =============================================================================
cat("Step 1: Loading social media post data...\n")

posts <- read.csv(DATA_PATH, stringsAsFactors = FALSE)
cat("  Total posts loaded:", nrow(posts), "\n")
cat("  Topics found:", paste(unique(posts$topic), collapse = ", "), "\n")

# Filter to specified topics if set
if (!is.null(TOPICS_TO_COMPARE)) {
  posts <- posts[posts$topic %in% TOPICS_TO_COMPARE, ]
  cat("  Posts after topic filter:", nrow(posts), "\n")
}

topics <- unique(posts$topic)

# =============================================================================
# STEP 2: Text Preprocessing Function
# =============================================================================
cat("\nStep 2: Preprocessing text...\n")

clean_text <- function(text_vector) {
  # Convert to lowercase
  text_vector <- tolower(text_vector)
  # Remove retweet markers
  text_vector <- gsub("rt", "", text_vector)
  # Remove @mentions
  text_vector <- gsub("@\\w+", "", text_vector)
  # Remove URLs
  text_vector <- gsub("http\\w+", "", text_vector)
  # Remove punctuation
  text_vector <- gsub("[[:punct:]]", "", text_vector)
  # Remove extra whitespace and tabs
  text_vector <- gsub("[ |\t]{2,}", " ", text_vector)
  # Trim leading/trailing spaces
  text_vector <- trimws(text_vector)
  return(text_vector)
}

# =============================================================================
# STEP 3: Generate Word Clouds Per Topic
# =============================================================================
cat("\nStep 3: Generating word clouds per topic...\n")

for (topic in topics) {
  cat("  Processing topic:", topic, "\n")

  topic_posts <- posts[posts$topic == topic, ]
  cleaned_text <- clean_text(topic_posts$text)

  # Create text corpus
  corpus <- Corpus(VectorSource(cleaned_text))
  corpus <- tm_map(corpus, function(x) removeWords(x, stopwords("english")))

  # Generate word cloud
  tryCatch({
    wordcloud(corpus,
              min.freq = WORDCLOUD_MIN_FREQ,
              colors = brewer.pal(8, "Dark2"),
              random.color = TRUE,
              max.words = WORDCLOUD_MAX_WORDS,
              main = paste("Word Cloud:", topic))
    title(main = paste("Word Cloud:", topic))
  }, error = function(e) {
    cat("    Word cloud skipped for", topic, "(insufficient unique words)\n")
  })
}

# =============================================================================
# STEP 4: NRC Sentiment Scoring
# =============================================================================
cat("\nStep 4: Scoring sentiment using NRC Emotion Lexicon...\n")
cat("  The NRC lexicon classifies words into 8 emotions + positive/negative.\n\n")

all_sentiment_results <- list()

for (topic in topics) {
  cat("  Scoring topic:", topic, "\n")

  topic_posts <- posts[posts$topic == topic, ]
  cleaned_text <- clean_text(topic_posts$text)

  # Get NRC sentiment scores for all posts in this topic
  sentiment_scores <- get_nrc_sentiment(cleaned_text)

  # Aggregate: total score per sentiment category
  totals <- data.frame(colSums(sentiment_scores))
  names(totals) <- "Score"
  totals$sentiment <- rownames(totals)
  totals$topic <- topic
  rownames(totals) <- NULL

  all_sentiment_results[[topic]] <- totals
}

# Combine all results
sentiment_combined <- bind_rows(all_sentiment_results)

# =============================================================================
# STEP 5: Sentiment Visualizations
# =============================================================================
cat("\nStep 5: Creating sentiment visualizations...\n")

# Individual bar charts per topic
for (topic in topics) {
  topic_data <- sentiment_combined[sentiment_combined$topic == topic, ]

  p <- ggplot(topic_data, aes(x = sentiment, y = Score, fill = sentiment)) +
    geom_bar(stat = "identity") +
    theme_minimal() +
    theme(
      legend.position = "none",
      axis.text.x = element_text(angle = 45, hjust = 1),
      plot.title = element_text(hjust = 0.5, face = "bold")
    ) +
    labs(
      title = paste("NRC Sentiment Scores:", topic),
      x = "Sentiment Category",
      y = "Total Score"
    )

  print(p)
}

# Comparative faceted chart (all topics side by side)
p_combined <- ggplot(sentiment_combined,
                     aes(x = sentiment, y = Score, fill = sentiment)) +
  geom_bar(stat = "identity") +
  facet_wrap(~ topic, scales = "free_y") +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    plot.title = element_text(hjust = 0.5, face = "bold")
  ) +
  labs(
    title = "NRC Sentiment Comparison Across Topics",
    x = "Sentiment Category",
    y = "Total Score"
  )

print(p_combined)

# =============================================================================
# STEP 6: Summary Table
# =============================================================================
cat("\nStep 6: Sentiment summary table...\n\n")

summary_wide <- sentiment_combined %>%
  tidyr::pivot_wider(names_from = sentiment, values_from = Score)

print(as.data.frame(summary_wide))

cat("\n=== Social Media Sentiment Analysis Complete ===\n")
cat("Compare sentiment distributions across topics to identify which\n")
cat("generate more positive/negative emotional responses. High 'trust'\n")
cat("and 'joy' scores indicate favorable perception; high 'anger' or\n")
cat("'fear' scores may warrant attention.\n")
