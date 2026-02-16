library(tidyverse)

# Load data ----
df <- read_csv("outputs/interventions_us_113_with_embeddings.csv") |>
  mutate(word_count = str_count(speech, "\\S+")) |> 
  select(-...1)

# Overall descriptives ----
cat("\n=========================================================\n")
cat("  Speech Length Descriptives — US 113th Congress\n")
cat("=========================================================\n\n")

overall <- df |>
  summarise(
    n        = n(),
    mean     = mean(word_count, na.rm = TRUE),
    median   = median(word_count, na.rm = TRUE),
    sd       = sd(word_count, na.rm = TRUE),
    min      = min(word_count, na.rm = TRUE),
    max      = max(word_count, na.rm = TRUE),
    q25      = quantile(word_count, 0.25, na.rm = TRUE),
    q75      = quantile(word_count, 0.75, na.rm = TRUE),
    q90      = quantile(word_count, 0.90, na.rm = TRUE),
    q95      = quantile(word_count, 0.95, na.rm = TRUE),
    iqr      = IQR(word_count, na.rm = TRUE),
    skewness = (mean(word_count, na.rm = TRUE) - median(word_count, na.rm = TRUE)) / sd(word_count, na.rm = TRUE)
  )

print(as.data.frame(overall))

# By party ----
cat("\n--- By party ---\n\n")
by_party <- df |>
  group_by(party) |>
  summarise(
    n      = n(),
    mean   = mean(word_count, na.rm = TRUE),
    median = median(word_count, na.rm = TRUE),
    sd     = sd(word_count, na.rm = TRUE),
    min    = min(word_count, na.rm = TRUE),
    max    = max(word_count, na.rm = TRUE),
    q25    = quantile(word_count, 0.25, na.rm = TRUE),
    q75    = quantile(word_count, 0.75, na.rm = TRUE),
    .groups = "drop"
  )
print(as.data.frame(by_party))

# By chamber ----
cat("\n--- By chamber ---\n\n")
by_chamber <- df |>
  group_by(chamber) |>
  summarise(
    n      = n(),
    mean   = mean(word_count, na.rm = TRUE),
    median = median(word_count, na.rm = TRUE),
    sd     = sd(word_count, na.rm = TRUE),
    min    = min(word_count, na.rm = TRUE),
    max    = max(word_count, na.rm = TRUE),
    q25    = quantile(word_count, 0.25, na.rm = TRUE),
    q75    = quantile(word_count, 0.75, na.rm = TRUE),
    .groups = "drop"
  )
print(as.data.frame(by_chamber))

# By party x chamber ----
cat("\n--- By party x chamber ---\n\n")
by_both <- df |>
  group_by(party, chamber) |>
  summarise(
    n      = n(),
    mean   = mean(word_count, na.rm = TRUE),
    median = median(word_count, na.rm = TRUE),
    sd     = sd(word_count, na.rm = TRUE),
    .groups = "drop"
  )
print(as.data.frame(by_both))

# Plots ----
party_colors <- c("D" = "royalblue", "R" = "firebrick3", "I" = "gray50")

p1 <- df |> 
  filter(!word_count > sd(word_count)) |> 
  ggplot(aes(x = word_count)) +
  geom_histogram(bins = 80, fill = "steelblue", color = "white", linewidth = 0.2) +
  geom_vline(aes(xintercept = mean(word_count)), color = "red", linetype = "dashed") +
  geom_vline(aes(xintercept = median(word_count)), color = "orange", linetype = "dashed") +
  annotate("text", x = mean(df$word_count), y = Inf, label = paste0("Mean: ", round(mean(df$word_count))),
           vjust = 2, hjust = -0.1, color = "red", size = 3) +
  annotate("text", x = median(df$word_count), y = Inf, label = paste0("Median: ", round(median(df$word_count))),
           vjust = 3.5, hjust = -0.1, color = "orange", size = 3) +
  labs(x = "Word count", y = "Frequency", title = "Overall distribution of speech lengths") +
  theme_minimal()

p2 <- ggplot(df, aes(x = word_count)) +
  geom_histogram(bins = 80, fill = "steelblue", color = "white", linewidth = 0.2) +
  scale_x_log10() +
  labs(x = "Word count (log scale)", y = "Frequency", title = "Log-scaled distribution") +
  theme_minimal()

p3 <- ggplot(df, aes(x = word_count, fill = party)) +
  geom_histogram(bins = 60, alpha = 0.6, position = "identity", color = "white", linewidth = 0.2) +
  scale_fill_manual(values = party_colors) +
  labs(x = "Word count", y = "Frequency", title = "Distribution by party", fill = "Party") +
  theme_minimal()

p4 <- df |>
  mutate(group = paste(party, chamber, sep = " – ")) |>
  ggplot(aes(x = group, y = word_count, fill = group)) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.3) +
  scale_fill_brewer(palette = "Set2") +
  labs(x = NULL, y = "Word count", title = "Box plot by party & chamber") +
  theme_minimal() +
  theme(legend.position = "none")

# Save combined figure
combined <- gridExtra::grid.arrange(p1, p2, p3, p4, ncol = 2,
                                     top = "Speech Length — US 113th Congress")

ggsave("outputs/speech_length_distribution_113.png", combined,
       width = 14, height = 10, dpi = 200)
cat("\nFigure saved to outputs/speech_length_distribution_113.png\n")

# Topics -----
topic_info <- read_csv("outputs/topic_info.csv")|> 
  select(-...1)

out_topics <- c(0, 6, 7, 8, 9, 11, 12, 15, 16, 17, 18, 23, 25, 26, 28, 29, 31, 34, 
                35, 38, 40, 43, 46, 48, 49, 50, 51, 52, 54, 55, 56, 59, 60, 62,
                63, 65, 66, 68, 69, 70, 71, 72, 73, 74, 75, 76) #procedural or irrelevant interventions

df_filtered <- df |> 
  filter(!topic %in% out_topics)

# =========================================================
# FILTERED SPEECHES — Substantive interventions only
# =========================================================

# Section 1: Basic descriptives on filtered speeches ----
cat("\n=========================================================\n")
cat("  Filtered Speeches — Substantive Interventions\n")
cat("=========================================================\n\n")

cat(sprintf("Total speeches:    %d\n", nrow(df)))
cat(sprintf("Retained:          %d (%.1f%%)\n", nrow(df_filtered), 100 * nrow(df_filtered) / nrow(df)))
cat(sprintf("Removed:           %d (%.1f%%)\n", nrow(df) - nrow(df_filtered), 100 * (nrow(df) - nrow(df_filtered)) / nrow(df)))

cat("\n--- Speech length summary (filtered) ---\n\n")
filt_overall <- df_filtered |>
  summarise(
    n      = n(),
    mean   = mean(word_count, na.rm = TRUE),
    median = median(word_count, na.rm = TRUE),
    sd     = sd(word_count, na.rm = TRUE),
    min    = min(word_count, na.rm = TRUE),
    max    = max(word_count, na.rm = TRUE)
  )
print(as.data.frame(filt_overall))

cat("\n--- By party (filtered) ---\n\n")
filt_by_party <- df_filtered |>
  group_by(party) |>
  summarise(n = n(), mean = mean(word_count, na.rm = TRUE), .groups = "drop")
print(as.data.frame(filt_by_party))

cat("\n--- By chamber (filtered) ---\n\n")
filt_by_chamber <- df_filtered |>
  group_by(chamber) |>
  summarise(n = n(), mean = mean(word_count, na.rm = TRUE), .groups = "drop")
print(as.data.frame(filt_by_chamber))

# Section 2: Topic distribution ----
cat("\n=========================================================\n")
cat("  Topic Distribution\n")
cat("=========================================================\n\n")

df_filtered <- df_filtered |>
  left_join(topic_info |> select(Topic, Name), by = c("topic" = "Topic"))

topic_freq <- df_filtered |>
  count(topic, Name, sort = TRUE) |>
  mutate(pct = 100 * n / sum(n))

cat("--- Topic frequency table ---\n\n")
print(as.data.frame(topic_freq), row.names = FALSE)

# Top 10 topics bar chart
top10 <- topic_freq |> slice_head(n = 10)

p_topic <- ggplot(top10, aes(x = reorder(Name, n), y = n)) +
  geom_col(fill = "steelblue") +
  geom_text(aes(label = sprintf("%d (%.1f%%)", n, pct)), hjust = -0.1, size = 3) +
  coord_flip() +
  labs(x = NULL, y = "Number of speeches",
       title = "Top 10 Topics — Substantive Interventions") +
  theme_minimal() +
  theme(plot.margin = margin(5, 40, 5, 5))

ggsave("outputs/topic_distribution_113.png", p_topic,
       width = 12, height = 6, dpi = 200)
cat("\nFigure saved to outputs/topic_distribution_113.png\n")

# Section 3: Topics by party ----
cat("\n=========================================================\n")
cat("  Topic Distribution by Party\n")
cat("=========================================================\n\n")

# Cross-tabulation: counts
topic_party <- df_filtered |>
  count(topic, Name, party, sort = TRUE)

cat("--- Topic x Party counts (top 20 rows) ---\n\n")
print(as.data.frame(topic_party |> slice_head(n = 20)), row.names = FALSE)

# Proportional table: within each party, % of speeches per topic
topic_party_pct <- df_filtered |>
  count(topic, Name, party) |>
  group_by(party) |>
  mutate(pct = 100 * n / sum(n)) |>
  ungroup() |>
  arrange(party, desc(pct))

cat("\n--- Topic proportions within each party (top 20 rows) ---\n\n")
print(as.data.frame(topic_party_pct |> slice_head(n = 20)), row.names = FALSE)

# Grouped bar chart: top 10 topics by party proportion
top10_topics <- topic_freq |> slice_head(n = 10) |> pull(topic)

p_topic_party <- topic_party_pct |>
  filter(topic %in% top10_topics) |>
  ggplot(aes(x = reorder(Name, pct), y = pct, fill = party)) +
  geom_col(position = "dodge") +
  scale_fill_manual(values = party_colors) +
  coord_flip() +
  labs(x = NULL, y = "% of party speeches",
       title = "Top 10 Topics by Party — Proportion of Each Party's Speeches",
       fill = "Party") +
  theme_minimal()


# Heatmap: topic x party
p_heatmap <- topic_party_pct |>
  filter(topic %in% top10_topics) |>
  ggplot(aes(x = party, y = reorder(Name, pct), fill = pct)) +
  geom_tile(color = "white") +
  geom_text(aes(label = sprintf("%.1f%%", pct)), size = 3) +
  scale_fill_gradient(low = "white", high = "steelblue") +
  labs(x = "Party", y = NULL, fill = "% speeches",
       title = "Topic × Party Heatmap — Top 10 Topics") +
  theme_minimal()


# Party share within each topic (topic "acaparation") ----
topic_party_share <- df_filtered |>
  count(topic, Name, party) |>
  group_by(topic, Name) |>
  mutate(share = 100 * n / sum(n)) |>
  ungroup()

p_acaparation <- topic_party_share |>
  filter(topic %in% top10_topics) |>
  ggplot(aes(x = reorder(Name, topic_freq$n[match(Name, topic_freq$Name)]),
             y = share, fill = party)) +
  geom_col(position = "stack", width = 0.7) +
  geom_text(aes(label = sprintf("%.0f%%", share)),
            position = position_stack(vjust = 0.5), size = 3, color = "white") +
  scale_fill_manual(values = party_colors) +
  coord_flip() +
  labs(x = NULL, y = "% of topic instances",
       title = "Party Share Within Each Topic — Top 10 Topics",
       fill = "Party") +
  theme_minimal()


# Unique speakers per topic ----
speakers_per_topic <- df_filtered |>
  group_by(topic, Name) |>
  summarise(n_speakers = n_distinct(speakerid), .groups = "drop") |>
  arrange(desc(n_speakers))

p_speakers <- speakers_per_topic |>
  filter(topic %in% top10_topics) |>
  ggplot(aes(x = reorder(Name, n_speakers), y = n_speakers)) +
  geom_col(fill = "steelblue", width = 0.7) +
  geom_text(aes(label = n_speakers), hjust = -0.1, size = 3) +
  coord_flip() +
  labs(x = NULL, y = "Unique speakers",
       title = "Number of Unique Speakers per Topic — Top 10 Topics") +
  theme_minimal() +
  theme(plot.margin = margin(5, 30, 5, 5))


# Speaker topic breadth ----
cat("\n=========================================================\n")
cat("  Speaker Topic Breadth\n")
cat("=========================================================\n\n")

speaker_breadth <- df_filtered |>
  group_by(speakerid, party) |>
  summarise(n_topics = n_distinct(topic), .groups = "drop")

cat(sprintf("Mean topics per speaker:   %.1f\n", mean(speaker_breadth$n_topics)))
cat(sprintf("Median topics per speaker: %.0f\n", median(speaker_breadth$n_topics)))

p_breadth <- ggplot(speaker_breadth, aes(x = n_topics)) +
  geom_histogram(binwidth = 1, fill = "steelblue", color = "white", linewidth = 0.2) +
  geom_vline(aes(xintercept = mean(n_topics)), color = "red", linetype = "dashed") +
  geom_vline(aes(xintercept = median(n_topics)), color = "orange", linetype = "dashed") +
  labs(x = "Number of unique topics", y = "Number of speakers",
       title = "Distribution of Topic Breadth per Speaker") +
  theme_minimal()

