import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Enable interactive mode
plt.ion()

# Set style
plt.style.use("seaborn-v0_8-darkgrid")
sns.set_palette("husl")

# Read CSV
csv_files = glob.glob("sentiment_results.csv/*.csv")
if not csv_files:
    csv_files = glob.glob("sentiment_results.csv/part-*.csv")

df = pd.read_csv(
    csv_files[0], quoting=1, escapechar="\\", on_bad_lines="skip", engine="python"
)

sentiment_counts = df["sentiment_label"].value_counts()

# ============================================================================
# FIGURE 1: MAIN OVERVIEW (5 plots)
# ============================================================================
fig1 = plt.figure(figsize=(18, 10))
fig1.suptitle(
    "Mental Health Social Media Sentiment Analysis - Overview",
    fontsize=16,
    fontweight="bold",
)

# 1. Sentiment Distribution Bar Chart
ax1 = plt.subplot(2, 3, 1)
colors = {"positive": "#2ecc71", "neutral": "#95a5a6", "negative": "#e74c3c"}
bars = ax1.bar(
    sentiment_counts.index,
    sentiment_counts.values,
    color=[colors.get(x, "gray") for x in sentiment_counts.index],
)
ax1.set_title("Sentiment Distribution", fontsize=12, fontweight="bold")
ax1.set_ylabel("Count", fontsize=11)
ax1.set_xlabel("Sentiment Label", fontsize=11)
for bar in bars:
    height = bar.get_height()
    ax1.text(
        bar.get_x() + bar.get_width() / 2.0,
        height,
        f"{int(height):,}\n({height / len(df) * 100:.1f}%)",
        ha="center",
        va="bottom",
        fontsize=9,
    )

# 2. Sentiment Score Distribution
ax2 = plt.subplot(2, 3, 2)
ax2.hist(df["sentiment_score"], bins=50, color="skyblue", edgecolor="black", alpha=0.7)
ax2.axvline(
    x=0.05, color="green", linestyle="--", linewidth=2, label="Positive (≥0.05)"
)
ax2.axvline(
    x=-0.05, color="red", linestyle="--", linewidth=2, label="Negative (≤-0.05)"
)
mean_score = df["sentiment_score"].mean()
ax2.axvline(
    x=mean_score,
    color="orange",
    linestyle="-",
    linewidth=2,
    label=f"Mean ({mean_score:.3f})",
)
ax2.set_title("Sentiment Score Distribution", fontsize=12, fontweight="bold")
ax2.set_xlabel("Sentiment Score", fontsize=11)
ax2.set_ylabel("Frequency", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

# 3. Mental Health Keywords Impact
ax3 = plt.subplot(2, 3, 3)
mh_sentiment = (
    pd.crosstab(
        df["has_mental_health_keyword"], df["sentiment_label"], normalize="index"
    )
    * 100
)
mh_sentiment.plot(
    kind="bar", ax=ax3, color=[colors.get(x, "gray") for x in mh_sentiment.columns]
)
ax3.set_title("Sentiment by MH Keyword Presence", fontsize=12, fontweight="bold")
ax3.set_xlabel("Has MH Keyword", fontsize=11)
ax3.set_ylabel("Percentage (%)", fontsize=11)
mh_categories = mh_sentiment.index.tolist()
label_map = {0: "No", 1: "Yes"}
ax3.set_xticklabels([label_map.get(cat, str(cat)) for cat in mh_categories], rotation=0)
ax3.legend(title="Sentiment", fontsize=9)

# 4. Hourly Posting Activity
ax4 = plt.subplot(2, 3, 4)
df_hour = df.copy()
df_hour["hour"] = pd.to_numeric(df_hour["hour"], errors="coerce")
hourly_posts = df_hour.groupby("hour").size()
ax4.plot(
    hourly_posts.index,
    hourly_posts.values,
    marker="o",
    linewidth=2,
    markersize=6,
    color="blue",
)
ax4.fill_between(hourly_posts.index, hourly_posts.values, alpha=0.3)
ax4.set_title("Posting Activity by Hour", fontsize=12, fontweight="bold")
ax4.set_xlabel("Hour of Day", fontsize=11)
ax4.set_ylabel("Number of Posts", fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(0, 24, 2))

# 5. Average Sentiment by Hour
ax5 = plt.subplot(2, 3, 5)
df_hour = df.copy()
df_hour["hour"] = pd.to_numeric(df_hour["hour"], errors="coerce")
hourly_sentiment = df_hour.groupby("hour")["sentiment_score"].mean()
ax5.plot(
    hourly_sentiment.index,
    hourly_sentiment.values,
    marker="o",
    linewidth=2,
    markersize=6,
    color="purple",
)
ax5.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.3)
ax5.axhline(y=0.05, color="green", linestyle="--", linewidth=1, alpha=0.5)
ax5.axhline(y=-0.05, color="red", linestyle="--", linewidth=1, alpha=0.5)
ax5.fill_between(
    hourly_sentiment.index, hourly_sentiment.values, alpha=0.3, color="purple"
)
ax5.set_title("Average Sentiment by Hour", fontsize=12, fontweight="bold")
ax5.set_xlabel("Hour of Day", fontsize=11)
ax5.set_ylabel("Avg Sentiment Score", fontsize=11)
ax5.grid(True, alpha=0.3)
ax5.set_xticks(range(0, 24, 2))

plt.tight_layout()

# ============================================================================
# FIGURE 2: ENGAGEMENT ANALYSIS (3 plots)
# ============================================================================
fig2 = plt.figure(figsize=(16, 10))
fig2.suptitle("Engagement and User Behavior Analysis", fontsize=16, fontweight="bold")

# 1. Box Plot - Sentiment by Label
ax1 = plt.subplot(2, 2, 1)
sentiment_labels = df["sentiment_label"].unique()
box_data = [
    df[df["sentiment_label"] == label]["sentiment_score"].values
    for label in sentiment_labels
]
bp = ax1.boxplot(box_data, labels=sentiment_labels, patch_artist=True, showmeans=True)
for i, (patch, label) in enumerate(zip(bp["boxes"], sentiment_labels)):
    patch.set_facecolor(colors.get(label, "gray"))
    patch.set_alpha(0.6)
ax1.set_title("Sentiment Score Distribution by Label", fontsize=12, fontweight="bold")
ax1.set_ylabel("Sentiment Score", fontsize=11)
ax1.grid(True, alpha=0.3, axis="y")

# 2. Retweets vs Original Posts
ax2 = plt.subplot(2, 2, 2)
retweet_sentiment = (
    pd.crosstab(df["is_retweet"], df["sentiment_label"], normalize="index") * 100
)
retweet_sentiment.plot(
    kind="bar", ax=ax2, color=[colors.get(x, "gray") for x in retweet_sentiment.columns]
)
ax2.set_title("Sentiment: Retweets vs Original Posts", fontsize=12, fontweight="bold")
ax2.set_xlabel("Post Type", fontsize=11)
ax2.set_ylabel("Percentage (%)", fontsize=11)
retweet_categories = retweet_sentiment.index.tolist()
retweet_label_map = {0: "Original", 1: "Retweets"}
ax2.set_xticklabels(
    [retweet_label_map.get(cat, str(cat)) for cat in retweet_categories], rotation=0
)
ax2.legend(title="Sentiment", fontsize=9)


# 3. Followers Distribution (log scale)
ax3 = plt.subplot(2, 2, 4)
followers_filtered = df[df["followers"] > 0]["followers"]
ax3.hist(
    np.log10(followers_filtered),
    bins=40,
    color="coral",
    edgecolor="black",
    alpha=0.7,
)
ax3.set_title("Followers Distribution (log scale)", fontsize=12, fontweight="bold")
ax3.set_xlabel("log10(Followers)", fontsize=11)
ax3.set_ylabel("Frequency", fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# ============================================================================
# FIGURE 3: CORRELATION HEATMAP
# ============================================================================
fig3, ax = plt.subplots(figsize=(10, 8))
fig3.suptitle("Feature Correlation Matrix", fontsize=16, fontweight="bold")

corr_columns = [
    "followers",
    "friends",
    "favourites",
    "statuses",
    "retweets",
    "sentiment_score",
    "engagement_ratio",
]
corr_columns = [col for col in corr_columns if col in df.columns]
corr_df = df[corr_columns].replace([np.inf, -np.inf], np.nan).dropna()

corr_matrix = corr_df.corr()
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    square=True,
    ax=ax,
    cbar_kws={"shrink": 0.8},
    linewidths=1,
    linecolor="white",
)
ax.set_title("Pearson Correlation Coefficients", fontsize=12, pad=20)

plt.tight_layout()

plt.show(block=True)
