import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from project1.dataset import AGE_MAP, GENRE_NAMES, OCCUPATION_MAP


def set_style():
    """Set a consistent plot style."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams["figure.dpi"] = 120


def plot_rating_distribution(ratings: pd.DataFrame, ax=None):
    """Bar plot of rating value distribution (1-5)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    counts = ratings["rating"].value_counts().sort_index()
    ax.bar(counts.index, counts.values, color=sns.color_palette("muted", 5), edgecolor="black")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Ratings")
    ax.set_xticks([1, 2, 3, 4, 5])
    return ax


def plot_ratings_per_user(ratings: pd.DataFrame, ax=None):
    """Histogram of number of ratings per user."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    per_user = ratings.groupby("user_id").size()
    ax.hist(per_user, bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Users")
    ax.set_title("Ratings per User")
    ax.axvline(per_user.mean(), color="red", linestyle="--", label=f"Mean={per_user.mean():.0f}")
    ax.legend()
    return ax


def plot_ratings_per_item(ratings: pd.DataFrame, ax=None):
    """Histogram of number of ratings per item."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    per_item = ratings.groupby("item_id").size()
    ax.hist(per_item, bins=50, edgecolor="black", alpha=0.7, color="orange")
    ax.set_xlabel("Number of Ratings")
    ax.set_ylabel("Number of Movies")
    ax.set_title("Ratings per Movie")
    ax.axvline(per_item.mean(), color="red", linestyle="--", label=f"Mean={per_item.mean():.0f}")
    ax.legend()
    return ax


def plot_age_distribution(users: pd.DataFrame, ax=None):
    """Bar plot of user age bracket distribution (ML-1M age codes)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    age_order = sorted(AGE_MAP.keys())
    age_labels = [AGE_MAP[k] for k in age_order]
    counts = users["age_code"].value_counts().reindex(age_order, fill_value=0)
    ax.bar(age_labels, counts.values, color="teal", edgecolor="black", alpha=0.8)
    ax.set_xlabel("Age Group")
    ax.set_ylabel("Count")
    ax.set_title("User Age Distribution")
    ax.tick_params(axis="x", rotation=30)
    return ax


def plot_gender_distribution(users: pd.DataFrame, ax=None):
    """Bar plot of gender distribution."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    counts = users["gender"].value_counts()
    ax.bar(counts.index, counts.values, color=["steelblue", "salmon"], edgecolor="black")
    ax.set_xlabel("Gender")
    ax.set_ylabel("Count")
    ax.set_title("Gender Distribution")
    return ax


def plot_occupation_distribution(users: pd.DataFrame, ax=None):
    """Horizontal bar plot of occupation distribution."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))
    occ_counts = users["occupation"].value_counts().sort_values()
    ax.barh(occ_counts.index, occ_counts.values, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Count")
    ax.set_title("Occupation Distribution")
    return ax


def plot_genre_distribution(items: pd.DataFrame, ax=None):
    """Horizontal bar plot of genre frequency across all movies."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    genre_counts = items[GENRE_NAMES].sum().sort_values()
    ax.barh(genre_counts.index, genre_counts.values, color="mediumpurple", edgecolor="black")
    ax.set_xlabel("Number of Movies")
    ax.set_title("Genre Distribution (All Movies)")
    return ax


def plot_genre_in_rated(merged: pd.DataFrame, ax=None):
    """Horizontal bar plot of genre frequency in rated movies."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    genre_counts = merged[GENRE_NAMES].sum().sort_values()
    ax.barh(genre_counts.index, genre_counts.values, color="coral", edgecolor="black")
    ax.set_xlabel("Number of Ratings")
    ax.set_title("Genre Frequency in Rated Movies")
    return ax


def plot_mean_rating_by_genre(merged: pd.DataFrame, ax=None):
    """Bar plot of mean rating per genre."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    means = {}
    for g in GENRE_NAMES:
        subset = merged[merged[g] == 1]
        if len(subset) > 0:
            means[g] = subset["rating"].mean()
    means_s = pd.Series(means).sort_values()
    ax.barh(means_s.index, means_s.values, color="goldenrod", edgecolor="black")
    ax.set_xlabel("Mean Rating")
    ax.set_title("Mean Rating by Genre")
    ax.set_xlim(0, 5)
    return ax


def plot_sparsity_summary(n_users: int, n_items: int, n_ratings: int, ax=None):
    """Visual summary of matrix sparsity."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    total = n_users * n_items
    sparsity = 1 - n_ratings / total
    filled = n_ratings / total
    ax.barh(
        ["Filled", "Empty"],
        [filled * 100, sparsity * 100],
        color=["seagreen", "lightgray"],
        edgecolor="black",
    )
    ax.set_xlabel("Percentage (%)")
    ax.set_title(f"Rating Matrix Sparsity ({sparsity * 100:.2f}% sparse)")
    for i, v in enumerate([filled * 100, sparsity * 100]):
        ax.text(v + 0.5, i, f"{v:.2f}%", va="center")
    return ax
