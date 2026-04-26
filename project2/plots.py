"""EDA plot functions for the Steam dataset."""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

def set_style():
    """Apply a consistent, clean style to all plots."""
    sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
    plt.rcParams.update({
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_recommend_distribution(df_reviews: pd.DataFrame, ax=None):
    """Pie + bar showing positive vs negative review split."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    counts = df_reviews["recommend"].value_counts()
    labels = {True: "Recommended", False: "Not Recommended"}
    colors = ["#4C9BE8", "#E8654C"]
    bars = ax.bar(
        [labels[k] for k in counts.index],
        counts.values,
        color=colors[: len(counts)],
        edgecolor="white",
        linewidth=1.5,
        width=0.5,
    )
    for bar, val in zip(bars, counts.values):
        pct = val / counts.sum() * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{val:,}\n({pct:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Review Sentiment Distribution")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return ax


def plot_reviews_per_user(df_reviews: pd.DataFrame, ax=None):
    """Log-scale histogram of reviews per user."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    per_user = df_reviews.groupby("user_id").size()
    ax.hist(per_user, bins=50, color="#4C9BE8", edgecolor="white", alpha=0.85, log=True)
    ax.axvline(per_user.mean(), color="crimson", linestyle="--", linewidth=1.5,
               label=f"Mean = {per_user.mean():.1f}")
    ax.axvline(per_user.median(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {per_user.median():.0f}")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Number of Users (log scale)")
    ax.set_title("Reviews per User")
    ax.legend()
    return ax


def plot_items_per_user(df_items: pd.DataFrame, ax=None):
    """Log-scale histogram of games owned per user."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    per_user = df_items.groupby("user_id")["item_id"].nunique()
    ax.hist(per_user, bins=60, color="#6FCF97", edgecolor="white", alpha=0.85, log=True)
    ax.axvline(per_user.mean(), color="crimson", linestyle="--", linewidth=1.5,
               label=f"Mean = {per_user.mean():.1f}")
    ax.axvline(per_user.median(), color="orange", linestyle="--", linewidth=1.5,
               label=f"Median = {per_user.median():.0f}")
    ax.set_xlabel("Number of Games Owned")
    ax.set_ylabel("Number of Users (log scale)")
    ax.set_title("Games Owned per User")
    ax.legend()
    return ax


def plot_reviews_per_item(df_reviews: pd.DataFrame, ax=None):
    """Log-scale histogram of review count per item."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    per_item = df_reviews.groupby("item_id").size()
    ax.hist(per_item, bins=60, color="#F2994A", edgecolor="white", alpha=0.85, log=True)
    ax.axvline(per_item.mean(), color="crimson", linestyle="--", linewidth=1.5,
               label=f"Mean = {per_item.mean():.1f}")
    ax.axvline(per_item.median(), color="navy", linestyle="--", linewidth=1.5,
               label=f"Median = {per_item.median():.0f}")
    ax.set_xlabel("Number of Reviews")
    ax.set_ylabel("Number of Games (log scale)")
    ax.set_title("Reviews per Game")
    ax.legend()
    return ax


def plot_top_reviewed_games(df_reviews: pd.DataFrame, df_games: pd.DataFrame,
                             n: int = 15, ax=None):
    """Horizontal bar chart of the N most-reviewed games."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    top = df_reviews.groupby("item_id").size().nlargest(n).reset_index(name="count")
    name_map = df_games.set_index("id")["app_name"].to_dict()
    top["name"] = top["item_id"].map(name_map).fillna(top["item_id"])
    top = top.sort_values("count")
    colors = sns.color_palette("Blues_r", len(top))
    ax.barh(top["name"], top["count"], color=colors, edgecolor="white")
    ax.set_xlabel("Number of Reviews")
    ax.set_title(f"Top {n} Most-Reviewed Games")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return ax


def plot_top_played_games(df_items: pd.DataFrame, df_games: pd.DataFrame,
                           n: int = 15, ax=None):
    """Horizontal bar chart of N games with the most total playtime."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))
    top = (
        df_items.groupby("item_id")["playtime_forever"]
        .sum()
        .nlargest(n)
        .reset_index(name="total_playtime")
    )
    name_map = df_games.set_index("id")["app_name"].to_dict()
    top["name"] = top["item_id"].map(name_map).fillna(top["item_id"])
    top = top.sort_values("total_playtime")
    top["total_hours"] = top["total_playtime"] / 60
    colors = sns.color_palette("Greens_r", len(top))
    ax.barh(top["name"], top["total_hours"], color=colors, edgecolor="white")
    ax.set_xlabel("Total Playtime (hours)")
    ax.set_title(f"Top {n} Games by Total Playtime")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return ax


def plot_playtime_distribution(df_items: pd.DataFrame, ax=None):
    """Log-scale histogram of playtime_forever (minutes) for played items only."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    played = df_items[df_items["playtime_forever"] > 0]["playtime_forever"]
    log_vals = np.log10(played + 1)
    ax.hist(log_vals, bins=60, color="#BB6BD9", edgecolor="white", alpha=0.85)
    ticks = [0, 1, 2, 3, 4, 5]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{10**t:,.0f}" for t in ticks])
    ax.set_xlabel("Playtime (minutes, log₁₀ scale)")
    ax.set_ylabel("Number of User-Game Pairs")
    ax.set_title("Playtime Distribution (played games only)")
    median_val = played.median()
    ax.axvline(np.log10(median_val + 1), color="crimson", linestyle="--", linewidth=1.5,
               label=f"Median = {median_val:.0f} min")
    ax.legend()
    return ax


def plot_playtime_vs_recommend(df_reviews: pd.DataFrame, df_items: pd.DataFrame, ax=None):
    """Box plot comparing playtime for recommended vs not-recommended games."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    avg_play = df_items.groupby(["user_id", "item_id"])["playtime_forever"].sum().reset_index()
    merged = df_reviews.merge(avg_play, on=["user_id", "item_id"], how="inner")
    merged = merged[merged["playtime_forever"] > 0]
    merged["log_playtime"] = np.log10(merged["playtime_forever"] + 1)
    merged["sentiment"] = merged["recommend"].map({True: "Recommended", False: "Not Recommended"})
    sns.boxplot(
        data=merged, x="sentiment", y="log_playtime",
        palette={"Recommended": "#4C9BE8", "Not Recommended": "#E8654C"},
        ax=ax, width=0.4, linewidth=1.2,
    )
    ticks = [0, 1, 2, 3, 4, 5]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{10**t:,.0f}" for t in ticks])
    ax.set_xlabel("")
    ax.set_ylabel("Playtime (minutes, log₁₀)")
    ax.set_title("Playtime by Review Sentiment")
    return ax


def plot_genre_distribution(df_games: pd.DataFrame, n: int = 20, ax=None):
    """Horizontal bar chart of the most frequent genres across all games."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    genre_series = df_games["genres"].dropna().explode()
    top_genres = genre_series.value_counts().head(n).sort_values()
    colors = sns.color_palette("viridis", len(top_genres))
    ax.barh(top_genres.index, top_genres.values, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Games")
    ax.set_title(f"Top {n} Genres in Steam Catalogue")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return ax


def plot_top_tags(df_games: pd.DataFrame, n: int = 20, ax=None):
    """Horizontal bar chart of top user-assigned tags."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    tag_series = df_games["tags"].dropna().explode()
    top_tags = tag_series.value_counts().head(n).sort_values()
    colors = sns.color_palette("magma", len(top_tags))
    ax.barh(top_tags.index, top_tags.values, color=colors, edgecolor="white")
    ax.set_xlabel("Number of Games")
    ax.set_title(f"Top {n} User Tags")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return ax


def plot_price_distribution(df_games: pd.DataFrame, ax=None):
    """Histogram of game prices (paid games only, capped at 99th percentile)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    prices = df_games["price_numeric"].dropna()
    prices = prices[prices > 0]
    cap = prices.quantile(0.99)
    prices_capped = prices[prices <= cap]
    ax.hist(prices_capped, bins=40, color="#F2C94C", edgecolor="white", alpha=0.9)
    ax.axvline(prices_capped.median(), color="crimson", linestyle="--", linewidth=1.5,
               label=f"Median = ${prices_capped.median():.2f}")
    ax.axvline(prices_capped.mean(), color="navy", linestyle="--", linewidth=1.5,
               label=f"Mean = ${prices_capped.mean():.2f}")
    ax.set_xlabel("Price (USD)")
    ax.set_ylabel("Number of Games")
    ax.set_title("Price Distribution (paid games, 99th pct cap)")
    ax.legend()
    return ax


def plot_release_year_distribution(df_games: pd.DataFrame, ax=None):
    """Bar chart of game releases by year."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    years = pd.to_datetime(df_games["release_date"], errors="coerce").dt.year.dropna()
    year_counts = years.value_counts().sort_index()
    year_counts = year_counts[(year_counts.index >= 2000) & (year_counts.index <= 2020)]
    ax.bar(year_counts.index.astype(int), year_counts.values,
           color="#56CCF2", edgecolor="white", alpha=0.9)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Games Released")
    ax.set_title("Game Releases per Year")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_early_access_distribution(df_games: pd.DataFrame, ax=None):
    """Bar chart showing Early Access vs full releases."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 4))
    counts = df_games["early_access"].value_counts()
    labels = {True: "Early Access", False: "Full Release"}
    colors = ["#EB5757", "#27AE60"]
    bars = ax.bar(
        [labels.get(k, str(k)) for k in counts.index],
        counts.values,
        color=colors[: len(counts)],
        edgecolor="white",
        width=0.5,
    )
    for bar, val in zip(bars, counts.values):
        pct = val / counts.sum() * 100
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + counts.max() * 0.01,
            f"{val:,}\n({pct:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )
    ax.set_ylabel("Number of Games")
    ax.set_title("Early Access vs Full Release")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    return ax


def plot_sparsity(n_users: int, n_items: int, n_interactions: int,
                  label: str = "Interaction Matrix", ax=None):
    """Horizontal stacked bar visualising matrix density vs sparsity."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 2))
    total = n_users * n_items
    density = n_interactions / total * 100
    sparsity = 100 - density
    ax.barh([""], [density], color="#27AE60", label=f"Filled ({density:.4f}%)")
    ax.barh([""], [sparsity], left=[density], color="#E0E0E0", label=f"Empty ({sparsity:.4f}%)")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentage (%)")
    ax.set_title(
        f"{label} — {n_users:,} users × {n_items:,} items\n"
        f"{n_interactions:,} interactions | sparsity = {sparsity:.4f}%"
    )
    ax.legend(loc="lower right")
    return ax


def plot_hours_distribution(df_steam_reviews: pd.DataFrame, ax=None):
    """Log-scale histogram of hours played at review time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    hours = df_steam_reviews["hours"].dropna()
    hours = hours[hours > 0]
    log_h = np.log10(hours + 1)
    ax.hist(log_h, bins=60, color="#9B51E0", edgecolor="white", alpha=0.85)
    ticks = [0, 1, 2, 3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{10**t:.0f}" for t in ticks])
    ax.axvline(log_h.median(), color="crimson", linestyle="--", linewidth=1.5,
               label=f"Median = {hours.median():.1f} h")
    ax.set_xlabel("Hours Played (log₁₀ scale)")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Hours Played at Time of Review")
    ax.legend()
    return ax


def plot_reviews_over_time(df_steam_reviews: pd.DataFrame, ax=None):
    """Line chart of review volume over time (by month)."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    dates = pd.to_datetime(df_steam_reviews["date"], errors="coerce").dropna()
    monthly = dates.dt.to_period("M").value_counts().sort_index()
    monthly.index = monthly.index.to_timestamp()
    ax.plot(monthly.index, monthly.values, color="#2D9CDB", linewidth=1.5)
    ax.fill_between(monthly.index, monthly.values, alpha=0.15, color="#2D9CDB")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Reviews")
    ax.set_title("Review Volume Over Time")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.tick_params(axis="x", rotation=30)
    return ax
