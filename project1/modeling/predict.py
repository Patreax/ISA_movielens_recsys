import numpy as np
import pandas as pd
from loguru import logger
from surprise import SVD

from project1.dataset import GENRE_NAMES, load_items
from project1.features import compute_genre_preference, encode_cold_user


def get_top_n_for_user(
    algo: SVD,
    user_id: int,
    all_item_ids: list[int],
    rated_item_ids: set[int],
    n: int = 10,
    genre_pref: pd.Series | None = None,
    items: pd.DataFrame | None = None,
    genre_boost: float = 0.0,
) -> list[tuple[int, float]]:
    """Get top-N recommendations for a warm user from SVD predictions.

    Optionally applies a genre preference boost to re-rank items.
    Returns list of (item_id, predicted_rating).
    """
    uid = str(user_id)
    candidates = [iid for iid in all_item_ids if iid not in rated_item_ids]

    predictions = []
    for iid in candidates:
        pred = algo.predict(uid, str(iid))
        score = pred.est

        if genre_boost > 0 and genre_pref is not None and items is not None:
            item_row = items[items["item_id"] == iid]
            if not item_row.empty:
                item_genres = item_row[GENRE_NAMES].values[0]
                pref_vec = genre_pref.reindex(GENRE_NAMES, fill_value=0).values
                overlap = np.dot(item_genres, pref_vec) / (np.sum(item_genres) + 1e-9)
                score += genre_boost * overlap

        predictions.append((iid, score))

    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:n]


def recommend_warm_users(
    algo: SVD,
    ratings: pd.DataFrame,
    items: pd.DataFrame | None = None,
    n: int = 10,
    genre_boost: float = 0.0,
) -> dict[int, list[tuple[int, float]]]:
    """Generate top-N recommendations for all warm users."""
    if items is None:
        items = load_items()

    all_item_ids = items["item_id"].tolist()
    user_rated = ratings.groupby("user_id")["item_id"].apply(set).to_dict()

    genre_pref_df = None
    if genre_boost > 0:
        genre_pref_df = compute_genre_preference(ratings, items)

    recommendations = {}
    for user_id, rated_ids in user_rated.items():
        gp = (
            genre_pref_df.loc[user_id]
            if genre_pref_df is not None and user_id in genre_pref_df.index
            else None
        )
        top_n = get_top_n_for_user(
            algo,
            user_id,
            all_item_ids,
            rated_ids,
            n,
            genre_pref=gp,
            items=items if genre_boost > 0 else None,
            genre_boost=genre_boost,
        )
        recommendations[user_id] = top_n

    logger.info(f"Generated top-{n} recommendations for {len(recommendations)} warm users")
    return recommendations


def recommend_cold_user(
    age_code: int,
    gender: str,
    occupation_code: int,
    preferred_genres: list[str],
    cold_start_model: dict,
    ratings: pd.DataFrame,
    items: pd.DataFrame | None = None,
    svd_algo: SVD | None = None,
    n: int = 10,
) -> list[tuple[int, str, float]]:
    """Recommend movies for a cold-start user using the similar-user strategy.

    1. Encode the cold user into the feature space.
    2. Find the nearest cluster via KMeans.
    3. Identify similar users in that cluster.
    4. Aggregate their highly-rated movies.

    Returns list of (item_id, title, score).
    """
    if items is None:
        items = load_items()

    cold_vec = encode_cold_user(
        age_code,
        gender,
        occupation_code,
        preferred_genres,
        cold_start_model["scaler"],
        cold_start_model["encoders"],
    )

    cluster = cold_start_model["kmeans"].predict([cold_vec])[0]
    user_features = cold_start_model["user_features"]
    cluster_users = user_features[user_features["cluster"] == cluster].index.tolist()

    if not cluster_users:
        logger.warning("No users in cluster; falling back to popular movies in preferred genres")
        return _fallback_genre_popular(preferred_genres, ratings, items, n)

    cluster_ratings = ratings[ratings["user_id"].isin(cluster_users)]
    high_ratings = cluster_ratings[cluster_ratings["rating"] >= 4]

    if high_ratings.empty:
        high_ratings = cluster_ratings

    movie_scores = high_ratings.groupby("item_id").agg(
        avg_rating=("rating", "mean"),
        count=("rating", "count"),
    )

    if preferred_genres:
        genre_mask = items[preferred_genres].max(axis=1) == 1
        genre_items = set(items[genre_mask]["item_id"])
        in_genre = movie_scores.index.isin(genre_items)
        movie_scores.loc[in_genre, "avg_rating"] += 0.5

    movie_scores = movie_scores.sort_values(
        ["avg_rating", "count"], ascending=[False, False]
    ).head(n)

    result = []
    for item_id, row in movie_scores.iterrows():
        title = items.loc[items["item_id"] == item_id, "title"].values
        title_str = title[0] if len(title) > 0 else f"Movie {item_id}"
        result.append((item_id, title_str, row["avg_rating"]))

    logger.info(
        f"Cold-start recommendation: cluster={cluster}, "
        f"{len(cluster_users)} similar users, {len(result)} movies"
    )
    return result


def _fallback_genre_popular(
    preferred_genres: list[str],
    ratings: pd.DataFrame,
    items: pd.DataFrame,
    n: int = 10,
) -> list[tuple[int, str, float]]:
    """Fallback: most popular movies in the user's preferred genres."""
    if preferred_genres:
        genre_mask = items[preferred_genres].max(axis=1) == 1
        candidate_items = items[genre_mask]["item_id"]
    else:
        candidate_items = items["item_id"]

    popular = (
        ratings[ratings["item_id"].isin(candidate_items)]
        .groupby("item_id")
        .agg(avg_rating=("rating", "mean"), count=("rating", "count"))
        .query("count >= 5")
        .sort_values(["avg_rating", "count"], ascending=[False, False])
        .head(n)
    )
    result = []
    for item_id, row in popular.iterrows():
        title = items.loc[items["item_id"] == item_id, "title"].values
        title_str = title[0] if len(title) > 0 else f"Movie {item_id}"
        result.append((item_id, title_str, row["avg_rating"]))
    return result


def precision_recall_at_k(
    predictions: list, k: int = 10, threshold: float = 4.0
) -> tuple[float, float]:
    """Compute Precision@K and Recall@K from Surprise predictions.

    A rating >= threshold is considered relevant.
    """
    user_est_true = {}
    for pred in predictions:
        uid = pred.uid
        if uid not in user_est_true:
            user_est_true[uid] = []
        user_est_true[uid].append((pred.est, pred.r_ui))

    precisions = []
    recalls = []
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_relevant = sum(1 for _, true_r in user_ratings if true_r >= threshold)
        n_relevant_in_k = sum(1 for _, true_r in top_k if true_r >= threshold)

        precisions.append(n_relevant_in_k / k if k > 0 else 0)
        recalls.append(n_relevant_in_k / n_relevant if n_relevant > 0 else 0)

    return float(np.mean(precisions)), float(np.mean(recalls))
