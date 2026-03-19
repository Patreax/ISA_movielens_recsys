import numpy as np
import pandas as pd
from loguru import logger
from sklearn.preprocessing import LabelEncoder, StandardScaler

from project1.dataset import GENRE_NAMES, load_items, load_users


def compute_genre_preference(
    ratings: pd.DataFrame, items: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Compute per-user genre preference vector from rated items.

    For each user, we calculate the mean rating given to movies of each genre.
    We only consider genres the user has actually rated and fill the NaN values with 0.
    """
    if items is None:
        items = load_items()
    merged = ratings.merge(items[["item_id"] + GENRE_NAMES], on="item_id")

    records = []
    for genre in GENRE_NAMES:
        genre_ratings = merged[merged[genre] == 1].groupby("user_id")["rating"].mean()
        genre_ratings.name = genre
        records.append(genre_ratings)

    genre_pref = pd.concat(records, axis=1).fillna(0.0)
    logger.info(
        f"Computed genre preference for {len(genre_pref)} users across {len(GENRE_NAMES)} genres"
    )
    return genre_pref


def build_user_features(
    ratings: pd.DataFrame,
    users: pd.DataFrame | None = None,
    items: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, StandardScaler, dict]:
    """Build a combined user feature matrix (demographics + genre preference).

    Demographics: age_code (categorical int), gender (M/F), occupation_code (int 0-20).
    Returns (feature_df, scaler, encoders) prepared for later use.
    """
    if users is None:
        users = load_users()
    if items is None:
        items = load_items()

    genre_pref = compute_genre_preference(ratings, items)

    user_feats = users.set_index("user_id").copy()

    le_gender = LabelEncoder()
    user_feats["gender_enc"] = le_gender.fit_transform(user_feats["gender"])

    numeric_cols = ["age_code", "gender_enc", "occupation_code"]
    demo_features = user_feats[numeric_cols]

    combined = demo_features.join(genre_pref, how="left").fillna(0.0)

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(combined)
    combined_scaled = pd.DataFrame(scaled_values, index=combined.index, columns=combined.columns)

    encoders = {"gender": le_gender}
    logger.info(f"Built user feature matrix: {combined_scaled.shape}")
    return combined_scaled, scaler, encoders


def encode_cold_user(
    age_code: int,
    gender: str,
    occupation_code: int,
    preferred_genres: list[str],
    scaler: StandardScaler,
    encoders: dict,
) -> np.ndarray:
    """Encode a cold-start user profile into the same feature space as existing users.

    age_code: one of 1, 18, 25, 35, 45, 50, 56
    occupation_code: integer 0-20
    """
    gender_enc = encoders["gender"].transform([gender])[0]

    genre_vector = [5.0 if g in preferred_genres else 0.0 for g in GENRE_NAMES]

    raw = [age_code, gender_enc, occupation_code] + genre_vector
    scaled = scaler.transform([raw])
    return scaled[0]
