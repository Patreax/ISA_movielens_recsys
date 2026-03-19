from pathlib import Path

import pandas as pd
from loguru import logger
from surprise import Dataset as SurpriseDataset
from surprise import Reader

from project1.config import RAW_DATA_DIR

ML_1M_DIR = RAW_DATA_DIR / "ml-1m"

GENRE_NAMES = [
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

AGE_MAP = {
    1: "Under 18",
    18: "18-24",
    25: "25-34",
    35: "35-44",
    45: "45-49",
    50: "50-55",
    56: "56+",
}

OCCUPATION_MAP = {
    0: "other",
    1: "academic/educator",
    2: "artist",
    3: "clerical/admin",
    4: "college/grad student",
    5: "customer service",
    6: "doctor/health care",
    7: "executive/managerial",
    8: "farmer",
    9: "homemaker",
    10: "K-12 student",
    11: "lawyer",
    12: "programmer",
    13: "retired",
    14: "sales/marketing",
    15: "scientist",
    16: "self-employed",
    17: "technician/engineer",
    18: "tradesman/craftsman",
    19: "unemployed",
    20: "writer",
}

SURPRISE_READER_NAME = "ml-1m"

def load_ratings(path: Path | None = None) -> pd.DataFrame:
    """Load ratings from ratings.dat (UserID::MovieID::Rating::Timestamp)."""
    if path is None:
        path = ML_1M_DIR / "ratings.dat"
    df = pd.read_csv(
        path,
        sep="::",
        names=["user_id", "item_id", "rating", "timestamp"],
        engine="python",
        encoding="latin-1",
    )
    logger.info(f"Loaded {len(df)} ratings from {path.name}")
    return df


def load_users(path: Path | None = None) -> pd.DataFrame:
    """Load user demographics from users.dat (UserID::Gender::Age::Occupation::Zip-code).

    Age is a categorical code mapped to a label. Occupation is also a numeric code mapped to a label.
    """
    if path is None:
        path = ML_1M_DIR / "users.dat"
    df = pd.read_csv(
        path,
        sep="::",
        names=["user_id", "gender", "age_code", "occupation_code", "zip_code"],
        engine="python",
        encoding="latin-1",
    )
    df["age_label"] = df["age_code"].map(AGE_MAP)
    df["occupation"] = df["occupation_code"].map(OCCUPATION_MAP)
    logger.info(f"Loaded {len(df)} users from {path.name}")
    return df


def load_items(path: Path | None = None) -> pd.DataFrame:
    """Load movie metadata from movies.dat (MovieID::Title::Genres).

    Genres are pipe-separated strings expanded into binary columns.
    """
    if path is None:
        path = ML_1M_DIR / "movies.dat"
    df = pd.read_csv(
        path,
        sep="::",
        names=["item_id", "title", "genres_str"],
        engine="python",
        encoding="latin-1",
    )
    for genre in GENRE_NAMES:
        df[genre] = df["genres_str"].apply(lambda g, gn=genre: int(gn in g.split("|")))
    logger.info(f"Loaded {len(df)} items from {path.name}")
    return df


def load_merged(
    ratings: pd.DataFrame | None = None,
    users: pd.DataFrame | None = None,
    items: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge ratings with user demographics and item metadata."""
    if ratings is None:
        ratings = load_ratings()
    if users is None:
        users = load_users()
    if items is None:
        items = load_items()
    merged = ratings.merge(users, on="user_id").merge(items, on="item_id")
    logger.info(f"Merged dataset: {merged.shape[0]} rows, {merged.shape[1]} columns")
    return merged


def ratings_to_surprise(ratings: pd.DataFrame) -> SurpriseDataset:
    """Convert a ratings DataFrame to a Surprise Dataset."""
    reader = Reader(name=SURPRISE_READER_NAME)
    data = SurpriseDataset.load_from_df(
        ratings[["user_id", "item_id", "rating"]].astype({"user_id": str, "item_id": str}),
        reader,
    )
    return data


def train_test_split_df(
    ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ratings into train/test DataFrames using stratified sampling per user."""
    test_dfs = []
    train_dfs = []
    for _, user_ratings in ratings.groupby("user_id"):
        test_sample = user_ratings.sample(frac=test_size, random_state=random_state)
        train_sample = user_ratings.drop(test_sample.index)
        test_dfs.append(test_sample)
        train_dfs.append(train_sample)
    train_df = pd.concat(train_dfs).reset_index(drop=True)
    test_df = pd.concat(test_dfs).reset_index(drop=True)
    logger.info(f"Train/test split: train={len(train_df)}, test={len(test_df)}")
    return train_df, test_df


def surprise_traintest(
    ratings: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
):
    """Split ratings and return Surprise trainset + list of test tuples."""
    train_df, test_df = train_test_split_df(ratings, test_size, random_state)
    surprise_data = ratings_to_surprise(train_df)
    trainset = surprise_data.build_full_trainset()
    testset = [
        (str(row.user_id), str(row.item_id), row.rating) for row in test_df.itertuples()
    ]
    return trainset, testset, train_df, test_df
