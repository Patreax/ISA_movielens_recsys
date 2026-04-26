"""Data preparation for NeuMF training on the Steam dataset."""

from __future__ import annotations

import ast
import html

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

def parse_list_col(series: pd.Series) -> pd.Series:
    """Convert genres/tags columns to actual Python lists.

    Handles three possible storage formats that PyArrow may produce:
      - Already a Python list  (rare, pass-through)
      - numpy.ndarray          (PyArrow list-type columns read back from parquet)
      - str repr of a list     (legacy: "['Action', 'Indie']" serialised as string)
    """
    def _parse(x):
        if isinstance(x, list):
            return x
        if isinstance(x, np.ndarray):
            return x.tolist()
        if not isinstance(x, str) or x.strip() in ("", "nan", "None"):
            return []
        try:
            result = ast.literal_eval(x)
            return result if isinstance(result, list) else []
        except (ValueError, SyntaxError):
            return []
    return series.apply(_parse)


def build_interaction_table(
    df_reviews: pd.DataFrame,
    df_games: pd.DataFrame,
    min_user_reviews: int = 5,
    min_item_reviews: int = 10,
    content_cols: tuple[str, ...] = ("genres", "tags", "specs"),
    include_negative_reviews: bool = False,
    df_items: pd.DataFrame | None = None,
    playtime_alpha: float = 1.0,
) -> tuple[pd.DataFrame, dict, dict, dict, np.ndarray, list[str]]:
    """
    Build a filtered interaction table from user reviews.
    """
    df = df_reviews[["user_id", "item_id", "recommend", "posted_date"]].copy()
    df = df.dropna(subset=["user_id", "item_id"])

    if not include_negative_reviews:
        df = df[df["recommend"] == True].copy()

    df["label"] = df["recommend"].astype(float)

    # Iterative core filtering (converges quickly, usually 2-3 rounds)
    for _ in range(10):
        prev_len = len(df)
        user_counts = df["user_id"].value_counts()
        df = df[df["user_id"].isin(user_counts[user_counts >= min_user_reviews].index)]
        item_counts = df["item_id"].value_counts()
        df = df[df["item_id"].isin(item_counts[item_counts >= min_item_reviews].index)]
        if len(df) == prev_len:
            break

    # Contiguous integer encoding
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())
    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    item_to_idx = {iid: i for i, iid in enumerate(unique_items)}
    idx_to_item = {i: iid for iid, i in item_to_idx.items()}

    df = df.copy()
    df["user_idx"] = df["user_id"].map(user_to_idx)
    df["item_idx"] = df["item_id"].map(item_to_idx)

    # Optional playtime → confidence weight (Hu/Koren/Volinsky 2008 style).
    if df_items is not None and "playtime_forever" in df_items.columns:
        play = df_items[["user_id", "item_id", "playtime_forever"]].copy()
        df = df.merge(play, on=["user_id", "item_id"], how="left")
        df["playtime_forever"] = df["playtime_forever"].fillna(0.0)
        playtime_hours = df["playtime_forever"] / 60.0
        boost = 1.0 + playtime_alpha * np.log1p(playtime_hours)
        df["confidence"] = np.where(df["label"] > 0, boost, 1.0).astype(np.float32)
    else:
        df["confidence"] = np.float32(1.0)

    n_items = len(unique_items)
    genre_matrix, genre_names = _build_content_matrix(
        df_games, item_to_idx, n_items, content_cols
    )

    return df, user_to_idx, item_to_idx, idx_to_item, genre_matrix, genre_names


def _build_content_matrix(
    df_games: pd.DataFrame,
    item_to_idx: dict,
    n_items: int,
    content_cols: tuple[str, ...],
) -> tuple[np.ndarray, list[str]]:
    """Build a multi-hot content matrix of shape (n_items, n_tokens)"""
    parsed_cols = {col: parse_list_col(df_games[col]) for col in content_cols}

    id_to_tokens: dict[str, list[str]] = {}
    for idx, item_id in enumerate(df_games["id"].astype(str)):
        merged: list[str] = []
        seen: set[str] = set()
        for col in content_cols:
            for raw_tok in parsed_cols[col].iloc[idx]:
                # Normalise HTML entities ("Animation &amp; Modeling" in genres
                # collides with "Animation & Modeling" in tags otherwise).
                tok = html.unescape(raw_tok).strip() if isinstance(raw_tok, str) else ""
                if tok and tok not in seen:
                    seen.add(tok)
                    merged.append(tok)
        id_to_tokens[item_id] = merged

    all_tokens = sorted({tok for toks in id_to_tokens.values() for tok in toks})
    token_to_col = {tok: i for i, tok in enumerate(all_tokens)}
    n_tokens = len(all_tokens)

    matrix = np.zeros((n_items, n_tokens), dtype=np.float32)
    for item_id, item_idx in item_to_idx.items():
        for tok in id_to_tokens.get(str(item_id), []):
            col = token_to_col.get(tok)
            if col is not None:
                matrix[item_idx, col] = 1.0

    return matrix, all_tokens


def leave_one_out_split(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Per-user temporal leave-one-out split.
    """
    interactions = interactions.copy()
    interactions = interactions.sort_values(
        ["user_idx", "posted_date"], na_position="first"
    )
    has_label = "label" in interactions.columns

    train_parts, val_parts, test_parts = [], [], []

    for _, group in interactions.groupby("user_idx", sort=False):
        group = group.sort_values("posted_date", na_position="first")
        if has_label:
            pos_mask = group["label"] > 0
            pos_group = group[pos_mask]
            if len(pos_group) < 3:
                continue
            test_idx = [pos_group.index[-1]]
            val_idx  = [pos_group.index[-2]]
            holdout = set(test_idx) | set(val_idx)
            train_parts.append(group.loc[~group.index.isin(holdout)])
            val_parts.append(group.loc[val_idx])
            test_parts.append(group.loc[test_idx])
        else:
            n = len(group)
            if n < 3:
                continue
            train_parts.append(group.iloc[:-2])
            val_parts.append(group.iloc[[-2]])
            test_parts.append(group.iloc[[-1]])

    df_train = pd.concat(train_parts, ignore_index=True)
    df_val   = pd.concat(val_parts,   ignore_index=True)
    df_test  = pd.concat(test_parts,  ignore_index=True)
    return df_train, df_val, df_test


def _build_neg_pool(
    rng: np.random.Generator,
    n_items: int,
    size: int,
    weights: np.ndarray | None,
) -> np.ndarray:
    """Pre-sample a flat pool of candidate negative item indices."""
    if weights is None:
        return rng.integers(0, n_items, size=size)
    return rng.choice(n_items, size=size, p=weights)


def _make_pop_weights(item_popularity: np.ndarray, alpha: float) -> np.ndarray:
    """word2vec-style p_j ∝ count_j^alpha, normalised."""
    w = np.power(np.asarray(item_popularity, dtype=np.float64) + 1e-9, alpha)
    return w / w.sum()


def sample_negatives(
    df_pos: pd.DataFrame,
    n_items: int,
    n_neg: int = 4,
    seed: int = 42,
    observed: set | None = None,
    item_popularity: np.ndarray | None = None,
    pop_alpha: float = 0.75,
) -> pd.DataFrame:
    """Build a (user, item, label) training table with sampled negatives."""
    rng = np.random.default_rng(seed)

    has_label = "label" in df_pos.columns
    has_conf  = "confidence" in df_pos.columns

    if has_label:
        pos_rows      = df_pos[df_pos["label"] > 0].copy()
        real_neg_rows = df_pos[df_pos["label"] == 0].copy()
    else:
        pos_rows      = df_pos.copy()
        pos_rows["label"] = 1.0
        real_neg_rows = df_pos.iloc[0:0].copy()
        real_neg_rows["label"] = 0.0

    if not has_conf:
        pos_rows["confidence"] = 1.0
        real_neg_rows["confidence"] = 1.0

    if observed is None:
        observed = set(zip(df_pos["user_idx"], df_pos["item_idx"]))

    weights = (
        _make_pop_weights(item_popularity, pop_alpha)
        if item_popularity is not None else None
    )

    n_pos = len(pos_rows)
    if n_pos == 0:
        out = real_neg_rows[["user_idx", "item_idx", "label", "confidence"]]
        return out.reset_index(drop=True)

    # Pre-sample ~5x what we need; refill if we run out.
    pool = _build_neg_pool(rng, n_items, size=n_pos * n_neg * 5, weights=weights)
    pool_ptr = 0

    users  = pos_rows["user_idx"].to_numpy()
    items  = pos_rows["item_idx"].to_numpy()
    neg_u  = np.empty(n_pos * n_neg, dtype=np.int64)
    neg_i  = np.empty(n_pos * n_neg, dtype=np.int64)
    write  = 0

    for k in range(n_pos):
        u = int(users[k])
        count = 0
        attempts = 0
        while count < n_neg:
            if pool_ptr >= len(pool):
                pool = _build_neg_pool(rng, n_items, size=n_pos * n_neg, weights=weights)
                pool_ptr = 0
            j = int(pool[pool_ptr])
            pool_ptr += 1
            attempts += 1
            if (u, j) in observed:
                if attempts > n_neg * 50:
                    # Extremely dense user — accept the collision so we make progress.
                    pass
                else:
                    continue
            neg_u[write] = u
            neg_i[write] = j
            write += 1
            count += 1

    neg_df = pd.DataFrame({
        "user_idx":   neg_u[:write],
        "item_idx":   neg_i[:write],
        "label":      0.0,
        "confidence": 1.0,
    })

    out_cols = ["user_idx", "item_idx", "label", "confidence"]
    parts = [pos_rows[out_cols], real_neg_rows[out_cols], neg_df[out_cols]]
    combined = pd.concat(parts, ignore_index=True)
    return combined.sample(frac=1, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)


def sample_bpr_triplets(
    df_pos: pd.DataFrame,
    n_items: int,
    seed: int = 42,
    observed: set | None = None,
    item_popularity: np.ndarray | None = None,
    pop_alpha: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (user, pos_item, neg_item) triplets for BPR training"""
    rng = np.random.default_rng(seed)

    if "label" in df_pos.columns:
        df_pos = df_pos[df_pos["label"] > 0]

    if observed is None:
        observed = set(zip(df_pos["user_idx"], df_pos["item_idx"]))

    weights = (
        _make_pop_weights(item_popularity, pop_alpha)
        if item_popularity is not None else None
    )

    users     = df_pos["user_idx"].to_numpy(dtype=np.int64)
    pos_items = df_pos["item_idx"].to_numpy(dtype=np.int64)
    n = len(users)
    neg_items = np.empty(n, dtype=np.int64)

    pool = _build_neg_pool(rng, n_items, size=n * 5, weights=weights)
    pool_ptr = 0

    for k in range(n):
        u = int(users[k])
        last_j = 0
        for _ in range(50):
            if pool_ptr >= len(pool):
                pool = _build_neg_pool(rng, n_items, size=n, weights=weights)
                pool_ptr = 0
            j = int(pool[pool_ptr])
            pool_ptr += 1
            last_j = j
            if (u, j) not in observed:
                break
        neg_items[k] = last_j

    return users, pos_items, neg_items


class InteractionDataset(Dataset):
    """Yields (user_idx, item_idx, genre_vector, label, confidence) per sample"""

    def __init__(self, df: pd.DataFrame, genre_matrix: np.ndarray):
        self.users  = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items  = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values,    dtype=torch.float32)
        if "confidence" in df.columns:
            conf = df["confidence"].astype(np.float32).to_numpy()
        else:
            conf = np.ones(len(df), dtype=np.float32)
        self.confs  = torch.tensor(conf, dtype=torch.float32)
        self.genre_mat = torch.tensor(genre_matrix, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        i = self.items[idx]
        return (
            self.users[idx],
            i,
            self.genre_mat[i],
            self.labels[idx],
            self.confs[idx],
        )


class BPRTripletDataset(Dataset):
    """Yields (user, pos_item, neg_item, pos_genre, neg_genre) per sample."""

    def __init__(
        self,
        users: np.ndarray,
        pos_items: np.ndarray,
        neg_items: np.ndarray,
        genre_matrix: np.ndarray,
    ):
        self.users     = torch.tensor(users,     dtype=torch.long)
        self.pos_items = torch.tensor(pos_items, dtype=torch.long)
        self.neg_items = torch.tensor(neg_items, dtype=torch.long)
        self.genre_mat = torch.tensor(genre_matrix, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx):
        pi = self.pos_items[idx]
        ni = self.neg_items[idx]
        return (
            self.users[idx],
            pi,
            ni,
            self.genre_mat[pi],
            self.genre_mat[ni],
        )
