"""Data preparation for NeuMF training on the Steam dataset."""

from __future__ import annotations

import ast
import html

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ── List-column parsing (genres / tags stored as str in parquet) ─────────────

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


# ── Filtering & encoding ─────────────────────────────────────────────────────

def build_interaction_table(
    df_reviews: pd.DataFrame,
    df_games: pd.DataFrame,
    min_user_reviews: int = 5,
    min_item_reviews: int = 10,
    content_cols: tuple[str, ...] = ("genres", "tags", "specs"),
) -> tuple[pd.DataFrame, dict, dict, dict, np.ndarray, list[str]]:
    """
    Build a filtered interaction table from user reviews (positive only).

    Returns:
        interactions  — DataFrame with columns [user_idx, item_idx, posted_date]
        user_to_idx   — str user_id → 0-based int
        item_to_idx   — str item_id → 0-based int
        idx_to_item   — int → str item_id
        genre_matrix  — np.ndarray (n_items, n_tokens), multi-hot float32 over the
                        union of tokens from `content_cols` (default: genres + tags).
                        Keeps the `genre_` name for backward compatibility with
                        downstream model / training code.
        genre_names   — list of token name strings (columns of genre_matrix)
    """
    df = df_reviews[["user_id", "item_id", "recommend", "posted_date"]].copy()
    df = df.dropna(subset=["user_id", "item_id"])
    # Use only positive interactions as the signal
    df = df[df["recommend"] == True].copy()

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
    """Build a multi-hot content matrix of shape (n_items, n_tokens).

    For each game we take the union of tokens across `content_cols`
    (e.g. genres + tags), deduplicated per game. `genres` is sparsely missing
    (~30% NaN in the catalogue); `tags` is much denser and overlaps heavily,
    so the union both fills the genre gaps and enriches the description.
    """
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


# ── Train / val / test split (leave-one-out by time) ────────────────────────

def leave_one_out_split(
    interactions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    For each user, sort interactions by date:
      test  = last  interaction
      val   = second-to-last
      train = everything else
    Users with <3 interactions are dropped.
    """
    interactions = interactions.copy()
    # Fallback: if date is all NaT, use original order
    interactions = interactions.sort_values(
        ["user_idx", "posted_date"], na_position="first"
    )

    train_parts, val_parts, test_parts = [], [], []

    for _, group in interactions.groupby("user_idx", sort=False):
        group = group.sort_values("posted_date", na_position="first")
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


# ── Negative sampling ────────────────────────────────────────────────────────

def sample_negatives(
    df_pos: pd.DataFrame,
    n_items: int,
    n_neg: int = 4,
    seed: int = 42,
    observed: set | None = None,
) -> pd.DataFrame:
    """
    For every positive (user_idx, item_idx) in df_pos, sample n_neg negatives.
    Returns a DataFrame with [user_idx, item_idx, label].
    """
    rng = np.random.default_rng(seed)

    if observed is None:
        observed = set(zip(df_pos["user_idx"], df_pos["item_idx"]))

    pos_rows = df_pos[["user_idx", "item_idx"]].copy()
    pos_rows["label"] = 1.0

    neg_list: list[dict] = []
    for u, i in zip(pos_rows["user_idx"], pos_rows["item_idx"]):
        count = 0
        while count < n_neg:
            j = int(rng.integers(0, n_items))
            if (u, j) not in observed:
                neg_list.append({"user_idx": u, "item_idx": j, "label": 0.0})
                count += 1

    neg_df = pd.DataFrame(neg_list)
    combined = pd.concat([pos_rows, neg_df], ignore_index=True)
    return combined.sample(frac=1, random_state=int(rng.integers(1_000_000))).reset_index(drop=True)


# ── PyTorch Dataset ──────────────────────────────────────────────────────────

class InteractionDataset(Dataset):
    """Yields (user_idx, item_idx, genre_vector, label) per sample."""

    def __init__(self, df: pd.DataFrame, genre_matrix: np.ndarray):
        self.users  = torch.tensor(df["user_idx"].values, dtype=torch.long)
        self.items  = torch.tensor(df["item_idx"].values, dtype=torch.long)
        self.labels = torch.tensor(df["label"].values,    dtype=torch.float32)
        self.genre_mat = torch.tensor(genre_matrix, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx):
        i = self.items[idx]
        return self.users[idx], i, self.genre_mat[i], self.labels[idx]
