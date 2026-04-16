"""Ranking metrics for the leave-one-out evaluation protocol.

For each user in the test set:
  - The single held-out item is the positive
  - All items NOT in that user's training history are candidate negatives
  - We score all candidates, rank them, and check if the positive lands in top-K

Metrics:
  Hit@K       — 1 if positive is in top-K, else 0
  NDCG@K      — normalised discounted cumulative gain (1/log2(rank+1) if positive in top-K)
  Precision@K — fraction of top-K that is positive (= Hit@K / K for leave-one-out)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@torch.no_grad()
def evaluate_loo(
    model: nn.Module,
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    genre_matrix: np.ndarray,
    n_items: int,
    device: torch.device,
    k: int = 10,
    batch_size: int = 1024,
) -> dict[str, float]:
    """
    Leave-one-out evaluation.

    For every user in df_eval, score all items not seen in training,
    rank them, and compute Hit@k, NDCG@k, Precision@k.
    """
    model.eval()
    genre_t = torch.tensor(genre_matrix, dtype=torch.float32, device=device)

    # Build per-user training history for quick lookup
    train_hist: dict[int, set[int]] = {}
    for u, i in zip(df_train["user_idx"], df_train["item_idx"]):
        train_hist.setdefault(int(u), set()).add(int(i))

    hits, ndcgs, precs = [], [], []
    all_items = np.arange(n_items)

    for _, row in df_eval.iterrows():
        u = int(row["user_idx"])
        pos_item = int(row["item_idx"])

        seen = train_hist.get(u, set())
        candidates = all_items[np.isin(all_items, list(seen), invert=True)]
        # Ensure the positive item is in candidates
        if pos_item not in candidates:
            candidates = np.append(candidates, pos_item)

        # Score in batches
        u_tensor = torch.full((len(candidates),), u, dtype=torch.long, device=device)
        i_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

        scores_list = []
        for start in range(0, len(candidates), batch_size):
            ub = u_tensor[start: start + batch_size]
            ib = i_tensor[start: start + batch_size]
            gb = genre_t[ib]
            scores_list.append(model(ub, ib, gb).cpu().numpy())
        scores = np.concatenate(scores_list)

        # Rank (higher score = better)
        ranked_items = candidates[np.argsort(-scores)]
        top_k = ranked_items[:k]

        hit = int(pos_item in top_k)
        if hit:
            rank = np.where(ranked_items == pos_item)[0][0] + 1  # 1-based
            ndcg = 1.0 / np.log2(rank + 1)
        else:
            ndcg = 0.0

        hits.append(hit)
        ndcgs.append(ndcg)
        precs.append(hit / k)

    return {
        f"Hit@{k}":       float(np.mean(hits)),
        f"NDCG@{k}":      float(np.mean(ndcgs)),
        f"Precision@{k}": float(np.mean(precs)),
    }


@torch.no_grad()
def evaluate_all_k(
    model: nn.Module,
    df_train: pd.DataFrame,
    df_eval: pd.DataFrame,
    genre_matrix: np.ndarray,
    n_items: int,
    device: torch.device,
    k_values: tuple[int, ...] = (5, 10, 20),
    batch_size: int = 1024,
) -> dict[str, float]:
    """Evaluate at multiple K values in a single pass (more efficient)."""
    model.eval()
    genre_t = torch.tensor(genre_matrix, dtype=torch.float32, device=device)

    train_hist: dict[int, set[int]] = {}
    for u, i in zip(df_train["user_idx"], df_train["item_idx"]):
        train_hist.setdefault(int(u), set()).add(int(i))

    results: dict[str, list[float]] = {
        f"{m}@{k}": [] for k in k_values for m in ("Hit", "NDCG", "Precision")
    }
    all_items = np.arange(n_items)

    for _, row in df_eval.iterrows():
        u = int(row["user_idx"])
        pos_item = int(row["item_idx"])

        seen = train_hist.get(u, set())
        candidates = all_items[np.isin(all_items, list(seen), invert=True)]
        if pos_item not in candidates:
            candidates = np.append(candidates, pos_item)

        u_tensor = torch.full((len(candidates),), u, dtype=torch.long, device=device)
        i_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

        scores_list = []
        for start in range(0, len(candidates), batch_size):
            ub = u_tensor[start: start + batch_size]
            ib = i_tensor[start: start + batch_size]
            scores_list.append(model(ub, ib, genre_t[ib]).cpu().numpy())
        scores = np.concatenate(scores_list)

        ranked_items = candidates[np.argsort(-scores)]
        pos_rank = int(np.where(ranked_items == pos_item)[0][0]) + 1  # 1-based

        for k in k_values:
            hit  = int(pos_rank <= k)
            ndcg = (1.0 / np.log2(pos_rank + 1)) if hit else 0.0
            results[f"Hit@{k}"].append(hit)
            results[f"NDCG@{k}"].append(ndcg)
            results[f"Precision@{k}"].append(hit / k)

    return {key: float(np.mean(vals)) for key, vals in results.items()}
