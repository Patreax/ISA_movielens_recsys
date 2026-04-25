"""Training loop for NeuMF and its ablation variants.

Supports two losses:
  - "bce" (default) — point-wise binary cross-entropy with logits, optionally
    weighted by a per-row confidence (Hu/Koren/Volinsky 2008).
  - "bpr" — pairwise Bayesian Personalised Ranking (Rendle et al. 2009),
    directly optimising the ranking the eval metric cares about.

Negative samples can be drawn uniformly (default) or weighted by item
popularity (``p_j ∝ count_j ** 0.75``).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from project2.modeling.data_prep import (
    BPRTripletDataset,
    InteractionDataset,
    sample_bpr_triplets,
    sample_negatives,
)
from project2.modeling.evaluate import evaluate_loo


def _train_epoch_bce(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_confidence: bool,
) -> float:
    """Point-wise BCE (with logits) — confidence is folded in as a per-sample weight."""
    model.train()
    total_loss = 0.0
    n_seen = 0
    for user_ids, item_ids, genres, labels, confs in loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        genres   = genres.to(device)
        labels   = labels.to(device)
        confs    = confs.to(device)

        optimizer.zero_grad()
        logits = model.score(user_ids, item_ids, genres)
        per_row = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        if use_confidence:
            per_row = per_row * confs
        loss = per_row.mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
        n_seen += len(labels)
    return total_loss / max(n_seen, 1)


def _train_epoch_bpr(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Pairwise BPR: maximise log σ(s_pos − s_neg)."""
    model.train()
    total_loss = 0.0
    n_seen = 0
    for users, pos_items, neg_items, pos_g, neg_g in loader:
        users     = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        pos_g     = pos_g.to(device)
        neg_g     = neg_g.to(device)

        optimizer.zero_grad()
        s_pos = model.score(users, pos_items, pos_g)
        s_neg = model.score(users, neg_items, neg_g)
        loss = -F.logsigmoid(s_pos - s_neg).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(users)
        n_seen += len(users)
    return total_loss / max(n_seen, 1)


def _compute_item_popularity(df_train: pd.DataFrame, n_items: int) -> np.ndarray:
    """Train-set item count vector, used for popularity-weighted negative sampling."""
    counts = np.zeros(n_items, dtype=np.float64)
    if "label" in df_train.columns:
        pos = df_train[df_train["label"] > 0]
    else:
        pos = df_train
    vc = pos["item_idx"].value_counts()
    counts[vc.index.to_numpy()] = vc.to_numpy()
    return counts


def train_model(
    model: nn.Module,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    genre_matrix: np.ndarray,
    n_items: int,
    n_neg: int = 4,
    n_epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    patience: int = 4,
    device: torch.device | None = None,
    k: int = 10,
    verbose: bool = True,
    loss: str = "bce",
    use_confidence: bool = False,
    pop_weighted_negatives: bool = False,
    pop_alpha: float = 0.75,
) -> dict:
    """
    Train a model with early stopping on validation NDCG@k.

    Negative samples are re-drawn each epoch to avoid overfitting.

    Args:
        loss: "bce" (point-wise) or "bpr" (pair-wise).
        use_confidence: when True and `loss="bce"`, each row's BCE term is
            multiplied by ``df_train["confidence"]``. Ignored for BPR.
        pop_weighted_negatives: when True, negatives are sampled with
            ``p_j ∝ count_j ** pop_alpha`` instead of uniformly.
        pop_alpha: exponent of the popularity weighting (default 0.75).

    Returns:
        model    — best-checkpoint model loaded back in
        history  — list of per-epoch dicts with loss and metrics
        best_ndcg — best validation NDCG@k achieved
    """
    if loss not in ("bce", "bpr"):
        raise ValueError(f"Unknown loss '{loss}', expected 'bce' or 'bpr'")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=2, factor=0.5, min_lr=1e-5
    )

    # Pre-build the observed set for fast negative sampling (positive *and*
    # explicit-negative reviews count as observed — both are real interactions
    # we don't want to resample as fake negatives).
    observed = set(zip(df_train["user_idx"].tolist(), df_train["item_idx"].tolist()))

    item_pop = _compute_item_popularity(df_train, n_items) if pop_weighted_negatives else None

    best_ndcg  = -1.0
    best_state: dict | None = None
    no_improve = 0
    history    = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        if loss == "bce":
            df_sampled = sample_negatives(
                df_train, n_items, n_neg=n_neg, seed=epoch,
                observed=observed,
                item_popularity=item_pop, pop_alpha=pop_alpha,
            )
            dataset = InteractionDataset(df_sampled, genre_matrix)
            loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            train_loss = _train_epoch_bce(model, loader, optimizer, device,
                                          use_confidence=use_confidence)
        else:
            users, pos_items, neg_items = sample_bpr_triplets(
                df_train, n_items, seed=epoch, observed=observed,
                item_popularity=item_pop, pop_alpha=pop_alpha,
            )
            dataset = BPRTripletDataset(users, pos_items, neg_items, genre_matrix)
            loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            train_loss = _train_epoch_bpr(model, loader, optimizer, device)

        metrics = evaluate_loo(model, df_train, df_val, genre_matrix, n_items,
                                device=device, k=k)
        ndcg = metrics[f"NDCG@{k}"]
        scheduler.step(ndcg)

        elapsed = time.time() - t0
        current_lr = optimizer.param_groups[0]["lr"]
        row = {"epoch": epoch, "loss": train_loss, "lr": current_lr,
               "time_s": round(elapsed, 1), **metrics}
        history.append(row)

        if verbose:
            print(f"Epoch {epoch:02d} | loss={train_loss:.4f} | "
                  + " | ".join(f"{m}={v:.4f}" for m, v in metrics.items())
                  + f" | lr={current_lr:.2e} | {elapsed:.1f}s")

        if ndcg > best_ndcg:
            best_ndcg  = ndcg
            best_state = {k2: v.cpu().clone() for k2, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  -> Early stopping (no improvement for {patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return {"model": model, "history": history, "best_ndcg": best_ndcg}
