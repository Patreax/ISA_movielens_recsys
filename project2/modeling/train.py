"""Training loop for NeuMF and its ablation variants."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from project2.modeling.data_prep import InteractionDataset, sample_negatives
from project2.modeling.evaluate import evaluate_loo


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    criterion = nn.BCELoss()
    total_loss = 0.0
    for user_ids, item_ids, genres, labels in loader:
        user_ids = user_ids.to(device)
        item_ids = item_ids.to(device)
        genres   = genres.to(device)
        labels   = labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(user_ids, item_ids, genres), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / max(len(loader.dataset), 1)


def train_model(
    model: nn.Module,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    genre_matrix: np.ndarray,
    n_items: int,
    n_neg: int = 4,
    n_epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    patience: int = 4,
    device: torch.device | None = None,
    k: int = 10,
    verbose: bool = True,
) -> dict:
    """
    Train a model with early stopping on validation NDCG@k.

    Negative samples are re-drawn each epoch to avoid overfitting.

    Returns:
        model    — best-checkpoint model loaded back in
        history  — list of per-epoch dicts with loss and metrics
        best_ndcg — best validation NDCG@k achieved
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Pre-build the observed set for fast negative sampling
    observed = set(zip(df_train["user_idx"].tolist(), df_train["item_idx"].tolist()))

    best_ndcg  = -1.0
    best_state: dict | None = None
    no_improve = 0
    history    = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Re-sample negatives every epoch
        df_sampled = sample_negatives(df_train, n_items, n_neg=n_neg, seed=epoch,
                                      observed=observed)
        dataset = InteractionDataset(df_sampled, genre_matrix)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        loss = _train_epoch(model, loader, optimizer, device)

        metrics = evaluate_loo(model, df_train, df_val, genre_matrix, n_items,
                                device=device, k=k)
        ndcg = metrics[f"NDCG@{k}"]

        elapsed = time.time() - t0
        row = {"epoch": epoch, "loss": loss, "time_s": round(elapsed, 1), **metrics}
        history.append(row)

        if verbose:
            print(f"Epoch {epoch:02d} | loss={loss:.4f} | "
                  + " | ".join(f"{m}={v:.4f}" for m, v in metrics.items())
                  + f" | {elapsed:.1f}s")

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

    # Restore best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    return {"model": model, "history": history, "best_ndcg": best_ndcg}
