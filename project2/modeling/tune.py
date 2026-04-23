"""Optuna hyperparameter search for NeuMF.

Tunes 5 parameters while keeping the rest fixed from TRAIN_KWARGS:
  lr, weight_decay, dropout, gmf_emb_dim, mlp_emb_dim
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

from project2.modeling.model import NeuMF
from project2.modeling.train import train_model


def build_objective(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    genre_matrix: np.ndarray,
    n_users: int,
    n_items: int,
    n_genres: int,
    device: torch.device,
    # fixed training kwargs
    n_neg: int = 8,
    batch_size: int = 512,
    n_epochs: int = 15,
    patience: int = 3,
    k: int = 10,
):
    """Return an Optuna objective that tunes lr, weight_decay, dropout,
    gmf_emb_dim, mlp_emb_dim while keeping everything else fixed."""

    def objective(trial: optuna.Trial) -> float:
        # --- 5 tuned parameters ---
        lr           = trial.suggest_float("lr",           1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        dropout      = trial.suggest_float("dropout",      0.0,  0.4,  step=0.05)
        gmf_emb_dim  = trial.suggest_categorical("gmf_emb_dim", [16, 32, 64])
        mlp_emb_dim  = trial.suggest_categorical("mlp_emb_dim", [32, 64, 128])

        model = NeuMF(
            n_users, n_items,
            gmf_emb_dim=gmf_emb_dim,
            mlp_emb_dim=mlp_emb_dim,
            n_genres=n_genres,
            genre_proj_dim=32,
            hidden_dims=(128, 64, 32),
            dropout=dropout,
        )

        result = train_model(
            model,
            df_train=df_train,
            df_val=df_val,
            genre_matrix=genre_matrix,
            n_items=n_items,
            n_neg=n_neg,
            n_epochs=n_epochs,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            patience=patience,
            device=device,
            k=k,
            verbose=False,
        )
        return result["best_ndcg"]

    return objective


def run_study(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    genre_matrix: np.ndarray,
    n_users: int,
    n_items: int,
    n_genres: int,
    device: torch.device,
    n_trials: int = 30,
    n_neg: int = 8,
    batch_size: int = 512,
    n_epochs: int = 15,
    patience: int = 3,
    k: int = 10,
    seed: int = 42,
) -> optuna.Study:
    """Run an Optuna study and return it.

    After completion:
        study.best_params  — best found hyperparameters
        study.best_value   — best validation NDCG@k
    """
    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        study_name="neumf_steam",
    )

    objective = build_objective(
        df_train, df_val, genre_matrix,
        n_users, n_items, n_genres, device,
        n_neg=n_neg,
        batch_size=batch_size,
        n_epochs=n_epochs,
        patience=patience,
        k=k,
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    return study
