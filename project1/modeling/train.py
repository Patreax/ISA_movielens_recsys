import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from surprise import SVD, accuracy
from surprise.model_selection import cross_validate, KFold

from project1.config import MODELS_DIR
from project1.dataset import load_items, load_ratings, load_users, ratings_to_surprise
from project1.features import build_user_features
import optuna

def train_svd(
    trainset,
    n_factors: int = 100,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
    random_state: int | None = None,
) -> SVD:
    """Train an SVD model on a Surprise trainset."""
    algo = SVD(
        n_factors=n_factors,
        n_epochs=n_epochs,
        lr_all=lr_all,
        reg_all=reg_all,
        random_state=random_state,
    )
    algo.fit(trainset)
    logger.info(
        f"SVD trained: n_factors={n_factors}, n_epochs={n_epochs}, lr={lr_all}, reg={reg_all}"
    )
    return algo


def evaluate_svd(algo: SVD, testset) -> dict:
    """Evaluate SVD predictions on a testset, returning RMSE and MAE."""
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    logger.info(f"SVD evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}")
    return {"rmse": rmse, "mae": mae, "predictions": predictions}


def cross_validate_svd(
    ratings: pd.DataFrame | None = None,
    n_factors: int = 100,
    n_epochs: int = 20,
    lr_all: float = 0.005,
    reg_all: float = 0.02,
    cv: int = 5,
) -> dict:
    """Run cross-validation on the full ratings dataset using Surprise's built-in splitter."""
    if ratings is None:
        ratings = load_ratings()
    data = ratings_to_surprise(ratings)
    algo = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    results = cross_validate(algo, data, measures=["RMSE", "MAE"], cv=cv, verbose=False)
    logger.info(
        f"CV results ({cv}-fold): "
        f"RMSE={np.mean(results['test_rmse']):.4f}±{np.std(results['test_rmse']):.4f}, "
        f"MAE={np.mean(results['test_mae']):.4f}±{np.std(results['test_mae']):.4f}"
    )
    return results


def train_cold_start_model(
    ratings: pd.DataFrame,
    n_clusters: int = 30,
    users: pd.DataFrame | None = None,
    items: pd.DataFrame | None = None,
) -> dict:
    """Train a KMeans clustering model on user features for cold-start recommendations.

    Returns a dict with the fitted KMeans, scaler, encoders, user features, and cluster assignments.
    """
    if users is None:
        users = load_users()
    if items is None:
        items = load_items()

    user_features, scaler, encoders = build_user_features(ratings, users, items)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(user_features.values)
    user_features["cluster"] = clusters

    logger.info(
        f"KMeans cold-start model trained: {n_clusters} clusters on {len(user_features)} users"
    )
    return {
        "kmeans": kmeans,
        "scaler": scaler,
        "encoders": encoders,
        "user_features": user_features,
    }


def bayesian_optimize_svd(
    data,
    n_trials: int = 30,
    cv: int = 3,
    metric: str = "rmse",
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """Bayesian hyperparameter optimisation for SVD using Optuna (TPE sampler).

    Each trial suggests a new configuration from the surrogate model, trains SVD
    using ``cv``-fold cross-validation on ``data``, and records the mean score.
    After ``n_trials`` the study returns the configuration that minimised ``metric``.

    Using K-fold CV instead of a single validation split reduces sensitivity to
    any particular split and produces a more stable estimate of generalisation.

    Parameters
    ----------
    data         : Surprise Dataset built from the training DataFrame (never include
                   the final test set here).
    n_trials     : Number of Optuna trials (function evaluations).
    cv           : Number of cross-validation folds used inside each trial.
    metric       : Optimisation target — ``"rmse"`` or ``"mae"``.
    random_state : Seed passed to the TPE sampler and KFold splitter for reproducibility.

    Returns
    -------
    results_df  : DataFrame with one row per trial (params + mean CV score), sorted by metric.
    best_params : Dict with the winning hyperparameter configuration.
    """
    if metric not in ("rmse", "mae"):
        raise ValueError(f"metric must be 'rmse' or 'mae', got '{metric}'")

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    history: list[dict] = []
    kf = KFold(n_splits=cv, random_state=random_state, shuffle=True)

    def _objective(trial: "optuna.Trial") -> float:
        params = {
            "n_factors": trial.suggest_int("n_factors", 20, 200),
            "n_epochs":  trial.suggest_int("n_epochs", 10, 50),
            "lr_all":    trial.suggest_float("lr_all", 0.001, 0.02, log=True),
            "reg_all":   trial.suggest_float("reg_all", 0.01,  0.1,  log=True),
        }
        fold_scores = []
        for fold_trainset, fold_valset in kf.split(data):
            algo = train_svd(
                fold_trainset,
                random_state=random_state + trial.number,
                **params,
            )
            fold_scores.append(evaluate_svd(algo, fold_valset)[metric])
        score = float(np.mean(fold_scores))
        history.append({**params, "trial": trial.number + 1, metric: score})
        return score

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    logger.info(
        f"BO best ({n_trials} trials, {cv}-fold CV): n_factors={best_params['n_factors']}, "
        f"n_epochs={best_params['n_epochs']}, lr={best_params['lr_all']:.4f}, "
        f"reg={best_params['reg_all']:.3f} → mean {metric.upper()}={study.best_value:.4f}"
    )
    results_df = pd.DataFrame(history).sort_values(metric).reset_index(drop=True)
    return results_df, best_params


def save_models(svd_algo: SVD, cold_start_model: dict, output_dir: Path | None = None):
    """Persist the SVD and cold-start models to disk."""
    if output_dir is None:
        output_dir = MODELS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "svd_model.pkl", "wb") as f:
        pickle.dump(svd_algo, f)
    with open(output_dir / "cold_start_model.pkl", "wb") as f:
        pickle.dump(cold_start_model, f)
    logger.info(f"Models saved to {output_dir}")


def load_models(model_dir: Path | None = None) -> tuple[SVD, dict]:
    """Load persisted SVD and cold-start models."""
    if model_dir is None:
        model_dir = MODELS_DIR

    with open(model_dir / "svd_model.pkl", "rb") as f:
        svd_algo = pickle.load(f)
    with open(model_dir / "cold_start_model.pkl", "rb") as f:
        cold_start_model = pickle.load(f)
    logger.info(f"Models loaded from {model_dir}")
    return svd_algo, cold_start_model
