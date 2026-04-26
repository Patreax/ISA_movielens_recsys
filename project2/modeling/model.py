"""NeuMF model variants for the Steam recommendation system.

Reference: He et al. (2017). Neural Collaborative Filtering. WWW 2017.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PopularityBaseline(nn.Module):
    """Non-personalised baseline: scores each item by its training frequency."""

    def __init__(self, item_counts: dict[int, int], n_items: int):
        super().__init__()
        counts = torch.zeros(n_items)
        for idx, cnt in item_counts.items():
            counts[idx] = float(cnt)
        total = counts.sum().clamp(min=1)
        self.register_buffer("scores", counts / total)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                genres: torch.Tensor | None = None) -> torch.Tensor:
        return self.scores[item_ids]


class GMFOnly(nn.Module):
    """Generalized Matrix Factorization branch only (linear CF)."""

    def __init__(self, n_users: int, n_items: int, emb_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, emb_dim)
        self.item_emb = nn.Embedding(n_items, emb_dim)
        self.out      = nn.Linear(emb_dim, 1)
        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
              genres: torch.Tensor | None = None) -> torch.Tensor:
        u = self.user_emb(user_ids)
        v = self.item_emb(item_ids)
        return self.out(u * v).squeeze(1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                genres: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.score(user_ids, item_ids, genres))


class MLPOnly(nn.Module):
    """MLP branch only — can incorporate genre side features."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        emb_dim: int = 64,
        n_genres: int = 0,
        genre_proj_dim: int = 32,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
    ):
        super().__init__()
        self.user_emb    = nn.Embedding(n_users, emb_dim)
        self.item_emb    = nn.Embedding(n_items, emb_dim)
        self.use_genres  = n_genres > 0
        if self.use_genres:
            self.genre_proj = nn.Linear(n_genres, genre_proj_dim)
            in_dim = emb_dim * 2 + genre_proj_dim
        else:
            in_dim = emb_dim * 2

        layers: list[nn.Module] = []
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
              genres: torch.Tensor | None = None) -> torch.Tensor:
        u, v = self.user_emb(user_ids), self.item_emb(item_ids)
        parts = [u, v]
        if self.use_genres and genres is not None:
            parts.append(torch.relu(self.genre_proj(genres)))
        return self.mlp(torch.cat(parts, dim=1)).squeeze(1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                genres: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.score(user_ids, item_ids, genres))


class NeuMF(nn.Module):
    """
    Neural Matrix Factorization (He et al., 2017).

    Architecture:
      GMF branch  : separate user/item embeddings (dim=gmf_emb_dim), element-wise product
      MLP branch  : separate larger embeddings (dim=mlp_emb_dim) + optional side features,
                    then FC 128→64→32 with ReLU
      Fusion layer: concat(gmf_out, mlp_out) → Linear(1) → sigmoid
    """

    def __init__(
        self,
        n_users: int,
        n_items: int,
        gmf_emb_dim: int = 32,
        mlp_emb_dim: int = 64,
        n_genres: int = 0,
        genre_proj_dim: int = 32,
        hidden_dims: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.0,
    ):
        super().__init__()

        # GMF branch
        self.gmf_user_emb = nn.Embedding(n_users, gmf_emb_dim)
        self.gmf_item_emb = nn.Embedding(n_items, gmf_emb_dim)

        # MLP branch
        self.mlp_user_emb = nn.Embedding(n_users, mlp_emb_dim)
        self.mlp_item_emb = nn.Embedding(n_items, mlp_emb_dim)

        self.use_genres = n_genres > 0
        if self.use_genres:
            self.genre_proj = nn.Linear(n_genres, genre_proj_dim)
            mlp_in = mlp_emb_dim * 2 + genre_proj_dim
        else:
            mlp_in = mlp_emb_dim * 2

        mlp_layers: list[nn.Module] = []
        for h in hidden_dims:
            mlp_layers += [nn.Linear(mlp_in, h), nn.ReLU(), nn.Dropout(p=dropout)]
            mlp_in = h
        self.mlp = nn.Sequential(*mlp_layers)

        # Fusion
        self.fusion = nn.Linear(gmf_emb_dim + hidden_dims[-1], 1)

        for emb in [self.gmf_user_emb, self.gmf_item_emb,
                    self.mlp_user_emb, self.mlp_item_emb]:
            nn.init.normal_(emb.weight, std=0.01)

    def score(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
              genres: torch.Tensor | None = None) -> torch.Tensor:
        """Pre-sigmoid score (logits). Used by BPR and BCEWithLogitsLoss."""
        # GMF
        gu = self.gmf_user_emb(user_ids)
        gi = self.gmf_item_emb(item_ids)
        gmf_out = gu * gi                          # (B, gmf_emb_dim)

        # MLP
        mu, mi = self.mlp_user_emb(user_ids), self.mlp_item_emb(item_ids)
        parts = [mu, mi]
        if self.use_genres and genres is not None:
            parts.append(torch.relu(self.genre_proj(genres)))
        mlp_out = self.mlp(torch.cat(parts, dim=1))  # (B, hidden_dims[-1])

        # Fusion
        x = torch.cat([gmf_out, mlp_out], dim=1)
        return self.fusion(x).squeeze(1)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor,
                genres: torch.Tensor | None = None) -> torch.Tensor:
        return torch.sigmoid(self.score(user_ids, item_ids, genres))
