# Jeremy's implementation of EGNN
# Does not use any graph lib, but does require padding of sequences 
from typing import Callable
import math

import torch
import torch.nn as nn
from einops import rearrange

from pp3.utils.constants import NUM_ATOMS


class SinusoidalEmbeddings(nn.Module):
    """A simple sinusoidal embedding layer."""

    def __init__(self, dim, theta: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        embeddings = math.log(self.theta) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)

        return embeddings


class EGNN_Layer(nn.Module):
    """A simple fully connected EGNN Layer."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        dist_dim: int,
        proj_dim: int,
        message_dim: int,
        dropout: float = 0.0,
        use_sinusoidal: bool = True,
        activation: Callable = nn.ReLU,
        update_feats: bool = True,
        update_coors: bool = True,
    ) -> None:
        super().__init__()
        self.update_feats = update_feats
        self.update_coors = update_coors

        if not update_feats and not update_coors:
            raise ValueError(
                "At least one of update_feats or update_coors must be True."
            )

        if use_sinusoidal:
            self.dist_embedding = SinusoidalEmbeddings(dist_dim)
        else:
            self.dist_embedding = nn.Linear(1, dist_dim)  # type: ignore

        self.phi_e = nn.Sequential(
            nn.Linear(
                2 * node_dim + edge_dim + dist_dim * (NUM_ATOMS**2), message_dim
            ),
            nn.Dropout(dropout),
            activation(),
            nn.Linear(message_dim, message_dim),
        )

        if update_coors:
            self.phi_x = nn.Sequential(
                nn.Linear(message_dim, proj_dim),
                nn.Dropout(dropout),
                activation(),
                nn.Linear(proj_dim, 1),
            )

        if update_feats:
            self.phi_h = nn.Sequential(
                nn.Linear(node_dim + message_dim, proj_dim),
                nn.Dropout(dropout),
                activation(),
                nn.Linear(proj_dim, node_dim),
            )

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor,
        edges: torch.Tensor,
        neighbor_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Compute pairwise distances
        B, N = embeddings.shape[:2]
        M = neighbor_ids.shape[-1]

        dispatch = nn.functional.one_hot(neighbor_ids, N)
        feats2 = torch.einsum("bikj,bjd -> bikd", dispatch, embeddings)
        n_coords = torch.einsum("bikj,bjd -> bikd", dispatch, coords)

        rel_coors = coords.unsqueeze(2) - n_coords
        rel_dist = torch.linalg.norm(rel_coors, dim=-1)
        dists = self.dist_embedding(rearrange(rel_dist, "b i j -> (b i j)"))
        dists = rearrange(dists, "(b i j) d -> b i j d", b=B, i=N, j=M)

        # Compute pairwise features
        feats1 = embeddings.unsqueeze(2).expand(-1, -1, M, -1)
        if edges is not None:
            feats_pair = torch.cat((feats1, feats2, dists, edges), dim=-1)
        else:
            feats_pair = torch.cat((feats1, feats2, dists), dim=-1)

        # Compute messages
        m_ij = self.phi_e(feats_pair)

        # Padding and self already ignored
        mask = torch.ones((B, N, M), device=m_ij.device)
        mask_sum = M

        # Compute coordinate update
        if self.update_coors:
            rel_coors = torch.nan_to_num(rel_coors / rel_dist.unsqueeze(-1)).detach()
            delta = rel_coors * self.phi_x(m_ij)
            delta = delta * mask.unsqueeze(-1)
            delta = delta.sum(dim=2) / mask_sum
            coords = coords + delta

        # Compute feature update
        if self.update_feats:
            m_ij = m_ij * padding_mask.view(B, N, 1, 1)
            m_ij = m_ij * mask.unsqueeze(-1)
            m_i = m_ij.sum(dim=2) / mask_sum
            embeddings = embeddings + self.phi_h(torch.cat((embeddings, m_i), dim=-1))

        return embeddings, coords


class EGNN(nn.Module):
    """A simple fully connected EGNN."""

    def __init__(
        self,
        node_dim: int,
        hidden_dim: int = 16,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        dist_dim = hidden_dim
        proj_dim = hidden_dim * 8
        message_dim = hidden_dim * 2

        self.dist_embedding = SinusoidalEmbeddings(dist_dim)
        layers = [
            EGNN_Layer(
                node_dim=node_dim,
                edge_dim=dist_dim,
                dist_dim=dist_dim,
                message_dim=message_dim,
                proj_dim=proj_dim,
                dropout=dropout,
                update_coors=i < (num_layers - 1),
                update_feats=True,
            )
            for i in range(num_layers)
        ]
        self.layers = nn.ModuleList(layers)

    def forward(
        self,
        embeddings: torch.Tensor,
        coords: torch.Tensor,
        padding_mask: torch.Tensor,
        dists: torch.Tensor,
        neighbor_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Only use c-a for coordinate updates
        coords = coords[:, :, 1]

        # Compute distance embeddings
        B, N, K = dists.shape[:3]
        dists = rearrange(dists, "b n k i j -> (b n k i j)")
        dists = self.dist_embedding(dists)
        dists = rearrange(dists, "(b n k d) -> b n k d", b=B, n=N, k=K)

        # Run layers, updating both features and coordinates
        for layer in self.layers:
            embeddings, coords = layer(
                embeddings, coords, padding_mask, dists, neighbor_ids
            )

        return embeddings