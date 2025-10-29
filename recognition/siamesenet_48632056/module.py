from __future__ import annotations

from typing import Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_dist(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine distance matrix between two sets of embeddings.
    Args:
        a: Tensor of shape [N, D]
        b: Tensor of shape [M, D]
    Returns:
        Tensor of shape [N, M] with cosine distances in [0, 2].
    """
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("cosine_dist expects 2D tensors [N, D] and [M, D].")
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    similarity = a_norm @ b_norm.t()
    return 1.0 - similarity


class SiameseBackbone(nn.Module):
    def __init__(
        self,
        backbone: str = "tf_efficientnet_b0_ns",
        out_dim: int = 512,
        p_drop: float = 0.2,
    ) -> None:
        super().__init__()
        if out_dim <= 0:
            raise ValueError("out_dim must be a positive integer.")
        if not 0.0 <= p_drop < 1.0:
            raise ValueError("p_drop must be in [0, 1).")

        self.encoder = timm.create_model(
            backbone, pretrained=True, num_classes=0, global_pool="avg"
        )
        in_features = getattr(self.encoder, "num_features", None)
        if in_features is None:
            raise AttributeError("Encoder returned by timm must expose num_features.")

        self.head = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Linear(in_features, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("Input to SiameseBackbone must be 4D [B, C, H, W].")
        features = self.encoder(x)
        embeddings = self.head(features)
        return F.normalize(embeddings, p=2, dim=-1)

    def freeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True


def batch_hard_triplet_loss(
    emb: torch.Tensor, labels: torch.Tensor, margin: float = 0.2
) -> torch.Tensor:
    if emb.ndim != 2:
        raise ValueError("emb must be a 2D tensor of shape [B, D].")
    if labels.ndim != 1:
        raise ValueError("labels must be a 1D tensor of shape [B].")
    if emb.shape[0] != labels.shape[0]:
        raise ValueError("emb and labels must have matching batch dimensions.")
    if labels.dtype not in (torch.int32, torch.int64):
        raise ValueError("labels must be integer tensor containing class indices.")

    unique_classes = labels.unique()
    if unique_classes.numel() < 2:
        raise ValueError("Triplet loss requires at least two classes in the batch.")

    dist_matrix = cosine_dist(emb, emb)
    batch_size = emb.shape[0]
    losses = []

    for anchor_idx in range(batch_size):
        anchor_label = labels[anchor_idx]
        mask_positive = labels == anchor_label
        mask_positive[anchor_idx] = False

        mask_negative = ~mask_positive & (labels != anchor_label)

        if not mask_positive.any():
            raise ValueError("Each anchor must have at least one positive sample.")
        if not mask_negative.any():
            raise ValueError("Each anchor must have at least one negative sample.")

        hardest_positive = dist_matrix[anchor_idx][mask_positive].max()
        hardest_negative = dist_matrix[anchor_idx][mask_negative].min()

        loss = F.relu(margin + hardest_positive - hardest_negative)
        losses.append(loss)

    if not losses:
        raise ValueError("Unable to compute triplet loss; no valid pairs found.")

    return torch.stack(losses).mean()


@torch.no_grad()
def build_prototypes(embs: torch.Tensor, ys: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if embs.ndim != 2:
        raise ValueError("embs must be 2D tensor [N, D].")
    if ys.ndim != 1:
        raise ValueError("ys must be 1D tensor [N].")
    if embs.shape[0] != ys.shape[0]:
        raise ValueError("embs and ys must have matching first dimension.")

    mask_normal = ys == 0
    mask_melanoma = ys == 1
    if not mask_normal.any() or not mask_melanoma.any():
        raise ValueError("Both classes (0 and 1) must be present to build prototypes.")

    proto0 = F.normalize(embs[mask_normal].mean(dim=0, keepdim=True), p=2, dim=-1)
    proto1 = F.normalize(embs[mask_melanoma].mean(dim=0, keepdim=True), p=2, dim=-1)
    return proto0, proto1


@torch.no_grad()
def score_by_prototypes(
    embs: torch.Tensor, proto0: torch.Tensor, proto1: torch.Tensor
) -> torch.Tensor:
    if embs.ndim != 2:
        raise ValueError("embs must be 2D tensor [N, D].")

    embs_norm = F.normalize(embs, p=2, dim=-1)
    proto0_norm = F.normalize(proto0, p=2, dim=-1)
    proto1_norm = F.normalize(proto1, p=2, dim=-1)

    cos_normal = (embs_norm @ proto0_norm.t()).squeeze(-1)
    cos_melanoma = (embs_norm @ proto1_norm.t()).squeeze(-1)

    logits = cos_melanoma - cos_normal
    return torch.sigmoid(logits)
