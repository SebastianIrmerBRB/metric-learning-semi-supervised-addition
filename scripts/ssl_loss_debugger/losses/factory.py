"""Factories and adapters for supervised metric-learning objectives."""

from __future__ import annotations

import torch
from pytorch_metric_learning.losses import ProxyAnchorLoss
from torch import nn


def make_supervised_criterion(
    name: str,
    num_classes: int,
    embedding_dim: int,
    device: torch.device,
) -> nn.Module:
    if name == "proxy_anchor":
        criterion = ProxyAnchorLoss(
            num_classes=num_classes,
            embedding_size=embedding_dim,
            margin=0.1,
            alpha=16.0,
        )
    else:
        raise ValueError(f"Unknown supervised loss: {name}")
    return criterion.to(device)


def compute_supervised_loss(
    criterion: nn.Module,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor,
) -> torch.Tensor:
    if getattr(criterion, "supports_sample_weights", False):
        return criterion(
            embeddings,
            labels,
            sample_weights=sample_weights.to(embeddings.device),
        )
    return criterion(embeddings, labels)
