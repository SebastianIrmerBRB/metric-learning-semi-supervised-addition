"""Tiny retrieval model used by the synthetic debugger workload."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from models.retrieval_model import (
    BACKBONE_TUNING_FROZEN,
    BACKBONE_TUNING_FULL,
    normalize_backbone_tuning,
)


class TinyRetrievalModel(nn.Module):
    """Tiny normalized retrieval model."""

    def __init__(
        self,
        image_size: int,
        feat_dim: int,
        backbone_tuning: str = BACKBONE_TUNING_FULL,
        precomputed_feature_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.backbone_tuning = normalize_backbone_tuning(backbone_tuning)
        if self.backbone_tuning not in {
            BACKBONE_TUNING_FULL,
            BACKBONE_TUNING_FROZEN,
        }:
            raise ValueError(
                "TinyRetrievalModel supports only 'full' or 'frozen' backbone tuning"
            )
        self.precomputed_feature_dim = (
            None if precomputed_feature_dim is None else int(precomputed_feature_dim)
        )
        if self.precomputed_feature_dim is None:
            flat_dim = 3 * int(image_size) * int(image_size)
            hidden_dim = max(12, 2 * self.feat_dim)
            self.encoder = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, hidden_dim),
                nn.ReLU(),
            )
        else:
            # The inputs are already backbone features, so the backbone left to
            # run is the identity. What remains is exactly DinoWrapper's frozen
            # path: fc is the first trainable layer over precomputed features.
            hidden_dim = self.precomputed_feature_dim
            self.encoder = nn.Sequential(nn.Flatten())
        self.fc = nn.Linear(hidden_dim, self.feat_dim)
        self._configure_backbone_tuning()

    def _configure_backbone_tuning(self) -> None:
        requires_grad = self.backbone_tuning == BACKBONE_TUNING_FULL
        for parameter in self.encoder.parameters():
            parameter.requires_grad = requires_grad
        if self.backbone_tuning == BACKBONE_TUNING_FROZEN:
            self.encoder.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_tuning == BACKBONE_TUNING_FROZEN:
            self.encoder.eval()
        return self

    def forward_backbone(self, inputs: torch.Tensor) -> torch.Tensor:
        inputs = inputs.float()
        if self.backbone_tuning == BACKBONE_TUNING_FROZEN:
            with torch.no_grad():
                return self.encoder(inputs)
        return self.encoder(inputs)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        embeddings = self.fc(self.forward_backbone(inputs))
        return F.normalize(embeddings, p=2.0, dim=1)
