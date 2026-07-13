"""Project-local metric-learning losses."""

import torch
import torch.nn.functional as F
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.reducers import DivisorReducer
from pytorch_metric_learning.utils import common_functions as c_f


class STMLLoss(torch.nn.Module):
    """Faithful two-head STML objective from Kim et al., CVPR 2022."""

    requires_stml_embeddings = True

    def __init__(
        self,
        sigma=3.0,
        delta=1.0,
        num_views=2,
        num_neighbors=5,
        teacher_momentum=0.999,
        normalize_student=False,
        eps=1e-12,
    ):
        super().__init__()
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if delta <= 0:
            raise ValueError("delta must be positive")
        if int(num_views) < 2:
            raise ValueError("num_views must be at least 2")
        if int(num_neighbors) <= 0:
            raise ValueError("num_neighbors must be positive")
        if not 0 <= teacher_momentum < 1:
            raise ValueError("teacher_momentum must be in [0, 1)")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.sigma = float(sigma)
        self.delta = float(delta)
        self.num_views = int(num_views)
        self.num_neighbors = int(num_neighbors)
        self.topk = self.num_views * self.num_neighbors
        self.teacher_momentum = float(teacher_momentum)
        self.normalize_student = bool(normalize_student)
        self.eps = float(eps)

    def forward(self, student_f, student_g, teacher_g, instance_ids):
        self._validate_inputs(student_f, student_g, teacher_g, instance_ids)
        relaxed_f = self._relaxed_contrastive(student_f, teacher_g, instance_ids)
        relaxed_g = self._relaxed_contrastive(student_g, teacher_g, instance_ids)
        relaxed_contrastive = (relaxed_f + relaxed_g) / 2
        self_distillation = self._kl_self_distillation(student_f, student_g)
        return relaxed_contrastive + self_distillation

    def _validate_inputs(self, student_f, student_g, teacher_g, instance_ids):
        for name, embeddings in (
            ("student_f", student_f),
            ("student_g", student_g),
            ("teacher_g", teacher_g),
        ):
            if embeddings.ndim != 2:
                raise ValueError(f"{name} must be a matrix")
        batch_size = len(student_f)
        if len(student_g) != batch_size or len(teacher_g) != batch_size:
            raise ValueError("student and teacher embedding batches must have the same length")
        if instance_ids.ndim != 1 or len(instance_ids) != batch_size:
            raise ValueError("instance_ids must be a vector aligned with embeddings")
        if batch_size < 2:
            raise ValueError("STMLLoss requires at least two samples per batch")

    def _relaxed_contrastive(self, student_embeddings, teacher_g, instance_ids):
        # Pairwise distance kernels are kept in float32 even under mixed
        # precision because cdist support and stability vary by device/dtype.
        student = student_embeddings.float()
        student = F.normalize(student, p=2, dim=1) if self.normalize_student else student
        student_distances = torch.cdist(student, student)
        student_distances = student_distances / student_distances.mean(dim=1, keepdim=True).clamp_min(self.eps)

        with torch.no_grad():
            teacher = F.normalize(teacher_g.float(), p=2, dim=1)
            teacher_distances = torch.cdist(teacher, teacher)
            pair_similarity = torch.exp(-teacher_distances.square() / self.sigma)
            contextual_similarity = self._contextual_similarity(pair_similarity, instance_ids)
            weights = (pair_similarity + contextual_similarity) / 2
            off_diagonal = ~torch.eye(len(student), device=student.device, dtype=torch.bool)
            positive_weights = weights.masked_fill(~off_diagonal, 0)
            negative_weights = (1 - weights).masked_fill(~off_diagonal, 0)

        pull = student_distances.square() * positive_weights
        push = F.relu(self.delta - student_distances).square() * negative_weights
        return (pull.sum() + push.sum()) / off_diagonal.sum()

    def _contextual_similarity(self, pair_similarity, instance_ids):
        """Compute the contextualized teacher similarity used by STML."""

        num_samples = len(pair_similarity)
        topk = min(self.topk, num_samples)
        ranking_similarity = pair_similarity.clone()
        same_instance = instance_ids.unsqueeze(1) == instance_ids.unsqueeze(0)
        ranking_similarity[same_instance] = 1
        topk_indices = ranking_similarity.topk(topk, dim=1).indices
        neighbor_mask = torch.zeros_like(pair_similarity)
        neighbor_mask.scatter_(1, topk_indices, 1)

        # V contains only reciprocal nearest-neighbor relationships.
        reciprocal_neighbors = ((neighbor_mask + neighbor_mask.t()) == 2).to(pair_similarity.dtype)
        reciprocal_counts = reciprocal_neighbors.sum(dim=1, keepdim=True).clamp_min(1)
        shared_neighbors = reciprocal_neighbors @ reciprocal_neighbors.t()
        contextual = (shared_neighbors / reciprocal_counts) * reciprocal_neighbors

        half_k = max(1, int(round(topk / 2)))
        contextual = contextual[topk_indices[:, :half_k]].mean(dim=1)
        return (contextual + contextual.t()) / 2

    def _kl_self_distillation(self, student_f, student_g):
        student_f = student_f.float()
        student_g = student_g.float()
        if self.normalize_student:
            student_f = F.normalize(student_f, p=2, dim=1)
            student_g = F.normalize(student_g, p=2, dim=1)
        distances_f = torch.cdist(student_f, student_f)
        distances_f = distances_f / distances_f.mean(dim=1, keepdim=True).clamp_min(self.eps)
        distances_g = torch.cdist(student_g, student_g)
        distances_g = distances_g / distances_g.mean(dim=1, keepdim=True).clamp_min(self.eps)
        return F.kl_div(
            F.log_softmax(-distances_f, dim=-1),
            F.softmax(-distances_g.detach(), dim=-1),
            reduction="sum",
        ) / len(student_f)


class MixedLabelPropagationProxyLoss(BaseMetricLossFunction):
    """Confidence-weighted proxy loss from Zhuang and Moulin, CVPR 2023.

    This implements equations (26)-(29) of "Deep Semi-supervised Metric
    Learning with Mixed Label Propagation". Sample weights are optional so the
    loss can also be used with fully supervised data or other SSL methods.
    """

    supports_sample_weights = True

    def __init__(self, num_classes, embedding_size, alpha=32.0, b=0.1, **kwargs):
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if embedding_size <= 0:
            raise ValueError("embedding_size must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if b < 0:
            raise ValueError("b must be non-negative")

        super().__init__(**kwargs)
        self.num_classes = int(num_classes)
        self.embedding_size = int(embedding_size)
        self.alpha = float(alpha)
        self.b = float(b)
        self.proxies = torch.nn.Parameter(torch.empty(self.num_classes, self.embedding_size))
        torch.nn.init.kaiming_normal_(self.proxies, mode="fan_out")
        self._sample_weights = None
        self.add_to_recordable_attributes(
            list_of_names=["num_classes", "embedding_size", "alpha", "b"],
            is_stat=False,
        )

    def forward(
        self,
        embeddings,
        labels=None,
        indices_tuple=None,
        ref_emb=None,
        ref_labels=None,
        sample_weights=None,
    ):
        if sample_weights is not None:
            if sample_weights.ndim != 1 or len(sample_weights) != len(embeddings):
                raise ValueError("sample_weights must be a vector aligned with embeddings")
            if not torch.isfinite(sample_weights).all():
                raise ValueError("sample_weights must be finite")
            if torch.any((sample_weights < 0) | (sample_weights > 1)):
                raise ValueError("sample_weights must be in [0, 1]")
            sample_weights = sample_weights.to(device=embeddings.device, dtype=embeddings.dtype)

        self._sample_weights = sample_weights
        try:
            return super().forward(embeddings, labels, indices_tuple, ref_emb, ref_labels)
        finally:
            self._sample_weights = None

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        if indices_tuple is not None:
            raise ValueError("MixedLabelPropagationProxyLoss does not support miners")
        if torch.any((labels < 0) | (labels >= self.num_classes)):
            raise ValueError("labels must be in [0, num_classes)")

        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_proxies = F.normalize(self.proxies, p=2, dim=1).to(dtype=embeddings.dtype)
        similarities = normalized_embeddings @ normalized_proxies.t()

        row_indices = torch.arange(len(labels), device=labels.device)
        positive_similarities = similarities[row_indices, labels]
        positive_loss = F.softplus(-self.alpha * (positive_similarities - self.b))

        negative_mask = ~F.one_hot(labels, num_classes=self.num_classes).bool()
        negative_losses = F.softplus(self.alpha * (similarities + self.b))
        negative_loss = (negative_losses * negative_mask).sum(dim=1)

        sample_weights = self._sample_weights
        if sample_weights is None:
            sample_weights = torch.ones_like(positive_loss)
        losses = sample_weights * (positive_loss + negative_loss)
        return {
            "loss": {
                "losses": losses,
                "indices": row_indices,
                "reduction_type": "element",
                # Equation (26) divides the summed per-sample objective by C.
                "divisor": self.num_classes,
            }
        }

    def get_default_reducer(self):
        return DivisorReducer()

    def get_logits(self, embeddings):
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        normalized_proxies = F.normalize(self.proxies, p=2, dim=1).to(dtype=embeddings.dtype)
        return normalized_embeddings @ normalized_proxies.t()


LOSS_REGISTRY = {
    "STMLLoss": STMLLoss,
    "MixedLabelPropagationProxyLoss": MixedLabelPropagationProxyLoss,
}
