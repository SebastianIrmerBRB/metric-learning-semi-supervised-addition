"""Project-local metric-learning losses."""

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_metric_learning.distances import (
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
    SNRDistance,
)
from pytorch_metric_learning.losses import ArcFaceLoss as UpstreamArcFaceLoss
from pytorch_metric_learning.losses import BaseMetricLossFunction
from pytorch_metric_learning.losses import CircleLoss as UpstreamCircleLoss
from pytorch_metric_learning.losses import MultiSimilarityLoss as UpstreamMultiSimilarityLoss
from pytorch_metric_learning.losses import ProxyAnchorLoss as UpstreamProxyAnchorLoss
from pytorch_metric_learning.losses import TripletMarginLoss as UpstreamTripletMarginLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner, TripletMarginMiner
from pytorch_metric_learning.reducers import DivisorReducer
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu


_upstream_to_dtype = c_f.to_dtype


def _device_aware_to_dtype(x, tensor=None, dtype=None):
    """``c_f.to_dtype`` with a guard that sees the tensor's own autocast state."""

    if torch.is_tensor(x) and torch.is_autocast_enabled(x.device.type):
        return x
    if not torch.is_tensor(x) and torch.is_autocast_enabled():
        return x
    dt = dtype if dtype is not None else tensor.dtype
    if x.dtype != dt:
        x = x.type(dt)
    return x


def install_device_aware_autocast_guard():
    """Give CPU runs the same autocast behaviour pytorch-metric-learning gives CUDA.

    ``c_f.to_dtype`` is how pml pushes tensors to the embedding dtype, and it
    deliberately does nothing while autocast is active so that autocast alone
    decides precision. Its guard calls ``torch.is_autocast_enabled()`` with no
    argument, which reports the *CUDA* autocast state only, so under this
    trainer's BF16 autocast the guard holds on CUDA and silently lapses on CPU.

    Two places in a proxy or classifier loss then diverge by device:

    * ``cast_types`` rewrites the loss's learnable weights to the embedding
      dtype. On CPU they become BF16 permanently, and are trained from then on
      -- along with their Adam moments, whose ``beta2=0.999`` decay is finer
      than BF16 can represent -- in eight bits of mantissa. CUDA keeps FP32.
    * ``convert_to_weights`` builds the miner-weight vector in FP32. CUDA
      leaves it FP32, which promotes ProxyAnchor's ``logsumexp`` accumulation
      to FP32; CPU drops it to BF16 and accumulates there instead.

    Making the guard device-aware reproduces the CUDA path on CPU at every call
    site. Losses without learnable weights were never affected either way.
    """

    if c_f.to_dtype is not _upstream_to_dtype:
        return
    c_f.to_dtype = _device_aware_to_dtype


install_device_aware_autocast_guard()


def classification_loss_float32(
    criterion,
    embeddings,
    labels,
    *,
    sample_weights=None,
):
    """Evaluate a proxy/classification loss outside mixed precision.

    Angular classifiers such as ArcFace take ``acos`` of a cosine similarity.
    BF16 autocast can round a near-one cosine to exactly one, leaving a finite
    forward loss but producing non-finite gradients at the ``acos`` endpoint.
    """

    with torch.autocast(device_type=embeddings.device.type, enabled=False):
        embeddings = embeddings.float()
        if sample_weights is not None:
            return criterion(
                embeddings,
                labels,
                sample_weights=sample_weights.float(),
            )
        return criterion(embeddings, labels)


def stml_contextual_similarity(pair_similarity, topk, instance_ids=None):
    """Compute the contextualized teacher similarity used by STML.

    Section 3.2 of Kim et al. (CVPR 2022): reciprocal nearest neighbors define a
    neighborhood per sample, the shared-neighbor overlap between two such
    neighborhoods becomes their contextual similarity, and query expansion over
    each sample's closest ``topk / 2`` neighbors smooths the result.

    ``instance_ids`` marks views of the same source image so they always rank as
    neighbors of each other; pass ``None`` for single-view batches.
    """

    num_samples = len(pair_similarity)
    topk = min(int(topk), num_samples)
    ranking_similarity = pair_similarity.clone()
    if instance_ids is not None:
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


def stml_teacher_pair_weights(teacher_embeddings, sigma, topk, instance_ids=None):
    """Return STML's teacher relation weight ``w = (s + c) / 2`` for every pair.

    ``s`` is the Gaussian kernel on teacher distances and ``c`` is
    :func:`stml_contextual_similarity`. STML reads ``w`` as the probability that
    two samples share a class: its relaxed contrastive loss pulls with ``w`` and
    pushes with ``1 - w``, and threshold-based variants cut it into hard
    relations instead.
    """

    teacher = F.normalize(teacher_embeddings.float(), p=2, dim=1)
    teacher_distances = torch.cdist(teacher, teacher)
    pair_similarity = torch.exp(-teacher_distances.square() / sigma)
    contextual_similarity = stml_contextual_similarity(pair_similarity, topk, instance_ids)
    return (pair_similarity + contextual_similarity) / 2


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
            weights = stml_teacher_pair_weights(teacher_g, self.sigma, self.topk, instance_ids)
            off_diagonal = ~torch.eye(len(student), device=student.device, dtype=torch.bool)
            positive_weights = weights.masked_fill(~off_diagonal, 0)
            negative_weights = (1 - weights).masked_fill(~off_diagonal, 0)

        pull = student_distances.square() * positive_weights
        push = F.relu(self.delta - student_distances).square() * negative_weights
        return (pull.sum() + push.sum()) / off_diagonal.sum()

    def _contextual_similarity(self, pair_similarity, instance_ids):
        return stml_contextual_similarity(pair_similarity, self.topk, instance_ids)

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
            sample_weights = sample_weights.to(device=embeddings.device, dtype=torch.float32)

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

        similarities = self._similarities(embeddings)

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

    def _similarities(self, embeddings):
        """Return the proxy cosine similarities in float32.

        Everything after this reads the result through ``alpha``, which is 32 by
        default. That scaling turns bfloat16's rounding near a cosine of 1 into a
        visible shift in the softplus, so the similarities are computed in
        float32 and the loss follows them.

        Autocast has to be switched off rather than cast around: it casts matmul
        down to bfloat16 even from float32 operands, so promoting the result
        afterwards returns a float32 tensor holding a bfloat16 cosine and buys
        nothing. This mirrors ``classification_loss_float32`` above, and
        SLADE/SERAPH/SimMatchV2 guard their own Gram matmuls the same way.

        Casting both operands rather than the proxies alone also keeps the two
        dtypes agreeing when autocast is off and the embedding is not float32,
        which is what the previous downcast to ``embeddings.dtype`` guarded.
        """

        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            normalized_embeddings = F.normalize(embeddings.float(), p=2, dim=1)
            normalized_proxies = F.normalize(self.proxies.float(), p=2, dim=1)
            return normalized_embeddings @ normalized_proxies.t()

    def get_default_reducer(self):
        return DivisorReducer()

    def get_logits(self, embeddings):
        return self._similarities(embeddings)


def _dense_pair_masks(labels):
    """Return all positive/negative pair masks without dynamic-size indices."""

    same_class = labels.unsqueeze(1) == labels.unsqueeze(0)
    negative_mask = ~same_class
    positive_mask = same_class
    positive_mask.fill_diagonal_(False)
    return positive_mask, negative_mask


class SyncFreeArcFaceLoss(UpstreamArcFaceLoss):
    """ArcFace with fixed-shape target selection and logit replacement."""

    def __init__(
        self,
        num_classes,
        embedding_size,
        margin=28.6,
        scale=64,
        **kwargs,
    ):
        super().__init__(
            num_classes=num_classes,
            embedding_size=embedding_size,
            margin=margin,
            scale=scale,
            **kwargs,
        )
        # Upstream stores these constants as NumPy scalars. AOTAutograd lifts
        # those as CPU graph inputs, which prevents CUDA graph capture. Device
        # scalar buffers produce the same dtype conversion and exact values.
        self.register_buffer(
            "_device_margin",
            torch.as_tensor(self.margin),
            persistent=False,
        )
        self.register_buffer(
            "_device_monotonic_threshold",
            torch.as_tensor(np.deg2rad(180) - self.margin),
            persistent=False,
        )
        self.register_buffer(
            "_device_monotonic_offset",
            torch.as_tensor(self.margin * np.sin(self.margin)),
            persistent=False,
        )

    @staticmethod
    def _scalar_operands(tensor, device_scalar):
        """Reproduce Tensor/NumPy-scalar arithmetic without a host scalar.

        Upstream stores ArcFace's constants as NumPy float64 scalars.  CUDA
        evaluates a low-precision tensor/host-scalar operation in FP32 and
        casts the result back, whereas a CUDA zero-dimensional tensor operand
        is rounded to FP16/BF16 *before* the operation.  That one-rounding
        difference is observable in both the loss and its gradient.  Promote
        just this CUDA low-precision case to CUDA's opmath dtype; all other
        dtypes already give the same arithmetic with the device scalar.
        """

        if tensor.device.type == "cuda" and tensor.dtype in (
            torch.float16,
            torch.bfloat16,
        ):
            return tensor.float(), device_scalar.float(), tensor.dtype
        return tensor, device_scalar, None

    def modify_cosine_of_target_classes(self, cosine_of_target_classes):
        angles = self.get_angles(cosine_of_target_classes)
        scalar_angles, margin, cast_dtype = self._scalar_operands(
            angles,
            self._device_margin,
        )
        angles_with_margin = scalar_angles + margin
        if cast_dtype is not None:
            angles_with_margin = angles_with_margin.to(cast_dtype)
        cosine_with_margin = torch.cos(angles_with_margin)
        cosine = torch.cos(angles)

        comparison_angles, threshold, _ = self._scalar_operands(
            angles,
            self._device_monotonic_threshold,
        )
        scalar_cosine, offset, cast_dtype = self._scalar_operands(
            cosine,
            self._device_monotonic_offset,
        )
        monotonic_fallback = scalar_cosine - offset
        if cast_dtype is not None:
            monotonic_fallback = monotonic_fallback.to(cast_dtype)
        return torch.where(
            comparison_angles <= threshold,
            cosine_with_margin,
            monotonic_fallback,
        )

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)

        cosine = self.get_cosine(embeddings)
        target_cosine = cosine.gather(1, labels.unsqueeze(1)).squeeze(1)
        modified_target = self.modify_cosine_of_target_classes(target_cosine)
        target_delta = (modified_target - target_cosine).unsqueeze(1)
        target_mask = F.one_hot(labels, num_classes=self.num_classes).to(dtype=dtype)
        # Keep upstream's add/multiply graph.  scatter_add has the same forward
        # values in FP32, but its FP16 backward associates the target-gradient
        # additions differently and can move them by one representable value.
        logits = cosine + (target_mask * target_delta)
        logits = self.scale_logits(logits, embeddings)
        losses = self.cross_entropy(logits, labels) * miner_weights
        loss_dict = {
            "loss": {
                "losses": losses,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element",
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, self.W.t())
        return loss_dict


class FusedMultiSimilarityLoss(UpstreamMultiSimilarityLoss):
    """Multi-similarity loss with a dense, synchronization-free miner path.

    Values and gradients match upstream bit for bit whenever the similarity
    matrix is finite.  One corner is knowingly left different: upstream returns
    its zero loss *before* building the similarity graph when the miner emits
    at most one pair of each sign, so a NaN similarity matrix cannot reach the
    embedding gradient there.  Reading that pair count is exactly the host
    synchronization this path exists to remove, so the mask below applies the
    same condition on-device and the graph is always built.  Both agree the
    loss is zero; with a NaN similarity matrix the fast path then propagates
    NaN into a gradient upstream leaves as exact zeros.  Reaching it needs a
    batch that is effectively single-class *and* an already non-finite
    similarity matrix, at which point training has diverged regardless.
    """

    @staticmethod
    def _matching_distance_configuration(left, right):
        attributes = ("normalize_embeddings", "p", "power", "is_inverted")
        supported_types = (
            CosineSimilarity,
            DotProductSimilarity,
            LpDistance,
            SNRDistance,
        )
        return type(left) is type(right) and type(left) in supported_types and all(
            getattr(left, name, None) == getattr(right, name, None)
            for name in attributes
        )

    def supports_fused_miner(self, miner):
        return (
            type(miner) is MultiSimilarityMiner
            and not self.collect_stats
            and not miner.collect_stats
            and not self.distance.collect_stats
            and not miner.distance.collect_stats
            and self._matching_distance_configuration(self.distance, miner.distance)
        )

    def forward_with_miner(self, embeddings, labels, miner):
        """Compute mining and loss from one similarity matrix.

        The upstream miner sorts masked rows and emits dynamic-size index
        vectors. The loss then recomputes the same similarity matrix and turns
        those vectors back into masks. Row extrema express the miner's two
        predicates directly and preserve the exact selected-pair masks.
        """

        if not self.supports_fused_miner(miner):
            raise ValueError("the supplied miner cannot use the fused multi-similarity path")

        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        labels = c_f.to_device(labels, embeddings)
        positive_mask, negative_mask = _dense_pair_masks(labels)
        similarity = self.distance(embeddings, embeddings)

        with torch.no_grad():
            mining_similarity = similarity.detach()
            # ``torch.sort`` orders NaN last, so the miner's *first* sorted
            # entry is the smallest non-NaN value while its *last* entry is NaN
            # as soon as the row holds one.  ``amin``/``amax`` both propagate
            # NaN, so only the ``amin`` side needs NaN folded into the ignore
            # sentinel to reproduce that asymmetry.  Every row already carries
            # the sentinel on its diagonal, so an all-NaN row still reduces to
            # the sentinel exactly as the sorted row does.
            if self.distance.is_inverted:
                hardest_positive = mining_similarity.masked_fill(
                    ~positive_mask | mining_similarity.isnan(),
                    float("inf"),
                ).amin(dim=1, keepdim=True)
                hardest_negative = mining_similarity.masked_fill(
                    ~negative_mask,
                    float("-inf"),
                ).amax(dim=1, keepdim=True)
                selected_positive = positive_mask & (
                    mining_similarity - miner.epsilon < hardest_negative
                )
                selected_negative = negative_mask & (
                    mining_similarity + miner.epsilon > hardest_positive
                )
            else:
                hardest_positive = mining_similarity.masked_fill(
                    ~positive_mask,
                    float("-inf"),
                ).amax(dim=1, keepdim=True)
                hardest_negative = mining_similarity.masked_fill(
                    ~negative_mask | mining_similarity.isnan(),
                    float("inf"),
                ).amin(dim=1, keepdim=True)
                selected_positive = positive_mask & (
                    mining_similarity + miner.epsilon > hardest_negative
                )
                selected_negative = negative_mask & (
                    mining_similarity - miner.epsilon < hardest_positive
                )

            # GenericPairLoss deliberately returns a zero loss when both mined
            # pair vectors contain at most one item.  Usually the miner emits
            # many pairs, which hid this upstream edge case in the original
            # dense implementation.  Apply the same global condition to the
            # masks on-device so the fast path remains synchronization-free.
            keep_mined_pairs = (selected_positive.sum() > 1) | (
                selected_negative.sum() > 1
            )
            selected_positive &= keep_mined_pairs
            selected_negative &= keep_mined_pairs

        loss_dict = self._compute_loss(
            similarity,
            selected_positive,
            selected_negative,
        )
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)


class FusedTripletMarginLoss(UpstreamTripletMarginLoss):
    """Triplet-margin loss with a lower-synchronization miner path.

    ``TripletMarginMiner`` first compacts every valid label triplet, then
    boolean-indexes each of its three index vectors to compact the violations.
    On CUDA that requires four device-to-host shape synchronizations before the
    loss starts.  Applying the same predicates to the dense triplet mask and
    compacting once produces the same lexicographically ordered index tuple
    with one synchronization.

    The mined loss vector and ``AvgNonZeroReducer`` are intentionally kept in
    their upstream form.  Replacing that compact-then-mean reduction with a
    masked dense sum changes floating-point reduction order, so the remaining
    reducer synchronizations are the price of bitwise identity.
    """

    # The mined tuple has a data-dependent length, so this path cannot be put in
    # the fixed-shape CUDA graph used by FusedMultiSimilarityLoss.
    supports_fused_miner_compilation = False

    def supports_fused_miner(self, miner):
        return (
            type(miner) is TripletMarginMiner
            and not self.collect_stats
            and not miner.collect_stats
            and not self.distance.collect_stats
            and not miner.distance.collect_stats
            and FusedMultiSimilarityLoss._matching_distance_configuration(
                self.distance,
                miner.distance,
            )
        )

    @staticmethod
    def _selected_triplet_mask(labels, distance_matrix, miner):
        positive_mask, negative_mask = _dense_pair_masks(labels)
        mining_distance = distance_matrix.detach()
        positive_distance = mining_distance.unsqueeze(2)
        negative_distance = mining_distance.unsqueeze(1)
        triplet_margin = (
            positive_distance - negative_distance
            if miner.distance.is_inverted
            else negative_distance - positive_distance
        )
        valid_triplet = positive_mask.unsqueeze(2) & negative_mask.unsqueeze(1)

        if miner.type_of_triplets == "easy":
            return valid_triplet & (triplet_margin > miner.margin)

        selected = valid_triplet & (triplet_margin <= miner.margin)
        if miner.type_of_triplets == "hard":
            selected &= triplet_margin <= 0
        elif miner.type_of_triplets == "semihard":
            selected &= triplet_margin > 0
        return selected

    def forward_with_miner(self, embeddings, labels, miner):
        """Compute the miner and loss from one distance matrix."""

        if not self.supports_fused_miner(miner):
            raise ValueError("the supplied miner cannot use the fused triplet-margin path")

        self.reset_stats()
        miner.reset_stats()
        miner.distance.reset_stats()
        c_f.check_shapes(embeddings, labels)
        labels = c_f.to_device(labels, embeddings)
        distance_matrix = self.distance(embeddings, embeddings)

        with torch.no_grad():
            indices_tuple = torch.where(
                self._selected_triplet_mask(labels, distance_matrix, miner)
            )
        miner.output_assertion(indices_tuple)

        anchor_idx, positive_idx, negative_idx = indices_tuple
        if len(anchor_idx) == 0:
            loss_dict = self.zero_losses()
        else:
            anchor_positive = distance_matrix[anchor_idx, positive_idx]
            anchor_negative = distance_matrix[anchor_idx, negative_idx]
            if self.swap:
                positive_negative = distance_matrix[positive_idx, negative_idx]
                anchor_negative = self.distance.smallest_dist(
                    anchor_negative,
                    positive_negative,
                )

            current_margins = self.distance.margin(
                anchor_positive,
                anchor_negative,
            )
            violation = current_margins + self.margin
            losses = (
                F.softplus(violation)
                if self.smooth_loss
                else F.relu(violation)
            )
            loss_dict = {
                "loss": {
                    "losses": losses,
                    "indices": indices_tuple,
                    "reduction_type": "triplet",
                }
            }

        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)


class SyncFreeCircleLoss(UpstreamCircleLoss):
    """Bit-identical ``CircleLoss`` with pair-construction syncs removed.

    Upstream's ``_compute_loss`` reaches for the pair entries with boolean-mask
    gathers (``mat[pos_mask_bool]``), writes them back with masked assignment,
    and selects the degenerate rows with ``torch.where(cond)[0]``. Every one of
    those lowers to ``nonzero``, which cannot report its output size without
    copying it to the host -- 18 device-to-host round trips per training step,
    against 5 for MultiSimilarity and 2 for NTXent.

    On a frozen backbone the step is latency-bound, not compute-bound, so those
    round trips are the loss's dominant cost. The upstream reducer is retained
    because replacing its compact-then-mean reduction changes floating-point
    order; it still performs a smaller dynamic selection after this loss core.

    This computes both pair terms and the pair masks densely, trading a few
    FLOPs on an ``(N, N)`` matrix for fewer round trips. The masks are disjoint
    by construction, so selecting positive over negative over zero rebuilds
    the same matrix. Both branches are finite polynomials in ``mat``, so the
    unselected branch cannot poison the gradient with a NaN.
    """

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_or_indices_tuple_required(labels, indices_tuple)
        if indices_tuple is not None or ref_emb is not embeddings:
            return super().compute_loss(
                embeddings,
                labels,
                indices_tuple,
                ref_emb,
                ref_labels,
            )
        positive_mask, negative_mask = _dense_pair_masks(labels)
        similarity = self.distance(embeddings, ref_emb)
        return self._compute_loss(similarity, positive_mask, negative_mask)

    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_mask_bool = pos_mask.bool()
        neg_mask_bool = neg_mask.bool()

        # Upstream applies these to the gathered entries; the gather is what
        # costs a sync, and the expression is elementwise, so this is the same
        # value computed at every position and then selected.
        positive_term = (
            -self.gamma * torch.relu(self.op - mat.detach()) * (mat - self.delta_p)
        )
        negative_term = (
            self.gamma * torch.relu(mat.detach() - self.on) * (mat - self.delta_n)
        )
        new_mat = torch.where(
            pos_mask_bool,
            positive_term,
            torch.where(neg_mask_bool, negative_term, torch.zeros_like(mat)),
        )

        logsumexp_pos = lmu.logsumexp(
            new_mat, keep_mask=pos_mask_bool, add_one=False, dim=1
        )
        logsumexp_neg = lmu.logsumexp(
            new_mat, keep_mask=neg_mask_bool, add_one=False, dim=1
        )

        losses = self.soft_plus(logsumexp_pos + logsumexp_neg)

        # Upstream zeroes whole rows that have no positive or no negative pair,
        # reaching them through nonzero indices. Same selection, as a mask.
        keep = (torch.sum(pos_mask, dim=1) != 0) & (torch.sum(neg_mask, dim=1) != 0)
        keep = keep.to(losses.dtype)
        while keep.dim() < losses.dim():
            keep = keep.unsqueeze(-1)
        losses = losses * keep

        return {
            "loss": {
                "losses": losses,
                "indices": c_f.torch_arange_from_size(new_mat),
                "reduction_type": "element",
            }
        }


class _DeviceDivisorReducer(DivisorReducer):
    """Divisor reduction whose tensor divisor stays entirely on the device."""

    @staticmethod
    def _divide_like_upstream_scalar(numerator, divisor):
        """Match ``numerator / int(divisor)`` without reading the device count.

        CUDA compiles floating-point division by a Python scalar as a multiply
        by its rounded reciprocal.  Division by a zero-dimensional CUDA tensor
        uses a division instruction instead and differs by one ULP for many
        values.  Computing the reciprocal on-device restores the scalar result
        for FP32/FP64; CUDA low precision and CPU use ordinary division, which
        is what their scalar overloads use.
        """

        typed_divisor = divisor.to(numerator.dtype)
        if numerator.device.type == "cuda" and numerator.dtype in (
            torch.float32,
            torch.float64,
        ):
            return numerator * torch.reciprocal(typed_divisor)
        return numerator / typed_divisor

    def sum_and_divide(self, losses, embeddings, divisor):
        if torch.is_tensor(divisor):
            safe_divisor = divisor.clamp_min(1)
            output = self._divide_like_upstream_scalar(
                torch.sum(losses),
                safe_divisor,
            )
            if losses.dtype == torch.float16:
                promoted_losses = c_f.to_dtype(losses, dtype=torch.float32)
                float32_output = self._divide_like_upstream_scalar(
                    torch.sum(promoted_losses),
                    safe_divisor,
                )
                output = torch.where(
                    torch.isnan(output),
                    float32_output.to(dtype=torch.float16),
                    output,
                )
            return torch.where(
                divisor != 0,
                output,
                self.zero_loss(embeddings),
            )
        if divisor == 0:
            return self.zero_loss(embeddings)

        output = torch.sum(losses) / divisor
        if losses.dtype == torch.float16:
            promoted_losses = c_f.to_dtype(losses, dtype=torch.float32)
            float32_output = torch.sum(promoted_losses) / divisor
            output = torch.where(
                torch.isnan(output),
                float32_output.to(dtype=torch.float16),
                output,
            )
        return output


class SyncFreeProxyAnchorLoss(UpstreamProxyAnchorLoss):
    """ProxyAnchor with a device-side count of represented classes."""

    def get_default_reducer(self):
        return _DeviceDivisorReducer()

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        c_f.labels_required(labels)
        c_f.ref_not_supported(embeddings, labels, ref_emb, ref_labels)
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(
            indices_tuple,
            labels,
            dtype=dtype,
        ).unsqueeze(1)
        miner_weights = miner_weights - 1

        cosine = self.get_logits(embeddings)
        positive_mask = F.one_hot(labels, self.num_classes)
        negative_mask = 1 - positive_mask
        represented_classes = (positive_mask.sum(dim=0) != 0).sum()

        positive_term = lmu.logsumexp(
            (self.alpha * self.distance.margin(cosine, self.margin)) + miner_weights,
            keep_mask=positive_mask.bool(),
            add_one=True,
            dim=0,
        )
        negative_term = lmu.logsumexp(
            (self.alpha * self.distance.margin(-self.margin, cosine)) + miner_weights,
            keep_mask=negative_mask.bool(),
            add_one=True,
            dim=0,
        )
        loss_indices = c_f.torch_arange_from_size(self.proxies)
        loss_dict = {
            "pos_loss": {
                "losses": positive_term.squeeze(0),
                "indices": loss_indices,
                "reduction_type": "element",
                "divisor": represented_classes,
            },
            "neg_loss": {
                "losses": negative_term.squeeze(0),
                "indices": loss_indices,
                "reduction_type": "element",
                "divisor": self.num_classes,
            },
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, self.proxies)
        return loss_dict


LOSS_REGISTRY = {
    "STMLLoss": STMLLoss,
    "MixedLabelPropagationProxyLoss": MixedLabelPropagationProxyLoss,
    # NTXentLoss is intentionally absent: production must use the untouched
    # upstream implementation so existing runs remain exactly reproducible.
    # Shadow synchronization-heavy upstream implementations under their same
    # public names, so configs, checkpoints, and HPO spaces remain unchanged.
    "ArcFaceLoss": SyncFreeArcFaceLoss,
    "MultiSimilarityLoss": FusedMultiSimilarityLoss,
    "TripletMarginLoss": FusedTripletMarginLoss,
    "CircleLoss": SyncFreeCircleLoss,
    "ProxyAnchorLoss": SyncFreeProxyAnchorLoss,
}
