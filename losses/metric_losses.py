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
from pytorch_metric_learning.losses import CircleLoss as UpstreamCircleLoss
from pytorch_metric_learning.losses import MultiSimilarityLoss as UpstreamMultiSimilarityLoss
from pytorch_metric_learning.losses import NTXentLoss as UpstreamNTXentLoss
from pytorch_metric_learning.losses import ProxyAnchorLoss as UpstreamProxyAnchorLoss
from pytorch_metric_learning.losses import TripletMarginLoss as UpstreamTripletMarginLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner, TripletMarginMiner
from pytorch_metric_learning.reducers import AvgNonZeroReducer, DivisorReducer
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


def _dense_pair_masks(labels):
    """Return all positive/negative pair masks without dynamic-size indices."""

    same_class = labels.unsqueeze(1) == labels.unsqueeze(0)
    negative_mask = ~same_class
    positive_mask = same_class
    positive_mask.fill_diagonal_(False)
    return positive_mask, negative_mask


def _miner_weight_dtype(embeddings):
    """The dtype ``lmu.convert_to_weights`` returns for these embeddings."""

    return c_f.to_dtype(
        torch.empty(0, device=embeddings.device),
        dtype=embeddings.dtype,
    ).dtype


def _promotes(tensor, other_dtype):
    """Whether combining ``tensor`` with an ``other_dtype`` tensor changes its dtype."""

    return torch.promote_types(tensor.dtype, other_dtype) != tensor.dtype


def _masked_logsumexp(x, keep_mask, add_one=True, dim=1):
    """``lmu.logsumexp`` selecting with ``where``; also returns the row-``any``.

    Upstream negates the keep mask twice -- once over the whole matrix to
    ``masked_fill`` it, once over the reduced ``any`` to zero empty rows.
    ``torch.where(keep, x, fill)`` writes exactly the values
    ``x.masked_fill(~keep, fill)`` writes, and its backward selects the same
    gradient, so the negations are pure launch overhead. The ``any`` is
    returned because callers that zero whole rows need exactly that predicate.
    """

    has_kept = torch.any(keep_mask, dim=dim, keepdim=True)
    x = torch.where(keep_mask, x, c_f.neg_inf(x.dtype))
    if add_one:
        zeros = torch.zeros(x.size(dim - 1), dtype=x.dtype, device=x.device).unsqueeze(
            dim
        )
        x = torch.cat([x, zeros], dim=dim)
    output = torch.logsumexp(x, dim=dim, keepdim=True)
    return torch.where(has_kept, output, 0), has_kept


def _all_pairs_indices(labels, ref_labels=None):
    """``lmu.get_all_pairs_indices`` with one host synchronization instead of two.

    Each ``torch.where`` on a device mask has to copy its output size to the
    host. Both sizes come from one transfer here and ``nonzero_static`` then
    compacts each mask with its known size, returning the same row-major index
    tensors.
    """

    matches, diffs = lmu.get_matches_and_diffs(labels, ref_labels)
    num_matches, num_diffs = torch.stack((matches.sum(), diffs.sum())).tolist()
    a1_idx, p_idx = torch.nonzero_static(matches, size=num_matches).unbind(1)
    a2_idx, n_idx = torch.nonzero_static(diffs, size=num_diffs).unbind(1)
    return a1_idx, p_idx, a2_idx, n_idx


class DenseMaskNTXentLoss(UpstreamNTXentLoss):
    """Bit-identical ``NTXentLoss`` without upstream's masking round trips.

    Upstream scores every positive pair against every negative pair, a
    ``(P, N)`` matrix that is 768 x 64,512 at batch 256 with four samples per
    class. It selects each row's own-anchor negatives by casting the anchor
    match to float, multiplying, and then overwriting the rest in place, and
    autograd has to undo each of those steps in the backward pass. One
    ``where`` writes the same matrix: ``x * 1`` is ``x`` exactly and the
    discarded entries are overwritten either way. The matrix, and every
    reduction over it, keeps upstream's shape and layout, so each sum
    associates identically.

    Only the unmined path also builds its pair indices with a single host
    synchronization. Mined tuples reach the same loss core unchanged, so SLADE
    may still hand this criterion to ``bounded_ntxent_loss``.
    """

    same_objective_as_upstream_ntxent = True

    def compute_loss(self, embeddings, labels, indices_tuple, ref_emb, ref_labels):
        if indices_tuple is not None:
            return super().compute_loss(
                embeddings, labels, indices_tuple, ref_emb, ref_labels
            )
        c_f.labels_required(labels)
        indices_tuple = _all_pairs_indices(labels, ref_labels)
        if all(len(x) <= 1 for x in indices_tuple):
            return self.zero_losses()
        mat = self.distance(embeddings, ref_emb)
        return self.loss_method(mat, indices_tuple)

    def _compute_loss(self, pos_pairs, neg_pairs, indices_tuple):
        a1, p, a2, _ = indices_tuple

        if len(a1) > 0 and len(a2) > 0:
            dtype = neg_pairs.dtype
            if not self.distance.is_inverted:
                pos_pairs = -pos_pairs
                neg_pairs = -neg_pairs

            pos_pairs = pos_pairs.unsqueeze(1) / self.temperature
            neg_pairs = neg_pairs / self.temperature
            neg_pairs = torch.where(
                a2.unsqueeze(0) == a1.unsqueeze(1),
                neg_pairs,
                c_f.neg_inf(dtype),
            )

            max_val = torch.max(
                pos_pairs, torch.max(neg_pairs, dim=1, keepdim=True)[0]
            ).detach()
            numerator = torch.exp(pos_pairs - max_val).squeeze(1)
            denominator = torch.sum(torch.exp(neg_pairs - max_val), dim=1) + numerator
            log_exp = torch.log((numerator / denominator) + c_f.small_val(dtype))
            return {
                "loss": {
                    "losses": -log_exp,
                    "indices": (a1, p),
                    "reduction_type": "pos_pair",
                }
            }
        return self.zero_losses()


class _IndexedAvgNonZeroReducer(AvgNonZeroReducer):
    """``AvgNonZeroReducer`` whose backward needs no host synchronization.

    Boolean-mask indexing compacts with ``nonzero`` in the forward pass and
    again in the backward pass, where autograd scatters the gradient through
    the same boolean mask. Indexing with the forward pass's ``nonzero`` result
    selects the same elements in the same order, and hands the backward
    integer indices it can scatter without asking the host for their count.
    The compacted mean itself is unchanged, so its reduction order is too.
    """

    def element_reduction_helper(self, losses, embeddings):
        low_condition, high_condition = None, None
        if self.low is not None:
            low_condition = losses > self.low
            losses = losses[low_condition.nonzero(as_tuple=True)]
        if self.high is not None:
            high_condition = losses < self.high
            losses = losses[high_condition.nonzero(as_tuple=True)]
        num_past_filter = len(losses)
        if num_past_filter >= 1:
            loss = torch.mean(losses)
        else:
            loss = self.zero_loss(embeddings)
        self.set_stats(low_condition, high_condition, num_past_filter)
        return loss


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
        losses = self.cross_entropy(logits, labels)
        # Without a miner every weight is exactly one, and multiplying by one
        # is exact in both directions, so the weights are skipped unless the
        # multiply would promote the per-sample losses.
        if indices_tuple is not None or _promotes(
            losses,
            _miner_weight_dtype(embeddings),
        ):
            losses = losses * lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
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

    def _compute_loss(self, mat, pos_mask, neg_mask):
        """Upstream's loss core with :func:`_masked_logsumexp` selecting."""

        pos_exp = self.distance.margin(mat, self.base)
        neg_exp = self.distance.margin(self.base, mat)
        pos_loss = (1.0 / self.alpha) * _masked_logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )[0]
        neg_loss = (1.0 / self.beta) * _masked_logsumexp(
            self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
        )[0]
        return {
            "loss": {
                "losses": pos_loss + neg_loss,
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }

    def supports_fused_miner(self, miner):
        return (
            type(miner) is MultiSimilarityMiner
            and not self.collect_stats
            and not miner.collect_stats
            and not self.distance.collect_stats
            and not miner.distance.collect_stats
            and self._matching_distance_configuration(self.distance, miner.distance)
        )

    def forward_with_pair_masks(self, embeddings, labels, positive_mask, negative_mask):
        """Score disjoint, symmetric mined masks without compacting pair indices.

        SLADE already selects pairs on the device. Converting those masks into
        variable-length indices and back forces host synchronizations without
        changing the loss. Keep upstream's loss core and reducer, including its
        embedding regularization. Symmetric masks contain either zero or at
        least two directed pairs, matching GenericPairLoss's empty-pair guard.
        """

        self.reset_stats()
        c_f.check_shapes(embeddings, labels)
        labels = c_f.to_device(labels, embeddings)
        similarity = self.distance(embeddings, embeddings)
        loss_dict = self._compute_loss(similarity, positive_mask, negative_mask)
        self.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
        return self.reducer(loss_dict, embeddings, labels)

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

    The mined loss vector and ``AvgNonZeroReducer``'s compact-then-mean are
    intentionally kept.  Replacing that reduction with a masked dense sum
    changes floating-point reduction order, so its forward compaction is the
    price of bitwise identity; only its backward synchronization is removed
    (see :class:`_IndexedAvgNonZeroReducer`).
    """

    # The mined tuple has a data-dependent length, so this path cannot be put in
    # the fixed-shape CUDA graph used by FusedMultiSimilarityLoss.
    supports_fused_miner_compilation = False

    def get_default_reducer(self):
        return _IndexedAvgNonZeroReducer()

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

    def get_default_reducer(self):
        return _IndexedAvgNonZeroReducer()

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

        logsumexp_pos, has_positive = _masked_logsumexp(
            new_mat, keep_mask=pos_mask_bool, add_one=False, dim=1
        )
        logsumexp_neg, has_negative = _masked_logsumexp(
            new_mat, keep_mask=neg_mask_bool, add_one=False, dim=1
        )

        losses = self.soft_plus(logsumexp_pos + logsumexp_neg)

        # Upstream zeroes whole rows that have no positive or no negative pair,
        # reaching them through nonzero indices. Same selection, as a mask; a
        # 0/1 mask row sums to zero exactly when it has no set entry.
        keep = (has_positive & has_negative).to(losses.dtype)
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
        cosine = self.get_logits(embeddings)
        positive_logits = self.alpha * self.distance.margin(cosine, self.margin)
        negative_logits = self.alpha * self.distance.margin(-self.margin, cosine)
        # Without a miner every offset is exactly +0.0. Adding it returns each
        # logit unchanged except that -0.0 becomes +0.0, which the logsumexp
        # below cannot tell apart (both give exp(x - max) == exp(-max)), and
        # the add's backward passes the gradient through. So it is skipped --
        # unless the add promotes: under autocast the weights stay float32
        # while the logits are low precision, and the promotion is what moves
        # the logsumexp to float32.
        if indices_tuple is not None or _promotes(
            positive_logits,
            _miner_weight_dtype(embeddings),
        ):
            miner_weights = lmu.convert_to_weights(
                indices_tuple,
                labels,
                dtype=dtype,
            ).unsqueeze(1)
            miner_weights = miner_weights - 1
            positive_logits = positive_logits + miner_weights
            negative_logits = negative_logits + miner_weights

        positive_mask = F.one_hot(labels, self.num_classes).bool()
        negative_mask = ~positive_mask
        positive_term, has_positive = _masked_logsumexp(
            positive_logits,
            keep_mask=positive_mask,
            add_one=True,
            dim=0,
        )
        negative_term, _ = _masked_logsumexp(
            negative_logits,
            keep_mask=negative_mask,
            add_one=True,
            dim=0,
        )
        represented_classes = has_positive.sum()
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
    # Shadow synchronization-heavy upstream implementations under their same
    # public names, so configs, checkpoints, and HPO spaces remain unchanged.
    # SLADE's mined unlabeled NTXent stream still goes through
    # losses.bounded_ntxent to avoid the P-by-N allocation altogether.
    "NTXentLoss": DenseMaskNTXentLoss,
    "ArcFaceLoss": SyncFreeArcFaceLoss,
    "MultiSimilarityLoss": FusedMultiSimilarityLoss,
    "TripletMarginLoss": FusedTripletMarginLoss,
    "CircleLoss": SyncFreeCircleLoss,
    "ProxyAnchorLoss": SyncFreeProxyAnchorLoss,
}
