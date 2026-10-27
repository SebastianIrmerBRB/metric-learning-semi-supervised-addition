"""NT-Xent on mined pairs without the upstream positive-by-negative product."""

import torch
from pytorch_metric_learning.losses import NTXentLoss
from pytorch_metric_learning.utils import common_functions as c_f


def computes_upstream_ntxent(criterion):
    """Whether ``criterion`` evaluates exactly upstream's NTXent objective.

    That is ``NTXentLoss`` itself, or a subclass that only changes how the
    objective is computed and says so with ``same_objective_as_upstream_ntxent``
    (``losses.metric_losses.DenseMaskNTXentLoss``, which the loss registry
    builds for the name ``NTXentLoss``). Any other subclass keeps its forward.
    """

    criterion_type = type(criterion)
    return criterion_type is NTXentLoss or (
        issubclass(criterion_type, NTXentLoss)
        and criterion_type.__dict__.get("same_objective_as_upstream_ntxent", False)
    )


def bounded_ntxent_loss(criterion, embeddings, labels, indices_tuple):
    """Evaluate an upstream NTXentLoss using O(B² + P + N) storage.

    Upstream masks a P-by-N matrix to select the negatives sharing each
    positive pair's anchor. Sum those negatives once per anchor instead. Every
    positive keeps its own numerator and denominator; other positives are NOT
    included in that denominator. Pair multiplicities, the configured distance,
    reducer, embedding regularizer and upstream's tiny-probability floor remain
    intact. Reassociated floating-point sums need not be bit-identical.

    This adapter is used only by SLADE's mined unlabeled objective. It does not
    replace the supervised loss or assume that custom NTXent subclasses use the
    upstream objective.
    """

    if not computes_upstream_ntxent(criterion):
        raise TypeError("bounded_ntxent_loss requires an unmodified NTXentLoss")
    if len(indices_tuple) != 4:
        raise ValueError("bounded_ntxent_loss requires a four-tensor pair tuple")
    criterion.reset_stats()
    c_f.check_shapes(embeddings, labels)
    if labels is not None:
        labels = c_f.to_device(labels, embeddings)
    a_pos, positives, a_neg, negatives = indices_tuple

    # Preserve GenericPairLoss's degenerate-pair check, as well as NTXent's
    # requirement for at least one positive and one negative globally.
    loss_dict = criterion.zero_losses()
    if not all(len(indices) <= 1 for indices in indices_tuple):
        scores = criterion.distance(embeddings, embeddings)
        if len(a_pos) > 0 and len(a_neg) > 0:
            if not criterion.distance.is_inverted:
                scores = -scores
            scores = scores / criterion.temperature

            # A mined pair set normally has no duplicates. Counts also preserve
            # upstream behavior if a caller intentionally repeats a negative.
            counts = torch.zeros_like(scores)
            counts.index_put_(
                (a_neg, negatives),
                torch.ones_like(a_neg, dtype=scores.dtype),
                accumulate=True,
            )
            negative_mask = counts > 0
            negative_scores = scores.masked_fill(~negative_mask, c_f.neg_inf(scores.dtype))
            negative_max = negative_scores.max(dim=1).values.detach()
            negative_sum = (
                torch.exp(negative_scores - negative_max[:, None]) * counts
            ).sum(dim=1)

            positive_scores = scores[a_pos, positives]
            pair_max = torch.maximum(positive_scores, negative_max[a_pos]).detach()
            numerator = torch.exp(positive_scores - pair_max)
            # Move each anchor's negative sum into this positive's coordinate
            # system. Rows with no negatives have sum zero and give loss zero.
            denominator = numerator + negative_sum[a_pos] * torch.exp(
                negative_max[a_pos] - pair_max
            )
            pair_losses = -torch.log(
                numerator / denominator + c_f.small_val(scores.dtype)
            )
            loss_dict = {
                "loss": {
                    "losses": pair_losses,
                    "indices": (a_pos, positives),
                    "reduction_type": "pos_pair",
                }
            }

    criterion.add_embedding_regularization_to_loss_dict(loss_dict, embeddings)
    return criterion.reducer(loss_dict, embeddings, labels)
