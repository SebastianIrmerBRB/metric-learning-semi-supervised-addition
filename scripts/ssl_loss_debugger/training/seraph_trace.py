"""Debugger-only reconstruction of the pairs SERAPH scored in one step.

``SeraphRegularizer.batch_diagnostics`` reports a step's aggregates. This rebuilds
the individual pairs behind them, so the harness can name both endpoints, show the
augmented images that were actually compared, and check the per-pair
contributions against the loss the regularizer returned.

Nothing here re-runs the posterior. ``SeraphRegularizer.pair_terms`` hands over
the distances, logits, thresholds and per-pair loss terms the step actually
scored, and the block weights are read back out of the diagnostics. Recomputing
instead would move the very numbers the trace exists to report -
``SeraphPosteriorHead.resolve_threshold`` advances the running quantile estimate
as a side effect - and for the per-anchor threshold modes it could not be done at
all: ``eta`` is one value per anchor, merged across two directed posteriors per
pair, so no scalar diagnostic can carry it.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from training.ssl.seraph import (
    PER_ANCHOR_THRESHOLD_MODES,
    binary_entropy_from_logits,
    cross_squared_distances,
    pairwise_squared_distances,
    sharpen_logits,
)


# Embeddings are printed inline per sample, so only a readable prefix is kept.
MAX_TRACED_EMBEDDING_VALUES = 8

# Why a traced pair was picked. ``distance_spread`` keeps the listing
# representative of the block as a whole; ``same_label`` deliberately breaks that
# representativeness to surface the true positives, which a random pair listing
# almost never hits: with C roughly balanced classes in the unlabeled pool, only
# about 1 / (C - 1) of the pairs share a label.
DISTANCE_SPREAD_SELECTION = "distance_spread"
SAME_LABEL_SELECTION = "same_label"

# Sentinel for samples whose true label the debugger cannot resolve.
UNKNOWN_LABEL = -1


def _representative_indices(count: int, limit: int) -> list[int]:
    """Return up to ``limit`` indices spread evenly across ``range(count)``."""

    if limit <= 0 or count <= 0:
        return []
    if count <= limit:
        return list(range(count))
    if limit == 1:
        return [count // 2]
    step = (count - 1) / (limit - 1)
    return sorted({int(round(index * step)) for index in range(limit)})


def _sample_entry(
    role: str,
    batch_index: int,
    embedding: torch.Tensor,
    position: int | None,
    mapped_label: int | None,
) -> dict[str, Any]:
    values = embedding.detach().float().cpu().flatten()
    return {
        "role": role,
        "batch_index": int(batch_index),
        "source_position": None if position is None else int(position),
        "mapped_label": None if mapped_label is None else int(mapped_label),
        "embedding": values[:MAX_TRACED_EMBEDDING_VALUES].tolist(),
        "embedding_truncated": bool(values.numel() > MAX_TRACED_EMBEDDING_VALUES),
    }


def debug_true_labels(
    dataset_labels: Any,
    positions: torch.Tensor | None,
) -> torch.Tensor | None:
    """Look up the ground-truth labels behind a batch's dataset positions.

    Debugger-only: SERAPH never sees these. They exist so the trace can say which
    scored pairs were in fact positives.
    """

    if positions is None:
        return None
    labels = torch.as_tensor(dataset_labels, dtype=torch.long)
    return labels[torch.as_tensor(positions, dtype=torch.long).cpu()]


def _same_label_mask(
    left_indices: torch.Tensor,
    right_indices: torch.Tensor,
    left_labels: torch.Tensor | None,
    right_labels: torch.Tensor | None,
) -> torch.Tensor | None:
    """Return which of a block's pairs share a true label, or ``None`` if unknown."""

    if left_labels is None or right_labels is None:
        return None
    left = left_labels[left_indices]
    right = right_labels[right_indices]
    known = (left != UNKNOWN_LABEL) & (right != UNKNOWN_LABEL)
    return (left == right) & known


def _masked_statistics(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> float | None:
    return None if not bool(mask.any()) else float(values[mask].mean())


def _pair_endpoints(
    block: str,
    labeled_count: int,
    unlabeled_count: int,
) -> tuple[torch.Tensor, torch.Tensor, str, str]:
    """Return the batch indices behind each pair, in the block's own flat order."""

    if block == "uu":
        pairs = torch.triu_indices(unlabeled_count, unlabeled_count, offset=1)
        return pairs[0], pairs[1], "unlabeled", "unlabeled"
    # ``cross_squared_distances(labeled, unlabeled).flatten()`` is row-major over
    # the labeled rows, so pair k compares labeled k // n_u with unlabeled k % n_u.
    flat = torch.arange(labeled_count * unlabeled_count)
    return flat // unlabeled_count, flat % unlabeled_count, "labeled", "unlabeled"


def _block_terms(
    regularizer,
    head,
    diagnostics,
    supervised_embeddings,
    regularizer_embeddings,
    include_labeled_unlabeled,
):
    """Return each block's per-pair distances, logits and per-pair loss terms.

    The regularizer keeps the tensors it actually scored while batch diagnostics
    are on, so the trace reads those rather than rebuilding them. That is the
    only way to report a per-anchor ``eta``: ``self_tuning``, ``umap`` and
    ``tsne`` calibrate one threshold per anchor and merge two directed posteriors
    per pair, none of which survives a round trip through a scalar diagnostic.

    Older callers, and any step whose regularizer predates ``pair_terms``, fall
    back to rebuilding the logit from the block's scalar ``eta`` and ``kappa`` -
    correct for ``fixed``, ``quantile`` and ``labeled_logistic``, which have one
    threshold for the whole block.
    """

    stashed = regularizer.pair_terms() if hasattr(regularizer, "pair_terms") else {}
    if stashed:
        return {
            block: dict(terms, block_weight=diagnostics[f"train/seraph/{block}/block_weight"])
            for block, terms in stashed.items()
        }

    projected_unlabeled = head.project(regularizer_embeddings)
    distances = pairwise_squared_distances(projected_unlabeled)
    pairs = torch.triu_indices(
        len(projected_unlabeled),
        len(projected_unlabeled),
        offset=1,
        device=distances.device,
    )
    block_distances = {"uu": distances[pairs[0], pairs[1]]}
    if include_labeled_unlabeled:
        projected_labeled = head.project(supervised_embeddings)
        block_distances["lu"] = cross_squared_distances(
            projected_labeled,
            projected_unlabeled,
        ).flatten()

    if head.threshold_mode in PER_ANCHOR_THRESHOLD_MODES:
        raise NotImplementedError(
            f"the SERAPH comparison trace cannot rebuild threshold_mode="
            f"{head.threshold_mode!r} without the regularizer's own pair terms: "
            "eta is per anchor, and the diagnostics carry only its block mean"
        )

    terms = {}
    for block, squared_distances in block_distances.items():
        # The values this step ran with, straight from the regularizer.
        eta = diagnostics[f"train/seraph/{block}/threshold"]
        kappa = diagnostics[f"train/seraph/{block}/kappa"]
        # Written out rather than routed through ``head.posterior_logits``, whose
        # debug logging would fire a second time for the same step.
        logits = kappa * (eta - squared_distances)
        if regularizer.objective == "em":
            per_pair = F.binary_cross_entropy_with_logits(
                logits,
                torch.sigmoid(sharpen_logits(logits, regularizer.em_sharpening)),
                reduction="none",
            )
        else:
            per_pair = binary_entropy_from_logits(logits)
        terms[block] = {
            "squared_distances": squared_distances,
            "logits": logits,
            "pair_term": per_pair,
            "threshold": torch.full_like(squared_distances, float(eta)),
            "kappa": float(kappa),
            "block_weight": diagnostics[f"train/seraph/{block}/block_weight"],
        }
    return terms


def build_seraph_comparison(
    *,
    regularizer: Any,
    head: torch.nn.Module,
    diagnostics: dict[str, float],
    supervised_embeddings: torch.Tensor,
    regularizer_embeddings: torch.Tensor,
    supervised_positions: torch.Tensor | None,
    supervised_labels: torch.Tensor | None,
    unlabeled_positions: torch.Tensor,
    max_pairs_per_block: int,
    true_labels: dict[str, torch.Tensor | None] | None = None,
) -> dict[str, Any]:
    """Rebuild one SERAPH step's pair table from the step's own diagnostics."""

    if not diagnostics:
        return {}

    include_labeled_unlabeled = bool(regularizer.include_labeled_unlabeled)
    role_labels = true_labels or {}
    with torch.no_grad():
        block_terms = _block_terms(
            regularizer,
            head,
            diagnostics,
            supervised_embeddings,
            regularizer_embeddings,
            include_labeled_unlabeled,
        )

        labeled_count = len(supervised_embeddings)
        unlabeled_count = len(regularizer_embeddings)
        samples: dict[str, list[dict[str, Any]]] = {
            "unlabeled": [
                _sample_entry(
                    "unlabeled",
                    index,
                    regularizer_embeddings[index],
                    int(unlabeled_positions[index]),
                    None,
                )
                for index in range(unlabeled_count)
            ],
            "labeled": [],
        }
        if include_labeled_unlabeled:
            samples["labeled"] = [
                _sample_entry(
                    "labeled",
                    index,
                    supervised_embeddings[index],
                    None if supervised_positions is None else int(supervised_positions[index]),
                    None if supervised_labels is None else int(supervised_labels[index]),
                )
                for index in range(labeled_count)
            ]

        loss_pairs: list[dict[str, Any]] = []
        loss_blocks: list[dict[str, Any]] = []
        reconstructed_loss = 0.0
        for block, terms in block_terms.items():
            squared_distances = terms["squared_distances"]
            logits = terms["logits"]
            per_pair = terms["pair_term"]
            # One eta per pair, whether the mode calibrated one for the block or
            # one per anchor.
            eta = terms["threshold"]
            kappa = terms["kappa"]
            block_weight = terms["block_weight"]

            targets = (
                torch.sigmoid(sharpen_logits(logits, regularizer.em_sharpening))
                if regularizer.objective == "em"
                else None
            )
            entropy = binary_entropy_from_logits(logits)
            probabilities = torch.sigmoid(logits)
            pair_count = len(squared_distances)
            reconstructed_loss += block_weight * float(per_pair.mean())

            left_indices, right_indices, left_role, right_role = _pair_endpoints(
                block,
                labeled_count,
                unlabeled_count,
            )
            same_label = _same_label_mask(
                left_indices,
                right_indices,
                role_labels.get(left_role),
                role_labels.get(right_role),
            )
            if same_label is not None:
                # The labels are debugger bookkeeping and live on the CPU; the
                # distances they index into do not.
                same_label = same_label.to(squared_distances.device)
            distance_order = torch.argsort(squared_distances)
            block_rank = torch.empty_like(distance_order)
            block_rank[distance_order] = torch.arange(
                pair_count,
                device=distance_order.device,
            )

            # Even spacing over the distance-sorted pairs keeps the listing
            # representative of the block instead of showing one corner of it.
            selections: dict[int, list[str]] = {}
            for rank in _representative_indices(pair_count, max_pairs_per_block):
                selections.setdefault(int(distance_order[rank]), []).append(
                    DISTANCE_SPREAD_SELECTION
                )
            # The same spread again, but restricted to the true positives, which
            # the representative listing is far too sparse to show.
            same_label_rank: dict[int, int] = {}
            same_label_count = None
            if same_label is not None:
                same_label_order = distance_order[same_label[distance_order]]
                same_label_count = int(len(same_label_order))
                for rank in _representative_indices(
                    same_label_count,
                    max_pairs_per_block,
                ):
                    index = int(same_label_order[rank])
                    same_label_rank[index] = rank
                    selections.setdefault(index, []).append(SAME_LABEL_SELECTION)

            for index in sorted(
                selections,
                key=lambda index: (
                    SAME_LABEL_SELECTION in selections[index],
                    int(block_rank[index]),
                ),
            ):
                loss_pairs.append(
                    {
                        "kind": block,
                        "selection": selections[index],
                        "left_role": left_role,
                        "left_index": int(left_indices[index]),
                        "right_role": right_role,
                        "right_index": int(right_indices[index]),
                        "distance_rank": int(block_rank[index]),
                        "block_pair_count": pair_count,
                        "same_label": (
                            None if same_label is None else bool(same_label[index])
                        ),
                        "same_label_rank": same_label_rank.get(index),
                        "same_label_pair_count": same_label_count,
                        "eta": float(eta[index]),
                        "kappa": float(kappa),
                        "squared_distance": float(squared_distances[index]),
                        "logit": float(logits[index]),
                        "same_probability": float(probabilities[index]),
                        "entropy": float(entropy[index]),
                        "target": None if targets is None else float(targets[index]),
                        "pair_term": float(per_pair[index]),
                        # What this one pair adds to the returned loss: the block
                        # is a mean, so each pair carries 1 / pair_count of it.
                        "loss_contribution": block_weight
                        * float(per_pair[index])
                        / pair_count,
                    }
                )
            positive = (probabilities > 0.5).float()
            block_entry = {
                "kind": block,
                "batch_pair_count": pair_count,
                "population_pair_count": _population_pair_count(regularizer, block),
                # A per-anchor mode has no single eta, so the block reports the
                # mean and the spread of what its pairs actually used.
                "eta": float(eta.mean()),
                "eta_std": float(eta.std(unbiased=False)),
                "kappa": float(kappa),
                "mean_squared_distance": float(squared_distances.mean()),
                "mean_entropy": float(entropy.mean()),
                "mean_pair_term": float(per_pair.mean()),
                "mean_absolute_logit": float(logits.abs().mean()),
                "positive_pair_rate": float(positive.mean()),
                "block_weight": float(block_weight),
                "listed_pairs": sum(1 for pair in loss_pairs if pair["kind"] == block),
                "same_label_pair_count": same_label_count,
            }
            if same_label is not None:
                # Whether the posterior separates the true positives at all: with
                # no separation both d² means coincide and the positive rates
                # match the block-wide one.
                different_label = ~same_label
                block_entry.update(
                    {
                        "same_label_mean_squared_distance": _masked_statistics(
                            squared_distances, same_label
                        ),
                        "different_label_mean_squared_distance": _masked_statistics(
                            squared_distances, different_label
                        ),
                        "same_label_mean_entropy": _masked_statistics(
                            entropy, same_label
                        ),
                        "different_label_mean_entropy": _masked_statistics(
                            entropy, different_label
                        ),
                        "same_label_positive_rate": _masked_statistics(
                            positive, same_label
                        ),
                        "different_label_positive_rate": _masked_statistics(
                            positive, different_label
                        ),
                    }
                )
            loss_blocks.append(block_entry)

    return {
        "objective": regularizer.objective,
        "em_sharpening": float(regularizer.em_sharpening),
        "metric": regularizer.metric,
        "threshold_mode": head.threshold_mode,
        "kappa_mode": head.kappa_mode,
        # The configured kappa, which is what each block's per-step kappa above is
        # derived from: batch_std divides this by the batch's distance spread.
        "kappa": float(head.kappa),
        # Both only shape eta in quantile mode; fixed mode returns the configured
        # threshold and never consults either.
        "positive_pair_prior": float(head.positive_pair_prior),
        "threshold_momentum": float(head.threshold_momentum),
        "include_labeled_unlabeled": include_labeled_unlabeled,
        "samples": samples,
        "loss_pairs": loss_pairs,
        "loss_blocks": loss_blocks,
        "reconstructed_loss": reconstructed_loss,
    }


def _population_pair_count(regularizer: Any, block: str) -> float | None:
    """Return the pool-level pair count the block's weight is derived from."""

    unlabeled_pool = regularizer.unlabeled_pool_size
    labeled_pool = regularizer.labeled_pool_size
    if unlabeled_pool is None:
        return None
    if block == "uu":
        return unlabeled_pool * (unlabeled_pool - 1) / 2
    return None if labeled_pool is None else float(labeled_pool) * unlabeled_pool
