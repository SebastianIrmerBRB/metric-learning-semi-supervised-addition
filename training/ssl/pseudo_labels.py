"""Pseudo-label filtering, relabeling, and quality diagnostics."""

import json
from pathlib import Path

import numpy as np
from loguru import logger

from .config import PSEUDO_LABEL_DIAGNOSTICS_MODES, PseudoLabelResult
from .data import RelabeledSubset


class PseudoLabelDiagnosticsTracker:
    """Track pseudo-label quality, distribution, and stability diagnostics."""

    def __init__(self, log_dir, mode="save"):
        if mode not in PSEUDO_LABEL_DIAGNOSTICS_MODES - {"off"}:
            raise ValueError(
                f"pseudo-label diagnostics mode must be one of {sorted(PSEUDO_LABEL_DIAGNOSTICS_MODES - {'off'})}"
            )
        self.mode = mode
        self.path = Path(log_dir) / "pseudo_label_diagnostics.jsonl" if mode == "save" else None
        self.previous_raw = None
        self.previous_accepted = None
        self.generation_index = 0

    def log(self, raw_pseudo_labels, accepted_pseudo_labels, train_dataset, config, epoch=None):
        true_labels = np.asarray(train_dataset.labels, dtype=np.int64)
        raw_summary = summarize_pseudo_label_result(raw_pseudo_labels, true_labels)
        accepted_summary = summarize_pseudo_label_result(accepted_pseudo_labels, true_labels)
        raw_changes = summarize_pseudo_label_changes(self.previous_raw, raw_pseudo_labels)
        accepted_changes = summarize_pseudo_label_changes(self.previous_accepted, accepted_pseudo_labels)

        record = {
            "generation_index": self.generation_index,
            "epoch": epoch,
            "method": config.method,
            "confidence_threshold": config.confidence_threshold,
            "raw": raw_summary,
            "accepted": accepted_summary,
            "raw_changes_from_previous_generation": raw_changes,
            "accepted_changes_from_previous_generation": accepted_changes,
            "audit_note": "Hidden labels are used only for post-prediction diagnostics, never for pseudo-label generation.",
        }
        if self.path is not None:
            with self.path.open("a") as jsonl_file:
                jsonl_file.write(json.dumps(record, sort_keys=True) + "\n")

        accepted_correctness = accepted_summary.get("confidence_correctness")
        accepted_auc = (
            None if not accepted_correctness else accepted_correctness.get("auc")
        )
        accepted_auc_text = "n/a" if accepted_auc is None else f"{accepted_auc:.3f}"

        logger.info(
            "Pseudo-label diagnostics: "
            f"raw={raw_summary['count']}, accepted={accepted_summary['count']}, "
            f"accepted_audit_accuracy={format_optional_metric(accepted_summary['audit_accuracy'])}, "
            f"accepted_confidence_auc={accepted_auc_text}, "
            f"accepted_confidence_mean={format_optional_metric(accepted_summary['confidence']['mean'])}, "
            f"accepted_changes={format_change_summary(accepted_changes)}"
        )
        self.previous_raw = pseudo_labels_to_position_map(raw_pseudo_labels)
        self.previous_accepted = pseudo_labels_to_position_map(accepted_pseudo_labels)
        self.generation_index += 1


def filter_pseudo_labels(pseudo_labels, confidence_threshold, valid_mapped_labels):
    """Drop low-confidence predictions and labels unknown to the train mapping."""

    # Start with predictions that refer to a class represented in the current
    # training label mapping. This protects against invalid estimator outputs.
    keep = np.isin(pseudo_labels.mapped_labels, list(valid_mapped_labels))

    if pseudo_labels.confidences is not None:
        # Combine conditions elementwise so positions, labels, and confidences
        # remain aligned after boolean indexing.
        keep = keep & (pseudo_labels.confidences >= confidence_threshold)

    dropped = int(len(keep) - keep.sum())
    if dropped > 0:
        logger.info(f"Dropped {dropped} pseudo-labels below confidence threshold or outside known classes")

    # Apply the same mask to every aligned result array.
    return PseudoLabelResult(
        positions=pseudo_labels.positions[keep],
        mapped_labels=pseudo_labels.mapped_labels[keep],
        confidences=None if pseudo_labels.confidences is None else pseudo_labels.confidences[keep],
    )


def _average_ranks(values):
    """Return 1-based ranks with ties broken to their average (Mann-Whitney AUC)."""

    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    n = len(values)
    sorted_ranks = np.empty(n, dtype=np.float64)
    index = 0
    while index < n:
        end = index
        while end + 1 < n and sorted_values[end + 1] == sorted_values[index]:
            end += 1
        sorted_ranks[index : end + 1] = (index + end) / 2.0 + 1.0
        index = end + 1
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = sorted_ranks
    return ranks


def confidence_correctness_diagnostics(correct, confidences, num_buckets=10):
    """Measure whether the confidence weight omega actually ranks correctness.

    The paper's per-sample loss weighting only protects training if confident
    pseudo-labels are more likely to be correct. This returns:
      - ``auc``: P(omega of a correct label > omega of an incorrect one), tie-aware.
        0.5 means omega is uninformative (the failure mode where confident-but-wrong
        labels get full weight); values well above 0.5 mean omega is usable.
      - ``buckets``: equal-count groups ordered low->high omega, each with its count,
        omega range/mean, and audit accuracy. A monotone accuracy rise across buckets
        confirms omega is calibrated; a flat profile means it is not.
    Returns None when confidences are absent or correctness is single-class.
    """

    if confidences is None:
        return None
    correct = np.asarray(correct, dtype=bool)
    confidences = np.asarray(confidences, dtype=np.float64)
    n = len(correct)
    if n == 0 or len(confidences) != n:
        return None

    n_correct = int(correct.sum())
    n_wrong = n - n_correct
    if n_correct == 0 or n_wrong == 0:
        auc = None
    else:
        ranks = _average_ranks(confidences)
        auc = float(
            (ranks[correct].sum() - n_correct * (n_correct + 1) / 2.0) / (n_correct * n_wrong)
        )

    num_buckets = max(1, min(int(num_buckets), n))
    order = np.argsort(confidences, kind="mergesort")
    confidence_sorted = confidences[order]
    correct_sorted = correct[order]
    edges = np.linspace(0, n, num_buckets + 1).astype(int)
    buckets = []
    for bucket_index in range(num_buckets):
        low, high = int(edges[bucket_index]), int(edges[bucket_index + 1])
        if high <= low:
            continue
        segment_confidence = confidence_sorted[low:high]
        buckets.append(
            {
                "quantile": f"{low / n:.2f}-{high / n:.2f}",
                "count": int(high - low),
                "confidence_min": float(segment_confidence.min()),
                "confidence_max": float(segment_confidence.max()),
                "confidence_mean": float(segment_confidence.mean()),
                "accuracy": float(correct_sorted[low:high].mean()),
            }
        )

    return {"auc": auc, "n_correct": n_correct, "n_wrong": n_wrong, "buckets": buckets}


def summarize_pseudo_label_result(pseudo_labels, true_labels):
    """Return JSON-safe distribution and hidden-label audit metrics."""

    positions = np.asarray(pseudo_labels.positions, dtype=np.int64)
    predicted_labels = np.asarray(pseudo_labels.mapped_labels, dtype=np.int64)
    if len(positions) == 0:
        audit_accuracy = None
        true_class_counts = {}
        correct_counts_by_true_class = {}
        audit_accuracy_by_true_class = {}
        confidence_correctness = None
    else:
        audit_labels = true_labels[positions]
        correct = predicted_labels == audit_labels
        audit_accuracy = float(np.mean(correct))
        confidence_correctness = confidence_correctness_diagnostics(correct, pseudo_labels.confidences)
        true_class_counts = count_values(audit_labels)
        correct_counts_by_true_class = {
            str(int(label)): int(correct[audit_labels == label].sum())
            for label in np.unique(audit_labels)
        }
        audit_accuracy_by_true_class = {
            label: float(correct_counts_by_true_class[label] / count)
            for label, count in true_class_counts.items()
        }

    return {
        "count": int(len(positions)),
        "predicted_class_counts": count_values(predicted_labels),
        "true_class_counts": true_class_counts,
        "correct_counts_by_true_class": correct_counts_by_true_class,
        "audit_accuracy": audit_accuracy,
        "audit_accuracy_by_true_class": audit_accuracy_by_true_class,
        "confidence": summarize_numeric_values(pseudo_labels.confidences),
        "confidence_correctness": confidence_correctness,
    }


def summarize_pseudo_label_changes(previous, current):
    """Compare position-aligned pseudo-label predictions across generations."""

    if previous is None:
        return None
    current_map = pseudo_labels_to_position_map(current)
    previous_positions = set(previous)
    current_positions = set(current_map)
    overlapping_positions = previous_positions & current_positions
    changed_count = sum(previous[position] != current_map[position] for position in overlapping_positions)
    return {
        "overlap_count": int(len(overlapping_positions)),
        "changed_count": int(changed_count),
        "changed_fraction": None
        if not overlapping_positions
        else float(changed_count / len(overlapping_positions)),
        "added_count": int(len(current_positions - previous_positions)),
        "removed_count": int(len(previous_positions - current_positions)),
    }


def pseudo_labels_to_position_map(pseudo_labels):
    return {
        int(position): int(label)
        for position, label in zip(pseudo_labels.positions, pseudo_labels.mapped_labels)
    }


def count_values(values):
    values = np.asarray(values)
    if len(values) == 0:
        return {}
    unique, counts = np.unique(values, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(unique, counts)}


def summarize_numeric_values(values):
    if values is None:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {
            "count": 0,
            "min": None,
            "p25": None,
            "median": None,
            "p75": None,
            "max": None,
            "mean": None,
            "std": None,
        }
    return {
        "count": int(len(values)),
        "min": float(np.min(values)),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
    }


def format_optional_metric(value):
    return "n/a" if value is None else f"{value:.4f}"


def format_change_summary(changes):
    if changes is None:
        return "n/a"
    return (
        f"{changes['changed_count']}/{changes['overlap_count']} changed, "
        f"{changes['added_count']} added, {changes['removed_count']} removed"
    )


def make_relabeled_training_dataset(
    train_dataset,
    train_labels_mapper,
    labeled_positions,
    pseudo_labels,
    return_indices=False,
):
    """Combine true and pseudo labels into the dataset view used for training."""

    # orig_labels contains source class IDs aligned with train_dataset positions.
    # Derive true-label dense IDs from the supplied mapper instead of reusing
    # train_dataset.labels: supervised class-subset runs deliberately narrow the
    # mapper after support selection, while the source dataset still carries its
    # pre-budget dense IDs.
    orig_labels = np.asarray(train_dataset.orig_labels, dtype=np.int64)
    # Pseudo-labelers predict dense mapped labels.  Convert them back to
    # original labels because RelabeledSubset.__getitem__ follows the same
    # contract as the source training dataset.
    inverse_labels_mapper = {mapped: original for original, mapped in train_labels_mapper.items()}

    # Build all result arrays in the same order: true-labeled samples first,
    # followed by accepted pseudo-labeled samples.
    all_positions = np.concatenate([labeled_positions, pseudo_labels.positions])
    # True-labeled positions use their known source labels remapped through the
    # active training mapper; pseudo-labeled positions already use its dense IDs.
    labeled_orig_labels = orig_labels[labeled_positions]
    try:
        labeled_mapped_labels = np.asarray(
            [train_labels_mapper[int(label)] for label in labeled_orig_labels],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(
            f"True-labeled class {int(exc.args[0])} is absent from the active training label mapper"
        ) from exc
    all_mapped_labels = np.concatenate([labeled_mapped_labels, pseudo_labels.mapped_labels])
    pseudo_orig_labels = np.asarray(
        [inverse_labels_mapper[int(label)] for label in pseudo_labels.mapped_labels],
        dtype=np.int64,
    )
    # RelabeledSubset returns original IDs, so concatenate known source labels
    # with inverse-mapped predicted source labels in the same sample order.
    all_orig_labels = np.concatenate([orig_labels[labeled_positions], pseudo_orig_labels])
    pseudo_confidences = (
        np.ones(len(pseudo_labels.positions), dtype=np.float32)
        if pseudo_labels.confidences is None
        else np.asarray(pseudo_labels.confidences, dtype=np.float32)
    )
    all_confidences = np.concatenate(
        [np.ones(len(labeled_positions), dtype=np.float32), pseudo_confidences]
    )

    return RelabeledSubset(
        dataset=train_dataset,
        positions=all_positions,
        orig_labels=all_orig_labels,
        mapped_labels=all_mapped_labels,
        confidences=all_confidences,
        return_indices=return_indices,
        labeled_count=len(labeled_positions),
    )
