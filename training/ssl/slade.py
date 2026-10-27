"""SLADE self-training regularization for deep metric learning.

Implements the student-side objective of "SLADE: A Self-Training Framework For
Distance Metric Learning" (Duan et al., CVPR 2021) with the paper's equation
numbers used throughout:

* Sec 3.2 - a teacher snapshot embeds the unlabeled pool and k-means cluster IDs
  become pseudo labels.
* Sec 3.3.1 / Eq 2-4 - feature basis learning. A learnable ``k x d`` basis matrix
  ``W_a`` maps the retrieval embedding ``f`` to a feature representation
  ``r = W_a f``. The basis is supervised by cross entropy on labeled data (Eq 3)
  and by a global similarity-distribution loss on pseudo-labeled pairs (Eq 5),
  whose two Gaussians are tracked with the momentum update of Eq 6.
* Sec 3.3.2 / Eq 7 - the basis similarity ``s = cos(r_i, r_j)`` mines
  high-confidence unlabeled positive and negative pairs at thresholds
  ``T1 = mu+`` and ``T2 = mu-``.
* Sec 3.3.3 / Eq 9 - the total objective
  ``L_rank(D^l) + lambda1 * L_rank(D^u) + lambda2 * L_Basis``.

Three departures from the paper follow from this project's setup rather than from
a choice about SLADE; ``docs/slade_implementation_vs_paper.md`` records them in
full alongside the ones that are genuine method differences.

* **No dataset-specific self-supervised stage.** Sec 3.1 initializes the teacher
  with SwAV and fine-tunes it on the union of labeled and unlabeled images. Here
  every run starts from the same frozen DINOv2 features. Neither the paper's
  ImageNet nor SwAV rows are directly comparable to that setup.
* **The teacher is a projection-head snapshot.** SLADE never consults the
  teacher between clustering events -- unlike STML, whose EMA teacher scores
  every step. With a frozen backbone, a saved projection head plus the shared
  DINO features is the complete teacher. The default schedule represents
  promotion by re-clustering with the current student. The opt-in
  ``within_fold_teacher_student`` lifecycle instead makes the boundary
  explicit: labeled-only teacher training, one fixed clustering, then student
  training from the teacher weights with fresh optimization state. Folds remain
  independent unless ``cross_fold_teacher_handoff`` is enabled separately.

  The default schedule puts on ``update_interval_epochs`` the length of one
  self-training iteration: the teacher and its clusters are frozen for exactly
  that long. The paper iterates "a few times" over a full student training run,
  so an interval of ``1`` is the one setting that genuinely departs from it --
  the student then re-clusters with a head the previous clustering just shaped.
* **k-means features come from the frozen backbone.** The teacher can only move
  the pool's embedding as far as the head allows, so cluster assignments drift
  much less between refreshes than they would with a tunable backbone.
"""

import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from pytorch_metric_learning import losses

import utils
from losses.bounded_ntxent import bounded_ntxent_loss, computes_upstream_ntxent
from .algorithms import faiss_gpu_flat_index, require_faiss
from .common import kmeans, parameter_validators
from .config import should_rebuild_on_epoch
from .data import CombinedTrainingLoader, UnlabeledSubset
from .embeddings import extract_embeddings
from .interfaces import BaseTrainingRegularizer

_validate_positive, _validate_non_negative, _, _ = parameter_validators("slade")


KMEANS_BACKENDS = ("auto", "faiss", "sklearn")

# Eq 1's hinges are gates, not scales, so a margin only matters where it changes
# which pairs are active. On L2-normalized embeddings ``d = sqrt(2 - 2cos)``, and
# unrelated pairs pile up near ``sqrt(2) = 1.414``: measured DINOv2-B Cars196
# features have a pairwise median of 1.35, with under 1.4% of pairs below 0.8.
# A negative margin beneath this floor therefore buys no repulsion at all.
CONTRASTIVE_NEG_MARGIN_LIVE_FLOOR = 1.2

# The same ceiling ``LRMLRegularizer.MAX_USEFUL_CONTRASTIVE_MARGIN`` records:
# two L2-normalized embeddings are at most 2 apart (antipodal), so a negative
# margin above this can never be satisfied and the repulsion never stops
# pushing. This is the bound that is genuinely about satisfiability -- unlike
# ``m_neg > m_pos``, which is about enforcing a separation gap.
MAX_USEFUL_CONTRASTIVE_MARGIN = 2.0

DEFAULT_KMEANS_GPU_MIN_WORK = 1_000_000

# Eq 9's unlabeled term applies a pair loss to the pairs Eq 7 mined.
#
# ``supervised`` reuses the configured supervised criterion, which is what Eq 9
# literally writes -- one ``L_rank`` on both streams. It ties SLADE to whatever
# the run's metric loss is, so a proxy or classification loss cannot be used at
# all and ``validate_run_args`` rejects one outright.
#
# ``contrastive`` gives the unlabeled term its own Eq 1 contrastive loss instead
# of borrowing the supervised one. Eq 1 is the paper's own worked example of
# ``L_rank`` ("for example, a constrastive loss [12]"), and Table 1 reports SLADE
# with it, so this is a re-parameterization of Eq 9 rather than a new objective.
# Decoupling the streams is what lets the supervised half be a classification
# loss: the mined pairs no longer have to be something that loss can consume.
UNLABELED_RANKING_LOSSES = ("supervised", "contrastive")

# How much of Eq 5's gradient reaches ``W_a`` through Eq 6's interpolation.
#
# ``damped`` is the literal composition: the stored half of Eq 6 is history and
# carries no gradient, so every path into ``W_a`` picks up the ``(1 - beta)``
# factor on the batch half. That makes beta a second weight on L_SD relative to
# the L_CE it is summed with in Eq 2 -- at the paper's beta = 0.99, L_SD pulls on
# the basis with 1% of L_CE's scale, and no downstream weight can restore the
# balance because lambda2 multiplies their sum.
#
# ``full`` keeps the same forward value -- Eq 5 still reads the global Gaussians,
# which is the paper's stated reason for using them -- and cancels the factor
# from the gradient alone, so beta means only what the paper calls it: the
# updating rate of Eq 6.
SD_GRADIENT_MODES = ("full", "damped")

# How the unlabeled batch is drawn.
#
# ``random`` is a uniform shuffle of the pool. Only ~1/k of a batch's pairs then
# share a cluster, whatever the batch size, so Eq 6's G+ is estimated from a few
# pairs per batch. Eq 7 thresholds on basis similarity alone, and the unrelated
# pairs outnumber the pseudo-positive ones ~k to 1, so their upper tail outweighs
# the true positives. Measured, ~25% of mined positives shared a cluster.
#
# ``pseudo_label_m_per_class`` draws ``unlabeled_batch_size / unlabeled_sampler_m``
# distinct clusters per batch and ``unlabeled_sampler_m`` members of each. That
# is how the framework the paper builds on (Musgrave et al. [21]) samples every
# stream, and the paper's "we use the pseudo labels to sample a set of pseudo
# positive pairs and pseudo negative pairs" most plausibly means it. The paper
# does not say so outright and no reference code is public, so ``random`` stays
# the default and this is an ablation.
UNLABELED_SAMPLERS = ("random", "pseudo_label_m_per_class")
DEFAULT_UNLABELED_SAMPLER_M = 4

# Which unlabeled pairs Eq 9's ranking term trains on. Table 4 of the paper
# builds SLADE up from a student trained on pseudo-label pairs alone, through
# "+ Basis", to "+ Basis + Mining"; the last one is the method.
#
# ``basis_mining`` is Eq 7: the pairs the basis similarity is confident about.
#
# ``pseudo_label`` is Table 4's "Student (pseudo label)" row: every same-cluster
# pair is a positive and every cross-cluster pair a negative, and the basis only
# regularizes through L_Basis. With ``lambda_basis = 0`` nothing trains the basis
# at all, which is the row without "+ Basis".
#
# ``pseudo_label_matched`` draws pseudo-label pairs at random, as many positives
# and negatives as Eq 7 mines in the same batch, and only on the steps where Eq 7
# mines. Eq 7 keeps fewer and easier pairs than ``pseudo_label``, so the two
# differ in both the number and the choice of pairs. This control keeps the
# number and changes only the choice.
PAIR_SOURCES = ("basis_mining", "pseudo_label", "pseudo_label_matched")

# The unlabeled half of Eq 2's L_Basis. Table 5 compares the paper's similarity
# distribution loss with two cross-entropy alternatives on CUB-200.
#
# ``sd`` is Eq 5 on the Eq 6 Gaussians.
#
# ``local_ce`` labels each batch pair 1 (same cluster) or 0 (different cluster)
# and takes the binary cross entropy of its basis cosine, read as the
# probability ``(1 + s) / 2``. The paper does not say how that probability is
# formed or how the two pair kinds are weighted. Each kind is averaged on its
# own and the two means are averaged, as Eq 5 treats G+ and G- as one term each
# whatever their pair counts; a plain mean would be almost all negatives.
#
# ``global_ce`` applies the same cross entropy to the Eq 6 global means, µ+
# against 1 and µ- against 0.
#
# ``none`` drops the unlabeled half, so L_Basis is Eq 3 alone. The Gaussians
# still update, because Eq 7 reads them.
BASIS_UNLABELED_LOSSES = ("sd", "local_ce", "global_ce", "none")

# Used when neither unlabeled_batch_size nor unlabeled_ratio is configured: one
# unlabeled sample per labeled sample, the paper's 32/32 split at batch_size=32.
DEFAULT_BASIS_WARMUP_STEPS = 100

DEFAULT_UNLABELED_RATIO = 1.0


# One label for Eq 2's basis term, shared by its diagnostics and its calibration
# so both report the same objective term under the same name.
SLADE_BASIS_COMPONENT = "slade_basis"
# The module ``configure_model`` attaches to the student model for W_a.
SLADE_BASIS_MODULE = "slade_basis"


# Eq 9's unlabeled term feeds the Eq 7 mined pairs straight into the configured
# criterion as a pytorch-metric-learning ``indices_tuple``. These losses reject
# that argument outright, on top of every entry in ``CLASSIFICATION_LOSSES``,
# which scores embeddings against learned proxies instead of against pairs.
PAIR_INCOMPATIBLE_LOSSES = frozenset(
    {
        "InstanceLoss",
        "ManifoldLoss",
        "P2SGradLoss",
        "PNPLoss",
        "RankedListLoss",
        "SelfSupervisedLoss",
        "VICRegLoss",
    }
)


def _both_directions(pairs):
    """Return an Eq 7 pair set as both ``(i, j)`` and ``(j, i)``.

    Eq 7 defines ``P`` and ``N`` as sets of *unordered* pairs, and ``_mined_pairs``
    returns each one once, from the upper triangle. Eq 1 reads them that way -- a
    sum of per-pair hinges, symmetric in the pair -- but a pytorch-metric-learning
    loss does not: it scatters the tuple into an ``n x n`` mask and then reduces
    *per anchor row*. An upper-triangular mask therefore gives row ``i`` only its
    partners ``j > i``, and the last row none at all, so half of every mined pair
    set never reaches the loss. Losses that additionally require both a positive
    and a negative on the same row (``CircleLoss``, ``NTXentLoss``) drop the row
    outright when mining put the pair's two ends on different rows.

    ``get_all_pairs_indices``, which is what these losses see whenever the caller
    passes labels instead of a tuple, emits both directions. Matching it makes the
    unlabeled term read the same pair set the labeled term would.
    """

    anchors, others = pairs
    return torch.cat([anchors, others]), torch.cat([others, anchors])


def make_equation_one_contrastive_loss(pos_margin, neg_margin):
    """Build Eq 1's contrastive loss for the unlabeled term.

    Eq 1 is

        L_rank = sum_{(i,j) in P} max(d(i, j) - m_pos, 0)
               + sum_{(i,j) in N} max(m_neg - d(i, j), 0)

    which is exactly what ``ContrastiveLoss`` computes: its ``pos_calc`` is
    ``relu(d - pos_margin)`` and its ``neg_calc`` is ``relu(neg_margin - d)``,
    over the default ``LpDistance`` -- Euclidean distance on L2-normalized
    embeddings, the metric Hadsell et al. [12] use and the one the paper cites
    for ``d``. Using the library loss rather than open-coding the two hinges
    also keeps the unlabeled term's reduction identical to the labeled term's,
    which is what makes the paper's ``lambda1 = 1`` mean what it says: Eq 9
    chooses the lambdas "to make the magnitudes of the losses similar in scale",
    and a literal sum over mined pairs against a mean-reduced labeled term would
    not be on a similar scale at all.

    Both margins live on ``d``, so on L2-normalized embeddings they are bounded
    by the antipodal distance 2.0. ``m_neg > m_pos`` is required for a different
    reason -- a separation gap, not satisfiability; see the checks below.

    The margin ordering below is only correct for a *non-inverted* distance,
    where a larger value means "farther apart". pytorch-metric-learning's own
    guidance is that the default margins suit a non-inverted measure like
    ``LpDistance``, and that an inverted one such as ``CosineSimilarity`` wants
    the pair swapped (``pos_margin = 1``, ``neg_margin = 0``) -- under an
    inverted measure the attractive hinge fires on *small* values and
    ``m_neg > m_pos`` becomes the unsatisfiable ordering rather than the
    required one. Eq 1 is written on a distance ``d``, so the non-inverted
    default is the faithful choice and this function keeps it; the assertion
    below pins the orientation the validation assumes rather than leaving it to
    the library's default staying put.

    The paper does not publish ``m_pos`` or ``m_neg``. The defaults here are the
    library's, i.e. the standard contrastive-loss setting of a zero positive
    margin and a unit negative margin; they are a repository choice in the same
    sense as Eq 5's ``sd_margin``.
    """

    pos_margin = _validate_non_negative("contrastive_pos_margin", pos_margin)
    neg_margin = _validate_non_negative("contrastive_neg_margin", neg_margin)
    if neg_margin <= pos_margin:
        # Not a satisfiability bound: with m_neg <= m_pos the loss still reaches
        # zero (positives inside m_pos, negatives outside m_neg is easier, not
        # harder). What it loses is discrimination -- every pair landing in the
        # overlap band [m_neg, m_pos] pays nothing whether it is a mined
        # positive or a mined negative, so Eq 1 stops separating there.
        # m_neg > m_pos is what leaves a genuine gap between the two branches.
        raise ValueError(
            "slade contrastive_neg_margin must exceed contrastive_pos_margin so Eq 1 "
            "leaves a separation gap; otherwise pairs in the overlap band pay nothing "
            f"whichever branch they are on. Got pos={pos_margin}, neg={neg_margin}"
        )
    if neg_margin > MAX_USEFUL_CONTRASTIVE_MARGIN:
        # This one *is* a satisfiability bound, and it is the one L2
        # normalization imposes: an antipodal pair sits at exactly 2.0 and can
        # be pushed no further, yet still pays m_neg - 2.
        logger.warning(
            f"slade contrastive_neg_margin={neg_margin} exceeds the largest distance "
            f"between L2-normalized embeddings ({MAX_USEFUL_CONTRASTIVE_MARGIN}), so no "
            "mined negative can ever satisfy it and Eq 1's repulsion never stops pushing"
        )
    if neg_margin < CONTRASTIVE_NEG_MARGIN_LIVE_FLOOR:
        # Both margins are pure gates: while a hinge is active its gradient is
        # +-1 regardless of the margin, so the only thing a margin changes is
        # which pairs are switched on. On L2-normalized embeddings
        # d = sqrt(2 - 2cos), and measured DINOv2-B Cars196 features put the
        # pairwise median at 1.35 -- so a negative margin under ~1.2 fires on
        # almost nothing and Eq 1 degenerates to attraction only.
        logger.warning(
            f"slade contrastive_neg_margin={neg_margin} is below "
            f"{CONTRASTIVE_NEG_MARGIN_LIVE_FLOOR}: on L2-normalized embeddings the "
            "Euclidean distance between unrelated pairs concentrates near sqrt(2), "
            "so Eq 1's repulsion will rarely fire and the unlabeled term becomes "
            "attraction-only. That is a collapse risk unless the small margin is a "
            "deliberate late-acting guard."
        )
    criterion = losses.ContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin)
    if criterion.distance.is_inverted:
        raise RuntimeError(
            "slade Eq 1 assumes a non-inverted distance, where m_neg > m_pos is the "
            f"satisfiable ordering, but {type(criterion.distance).__name__} is inverted. "
            "The margins would have to be swapped for a similarity measure."
        )
    return criterion


def kmeans_cluster_labels(
    features,
    num_clusters,
    seed,
    iterations=25,
    backend="auto",
    n_init=10,
    use_gpu=None,
    stats=None,
):
    """Cluster teacher features and return one cluster ID per row (Sec 3.2).

    FAISS is preferred because the unlabeled pools here can be large, but the
    project also runs on hosts without a FAISS build, so ``auto`` silently falls
    back to scikit-learn instead of failing the run.

    ``use_gpu`` is ``None`` for the size rule of ``DEFAULT_KMEANS_GPU_MIN_WORK``,
    ``True`` to always try the GPU, or ``False`` to stay on the CPU. Any GPU
    failure falls back to the CPU index rather than failing the refresh.

    Pass a dict as ``stats`` to receive the backend that actually ran and how
    long it took. Which backend was used is otherwise unobservable from the
    outside, and a silent GPU fallback would look exactly like a GPU run.
    """

    started_at = time.perf_counter()

    def record(name):
        if stats is not None:
            stats["backend"] = name
            stats["seconds"] = time.perf_counter() - started_at

    features = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
    if features.ndim != 2 or len(features) == 0 or features.shape[1] == 0:
        raise ValueError("slade k-means needs a non-empty feature matrix")
    num_clusters = int(num_clusters)
    if num_clusters < 2:
        raise ValueError("slade num_clusters must be at least 2")
    if num_clusters > len(features):
        raise ValueError(
            f"slade num_clusters={num_clusters} exceeds the {len(features)} unlabeled samples"
        )
    if backend not in KMEANS_BACKENDS:
        raise ValueError(f"slade kmeans_backend must be one of {list(KMEANS_BACKENDS)}")

    return kmeans(
        features,
        num_clusters,
        seed=seed,
        iterations=iterations,
        n_init=n_init,
        backend=backend,
        use_gpu=use_gpu,
        min_gpu_work=DEFAULT_KMEANS_GPU_MIN_WORK,
        purpose="slade pseudo-label clustering",
        require_faiss=require_faiss,
        faiss_gpu_flat_index=faiss_gpu_flat_index,
        record=record,
        return_labels=True,
    )


def pseudo_label_cluster_quality(cluster_ids, true_labels):
    """Score one clustering against the hidden labels of the same samples.

    Diagnostics only: the labels never reach a loss. Rows whose label is unknown
    (negative, e.g. external images) are left out. Returns ``None`` when fewer
    than two labeled rows remain.

    * ``purity``: share of samples whose cluster's majority class is their own.
    * ``pair_precision``: of the pairs sharing a cluster, the share sharing a
      class -- how trustworthy a same-cluster pair is as a positive.
    * ``pair_recall``: of the pairs sharing a class, the share put in one cluster.
    * ``nmi``: normalized mutual information between clusters and classes.
    * ``singleton_fraction``: share of samples alone in their cluster, which can
      never form a same-cluster pair for Eq 6.
    """

    cluster_ids = np.asarray(cluster_ids).reshape(-1)
    true_labels = np.asarray(true_labels).reshape(-1)
    if len(cluster_ids) != len(true_labels):
        raise ValueError("cluster ids and true labels must align")
    known = true_labels >= 0
    if int(known.sum()) < 2:
        return None
    _, cluster_index = np.unique(cluster_ids[known], return_inverse=True)
    _, label_index = np.unique(true_labels[known], return_inverse=True)
    num_labels = int(label_index.max()) + 1
    joint_codes, joint_counts = np.unique(
        cluster_index.astype(np.int64) * num_labels + label_index,
        return_counts=True,
    )
    joint_counts = joint_counts.astype(np.float64)
    majority = np.zeros(int(cluster_index.max()) + 1, dtype=np.float64)
    np.maximum.at(majority, joint_codes // num_labels, joint_counts)
    cluster_sizes = np.bincount(cluster_index).astype(np.float64)
    class_sizes = np.bincount(label_index).astype(np.float64)

    def pair_count(sizes):
        return float((sizes * (sizes - 1.0) / 2.0).sum())

    same_both = pair_count(joint_counts)
    same_cluster = pair_count(cluster_sizes)
    same_class = pair_count(class_sizes)
    from sklearn.metrics import normalized_mutual_info_score

    count = float(len(cluster_index))
    return {
        "purity": float(majority.sum() / count),
        "pair_precision": same_both / same_cluster if same_cluster > 0 else float("nan"),
        "pair_recall": same_both / same_class if same_class > 0 else float("nan"),
        "nmi": float(normalized_mutual_info_score(label_index, cluster_index)),
        "singleton_fraction": float((cluster_sizes[cluster_index] == 1).mean()),
        "labeled_samples": count,
    }


# Where a run with an artifact directory keeps every pseudo-label refresh.
SLADE_CLUSTER_DIR = "slade_clusters"
SLADE_CLUSTER_INDEX = "refreshes.jsonl"


def drop_refresh_centers(path):
    """Rewrite one saved refresh without its ``centers``; returns the bytes freed.

    Every other array stays bit-identical. The file is replaced atomically, so a
    reader sees either the old file or the new one. The partial file's name does
    not match ``refresh_*.npz``.
    """

    path = Path(path)
    with np.load(path) as saved:
        if "centers" not in saved.files:
            return 0
        arrays = {name: saved[name] for name in saved.files if name != "centers"}
    before = path.stat().st_size
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(partial, path)
    return before - path.stat().st_size
# A sample of the pairs Eq 7 actually mined, per epoch, for the embedding viewer.
SLADE_PAIR_DIR = "slade_pairs"
SLADE_PAIR_SAMPLES_PER_EPOCH = 4096
SLADE_PAIR_SAMPLES_PER_STEP = 32


# Device-side mining counters, summed without a host sync on every step and read
# once per epoch. Every "pair" is an unordered pair of one unlabeled batch.
_MINED_POSITIVE_PAIRS = 0
_MINED_POSITIVE_KNOWN = 1
_MINED_POSITIVE_SAME_CLASS = 2
_MINED_NEGATIVE_KNOWN = 3
_MINED_NEGATIVE_SAME_CLASS = 4
_STEPS_WITHOUT_MINED_POSITIVE = 5
_LABELED_STEPS_WITH_SAME_CLASS_PAIR = 6
_STEPS_MISSING_AN_AVAILABLE_POSITIVE = 7
# Eq 7 mines on basis similarity alone, so a mined pair may cross clusters.
_MINED_POSITIVE_SAME_CLUSTER = 8
_MINED_NEGATIVE_PAIRS = 9
_MINED_NEGATIVE_SAME_CLUSTER = 10
_MINING_COUNTER_COUNT = 11


def upper_triangle_pairs(count, device):
    """Return the distinct unordered index pairs of a batch."""

    return torch.triu_indices(int(count), int(count), offset=1, device=device)


class PseudoLabelMPerClassBatchSampler(torch.utils.data.Sampler):
    """Unlabeled batches of ``m`` members from each of ``batch_size / m`` clusters.

    Yields row indices into the unlabeled subset. The cluster assignment is
    replaced in place by ``set_labels`` after every pseudo-label refresh; both
    the DataLoader and the GPU-resident loader iterate the sampler in the main
    process once per epoch, so the next epoch reads the new clusters.

    Only clusters with at least ``m`` members are drawn. Topping a smaller
    cluster up with replacement, as pytorch-metric-learning's MPerClassSampler
    does, would put the same image twice in a batch. That pair has similarity
    exactly 1, which would inflate Eq 6's mu+ and add positives Eq 1 already
    satisfies.

    Clusters are chosen uniformly, as MPerClassSampler chooses classes, and
    members without replacement within a batch. When fewer clusters qualify
    than a batch has ``m``-sized slots, e.g. a small ``cluster_ratio`` against a
    large unlabeled batch, the slots go round-robin over a shuffled cluster
    order. A cluster then contributes ``2m``, ``3m``, ... distinct members, up
    to what it holds. The epoch length is fixed at
    ``len(pool) // batch_size`` batches, the same as the ``random`` shuffle with
    ``drop_last``. So both samplers give the same number of unlabeled steps, and
    the GPU-resident loader's length, read once at construction, stays valid.
    """

    def __init__(self, num_rows, batch_size, m, seed=0):
        self.num_rows = int(num_rows)
        self.batch_size = int(batch_size)
        self.m = int(m)
        if self.m < 2:
            raise ValueError("slade unlabeled_sampler_m must be at least 2 so clusters form pairs")
        if self.batch_size % self.m != 0:
            raise ValueError(
                f"slade unlabeled batch size {self.batch_size} must be divisible by "
                f"unlabeled_sampler_m={self.m}"
            )
        self.clusters_per_batch = self.batch_size // self.m
        self.num_batches = self.num_rows // self.batch_size
        if self.num_batches < 1:
            raise ValueError("slade pseudo-label sampler needs at least one full batch")
        self.generator = np.random.default_rng(seed)
        self._members = None
        self.eligible_fraction = None

    def set_labels(self, labels):
        """Regroup the rows by their new cluster ID; returns the share of rows drawable."""

        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(labels) != self.num_rows:
            raise ValueError("slade pseudo-label sampler labels must cover every unlabeled row")
        order = np.argsort(labels, kind="stable")
        boundaries = np.flatnonzero(np.diff(labels[order])) + 1
        members = [
            group for group in np.split(order, boundaries) if len(group) >= self.m
        ]
        capacity = np.asarray([len(group) // self.m for group in members], dtype=np.int64)
        if int(capacity.sum()) < self.clusters_per_batch:
            raise ValueError(
                f"slade pseudo-label sampling needs {self.clusters_per_batch} groups of "
                f"unlabeled_sampler_m={self.m} same-cluster members per batch, but the "
                f"{len(members)} clusters with at least {self.m} members hold only "
                f"{int(capacity.sum())}. Lower unlabeled_batch_size or unlabeled_sampler_m"
            )
        if len(members) < self.clusters_per_batch:
            logger.warning(
                f"slade pseudo-label sampling: only {len(members)} clusters have at least "
                f"m={self.m} members, fewer than the {self.clusters_per_batch} a batch of "
                f"{self.batch_size} has slots for, so some clusters contribute a multiple of m"
            )
        self._members = members
        self._capacity = capacity
        self.eligible_fraction = float(sum(len(group) for group in members)) / self.num_rows
        return self.eligible_fraction

    def _slots_per_cluster(self):
        """How many ``m``-sized groups each cluster gives this batch."""

        order = self.generator.permutation(len(self._members))
        slots = np.zeros(len(self._members), dtype=np.int64)
        if len(order) >= self.clusters_per_batch:
            slots[order[: self.clusters_per_batch]] = 1
            return slots
        remaining = self.clusters_per_batch
        while remaining:
            for index in order:
                if slots[index] < self._capacity[index]:
                    slots[index] += 1
                    remaining -= 1
                    if not remaining:
                        break
        return slots

    def __len__(self):
        return self.num_batches

    def __iter__(self):
        if self._members is None:
            raise RuntimeError("slade pseudo-label sampler has no clusters; call set_labels first")
        members = self._members
        for _ in range(self.num_batches):
            slots = self._slots_per_cluster()
            batch = np.concatenate(
                [
                    self.generator.choice(members[index], size=int(count) * self.m, replace=False)
                    for index, count in enumerate(slots)
                    if count
                ]
            )
            yield batch.tolist()


class SladeBasisLayer(nn.Module):
    """Basis vectors ``W_a`` plus the two global similarity Gaussians.

    ``weight`` holds the ``k x d`` basis matrix of Sec 3.3.1, so row ``i`` is the
    basis vector ``a_i`` and ``r = W_a f`` is a class-wise similarity
    representation. The running means and variances of Eq 6 are buffers rather
    than plain attributes so they follow the model across devices and into the
    checkpoint alongside ``weight``.
    """

    def __init__(self, feat_dim, num_basis, logit_scale=1.0):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_basis = int(num_basis)
        self.logit_scale = _validate_positive("logit_scale", logit_scale)
        if self.feat_dim <= 0:
            raise ValueError("slade feat_dim must be positive")
        if self.num_basis <= 1:
            raise ValueError("slade requires at least two basis vectors")

        # Eq 3 applies a softmax cross entropy over the k basis responses, which
        # is exactly a bias-free linear classifier over the embedding.
        self.weight = nn.Parameter(torch.empty(self.num_basis, self.feat_dim))
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.feat_dim))

        for name in ("positive_mean", "negative_mean", "positive_variance", "negative_variance"):
            self.register_buffer(name, torch.zeros((), dtype=torch.float32))
        # Eq 6 is an interpolation, so it needs a real first observation rather
        # than the arbitrary zeros above before the loss or Eq 7 may use it.
        # Tracked per distribution: G+ and G- are seeded by different batches
        # whenever pseudo-positive pairs are rarer than pseudo-negative ones,
        # which is the normal case -- a batch of ``b`` samples over ``k``
        # clusters holds only about ``C(b, 2) / k`` same-cluster pairs.
        for name in ("positive_initialized", "negative_initialized"):
            self.register_buffer(name, torch.zeros((), dtype=torch.bool))
        self.register_buffer("distributions_initialized", torch.zeros((), dtype=torch.bool))

    def forward(self, embeddings):
        """Return the feature representation ``r = W_a f`` (Sec 3.3.1)."""

        return embeddings @ self.weight.t().to(dtype=embeddings.dtype)

    def logits(self, embeddings):
        """Return the Eq 3 basis responses used by the cross-entropy term."""

        return self.logit_scale * self(embeddings)

    def pairwise_similarity(self, embeddings, eps=1e-8):
        """Return ``s(x_i, x_j) = cos(W_a f_i, W_a f_j)`` for a batch (Eq 4).

        Autocast would run both the basis projection and the Gram matmul in
        bfloat16 whatever dtype the embedding arrives in, because matmul is cast
        down even from float32. Eq 5 then reads the mean and the *variance* of
        these cosines, and a spread around a common mean is exactly what eight
        mantissa bits cannot resolve - the variance of a tight positive cluster
        would collapse into rounding noise. Everything downstream of here is
        elementwise, so it follows the float32 similarities on its own.
        """

        with torch.autocast(device_type=embeddings.device.type, enabled=False):
            representations = self(embeddings.float())
            normalized = representations / representations.norm(dim=1, keepdim=True).clamp_min(eps)
            return normalized @ normalized.t()

    @property
    def is_calibrated(self):
        return bool(self.distributions_initialized.item())

    def update_distributions(
        self,
        positive_mean=None,
        positive_variance=None,
        negative_mean=None,
        negative_variance=None,
        *,
        beta,
        full_gradient=True,
    ):
        """Apply Eq 6 and return the updated statistics with gradients intact.

        The returned values are differentiable in the batch statistics so Eq 5
        can train ``W_a``, while the stored buffers keep only detached history.

        ``full_gradient`` selects between the two ``SD_GRADIENT_MODES``. It
        changes only what the returned statistics differentiate to; the buffers
        this writes, and the values Eq 5 and Eq 7 read, are identical either way.

        Each distribution is updated on its own, which is how the paper states
        Eq 6 -- it gives the update for G+ and adds that "the parameters of G-
        are updated in a similar way". Pass ``None`` for a statistic the batch
        could not form: Eq 6 has no ``mu_b`` for a pair kind that did not occur,
        so that Gaussian keeps its global value and the other one still moves.
        Requiring both at once, which this used to do, stalls the whole of Eq 5
        on the batches where only one kind occurs -- and with many clusters
        those are most batches.
        """

        beta = float(beta)
        statistics = {}
        sides = (
            ("positive", positive_mean, positive_variance),
            ("negative", negative_mean, negative_variance),
        )
        for side, mean, variance in sides:
            names = (f"{side}_mean", f"{side}_variance")
            initialized = getattr(self, f"{side}_initialized")
            if mean is None or variance is None:
                # Eq 5 still reads this side, just at its stored global value.
                statistics.update(
                    (name, getattr(self, name).detach().clone()) for name in names
                )
                continue
            # Choose on-device: reading these flags on the host stalls CUDA
            # every batch. Clone because where saves its condition for backward
            # and the checkpointed flag is updated in-place below.
            was_initialized = initialized.detach().clone()
            updated = tuple(
                torch.where(
                    was_initialized,
                    self._blend(name, value, beta, full_gradient),
                    value,
                )
                for name, value in zip(names, (mean, variance))
            )
            with torch.no_grad():
                for name, value in zip(names, updated):
                    getattr(self, name).copy_(value.detach().to(torch.float32))
                initialized.fill_(True)
            statistics.update(zip(names, updated))
        with torch.no_grad():
            self.distributions_initialized.copy_(
                self.positive_initialized & self.negative_initialized
            )
        return statistics

    def _blend(self, name, batch_value, beta, full_gradient):
        """Return Eq 6's updated statistic under the selected gradient mode."""

        running = getattr(self, name).detach()
        blended = (1.0 - beta) * batch_value + beta * running
        if not full_gradient:
            return blended
        # ``beta * (x - x.detach())`` is zero in the forward pass and has
        # derivative ``beta`` in the backward one, so it restores the batch
        # statistic's full gradient without touching the value Eq 5 reads:
        # (1 - beta) + beta = 1.
        return blended + beta * (batch_value - batch_value.detach())

    @torch.no_grad()
    def reset_parameters(self):
        """Re-initialize ``W_a`` and discard the two Gaussians (Sec 3.3.1/Eq 6).

        Used when a self-training iteration starts a new student. The stored
        Gaussians summarize similarities under the *previous* iteration's cluster
        IDs, and Eq 7 turns them straight into mining thresholds, so carrying
        them past a re-clustering mines the new pseudo labels against a stale
        notion of what a confident pair looks like.
        """

        nn.init.normal_(self.weight, std=1.0 / math.sqrt(self.feat_dim))
        for name in ("positive_mean", "negative_mean", "positive_variance", "negative_variance"):
            getattr(self, name).zero_()
        for name in ("positive_initialized", "negative_initialized", "distributions_initialized"):
            getattr(self, name).fill_(False)

    def mining_thresholds(self):
        """Return ``(T1, T2) = (mu+, mu-)`` used by Eq 7, or ``None``.

        Two conditions gate this, and only the first is the paper's.

        ``is_calibrated`` is Eq 6, not Eq 7: until a batch has seeded each
        distribution the buffers hold their placeholder zeros, so both thresholds
        would be ``0.0`` and would split the batch at an arbitrary cut.

        ``mu+ > mu-`` is **an addition to Eq 7**, which sets ``T1 = mu+`` and
        ``T2 = mu-`` with no condition on their order. It is kept because the
        overlapping case has no usable behaviour rather than because the paper
        asks for it: while ``mu+ <= mu-`` the two bands intersect, and Eq 7 puts
        every pair between them in *both* sets. pytorch-metric-learning rejects a
        duplicated pair outright -- ``GenericPairLoss._assert_either_pos_or_neg``
        raises -- so the mined-pair term would abort the run rather than pull the
        pair together and apart on the same step. The paper's own defence is
        Sec 3.3.3's basis warm-up, training ``W_a`` alone for "a few iterations"
        so the distributions separate before mining opens; ``basis_warmup_steps``
        is that warm-up here, and this guard covers the case where it was too
        short.

        The cost of the guard is that it suppresses *all* mining while the bands
        overlap, including the confidently separated pairs outside the
        intersection. Watch ``train/slade/mining_active``: if it stays at zero
        well past the warm-up, the basis is not separating and the fix is
        ``basis_warmup_steps`` or ``sd_margin``, not this threshold.
        """

        return self.mining_state()[1]

    def mining_state(self):
        """Read calibration and mining eligibility in a single device transfer.

        A calibrated but overlapping pair of Gaussians still contributes the
        distribution loss, even though it cannot supply mining thresholds.
        """

        calibrated, positive_mean, negative_mean = torch.stack(
            (self.distributions_initialized, self.positive_mean, self.negative_mean)
        ).tolist()
        thresholds = None
        if calibrated and not positive_mean <= negative_mean:
            thresholds = (positive_mean, negative_mean)
        return bool(calibrated), thresholds


def similarity_distribution_loss(
    positive_mean,
    positive_variance,
    negative_mean,
    negative_variance,
    margin,
    variance_weight,
):
    """Eq 5: separate the two Gaussians and penalize their spread."""

    margin_term = F.relu(negative_mean - positive_mean + float(margin))
    return margin_term + float(variance_weight) * (positive_variance + negative_variance)


def _pair_cross_entropy(similarity, positive, eps):
    """Binary cross entropy of a cosine read as the probability ``(1 + s) / 2``."""

    probability = (1.0 + similarity) / 2.0 if positive else (1.0 - similarity) / 2.0
    return -torch.log(probability.clamp_min(eps))


def local_pair_cross_entropy(positive_similarity, negative_similarity, eps=1e-6):
    """Table 5's local-CE: cross entropy of each batch pair against its pseudo label.

    See ``BASIS_UNLABELED_LOSSES`` for the probability and the per-kind averaging.
    A kind the batch has no pairs of is left out rather than counted as zero.
    """

    terms = [
        _pair_cross_entropy(similarity, positive, eps).mean()
        for similarity, positive in ((positive_similarity, True), (negative_similarity, False))
        if similarity.numel()
    ]
    if not terms:
        raise ValueError("local pair cross entropy needs at least one pair")
    return torch.stack(terms).mean()


def global_pair_cross_entropy(positive_mean, negative_mean, eps=1e-6):
    """Table 5's global-CE: the same cross entropy on the Eq 6 global means."""

    return 0.5 * (
        _pair_cross_entropy(positive_mean, True, eps)
        + _pair_cross_entropy(negative_mean, False, eps)
    )


def oracle_cluster_labels(true_labels, positions, noise=0.0, seed=0):
    """Hidden true classes as cluster IDs, a seeded share of them reassigned.

    This is the SLADE counterpart of ``apply_pseudo_label_oracle``: a perfect
    clustering, so a run shows how much of SLADE's effect depends on how well
    k-means recovers the classes. ``true_labels`` is aligned with ``positions``.
    Which samples are corrupted, and the class each one gets, are drawn once per
    training-subset position from ``seed``. A sample then stays wrong in the same
    way across refreshes, and a larger ``noise`` corrupts a superset of what a
    smaller one does. A wrong ID is drawn uniformly from the pool's other classes.

    Returns ``(cluster_ids, num_clusters, stats)`` with IDs in ``[0, num_clusters)``.
    """

    noise = float(noise)
    if not 0.0 <= noise < 1.0:
        raise ValueError(f"oracle cluster noise must lie in [0, 1): {noise}")
    positions = np.asarray(positions, dtype=np.int64)
    true_labels = np.asarray(true_labels, dtype=np.int64)
    if len(true_labels) != len(positions):
        raise ValueError("oracle cluster labels need one true label per position")
    if np.any(true_labels < 0):
        raise ValueError("oracle clusters need a hidden true label for every unlabeled sample")
    _, cluster_ids = np.unique(true_labels, return_inverse=True)
    count = int(cluster_ids.max()) + 1
    generator = np.random.default_rng(int(seed))
    size = int(positions.max()) + 1
    draws = generator.random(size)
    offsets = generator.integers(0, np.iinfo(np.int64).max, size)
    corrupt = draws[positions] < noise
    if corrupt.any():
        if count < 2:
            raise ValueError("oracle cluster noise needs at least two classes in the pool")
        # Step 1..count-1 classes forward, so the new ID is never the true one.
        wrong = (cluster_ids + 1 + offsets[positions] % (count - 1)) % count
        cluster_ids = np.where(corrupt, wrong, cluster_ids)
    stats = {
        "count": int(len(positions)),
        "classes": count,
        "corrupted": int(corrupt.sum()),
        "corrupted_fraction": float(corrupt.mean()) if len(positions) else 0.0,
    }
    return cluster_ids.astype(np.int64), count, stats


class SladeRegularizer(BaseTrainingRegularizer):
    """SLADE's student objective as an unlabeled regularization term (Eq 9)."""

    name = "slade"
    # One deterministic view per unlabeled image is enough, so the frozen
    # backbone feature cache stays usable.
    supports_frozen_feature_precompute = True
    uses_joint_forward = True
    requires_supervised_objective = True
    # ``pseudo_label_source='oracle'`` replaces the k-means clusters with the
    # hidden true classes (``oracle_cluster_labels``). Diagnostic only.
    supports_oracle_pseudo_labels = True
    extra_component_names = (SLADE_BASIS_COMPONENT,)
    # Eq 1's margins are not published; these are the library's standard
    # contrastive settings. Named so the constructor can tell an explicit value
    # from an inherited default when the mode does not use them.
    DEFAULT_CONTRASTIVE_POS_MARGIN = 0.0
    DEFAULT_CONTRASTIVE_NEG_MARGIN = 1.0

    def __init__(
        self,
        regularizer_weight=1.0,
        supervised_weight=1.0,
        lambda_basis=0.25,
        basis_target_ratio=None,
        beta=0.99,
        sd_margin=0.5,
        sd_variance_weight=1.0,
        sd_gradient_mode="damped",
        logit_scale=1.0,
        basis_warmup_steps=100,
        basis_warmup_epochs=None,
        num_clusters=None,
        cluster_ratio=None,
        kmeans_backend="auto",
        kmeans_iterations=25,
        kmeans_n_init=10,
        kmeans_gpu=True,
        unlabeled_ranking_loss="supervised",
        contrastive_pos_margin=0.0,
        contrastive_neg_margin=1.0,
        contrastive_positive_margin=None,
        contrastive_negative_margin=None,
        unlabeled_ratio=1.0,
        unlabeled_batch_size=None,
        unlabeled_sampler="random",
        unlabeled_sampler_m=DEFAULT_UNLABELED_SAMPLER_M,
        mining_pseudo_label_filter=False,
        pair_source="basis_mining",
        basis_unlabeled_loss="sd",
        within_fold_teacher_student=False,
        cross_fold_teacher_handoff=False,
        reset_basis_on_refresh=False,
        reset_basis_warmup_on_refresh=False,
    ):
        super().__init__(
            regularizer_weight=regularizer_weight,
            supervised_weight=supervised_weight,
        )
        self.lambda_basis = _validate_non_negative("lambda_basis", lambda_basis)
        # Calibrating lambda_basis makes the configured value only a probe, the
        # same relationship regularizer_target_ratio has with regularizer_weight.
        self.basis_target_ratio = (
            None
            if basis_target_ratio is None
            else _validate_positive("basis_target_ratio", basis_target_ratio)
        )
        self.beta = float(beta)
        if not math.isfinite(self.beta) or not (0 < self.beta <= 1):
            raise ValueError("slade beta must be in (0, 1]")
        self.sd_margin = _validate_non_negative("sd_margin", sd_margin)
        self.sd_variance_weight = _validate_non_negative(
            "sd_variance_weight",
            sd_variance_weight,
        )
        self.sd_gradient_mode = str(sd_gradient_mode)
        if self.sd_gradient_mode not in SD_GRADIENT_MODES:
            raise ValueError(
                f"slade sd_gradient_mode must be one of {list(SD_GRADIENT_MODES)}"
            )
        self.logit_scale = _validate_positive("logit_scale", logit_scale)
        # Two alternative ways to length the same warm-up: ``basis_warmup_steps``
        # is absolute, ``basis_warmup_epochs`` is relative to the sampler epoch.
        # Steps are what the warm-up actually runs on -- W_a moves once per
        # optimizer step and Eq 6's statistics lag the live basis by ~1/(1-beta)
        # steps, neither of which depends on the batch size. But the number of
        # steps in an epoch is the fold training pool over batch_size, so a study
        # that searches batch_sampler gives one absolute step count wildly
        # different meanings: 100 steps is half an epoch at batch 32 and four
        # epochs at batch 256. Searching the epoch form instead means the same
        # thing at every batch size, so it never has to be filtered against one.
        self.basis_warmup_steps = (
            None if basis_warmup_steps is None else int(basis_warmup_steps)
        )
        if self.basis_warmup_steps is not None and self.basis_warmup_steps < 0:
            raise ValueError("slade basis_warmup_steps must be non-negative")
        self.basis_warmup_epochs = (
            None
            if basis_warmup_epochs is None
            else _validate_non_negative("basis_warmup_epochs", basis_warmup_epochs)
        )
        if self.basis_warmup_steps is None and self.basis_warmup_epochs is None:
            self.basis_warmup_steps = DEFAULT_BASIS_WARMUP_STEPS
        elif self.basis_warmup_steps is not None and self.basis_warmup_epochs is not None:
            logger.warning(
                f"slade received both basis_warmup_steps={self.basis_warmup_steps} and "
                f"basis_warmup_epochs={self.basis_warmup_epochs}; the absolute step count "
                "wins and the epoch count is ignored"
            )
            self.basis_warmup_epochs = None
        # Two alternative ways to set the same k: ``num_clusters`` is absolute,
        # ``cluster_ratio`` is relative to the C labeled classes, so 0.5 is
        # k = C/2 and 2.0 is k = 2C -- the default of one cluster per class is
        # the ratio 1.0. The absolute form means a different thing on every
        # dataset (98 clusters is a fifth of Cars-196 and a thirtieth of
        # In-Shop), so a study that searches it is pinned to one dataset and its
        # winner cannot be replayed on another. The ratio transfers, which makes
        # it the form to search.
        self.num_clusters = None if num_clusters is None else int(num_clusters)
        if self.num_clusters is not None and self.num_clusters < 2:
            raise ValueError("slade num_clusters must be at least 2 when set")
        self.cluster_ratio = (
            None
            if cluster_ratio is None
            else _validate_positive("cluster_ratio", cluster_ratio)
        )
        if self.num_clusters is not None and self.cluster_ratio is not None:
            logger.warning(
                f"slade received both num_clusters={self.num_clusters} and "
                f"cluster_ratio={self.cluster_ratio}; the absolute cluster count "
                "wins and the ratio is ignored"
            )
        self.kmeans_backend = str(kmeans_backend)
        if self.kmeans_backend not in KMEANS_BACKENDS:
            raise ValueError(f"slade kmeans_backend must be one of {list(KMEANS_BACKENDS)}")
        self.kmeans_iterations = int(kmeans_iterations)
        if self.kmeans_iterations <= 0:
            raise ValueError("slade kmeans_iterations must be positive")
        self.kmeans_n_init = int(kmeans_n_init)
        if self.kmeans_n_init <= 0:
            raise ValueError("slade kmeans_n_init must be positive")
        if kmeans_gpu is not None and not isinstance(kmeans_gpu, bool):
            raise ValueError("slade kmeans_gpu must be a boolean or null")
        self.kmeans_gpu = kmeans_gpu
        self.unlabeled_ranking_loss = str(unlabeled_ranking_loss)
        if self.unlabeled_ranking_loss not in UNLABELED_RANKING_LOSSES:
            raise ValueError(
                f"slade unlabeled_ranking_loss must be one of {list(UNLABELED_RANKING_LOSSES)}"
            )
        # The mode existed under these names before; ``configs/old_runs`` still
        # carries one. Name the replacement instead of letting the constructor
        # report an unexpected keyword.
        renamed = {
            "contrastive_positive_margin": ("contrastive_pos_margin", contrastive_positive_margin),
            "contrastive_negative_margin": ("contrastive_neg_margin", contrastive_negative_margin),
        }
        for old_name, (new_name, value) in renamed.items():
            if value is not None:
                raise ValueError(
                    f"slade {old_name} was renamed to {new_name}, matching the "
                    "pos_margin/neg_margin arguments of the underlying "
                    "ContrastiveLoss it now builds"
                )
        self.contrastive_pos_margin = _validate_non_negative(
            "contrastive_pos_margin",
            contrastive_pos_margin,
        )
        self.contrastive_neg_margin = _validate_non_negative(
            "contrastive_neg_margin",
            contrastive_neg_margin,
        )
        # Built eagerly so a bad margin pair fails at construction rather than at
        # the first mined batch. It owns no parameters, so unlike a proxy loss it
        # needs no optimizer of its own -- nothing in Eq 1 is learnable.
        self._contrastive_criterion = None
        if self.unlabeled_ranking_loss == "contrastive":
            self._contrastive_criterion = make_equation_one_contrastive_loss(
                pos_margin=self.contrastive_pos_margin,
                neg_margin=self.contrastive_neg_margin,
            )
        elif (
            contrastive_pos_margin != self.DEFAULT_CONTRASTIVE_POS_MARGIN
            or contrastive_neg_margin != self.DEFAULT_CONTRASTIVE_NEG_MARGIN
        ):
            logger.warning(
                "slade contrastive_pos_margin/contrastive_neg_margin only apply to "
                f"unlabeled_ranking_loss='contrastive'; the configured "
                f"'{self.unlabeled_ranking_loss}' mode reuses the supervised loss "
                "and its own margins, so these values are ignored"
            )
        # Two alternative ways to size the same unlabeled batch:
        # ``unlabeled_batch_size`` is absolute, ``unlabeled_ratio`` is relative to
        # the labeled batch. Setting either one leaves the other free to be null,
        # so a config that pins an absolute size does not also have to carry a
        # ratio that would be ignored. Null for both keeps the historical default
        # of one unlabeled sample per labeled sample.
        self.set_unlabeled_batch_sizing(unlabeled_batch_size, unlabeled_ratio, DEFAULT_UNLABELED_RATIO)
        self.unlabeled_sampler = str(unlabeled_sampler)
        if self.unlabeled_sampler not in UNLABELED_SAMPLERS:
            raise ValueError(
                f"slade unlabeled_sampler must be one of {list(UNLABELED_SAMPLERS)}"
            )
        if isinstance(unlabeled_sampler_m, bool) or int(unlabeled_sampler_m) != unlabeled_sampler_m:
            raise ValueError("slade unlabeled_sampler_m must be an integer")
        self.unlabeled_sampler_m = int(unlabeled_sampler_m)
        if self.unlabeled_sampler_m < 2:
            raise ValueError("slade unlabeled_sampler_m must be at least 2 so clusters form pairs")
        if (
            self.unlabeled_sampler != "pseudo_label_m_per_class"
            and self.unlabeled_sampler_m != DEFAULT_UNLABELED_SAMPLER_M
        ):
            logger.warning(
                f"slade unlabeled_sampler_m={self.unlabeled_sampler_m} only applies to "
                "unlabeled_sampler='pseudo_label_m_per_class'; the configured "
                f"'{self.unlabeled_sampler}' sampler ignores it"
            )
        # Eq 7 read literally mines on basis similarity alone, so a mined
        # positive may cross clusters and a mined negative may share one. Table
        # 4 lists mining as a step added on top of pseudo-label pairs, and Fig 2
        # says the basis selects "high-confidence" pairs, which reads more like
        # a filter on those pairs. True keeps a mined positive only when it
        # shares a cluster and a mined negative only when it does not. The
        # thresholds T1 = mu+ and T2 = mu- stay as they are.
        if not isinstance(mining_pseudo_label_filter, bool):
            raise ValueError("slade mining_pseudo_label_filter must be a boolean")
        self.mining_pseudo_label_filter = mining_pseudo_label_filter
        self.pair_source = str(pair_source)
        if self.pair_source not in PAIR_SOURCES:
            raise ValueError(f"slade pair_source must be one of {list(PAIR_SOURCES)}")
        if self.pair_source == "pseudo_label" and self.mining_pseudo_label_filter:
            raise ValueError(
                "slade mining_pseudo_label_filter filters Eq 7's mined pairs by cluster, "
                "but pair_source='pseudo_label' trains on the cluster pairs themselves, "
                "so the filter would change nothing"
            )
        self.basis_unlabeled_loss = str(basis_unlabeled_loss)
        if self.basis_unlabeled_loss not in BASIS_UNLABELED_LOSSES:
            raise ValueError(
                f"slade basis_unlabeled_loss must be one of {list(BASIS_UNLABELED_LOSSES)}"
            )
        # Seeds the random draw of ``pair_source='pseudo_label_matched'``; set
        # from the run seed in make_loader, the generator built on first use.
        self._pair_selection_seed = 0
        self._pair_generator = None
        self._unlabeled_batch_sampler = None
        if not isinstance(within_fold_teacher_student, bool):
            raise ValueError("slade within_fold_teacher_student must be a boolean")
        if not isinstance(cross_fold_teacher_handoff, bool):
            raise ValueError("slade cross_fold_teacher_handoff must be a boolean")
        # The engine owns both lifecycle switches.  The first creates a genuine
        # optimization/model-selection boundary inside a run.  The second spans
        # otherwise independent runs and is deliberately separate, so a user
        # can run teacher -> student in every fold without leaking a trained
        # projection into the next fold.
        self.within_fold_teacher_student = within_fold_teacher_student
        self.cross_fold_teacher_handoff = cross_fold_teacher_handoff
        for name, value in (
            ("reset_basis_on_refresh", reset_basis_on_refresh),
            ("reset_basis_warmup_on_refresh", reset_basis_warmup_on_refresh),
        ):
            if not isinstance(value, bool):
                raise ValueError(f"slade {name} must be a boolean")
        # Sec 3.3.3 promotes the student to teacher and starts the next
        # iteration. Whether that iteration's student inherits the previous
        # one's basis and its consumed basis warm-up is left open by the paper,
        # so each is its own switch rather than one lumped "new student" flag.
        self.reset_basis_on_refresh = reset_basis_on_refresh
        self.reset_basis_warmup_on_refresh = reset_basis_warmup_on_refresh

        if self.beta < 0.5:
            logger.warning(
                f"slade beta={self.beta} weights the running estimate in Eq 6 "
                "((1-beta) * batch + beta * running), so the 'global' Gaussians track the "
                "current batch almost exclusively. The paper uses beta=0.99 for a "
                "slow-moving global estimate."
            )

        self.num_basis = None
        self.dataset = None
        self._cluster_by_position = None
        self._resolved_num_clusters = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_cluster_rebuild_epoch = None
        self._optimizer_steps = 0
        # Tracked outside the diagnostics dict because target-ratio calibration
        # reads it whether or not batch diagnostics are being collected.
        self._mining_active = False
        # This step's unweighted L_Basis, read by combine_losses, the gradient
        # diagnostics, and the basis weight calibration.
        self._basis_loss = None
        self._bounded_ntxent_logged = False
        self._last_diagnostics = {}
        # Hidden labels by training-subset position, for the activity
        # diagnostics only; ``None`` when the pool has no labels to audit.
        self._true_label_by_position = None
        self._reset_activity_counters()
        # Parameters re-initialized by a refresh reset, held until the trainer
        # collects them and drops their stale optimizer moments.
        self._pending_optimizer_state_resets = []

    def set_steps_per_epoch(self, steps_per_epoch):
        """Turn ``basis_warmup_epochs`` into the step count the warm-up gate reads."""

        if self.basis_warmup_epochs is None:
            return None
        steps_per_epoch = int(steps_per_epoch)
        if steps_per_epoch < 1:
            raise ValueError("slade requires at least one optimizer step per epoch")
        self.basis_warmup_steps = int(round(self.basis_warmup_epochs * steps_per_epoch))
        logger.info(
            f"slade basis warm-up: {self.basis_warmup_epochs:g} epochs x "
            f"{steps_per_epoch} steps/epoch = {self.basis_warmup_steps} optimizer steps"
        )
        return self.basis_warmup_steps

    def steady_state_active(self):
        """Wait until both Eq 9 unlabeled terms actually reach the model.

        Two start-up phases run after the engine's ``warmup_epochs`` boundary,
        and during either one this regularizer is not yet the method it is
        supposed to be:

        * During ``basis_warmup_steps`` the embedding is detached, so the
          regularizer's gradient reaches ``W_a`` alone -- non-zero, so the
          engine's zero-guard does not catch it, but it carries none of the
          unlabeled ranking term ``regularizer_weight`` is meant to balance.
        * Eq 7 mines nothing until the two Gaussians separate (``mu+ > mu-``), so
          until then the ranking term is exactly zero whatever the batch holds.

        Target-ratio calibration would otherwise measure a start-up phase and
        freeze it: with the default 100 warmup steps against 20 calibration
        batches, the whole calibration completes before either term exists. Model
        selection under ``restart_selection_after_warmup`` would otherwise
        rebaseline on epochs the ranking term never reached, which matters more
        here than elsewhere because ``basis_warmup_epochs`` is a searched
        hyperparameter -- a baseline taken before it elapses would let a trial
        score better simply by starting up more slowly.

        Not monotone: the second condition is the mining gate, which switches
        back off if the Gaussians stop separating. Callers latch on first True.
        """

        if self._optimizer_steps < self.basis_warmup_steps:
            return False
        return self._mining_active

    def validate_run_args(self, args):
        from ..types import CLASSIFICATION_LOSSES

        if self.cross_fold_teacher_handoff:
            if not self.within_fold_teacher_student:
                raise ValueError(
                    "slade cross_fold_teacher_handoff requires "
                    "within_fold_teacher_student=true"
                )
            if int(args.cv_k) <= 1:
                raise ValueError(
                    "slade cross_fold_teacher_handoff requires cv_k > 1"
                )
            if str(args.backbone_tuning) != "frozen":
                raise ValueError(
                    "slade cross_fold_teacher_handoff currently requires "
                    "backbone_tuning='frozen' so the promoted projection head is "
                    "the complete trainable embedding transform"
                )
        if int(args.batch_size) < 2:
            raise ValueError("slade requires batch_size >= 2 so unlabeled pairs exist")
        supervised_is_classification = args.loss in CLASSIFICATION_LOSSES
        # A proxy loss scores each embedding against learned class centers, so it
        # needs no in-batch positive for Eq 9's labeled term. The requirement is
        # the pair-based term's, not SLADE's.
        if not supervised_is_classification and int(args.sampler_m) < 2:
            raise ValueError(
                "slade requires sampler_m >= 2 so the labeled ranking loss of Eq 9 sees positives"
            )
        if self.unlabeled_ranking_loss != "contrastive" and args.loss in (
            set(CLASSIFICATION_LOSSES) | PAIR_INCOMPATIBLE_LOSSES
        ):
            reason = (
                "is proxy/classification based and scores against learned class centers"
                if supervised_is_classification
                else "does not accept a mined indices_tuple"
            )
            raise ValueError(
                f"slade applies the configured loss to the pairs mined by Eq 7, but "
                f"{args.loss} {reason}. Choose a pair-based loss such as ContrastiveLoss, "
                "MultiSimilarityLoss, TripletMarginLoss, or CircleLoss, or set "
                "regularizer_params.unlabeled_ranking_loss='contrastive' so Eq 9's "
                "unlabeled term uses the paper's own Eq 1 loss instead of this one."
            )

    def configure_model(self, student_model, train_dataset, split, train_labels_mapper, device):
        """Attach ``W_a`` before the optimizer collects trainable parameters."""

        if self.regularizer_weight == 0:
            return None
        if hasattr(student_model, "slade_basis"):
            raise RuntimeError("slade basis vectors are already configured on this model")
        num_classes = len(train_labels_mapper)
        if num_classes <= 1:
            raise ValueError("slade needs at least two training classes for Eq 3")
        basis = SladeBasisLayer(
            feat_dim=student_model.feat_dim,
            num_basis=num_classes,
            logit_scale=self.logit_scale,
        ).to(device)
        student_model.add_module("slade_basis", basis)
        self.num_basis = int(num_classes)
        logger.info(
            "Configured SLADE feature basis: "
            f"{self.num_basis} basis vectors of dim {student_model.feat_dim}, "
            f"lambda1/regularizer_weight={self.regularizer_weight}, "
            f"lambda2={self.lambda_basis}, "
            f"beta={self.beta}, unlabeled_ranking_loss={self.unlabeled_ranking_loss}, "
            f"sd_gradient_mode={self.sd_gradient_mode}"
        )
        if self._contrastive_criterion is not None:
            self._contrastive_criterion = self._contrastive_criterion.to(device)
            logger.info(
                "slade unlabeled_ranking_loss='contrastive': Eq 9's unlabeled term uses "
                "the paper's Eq 1 contrastive loss "
                f"(m_pos={self.contrastive_pos_margin}, m_neg={self.contrastive_neg_margin}) "
                "on Eq 7's mined pairs instead of the supervised criterion, so the two "
                "streams of Eq 9 no longer share one L_rank. The margins are repository "
                "defaults; the paper does not publish them."
            )
        if self.sd_gradient_mode == "damped":
            logger.info(
                "slade sd_gradient_mode='damped': Eq 5's gradient into W_a carries the "
                f"literal Eq 6 factor (1 - beta) = {1 - self.beta:g}, so beta weights L_SD "
                "against L_CE inside Eq 2 as well as setting the updating rate. "
                "sd_gradient_mode='full' removes that coupling; it is not the default "
                "because it changes the objective's balance relative to earlier studies."
            )
        if self.unlabeled_sampler != "random" or self.mining_pseudo_label_filter:
            logger.info(
                f"SLADE unlabeled_sampler={self.unlabeled_sampler}"
                + (
                    f" (m={self.unlabeled_sampler_m} per cluster)"
                    if self.unlabeled_sampler == "pseudo_label_m_per_class"
                    else ""
                )
                + f", mining_pseudo_label_filter={self.mining_pseudo_label_filter}"
            )
        if self.pair_source != "basis_mining" or self.basis_unlabeled_loss != "sd":
            logger.info(
                f"SLADE ablation: pair_source={self.pair_source}, "
                f"basis_unlabeled_loss={self.basis_unlabeled_loss}"
                + (
                    "; sd_margin and sd_variance_weight are unused"
                    if self.basis_unlabeled_loss != "sd"
                    else ""
                )
            )
        if self.reset_basis_on_refresh or self.reset_basis_warmup_on_refresh:
            logger.info(
                "SLADE self-training iterations: "
                f"reset_basis_on_refresh={self.reset_basis_on_refresh}, "
                f"reset_basis_warmup_on_refresh={self.reset_basis_warmup_on_refresh}"
            )
        return None

    def build_dataset(self, train_dataset, split, use_cache=False):
        self._unlabeled_batch_sampler = None
        self._last_cluster_rebuild_epoch = None
        self._cluster_by_position = None
        self._pair_generator = None
        unlabeled_positions = self.begin_unlabeled_pool(split)
        if unlabeled_positions is None:
            return None

        if self.num_clusters is not None:
            resolved_clusters = self.num_clusters
        elif self.num_basis is None:
            raise RuntimeError("slade configure_model must run before build_dataset")
        elif self.cluster_ratio is not None:
            scaled_clusters = int(round(self.cluster_ratio * self.num_basis))
            if scaled_clusters < 2:
                # Every ratio below 2/C rounds to the same two clusters, so a
                # search range that reaches down there spends its trials
                # re-running one configuration under different names.
                logger.warning(
                    f"slade cluster_ratio={self.cluster_ratio:g} x {self.num_basis} labeled "
                    f"classes rounds to k={scaled_clusters}; using the minimum of 2 instead"
                )
            resolved_clusters = max(2, scaled_clusters)
            logger.info(
                f"slade cluster_ratio={self.cluster_ratio:g} x {self.num_basis} labeled "
                f"classes gives k={resolved_clusters}"
            )
        else:
            resolved_clusters = self.num_basis
            logger.info(
                f"slade num_clusters defaults to the {self.num_basis} labeled classes; set it "
                "(or cluster_ratio) explicitly when the unlabeled pool comes from another dataset"
            )
        if resolved_clusters > len(unlabeled_positions):
            logger.warning(
                f"slade k={resolved_clusters} exceeds the "
                f"{len(unlabeled_positions)} unlabeled samples; clustering into "
                f"{len(unlabeled_positions)} clusters instead"
            )
            resolved_clusters = len(unlabeled_positions)
        if resolved_clusters < 2:
            raise ValueError("slade needs at least two pseudo-label clusters")
        self._resolved_num_clusters = int(resolved_clusters)
        self._reset_activity_counters()
        self._cluster_refresh_count = 0
        self._refresh_with_droppable_centers = None
        self._true_label_by_position = self._hidden_label_lookup(
            train_dataset,
            unlabeled_positions,
        )
        regularizer_dataset = self.make_regularizer_source_dataset(
            train_dataset,
            use_cache=use_cache,
        )
        self.dataset = UnlabeledSubset(regularizer_dataset, unlabeled_positions, num_views=1)
        return self.dataset

    def _refresh_pseudo_labels(self, model, train_dataset, device, config, seed, start_method, epoch):
        """Use the current embedding as the offline teacher and re-cluster (Sec 3.2).

        On the first explicit teacher/student boundary this is the supervised
        teacher. On a periodic default run it is the current student promoted
        to the next teacher role. In both cases clustering finishes before any
        subsequent student optimizer step, so the resulting IDs are a frozen
        teacher target until the next requested refresh.

        Under ``pseudo_label_source='oracle'`` the hidden true classes replace
        k-means, so k is the number of classes in the pool rather than the
        configured one.
        """

        num_clusters = self._resolved_num_clusters
        if getattr(config, "pseudo_label_source", "method") == "oracle":
            if self._true_label_by_position is None:
                raise ValueError(
                    "slade pseudo_label_source='oracle' needs hidden labels for the unlabeled "
                    "pool; an external pool has none"
                )
            noise = float(getattr(config, "pseudo_label_oracle_noise", 0.0))
            clusters, num_clusters, oracle_stats = oracle_cluster_labels(
                self._true_label_by_position[self.dataset.positions],
                self.dataset.positions,
                noise=noise,
                seed=0 if config.seed is None else int(config.seed),
            )
            features = None
            kmeans_stats = {"backend": "oracle", "seconds": 0.0}
            logger.warning(
                "SLADE CLUSTER ORACLE uses hidden true labels: "
                f"{oracle_stats['count']} unlabeled samples in {num_clusters} classes, "
                f"{oracle_stats['corrupted']} ({oracle_stats['corrupted_fraction']:.1%}) "
                f"given a wrong class at noise={noise:g}; the configured "
                f"k={self._resolved_num_clusters} is not used"
            )
        else:
            features = extract_embeddings(
                model=model,
                dataset=train_dataset,
                positions=self.dataset.positions,
                device=self.get_ssl_device(device),
                batch_size=config.embedding_batch_size,
                num_workers=config.embedding_num_workers,
                seed=seed,
                start_method=start_method,
                desc=f"SLADE teacher embeddings - epoch {epoch}",
            )
            kmeans_stats = {}
            clusters = kmeans_cluster_labels(
                features,
                num_clusters=num_clusters,
                seed=seed,
                iterations=self.kmeans_iterations,
                backend=self.kmeans_backend,
                n_init=self.kmeans_n_init,
                use_gpu=self.kmeans_gpu,
                stats=kmeans_stats,
            )
        # Batches identify samples by training-subset position, so index the
        # cluster IDs the same way instead of by row in the unlabeled subset.
        lookup = np.full(int(self.dataset.positions.max()) + 1, -1, dtype=np.int64)
        lookup[self.dataset.positions] = clusters
        self._cluster_by_position = lookup
        occupied = int(len(np.unique(clusters)))
        quality = None
        if self._true_label_by_position is not None:
            quality = pseudo_label_cluster_quality(
                clusters,
                self._true_label_by_position[self.dataset.positions],
            )
            if quality is not None:
                self._cluster_quality_history.append(quality)
                self._pending_cluster_quality = quality
                logger.info(
                    f"SLADE pseudo-label quality - epoch {epoch} (hidden labels, "
                    f"diagnostic only): purity={quality['purity']:.3f}, "
                    f"same-cluster pair precision={quality['pair_precision']:.3f}, "
                    f"same-class pair recall={quality['pair_recall']:.3f}, "
                    f"NMI={quality['nmi']:.3f}, "
                    f"singletons={quality['singleton_fraction']:.1%}"
                )
        self._save_cluster_refresh(
            epoch, clusters, occupied, quality, features=features, num_clusters=num_clusters
        )
        logger.info(
            f"SLADE pseudo labels - epoch {epoch}: "
            f"{len(clusters)} unlabeled samples, k={num_clusters}, "
            f"{occupied} non-empty clusters "
            f"({kmeans_stats.get('backend', 'unknown')}, "
            f"{kmeans_stats.get('seconds', float('nan')):.2f}s)"
        )

    def _begin_self_training_iteration(self, model, epoch):
        """Apply the configured per-iteration resets after a re-clustering.

        Sec 3.3.3 ends an iteration by promoting the student to teacher and
        starting the next one. The engine owns the optimizer-side half of that
        boundary; these are the two pieces of state this regularizer owns.
        """

        if not (self.reset_basis_on_refresh or self.reset_basis_warmup_on_refresh):
            return
        actions = []
        if self.reset_basis_on_refresh:
            basis = getattr(model, SLADE_BASIS_MODULE, None)
            if basis is None:
                raise RuntimeError(
                    "slade reset_basis_on_refresh needs the basis module configure_model attaches"
                )
            basis.reset_parameters()
            # Adam's moments were fitted to the basis this just discarded.
            self._pending_optimizer_state_resets.append(basis.weight)
            self._mining_active = False
            actions.append("re-initialized W_a and cleared the Eq 6 Gaussians")
        if self.reset_basis_warmup_on_refresh:
            self._optimizer_steps = 0
            actions.append(
                f"restarted the {self.basis_warmup_steps}-step Eq 2 basis warm-up"
            )
        logger.info(
            f"SLADE self-training iteration - epoch {epoch}: " + "; ".join(actions)
        )

    def consume_pending_optimizer_state_resets(self):
        parameters = tuple(self._pending_optimizer_state_resets)
        self._pending_optimizer_state_resets = []
        return parameters

    def make_loader(
        self,
        model,
        train_dataset,
        supervised_loader,
        device,
        config,
        batch_size,
        seed,
        num_workers,
        start_method,
        epoch,
        log_dir=None,
    ):
        if self.dataset is None:
            raise RuntimeError("slade build_dataset must run before make_loader")
        self._pair_selection_seed = int(seed)
        first_refresh = self._cluster_by_position is None
        refreshed = first_refresh or should_rebuild_on_epoch(
            config.update_mode,
            config.update_interval_epochs,
            epoch,
            self._last_cluster_rebuild_epoch,
        )
        if refreshed:
            self._refresh_pseudo_labels(
                model=model,
                train_dataset=train_dataset,
                device=device,
                config=config,
                seed=seed,
                start_method=start_method,
                epoch=epoch,
            )
            if not first_refresh:
                # The first refresh produces the first teacher's labels; the
                # student it feeds has not trained yet, so there is nothing to
                # reset. Every later one opens a new self-training iteration.
                self._begin_self_training_iteration(model, epoch)
            self._last_cluster_rebuild_epoch = None if epoch is None else int(epoch)

        # ``drop_last`` means a shrunken batch also changes the step count, so it
        # moves both the Eq 6 statistics and how often Eq 9's unlabeled term fires.
        unlabeled_batch_size, sizing = self.resolve_unlabeled_batch_size(batch_size)
        worker_count = utils.dataloader_num_workers_for_dataset(self.dataset, num_workers)
        cache_key = (
            id(self.dataset),
            int(unlabeled_batch_size),
            int(worker_count),
            str(start_method),
            self.unlabeled_sampler,
        )
        rebuild_loader = (
            self._regularizer_loader is None or self._regularizer_loader_cache_key != cache_key
        )
        if rebuild_loader and self.unlabeled_sampler == "pseudo_label_m_per_class":
            self._unlabeled_batch_sampler = PseudoLabelMPerClassBatchSampler(
                num_rows=len(self.dataset),
                batch_size=unlabeled_batch_size,
                m=self.unlabeled_sampler_m,
                seed=seed,
            )
        if self._unlabeled_batch_sampler is not None and (refreshed or rebuild_loader):
            # The loaders iterate the sampler in the main process each epoch, so
            # regrouping it in place is enough: a cached loader picks it up.
            eligible = self._unlabeled_batch_sampler.set_labels(
                self._cluster_by_position[self.dataset.positions]
            )
            logger.info(
                f"SLADE pseudo-label sampler - epoch {epoch}: "
                f"{self._unlabeled_batch_sampler.clusters_per_batch} clusters x "
                f"m={self.unlabeled_sampler_m} per batch; {eligible:.1%} of the pool sits in "
                f"clusters with at least m members and can be drawn"
            )
        if rebuild_loader:
            utils.shutdown_dataloaders(self._regularizer_loader)
            self._regularizer_loader = utils.make_unlabeled_stream_loader(
                self.dataset,
                batch_size=unlabeled_batch_size,
                seed=seed,
                num_workers=worker_count,
                start_method=start_method,
                supervised_loader=supervised_loader,
                # Eq 5 estimates batch means and variances, so a ragged final
                # batch would inject noisy statistics into Eq 6.
                drop_last=True,
                persistent_workers=True,
                pin_memory=True,
                desc="slade unlabeled",
                batch_sampler=self._unlabeled_batch_sampler,
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                "SLADE unlabeled loader: "
                f"pool={len(self.dataset)}, batch_size={unlabeled_batch_size} ({sizing}), "
                f"sampler={self.unlabeled_sampler}, "
                f"clusters={self._resolved_num_clusters}, steps={len(supervised_loader)}"
            )
        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    def _cluster_labels(self, positions, device):
        if self._cluster_by_position is None:
            raise RuntimeError("slade pseudo labels are missing; make_loader must run first")
        positions = np.asarray(
            torch.as_tensor(positions).detach().cpu().numpy(),
            dtype=np.int64,
        ).reshape(-1)
        clusters = self._cluster_by_position[positions]
        if np.any(clusters < 0):
            raise RuntimeError("slade batch contains a sample without a pseudo label")
        return torch.as_tensor(clusters, dtype=torch.long, device=device)

    @staticmethod
    def _eq7_selection(pair_similarity, thresholds, same_cluster=None):
        """Eq 7's positive and negative flags for each upper-triangle pair.

        ``same_cluster`` is passed only under ``mining_pseudo_label_filter``.
        A mined positive must then share a cluster and a mined negative must
        not.
        """

        positive = pair_similarity >= thresholds[0]
        negative = pair_similarity <= thresholds[1]
        if same_cluster is not None:
            positive = positive & same_cluster
            negative = negative & ~same_cluster
        return positive, negative

    def _pair_source_selection(self, pair_similarity, thresholds, same_cluster):
        """Positive and negative flags per upper-triangle pair for a non-Eq-7 ``pair_source``."""

        if self.pair_source == "pseudo_label":
            return same_cluster, ~same_cluster
        if self.pair_source != "pseudo_label_matched":
            raise ValueError(f"slade pair_source {self.pair_source!r} has no alternative selection")
        mining_filter = same_cluster if self.mining_pseudo_label_filter else None
        positive, negative = self._eq7_selection(pair_similarity, thresholds, mining_filter)
        return (
            self._random_subset(same_cluster, positive.sum()),
            self._random_subset(~same_cluster, negative.sum()),
        )

    def _random_subset(self, candidates, count):
        """Flag ``count`` uniformly drawn ``candidates``, or all of them if fewer.

        ``count`` stays a tensor: ranking random scores avoids reading it on the host.
        """

        device = candidates.device
        # Keyed by the tensor's device: a generator built for "cuda:0" reports "cuda".
        if self._pair_generator is None or self._pair_generator[0] != device:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._pair_selection_seed)
            self._pair_generator = (device, generator)
        scores = torch.rand(candidates.shape, generator=self._pair_generator[1], device=device)
        # Non-candidates sort last, so the first ``count`` ranks are candidates.
        rank = scores.masked_fill(~candidates, 2.0).argsort().argsort()
        return candidates & (rank < count)

    @staticmethod
    def _mined_pairs(similarity, pair_indices, thresholds, same_cluster=None, selection=None):
        """Eq 7: keep pairs the basis similarity is confident about.

        ``mining_thresholds`` guarantees ``T1 > T2``, so the two bands are
        disjoint and no pair is mined as both positive and negative. That
        guarantee is an addition to Eq 7 rather than part of it; see
        ``mining_thresholds`` for why it is kept.

        ``selection`` replaces Eq 7 with precomputed flags (``pair_source``).
        """

        pair_similarity = similarity[pair_indices[0], pair_indices[1]].detach()
        positive_mask, negative_mask = (
            selection
            if selection is not None
            else SladeRegularizer._eq7_selection(pair_similarity, thresholds, same_cluster)
        )
        # Each mask used to be compacted twice, once per endpoint. On CUDA
        # nonzero synchronizes to learn the output size; reuse the integer IDs.
        positive_indices = positive_mask.nonzero(as_tuple=True)[0]
        negative_indices = negative_mask.nonzero(as_tuple=True)[0]
        return (
            (pair_indices[0][positive_indices], pair_indices[1][positive_indices]),
            (pair_indices[0][negative_indices], pair_indices[1][negative_indices]),
        )

    @staticmethod
    def _mined_pair_masks(similarity, pair_indices, thresholds, same_cluster=None, selection=None):
        """The same Eq 7 pairs as _mined_pairs, kept as symmetric dense masks."""

        if selection is None:
            pair_similarity = similarity[pair_indices[0], pair_indices[1]].detach()
            selection = SladeRegularizer._eq7_selection(pair_similarity, thresholds, same_cluster)
        masks = []
        for selected in selection:
            mask = torch.zeros_like(similarity, dtype=torch.bool)
            # Mirror the upper triangle explicitly, just as _both_directions
            # does; do not assume the two halves of a GEMM are bit-identical.
            mask[pair_indices[0], pair_indices[1]] = selected
            mask[pair_indices[1], pair_indices[0]] = selected
            masks.append(mask)
        return tuple(masks)

    def _unlabeled_ranking_loss(
        self,
        embeddings,
        cluster_labels,
        positive_pairs,
        negative_pairs,
        supervised_criterion,
        supervised_is_classification,
    ):
        """Apply the configured pair loss to the mined pairs.

        The configured miner is deliberately not consulted: Eq 7's basis
        thresholds already are SLADE's mining rule for the unlabeled stream.

        Under ``unlabeled_ranking_loss='contrastive'`` the term uses SLADE's own
        Eq 1 loss instead of the supervised criterion, so the supervised half is
        free to be a proxy or classification loss. The mined pairs, the
        embeddings they index, and the reduction are otherwise unchanged -- only
        the callable differs.
        """

        if len(positive_pairs[0]) == 0 and len(negative_pairs[0]) == 0:
            return embeddings.sum() * 0.0
        criterion = supervised_criterion
        if self._contrastive_criterion is not None:
            criterion = self._contrastive_criterion
        elif supervised_is_classification:
            # ``validate_run_args`` rejects these before training starts; this is
            # the backstop for a criterion built outside that path.
            raise ValueError("slade needs a pair-based supervised loss for Eq 9's unlabeled term")
        # pytorch-metric-learning reads a four-tuple as
        # (anchors+, positives, anchors-, negatives) and then ignores labels, so
        # the mined pairs drive the configured ranking loss directly. This is the
        # paper's claim that the framework accepts any pair-based ranking loss.
        indices_tuple = (
            *_both_directions(positive_pairs),
            *_both_directions(negative_pairs),
        )
        if computes_upstream_ntxent(criterion):
            # Upstream expands every positive against every negative before
            # masking anchors. Dense SLADE mining can make that product tens of
            # GiB at batch size 512, even when backbone features are frozen.
            if not self._bounded_ntxent_logged:
                logger.info(
                    "SLADE NTXentLoss: using bounded-memory per-anchor negative "
                    "sums; no positive-pair x negative-pair matrix"
                )
                self._bounded_ntxent_logged = True
            return bounded_ntxent_loss(criterion, embeddings, cluster_labels, indices_tuple)
        return criterion(embeddings, cluster_labels, indices_tuple)

    def compute_loss(
        self,
        student_model,
        state,
        batch,
        device,
        timings=None,
        supervised_embeddings=None,
        supervised_labels=None,
        regularizer_embeddings=None,
        supervised_inputs=None,
        supervised_indices=None,
        supervised_criterion=None,
        supervised_miner=None,
        supervised_is_classification=False,
    ):
        basis = getattr(student_model, "slade_basis", None)
        if basis is None:
            raise RuntimeError("slade requires configure_model to attach the basis vectors")
        if supervised_embeddings is None or supervised_labels is None or regularizer_embeddings is None:
            raise ValueError("slade requires the joint labeled/unlabeled forward context")
        if supervised_criterion is None:
            raise ValueError("slade requires the configured supervised objective")
        if batch is None or len(batch) < 3:
            raise ValueError("slade unlabeled batches must carry their training-subset positions")
        if len(regularizer_embeddings) < 2:
            raise ValueError("slade needs at least two unlabeled embeddings per batch")

        cluster_labels = self._cluster_labels(batch[2], regularizer_embeddings.device)
        if len(cluster_labels) != len(regularizer_embeddings):
            raise ValueError("slade pseudo labels must align with the unlabeled embeddings")

        # The paper trains the basis vectors alone for a few iterations before
        # going end-to-end, so detaching f leaves W_a as the only thing Eq 2 moves.
        basis_warmup_active = self._optimizer_steps < self.basis_warmup_steps
        labeled_features = (
            supervised_embeddings.detach() if basis_warmup_active else supervised_embeddings
        )
        unlabeled_features = (
            regularizer_embeddings.detach() if basis_warmup_active else regularizer_embeddings
        )

        # Autocast off around the projection, not a cast after it: matmul is cast
        # down to bfloat16 even from float32 operands, so promoting W_a f
        # afterwards would hand cross_entropy a float32 tensor holding bfloat16
        # logits. ``logit_scale`` multiplies whatever rounding survives.
        with torch.autocast(device_type=labeled_features.device.type, enabled=False):
            cross_entropy_loss = F.cross_entropy(
                basis.logits(labeled_features.float()),
                supervised_labels,
            )

        similarity = basis.pairwise_similarity(unlabeled_features)
        pair_indices = upper_triangle_pairs(len(cluster_labels), similarity.device)
        same_cluster = (
            cluster_labels[pair_indices[0]] == cluster_labels[pair_indices[1]]
        )
        pair_similarity = similarity[pair_indices[0], pair_indices[1]]
        positive_similarity = pair_similarity[same_cluster]
        negative_similarity = pair_similarity[~same_cluster]

        zero = regularizer_embeddings.sum() * 0.0
        distribution_loss = zero
        statistics = None
        if len(positive_similarity) > 0 or len(negative_similarity) > 0:
            statistics = basis.update_distributions(
                # Eq 6 moves only the Gaussian this batch has pairs for.
                positive_mean=positive_similarity.mean() if len(positive_similarity) else None,
                positive_variance=(
                    positive_similarity.var(unbiased=False)
                    if len(positive_similarity)
                    else None
                ),
                negative_mean=negative_similarity.mean() if len(negative_similarity) else None,
                negative_variance=(
                    negative_similarity.var(unbiased=False)
                    if len(negative_similarity)
                    else None
                ),
                beta=self.beta,
                full_gradient=self.sd_gradient_mode == "full",
            )
        calibrated, thresholds = basis.mining_state()
        if self.basis_unlabeled_loss == "local_ce":
            # Per-pair, so it needs no calibrated Gaussians.
            distribution_loss = local_pair_cross_entropy(positive_similarity, negative_similarity)
        elif statistics is not None and calibrated:
            # Eq 5 still trains overlapping distributions; only mining is gated.
            if self.basis_unlabeled_loss == "sd":
                distribution_loss = similarity_distribution_loss(
                    positive_mean=statistics["positive_mean"],
                    positive_variance=statistics["positive_variance"],
                    negative_mean=statistics["negative_mean"],
                    negative_variance=statistics["negative_variance"],
                    margin=self.sd_margin,
                    variance_weight=self.sd_variance_weight,
                )
            elif self.basis_unlabeled_loss == "global_ce":
                distribution_loss = global_pair_cross_entropy(
                    statistics["positive_mean"], statistics["negative_mean"]
                )
        ranking_loss = zero
        positive_pairs = (pair_indices[0][:0], pair_indices[1][:0])
        negative_pairs = positive_pairs
        positive_pair_count = negative_pair_count = 0.0
        # Cluster pairs need no thresholds; Eq 7 and the count it sets for the
        # matched control do. All sources wait for the basis warm-up, so the
        # ranking term starts on the same step whichever one is configured.
        self._mining_active = not basis_warmup_active and (
            self.pair_source == "pseudo_label" or thresholds is not None
        )
        mining_filter = same_cluster if self.mining_pseudo_label_filter else None
        selection = None
        if self._mining_active and self.pair_source != "basis_mining":
            selection = self._pair_source_selection(
                pair_similarity.detach(), thresholds, same_cluster
            )
        self._record_mining_activity(
            pair_similarity.detach(),
            pair_indices,
            thresholds if self._mining_active else None,
            batch[2],
            cluster_labels=cluster_labels,
            same_cluster_filter=mining_filter,
            selection=selection,
        )
        if self._mining_active:
            dense_forward = getattr(supervised_criterion, "forward_with_pair_masks", None)
            if (
                self._contrastive_criterion is None
                and not supervised_is_classification
                and callable(dense_forward)
            ):
                positive_mask, negative_mask = self._mined_pair_masks(
                    similarity, pair_indices, thresholds, mining_filter, selection=selection,
                )
                ranking_loss = dense_forward(
                    regularizer_embeddings, cluster_labels, positive_mask, negative_mask,
                )
                if self.collect_batch_diagnostics:
                    positive_pair_count = positive_mask.sum().detach() / 2
                    negative_pair_count = negative_mask.sum().detach() / 2
            else:
                positive_pairs, negative_pairs = self._mined_pairs(
                    similarity,
                    pair_indices,
                    thresholds,
                    mining_filter,
                    selection=selection,
                )
                ranking_loss = self._unlabeled_ranking_loss(
                    embeddings=regularizer_embeddings,
                    cluster_labels=cluster_labels,
                    positive_pairs=positive_pairs,
                    negative_pairs=negative_pairs,
                    supervised_criterion=supervised_criterion,
                    supervised_is_classification=supervised_is_classification,
                )
                positive_pair_count = float(len(positive_pairs[0]))
                negative_pair_count = float(len(negative_pairs[0]))

        # Eq 9's two unlabeled terms are kept apart rather than summed here.
        # L_Basis trains W_a, which the supervised loss never touches, and its
        # scale has nothing to do with the ranking term's; folding them into one
        # number would let the basis gradient drive a weight meant for the
        # ranking gradient, and would hide both inside a single reported norm.
        # ``combine_losses`` adds the basis term with its own weight.
        self._basis_loss = cross_entropy_loss + distribution_loss
        # Return the raw unlabeled ranking term. BaseTrainingRegularizer applies
        # regularizer_weight in combine_losses, so that single weight is Eq 9's
        # lambda1 (and is the weight GradNorm learns for this component).
        total = ranking_loss
        if not (torch.isfinite(total) & torch.isfinite(self._basis_loss)):
            raise FloatingPointError("slade produced a non-finite regularization loss")

        if self.collect_batch_diagnostics:
            self._last_diagnostics = {
                "train/slade/cross_entropy_loss": cross_entropy_loss.detach(),
                "train/slade/similarity_distribution_loss": distribution_loss.detach(),
                "train/slade/unlabeled_ranking_loss": ranking_loss.detach(),
                "train/slade/positive_mean": basis.positive_mean.detach(),
                "train/slade/negative_mean": basis.negative_mean.detach(),
                "train/slade/positive_variance": basis.positive_variance.detach(),
                "train/slade/negative_variance": basis.negative_variance.detach(),
                "train/slade/mined_positive_pairs": positive_pair_count,
                "train/slade/mined_negative_pairs": negative_pair_count,
                "train/slade/pseudo_positive_pairs": float(len(positive_similarity)),
                "train/slade/basis_warmup_active": float(basis_warmup_active),
                "train/slade/mining_active": float(thresholds is not None),
            }
        else:
            self._last_diagnostics = {}
        return total

    def combine_losses(self, supervised_loss, regularization_loss):
        """Eq 9 with all three terms weighted independently.

        ``regularization_loss`` is the raw ``L_rank(D_u)``;
        ``regularizer_weight`` supplies lambda1 through the base composition.
        The basis term is added here under ``lambda_basis`` so each carries its
        own weight -- and, when target ratios are configured, its own calibration.
        """

        total = super().combine_losses(supervised_loss, regularization_loss)
        if self._basis_loss is None:
            return total
        return total + self.lambda_basis * self._basis_loss

    def extra_loss_components(self):
        if self._basis_loss is None:
            return {}
        return {SLADE_BASIS_COMPONENT: (self._basis_loss, self.lambda_basis)}

    def extra_target_ratios(self):
        if self.basis_target_ratio is None:
            return {}
        return {SLADE_BASIS_COMPONENT: self.basis_target_ratio}

    def clear_extra_target_ratios(self, keep=()):
        if SLADE_BASIS_COMPONENT in keep:
            return
        self.basis_target_ratio = None

    def grad_norm_extra_components(self):
        """Eq 9's basis term, so GradNorm learns lambda2 alongside the other two.

        The paper picks lambda1 and lambda2 "empirically ... to make the
        magnitudes of the losses similar in scale", which is what GradNorm does
        from measured gradient norms. Leaving the basis term out would balance
        two of Eq 9's three terms and leave the third at a fixed scale.
        """

        return {SLADE_BASIS_COMPONENT: self.lambda_basis}

    def private_model_module_names(self):
        # ``W_a``: Eq 9 is the only thing that reaches it, and during
        # ``basis_warmup_steps`` it is the *only* thing the basis term reaches,
        # so a GradNorm norm taken over it would measure a term that is not in
        # the shared trunk at all.
        return (SLADE_BASIS_MODULE,)

    def calibratable_component_loss(self, name):
        if name != SLADE_BASIS_COMPONENT:
            return None
        return self._basis_loss

    def calibrated_component_weight(self, name):
        if name != SLADE_BASIS_COMPONENT:
            raise KeyError(name)
        return self.lambda_basis

    def apply_calibrated_component_weight(self, name, weight):
        if name != SLADE_BASIS_COMPONENT:
            raise KeyError(name)
        self.lambda_basis = float(weight)

    def after_optimizer_step(self, student_model, state):
        self._optimizer_steps += 1
        return None

    def batch_diagnostics(self):
        return {
            name: float(value.detach().item()) if torch.is_tensor(value) else float(value)
            for name, value in self._last_diagnostics.items()
        }

    def _save_cluster_refresh(self, epoch, clusters, occupied, quality, features=None, num_clusters=None):
        """Keep the clusters this refresh assigned, when the run has an artifact directory.

        One ``refresh_<n>_epoch_<e>.npz`` per refresh holds the unlabeled pool's
        training-subset positions and their cluster ids -- the exact pseudo labels
        the student trained against until the next refresh, not a recomputation --
        and ``refreshes.jsonl`` indexes them with k, occupancy and, where hidden
        labels exist, the clustering's quality.

        With the teacher ``features`` it also keeps every member's cosine
        similarity to its own cluster center, and the centers themselves (the
        mean teacher embedding, a row of NaN for an empty cluster). SLADE itself
        never reads the centers: only the IDs enter Eq 5, so these are for
        inspection. They are k x d float16, ~8 MB at In-Shop's k=5996, and barely
        compress, so only the first refresh (the supervised teacher's) and the
        newest one keep them: each new refresh drops them from the one before.
        """

        if self.artifact_dir is None:
            return None
        # An oracle refresh groups by true class, so its k is not the configured one.
        k = int(self._resolved_num_clusters if num_clusters is None else num_clusters)
        directory = Path(self.artifact_dir) / SLADE_CLUSTER_DIR
        directory.mkdir(parents=True, exist_ok=True)
        refresh_index = getattr(self, "_cluster_refresh_count", 0)
        self._cluster_refresh_count = refresh_index + 1
        epoch_value = -1 if epoch is None else int(epoch)
        name = f"refresh_{refresh_index:03d}_epoch_{max(epoch_value, 0):04d}.npz"
        arrays = {}
        if features is not None:
            features = np.asarray(features, dtype=np.float32)
            cluster_ids = np.asarray(clusters, dtype=np.int64)
            sums = np.zeros((k, features.shape[1]), dtype=np.float64)
            np.add.at(sums, cluster_ids, features)
            sizes = np.bincount(cluster_ids, minlength=k).astype(np.float64)
            centers = np.divide(sums, sizes[:, None], out=np.full_like(sums, np.nan), where=sizes[:, None] > 0)
            unit = lambda values: values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
            arrays["centers"] = centers.astype(np.float16)
            arrays["center_similarity"] = np.sum(
                unit(features.astype(np.float64)) * unit(np.nan_to_num(centers[cluster_ids])), axis=1
            ).astype(np.float16)
        np.savez_compressed(
            directory / name,
            positions=np.asarray(self.dataset.positions, dtype=np.int64),
            clusters=np.asarray(clusters, dtype=np.int32),
            num_clusters=np.int64(k),
            epoch=np.int64(epoch_value),
            **arrays,
        )
        record = {
            "file": name,
            "refresh": refresh_index,
            "epoch": epoch_value,
            "num_clusters": k,
            "occupied_clusters": int(occupied),
            "samples": int(len(clusters)),
        }
        if quality is not None:
            record["quality"] = {key: float(value) for key, value in quality.items()}
        with (directory / SLADE_CLUSTER_INDEX).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        # The new file is complete, so the previous one's centers can go.
        previous = getattr(self, "_refresh_with_droppable_centers", None)
        if previous is not None:
            drop_refresh_centers(previous)
        self._refresh_with_droppable_centers = (directory / name) if refresh_index > 0 and "centers" in arrays else None
        return directory / name

    # -- Activity diagnostics -------------------------------------------------
    #
    # Whether SLADE helped cannot be read off a score alone: a flat result looks
    # the same whether the unlabeled term was trained on good pairs, on wrong
    # pairs, or barely at all. These counters answer which, per epoch in
    # metrics.csv and per run in the training result. Hidden labels are read
    # here and in the refresh log only; nothing below feeds a loss.

    def _reset_activity_counters(self):
        self._epoch_mining_counts = None
        self._epoch_steps = 0
        self._epoch_mining_active_steps = 0
        self._epoch_labeled_steps = 0
        self._run_mining_counts = [0.0] * _MINING_COUNTER_COUNT
        self._run_steps = 0
        self._run_mining_active_steps = 0
        self._run_labeled_steps = 0
        self._cluster_quality_history = []
        self._pending_cluster_quality = None
        self._pair_samples = []
        self._pair_sample_counts = [0, 0]
        self._pair_rng = np.random.default_rng(0)
        self._pair_epoch_files = 0

    @staticmethod
    def _hidden_label_lookup(train_dataset, positions):
        labels = getattr(train_dataset, "labels", None)
        positions = np.asarray(positions, dtype=np.int64)
        if labels is None or len(positions) == 0:
            return None
        labels = np.asarray(labels).reshape(-1)
        if len(labels) <= int(positions.max()):
            # The pool indexes a dataset these labels do not cover (external
            # images); there is nothing to audit against.
            return None
        lookup = np.full(int(positions.max()) + 1, -1, dtype=np.int64)
        lookup[positions] = labels[positions].astype(np.int64)
        return lookup

    def _record_mining_activity(
        self,
        pair_similarity,
        pair_indices,
        thresholds,
        positions,
        cluster_labels=None,
        same_cluster_filter=None,
        selection=None,
    ):
        """Count this step's mining outcome; ``thresholds`` is ``None`` when mining is off.

        ``same_cluster_filter`` is the pseudo-label filter the loss applied, so the
        counters describe the pairs the loss actually trained on. ``selection``
        is the loss's own pair flags when ``pair_source`` is not Eq 7; it counts
        as mining being on even without thresholds.
        """

        self._epoch_steps += 1
        if thresholds is None and selection is None:
            return
        self._epoch_mining_active_steps += 1
        self._last_mining_thresholds = thresholds
        device = pair_similarity.device
        if self._epoch_mining_counts is None:
            self._epoch_mining_counts = torch.zeros(
                _MINING_COUNTER_COUNT, dtype=torch.float64, device=device
            )
        positive, negative = (
            selection
            if selection is not None
            else self._eq7_selection(pair_similarity, thresholds, same_cluster_filter)
        )
        no_positive = ~positive.any()
        zero = torch.zeros((), dtype=torch.float64, device=device)
        known_positive = same_positive = known_negative = same_negative = zero
        has_same_class = missed = zero
        if self._true_label_by_position is not None:
            index = np.asarray(
                torch.as_tensor(positions).detach().cpu().numpy(), dtype=np.int64
            ).reshape(-1)
            labels = torch.as_tensor(
                self._true_label_by_position[index], dtype=torch.long, device=device
            )
            left = labels[pair_indices[0]]
            right = labels[pair_indices[1]]
            known = (left >= 0) & (right >= 0)
            same = known & (left == right)
            known_positive = (positive & known).sum()
            same_positive = (positive & same).sum()
            known_negative = (negative & known).sum()
            same_negative = (negative & same).sum()
            has_same_class = same.any()
            missed = has_same_class & no_positive
            self._epoch_labeled_steps += 1
        update = torch.zeros(_MINING_COUNTER_COUNT, dtype=torch.float64, device=device)
        update[_MINED_POSITIVE_PAIRS] = positive.sum()
        update[_MINED_POSITIVE_KNOWN] = known_positive
        update[_MINED_POSITIVE_SAME_CLASS] = same_positive
        update[_MINED_NEGATIVE_KNOWN] = known_negative
        update[_MINED_NEGATIVE_SAME_CLASS] = same_negative
        update[_STEPS_WITHOUT_MINED_POSITIVE] = no_positive
        update[_LABELED_STEPS_WITH_SAME_CLASS_PAIR] = has_same_class
        update[_STEPS_MISSING_AN_AVAILABLE_POSITIVE] = missed
        update[_MINED_NEGATIVE_PAIRS] = negative.sum()
        if cluster_labels is not None:
            same_cluster = cluster_labels[pair_indices[0]] == cluster_labels[pair_indices[1]]
            update[_MINED_POSITIVE_SAME_CLUSTER] = (positive & same_cluster).sum()
            update[_MINED_NEGATIVE_SAME_CLUSTER] = (negative & same_cluster).sum()
        self._epoch_mining_counts += update
        if self.artifact_dir is not None:
            self._sample_mined_pairs(pair_similarity, pair_indices, positive, negative, positions, cluster_labels)

    def _sample_mined_pairs(self, pair_similarity, pair_indices, positive, negative, positions, cluster_labels):
        """Keep a few of this step's mined pairs for the epoch's ``slade_pairs`` file.

        Runs only for an inspected run (one with an artifact directory). The
        choice uses its own NumPy generator, so the training stream's random
        state is exactly what an uninspected run would see.
        """

        index = np.asarray(torch.as_tensor(positions).detach().cpu().numpy(), dtype=np.int64).reshape(-1)
        for kind, mask in ((1, positive), (0, negative)):
            room = SLADE_PAIR_SAMPLES_PER_EPOCH - self._pair_sample_counts[kind]
            if room <= 0:
                continue
            chosen = mask.nonzero(as_tuple=True)[0].cpu().numpy()
            if len(chosen) == 0:
                continue
            take = min(room, SLADE_PAIR_SAMPLES_PER_STEP, len(chosen))
            chosen = torch.as_tensor(self._pair_rng.choice(chosen, size=take, replace=False), device=mask.device)
            left = pair_indices[0][chosen]
            right = pair_indices[1][chosen]
            same_cluster = (
                np.full(take, -1, dtype=np.int8)
                if cluster_labels is None
                else (cluster_labels[left] == cluster_labels[right]).cpu().numpy().astype(np.int8)
            )
            self._pair_samples.append(
                {
                    "left": index[left.cpu().numpy()],
                    "right": index[right.cpu().numpy()],
                    "similarity": pair_similarity[chosen].float().cpu().numpy(),
                    "positive": np.full(take, kind, dtype=np.int8),
                    "same_cluster": same_cluster,
                }
            )
            self._pair_sample_counts[kind] += take

    def _save_pair_samples(self, epoch, counts):
        """Write the epoch's mined-pair sample, with the totals it was drawn from."""

        samples, self._pair_samples = self._pair_samples, []
        self._pair_sample_counts = [0, 0]
        if self.artifact_dir is None or not samples:
            return None
        directory = Path(self.artifact_dir) / SLADE_PAIR_DIR
        directory.mkdir(parents=True, exist_ok=True)
        epoch_value = self._pair_epoch_files if epoch is None else int(epoch)
        self._pair_epoch_files += 1
        # The thresholds of the epoch's last mining step (Eq 6 moves them slowly).
        thresholds = getattr(self, "_last_mining_thresholds", None)
        arrays = {
            name: np.concatenate([sample[name] for sample in samples])
            for name in ("left", "right", "similarity", "positive", "same_cluster")
        }
        arrays["similarity"] = arrays["similarity"].astype(np.float16)
        path = directory / f"epoch_{max(epoch_value, 0):04d}.npz"
        np.savez_compressed(
            path,
            **arrays,
            epoch=np.int64(epoch_value),
            mined_positive_pairs=np.float64(counts[_MINED_POSITIVE_PAIRS]),
            mined_negative_pairs=np.float64(counts[_MINED_NEGATIVE_PAIRS]),
            thresholds=np.asarray(
                [np.nan, np.nan] if thresholds is None else [float(value) for value in thresholds], dtype=np.float32
            ),
        )
        return path

    @staticmethod
    def _mining_statistics(counts, steps, active_steps, labeled_steps):
        def ratio(numerator, denominator):
            return float(numerator) / float(denominator) if denominator else None

        statistics = {
            "steps": float(steps),
            "mining_active_fraction": ratio(active_steps, steps),
            "mined_positive_pairs_per_step": ratio(counts[_MINED_POSITIVE_PAIRS], active_steps),
            "steps_without_mined_positive_fraction": ratio(
                counts[_STEPS_WITHOUT_MINED_POSITIVE], active_steps
            ),
            "mined_positive_precision": ratio(
                counts[_MINED_POSITIVE_SAME_CLASS], counts[_MINED_POSITIVE_KNOWN]
            ),
            "mined_negative_same_class_fraction": ratio(
                counts[_MINED_NEGATIVE_SAME_CLASS], counts[_MINED_NEGATIVE_KNOWN]
            ),
            "steps_without_same_class_pair_fraction": (
                None
                if not labeled_steps
                else 1.0 - float(counts[_LABELED_STEPS_WITH_SAME_CLASS_PAIR]) / labeled_steps
            ),
            "steps_missing_an_available_positive_fraction": ratio(
                counts[_STEPS_MISSING_AN_AVAILABLE_POSITIVE], labeled_steps
            ),
            "mined_positive_same_cluster_fraction": ratio(
                counts[_MINED_POSITIVE_SAME_CLUSTER], counts[_MINED_POSITIVE_PAIRS]
            ),
            "mined_negative_same_cluster_fraction": ratio(
                counts[_MINED_NEGATIVE_SAME_CLUSTER], counts[_MINED_NEGATIVE_PAIRS]
            ),
        }
        return {name: value for name, value in statistics.items() if value is not None}

    def _flush_epoch_activity(self, epoch=None):
        counts = (
            [0.0] * _MINING_COUNTER_COUNT
            if self._epoch_mining_counts is None
            else self._epoch_mining_counts.tolist()
        )
        try:
            self._save_pair_samples(epoch, counts)
        except Exception as exc:  # pragma: no cover - diagnostics must not stop training
            logger.warning(f"SLADE mined-pair sample was not saved: {exc}")
        statistics = self._mining_statistics(
            counts,
            self._epoch_steps,
            self._epoch_mining_active_steps,
            self._epoch_labeled_steps,
        )
        self._run_mining_counts = [
            total + value for total, value in zip(self._run_mining_counts, counts)
        ]
        self._run_steps += self._epoch_steps
        self._run_mining_active_steps += self._epoch_mining_active_steps
        self._run_labeled_steps += self._epoch_labeled_steps
        self._epoch_mining_counts = None
        self._epoch_steps = 0
        self._epoch_mining_active_steps = 0
        self._epoch_labeled_steps = 0
        return statistics

    def epoch_activity_diagnostics(self, epoch=None):
        if not self._epoch_steps and self._pending_cluster_quality is None:
            return {}
        diagnostics = {
            f"train/slade/{name}": value
            for name, value in self._flush_epoch_activity(epoch).items()
        }
        if self._pending_cluster_quality is not None:
            diagnostics.update(
                {
                    f"train/slade/cluster_{name}": value
                    for name, value in self._pending_cluster_quality.items()
                    if math.isfinite(value)
                }
            )
            self._pending_cluster_quality = None
        return diagnostics

    def activity_summary(self):
        if self._epoch_steps:
            self._flush_epoch_activity()
        if not self._run_steps and not self._cluster_quality_history:
            return {}
        summary = {
            f"slade_{name}": value
            for name, value in self._mining_statistics(
                self._run_mining_counts,
                self._run_steps,
                self._run_mining_active_steps,
                self._run_labeled_steps,
            ).items()
        }
        if self._cluster_quality_history:
            for name in self._cluster_quality_history[-1]:
                values = [
                    quality[name]
                    for quality in self._cluster_quality_history
                    if math.isfinite(quality[name])
                ]
                if values:
                    summary[f"slade_cluster_{name}_mean"] = float(np.mean(values))
                    summary[f"slade_cluster_{name}_last"] = float(values[-1])
            summary["slade_cluster_refreshes"] = float(len(self._cluster_quality_history))
        return summary
