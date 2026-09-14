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

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

from pytorch_metric_learning import losses

import utils
from .algorithms import faiss_gpu_flat_index, require_faiss
from .config import should_rebuild_on_epoch
from .data import CombinedTrainingLoader, UnlabeledSubset
from .embeddings import extract_embeddings
from .interfaces import BaseTrainingRegularizer


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


def _validate_positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"slade {name} must be finite and positive")
    return value


def _validate_non_negative(name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"slade {name} must be finite and non-negative")
    return value


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

    if backend != "sklearn":
        try:
            faiss = require_faiss("slade pseudo-label clustering")
        except ImportError:
            if backend == "faiss":
                raise
            faiss = None
        if faiss is not None:
            purpose = "slade pseudo-label clustering"
            work = len(features) * num_clusters
            prefer_gpu = (
                work >= DEFAULT_KMEANS_GPU_MIN_WORK if use_gpu is None else bool(use_gpu)
            )
            # Held for as long as the index is: a GPU index does not own them.
            gpu_resources = None
            gpu_index = None
            if prefer_gpu:
                built = faiss_gpu_flat_index(faiss, features.shape[1], purpose)
                if built is not None:
                    gpu_index, gpu_resources = built

            def make_clustering():
                return faiss.Kmeans(
                    d=features.shape[1],
                    k=num_clusters,
                    niter=int(iterations),
                    seed=int(seed) % (2**31 - 1),
                    # A single unlucky initialization can collapse clusters, and
                    # the pseudo labels are the only supervision the unlabeled
                    # stream receives, so keep FAISS's redundant restarts.
                    nredo=max(1, int(n_init) // 5),
                    verbose=False,
                    spherical=False,
                    # The GPU index is attached below instead: gpu=True would
                    # build uncapped resources of its own.
                    gpu=False,
                )

            if gpu_index is not None:
                try:
                    clustering = make_clustering()
                    clustering.index = gpu_index
                    clustering.train(features)
                    _, assignments = clustering.index.search(features, 1)
                    record("faiss-gpu")
                    return np.asarray(assignments, dtype=np.int64).reshape(-1)
                except Exception as exc:
                    logger.warning(
                        f"{purpose}: FAISS GPU k-means failed ({exc}); retrying on CPU"
                    )
                finally:
                    # Drop the index first, then the resources behind it.
                    gpu_index = None
                    clustering = None
                    gpu_resources = None

            clustering = make_clustering()
            clustering.train(features)
            _, assignments = clustering.index.search(features, 1)
            record("faiss-cpu")
            return np.asarray(assignments, dtype=np.int64).reshape(-1)

    from sklearn.cluster import KMeans

    estimator = KMeans(
        n_clusters=num_clusters,
        n_init=int(n_init),
        max_iter=int(iterations),
        random_state=int(seed) % (2**31 - 1),
    )
    labels = np.asarray(estimator.fit_predict(features), dtype=np.int64)
    record("sklearn")
    return labels


def upper_triangle_pairs(count, device):
    """Return the distinct unordered index pairs of a batch."""

    return torch.triu_indices(int(count), int(count), offset=1, device=device)


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
        # Read both flags together; separate scalar reads serialize CUDA work.
        initialized_sides = torch.stack(
            (self.positive_initialized, self.negative_initialized)
        ).tolist()
        sides = (
            ("positive", positive_mean, positive_variance),
            ("negative", negative_mean, negative_variance),
        )
        for was_initialized, (side, mean, variance) in zip(initialized_sides, sides):
            names = (f"{side}_mean", f"{side}_variance")
            initialized = getattr(self, f"{side}_initialized")
            if mean is None or variance is None:
                # Eq 5 still reads this side, just at its stored global value.
                statistics.update(
                    (name, getattr(self, name).detach().clone()) for name in names
                )
                continue
            if was_initialized:
                # Eq 6: mu+ = (1 - beta) * mu+_b + beta * mu+, where mu+_b is the
                # batch statistic. beta is the paper's "updating rate" and weights
                # the *running* estimate, which is what makes the two Gaussians
                # global rather than per-batch at the paper's beta = 0.99.
                updated = tuple(
                    self._blend(name, value, beta, full_gradient)
                    for name, value in zip(names, (mean, variance))
                )
            else:
                # The first batch carrying this pair kind seeds its estimate;
                # interpolating against zeros would otherwise halve the first
                # margin.
                updated = (mean, variance)
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

        # Packing these scalars preserves their values and replaces three
        # consecutive device-to-host waits with one.
        calibrated, positive_mean, negative_mean = torch.stack(
            (self.distributions_initialized, self.positive_mean, self.negative_mean)
        ).tolist()
        if not calibrated:
            return None
        if positive_mean <= negative_mean:
            return None
        return positive_mean, negative_mean


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





class SladeRegularizer(BaseTrainingRegularizer):
    """SLADE's student objective as an unlabeled regularization term (Eq 9)."""

    name = "slade"
    # One deterministic view per unlabeled image is enough, so the frozen
    # backbone feature cache stays usable.
    supports_frozen_feature_precompute = True
    uses_joint_forward = True
    requires_supervised_objective = True
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
        self.unlabeled_batch_size = (
            None if unlabeled_batch_size is None else int(unlabeled_batch_size)
        )
        if self.unlabeled_batch_size is not None and self.unlabeled_batch_size < 2:
            raise ValueError("slade unlabeled_batch_size must be at least two")
        self.unlabeled_ratio = (
            None
            if unlabeled_ratio is None
            else _validate_positive("unlabeled_ratio", unlabeled_ratio)
        )
        if self.unlabeled_ratio is None and self.unlabeled_batch_size is None:
            self.unlabeled_ratio = DEFAULT_UNLABELED_RATIO
        elif self.unlabeled_ratio is not None and self.unlabeled_batch_size is not None:
            logger.warning(
                f"slade received both unlabeled_batch_size={self.unlabeled_batch_size} "
                f"and unlabeled_ratio={self.unlabeled_ratio}; the absolute batch size "
                "wins and the ratio is ignored"
            )
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
        self._last_diagnostics = {}
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
        if self.reset_basis_on_refresh or self.reset_basis_warmup_on_refresh:
            logger.info(
                "SLADE self-training iterations: "
                f"reset_basis_on_refresh={self.reset_basis_on_refresh}, "
                f"reset_basis_warmup_on_refresh={self.reset_basis_warmup_on_refresh}"
            )
        return None

    def build_dataset(self, train_dataset, split, use_cache=False):
        utils.shutdown_dataloaders(self._regularizer_loader)
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_cluster_rebuild_epoch = None
        self._cluster_by_position = None
        if self.regularizer_weight == 0:
            self.dataset = None
            return None
        unlabeled_positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
        if len(unlabeled_positions) < 2:
            raise ValueError("slade requires at least two unlabeled samples")

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
        """

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
            num_clusters=self._resolved_num_clusters,
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
        logger.info(
            f"SLADE pseudo labels - epoch {epoch}: "
            f"{len(clusters)} unlabeled samples, k={self._resolved_num_clusters}, "
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
        first_refresh = self._cluster_by_position is None
        if first_refresh or should_rebuild_on_epoch(
            config.update_mode,
            config.update_interval_epochs,
            epoch,
            self._last_cluster_rebuild_epoch,
        ):
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

        if self.unlabeled_batch_size is not None:
            requested_batch_size = self.unlabeled_batch_size
            sizing = f"unlabeled_batch_size={self.unlabeled_batch_size}"
        else:
            requested_batch_size = max(2, int(round(float(batch_size) * self.unlabeled_ratio)))
            sizing = (
                f"unlabeled_ratio={self.unlabeled_ratio} x labeled batch_size={batch_size}"
            )
        unlabeled_batch_size = min(requested_batch_size, len(self.dataset))
        if unlabeled_batch_size < 2:
            raise ValueError("slade needs at least two unlabeled samples per batch")
        if unlabeled_batch_size < requested_batch_size:
            # ``drop_last`` means a shrunken batch also changes the step count,
            # so silently clamping would move both the Eq 6 statistics and how
            # often Eq 9's unlabeled term fires with nothing in the log saying so.
            logger.warning(
                f"slade requested {requested_batch_size} unlabeled samples per step "
                f"({sizing}) but the unlabeled pool holds only {len(self.dataset)}; "
                f"using {unlabeled_batch_size}"
            )
        worker_count = utils.dataloader_num_workers_for_dataset(self.dataset, num_workers)
        cache_key = (
            id(self.dataset),
            int(unlabeled_batch_size),
            int(worker_count),
            str(start_method),
        )
        if self._regularizer_loader is None or self._regularizer_loader_cache_key != cache_key:
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
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                "SLADE unlabeled loader: "
                f"pool={len(self.dataset)}, batch_size={unlabeled_batch_size} ({sizing}), "
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
    def _mined_pairs(similarity, pair_indices, thresholds):
        """Eq 7: keep pairs the basis similarity is confident about.

        ``mining_thresholds`` guarantees ``T1 > T2``, so the two bands are
        disjoint and no pair is mined as both positive and negative. That
        guarantee is an addition to Eq 7 rather than part of it; see
        ``mining_thresholds`` for why it is kept.
        """

        positive_threshold, negative_threshold = thresholds
        pair_similarity = similarity[pair_indices[0], pair_indices[1]].detach()
        positive_mask = pair_similarity >= positive_threshold
        negative_mask = pair_similarity <= negative_threshold
        # Each mask used to be compacted twice, once per endpoint. On CUDA
        # nonzero synchronizes to learn the output size; reuse the integer IDs.
        positive_indices = positive_mask.nonzero(as_tuple=True)[0]
        negative_indices = negative_mask.nonzero(as_tuple=True)[0]
        return (
            (pair_indices[0][positive_indices], pair_indices[1][positive_indices]),
            (pair_indices[0][negative_indices], pair_indices[1][negative_indices]),
        )

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
            if basis.is_calibrated:
                # Eq 5 is a margin between the two Gaussians, so it needs both of
                # them seeded; until then there is nothing to separate.
                distribution_loss = similarity_distribution_loss(
                    positive_mean=statistics["positive_mean"],
                    positive_variance=statistics["positive_variance"],
                    negative_mean=statistics["negative_mean"],
                    negative_variance=statistics["negative_variance"],
                    margin=self.sd_margin,
                    variance_weight=self.sd_variance_weight,
                )

        thresholds = basis.mining_thresholds()
        ranking_loss = zero
        positive_pairs = (pair_indices[0][:0], pair_indices[1][:0])
        negative_pairs = positive_pairs
        self._mining_active = thresholds is not None and not basis_warmup_active
        if thresholds is not None and not basis_warmup_active:
            positive_pairs, negative_pairs = self._mined_pairs(
                similarity,
                pair_indices,
                thresholds,
            )
            ranking_loss = self._unlabeled_ranking_loss(
                embeddings=regularizer_embeddings,
                cluster_labels=cluster_labels,
                positive_pairs=positive_pairs,
                negative_pairs=negative_pairs,
                supervised_criterion=supervised_criterion,
                supervised_is_classification=supervised_is_classification,
            )

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
                "train/slade/mined_positive_pairs": float(len(positive_pairs[0])),
                "train/slade/mined_negative_pairs": float(len(negative_pairs[0])),
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
