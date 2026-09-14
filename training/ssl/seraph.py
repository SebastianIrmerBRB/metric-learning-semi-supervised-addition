"""SERAPH entropy regularization for deep metric learning.

Implements the *unlabeled* part of "Information-theoretic Semi-supervised Metric
Learning via Entropy Regularization" (Niu et al., ICML 2012, arXiv:1206.4614),
with the paper's equation numbers used throughout.

SERAPH parameterizes a posterior over the weak pair label ``y = +-1`` with a
Mahalanobis metric ``A``, a distance threshold ``eta`` and an inverse temperature
``kappa`` (Eq 5):

    p^A(y | x, x') = 1 / (1 + exp(-kappa * y * (||x - x'||_A^2 - eta)))

so ``p(y = +1) = sigmoid(kappa * (eta - d_A^2))``. Its simplified objective
(Eq 7) is

    max_A  sum_{S u D} ln p(y_ij)                 # labeled pair log-likelihood
           + mu * sum_U sum_y p_ij(y) ln p_ij(y)  # = -mu * H(p) on unlabeled pairs
           - lambda * tr(A)                       # hyper-sparsity / low rank

solved by the EM-like scheme of Sec 2.4. Its E-step (Eq 9)

    min_q KL(q || p^{A(t)}) + mu * E_q[-ln p^A(y)]

has the analytical solution ``q(y) ~ p(y)^(1+mu)``, which in logit space is just
the rescaling ``logit q = (1 + mu) * logit p``. The M-step then maximizes the
same objective with ``q`` held fixed, i.e. a cross entropy against those soft
pair targets. ``objective="em"`` follows that scheme; ``objective="entropy"``
descends Eq 7's entropy term directly instead.

Deep / frozen-feature form
--------------------------
``x`` is the model's L2-normalized retrieval embedding ``z`` (frozen DINO
backbone plus the trainable projection head), and the metric is factored as
``A = B^T B``, scored as ``d_A^2 = ||B (z_i - z_j)||^2``. This is the paper's own
footnote 1 - learning a metric is learning a projection - so ``B`` is the
projection SERAPH learns on top of the embedding. Gradients flow through ``z``
too, so the entropy term shapes the retrieval embedding itself instead of only an
isolated head. Every parameterization of ``B`` (see ``SeraphPosteriorHead``) is
PSD by construction and starts at the plain squared Euclidean distance of the
L2-normalized embedding.

What deviates from the paper, and why
-------------------------------------
1. **The labeled term of Eq 7 is not implemented.** The configured deep metric
   learning loss (triplet, multi-similarity, proxy anchor, ...) is the supervised
   objective, and ``supervised_weight`` / ``regularizer_weight`` take the role of
   the paper's supervised/unsupervised tradeoff ``mu``.
2. **The trace term of Eq 7 is opt-in, and off by default** (``trace_weight``,
   default ``0.0``). ``tr(A)`` is the convex relaxation of ``rank(A)``, and it
   relaxes rank only where ``A``'s scale is identifiable, which deviations 3
   and 4 remove:

   - Under ``threshold_mode`` and ``kappa_mode`` both estimated from the batch's
     own distances, the objective is homogeneous of degree 0 in ``A``: scaling
     ``A -> cA`` sends ``eta -> c eta`` and ``kappa -> kappa / c``, which cancel
     in the logit. ``tr(A)`` is then unopposed and drives ``A -> 0`` instead of
     leashing it. Fixing either calibration restores something for it to push
     against.
   - ``trace_target="projection"`` (below) is scale-free for a second,
     independent reason: ``project_features`` L2-normalizes the head's output,
     so the objective is invariant to ``(W, b) -> (cW, cb)`` for *any*
     calibration. The loss gradient is therefore orthogonal to the head and
     ``tr(A)``'s gradient is parallel to it, so the term rescales the head -- an
     effective learning rate, in the sense of arXiv:1706.05350 -- and cannot
     move its spectrum at all.

   ``trace_mode="soft_rank"`` is the way out that keeps the paper's intent:
   ``tr(A)^2 / ||A||_F^2``, the participation ratio of ``A``'s eigenvalues. It
   is the same rank relaxation with the unidentifiable scale divided out, so it
   is homogeneous of degree 0 like the objective around it, and it concentrates
   the spectrum without touching the norm.

   ``trace_target`` picks which ``A`` is penalized. ``"metric"`` is the paper's
   literal variable, the posterior head's ``A = B^T B``; it is rejected for
   ``metric="identity"``, where ``tr(I) = D`` is a constant. ``"projection"``
   penalizes the metric the *model's* projection head induces -- with frozen
   cached features ``x``, the head maps ``x -> W x + b`` and so induces
   ``A = M^T M`` for ``M = [W | b]``, which is the projection actually being
   learned when the posterior's own metric has no parameters. The bias is
   carried along because penalizing ``W`` alone shrinks the head relative to its
   own bias, pulling the normalized embedding towards the constant direction
   ``b / ||b||``: a collapse rather than a low rank.

   ``trace_weight`` is calibrated rather than taken literally where it can be:
   ``trace_target_ratio`` (default ``0.1`` for the projection target, off for the
   metric one) makes it a probe and sets the weight that hits a measured gradient
   ratio against the supervised loss, because neither a squared Frobenius norm
   nor a dimensionless participation ratio is commensurable with a
   metric-learning loss. ``grad_norm_alpha`` can learn it instead. Both are
   confined to the projection target: the posterior's metric contributes exactly
   zero gradient to the shared trunk, which is what both mechanisms measure over.

   Note the consequence of leaving ``trace_weight`` at ``0``: nothing bounds the
   metric scale, and a learnable ``A`` lets entropy minimization sharpen the
   posterior by inflating ``A`` rather than by separating embeddings.
   ``metric="identity"`` has no scale to inflate.
3. **``kappa`` is kept as an explicit fixed hyperparameter.** Eq 8 drops it
   because Theorem 2 folds it into an unconstrained ``A``; with L2-normalized
   embeddings ``d^2`` is confined to ``[0, 4]``, so with ``metric="identity"``
   there is no ``A`` left to absorb it and ``kappa`` is the only thing that can
   give the posterior a usable confidence range. It is never trainable: that
   would re-add the redundant scale Theorem 2 removes, and entropy minimization
   would inflate it without moving any embedding.
   ``kappa_mode="batch_std"`` divides it by the spread of the batch's own
   distances, which makes the posterior independent of how concentrated the
   frozen embedding's distances happen to be.
4. **``eta`` defaults to a self-calibrated quantile** rather than the paper's
   fixed hyperparameter. The labeled likelihood dropped in (1) was what anchored
   ``eta``; without it, minimizing entropy over a free ``eta`` would simply
   declare every pair dissimilar. ``threshold_mode="quantile"`` places ``eta`` at
   the ``positive_pair_prior`` quantile of the batch distances, which pins the
   fraction of pairs on the similar side and makes that collapse unreachable.
   ``threshold_mode="fixed"`` restores the paper's reading.
5. **Pairs are formed inside the regularizer batch**, so ``U`` is a subset of the
   paper's transductive pair set over the whole pool, and the unlabeled term is a
   mean rather than a sum so its scale does not depend on the batch size.
   ``U`` is also split by pair kind. Unlabeled-unlabeled ("uu") pairs are always
   scored. Labeled-unlabeled ("lu") pairs carry an equally unknown weak label and
   so belong to ``U`` too, but they are opt-in via ``include_labeled_unlabeled``
   because they need their own calibration: labeled embeddings are already
   organized by the supervised loss, so the two blocks' distance distributions
   differ and each keeps a separate running ``eta``. The blocks are then combined
   by their share of the *pool's* pair counts rather than the batch's, so batch
   composition cannot reweight the objective. Labeled-labeled pairs are excluded
   on principle: their weak label is known, which puts them in the likelihood term
   of (1) rather than in an entropy term that would sharpen toward the model's
   current belief instead of the truth.
6. **``mu`` is not a parameter.** In the paper it both weights the unlabeled term
   and sets the E-step's sharpening exponent ``1 + mu``. The weighting is what
   ``regularizer_weight`` already does, so only the exponent survives, as
   ``em_sharpening`` (``2.0`` reproduces the paper's ``mu = 1``).
7. **Optimization is plain SGD**, not the paper's alternating convex M-step: the
   E-step of Eq 9 is applied per batch to the current posterior (detached), and
   the PSD constraint needs no projection step because ``A = B^T B``.
8. **``threshold_mode`` also offers three per-anchor calibrations.** (4) sets one
   ``eta`` for the whole block, which fixes a global positive rate. Fixing an
   *effective neighbor count per anchor* instead is the stronger version of the
   same idea, and three mature methods do exactly that. Each is a separate mode,
   each is cited where it is implemented, and each keeps its own paper's
   symmetrization, because ``eta`` has to stay a property of the *pair* and a
   per-anchor quantity is not one:

   ``self_tuning``
       Zelnik-Manor and Perona, "Self-Tuning Spectral Clustering" (NIPS 2004).
       ``sigma_i`` is the distance from ``i`` to its ``K``-th neighbor (the
       paper's ``K = 7``) and the affinity uses the two scales jointly as
       ``exp(-d^2 / (sigma_i sigma_j))``, so ``eta_ij = sigma_i sigma_j``. In
       this module's squared-distance units that is the geometric mean of the two
       anchors' ``eta_i = sigma_i^2``, and it is symmetric by construction, so
       this mode needs no second step.
   ``umap``
       McInnes, Healy and Melville (arXiv:1802.03426, Sec 3.1). ``rho_i`` is the
       distance to the nearest neighbor and ``sigma_i`` is calibrated per point
       by bisection so the neighbor weights ``exp(-(d - rho_i) / sigma_i)`` sum
       to ``log2(k)``. That weight passes ``1/2`` at ``d = rho_i + sigma_i ln 2``,
       which is exactly where SERAPH's posterior passes ``1/2``, so
       ``eta_i = (rho_i + sigma_i ln 2)^2``. The two directed posteriors are then
       merged by UMAP's probabilistic t-conorm ``p_ij + p_ji - p_ij p_ji``.
   ``tsne``
       van der Maaten and Hinton (JMLR 2008, Sec 2). ``sigma_j`` is found by
       bisection so that the perplexity ``2^H(p_.|i)`` of the conditional
       ``p_j|i ~ exp(-d^2 / (2 sigma_i^2))`` matches the configured value,
       searched over the ``3 * perplexity`` nearest neighbors of the Barnes-Hut
       variant rather than the whole pool. The kernel passes ``1/2`` at
       ``d^2 = 2 sigma_i^2 ln 2``, giving ``eta_i`` directly, and the two directed
       posteriors are averaged as t-SNE averages its two conditionals. Its
       ``1 / 2n`` normalization is dropped: SERAPH's ``p`` is a per-pair
       Bernoulli, not a joint distribution over all pairs.

   All three read their neighbors from the memory bank of (9) when one is
   configured. Without it the k-th order statistic is taken over the batch alone,
   which at these batch sizes is noisy enough to be the mode's dominant error
   term. Per-anchor ``eta`` is recomputed every step instead of being smoothed by
   ``threshold_momentum``: the anchors change every step, so a running estimate
   has nothing to attach to, and the bank supplies the stability the EMA
   supplied before.
9. **An optional memory bank of recent unlabeled embeddings.** ``eta`` and
   ``kappa`` are estimated from batch statistics, and the quantile of (4) is the
   worst case: at ``C = 200`` a prior of ``1 / C`` over the roughly 2000 pairs of
   a 64-sample batch lands on about the 10th order statistic, whose sampling
   noise the EMA can smooth but not remove. ``memory_bank_size`` keeps a FIFO of
   the last N unlabeled *retrieval* embeddings and adds the batch-to-bank
   distances to the sample that the quantile, the ``batch_std`` kappa and the
   per-anchor neighbor searches read. It stores the retrieval embedding rather
   than the projected one, so a trainable ``B`` still applies at its current
   value; what it cannot undo is that the embeddings themselves are stale, as in
   any MoCo-style bank. The pairs the objective *scores* are unchanged - the bank
   calibrates the posterior, it does not add terms to the loss.
10. **``prior_mode="freematch"`` lets the target rate track learning status.**
    (4) pins the similar-pair fraction at ``positive_pair_prior``, which defaults
    to ``1 / C``: a balanced-class, stationary target. FreeMatch (Wang et al.,
    "FreeMatch: Self-adaptive Thresholding for Semi-supervised Learning", ICLR
    2023, arXiv:2205.07246) replaces exactly that kind of fixed threshold with
    EMAs of the model's own predictions on unlabeled data, and this mode is its
    binary-pair-label form: the weak label ``y = +-1`` is the two-outcome
    variable its ``C`` classes become. Its global threshold (Eq 9) rises as the
    mean pair confidence does, its per-class list (Eq 10) and MaxNorm (Eq 11)
    give the rarer of the two pair labels a lower bar, and the fraction of pairs
    that pass on the similar side - not a constant - becomes the quantile level
    ``eta`` is placed at. ``fairness_weight`` adds FreeMatch's self-adaptive
    fairness penalty (Sec 3.3), which is what keeps a rising threshold from
    ratcheting the rate towards the all-dissimilar solution.
11. **``threshold_mode="labeled_logistic"`` calibrates on the labeled pairs.**
    (5) excludes labeled-labeled pairs from ``U`` because their weak label is
    known - but that is exactly what makes them a *calibration set* rather than
    an objective term. A one-dimensional logistic regression of ``y`` on ``d_A^2``
    over that block gives ``kappa`` and ``eta`` jointly, since Eq 5's logit is
    ``a d_A^2 + b`` with ``a = -kappa`` and ``b = kappa eta``. The fit is
    detached, EMA'd across steps, and never enters the loss, which restores what
    the dropped Eq-7 likelihood was doing for ``eta`` without restoring the term
    itself. ``kappa_mode`` must select it too: one fit yields both numbers.

    The precedent is close to exact. SigLIP (Zhai et al., ICCV 2023,
    arXiv:2303.15343, Sec 2.2) scores each pair with a sigmoid of a learnable
    temperature times the similarity plus a learnable bias - the same two degrees
    of freedom as ``kappa`` and ``kappa * eta`` - and initializes that bias to
    ``-10`` so the heavy negative majority does not dominate the loss at
    initialization. The class-balanced form of the same trick sets it to the
    logit of the prior, ``b = -log((1 - pi) / pi)``, which is what the opt-in
    ``prior_correction`` computes here from the fit's own base rate rather than
    from a hand-chosen constant - see ``prior_corrected_intercept`` for why it is
    opt-in. HIB (Oh et al., "Modeling Uncertainty with
    Hedged Instance Embeddings", ICLR 2019, arXiv:1810.00319) is the
    metric-learning ancestor: a sigmoid of ``-a ||z - z'||^2 + b`` with both
    scalars tunable, anchored by a labeled match likelihood - which is this
    mode's calibration set, one paper earlier.

    Two consequences worth stating:

    * That literature also removes weight decay from the bias specifically, and
      here that comes for free: ``kappa`` and ``eta`` are buffers filled by a
      detached fit, not parameters, so the ``weight_decay`` that (2) deliberately
      aims at this head cannot reach them. ``eta`` is a buffer under every other
      mode too; what changes is that this one has a fitted value worth
      protecting.
    * The fit is single, and it is applied to both pair blocks. That is Eq 7's
      own structure - one ``(A, eta, kappa)`` shared by the likelihood over
      ``S u D`` and the entropy over ``U`` - but it does give up the per-block
      calibration (5) introduced, so the mismatch between labeled and unlabeled
      distance distributions lands on the fit instead of being split around it.
      ``prior_correction`` addresses the part of that mismatch that is a base
      rate; nothing here addresses the part that is a shape.
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

import utils
from .data import CombinedTrainingLoader, UnlabeledSubset
from .interfaces import BaseTrainingRegularizer


# "em" follows Sec 2.4: the analytical E-step of Eq 9 followed by a cross entropy
# M-step. "entropy" descends the entropy term of Eq 7 directly.
SERAPH_OBJECTIVES = ("em", "entropy")

# Parameterizations of A = B^T B, all PSD by construction and all starting at the
# plain squared Euclidean distance of the L2-normalized embedding.
SERAPH_METRICS = ("identity", "lowrank", "diag", "scale")

# Per-anchor eta: each mode fixes an effective neighbor count for every anchor
# instead of a global positive rate, and each carries its own symmetrization back
# to a pair quantity. See deviation 8 of the module docstring for the citations.
PER_ANCHOR_THRESHOLD_MODES = ("self_tuning", "umap", "tsne")

THRESHOLD_MODES = ("quantile", "fixed", "labeled_logistic") + PER_ANCHOR_THRESHOLD_MODES

# How each per-anchor mode turns the two anchors' eta into one pair posterior.
# "scale_product" is symmetric already and stays in eta space; the other two
# score both directions and merge the resulting posteriors.
PER_ANCHOR_SYMMETRIZATIONS = {
    "self_tuning": "scale_product",
    "umap": "t_conorm",
    "tsne": "mean",
}

# Which A the trace term of Eq 7 is taken of. "metric" is the paper's literal
# variable, the posterior head's own A = B^T B. "projection" is the A induced by
# the model's trainable projection head, which is the A that exists when
# metric="identity" leaves the head with no parameters of its own: the head maps
# the frozen feature x to W x + b, so the metric it induces on the feature space
# is A = W^T W and the paper's tr(A) is ||W||_F^2. Footnote 1's "learning a
# metric is learning a projection" read backwards.
TRACE_TARGETS = ("metric", "projection")

# "nuclear" is Eq 7's term as written, tr(A) = ||A||_* for PSD A, the convex
# relaxation of rank(A). "soft_rank" is tr(A)^2 / ||A||_F^2, the same relaxation
# with the scale divided out; see the trace-term deviation in the module
# docstring for why the difference is the whole story here.
TRACE_MODES = ("nuclear", "soft_rank")

# Component name for the fairness term of arXiv:2205.07246, reported alongside
# the trace term wherever the trainer names objective terms.
FAIRNESS_COMPONENT = "seraph_fairness"

# Component name for the trace term wherever the trainer reports, calibrates or
# GradNorm-balances objective terms by name.
TRACE_COMPONENT = "seraph_trace"

# The trace's units are arbitrary -- a participation ratio and a squared
# Frobenius norm are not commensurable with a metric-learning loss -- so
# calibrating lambda from measured gradients is the honest way to pick it, and
# ``trace_target="projection"`` defaults to doing so. The value is deliberately
# an order of magnitude below the supervised gradient: a rank prior should be
# visible in the objective, not competitive with it. It is a starting point, not
# a tuned constant.
DEFAULT_TRACE_TARGET_RATIO = 0.1

KAPPA_MODES = ("fixed", "batch_std", "labeled_logistic")

# "labeled_logistic" produces kappa and eta from one fit, so it is the one value
# that has to appear in both enums at once.
JOINT_CALIBRATION_MODE = "labeled_logistic"

# "fixed" keeps positive_pair_prior as configured. "freematch" replaces it with
# the self-adaptive rate of arXiv:2205.07246.
PRIOR_MODES = ("fixed", "freematch")

# The labeled-pair fit: two parameters, so Newton converges in a few steps and
# the budget is there for the pathological batch rather than the typical one.
NEWTON_STEPS = 25
NEWTON_BACKTRACKS = 20
CALIBRATION_RIDGE = 1e-6

# A fit whose slope is not clearly negative says larger d^2 does not mean less
# similar, which is not a calibration - it is a broken embedding, and kappa = -a
# would come out non-positive. Such a step is skipped rather than applied.
MIN_CALIBRATION_SLOPE = 1e-6

# Zelnik-Manor and Perona use K = 7 throughout their experiments (NIPS 2004,
# Sec 3); UMAP's own default n_neighbors is 15 (arXiv:1802.03426, Sec 4).
DEFAULT_NEIGHBORS = {"self_tuning": 7, "umap": 15}

# t-SNE's usual range is 5 to 50 with 30 the common default (JMLR 2008, Sec 2).
DEFAULT_PERPLEXITY = 30.0

# Barnes-Hut t-SNE searches 3 * perplexity neighbors per point, which is the
# bound this module borrows so every per-anchor mode costs one topk.
TSNE_NEIGHBORS_PER_PERPLEXITY = 3

# Bisection budget for the UMAP and t-SNE bandwidth solves. UMAP's own
# smooth_knn_dist runs 64 iterations; t-SNE's reference code runs 50.
BISECTION_STEPS = 64

# FreeMatch's EMA weight on the *previous* value (its lambda; the paper and the
# reference implementation both use 0.999). Note this is the opposite convention
# from threshold_momentum next door, which weights the new value.
DEFAULT_FREEMATCH_MOMENTUM = 0.999

# The adaptive rate may travel one order of magnitude either side of the
# balanced-class prior unless the config says otherwise. A rate of exactly zero
# would put eta at the smallest distance in the sample, which is the
# all-dissimilar collapse deviation 4 exists to prevent.
DEFAULT_PRIOR_RANGE = 10.0

# Beyond this the posterior is saturated to the last float32 bit and the entropy
# term's gradient is ~1e-12, but the t-conorm can still compose two saturated
# logits into an infinity. Clamping the directed logits keeps that finite.
MAX_DIRECTED_LOGIT = 30.0

EPSILON = 1e-12

# The two pair blocks the unlabeled term can score. Both carry an unknown weak
# label, which is what puts them in the paper's U; labeled-labeled pairs have a
# known label and belong to the likelihood term this project delegates to the
# configured supervised loss.
PAIR_BLOCKS = ("uu", "lu")

# Used when neither unlabeled_batch_size nor unlabeled_ratio is configured: one
# unlabeled sample per labeled sample.
DEFAULT_UNLABELED_RATIO = 1.0

# q(y) ~ p(y)^exponent in Eq 9's E-step. 2.0 is the paper's mu = 1.
DEFAULT_EM_SHARPENING = 2.0


def _validate_positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"seraph {name} must be finite and positive")
    return value


def _validate_non_negative(name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"seraph {name} must be finite and non-negative")
    return value


def _validate_probability(name, value):
    value = float(value)
    if not math.isfinite(value) or not (0.0 < value < 1.0):
        raise ValueError(f"seraph {name} must lie strictly between 0 and 1")
    return value


def _validate_positive_integer(name, value):
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"seraph {name} must be a positive integer")
    return integer


def cross_squared_distances(left, right):
    """Return the ``[len(left), len(right)]`` squared distances between two sets.

    The labeled-unlabeled block needs every cross pair but no within-set pair, so
    this is the rectangular counterpart of ``pairwise_squared_distances`` and
    carries the same Gram-form caveat about cancellation and the same clamp.
    """

    left_norms = left.pow(2).sum(dim=1, keepdim=True)
    right_norms = right.pow(2).sum(dim=1, keepdim=True)
    distances = left_norms + right_norms.t() - 2.0 * (left @ right.t())
    return distances.clamp_min(0.0)


def pairwise_squared_distances(projected):
    """Return the squared Euclidean distance matrix of already projected rows.

    The Gram form keeps memory at ``O(n^2)`` instead of ``O(n^2 d)``, which
    matters because every pair of the batch is scored. Rounding can make the
    difference slightly negative, so the result is clamped at zero.
    """

    squared_norms = projected.pow(2).sum(dim=1, keepdim=True)
    distances = squared_norms + squared_norms.t() - 2.0 * (projected @ projected.t())
    return distances.clamp_min(0.0)


def binary_entropy_from_logits(logits):
    """Return ``H(p)`` in nats for ``p = sigmoid(logits)``, elementwise.

    ``ln p = -softplus(-l)`` and ``ln(1 - p) = -softplus(l)`` keep this finite for
    saturated logits, where a naive ``p * log(p)`` would produce ``0 * -inf`` -
    and a confident posterior is exactly what the objective drives towards.
    """

    probabilities = torch.sigmoid(logits)
    return probabilities * F.softplus(-logits) + (1.0 - probabilities) * F.softplus(logits)


def sharpen_logits(logits, exponent):
    """Eq 9's analytical E-step: ``q(y) ~ p(y)^exponent``.

    In logit space the solution of ``min_q KL(q || p) + mu * E_q[-ln p]`` is the
    rescaling below with ``exponent = 1 + mu``, so the E-step is a temperature
    ``1 / exponent`` sharpening of the current posterior. The result is detached
    because the M-step treats ``q`` as fixed data.

    The exponent is passed directly rather than as the paper's ``mu``: ``mu`` also
    scaled the whole unlabeled term, which ``regularizer_weight`` already does, so
    conflating the two meant one number moved both the objective's weight and its
    E-step temperature.
    """

    return float(exponent) * logits.detach()


def log1mexp(values):
    """Return ``ln(1 - exp(x))`` for ``x < 0`` without losing the small-x case.

    ``exp`` then ``log1p`` cancels catastrophically as ``x -> 0``, and ``expm1``
    then ``log`` does the same as ``x -> -inf``, so each branch is used where it
    is the accurate one. The switch sits at ``-ln 2``, where the two are equally
    good.
    """

    switch = -math.log(2.0)
    # Both branches are evaluated, so each one's input is kept inside its own
    # domain to stop the unused half from manufacturing a NaN.
    below = values.clamp_max(switch)
    above = values.clamp_min(switch)
    return torch.where(
        values > switch,
        torch.log(-torch.expm1(above)),
        torch.log1p(-torch.exp(below)),
    )


def self_tuning_scales(neighbor_squared_distances, neighbors):
    """Return ``eta_i = sigma_i^2`` for self-tuning spectral clustering.

    Zelnik-Manor and Perona, "Self-Tuning Spectral Clustering" (NIPS 2004),
    Sec 2: ``sigma_i`` is the distance from ``i`` to its ``K``-th neighbor, and
    the affinity uses the two scales jointly as ``exp(-d^2 / (sigma_i sigma_j))``.
    The paper's ``K = 7`` is this module's default.

    ``sigma_i`` is returned squared so every per-anchor mode speaks the same
    units as ``d_A^2``; the pair threshold ``sigma_i sigma_j`` is then the
    geometric mean of two such values, which is what
    ``PER_ANCHOR_SYMMETRIZATIONS["self_tuning"]`` selects.
    """

    kth = torch.topk(
        neighbor_squared_distances,
        k=int(neighbors),
        dim=1,
        largest=False,
        sorted=True,
    ).values[:, -1]
    return kth.clamp_min(0.0)


def umap_scales(neighbor_squared_distances, neighbors, steps=BISECTION_STEPS):
    """Return ``eta_i`` from UMAP's per-point bandwidth calibration.

    McInnes, Healy and Melville (arXiv:1802.03426, Sec 3.1 and Alg 3): ``rho_i``
    is the distance to the nearest neighbor and ``sigma_i`` is chosen so that

        sum_j exp(-(d(i, j) - rho_i)_+ / sigma_i) = log2(k)

    over the ``k`` nearest neighbors, which fixes the *effective* neighbor count
    at ``log2(k)`` for every point regardless of local density. UMAP solves it by
    bisection with a doubling upper bracket; this is that loop vectorized over
    anchors, so every row runs the same fixed number of steps instead of
    stopping at its own tolerance.

    The returned ``eta_i = (rho_i + sigma_i ln 2)^2`` is the squared distance at
    which UMAP's membership strength passes ``1/2`` - the same crossing SERAPH's
    posterior has at ``d_A^2 = eta``. UMAP works in distance rather than squared
    distance, so the calibration runs on the square root and the result is
    squared back.
    """

    neighbors = int(neighbors)
    nearest = torch.topk(
        neighbor_squared_distances,
        k=neighbors,
        dim=1,
        largest=False,
        sorted=True,
    ).values.clamp_min(0.0).sqrt()
    rho = nearest[:, :1]
    # UMAP's `d - rho` is clamped at zero, so every neighbor at or inside rho
    # contributes a full 1.0 to the sum.
    offsets = (nearest - rho).clamp_min(0.0)
    target = math.log2(neighbors) if neighbors > 1 else 1.0

    low = torch.zeros_like(rho[:, 0])
    high = torch.full_like(low, math.inf)
    middle = torch.ones_like(low)
    for _ in range(int(steps)):
        weights = torch.exp(-offsets / middle.clamp_min(EPSILON).unsqueeze(1))
        overshoot = weights.sum(dim=1) > target
        high = torch.where(overshoot, middle, high)
        low = torch.where(overshoot, low, middle)
        # An infinite upper bracket has not been found yet, so double instead of
        # bisecting - Alg 3's `mid *= 2` branch.
        middle = torch.where(torch.isinf(high), low * 2.0, 0.5 * (low + high))

    # Alg 3's MIN_K_DIST_SCALE floor, which stops a point sitting on top of its
    # neighbors from getting a zero bandwidth.
    floor = 1e-3 * nearest.mean(dim=1)
    sigma = torch.maximum(middle, floor).clamp_min(EPSILON)
    half_membership = rho[:, 0] + sigma * math.log(2.0)
    return half_membership.pow(2)


def tsne_scales(
    neighbor_squared_distances,
    perplexity,
    max_neighbors=None,
    steps=BISECTION_STEPS,
):
    """Return ``eta_i`` from t-SNE's per-point perplexity calibration.

    van der Maaten and Hinton (JMLR 2008, Sec 2): ``sigma_i`` is found by binary
    search so that the conditional ``p_j|i ~ exp(-d^2 / (2 sigma_i^2))`` has a
    fixed perplexity ``2^H(p_.|i)``, which again fixes an effective neighbor
    count per point. The search runs on ``beta = 1 / (2 sigma_i^2)`` as the
    reference implementation does, and entropy is accumulated in nats against
    ``ln(perplexity)`` - the same condition as ``2^H`` in bits.

    Only the ``3 * perplexity`` nearest neighbors are searched, which is the
    neighbor budget of the Barnes-Hut variant (van der Maaten, JMLR 2014,
    Sec 3.1) rather than the original's full row.

    The returned ``eta_i = 2 sigma_i^2 ln 2 = ln 2 / beta_i`` is the squared
    distance at which the kernel passes ``1/2``, matching SERAPH's posterior
    crossing.
    """

    perplexity = float(perplexity)
    if max_neighbors is None:
        max_neighbors = neighbor_squared_distances.shape[1]
    budget = max(
        2,
        min(int(math.ceil(TSNE_NEIGHBORS_PER_PERPLEXITY * perplexity)), int(max_neighbors)),
    )
    nearest = torch.topk(
        neighbor_squared_distances,
        k=budget,
        dim=1,
        largest=False,
        sorted=False,
    ).values.clamp_min(0.0)
    # A perplexity the neighbor budget cannot reach would send the bisection to
    # its upper bracket and stay there, so the target is capped at the entropy of
    # a uniform distribution over the neighbors actually searched.
    target_entropy = min(math.log(perplexity), math.log(budget))

    beta = torch.ones(len(nearest), device=nearest.device, dtype=nearest.dtype)
    # beta > 0 always, so zero doubles as "no lower bracket yet" and drives the
    # reference implementation's `beta / 2` branch.
    low = torch.zeros_like(beta)
    high = torch.full_like(beta, math.inf)
    for _ in range(int(steps)):
        log_conditional = F.log_softmax(-beta.unsqueeze(1) * nearest, dim=1)
        weights = log_conditional.exp()
        # A neighbor that underflows to zero probability contributes nothing, but
        # writing that as 0 * -inf would make it a NaN instead.
        entropy = -torch.where(
            weights > 0,
            weights * log_conditional,
            torch.zeros_like(weights),
        ).sum(dim=1)
        # Entropy falls as beta rises, so too much entropy means too small a beta.
        too_broad = entropy > target_entropy
        low = torch.where(too_broad, beta, low)
        high = torch.where(too_broad, high, beta)
        beta = torch.where(
            too_broad,
            torch.where(torch.isinf(high), beta * 2.0, 0.5 * (low + high)),
            torch.where(low > 0, 0.5 * (low + high), beta * 0.5),
        )

    return math.log(2.0) / beta.clamp_min(EPSILON)


def fit_pair_logistic(
    squared_distances,
    targets,
    slope,
    intercept,
    steps=NEWTON_STEPS,
    ridge=CALIBRATION_RIDGE,
):
    """Fit ``p(y = +1) = sigmoid(a d^2 + b)`` by regularized Newton descent.

    This is the one-dimensional logistic regression behind
    ``threshold_mode="labeled_logistic"``: two parameters, so the Newton system is
    a 2x2 solve written out in closed form and the whole fit costs a handful of
    reductions over the pair vector.

    Two things keep it from diverging on the pair sets it actually sees:

    * The targets are Platt's smoothed ones (Platt, "Probabilistic Outputs for
      Support Vector Machines", 1999, Sec 2.1): ``(N+ + 1) / (N+ + 2)`` for the
      positives and ``1 / (N- + 2)`` for the negatives. A labeled batch whose
      similar and dissimilar pairs are linearly separable in ``d^2`` - which a
      well-trained metric makes likely - has no finite maximum-likelihood fit at
      all, and these targets are what put one back.
    * A ridge on both parameters and a backtracking line search on the same
      penalized objective, because a Newton step taken from a nearly singular
      Hessian is otherwise free to jump anywhere.

    Runs in float64: the Hessian entry ``sum w d^4`` spans twice the dynamic
    range of the distances themselves, and there are only two parameters to
    solve for, so the precision is nearly free.

    Returns ``(a, b)`` as float64 scalars; the caller decides whether the fit is
    usable and how to blend it into the running calibration.
    """

    x = squared_distances.detach().double().flatten()
    y = targets.detach().double().flatten()
    positives = float(y.sum())
    negatives = float(len(y) - positives)
    smoothed = torch.where(
        y > 0.5,
        torch.full_like(y, (positives + 1.0) / (positives + 2.0)),
        torch.full_like(y, 1.0 / (negatives + 2.0)),
    )

    slope = torch.as_tensor(float(slope), dtype=torch.float64, device=x.device)
    intercept = torch.as_tensor(float(intercept), dtype=torch.float64, device=x.device)
    ridge = float(ridge)

    def objective(a, b):
        z = a * x + b
        return float(
            (smoothed * F.softplus(-z) + (1.0 - smoothed) * F.softplus(z)).mean()
            + 0.5 * ridge * (a * a + b * b)
        )

    current = objective(slope, intercept)
    for _ in range(int(steps)):
        z = slope * x + intercept
        probabilities = torch.sigmoid(z)
        residual = probabilities - smoothed
        weights = probabilities * (1.0 - probabilities)
        # Means rather than sums, so the ridge means the same thing whatever the
        # pair count is.
        gradient_slope = (residual * x).mean() + ridge * slope
        gradient_intercept = residual.mean() + ridge * intercept
        hessian_ss = (weights * x * x).mean() + ridge
        hessian_si = (weights * x).mean()
        hessian_ii = weights.mean() + ridge
        determinant = hessian_ss * hessian_ii - hessian_si * hessian_si
        if not torch.isfinite(determinant) or float(determinant) <= 0.0:
            break
        step_slope = (hessian_ii * gradient_slope - hessian_si * gradient_intercept) / determinant
        step_intercept = (hessian_ss * gradient_intercept - hessian_si * gradient_slope) / determinant

        scale = 1.0
        improved = False
        for _ in range(NEWTON_BACKTRACKS):
            candidate_slope = slope - scale * step_slope
            candidate_intercept = intercept - scale * step_intercept
            candidate = objective(candidate_slope, candidate_intercept)
            if math.isfinite(candidate) and candidate <= current:
                slope, intercept, current = candidate_slope, candidate_intercept, candidate
                improved = True
                break
            scale *= 0.5
        if not improved:
            break

    return float(slope), float(intercept)


def prior_corrected_intercept(intercept, sample_rate, target_rate):
    """Shift a logistic intercept from the fitted rate to a target base rate.

    The labeled batch's positive-pair rate is set by the M-per-class sampler, not
    by ``C``: with ``M`` samples from each of ``B / M`` classes it is roughly
    ``(M - 1) / (B - 1)``, which at typical settings is an order of magnitude
    above the ``1 / C`` an unlabeled pair actually has. Fitting there and scoring
    here is case-control sampling, whose standard correction leaves the slope
    alone and moves the intercept by the difference of the two log-odds (King and
    Zeng, "Logistic Regression in Rare Events Data", 2001, Sec 4.1).

    It is the same expression the class-balanced sigmoid loss uses when it
    initializes its bias to ``-log((1 - pi) / pi)``, and the data-driven version
    of SigLIP's hand-set ``-10``: all three are a bias that carries the base rate
    so the slope does not have to.

    Off by default, because it is only sound where the fit is. The shift moves
    ``eta`` by ``(logit(target) - logit(sample)) / kappa``, which a shallow slope
    turns into a large move, and a metric that has not separated its classes yet
    has a shallow slope. Measured on an untrained embedding it takes a healthy
    ``eta`` of 0.70 to ``-0.02`` - the all-dissimilar collapse - in one step. The
    correction is directionally right the whole time (a rarer base rate really
    does mean a nearer half-way distance); what it cannot do is stay inside the
    range where a linear-in-``d^2`` logit is a reasonable model. Turn it on with
    ``calibration/step_intercept_raw`` and ``calibration/threshold_clamped`` in
    view.
    """

    sample_rate = min(max(float(sample_rate), EPSILON), 1.0 - EPSILON)
    target_rate = min(max(float(target_rate), EPSILON), 1.0 - EPSILON)
    return (
        float(intercept)
        - math.log(sample_rate / (1.0 - sample_rate))
        + math.log(target_rate / (1.0 - target_rate))
    )


def combine_directed_logits(left_logits, right_logits, symmetrization):
    """Merge the two directions of a per-anchor posterior into one pair logit.

    ``eta_i`` is a property of an anchor, so scoring pair ``(i, j)`` against
    ``eta_i`` and against ``eta_j`` gives two different posteriors for the same
    unordered pair. Each mode merges them the way its own paper merges its
    directed neighbor weights:

    ``t_conorm``
        UMAP's probabilistic t-conorm ``p_ij + p_ji - p_ij p_ji``
        (arXiv:1802.03426, Sec 3.1), the fuzzy-set union of the two memberships.
        It is evaluated through ``1 - p = (1 - p_ij)(1 - p_ji)``, whose log is a
        sum of ``softplus`` terms, because forming the union directly loses the
        whole confident tail to rounding.
    ``mean``
        t-SNE's ``(p_j|i + p_i|j) / 2`` (JMLR 2008, Eq 4), in log space so a
        saturated direction cannot round the other one away. The ``1/2`` cancels
        between the two log-sums and never has to be written down.

    Both inputs are clamped first: two saturated directions compose to a
    posterior that is 1 to the last float32 bit, and the resulting infinite logit
    would turn the entropy term's ``0 * inf`` into a NaN.
    """

    left_logits = left_logits.clamp(-MAX_DIRECTED_LOGIT, MAX_DIRECTED_LOGIT)
    right_logits = right_logits.clamp(-MAX_DIRECTED_LOGIT, MAX_DIRECTED_LOGIT)
    if symmetrization == "t_conorm":
        log_negative = -(F.softplus(left_logits) + F.softplus(right_logits))
        return log1mexp(log_negative) - log_negative
    if symmetrization == "mean":
        log_positive = torch.logaddexp(
            F.logsigmoid(left_logits), F.logsigmoid(right_logits)
        )
        log_negative = torch.logaddexp(
            F.logsigmoid(-left_logits), F.logsigmoid(-right_logits)
        )
        return log_positive - log_negative
    raise ValueError(f"seraph symmetrization must be one of {list(PER_ANCHOR_SYMMETRIZATIONS.values())}")


def histogram_normalized(marginal, histogram):
    """Return FreeMatch's ``SumNorm(p / h)`` (arXiv:2205.07246, Sec 3.3).

    Dividing the mean posterior by the frequency of the corresponding hard
    prediction is what makes the fairness term self-adaptive rather than a push
    towards a uniform label marginal - which for a pair label would be a demand
    that half of all pairs be similar, when the truth is nearer ``1 / C``. A
    calibrated model has ``p ~ h`` in both entries, so the reference sits at
    ``[0.5, 0.5]`` no matter how rare similar pairs are.
    """

    scaled = marginal / histogram
    return scaled / scaled.sum().clamp_min(EPSILON)


def factored_trace_terms(matrix):
    """Return ``(tr(A), ||A||_F^2)`` for ``A = M^T M``, without forming ``A``.

    ``tr(M^T M) = ||M||_F^2`` needs no product at all, and
    ``||M^T M||_F^2 = ||M M^T||_F^2`` lets the Gram matrix be taken on whichever
    side is smaller, so a ``64 x 768`` factor costs a ``64 x 64`` product rather
    than a ``768 x 768`` one.
    """

    matrix = matrix.float()
    rows, cols = matrix.shape
    gram = matrix @ matrix.t() if rows <= cols else matrix.t() @ matrix
    return matrix.pow(2).sum(), gram.pow(2).sum()


def affine_map(linear):
    """Return one ``nn.Linear`` as the single matrix ``[W | b]``.

    The paper's ``A`` is the metric of a *linear* map, and this head is affine.
    Augmenting the input with a constant 1 turns it back into a linear map, so
    ``A = M^T M`` stays the metric the head actually induces instead of the
    metric of the half of it that happens to have no bias. That difference is
    not cosmetic: penalizing ``W`` while leaving ``b`` alone shrinks the head
    relative to its own bias, which drives the normalized embedding towards the
    constant direction ``b / ||b||`` -- a collapse, not a low rank.
    """

    weight = linear.weight.float()
    if linear.bias is None:
        return weight
    return torch.cat([weight, linear.bias.float().unsqueeze(1)], dim=1)


def projection_affine_maps(projection_head):
    """Return the model projection head's linear maps, input side first."""

    return [affine_map(module) for module in projection_head.modules()
            if isinstance(module, nn.Linear)]


class SeraphPosteriorHead(nn.Module):
    """The metric ``A``, threshold ``eta`` and temperature of the pair posterior.

    ``A = B^T B`` is never materialized: ``project`` applies ``B`` and the
    posterior reads the squared Euclidean distance in that projected space, which
    is ``d_A^2``. The parameterizations differ only in the shape of ``B``:

    ``identity``
        ``A = I``, no metric parameters at all. The posterior scores the
        retrieval embedding directly, so only the embedding itself can lower the
        entropy and there is no metric scale for it to inflate instead.
    ``lowrank``
        ``B = L`` of shape ``metric_dim x feat_dim``, the paper's own setting: a
        full PSD metric whose induced projection Sec 2.3 pushes towards low rank.
    ``diag``
        ``B = diag(sqrt(a))`` with ``a = softplus(raw) >= 0``, a per-dimension
        weighting of the embedding the posterior trusts.
    ``scale``
        ``B = sqrt(s) I``, a single learnable scalar.

    It also owns everything the posterior is calibrated from, because all of it
    is state a checkpoint has to carry: the per-block running ``eta``, the
    optional memory bank of recent retrieval embeddings (deviation 9) and the
    optional FreeMatch EMAs behind the self-adaptive rate (deviation 10).
    """

    def __init__(
        self,
        feat_dim,
        metric="identity",
        metric_dim=None,
        threshold=1.0,
        threshold_mode="quantile",
        threshold_momentum=0.1,
        positive_pair_prior=0.01,
        kappa=1.0,
        kappa_mode="batch_std",
        neighbors=None,
        perplexity=DEFAULT_PERPLEXITY,
        memory_bank_size=None,
        prior_mode="fixed",
        freematch_momentum=DEFAULT_FREEMATCH_MOMENTUM,
        positive_pair_prior_min=None,
        positive_pair_prior_max=None,
        track_fairness=False,
        prior_correction=False,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        if self.feat_dim <= 0:
            raise ValueError("seraph feat_dim must be positive")
        self.metric = str(metric)
        if self.metric not in SERAPH_METRICS:
            raise ValueError(f"seraph metric must be one of {list(SERAPH_METRICS)}")
        self.kappa = _validate_positive("kappa", kappa)
        self.kappa_mode = str(kappa_mode)
        if self.kappa_mode not in KAPPA_MODES:
            raise ValueError(f"seraph kappa_mode must be one of {list(KAPPA_MODES)}")
        self.threshold_mode = str(threshold_mode)
        if self.threshold_mode not in THRESHOLD_MODES:
            raise ValueError(f"seraph threshold_mode must be one of {list(THRESHOLD_MODES)}")
        # One fit yields both, so taking only half of it would mean pairing a
        # fitted eta with an unrelated kappa.
        if (self.threshold_mode == JOINT_CALIBRATION_MODE) != (
            self.kappa_mode == JOINT_CALIBRATION_MODE
        ):
            raise ValueError(
                f"seraph {JOINT_CALIBRATION_MODE!r} fits kappa and eta jointly, so "
                "threshold_mode and kappa_mode must both select it or neither "
                f"(got threshold_mode={self.threshold_mode!r}, "
                f"kappa_mode={self.kappa_mode!r})"
            )
        self.calibrates_on_labeled_pairs = self.threshold_mode == JOINT_CALIBRATION_MODE
        self.prior_correction = bool(prior_correction)
        # Eq 4 requires eta > 0 as the threshold separating S from D under d_A^2.
        self.initial_threshold = _validate_positive("threshold", threshold)
        self.threshold_momentum = _validate_probability(
            "threshold_momentum",
            threshold_momentum,
        )
        self.positive_pair_prior = _validate_probability(
            "positive_pair_prior",
            positive_pair_prior,
        )
        self.prior_mode = str(prior_mode)
        if self.prior_mode not in PRIOR_MODES:
            raise ValueError(f"seraph prior_mode must be one of {list(PRIOR_MODES)}")
        # FreeMatch's lambda weights the *previous* value, unlike
        # threshold_momentum right above it, which weights the new one.
        self.freematch_momentum = _validate_probability(
            "freematch_momentum",
            freematch_momentum,
        )
        self.positive_pair_prior_min = (
            self.positive_pair_prior / DEFAULT_PRIOR_RANGE
            if positive_pair_prior_min is None
            else _validate_probability("positive_pair_prior_min", positive_pair_prior_min)
        )
        self.positive_pair_prior_max = (
            min(0.5, self.positive_pair_prior * DEFAULT_PRIOR_RANGE)
            if positive_pair_prior_max is None
            else _validate_probability("positive_pair_prior_max", positive_pair_prior_max)
        )
        if self.positive_pair_prior_min >= self.positive_pair_prior_max:
            raise ValueError(
                "seraph positive_pair_prior_min must be below positive_pair_prior_max "
                f"({self.positive_pair_prior_min:.6g} vs {self.positive_pair_prior_max:.6g})"
            )
        self.track_fairness = bool(track_fairness)
        # The EMAs are shared: the fairness term reads the same running marginal
        # and histogram the self-adaptive threshold is built from.
        self.tracks_freematch = self.prior_mode == "freematch" or self.track_fairness

        self.perplexity = _validate_positive("perplexity", perplexity)
        if self.perplexity <= 1.0:
            # ln(perplexity) is the target entropy, and entropy is non-negative.
            raise ValueError("seraph perplexity must be greater than one")
        self.neighbors = (
            DEFAULT_NEIGHBORS.get(self.threshold_mode)
            if neighbors is None
            else _validate_positive_integer("neighbors", neighbors)
        )
        if self.threshold_mode in PER_ANCHOR_THRESHOLD_MODES:
            self.symmetrization = PER_ANCHOR_SYMMETRIZATIONS[self.threshold_mode]
        else:
            self.symmetrization = None
        self.memory_bank_size = (
            None
            if memory_bank_size is None
            else _validate_positive_integer("memory_bank_size", memory_bank_size)
        )

        self.metric_dim = None
        if self.metric == "lowrank":
            self.metric_dim = self.feat_dim if metric_dim is None else int(metric_dim)
            if not (0 < self.metric_dim <= self.feat_dim):
                raise ValueError(
                    "seraph metric_dim must be positive and at most feat_dim "
                    f"({self.metric_dim} vs {self.feat_dim})"
                )
            projection = torch.empty(self.metric_dim, self.feat_dim)
            # Orthonormal rows scaled by sqrt(feat_dim / metric_dim) keep
            # E[d_A^2] = ||z_i - z_j||^2 at initialization; a square L is then a
            # rotation, which leaves distances untouched.
            nn.init.orthogonal_(projection)
            projection *= math.sqrt(self.feat_dim / self.metric_dim)
            self.projection = nn.Parameter(projection)
        elif self.metric == "diag":
            # softplus(raw) = 1, so A starts at the identity.
            self.diagonal_logits = nn.Parameter(
                torch.full((self.feat_dim,), math.log(math.e - 1.0))
            )
        elif self.metric == "scale":
            self.log_scale = nn.Parameter(torch.zeros(()))
        elif metric_dim is not None:
            raise ValueError(
                f"seraph metric_dim only applies to metric='lowrank', not {self.metric!r}"
            )

        # The quantile threshold is a running estimate so the decision boundary
        # does not jitter with every batch, and so a checkpoint keeps a usable
        # eta. It is seeded by the first batch instead of interpolating away from
        # the configured value.
        # ``threshold`` is the fixed-mode eta; each pair block additionally keeps
        # its own running quantile estimate.
        self.register_buffer("threshold", torch.tensor(float(self.initial_threshold)))
        for block in PAIR_BLOCKS:
            self.register_buffer(
                f"threshold_{block}", torch.tensor(float(self.initial_threshold))
            )
            self.register_buffer(
                f"threshold_{block}_initialized", torch.zeros((), dtype=torch.bool)
            )

        # Both groups below are registered only when their feature is configured,
        # so a default run's state dict stays exactly what it was before they
        # existed and old checkpoints keep loading.
        if self.memory_bank_size is not None:
            self.register_buffer(
                "memory_bank", torch.zeros(self.memory_bank_size, self.feat_dim)
            )
            self.register_buffer("memory_bank_pointer", torch.zeros((), dtype=torch.long))
            self.register_buffer("memory_bank_count", torch.zeros((), dtype=torch.long))

        if self.calibrates_on_labeled_pairs:
            # Seeded so that before the first fit the posterior is exactly what
            # threshold_mode="fixed" with kappa_mode="fixed" would have given;
            # the first fit then replaces both outright, as the running quantile
            # does with its own first batch.
            self.register_buffer("calibration_slope", torch.tensor(-self.kappa))
            self.register_buffer(
                "calibration_intercept",
                torch.tensor(self.kappa * self.initial_threshold),
            )
            self.register_buffer(
                "calibration_initialized", torch.zeros((), dtype=torch.bool)
            )
            self.register_buffer(
                "calibration_rejections", torch.zeros((), dtype=torch.long)
            )

        if self.tracks_freematch:
            for block in PAIR_BLOCKS:
                # FreeMatch seeds tau_0 and p~_0 at 1 / C (Eqs 9 and 10); the pair
                # label has C = 2 outcomes, so both start at one half. Index 0 is
                # y = +1 (similar) and index 1 is y = -1 throughout.
                self.register_buffer(f"freematch_tau_{block}", torch.tensor(0.5))
                self.register_buffer(f"freematch_marginal_{block}", torch.full((2,), 0.5))
                self.register_buffer(f"freematch_histogram_{block}", torch.full((2,), 0.5))
                # The adaptive rate starts where the fixed one would have sat, so
                # switching prior_mode on does not move eta on the first step.
                self.register_buffer(
                    f"freematch_prior_{block}",
                    torch.tensor(float(self.positive_pair_prior)),
                )

    def project(self, embeddings):
        """Apply ``B`` to the retrieval embedding, in float32.

        Training runs under bfloat16 autocast, which is too coarse for the
        pairwise distances the posterior is built on.
        """

        embeddings = embeddings.float()
        if self.metric == "identity":
            return embeddings
        if self.metric == "lowrank":
            return embeddings @ self.projection.t().float()
        if self.metric == "diag":
            return embeddings * F.softplus(self.diagonal_logits.float()).sqrt()
        return embeddings * torch.exp(0.5 * self.log_scale.float())

    def metric_trace_terms(self):
        """Return ``(tr(A), ||A||_F^2)`` for this parameterization's own ``A``.

        Eq 7's hyper-sparsity term reads the first; ``soft_rank`` divides the
        scale out with the second. ``identity`` is rejected rather than answered
        with ``tr(I) = D``: a constant is not a regularizer, and a caller asking
        for one has configured something that cannot do what it was asked for.
        """

        if self.metric == "identity":
            raise RuntimeError(
                "seraph metric='identity' fixes A = I, whose trace is feat_dim "
                "and whose gradient is zero; trace_target='projection' penalizes "
                "the projection head that is learnable in that setting"
            )
        if self.metric == "lowrank":
            return factored_trace_terms(self.projection)
        if self.metric == "diag":
            diagonal = F.softplus(self.diagonal_logits.float())
            return diagonal.sum(), diagonal.pow(2).sum()
        # A = s I, so both terms are the scalar's, D times over.
        scale = torch.exp(self.log_scale.float())
        return self.feat_dim * scale, self.feat_dim * scale.pow(2)

    @staticmethod
    def _calibration_sample(squared_distances, reference_squared_distances):
        """Return the detached distances an ``eta`` or ``kappa`` estimate reads.

        The scored pairs always count; the batch-to-bank distances of deviation 9
        are appended when a memory bank is configured, which is the whole point of
        keeping one - the same distribution, sampled far more densely than a
        batch can sample it.
        """

        sample = squared_distances.detach().flatten().float()
        if reference_squared_distances is None:
            return sample
        return torch.cat(
            [sample, reference_squared_distances.detach().flatten().float()]
        )

    def resolve_threshold(self, squared_distances, block="uu", reference_squared_distances=None):
        """Return the ``eta`` used for this batch's posterior.

        ``quantile`` mode places ``eta`` at the ``positive_pair_prior`` quantile of
        the batch's own distances, so that fraction of pairs falls on the similar
        side and the all-dissimilar solution stops being reachable by moving
        ``eta``. The estimate is detached: it calibrates the posterior, it is not
        something the entropy term may optimize.

        Under ``prior_mode="freematch"`` the quantile level is not the configured
        prior but the self-adaptive rate of deviation 10, read from the previous
        step's statistics - this step's posterior does not exist yet.

        Each pair block keeps its own running estimate. Labeled embeddings are
        already organized by the supervised loss while unlabeled ones are not, so
        the two blocks' distance distributions differ systematically and a shared
        quantile would sit in the wrong place for both.

        The per-anchor modes do not come through here at all: they have no single
        ``eta`` to smooth. See ``anchor_thresholds``.
        """

        if block not in PAIR_BLOCKS:
            raise ValueError(f"seraph pair block must be one of {list(PAIR_BLOCKS)}")
        if self.threshold_mode == "fixed":
            return self.threshold
        if self.calibrates_on_labeled_pairs:
            return self.calibrated_threshold()
        if self.threshold_mode in PER_ANCHOR_THRESHOLD_MODES:
            raise RuntimeError(
                f"seraph threshold_mode={self.threshold_mode!r} calibrates eta per "
                "anchor; call anchor_thresholds instead of resolve_threshold"
            )
        threshold = getattr(self, f"threshold_{block}")
        initialized = getattr(self, f"threshold_{block}_initialized")
        batch_threshold = torch.quantile(
            self._calibration_sample(squared_distances, reference_squared_distances),
            self.quantile_level(block),
        )
        with torch.no_grad():
            if bool(initialized.item()):
                threshold.mul_(1.0 - self.threshold_momentum).add_(
                    self.threshold_momentum * batch_threshold
                )
            else:
                threshold.copy_(batch_threshold)
                initialized.fill_(True)
        return threshold

    def quantile_level(self, block="uu"):
        """Return the similar-pair fraction ``eta`` is currently placed at.

        ``prior_mode="fixed"`` returns the configured prior, which for the default
        ``1 / C`` is the balanced-class, stationary target of deviation 10.
        ``"freematch"`` returns the running self-adaptive rate instead, clamped to
        ``[positive_pair_prior_min, positive_pair_prior_max]``: a rate of zero
        would put ``eta`` at the smallest distance in the sample and hand the
        entropy term the all-dissimilar collapse.
        """

        if self.prior_mode == "fixed":
            return self.positive_pair_prior
        prior = getattr(self, f"freematch_prior_{block}")
        return float(
            prior.clamp(self.positive_pair_prior_min, self.positive_pair_prior_max)
        )

    def anchor_thresholds(self, neighbor_squared_distances):
        """Return the per-anchor ``eta_i`` of the configured per-anchor mode.

        ``neighbor_squared_distances`` is ``[anchors, pool]`` against the neighbor
        pool of deviation 8, with an anchor's own entry already masked out. Every
        mode returns ``eta_i`` in the units of ``d_A^2``, so the three differ only
        in how they decide what counts as "near" for anchor ``i``:
        ``self_tuning`` reads the ``K``-th order statistic, ``umap`` and ``tsne``
        each solve for the bandwidth that fixes an effective neighbor count.

        Detached for the same reason ``resolve_threshold`` detaches: this is the
        posterior's calibration, not a term the entropy objective may optimize by
        moving the neighbors.
        """

        if self.threshold_mode not in PER_ANCHOR_THRESHOLD_MODES:
            raise RuntimeError(
                f"seraph threshold_mode={self.threshold_mode!r} has a single eta; "
                "call resolve_threshold instead of anchor_thresholds"
            )
        distances = neighbor_squared_distances.detach().float()
        # An anchor's own entry arrives masked to +inf, and a row that is short of
        # neighbors would otherwise pull that mask into its order statistic and
        # return an infinite eta. The scarcest row sets the budget for all of them
        # so every anchor's eta keeps meaning the same thing.
        available = int(torch.isfinite(distances).sum(dim=1).min().item())
        if available < 2:
            # One neighbor gives every bandwidth solve a degenerate bracket: the
            # UMAP sum and the t-SNE entropy both hit their target at the boundary
            # and the bisection runs off to an infinite eta.
            raise ValueError(
                "seraph per-anchor thresholds need at least two neighbors per "
                f"anchor, found {available}; enlarge the unlabeled batch or set "
                "regularizer_params.memory_bank_size"
            )
        if self.threshold_mode == "tsne":
            return tsne_scales(distances, self.perplexity, max_neighbors=available)
        neighbors = max(2, min(self.neighbors, available))
        if self.threshold_mode == "self_tuning":
            return self_tuning_scales(distances, neighbors)
        return umap_scales(distances, neighbors)

    def per_anchor_logits(self, squared_distances, left_thresholds, right_thresholds, kappa):
        """Return the pair logit of a per-anchor ``eta``, already symmetrized.

        ``self_tuning`` combines in ``eta`` space - the paper's ``sigma_i sigma_j``
        is the geometric mean of the two ``eta_i = sigma_i^2`` - so it yields one
        logit directly. The other two score both directions and merge the
        posteriors through their own paper's rule; see ``combine_directed_logits``.
        """

        if self.symmetrization == "scale_product":
            pair_thresholds = (left_thresholds * right_thresholds).clamp_min(0.0).sqrt()
            return self.posterior_logits(squared_distances, pair_thresholds, kappa)
        return combine_directed_logits(
            self.posterior_logits(squared_distances, left_thresholds, kappa),
            self.posterior_logits(squared_distances, right_thresholds, kappa),
            self.symmetrization,
        )

    def inverse_temperature(self, squared_distances, reference_squared_distances=None):
        """Return the ``kappa`` of Eq 5 used for this batch.

        ``batch_std`` turns the logit into a standardized score by dividing by the
        spread of the batch's distances, so the posterior does not depend on how
        concentrated the frozen embedding's distances are. It is detached for the
        same reason ``eta`` is, but recomputed per batch rather than smoothed,
        because it only scales the gradient instead of deciding its sign. A
        configured memory bank widens the sample the spread is read from, exactly
        as it does for the quantile.
        """

        if self.kappa_mode == "fixed":
            return self.kappa
        if self.calibrates_on_labeled_pairs:
            return self.calibrated_kappa()
        spread = self._calibration_sample(
            squared_distances, reference_squared_distances
        ).std()
        # A batch of equidistant pairs carries no ordering to sharpen; falling
        # back to the raw kappa keeps the term finite and near-inert.
        if not torch.isfinite(spread) or float(spread) <= 0.0:
            return self.kappa
        return self.kappa / spread

    def posterior_logits(self, squared_distances, threshold, kappa):
        """Return the logit of ``p(y = +1)``: ``kappa * (eta - d_A^2)`` (Eq 5)."""

        return kappa * (threshold - squared_distances)

    def calibrated_kappa(self):
        """Return ``kappa = -a`` from the labeled-pair fit."""

        return -self.calibration_slope

    def calibrated_threshold(self):
        """Return ``eta = b / kappa`` from the labeled-pair fit.

        Eq 4 needs ``eta > 0``. A fitted intercept at or below zero says the
        calibration puts *no* distance on the similar side, which is the
        all-dissimilar solution rather than a threshold; it is clamped to keep the
        posterior finite and surfaced as ``calibration/threshold_clamped`` so it
        is visible instead of merely survivable. Under ``prior_correction`` it is
        also the expected symptom of a target rate the linear-in-``d^2`` fit has
        to extrapolate to reach.
        """

        return (self.calibration_intercept / self.calibrated_kappa()).clamp_min(EPSILON)

    @torch.no_grad()
    def calibrate_from_labeled_pairs(self, squared_distances, same_class, target_rate=None):
        """Fit ``(kappa, eta)`` on the labeled-labeled block and blend it in.

        Deviation 11. The pair label is *known* here, which is what makes this
        block a calibration set rather than an objective term: the fit reads it,
        the loss never does. ``prior_correction`` then moves the intercept from
        the sampler's positive-pair rate to the one an unlabeled pair actually
        has; see ``prior_corrected_intercept``.

        The EMA runs on the fit's own ``(a, b)`` rather than on ``(kappa, eta)``,
        because ``eta = -b / a`` is a ratio and smoothing a ratio is not the same
        as smoothing what it is made of. Like the running quantile, the first
        usable fit replaces the seed outright instead of interpolating away from
        it.

        Returns the fit's diagnostics, or ``None`` when the step had nothing to
        fit on: a batch whose labeled pairs are all similar or all dissimilar
        carries no slope, and a slope that does not come out negative is reported
        rather than applied.
        """

        if not self.calibrates_on_labeled_pairs:
            raise RuntimeError(
                f"seraph calibrate_from_labeled_pairs needs threshold_mode="
                f"{JOINT_CALIBRATION_MODE!r}"
            )
        targets = same_class.detach().float().flatten()
        positives = float(targets.sum())
        if positives < 1.0 or positives > len(targets) - 1.0:
            self.calibration_rejections += 1
            return None

        slope, intercept = fit_pair_logistic(
            squared_distances,
            targets,
            slope=float(self.calibration_slope),
            intercept=float(self.calibration_intercept),
        )
        if not (math.isfinite(slope) and math.isfinite(intercept)):
            self.calibration_rejections += 1
            return None
        if slope > -MIN_CALIBRATION_SLOPE:
            self.calibration_rejections += 1
            return None

        sample_rate = positives / len(targets)
        raw_intercept = intercept
        if self.prior_correction and target_rate is not None:
            intercept = prior_corrected_intercept(intercept, sample_rate, target_rate)

        if bool(self.calibration_initialized.item()):
            momentum = self.threshold_momentum
            self.calibration_slope.mul_(1.0 - momentum).add_(momentum * slope)
            self.calibration_intercept.mul_(1.0 - momentum).add_(momentum * intercept)
        else:
            self.calibration_slope.fill_(slope)
            self.calibration_intercept.fill_(intercept)
            self.calibration_initialized.fill_(True)
        return {
            "slope": slope,
            "intercept": intercept,
            "raw_intercept": raw_intercept,
            "sample_positive_rate": sample_rate,
            "pairs": float(len(targets)),
        }

    @torch.no_grad()
    def update_memory_bank(self, embeddings):
        """Push this step's unlabeled retrieval embeddings into the FIFO bank.

        Retrieval embeddings, not projected ones: a trainable ``B`` moves every
        step, and re-projecting the bank at read time keeps the stored rows
        consistent with the metric the posterior is currently using. Only the
        embeddings themselves go stale, which is the one approximation a
        MoCo-style bank cannot avoid.
        """

        if self.memory_bank_size is None:
            return
        embeddings = embeddings.detach().to(
            device=self.memory_bank.device, dtype=self.memory_bank.dtype
        )
        if len(embeddings) == 0:
            return
        if len(embeddings) > self.memory_bank_size:
            embeddings = embeddings[-self.memory_bank_size :]
        pointer = int(self.memory_bank_pointer.item())
        first = min(len(embeddings), self.memory_bank_size - pointer)
        self.memory_bank[pointer : pointer + first] = embeddings[:first]
        remaining = len(embeddings) - first
        if remaining:
            self.memory_bank[:remaining] = embeddings[first:]
        self.memory_bank_pointer.fill_((pointer + len(embeddings)) % self.memory_bank_size)
        self.memory_bank_count.fill_(
            min(self.memory_bank_size, int(self.memory_bank_count.item()) + len(embeddings))
        )

    def memory_bank_embeddings(self):
        """Return the filled prefix of the bank, or ``None`` when it is empty."""

        if self.memory_bank_size is None:
            return None
        filled = int(self.memory_bank_count.item())
        if filled == 0:
            return None
        return self.memory_bank[:filled]

    @torch.no_grad()
    def update_freematch_statistics(self, logits, block="uu"):
        """Advance FreeMatch's EMAs and return this step's confidence mask.

        arXiv:2205.07246 for the binary pair label, whose two outcomes are the
        ``C`` classes its equations range over:

        * Eq 9, the global threshold: ``tau_t`` is an EMA of the mean confidence
          ``max(p, 1 - p)``, so it rises as the model's confidence does.
        * Eq 10, the local list: ``p~_t`` is an EMA of the mean posterior over the
          two labels.
        * Eq 11, MaxNorm: ``tau_t(y) = p~_t(y) / max_y p~_t(y) * tau_t``. This is
          what makes the transplant safe. Similar pairs are the rare label by a
          factor of ``C``, so ``p~_t(+1)`` stays far below ``p~_t(-1)`` and the
          similar side's bar drops with it; a rising global threshold therefore
          cannot ratchet the similar-pair rate to zero on its own.

        The rate itself is the fraction of pairs that both pass the mask and are
        predicted similar, EMA'd like everything else, and it is what
        ``quantile_level`` places ``eta`` at on the next step. Updating before
        masking follows the reference implementation, whose thresholding hook
        updates and then masks inside one call.
        """

        if not self.tracks_freematch:
            raise RuntimeError(
                "seraph freematch statistics need prior_mode='freematch' or a "
                "non-zero fairness_weight"
            )
        logits = logits.detach().float()
        positive = torch.sigmoid(logits)
        probabilities = torch.stack([positive, 1.0 - positive], dim=1)
        confidence, predicted = probabilities.max(dim=1)

        momentum = self.freematch_momentum
        tau = getattr(self, f"freematch_tau_{block}")
        marginal = getattr(self, f"freematch_marginal_{block}")
        histogram = getattr(self, f"freematch_histogram_{block}")
        prior = getattr(self, f"freematch_prior_{block}")

        tau.mul_(momentum).add_((1.0 - momentum) * confidence.mean())
        marginal.mul_(momentum).add_((1.0 - momentum) * probabilities.mean(dim=0))
        batch_histogram = torch.zeros_like(histogram)
        batch_histogram.scatter_add_(0, predicted, torch.ones_like(confidence))
        batch_histogram /= batch_histogram.sum().clamp_min(1.0)
        histogram.mul_(momentum).add_((1.0 - momentum) * batch_histogram)

        mask = confidence >= self.freematch_thresholds(block)[predicted]
        confident_positive_rate = (mask & (predicted == 0)).float().mean()
        prior.mul_(momentum).add_((1.0 - momentum) * confident_positive_rate)
        return mask

    def freematch_thresholds(self, block="uu"):
        """Return Eq 11's ``[tau_t(+1), tau_t(-1)]`` for the current statistics."""

        marginal = getattr(self, f"freematch_marginal_{block}")
        tau = getattr(self, f"freematch_tau_{block}")
        return tau * marginal / marginal.max().clamp_min(EPSILON)

    def fairness_loss(self, logits, mask, block="uu"):
        """Return FreeMatch's self-adaptive fairness penalty for one pair block.

        arXiv:2205.07246 Sec 3.3 pairs its self-adaptive threshold with a term
        over the *marginal* of the confident predictions, divided by the histogram
        of the hard predictions so the reference is not a demand for a uniform
        label marginal. That normalization is what makes the term usable here at
        all: for a pair label, uniform would mean half of all pairs are similar,
        when the truth is nearer ``1 / C``. A calibrated model has mean posterior
        and prediction frequency in step on both labels, which puts both the
        reference and the batch at ``[0.5, 0.5]`` and the term at its floor no
        matter how rare similar pairs really are.

        This is the counterweight the self-adaptive rate needs. If the similar
        label dries up, the batch's normalized marginal collapses onto the
        dissimilar entry while the EMA reference still has mass on both, and the
        cross entropy between them grows.

        Two deliberate departures:

        * The paper writes ``L_f = -H(SumNorm(p~/h~), SumNorm(p~'/h~'))`` and its
          reference implementation minimizes ``sum p log q``, which is maximized
          rather than minimized at ``q = p``. For a two-outcome variable that sign
          drives the marginal to a corner - it would *reward* exactly the collapse
          the term is described as preventing. This minimizes the cross entropy
          ``-sum p log q`` instead, which is what the paper's stated aim of
          encouraging diverse predictions asks for.
        * A label the batch never hard-predicts gets its histogram entry floored
          at one sample's worth rather than the reference's ``1/h -> 0``, which
          would leave the collapse a flat region with no gradient out of it. With
          only two labels an empty one is common enough to matter.

        Returns ``None`` when the mask is empty, as the reference implementation
        skips the term when no sample passes.
        """

        if not self.tracks_freematch:
            raise RuntimeError(
                "seraph fairness_loss needs the freematch statistics to be tracked"
            )
        selected = int(mask.sum().item())
        if selected == 0:
            return None
        positive = torch.sigmoid(logits[mask].float())
        probabilities = torch.stack([positive, 1.0 - positive], dim=1)
        with torch.no_grad():
            batch_histogram = torch.zeros(2, device=logits.device, dtype=torch.float32)
            batch_histogram.scatter_add_(
                0,
                probabilities.argmax(dim=1),
                torch.ones(selected, device=logits.device, dtype=torch.float32),
            )
            batch_histogram = (batch_histogram / selected).clamp_min(1.0 / selected)
            reference = histogram_normalized(
                getattr(self, f"freematch_marginal_{block}"),
                getattr(self, f"freematch_histogram_{block}"),
            )
        batch_marginal = histogram_normalized(
            probabilities.mean(dim=0),
            batch_histogram,
        )
        return -(reference * batch_marginal.clamp_min(EPSILON).log()).sum()


class SeraphRegularizer(BaseTrainingRegularizer):
    """SERAPH's unlabeled entropy term as a regularizer on unlabeled pairs."""

    name = "seraph"
    # One deterministic view per unlabeled image is all the posterior needs, so
    # the frozen-backbone feature cache stays usable.
    supports_frozen_feature_precompute = True
    # The unlabeled batch rides along in the labeled stream's forward pass.
    uses_joint_forward = True
    # Only the trace is ever handed to GradNorm; the fairness term is named here
    # so an exclusion list can still refer to it by the label it is reported under.
    extra_component_names = (FAIRNESS_COMPONENT, TRACE_COMPONENT)

    def __init__(
        self,
        regularizer_weight=1.0,
        supervised_weight=1.0,
        objective="em",
        em_sharpening=DEFAULT_EM_SHARPENING,
        include_labeled_unlabeled=False,
        metric="identity",
        metric_dim=None,
        kappa=1.0,
        kappa_mode="batch_std",
        threshold=1.0,
        threshold_mode="quantile",
        threshold_momentum=0.1,
        positive_pair_prior=None,
        neighbors=None,
        perplexity=DEFAULT_PERPLEXITY,
        memory_bank_size=None,
        prior_mode="fixed",
        freematch_momentum=DEFAULT_FREEMATCH_MOMENTUM,
        positive_pair_prior_min=None,
        positive_pair_prior_max=None,
        fairness_weight=0.0,
        trace_weight=0.0,
        trace_target="projection",
        trace_mode="soft_rank",
        trace_target_ratio=None,
        prior_correction=False,
        unlabeled_ratio=1.0,
        unlabeled_batch_size=None,
    ):
        super().__init__(
            regularizer_weight=regularizer_weight,
            supervised_weight=supervised_weight,
        )
        self.objective = str(objective)
        if self.objective not in SERAPH_OBJECTIVES:
            raise ValueError(f"seraph objective must be one of {list(SERAPH_OBJECTIVES)}")
        # The paper's mu did two unrelated jobs: it weighted the whole unlabeled
        # term and it set the E-step's sharpening exponent 1 + mu. The weighting
        # is what regularizer_weight already does, so only the exponent survives
        # here, named for the one thing it now controls.
        self.em_sharpening = _validate_positive("em_sharpening", em_sharpening)
        if self.em_sharpening < 1.0:
            # exponent = 1 + mu with mu >= 0; below 1 the E-step would flatten the
            # posterior instead of sharpening it, inverting Eq 9.
            raise ValueError("seraph em_sharpening must be at least 1")
        self.include_labeled_unlabeled = bool(include_labeled_unlabeled)
        self.metric = str(metric)
        if self.metric not in SERAPH_METRICS:
            raise ValueError(f"seraph metric must be one of {list(SERAPH_METRICS)}")
        self.metric_dim = None if metric_dim is None else int(metric_dim)
        if self.metric_dim is not None and self.metric != "lowrank":
            raise ValueError("seraph metric_dim only applies to metric='lowrank'")
        if self.metric_dim is not None and self.metric_dim <= 0:
            raise ValueError("seraph metric_dim must be positive when set")
        self.kappa = _validate_positive("kappa", kappa)
        self.kappa_mode = str(kappa_mode)
        if self.kappa_mode not in KAPPA_MODES:
            raise ValueError(f"seraph kappa_mode must be one of {list(KAPPA_MODES)}")
        self.threshold = _validate_positive("threshold", threshold)
        self.threshold_mode = str(threshold_mode)
        if self.threshold_mode not in THRESHOLD_MODES:
            raise ValueError(f"seraph threshold_mode must be one of {list(THRESHOLD_MODES)}")
        self.threshold_momentum = _validate_probability(
            "threshold_momentum",
            threshold_momentum,
        )
        self.positive_pair_prior = (
            None
            if positive_pair_prior is None
            else _validate_probability("positive_pair_prior", positive_pair_prior)
        )
        self.prior_mode = str(prior_mode)
        if self.prior_mode not in PRIOR_MODES:
            raise ValueError(f"seraph prior_mode must be one of {list(PRIOR_MODES)}")
        self.freematch_momentum = _validate_probability(
            "freematch_momentum",
            freematch_momentum,
        )
        self.positive_pair_prior_min = (
            None
            if positive_pair_prior_min is None
            else _validate_probability("positive_pair_prior_min", positive_pair_prior_min)
        )
        self.positive_pair_prior_max = (
            None
            if positive_pair_prior_max is None
            else _validate_probability("positive_pair_prior_max", positive_pair_prior_max)
        )
        self.fairness_weight = _validate_non_negative("fairness_weight", fairness_weight)
        self.trace_weight = _validate_non_negative("trace_weight", trace_weight)
        self.trace_target = str(trace_target)
        if self.trace_target not in TRACE_TARGETS:
            raise ValueError(f"seraph trace_target must be one of {list(TRACE_TARGETS)}")
        self.trace_mode = str(trace_mode)
        if self.trace_mode not in TRACE_MODES:
            raise ValueError(f"seraph trace_mode must be one of {list(TRACE_MODES)}")
        if trace_target_ratio is not None and self.trace_weight == 0:
            raise ValueError(
                "seraph trace_target_ratio has no effect without trace_weight: the "
                "ratio calibrates that weight, and a zero weight leaves the trace "
                "term out of the objective entirely"
            )
        # ``0`` is the opt-out, so that "projection defaults to calibrated" stays
        # a default rather than a rule with no exception.
        self.trace_target_ratio = (
            0.0
            if trace_target_ratio is None
            else _validate_non_negative("trace_target_ratio", trace_target_ratio)
        )
        if self.trace_weight > 0:
            if trace_target_ratio is None and self.trace_target == "projection":
                self.trace_target_ratio = DEFAULT_TRACE_TARGET_RATIO
            self._validate_trace_term()
        self.prior_correction = bool(prior_correction)
        self.calibrates_on_labeled_pairs = self.threshold_mode == JOINT_CALIBRATION_MODE
        if self.calibrates_on_labeled_pairs != (self.kappa_mode == JOINT_CALIBRATION_MODE):
            raise ValueError(
                f"seraph {JOINT_CALIBRATION_MODE!r} fits kappa and eta jointly, so "
                "threshold_mode and kappa_mode must both select it or neither "
                f"(got threshold_mode={self.threshold_mode!r}, "
                f"kappa_mode={self.kappa_mode!r})"
            )
        self.neighbors = (
            None if neighbors is None else _validate_positive_integer("neighbors", neighbors)
        )
        self.perplexity = _validate_positive("perplexity", perplexity)
        self.memory_bank_size = (
            None
            if memory_bank_size is None
            else _validate_positive_integer("memory_bank_size", memory_bank_size)
        )
        # The adaptive rate enters the objective in exactly two places: as the
        # level the quantile is taken at, and as the base rate the labeled fit's
        # intercept is corrected to. Anywhere else it is tracked and logged but
        # inert, which is worth saying out loud rather than leaving in a
        # diagnostic nobody reads.
        adaptive_rate_used = self.threshold_mode == "quantile" or (
            self.calibrates_on_labeled_pairs and self.prior_correction
        )
        if self.prior_mode == "freematch" and not adaptive_rate_used:
            logger.warning(
                f"seraph prior_mode='freematch' does not reach the objective with "
                f"threshold_mode={self.threshold_mode!r} and "
                f"prior_correction={self.prior_correction}; the adaptive rate is "
                "tracked and logged but nothing reads it"
            )
        if self.neighbors is not None and self.threshold_mode not in ("self_tuning", "umap"):
            raise ValueError(
                "seraph neighbors only applies to threshold_mode='self_tuning' or "
                f"'umap', not {self.threshold_mode!r}"
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
            raise ValueError("seraph unlabeled_batch_size must be at least two")
        self.unlabeled_ratio = (
            None
            if unlabeled_ratio is None
            else _validate_positive("unlabeled_ratio", unlabeled_ratio)
        )
        if self.unlabeled_ratio is None and self.unlabeled_batch_size is None:
            self.unlabeled_ratio = DEFAULT_UNLABELED_RATIO
        elif self.unlabeled_ratio is not None and self.unlabeled_batch_size is not None:
            logger.warning(
                f"seraph received both unlabeled_batch_size={self.unlabeled_batch_size} "
                f"and unlabeled_ratio={self.unlabeled_ratio}; the absolute batch size "
                "wins and the ratio is ignored"
            )
        self.head = None
        self.dataset = None
        # Population sizes, not batch sizes: the two blocks are weighted by how
        # many pairs of each kind the whole pool holds, so a batch that happens to
        # be labeled-heavy does not reweight the objective.
        self.labeled_pool_size = None
        self.unlabeled_pool_size = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_diagnostics = {}
        # Set by compute_loss, read by combine_losses one call later. The
        # fairness term is weighted independently of regularizer_weight because
        # FreeMatch weights its own by a separate lambda.
        self._fairness_loss = None
        # Same contract as the fairness term above, and separate from it because
        # Eq 7 gives the trace its own lambda.
        self._trace_loss = None
        self._projection_head = None
        self._calibration = None
        # The exact per-pair tensors the step scored, kept only while batch
        # diagnostics are on. The debugger's pair trace reads these rather than
        # rebuilding logits from a scalar eta, which a per-anchor mode does not
        # have and which even the scalar modes could only approximate.
        self._last_pair_terms = {}

    def _trace_penalty(self, head):
        """Return Eq 7's hyper-sparsity term for this step, or ``None``.

        Batch-independent by construction -- it reads parameters, not pairs --
        but it is evaluated per step like every other term so it lands in the
        same graph and the same backward.

        ``nuclear`` is the paper's ``tr(A)``. Across a multi-layer projection
        head it becomes the sum of the layers' traces, which is the variational
        form of the nuclear norm (``||M||_* = min_{M = UV} (||U||_F^2 +
        ||V||_F^2) / 2``) and so is exact for a linear stack and an
        approximation once a ReLU sits between the factors.

        ``soft_rank`` is ``tr(A)^2 / ||A||_F^2``: the participation ratio of
        ``A``'s eigenvalues, ``1`` for a rank-one metric and ``feat_dim`` for the
        identity. It relaxes rank the same way the trace does, and unlike the
        trace it is homogeneous of degree 0 in ``A``, so it survives the two
        deviations that make this module's objective scale-free.
        """

        if self.trace_weight == 0:
            return None
        if self.trace_target == "metric":
            trace, frobenius_squared = head.metric_trace_terms()
        else:
            if self._projection_head is None:
                raise RuntimeError(
                    "seraph trace_target='projection' requires configure_model to "
                    "resolve the model's projection head"
                )
            maps = projection_affine_maps(self._projection_head)
            if self.trace_mode == "nuclear":
                return torch.stack([m.pow(2).sum() for m in maps]).sum()
            trace, frobenius_squared = factored_trace_terms(maps[0])
        if self.trace_mode == "nuclear":
            return trace
        return trace.pow(2) / frobenius_squared.clamp_min(EPSILON)

    def _trace_diagnostics(self, head):
        """Report the traced metric's scale and effective rank side by side.

        ``tr(A)`` says how large the metric is; ``tr(A)^2 / ||A||_F^2`` says how
        many directions it spreads over, running from ``1`` for a rank-one metric
        to ``feat_dim`` for the identity. A trace term that is working shows the
        second falling; a trace term that is only shrinking the parameter shows
        the first falling with the second flat.
        """

        with torch.no_grad():
            if self.trace_target == "metric":
                trace, frobenius_squared = head.metric_trace_terms()
            else:
                maps = projection_affine_maps(self._projection_head)
                if len(maps) > 1:
                    # No single A exists across a ReLU, so a rank for it would
                    # be invented rather than measured. The nuclear term is still
                    # a sum of well-defined layer traces.
                    return {
                        "train/seraph/trace/nuclear": torch.stack(
                            [m.pow(2).sum() for m in maps]
                        ).sum()
                    }
                trace, frobenius_squared = factored_trace_terms(maps[0])
        return {
            "train/seraph/trace/nuclear": trace,
            "train/seraph/trace/soft_rank": trace.pow(2)
            / frobenius_squared.clamp_min(EPSILON),
        }

    def _resolve_projection_head(self, student_model):
        """Return the model's trainable projection head, or explain its absence.

        ``trace_target="projection"`` penalizes the metric that head induces, so
        a model without one has nothing for the term to act on. Frozen-backbone
        runs always have one -- a null ``feat_dim`` leaves ``fc`` an
        ``nn.Identity`` and cannot train at all -- but the check is cheap and the
        failure is otherwise a silent zero.
        """

        projection_head = getattr(student_model, "fc", None)
        maps = [] if projection_head is None else projection_affine_maps(projection_head)
        if not maps:
            raise ValueError(
                "seraph trace_target='projection' needs a trainable projection head, "
                "but this model has none (feat_dim=None leaves fc as nn.Identity). "
                "Set feat_dim, or move the term to the posterior's own metric with "
                "trace_target='metric' and a metric other than 'identity'"
            )
        if len(maps) > 1 and self.trace_mode == "soft_rank":
            raise ValueError(
                f"seraph trace_mode='soft_rank' needs one linear projection, but the "
                f"head has {len(maps)} layers. The ReLU between them means the induced "
                "metric is not A = W^T W and its spectrum is not the layers' spectra, "
                "so the ratio would not measure the composite's rank. Use "
                "projection_layers=1, or trace_mode='nuclear', which is defined "
                "layerwise"
            )
        return projection_head

    def _validate_trace_term(self):
        """Reject the trace configurations that provably cannot regularize.

        Eq 7's ``-lambda tr(A)`` is a rank relaxation only where ``A``'s scale is
        identifiable. Two of this module's deviations remove exactly that, so the
        combinations below are a constant or an unopposed pull to zero rather
        than the paper's low-rank pressure, and each is caught here instead of
        being discovered as a run that trained to the same number.
        """

        if self.trace_target == "metric" and self.trace_target_ratio > 0:
            raise ValueError(
                "seraph trace_target_ratio cannot calibrate a trace on the posterior's "
                "own metric: that metric lives in seraph_posterior, which "
                "private_model_module_names() excludes from every gradient comparison, "
                "so the term's measured norm over shared parameters is exactly zero. "
                "The calibration would burn its dead-term budget and raise. Set "
                "trace_weight directly, or use trace_target='projection'"
            )

        if self.trace_target == "metric":
            if self.metric == "identity":
                raise ValueError(
                    "seraph trace_weight with trace_target='metric' needs a learnable "
                    "metric: metric='identity' fixes A = I, so tr(A) = feat_dim is a "
                    "constant with zero gradient. Use trace_target='projection' to "
                    "penalize the model's projection head, which is the projection "
                    "being learned when the posterior's own metric has no parameters"
                )
            if self.metric == "scale" and self.trace_mode == "soft_rank":
                raise ValueError(
                    "seraph trace_mode='soft_rank' with metric='scale' is a constant: "
                    "A = s I has tr(A)^2 / ||A||_F^2 = feat_dim for every s. A scalar "
                    "metric is isotropic, so it has no spectrum to concentrate; use "
                    "metric='diag' or 'lowrank', or trace_mode='nuclear'"
                )

        if self.trace_mode != "nuclear":
            return
        # Both remaining checks are the same statement: tr(A) is a pure scale
        # penalty, so it does nothing unless something else in the objective
        # holds the scale up.
        if self.trace_target == "projection":
            logger.warning(
                "seraph trace_mode='nuclear' on the projection head is a pure scale "
                "penalty: project_features L2-normalizes the head's output, so the "
                "objective is invariant to (W, b) -> (cW, cb) and the loss gradient "
                "is orthogonal to the head while tr(A)'s gradient is parallel to it. "
                "It rescales the head - an effective learning rate - and cannot move "
                "its spectrum. trace_mode='soft_rank' is the same relaxation with "
                "that scale divided out"
            )
        elif self.threshold_mode != "fixed" and self.kappa_mode != "fixed":
            logger.warning(
                f"seraph trace_mode='nuclear' with threshold_mode={self.threshold_mode!r} "
                f"and kappa_mode={self.kappa_mode!r} penalizes a scale the objective "
                "cannot see: both eta and kappa are estimated from the batch's own "
                "distances, so scaling A leaves every logit unchanged and tr(A) is "
                "unopposed - it drives A to zero. Fix one of the two calibrations, "
                "or use trace_mode='soft_rank'"
            )

    def validate_run_args(self, args):
        if int(args.batch_size) < 2:
            raise ValueError("seraph requires batch_size >= 2 so unlabeled pairs exist")

    def private_model_module_names(self):
        # The posterior head is the regularizer's alone; the supervised loss
        # never reaches it, so GradNorm must not weigh it as shared.
        return ("seraph_posterior",)

    def configure_model(self, student_model, train_dataset, split, train_labels_mapper, device):
        """Attach the posterior head before the optimizer collects parameters."""

        if self.regularizer_weight == 0:
            return None
        if hasattr(student_model, "seraph_posterior"):
            raise RuntimeError("seraph posterior head is already configured on this model")
        prior = self.positive_pair_prior
        if prior is None:
            # Two samples drawn from C balanced classes are similar with
            # probability 1 / C, which is the expected positive rate in U.
            num_classes = len(train_labels_mapper)
            if num_classes < 2:
                raise ValueError(
                    "seraph needs at least two training classes to estimate the "
                    "positive-pair prior; set regularizer_params.positive_pair_prior instead"
                )
            prior = 1.0 / float(num_classes)
        head = SeraphPosteriorHead(
            feat_dim=student_model.feat_dim,
            metric=self.metric,
            metric_dim=self.metric_dim,
            threshold=self.threshold,
            threshold_mode=self.threshold_mode,
            threshold_momentum=self.threshold_momentum,
            positive_pair_prior=prior,
            kappa=self.kappa,
            kappa_mode=self.kappa_mode,
            neighbors=self.neighbors,
            perplexity=self.perplexity,
            memory_bank_size=self.memory_bank_size,
            prior_mode=self.prior_mode,
            freematch_momentum=self.freematch_momentum,
            positive_pair_prior_min=self.positive_pair_prior_min,
            positive_pair_prior_max=self.positive_pair_prior_max,
            track_fairness=self.fairness_weight > 0,
            prior_correction=self.prior_correction,
        ).to(device)
        student_model.add_module("seraph_posterior", head)
        self.head = head
        if self.trace_weight > 0 and self.trace_target == "projection":
            self._projection_head = self._resolve_projection_head(student_model)
        if self.threshold_mode in PER_ANCHOR_THRESHOLD_MODES and self.memory_bank_size is None:
            # Deviation 8's stated failure mode, and the one worth a line in the
            # log rather than a line in a docstring nobody opened.
            logger.warning(
                f"seraph threshold_mode={self.threshold_mode!r} estimates a k-th "
                "nearest neighbor per anchor from the unlabeled batch alone; set "
                "regularizer_params.memory_bank_size to give that order statistic "
                "a pool worth searching"
            )
        logger.info(
            "Configured SERAPH posterior head: "
            f"metric={self.metric}, metric_dim={head.metric_dim}, feat_dim={head.feat_dim}, "
            f"objective={self.objective}, em_sharpening={self.em_sharpening}, "
            f"include_labeled_unlabeled={self.include_labeled_unlabeled}, "
            f"kappa={self.kappa} ({self.kappa_mode}), "
            f"threshold_mode={self.threshold_mode}, eta_init={self.threshold}, "
            f"positive_pair_prior={prior:.6g} ({self.prior_mode}), "
            f"neighbors={head.neighbors}, perplexity={head.perplexity}, "
            f"memory_bank_size={self.memory_bank_size}, "
            f"fairness_weight={self.fairness_weight}, "
            f"trace_weight={self.trace_weight}"
            + (
                ""
                if self.trace_weight == 0
                else (
                    f" ({self.trace_mode} on the {self.trace_target}"
                    + (
                        ", uncalibrated)"
                        if self.trace_target_ratio <= 0
                        else f", calibrated to ratio {self.trace_target_ratio:g})"
                    )
                )
            )
        )
        return None

    def build_dataset(self, train_dataset, split, use_cache=False):
        utils.shutdown_dataloaders(self._regularizer_loader)
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        if self.regularizer_weight == 0:
            self.dataset = None
            return None
        unlabeled_positions = np.asarray(split.unlabeled_positions, dtype=np.int64)
        if len(unlabeled_positions) < 2:
            raise ValueError("seraph requires at least two unlabeled samples to form a pair")
        self.unlabeled_pool_size = int(len(unlabeled_positions))
        self.labeled_pool_size = int(len(np.asarray(split.labeled_positions, dtype=np.int64)))
        if self.include_labeled_unlabeled and self.labeled_pool_size < 1:
            raise ValueError(
                "seraph include_labeled_unlabeled needs at least one labeled sample"
            )
        regularizer_dataset = self.make_regularizer_source_dataset(
            train_dataset,
            use_cache=use_cache,
        )
        self.dataset = UnlabeledSubset(regularizer_dataset, unlabeled_positions, num_views=1)
        return self.dataset

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
        """Pair the labeled loader with a shuffled unlabeled stream.

        SERAPH keeps no epoch-scoped artifact: every step forms its pairs inside
        the current batch, so the shared ``update_mode`` cadence has nothing to
        rebuild here.
        """

        if self.dataset is None:
            raise RuntimeError("seraph build_dataset must run before make_loader")
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
            raise ValueError("seraph needs at least two unlabeled samples per batch")
        if unlabeled_batch_size < requested_batch_size:
            # Silently shrinking would change the pair count, and with it eta and
            # kappa, without anything in the log saying so.
            logger.warning(
                f"seraph requested {requested_batch_size} unlabeled samples per step "
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
                # A ragged final batch would shift both the pair count and the
                # batch statistics eta and kappa are read from.
                drop_last=True,
                persistent_workers=True,
                pin_memory=True,
                desc="seraph unlabeled",
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                "SERAPH unlabeled loader: "
                f"pool={len(self.dataset)}, batch_size={unlabeled_batch_size} ({sizing}), "
                f"pairs_per_step={unlabeled_batch_size * (unlabeled_batch_size - 1) // 2}, "
                f"steps={len(supervised_loader)}"
            )
        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

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
    ):
        head = getattr(student_model, "seraph_posterior", None)
        if head is None:
            raise RuntimeError("seraph requires configure_model to attach the posterior head")
        if regularizer_embeddings is None:
            raise ValueError("seraph requires the joint labeled/unlabeled forward context")
        if len(regularizer_embeddings) < 2:
            raise ValueError("seraph needs at least two unlabeled embeddings per batch")

        if self.calibrates_on_labeled_pairs:
            if supervised_embeddings is None or len(supervised_embeddings) < 2:
                raise ValueError(
                    f"seraph threshold_mode={JOINT_CALIBRATION_MODE!r} needs at least "
                    "two labeled embeddings per step to form a calibration pair"
                )
            if supervised_labels is None:
                raise ValueError(
                    f"seraph threshold_mode={JOINT_CALIBRATION_MODE!r} needs the "
                    "labeled batch's labels; they are what make the labeled-labeled "
                    "block a calibration set"
                )
        if self.include_labeled_unlabeled:
            if supervised_embeddings is None or len(supervised_embeddings) == 0:
                raise ValueError(
                    "seraph include_labeled_unlabeled needs the labeled half of the "
                    "joint forward"
                )
            if self.unlabeled_pool_size is None:
                raise RuntimeError(
                    "seraph include_labeled_unlabeled must build its dataset before "
                    "training so the pair blocks can be weighted by population"
                )

        # ``project`` casts its input, but autocast would still run the projection
        # and Gram matmuls in bfloat16, and the Gram form of a squared distance
        # cancels two large terms against each other - the last place that can
        # afford three decimal digits. Everything after this is elementwise and
        # follows the float32 distances.
        per_anchor = self.threshold_mode in PER_ANCHOR_THRESHOLD_MODES
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            projected = head.project(regularizer_embeddings)

            distances = pairwise_squared_distances(projected)
            # Each unordered pair of the unlabeled batch is scored once.
            pairs = torch.triu_indices(len(projected), len(projected), offset=1, device=distances.device)
            block_distances = {"uu": distances[pairs[0], pairs[1]]}
            projected_labeled = None
            if self.include_labeled_unlabeled:
                # One endpoint is labeled, but the pair's own weak label stays
                # unknown, so these belong to the paper's U alongside the uu block.
                projected_labeled = head.project(supervised_embeddings)
                block_distances["lu"] = cross_squared_distances(
                    projected_labeled, projected
                ).flatten()

            with torch.no_grad():
                self._calibration = (
                    self._calibrate_on_labeled_pairs(
                        head,
                        # Reuse the cross block's projection when there is one;
                        # otherwise the labeled rows are needed for the fit alone
                        # and never have to carry a graph.
                        head.project(supervised_embeddings)
                        if projected_labeled is None
                        else projected_labeled,
                        supervised_labels,
                    )
                    if self.calibrates_on_labeled_pairs
                    else None
                )
                # Everything below calibrates the posterior rather than entering
                # it, and the bank rows are stale data the objective must not
                # push back on, so the whole block stays outside the graph.
                bank = head.memory_bank_embeddings()
                projected_bank = None if bank is None else head.project(bank)
                reference_distances = self._reference_distances(
                    projected, projected_labeled, projected_bank
                )
                anchor_thresholds = (
                    self._anchor_thresholds(head, projected, projected_labeled, projected_bank)
                    if per_anchor
                    else {}
                )

        block_weights = self._block_weights(len(block_distances) > 1)
        total = None
        fairness_total = None
        block_terms = {}
        for block, squared_distances in block_distances.items():
            kappa = head.inverse_temperature(
                squared_distances, reference_distances.get(block)
            )
            if per_anchor:
                left, right = self._anchor_threshold_pairs(
                    block, anchor_thresholds, pairs, len(projected)
                )
                logits = head.per_anchor_logits(squared_distances, left, right, kappa)
                # No single eta exists; the pair-level summary is what the
                # diagnostics can honestly report. See _diagnostics.
                threshold = (
                    (left * right).clamp_min(0.0).sqrt()
                    if head.symmetrization == "scale_product"
                    else 0.5 * (left + right)
                )
            else:
                threshold = head.resolve_threshold(
                    squared_distances,
                    block=block,
                    reference_squared_distances=reference_distances.get(block),
                )
                logits = head.posterior_logits(squared_distances, threshold, kappa)

            if self.objective == "em":
                # Eq 9's E-step, then its M-step: a cross entropy against the fixed
                # soft pair targets q.
                targets = torch.sigmoid(sharpen_logits(logits, self.em_sharpening))
                per_pair = F.binary_cross_entropy_with_logits(
                    logits, targets, reduction="none"
                )
            else:
                # Eq 7's unlabeled term descended directly.
                per_pair = binary_entropy_from_logits(logits)
            # Each block contributes its own mean, so the weights below decide the
            # tradeoff rather than whichever block happens to hold more pairs.
            weighted = block_weights[block] * per_pair.mean()
            total = weighted if total is None else total + weighted

            mask = None
            fairness = None
            if head.tracks_freematch:
                # Advances the EMAs this step's posterior feeds, which the next
                # step's eta is placed from. The mask comes back out of the same
                # call because Eq 11 reads the statistics that were just updated.
                mask = head.update_freematch_statistics(logits, block=block)
                if self.fairness_weight > 0:
                    fairness = head.fairness_loss(logits, mask, block=block)
                    if fairness is not None:
                        weighted_fairness = block_weights[block] * fairness
                        fairness_total = (
                            weighted_fairness
                            if fairness_total is None
                            else fairness_total + weighted_fairness
                        )
            block_terms[block] = (
                squared_distances, logits, per_pair, threshold, kappa, mask, fairness
            )

        if not torch.isfinite(total):
            raise FloatingPointError("seraph produced a non-finite regularization loss")
        if fairness_total is not None and not torch.isfinite(fairness_total):
            raise FloatingPointError("seraph produced a non-finite fairness loss")
        self._fairness_loss = fairness_total

        trace = self._trace_penalty(head)
        if trace is not None and not torch.isfinite(trace):
            raise FloatingPointError("seraph produced a non-finite trace penalty")
        self._trace_loss = trace

        # After the step's own statistics, so an embedding is never its own
        # neighbor in the pool that calibrated it.
        head.update_memory_bank(regularizer_embeddings)

        if self.collect_batch_diagnostics:
            self._last_diagnostics = self._diagnostics(
                head, block_terms, block_weights, regularizer_embeddings
            )
            self._last_pair_terms = self._pair_terms(block_terms)
        else:
            self._last_diagnostics = {}
            self._last_pair_terms = {}
        return total

    @staticmethod
    def _pair_terms(block_terms):
        """Return each block's per-pair tensors, detached and aligned.

        ``eta`` is broadcast to one value per pair whatever mode produced it, so
        a reader never has to branch on whether this step had a single threshold
        or one per anchor. The pair order is the one the blocks were built in:
        the upper triangle for ``uu``, row-major over the labeled rows for ``lu``.
        """

        terms = {}
        for block, (squared_distances, logits, per_pair, threshold, kappa, _, _) in (
            block_terms.items()
        ):
            squared_distances = squared_distances.detach()
            terms[block] = {
                "squared_distances": squared_distances,
                "logits": logits.detach(),
                "pair_term": per_pair.detach(),
                "threshold": torch.broadcast_to(
                    torch.as_tensor(
                        threshold,
                        dtype=squared_distances.dtype,
                        device=squared_distances.device,
                    ).detach(),
                    squared_distances.shape,
                ),
                "kappa": float(kappa),
            }
        return terms

    def pair_terms(self):
        """Return the most recent step's per-pair tensors, or ``{}``.

        Populated only while ``collect_batch_diagnostics`` is enabled, which is
        the same condition under which ``batch_diagnostics`` reports anything.
        """

        return self._last_pair_terms

    def combine_losses(self, supervised_loss, regularization_loss):
        """Add FreeMatch's fairness penalty and Eq 7's trace term.

        Each carries its own weight rather than riding on ``regularizer_weight``.
        The fairness term because arXiv:2205.07246 weights it by a separate
        lambda from its unlabeled term - and because it exists to push back on
        what the entropy term does, so scaling the two together would cancel the
        point of it. The trace term because Eq 7 gives it its own ``lambda`` and
        because it does not read the batch at all: folding it into
        ``regularizer_weight`` would put a constant-per-step term inside the
        weight that ``regularizer_target_ratio`` calibrates from measured
        gradients.
        """

        combined = super().combine_losses(supervised_loss, regularization_loss)
        if self._fairness_loss is not None:
            combined = combined + self.fairness_weight * self._fairness_loss
        if self._trace_loss is not None:
            combined = combined + self.trace_weight * self._trace_loss
        return combined

    def extra_loss_components(self):
        components = {}
        if self._fairness_loss is not None:
            components[FAIRNESS_COMPONENT] = (self._fairness_loss, self.fairness_weight)
        if self._trace_loss is not None:
            components[TRACE_COMPONENT] = (self._trace_loss, self.trace_weight)
        return components

    def extra_target_ratios(self):
        """Calibrate ``trace_weight`` from measured gradients, not from its units.

        ``tr(A)`` is a squared Frobenius norm and ``soft_rank`` is a
        dimensionless ratio in ``[1, feat_dim]``; neither is commensurable with a
        metric-learning loss, so a hand-picked ``lambda`` is a guess at a scale
        rather than a choice about strength. The ratio makes the configured
        ``trace_weight`` a probe, exactly as ``regularizer_target_ratio`` does for
        ``regularizer_weight``.
        """

        if self.trace_target_ratio <= 0:
            return {}
        return {TRACE_COMPONENT: self.trace_target_ratio}

    def clear_extra_target_ratios(self, keep=()):
        if TRACE_COMPONENT in keep:
            # GradNorm is not balancing the trace, so nothing supersedes its
            # calibration and the ratio is still the honest way to size it.
            return
        self.trace_target_ratio = 0.0

    def grad_norm_extra_components(self):
        """Let GradNorm learn ``trace_weight``, but only where it can measure it.

        Under ``trace_target="metric"`` the term's gradient over the shared trunk
        is exactly zero, and GradNorm skips its whole update on any non-positive
        component norm -- so registering it there would not balance the trace, it
        would silently freeze the supervised and regularizer weights too. The
        term keeps its configured weight instead.

        Note what GradNorm does with the one it *can* measure: its signal is the
        inverse training rate ``L(t)/L(0)``, and for ``soft_rank`` that ratio is
        how much rank has already been surrendered, so a falling value reads as
        "learning fast, downweight". That is backwards for a structural prior,
        which is why ``extra_target_ratios`` above -- periodically measure, then
        hold -- is the default for the projection target and this is opt-in
        behind ``grad_norm_alpha``.
        ``grad_norm_exclude_components=["seraph_trace"]`` is the way back to
        that default while GradNorm balances the rest.
        """

        if self.trace_weight <= 0 or self.trace_target != "projection":
            return {}
        return {TRACE_COMPONENT: self.trace_weight}

    def calibratable_component_loss(self, name):
        if name != TRACE_COMPONENT:
            return None
        return self._trace_loss

    def calibrated_component_weight(self, name):
        if name != TRACE_COMPONENT:
            raise KeyError(name)
        return self.trace_weight

    def apply_calibrated_component_weight(self, name, weight):
        if name != TRACE_COMPONENT:
            raise KeyError(name)
        self.trace_weight = float(weight)

    def _calibrate_on_labeled_pairs(self, head, projected_labeled, supervised_labels):
        """Fit this step's ``(kappa, eta)`` on the labeled-labeled block.

        Deviation 11's one point of contact with the training loop. The block is
        formed exactly as the other two are - every unordered pair of the labeled
        batch, once - except that its weak label ``y = +-1`` is read off the class
        labels instead of being inferred, which is what keeps it a calibration set
        and out of ``block_distances``.
        """

        labels = supervised_labels.detach().reshape(-1)
        if len(labels) != len(projected_labeled):
            raise ValueError(
                "seraph labeled embeddings and labels must align for the "
                "labeled-pair calibration"
            )
        labeled_distances = pairwise_squared_distances(projected_labeled)
        labeled_pairs = torch.triu_indices(
            len(projected_labeled),
            len(projected_labeled),
            offset=1,
            device=labeled_distances.device,
        )
        same_class = labels[labeled_pairs[0]] == labels[labeled_pairs[1]]
        return head.calibrate_from_labeled_pairs(
            labeled_distances[labeled_pairs[0], labeled_pairs[1]],
            same_class,
            # The rate the calibration is being moved *to* is the one unlabeled
            # pairs have, which under prior_mode="freematch" is itself adaptive.
            target_rate=head.quantile_level("uu"),
        )

    def _reference_distances(self, projected, projected_labeled, projected_bank):
        """Return the batch-to-bank distances each block's calibration may read.

        The bank holds unlabeled embeddings, so the labeled rows crossed with it
        are the same kind of pair the ``lu`` block scores and the unlabeled rows
        crossed with it are the same kind the ``uu`` block scores. Each block's
        quantile and spread therefore widen without either one being handed the
        other's distribution.
        """

        if projected_bank is None:
            return {}
        reference = {"uu": cross_squared_distances(projected, projected_bank)}
        if projected_labeled is not None:
            reference["lu"] = cross_squared_distances(projected_labeled, projected_bank)
        return reference

    def _anchor_thresholds(self, head, projected, projected_labeled, projected_bank):
        """Return the per-anchor ``eta`` of every anchor the blocks will need.

        One neighbor pool serves every anchor: the memory bank plus the current
        unlabeled batch. Using the unlabeled pool for labeled anchors too is
        deliberate - the local scale is a property of the embedding space, and
        estimating it from the largest available sample of that space beats
        estimating it from whichever half of the batch the anchor came from. It
        also removes the reason ``eta`` was calibrated per block in the first
        place: each anchor now carries its own.
        """

        pool = (
            projected
            if projected_bank is None
            else torch.cat([projected, projected_bank], dim=0)
        )
        unlabeled_distances = cross_squared_distances(projected, pool)
        # An anchor is in the pool, so its own zero distance would be its nearest
        # neighbor and shift every order statistic by one.
        self_index = torch.arange(len(projected), device=pool.device)
        unlabeled_distances[self_index, self_index] = math.inf
        thresholds = {"unlabeled": head.anchor_thresholds(unlabeled_distances)}
        if projected_labeled is not None:
            thresholds["labeled"] = head.anchor_thresholds(
                cross_squared_distances(projected_labeled, pool)
            )
        return thresholds

    @staticmethod
    def _anchor_threshold_pairs(block, anchor_thresholds, pairs, unlabeled_count):
        """Return the two endpoints' ``eta`` aligned with a block's flat pairs."""

        if block == "uu":
            unlabeled = anchor_thresholds["unlabeled"]
            return unlabeled[pairs[0]], unlabeled[pairs[1]]
        # cross_squared_distances(labeled, unlabeled).flatten() is row-major over
        # the labeled rows, so pair k compares labeled k // n_u with unlabeled
        # k % n_u.
        return (
            anchor_thresholds["labeled"].repeat_interleave(unlabeled_count),
            anchor_thresholds["unlabeled"].repeat(len(anchor_thresholds["labeled"])),
        )

    def _block_weights(self, labeled_unlabeled_active):
        """Split the objective between blocks by their share of population pairs.

        Batch pair counts would weight the blocks by batch composition instead:
        ``lu`` grows as ``n_l * n_u`` while ``uu`` grows as ``n_u^2 / 2``, so a
        labeled-heavy batch would silently amplify the cross block. The pool's own
        proportions are the fixed quantity the paper's sum over U implies.
        """

        if not labeled_unlabeled_active:
            return {"uu": 1.0}
        unlabeled_pairs = self.unlabeled_pool_size * (self.unlabeled_pool_size - 1) / 2
        labeled_pairs = float(self.labeled_pool_size) * self.unlabeled_pool_size
        population = unlabeled_pairs + labeled_pairs
        if population <= 0:
            raise RuntimeError("seraph found no unlabeled pairs to weight")
        return {"uu": unlabeled_pairs / population, "lu": labeled_pairs / population}

    def _diagnostics(self, head, block_terms, block_weights, regularizer_embeddings):
        with torch.no_grad():
            diagnostics = {
                "train/seraph/positive_pair_prior": head.positive_pair_prior,
                # What actually arrived in this step, as opposed to what the
                # loader was asked for: the only per-step evidence that the
                # unlabeled stream is feeding the objective.
                "train/seraph/unlabeled_batch_size": float(len(regularizer_embeddings)),
            }
            if head.memory_bank_size is not None:
                filled = float(head.memory_bank_count.item())
                diagnostics["train/seraph/memory_bank_filled"] = filled
                diagnostics["train/seraph/memory_bank_fraction"] = (
                    filled / head.memory_bank_size
                )
            if self._fairness_loss is not None:
                diagnostics["train/seraph/fairness_loss"] = self._fairness_loss
            if self._trace_loss is not None:
                diagnostics["train/seraph/trace_loss"] = self._trace_loss
                # Both are logged whichever term is being descended, because the
                # pair is what makes the run readable: the trace alone cannot
                # distinguish a metric that shrank from one that lost rank.
                diagnostics.update(self._trace_diagnostics(head))
            if head.calibrates_on_labeled_pairs:
                diagnostics.update(
                    {
                        "train/seraph/calibration/slope": head.calibration_slope,
                        "train/seraph/calibration/intercept": head.calibration_intercept,
                        # eta before the positivity clamp: a negative value here
                        # is the signal that the fit, or the correction applied to
                        # it, has put the whole distance axis on the dissimilar
                        # side.
                        "train/seraph/calibration/threshold_raw": (
                            head.calibration_intercept / head.calibrated_kappa()
                        ),
                        "train/seraph/calibration/threshold_clamped": float(
                            float(head.calibration_intercept) <= 0.0
                        ),
                        # Monotone counter: a step that never rises means every
                        # step found a usable fit.
                        "train/seraph/calibration/rejections": (
                            head.calibration_rejections.float()
                        ),
                    }
                )
                if self._calibration is not None:
                    diagnostics.update(
                        {
                            "train/seraph/calibration/step_slope":
                                self._calibration["slope"],
                            "train/seraph/calibration/step_intercept":
                                self._calibration["intercept"],
                            # The uncorrected fit next to the corrected one, so the
                            # size of the case-control shift is visible rather than
                            # inferred.
                            "train/seraph/calibration/step_intercept_raw":
                                self._calibration["raw_intercept"],
                            "train/seraph/calibration/sample_positive_rate":
                                self._calibration["sample_positive_rate"],
                            "train/seraph/calibration/target_positive_rate":
                                head.quantile_level("uu"),
                            "train/seraph/calibration/pairs":
                                self._calibration["pairs"],
                        }
                    )
            scored_pairs = 0
            for block, (
                    squared_distances,
                    logits,
                    per_pair,
                    threshold,
                    kappa,
                    mask,
                    fairness,
            ) in block_terms.items():
                if head.tracks_freematch:
                    thresholds = head.freematch_thresholds(block)
                    marginal = getattr(head, f"freematch_marginal_{block}")
                    histogram = getattr(head, f"freematch_histogram_{block}")
                    diagnostics.update(
                        {
                            # Eq 9's global threshold: the series to watch to see
                            # whether the rate is tracking anything at all.
                            f"train/seraph/{block}/freematch_tau": (
                                getattr(head, f"freematch_tau_{block}")
                            ),
                            # Eq 11 after MaxNorm. The similar side sitting far
                            # below the dissimilar one is the mechanism working,
                            # not a fault.
                            f"train/seraph/{block}/freematch_threshold_positive":
                                thresholds[0],
                            f"train/seraph/{block}/freematch_threshold_negative":
                                thresholds[1],
                            f"train/seraph/{block}/freematch_marginal_positive":
                                marginal[0],
                            f"train/seraph/{block}/freematch_histogram_positive":
                                histogram[0],
                            f"train/seraph/{block}/freematch_mask_rate":
                                mask.float().mean(),
                            # What eta will actually be placed at next step, after
                            # the clamp - the headline number of deviation 10.
                            f"train/seraph/{block}/adaptive_positive_pair_prior":
                                head.quantile_level(block),
                        }
                    )
                    if fairness is not None:
                        diagnostics[f"train/seraph/{block}/fairness_loss"] = fairness
                if torch.is_tensor(threshold) and threshold.numel() > 1:
                    # Per-anchor eta has no single value to report; these are the
                    # spread of the pair-level thresholds the step actually used.
                    diagnostics.update(
                        {
                            f"train/seraph/{block}/threshold_std":
                                threshold.float().std(unbiased=False),
                            f"train/seraph/{block}/threshold_min": threshold.min(),
                            f"train/seraph/{block}/threshold_max": threshold.max(),
                        }
                    )
                posterior = torch.sigmoid(logits)
                entropy = (
                    per_pair
                    if self.objective == "entropy"
                    else binary_entropy_from_logits(logits)
                )
                scored_pairs += len(squared_distances)
                diagnostics.update(
                    {
                        f"train/seraph/{block}/pair_entropy": entropy.mean(),
                        f"train/seraph/{block}/posterior_mean": posterior.mean(),
                        f"train/seraph/{block}/positive_pair_rate": (posterior > 0.5).float().mean(),
                        # Per-anchor modes have one eta per pair, so this is their
                        # mean and the spread is reported alongside it above.
                        f"train/seraph/{block}/threshold": (
                            threshold.float().mean()
                            if torch.is_tensor(threshold) and threshold.numel() > 1
                            else threshold
                        ),
                        f"train/seraph/{block}/kappa": kappa,
                        f"train/seraph/{block}/mean_squared_distance": squared_distances.mean(),
                        f"train/seraph/{block}/scored_pairs": float(len(squared_distances)),
                        f"train/seraph/{block}/block_weight": block_weights[block],
                    }
                )
            for block, (
                    squared_distances,
                    logits,
                    per_pair,
                    threshold,
                    kappa,
                    mask,
                    fairness,
            ) in block_terms.items():
                distances = squared_distances.detach().float()
                posterior = torch.sigmoid(logits)

                quantile_levels = torch.tensor(
                    [0.10, 0.25, 0.50, 0.75, 0.90],
                    device=distances.device,
                )
                q10, q25, q50, q75, q90 = torch.quantile(
                    distances,
                    quantile_levels,
                )

                # Exact conversion because the embeddings are L2-normalized and
                # metric="identity".
                cosine_similarities = 1.0 - distances / 2.0

                diagnostics.update(
                    {
                        f"train/seraph/{block}/distance_mean":
                            distances.mean(),
                        f"train/seraph/{block}/distance_std":
                            distances.std(unbiased=False),
                        f"train/seraph/{block}/distance_min":
                            distances.min(),
                        f"train/seraph/{block}/distance_q10":
                            q10,
                        f"train/seraph/{block}/distance_q25":
                            q25,
                        f"train/seraph/{block}/distance_median":
                            q50,
                        f"train/seraph/{block}/distance_q75":
                            q75,
                        f"train/seraph/{block}/distance_q90":
                            q90,
                        f"train/seraph/{block}/distance_max":
                            distances.max(),

                        f"train/seraph/{block}/fraction_below_threshold":
                            (distances < threshold).float().mean(),
                        f"train/seraph/{block}/fraction_near_threshold":
                            (
                                    (distances - threshold).abs() < 0.1
                            ).float().mean(),

                        f"train/seraph/{block}/cosine_mean":
                            cosine_similarities.mean(),
                        f"train/seraph/{block}/cosine_std":
                            cosine_similarities.std(unbiased=False),

                        f"train/seraph/{block}/logit_mean":
                            logits.mean(),
                        f"train/seraph/{block}/logit_abs_mean":
                            logits.abs().mean(),
                        f"train/seraph/{block}/posterior_std":
                            posterior.std(unbiased=False),
                    }
                )
            diagnostics["train/seraph/scored_pairs"] = float(scored_pairs)
            # The uu block is the objective whenever the cross block is off, so
            # keep its series addressable under the original unprefixed names too.
            for name in ("pair_entropy", "posterior_mean", "positive_pair_rate",
                         "threshold", "kappa", "mean_squared_distance"):
                diagnostics[f"train/seraph/{name}"] = diagnostics[f"train/seraph/uu/{name}"]
            return diagnostics

    def batch_diagnostics(self):
        return {
            name: float(value.detach().item()) if torch.is_tensor(value) else float(value)
            for name, value in self._last_diagnostics.items()
        }
