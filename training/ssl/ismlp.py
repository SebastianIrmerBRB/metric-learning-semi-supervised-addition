"""ISMLP proxy-entropy regularization for deep metric learning.

Implements the *unlabeled* part of "Efficient Information-Theoretic Large-Scale
Semi-Supervised Metric Learning via Proxies" (Chen and Wang, Applied Sciences
13(15):8993, 2023, doi:10.3390/app13158993), with the paper's equation numbers
used throughout.

ISMLP replaces SERAPH's pairwise posterior with an NCA-style posterior over a
set of ``m`` proxy vectors ``Z = {z_1, ..., z_m}``. For an instance ``x_i`` the
probability of choosing proxy ``z_k`` is (Eq 6)

    q_ik = exp(-d_M^2(x_i, z_k)) / sum_j exp(-d_M^2(x_i, z_j))

and ``q_i = (q_i1, ..., q_im)`` is a valid discrete distribution (Eq 7). The
unlabeled term (Eq 9) minimizes its entropy, averaged over the unlabeled pool:

    min_M  -(1 / n_u) sum_{i in U} sum_k q_ik log q_ik

which is the low-density separation assumption stated over proxy assignments:
every unlabeled instance should sit decisively near *one* proxy. Because each
instance is scored against ``m`` proxies rather than against every other
instance, the cost is ``O(n m d)`` instead of SERAPH's ``O(n^2 d)``, which is
the paper's whole reason for existing.

Deep form
---------
``x`` is the model's L2-normalized retrieval embedding (frozen backbone plus the
trainable projection head), and ``M = I`` on it: ``d_M^2`` is the plain squared
Euclidean distance between the embedding and a proxy. The metric ISMLP learns is
therefore the projection head itself, exactly as the paper's own Sec 3.4 reading
of ``M = P R P^T`` intends -- first project into a lower-dimensional space, then
measure there. Gradients flow through the embedding, so the entropy term shapes
the retrieval space rather than an isolated head.

Deviations from the paper
-------------------------
1. **Eq 8 is optional; only Eq 9 is always on.** The labeled proxy alignment
   over the similar set ``S`` is ``labeled_weight``, which defaults *on* under
   ``proxy_mode="learned"`` -- the one mode with no other force on ``Z`` -- and
   off everywhere else, where the run's configured metric loss takes its place
   on the labeled stream, exactly as this project's SERAPH port drops Eq 7's
   labeled log-likelihood. ``regularizer_weight`` is Eq 11's ``lambda`` on the
   entropy term either way. Deviation 9 is how Eq 8 is read here.

   Turning it on does not by itself make the objective Eq 11: the run's metric
   loss is still there at ``supervised_weight``, so the labeled stream carries
   two terms where Eq 11 has one. ``supervised_weight: 0.0`` is Eq 11 literally,
   at the cost of making the run's ``--loss`` inert.

2. **Sec 3.4's ``M = P R P^T`` factorization is dropped, and with it Eq 12's
   Burg divergence ``mu * r(R, R_0)``.** The projection head is the metric, so
   there is no separate ``R`` for a structural prior to be placed on. ``mu`` has
   nothing to weight and is not exposed.

3. **The optimizer is SGD on the whole objective, not Sec 4's alternating
   scheme.** The paper fixes ``Z`` and takes Riemannian steps on the product
   manifold ``St(p, d) x S^p_++``, then fixes ``(P, R)`` and solves each ``z_k``
   in closed form (Eq 29). Neither manifold survives deviation 2, and the
   closed-form ``Z`` update is derived from an objective that includes Eq 8, so
   it does not apply once Eq 8 is gone. ``proxy_mode`` below is what replaces
   it.

4. **``q`` is read at an inverse temperature ``kappa``:**
   ``q_ik ~ exp(-kappa * d^2)``. The paper's Eq 6 has ``kappa = 1``, which is
   well scaled for its own features -- unnormalized, PCA-reduced to 150 dims --
   but not for an L2-normalized retrieval embedding, where ``d^2`` is confined to
   ``[0, 4]``. At ``kappa = 1`` the softmax over a few hundred proxies is nearly
   flat whatever the embedding does, the entropy sits within a hair of
   ``log m``, and the term is close to inert. ``kappa_mode="batch_std"`` divides
   the configured ``kappa`` by the spread of the batch's own distances, so the
   knob means the same thing across datasets; it is the same mechanism, and the
   same name, as ``SeraphPosteriorHead.inverse_temperature``.

   Measured, on ``m = 200`` unit proxies in ``d = 512`` and ``n = 256`` rows
   drawn as ``normalize(z_k + 0.35 * N(0, I))`` under torch seed 0 -- the noise
   scale matters, because plain uniform rows on the sphere give a visibly
   different curve (0.908 at configured 1, not 0.888). On that sample ``d^2`` has
   mean 2.00 and standard deviation 0.090, so ``kappa_mode="fixed"`` at the
   paper's ``kappa = 1`` leaves the mean assignment entropy at 0.9992 of
   ``log m`` -- the term is, numerically, not there. ``kappa_mode="batch_std"``
   turns the same configured 1 into an effective 11.1 (entropy 0.888 of the
   maximum), 2 into 22.2 (0.559, mean top assignment 0.36), and 3 into 33.3
   (0.322, 0.576).

   ``batch_std`` carries a mild negative feedback worth knowing about: as rows
   tighten onto their proxies the distance distribution goes bimodal and its
   spread *rises*, which lowers the effective ``kappa``. Interpolating the same
   rows 90% of the way onto their nearest proxy raises the spread from 0.090 to
   0.166 and damps a configured 2 from an effective 22.2 to 12.1. It is a real
   feedback and far too weak to be a floor -- the entropy is already at 0.0000 of
   the maximum a quarter of the way along that path -- so a run whose
   ``assignment_entropy_normalized`` stops falling has stalled for some other
   reason, not been throttled by the temperature.

5. **Eq 24 is a sign typo, and Eq 9/11 are followed instead.** The ``Z``
   subproblem writes ``+ lambda/n_u sum q log q``, which is entropy
   *maximization*; Eq 9, Eq 11 and the surrounding text ("the distribution
   should be a perky one") all say minimization. This module minimizes.

6. **Proxy vectors are not solved in closed form; ``proxy_mode`` selects one of
   three substitutes.** Some substitute is forced rather than optional: Eq 29 is
   derived from an objective that contains Eq 8, so dropping the labeled term
   (deviation 1) removes the closed form's derivation along with the only
   label-anchoring force on ``Z``. Be precise about what ``centroids`` then
   preserves. It keeps the paper's *alternation structure* -- the metric moves
   for an interval, then ``Z`` is re-solved against it -- but k-means is not the
   paper's ``Z``-step: it minimizes within-cluster variance, reads no labels, and
   is blind to the entropy the refit is supposed to serve. If fidelity to Eq 11's
   joint objective is what matters and the run's metric loss is proxy-based,
   ``supervised_loss`` is the closer analogue; ``centroids`` is the safer default.

   ``centroids`` (default)
       ``Z`` is the k-means (or GMM) centroid set of the unlabeled pool's
       embeddings, recomputed on the SSL config's ``update_mode`` cadence and
       held fixed in between, as a non-trainable buffer. It is the closest
       reading of the paper's own description of a proxy -- "the mean center of
       each class, or anchor that aggregates the local similarity information of
       some local instances" -- and it is the only mode in which the entropy term
       is unambiguously a regularizer *on the embedding*.
   ``learned``
       ``Z`` is an ``nn.Parameter`` trained by SGD alongside the model,
       initialized from those same centroids (``proxy_init="centroids"``) or
       from random unit vectors. It is stepped by *this module's own* AdamW at
       ``proxy_lr``, which defaults to the run's ``--classifier_lr`` -- the same
       rate every proxy loss in ``CLASSIFICATION_LOSSES`` is optimized at -- and
       not by the model optimizer at ``--lr``. The distinction is the difference
       between the mode working and the mode being a no-op: those two defaults
       are 1.0 and 1e-6, and measured over 2000 steps a proxy set at 1e-6 moves
       0.023 of a unit and leaves the entropy at 0.62 of ``log m``, against 1.40
       and 0.03 at 1.0. See ``after_optimizer_step`` for how the parameter is
       kept out of the model optimizer. This is the paper's joint optimization of
       ``(M, Z)``, minus the closed form. It is also the one mode
       ``labeled_weight`` defaults *on* for, and this is the reason: with Eq 8
       off, the entropy term is the *only* thing pulling on ``Z``, and it has a
       trivial minimizer -- push ``m - 1`` proxies away until every ``q_i`` is
       one-hot at no cost to the embedding. Eq 8 is what the paper anchors ``Z``
       against exactly that with, so what the rest of this paragraph describes is
       ``labeled_weight=0``. Two further mitigations hold either way. The default
       ``normalize_proxies=True`` confines
       ``Z`` to the unit sphere, so "away" bounds out at antipodal crowding
       rather than running to infinity, and ``after_optimizer_step`` keeps the
       stored parameter there. Measured, 400 AdamW steps on ``Z`` alone against
       fixed embeddings (``n=256`` rows around ``m=64`` true clusters, ``m=64``
       proxies, random init): the entropy reached 0.0007 of ``log m`` with
       occupancy 1.00 and a marginal entropy of 0.98 -- the minimum was reached
       by *spreading* ``Z`` over the data, not by collapsing it, because on a
       sphere making every ``q_i`` peaked is what separating the proxies buys.
       Read that as one favorable case rather than a guarantee: ``m`` matched the
       cluster count exactly and only ``Z`` could move. The warning stands for
       the joint embedding-and-``Z`` dynamic, where the embedding can also travel
       to meet a degenerate ``Z``. Watch ``train/ismlp/proxy_occupancy``, and see
       ``marginal_entropy_weight``.
   ``supervised_loss``
       ``Z`` is the proxy table of the run's supervised criterion
       (ProxyAnchor, ProxyNCA, MixedLabelPropagationProxyLoss, SoftTriple).
       The supervised loss then plays the role of Eq 8 literally: one proxy set,
       aligned by labels on the labeled stream and sharpened by entropy on the
       unlabeled one, which is what Eq 11 optimizes jointly. ``m`` is whatever
       the criterion has, so it cannot be configured here.

7. **``marginal_entropy_weight`` is an addition, off by default.** Eq 9 alone is
   minimized perfectly by assigning the entire unlabeled pool to one proxy.
   Setting this subtracts the entropy of the batch-averaged assignment,
   ``H(mean_i q_i)``, turning Eq 9 into the mutual-information objective of
   RIM / IMSAT. It is reported and weighted as its own component -- as
   ``log m - H(mean_i q_i)``, so that the number logged is a non-negative
   deficit that reads as zero when every proxy is used equally.

8. **k-means replaces the paper's GMM initialization by default.** The paper
   fits a Gaussian mixture and takes its component means. Full covariances are
   not affordable here -- ``m`` in the hundreds times ``512 x 512`` per component,
   against a paper whose features were PCA-reduced to 150 dims first -- so
   ``centroid_method="gmm"`` fits diagonal ones. That is also why it is not the
   default, and the reason is about the *diagonal* mixture rather than the full
   one: on L2-normalized embeddings a per-dimension variance buys little over a
   plain centroid, the mixture has no FAISS or GPU path where k-means has both,
   and ``normalize_proxies`` projects the component means back onto the sphere
   afterwards, which erases most of what remains. The option is kept for the
   ablation, not because it is expected to differ.

9. **Eq 8 is read three ways the paper leaves to its own setting.**
   ``labeled_weight`` scales it; deviation 1 says when it is on.

   *What ``S`` is.* The paper is handed a set of similar pairs. Here the labeled
   stream carries class labels, so ``S`` is every ordered same-class pair inside
   the labeled batch, ``i != j``. ``|S|`` is therefore a property of the batch
   rather than a constant -- ``sum_c n_c (n_c - 1)`` over its class counts, about
   ``3 n_l`` at the shipped ``sampler_m=4``. A labeled batch whose classes are
   all distinct has ``S`` empty; Eq 8 is then undefined rather than zero, so the
   term sits that step out instead of contributing a zero that the target-ratio
   probe would average in. Watch ``train/ismlp/labeled_pairs``.

   *What ``p(x_j)`` is.* ``labeled_assignment="nearest"`` is Eq 5 read literally:
   the target is ``argmin_k d^2(x_j, z_k)``, discovered rather than assigned,
   which is what lets ``m`` exceed ``#Class`` the way the paper's own sweep over
   ``{#Class, 2#Class, 3#Class}`` requires. That argmin is discrete and carries
   no gradient, so ``x_j`` reaches the objective only through the index it picks
   and only ``x_i`` and ``Z`` are pulled. ``labeled_assignment="class"`` is
   ProxyNCA proper instead: ``Z`` is blocked by class, ``proxies_per_class`` rows
   each, and the target is the nearest of the row's *own* class's block -- at one
   proxy per class exactly PML's ``ProxyNCALoss``, which is Eq 8 with
   ``softmax_scale`` for ``kappa``. It requires ``proxy_mode="learned"`` and
   ``proxy_init="random"``: a k-means centroid of the pooled unlabeled
   embeddings has no class to be placed in a block of.

   *What temperature it is read at.* Deviation 4's ``kappa`` applies to Eq 8's
   softmax too -- it is the same Eq 6 posterior -- and under
   ``kappa_mode="batch_std"`` it is standardized by the spread of the *labeled*
   batch's own instance-to-proxy distances, the same rule the unlabeled term
   applies to its own. The two run at the same configured ``kappa`` and, early
   on, at close to the same effective one, but they are not pinned together:
   they are reported separately as ``train/ismlp/kappa`` and
   ``train/ismlp/labeled_kappa``.

   The weight is its own component (``ismlp_labeled``) rather than part of
   ``regularizer_weight``, for the reason deviation 7's guard is not either:
   ``regularizer_weight`` is Eq 11's ``lambda`` on the entropy alone, and the
   calibration has to see the two gradients separately. ``labeled_target_ratio``
   sizes it from measured gradients the same way.
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.utils.data import DataLoader

import utils
from .algorithms import faiss_gpu_flat_index, require_faiss
from .config import should_rebuild_on_epoch
from .data import CombinedTrainingLoader, UnlabeledSubset
from .embeddings import extract_embeddings
from .interfaces import BaseTrainingRegularizer


# Where ``Z`` comes from; see deviation 6.
PROXY_MODES = ("centroids", "learned", "supervised_loss")

# How ``proxy_mode="learned"`` seeds its parameter before training touches it.
PROXY_INITS = ("centroids", "random")

# ``fixed`` is Eq 6 read literally at the configured kappa. ``batch_std``
# standardizes by the batch's own distance spread; see deviation 4.
KAPPA_MODES = ("fixed", "batch_std")

# How Eq 8 picks the target proxy ``p(x_j)``; see deviation 9.
LABELED_ASSIGNMENTS = ("nearest", "class")

CENTROID_METHODS = ("kmeans", "gmm")
CENTROID_BACKENDS = ("auto", "faiss", "sklearn")

# Below this many ``rows x centroids`` the GPU index costs more to build than it
# saves, so ``centroid_gpu=None`` stays on the CPU. Deliberately four times
# ``slade.DEFAULT_KMEANS_GPU_MIN_WORK``, which is calibrated for a different
# call: SLADE clusters once per self-training iteration and then *searches* the
# index for per-row assignments, while this module keeps only the centroids and
# throws the index away.
#
# Measured here at d=512, warmed, best of two, against the FAISS CPU path:
#
#     5864 x 100  (0.59M)   CPU 0.16s   GPU 0.32s   0.49x
#     5864 x 200  (1.17M)   CPU 0.16s   GPU 0.32s   0.51x
#     7000 x 196  (1.37M)   CPU 0.17s   GPU 0.32s   0.53x
#    12000 x 392  (4.70M)   CPU 0.53s   GPU 0.44s   1.19x
#    25882 x 3997 (103M)    CPU 7.90s   GPU 0.86s   9.15x
#
# The GPU path has a ~0.32s floor it cannot go below -- resource allocation,
# index build, host-to-device copy -- so everything under a few million rows x
# centroids pays that floor for nothing. A Cars196 or CUB fold sits at 1-1.4M
# and is twice as fast on the CPU; In-Shop and SOP are where the GPU earns its
# place. Set ``centroid_gpu=true`` to force it anyway, at the cost of FAISS's
# default temp arena per concurrent trial -- ``faiss_gpu_flat_index`` only caps
# that when the SSL low-memory policy is active.
DEFAULT_KMEANS_GPU_MIN_WORK = 4_000_000

# One label for the deviation-7 term, shared by its diagnostics, its weight and
# its calibration.
MARGINAL_COMPONENT = "ismlp_marginal"

# The same, for Eq 8's labeled proxy alignment; see deviation 9.
LABELED_COMPONENT = "ismlp_labeled"

# Eq 11 puts no coefficient on Eq 8 -- ``lambda`` scales the entropy term against
# it -- so wherever the labeled term is on at all, one is what it means.
DEFAULT_LABELED_WEIGHT = 1.0

# The module ``configure_model`` attaches for the two modes that own their
# proxies. ``supervised_loss`` attaches nothing: the proxies are the criterion's.
PROXY_MODULE = "ismlp_proxies"

# The paper tunes m over {#Class, 2#Class, 3#Class}; the smallest of the three is
# the default, so that "one proxy per class" is what an unset config means.
DEFAULT_PROXIES_PER_CLASS = 1.0

# One unlabeled sample per labeled sample, matching the other regularizers here.
DEFAULT_UNLABELED_RATIO = 1.0

# Fallback rate for Z when no run args have been seen, matching the repo's own
# ``--classifier_lr`` default. A real run resolves it in validate_run_args.
DEFAULT_PROXY_LR = 1.0

# Restarts for the k-means path. The GMM path fixes its own at one; see the
# warning in ``IsmlpRegularizer.__init__``.
DEFAULT_CENTROID_N_INIT = 10


def _validate_positive(name, value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"ismlp {name} must be finite and positive")
    return value


def _validate_non_negative(name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"ismlp {name} must be finite and non-negative")
    return value


def _validate_positive_integer(name, value):
    integer = int(value)
    if integer <= 0:
        raise ValueError(f"ismlp {name} must be a positive integer")
    return integer


def cross_squared_distances(rows, proxies):
    """Return the ``[len(rows), len(proxies)]`` squared distances of Eq 6.

    The Gram form keeps memory at ``O(n m)`` instead of ``O(n m d)``, which is
    the whole point of scoring against proxies rather than against pairs.
    Rounding can make the difference slightly negative, so the result is clamped
    at zero before it reaches a softmax.
    """

    row_norms = rows.pow(2).sum(dim=1, keepdim=True)
    proxy_norms = proxies.pow(2).sum(dim=1, keepdim=True)
    distances = row_norms + proxy_norms.t() - 2.0 * (rows @ proxies.t())
    return distances.clamp_min(0.0)


def inverse_temperature(kappa, kappa_mode, squared_distances):
    """Return the ``kappa`` Eq 6's exponent is read at for this batch.

    ``batch_std`` divides by the spread of the batch's own instance-to-proxy
    distances, so the sharpness of ``q`` does not depend on how concentrated the
    embedding happens to be at this point in training. It is detached for the
    same reason SERAPH detaches its own: it scales the gradient rather than
    deciding its sign, so letting the objective flatten ``q`` by shrinking the
    spread would be a way to lower the entropy without separating anything.
    """

    if kappa_mode == "fixed":
        return kappa
    spread = squared_distances.detach().std()
    # A batch whose rows are equidistant from every proxy carries no ordering to
    # sharpen; falling back to the raw kappa keeps the term finite and inert.
    if not torch.isfinite(spread) or float(spread) <= 0.0:
        return kappa
    return kappa / spread


def assignment_log_probabilities(squared_distances, kappa):
    """Return ``log q`` of Eq 6 at inverse temperature ``kappa``.

    ``log_softmax`` rather than ``log(softmax(...))``: ``q`` underflows to zero
    for distant proxies long before its logit does, and the entropy needs the
    log of exactly those terms.
    """

    return F.log_softmax(-kappa * squared_distances, dim=1)


def labeled_target_proxies(squared_distances, labels, assignment, proxies_per_class):
    """Return Eq 5's ``p(x_j)`` for every labeled row, as indices into ``Z``.

    ``nearest`` is Eq 5 unrestricted -- the closest proxy under the current
    metric, whichever class it has come to serve. ``class`` restricts the same
    argmin to the ``proxies_per_class`` rows blocked under the row's own label,
    which is ProxyNCA's label assignment and, at one proxy per class, is the
    label itself.

    The result indexes a softmax and is never differentiated; the caller passes
    detached distances for that reason. See deviation 9.
    """

    if assignment == "nearest":
        return squared_distances.argmin(dim=1)
    per_class = int(proxies_per_class)
    num_proxies = squared_distances.shape[1]
    if num_proxies % per_class:
        raise ValueError(
            f"ismlp labeled_assignment='class' needs m divisible by "
            f"proxies_per_class={per_class}, got m={num_proxies}"
        )
    num_classes = num_proxies // per_class
    if int(labels.max()) >= num_classes:
        raise ValueError(
            f"ismlp labeled_assignment='class' blocks Z into {num_classes} classes, "
            f"but the batch carries label {int(labels.max())}. Z is sized from the "
            "run's label mapper, so this means the labels reaching the loss were not "
            "the mapped ones"
        )
    blocks = squared_distances.view(len(squared_distances), num_classes, per_class)
    rows = torch.arange(len(blocks), device=blocks.device)
    own_block = blocks[rows, labels]
    return labels * per_class + own_block.argmin(dim=1)


def labeled_pair_counts(targets, labels, num_proxies, dtype):
    """Return the ``[n_l, m]`` multiplicities of Eq 8's similar pairs.

    ``S`` is every ordered same-class pair in the labeled batch, and Eq 8 reads
    ``log q_{i, p(x_j)}`` once per pair. Two rows of ``i``'s class that landed on
    the same proxy therefore contribute that proxy's log twice, which makes the
    pair sum a weighted sum over ``log q_i`` rather than a gather:
    ``counts[i, k]`` is how many ``j != i`` of ``i``'s class chose proxy ``k``.
    One ``[n_l, n_l] @ [n_l, m]`` matmul replaces ``|S|`` lookups, and
    ``counts.sum()`` is exactly ``|S|``.
    """

    same = (labels.unsqueeze(1) == labels.unsqueeze(0)).to(dtype)
    same.fill_diagonal_(0.0)
    return same @ F.one_hot(targets, num_proxies).to(dtype)


def proxy_centroids(
    features,
    num_proxies,
    seed,
    method="kmeans",
    iterations=25,
    n_init=DEFAULT_CENTROID_N_INIT,
    backend="auto",
    use_gpu=None,
    stats=None,
):
    """Return ``num_proxies`` centroid rows for ``features``.

    The k-means path mirrors ``slade.kmeans_cluster_labels`` -- FAISS where it is
    installed, scikit-learn otherwise, GPU index by size rule -- but keeps the
    centroids instead of the assignments, because that is what ``Z`` is.

    ``method="gmm"`` is the paper's own initialization, restricted to diagonal
    covariances; see deviation 8.

    Pass a dict as ``stats`` to receive the backend that actually ran and how
    long it took. A silent GPU fallback is otherwise indistinguishable from a
    GPU run.
    """

    started_at = time.perf_counter()

    def record(name):
        if stats is not None:
            stats["backend"] = name
            stats["seconds"] = time.perf_counter() - started_at

    features = np.ascontiguousarray(np.asarray(features, dtype=np.float32))
    if features.ndim != 2 or len(features) == 0 or features.shape[1] == 0:
        raise ValueError("ismlp proxy initialization needs a non-empty feature matrix")
    num_proxies = int(num_proxies)
    if num_proxies < 2:
        raise ValueError("ismlp num_proxies must be at least 2")
    if num_proxies > len(features):
        raise ValueError(
            f"ismlp num_proxies={num_proxies} exceeds the {len(features)} unlabeled samples"
        )
    if method not in CENTROID_METHODS:
        raise ValueError(f"ismlp centroid_method must be one of {list(CENTROID_METHODS)}")
    if backend not in CENTROID_BACKENDS:
        raise ValueError(f"ismlp centroid_backend must be one of {list(CENTROID_BACKENDS)}")

    if method == "gmm":
        from sklearn.mixture import GaussianMixture

        mixture = GaussianMixture(
            n_components=num_proxies,
            # Full covariances are d x d per component: at d=512 and m=200 that
            # is 52M parameters fitted from a few thousand rows. Diagonal is the
            # only affordable choice at embedding dimensionality.
            covariance_type="diag",
            max_iter=int(iterations),
            # Fixed at one restart, unlike the k-means path: a mixture refit is
            # far more expensive per restart, and ``init_params`` already seeds
            # it from k-means++ rather than at random. ``n_init``, ``backend``
            # and ``use_gpu`` do not reach this branch at all, which the
            # constructor warns about rather than leaving to be discovered here.
            n_init=1,
            init_params="k-means++",
            random_state=int(seed) % (2**31 - 1),
        )
        mixture.fit(features)
        record("sklearn-gmm")
        return np.ascontiguousarray(mixture.means_.astype(np.float32))

    if backend != "sklearn":
        try:
            faiss = require_faiss("ismlp proxy initialization")
        except ImportError:
            if backend == "faiss":
                raise
            faiss = None
        if faiss is not None:
            purpose = "ismlp proxy initialization"
            work = len(features) * num_proxies
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
                    k=num_proxies,
                    niter=int(iterations),
                    seed=int(seed) % (2**31 - 1),
                    # A single unlucky initialization can collapse proxies onto
                    # each other, which the entropy term then cannot tell apart
                    # from a genuinely dense region.
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
                    record("faiss-gpu")
                    return np.ascontiguousarray(
                        np.asarray(clustering.centroids, dtype=np.float32)
                    )
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
            record("faiss-cpu")
            return np.ascontiguousarray(np.asarray(clustering.centroids, dtype=np.float32))

    from sklearn.cluster import KMeans

    estimator = KMeans(
        n_clusters=num_proxies,
        n_init=int(n_init),
        max_iter=int(iterations),
        random_state=int(seed) % (2**31 - 1),
    )
    estimator.fit(features)
    record("sklearn")
    return np.ascontiguousarray(estimator.cluster_centers_.astype(np.float32))


class IsmlpProxyHead(nn.Module):
    """The proxy set ``Z`` of Eq 5-7, owned by the model so a checkpoint keeps it.

    ``proxy_mode="learned"`` makes ``proxies`` an ``nn.Parameter`` the optimizer
    collects; ``proxy_mode="centroids"`` makes it a buffer the refresh writes and
    nothing differentiates. Both are reached the same way, and both are saved and
    restored with the model, which is what makes a resumed run score against the
    proxies it was trained against rather than a fresh clustering.

    ``normalize`` projects the rows back onto the unit sphere the embeddings live
    on. It matters most in ``centroids`` mode: a k-means centroid is the mean of
    unit vectors and therefore sits *inside* the ball, by an amount that measures
    how tight its cluster is -- so without normalization a diffuse cluster's
    centroid is closer to everything and its proxy wins assignments it has no
    claim to.
    """

    def __init__(self, feat_dim, num_proxies, trainable, normalize=True):
        super().__init__()
        self.feat_dim = _validate_positive_integer("feat_dim", feat_dim)
        self.num_proxies = int(num_proxies)
        if self.num_proxies < 2:
            raise ValueError("ismlp needs at least two proxy vectors")
        self.trainable = bool(trainable)
        self.normalize = bool(normalize)

        proxies = F.normalize(torch.randn(self.num_proxies, self.feat_dim), p=2.0, dim=1)
        if self.trainable:
            # Deliberately created frozen. The engine builds the model optimizer
            # from ``[p for p in model.parameters() if p.requires_grad]`` after
            # ``configure_model`` has run, so a parameter that is frozen at this
            # moment is excluded from it for the life of the run -- which is what
            # lets ``IsmlpRegularizer`` drive Z with its own optimizer at a proxy
            # learning rate instead of the backbone's. ``enable_gradients`` flips
            # it back on once that optimizer exists. It stays an ``nn.Parameter``
            # rather than a buffer so ``.to()`` preserves leaf-ness and
            # ``state_dict`` keeps carrying it.
            self.proxies = nn.Parameter(proxies, requires_grad=False)
        else:
            self.register_buffer("proxies", proxies)
        # ``centroids`` mode must never score against the random seed values, and
        # ``learned`` mode must not either when it is initialized from centroids;
        # this is the flag compute_loss checks rather than trusting the schedule.
        self.register_buffer("proxies_initialized", torch.zeros((), dtype=torch.bool))

    def proxy_vectors(self):
        """Return ``Z`` in float32, on the sphere when ``normalize`` is set."""

        proxies = self.proxies.float()
        return F.normalize(proxies, p=2.0, dim=1) if self.normalize else proxies

    def set_proxies(self, values):
        """Replace ``Z`` in place and mark it initialized."""

        values = torch.as_tensor(values, dtype=torch.float32)
        if values.shape != (self.num_proxies, self.feat_dim):
            raise ValueError(
                f"ismlp expected proxies of shape {(self.num_proxies, self.feat_dim)}, "
                f"got {tuple(values.shape)}"
            )
        with torch.no_grad():
            self.proxies.copy_(values.to(self.proxies.device))
            self.proxies_initialized.fill_(True)

    def enable_gradients(self):
        """Make the trainable proxies differentiable, once past optimizer setup."""

        if self.trainable and not self.proxies.requires_grad:
            self.proxies.requires_grad_(True)

    def mark_initialized(self):
        """Accept the random rows as ``Z``; only ``proxy_init="random"`` does this."""

        with torch.no_grad():
            self.proxies_initialized.fill_(True)


class IsmlpRegularizer(BaseTrainingRegularizer):
    """ISMLP's unlabeled proxy-assignment entropy as a regularizer (Eq 9)."""

    name = "ismlp"
    # One deterministic view per unlabeled image is all Eq 6 needs, so the
    # frozen-backbone feature cache stays usable.
    supports_frozen_feature_precompute = True
    # The unlabeled batch rides along in the labeled stream's forward pass.
    uses_joint_forward = True
    # A centroid refresh is one call the engine can repeat mid-epoch. The other
    # two proxy modes reject the schedule in make_loader rather than silently
    # ignoring its requests.
    supports_sample_scoped_refresh = True
    extra_component_names = (LABELED_COMPONENT, MARGINAL_COMPONENT)

    def __init__(
        self,
        regularizer_weight=1.0,
        supervised_weight=1.0,
        proxy_mode="centroids",
        proxy_init=None,
        proxy_gradient=None,
        proxy_lr=None,
        num_proxies=None,
        proxies_per_class=None,
        normalize_proxies=True,
        kappa=1.0,
        kappa_mode="batch_std",
        labeled_weight=None,
        labeled_assignment="nearest",
        labeled_target_ratio=None,
        marginal_entropy_weight=0.0,
        marginal_entropy_target_ratio=None,
        centroid_method="kmeans",
        centroid_iterations=25,
        centroid_n_init=10,
        centroid_backend="auto",
        centroid_gpu=None,
        unlabeled_ratio=None,
        unlabeled_batch_size=None,
    ):
        super().__init__(
            regularizer_weight=regularizer_weight,
            supervised_weight=supervised_weight,
        )
        self.proxy_mode = str(proxy_mode)
        if self.proxy_mode not in PROXY_MODES:
            raise ValueError(f"ismlp proxy_mode must be one of {list(PROXY_MODES)}")
        self.owns_proxies = self.proxy_mode in ("centroids", "learned")

        # Read before proxy_init because it decides what a sane default init is.
        self.labeled_assignment = str(labeled_assignment)
        if self.labeled_assignment not in LABELED_ASSIGNMENTS:
            raise ValueError(
                f"ismlp labeled_assignment must be one of {list(LABELED_ASSIGNMENTS)}"
            )
        if self.labeled_assignment == "class" and self.proxy_mode != "learned":
            raise ValueError(
                "ismlp labeled_assignment='class' blocks Z by label, which only "
                f"proxy_mode='learned' owns; {self.proxy_mode!r} does not tie its "
                "proxy rows to classes. Eq 5 -- labeled_assignment='nearest' -- is "
                "what the paper does and works in every mode"
            )

        # ``class`` blocks Z by row order, and a k-means centroid of the pooled
        # unlabeled embeddings has no class to be placed in one; ProxyNCA's own
        # initialization is random anyway.
        default_proxy_init = "centroids" if self.labeled_assignment == "nearest" else "random"
        if proxy_init is None:
            self.proxy_init = default_proxy_init if self.proxy_mode == "learned" else None
        else:
            self.proxy_init = str(proxy_init)
            if self.proxy_mode != "learned":
                raise ValueError(
                    "ismlp proxy_init only applies to proxy_mode='learned'; "
                    f"{self.proxy_mode!r} has nothing to initialize"
                )
            if self.proxy_init not in PROXY_INITS:
                raise ValueError(f"ismlp proxy_init must be one of {list(PROXY_INITS)}")
            if self.labeled_assignment == "class" and self.proxy_init == "centroids":
                raise ValueError(
                    "ismlp labeled_assignment='class' needs proxy_init='random': the "
                    "class blocks of Z are defined by row order, and a centroid of the "
                    "pooled unlabeled embeddings has no class to be placed in one"
                )

        if proxy_gradient is None:
            # A buffer has no gradient to give; the other two modes hold real
            # parameters and the paper optimizes Z, so they default to letting it.
            self.proxy_gradient = self.proxy_mode != "centroids"
        else:
            self.proxy_gradient = bool(proxy_gradient)
            if self.proxy_gradient and self.proxy_mode == "centroids":
                raise ValueError(
                    "ismlp proxy_gradient=True needs proxies the objective can move, "
                    "but proxy_mode='centroids' stores them as a buffer refreshed from "
                    "k-means. Use proxy_mode='learned' to optimize Z"
                )
            if not self.proxy_gradient and self.proxy_mode == "learned":
                # Not an error - it is a legitimate ablation - but nothing then
                # steps Z: the parameter is created frozen and excluded from the
                # model optimizer, and this module only builds its own optimizer
                # when proxy_gradient is on.
                logger.warning(
                    "ismlp proxy_mode='learned' with proxy_gradient=False freezes Z at "
                    "its initialization and no optimizer steps it; "
                    "proxy_mode='centroids' with update_mode='once' is the same "
                    "objective, and says so in the config"
                )

        self.proxy_lr = (
            None if proxy_lr is None else _validate_positive("proxy_lr", proxy_lr)
        )
        if self.proxy_lr is not None and not (
            self.proxy_mode == "learned" and self.proxy_gradient
        ):
            raise ValueError(
                "ismlp proxy_lr only applies to proxy_mode='learned' with "
                "proxy_gradient=True, which is the one configuration this module "
                "steps Z itself. Under 'centroids' Z is fitted, and under "
                "'supervised_loss' it belongs to the criterion's own optimizer at "
                "--classifier_lr"
            )
        self.num_proxies = (
            None if num_proxies is None else _validate_positive_integer("num_proxies", num_proxies)
        )
        self.proxies_per_class = (
            None
            if proxies_per_class is None
            else _validate_positive("proxies_per_class", proxies_per_class)
        )
        if self.proxy_mode == "supervised_loss":
            configured = [
                name
                for name, value in (
                    ("num_proxies", self.num_proxies),
                    ("proxies_per_class", self.proxies_per_class),
                )
                if value is not None
            ]
            if configured:
                raise ValueError(
                    f"ismlp proxy_mode='supervised_loss' reads Z from the run's "
                    f"criterion, so {configured} cannot set its size; m is whatever "
                    "that loss was built with"
                )
        elif self.num_proxies is not None and self.proxies_per_class is not None:
            logger.warning(
                f"ismlp received both num_proxies={self.num_proxies} and "
                f"proxies_per_class={self.proxies_per_class}; the absolute count wins "
                "and the per-class multiplier is ignored"
            )
        elif self.num_proxies is None and self.proxies_per_class is None:
            self.proxies_per_class = DEFAULT_PROXIES_PER_CLASS

        if self.labeled_assignment == "class":
            # Eq 8's target has to be locatable from the label alone, so m is
            # k * #Class with class c owning rows [c*k, (c+1)*k).
            if self.num_proxies is not None:
                raise ValueError(
                    "ismlp labeled_assignment='class' sizes Z as proxies_per_class x "
                    "#Class so its blocks line up with the labels; num_proxies cannot "
                    "set it"
                )
            per_class = float(self.proxies_per_class)
            if per_class != int(per_class) or int(per_class) < 1:
                raise ValueError(
                    "ismlp labeled_assignment='class' needs a whole number of proxies "
                    f"per class to block Z by label, got proxies_per_class={per_class:g}"
                )

        self.normalize_proxies = bool(normalize_proxies)
        self.kappa = _validate_positive("kappa", kappa)
        self.kappa_mode = str(kappa_mode)
        if self.kappa_mode not in KAPPA_MODES:
            raise ValueError(f"ismlp kappa_mode must be one of {list(KAPPA_MODES)}")

        # Eq 11 carries Eq 8 unweighted. It gets a weight here because this port
        # also keeps the run's metric loss on the same stream (deviation 1), and
        # because ``learned`` is the only mode that needs it to anchor Z.
        if labeled_weight is None:
            self.labeled_weight = (
                DEFAULT_LABELED_WEIGHT if self.proxy_mode == "learned" else 0.0
            )
        else:
            self.labeled_weight = _validate_non_negative("labeled_weight", labeled_weight)
        if self.labeled_weight > 0 and self.proxy_mode == "supervised_loss":
            raise ValueError(
                "ismlp proxy_mode='supervised_loss' already has Eq 8 in the objective: "
                "Z is the run's proxy criterion's own table, and that criterion is the "
                "labeled alignment. labeled_weight would count it a second time; leave "
                "it unset, or use proxy_mode='learned'"
            )
        if labeled_target_ratio is not None and self.labeled_weight == 0:
            raise ValueError(
                "ismlp labeled_target_ratio has no effect without labeled_weight: the "
                "ratio calibrates that weight, and a zero weight leaves Eq 8 out of the "
                "objective entirely"
            )
        # ``0`` is the opt-out, matching marginal_entropy_target_ratio: a
        # configured weight without a ratio stays fixed rather than calibrated.
        self.labeled_target_ratio = (
            0.0
            if labeled_target_ratio is None
            else _validate_non_negative("labeled_target_ratio", labeled_target_ratio)
        )

        self.marginal_entropy_weight = _validate_non_negative(
            "marginal_entropy_weight",
            marginal_entropy_weight,
        )
        if marginal_entropy_target_ratio is not None and self.marginal_entropy_weight == 0:
            raise ValueError(
                "ismlp marginal_entropy_target_ratio has no effect without "
                "marginal_entropy_weight: the ratio calibrates that weight, and a zero "
                "weight leaves the term out of the objective entirely"
            )
        # ``0`` is the opt-out, so a configured weight without a ratio stays a
        # fixed weight rather than being quietly calibrated.
        self.marginal_entropy_target_ratio = (
            0.0
            if marginal_entropy_target_ratio is None
            else _validate_non_negative(
                "marginal_entropy_target_ratio",
                marginal_entropy_target_ratio,
            )
        )

        self.centroid_method = str(centroid_method)
        if self.centroid_method not in CENTROID_METHODS:
            raise ValueError(f"ismlp centroid_method must be one of {list(CENTROID_METHODS)}")
        self.centroid_iterations = _validate_positive_integer(
            "centroid_iterations",
            centroid_iterations,
        )
        self.centroid_n_init = _validate_positive_integer("centroid_n_init", centroid_n_init)
        self.centroid_backend = str(centroid_backend)
        if self.centroid_backend not in CENTROID_BACKENDS:
            raise ValueError(f"ismlp centroid_backend must be one of {list(CENTROID_BACKENDS)}")
        if centroid_gpu is not None and not isinstance(centroid_gpu, bool):
            raise ValueError("ismlp centroid_gpu must be true, false, or null")
        self.centroid_gpu = centroid_gpu
        if self.centroid_method == "gmm":
            ignored = sorted(
                name
                for name, configured in (
                    ("centroid_n_init", self.centroid_n_init != DEFAULT_CENTROID_N_INIT),
                    ("centroid_backend", self.centroid_backend != "auto"),
                    ("centroid_gpu", self.centroid_gpu is not None),
                )
                if configured
            )
            if ignored:
                logger.warning(
                    f"ismlp centroid_method='gmm' does not read {ignored}: the mixture "
                    "is fitted by scikit-learn with a single k-means++ seeded restart "
                    "and has no FAISS or GPU path. They apply to "
                    "centroid_method='kmeans' only"
                )
        self.fits_centroids = self.proxy_mode == "centroids" or (
            self.proxy_mode == "learned" and self.proxy_init == "centroids"
        )

        # Two alternative ways to size the same unlabeled batch, matching the
        # sibling regularizers: ``unlabeled_batch_size`` is absolute,
        # ``unlabeled_ratio`` is relative to the labeled batch.
        self.unlabeled_batch_size = (
            None if unlabeled_batch_size is None else int(unlabeled_batch_size)
        )
        if self.unlabeled_batch_size is not None and self.unlabeled_batch_size < 2:
            raise ValueError("ismlp unlabeled_batch_size must be at least two")
        self.unlabeled_ratio = (
            None
            if unlabeled_ratio is None
            else _validate_positive("unlabeled_ratio", unlabeled_ratio)
        )
        if self.unlabeled_ratio is None and self.unlabeled_batch_size is None:
            self.unlabeled_ratio = DEFAULT_UNLABELED_RATIO
        elif self.unlabeled_ratio is not None and self.unlabeled_batch_size is not None:
            logger.warning(
                f"ismlp received both unlabeled_batch_size={self.unlabeled_batch_size} "
                f"and unlabeled_ratio={self.unlabeled_ratio}; the absolute batch size "
                "wins and the ratio is ignored"
            )

        # Only this mode needs the trainer to hand compute_loss the run's
        # criterion; the engine reads the flag off the instance. Eq 8 does not
        # need it: uses_joint_forward already brings the labeled embeddings and
        # their labels, which is all deviation 9 reads.
        self.requires_supervised_objective = self.proxy_mode == "supervised_loss"

        if self.regularizer_weight == 0 and self.labeled_weight > 0:
            logger.warning(
                "ismlp regularizer_weight=0 switches the whole regularizer off, Eq 8 "
                "included: the proxy head is never attached and compute_loss is never "
                "reached, so labeled_weight does nothing on its own"
            )

        self.head = None
        self.dataset = None
        self.unlabeled_pool_size = None
        self.labeled_pool_size = None
        self._resolved_num_proxies = None
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_refresh_epoch = None
        # Z's own optimizer; see after_optimizer_step. Built lazily so it does not
        # depend on where make_loader and initialize_state fall relative to each
        # other, and its state is cleared rather than the optimizer rebuilt when a
        # refresh reseeds Z.
        self._proxy_optimizer = None
        self._proxy_optimizer_needs_reset = False
        self._resolved_proxy_lr = None
        self._last_diagnostics = {}
        # Set by compute_loss, read by combine_losses one call later. Weighted
        # separately from regularizer_weight because it exists to push back on
        # what the entropy term does, so scaling the two together would cancel
        # the point of it.
        self._marginal_loss = None
        # Eq 8's term for this step, or None when it is switched off or the
        # batch's similar set came out empty; see deviation 9.
        self._labeled_loss = None

    def validate_run_args(self, args):
        if int(args.batch_size) < 2:
            raise ValueError("ismlp requires batch_size >= 2")
        if not (self.proxy_mode == "learned" and self.proxy_gradient):
            return None
        if self.proxy_lr is not None:
            self._resolved_proxy_lr = self.proxy_lr
        else:
            # Every proxy loss in CLASSIFICATION_LOSSES is optimized at
            # ``--classifier_lr`` rather than ``--lr``, and Z is the same kind of
            # object, so it inherits the same rate by default. The gap is not
            # cosmetic: the shipped defaults are 1.0 and 1e-6.
            self._resolved_proxy_lr = _validate_positive(
                "classifier_lr", getattr(args, "classifier_lr", 1.0)
            )
        model_lr = float(getattr(args, "lr", 0.0) or 0.0)
        logger.info(
            f"ismlp steps Z with its own AdamW at lr={self._resolved_proxy_lr:g}"
            + (f", against a model lr of {model_lr:g}" if model_lr else "")
        )
        return None

    def private_model_module_names(self):
        # The proxy head is the regularizer's alone; the supervised loss never
        # reaches it, so GradNorm and the target-ratio probes must not weigh it
        # as shared. Under proxy_mode='supervised_loss' there is no such module:
        # the proxies belong to the criterion, which is not part of the model.
        return (PROXY_MODULE,) if self.owns_proxies else ()

    def _resolve_num_proxies(self, train_labels_mapper):
        if self.num_proxies is not None:
            return int(self.num_proxies)
        num_classes = len(train_labels_mapper)
        if num_classes < 2:
            raise ValueError(
                "ismlp needs at least two training classes to size Z from "
                "proxies_per_class; set regularizer_params.num_proxies instead"
            )
        resolved = int(round(float(self.proxies_per_class) * num_classes))
        if resolved < 2:
            raise ValueError(
                f"ismlp proxies_per_class={self.proxies_per_class} over {num_classes} "
                f"classes rounds to {resolved} proxies; Eq 6 needs at least two"
            )
        return resolved

    def configure_model(self, student_model, train_dataset, split, train_labels_mapper, device):
        """Attach the proxy head before the optimizer collects parameters."""

        if self.regularizer_weight == 0:
            return None
        if not self.owns_proxies:
            logger.info(
                "Configured ISMLP: proxy_mode='supervised_loss' (Z is the run's "
                f"criterion's proxy table), kappa={self.kappa} ({self.kappa_mode}), "
                f"normalize_proxies={self.normalize_proxies}, "
                f"proxy_gradient={self.proxy_gradient}, "
                f"marginal_entropy_weight={self.marginal_entropy_weight}"
            )
            return None
        if hasattr(student_model, PROXY_MODULE):
            raise RuntimeError("ismlp proxy head is already configured on this model")
        self._resolved_num_proxies = self._resolve_num_proxies(train_labels_mapper)
        head = IsmlpProxyHead(
            feat_dim=student_model.feat_dim,
            num_proxies=self._resolved_num_proxies,
            trainable=self.proxy_mode == "learned",
            normalize=self.normalize_proxies,
        ).to(device)
        student_model.add_module(PROXY_MODULE, head)
        self.head = head
        logger.info(
            "Configured ISMLP proxy head: "
            f"proxy_mode={self.proxy_mode}, m={head.num_proxies}, "
            f"feat_dim={head.feat_dim}, trainable={head.trainable}, "
            f"proxy_gradient={self.proxy_gradient}, "
            f"normalize_proxies={self.normalize_proxies}, "
            f"kappa={self.kappa} ({self.kappa_mode}), "
            + (
                f"init={self.proxy_init}, "
                if self.proxy_mode == "learned"
                else ""
            )
            + (
                f"centroids={self.centroid_method}, "
                if self.fits_centroids
                else ""
            )
            + f"marginal_entropy_weight={self.marginal_entropy_weight}"
            + (
                ""
                if self.marginal_entropy_weight == 0
                else (
                    " (uncalibrated)"
                    if self.marginal_entropy_target_ratio <= 0
                    else f" (calibrated to ratio {self.marginal_entropy_target_ratio:g})"
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
            raise ValueError("ismlp requires at least two unlabeled samples")
        self.unlabeled_pool_size = int(len(unlabeled_positions))
        self.labeled_pool_size = int(len(np.asarray(split.labeled_positions, dtype=np.int64)))
        if (
            self.fits_centroids
            and self._resolved_num_proxies is not None
            and self._resolved_num_proxies > self.unlabeled_pool_size
        ):
            raise ValueError(
                f"ismlp needs at least as many unlabeled samples as proxies to fit Z: "
                f"m={self._resolved_num_proxies} over a pool of "
                f"{self.unlabeled_pool_size}. Lower num_proxies/proxies_per_class, or "
                "use proxy_mode='learned' with proxy_init='random'"
            )
        regularizer_dataset = self.make_regularizer_source_dataset(
            train_dataset,
            use_cache=use_cache,
        )
        self.dataset = UnlabeledSubset(regularizer_dataset, unlabeled_positions, num_views=1)
        # A new fold's pool is a different pool, so its proxies must be refit
        # rather than carried over from the previous one.
        self._last_refresh_epoch = None
        return self.dataset

    def _refresh_proxies(self, model, train_dataset, device, config, seed, start_method, epoch):
        """Refit ``Z`` on the current embedding of the unlabeled pool.

        This is what replaces Sec 4's closed-form ``z_k`` update: the paper
        alternates between the metric and the proxies, and so does a run that
        refreshes here on the ``update_mode`` cadence -- the metric moves for an
        interval, then the proxies are re-solved against it.
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
            desc=f"ISMLP proxy embeddings - epoch {epoch}",
        )
        stats = {}
        centroids = proxy_centroids(
            features,
            num_proxies=self.head.num_proxies,
            seed=seed,
            method=self.centroid_method,
            iterations=self.centroid_iterations,
            n_init=self.centroid_n_init,
            backend=self.centroid_backend,
            use_gpu=self.centroid_gpu,
            stats=stats,
        )
        self.head.set_proxies(torch.from_numpy(centroids))
        if self.head.trainable:
            # Adam keeps its moments keyed by parameter object, so the freshly
            # seeded proxies would otherwise take their first steps under a
            # second moment fitted to the random rows they just replaced. The
            # optimizer holding those moments is this module's own, not the
            # engine's -- see after_optimizer_step -- so the reset is ours too.
            self._proxy_optimizer_needs_reset = True
        norms = np.linalg.norm(centroids, axis=1)
        logger.info(
            f"ISMLP proxies - epoch {epoch}: m={self.head.num_proxies} fitted on "
            f"{len(features)} unlabeled embeddings "
            f"({stats.get('backend', 'unknown')}, {stats.get('seconds', float('nan')):.2f}s); "
            f"centroid norm mean={float(norms.mean()):.4f}, min={float(norms.min()):.4f}"
        )

    def after_optimizer_step(self, student_model, state):
        """Re-project the learned proxies onto the unit sphere.

        ``proxy_vectors`` normalizes at read time, so the objective is invariant
        to each row's norm and its gradient has no radial component: nothing in
        the loss pins ``||z_k||``, so the norm drifts under whatever the
        optimizer does, and *which way* it drifts is not fixed -- decoupled decay
        pulls it down while the accumulated tangential update pushes it out.
        Measured over 400 AdamW steps at ``lr=0.05, weight_decay=0.01``, the mean
        row norm grew to 5.28, which shrinks the tangential step by the same
        factor: an effective learning rate on ``Z`` of 0.19x, configured nowhere.
        Re-projecting costs one normalize per step, leaves the loss bit-for-bit
        unchanged -- the read path already normalized -- and makes the unit norm
        an invariant of the stored parameter rather than a view applied on the
        way out.

        The step above it is the other half. ``Z`` is created frozen so the
        engine's model optimizer, built from ``model.parameters()`` after
        ``configure_model``, never collects it; gradients are switched on once
        that optimizer exists, and this module steps ``Z`` itself at
        ``proxy_lr``. That is what gives ``Z`` a proxy-scale learning rate
        instead of the backbone's, matching how the engine already optimizes
        every ``CLASSIFICATION_LOSSES`` proxy table at ``--classifier_lr``.
        Nothing else clears this gradient, so it is zeroed here too.

        Only this module's own trainable proxies are touched. Under
        ``proxy_mode="centroids"`` the buffer is never stepped, and keeping it
        un-normalized is what makes the refresh log's centroid norms readable.
        Under ``proxy_mode="supervised_loss"`` the table belongs to the
        criterion, whose own loss reads it un-normalized through its distance.
        """

        if not self.owns_proxies:
            return None
        head = getattr(student_model, PROXY_MODULE, None)
        if head is None or not head.trainable:
            return None
        if self.proxy_gradient and head.proxies.grad is not None:
            optimizer = self._ensure_proxy_optimizer(head)
            optimizer.step()
            # The model optimizer never sees this parameter, so nothing else
            # clears its gradient before the next backward accumulates into it.
            optimizer.zero_grad(set_to_none=True)
        if self.normalize_proxies:
            with torch.no_grad():
                head.proxies.copy_(F.normalize(head.proxies, p=2.0, dim=1))
        return None

    def _ensure_proxy_optimizer(self, head):
        """Return the optimizer that steps ``Z``, building or resetting as needed."""

        if self._proxy_optimizer is None:
            head.enable_gradients()
            self._proxy_optimizer = torch.optim.AdamW(
                [head.proxies],
                lr=(
                    DEFAULT_PROXY_LR
                    if self._resolved_proxy_lr is None
                    else self._resolved_proxy_lr
                ),
                # Weight decay would be doing nothing here anyway: the objective
                # is invariant to each row's norm and the step below re-projects
                # onto the sphere, so a decay term is a pull along the one
                # direction that is immediately undone.
                weight_decay=0.0,
            )
            self._proxy_optimizer_needs_reset = False
        elif self._proxy_optimizer_needs_reset:
            self._proxy_optimizer.state.clear()
            self._proxy_optimizer_needs_reset = False
        return self._proxy_optimizer

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
        """Pair the labeled loader with a shuffled unlabeled stream, refitting ``Z``.

        Every step scores its own batch against ``Z``, so the only epoch-scoped
        artifact is the proxy set itself.
        """

        if self.dataset is None:
            raise RuntimeError("ismlp build_dataset must run before make_loader")
        refresh_requested = self.consume_refresh_request()
        if config.update_mode == "every_n_samples" and self.proxy_mode != "centroids":
            raise ValueError(
                f"ismlp proxy_mode={self.proxy_mode!r} has no artifact for "
                "update_mode='every_n_samples' to rebuild: Z is optimized by SGD, not "
                "refitted on a schedule. Use proxy_mode='centroids', or a schedule "
                "without sample-scoped refreshes"
            )
        if self.head is not None:
            needs_initialization = not bool(self.head.proxies_initialized)
            periodic = self.proxy_mode == "centroids" and (
                refresh_requested
                or should_rebuild_on_epoch(
                    config.update_mode,
                    config.update_interval_epochs,
                    epoch,
                    self._last_refresh_epoch,
                )
            )
            if needs_initialization and not self.fits_centroids:
                # proxy_init='random': the constructor's rows are Z, and there is
                # no pool pass to make.
                self.head.mark_initialized()
            elif needs_initialization or periodic:
                self._refresh_proxies(
                    model=model,
                    train_dataset=train_dataset,
                    device=device,
                    config=config,
                    seed=seed,
                    start_method=start_method,
                    epoch=epoch,
                )
                self._last_refresh_epoch = None if epoch is None else int(epoch)

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
            raise ValueError("ismlp needs at least two unlabeled samples per batch")
        if unlabeled_batch_size < requested_batch_size:
            # The batch size sets both the ``batch_std`` temperature and how many
            # rows the marginal term averages over, so silently shrinking it
            # would move the objective with nothing in the log saying so.
            logger.warning(
                f"ismlp requested {requested_batch_size} unlabeled samples per step "
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
                # A ragged final batch would shift both the ``batch_std``
                # temperature and the marginal assignment the guard reads.
                drop_last=True,
                persistent_workers=True,
                pin_memory=True,
                desc="ismlp unlabeled",
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                "ISMLP unlabeled loader: "
                f"pool={len(self.dataset)}, batch_size={unlabeled_batch_size} ({sizing}), "
                f"steps={len(supervised_loader)}"
            )
        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    def _criterion_proxies(self, criterion):
        """Return the run's supervised criterion's proxy table as ``Z``.

        ``proxies`` covers ProxyAnchor, ProxyNCA and this project's
        MixedLabelPropagationProxyLoss, all of which store ``[num_classes,
        embedding_size]``. SoftTriple keeps ``[embedding_size, num_classes *
        centers_per_class]`` under ``fc`` instead, and it is worth supporting
        because its centers-per-class is literally the paper's ``m = k * #Class``.
        """

        if criterion is None:
            raise RuntimeError(
                "ismlp proxy_mode='supervised_loss' needs the run's criterion; the "
                "trainer did not provide one"
            )
        proxies = getattr(criterion, "proxies", None)
        if proxies is None:
            fc = getattr(criterion, "fc", None)
            if fc is not None and fc.dim() == 2:
                proxies = fc.t()
        if proxies is None:
            raise ValueError(
                "ismlp proxy_mode='supervised_loss' needs a proxy-based supervised "
                f"loss, but {type(criterion).__name__} exposes no proxy table. Use "
                "ProxyAnchorLoss, ProxyNCALoss, SoftTripleLoss or "
                "MixedLabelPropagationProxyLoss, or set proxy_mode='centroids'"
            )
        if proxies.dim() != 2 or len(proxies) < 2:
            raise ValueError(
                f"ismlp expected at least two proxy rows from "
                f"{type(criterion).__name__}, got {tuple(proxies.shape)}"
            )
        return proxies

    def _resolve_proxies(self, student_model, supervised_criterion):
        """Return this step's ``Z`` in float32, normalized and detached as configured."""

        if self.owns_proxies:
            head = getattr(student_model, PROXY_MODULE, None)
            if head is None:
                raise RuntimeError(
                    "ismlp requires configure_model to attach the proxy head"
                )
            if not bool(head.proxies_initialized):
                raise RuntimeError(
                    "ismlp proxies were never initialized; make_loader must run before "
                    "the first regularized step"
                )
            if self.proxy_gradient:
                # Safe to do here rather than in configure_model: the model
                # optimizer was built from the frozen parameter long before the
                # first forward reaches this line.
                head.enable_gradients()
            proxies = head.proxy_vectors()
        else:
            proxies = self._criterion_proxies(supervised_criterion).float()
            if self.normalize_proxies:
                proxies = F.normalize(proxies, p=2.0, dim=1)
        return proxies if self.proxy_gradient else proxies.detach()

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
        supervised_is_classification=None,
    ):
        if regularizer_embeddings is None:
            raise ValueError("ismlp requires the joint labeled/unlabeled forward context")
        if len(regularizer_embeddings) < 2:
            raise ValueError("ismlp needs at least two unlabeled embeddings per batch")

        proxies = self._resolve_proxies(student_model, supervised_criterion)
        if proxies.shape[1] != regularizer_embeddings.shape[1]:
            raise ValueError(
                f"ismlp proxy dimension {proxies.shape[1]} does not match the "
                f"{regularizer_embeddings.shape[1]}-dimensional embedding"
            )

        # Autocast would run the Gram matmul in bfloat16, and the Gram form of a
        # squared distance cancels two large terms against each other - the last
        # place that can afford three decimal digits. Everything after this is a
        # softmax over those distances and follows them in float32.
        with torch.autocast(device_type=torch.device(device).type, enabled=False):
            embeddings = regularizer_embeddings.float()
            proxies = proxies.to(embeddings.device)
            squared_distances = cross_squared_distances(embeddings, proxies)
            kappa = inverse_temperature(self.kappa, self.kappa_mode, squared_distances)
            log_q = assignment_log_probabilities(squared_distances, kappa)
            q = log_q.exp()
            # Eq 9: the mean over U of H(q_i), in nats.
            row_entropy = -(q * log_q).sum(dim=1)
            total = row_entropy.mean()

            marginal_entropy = None
            if self.marginal_entropy_weight > 0 or self.collect_batch_diagnostics:
                # log of the column mean of q, straight from the logs: the
                # averaged assignment of a distant proxy underflows to zero long
                # before its log does, and this term is exactly about those.
                log_marginal = torch.logsumexp(log_q, dim=0) - math.log(len(log_q))
                marginal_entropy = -(log_marginal.exp() * log_marginal).sum()

            # Eq 8, on the labeled half of the same joint forward. It shares this
            # block for the same reason Eq 9 needs it: the Gram form of the
            # squared distance cancels two large terms against each other.
            labeled_loss = None
            labeled_diagnostics = {}
            if self.labeled_weight > 0:
                labeled_loss, labeled_diagnostics = self._labeled_term(
                    supervised_embeddings,
                    supervised_labels,
                    proxies,
                )

        if not torch.isfinite(total):
            raise FloatingPointError("ismlp produced a non-finite regularization loss")
        if marginal_entropy is not None and self.marginal_entropy_weight > 0:
            # Reported as a deficit from the uniform maximum so the logged number
            # is non-negative and reads as zero when every proxy is used equally;
            # the gradient is the same as that of -H(mean q).
            self._marginal_loss = (
                math.log(proxies.shape[0]) - marginal_entropy
            ).clamp_min(0.0)
            if not torch.isfinite(self._marginal_loss):
                raise FloatingPointError("ismlp produced a non-finite marginal entropy term")
        else:
            self._marginal_loss = None

        self._labeled_loss = labeled_loss
        if self._labeled_loss is not None and not torch.isfinite(self._labeled_loss):
            raise FloatingPointError("ismlp produced a non-finite Eq 8 labeled term")

        if self.collect_batch_diagnostics:
            self._last_diagnostics = self._diagnostics(
                q, row_entropy, squared_distances, marginal_entropy, kappa, proxies
            )
            self._last_diagnostics.update(labeled_diagnostics)
        else:
            self._last_diagnostics = {}
        return total

    def _labeled_term(self, supervised_embeddings, supervised_labels, proxies):
        """Return Eq 8 as a minimizable loss, plus its diagnostics.

        Eq 8 is written as a maximization of the mean log-likelihood over ``S``;
        this returns its negation, so descending it is what the paper's ``max``
        asks for. ``S`` and ``p(x_j)`` are deviation 9's readings.

        Returns ``(None, ...)`` when the batch has no similar pair at all. That
        is not a zero: Eq 8 divides by ``|S|``, so an empty ``S`` leaves it
        undefined, and a term reported as zero would be averaged into the
        target-ratio probe as if it had been measured.
        """

        if supervised_embeddings is None or supervised_labels is None:
            raise ValueError(
                "ismlp labeled_weight needs the joint forward's labeled embeddings "
                "and labels; the trainer did not provide them"
            )
        labeled = supervised_embeddings.float()
        labels = supervised_labels.to(device=labeled.device, dtype=torch.long).reshape(-1)
        if len(labels) != len(labeled):
            raise ValueError(
                f"ismlp got {len(labeled)} labeled embeddings and {len(labels)} labels"
            )
        if len(labeled) < 2:
            return None, {"train/ismlp/labeled_pairs": 0.0}
        if labeled.shape[1] != proxies.shape[1]:
            raise ValueError(
                f"ismlp proxy dimension {proxies.shape[1]} does not match the "
                f"{labeled.shape[1]}-dimensional labeled embedding"
            )

        squared_distances = cross_squared_distances(labeled, proxies.to(labeled.device))
        # Same mechanism as Eq 9's, read on the distances this term scores; see
        # deviation 9.
        kappa = inverse_temperature(self.kappa, self.kappa_mode, squared_distances)
        log_q = assignment_log_probabilities(squared_distances, kappa)
        # Eq 5. Detached because an argmin has no gradient to carry anyway, which
        # is also what keeps x_j out of the objective: it contributes an index.
        targets = labeled_target_proxies(
            squared_distances.detach(),
            labels,
            self.labeled_assignment,
            self.proxies_per_class,
        )
        counts = labeled_pair_counts(targets, labels, proxies.shape[0], log_q.dtype)
        pair_count = counts.sum()
        if float(pair_count) <= 0.0:
            return None, {"train/ismlp/labeled_pairs": 0.0}
        loss = -(log_q * counts).sum() / pair_count

        diagnostics = {}
        if self.collect_batch_diagnostics:
            with torch.no_grad():
                num_proxies = int(proxies.shape[0])
                diagnostics = {
                    "train/ismlp/labeled_loss": loss,
                    # A property of the batch, not a constant; deviation 9.
                    "train/ismlp/labeled_pairs": pair_count,
                    # Mean over S of q at the pair's target proxy. This is what
                    # Eq 8 pushes to one, and it reads on the same 0-1 scale as
                    # max_probability does for the unlabeled term.
                    "train/ismlp/labeled_target_probability": (
                        (counts * log_q.exp()).sum() / pair_count
                    ),
                    "train/ismlp/labeled_kappa": kappa,
                    # How much of Z the labeled batch actually claims. Normalized
                    # like proxy_occupancy, by the attainable maximum.
                    "train/ismlp/labeled_target_occupancy": (
                        float(len(torch.unique(targets)))
                        / float(min(len(targets), num_proxies))
                    ),
                }
        return loss, diagnostics

    def _diagnostics(self, q, row_entropy, squared_distances, marginal_entropy, kappa, proxies):
        with torch.no_grad():
            num_proxies = int(proxies.shape[0])
            uniform_entropy = math.log(num_proxies)
            assignments = q.argmax(dim=1)
            diagnostics = {
                # The quantity Eq 9 descends, and the same quantity as a fraction
                # of its maximum - which is what says whether the term is doing
                # anything at all at this temperature.
                "train/ismlp/assignment_entropy": row_entropy.mean(),
                "train/ismlp/assignment_entropy_normalized": (
                    row_entropy.mean() / uniform_entropy
                ),
                "train/ismlp/max_probability": q.max(dim=1).values.mean(),
                # The collapse Eq 9 alone is minimized by. Normalized by the
                # attainable maximum, not by m: a batch of n rows can occupy at
                # most n proxies, so dividing by m would cap a perfectly healthy
                # step at n/m -- 0.33 at the shipped batch of 128 against a
                # 392-proxy Cars196 fold -- and read as collapse. Even against
                # min(n, m) the healthy value sits below 1 by the birthday
                # collisions of n draws over m proxies (~0.85 at n=128, m=392);
                # what marks the collapse is the approach to 1/n.
                "train/ismlp/proxy_occupancy": (
                    float(len(torch.unique(assignments)))
                    / float(min(len(q), num_proxies))
                ),
                "train/ismlp/nearest_proxy_distance": (
                    squared_distances.min(dim=1).values.mean()
                ),
                "train/ismlp/kappa": kappa,
                "train/ismlp/num_proxies": float(num_proxies),
                # What actually arrived in this step, as opposed to what the
                # loader was asked for.
                "train/ismlp/unlabeled_batch_size": float(len(q)),
            }
            if marginal_entropy is not None:
                diagnostics["train/ismlp/marginal_entropy"] = marginal_entropy
                diagnostics["train/ismlp/marginal_entropy_normalized"] = (
                    marginal_entropy / uniform_entropy
                )
            if self._marginal_loss is not None:
                diagnostics["train/ismlp/marginal_loss"] = self._marginal_loss
        return diagnostics

    def combine_losses(self, supervised_loss, regularization_loss):
        """Add Eq 8 and the deviation-7 marginal-entropy guard, each at its own weight.

        Both carry separate weights rather than riding inside
        ``regularizer_weight``. The guard because it exists to push back on what
        the entropy term does, so scaling the two together would cancel the point
        of it; Eq 8 because ``regularizer_weight`` is Eq 11's ``lambda`` on the
        entropy alone. Folding either in would also put a term with a different
        gradient scale inside the weight that ``regularizer_target_ratio``
        calibrates from measured gradients.
        """

        combined = super().combine_losses(supervised_loss, regularization_loss)
        if self._labeled_loss is not None:
            combined = combined + self.labeled_weight * self._labeled_loss
        if self._marginal_loss is not None:
            combined = combined + self.marginal_entropy_weight * self._marginal_loss
        return combined

    def extra_loss_components(self):
        components = {}
        if self._labeled_loss is not None:
            components[LABELED_COMPONENT] = (self._labeled_loss, self.labeled_weight)
        if self._marginal_loss is not None:
            components[MARGINAL_COMPONENT] = (
                self._marginal_loss,
                self.marginal_entropy_weight,
            )
        return components

    def extra_target_ratios(self):
        """Calibrate the side weights from measured gradients.

        None of these three terms are commensurable. Eq 9's entropy is a per-row
        average whose gradient reaches every unlabeled embedding; the guard's is
        one scalar over the batch mean; Eq 8's is a log-likelihood over ``|S|``
        pairs on the *labeled* embeddings and on ``Z``. A hand-picked weight is a
        guess at those ratios; a ratio makes the configured weight a probe,
        exactly as ``regularizer_target_ratio`` does for ``regularizer_weight``.
        """

        ratios = {}
        if self.labeled_target_ratio > 0:
            ratios[LABELED_COMPONENT] = self.labeled_target_ratio
        if self.marginal_entropy_target_ratio > 0:
            ratios[MARGINAL_COMPONENT] = self.marginal_entropy_target_ratio
        return ratios

    def clear_extra_target_ratios(self, keep=()):
        # A name in ``keep`` is one GradNorm is not balancing, so nothing
        # supersedes its calibration and the ratio is still the honest way to
        # size it.
        if LABELED_COMPONENT not in keep:
            self.labeled_target_ratio = 0.0
        if MARGINAL_COMPONENT not in keep:
            self.marginal_entropy_target_ratio = 0.0

    def grad_norm_extra_components(self):
        """Let GradNorm learn the side weights when they are in the objective."""

        components = {}
        if self.labeled_weight > 0:
            components[LABELED_COMPONENT] = self.labeled_weight
        if self.marginal_entropy_weight > 0:
            components[MARGINAL_COMPONENT] = self.marginal_entropy_weight
        return components

    def calibratable_component_loss(self, name):
        if name == LABELED_COMPONENT:
            return self._labeled_loss
        if name == MARGINAL_COMPONENT:
            return self._marginal_loss
        return None

    def calibrated_component_weight(self, name):
        if name == LABELED_COMPONENT:
            return self.labeled_weight
        if name == MARGINAL_COMPONENT:
            return self.marginal_entropy_weight
        raise KeyError(name)

    def apply_calibrated_component_weight(self, name, weight):
        if name == LABELED_COMPONENT:
            self.labeled_weight = float(weight)
        elif name == MARGINAL_COMPONENT:
            self.marginal_entropy_weight = float(weight)
        else:
            raise KeyError(name)

    def batch_diagnostics(self):
        return {
            name: float(value.detach().item()) if torch.is_tensor(value) else float(value)
            for name, value in self._last_diagnostics.items()
        }
