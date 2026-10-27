"""Graph and embedding structure over the epochs of one fold.

A run already records what its model *scores* every epoch. This records what
its representation and its neighbourhood graph *look like* while it gets there:
one bundle of artifacts at epoch -1, before any optimizer step, and then every
``--track_epoch_interval`` epochs. The baseline is the point of the exercise --
without it a curve cannot say whether the SSL phase built structure or merely
failed to destroy the structure the pretrained backbone already had.

Three artifact kinds, requested by name with ``--track_epoch_artifacts``:

``graph``
    The kNN graph over the SSL pool (labeled positions first, then unlabeled),
    rebuilt from the current embeddings with the method's own graph parameters
    and handed to :mod:`training.ssl.graph_diagnostics`, which already knows how
    to draw a graph on its own 2-D layout and how to measure purity, degree,
    components and spectral gap. The graph is built here rather than captured
    from the regularizer on purpose: at epoch -1 no method has built one yet, a
    method's own rebuild schedule is not the interval being asked about, and a
    supervised baseline builds none at all -- yet "what would the graph have
    looked like" is exactly the comparison a supervised row in the table is for.
    It is read-only. Nothing it computes is fed back into training.

``embeddings``
    The same pool's embeddings, projected with the chosen visualizer, plus the
    geometry that says whether a representation is collapsing: effective rank,
    alignment and uniformity, and the intra/inter-class cosine gap.

``geometry``
    What the representation looks like on each split, as Roth et al. (2020) and
    Wang and Isola (2020) measure it: the singular spectrum of the normalized
    embeddings with its effective rank and spectral decay (with and without the
    dominant direction), intra/inter-class distances and their ratio, alignment
    and uniformity. Measured separately on the labeled and unlabeled part of the
    training pool, the fold's validation classes and the test classes, and once
    per fold on the frozen backbone features the head receives, so every curve
    has the untrained representation as its reference. One JSON line per split
    and epoch in ``geometry.jsonl``, including the leading spectrum shares.

``test``
    D_test evaluated off the selection path, so the test curve can be read
    against the structure at the same epochs. It is never consulted by model
    selection, early stopping, or Optuna -- it is written and nothing reads it
    back. It is forced off for HPO trials, where a test number per epoch would
    be an invitation to select on it.

Every tracked epoch appends one row to ``epoch_tracking/epoch_metrics.csv``, so
a fold's whole history is one table; the per-epoch directories beside it hold
the plots and the full JSON summaries.

Tracking costs an extra embedding pass over the pool per tracked epoch, plus a
graph build and a 2-D projection. The projection dominates, which is why
``--track_epoch_max_samples`` caps the pool and why the whole feature is off
unless an interval is given. The ``geometry`` rows are the exception to that
cost: they are CPU work on embeddings the epoch has already produced, so
``--track_epoch_workers`` processes compute them while training goes on and the
fold only waits for the last of them when it closes.

The files are meant to be read while the run is still going: the Trial Atlas
dev server (``evaluate/live_tracking.py``) polls ``epoch_tracking/`` and shows
an epoch as soon as its 2-D projection exists, before the graph and class
evidence of that epoch are done. Layout CSVs and npz archives are therefore
renamed into place once complete, and ``status.json`` says which epoch and
which stage the tracker is in, so a viewer can tell "still computing" from
"the run stopped".

Torch is never imported at module level, so :mod:`training.cli` can name these
options without pulling the training stack in.
"""

import contextlib
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from loguru import logger

TRACK_EPOCH_NONE = "none"
TRACK_EPOCH_GRAPH = "graph"
TRACK_EPOCH_EMBEDDINGS = "embeddings"
TRACK_EPOCH_TEST = "test"
TRACK_EPOCH_GEOMETRY = "geometry"
TRACK_EPOCH_ARTIFACTS = (
    TRACK_EPOCH_NONE,
    TRACK_EPOCH_GRAPH,
    TRACK_EPOCH_EMBEDDINGS,
    TRACK_EPOCH_TEST,
    TRACK_EPOCH_GEOMETRY,
)
# Artifacts that need the SSL pool embedded every tracked epoch.
_POOL_ARTIFACTS = {TRACK_EPOCH_GRAPH, TRACK_EPOCH_EMBEDDINGS, TRACK_EPOCH_GEOMETRY}

GEOMETRY_NAME = "geometry.jsonl"
# Splits the geometry is measured on. ``train`` is the whole tracked training pool
# (Roth et al. measure rho on the embedded training data); labeled and unlabeled
# are its two halves; validation and test hold classes the head never trained on.
GEOMETRY_SPLITS = ("train", "labeled", "unlabeled", "validation", "test")
# Leading spectrum shares kept per row, enough to draw the decay with and
# without the dominant direction.
GEOMETRY_SPECTRUM_POINTS = 128
# Roth et al. (2020), supp. E / Fig. 11 plot the first 35 per-class singular values.
CLASS_SPECTRUM_POINTS = 35

# Layouts are the graph diagnostics' own, so a tracked graph and a tracked
# embedding space can be drawn with the same projection and read together.
TRACK_EPOCH_LAYOUTS = ("pca", "pacmap", "tsne")
DEFAULT_TRACK_EPOCH_LAYOUT = "tsne"

# The epoch before the first optimizer step. Negative rather than 0 so it can
# never be confused with "after the first epoch" in a plot or a join.
BASELINE_EPOCH = -1

# Enough nodes for a purity or component estimate to mean something, few enough
# that PaCMAP/t-SNE stay a small fraction of an epoch on a fold of In-Shop.
DEFAULT_TRACK_EPOCH_MAX_SAMPLES = 4000

# Worker processes that compute the geometry rows behind the training loop. An
# uncapped Semi-iNat epoch costs ~24 s of single-threaded geometry against ~6.5 s
# of training, so a few workers are what keeps training from waiting on it.
# 0 computes the rows in the training process, as before.
DEFAULT_TRACK_EPOCH_WORKERS = 4

# Epochs whose geometry rows may be queued at once. A queued epoch holds its pool,
# validation and test embeddings until a worker has used them, so a run whose
# workers fall behind would grow by a full set of them every epoch; past this
# the training loop waits for the oldest epoch instead.
MAX_QUEUED_GEOMETRY_EPOCHS = 2

# How the capped pool is drawn. ``images`` (the default, and every run before
# 2026-09-29): single images at the pool's labeled/unlabeled ratio. ``classes``:
# whole classes, the way the validation and test splits are drawn. With a few
# images per class (In-Shop) ``images`` leaves about half the tracked classes with
# one image, on which pi_intra is not defined; ``classes`` leaves none.
TRACK_EPOCH_POOL_SAMPLINGS = ("images", "classes")
DEFAULT_TRACK_EPOCH_POOL_SAMPLING = "images"

# Pairs drawn for the sampled geometry. The full N^2 cosine matrix is what this
# avoids; 200k pairs put the standard error on a mean cosine below 0.005.
DEFAULT_STRUCTURE_PAIRS = 200_000
# Pairs gathered at once. All 200k pairs of 768-d float64 rows at once cost
# ~2.5 GB per call, which multiplied by the geometry workers of every run is what
# a server runs out of; each pair's cosine is computed the same way either way.
STRUCTURE_PAIR_CHUNK = 16_384

OUTPUT_DIR_NAME = "epoch_tracking"
METRICS_CSV_NAME = "epoch_metrics.csv"
# Where the tracker is right now: which epoch, which stage, running or finished.
STATUS_NAME = "status.json"
# Every edge of the tracked graph; graph_sampled_edges.csv keeps only a sample.
GRAPH_EDGES_NAME = "graph_edges.npz"
# How label propagation and Hoffer's entropy term would read each tracked epoch.
CLASS_EVIDENCE_NAME = "class_evidence.npz"
CLASS_EVIDENCE_TOP_K = 5
# How the labels reached each node: every step of the same diffusion, saved
# beside the answer it converges to.
PROPAGATION_NAME = "label_propagation.npz"
# Steps of that series. A cosine kNN graph over a few thousand nodes has a
# diameter of a handful of hops, so the front has long arrived by here. The
# magnitudes have not converged (alpha^16 is 0.85 at alpha=0.99), which is why
# the saved series is each node's share of its own final mass, not the mass.
PROPAGATION_STEPS = 16
# Incoming contributors kept per node, strongest first.
PROPAGATION_TOP_SOURCES = 3
# Hoffer draws one labeled reference per class per step; this many draws average
# that randomness out of a per-epoch picture.
HOFFER_REFERENCE_DRAWS = 16
# The network's own class prediction: a softmax over cosine similarity to class
# prototypes (the mean labeled embedding per class) at this temperature, the
# retrieval analogue of a classifier head. Its entropy is the conditional
# entropy H(y|x) Lee (2013) tracks to show pseudo-labeling acting as entropy
# regularization.
PROTOTYPE_TEMPERATURE = 0.05
# Accuracy among the most confident unlabeled predictions, as shares of them:
# whether a confidence weight ranks correctness (Iscen et al., Sec 5 / Fig 5).
CONFIDENT_FRACTIONS = (10, 25, 50)
# Hoffer's entropy depends on distance_scale as much as on the embedding: the same
# space reads as sharper under a larger scale. A fixed evaluation scale (the
# regularizer's default) makes entropy comparable across runs whose tuned scales
# differ, next to the reading at the run's own scale.
HOFFER_FIXED_DISTANCE_SCALE = 1.0
# "Confident": Hoffer's certainty (1 - normalized entropy) or Iscen's weight omega
# (that certainty over the pool's largest) of at least one half.
CONFIDENT_CERTAINTY = 0.5

# Graph parameters are read from the method that owns them when it has them, so
# a tracked graph is the method's graph rather than a second opinion about it.
DEFAULT_TRACKING_N_NEIGHBORS = 10
DEFAULT_TRACKING_GAMMA = 1.0

_EPOCH_COLUMNS = ("epoch", "tracked_seconds")
_TEST_COLUMNS = (
    "test_precision_at_1",
    "test_mean_average_precision_at_r",
)
_EMBEDDING_COLUMNS = (
    "embedding_count",
    "embedding_dim",
    "embedding_norm_mean",
    "embedding_norm_std",
    "embedding_effective_rank",
    "embedding_top1_variance_share",
    "embedding_intra_class_cosine",
    "embedding_inter_class_cosine",
    "embedding_class_separation",
    "embedding_alignment",
    "embedding_uniformity",
)
_GRAPH_COLUMNS = (
    "graph_node_count",
    "graph_undirected_edge_count",
    "graph_mean_degree",
    "graph_isolated_node_count",
    "graph_component_count",
    "graph_nodes_without_labeled_component",
    "graph_edge_purity",
    "graph_weighted_edge_purity",
)
_CLASS_EVIDENCE_COLUMNS = (
    "lp_unlabeled_accuracy",
    "lp_unlabeled_certainty_mean",
    "lp_weight_auc",
    "lp_confident_wrong_share",
    "hoffer_unlabeled_accuracy",
    "hoffer_unlabeled_certainty_mean",
    "hoffer_unlabeled_entropy_mean",
    "hoffer_fixed_scale_unlabeled_accuracy",
    "hoffer_fixed_scale_unlabeled_entropy_mean",
    "hoffer_assignment_entropy",
    "hoffer_assignment_gini",
    "hoffer_assignment_max_share",
    "lp_assignment_entropy",
    "lp_assignment_gini",
    "lp_assignment_max_share",
    "hoffer_certainty_auc",
    "hoffer_confident_wrong_share",
    *(f"{prefix}_top{share}_accuracy" for prefix in ("lp", "hoffer") for share in CONFIDENT_FRACTIONS),
    "prototype_unlabeled_accuracy",
    "prototype_certainty_auc",
    *(f"prototype_top{share}_accuracy" for share in CONFIDENT_FRACTIONS),
    "conditional_entropy_labeled",
    "conditional_entropy_unlabeled",
    "conditional_entropy_test",
)
# The same diffusion read step by step (label_propagation.npz).
_PROPAGATION_COLUMNS = (
    "lp_reached_share",
    "lp_mean_hops",
    "lp_tree_edge_purity",
    "lp_source_accuracy",
    "lp_influence_share",
    "lp_influence_purity",
    "lp_trace_agreement",
    "lp_trace_steps",
)
# summary["overview"] already names these; the CSV only renames them so a row
# reads as one namespace.
_GRAPH_OVERVIEW_COLUMNS = {
    "graph_node_count": "node_count",
    "graph_undirected_edge_count": "undirected_edge_count",
    "graph_mean_degree": "mean_degree",
    "graph_isolated_node_count": "isolated_node_count",
    "graph_component_count": "component_count",
    "graph_nodes_without_labeled_component": "nodes_in_components_without_labeled_node",
    "graph_edge_purity": "oracle_edge_purity",
    "graph_weighted_edge_purity": "oracle_weighted_edge_purity",
}


def normalize_absorbing_choices(value, option, choices, none_choice, noun):
    """Return the requested choices as an ordered tuple without repeats.

    Accepts a bare string, as configs that predate several choices recorded
    one. ``none_choice`` is absorbing: it means "nothing", so combining it with
    a real choice is a contradiction rather than a request for both.
    """

    if value is None:
        return ()
    selected = (value,) if isinstance(value, str) else tuple(value)
    unknown = [item for item in selected if item not in choices]
    if unknown:
        raise ValueError(f"{option} must be one or more of {choices}: {unknown}")
    if none_choice in selected:
        if len(set(selected)) > 1:
            raise ValueError(
                f"{option}={list(selected)} combines {none_choice!r} with {noun}; "
                "request one or the other"
            )
        return ()
    return tuple(dict.fromkeys(selected))


def normalize_track_epoch_artifacts(value):
    return normalize_absorbing_choices(
        value, "track_epoch_artifacts", TRACK_EPOCH_ARTIFACTS, TRACK_EPOCH_NONE, "an artifact"
    )


def get_track_epoch_artifacts(args):
    return normalize_track_epoch_artifacts(
        getattr(args, "track_epoch_artifacts", TRACK_EPOCH_NONE)
    )


def get_track_epoch_interval(args):
    """Return the epoch interval, or None when tracking is off."""

    interval = getattr(args, "track_epoch_interval", None)
    if interval is None:
        return None
    interval = int(interval)
    return interval if interval > 0 else None


def get_track_epoch_max_samples(args):
    value = getattr(args, "track_epoch_max_samples", None)
    if value is None:
        return DEFAULT_TRACK_EPOCH_MAX_SAMPLES
    value = int(value)
    return value if value > 0 else None


def get_track_epoch_workers(args):
    value = getattr(args, "track_epoch_workers", None)
    if value is None:
        return DEFAULT_TRACK_EPOCH_WORKERS
    value = int(value)
    if value < 0:
        raise ValueError(f"track_epoch_workers must be >= 0: {value}")
    return value


def tracks_pool(args):
    """Whether tracking embeds the training pool (every artifact but ``test`` does)."""

    return get_track_epoch_interval(args) is not None and bool(
        _POOL_ARTIFACTS & set(get_track_epoch_artifacts(args))
    )


def get_track_epoch_pool_sampling(args):
    value = getattr(args, "track_epoch_pool_sampling", None) or DEFAULT_TRACK_EPOCH_POOL_SAMPLING
    if value not in TRACK_EPOCH_POOL_SAMPLINGS:
        raise ValueError(
            f"track_epoch_pool_sampling must be one of {TRACK_EPOCH_POOL_SAMPLINGS}: {value!r}"
        )
    return value


def validate_track_epoch_args(args):
    """Reject a tracking request that would silently produce nothing."""

    artifacts = get_track_epoch_artifacts(args)
    interval = get_track_epoch_interval(args)
    if artifacts and interval is None:
        raise ValueError(
            f"track_epoch_artifacts={list(artifacts)} needs --track_epoch_interval; without an "
            "interval there is no epoch to track"
        )
    if interval is not None and not artifacts:
        raise ValueError(
            "--track_epoch_interval is set but track_epoch_artifacts is 'none'; name what to "
            f"track with --track_epoch_artifacts {{{','.join(TRACK_EPOCH_ARTIFACTS[1:])}}}"
        )
    layout = getattr(args, "track_epoch_layout", DEFAULT_TRACK_EPOCH_LAYOUT)
    if layout not in TRACK_EPOCH_LAYOUTS:
        raise ValueError(f"track_epoch_layout must be one of {TRACK_EPOCH_LAYOUTS}: {layout!r}")
    get_track_epoch_pool_sampling(args)
    if TRACK_EPOCH_TEST in artifacts and not bool(getattr(args, "evaluate_test", False)):
        # A null result is worse than a failed launch: the curve would simply be
        # missing its test column with nothing in the log to say why.
        raise ValueError(
            "track_epoch_artifacts includes 'test' but this run does not evaluate the test "
            "set; pass --evaluate_test, or drop 'test' from --track_epoch_artifacts"
        )
    return artifacts


def is_tracked_epoch(epoch, interval, epochs):
    """Return whether one epoch gets a bundle.

    The baseline and every ``interval``-th epoch, plus the last epoch, so a
    curve always ends on the model the run's own metrics describe rather than
    stopping wherever the interval happened to land.
    """

    if interval is None or interval <= 0:
        return False
    epoch = int(epoch)
    if epoch <= BASELINE_EPOCH:
        return True
    if epoch % int(interval) == 0:
        return True
    return epoch == int(epochs) - 1


def epoch_slug(epoch):
    epoch = int(epoch)
    return f"epoch_{epoch:04d}" if epoch >= 0 else f"epoch_m{abs(epoch):03d}"


def sample_same_class_partners(labels, rng):
    """Pair each sample with another of its class, or mark it unpairable.

    Drawn per sample rather than from uniform random pairs because the fraction
    of same-class pairs in a uniform sample is ~1/C: on In-Shop's thousands of
    classes that leaves too few pairs to estimate alignment from.
    """

    labels = np.asarray(labels).reshape(-1)
    order = np.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    _, starts, counts = np.unique(sorted_labels, return_index=True, return_counts=True)
    starts_per_element = np.repeat(starts, counts)
    counts_per_element = np.repeat(counts, counts)
    within_block = np.arange(len(labels), dtype=np.int64) - starts_per_element
    # Draw among the block's other members, then step over the sample itself so
    # the partner is uniform over the class without ever being the anchor. A
    # class of one has no other member: it would step off the end of its own
    # block, so it points at itself and the pairable mask below drops it.
    draw = rng.integers(0, np.maximum(counts_per_element - 1, 1))
    partner_within = np.where(
        counts_per_element >= 2,
        draw + (draw >= within_block),
        within_block,
    )
    partners = np.empty(len(labels), dtype=np.int64)
    partners[order] = order[starts_per_element + partner_within]
    pairable = np.empty(len(labels), dtype=bool)
    pairable[order] = counts_per_element >= 2
    return partners, pairable


def _paired_cosines(unit, left, right, chunk=STRUCTURE_PAIR_CHUNK):
    """Row-wise cosines of ``unit[left]`` and ``unit[right]``, ``chunk`` pairs at a time."""

    cosines = np.empty(len(left), dtype=np.result_type(unit.dtype, np.float64))
    for start in range(0, len(left), chunk):
        stop = start + chunk
        cosines[start:stop] = np.einsum("ij,ij->i", unit[left[start:stop]], unit[right[start:stop]])
    return cosines


def embedding_structure_metrics(
    embeddings,
    labels,
    *,
    seed=0,
    max_pairs=DEFAULT_STRUCTURE_PAIRS,
    singular_values=None,
):
    """Measure whether an embedding space is spreading out or collapsing.

    ``effective_rank`` is Roy and Vetterli's: the exponential of the entropy of
    the normalized singular-value spectrum, so a space whose variance has
    fallen into one direction reports ~1 however many dimensions it nominally
    has. ``alignment`` and ``uniformity`` are Wang and Isola's, computed on the
    unit sphere; ``class_separation`` is the plain intra-minus-inter cosine gap,
    which is the one a reader of a metric-learning table already has intuition
    for. All three answer the same question from different directions, and a
    collapse shows up in all three at once.

    ``singular_values`` takes :func:`singular_spectrum` of the same embeddings
    when the caller already has it; it is the same decomposition of the same
    matrix, so the metrics do not change.
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim != 2 or len(embeddings) == 0:
        raise ValueError("embedding structure metrics need a non-empty embedding matrix")
    labels = np.asarray(labels).reshape(-1)
    if len(labels) != len(embeddings):
        raise ValueError("embedding structure metrics need one label per embedding")

    norms = np.linalg.norm(embeddings, axis=1)
    unit = embeddings / np.clip(norms, 1e-12, None)[:, None]
    metrics = {
        "embedding_count": int(len(embeddings)),
        "embedding_dim": int(embeddings.shape[1]),
        "embedding_norm_mean": float(norms.mean()),
        "embedding_norm_std": float(norms.std()),
    }

    if singular_values is None:
        centered = unit - unit.mean(axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
    power = np.square(np.asarray(singular_values, dtype=np.float64))
    total_power = float(power.sum())
    if total_power > 0.0:
        shares = power / total_power
        positive = shares[shares > 0.0]
        entropy = float(-(positive * np.log(positive)).sum())
        metrics["embedding_effective_rank"] = float(np.exp(entropy))
        metrics["embedding_top1_variance_share"] = float(shares[0])
    else:
        # Every sample at the same point: one direction, no variance to share.
        metrics["embedding_effective_rank"] = 1.0
        metrics["embedding_top1_variance_share"] = 1.0

    rng = np.random.default_rng(int(seed))
    count = len(unit)
    pair_count = int(min(max_pairs, max(count * 4, 1)))
    left = rng.integers(0, count, size=pair_count)
    right = rng.integers(0, count, size=pair_count)
    distinct = left != right
    left, right = left[distinct], right[distinct]
    if len(left):
        random_cosine = _paired_cosines(unit, left, right)
        different_class = labels[left] != labels[right]
        if np.any(different_class):
            metrics["embedding_inter_class_cosine"] = float(
                random_cosine[different_class].mean()
            )
        # Uniformity reads the whole sampled cloud, same-class pairs included,
        # because it is a statement about how the sphere is covered.
        squared_distance = np.clip(2.0 - 2.0 * random_cosine, 0.0, None)
        metrics["embedding_uniformity"] = float(
            np.log(np.mean(np.exp(-2.0 * squared_distance)))
        )

    partners, pairable = sample_same_class_partners(labels, rng)
    if np.any(pairable):
        anchors = np.flatnonzero(pairable)
        if len(anchors) > max_pairs:
            anchors = rng.choice(anchors, size=max_pairs, replace=False)
        same_cosine = _paired_cosines(unit, anchors, partners[anchors])
        metrics["embedding_intra_class_cosine"] = float(same_cosine.mean())
        metrics["embedding_alignment"] = float(
            np.clip(2.0 - 2.0 * same_cosine, 0.0, None).mean()
        )
        if "embedding_inter_class_cosine" in metrics:
            metrics["embedding_class_separation"] = float(
                metrics["embedding_intra_class_cosine"]
                - metrics["embedding_inter_class_cosine"]
            )
    return metrics


def singular_spectrum(embeddings):
    """Singular values of the centered, L2-normalized embeddings, largest first."""

    embeddings = np.asarray(embeddings, dtype=np.float64)
    unit = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    centered = unit - unit.mean(axis=0, keepdims=True)
    return np.linalg.svd(centered, compute_uv=False)


def spectral_decay(singular_values, *, drop_first=False):
    """Roth et al. (2020)'s rho: KL(uniform || normalized singular spectrum).

    Larger means a faster decay, i.e. a representation compressed into fewer
    directions. ``drop_first`` repeats it without the dominant direction.
    """

    values = np.asarray(singular_values, dtype=np.float64)
    if drop_first:
        values = values[1:]
    values = values[values > 0]
    if len(values) == 0:
        return float("nan")
    shares = values / values.sum()
    uniform = np.full_like(shares, 1.0 / len(shares))
    return float(np.sum(uniform * np.log(uniform / shares)))


def class_geometry(embeddings, labels):
    """Intra-class spread and centroid separation on the unit sphere.

    ``intra_inter_ratio`` is Roth et al.'s pi_ratio in spirit: mean distance of a
    sample to its class centroid over mean distance between class centroids.
    """

    embeddings = np.asarray(embeddings, dtype=np.float64)
    unit = embeddings / np.clip(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)
    labels = np.asarray(labels).reshape(-1)
    classes = np.unique(labels)
    if len(classes) < 2:
        return {}
    centroids = []
    intra = []
    for label in classes:
        members = unit[labels == label]
        centroid = members.mean(axis=0)
        centroids.append(centroid / max(np.linalg.norm(centroid), 1e-12))
        intra.append(np.mean(np.linalg.norm(members - centroid, axis=1)))
    centroids = np.stack(centroids)
    centroid_cos = centroids @ centroids.T
    diagonal = np.eye(len(classes), dtype=bool)
    off_diagonal = centroid_cos[~diagonal]
    nearest = np.max(np.where(diagonal, -np.inf, centroid_cos), axis=1)
    mean_intra = float(np.mean(intra))
    inter_distance = float(np.mean(np.sqrt(np.clip(2 - 2 * off_diagonal, 0, None))))
    return {
        "intra_class_distance": mean_intra,
        "inter_centroid_distance": inter_distance,
        "nearest_centroid_distance": float(np.mean(np.sqrt(np.clip(2 - 2 * nearest, 0, None)))),
        "intra_inter_ratio": mean_intra / max(inter_distance, 1e-12),
    }


def roth_rho_spectrum(embeddings, mode):
    """Roth et al. (2020)'s ``rho_spectrum@mode``, ported from their reference code.

    Singular values of the embedding matrix as it is -- no centering -- normalized
    to sum to one. Their default evaluation path (``--evaluate_on_gpu`` off, also
    in their sample runs) takes them from ``TruncatedSVD(n_components=D - 1)``, so
    the smallest of the D values never enters; with D - 1 + 10 oversamples >= D
    the randomized solver spans the whole space and matches the exact top D - 1.
    ``mode`` > 0 then drops the ``mode - 1`` largest values and returns
    KL(uniform || spectrum); the paper reports mode 2 ("excluding highest SV
    which can obfuscate remaining DoVs", footnote 3; ``Rho3`` in their
    Result_Evaluations.py). ``mode`` < 0 returns KL(spectrum || uniform); 0
    returns the normalized spectrum itself.

    With fewer samples than dimensions their code would normalize over D - N
    zero singular values and report a divergence set by numerical noise; here
    the rank-N spectrum is used instead, so ``roth_rho_dims`` tells whether a
    value is Roth's (it equals D - mode) or only the rank-deficient stand-in.
    """

    return roth_rho_from_singular(roth_singular_values(embeddings), mode)


def roth_singular_values(embeddings):
    """The singular values :func:`roth_rho_spectrum` reads: uncentered, largest first, the smallest dropped."""

    features = np.asarray(embeddings, dtype=np.float64)
    return np.linalg.svd(features, compute_uv=False)[: features.shape[1] - 1]


def roth_rho_from_singular(singular, mode):
    """:func:`roth_rho_spectrum` from :func:`roth_singular_values`, so several modes share one SVD."""

    singular = np.asarray(singular, dtype=np.float64)
    if mode != 0:
        singular = singular[abs(int(mode)) - 1:]
    shares = singular / singular.sum()
    uniform = np.full(len(shares), 1.0 / len(shares))
    if mode == 0:
        return shares
    p, q = (uniform, shares) if mode > 0 else (shares, uniform)
    positive = p > 0
    with np.errstate(divide="ignore"):
        return float(np.sum(p[positive] * np.log(p[positive] / q[positive])))


def roth_class_distances(embeddings, labels):
    """Roth et al. (2020)'s ``dists@intra``, ``dists@inter`` and their ratio, ported from the reference code.

    ``pi_intra``: per class the mean pairwise cosine distance between its samples
    (diagonal excluded), averaged over classes; a single-sample class (0/0 in their
    code) takes the largest finite class value -- what their fill intends; the
    code itself indexes ``intra[1 - isnan]``, i.e. elements 0 and 1, which never
    mattered on their benchmarks because no class there has a single sample.
    ``pi_inter``: mean pairwise cosine distance between the L2-normalized class
    means (diagonal excluded).
    ``*_multi``: the same over the classes with at least two samples only, without
    the fill (pi_inter and the ratio only when two such classes exist);
    ``pi_class_count`` and ``pi_single_class_count`` count the classes.
    """

    features = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    unit = features / np.clip(np.linalg.norm(features, axis=1, keepdims=True), 1e-12, None)
    classes = np.unique(labels)
    if len(classes) < 2:
        return {}
    intra, means = [], []
    for label in classes:
        members = unit[labels == label]
        n = len(members)
        if n > 1:
            cosine = members @ members.T
            intra.append(float(np.sum(1.0 - cosine) / (n * n - n)))
        else:
            intra.append(np.nan)
        mean = features[labels == label].mean(axis=0)
        means.append(mean / max(np.linalg.norm(mean), 1e-12))
    intra = np.asarray(intra)
    finite = np.isfinite(intra)
    if not finite.any():
        return {}
    observed = intra[finite].copy()
    intra[~finite] = intra[finite].max()
    means = np.stack(means)
    inter_matrix = 1.0 - means @ means.T
    c = len(classes)
    pi_intra = float(intra.mean())
    pi_inter = float((inter_matrix.sum() - np.trace(inter_matrix)) / (c * c - c))
    result = {"pi_intra_roth": pi_intra, "pi_inter_roth": pi_inter,
              "pi_ratio_roth": pi_intra / max(pi_inter, 1e-8)}
    # The same over the classes with at least two samples: no fill, and pi_inter
    # over the same classes, so the ratio describes one class set.
    m = int(finite.sum())
    result.update({"pi_intra_multi": float(observed.mean()),
                   "pi_class_count": int(c), "pi_single_class_count": int(c - m)})
    if m >= 2:
        block = inter_matrix[np.ix_(finite, finite)]
        pi_inter_multi = float((block.sum() - np.trace(block)) / (m * m - m))
        result.update({"pi_inter_multi": pi_inter_multi,
                       "pi_ratio_multi": result["pi_intra_multi"] / max(pi_inter_multi, 1e-8)})
    return result


def roth_class_spectrum(embeddings, labels, points=CLASS_SPECTRUM_POINTS):
    """Roth et al. (2020)'s class-conditioned spectrum (supp. E, Fig. 11, after Verma et al. 2018).

    "We perform SVD on Phi_yl and subsequently average over all classes": per
    class the sorted singular values of its (uncentered, like their rho) embedding
    matrix, the first ``points`` of them, averaged over classes. A class with fewer
    samples than ``points`` has exactly zero singular values beyond its rank, so it
    is zero-padded rather than dropped. Returns (mean spectrum, class count).
    """

    features = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels).reshape(-1)
    spectra = []
    for label in np.unique(labels):
        singular = np.linalg.svd(features[labels == label], compute_uv=False)[:points]
        spectra.append(np.pad(singular, (0, points - len(singular))))
    if not spectra:
        return [], 0
    return np.mean(spectra, axis=0), len(spectra)


def embedding_geometry_metrics(embeddings, labels, *, seed=0, spectrum_points=GEOMETRY_SPECTRUM_POINTS):
    """Everything ``geometry`` records for one split: structure, decay, class distances, spectrum."""

    # Two decompositions serve the whole row: the centered spectrum (effective
    # rank, spectral decay, spectrum) and Roth's uncentered one (every rho mode).
    singular = singular_spectrum(embeddings)
    metrics = embedding_structure_metrics(embeddings, labels, seed=seed, singular_values=singular)
    metrics["spectral_decay"] = spectral_decay(singular)
    metrics["spectral_decay_wo_first"] = spectral_decay(singular, drop_first=True)
    metrics.update(class_geometry(embeddings, labels))
    metrics["class_count"] = int(len(np.unique(np.asarray(labels))))
    power = np.square(singular)
    total = float(power.sum())
    shares = power / total if total > 0 else power
    metrics["spectrum"] = [round(float(value), 7) for value in shares[:spectrum_points]]
    # Roth et al.'s definitions exactly as their code computes them, on the
    # L2-normalized embeddings (a no-op for the head, which already outputs unit
    # vectors; it puts the backbone reference on the same footing).
    unit = np.asarray(embeddings, dtype=np.float64)
    unit = unit / np.clip(np.linalg.norm(unit, axis=1, keepdims=True), 1e-12, None)
    roth_singular = roth_singular_values(unit)
    metrics["rho_roth"] = roth_rho_from_singular(roth_singular, 2)
    metrics["rho_roth_all"] = roth_rho_from_singular(roth_singular, 1)
    metrics["roth_rho_dims"] = int(min(unit.shape[0], unit.shape[1] - 1) - 1)
    metrics.update(roth_class_distances(unit, labels))
    metrics["spectrum_roth"] = [
        round(float(value), 7) for value in roth_rho_from_singular(roth_singular, 0)[:spectrum_points]
    ]
    class_spectrum, class_count = roth_class_spectrum(unit, labels)
    metrics["class_spectrum_roth"] = [round(float(value), 6) for value in class_spectrum]
    metrics["class_spectrum_roth_classes"] = int(class_count)
    return metrics


def geometry_row(epoch, representation, split, embeddings, labels, seed, known_mask=None, extra=None):
    """One ``geometry.jsonl`` row. Module-level, so a worker process can run it.

    ``known_mask`` (the whole pool's row only) adds the prototype accuracy of the
    unlabeled samples; ``extra`` holds what the training process measured on the
    same embeddings (the kNN graph's edge purity).
    """

    metrics = embedding_geometry_metrics(embeddings, labels, seed=seed)
    if known_mask is not None:
        metrics.update(prototype_unlabeled_accuracy(embeddings, labels, known_mask))
    return {"epoch": epoch, "representation": representation, "split": split, **metrics, **(extra or {})}


# Unlabeled rows scored against the prototypes at once: rows x classes float64
# probabilities, which for In-Shop's ~3000 classes would otherwise be ~1.5 GB.
PROTOTYPE_CHUNK = 2048


def prototype_unlabeled_accuracy(embeddings, labels, known_mask, chunk=PROTOTYPE_CHUNK):
    """The graph runs' ``prototype_unlabeled_accuracy`` over every unlabeled sample of the pool.

    :func:`prototype_evidence` against the labeled samples' class means, scored
    as :func:`_evidence_summary` does: the share of unlabeled samples whose most
    probable prototype is their hidden class.
    """

    embeddings = np.asarray(embeddings)
    labels = np.asarray(labels).reshape(-1)
    known_mask = np.asarray(known_mask, dtype=bool)
    unlabeled = np.flatnonzero(~known_mask)
    if not known_mask.any() or len(unlabeled) == 0:
        return {}
    reference, reference_labels = embeddings[known_mask], labels[known_mask]
    correct = 0
    for start in range(0, len(unlabeled), chunk):
        rows = unlabeled[start:start + chunk]
        probabilities, classes, _ = prototype_evidence(embeddings[rows], reference, reference_labels)
        correct += int(np.sum(np.asarray(classes)[np.argmax(probabilities, axis=1)] == labels[rows]))
    return {"prototype_unlabeled_accuracy": correct / len(unlabeled)}


def graph_edge_purity(embeddings, labels, ssl_config):
    """The graph runs' ``graph_edge_purity`` over the given embeddings.

    The kNN graph is the one the graph runs build (:func:`build_tracking_graph`
    with the method's neighbour count and weighting, the documented defaults for
    a method without a graph), searched on the GPU as the training's own graphs
    are. Purity is graph_diagnostics' oracle edge purity: the share of undirected
    edges whose two samples share their hidden class.
    """

    from scipy import sparse

    n_neighbors, edge_weighting, gamma = graph_parameters(ssl_config)
    adjacency, metadata = build_tracking_graph(
        embeddings, n_neighbors=n_neighbors, edge_weighting=edge_weighting, gamma=gamma
    )
    upper = sparse.triu(adjacency, k=1).tocoo()
    labels = np.asarray(labels).reshape(-1)
    same = labels[upper.row] == labels[upper.col]
    return {
        "graph_edge_purity": float(same.mean()) if len(same) else float("nan"),
        "graph_undirected_edge_count": int(len(same)),
        "graph_n_neighbors": int(metadata["search_n_neighbors"]),
        "graph_kind": metadata["graph_kind"],
    }


def _blas_threads():
    """The BLAS thread count this process computes with (the run's ``--num_threads``).

    The last bits of an SVD depend on it, so the workers are held to the same
    count: a row then comes out identical whichever process computed it.
    """

    try:
        from threadpoolctl import threadpool_info
    except ImportError:  # pragma: no cover - a dependency of scikit-learn
        return 1
    counts = [int(info["num_threads"]) for info in threadpool_info() if info.get("user_api") == "blas"]
    return max(1, min(counts)) if counts else 1


def _geometry_worker_init(threads):
    import signal

    from .cpu_threads import limit_native_thread_pools

    # The training process decides when the run stops; a Ctrl-C aimed at it must
    # not take a half-written row down with the worker.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    limit_native_thread_pools(threads)


@contextlib.contextmanager
def _starting_without_main():
    """Start worker processes without re-running the training entry point.

    A spawned child re-imports ``__main__`` before it runs anything; for
    main.py that is torch and the whole training stack (~6 s and ~1.2 GB per
    worker), while a geometry worker needs numpy and this module only.
    multiprocessing reads the main module's ``__spec__`` and ``__file__`` when
    it starts a child, and ProcessPoolExecutor starts its children inside
    ``submit``, so the two are hidden for exactly that call.
    """

    main = sys.modules.get("__main__")
    hidden = {name: getattr(main, name) for name in ("__spec__", "__file__") if hasattr(main, name)}
    try:
        for name in hidden:
            if name == "__spec__":
                main.__spec__ = None
            else:
                delattr(main, name)
        yield
    finally:
        for name, value in hidden.items():
            setattr(main, name, value)


# One pool per process, reused by every fold it trains.
_GEOMETRY_POOL = None
_GEOMETRY_POOL_KEY = None


def _geometry_pool(workers, threads):
    global _GEOMETRY_POOL, _GEOMETRY_POOL_KEY
    if _GEOMETRY_POOL is None or _GEOMETRY_POOL_KEY != (workers, threads):
        if _GEOMETRY_POOL is not None:
            _GEOMETRY_POOL.shutdown(wait=True)
        import multiprocessing
        from concurrent.futures import ProcessPoolExecutor

        _GEOMETRY_POOL = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_geometry_worker_init,
            initargs=(threads,),
        )
        _GEOMETRY_POOL_KEY = (workers, threads)
    return _GEOMETRY_POOL


def _discard_geometry_pool():
    global _GEOMETRY_POOL, _GEOMETRY_POOL_KEY
    pool, _GEOMETRY_POOL, _GEOMETRY_POOL_KEY = _GEOMETRY_POOL, None, None
    if pool is not None:
        pool.shutdown(wait=False, cancel_futures=True)


def _geometry_row_here(task, threads):
    """The in-process fallback, under the same thread count a worker uses."""

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=threads):
        return geometry_row(*task)


def class_subsample(labels, max_samples, seed):
    """Indices of whole, randomly chosen classes up to ``max_samples`` samples.

    Whole classes rather than random samples: on a split with a few images per
    class (In-Shop), a random cap leaves most classes with one sample and the
    intra-class statistics with nothing to measure.
    """

    labels = np.asarray(labels).reshape(-1)
    if max_samples is None or len(labels) <= max_samples:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    classes, counts = np.unique(labels, return_counts=True)
    chosen, total = [], 0
    for index in rng.permutation(len(classes)):
        if total + counts[index] > max_samples and chosen:
            continue
        chosen.append(classes[index])
        total += counts[index]
        if total >= max_samples:
            break
    return np.flatnonzero(np.isin(labels, chosen)).astype(np.int64)


def build_tracking_graph(embeddings, *, n_neighbors, edge_weighting, gamma):
    """Build the pool's kNN graph with the method's own edge weighting."""

    from .ssl.algorithms import build_lrml_knn_graph, make_mixed_label_affinity

    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    requested = int(n_neighbors)
    k = max(1, min(requested, len(embeddings) - 1))
    if edge_weighting == "binary":
        neighbor_indices, adjacency, _ = build_lrml_knn_graph(embeddings, n_neighbors=k)
        graph_kind = "binary_knn_union"
    else:
        adjacency, construction = make_mixed_label_affinity(
            embeddings,
            n_neighbors=k,
            gamma=float(gamma),
            return_diagnostics=True,
        )
        neighbor_indices = construction["neighbor_indices"]
        graph_kind = "positive_cosine_knn"
    metadata = {
        "graph_kind": graph_kind,
        "requested_n_neighbors": requested,
        "search_n_neighbors": k,
        "neighbor_indices": neighbor_indices,
    }
    return adjacency, metadata


def save_npz_atomically(path, **arrays):
    """``np.savez_compressed`` into a temporary file renamed into place.

    A live reader polls these archives while the run writes them; the rename
    means it sees either the previous state or the complete archive.
    """

    path = Path(path)
    if path.suffix != ".npz":
        # What np.savez_compressed does with a name, so callers see the same file.
        path = path.with_name(f"{path.name}.npz")
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return path


def write_graph_edges(path, adjacency, positions, neighbor_indices=None, embeddings=None):
    """Save every edge of a tracked graph, keyed by the nodes' sample positions.

    ``source``/``target`` index ``positions`` (the graph_nodes.csv order). A
    symmetric graph stores each undirected edge once (source < target); any
    other graph stores every directed entry.

    ``knn`` keeps each node's own neighbor search in rank order, with the
    cosine similarity of every pair in ``knn_similarity``: the graph above is
    unordered, and on a binary kNN union every edge weighs the same, so this is
    the only record of which neighbors were nearest. -1 marks an empty slot.
    """

    from scipy import sparse

    adjacency = sparse.csr_matrix(adjacency)
    symmetric = (adjacency != adjacency.T).nnz == 0
    stored = sparse.triu(adjacency, k=1) if symmetric else adjacency
    stored = sparse.coo_matrix(stored)
    index_dtype = np.uint16 if adjacency.shape[0] <= np.iinfo(np.uint16).max + 1 else np.uint32
    arrays = {
        "positions": np.asarray(positions, dtype=np.int64),
        "source": stored.row.astype(index_dtype),
        "target": stored.col.astype(index_dtype),
        "weight": stored.data.astype(np.float32),
        "symmetric": np.bool_(symmetric),
    }
    if neighbor_indices is not None:
        knn = np.asarray(neighbor_indices, dtype=np.int64).reshape(adjacency.shape[0], -1)
        arrays["knn"] = knn.astype(np.int32)
        if embeddings is not None:
            normalized = np.asarray(embeddings, dtype=np.float32)
            normalized = normalized / np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
            similarity = np.einsum("nd,nkd->nk", normalized, normalized[np.maximum(knn, 0)])
            arrays["knn_similarity"] = np.where(knn >= 0, similarity, np.nan).astype(np.float16)
    save_npz_atomically(path, **arrays)
    return path


def _normalized_certainty(probabilities, class_count):
    """1 - H(p) / log(C): 0 for a uniform distribution over C classes, 1 for one-hot."""

    probabilities = np.clip(np.asarray(probabilities, dtype=np.float64), 0.0, 1.0)
    entropy = -np.sum(np.where(probabilities > 0.0, probabilities * np.log(np.maximum(probabilities, 1e-300)), 0.0), axis=1)
    return 1.0 - entropy / np.log(max(int(class_count), 2))


def _top_classes(probabilities, class_ids, top_k):
    """Each row's ``top_k`` classes (dataset labels, -1 where the row has no mass) and their probabilities."""

    order = np.argsort(-probabilities, axis=1)[:, :top_k]
    top_probabilities = np.zeros((len(probabilities), top_k), dtype=np.float64)
    top_classes = np.full((len(probabilities), top_k), -1, dtype=np.int64)
    # Fewer classes than top_k leave the trailing columns empty.
    width = order.shape[1]
    top_probabilities[:, :width] = np.take_along_axis(probabilities, order, axis=1)
    top_classes[:, :width] = np.where(top_probabilities[:, :width] > 0.0, np.asarray(class_ids)[order], -1)
    return top_classes.astype(np.int32), top_probabilities.astype(np.float16)


def label_propagation_params(ssl_config):
    """The label propagation settings to read one epoch with.

    A run of Iscen's method is read with its own settings; any other run is read
    with label propagation's defaults, because the question those rows answer is
    what Iscen's diffusion would have made of *this* embedding. ``ssl_config``
    may be the config object or the ``ssl_config`` block of a run_config.json.
    """

    if isinstance(ssl_config, dict):
        method = ssl_config.get("method")
        method_params = dict(ssl_config.get("method_params") or {})
    else:
        method = getattr(ssl_config, "method", None)
        method_params = dict(getattr(ssl_config, "method_params", None) or {})
    if method != "iscen_label_spreading":
        return {}
    return {
        key: method_params[key]
        for key in ("n_neighbors", "gamma", "alpha", "cg_rtol", "cg_max_iter", "linear_solver")
        if key in method_params
    }


def label_propagation_evidence(embeddings, labels, known_mask, params=None):
    """Iscen et al.'s label spreading from the labeled nodes over this embedding.

    Returns row-normalized class probabilities over the labeled classes (all
    zero for a node no labeled node can reach), the class ids of their columns,
    the normalized-entropy certainty, and the method's own weight omega (that
    certainty divided by the pool's largest, 1 for labeled nodes).
    """

    from .ssl.algorithms import UNLABELED_TARGET, iscen_label_spreading

    classes = np.unique(labels[known_mask])
    column_of = {int(label): column for column, label in enumerate(classes)}
    targets = np.full(len(labels), UNLABELED_TARGET, dtype=np.int64)
    targets[known_mask] = [column_of[int(label)] for label in labels[known_mask]]
    params = dict(params or {})
    params["n_neighbors"] = max(1, min(int(params.get("n_neighbors", 50)), len(labels) - 1))
    scores, weights = iscen_label_spreading(
        features=embeddings,
        targets=targets,
        num_classes=len(classes),
        **params,
    )
    scores = np.clip(np.asarray(scores, dtype=np.float64), 0.0, None)
    mass = scores.sum(axis=1, keepdims=True)
    probabilities = np.divide(scores, mass, out=np.zeros_like(scores), where=mass > 0.0)
    certainty = np.where(mass[:, 0] > 0.0, _normalized_certainty(probabilities, len(classes)), 0.0)
    return probabilities, classes, certainty, np.asarray(weights, dtype=np.float64)


def hoffer_evidence(embeddings, labels, known_mask, distance_scale=1.0, draws=HOFFER_REFERENCE_DRAWS, seed=0):
    """Hoffer & Ailon's reference softmax, averaged over random reference draws.

    Each draw takes one labeled sample per class as that class's reference and
    forms softmax(-distance_scale * ||f(x) - f(z_c)||^2) on L2-normalized
    embeddings, as the regularizer does every step. Returns the mean
    distribution, its class ids, and the mean normalized certainty of the
    per-draw distributions (1 - the entropy the regularizer minimizes, over log C).
    """

    normalized = np.asarray(embeddings, dtype=np.float64)
    normalized = normalized / np.maximum(np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12)
    classes = np.unique(labels[known_mask])
    members = [np.flatnonzero(known_mask & (labels == label)) for label in classes]
    rng = np.random.default_rng(seed)
    squared_norms = np.sum(normalized * normalized, axis=1)
    mean_probabilities = np.zeros((len(labels), len(classes)), dtype=np.float64)
    certainty = np.zeros(len(labels), dtype=np.float64)
    for _ in range(int(draws)):
        references = normalized[[rng.choice(indices) for indices in members]]
        squared_distances = (
            squared_norms[:, None] + np.sum(references * references, axis=1)[None, :] - 2.0 * normalized @ references.T
        )
        logits = -float(distance_scale) * np.maximum(squared_distances, 0.0)
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        mean_probabilities += probabilities
        certainty += _normalized_certainty(probabilities, len(classes))
    return mean_probabilities / draws, classes, certainty / draws


def prototype_evidence(embeddings, reference_embeddings, reference_labels, temperature=PROTOTYPE_TEMPERATURE):
    """softmax(cos(x, mu_c) / temperature) over the reference classes' mean embeddings.

    Returns the class probabilities, their class ids, and each row's entropy in nats.
    """

    def normalize(values):
        values = np.asarray(values, dtype=np.float64)
        return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)

    reference = normalize(reference_embeddings)
    reference_labels = np.asarray(reference_labels, dtype=np.int64)
    classes = np.unique(reference_labels)
    prototypes = normalize(np.stack([reference[reference_labels == label].mean(axis=0) for label in classes]))
    logits = normalize(embeddings) @ prototypes.T / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    probabilities = np.exp(logits)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    entropy = -np.sum(np.where(probabilities > 0.0, probabilities * np.log(np.maximum(probabilities, 1e-300)), 0.0), axis=1)
    return probabilities, classes, entropy


def _evidence_summary(prefix, probabilities, classes, certainty, labels, known_mask, confidence=None):
    """Hidden-label scores of one evidence map; ``confidence`` (default: certainty) decides "confident"."""

    from .ssl.pseudo_labels import confidence_correctness_diagnostics

    unlabeled = ~known_mask
    confidence = certainty if confidence is None else confidence
    reachable = probabilities.sum(axis=1) > 0.0
    predicted = np.asarray(classes)[np.argmax(probabilities, axis=1)]
    correct = (predicted == labels) & reachable
    summary = {}
    if np.any(unlabeled):
        summary[f"{prefix}_unlabeled_accuracy"] = float(np.mean(correct[unlabeled]))
        summary[f"{prefix}_unlabeled_certainty_mean"] = float(np.mean(certainty[unlabeled]))
        summary[f"{prefix}_confident_wrong_share"] = float(
            np.mean(~correct[unlabeled] & (confidence[unlabeled] >= CONFIDENT_CERTAINTY))
        )
        order = np.argsort(-confidence[unlabeled], kind="stable")
        unlabeled_correct = correct[unlabeled][order]
        for share in CONFIDENT_FRACTIONS:
            kept = max(1, int(round(len(order) * share / 100.0)))
            summary[f"{prefix}_top{share}_accuracy"] = float(np.mean(unlabeled_correct[:kept]))
        diagnostics = confidence_correctness_diagnostics(correct[unlabeled], confidence[unlabeled])
        if diagnostics and diagnostics.get("auc") is not None:
            summary[f"{prefix}_{'weight' if prefix == 'lp' else 'certainty'}_auc"] = float(diagnostics["auc"])
    return summary


def assignment_concentration(prefix, probabilities, classes, known_mask):
    """How evenly the unlabeled samples' top-1 assignments spread over the reference classes.

    ``assignment_entropy`` is the entropy of the assignment distribution over log C
    (1 = uniform over all classes), ``assignment_gini`` the Gini coefficient of the
    per-class counts (0 = even, towards 1 = a few classes take everything), and
    ``assignment_max_share`` the largest single class's share. A method that only
    sharpens assignments onto a few classes shows falling entropy and rising Gini.
    """

    unlabeled = ~np.asarray(known_mask, dtype=bool)
    classes = np.asarray(classes)
    if not np.any(unlabeled) or len(classes) < 2:
        return {}
    reachable = probabilities[unlabeled].sum(axis=1) > 0.0
    top = np.argmax(probabilities[unlabeled][reachable], axis=1)
    counts = np.bincount(top, minlength=len(classes)).astype(np.float64)
    total = counts.sum()
    if total <= 0:
        return {}
    shares = counts / total
    positive = shares[shares > 0]
    ordered = np.sort(counts)
    n = len(ordered)
    gini = float(np.sum((2 * np.arange(1, n + 1) - n - 1) * ordered) / (n * ordered.sum()))
    return {
        f"{prefix}_assignment_entropy": float(-(positive * np.log(positive)).sum() / np.log(len(classes))),
        f"{prefix}_assignment_gini": gini,
        f"{prefix}_assignment_max_share": float(shares.max()),
    }


def write_class_evidence(
    path,
    embeddings,
    positions,
    labels,
    known_mask,
    ssl_config=None,
    seed=0,
    test_embeddings=None,
    test_labels=None,
    trace_path=None,
):
    """Save how label propagation and Hoffer's entropy term read one tracked epoch.

    Both are computed for every method, so an LRML or SLADE run shows how
    propagatable its embedding is too. A run of the method itself uses its own
    settings (Iscen's n_neighbors/gamma/alpha, Hoffer's distance_scale); any
    other run uses their defaults. Labels are the run's own; the hidden labels of
    unlabeled nodes only score the predictions afterwards.

    The network's own prediction (``prototype_*``) is a softmax over the labeled
    class prototypes; its mean entropy is reported for labeled and unlabeled
    nodes, and for D_test against D_test's own class prototypes (its classes
    are disjoint from training's, so the hidden test labels form them; analysis
    only).

    ``trace_path`` additionally saves the same propagation step by step, with
    the settings this function has already read off the method
    (:func:`label_propagation_trace`).
    """

    labels = np.asarray(labels, dtype=np.int64)
    known_mask = np.asarray(known_mask, dtype=bool)
    if not np.any(known_mask):
        return {}
    method_params = dict(getattr(ssl_config, "method_params", None) or {})
    lp_params = label_propagation_params(ssl_config)
    distance_scale = 1.0
    if method_params.get("regularizer") == "hoffer_entropy":
        distance_scale = float((method_params.get("regularizer_params") or {}).get("distance_scale", 1.0))

    arrays = {"positions": np.asarray(positions, dtype=np.int64)}
    summary = {}
    lp_probabilities, lp_classes, lp_certainty, lp_weight = label_propagation_evidence(
        embeddings, labels, known_mask, lp_params
    )
    arrays["lp_top_classes"], arrays["lp_top_probabilities"] = _top_classes(lp_probabilities, lp_classes, CLASS_EVIDENCE_TOP_K)
    arrays["lp_certainty"] = lp_certainty.astype(np.float16)
    arrays["lp_weight"] = lp_weight.astype(np.float16)
    # Iscen trains with omega, which is relative to the pool's most certain node,
    # so "confident" is judged on omega; the absolute certainty is mostly tiny.
    summary.update(
        _evidence_summary("lp", lp_probabilities, lp_classes, lp_certainty, labels, known_mask, confidence=lp_weight)
    )
    if trace_path is not None:
        try:
            summary.update(
                write_label_propagation_trace(
                    trace_path,
                    embeddings,
                    positions,
                    labels,
                    known_mask,
                    lp_params,
                    reference_probabilities=lp_probabilities,
                )
            )
        except Exception as exc:
            # The trace is a second reading of a diffusion this function has
            # already recorded; losing it must not cost the evidence file.
            logger.warning(f"Epoch tracking label propagation trace failed: {exc}")

    hoffer_probabilities, hoffer_classes, hoffer_certainty = hoffer_evidence(
        embeddings, labels, known_mask, distance_scale=distance_scale, seed=seed
    )
    arrays["hoffer_top_classes"], arrays["hoffer_top_probabilities"] = _top_classes(
        hoffer_probabilities, hoffer_classes, CLASS_EVIDENCE_TOP_K
    )
    arrays["hoffer_certainty"] = hoffer_certainty.astype(np.float16)
    arrays["hoffer_distance_scale"] = np.float32(distance_scale)
    summary.update(_evidence_summary("hoffer", hoffer_probabilities, hoffer_classes, hoffer_certainty, labels, known_mask))
    summary.update(assignment_concentration("hoffer", hoffer_probabilities, hoffer_classes, known_mask))
    summary.update(assignment_concentration("lp", lp_probabilities, lp_classes, known_mask))
    if np.any(~known_mask):
        summary["hoffer_unlabeled_entropy_mean"] = float(1.0 - np.mean(hoffer_certainty[~known_mask]))
    fixed_probabilities, fixed_classes, fixed_certainty = hoffer_evidence(
        embeddings, labels, known_mask, distance_scale=HOFFER_FIXED_DISTANCE_SCALE, seed=seed
    )
    arrays["hoffer_fixed_scale_certainty"] = fixed_certainty.astype(np.float16)
    arrays["hoffer_fixed_distance_scale"] = np.float32(HOFFER_FIXED_DISTANCE_SCALE)
    fixed_summary = _evidence_summary("hoffer_fixed", fixed_probabilities, fixed_classes, fixed_certainty, labels, known_mask)
    if "hoffer_fixed_unlabeled_accuracy" in fixed_summary:
        summary["hoffer_fixed_scale_unlabeled_accuracy"] = fixed_summary["hoffer_fixed_unlabeled_accuracy"]
        summary["hoffer_fixed_scale_unlabeled_entropy_mean"] = float(1.0 - np.mean(fixed_certainty[~known_mask]))

    prototype_probabilities, prototype_classes, prototype_entropy = prototype_evidence(
        embeddings, embeddings[known_mask], labels[known_mask]
    )
    prototype_certainty = 1.0 - prototype_entropy / np.log(max(len(prototype_classes), 2))
    arrays["prototype_top_classes"], arrays["prototype_top_probabilities"] = _top_classes(
        prototype_probabilities, prototype_classes, CLASS_EVIDENCE_TOP_K
    )
    arrays["prototype_certainty"] = prototype_certainty.astype(np.float16)
    arrays["prototype_entropy"] = prototype_entropy.astype(np.float32)
    arrays["prototype_temperature"] = np.float32(PROTOTYPE_TEMPERATURE)
    prototype_summary = _evidence_summary(
        "prototype", prototype_probabilities, prototype_classes, prototype_certainty, labels, known_mask
    )
    for key in (
        "prototype_unlabeled_accuracy",
        "prototype_certainty_auc",
        *(f"prototype_top{share}_accuracy" for share in CONFIDENT_FRACTIONS),
    ):
        if key in prototype_summary:
            summary[key] = prototype_summary[key]
    summary["conditional_entropy_labeled"] = float(np.mean(prototype_entropy[known_mask]))
    if np.any(~known_mask):
        summary["conditional_entropy_unlabeled"] = float(np.mean(prototype_entropy[~known_mask]))
    if test_embeddings is not None and test_labels is not None and len(test_labels) > 1:
        _, _, test_entropy = prototype_evidence(test_embeddings, test_embeddings, test_labels)
        summary["conditional_entropy_test"] = float(np.mean(test_entropy))
        arrays["test_entropy_mean"] = np.float32(summary["conditional_entropy_test"])
    save_npz_atomically(path, **arrays)
    return summary


def label_propagation_trace(
    embeddings,
    labels,
    known_mask,
    params=None,
    steps=PROPAGATION_STEPS,
    reference_probabilities=None,
    affinity=None,
):
    """Iscen's diffusion replayed as its own Neumann series, one hop at a time.

    :func:`label_propagation_evidence` solves ``(I - alpha S) F = Y`` and keeps
    the answer. The same answer is the limit of ``F_0 = Y``,
    ``F_t = alpha S F_{t-1} + Y``, and that series *is* the propagation: at step
    ``t`` a node holds only the label mass that reached it over at most ``t``
    edges. Walking it therefore recovers what the solve collapses -- when each
    node was first reached, which neighbour delivered the mass, which labeled
    node it ultimately came from, and which class the node held while the front
    was still moving.

    The graph is the one label propagation uses (the method's own settings for
    an Iscen run, the defaults otherwise), not the tracked graph, so the last
    step and ``class_evidence.npz`` describe the same diffusion.
    ``lp_trace_agreement`` says how far apart the truncated series and the run's
    solve ended up. Pass ``affinity`` to diffuse over a graph that is already
    built -- what :mod:`scripts.backfill_label_propagation` does with the
    ``graph_edges.npz`` of a run that finished before this file existed --
    in which case ``embeddings`` is ignored.

    Returns ``(arrays, summary)``: the arrays are written to
    ``label_propagation.npz``, the summary becomes epoch_metrics.csv columns.
    """

    from scipy import sparse

    from .ssl.algorithms import make_mixed_label_affinity, symmetric_normalized_affinity

    labels = np.asarray(labels, dtype=np.int64)
    known_mask = np.asarray(known_mask, dtype=bool)
    node_count = len(labels)
    params = dict(params or {})
    n_neighbors = max(1, min(int(params.get("n_neighbors", 50)), node_count - 1))
    gamma = float(params.get("gamma", 3.0))
    alpha = float(params.get("alpha", 0.99))
    steps = int(steps)

    if affinity is None:
        affinity = make_mixed_label_affinity(embeddings, n_neighbors=n_neighbors, gamma=gamma)
    elif "n_neighbors" not in params or "gamma" not in params:
        # A graph that was handed in fixes its own neighborhood; saying k=50 and
        # gamma=3 because those are the defaults would describe someone else's.
        n_neighbors = int(params.get("n_neighbors", -1))
        gamma = float(params.get("gamma", np.nan))
    affinity = sparse.csr_matrix(affinity)
    if affinity.shape != (node_count, node_count):
        raise ValueError(f"affinity is {affinity.shape}, expected one row per node ({node_count})")
    degrees = np.asarray(affinity.sum(axis=1), dtype=np.float64).ravel()
    normalized_affinity = symmetric_normalized_affinity(affinity, degrees)

    # Iscen's class-balanced seeds: every class starts with one unit of mass,
    # so a class with more labeled examples does not simply diffuse further.
    classes = np.unique(labels[known_mask])
    column_of = {int(label): column for column, label in enumerate(classes)}
    labeled_indices = np.flatnonzero(known_mask)
    seed_columns = np.asarray([column_of[int(label)] for label in labels[labeled_indices]], dtype=np.int64)
    class_seed_counts = np.bincount(seed_columns, minlength=len(classes)).astype(np.float64)
    seeds = np.zeros((node_count, len(classes)), dtype=np.float64)
    seeds[labeled_indices, seed_columns] = 1.0 / class_seed_counts[seed_columns]

    state = seeds.copy()
    step_top_classes = np.full((node_count, steps + 1), -1, dtype=np.int32)
    step_top_probabilities = np.zeros((node_count, steps + 1), dtype=np.float64)
    masses = np.zeros((node_count, steps + 1), dtype=np.float64)
    arrival_step = np.full(node_count, -1, dtype=np.int32)
    parent = np.full(node_count, -1, dtype=np.int64)
    parent_share = np.zeros(node_count, dtype=np.float64)
    rows = np.arange(node_count)
    previous_mass = None
    for step in range(steps + 1):
        if step:
            state = alpha * (normalized_affinity @ state) + seeds
        mass = state.sum(axis=1)
        reached = mass > 0.0
        top = np.argmax(state, axis=1)
        masses[:, step] = mass
        step_top_classes[:, step] = np.where(reached, classes[top], -1)
        step_top_probabilities[:, step] = np.where(reached, state[rows, top] / np.maximum(mass, 1e-300), 0.0)
        newly_reached = np.flatnonzero(reached & (arrival_step < 0))
        arrival_step[newly_reached] = step
        if step and len(newly_reached):
            # Who delivered it: the neighbour whose share of the mass arriving
            # over this step's edges is largest. alpha is common to every term,
            # so it cancels out of both the argmax and the share.
            delivered = normalized_affinity.multiply(previous_mass[None, :]).tocsr()
            for row in newly_reached:
                start, end = delivered.indptr[row], delivered.indptr[row + 1]
                if end <= start:
                    continue
                contributions = delivered.data[start:end]
                total = contributions.sum()
                best = int(np.argmax(contributions))
                if contributions[best] <= 0.0 or total <= 0.0:
                    continue
                parent[row] = int(delivered.indices[start + best])
                parent_share[row] = contributions[best] / total
        previous_mass = mass

    # The seed each node's mass came from, by following the parents back. A
    # parent was reached strictly earlier than its child, so one pass in
    # arrival order resolves every chain.
    source = np.full(node_count, -1, dtype=np.int64)
    hop = np.full(node_count, -1, dtype=np.int32)
    for row in np.argsort(np.where(arrival_step < 0, steps + 1, arrival_step), kind="stable"):
        if arrival_step[row] < 0:
            break
        if known_mask[row]:
            source[row], hop[row] = row, 0
        elif parent[row] >= 0 and source[parent[row]] >= 0:
            source[row], hop[row] = source[parent[row]], hop[parent[row]] + 1

    # Where a node's label is coming from once the mass has settled, which is
    # not the same question as which edge first delivered it: on a graph with
    # k=50 almost every node touches a labeled node directly, so the arrival
    # tree is one hop deep while the converged flow still runs over several.
    # For node i and its winning class c, neighbour j contributes S_ij F_jc.
    # No one edge carries a label: with k=50 the strongest contributor is
    # typically a tenth of the incoming mass, so the top few are kept rather
    # than an argmax pretending to be the route.
    influence = np.full((node_count, PROPAGATION_TOP_SOURCES), -1, dtype=np.int64)
    influence_share = np.zeros((node_count, PROPAGATION_TOP_SOURCES), dtype=np.float64)
    settled = state.sum(axis=1) > 0.0
    winning_column = np.argmax(state, axis=1)
    for column in np.unique(winning_column[settled]):
        winners = np.flatnonzero(settled & (winning_column == column))
        contributions = normalized_affinity[winners].multiply(state[:, column][None, :]).tocsr()
        for offset, row in enumerate(winners):
            start, end = contributions.indptr[offset], contributions.indptr[offset + 1]
            values = contributions.data[start:end]
            total = values.sum()
            if end <= start or total <= 0.0:
                continue
            best = np.argsort(-values)[:PROPAGATION_TOP_SOURCES]
            best = best[values[best] > 0.0]
            influence[row, : len(best)] = contributions.indices[start:end][best]
            influence_share[row, : len(best)] = values[best] / total

    final_mass = masses[:, steps]
    reached = final_mass > 0.0
    share = np.divide(
        masses,
        np.maximum(final_mass, 1e-300)[:, None],
        out=np.zeros_like(masses),
        where=(final_mass > 0.0)[:, None],
    )
    unlabeled = ~known_mask
    correct = (step_top_classes == labels[:, None]) & (step_top_classes >= 0)
    step_reached = (masses > 0.0)[unlabeled].mean(axis=0) if np.any(unlabeled) else np.zeros(steps + 1)
    step_accuracy = np.zeros(steps + 1)
    for step in range(steps + 1):
        counted = unlabeled & (masses[:, step] > 0.0)
        step_accuracy[step] = float(np.mean(correct[counted, step])) if np.any(counted) else np.nan

    arrays = {
        "step_top_classes": step_top_classes,
        "step_top_probabilities": step_top_probabilities.astype(np.float16),
        "step_share": share.astype(np.float16),
        "arrival_step": arrival_step.astype(np.int16),
        "parent": parent.astype(np.int32),
        "parent_share": parent_share.astype(np.float16),
        "source": source.astype(np.int32),
        "hop": hop.astype(np.int16),
        "influence": influence.astype(np.int32),
        "influence_share": influence_share.astype(np.float16),
        "influence_top_k": np.int32(PROPAGATION_TOP_SOURCES),
        "step_accuracy": step_accuracy.astype(np.float32),
        "step_reached": step_reached.astype(np.float32),
        "steps": np.int32(steps),
        "alpha": np.float32(alpha),
        "gamma": np.float32(gamma),
        "n_neighbors": np.int32(n_neighbors),
    }

    summary = {"lp_trace_steps": int(steps)}
    if np.any(unlabeled):
        unlabeled_reached = unlabeled & reached
        summary["lp_reached_share"] = float(np.mean(reached[unlabeled]))
        if np.any(unlabeled_reached):
            summary["lp_mean_hops"] = float(np.mean(hop[unlabeled_reached & (hop >= 0)]))
            summary["lp_source_accuracy"] = float(
                np.mean(labels[source[unlabeled_reached]] == labels[unlabeled_reached])
            )
        # Every edge a label actually travelled along, once per node: the share
        # of them that stayed inside a class is how often the diffusion had to
        # cross a class boundary to arrive at all.
        with_parent = unlabeled & (parent >= 0)
        if np.any(with_parent):
            summary["lp_tree_edge_purity"] = float(np.mean(labels[parent[with_parent]] == labels[with_parent]))
        with_influence = unlabeled & (influence[:, 0] >= 0)
        if np.any(with_influence):
            summary["lp_influence_share"] = float(np.mean(influence_share[with_influence, 0]))
            summary["lp_influence_purity"] = float(
                np.mean(labels[influence[with_influence, 0]] == labels[with_influence])
            )
    if reference_probabilities is not None:
        reference = np.asarray(reference_probabilities, dtype=np.float64)
        reference_reached = reference.sum(axis=1) > 0.0
        both = reference_reached & reached
        if np.any(both):
            summary["lp_trace_agreement"] = float(
                np.mean(classes[np.argmax(reference[both], axis=1)] == step_top_classes[both, steps])
            )
    return arrays, summary


def write_label_propagation_trace(path, embeddings, positions, labels, known_mask, params=None, graph_source="label_propagation", **kwargs):
    """Save :func:`label_propagation_trace` for one tracked epoch.

    ``parent``, ``source`` and ``influence`` index ``positions``, exactly as the
    edges of ``graph_edges.npz`` do. ``graph_source`` names the graph the mass
    moved over: label propagation's own (the default) or the epoch's tracked
    graph, which a backfill has to use because the embeddings are gone.
    """

    arrays, summary = label_propagation_trace(embeddings, labels, known_mask, params, **kwargs)
    save_npz_atomically(
        path,
        positions=np.asarray(positions, dtype=np.int64),
        graph_source=np.str_(graph_source),
        # The summary rides along so a reader has the epoch's numbers without
        # epoch_metrics.csv, which a backfilled epoch cannot add columns to.
        **{key: np.float32(value) for key, value in summary.items()},
        **arrays,
    )
    return summary


def graph_parameters(ssl_config):
    """Read the graph parameters from the method that owns them.

    A method without them (a supervised baseline, or one that never builds a
    graph) still gets a graph built to the documented defaults, because the
    comparison the supervised row exists for is against the other rows' graphs.
    """

    params = dict(
        (ssl_config.get("method_params") if isinstance(ssl_config, dict) else getattr(ssl_config, "method_params", None))
        or {}
    )
    # The graph settings belong to the regularizer, which nests them one level
    # down under ``regularizer_params`` -- that is the level an HPO space tunes
    # (``ssl_config.method_params.regularizer_params.n_neighbors``). Reading only
    # the top level would quietly build the default 10-neighbour graph for a run
    # tuned to 52 and report it as that run's graph.
    nested = dict(params.get("regularizer_params") or {})

    def pick(name, default):
        if name in nested:
            return nested[name]
        if name in params:
            return params[name]
        return default

    n_neighbors = pick("n_neighbors", DEFAULT_TRACKING_N_NEIGHBORS)
    edge_weighting = pick("edge_weighting", "cosine")
    gamma = pick("gamma", DEFAULT_TRACKING_GAMMA)
    return int(n_neighbors), str(edge_weighting), float(gamma)


class EpochTracker:
    """Writes one structure bundle per tracked epoch of one fold."""

    def __init__(
        self,
        *,
        args,
        ssl_config,
        output_dir,
        artifacts,
        interval,
        layout,
        max_samples,
        train_dataset,
        positions,
        labels,
        known_mask,
        dataset_name=None,
        test_embedder=None,
        split_embedder=None,
        backbone_model=None,
        pool_sampling=DEFAULT_TRACK_EPOCH_POOL_SAMPLING,
    ):
        self.args = args
        self.ssl_config = ssl_config
        self.output_dir = Path(output_dir)
        self.artifacts = tuple(artifacts)
        self.interval = int(interval)
        self.layout = layout
        self.max_samples = max_samples
        self.pool_sampling = pool_sampling
        self.train_dataset = train_dataset
        self.positions = np.asarray(positions, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.known_mask = np.asarray(known_mask, dtype=bool)
        self.dataset_name = dataset_name
        # ``test_embedder(epoch) -> (embeddings, labels)`` embeds D_test through the
        # engine's own (feature-cached) test loader, for the conditional-entropy
        # diagnostic only; None skips it.
        self.test_embedder = test_embedder
        # ``split_embedder(split, representation, epoch) -> (embeddings, labels)``
        # for split in ("validation", "test") and representation in ("head",
        # "backbone"); None, or a None return, skips that split. The backbone
        # model stands in for the network when the pool's backbone features are
        # measured. Both are used by ``geometry`` only.
        self.split_embedder = split_embedder
        self.backbone_model = backbone_model
        self._geometry_indices = {}
        self._backbone_geometry_done = False
        self.epochs = int(getattr(args, "epochs", 0))
        self.seed = int(getattr(args, "seed", 0) or 0)
        self.geometry_workers = get_track_epoch_workers(args)
        self._geometry_threads = _blas_threads()
        # Queued geometry rows, oldest first: (description, task, future). They are
        # written in this order, so geometry.jsonl reads as if computed in line.
        self._geometry_pending = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        columns = list(_EPOCH_COLUMNS)
        if TRACK_EPOCH_TEST in self.artifacts:
            columns.extend(_TEST_COLUMNS)
        if TRACK_EPOCH_EMBEDDINGS in self.artifacts:
            columns.extend(_EMBEDDING_COLUMNS)
        if TRACK_EPOCH_GRAPH in self.artifacts:
            columns.extend(_GRAPH_COLUMNS)
            columns.extend(_CLASS_EVIDENCE_COLUMNS)
            columns.extend(_PROPAGATION_COLUMNS)
        self.csv_path = self.output_dir / METRICS_CSV_NAME
        self._csv_file = self.csv_path.open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=columns)
        self._csv_writer.writeheader()
        self._csv_file.flush()

        # The pool is capped once rather than per epoch, so every tracked epoch
        # describes the same samples and the rows form a series.
        self.sample_indices = self._choose_sample_indices()
        logger.info(
            f"Epoch tracking writes {list(self.artifacts)} to {self.output_dir} at epoch "
            f"{BASELINE_EPOCH} and every {self.interval} epochs, over "
            f"{len(self.sample_indices)} of {len(self.positions)} pool samples "
            f"({int(self.known_mask.sum())} labeled), layout={self.layout}"
        )
        self.status_path = self.output_dir / STATUS_NAME
        self._last_completed_epoch = None
        self._write_status("running", stage="waiting")

    def _write_status(self, state, epoch=None, stage=None):
        """Replace status.json. Never raises: a viewer's heartbeat is not the run."""

        status = {
            "state": state,
            "epoch": None if epoch is None else int(epoch),
            "stage": stage,
            "lastCompletedEpoch": self._last_completed_epoch,
            "epochs": self.epochs,
            "interval": self.interval,
            "layout": self.layout,
            "artifacts": list(self.artifacts),
            "samples": int(len(self.sample_indices)),
            "poolSampling": self.pool_sampling,
            "geometryPending": len(getattr(self, "_geometry_pending", ())),
            "pid": os.getpid(),
            "updatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        try:
            temporary = self.status_path.with_name(f".{STATUS_NAME}.tmp")
            temporary.write_text(json.dumps(status, indent=2), encoding="utf-8")
            os.replace(temporary, self.status_path)
        except OSError as exc:  # pragma: no cover - a full disk must not stop training
            logger.debug(f"Epoch tracking status was not written: {exc}")

    @property
    def tracks_test(self):
        return TRACK_EPOCH_TEST in self.artifacts

    @property
    def tracks_structure(self):
        return bool(_POOL_ARTIFACTS & set(self.artifacts))

    @property
    def tracks_geometry(self):
        return TRACK_EPOCH_GEOMETRY in self.artifacts

    def is_due(self, epoch):
        return is_tracked_epoch(epoch, self.interval, self.epochs)

    def _choose_sample_indices(self):
        count = len(self.positions)
        if self.max_samples is None or count <= self.max_samples:
            return np.arange(count, dtype=np.int64)
        if self.pool_sampling == "classes" and len(np.unique(self.labels)) > 1:
            # Whole classes, drawn like the validation and test splits: the cap
            # leaves no tracked class with one image. The labeled share then
            # follows the chosen classes instead of being held fixed.
            return class_subsample(self.labels, self.max_samples, self.seed)
        # Keep the labeled/unlabeled proportion: purity and "nodes in components
        # without a labeled node" are both read against how much of the pool
        # carries a label, so the cap must not quietly change that ratio.
        rng = np.random.default_rng(self.seed)
        labeled = np.flatnonzero(self.known_mask)
        unlabeled = np.flatnonzero(~self.known_mask)
        labeled_quota = int(round(self.max_samples * len(labeled) / count))
        labeled_quota = min(len(labeled), max(0, labeled_quota))
        unlabeled_quota = min(len(unlabeled), self.max_samples - labeled_quota)
        chosen = np.concatenate(
            [
                rng.choice(labeled, size=labeled_quota, replace=False),
                rng.choice(unlabeled, size=unlabeled_quota, replace=False),
            ]
        )
        chosen.sort()
        return chosen

    def track(self, epoch, model, test_outcome=None):
        """Write this epoch's bundle. Never raises: tracking is not the run."""

        if not self.is_due(epoch):
            return None
        started_at = time.perf_counter()
        row = {"epoch": int(epoch)}
        try:
            if test_outcome is not None:
                row["test_precision_at_1"] = test_outcome.precision_at_1
                row["test_mean_average_precision_at_r"] = (
                    test_outcome.mean_average_precision_at_r
                )
            if self.tracks_structure:
                row.update(self._track_structure(epoch, model))
        except Exception as exc:  # pragma: no cover - tracking must not stop training
            logger.warning(f"Epoch tracking failed at epoch {epoch}: {exc}")
        row["tracked_seconds"] = round(time.perf_counter() - started_at, 3)
        self._write_row(row)
        self._last_completed_epoch = int(epoch)
        self._write_status("running", stage="waiting")
        return row

    def _write_row(self, row):
        known = set(self._csv_writer.fieldnames)
        self._csv_writer.writerow({key: row[key] for key in row if key in known})
        self._csv_file.flush()

    def _embedding_workers(self):
        """Loader workers for the tracker's own passes over the pool.

        Over precomputed backbone features a pass only reads cached vectors, which
        the training process does in well under a second. Loader workers would be
        spawned anew on every tracked epoch, each re-importing the training stack
        (~5 s for the default eight), so they are only used for image datasets.
        """

        import utils

        if utils.dataset_has_precomputed_backbone_features(self.train_dataset):
            return 0
        return getattr(self.ssl_config, "embedding_num_workers", 0)

    def _track_structure(self, epoch, model):
        from . import semi_supervised

        epoch_dir = self.output_dir / epoch_slug(epoch)
        epoch_dir.mkdir(parents=True, exist_ok=True)
        positions = self.positions[self.sample_indices]
        labels = self.labels[self.sample_indices]
        known_mask = self.known_mask[self.sample_indices]
        self._write_status("running", epoch=epoch, stage="embedding")
        embeddings = semi_supervised.extract_embeddings(
            model=model,
            dataset=self.train_dataset,
            positions=positions,
            device=getattr(self.args, "ssl_device", None) or self.args.device,
            batch_size=self.ssl_config.embedding_batch_size,
            num_workers=self._embedding_workers(),
            # Offset from the training stream's seeds so a tracked epoch cannot
            # reproduce the same shuffle an SSL rebuild would have used.
            seed=self.seed + 977 + int(epoch),
            start_method=getattr(self.args, "dataloader_start_method", None),
            desc=f"epoch tracking embeddings - epoch {epoch}",
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        row = {}
        if TRACK_EPOCH_EMBEDDINGS in self.artifacts:
            row.update(self._write_embedding_artifacts(epoch, epoch_dir, embeddings, labels))
        if self.tracks_geometry:
            self._write_status("running", epoch=epoch, stage="geometry")
            try:
                self._track_geometry(epoch, model, embeddings, positions, labels, known_mask)
            except Exception as exc:
                logger.warning(f"Epoch tracking geometry failed at epoch {epoch}: {exc}")
        if TRACK_EPOCH_GRAPH in self.artifacts:
            self._write_status("running", epoch=epoch, stage="graph")
            row.update(
                self._write_graph_artifacts(
                    epoch, epoch_dir, embeddings, positions, labels, known_mask
                )
            )
            self._write_status("running", epoch=epoch, stage="class_evidence")
            try:
                test_embeddings, test_labels = self._test_embeddings(epoch, model)
                row.update(
                    write_class_evidence(
                        epoch_dir / CLASS_EVIDENCE_NAME,
                        embeddings,
                        positions,
                        labels,
                        known_mask,
                        self.ssl_config,
                        seed=self.seed + int(epoch),
                        test_embeddings=test_embeddings,
                        test_labels=test_labels,
                        trace_path=epoch_dir / PROPAGATION_NAME,
                    )
                )
            except Exception as exc:
                logger.warning(f"Epoch tracking class evidence failed at epoch {epoch}: {exc}")
        return row

    def _embed_split(self, split, representation, epoch):
        """One eval split, subsampled by whole classes the same way every epoch."""

        if self.split_embedder is None:
            return None
        embedded = self.split_embedder(split, representation, epoch)
        if embedded is None:
            return None
        embeddings, labels = embedded
        embeddings = np.asarray(embeddings, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if split not in self._geometry_indices:
            self._geometry_indices[split] = class_subsample(
                labels, self.max_samples, self.seed + 31 * (GEOMETRY_SPLITS.index(split) + 1)
            )
        indices = self._geometry_indices[split]
        if len(indices) and indices.max() >= len(labels):
            return None
        return embeddings[indices], labels[indices]

    def _geometry_splits(self, representation, epoch, pool_embeddings, labels, known_mask):
        splits = {}
        if pool_embeddings is not None:
            splits["train"] = (pool_embeddings, labels)
            if np.any(known_mask):
                splits["labeled"] = (pool_embeddings[known_mask], labels[known_mask])
            if np.any(~known_mask):
                splits["unlabeled"] = (pool_embeddings[~known_mask], labels[~known_mask])
        for split in ("validation", "test"):
            try:
                embedded = self._embed_split(split, representation, epoch)
            except Exception as exc:
                logger.warning(f"Epoch tracking geometry could not embed {split} ({representation}): {exc}")
                embedded = None
            if embedded is not None:
                splits[split] = embedded
        return splits

    def _write_geometry_rows(self, epoch, representation, splits, known_mask=None, pool_extra=None):
        """Queue one row per split; rows reach geometry.jsonl in this order as they finish.

        The whole pool's row also gets the prototype accuracy (``known_mask``) and
        what the training process measured on it (``pool_extra``).
        """

        self._wait_for_geometry_backlog()
        for split in GEOMETRY_SPLITS:
            if split not in splits:
                continue
            embeddings, labels = splits[split]
            if len(embeddings) < 3:
                continue
            task = (epoch, representation, split, np.asarray(embeddings), np.asarray(labels), self.seed)
            if split == "train":
                task += (None if known_mask is None else np.asarray(known_mask, dtype=bool), pool_extra)
            description = f"{representation} {split} at epoch {epoch}"
            self._geometry_pending.append((description, task, self._submit_geometry(task)))
        self._flush_geometry(block=False)

    def _wait_for_geometry_backlog(self):
        """Hold the training loop while MAX_QUEUED_GEOMETRY_EPOCHS epochs are still queued."""

        from concurrent.futures import wait

        while len({task[0] for _, task, _ in self._geometry_pending}) >= MAX_QUEUED_GEOMETRY_EPOCHS:
            wait([self._geometry_pending[0][2]])
            self._flush_geometry(block=False)

    def _submit_geometry(self, task):
        from concurrent.futures import Future

        if self.geometry_workers > 0:
            try:
                with _starting_without_main():
                    return _geometry_pool(self.geometry_workers, self._geometry_threads).submit(
                        geometry_row, *task
                    )
            except Exception as exc:
                logger.warning(f"Epoch tracking geometry workers unavailable, computing in-process: {exc}")
                _discard_geometry_pool()
                self.geometry_workers = 0
        future = Future()
        try:
            future.set_result(_geometry_row_here(task, self._geometry_threads))
        except Exception as exc:
            future.set_exception(exc)
        return future

    def _flush_geometry(self, block):
        """Write the finished rows at the head of the queue; with ``block``, all of them."""

        from concurrent.futures.process import BrokenProcessPool

        rows = []
        while self._geometry_pending and (block or self._geometry_pending[0][2].done()):
            description, task, future = self._geometry_pending.pop(0)
            try:
                rows.append(future.result())
            except BrokenProcessPool as exc:
                # A worker died (killed, out of memory); the task is still here, so
                # the row is computed in-process instead of lost.
                logger.warning(f"Epoch tracking geometry worker lost for {description}, computing in-process: {exc}")
                _discard_geometry_pool()
                self.geometry_workers = 0
                try:
                    rows.append(_geometry_row_here(task, self._geometry_threads))
                except Exception as retry_exc:
                    logger.warning(f"Epoch tracking geometry failed for {description}: {retry_exc}")
            except Exception as exc:
                logger.warning(f"Epoch tracking geometry failed for {description}: {exc}")
        if not rows:
            return
        with (self.output_dir / GEOMETRY_NAME).open("a", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    def _track_geometry(self, epoch, model, embeddings, positions, labels, known_mask):
        from . import semi_supervised

        if not self._backbone_geometry_done and self.backbone_model is not None:
            # The frozen backbone does not change, so its geometry is measured once
            # per fold, on exactly the samples every head row below describes.
            self._backbone_geometry_done = True
            try:
                backbone_pool = semi_supervised.extract_embeddings(
                    model=self.backbone_model,
                    dataset=self.train_dataset,
                    positions=positions,
                    device=getattr(self.args, "ssl_device", None) or self.args.device,
                    batch_size=self.ssl_config.embedding_batch_size,
                    num_workers=self._embedding_workers(),
                    seed=self.seed + 977,
                    start_method=getattr(self.args, "dataloader_start_method", None),
                    desc="epoch tracking backbone features",
                )
                backbone_pool = np.asarray(backbone_pool, dtype=np.float32)
            except Exception as exc:
                logger.warning(f"Epoch tracking geometry could not embed the pool's backbone features: {exc}")
                backbone_pool = None
            self._write_geometry_rows(
                None, "backbone", self._geometry_splits("backbone", epoch, backbone_pool, labels, known_mask),
                known_mask=known_mask, pool_extra=self._pool_graph_purity(backbone_pool, labels, epoch),
            )
        self._write_geometry_rows(
            int(epoch), "head", self._geometry_splits("head", epoch, embeddings, labels, known_mask),
            known_mask=known_mask, pool_extra=self._pool_graph_purity(embeddings, labels, epoch),
        )

    def _pool_graph_purity(self, embeddings, labels, epoch):
        """Edge purity of the whole pool's kNN graph, built here on the run's GPU; never raises."""

        if embeddings is None:
            return None
        self._write_status("running", epoch=epoch, stage="graph_purity")
        try:
            return graph_edge_purity(embeddings, labels, self.ssl_config)
        except Exception as exc:
            logger.warning(f"Epoch tracking graph purity failed at epoch {epoch}: {exc}")
            return None

    def _test_embeddings(self, epoch, model):
        """D_test embedded for the conditional-entropy diagnostic, or (None, None)."""

        if self.test_embedder is None or TRACK_EPOCH_TEST not in self.artifacts:
            return None, None
        try:
            embedded = self.test_embedder(epoch)
        except Exception as exc:
            logger.warning(f"Epoch tracking could not embed D_test for conditional entropy: {exc}")
            self.test_embedder = None
            return None, None
        if embedded is None:
            return None, None
        embeddings, labels = embedded
        return np.asarray(embeddings, dtype=np.float32), np.asarray(labels, dtype=np.int64).reshape(-1)

    def _write_embedding_artifacts(self, epoch, epoch_dir, embeddings, labels):
        import utils

        metrics = embedding_structure_metrics(
            embeddings,
            labels,
            seed=self.seed + int(epoch),
        )
        with (epoch_dir / "embedding_metrics.json").open("w", encoding="utf-8") as handle:
            json.dump({"epoch": int(epoch), **metrics}, handle, indent=2, sort_keys=True)
        title = f"{getattr(self.args, 'dataset', 'pool')} SSL pool - epoch {epoch}"
        self._write_status("running", epoch=epoch, stage="projection")
        # Coordinates only: the viewer draws them itself, and evaluate/filmstrip.py
        # renders the PNG from the CSV when a picture is wanted. Per tracked
        # epoch that is a few hundred kB of PNG that nothing needs to keep.
        try:
            if self.layout in utils.PROJECTION_DISPLAY_NAMES:
                utils.write_projection_visualization(
                    self.layout,
                    embeddings,
                    labels,
                    output_dir=epoch_dir,
                    stem=f"embeddings_{self.layout}",
                    title=title,
                    dataset=self.train_dataset,
                    dataset_name=self.dataset_name,
                    seed=self.seed,
                    write_plot=False,
                )
        except Exception as exc:
            # The metrics are the part a table is built from; a projection that
            # fails costs a layout, not the row.
            logger.warning(f"Epoch tracking projection failed at epoch {epoch}: {exc}")
        if bool(getattr(self.args, "projection_3d", False)) and self.layout in ("tsne", "pacmap", "pca"):
            self._write_status("running", epoch=epoch, stage="projection_3d")
            try:
                utils.write_projection_3d_csv(
                    embeddings,
                    labels,
                    output_dir=epoch_dir,
                    stem=f"embeddings_{self.layout}_3d",
                    layout=self.layout,
                    seed=self.seed,
                    dataset=self.train_dataset,
                )
            except Exception as exc:
                logger.warning(f"Epoch tracking 3-D projection failed at epoch {epoch}: {exc}")
        return metrics

    def _write_graph_artifacts(self, epoch, epoch_dir, embeddings, positions, labels, known_mask):
        from .ssl import graph_diagnostics
        from .ssl.config import GraphDiagnosticsRequest

        n_neighbors, edge_weighting, gamma = graph_parameters(self.ssl_config)
        adjacency, metadata = build_tracking_graph(
            embeddings,
            n_neighbors=n_neighbors,
            edge_weighting=edge_weighting,
            gamma=gamma,
        )
        try:
            write_graph_edges(
                epoch_dir / GRAPH_EDGES_NAME,
                adjacency,
                positions,
                neighbor_indices=metadata.get("neighbor_indices"),
                embeddings=embeddings,
            )
        except Exception as exc:
            logger.warning(f"Epoch tracking graph edges were not saved at epoch {epoch}: {exc}")
        request = GraphDiagnosticsRequest(
            output_dir=epoch_dir,
            slug="graph",
            title=f"{getattr(self.args, 'dataset', 'pool')} SSL pool graph - epoch {epoch}",
            # The caps the run already configured for its own graph diagnostics
            # apply here too, so one setting governs both.
            max_nodes=self.ssl_config.graph_diagnostics_max_nodes,
            max_edges=self.ssl_config.graph_diagnostics_max_edges,
            max_labels=self.ssl_config.graph_diagnostics_max_labels,
            seed=self.seed + int(epoch),
            layout=self.layout,
            # Set explicitly: the slug carries no epoch, so the series regex
            # would otherwise read every epoch as its own series and the
            # epoch-to-epoch edge churn would never be computed.
            series_slug="epoch_tracking_graph",
            epoch=int(epoch),
            class_focus=self.ssl_config.graph_diagnostics_class_focus,
            class_count=self.ssl_config.graph_diagnostics_class_count,
            classes=(
                None
                if self.ssl_config.graph_diagnostics_classes is None
                else tuple(self.ssl_config.graph_diagnostics_classes)
            ),
            class_context=self.ssl_config.graph_diagnostics_class_context,
            # The graph PNG would need its own 2-D projection of the embeddings
            # every tracked epoch; evaluate/filmstrip.py draws it on demand from
            # graph_sampled_edges.csv over the embeddings layout instead.
            write_plots=False,
        )
        graph_diagnostics.maybe_save_graph_diagnostics(
            request=request,
            embeddings=embeddings,
            adjacency=adjacency,
            positions=positions,
            labels=labels,
            known_mask=known_mask,
            graph_metadata=metadata,
        )
        summary_path = graph_diagnostics.graph_diagnostic_artifact_paths(request)["summary_json"]
        if not summary_path.is_file():
            return {}
        # Read back rather than recompute: save_graph_diagnostics has already
        # written the full summary, and the CSV wants eight of its numbers.
        with summary_path.open(encoding="utf-8") as handle:
            overview = json.load(handle).get("overview", {})
        return {
            column: overview.get(key)
            for column, key in _GRAPH_OVERVIEW_COLUMNS.items()
            if overview.get(key) is not None
        }

    def close(self):
        if self._geometry_pending:
            # The rows run behind training; the fold waits here for the last of them.
            logger.info(f"Epoch tracking waits for {len(self._geometry_pending)} queued geometry rows")
            self._write_status("running", stage="geometry")
            self._flush_geometry(block=True)
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()
            logger.info(f"Epoch tracking metrics written to {self.csv_path}")
            self._write_status("finished")


def make_epoch_tracker(
    args,
    ssl_config,
    *,
    train_dataset,
    split,
    log_dir,
    test_embedder=None,
    split_embedder=None,
    backbone_model=None,
):
    """Build this fold's tracker, or return None when tracking is off."""

    artifacts = get_track_epoch_artifacts(args)
    interval = get_track_epoch_interval(args)
    if not artifacts or interval is None:
        return None
    if not _POOL_ARTIFACTS & set(artifacts):
        # Test-only tracking needs no pool, no embeddings and no graph; the
        # engine evaluates D_test and the tracker only tabulates it.
        positions = np.zeros(0, dtype=np.int64)
        labels = np.zeros(0, dtype=np.int64)
        known_mask = np.zeros(0, dtype=bool)
    else:
        from .ssl import graph_diagnostics

        positions = np.concatenate(
            [
                np.asarray(split.labeled_positions, dtype=np.int64),
                np.asarray(split.unlabeled_positions, dtype=np.int64),
            ]
        )
        labels = graph_diagnostics.dataset_labels_for_positions(train_dataset, positions)
        if labels is None:
            labels = np.zeros(len(positions), dtype=np.int64)
        known_mask = np.arange(len(positions)) < len(split.labeled_positions)
    return EpochTracker(
        args=args,
        ssl_config=ssl_config,
        output_dir=Path(log_dir) / OUTPUT_DIR_NAME,
        artifacts=artifacts,
        interval=interval,
        layout=getattr(args, "track_epoch_layout", DEFAULT_TRACK_EPOCH_LAYOUT),
        max_samples=get_track_epoch_max_samples(args),
        pool_sampling=get_track_epoch_pool_sampling(args),
        train_dataset=train_dataset,
        positions=positions,
        labels=labels,
        known_mask=known_mask,
        dataset_name=getattr(args, "dataset", None),
        test_embedder=test_embedder,
        split_embedder=split_embedder,
        backbone_model=backbone_model,
    )
