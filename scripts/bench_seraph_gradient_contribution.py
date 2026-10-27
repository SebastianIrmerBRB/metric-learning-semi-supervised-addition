"""Measure when SERAPH's entropy term actually competes with the supervised loss.

``ssl_gradient_contribution_log_interval`` already logs
``train/gradient_contribution/regularizer_to_supervised_ratio`` during real runs,
but one run only ever reports the ratio at the hyperparameters that run happened
to draw. This benchmark sweeps the knobs one at a time around a fixed operating
point and reports the same ratio, so the question "which setting makes the
regularizer's gradient large?" is answered by measurement rather than by reading
Eq 5 and guessing.

What is measured
----------------
Per condition, on the same batches and from the same head weights::

    ||grad(regularizer_weight * L_seraph)||  /  ||grad(supervised_weight * L_sup)||

over the model parameters, using ``engine.measure_gradient_component_norms`` -
the function the trainer itself calls - so the numbers are comparable with the
``train/gradient_contribution/*`` series of an actual run. Three splits of the
same measurement are reported because they answer different questions:

``model``
    All model parameters, which is what production logs. With a learnable metric
    this includes SERAPH's own ``B``.
``projection``
    Only the retrieval projection ``fc``, i.e. the parameters the two losses
    genuinely compete over. This is the number that says whether the entropy term
    is shaping the *embedding*.
``head_share``
    The fraction of the regularizer's gradient norm that lands on SERAPH's own
    metric parameters instead of the embedding. Module docstring deviation 2
    warns that a learnable ``A`` lets entropy minimization sharpen the posterior
    by inflating the metric rather than by moving embeddings; this quantifies it.

Alongside those, the saturation probe (mean ``|logit|``, the fraction of pairs
past ``|logit| > 4``, the pair entropy, ``eta`` and the effective ``kappa``)
explains *why* a condition is where it is: the per-pair gradient of the entropy
objective is ``-l * sigmoid'(l)``, which vanishes both at ``l = 0`` (undecided)
and at large ``|l|`` (saturated) and peaks near ``|l| = 1.54``.

Setup
-----
Real frozen DINOv2 features from the project's ``backbone_cache`` (so the pair
distance distribution ``eta`` and ``kappa`` calibrate against is the real one),
a trainable ``Linear + L2 normalize`` head mirroring ``DinoWrapper.fc`` /
``project_features``, an m-per-class labeled batch, a random unlabeled batch, and
the real ``SeraphRegularizer``. Batches are shared across conditions, so the
supervised norm is a constant within a stage and only the regularizer moves.

The default operating point is trial 0006 of ``cars_imb_search``
(``logs/ssl/seraph_prx/.../cars_imb_search``): the tuned Cars196 SERAPH run, so
the sweep brackets a point the HPO actually selected, and every sweep range stays
inside that study's search space.

Examples::

    python scripts/bench_seraph_gradient_contribution.py
    python scripts/bench_seraph_gradient_contribution.py --axes kappa prior
    python scripts/bench_seraph_gradient_contribution.py --dataset CUB --warmup-steps 0 2000
"""

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training import engine  # noqa: E402
from training.ssl.seraph import (  # noqa: E402
    SeraphRegularizer,
    binary_entropy_from_logits,
    cross_squared_distances,
    pairwise_squared_distances,
)
from utils.dataset_protocols import get_dataset_class  # noqa: E402


# Trial 0006 of the tuned Cars196 study. Sweeps move one of these at a time so
# every condition is read against a point the HPO actually chose.
BASELINE_REGULARIZER_PARAMS = {
    "objective": "entropy",
    "em_sharpening": 2.0,
    "metric": "lowrank",
    "metric_dim": None,
    "kappa": 1.7217509183057627,
    "kappa_mode": "fixed",
    "threshold": 0.1,
    "threshold_mode": "quantile",
    "threshold_momentum": 0.7577182057524012,
    "positive_pair_prior": 0.018019235760867558,
    "include_labeled_unlabeled": False,
    "unlabeled_batch_size": 64,
    "unlabeled_ratio": None,
}
BASELINE_REGULARIZER_WEIGHT = 2.0264726922580607
BASELINE_SUPERVISED_WEIGHT = 1.0

# ``batch_sampler='128:4'``, ``lr``, and ``classifier_lr`` of the same trial.
BASELINE_BATCH_SIZE = 128
BASELINE_SAMPLER_M = 4
BASELINE_LR = 1.0283387717258826e-05
BASELINE_CLASSIFIER_LR = 0.03323081571675744
BASELINE_LOSS = "ProxyAnchorLoss"
BASELINE_LOSS_PARAMS = {"alpha": 1.291216318983798, "margin": 0.46850485585776586}
BASELINE_LABELED_PER_CLASS = 16

# A logit this large puts sigmoid within ~2% of a hard 0/1, where both the
# entropy and the EM cross entropy have effectively no gradient left.
SATURATED_LOGIT = 4.0


def _median(values):
    return float(statistics.median(values)) if values else float("nan")


# --------------------------------------------------------------------------- #
# Frozen features
# --------------------------------------------------------------------------- #


@dataclass
class FeatureSource:
    """Frozen backbone features plus the labels they are aligned with."""

    name: str
    features: torch.Tensor
    labels: torch.Tensor
    cache_path: str
    nn_label_agreement: float

    @property
    def backbone_dim(self):
        return int(self.features.shape[1])


def find_cached_feature_matrix(dataset_name, num_rows, data_root):
    """Return the cached feature matrix whose row count matches the dataset.

    ``_indexed_feature_matrix_path`` names caches by a hash of the run's cache
    key, so the file cannot be found by dataset name alone. Row count is the
    identifying property that matters here: the matrix is indexed by dataset
    position, so a matching row count means row ``i`` is sample ``i``. That
    assumption is verified rather than trusted - see ``nn_label_agreement``.
    """

    cache_dir = Path(data_root) / dataset_name / "backbone_cache"
    candidates = sorted(cache_dir.glob("v*/*/*.features.npy"), reverse=True)
    for path in candidates:
        if np.load(path, mmap_mode="r").shape[0] == num_rows:
            return path
    return None


def nearest_neighbor_label_agreement(features, labels, sample_size=4096, seed=0):
    """Return 1-NN label agreement, which detects a feature/label misalignment.

    Frozen DINOv2 features of a retrieval dataset land far above chance; a value
    near ``1 / num_classes`` means the cache rows do not correspond to the
    dataset positions the labels were read from, which would silently invalidate
    every supervised gradient in this benchmark.
    """

    generator = np.random.default_rng(seed)
    if len(features) > sample_size:
        rows = generator.choice(len(features), size=sample_size, replace=False)
    else:
        rows = np.arange(len(features))
    rows = torch.as_tensor(np.sort(rows))
    sampled = F.normalize(features[rows].float(), dim=1)
    similarity = sampled @ sampled.t()
    similarity.fill_diagonal_(-2.0)
    neighbors = similarity.argmax(dim=1)
    sampled_labels = labels[rows]
    return float((sampled_labels[neighbors] == sampled_labels).float().mean())


def load_feature_source(dataset_name, split, data_root, device):
    """Load a dataset's labels and its cached frozen features, position-aligned."""

    dataset_class = get_dataset_class(dataset_name)
    dataset = dataset_class(
        root=str(Path(data_root) / dataset_name),
        split=split,
        transform=None,
        download=False,
    )
    labels = torch.as_tensor(np.asarray(dataset.labels), dtype=torch.long)
    cache_path = find_cached_feature_matrix(dataset_name, len(labels), data_root)
    if cache_path is None:
        raise FileNotFoundError(
            f"No cached backbone features with {len(labels)} rows under "
            f"{Path(data_root) / dataset_name / 'backbone_cache'}; run a "
            f"--use_cache training job on {dataset_name} first, or pass a dataset "
            "whose cache exists"
        )
    features = torch.from_numpy(np.load(cache_path)).float()
    agreement = nearest_neighbor_label_agreement(features, labels)
    return FeatureSource(
        name=f"{dataset_name}/{split}",
        features=features.to(device),
        labels=labels.to(device),
        cache_path=str(cache_path),
        nn_label_agreement=agreement,
    )


def make_k_shot_split(labels, labeled_per_class, seed):
    """Return ``(labeled_positions, unlabeled_positions)`` for k labeled per class.

    This is the runs' ``label_sampling_mode='class_subset_k_shot'`` with
    ``labeled_fraction=1.0``: every class keeps ``labeled_per_class`` labeled
    samples and contributes the rest to the unlabeled pool.
    """

    generator = np.random.default_rng(seed)
    labels = labels.cpu().numpy()
    labeled, unlabeled = [], []
    for value in np.unique(labels):
        positions = np.flatnonzero(labels == value)
        generator.shuffle(positions)
        labeled.append(positions[:labeled_per_class])
        unlabeled.append(positions[labeled_per_class:])
    return (
        np.sort(np.concatenate(labeled)),
        np.sort(np.concatenate(unlabeled)),
    )


# --------------------------------------------------------------------------- #
# The trainable half of a frozen-backbone run
# --------------------------------------------------------------------------- #


class FrozenFeatureModel(nn.Module):
    """``DinoWrapper.fc`` plus ``project_features``' L2 normalization.

    ``SeraphRegularizer.configure_model`` reads ``feat_dim`` and attaches its
    posterior head as a submodule, so this stands in for the real model without
    loading a backbone; the backbone is frozen and precomputed in the runs this
    mirrors anyway.
    """

    def __init__(self, backbone_dim, feat_dim):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.fc = nn.Linear(int(backbone_dim), self.feat_dim)

    def forward(self, features):
        return F.normalize(self.fc(features), p=2.0, dim=1)


class BatchStream:
    """Deterministic labeled (m-per-class) and unlabeled batches.

    Batches are a pure function of the step index, so two conditions measured at
    the same step see the same data and their supervised gradients are identical
    by construction - which makes the supervised norm a control column rather
    than another source of variance.
    """

    def __init__(self, labels, labeled_positions, unlabeled_positions, batch_size, sampler_m, seed):
        self.labels = labels.cpu().numpy()
        self.labeled_positions = np.asarray(labeled_positions)
        self.unlabeled_positions = np.asarray(unlabeled_positions)
        self.batch_size = int(batch_size)
        self.sampler_m = int(sampler_m)
        self.seed = int(seed)
        labeled_labels = self.labels[self.labeled_positions]
        self.positions_by_class = {
            value: self.labeled_positions[labeled_labels == value]
            for value in np.unique(labeled_labels)
        }
        self.classes = np.asarray(sorted(self.positions_by_class))

    def labeled_batch(self, step):
        """``batch_size // sampler_m`` classes, ``sampler_m`` samples each."""

        generator = np.random.default_rng((self.seed, step, 0))
        classes_per_batch = max(1, self.batch_size // self.sampler_m)
        chosen = generator.choice(
            self.classes,
            size=min(classes_per_batch, len(self.classes)),
            replace=False,
        )
        indices = []
        for value in chosen:
            positions = self.positions_by_class[value]
            replace = len(positions) < self.sampler_m
            indices.append(
                generator.choice(positions, size=self.sampler_m, replace=replace)
            )
        return np.concatenate(indices)

    def unlabeled_batch(self, step, size):
        generator = np.random.default_rng((self.seed, step, 1))
        size = min(int(size), len(self.unlabeled_positions))
        return generator.choice(self.unlabeled_positions, size=size, replace=False)


# --------------------------------------------------------------------------- #
# Conditions
# --------------------------------------------------------------------------- #


@dataclass
class Condition:
    axis: str
    label: str
    regularizer_params: dict = field(default_factory=dict)
    regularizer_weight: float = BASELINE_REGULARIZER_WEIGHT


def _condition(axis, label, weight=BASELINE_REGULARIZER_WEIGHT, **overrides):
    params = dict(BASELINE_REGULARIZER_PARAMS)
    params.update(overrides)
    return Condition(axis, label, params, weight)


def build_axis_conditions(axis, values=None):
    """Return the conditions of one sweep axis, all inside the HPO search space.

    ``values`` replaces the axis' default grid, which is how a coarse sweep is
    followed by a fine one around whichever region it pointed at.
    """

    if axis == "kappa":
        # ssl_config...kappa was searched log-uniformly over [0.01, 5].
        return [
            _condition(axis, f"kappa={value:g}", kappa=value)
            for value in (values or (0.01, 0.05, 0.2, 0.5, 1.72, 5.0))
        ]
    if axis == "kappa_mode":
        # batch_std divides kappa by the spread of the batch's own distances, so
        # the same kappa means something different in the two modes.
        return [
            _condition(axis, f"{mode}/kappa={value:g}", kappa=value, kappa_mode=mode)
            for mode in ("fixed", "batch_std")
            for value in (values or (0.5, 1.72, 5.0))
        ]
    if axis == "prior":
        # positive_pair_prior places eta at that quantile of the batch distances,
        # so it decides how many pairs sit near the decision boundary.
        return [
            _condition(axis, f"prior={value:g}", positive_pair_prior=value)
            for value in (values or (0.01, 0.05, 0.1, 0.3, 0.5, 0.9))
        ]
    if axis == "objective":
        # em_sharpening=1.0 is included as the control the validation permits but
        # the M-step cannot use: q = p exactly, so the cross entropy is taken
        # against the posterior's own detached value and its gradient is
        # identically zero. Any non-zero reading in that row is a bug.
        return [_condition(axis, "entropy", objective="entropy")] + [
            _condition(axis, f"em/sharpening={value:g}", objective="em", em_sharpening=value)
            for value in (1.0, 1.5, 2.0, 4.0, 8.0)
        ]
    if axis == "metric":
        return [
            _condition(axis, f"metric={value}", metric=value, metric_dim=None)
            for value in ("identity", "scale", "diag", "lowrank")
        ]
    if axis == "unlabeled_batch":
        return [
            _condition(axis, f"unlabeled_batch={int(value)}", unlabeled_batch_size=int(value))
            for value in (values or (32, 64, 128, 256, 512))
        ]
    if axis == "blocks":
        return [
            _condition(axis, f"include_lu={value}", include_labeled_unlabeled=value)
            for value in (False, True)
        ]
    if axis == "threshold":
        return [_condition(axis, "quantile", threshold_mode="quantile")] + [
            _condition(axis, f"fixed/eta={value:g}", threshold_mode="fixed", threshold=value)
            for value in (values or (0.05, 0.1, 0.5, 1.0, 2.0))
        ]
    if axis == "weight":
        # regularizer_weight was searched log-uniformly over [0.5, 5].
        return [
            _condition(axis, f"weight={value:g}", weight=value)
            for value in (values or (0.5, 1.0, 2.03, 5.0))
        ]
    raise ValueError(f"unknown axis {axis!r}")


AXES = (
    "kappa",
    "kappa_mode",
    "prior",
    "objective",
    "metric",
    "unlabeled_batch",
    "blocks",
    "threshold",
    "weight",
)


# --------------------------------------------------------------------------- #
# Measurement
# --------------------------------------------------------------------------- #


def gradient_norm(loss, parameters):
    """``||grad(loss)||`` without touching ``.grad``, as the trainer measures it."""

    parameters = [parameter for parameter in parameters if parameter.requires_grad]
    if not parameters or loss is None or not loss.requires_grad:
        return 0.0
    gradients = [
        gradient
        for gradient in torch.autograd.grad(
            loss, parameters, allow_unused=True, retain_graph=True
        )
        if gradient is not None
    ]
    if not gradients:
        return 0.0
    squared = sum(gradient.detach().float().square().sum() for gradient in gradients)
    return float(squared.clamp_min(0.0).sqrt())


def saturation_probe(head, embeddings, supervised_embeddings, include_labeled_unlabeled):
    """Read the posterior the objective is currently descending, without changing it.

    ``resolve_threshold`` mutates the running ``eta``, so the probe reads the
    buffers instead of calling it: this is a report on the state the measured
    step used, not another update of it.
    """

    with torch.no_grad():
        projected = head.project(embeddings)
        distances = pairwise_squared_distances(projected)
        pairs = torch.triu_indices(len(projected), len(projected), offset=1, device=distances.device)
        blocks = {"uu": distances[pairs[0], pairs[1]]}
        if include_labeled_unlabeled:
            blocks["lu"] = cross_squared_distances(
                head.project(supervised_embeddings), projected
            ).flatten()
        probe = {}
        for block, squared_distances in blocks.items():
            if head.threshold_mode == "fixed":
                threshold = head.threshold
            else:
                threshold = getattr(head, f"threshold_{block}")
            kappa = head.inverse_temperature(squared_distances)
            logits = head.posterior_logits(squared_distances, threshold, kappa)
            probe[block] = {
                "mean_abs_logit": float(logits.abs().mean()),
                "saturated_fraction": float((logits.abs() > SATURATED_LOGIT).float().mean()),
                "pair_entropy": float(binary_entropy_from_logits(logits).mean()),
                "positive_pair_rate": float((logits > 0).float().mean()),
                "threshold": float(threshold),
                "kappa_effective": float(kappa),
                "mean_squared_distance": float(squared_distances.mean()),
            }
        return probe


def measure_condition(
    condition,
    stage_state,
    source,
    stream,
    split_sizes,
    label_lookup,
    num_classes,
    args,
):
    """Run one condition from the stage's head weights and return its statistics."""

    device = args.device
    model = FrozenFeatureModel(source.backbone_dim, args.feat_dim).to(device)
    model.load_state_dict(stage_state["model"])
    criterion = engine.make_named_loss(
        name=args.loss,
        params=args.loss_params,
        num_classes=num_classes,
        embedding_size=args.feat_dim,
    ).to(device)
    if stage_state["criterion"] is not None:
        criterion.load_state_dict(stage_state["criterion"])

    regularizer = SeraphRegularizer(
        regularizer_weight=condition.regularizer_weight,
        supervised_weight=BASELINE_SUPERVISED_WEIGHT,
        **condition.regularizer_params,
    )
    regularizer.configure_model(
        student_model=model,
        train_dataset=None,
        split=None,
        train_labels_mapper={index: index for index in range(num_classes)},
        device=device,
    )
    # ``build_dataset`` would size these from the split while building the
    # unlabeled loader; this benchmark feeds batches itself, so the population
    # sizes the lu/uu block weights need are set from the same split directly.
    regularizer.labeled_pool_size, regularizer.unlabeled_pool_size = split_sizes
    head = model.seraph_posterior
    projection_parameters = list(model.fc.parameters())
    head_parameters = list(head.parameters())

    unlabeled_batch_size = condition.regularizer_params["unlabeled_batch_size"]

    def embed(step):
        labeled_indices = stream.labeled_batch(step)
        unlabeled_indices = stream.unlabeled_batch(step, unlabeled_batch_size)
        # Production runs SERAPH through a joint forward with the labeled half.
        inputs = torch.cat(
            [
                source.features[torch.as_tensor(labeled_indices, device=device)],
                source.features[torch.as_tensor(unlabeled_indices, device=device)],
            ]
        )
        embeddings = model(inputs)
        labels = label_lookup[source.labels[torch.as_tensor(labeled_indices, device=device)]]
        return embeddings[: len(labeled_indices)], labels, embeddings[len(labeled_indices) :]

    # The quantile eta is a running estimate seeded by its first batch. Measuring
    # before it settles would report the transient rather than the eta a run
    # spends its epochs at.
    with torch.no_grad():
        for step in range(args.settle_batches):
            supervised_embeddings, _, regularizer_embeddings = embed(step)
            regularizer.compute_loss(
                model,
                None,
                None,
                device,
                supervised_embeddings=supervised_embeddings,
                supervised_labels=None,
                regularizer_embeddings=regularizer_embeddings,
            )

    samples = []
    for offset in range(args.measure_batches):
        step = args.settle_batches + offset
        supervised_embeddings, supervised_labels, regularizer_embeddings = embed(step)
        supervised_loss = criterion(supervised_embeddings, supervised_labels)
        regularization_loss = regularizer.compute_loss(
            model,
            None,
            None,
            device,
            supervised_embeddings=supervised_embeddings,
            supervised_labels=supervised_labels,
            regularizer_embeddings=regularizer_embeddings,
        )
        model_norms = engine.measure_gradient_component_norms(
            supervised_loss,
            regularization_loss,
            model.parameters(),
            supervised_weight=BASELINE_SUPERVISED_WEIGHT,
            regularizer_weight=condition.regularizer_weight,
        )
        projection_norms = engine.measure_gradient_component_norms(
            supervised_loss,
            regularization_loss,
            projection_parameters,
            supervised_weight=BASELINE_SUPERVISED_WEIGHT,
            regularizer_weight=condition.regularizer_weight,
        )
        combined_norm = gradient_norm(
            BASELINE_SUPERVISED_WEIGHT * supervised_loss
            + condition.regularizer_weight * regularization_loss,
            list(model.parameters()),
        )
        diagnostics = engine.make_gradient_contribution_diagnostics(
            model_norms, combined_norm
        )
        head_norm = (
            gradient_norm(
                condition.regularizer_weight * regularization_loss, head_parameters
            )
            if head_parameters
            else 0.0
        )
        probe = saturation_probe(
            head,
            regularizer_embeddings,
            supervised_embeddings,
            condition.regularizer_params["include_labeled_unlabeled"],
        )
        samples.append(
            {
                "supervised_loss": float(supervised_loss.detach()),
                "regularization_loss": float(regularization_loss.detach()),
                "supervised_norm": model_norms.supervised,
                "regularizer_norm": model_norms.regularizer,
                "ratio": (
                    model_norms.regularizer / model_norms.supervised
                    if model_norms.supervised > 0
                    else float("inf")
                ),
                "share": diagnostics["train/gradient_contribution/regularizer_fraction"],
                "cosine": diagnostics["train/gradient_contribution/cosine_similarity"],
                "projection_supervised_norm": projection_norms.supervised,
                "projection_regularizer_norm": projection_norms.regularizer,
                "projection_ratio": (
                    projection_norms.regularizer / projection_norms.supervised
                    if projection_norms.supervised > 0
                    else float("inf")
                ),
                "head_share": (
                    head_norm / model_norms.regularizer
                    if model_norms.regularizer > 0
                    else 0.0
                ),
                **{f"uu_{name}": value for name, value in probe["uu"].items()},
                **{
                    f"lu_{name}": value
                    for name, value in probe.get("lu", {}).items()
                },
            }
        )

    result = {
        "axis": condition.axis,
        "condition": condition.label,
        "regularizer_weight": condition.regularizer_weight,
        "regularizer_params": condition.regularizer_params,
    }
    for name in samples[0]:
        result[name] = _median([sample[name] for sample in samples])
    return result


def train_supervised(model, criterion, optimizer, criterion_optimizer, source, stream, label_lookup, steps, start_step):
    """Advance the supervised objective, so a stage measures a partly trained head.

    SERAPH's share is a ratio, and the denominator is what training moves: the
    supervised gradient is largest at initialization and shrinks as the labeled
    batch gets organized. Stages are what let the sweep separate "the entropy
    term is strong" from "the supervised term has gone quiet".
    """

    device = next(model.parameters()).device
    for offset in range(steps):
        indices = stream.labeled_batch(start_step + offset)
        indices = torch.as_tensor(indices, device=device)
        embeddings = model(source.features[indices])
        loss = criterion(embeddings, label_lookup[source.labels[indices]])
        optimizer.zero_grad(set_to_none=True)
        if criterion_optimizer is not None:
            criterion_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if criterion_optimizer is not None:
            criterion_optimizer.step()
    return float(loss.detach()) if steps else float("nan")


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

COLUMNS = (
    ("condition", "condition", "{:<26}", 26),
    ("supervised_norm", "sup_norm", "{:>10.4g}", 10),
    ("regularizer_norm", "reg_norm", "{:>10.4g}", 10),
    ("ratio", "reg/sup", "{:>9.3g}", 9),
    ("projection_ratio", "proj r/s", "{:>9.3g}", 9),
    ("head_share", "on_head", "{:>8.2f}", 8),
    ("share", "share", "{:>7.3f}", 7),
    ("cosine", "cos", "{:>7.3f}", 7),
    ("uu_pair_entropy", "H(pair)", "{:>8.4f}", 8),
    ("uu_mean_abs_logit", "|logit|", "{:>8.3g}", 8),
    ("uu_saturated_fraction", "sat", "{:>6.2f}", 6),
    ("uu_kappa_effective", "kappa_eff", "{:>10.4g}", 10),
    ("uu_threshold", "eta", "{:>9.4g}", 9),
)


LEGEND = """
columns: reg/sup   ||grad(w_r L_seraph)|| / ||grad(w_s L_sup)|| over the model parameters
         proj r/s  the same ratio over the retrieval projection alone
         on_head   fraction of the regularizer gradient landing on SERAPH's own metric B
         share     regularizer_fraction, as production logs it
         cos       cosine between the two gradients (0 = orthogonal, <0 = opposing)
         |logit|   mean |kappa (eta - d^2)|; the per-pair gradient peaks near 1.54
         sat       fraction of pairs past |logit| > 4, where the term is inert
"""


def print_stage_report(stage, results):
    print()
    print("=" * 132)
    print(
        f"stage: {stage['warmup_steps']} supervised steps  "
        f"(supervised loss {stage['supervised_loss']:.4g})"
    )
    print("=" * 132)
    header = "".join(
        f"{title:<{width}}" if index == 0 else f"{title:>{width}} "
        for index, (_key, title, _fmt, width) in enumerate(COLUMNS)
    )
    current_axis = None
    for result in results:
        if result["axis"] != current_axis:
            current_axis = result["axis"]
            print()
            print(f"-- {current_axis} " + "-" * (128 - len(current_axis)))
            print(header)
        row = []
        for index, (key, _title, fmt, width) in enumerate(COLUMNS):
            value = result[key]
            try:
                text = fmt.format(value)
            except (TypeError, ValueError):
                text = str(value)
            row.append(f"{text:<{width}}" if index == 0 else f"{text:>{width}} ")
        print("".join(row))


def print_ranking(results):
    print()
    print("=" * 132)
    print("Highest regularizer/supervised gradient ratio, all stages and axes")
    print("=" * 132)
    ranked = sorted(
        results,
        key=lambda result: result["ratio"] if math.isfinite(result["ratio"]) else -1.0,
        reverse=True,
    )
    print(f"{'stage':>6} {'axis':<16} {'condition':<26} {'reg/sup':>9} {'proj r/s':>9} {'on_head':>8} {'|logit|':>8}")
    for result in ranked[:15]:
        print(
            f"{result['warmup_steps']:>6} {result['axis']:<16} {result['condition']:<26} "
            f"{result['ratio']:>9.3g} {result['projection_ratio']:>9.3g} "
            f"{result['head_share']:>8.2f} {result['uu_mean_abs_logit']:>8.3g}"
        )


# --------------------------------------------------------------------------- #


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="Cars196")
    parser.add_argument("--split", default="train")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--feat-dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=BASELINE_BATCH_SIZE)
    parser.add_argument("--sampler-m", type=int, default=BASELINE_SAMPLER_M)
    parser.add_argument("--labeled-per-class", type=int, default=BASELINE_LABELED_PER_CLASS)
    parser.add_argument("--loss", default=BASELINE_LOSS)
    parser.add_argument(
        "--loss-params",
        type=json.loads,
        default=json.dumps(BASELINE_LOSS_PARAMS),
        help="JSON object of loss constructor parameters",
    )
    parser.add_argument("--lr", type=float, default=BASELINE_LR)
    parser.add_argument("--classifier-lr", type=float, default=BASELINE_CLASSIFIER_LR)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        nargs="+",
        default=[0, 200, 2000],
        help="supervised-only steps before each measured stage (cumulative)",
    )
    parser.add_argument("--settle-batches", type=int, default=20)
    parser.add_argument("--measure-batches", type=int, default=12)
    parser.add_argument("--axes", nargs="+", choices=AXES, default=list(AXES))
    parser.add_argument(
        "--values",
        type=float,
        nargs="+",
        default=None,
        help="replace the swept grid of a single --axes entry",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default=None, help="write the full results as JSON")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="keep the INFO logs the loss and regularizer emit per condition",
    )
    args = parser.parse_args(argv)
    if isinstance(args.loss_params, str):
        args.loss_params = json.loads(args.loss_params)
    if args.values and len(args.axes) != 1:
        parser.error("--values replaces one axis' grid, so pass exactly one --axes")
    return args


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if not args.verbose:
        # Every condition rebuilds the loss and the posterior head, and each logs
        # its configuration at INFO; that is one line per row of the table.
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    source = load_feature_source(args.dataset, args.split, args.data_root, args.device)
    labeled_positions, unlabeled_positions = make_k_shot_split(
        source.labels, args.labeled_per_class, args.seed
    )
    unique_labels = torch.unique(source.labels)
    num_classes = int(len(unique_labels))
    # Contiguous class ids, as ``make_label_lookup_tensor`` provides in a run.
    label_lookup = torch.full(
        (int(unique_labels.max()) + 1,), -1, dtype=torch.long, device=args.device
    )
    label_lookup[unique_labels] = torch.arange(num_classes, device=args.device)

    print(f"features:  {source.name} {tuple(source.features.shape)} from {source.cache_path}")
    print(
        f"alignment: 1-NN label agreement {source.nn_label_agreement:.3f} "
        f"over {num_classes} classes (chance {1.0 / num_classes:.3f})"
    )
    if source.nn_label_agreement < 5.0 / num_classes:
        raise RuntimeError(
            "cached features look unaligned with the dataset labels; refusing to "
            "report supervised gradients measured against the wrong labels"
        )
    print(
        f"split:     {len(labeled_positions)} labeled ({args.labeled_per_class}/class), "
        f"{len(unlabeled_positions)} unlabeled"
    )
    print(
        f"objective: {args.loss}{args.loss_params}, batch {args.batch_size}:{args.sampler_m}, "
        f"device {args.device}"
    )

    stream = BatchStream(
        source.labels,
        labeled_positions,
        unlabeled_positions,
        args.batch_size,
        args.sampler_m,
        args.seed,
    )
    conditions = [
        condition
        for axis in args.axes
        for condition in build_axis_conditions(axis, args.values)
    ]
    split_sizes = (len(labeled_positions), len(unlabeled_positions))

    # One head, trained forward through the requested stages, so stage N is stage
    # N-1 plus more supervised steps rather than an independent run.
    model = FrozenFeatureModel(source.backbone_dim, args.feat_dim).to(args.device)
    criterion = engine.make_named_loss(
        name=args.loss,
        params=args.loss_params,
        num_classes=num_classes,
        embedding_size=args.feat_dim,
    ).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_parameters = list(criterion.parameters())
    criterion_optimizer = (
        torch.optim.Adam(criterion_parameters, lr=args.classifier_lr)
        if criterion_parameters
        else None
    )

    print(LEGEND)

    all_results = []
    trained_steps = 0
    for warmup_steps in sorted(set(args.warmup_steps)):
        supervised_loss = train_supervised(
            model,
            criterion,
            optimizer,
            criterion_optimizer,
            source,
            stream,
            label_lookup,
            steps=warmup_steps - trained_steps,
            # Warmup draws from a step range the measurement never visits, so a
            # stage is not measured on the batches it just fitted.
            start_step=10_000 + trained_steps,
        )
        trained_steps = warmup_steps
        stage_state = {
            "model": {name: tensor.detach().clone() for name, tensor in model.state_dict().items()},
            "criterion": (
                {name: tensor.detach().clone() for name, tensor in criterion.state_dict().items()}
                if criterion_parameters
                else None
            ),
        }
        stage_results = []
        for condition in conditions:
            result = measure_condition(
                condition,
                stage_state,
                source,
                stream,
                split_sizes,
                label_lookup,
                num_classes,
                args,
            )
            result["warmup_steps"] = warmup_steps
            stage_results.append(result)
        stage = {
            "warmup_steps": warmup_steps,
            "supervised_loss": (
                stage_results[0]["supervised_loss"] if stage_results else supervised_loss
            ),
        }
        print_stage_report(stage, stage_results)
        all_results.extend(stage_results)

    print_ranking(all_results)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "setup": {
                        "dataset": source.name,
                        "cache_path": source.cache_path,
                        "nn_label_agreement": source.nn_label_agreement,
                        "num_classes": num_classes,
                        "labeled": len(labeled_positions),
                        "unlabeled": len(unlabeled_positions),
                        "baseline_regularizer_params": BASELINE_REGULARIZER_PARAMS,
                        "baseline_regularizer_weight": BASELINE_REGULARIZER_WEIGHT,
                        "args": {
                            key: value
                            for key, value in vars(args).items()
                            if key != "loss_params"
                        },
                        "loss_params": args.loss_params,
                    },
                    "results": all_results,
                },
                indent=2,
            )
        )
        print(f"\nWrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
