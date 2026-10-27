"""Measure whether ``hoffer_entropy``'s ``distance_scale`` earns its search dimension.

Hoffer & Ailon (arXiv:1611.01449) Eq. 1 has no temperature: the softmax logit is
``-||F(x) - F(z_i)||^2`` over *unnormalized* embeddings, so the network sets its
own sharpness by growing the embedding norm -- scaling ``F`` by ``a`` scales every
logit by ``a^2``. This repository feeds the regularizer L2-normalized embeddings,
which pins ``d^2 = 2 - 2 cos`` into ``[0, 4]`` and removes that degree of freedom;
``distance_scale`` was added to restore it. This benchmark asks whether that knob
does anything the existing ``regularizer_weight`` does not already do.

Note that the released code
(https://github.com/eladhoffer/SemiSupContrast, ``ContrastingMinEntropyCriterion``)
uses ``nn.PairwiseDistance(2)`` followed by ``MulConstant(-1)``, i.e. ``-d``
*unsquared*, which is not Eq. 1. ``distance_power=1`` reproduces that variant so
the two can be compared; ``distance_power=2`` is the paper's form and this
repository's.

The redundancy is a real risk, not a hypothetical one. Write ``l_j = -s d_j^2``.
Near a uniform softmax the entropy gradient linearizes to

    dH/d(d_j^2) ~ (s^2 / C) * (mean_k d_k^2 - d_j^2),

so in the flat regime ``distance_scale`` and ``regularizer_weight`` are the same
knob: ``w`` at scale ``s`` behaves like ``w s^2 / s'^2`` at scale ``s'``. The
scale only buys something once the softmax leaves that regime and the term starts
committing to one nearest reference instead of applying a quadratic pull toward
the mean distance. So the question is measurable: **does the searched range cross
out of the flat regime, and does anything change when it does?**

Three stages answer that, all on real cached Cars196 DINOv2 features:

``probe``
    Per ``distance_scale``, from a head at several points of supervised training:
    the softmax sharpness it produces (``p_max`` against the uniform ``1/C``,
    normalized entropy), whether the reference the term sharpens toward is even
    the right class (``argmax_purity``), and the regularizer's share of the model
    gradient, measured with ``engine.measure_gradient_component_norms`` -- the
    function the trainer itself logs from. The ``grad_slope`` column is the local
    log-log slope of that gradient norm in ``s``: ``~2`` is the flat regime where
    the scale is redundant with the weight, ``<2`` means the softmax has started
    to sharpen and the knob is doing something of its own.

``degeneracy``
    The direct test of the redundancy claim: pairs of training arms that hold
    ``regularizer_weight * distance_scale^2`` constant. If those arms land on the
    same held-out retrieval, the two knobs are one knob over that range.

``train``
    Full short training arms -- a supervised-only control, a scale ladder at a
    fixed weight, and the released code's ``-d`` form at ``--reference-scales`` --
    scored by held-out precision@1 on the Cars196 test split.

Setup mirrors ``scripts/bench_lrml_collapse.py`` and
``scripts/bench_seraph_gradient_contribution.py``: real cached frozen DINOv2
features, a trainable ``Linear + L2 normalize`` head standing in for
``DinoWrapper.fc``/``project_features``, a ``class_subset_k_shot`` split, and the
repository's own ``HofferEntropyRegularizer`` for the loss itself (the local
closed form used by the training arms is asserted equal to it at startup).

Examples::

    python scripts/bench_hoffer_distance_scale.py
    python scripts/bench_hoffer_distance_scale.py --stages probe
    python scripts/bench_hoffer_distance_scale.py --stages train --scales 1 4 16
    python scripts/bench_hoffer_distance_scale.py --dataset CUB --epochs 10
"""

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import engine  # noqa: E402
from training.semi_supervised import HofferEntropyRegularizer  # noqa: E402

from bench_seraph_gradient_contribution import (  # noqa: E402
    BatchStream,
    FeatureSource,
    FrozenFeatureModel,
    make_k_shot_split,
    nearest_neighbor_label_agreement,
)
from utils.dataset_protocols import get_dataset_class  # noqa: E402


# ``configs/phase_1/ssl_methods/hoffer/cars/*.json`` plus
# ``configs/ssl_methods/hoffer.json``: the shipped Cars196 operating point.
DEFAULT_FEAT_DIM = 768
DEFAULT_LABELED_PER_CLASS = 16
DEFAULT_BATCH_SIZE = 128
DEFAULT_SAMPLER_M = 4
DEFAULT_REFERENCE_SETS = 1
DEFAULT_REGULARIZER_WEIGHT = 0.5
# ``configs/runs/hoffer/cars.json`` searches distance_scale log-uniformly over
# [0.1, 16], so the ladder brackets that range and adds one point past it.
DEFAULT_SCALES = (0.1, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0)

LEGEND = """
columns
  dist_span  p95-p05 of the distance term one unlabeled row sees, before scaling
             (d^2 for the paper's form, d for the released code's), so it is the
             per-row spread the softmax reads once the offset cancels
  logit_span distance_scale * dist_span: the softmax's actual logit range
  p_max      mean largest softmax probability
  x_unif     p_max / (1/C): 1.0 = the term cannot tell the references apart
  H/logC     normalized entropy; 1.0 = uniform, 0.0 = one reference wins
  purity     fraction of unlabeled rows whose nearest reference is their own class
  grad_ratio || grad(w * H) || / || grad(L_sup) || over the model parameters
  grad_slope local d log(grad) / d log(s); ~2.0 = redundant with regularizer_weight
""".strip()


# --------------------------------------------------------------------------- #
# Feature source
# --------------------------------------------------------------------------- #


def resolve_feature_source(dataset_name, split, data_root, device, backbone=None):
    """Load labels plus the cached feature matrix that is actually aligned with them.

    ``find_cached_feature_matrix`` in the SERAPH benchmark takes the first cache
    whose row count matches, which on this machine is a stale ``dinov2_vitg14``
    Cars196 matrix whose rows do not correspond to the dataset positions (1-NN
    label agreement at chance). Row count alone is therefore not enough to
    identify a usable cache; this picks the candidate with the best agreement and
    reports it, so a misaligned cache is skipped rather than silently used.
    """

    dataset = get_dataset_class(dataset_name)(
        root=str(Path(data_root) / dataset_name),
        split=split,
        transform=None,
        download=False,
    )
    labels = torch.as_tensor(np.asarray(dataset.labels), dtype=torch.long)
    cache_dir = Path(data_root) / dataset_name / "backbone_cache"
    candidates = sorted(cache_dir.glob("v*/*/*.features.npy"), reverse=True)
    if backbone is not None:
        candidates = [path for path in candidates if path.parent.name == backbone]

    best = None
    for path in candidates:
        if np.load(path, mmap_mode="r").shape[0] != len(labels):
            continue
        features = torch.from_numpy(np.load(path)).float()
        agreement = nearest_neighbor_label_agreement(features, labels)
        if best is None or agreement > best[1]:
            best = (path, agreement, features)
    if best is None:
        raise FileNotFoundError(
            f"no cached backbone features with {len(labels)} rows under {cache_dir}; "
            f"run a --use_cache training job on {dataset_name} first"
        )
    path, agreement, features = best
    num_classes = int(len(torch.unique(labels)))
    if agreement < 5.0 / num_classes:
        raise RuntimeError(
            f"the best Cars196 cache ({path}) has 1-NN label agreement {agreement:.3f}, "
            f"near the {1.0 / num_classes:.3f} chance level: no cache here is aligned "
            "with the dataset labels"
        )
    return FeatureSource(
        name=f"{dataset_name}/{split}",
        features=features.to(device),
        labels=labels.to(device),
        cache_path=str(path),
        nn_label_agreement=agreement,
    )


# --------------------------------------------------------------------------- #
# The Hoffer entropy term
# --------------------------------------------------------------------------- #


def hoffer_entropy(unlabeled, references, reference_sets, distance_scale, distance_power=2):
    """``HofferEntropyRegularizer.compute_loss``'s tail, as a plain function.

    ``distance_power`` is the only addition: ``2`` is the paper's Eq. 1 and this
    repository's form, ``1`` is the released ``ContrastingMinEntropyCriterion``'s
    ``-d`` from ``nn.PairwiseDistance(2)`` followed by ``nn.MulConstant(-1)``,
    which does not match Eq. 1.
    ``check_matches_regularizer`` asserts the ``distance_power=2`` branch against
    the real class, so the training arms measure the shipped objective.
    """

    num_classes = len(references) // reference_sets
    sets = references.reshape(num_classes, reference_sets, -1).permute(1, 0, 2)
    squared = (unlabeled[None, :, None, :] - sets[:, None, :, :]).pow(2).sum(dim=-1)
    if distance_power == 2:
        distances = squared
    elif distance_power == 1:
        distances = squared.clamp_min(1e-12).sqrt()
    else:
        raise ValueError(f"distance_power must be 1 or 2, got {distance_power}")
    log_probabilities = F.log_softmax(-distance_scale * distances, dim=-1)
    entropy = -(log_probabilities.exp() * log_probabilities).sum(dim=-1)
    # ``distances`` rather than ``squared``: every caller reports the spread of
    # what the softmax actually reads, which is not sqrt(spread of d^2).
    return entropy.mean(), distances, log_probabilities


def check_matches_regularizer(device, num_classes=17, reference_sets=2, batch=11, dim=13):
    """Assert the local closed form reproduces the shipped regularizer exactly."""

    generator = torch.Generator(device="cpu").manual_seed(0)
    embeddings = F.normalize(
        torch.randn(batch + num_classes * reference_sets, dim, generator=generator), dim=1
    ).to(device)
    is_reference = torch.zeros(len(embeddings), dtype=torch.bool)
    is_reference[batch:] = True

    regularizer = HofferEntropyRegularizer(
        regularizer_weight=1.0,
        supervised_weight=1.0,
        distance_scale=3.5,
        reference_sets=reference_sets,
    )
    regularizer.class_candidates = [np.array([index]) for index in range(num_classes)]
    reference = regularizer.compute_loss(
        torch.nn.Identity(), None, (embeddings, is_reference), device
    )
    local, _, _ = hoffer_entropy(
        embeddings[: batch],
        embeddings[batch:],
        reference_sets,
        distance_scale=3.5,
        distance_power=2,
    )
    if not torch.allclose(reference, local, atol=1e-6):
        raise AssertionError(
            f"local hoffer_entropy {float(local):.8f} != HofferEntropyRegularizer "
            f"{float(reference):.8f}"
        )


class ReferenceStream:
    """``HofferReferenceBatchSampler``: one uniform draw per class, per step.

    Every class is present in every batch and every denominator, and the draw is
    a pure function of the step, so two conditions measured at the same step see
    the same references.
    """

    def __init__(self, labels, labeled_positions, reference_sets, seed):
        labels = labels.cpu().numpy()
        labeled_positions = np.asarray(labeled_positions)
        labeled_labels = labels[labeled_positions]
        self.classes = np.unique(labeled_labels)
        self.candidates = [labeled_positions[labeled_labels == value] for value in self.classes]
        self.reference_sets = int(reference_sets)
        self.seed = int(seed)

    @property
    def num_classes(self):
        return len(self.classes)

    def references(self, step):
        """Positions laid out as ``(class, draw)``, matching the sampler's order."""

        generator = np.random.default_rng((self.seed, step, 2))
        return np.concatenate(
            [generator.choice(candidates, size=self.reference_sets) for candidates in self.candidates]
        )

    def reference_labels(self):
        return np.repeat(self.classes, self.reference_sets)


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #


def recall_at_1(embeddings, labels, chunk=1024):
    """Leave-one-out precision@1 by exact inner product on unit-norm rows."""

    correct = 0
    for start in range(0, len(embeddings), chunk):
        block = embeddings[start : start + chunk]
        similarity = block @ embeddings.t()
        rows = torch.arange(len(block), device=embeddings.device)
        similarity[rows, rows + start] = -2.0
        correct += int((labels[similarity.argmax(dim=1)] == labels[start : start + chunk]).sum())
    return correct / len(embeddings)


@torch.no_grad()
def embed_positions(model, features, positions, batch_size=4096):
    model.eval()
    chunks = []
    for start in range(0, len(positions), batch_size):
        rows = torch.as_tensor(positions[start : start + batch_size], device=features.device)
        chunks.append(model(features[rows]))
    model.train()
    return torch.cat(chunks) if chunks else torch.empty(0, device=features.device)


def sharpness_metrics(distances, log_probabilities, unlabeled_labels, reference_labels):
    """What the entropy term currently sees: how sharp it is, and whether it is right."""

    with torch.no_grad():
        probabilities = log_probabilities.exp()
        num_classes = probabilities.shape[-1]
        entropy = -(probabilities * log_probabilities).sum(dim=-1)
        # Spread within a row is what the softmax reads; an offset shared by the
        # whole row cancels, so the row-wise quantile span is the honest number.
        quantiles = torch.quantile(
            distances.flatten(0, 1).float(),
            torch.tensor([0.05, 0.95], device=distances.device),
            dim=-1,
        )
        # Nearest reference over every set, as an assignment of the row.
        nearest = distances.permute(1, 0, 2).flatten(1).argmin(dim=1)
        assigned = reference_labels[nearest]
        return {
            "dist_mean": float(distances.mean()),
            "dist_span": float((quantiles[1] - quantiles[0]).mean()),
            "p_max": float(probabilities.max(dim=-1).values.mean()),
            "x_unif": float(probabilities.max(dim=-1).values.mean()) * num_classes,
            "normalized_entropy": float(entropy.mean()) / math.log(num_classes),
            "argmax_purity": float((assigned == unlabeled_labels).float().mean()),
        }


# --------------------------------------------------------------------------- #
# Shared harness
# --------------------------------------------------------------------------- #


@dataclass
class Harness:
    source: FeatureSource
    eval_source: FeatureSource
    stream: BatchStream
    references: ReferenceStream
    label_lookup: torch.Tensor
    num_classes: int
    unlabeled_positions: np.ndarray
    eval_positions: np.ndarray


def build_model_and_criterion(harness, args, state=None):
    torch.manual_seed(args.seed)
    model = FrozenFeatureModel(harness.source.backbone_dim, args.feat_dim).to(args.device)
    criterion = engine.make_named_loss(
        name=args.loss,
        params=args.loss_params,
        num_classes=harness.num_classes,
        embedding_size=args.feat_dim,
    ).to(args.device)
    if state is not None:
        model.load_state_dict(state["model"])
        if state["criterion"] is not None:
            criterion.load_state_dict(state["criterion"])
    return model, criterion


def batch_embeddings(model, harness, args, step, unlabeled_size):
    """One step's labeled batch, unlabeled batch, and reference batch, embedded jointly."""

    device = args.device
    labeled_indices = harness.stream.labeled_batch(step)
    unlabeled_indices = harness.stream.unlabeled_batch(step, unlabeled_size)
    reference_indices = harness.references.references(step)
    sizes = (len(labeled_indices), len(unlabeled_indices), len(reference_indices))
    rows = torch.as_tensor(
        np.concatenate([labeled_indices, unlabeled_indices, reference_indices]), device=device
    )
    embeddings = model(harness.source.features[rows])
    labeled, unlabeled, references = torch.split(embeddings, sizes)
    labels = harness.label_lookup[harness.source.labels[rows[: sizes[0]]]]
    unlabeled_labels = harness.source.labels[rows[sizes[0] : sizes[0] + sizes[1]]]
    return labeled, labels, unlabeled, unlabeled_labels, references


def supervised_step(criterion, miner, embeddings, labels):
    mined = miner(embeddings, labels) if miner is not None else None
    return criterion(embeddings, labels, mined) if mined is not None else criterion(embeddings, labels)


# --------------------------------------------------------------------------- #
# Stage: probe
# --------------------------------------------------------------------------- #


def train_to_state(harness, args, steps):
    """Supervised-only training to a given step, returning the head/criterion state."""

    model, criterion = build_model_and_criterion(harness, args)
    miner = engine.make_named_miner(args.miner, args.miner_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_parameters = list(criterion.parameters())
    criterion_optimizer = (
        torch.optim.Adam(criterion_parameters, lr=args.classifier_lr) if criterion_parameters else None
    )
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        if criterion_optimizer is not None:
            criterion_optimizer.zero_grad(set_to_none=True)
        indices = torch.as_tensor(harness.stream.labeled_batch(step), device=args.device)
        embeddings = model(harness.source.features[indices])
        labels = harness.label_lookup[harness.source.labels[indices]]
        supervised_step(criterion, miner, embeddings, labels).backward()
        optimizer.step()
        if criterion_optimizer is not None:
            criterion_optimizer.step()
    return {
        "model": {name: value.detach().clone() for name, value in model.state_dict().items()},
        "criterion": (
            {name: value.detach().clone() for name, value in criterion.state_dict().items()}
            if criterion_parameters
            else None
        ),
    }


def probe_stage(harness, args):
    """Sharpness and gradient share per distance_scale, at several training states."""

    rows = []
    miner = engine.make_named_miner(args.miner, args.miner_params)
    reference_labels = torch.as_tensor(harness.references.reference_labels(), device=args.device)

    for warmup_steps in args.probe_steps:
        state = train_to_state(harness, args, warmup_steps)
        for power in args.distance_powers:
            previous = None
            for scale in args.scales:
                model, criterion = build_model_and_criterion(harness, args, state)
                samples = []
                for offset in range(args.measure_batches):
                    step = args.probe_measure_offset + offset
                    labeled, labels, unlabeled, unlabeled_labels, references = batch_embeddings(
                        model, harness, args, step, args.unlabeled_batch_size
                    )
                    supervised = supervised_step(criterion, miner, labeled, labels)
                    entropy, distances, log_probabilities = hoffer_entropy(
                        unlabeled, references, args.reference_sets, scale, power
                    )
                    norms = engine.measure_gradient_component_norms(
                        supervised,
                        entropy,
                        list(model.parameters()),
                        supervised_weight=1.0,
                        regularizer_weight=args.regularizer_weight,
                        include_pair_norm=False,
                    )
                    metrics = sharpness_metrics(
                        distances, log_probabilities, unlabeled_labels, reference_labels
                    )
                    metrics["supervised_norm"] = float(norms.supervised)
                    metrics["regularizer_norm"] = float(norms.regularizer)
                    metrics["grad_ratio"] = (
                        metrics["regularizer_norm"] / metrics["supervised_norm"]
                        if metrics["supervised_norm"] > 0
                        else float("nan")
                    )
                    samples.append(metrics)

                row = {key: float(np.mean([sample[key] for sample in samples])) for key in samples[0]}
                row["logit_span"] = scale * row["dist_span"]
                # Local log-log slope: 2.0 is the flat-softmax regime in which
                # w * s^2 is the only thing the objective depends on.
                if previous is not None and previous[1] > 0 and row["regularizer_norm"] > 0:
                    row["grad_slope"] = math.log(row["regularizer_norm"] / previous[1]) / math.log(
                        scale / previous[0]
                    )
                else:
                    row["grad_slope"] = float("nan")
                previous = (scale, row["regularizer_norm"])
                row.update(
                    {
                        "stage": "probe",
                        "warmup_steps": warmup_steps,
                        "distance_power": power,
                        "distance_scale": scale,
                    }
                )
                rows.append(row)
    return rows


def print_probe(rows, num_classes, args):
    header = (
        f"{'scale':>7} {'dist_span':>10} {'logit_span':>10} {'p_max':>8} {'x_unif':>7} "
        f"{'H/logC':>7} {'purity':>7} {'grad_ratio':>11} {'grad_slope':>10}"
    )
    for warmup_steps in args.probe_steps:
        for power in args.distance_powers:
            selected = [
                row
                for row in rows
                if row["warmup_steps"] == warmup_steps and row["distance_power"] == power
            ]
            if not selected:
                continue
            form = "-s*d^2 (paper Eq. 1, this repo)" if power == 2 else "-s*d (released code)"
            print(
                f"\n[probe] {form}, head after {warmup_steps} supervised steps, "
                f"C={num_classes}, uniform p={1.0 / num_classes:.4f}"
            )
            print(header)
            print("-" * len(header))
            for row in selected:
                print(
                    f"{row['distance_scale']:>7.2f} {row['dist_span']:>10.3f} {row['logit_span']:>10.2f} "
                    f"{row['p_max']:>8.4f} {row['x_unif']:>7.2f} {row['normalized_entropy']:>7.4f} "
                    f"{row['argmax_purity']:>7.3f} {row['grad_ratio']:>11.2e} "
                    f"{row['grad_slope']:>10.2f}"
                )


# --------------------------------------------------------------------------- #
# Stage: training arms
# --------------------------------------------------------------------------- #


@dataclass
class Arm:
    label: str
    regularizer_weight: float
    distance_scale: float
    distance_power: int = 2


def run_arm(arm, harness, args):
    """Train one arm from the shared initialization and score it per epoch."""

    device = args.device
    model, criterion = build_model_and_criterion(harness, args)
    miner = engine.make_named_miner(args.miner, args.miner_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_parameters = list(criterion.parameters())
    criterion_optimizer = (
        torch.optim.Adam(criterion_parameters, lr=args.classifier_lr) if criterion_parameters else None
    )
    reference_labels = torch.as_tensor(harness.references.reference_labels(), device=device)

    rows = []
    global_step = 0
    for epoch in range(args.epochs):
        regularized = arm.regularizer_weight > 0 and epoch >= args.warmup_epochs
        epoch_supervised, epoch_entropy, epoch_metrics = [], [], []
        for _ in range(args.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            if criterion_optimizer is not None:
                criterion_optimizer.zero_grad(set_to_none=True)

            labeled, labels, unlabeled, unlabeled_labels, references = batch_embeddings(
                model, harness, args, global_step, args.unlabeled_batch_size
            )
            supervised = supervised_step(criterion, miner, labeled, labels)
            total = supervised
            epoch_supervised.append(float(supervised.detach()))
            if regularized:
                entropy, distances, log_probabilities = hoffer_entropy(
                    unlabeled, references, args.reference_sets, arm.distance_scale, arm.distance_power
                )
                total = total + arm.regularizer_weight * entropy
                epoch_entropy.append(float(entropy.detach()))
                if len(epoch_metrics) < args.measure_batches:
                    epoch_metrics.append(
                        sharpness_metrics(distances, log_probabilities, unlabeled_labels, reference_labels)
                    )
            total.backward()
            optimizer.step()
            if criterion_optimizer is not None:
                criterion_optimizer.step()
            global_step += 1

        evaluation = embed_positions(model, harness.eval_source.features, harness.eval_positions)
        row = {
            "stage": "train",
            "arm": arm.label,
            "epoch": epoch,
            "regularizer_weight": arm.regularizer_weight,
            "distance_scale": arm.distance_scale,
            "distance_power": arm.distance_power,
            "supervised_loss": float(np.mean(epoch_supervised)),
            "entropy": float(np.mean(epoch_entropy)) if epoch_entropy else float("nan"),
            "recall_at_1": recall_at_1(
                evaluation, harness.eval_source.labels[harness.eval_positions]
            ),
        }
        for key in ("normalized_entropy", "p_max", "argmax_purity"):
            row[key] = (
                float(np.mean([metrics[key] for metrics in epoch_metrics]))
                if epoch_metrics
                else float("nan")
            )
        rows.append(row)
        if args.verbose:
            print(
                f"    epoch {epoch:>3}  R@1 {row['recall_at_1']:.4f}  "
                f"H/logC {row['normalized_entropy']:.4f}  purity {row['argmax_purity']:.3f}"
            )
    return rows


def build_train_arms(args):
    """The scale ladder on ``-s d^2``, plus the reference's ``-s d`` at its own scales.

    ``--reference-scales`` exists so ``-s d`` can be run at a *tuned* scale and
    not only at the reference's implicit ``s = 1``. Since ``-s d`` is flatter than
    ``-s d^2``, comparing the two at equal ``s`` only re-measures the temperature;
    comparing them at equal softmax sharpness is what says whether the choice of
    ``d`` vs ``d^2`` matters on its own.
    """

    arms = [Arm("sup_only", 0.0, float("nan"))]
    for scale in args.scales:
        arms.append(Arm(f"s={scale:g}", args.regularizer_weight, scale))
    if 1 in args.distance_powers:
        for scale in args.reference_scales:
            label = "released -d s=1" if scale == 1.0 else f"-d s={scale:g}"
            arms.append(Arm(label, args.regularizer_weight, scale, distance_power=1))
    return arms


def build_degeneracy_arms(args):
    """Pairs holding ``w * s^2`` constant: identical results mean one knob, not two."""

    arms = []
    for scale in args.degeneracy_scales:
        weight = args.degeneracy_product / (scale**2)
        arms.append(Arm(f"s={scale:g} w={weight:.4g}", weight, scale))
    return arms


def print_train(arm_rows, title):
    header = (
        f"{'arm':<22} {'weight':>9} {'scale':>7} {'R@1':>8} {'dR@1':>8} "
        f"{'H/logC':>8} {'p_max':>8} {'purity':>7}"
    )
    print(f"\n[{title}] final epoch")
    print(header)
    print("-" * len(header))
    baseline = None
    for arm, rows in arm_rows:
        final = rows[-1]
        if baseline is None:
            baseline = final["recall_at_1"]
        print(
            f"{arm.label:<22} {arm.regularizer_weight:>9.4g} {arm.distance_scale:>7.4g} "
            f"{final['recall_at_1']:>8.4f} {final['recall_at_1'] - baseline:>+8.4f} "
            f"{final['normalized_entropy']:>8.4f} {final['p_max']:>8.4f} "
            f"{final['argmax_purity']:>7.3f}"
        )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nrows: {path}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="Cars196")
    parser.add_argument("--split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--backbone", default=None, help="restrict to one backbone_cache subdirectory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--stages", nargs="+", default=["probe", "degeneracy", "train"],
        choices=["probe", "degeneracy", "train"],
    )
    parser.add_argument("--scales", type=float, nargs="+", default=list(DEFAULT_SCALES))
    parser.add_argument("--distance-powers", type=int, nargs="+", default=[2, 1], choices=[1, 2])
    parser.add_argument(
        "--reference-scales", type=float, nargs="+", default=[1.0],
        help="distance_scale values the -s*d (reference form) training arms are run at",
    )
    parser.add_argument("--regularizer-weight", type=float, default=DEFAULT_REGULARIZER_WEIGHT)
    parser.add_argument("--reference-sets", type=int, default=DEFAULT_REFERENCE_SETS)
    parser.add_argument("--unlabeled-batch-size", type=int, default=DEFAULT_BATCH_SIZE)

    parser.add_argument("--probe-steps", type=int, nargs="+", default=[0, 200, 1000],
                        help="supervised-only steps the probe reads the head at")
    parser.add_argument("--probe-measure-offset", type=int, default=10_000,
                        help="step index the probe draws its measurement batches from")
    parser.add_argument("--measure-batches", type=int, default=8)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--degeneracy-scales", type=float, nargs="+", default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--degeneracy-product", type=float, default=8.0,
                        help="the constant regularizer_weight * distance_scale^2 of the pairs")

    parser.add_argument("--feat-dim", type=int, default=DEFAULT_FEAT_DIM)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--classifier-lr", type=float, default=1e-2)
    parser.add_argument("--loss", default="MultiSimilarityLoss")
    parser.add_argument("--loss-params", default={}, type=json.loads)
    parser.add_argument("--miner", default="MultiSimilarityMiner")
    parser.add_argument("--miner-params", default={}, type=json.loads)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--sampler-m", type=int, default=DEFAULT_SAMPLER_M)
    parser.add_argument("--labeled-per-class", type=int, default=DEFAULT_LABELED_PER_CLASS)
    parser.add_argument("--eval-samples", type=int, default=4096)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    check_matches_regularizer(args.device)

    source = resolve_feature_source(args.dataset, args.split, args.data_root, args.device, args.backbone)
    eval_source = resolve_feature_source(
        args.dataset, args.eval_split, args.data_root, args.device, args.backbone
    )
    labeled_positions, unlabeled_positions = make_k_shot_split(
        source.labels, args.labeled_per_class, args.seed
    )
    unique_labels = torch.unique(source.labels)
    num_classes = int(len(unique_labels))
    label_lookup = torch.full(
        (int(unique_labels.max()) + 1,), -1, dtype=torch.long, device=args.device
    )
    label_lookup[unique_labels] = torch.arange(num_classes, device=args.device)

    eval_positions = np.arange(len(eval_source.labels))
    if len(eval_positions) > args.eval_samples:
        eval_positions = np.sort(
            np.random.default_rng(args.seed).choice(
                len(eval_positions), size=args.eval_samples, replace=False
            )
        )

    harness = Harness(
        source=source,
        eval_source=eval_source,
        stream=BatchStream(
            source.labels, labeled_positions, unlabeled_positions, args.batch_size, args.sampler_m, args.seed
        ),
        references=ReferenceStream(source.labels, labeled_positions, args.reference_sets, args.seed),
        label_lookup=label_lookup,
        num_classes=num_classes,
        unlabeled_positions=unlabeled_positions,
        eval_positions=eval_positions,
    )

    print(f"features:  {source.name} {tuple(source.features.shape)} from {source.cache_path}")
    print(
        f"alignment: 1-NN label agreement {source.nn_label_agreement:.3f} over {num_classes} "
        f"classes (chance {1.0 / num_classes:.3f})"
    )
    print(
        f"split:     {len(labeled_positions)} labeled ({args.labeled_per_class}/class), "
        f"{len(unlabeled_positions)} unlabeled"
    )
    print(
        f"regularizer batch: {args.unlabeled_batch_size} unlabeled + "
        f"{args.reference_sets * num_classes} references "
        f"({args.reference_sets} set(s) x {num_classes} classes)"
    )
    print(
        f"objective: {args.loss}{args.loss_params} + {args.miner}, batch "
        f"{args.batch_size}:{args.sampler_m}, feat_dim {args.feat_dim}, device {args.device}"
    )
    print(f"held-out:  {len(eval_positions)} rows of {eval_source.name}")
    print()
    print(LEGEND)

    all_rows = []
    if "probe" in args.stages:
        rows = probe_stage(harness, args)
        print_probe(rows, num_classes, args)
        all_rows.extend(rows)

    if "degeneracy" in args.stages:
        arm_rows = []
        for arm in build_degeneracy_arms(args):
            if args.verbose:
                print(f"[degeneracy] {arm.label}")
            rows = run_arm(arm, harness, args)
            for row in rows:
                row["stage"] = "degeneracy"
            arm_rows.append((arm, rows))
            all_rows.extend(rows)
        print_train(arm_rows, f"degeneracy: w * s^2 = {args.degeneracy_product:g} held constant")

    if "train" in args.stages:
        arm_rows = []
        for arm in build_train_arms(args):
            if args.verbose:
                print(f"[train] {arm.label}")
            rows = run_arm(arm, harness, args)
            arm_rows.append((arm, rows))
            all_rows.extend(rows)
        print_train(arm_rows, f"training arms at regularizer_weight={args.regularizer_weight:g}")

    if args.csv:
        write_csv(args.csv, all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
