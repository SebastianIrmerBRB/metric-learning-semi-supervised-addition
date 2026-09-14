"""Single-fold training, cross-validation, and model component construction."""

import copy
import csv
import gc
import time
from collections import OrderedDict, defaultdict, deque
from dataclasses import replace
from datetime import datetime
from functools import partial
from numbers import Integral, Real
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytorch_metric_learning.losses as losses
import pytorch_metric_learning.miners as miners
import torch
from loguru import logger
from tqdm import tqdm

import utils
from losses import metric_losses
from models.retrieval_model import (
    BACKBONE_TUNING_FROZEN,
    DEFAULT_PROJECTION_LAYERS,
    DinoWrapper,
)

from . import ablation, gpu_feature_loader, resume, semi_supervised
from .ssl import algorithms as ssl_algorithms
from .ssl import grad_norm
from .ssl import gradient_surgery
from .ssl import step_visualization
from .cli import (
    DEFAULT_DATA_SPLIT_SEED,
    FOLD_TEST_EMBEDDING_STORAGE_MEMORY,
    FOLD_TEST_EMBEDDING_STORAGE_TEMPORARY,
    get_fold_test_embedding_storage,
    FINAL_TEST_VISUALIZATION_NONE,
    FINAL_TEST_VISUALIZATION_PACMAP,
    FINAL_TEST_VISUALIZATION_TSNE,
    normalize_final_test_visualization,
    LR_SCHEDULER_COSINE,
    LR_SCHEDULER_COSINE_WARM_RESTARTS,
    LR_SCHEDULER_NONE,
    LR_SCHEDULER_STEP,
    LR_SCHEDULERS,
    get_ssl_device,
    normalize_backbone_tuning_args,
    resolve_scheduler_batch_device,
)
from .frozen_feature_cache import (
    FrozenFeatureDatasetCache,
    make_frozen_feature_cache_key,
    make_frozen_feature_index_spec,
)
from .io import namespace_to_dict, result_to_dict, write_json
from .types import (
    ALL_LOSSES,
    ALL_MINERS,
    CLASSIFICATION_LOSSES,
    WARMUP_LOSS_SAME_AS_LOSS,
    DATASETS,
    SELECTION_METRIC_MAP_AT_R,
    SELECTION_METRIC_PRECISION_AT_1,
    SELECTION_METRICS,
    TrainingResult,
)

BATCH_EASY_HARD_MINER_STRATEGIES = {"all", "easy", "hard", "semihard"}
BATCH_EASY_HARD_DEFAULT_POS_STRATEGY = "easy"
BATCH_EASY_HARD_DEFAULT_NEG_STRATEGY = "semihard"
BATCH_EASY_HARD_RANGE_PARAMS = ("allowed_pos_range", "allowed_neg_range")
FROZEN_BACKBONE_STATE_PREFIX = "dinov2."

HEAD_STATE_FILENAME = "head_state.pt"
# Default cross-trial frozen-feature budget under --low_memory_mode. Large
# enough to keep a study's active folds resident, small enough that a sweep over
# seeds or label budgets cannot grow without bound.
LOW_MEMORY_FROZEN_FEATURE_CACHE_BYTES = 8_000_000_000


class TrainingLossComponents(NamedTuple):
    criterion: torch.nn.Module
    is_classification: bool
    miner: object | None
    optimizer: torch.optim.Optimizer | None

    @property
    def requires_float32_loss(self) -> bool:
        return isinstance(self.criterion, losses.ArcFaceLoss)

class _EpochPhase(NamedTuple):
    objective: TrainingLossComponents
    standalone_stml: bool
    regularization_active: bool
    regularizer_disabled: bool
    description: str


def _sync_if_cuda(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _now(device):
    _sync_if_cuda(device)
    return time.perf_counter()


class _BatchTimer:
    """Collect optional synchronized batch timings without cluttering the loop."""

    def __init__(self, device, enabled):
        self.device = device
        self.enabled = enabled
        self.totals = defaultdict(float)

    def start(self):
        return _now(self.device) if self.enabled else None

    def stop(self, name, start, end=None):
        if start is not None:
            self.totals[name] += (self.start() if end is None else end) - start


class GradientContributionNorms(NamedTuple):
    supervised: float
    regularizer: float
    # ((name, norm), ...) for objective terms a regularizer adds beyond the two
    # weighted losses.
    extra: tuple = ()
    # ||g_supervised + g_regularizer||, measured only when extra terms make the
    # combined gradient norm an unusable stand-in for it.
    supervised_regularizer_sum: float = None


# A component gradient this far below the supervised one carries no usable
# calibration signal. The implied weight is target_ratio / ratio, so one denormal
# measurement freezes a weight of 1e15 and the next batch that does produce a
# gradient blows the model up. Measured on SLADE + NTXent at temperature 0.0023,
# where Eq 7's mining leaves the ranking term exactly zero on the pairs it
# separated and enormous on the occasional loose one: a median ratio of 2.2e16
# froze regularizer_weight=4.6e15 and training went non-finite three epochs later.
# How a gradient is summarized before the two terms are compared. "l2" is the
# Euclidean norm; "max_mean" is Wang et al.'s statistic (doi:10.1137/20M1318043).
TARGET_RATIO_STATISTIC_L2 = "l2"
TARGET_RATIO_STATISTIC_MAX_MEAN = "max_mean"
TARGET_RATIO_STATISTICS = (TARGET_RATIO_STATISTIC_L2, TARGET_RATIO_STATISTIC_MAX_MEAN)

MINIMUM_CALIBRATION_NORM_RATIO = 1e-6

# Consecutive unusable batches before a term is declared dead. A single one is
# ordinary -- a regularizer whose objective starts up in stages produces them --
# so the budget has to be long enough to outlast a transient and short enough to
# fail in the epoch that caused it rather than the one that goes non-finite.
# Counted in eligible probes, so a wider sample update interval stretches the
# same budget over proportionally more training batches.
MAXIMUM_UNUSABLE_CALIBRATION_BATCHES = 50

# Backward-compatible default window. Individual regularizers may override it
# through regularizer_target_ratio_recalibration_interval_samples.
TARGET_RATIO_RECALIBRATION_INTERVAL_SAMPLES = 10_000


def measure_gradient_component_norms(
    supervised_loss,
    regularization_loss,
    parameters,
    *,
    supervised_weight,
    regularizer_weight,
    extra_components=None,
    include_pair_norm=True,
    statistic=TARGET_RATIO_STATISTIC_L2,
):
    """Measure weighted model-gradient component norms without accumulating them.

    Each component is summarized and released before computing the next one, so
    interval logging needs at most one additional model-sized gradient tuple.
    ``autograd.grad`` leaves ``parameter.grad`` untouched, and retaining the
    graph lets the caller run the ordinary combined ``loss.backward()``
    afterwards.

    ``extra_components`` maps a diagnostic name to ``(loss, weight)`` for terms
    the regularizer adds in ``combine_losses``. They are scored separately, and
    their presence also costs one measurement of the supervised plus regularizer
    gradient, which the caller's combined norm no longer represents.

    ``include_pair_norm=False`` drops that extra measurement. Only the cosine in
    ``make_gradient_contribution_diagnostics`` reads it, so a caller that wants
    the per-term norms alone -- GradNorm -- should not pay a full-model backward
    pass per update for it.

    ``statistic`` selects how a gradient is summarized into one number.
    ``"l2"`` is the Euclidean norm, which is what a ratio of gradient
    *magnitudes* usually means. ``"max_mean"`` is the statistic Wang et al. use
    for their learning-rate annealing in physics-informed networks
    (doi:10.1137/20M1318043, Algorithm 1): the primary term is summarized by the
    largest absolute coordinate of its gradient and every auxiliary term by the
    mean absolute coordinate, so the resulting weight equalizes the auxiliary
    term's *typical* coordinate against the primary term's *largest* one. The
    two disagree whenever a gradient is concentrated on few coordinates rather
    than spread across them, which is exactly when an L2 ratio is unstable.
    """

    extra_components = dict(extra_components or {})
    parameters = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not parameters:
        return GradientContributionNorms(
            0.0,
            0.0,
            tuple((name, 0.0) for name in extra_components),
        )

    def component_norm(*weighted_terms, reduction=TARGET_RATIO_STATISTIC_L2):
        weighted_loss = None
        for loss, weight in weighted_terms:
            weight = float(weight)
            if loss is None or weight == 0.0 or not loss.requires_grad:
                continue
            term = weight * loss
            weighted_loss = term if weighted_loss is None else weighted_loss + term
        if weighted_loss is None:
            return 0.0
        gradients = torch.autograd.grad(
            weighted_loss,
            parameters,
            allow_unused=True,
            retain_graph=True,
        )
        device = parameters[0].device
        squared_norm = torch.zeros((), dtype=torch.float32, device=device)
        absolute_sum = torch.zeros((), dtype=torch.float32, device=device)
        absolute_max = torch.zeros((), dtype=torch.float32, device=device)
        counted = 0
        for gradient in gradients:
            if gradient is None:
                continue
            gradient = gradient.detach()
            if gradient.is_sparse:
                gradient = gradient.coalesce().values()
            gradient = gradient.float()
            if reduction == "l2":
                squared_norm.add_(gradient.square().sum())
                continue
            magnitude = gradient.abs()
            absolute_sum.add_(magnitude.sum())
            absolute_max = torch.maximum(absolute_max, magnitude.max())
            counted += magnitude.numel()
        if reduction == "l2":
            return max(float(squared_norm.item()), 0.0) ** 0.5
        if reduction == "max":
            return max(float(absolute_max.item()), 0.0)
        if reduction == "mean":
            return max(float(absolute_sum.item()), 0.0) / counted if counted else 0.0
        raise ValueError(f"unknown gradient reduction {reduction!r}")

    if statistic == TARGET_RATIO_STATISTIC_MAX_MEAN:
        # Wang et al., Algorithm 1: max over the primary gradient, mean over each
        # auxiliary one. The pair norm stays Euclidean because only the cosine
        # diagnostic reads it and that is an L2 quantity by definition.
        supervised_reduction, regularizer_reduction = "max", "mean"
    else:
        supervised_reduction = regularizer_reduction = "l2"

    # Do not keep several model-sized component gradient tuples alive together.
    supervised_norm = component_norm(
        (supervised_loss, supervised_weight),
        reduction=supervised_reduction,
    )
    regularizer_norm = component_norm(
        (regularization_loss, regularizer_weight),
        reduction=regularizer_reduction,
    )
    extra = tuple(
        (name, component_norm((loss, weight), reduction=regularizer_reduction))
        for name, (loss, weight) in extra_components.items()
    )
    # The caller's combined norm is taken over the whole model, so it only
    # stands in for ``||grad(sup) + grad(reg)||`` when nothing else contributes
    # and nothing was excluded from ``parameters``. A regularizer declaring extra
    # components is exactly the case where both of those can fail, including on
    # the batches where the extra term happens to measure zero.
    supervised_regularizer_sum = (
        component_norm(
            (supervised_loss, supervised_weight),
            (regularization_loss, regularizer_weight),
        )
        if extra and include_pair_norm
        else None
    )
    return GradientContributionNorms(
        supervised_norm,
        regularizer_norm,
        extra,
        supervised_regularizer_sum,
    )


def make_gradient_contribution_diagnostics(
    component_norms,
    combined_norm,
    regularizer_weight=None,
):
    """Build scalar diagnostics after the real combined backward pass.

    ``regularizer_norm`` here is ``||grad(w_reg * L_reg)||`` -- the weighted
    term, which is what the objective actually contributes. Under a calibrated
    weight that quantity is pinned near ``target_ratio * supervised_norm`` by
    construction, so it cannot show the regularizer's *own* gradient collapsing:
    a shrinking ``||grad(L_reg)||`` and a growing ``w_reg`` cancel in it. Passing
    ``regularizer_weight`` adds the unweighted norm, which is the quantity that
    does show it. It costs nothing extra, because the gradient norm is linear in
    the weight and the division is exact.
    """

    supervised_norm = float(component_norms.supervised)
    regularizer_norm = float(component_norms.regularizer)
    combined_norm = 0.0 if combined_norm is None else float(combined_norm)
    supervised_squared = supervised_norm * supervised_norm
    regularizer_squared = regularizer_norm * regularizer_norm
    # The cosine follows from ||a + b||^2 = ||a||^2 + ||b||^2 + 2 a.b, so it
    # needs the norm of those two terms alone. Without extra objective terms the
    # combined gradient is exactly their sum.
    pair_norm = (
        combined_norm
        if component_norms.supervised_regularizer_sum is None
        else float(component_norms.supervised_regularizer_sum)
    )
    dot_product = (
        pair_norm * pair_norm
        - supervised_squared
        - regularizer_squared
    ) / 2.0
    component_norm_sum = supervised_norm + regularizer_norm
    norm_product = supervised_norm * regularizer_norm
    cosine_similarity = (
        max(-1.0, min(1.0, dot_product / norm_product))
        if norm_product > 0.0
        else 0.0
    )
    diagnostics = {
        "train/gradient_contribution/supervised_norm": supervised_norm,
        "train/gradient_contribution/regularizer_norm": regularizer_norm,
        "train/gradient_contribution/combined_norm": combined_norm,
        "train/gradient_contribution/regularizer_fraction": (
            regularizer_norm / component_norm_sum
            if component_norm_sum > 0.0
            else 0.0
        ),
        "train/gradient_contribution/cosine_similarity": cosine_similarity,
    }
    if regularizer_weight is not None and float(regularizer_weight) > 0.0:
        # ||grad(w * L)|| == w * ||grad(L)||, so the unweighted norm is exact.
        weight = float(regularizer_weight)
        diagnostics["train/gradient_contribution/regularizer_weight"] = weight
        diagnostics[
            "train/gradient_contribution/regularizer_norm_at_unit_weight"
        ] = regularizer_norm / weight
        diagnostics[
            "train/gradient_contribution/unit_regularizer_to_supervised_ratio"
        ] = (
            (regularizer_norm / weight) / supervised_norm
            if supervised_norm > 0.0
            else 0.0
        )
    if supervised_norm > 0.0:
        diagnostics[
            "train/gradient_contribution/regularizer_to_supervised_ratio"
        ] = regularizer_norm / supervised_norm
    for name, norm in component_norms.extra:
        norm = float(norm)
        diagnostics[f"train/gradient_contribution/{name}_norm"] = norm
        if supervised_norm > 0.0:
            diagnostics[
                f"train/gradient_contribution/{name}_to_supervised_ratio"
            ] = norm / supervised_norm
    return diagnostics


def concatenate_joint_forward_inputs(supervised_inputs, regularizer_batch):
    """Concatenate labeled and regularizer inputs without changing either stream."""

    regularizer_inputs = regularizer_batch[0]
    if not torch.is_tensor(supervised_inputs) or not torch.is_tensor(regularizer_inputs):
        raise TypeError("joint regularizer forwards require tensor input batches")
    if supervised_inputs.ndim != regularizer_inputs.ndim:
        raise ValueError("labeled and unlabeled inputs must have the same rank")
    if supervised_inputs.shape[1:] != regularizer_inputs.shape[1:]:
        raise ValueError(
            "labeled and unlabeled inputs must have matching non-batch dimensions; "
            f"got {tuple(supervised_inputs.shape)} and {tuple(regularizer_inputs.shape)}"
        )
    return torch.cat([supervised_inputs, regularizer_inputs], dim=0)


class _ProjectionGraphModule(torch.nn.Module):
    """``model.project_features`` exposed as a Module, for CUDA-graph capture.

    ``make_graphed_callables`` differentiates the captured forward with respect
    to ``self.parameters()`` and, with ``allow_unused_input=False``, rejects any
    parameter the forward does not reach. So this must register the projection
    head and nothing else: not the backbone -- frozen or not -- and not the
    parameters a regularizer hangs on the model, such as SLADE's ``slade_basis``.
    The model itself is held inside a tuple, which ``nn.Module.__setattr__``
    does not walk, so calling the real ``project_features`` here costs no
    duplicated projection logic and adds no parameters.
    """

    def __init__(self, model, head):
        super().__init__()
        self.head = head
        self._model_ref = (model,)

    def forward(self, features):
        return self._model_ref[0].project_features(features)


def make_graphed_projection(model, sample_features):
    """Capture ``project_features`` for one batch shape, or return ``None``.

    Returns the graphed callable only when replaying it reproduces the eager
    projection exactly. The capture is the head's own kernels, so that is the
    expected outcome rather than a hopeful one -- but a graph that silently
    diverged would corrupt every subsequent step, and the check costs one
    forward per fold.
    """

    head_accessor = getattr(model, "projection_head", None)
    head = head_accessor() if callable(head_accessor) else None
    if not isinstance(head, torch.nn.Module):
        return None, "the model exposes no projection_head() to capture"
    if not any(parameter.requires_grad for parameter in head.parameters()):
        return None, "the projection head has no trainable parameters"
    if any(
        module._forward_hooks or module._forward_pre_hooks or module._backward_hooks
        for module in head.modules()
    ):
        return None, "the projection head carries module hooks, which capture forbids"

    graphed = torch.cuda.make_graphed_callables(
        _ProjectionGraphModule(model, head),
        (sample_features,),
    )
    with torch.no_grad():
        expected = model.project_features(sample_features)
        replayed = graphed(sample_features)
        if not torch.equal(expected, replayed):
            return None, "the replayed projection did not match the eager one"
    return graphed, None


def should_precompute_frozen_features(args, ssl_config):
    regularized_ssl = is_cacheable_regularized_ssl(ssl_config)
    loss_driven_ssl = ssl_config.method in semi_supervised.LOSS_DRIVEN_METHODS
    # Every registered pseudo-label method consumes the common deterministic
    # embedding-extraction path. Materializing the raw frozen-backbone matrix is
    # therefore method-agnostic: Iscen, mixed propagation, sklearn, and FAISS
    # methods all re-project the same matrix through the current trainable head.
    pseudo_label_ssl = semi_supervised.is_pseudo_label_method(ssl_config)
    supervised = is_supervised_mode(args)
    return (
        bool(getattr(args, "use_cache", False))
        and args.backbone_tuning == BACKBONE_TUNING_FROZEN
        and (
            (supervised and not ssl_config.enabled)
            or (not supervised and (loss_driven_ssl or regularized_ssl or pseudo_label_ssl))
        )
    )


def is_cacheable_regularized_ssl(ssl_config):
    if not ssl_config.enabled:
        return False
    ssl_method = semi_supervised.get_method(ssl_config)
    if ssl_method is None or not ssl_method.is_regularization_method:
        return False
    regularizer = ssl_method.make_regularizer(ssl_config)
    return regularizer.supports_frozen_feature_precompute


def get_frozen_feature_batch_size(args):
    batch_size = getattr(args, "frozen_feature_batch_size", None)
    return int(args.batch_size if batch_size is None else batch_size)


def get_frozen_feature_train_views(args):
    return int(getattr(args, "frozen_feature_train_views", 1))


def get_projection_layers(args):
    projection_layers = getattr(args, "projection_layers", DEFAULT_PROJECTION_LAYERS)
    if projection_layers is None:
        return DEFAULT_PROJECTION_LAYERS
    return int(projection_layers)


def get_frozen_feature_residency(args):
    return str(getattr(args, "frozen_feature_residency", utils.FEATURE_RESIDENCY_MMAP))


def get_evaluation_embedding_residency(args):
    return str(
        getattr(
            args,
            "evaluation_embedding_residency",
            utils.EVALUATION_EMBEDDING_RESIDENCY_CPU,
        )
    ).lower()


def get_frozen_feature_residency_max_bytes(args):
    budget_gb = getattr(args, "frozen_feature_residency_max_gb", None)
    if budget_gb is None:
        return None
    return int(float(budget_gb) * 1e9)


def get_low_memory_mode(args):
    return bool(getattr(args, "low_memory_mode", False))


def get_faiss_temp_memory_bytes(args):
    """Return the FAISS scratch cap, or None to keep the built-in default."""

    temp_memory_mb = getattr(args, "faiss_temp_memory_mb", None)
    if temp_memory_mb is None:
        return None
    return int(float(temp_memory_mb) * 1024 * 1024)


def get_frozen_feature_cache_max_bytes(args):
    """Return the cross-trial feature-cache budget, or None to keep every view."""

    budget_gb = getattr(args, "frozen_feature_cache_max_gb", None)
    if budget_gb is not None:
        return int(float(budget_gb) * 1e9)
    if get_low_memory_mode(args):
        return LOW_MEMORY_FROZEN_FEATURE_CACHE_BYTES
    return None


def apply_low_memory_mode(args):
    """Publish the process-wide FAISS buffer policy selected on the CLI.

    Evaluation retrieval and SSL graph construction build their FAISS resources
    independently, so both policies are set from the one flag.
    """

    enabled = get_low_memory_mode(args)
    temp_memory_bytes = get_faiss_temp_memory_bytes(args)
    utils.set_faiss_low_memory(enabled, temp_memory_bytes=temp_memory_bytes)
    ssl_algorithms.set_ssl_faiss_low_memory(
        (
            utils.LOW_MEMORY_FAISS_TEMP_BYTES
            if temp_memory_bytes is None
            else temp_memory_bytes
        )
        if enabled
        else None
    )


def get_gpu_resident_device(args):
    """Return the device to hold training features on, or None to keep the loader."""

    if get_frozen_feature_residency(args) != utils.FEATURE_RESIDENCY_GPU:
        return None
    return args.device


def _precompute_backbone_features(
    args,
    model,
    dataset,
    desc,
    pin_memory,
    *,
    require_feature_transform=False,
    use_feature_transform=True,
    num_views=1,
    frozen_feature_cache=None,
):
    index_spec = None
    if (
        bool(getattr(model, "use_cache", False))
        and hasattr(model, "materialize_cached_backbone_features")
    ):
        index_spec = make_frozen_feature_index_spec(
            args,
            model,
            dataset,
            require_feature_transform=require_feature_transform,
            use_feature_transform=use_feature_transform,
            num_views=num_views,
        )

    def compute():
        cache_kwargs = {}
        if index_spec is not None:
            cache_kwargs = {
                "cache_key": index_spec.key,
                "cache_indices": index_spec.row_indices,
                "cache_size": index_spec.capacity,
            }
        return utils.precompute_backbone_feature_dataset(
            model=model,
            dataset=dataset,
            device=args.device,
            batch_size=get_frozen_feature_batch_size(args),
            seed=args.seed,
            num_workers=args.num_workers,
            start_method=args.dataloader_start_method,
            desc=desc,
            pin_memory=pin_memory,
            require_feature_transform=require_feature_transform,
            use_feature_transform=use_feature_transform,
            num_views=num_views,
            residency=get_frozen_feature_residency(args),
            residency_max_bytes=get_frozen_feature_residency_max_bytes(args),
            **cache_kwargs,
        )

    if frozen_feature_cache is None:
        return compute()
    cache_key = make_frozen_feature_cache_key(
        args,
        model,
        dataset,
        require_feature_transform=require_feature_transform,
        use_feature_transform=use_feature_transform,
        num_views=num_views,
    )
    if cache_key is None:
        logger.warning(
            f"Cannot safely identify the frozen backbone for {desc}; "
            "skipping cross-trial in-memory feature reuse"
        )
        return compute()
    return frozen_feature_cache.get_or_compute(cache_key, compute, desc)


def _make_train_loader(
    args,
    dataset,
    seed,
    pin_memory,
    *,
    persistent_workers=True,
    labeled_batch_size=None,
    class_overlap="independent",
):
    return utils.make_train_loader(
        dataset,
        args.batch_size,
        args.sampler_m,
        seed=seed,
        length_before_new_iter=args.length_before_new_iter,
        num_workers=args.num_workers,
        start_method=args.dataloader_start_method,
        persistent_workers=persistent_workers,
        pin_memory=pin_memory,
        labeled_batch_size=labeled_batch_size,
        class_overlap=class_overlap,
        gpu_resident_device=get_gpu_resident_device(args),
        gpu_resident_max_bytes=get_frozen_feature_residency_max_bytes(args),
    )


def _pseudo_label_labeled_batch_size(ssl_config):
    """Return the optional two-stream quota for post-warmup loaders."""

    if ssl_config.method not in semi_supervised.TWO_STREAM_SAMPLER_METHODS:
        return None
    return ssl_config.labeled_batch_size


def _pseudo_label_capacity_filter_kwargs(args, ssl_config):
    """Describe the pseudo-label diversity needed by a two-stream batch."""

    labeled_batch_size = _pseudo_label_labeled_batch_size(ssl_config)
    if labeled_batch_size is None:
        return {}
    sampler_m = int(args.sampler_m)
    pseudo_batch_size = int(args.batch_size) - int(labeled_batch_size)
    kwargs = {
        "required_pseudo_label_classes": (pseudo_batch_size + sampler_m - 1) // sampler_m,
        # Unless explicitly overridden in the SSL config, rescue up to one full
        # M-per-class group for every newly admitted pseudo class.
        "pseudo_label_rescue_top_k": sampler_m,
    }
    if ssl_config.class_overlap == "disjoint":
        # Only disjoint class selection needs the streams to jointly cover a
        # whole batch's worth of classes; independent streams each stand alone.
        kwargs["required_combined_label_classes"] = (
            int(args.batch_size) + sampler_m - 1
        ) // sampler_m
    return kwargs


def _make_eval_loader(
    args,
    model,
    dataset,
    desc,
    pin_memory,
    *,
    precompute_features,
    frozen_feature_cache=None,
):
    if precompute_features:
        dataset = _precompute_backbone_features(
            args,
            model,
            dataset,
            desc,
            pin_memory,
            frozen_feature_cache=frozen_feature_cache,
        )
    return utils.make_eval_loader(
        dataset,
        batch_size=utils.frozen_feature_eval_batch_size(
            dataset,
            args.device,
        ),
        seed=args.seed,
        num_workers=args.num_workers,
        start_method=args.dataloader_start_method,
        pin_memory=pin_memory,
        gpu_resident_device=get_gpu_resident_device(args),
        gpu_resident_max_bytes=get_frozen_feature_residency_max_bytes(args),
    )


def _retrieval_backend_name(device):
    return torch.device(device).type


def resolve_run_evaluation_retrieval_devices(
    args,
    valid_loader,
    test_loader,
):
    """Use one retrieval backend for validation and its associated test run."""

    inherited_backend = getattr(args, "validation_retrieval_backend", None)
    validation_device = None
    if valid_loader is not None:
        validation_device = utils.resolve_evaluation_retrieval_device(
            getattr(valid_loader, "dataset", None),
            args.device,
            backend=inherited_backend,
        )
        validation_backend = _retrieval_backend_name(validation_device)
    else:
        # A final full-development fit has no validation loader. New HPO runs
        # provide the winning validation backend explicitly; legacy studies do
        # not, and conservatively retain the historical CPU test evaluation.
        validation_backend = (
            utils.EVALUATION_RETRIEVAL_BACKEND_CPU
            if inherited_backend is None
            else str(inherited_backend).lower()
        )
        if inherited_backend is None and test_loader is not None:
            logger.warning(
                "No HPO validation retrieval backend was recorded; "
                "defaulting final test retrieval to CPU"
            )

    test_device = None
    if test_loader is not None:
        if validation_device is not None:
            test_device = validation_device
        else:
            test_device = utils.resolve_evaluation_retrieval_device(
                getattr(test_loader, "dataset", None),
                args.device,
                backend=validation_backend,
            )

    return validation_device, test_device, validation_backend


def make_label_lookup_tensor(train_labels_mapper, device):
    label_mapping = [
        (int(original_label), int(mapped_label))
        for original_label, mapped_label in train_labels_mapper.items()
    ]
    if not label_mapping:
        return None
    label_ids, mapped_labels = zip(*label_mapping)
    if min(label_ids) < 0:
        return None
    max_label = max(label_ids)
    if max_label > 10_000_000:
        return None
    lookup = torch.full((max_label + 1,), -1, dtype=torch.long, device=device)
    lookup[torch.as_tensor(label_ids, dtype=torch.long, device=device)] = torch.as_tensor(
        mapped_labels,
        dtype=torch.long,
        device=device,
    )
    return lookup


def map_training_labels(labels, label_lookup, train_labels_mapper, device):
    if torch.is_tensor(labels) and label_lookup is not None:
        labels = labels.to(device, dtype=torch.long, non_blocking=True)
        return label_lookup[labels]
    return torch.tensor([train_labels_mapper[int(label)] for label in labels], device=device, dtype=torch.long)


def shutdown_epoch_train_loader(train_loader, warmup_train_loader=None, static_train_loader=None, *reusable_loaders):
    """Shutdown loaders that are rebuilt for a single epoch."""

    reusable_loaders = (warmup_train_loader, static_train_loader, *reusable_loaders)
    if train_loader is None or any(train_loader is loader for loader in reusable_loaders):
        return
    shutdown = getattr(train_loader, "shutdown", None)
    if callable(shutdown):
        shutdown()
        return
    utils.shutdown_dataloaders(train_loader)


def resolve_warmup_objective(args, ssl_config):
    """Turn a ``same_as_loss`` warm-up objective into the run's own objective.

    Runs for every job, not only HPO trials. ``resolve_hpo_warmup_objective``
    does the same thing one layer up so the choice is recorded in the trial's
    ``resolved_args``, but it only ever ran on the Optuna path -- so a directly
    configured run silently warmed up with whatever ``--warmup_loss`` defaulted
    to, at stock parameters, no matter which loss the run was about. Both
    entry points now converge here.

    STML is the exception it has always been: an unsupervised multi-view
    objective cannot run on labeled-only warm-up batches, so it falls back to a
    standard pair loss.
    """

    if not uses_ssl_warmup_objective(ssl_config):
        return args
    if args.warmup_loss != WARMUP_LOSS_SAME_AS_LOSS:
        return args

    resolved = copy.deepcopy(args)
    if args.loss == "STMLLoss":
        resolved.warmup_loss = "NTXentLoss"
        resolved.warmup_loss_params = {}
        resolved.warmup_miner = "no_miner"
        resolved.warmup_miner_params = {}
        logger.info(
            "warmup_loss=same_as_loss cannot use STMLLoss on labeled-only "
            "warm-up batches; warming up with NTXentLoss instead"
        )
        return resolved

    resolved.warmup_loss = resolved.loss
    resolved.warmup_loss_params = copy.deepcopy(resolved.loss_params)
    resolved.warmup_miner = resolved.miner
    resolved.warmup_miner_params = copy.deepcopy(resolved.miner_params)
    logger.info(
        "warmup_loss=same_as_loss resolved to the run's own objective: "
        f"{resolved.warmup_loss} / {resolved.warmup_miner} with its tuned "
        "parameters, so warm-up and post-warm-up training share one criterion "
        "instance and its trained state carries across the boundary"
    )
    return resolved


def run_experiment(
    args,
    ssl_config,
    optuna_trial=None,
    optuna_metric=None,
    frozen_feature_cache=None,
):
    """Run either one holdout training job or all requested CV folds."""

    args = resolve_warmup_objective(args, ssl_config)
    args = resolve_loss_driven_supervised_args(args)
    resolve_scheduler_batch_device(args, ssl_config)
    args, ssl_config = resolve_platform_dataloader_workers(args, ssl_config)
    if (
        frozen_feature_cache is None
        and bool(getattr(args, "use_cache", False))
        and args.backbone_tuning == BACKBONE_TUNING_FROZEN
    ):
        # One coordinator spans every fold in a normal run as well as every
        # trial when HPO supplies its own longer-lived instance.
        frozen_feature_cache = FrozenFeatureDatasetCache(
            max_bytes=get_frozen_feature_cache_max_bytes(args),
        )
    if args.cv_k > 1:
        # The Optuna trial is reported only after folds complete; individual
        # folds do not independently prune the same trial.
        return run_cross_validation(
            args,
            ssl_config,
            optuna_trial=optuna_trial,
            optuna_metric=optuna_metric,
            frozen_feature_cache=frozen_feature_cache,
        )
    return run_training(
        args,
        ssl_config,
        optuna_trial=optuna_trial,
        optuna_metric=optuna_metric,
        frozen_feature_cache=frozen_feature_cache,
    )


def resolve_platform_dataloader_workers(args, ssl_config, platform_name=None):
    """Apply platform-specific effective worker counts without changing source configs."""

    effective_train_workers = utils.effective_num_workers(args.num_workers, platform_name=platform_name)
    effective_embedding_workers = utils.effective_num_workers(
        ssl_config.embedding_num_workers,
        platform_name=platform_name,
    )
    if (
        effective_train_workers == args.num_workers
        and effective_embedding_workers == ssl_config.embedding_num_workers
    ):
        return args, ssl_config

    resolved_args = copy.deepcopy(args)
    resolved_args.configured_num_workers = args.num_workers
    resolved_args.configured_ssl_embedding_num_workers = ssl_config.embedding_num_workers
    resolved_args.num_workers = effective_train_workers
    resolved_args.windows_dataloader_workers_forced_to_zero = True
    resolved_ssl_config = replace(ssl_config, embedding_num_workers=effective_embedding_workers)
    return resolved_args, resolved_ssl_config


def is_supervised_mode(args):
    return getattr(args, "mode", "supervised") == "supervised"


def resolve_loss_driven_supervised_args(args):
    """Use the configured warm-up objective for an STML supervised baseline."""

    if not is_supervised_mode(args) or getattr(args, "loss", None) != "STMLLoss":
        return args
    resolved = copy.deepcopy(args)
    resolved.loss = resolved.warmup_loss
    resolved.loss_params = dict(resolved.warmup_loss_params)
    resolved.miner = resolved.warmup_miner
    resolved.miner_params = dict(resolved.warmup_miner_params)
    return resolved


def resolve_mode_ssl_config(args, ssl_config):
    """Resolve the config a run actually trains with, from its mode and its args.

    Every executed path funnels through here -- single run, outer-grid scenario,
    and HPO trial -- which is why the unlabeled-pool arguments and the ablation
    config are applied here rather than at each ``load_ssl_config`` call site.

    The ablation is applied in two halves around the unlabeled-pool arguments,
    and ``args`` is updated in place by the first: an ablation of a CLI argument
    has to land before the arguments shape the config, while an ablation of the
    config itself has to land after, because the whole contract of an ablation
    config is that it is the last word. See :mod:`training.ablation`.
    """

    ablation.apply_ablation_args(args)
    ssl_config = apply_unlabeled_pool_args(args, ssl_config)
    ssl_config = ablation.apply_ablation_ssl_config(args, ssl_config)
    if is_supervised_mode(args):
        return make_supervised_split_config(ssl_config)
    if not ssl_config.enabled:
        raise ValueError(
            "--mode ssl requires --ssl_config to name an SSL method. "
            + (
                "No --ssl_config was given"
                if getattr(args, "ssl_config", None) is None
                else f"{args.ssl_config} sets method='none'"
            )
            + f". Available methods: {semi_supervised.available_methods()}"
        )
    return ssl_config


def apply_unlabeled_pool_args(args, ssl_config):
    """Let the unlabeled-pool CLI arguments override the SSL config file.

    These three shape the unlabeled candidate pool rather than the method, so
    they belong to the experiment rather than to the method config that a whole
    sweep shares. An omitted argument is ``None`` and leaves the config's own
    value alone, so a config that sets them keeps working.
    """

    overrides = {}
    for name in ("unlabeled_class_scope", "unlabeled_fraction", "max_unlabeled_samples"):
        value = getattr(args, name, None)
        if value is not None:
            overrides[name] = value
    if not overrides:
        return ssl_config
    resolved = replace(ssl_config, **overrides)
    semi_supervised.validate_ssl_config(resolved)
    return resolved


def restarts_selection_after_warmup(ssl_config, regularizer):
    """Whether this run discards its warm-up validation history at some point.

    Two settings ask for it, and they fire at different epochs. SLADE's
    ``within_fold_teacher_student`` has no choice and no latitude: its warm-up
    model is the teacher, a training input rather than a candidate student, so a
    teacher checkpoint must never win selection for a run whose output promises a
    student. That restart is pinned to ``warmup_epochs``.

    ``restart_selection_after_warmup`` asks for it deliberately, so that
    ``best_valid_*`` and the patience budget both describe the SSL phase alone --
    without it a warm-up that already peaks makes the SSL phase unmeasurable,
    since the objective is then a warm-up epoch's score no matter what the
    method's own hyperparameters do. That restart waits for
    ``selection_restart_is_due`` below, because ``warmup_epochs`` is not
    necessarily where the method starts being itself.
    """

    return bool(
        (
            regularizer is not None
            and getattr(regularizer, "within_fold_teacher_student", False)
        )
        or ssl_config.restart_selection_after_warmup
    )


def selection_restart_is_due(ssl_config, regularizer, num_epoch):
    """Whether the SSL phase's own baseline may be taken after this epoch.

    ``warmup_epochs`` is the floor, not the answer. A regularizer whose objective
    starts up in stages is still not itself for some epochs past that boundary --
    SLADE detaches the embedding for ``basis_warmup_steps`` and mines nothing
    until its two Gaussians separate -- and rebaselining inside such a stage just
    renames the problem the restart exists to solve: the stage boundary becomes
    the new warm-up boundary, and where its length is a searched hyperparameter
    (``basis_warmup_epochs`` is, on every shipped SLADE space) HPO can score a
    trial better by starting up more slowly.

    A regularizer that declares no start-up stages, and a pseudo-label method
    with no regularizer at all, are ready at the boundary itself, which is the
    behavior this had before the stages were taken into account.

    The result is not monotone -- SLADE's mining gate can switch back off -- so
    the caller latches the first ``True`` rather than tracking this per epoch.
    """

    if num_epoch < ssl_config.warmup_epochs:
        return False
    if regularizer is None:
        return True
    return bool(regularizer.steady_state_active())


def get_selection_metric_value(selection_metric, precision_at_1, mean_average_precision_at_r):
    if selection_metric == SELECTION_METRIC_PRECISION_AT_1:
        return precision_at_1
    if selection_metric == SELECTION_METRIC_MAP_AT_R:
        return mean_average_precision_at_r
    raise ValueError(f"Unknown selection metric: {selection_metric}")


def capture_non_backbone_model_state(model):
    """Clone all model state except the frozen DINO backbone into CPU memory."""

    full_state = model.state_dict()
    snapshot = OrderedDict()
    for name, value in full_state.items():
        if name.startswith(FROZEN_BACKBONE_STATE_PREFIX):
            continue
        if torch.is_tensor(value):
            snapshot[name] = value.detach().to(device="cpu", copy=True)
        else:
            snapshot[name] = copy.deepcopy(value)

    # Preserve per-module serialization versions for custom trainable heads.
    metadata = getattr(full_state, "_metadata", None)
    if metadata is not None:
        snapshot._metadata = {
            name: copy.deepcopy(value)
            for name, value in metadata.items()
            if name != "dinov2" and not name.startswith(FROZEN_BACKBONE_STATE_PREFIX)
        }
    return snapshot


def restore_non_backbone_model_state(model, state):
    """Restore a partial state while requiring every omitted key to be DINO state."""

    incompatible = model.load_state_dict(state, strict=False)
    missing_non_backbone = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(FROZEN_BACKBONE_STATE_PREFIX)
    ]
    if missing_non_backbone or incompatible.unexpected_keys:
        raise RuntimeError(
            "Invalid non-backbone checkpoint: "
            f"missing non-backbone keys={missing_non_backbone}, "
            f"unexpected keys={incompatible.unexpected_keys}"
        )


class _BestModelCheckpoint:
    """Store either the full model on disk or only non-backbone state in memory."""

    def __init__(self, model, path, *, non_backbone_in_memory):
        self.model = model
        self.path = Path(path)
        self.non_backbone_in_memory = bool(non_backbone_in_memory)
        self._state = None

    def save(self):
        if self.non_backbone_in_memory:
            self._state = capture_non_backbone_model_state(self.model)
        else:
            torch.save(self.model.state_dict(), self.path)

    def restore(self, *, consume=True):
        if self.non_backbone_in_memory:
            if self._state is None:
                raise RuntimeError("Cannot restore an in-memory checkpoint before saving it")
            restore_non_backbone_model_state(self.model, self._state)
            # The warm-up boundary restores without claiming: selection may
            # still be live past it, and the end-of-run restore needs the state
            # to still be here.
            if consume:
                self._state = None
        else:
            self.model.load_state_dict(torch.load(self.path, weights_only=True))

    def cleanup(self):
        self._state = None
        if not self.non_backbone_in_memory:
            self.path.unlink(missing_ok=True)


def make_run_model(args, ssl_config, regularizer=None):
    """Build the model a run trains, including any method-specific extra head.

    Reloading a saved head means rebuilding exactly the module tree that head
    came from, so the layout is decided here instead of inline in
    :func:`run_training`.
    """

    if ssl_config.method in semi_supervised.LOSS_DRIVEN_METHODS:
        stml_params = dict(getattr(args, "loss_params", {}))
        head_kwargs = {
            "stml": True,
            "stml_g_dim": getattr(args, "stml_g_dim", None),
            "stml_normalize_student": bool(stml_params.get("normalize_student", False)),
        }
    else:
        head_kwargs = {} if regularizer is None else regularizer.model_kwargs(args)
    return DinoWrapper(
        dino_size=args.dino_size,
        feat_dim=args.feat_dim,
        backbone_tuning=args.backbone_tuning,
        use_cache=bool(args.use_cache),
        cache_dir=Path("data") / args.dataset / "backbone_cache",
        projection_layers=get_projection_layers(args),
        projection_hidden_dim=getattr(args, "projection_hidden_dim", None),
        **head_kwargs,
    )


def save_head_state(model, args, ssl_config, selected_epoch, cv_fold, output_dir):
    """Persist the fitted projection head so it can be measured again later.

    With a frozen backbone the non-backbone state is the whole fitted model, so
    this file and the run's own ``run_config.json`` are everything a later
    evaluation needs. The trained criterion is deliberately left out: retrieval
    reads embeddings, and proxies or class weights produce none.
    """

    payload = {
        "format_version": 1,
        "state_dict": capture_non_backbone_model_state(model),
        "metadata": {
            "backbone_tuning": args.backbone_tuning,
            "dino_size": args.dino_size,
            "feat_dim": int(model.feat_dim),
            "projection_layers": get_projection_layers(args),
            "projection_hidden_dim": getattr(args, "projection_hidden_dim", None),
            "ssl_method": ssl_config.method,
            "selected_epoch": int(selected_epoch),
            "cv_fold": None if cv_fold is None else int(cv_fold),
            "cv_k": int(args.cv_k),
            "cv_mode": args.cv_mode,
            "dataset": args.dataset,
            "dataset_protocol": args.dataset_protocol,
            "seed": int(args.seed),
            "data_split_seed": int(get_data_split_seed(args)),
            "support_seed": getattr(ssl_config, "support_seed", None),
            "log_dir": str(args.log_dir),
        },
    }
    output_path = Path(output_dir) / HEAD_STATE_FILENAME
    torch.save(payload, output_path)
    logger.info(
        f"Saved the projection head selected at epoch {selected_epoch} to {output_path}"
    )
    return output_path


def load_head_state(path):
    """Read one saved head, rejecting a file this revision cannot interpret."""

    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path} is not a saved projection head")
    format_version = payload.get("format_version")
    if format_version != 1:
        raise ValueError(
            f"{path} was written in head-state format {format_version}, "
            "which this revision cannot read"
        )
    return payload


def restore_head_state(model, payload):
    """Load a saved head into a model rebuilt from the same run configuration.

    Methods such as SLADE and SERAPH attach training-only heads sized from the
    train split, which an evaluation never rebuilds. Their entries are dropped
    rather than failing the load; every parameter the rebuilt model does have
    must still be present, which ``restore_non_backbone_model_state`` enforces.
    """

    saved_state = payload["state_dict"]
    expected_names = set(model.state_dict())
    training_only_names = sorted(set(saved_state) - expected_names)
    if training_only_names:
        logger.info(
            f"Ignoring {len(training_only_names)} training-only entries of the saved head: "
            f"{training_only_names}"
        )
    restore_non_backbone_model_state(
        model,
        {name: value for name, value in saved_state.items() if name in expected_names},
    )


def restore_slade_teacher_projection_head(model, payload):
    """Promote a saved SLADE student's retrieval head into a fresh fold.

    The frozen DINO backbone is shared implicitly and the paper's teacher uses
    only the retrieval embedding to generate cluster IDs.  ``slade_basis`` is a
    student-only module whose rows correspond to the current fold's labeled
    classes, so neither it nor its running Gaussian statistics may cross the
    fold boundary.  A fresh optimizer is built later by :func:`run_training`.
    """

    metadata = payload.get("metadata", {})
    saved_tuning = metadata.get("backbone_tuning")
    if saved_tuning is not None and saved_tuning != BACKBONE_TUNING_FROZEN:
        raise RuntimeError(
            "SLADE teacher handoff requires a head saved from a frozen backbone, "
            f"got backbone_tuning={saved_tuning!r}"
        )

    current_state = model.state_dict()
    projection_names = sorted(
        name for name in current_state if name == "fc" or name.startswith("fc.")
    )
    if not projection_names:
        raise RuntimeError("SLADE teacher handoff found no trainable projection-head state")

    saved_state = payload.get("state_dict", {})
    missing = [name for name in projection_names if name not in saved_state]
    if missing:
        raise RuntimeError(
            "SLADE teacher handoff checkpoint is missing projection-head entries: "
            f"{missing}"
        )
    shape_mismatches = [
        (name, tuple(saved_state[name].shape), tuple(current_state[name].shape))
        for name in projection_names
        if torch.is_tensor(saved_state[name])
        and torch.is_tensor(current_state[name])
        and saved_state[name].shape != current_state[name].shape
    ]
    if shape_mismatches:
        raise RuntimeError(
            "SLADE teacher handoff projection shape mismatch: "
            f"{shape_mismatches}"
        )

    incompatible = model.load_state_dict(
        {name: saved_state[name] for name in projection_names},
        strict=False,
    )
    missing_projection = [
        name for name in incompatible.missing_keys if name in projection_names
    ]
    if missing_projection or incompatible.unexpected_keys:
        raise RuntimeError(
            "Invalid SLADE teacher projection checkpoint: "
            f"missing projection keys={missing_projection}, "
            f"unexpected keys={incompatible.unexpected_keys}"
        )
    logger.info(
        f"Promoted {len(projection_names)} projection-head state entries from the "
        "previous SLADE student; initialized a fresh fold-local basis"
    )


def _slade_lifecycle_flag_enabled(ssl_config, name):
    """Read one boolean SLADE lifecycle flag without constructing a regularizer."""

    if getattr(ssl_config, "method", None) != "slade":
        return False
    method_params = getattr(ssl_config, "method_params", {})
    if not isinstance(method_params, dict):
        return False
    regularizer_params = method_params.get("regularizer_params", {})
    if not isinstance(regularizer_params, dict):
        return False
    return regularizer_params.get(name, False) is True


def slade_cross_fold_teacher_handoff_enabled(ssl_config):
    """Whether a selected SLADE student also initializes the following fold."""

    return _slade_lifecycle_flag_enabled(ssl_config, "cross_fold_teacher_handoff")


def uses_ssl_warmup_objective(ssl_config):
    """Return whether this run has labeled-only SSL warmup epochs."""

    return ssl_config.enabled and ssl_config.warmup_epochs > 0


def uses_shared_mixed_lp_proxy_warmup(args, ssl_config):
    """Whether labeled warmup must continue the final method's proxy state."""

    return (
        uses_ssl_warmup_objective(ssl_config)
        and ssl_config.method == "mixed_label_propagation"
        and args.loss == "MixedLabelPropagationProxyLoss"
    )


def uses_main_objective_for_warmup(args, ssl_config):
    """Whether warmup and main training use one objective instance."""

    if uses_shared_mixed_lp_proxy_warmup(args, ssl_config):
        return True
    if not uses_ssl_warmup_objective(ssl_config):
        return False
    return (
        args.warmup_loss == args.loss
        and dict(args.warmup_loss_params) == dict(args.loss_params)
        and args.warmup_miner == args.miner
        and dict(args.warmup_miner_params) == dict(args.miner_params)
    )


def should_rebuild_pseudo_label_training_dataset(
    ssl_config,
    num_epoch,
    last_rebuild_epoch,
    refresh_due=False,
):
    """Return whether an enabled pseudo-label method needs a new train dataset.

    ``refresh_due`` carries the sample-scoped schedule, whose interval an epoch
    index cannot express; see :class:`_SampleScopedRefresh`.
    """

    if not ssl_config.enabled:
        return False
    if refresh_due:
        return True
    return semi_supervised.should_rebuild_on_epoch(
        ssl_config.update_mode,
        ssl_config.update_interval_epochs,
        num_epoch,
        last_rebuild_epoch,
    )


class _SampleScopedRefresh:
    """Rebuild an SSL graph on a consumed-sample schedule instead of per epoch.

    An epoch-scoped interval ties how stale the graph gets to how large the
    training pool is: ``length_before_new_iter`` is resolved from the fold's own
    labeled + unlabeled pool, so one ``every_epoch`` rebuild covers ~46k samples
    of drift on semi-iNat and a fraction of that on Cars196. Counting samples
    instead keeps one configured cadence meaning the same amount of training
    everywhere, at the price of rebuilding inside an epoch when the interval is
    shorter than one.

    The interval is configured in samples rather than steps so it stays
    independent of the batch size a trial draws; ``interval_steps`` is that
    conversion, done once per run.
    """

    def __init__(self, interval_steps, rebuild):
        self.interval_steps = int(interval_steps)
        if self.interval_steps <= 0:
            raise ValueError("sample-scoped refresh interval must be at least one step")
        self._rebuild = rebuild
        self.steps_since_rebuild = 0
        self.rebuilds = 0
        # The loader currently being iterated, which a mid-epoch rebuild replaces
        # and the caller shuts down once the epoch ends.
        self.current_loader = None

    @property
    def due(self):
        return self.steps_since_rebuild >= self.interval_steps

    def count_step(self):
        self.steps_since_rebuild += 1

    def note_rebuild(self, loader):
        """Record a rebuild the epoch-start path performed."""

        self.steps_since_rebuild = 0
        self.rebuilds += 1
        self.current_loader = loader

    def rebuild_loader(self, epoch):
        """Rebuild mid-epoch and return the loader the rest of the epoch uses."""

        loader = self._rebuild(epoch, self.rebuilds, self.current_loader)
        self.note_rebuild(loader)
        return loader


def resolve_sample_scoped_refresh_interval_steps(args, ssl_config):
    """Return the sample-scoped refresh interval in training steps, or ``None``.

    ``None`` means the run keeps the epoch-scoped schedule, which is the default.
    """

    if ssl_config.update_mode != "every_n_samples":
        return None
    batch_size = int(args.batch_size)
    interval_steps = semi_supervised.sample_scoped_refresh_interval_steps(
        ssl_config.update_interval_samples,
        batch_size,
    )
    epoch_samples = getattr(args, "length_before_new_iter", None)
    epoch_steps = (
        None if epoch_samples is None else max(1, int(epoch_samples) // batch_size)
    )
    logger.info(
        "SSL refresh schedule: every "
        f"{int(ssl_config.update_interval_samples)} training samples = "
        f"{interval_steps} steps at batch_size={batch_size}"
        + (
            ""
            if epoch_steps is None
            else (
                f" ({epoch_steps} steps per epoch over {int(epoch_samples)} samples, "
                f"{epoch_steps / interval_steps:.2f} rebuilds per epoch)"
            )
        )
    )
    return interval_steps


def should_write_split_manifest(args):
    """Return whether this run persists its split arrays under ``split/``.

    An HPO study reuses one dataset/support seed for every trial, so each trial
    and cross-validation fold directory would otherwise repeat the same arrays.
    Runs outside a study, including the final and train/validation replays,
    always write them.
    """

    if bool(getattr(args, "save_trial_split_data", False)):
        return True
    return not utils.is_hpo_trial_run(args)


def write_split_manifest(log_dir, dataset_bundle, ssl_config, ssl_split):
    """Persist the exact split so an experiment can be audited or reproduced.

    ``*_positions`` refer to offsets inside the current training subset,
    whereas ``*_indices`` refer to samples in the underlying source dataset.
    """

    split_dir = Path(log_dir) / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    if ssl_split is None:
        # No explicit label budget means every position in the current training
        # subset is treated as labeled and there is no unlabeled candidate pool.
        labeled_positions = np.arange(len(dataset_bundle.train_dataset), dtype=np.int64)
        unlabeled_positions = np.array([], dtype=np.int64)
    else:
        # Copy into stable integer arrays before saving/indexing, regardless of
        # which selector produced the split.
        labeled_positions = np.asarray(ssl_split.labeled_positions, dtype=np.int64)
        unlabeled_positions = np.asarray(ssl_split.unlabeled_positions, dtype=np.int64)

    # Subset.indices maps current positions back to the underlying source
    # dataset. Saving both forms makes later debugging much less ambiguous.
    train_indices = get_subset_indices(dataset_bundle.train_dataset)
    val_indices = get_subset_indices(dataset_bundle.valid_dataset)

    np.save(split_dir / "labeled_positions.npy", labeled_positions)
    np.save(split_dir / "unlabeled_positions.npy", unlabeled_positions)
    np.save(split_dir / "val_indices.npy", val_indices)
    np.save(split_dir / "train_indices.npy", train_indices)
    # Convert labeled/unlabeled positions through train_indices so these files
    # refer to source samples even if the training subset is later rebuilt.
    np.save(split_dir / "labeled_indices.npy", positions_to_indices(train_indices, labeled_positions))
    np.save(split_dir / "unlabeled_indices.npy", positions_to_indices(train_indices, unlabeled_positions))

    write_json(
        split_dir / "split_info.json",
        {
            "ssl_config": ssl_config.to_dict(),
            "dataset_split": dataset_bundle.split_info,
            "train_size": len(dataset_bundle.train_dataset),
            "valid_size": len(dataset_bundle.valid_dataset),
            "num_labeled": len(labeled_positions),
            "num_unlabeled": len(unlabeled_positions),
            "labeled_label_counts": label_counts(dataset_bundle.train_dataset.labels, labeled_positions),
            "unlabeled_label_counts": label_counts(dataset_bundle.train_dataset.labels, unlabeled_positions),
        },
    )
    write_json(split_dir / "test_info.json", make_test_info(dataset_bundle.test_dataset))


NATIVE_UNLABELED_POOLS = {
    utils.NATIVE_UNLABELED_SEMI_AVES: {
        "label": "Semi-Aves",
        "fraction_arg": "semi_aves_ood_fraction",
        "seed_arg": "semi_aves_ood_seed",
        "default_fraction": utils.SEMI_AVES_DEFAULT_OOD_FRACTION,
        "default_seed": utils.SEMI_AVES_DEFAULT_OOD_SEED,
        "source": "semi_aves_original_u_train_out",
        "append": lambda *call_args, **call_kwargs: (
            utils.append_semi_aves_native_unlabeled_dataset(
                *call_args, **call_kwargs
            )
        ),
    },
    utils.NATIVE_UNLABELED_SEMI_INAT: {
        "label": "Semi-iNat",
        "fraction_arg": "semi_inat_ood_fraction",
        "seed_arg": "semi_inat_ood_seed",
        "default_fraction": utils.SEMI_INAT_DEFAULT_OOD_FRACTION,
        "default_seed": utils.SEMI_INAT_DEFAULT_OOD_SEED,
        "source": "semi_inat_original_u_train_out",
        "append": lambda *call_args, **call_kwargs: (
            utils.append_semi_inat_native_unlabeled_dataset(
                *call_args, **call_kwargs
            )
        ),
    },
}


# Sources that attach images from outside the training split; the split's own
# candidates and the labeled-support ablation need none of the external checks.
EXTERNAL_UNLABELED_SOURCES = ("external", "split_and_external")


def _apply_labeled_unlabeled_pool(args, dataset_bundle, ssl_split):
    """Point the unlabeled objective at the labeled support, labels hidden.

    The loss control for a semi-supervised claim: the regularizer still runs,
    with the same weight and the same number of updates, but every sample it
    sees is one the supervised term already trains on. What separates this run
    from ``unlabeled_source='split'`` is therefore the unlabeled *data*, not the
    unlabeled *objective*, which is the confound a plain SSL-vs-supervised gap
    cannot resolve. Only the pool changes here: the labels stay hidden from the
    regularizer exactly as they are for genuine unlabeled candidates, because
    the positions are handed over without their labels.
    """

    if ssl_split is None:
        raise ValueError("unlabeled_source='labeled' requires a semi-supervised split")
    labeled_positions = np.asarray(ssl_split.labeled_positions, dtype=np.int64)
    if len(labeled_positions) == 0:
        raise ValueError("unlabeled_source='labeled' requires a non-empty labeled support")
    discarded = int(len(np.asarray(ssl_split.unlabeled_positions, dtype=np.int64)))
    ssl_split = semi_supervised.SemiSupervisedSplit(
        labeled_positions=labeled_positions,
        unlabeled_positions=labeled_positions.copy(),
    )
    if dataset_bundle.split_info is None:
        dataset_bundle.split_info = {}
    dataset_bundle.split_info["labeled_unlabeled_pool"] = {
        "source": "labeled",
        "unlabeled_size": int(len(labeled_positions)),
        "discarded_split_unlabeled_size": discarded,
    }
    # The sampler budget is resolved later from the split. Record what the split
    # offered before the swap so the control keeps the semi-supervised arm's
    # updates per epoch instead of shrinking to the labeled support.
    args.labeled_pool_replaced_unlabeled_size = discarded
    logger.info(
        "unlabeled_source='labeled' (loss control): the unlabeled objective is applied to "
        f"the {len(labeled_positions)} labeled samples with their labels hidden; the split's "
        f"{discarded} genuinely unlabeled candidates are discarded"
    )
    return dataset_bundle, ssl_split


def configure_external_unlabeled_pool(args, dataset_bundle, ssl_split):
    """Optionally replace or extend split-derived unlabeled candidates."""

    source = getattr(args, "unlabeled_source", "split")
    external_dir = getattr(args, "external_unlabeled_dir", None)
    external_filter = getattr(args, "external_unlabeled_filter", utils.EXTERNAL_UNLABELED_FILTER_NONE)
    if source == "labeled":
        return _apply_labeled_unlabeled_pool(args, dataset_bundle, ssl_split)
    protocol_info = (
        (dataset_bundle.split_info or {}).get("dataset_protocol", {})
    )
    if source == "split" and protocol_info.get("auto_native_unlabeled_pool"):
        pool_spec = NATIVE_UNLABELED_POOLS[
            protocol_info.get(
                "native_unlabeled_dataset",
                utils.NATIVE_UNLABELED_SEMI_AVES,
            )
        ]
        pool_label = pool_spec["label"]
        out_of_class_fraction = float(
            getattr(
                args,
                pool_spec["fraction_arg"],
                pool_spec["default_fraction"],
            )
        )
        out_of_class_seed = getattr(
            args,
            pool_spec["seed_arg"],
            pool_spec["default_seed"],
        )
        if out_of_class_fraction <= 0:
            # The out-of-class pool is opt-in, so a zero mismatch level keeps
            # the in-class split untouched instead of appending an empty pool.
            dataset_bundle.split_info["native_unlabeled_pool"] = {
                "source": pool_spec["source"],
                "dataset_root": str(protocol_info["dataset_root"]),
                "attached": False,
                "out_of_class_fraction_requested": out_of_class_fraction,
            }
            logger.info(
                f"{pool_label} out-of-class unlabeled pool disabled "
                f"({pool_spec['fraction_arg']}=0); training on in-class data only"
            )
            return dataset_bundle, ssl_split
        if ssl_split is None:
            raise ValueError(
                f"The {pool_label} native unlabeled pool requires a "
                "semi-supervised split"
            )
        internal_train_size = len(dataset_bundle.train_dataset)
        combined_dataset, native_dataset = pool_spec["append"](
            dataset_bundle.train_dataset,
            protocol_info["dataset_root"],
            out_of_class_fraction=out_of_class_fraction,
            out_of_class_seed=out_of_class_seed,
        )
        native_positions = np.arange(
            internal_train_size,
            internal_train_size + len(native_dataset),
            dtype=np.int64,
        )
        internal_unlabeled_positions = np.asarray(
            ssl_split.unlabeled_positions,
            dtype=np.int64,
        )
        dataset_bundle.train_dataset = combined_dataset
        ssl_split = semi_supervised.SemiSupervisedSplit(
            labeled_positions=np.asarray(
                ssl_split.labeled_positions,
                dtype=np.int64,
            ),
            unlabeled_positions=np.concatenate(
                [internal_unlabeled_positions, native_positions]
            ),
        )
        filter_info = dict(native_dataset.filter_info)
        dataset_bundle.split_info["native_unlabeled_pool"] = {
            "source": pool_spec["source"],
            "dataset_root": str(protocol_info["dataset_root"]),
            "attached": True,
            "internal_train_size": int(internal_train_size),
            "internal_unlabeled_size": int(len(internal_unlabeled_positions)),
            "native_unlabeled_size": int(len(native_dataset)),
            "filter_info": filter_info,
        }
        logger.info(
            f"{pool_label} out-of-class unlabeled pool: "
            f"{filter_info['kept_images']} hidden-label U-out images "
            f"({filter_info['out_of_class_fraction_realized']:.3f} of the "
            f"{filter_info['available_out_of_class_images']} out-of-class "
            f"images; {filter_info['excluded_out_of_class_images']} excluded)"
        )
        return dataset_bundle, ssl_split
    if source == "split":
        return dataset_bundle, ssl_split
    if ssl_split is None:
        raise ValueError(f"unlabeled_source={source!r} requires a semi-supervised split")

    internal_train_size = len(dataset_bundle.train_dataset)
    combined_dataset, external_dataset = utils.append_external_unlabeled_dataset(
        dataset_bundle.train_dataset,
        external_dir,
        external_filter=external_filter,
        compcars_min_model_images=getattr(args, "compcars_min_model_images", 100),
        compcars_strict_paper_counts=getattr(args, "compcars_strict_paper_counts", False),
        compcars_paper_threshold_calibration=getattr(
            args,
            "compcars_paper_threshold_calibration",
            utils.COMPCARS_PAPER_THRESHOLD_CALIBRATION_AUTO,
        ),
    )
    external_positions = np.arange(
        internal_train_size,
        internal_train_size + len(external_dataset),
        dtype=np.int64,
    )
    internal_unlabeled_positions = (
        np.asarray(ssl_split.unlabeled_positions, dtype=np.int64)
        if source == "split_and_external"
        else np.array([], dtype=np.int64)
    )
    dataset_bundle.train_dataset = combined_dataset
    ssl_split = semi_supervised.SemiSupervisedSplit(
        labeled_positions=np.asarray(ssl_split.labeled_positions, dtype=np.int64),
        unlabeled_positions=np.concatenate([internal_unlabeled_positions, external_positions]),
    )
    if dataset_bundle.split_info is None:
        dataset_bundle.split_info = {}
    dataset_bundle.split_info["external_unlabeled_pool"] = {
        "source": source,
        "external_root": str(external_dir),
        "internal_train_size": int(internal_train_size),
        "internal_unlabeled_size": int(len(internal_unlabeled_positions)),
        "external_unlabeled_size": int(len(external_dataset)),
        "external_dataset_type": type(external_dataset).__name__,
        "external_unlabeled_filter": external_filter,
    }
    filter_info = getattr(external_dataset, "filter_info", None)
    if filter_info is not None:
        dataset_bundle.split_info["external_unlabeled_pool"]["filter_info"] = filter_info
        category_unit = filter_info.get("category_unit", "model classes")
        logger.info(
            "External unlabeled filter: "
            f"mode={filter_info.get('mode')}, "
            f"candidate_source={filter_info.get('candidate_source')}, "
            f"kept={filter_info.get('kept_images')} images / "
            f"{filter_info.get('kept_model_classes')} {category_unit}, "
            f"dropped={filter_info.get('dropped_images')} images / "
            f"{filter_info.get('dropped_model_classes')} {category_unit}"
        )
        if filter_info.get("calibrated_min_images_per_model"):
            logger.info(
                "CompCars paper filter recalibrated min_images_per_model from "
                f"{filter_info.get('requested_min_images_per_model')} to "
                f"{filter_info.get('min_images_per_model')} to reproduce the published "
                f"{filter_info.get('expected_images')} images / "
                f"{filter_info.get('expected_model_classes')} model classes"
            )
        if filter_info.get("matches_expected_candidate_pool") is False:
            logger.warning(
                "CompCars candidate pool before filtering does not match the official "
                "classification split: "
                f"expected={filter_info.get('expected_candidate_images')} images / "
                f"{filter_info.get('expected_candidate_model_classes')} model classes, "
                f"got={filter_info.get('discovered_images')} images / "
                f"{filter_info.get('discovered_model_classes')} model classes"
            )
        if filter_info.get("matches_expected_counts") is False:
            paper_label = filter_info.get("paper_label") or "paper"
            logger.warning(
                f"External unlabeled filter did not match documented {paper_label} CompCars counts: "
                f"expected={filter_info.get('expected_images')} images / "
                f"{filter_info.get('expected_model_classes')} model classes, "
                f"got={filter_info.get('kept_images')} images / "
                f"{filter_info.get('kept_model_classes')} model classes. "
                f"Nearest thresholds={filter_info.get('nearest_count_thresholds')}"
            )
    logger.info(
        "External unlabeled pool: "
        f"source={source}, {len(internal_unlabeled_positions)} internal candidates, "
        f"{len(external_dataset)} external candidates from {external_dir}, "
        f"filter={external_filter}"
    )
    return dataset_bundle, ssl_split


def get_subset_indices(dataset):
    # Plain datasets already use source indices 0..N-1; Subset instances expose
    # an explicit mapping through their indices attribute.
    indices = getattr(dataset, "indices", None)
    if indices is None:
        return np.arange(len(dataset), dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def positions_to_indices(indices, positions):
    # Array indexing performs the position -> source-index lookup in one step.
    if len(indices) == 0 or len(positions) == 0:
        return np.array([], dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)[np.asarray(positions, dtype=np.int64)]


def label_counts(labels, positions=None):
    # Restrict counts to the requested subset positions when supplied.
    labels = np.asarray(labels, dtype=np.int64)
    if positions is not None:
        labels = labels[np.asarray(positions, dtype=np.int64)]
    if len(labels) == 0:
        return {}
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def make_test_info(test_dataset):
    labels = getattr(test_dataset, "labels", None)
    info = {
        "size": len(test_dataset),
        "dataset_type": type(test_dataset).__name__,
    }
    if labels is not None:
        info["num_classes"] = int(len(set(int(label) for label in labels)))
        info["label_counts"] = label_counts(labels)
    query_indices = getattr(test_dataset, "query_indices", None)
    gallery_indices = getattr(test_dataset, "gallery_indices", None)
    if query_indices is not None and gallery_indices is not None:
        info["retrieval_mode"] = "query_gallery"
        info["num_queries"] = int(len(query_indices))
        info["num_gallery"] = int(len(gallery_indices))
    return info


def get_data_split_seed(args):
    """Return the fixed seed used for validation/test split construction."""

    data_split_seed = getattr(args, "data_split_seed", None)
    return DEFAULT_DATA_SPLIT_SEED if data_split_seed is None else int(data_split_seed)


def _make_training_split(args, ssl_config, train_dataset):
    """Apply the label budget while keeping supervised and SSL supports aligned."""

    if ssl_config.enabled:
        return semi_supervised.prepare_ssl_split(train_dataset, ssl_config)
    if is_supervised_mode(args):
        logger.info("Training supervised baseline")
        return semi_supervised.prepare_label_split(train_dataset, ssl_config)
    return None


def make_labeled_support_mapper(train_dataset, split):
    """Map only the original class labels exposed by the supervised support."""

    if split is None:
        raise ValueError("A supervised label-budget run requires a labeled-position split")

    original_labels = getattr(train_dataset, "orig_labels", None)
    if original_labels is None or len(original_labels) != len(train_dataset):
        raise ValueError(
            "The supervised training dataset must expose one original label per sample "
            "before its support label mapper can be built"
        )

    labeled_positions = np.asarray(split.labeled_positions, dtype=np.int64)
    if len(labeled_positions) == 0:
        raise ValueError("The supervised labeled support must contain at least one sample")
    if np.any(labeled_positions < 0) or np.any(labeled_positions >= len(train_dataset)):
        raise ValueError("The supervised labeled support contains an out-of-range position")

    support_labels = np.asarray(original_labels, dtype=np.int64)[labeled_positions]
    return {
        int(original_label): mapped_label
        for mapped_label, original_label in enumerate(sorted(np.unique(support_labels).tolist()))
    }


def restrict_supervised_label_mapper(args, dataset_bundle, ssl_split):
    """Prevent hidden, non-support classes from allocating supervised proxies."""

    if not is_supervised_mode(args) or ssl_split is None:
        return dataset_bundle

    source_class_count = len(dataset_bundle.train_labels_mapper)
    dataset_bundle.train_labels_mapper = make_labeled_support_mapper(
        dataset_bundle.train_dataset,
        ssl_split,
    )
    support_class_count = len(dataset_bundle.train_labels_mapper)
    if support_class_count != source_class_count:
        logger.info(
            "Restricted supervised label mapping to "
            f"{support_class_count} labeled support classes "
            f"({source_class_count} classes existed in the source training pool)"
        )
    return dataset_bundle


def get_holdout_val_ratio(args):
    """Return the requested validation-slice size, or ``None`` for the default."""

    holdout_val_ratio = getattr(args, "holdout_val_ratio", None)
    if holdout_val_ratio is None:
        return None
    holdout_val_ratio = float(holdout_val_ratio)
    if not 0 < holdout_val_ratio < 1:
        raise ValueError(f"holdout_val_ratio must be in (0, 1), got {holdout_val_ratio}")
    return holdout_val_ratio


def get_validation_retrieval_mode(args):
    """Return how validation retrieval is scored: same-source or query/gallery."""

    mode = getattr(args, "validation_retrieval_mode", None)
    if mode is None:
        return utils.SAME_SOURCE_EVALUATION
    mode = str(mode).lower()
    if mode not in utils.VALIDATION_RETRIEVAL_MODES:
        raise ValueError(
            f"validation_retrieval_mode must be one of {utils.VALIDATION_RETRIEVAL_MODES}: {mode!r}"
        )
    return mode


def get_validation_gallery_fraction(args):
    """Return the per-class gallery share of a derived validation partition."""

    gallery_fraction = getattr(args, "validation_gallery_fraction", None)
    if gallery_fraction is None:
        return utils.DEFAULT_VALIDATION_GALLERY_FRACTION
    gallery_fraction = float(gallery_fraction)
    if not 0 < gallery_fraction < 1:
        raise ValueError(
            f"validation_gallery_fraction must be in (0, 1), got {gallery_fraction}"
        )
    return gallery_fraction


def apply_validation_retrieval_mode(args, dataset_bundle):
    """Optionally derive the validation query/gallery partition for this run."""

    if get_validation_retrieval_mode(args) != utils.QUERY_GALLERY_EVALUATION:
        return dataset_bundle
    valid_dataset = dataset_bundle.valid_dataset
    if valid_dataset is None or len(valid_dataset) == 0:
        # A final full-development fit trains without validation; there is
        # nothing to partition and nothing that reads the partition.
        return dataset_bundle
    return utils.apply_validation_query_gallery_split(
        dataset_bundle,
        gallery_fraction=get_validation_gallery_fraction(args),
        # The data split seed, not the run seed: every trial of one study must
        # see the same validation partition or their objectives differ by more
        # than the hyperparameters under test.
        seed=get_data_split_seed(args),
    )


def get_image_resize_mode(args):
    """Return the configured image geometry, defaulting to the historical squash."""

    image_resize_mode = getattr(args, "image_resize_mode", None)
    if image_resize_mode is None:
        # Studies saved before the mode existed record no value for it.
        return utils.DEFAULT_IMAGE_RESIZE_MODE
    return utils.validate_image_resize_mode(image_resize_mode)


def _apply_validation_configuration(args, ssl_config, dataset_bundle, ssl_split, cv_fold):
    """Materialize the requested holdout/CV validation split."""

    if getattr(args, "final_full_train", False):
        logger.info("Final HPO fit uses the complete development set without validation or early stopping")
        return dataset_bundle, ssl_split

    if args.val_mode == utils.VAL_MODE_SPLIT_AFTER_APPORTION:
        labeled_positions = None if ssl_split is None else ssl_split.labeled_positions
        unlabeled_positions = None if ssl_split is None else ssl_split.unlabeled_positions
        if cv_fold is None:
            holdout_val_ratio = get_holdout_val_ratio(args)
            dataset_bundle, labeled_positions, unlabeled_positions = utils.apply_post_apportion_validation_split(
                dataset_bundle=dataset_bundle,
                labeled_positions=labeled_positions,
                unlabeled_positions=unlabeled_positions,
                seed=ssl_config.support_seed,
                val_ratio=(
                    utils.POST_APPORTION_VAL_RATIO
                    if holdout_val_ratio is None
                    else holdout_val_ratio
                ),
            )
        else:
            dataset_bundle, labeled_positions, unlabeled_positions = utils.apply_apportioned_cross_validation_split(
                dataset_bundle=dataset_bundle,
                labeled_positions=labeled_positions,
                unlabeled_positions=unlabeled_positions,
                include_unlabeled=ssl_config.enabled,
                cv_k=args.cv_k,
                cv_fold=cv_fold,
                cv_mode=args.cv_mode,
                seed=ssl_config.support_seed,
            )
        if ssl_split is not None:
            ssl_split = semi_supervised.SemiSupervisedSplit(
                labeled_positions=labeled_positions,
                unlabeled_positions=unlabeled_positions,
            )
        return dataset_bundle, ssl_split

    if ssl_split is None:
        target_train_size = len(dataset_bundle.train_dataset)
        target_train_num_classes = len(set(int(label) for label in dataset_bundle.train_dataset.labels))
    else:
        target_train_size = len(ssl_split.labeled_positions)
        train_labels = np.asarray(dataset_bundle.train_dataset.labels, dtype=np.int64)
        target_train_num_classes = int(len(np.unique(train_labels[np.asarray(ssl_split.labeled_positions)])))
    dataset_bundle = utils.apply_validation_mode(
        dataset_bundle=dataset_bundle,
        val_mode=args.val_mode,
        target_train_size=target_train_size,
        target_train_num_classes=target_train_num_classes,
        seed=ssl_config.support_seed,
    )
    return dataset_bundle, ssl_split


def _prepare_datasets(
    args,
    ssl_config,
    cv_fold,
    *,
    precompute_frozen_features,
    augmented_frozen_feature_precompute,
    frozen_feature_train_views,
):
    """Build datasets, apply support/validation splits, and attach external data."""

    dataset_bundle = utils.setup_dataset_bundle(
        args.dataset,
        seed=args.seed,
        data_split_seed=get_data_split_seed(args),
        cv_k=args.cv_k if cv_fold is not None else 1,
        cv_fold=cv_fold,
        cv_mode=args.cv_mode,
        val_mode=args.val_mode,
        dataset_protocol=args.dataset_protocol,
        cifar_imbalance_factor=args.cifar_imbalance_factor,
        cifar_train_fraction=args.cifar_train_fraction,
        cifar_test_fraction=args.cifar_test_fraction,
        full_train=bool(getattr(args, "final_full_train", False)),
        holdout_val_ratio=get_holdout_val_ratio(args),
        image_resize_mode=get_image_resize_mode(args),
    )
    if args.use_cache and not augmented_frozen_feature_precompute:
        utils.use_feature_transform_for_training(dataset_bundle.train_dataset)
        if precompute_frozen_features:
            logger.info(
                "Frozen feature precompute enabled: training uses deterministic transforms and "
                "a source-indexed memory-mapped backbone feature matrix"
            )
        else:
            logger.info("Cache mode enabled: training uses deterministic transforms and cached DINO embeddings")
    elif augmented_frozen_feature_precompute:
        logger.info(
            "Augmented frozen feature precompute enabled: training keeps stochastic transforms while "
            f"precomputing {frozen_feature_train_views} backbone feature views per sample"
        )

    ssl_split = _make_training_split(args, ssl_config, dataset_bundle.train_dataset)
    dataset_bundle, ssl_split = _apply_validation_configuration(
        args,
        ssl_config,
        dataset_bundle,
        ssl_split,
        cv_fold,
    )
    if ssl_config.enabled:
        dataset_bundle, ssl_split = configure_external_unlabeled_pool(args, dataset_bundle, ssl_split)
    else:
        dataset_bundle = restrict_supervised_label_mapper(args, dataset_bundle, ssl_split)
    # Last, so the partition addresses the validation split the loaders will
    # actually iterate rather than an earlier version of it.
    dataset_bundle = apply_validation_retrieval_mode(args, dataset_bundle)
    return dataset_bundle, ssl_split


def override_length_before_new_iter_from_fold(args, train_dataset, ssl_split):
    """Resolve the sampler budget from an explicit override or the fold pool."""

    configured_length = getattr(
        args,
        "configured_length_before_new_iter",
        getattr(args, "length_before_new_iter", None),
    )
    if ssl_split is None:
        num_labeled = len(train_dataset)
        num_unlabeled = 0
    else:
        labeled_positions = np.asarray(ssl_split.labeled_positions, dtype=np.int64)
        unlabeled_positions = np.asarray(ssl_split.unlabeled_positions, dtype=np.int64)
        combined_positions = np.concatenate((labeled_positions, unlabeled_positions))
        if np.any((combined_positions < 0) | (combined_positions >= len(train_dataset))):
            raise ValueError("The labeled/unlabeled fold split contains an out-of-range position")
        pools_are_the_same_samples = getattr(args, "unlabeled_source", "split") == "labeled"
        if (
            not pools_are_the_same_samples
            and len(np.unique(combined_positions)) != len(combined_positions)
        ):
            raise ValueError("The labeled/unlabeled fold split must not contain overlapping positions")
        num_labeled = len(labeled_positions)
        num_unlabeled = len(unlabeled_positions)

    fold_training_length = int(num_labeled + num_unlabeled)
    if ssl_split is not None and pools_are_the_same_samples:
        # unlabeled_source='labeled' points the unlabeled objective at the
        # labeled support, so the two pools are the same samples and summing
        # them double-counts every one. Sizing the budget off the pool the
        # split actually offered keeps this control matched to the
        # semi-supervised arm in updates per epoch, which is the whole point of
        # running it: a tenth of the updates would confound the comparison with
        # a training-budget difference.
        replaced_unlabeled = getattr(args, "labeled_pool_replaced_unlabeled_size", None)
        if replaced_unlabeled is not None:
            fold_training_length = int(num_labeled + int(replaced_unlabeled))
    if fold_training_length <= 0:
        raise ValueError("The labeled + unlabeled fold training pool must not be empty")

    explicit_override = getattr(args, "length_before_new_iter_override", None)
    if explicit_override is None:
        resolved_length = fold_training_length
        source = "fold_labeled_plus_unlabeled"
    else:
        resolved_length = int(explicit_override)
        if resolved_length <= 0:
            raise ValueError("length_before_new_iter_override must be positive")
        source = "explicit_override"

    args.configured_length_before_new_iter = configured_length
    args.length_before_new_iter = resolved_length
    args.length_before_new_iter_source = source
    args.length_before_new_iter_num_labeled = int(num_labeled)
    args.length_before_new_iter_num_unlabeled = int(num_unlabeled)
    if explicit_override is None:
        if ssl_split is not None and pools_are_the_same_samples:
            logger.info(
                "Resolved length_before_new_iter to match the semi-supervised arm: "
                f"{fold_training_length} = {num_labeled} labeled + "
                f"{getattr(args, 'labeled_pool_replaced_unlabeled_size', 0)} unlabeled the split "
                f"offered, while unlabeled_source='labeled' applies the unlabeled objective to the "
                f"same {num_unlabeled} labeled samples "
                f"(configured value {configured_length!r} was overridden)"
            )
        else:
            logger.info(
                "Resolved length_before_new_iter from the complete fold training pool: "
                f"{fold_training_length} = {num_labeled} labeled + {num_unlabeled} unlabeled "
                f"(configured value {configured_length!r} was overridden)"
            )
    else:
        logger.info(
            f"Using explicit length_before_new_iter override {resolved_length} for a fold training pool "
            f"of {fold_training_length} = {num_labeled} labeled + {num_unlabeled} unlabeled"
        )
    return resolved_length


class _EpochTrainer:
    """Own the mutable state and batch mechanics for one fold's epoch loop."""

    def __init__(
        self,
        args,
        ssl_config,
        model,
        model_optimizer,
        main_objective,
        warmup_objective,
        regularizer,
        metrics_logger,
        train_labels_mapper,
        train_label_lookup,
        model_use_cache,
        step_visualizer=None,
    ):
        self.args = args
        self.ssl_config = ssl_config
        self.model = model
        self.model_optimizer = model_optimizer
        self.main_objective = main_objective
        self.warmup_objective = warmup_objective
        self.regularizer = regularizer
        self.metrics_logger = metrics_logger
        self.train_labels_mapper = train_labels_mapper
        self.train_label_lookup = train_label_lookup
        self.model_use_cache = model_use_cache
        self.step_visualizer = step_visualizer
        self.device_type = torch.device(args.device).type
        self.train_amp_enabled = bool(getattr(args, "train_amp", False))
        self.batch_timing_enabled = bool(getattr(args, "debug_batch_timing", False))
        self.batch_timing_interval = int(getattr(args, "debug_batch_timing_interval", 5))
        self.batch_loss_log_interval = int(getattr(args, "batch_loss_log_interval", 0))
        self.batch_diagnostics_enabled = bool(getattr(args, "log_batch_diagnostics", False))
        self.gradient_contribution_log_interval = int(
            getattr(args, "ssl_gradient_contribution_log_interval", 0) or 0
        )
        if (
            self.gradient_contribution_log_interval > 0
            and regularizer is not None
            and regularizer.uses_separate_optimizer_steps
        ):
            logger.warning(
                "SSL gradient contribution logging disabled for "
                f"{regularizer.name}: its ordered optimizer steps evaluate losses "
                "at different parameter points, so the joint-gradient ratios and "
                "cosine are not defined"
            )
            self.gradient_contribution_log_interval = 0
        if self.gradient_contribution_log_interval > 0 and regularizer is not None:
            logger.info(
                "SSL gradient contribution logging enabled: "
                f"regularizer={regularizer.name}, "
                f"active_batch_interval={self.gradient_contribution_log_interval}"
            )
        self._lr_scheduler_uses_batch_steps = (
            getattr(args, "lr_scheduler", LR_SCHEDULER_NONE)
            == LR_SCHEDULER_COSINE_WARM_RESTARTS
        )
        self._lr_schedulers = {}
        for optimizer in (
            model_optimizer,
            main_objective.optimizer,
            None if warmup_objective is None else warmup_objective.optimizer,
        ):
            if optimizer is None or id(optimizer) in self._lr_schedulers:
                continue
            scheduler = make_lr_scheduler(args, optimizer)
            if scheduler is not None:
                self._lr_schedulers[id(optimizer)] = scheduler
        self._lr_scheduler_active_epochs = {
            optimizer_id: 0 for optimizer_id in self._lr_schedulers
        }
        if self._lr_schedulers:
            logger.info(
                f"LR scheduler: {args.lr_scheduler}, "
                f"params={resolve_lr_scheduler_params(args)}"
            )
        # An explicit SLADE teacher/student run starts the student with the
        # teacher's fitted weights but not with its optimizer moments or its
        # partly consumed learning-rate schedule. Capture the pristine state
        # after scheduler construction (which may add ``initial_lr`` fields) so
        # the transition can reset every optimizer exactly once.
        all_optimizers = {}
        for optimizer in (
            model_optimizer,
            main_objective.optimizer,
            None if warmup_objective is None else warmup_objective.optimizer,
        ):
            if optimizer is not None:
                all_optimizers.setdefault(id(optimizer), optimizer)
        self._student_stage_optimizers = all_optimizers
        self._student_stage_optimizer_states = {
            optimizer_id: copy.deepcopy(optimizer.state_dict())
            for optimizer_id, optimizer in all_optimizers.items()
        }
        self._student_stage_scheduler_states = {
            optimizer_id: copy.deepcopy(scheduler.state_dict())
            for optimizer_id, scheduler in self._lr_schedulers.items()
        }
        self._slade_student_stage_started = False
        self.grad_norm_weights = None
        if regularizer is not None and regularizer.grad_norm_alpha is not None:
            self.grad_norm_weights = grad_norm.GradNormWeights(
                regularizer.grad_norm_alpha,
                extra_weights=regularizer.grad_norm_balanced_components(),
                supervised_weight=regularizer.supervised_weight,
                regularizer_weight=regularizer.regularizer_weight,
                lr=regularizer.grad_norm_lr,
                update_interval=regularizer.grad_norm_update_interval,
                renormalize=regularizer.grad_norm_renormalize,
                parameterization=regularizer.grad_norm_parameterization,
                device=args.device,
            )
            grad_norm.log_grad_norm_configuration(regularizer)
        self.target_ratio_statistic = (
            TARGET_RATIO_STATISTIC_L2
            if regularizer is None
            else getattr(
                regularizer,
                "regularizer_target_ratio_statistic",
                TARGET_RATIO_STATISTIC_L2,
            )
        )
        self.target_ratio_statistic_anchor = (
            True
            if regularizer is None
            else bool(
                getattr(regularizer, "regularizer_target_ratio_statistic_anchor", True)
            )
        )
        # Measured once, at the first usable probe, then held for the run.
        self._statistic_anchor_scale = (
            1.0
            if self.target_ratio_statistic == TARGET_RATIO_STATISTIC_L2
            or not self.target_ratio_statistic_anchor
            else None
        )
        self.gradient_surgery = (
            None if regularizer is None else getattr(regularizer, "gradient_surgery", None)
        )
        if self.gradient_surgery is not None:
            if regularizer.uses_separate_optimizer_steps:
                raise ValueError(
                    f"gradient_surgery is not supported for {regularizer.name}: it "
                    "steps the optimizer between its own ordered losses, so there is "
                    "no single pair of gradients to project apart"
                )
            if regularizer.extra_loss_components() or regularizer.extra_component_names:
                raise ValueError(
                    f"gradient_surgery is not supported for {regularizer.name}: it adds "
                    f"{sorted(regularizer.extra_component_names)} to the objective "
                    "beyond the two weighted losses, and the projection is defined "
                    "here for the supervised/regularizer pair only"
                )

        self.target_ratio = (
            None if regularizer is None else regularizer.regularizer_target_ratio
        )
        self.target_ratio_configured_batches = (
            0 if regularizer is None else regularizer.regularizer_target_ratio_batches
        )
        self.target_ratio_batches = self.target_ratio_configured_batches
        self.target_ratio_recalibration_interval_samples = (
            TARGET_RATIO_RECALIBRATION_INTERVAL_SAMPLES
            if regularizer is None
            else regularizer.regularizer_target_ratio_recalibration_interval_samples
        )
        self.target_ratio_update_interval_samples = (
            None
            if regularizer is None
            else regularizer.regularizer_target_ratio_update_interval_samples
        )
        self.target_ratio_probe_memory = (
            None
            if regularizer is None
            else getattr(regularizer, "regularizer_target_ratio_probe_memory", None)
        )
        self.target_ratio_ema_alpha = (
            None
            if regularizer is None
            else getattr(regularizer, "regularizer_target_ratio_ema_alpha", None)
        )
        self.target_ratio_aggregation = (
            "median"
            if regularizer is None
            else getattr(
                regularizer,
                "regularizer_target_ratio_aggregation",
                "median",
            )
        )
        # The sliding schedule keeps the last ``probe_memory`` measurements and
        # probes forever, so there is no window to reset, nothing to freeze once
        # a window's probes are spent, and the lookback stays constant instead of
        # sweeping from "this batch only" to "the whole window".
        self._sliding_calibration = self.target_ratio_probe_memory is not None
        # The moving average drops the buffer too: each probe folds in with
        # weight alpha. Its reciprocal is used as a reporting horizon, while
        # older probes retain exponentially decaying non-zero weight forever.
        self._ema_calibration = self.target_ratio_ema_alpha is not None
        # Both are windowless, and every scheduling decision below keys off that
        # rather than off which estimator is in use.
        self._continuous_calibration = (
            self._sliding_calibration or self._ema_calibration
        )
        self._ratio_ema = {}
        self.target_ratio_diagnostics_enabled = bool(
            regularizer is not None
            and regularizer.regularizer_target_ratio_diagnostics
        )
        # Ratios of the supervised to the unit-weight regularizer gradient norm,
        # one per scheduled gradient probe. The configured finite-buffer
        # statistic sets the calibrated weight; EMA keeps its own state below.
        self._target_ratio_measurements = self._new_calibration_buffer()
        self._target_ratio_frozen = self.target_ratio is None
        # Terms the regularizer adds in combine_losses and calibrates separately,
        # each against the supervised gradient rather than against each other.
        self._extra_target_ratios = (
            {} if regularizer is None else dict(regularizer.extra_target_ratios())
        )
        self._extra_ratio_measurements = {
            name: self._new_calibration_buffer()
            for name in self._extra_target_ratios
        }
        self._extra_ratio_frozen = set()
        # Consecutive eligible probes whose gradient was too small to invert.
        self._unusable_calibration_batches = {}
        target_ratio_enabled = bool(
            self.target_ratio is not None or self._extra_target_ratios
        )
        self._regularized_samples_per_step = max(
            1,
            int(getattr(args, "batch_size", 1)),
        )
        self._target_ratio_calibration_cycle = 1 if target_ratio_enabled else 0
        self._target_ratio_calibrations_completed = 0
        self._target_ratio_cycle_start_sample = None
        self._target_ratio_measurement_events = 0
        self._target_ratio_measurement_samples = self._new_calibration_buffer()
        self._target_ratio_probe_samples = []
        self._target_ratio_probe_diagnostics = {}
        self._next_target_ratio_measurement_sample = None
        self._next_target_ratio_recalibration_sample = None
        if target_ratio_enabled:
            self.args.regularizer_target_ratio_recalibration_interval_samples = (
                self.target_ratio_recalibration_interval_samples
            )
            self.args.regularizer_target_ratio_update_interval_samples = (
                self.target_ratio_update_interval_samples
            )
            epoch_length = getattr(args, "length_before_new_iter", None)
            effective_epoch_samples = (
                None
                if epoch_length is None
                else (
                    int(epoch_length)
                    - int(epoch_length) % self._regularized_samples_per_step
                )
            )
            if effective_epoch_samples is None:
                schedule_context = ""
            elif self._continuous_calibration:
                schedule_context = (
                    f", {effective_epoch_samples} sampled examples per epoch from "
                    "length_before_new_iter"
                )
            else:
                schedule_context = (
                    f", {effective_epoch_samples} sampled examples per epoch from "
                    "length_before_new_iter "
                    f"(~{effective_epoch_samples / self.target_ratio_recalibration_interval_samples:.2f} "
                    "windows per epoch)"
                )
            if self._continuous_calibration:
                probe_cadence = (
                    "every regularized step"
                    if self.target_ratio_update_interval_samples == 0
                    else (
                        "approximately every "
                        f"{self.target_ratio_update_interval_samples} samples"
                    )
                )
                span = self._continuous_estimator_span()
                lookback = span * max(
                    self.target_ratio_update_interval_samples,
                    self._regularized_samples_per_step,
                )
                estimator = (
                    "exponential moving average with alpha="
                    f"{self.target_ratio_ema_alpha:g} "
                    f"(~{span}-probe reciprocal-alpha horizon)"
                    if self._ema_calibration
                    else (
                        f"sliding {self.target_ratio_aggregation} over the last "
                        f"{span} probes"
                    )
                )
                logger.info(
                    "Target-ratio calibration schedule: one gradient probe "
                    f"{probe_cadence}; {estimator} "
                    f"(~{lookback} samples of nominal horizon), no window reset, "
                    f"batch_size={self._regularized_samples_per_step}"
                    f"{schedule_context}"
                )
                probe_schedule = None
            elif self.target_ratio_update_interval_samples == 0:
                probe_schedule = "one gradient probe on every regularized step"
            elif self.target_ratio_update_interval_samples is not None:
                probe_schedule = (
                    "one gradient probe approximately every "
                    f"{self.target_ratio_update_interval_samples} samples"
                )
            else:
                probe_schedule = (
                    f"{self.target_ratio_configured_batches} gradient probes "
                    "distributed over each window"
                )
            if probe_schedule is not None:
                logger.info(
                    "Target-ratio calibration schedule: "
                    f"{probe_schedule}; running {self.target_ratio_aggregation} "
                    "reset every "
                    f"{self.target_ratio_recalibration_interval_samples} samples, "
                    f"batch_size={self._regularized_samples_per_step}"
                    f"{schedule_context}"
                )
            if self.target_ratio is not None:
                logger.info(
                    "Regularizer weight calibration enabled: target gradient ratio "
                    f"{self.target_ratio:g}"
                )
            for name, ratio in self._extra_target_ratios.items():
                logger.info(
                    f"{name} weight calibration enabled: target gradient ratio {ratio:g}"
                )
        self.teacher_model = None
        self.regularizer_state = None
        self.global_step = 0
        self.regularization_step = 0
        self.regularization_samples_seen = 0
        self.progress_bar = None
        self._compile_requested = bool(getattr(args, "compile_train_step", False))
        self._compiled_forward_loss = {}
        # Set once a compile fails: the capture is a throughput optimization, so
        # the rest of the run continues down the eager path it is tested against.
        self._compile_fallback = False
        self._graph_projection_requested = bool(getattr(args, "graph_projection", False))
        # Keyed by batch shape: the joint labeled+unlabeled forward and a
        # labeled-only warm-up epoch are different shapes, and a graph replays
        # one shape only.
        self._graphed_projections = {}
        self._graph_projection_skip_logged = False

    def _fused_forward_loss(self, phase, images, sample_weights):
        """Return a bit-identical CUDA-graph projection+loss, when supported.

        The ``cudagraphs`` backend replays the same ATen kernels rather than
        fusing and reassociating their arithmetic as Inductor does.  That keeps
        the eager loss and gradients bit-identical while removing Python and
        kernel-launch overhead.  Only fixed-shape loss paths with no dynamic
        pair compaction are admitted here.
        """

        if not self._compile_requested:
            return None
        objective = phase.objective
        if (
            self.device_type != "cuda"
            or self.train_amp_enabled
            or phase.standalone_stml
            or phase.regularization_active
            or phase.regularizer_disabled
            or getattr(objective.criterion, "supports_sample_weights", False)
            or not utils.is_precomputed_feature_batch(images)
            or not hasattr(self.model, "project_features")
        ):
            return None

        criterion = objective.criterion
        miner = objective.miner
        fused_miner_forward = getattr(criterion, "forward_with_miner", None)
        supports_fused_miner = getattr(criterion, "supports_fused_miner", None)
        compiled_miner_path = (
            miner is not None
            and not self.batch_diagnostics_enabled
            and getattr(criterion, "supports_fused_miner_compilation", True)
            and callable(fused_miner_forward)
            and callable(supports_fused_miner)
            and supports_fused_miner(miner)
        )
        supported_path = (
            compiled_miner_path
            or (
                objective.is_classification
                and isinstance(
                    criterion,
                    (
                        metric_losses.SyncFreeArcFaceLoss,
                        metric_losses.SyncFreeProxyAnchorLoss,
                    ),
                )
            )
        )
        if not supported_path:
            return None

        key = id(objective.criterion)
        if key not in self._compiled_forward_loss:
            model = self.model

            def forward_loss(features, labels):
                embeddings = model.project_features(features)
                if isinstance(criterion, losses.ArcFaceLoss):
                    return metric_losses.classification_loss_float32(
                        criterion,
                        embeddings,
                        labels,
                    )
                if compiled_miner_path:
                    return fused_miner_forward(embeddings, labels, miner)
                return criterion(embeddings, labels)

            logger.info(
                f"Capturing the bit-identical projection+loss step for {self.args.loss} "
                "(the first batches pay one-time CUDA graph setup)"
            )
            self._compiled_forward_loss[key] = self._compile_forward_loss(forward_loss)
        return self._compiled_forward_loss[key]

    def _compile_forward_loss(self, forward_loss):
        """Compile one projection+loss closure without letting it end the run.

        Dynamo keys its compiled-code cache on the *code object*, so every
        trainer's closure over its own model and criterion shares the single
        cache line belonging to this one ``def``.  Nothing separates those
        entries by instance, so the eight entries ``recompile_limit`` allows are
        spent across folds and trials rather than within one of them, and a
        class-disjoint CV fold spends one every time: it changes the
        classification criterion's weight shape, and parameter shapes are never
        made dynamic (``force_parameter_static_shapes``).  Under
        ``fullgraph=True`` the ninth compile then raises instead of falling back
        to eager, which is enough to kill a study.

        Three things keep that from happening.  Dropping the previous trainer's
        entries at this trainer's first compile keeps the cache line short,
        isolation stops a warm-up criterion from spending the main objective's
        budget, and the wrapper below turns any remaining compile failure into
        an eager run rather than a dead study.
        """

        import torch._dynamo as dynamo

        if not self._compiled_forward_loss:
            # Folds and trials run sequentially, and every entry is specialized
            # to the shapes of the trainer that made it, so no live caller can
            # reuse what this drops.
            dynamo.reset()
        compiled = torch.compile(
            forward_loss,
            backend="cudagraphs",
            fullgraph=True,
            isolate_recompiles=True,
        )

        def compiled_or_eager(features, labels):
            if self._compile_fallback:
                return forward_loss(features, labels)
            try:
                return compiled(features, labels)
            except (
                dynamo.exc.FailOnRecompileLimitHit,
                dynamo.exc.TorchDynamoException,
            ) as exc:
                # Compilation fails before any of the traced code runs, and the
                # capture is tested bit-identical to this eager call, so the
                # fallback costs throughput and nothing else.
                logger.warning(
                    "Could not compile the projection+loss step "
                    f"({type(exc).__name__}: {exc}). "
                    "Continuing this run on the eager path."
                )
                self._compile_fallback = True
                return forward_loss(features, labels)

        return compiled_or_eager

    def _graph_projection_block_reason(self):
        """Why the projection graph stays off for this run, or ``None``.

        Everything listed here runs a second backward through the projection:
        gradient surgery projects the per-component gradients, GradNorm and the
        target-ratio calibration measure them, and the contribution diagnostic
        logs them. A graphed node hands those passes detached views of its own
        static gradient buffers, so instead of reasoning about the lifetime of
        each extra pass, a run that uses any of them keeps the eager head. That
        is a coverage limit, not a numerical one -- lifting it needs a
        bit-identity test for the multi-pass case, not a smaller condition here.
        """

        if self.device_type != "cuda":
            return "the run is not on CUDA"
        if self.train_amp_enabled:
            # make_graphed_callables rejects autocast unless its weight cache is
            # disabled, and that cache is most of what autocast buys.
            return "--train_amp is set"
        if self.gradient_surgery is not None:
            return "gradient surgery runs a second backward per component"
        if self.grad_norm_weights is not None:
            return "GradNorm measures per-component gradients"
        if self.target_ratio is not None or self._extra_target_ratios:
            return "regularizer weight calibration measures per-component gradients"
        if self.gradient_contribution_log_interval > 0:
            return "gradient-contribution diagnostics measure per-component gradients"
        if not hasattr(self.model, "project_features"):
            return "the model has no project_features path"
        return None

    def _graphed_projection(self, images):
        """Return a CUDA-graphed ``project_features`` for this batch, or ``None``.

        The graph replays the head's own forward and backward kernels, so the
        embeddings and their gradients are the eager ones bit for bit; what it
        removes is the launch per kernel, which is what a frozen-backbone step
        is bound by. Only precomputed-feature batches qualify: an image batch
        enters through the backbone, which this does not capture.
        """

        if not self._graph_projection_requested:
            return None
        if not utils.is_precomputed_feature_batch(images):
            return None
        reason = self._graph_projection_block_reason()
        if reason is not None:
            if not self._graph_projection_skip_logged:
                logger.info(
                    f"--graph_projection requested but not used: {reason}. "
                    "The eager projection runs instead."
                )
                self._graph_projection_skip_logged = True
            return None

        # One graph per batch shape, kept for the fold: the joint
        # labeled+unlabeled forward, a labeled-only warm-up epoch, and a partial
        # last batch are three shapes, and each capture amortizes over the
        # epochs that replay it. A shape whose capture fails caches ``None`` so
        # it is not retried every step.
        key = tuple(images.shape)
        cached = self._graphed_projections.get(key)
        # A graph reads the parameter storage it captured, so it belongs to one
        # model object. Nothing replaces a trainer's model today; this keeps the
        # cache self-invalidating if something ever does.
        if cached is not None and cached[0] is self.model:
            return cached[1]

        sample = images.to(self.args.device, non_blocking=True).detach().clone()
        try:
            graphed, failure = make_graphed_projection(self.model, sample)
        except Exception as error:
            # Capture is an optimization; a run must not die for it.
            graphed = None
            failure = f"capture raised {type(error).__name__}: {error}"
        if graphed is None:
            logger.warning(
                f"Projection CUDA graph unavailable for batch shape {key}: "
                f"{failure}. Using the eager projection."
            )
        else:
            logger.info(
                f"Captured the projection head for batch shape {key}: replayed "
                "output verified equal to the eager projection"
            )
        self._graphed_projections[key] = (self.model, graphed)
        return graphed

    def _resolve_phase(self, epoch):
        warmup_active = self.ssl_config.enabled and epoch < self.ssl_config.warmup_epochs
        standalone_stml = self.ssl_config.method in semi_supervised.LOSS_DRIVEN_METHODS and not warmup_active
        regularized_phase = self.regularizer is not None and not warmup_active
        regularization_active = regularized_phase and self.regularizer.regularizer_weight > 0
        regularizer_disabled = regularized_phase and self.regularizer.regularizer_weight == 0
        objective = self.warmup_objective if warmup_active else self.main_objective

        if warmup_active:
            warmup_loss_name = (
                self.args.loss
                if uses_main_objective_for_warmup(self.args, self.ssl_config)
                else self.args.warmup_loss
            )
            description = f"warmup ({warmup_loss_name})"
        elif regularization_active:
            description = f"{self.args.loss} + {self.regularizer.name} regularization"
        elif regularizer_disabled:
            description = f"{self.args.loss} (regularizer weight 0; supervised-only sanity path)"
        else:
            description = self.args.loss
        return _EpochPhase(
            objective,
            standalone_stml,
            regularization_active,
            regularizer_disabled,
            description,
        )

    def _apply_pending_regularizer_optimizer_resets(self):
        """Drop optimizer state for parameters the regularizer re-initialized.

        Adam keeps its moments keyed by parameter object, so a module that is
        re-initialized in place would otherwise take its first steps under a
        second moment fitted to the values it just discarded. Popping the entry
        makes the optimizer treat the parameter as new.
        """

        if self.regularizer is None:
            return
        parameters = list(self.regularizer.consume_pending_optimizer_state_resets())
        if not parameters:
            return
        cleared = 0
        for optimizer in self._student_stage_optimizers.values():
            for parameter in parameters:
                if optimizer.state.pop(parameter, None) is not None:
                    cleared += 1
        logger.info(
            f"Cleared optimizer state for {cleared} of {len(parameters)} "
            f"re-initialized {self.regularizer.name} parameter(s)"
        )

    def begin_slade_student_stage(self):
        """Keep teacher weights while restarting optimization for its student.

        SLADE's teacher is needed only long enough to produce the fixed cluster
        IDs. Reusing the live module after that point is equivalent to cloning
        the teacher into a student, but carrying Adam moments or a decayed
        scheduler across the boundary would still make it one continuous
        optimization run. This reset is the material stage separation.
        """

        if self._slade_student_stage_started:
            raise RuntimeError("SLADE student stage has already started")
        for optimizer_id, optimizer in self._student_stage_optimizers.items():
            optimizer.load_state_dict(
                copy.deepcopy(self._student_stage_optimizer_states[optimizer_id])
            )
            optimizer.zero_grad(set_to_none=True)
        for optimizer_id, scheduler in self._lr_schedulers.items():
            scheduler.load_state_dict(
                copy.deepcopy(self._student_stage_scheduler_states[optimizer_id])
            )
            self._lr_scheduler_active_epochs[optimizer_id] = 0
        self._slade_student_stage_started = True
        logger.info(
            "SLADE teacher -> student boundary: retained the teacher embedding "
            "weights and reset optimizer/scheduler state for the student"
        )

    def trained_state_holders(self):
        """Every trained module whose state does not live inside ``model``.

        ``model.state_dict()`` is not the whole of what a run trains. A proxy
        loss owns the class centers the embedding is pulled toward, STML owns a
        student head, and a regularizer can own modules of its own -- all of them
        nn.Modules held beside the model rather than inside it. This is
        deliberately keyed by role and typed by ``isinstance`` rather than by a
        list of loss names: whether a criterion carries state is a property of
        the object, not of whether its name appears in CLASSIFICATION_LOSSES.
        """

        holders = {}
        for role, objective in (
            ("main_criterion", self.main_objective),
            ("warmup_criterion", self.warmup_objective),
        ):
            criterion = None if objective is None else getattr(objective, "criterion", None)
            if isinstance(criterion, torch.nn.Module):
                holders[role] = criterion
        if isinstance(self.regularizer, torch.nn.Module):
            holders["regularizer"] = self.regularizer
        return holders

    def capture_optimization_state(self):
        """Snapshot everything a checkpoint needs to be restored consistently.

        Taken next to a model checkpoint so the whole set can be restored
        together. Rolling the embedding back to an earlier epoch while leaving
        Adam's moments describing a later one is not a continuation of that
        epoch: the moments are wrong for the restored weights. Neither is
        clearing them -- a cold optimizer takes its first steps with no
        second-moment estimate to scale them by, which at a large learning rate
        is enough on its own to destroy the embedding. The same argument applies
        to every module in :meth:`trained_state_holders`: restoring an embedding
        to epoch N while its loss's proxies stay at epoch M leaves the loss
        scoring a geometry that no longer exists.
        """

        return {
            "optimizers": {
                optimizer_id: copy.deepcopy(optimizer.state_dict())
                for optimizer_id, optimizer in self._student_stage_optimizers.items()
            },
            "schedulers": {
                optimizer_id: copy.deepcopy(scheduler.state_dict())
                for optimizer_id, scheduler in self._lr_schedulers.items()
            },
            "scheduler_epochs": dict(self._lr_scheduler_active_epochs),
            "modules": {
                role: copy.deepcopy(module.state_dict())
                for role, module in self.trained_state_holders().items()
            },
        }

    def restore_optimization_state(self, snapshot, reason):
        """Reinstate a snapshot taken by :meth:`capture_optimization_state`."""

        holders = self.trained_state_holders()
        restored_modules = []
        for role, state in snapshot.get("modules", {}).items():
            module = holders.get(role)
            if module is None:
                continue
            # strict: a mismatch here means the snapshot and the live module
            # disagree about what the run trains, which is exactly the silent
            # inconsistency this method exists to prevent.
            module.load_state_dict(copy.deepcopy(state))
            restored_modules.append(role)
        for optimizer_id, state in snapshot["optimizers"].items():
            optimizer = self._student_stage_optimizers.get(optimizer_id)
            if optimizer is None:
                continue
            optimizer.load_state_dict(copy.deepcopy(state))
            optimizer.zero_grad(set_to_none=True)
        for optimizer_id, state in snapshot["schedulers"].items():
            scheduler = self._lr_schedulers.get(optimizer_id)
            if scheduler is None:
                continue
            scheduler.load_state_dict(copy.deepcopy(state))
        self._lr_scheduler_active_epochs.update(snapshot["scheduler_epochs"])
        also = f", plus {', '.join(restored_modules)}" if restored_modules else ""
        logger.info(
            "Restored the optimizer/scheduler state saved with that checkpoint"
            f"{also}: {reason}"
        )

    def reset_optimization_state(self, reason):
        """Drop optimizer moments and schedule progress while keeping the weights.

        Restoring an earlier checkpoint mid-run leaves Adam's moments, and any
        partly consumed schedule, describing the trajectory that was just thrown
        away. Carrying them onto the restored weights raises exactly the
        objection :meth:`begin_slade_student_stage` exists to answer, so the
        reset is the same one.
        """

        for optimizer_id, optimizer in self._student_stage_optimizers.items():
            optimizer.load_state_dict(
                copy.deepcopy(self._student_stage_optimizer_states[optimizer_id])
            )
            optimizer.zero_grad(set_to_none=True)
        for optimizer_id, scheduler in self._lr_schedulers.items():
            scheduler.load_state_dict(
                copy.deepcopy(self._student_stage_scheduler_states[optimizer_id])
            )
            self._lr_scheduler_active_epochs[optimizer_id] = 0
        logger.info(f"Reset optimizer/scheduler state: {reason}")

    def _compute_batch_loss(self, batch, phase, timer, data_ready_time):
        objective = phase.objective
        if phase.standalone_stml:
            images, _, instance_ids = batch
            if not isinstance(images, (list, tuple)) or len(images) != objective.criterion.num_views:
                raise ValueError(
                    f"STML batches must contain {objective.criterion.num_views} augmented views per sample"
                )
            images = torch.cat(images, dim=0)
            instance_ids = instance_ids.repeat(objective.criterion.num_views).to(
                self.args.device,
                non_blocking=True,
            )
        else:
            supervised_batch, regularizer_batch = batch if phase.regularization_active else (batch, None)
            images, labels, sample_weights, supervised_indices = unpack_training_batch(supervised_batch)
            supervised_inputs = images
            supervised_batch_size = len(labels)
            joint_forward_active = bool(
                phase.regularization_active and self.regularizer.uses_joint_forward
            )
            if joint_forward_active:
                images = concatenate_joint_forward_inputs(images, regularizer_batch)
        timer.stop("unpack_batch", data_ready_time)

        miner_outputs = None
        supervised_loss = None
        regularization_loss = None
        with torch.autocast(
            device_type=self.device_type,
            dtype=torch.bfloat16,
            enabled=self.train_amp_enabled,
        ):
            if phase.standalone_stml:
                started = timer.start()
                student_g, student_f = self.model.forward_stml_cached(images, self.args.device)
                with torch.no_grad():
                    teacher_g = self.teacher_model.forward_stml_teacher_cached(images, self.args.device)
                timer.stop("forward", started)

                started = timer.start()
                loss = objective.criterion(student_f, student_g, teacher_g, instance_ids)
                timer.stop("loss", started)
                return loss, supervised_loss, regularization_loss, miner_outputs

            started = timer.start()
            fused = self._fused_forward_loss(phase, images, sample_weights)
            if fused is not None:
                started = timer.start()
                labels = map_training_labels(
                    labels,
                    self.train_label_lookup,
                    self.train_labels_mapper,
                    self.args.device,
                )
                # The eager path transfers inside forward_model_inputs, which the
                # fused path skips. A GPU-resident loader already yields device
                # tensors and this is a no-op; a DataLoader does not.
                images = images.to(self.args.device, non_blocking=True)
                timer.stop("label_weight_prep", started)
                started = timer.start()
                supervised_loss = fused(images, labels)
                timer.stop("fused_forward_loss", started)
                # _fused_forward_loss only accepts phases where the plain
                # supervised loss is the whole objective.
                return supervised_loss, supervised_loss, regularization_loss, miner_outputs

            graphed_projection = self._graphed_projection(images)
            if graphed_projection is None:
                embeddings = utils.forward_model_inputs(
                    self.model,
                    images,
                    self.args.device,
                    use_cache=self.model_use_cache,
                )
            else:
                # .clone() because a graph writes its output into static storage
                # that the next replay overwrites, and these embeddings outlive
                # the step in places a regularizer chooses -- SLADE's Eq 6
                # statistics, a teacher snapshot. The copy is one kernel on an
                # (N, d) tensor against the launches the replay saves, and it is
                # exact, so the bit-identity the capture guarantees survives it.
                embeddings = graphed_projection(
                    images.to(self.args.device, non_blocking=True)
                ).clone()
            regularizer_embeddings = None
            if joint_forward_active:
                regularizer_embeddings = embeddings[supervised_batch_size:]
                embeddings = embeddings[:supervised_batch_size]
            timer.stop("forward", started)

            started = timer.start()
            labels = map_training_labels(
                labels,
                self.train_label_lookup,
                self.train_labels_mapper,
                self.args.device,
            )
            sample_weights = sample_weights.to(self.args.device, non_blocking=True)
            timer.stop("label_weight_prep", started)

            started = timer.start()
            supports_sample_weights = getattr(
                objective.criterion,
                "supports_sample_weights",
                False,
            )
            if isinstance(objective.criterion, losses.ArcFaceLoss):
                supervised_loss = metric_losses.classification_loss_float32(
                    objective.criterion,
                    embeddings,
                    labels,
                    sample_weights=sample_weights if supports_sample_weights else None,
                )
            elif supports_sample_weights:
                supervised_loss = objective.criterion(
                    embeddings,
                    labels,
                    sample_weights=sample_weights,
                )
            elif not objective.is_classification and objective.miner is not None:
                fused_miner_forward = getattr(
                    objective.criterion,
                    "forward_with_miner",
                    None,
                )
                supports_fused_miner = getattr(
                    objective.criterion,
                    "supports_fused_miner",
                    None,
                )
                fused_miner_active = (
                    not self.batch_diagnostics_enabled
                    and callable(fused_miner_forward)
                    and callable(supports_fused_miner)
                    and supports_fused_miner(objective.miner)
                )
                if fused_miner_active:
                    supervised_loss = fused_miner_forward(
                        embeddings,
                        labels,
                        objective.miner,
                    )
                else:
                    miner_outputs = objective.miner(embeddings, labels)
                    supervised_loss = objective.criterion(embeddings, labels, miner_outputs)
            else:
                supervised_loss = objective.criterion(embeddings, labels)
            timer.stop("miner_and_loss", started)

            if phase.regularization_active:
                visualize_step = (
                    self.step_visualizer is not None
                    and self.step_visualizer.is_due(self.regularization_step)
                )
                if visualize_step:
                    # Some artifacts describe state this step is about to
                    # overwrite, so they are drawn before the loss runs.
                    self.step_visualizer.before_regularizer_loss(
                        step=self.global_step,
                        state=self.regularizer_state,
                        supervised_inputs=supervised_inputs,
                        regularizer_batch=regularizer_batch,
                    )
                started = timer.start()
                if self.regularizer.uses_separate_optimizer_steps:
                    # This is the first of an ordered sequence of updates. The
                    # trainer steps it before asking the regularizer to produce
                    # its fresh attraction and repulsion forwards.
                    loss = self.regularizer.supervised_weight * supervised_loss
                else:
                    regularizer_loss_kwargs = {}
                    if joint_forward_active:
                        regularizer_loss_kwargs = {
                            "supervised_embeddings": embeddings,
                            "supervised_labels": labels,
                            "regularizer_embeddings": regularizer_embeddings,
                        }
                    if self.regularizer.requires_labeled_indices:
                        regularizer_loss_kwargs["supervised_indices"] = supervised_indices
                    if self.regularizer.requires_supervised_objective:
                        regularizer_loss_kwargs.update(
                            {
                                "supervised_inputs": supervised_inputs,
                                "supervised_indices": supervised_indices,
                                "supervised_criterion": objective.criterion,
                                "supervised_miner": objective.miner,
                                "supervised_is_classification": objective.is_classification,
                            }
                        )
                    regularization_loss = self.regularizer.compute_loss(
                        student_model=self.model,
                        state=self.regularizer_state,
                        batch=regularizer_batch,
                        device=self.args.device,
                        timings=timer.totals if self.batch_timing_enabled else None,
                        **regularizer_loss_kwargs,
                    )
                    loss = self.regularizer.combine_losses(
                        supervised_loss,
                        regularization_loss,
                    )
                timer.stop("regularization_loss", started)
                if visualize_step and not self.regularizer.uses_separate_optimizer_steps:
                    # The remaining artifacts are read back out of the step's own
                    # diagnostics, so they are drawn once the loss has produced them.
                    self.step_visualizer.after_regularizer_loss(
                        step=self.global_step,
                        state=self.regularizer_state,
                        diagnostics=self.regularizer.batch_diagnostics(),
                        supervised_inputs=supervised_inputs,
                        supervised_embeddings=embeddings,
                        supervised_labels=labels,
                        supervised_indices=supervised_indices,
                        regularizer_batch=regularizer_batch,
                        regularizer_embeddings=regularizer_embeddings,
                    )
            elif phase.regularizer_disabled:
                loss = self.regularizer.supervised_weight * supervised_loss
            else:
                loss = supervised_loss
        return loss, supervised_loss, regularization_loss, miner_outputs

    def _make_batch_diagnostics(
        self,
        phase,
        loss_value,
        supervised_loss,
        regularization_loss,
        miner_diagnostics,
    ):
        if not self.batch_diagnostics_enabled:
            return None
        diagnostics = {
            "train/zero_loss_batch": float(loss_value == 0.0),
            "train/gradient_norm/model": utils.gradient_l2_norm(self.model.parameters()),
            **utils.optimizer_learning_rates(self.model_optimizer, "model"),
            **miner_diagnostics,
            "train/stml_active": float(phase.standalone_stml),
            "train/regularization_active": float(phase.regularization_active),
        }
        if phase.objective.is_classification:
            diagnostics["train/gradient_norm/criterion"] = utils.gradient_l2_norm(
                phase.objective.criterion.parameters()
            )
            diagnostics.update(utils.optimizer_learning_rates(phase.objective.optimizer, "criterion"))
        if supervised_loss is not None:
            diagnostics["train/supervised_loss"] = supervised_loss.detach().item()
        if regularization_loss is not None:
            diagnostics["train/regularization_loss"] = regularization_loss.detach().item()
            diagnostics.update(self.regularizer.batch_diagnostics())
        return diagnostics

    def _update_grad_norm_weights(self, phase, supervised_loss, regularization_loss):
        """Run one GradNorm step on the two loss weights.

        The measurement has to happen before the combined backward frees the
        graph. Following the paper, this batch still trains with the weights it
        was built from; the update lands on the next one.
        """

        if (
            self.grad_norm_weights is None
            or not phase.regularization_active
            or supervised_loss is None
            or regularization_loss is None
            or not regularization_loss.requires_grad
            or not self.grad_norm_weights.is_due(self.regularization_step)
        ):
            return
        # The regularizer's own objective terms are balanced alongside the two,
        # at unit weight like them; GradNorm applies the live weights itself.
        present = self.regularizer.extra_loss_components()
        extra_components = {}
        extra_losses = {}
        for name in self.grad_norm_weights.extra_names:
            term = present.get(name)
            if term is None or term[0] is None or not term[0].requires_grad:
                # Declared but absent from this batch; update() skips it too.
                return
            extra_components[name] = (term[0], 1.0)
            extra_losses[name] = float(term[0].detach().item())
        component_norms = measure_gradient_component_norms(
            supervised_loss,
            regularization_loss,
            self._shared_trunk_parameters(),
            supervised_weight=1.0,
            regularizer_weight=1.0,
            extra_components=extra_components,
            # GradNorm reads the per-term norms only; the pair norm exists for
            # the diagnostic cosine and would cost a full-model backward pass on
            # every batch at the default update interval.
            include_pair_norm=False,
        )
        updated = self.grad_norm_weights.update(
            component_norms,
            supervised_loss.detach().item(),
            regularization_loss.detach().item(),
            extra_losses=extra_losses,
        )
        if updated:
            self.regularizer.supervised_weight = self.grad_norm_weights.supervised_weight
            self.regularizer.regularizer_weight = self.grad_norm_weights.regularizer_weight
            for name, weight in self.grad_norm_weights.extra_weights().items():
                self.regularizer.apply_calibrated_component_weight(name, weight)

    def _shared_trunk_parameters(self):
        """The shared weights every objective term's gradient can reach.

        GradNorm's paper chooses its ``W`` among exactly these, and the same
        argument applies to target-ratio calibration and to the gradient
        contribution diagnostics: all three compare one term's gradient against
        another's, which is only meaningful where both terms can act.

        Modules a regularizer attaches to the student model are reached by that
        regularizer's terms alone, so counting them would let a term's private
        head decide the balance of the shared trunk -- and for SLADE it would
        hide the basis warmup entirely, since a detached embedding leaves ``W_a``
        as the only thing the basis term moves. Measured on the Cars196 setup at
        768 dimensions with 73 basis vectors, ``W_a`` carried half the reported
        basis gradient norm on an untrained head and more once the head had
        converged, so a basis term barely reaching the embedding still logged a
        ratio close to its target.
        """

        private = set()
        for name in self.regularizer.private_model_module_names():
            module = getattr(self.model, name, None)
            if module is None:
                continue
            private.update(id(parameter) for parameter in module.parameters())
        parameters = self.model.parameters()
        if not private:
            return parameters
        return [parameter for parameter in parameters if id(parameter) not in private]

    def _new_calibration_buffer(self):
        """A finite-statistic buffer, bounded under the sliding schedule."""

        if self.target_ratio_probe_memory is not None:
            return deque(maxlen=int(self.target_ratio_probe_memory))
        if self._ema_calibration:
            # Nothing reads it back; the estimate lives in _ratio_ema.
            return deque(maxlen=1)
        return []

    def _target_ratio_batch_boundary_at_or_after(self, sample_count):
        batch_size = self._regularized_samples_per_step
        return ((int(sample_count) + batch_size - 1) // batch_size) * batch_size

    def _plan_target_ratio_probe_samples(
        self,
        window_start,
        first_available_sample,
    ):
        """Map sample-scoped probe targets onto this loader's batch boundaries."""

        window_start = int(window_start)
        first_available_sample = int(first_available_sample)
        window_samples = self.target_ratio_recalibration_interval_samples
        window_end = window_start + window_samples
        update_interval = self.target_ratio_update_interval_samples

        if self._continuous_calibration:
            # One probe now; the next is scheduled from the sample count this one
            # actually lands on, so the cadence never drifts against the window
            # arithmetic that no longer applies.
            return [
                max(
                    first_available_sample,
                    self._target_ratio_batch_boundary_at_or_after(window_start),
                )
            ]

        if update_interval == 0:
            first = max(
                first_available_sample,
                self._target_ratio_batch_boundary_at_or_after(window_start),
            )
            planned = list(
                range(first, window_end, self._regularized_samples_per_step)
            )
        else:
            if update_interval is None:
                nominal_samples = (
                    window_start
                    + round(index * window_samples / self.target_ratio_configured_batches)
                    for index in range(self.target_ratio_configured_batches)
                )
            else:
                nominal_samples = range(
                    window_start,
                    window_end,
                    update_interval,
                )
            planned = []
            for nominal_sample in nominal_samples:
                actual_sample = max(
                    first_available_sample,
                    self._target_ratio_batch_boundary_at_or_after(nominal_sample),
                )
                if actual_sample >= window_end:
                    continue
                if not planned or planned[-1] != actual_sample:
                    planned.append(actual_sample)

        # A window shorter than one batch still needs one chance to update. Its
        # completion schedules the following window from this actual boundary.
        if not planned:
            planned = [first_available_sample]
        return planned

    def _start_target_ratio_probe_schedule(
        self,
        window_start,
        first_available_sample,
    ):
        self._target_ratio_cycle_start_sample = int(window_start)
        self._target_ratio_measurement_events = 0
        self._target_ratio_measurement_samples = self._new_calibration_buffer()
        self._target_ratio_probe_samples = self._plan_target_ratio_probe_samples(
            window_start,
            first_available_sample,
        )
        self.target_ratio_batches = len(self._target_ratio_probe_samples)
        self._next_target_ratio_measurement_sample = (
            self._target_ratio_probe_samples[0]
        )
        self.args.regularizer_target_ratio_measurements_per_window = (
            self._continuous_estimator_span()
            if self._continuous_calibration
            else self.target_ratio_batches
        )

    def _surgery_parameters(self, phase):
        """Every parameter the two terms can both reach, deduplicated.

        The projection is defined over the shared trunk. Parameters only one
        term reaches -- a criterion's proxies, a regularizer's private head --
        contribute nothing to the dot product and are unchanged by subtracting a
        multiple of the other term's gradient, so including them is exact and
        saves a second backward pass for them.
        """

        seen = set()
        parameters = []
        for optimizer in self._active_optimizers(phase):
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    if not parameter.requires_grad or id(parameter) in seen:
                        continue
                    seen.add(id(parameter))
                    parameters.append(parameter)
        return tuple(parameters)

    def _measure_projected_component_norms(self, supervised_loss, regularization_loss):
        """Supervised norm and *projected* unit-weight regularizer norm."""

        # A frozen backbone leaves parameters in the trunk that autograd cannot
        # differentiate; measure_gradient_component_norms filters them the same way.
        parameters = tuple(
            parameter
            for parameter in self._shared_trunk_parameters()
            if parameter.requires_grad
        )
        if not parameters:
            return GradientContributionNorms(0.0, 0.0, ())
        supervised = gradient_surgery.component_gradients(
            supervised_loss,
            parameters,
            self.regularizer.supervised_weight,
        )
        regularizer = gradient_surgery.component_gradients(
            regularization_loss,
            parameters,
            1.0,
        )
        projected_supervised, projected_regularizer, _ = gradient_surgery.apply_surgery(
            supervised,
            regularizer,
            self.gradient_surgery,
        )
        norm = lambda vectors: float(
            sum(float(v.float().square().sum()) for v in vectors)
        ) ** 0.5
        # Both sides must be the gradients the objective applies. Under
        # supervised_priority the supervised one is returned unchanged, so this
        # is only a difference under pcgrad -- where projecting the denominator
        # but not the numerator would miscalibrate by 1 / sqrt(1 - cos^2).
        return GradientContributionNorms(
            norm(projected_supervised),
            norm(projected_regularizer),
            (),
        )

    def _surgical_backward(self, phase, supervised_loss, regularization_loss):
        """Backward pass with the conflicting component projected out.

        Replaces ``loss.backward()``. The two terms are differentiated
        separately, the regularizer gradient is projected off the supervised one
        where they conflict, and the weighted sum is accumulated into
        ``parameter.grad`` exactly as ``backward`` would have.
        """

        parameters = self._surgery_parameters(phase)
        if not parameters:
            return {}
        supervised = gradient_surgery.component_gradients(
            supervised_loss,
            parameters,
            self.regularizer.supervised_weight,
        )
        regularizer = gradient_surgery.component_gradients(
            regularization_loss,
            parameters,
            self.regularizer.regularizer_weight,
        )
        supervised, regularizer, diagnostics = gradient_surgery.apply_surgery(
            supervised,
            regularizer,
            self.gradient_surgery,
        )
        gradient_surgery.accumulate(parameters, (supervised, 1.0), (regularizer, 1.0))
        norm = lambda vectors: float(
            sum(float(v.float().square().sum()) for v in vectors)
        ) ** 0.5
        supervised_norm = norm(supervised)
        surgical = {
            "train/gradient_surgery/cosine_before": diagnostics["cosine_before"],
            "train/gradient_surgery/fired": float(diagnostics["fired"]),
            "train/gradient_surgery/regularizer_length_kept": (
                diagnostics["regularizer_length_kept"]
            ),
            "train/gradient_surgery/supervised_length_kept": (
                diagnostics["supervised_length_kept"]
            ),
        }
        if supervised_norm > 0.0:
            # The contribution log measures the gradients before the projection,
            # so it overstates the ratio by 1 / regularizer_length_kept. This is
            # the ratio the objective actually applies.
            surgical["train/gradient_surgery/applied_ratio"] = (
                norm(regularizer) / supervised_norm
            )
        return surgical

    def _calibrate_regularizer_weight(self, phase, supervised_loss, regularization_loss):
        """Rescale ``regularizer_weight`` to hit the configured gradient ratio.

        ``||grad(w * L_reg)||`` is linear in ``w``, so measuring the components at
        ``w = 1`` gives the weight that reaches the target exactly:
        ``w = target * ||grad(w_sup * L_sup)|| / ||grad(L_reg)||``. Each measured
        batch refines the estimate with the configured finite-buffer statistic.
        The configured
        probes are distributed across a configurable sample-count window, after
        which a fresh calibration cycle starts. Scheduling by samples rather
        than batch indices keeps the measured part of training invariant to
        batch size.

        Returns whether the weight changed, i.e. whether the caller has to rebuild
        the combined loss before its backward pass.
        """

        if (
            not self._target_ratio_calibration_pending()
            or not phase.regularization_active
            or supervised_loss is None
            or regularization_loss is None
            or not regularization_loss.requires_grad
        ):
            return False
        if not self.regularizer.ready_for_target_ratio_calibration():
            # A regularizer with its own start-up phase can produce a gradient
            # that is real but unrepresentative -- measuring it would freeze a
            # ratio computed from terms that are not yet in the objective.
            logger.debug(
                "Skipping regularizer weight calibration: "
                f"{self.regularizer.name} reports it is still starting up"
            )
            return False

        if self._target_ratio_cycle_start_sample is None:
            # A method-specific startup phase may delay the first representative
            # gradient. Anchor its first complete window here instead of spending
            # old probe targets in a burst.
            self._start_target_ratio_probe_schedule(
                self.regularization_samples_seen,
                self.regularization_samples_seen,
            )

        # The first eligible batch starts the window and is measured immediately.
        # Later probes are placed at equal sample-count offsets in that window.
        # Batch boundaries can overshoot a target by at most one batch, but the
        # offsets themselves never change with batch size. An unusable probe does
        # not consume its slot, so a transiently quiet term is retried on the next
        # batch rather than leaving the initial probe weight in place.
        if (
            self._next_target_ratio_measurement_sample is not None
            and self.regularization_samples_seen
            < self._next_target_ratio_measurement_sample
        ):
            return False

        pending_extras = {}
        for name in self._extra_target_ratios:
            if name in self._extra_ratio_frozen:
                continue
            loss = self.regularizer.calibratable_component_loss(name)
            if loss is None or not loss.requires_grad:
                continue
            # Unit weight: the norm is linear in the weight, so the measurement
            # divides out and the configured value never biases the result.
            pending_extras[name] = (loss, 1.0)

        if self.gradient_surgery is not None:
            # The objective applies the *projected* regularizer gradient, so that
            # is what the ratio has to be measured against; calibrating on the
            # unprojected norm would undershoot the target by sqrt(1 - cos^2).
            component_norms = self._measure_projected_component_norms(
                supervised_loss,
                regularization_loss,
            )
        else:
            component_norms = measure_gradient_component_norms(
                supervised_loss,
                regularization_loss,
                self._shared_trunk_parameters(),
                supervised_weight=self.regularizer.supervised_weight,
                # Probe the unweighted regularizer gradient; extra objective terms
                # carry their own weights and are calibrated separately below.
                regularizer_weight=1.0,
                extra_components=pending_extras,
                statistic=self.target_ratio_statistic,
            )
        if component_norms.supervised <= 0.0:
            logger.debug(
                "Skipping weight calibration on a batch with no supervised gradient"
            )
            return False

        if component_norms.regularizer <= 0.0:
            # A ratio needs a non-zero denominator, and every consumer below --
            # the statistic anchor, the target-ratio measurement, and GradNorm --
            # divides by this norm. It reaches exactly zero when a weighted graph
            # underflows: the manifold-preserving Gaussian weights span enough
            # decades that a whole sampled edge batch can flush to zero in
            # float32. Wait for a batch that carries gradient rather than
            # dividing by it.
            logger.debug(
                "Skipping weight calibration on a batch with no regularizer gradient"
            )
            return False

        if self._statistic_anchor_scale is None:
            # One extra pair of gradient evaluations, once per run.
            euclidean = measure_gradient_component_norms(
                supervised_loss,
                regularization_loss,
                self._shared_trunk_parameters(),
                supervised_weight=self.regularizer.supervised_weight,
                regularizer_weight=1.0,
                include_pair_norm=False,
                statistic=TARGET_RATIO_STATISTIC_L2,
            )
            configured = component_norms.supervised / component_norms.regularizer
            if euclidean.regularizer > 0.0 and configured > 0.0:
                self._statistic_anchor_scale = (
                    euclidean.supervised / euclidean.regularizer
                ) / configured
            else:
                self._statistic_anchor_scale = 1.0
            logger.info(
                f"Anchored the {self.target_ratio_statistic!r} gradient statistic to "
                f"the Euclidean one with factor {self._statistic_anchor_scale:g}, so "
                f"regularizer_target_ratio={self.target_ratio if self.target_ratio is not None else float('nan'):g} "
                "keeps the meaning it has under statistic='l2'"
            )

        changed = False
        measured = False
        if not self._target_ratio_frozen:
            recorded = self._record_calibration_measurement(
                measurements=self._target_ratio_measurements,
                supervised_norm=component_norms.supervised,
                component_norm=component_norms.regularizer,
                target_ratio=self.target_ratio,
                apply_weight=self._set_regularizer_weight,
                on_frozen=self._freeze_calibrated_regularizer_weight,
                key=None,
            )
            changed |= recorded
            measured |= recorded
            if recorded and self.target_ratio_diagnostics_enabled:
                applied_weight = self.regularizer.regularizer_weight
                applied_ratio = (
                    applied_weight
                    * component_norms.regularizer
                    / component_norms.supervised
                )
                self._target_ratio_probe_diagnostics.update(
                    {
                        "train/target_ratio/target": self.target_ratio,
                        "train/target_ratio/applied_ratio": applied_ratio,
                        "train/target_ratio/target_relative_error": (
                            applied_ratio / self.target_ratio - 1.0
                        ),
                        "train/target_ratio/supervised_gradient_norm": (
                            component_norms.supervised
                        ),
                        "train/target_ratio/unit_regularizer_gradient_norm": (
                            component_norms.regularizer
                        ),
                        "train/target_ratio/regularizer_weight": applied_weight,
                        "train/target_ratio/cycle": float(
                            self._target_ratio_calibration_cycle
                        ),
                        # Under the sliding schedule the buffer saturates at
                        # probe_memory, so report the running probe count there.
                        "train/target_ratio/probe": float(
                            self._target_ratio_measurement_events + 1
                            if self._continuous_calibration
                            else len(self._target_ratio_measurements)
                        ),
                        "train/target_ratio/regularized_samples_seen": float(
                            self.regularization_samples_seen
                        ),
                    }
                )
            # A batch that produced no usable gradient for one side says nothing
            # about their relative scale; wait for one that does.
            self._count_unusable_calibration_batch(
                self.regularizer.name,
                recorded,
                component_norms.supervised,
                component_norms.regularizer,
            )

        for name, norm in component_norms.extra:
            recorded = self._record_calibration_measurement(
                measurements=self._extra_ratio_measurements[name],
                supervised_norm=component_norms.supervised,
                component_norm=float(norm),
                target_ratio=self._extra_target_ratios[name],
                apply_weight=partial(
                    self.regularizer.apply_calibrated_component_weight,
                    name,
                ),
                on_frozen=partial(self._freeze_calibrated_component_weight, name),
                key=name,
            )
            changed |= recorded
            measured |= recorded
            if recorded and self.target_ratio_diagnostics_enabled:
                applied_weight = self.regularizer.calibrated_component_weight(name)
                applied_ratio = (
                    applied_weight * float(norm) / component_norms.supervised
                )
                prefix = f"train/target_ratio/{name}"
                self._target_ratio_probe_diagnostics.update(
                    {
                        f"{prefix}/target": self._extra_target_ratios[name],
                        f"{prefix}/applied_ratio": applied_ratio,
                        f"{prefix}/target_relative_error": (
                            applied_ratio / self._extra_target_ratios[name] - 1.0
                        ),
                        f"{prefix}/unit_gradient_norm": float(norm),
                        f"{prefix}/weight": applied_weight,
                        f"{prefix}/probe": float(
                            len(self._extra_ratio_measurements[name])
                        ),
                    }
                )
            self._count_unusable_calibration_batch(
                name,
                recorded,
                component_norms.supervised,
                float(norm),
            )
        if measured:
            self._target_ratio_measurement_events += 1
            self._target_ratio_measurement_samples.append(
                self.regularization_samples_seen
            )
            if self._continuous_calibration:
                self._next_target_ratio_measurement_sample = (
                    self._target_ratio_batch_boundary_at_or_after(
                        self.regularization_samples_seen
                        + max(
                            self.target_ratio_update_interval_samples,
                            self._regularized_samples_per_step,
                        )
                    )
                )
                self._maybe_log_sliding_calibration()
            elif not self._target_ratio_calibration_pending():
                self._complete_target_ratio_calibration_cycle()
            elif self._target_ratio_measurement_events < len(
                self._target_ratio_probe_samples
            ):
                self._next_target_ratio_measurement_sample = (
                    self._target_ratio_probe_samples[
                        self._target_ratio_measurement_events
                    ]
                )
            else:
                # One component may have missed a scheduled probe while another
                # consumed it. Retry the unfinished component on the next step.
                self._next_target_ratio_measurement_sample = (
                    self.regularization_samples_seen
                    + self._regularized_samples_per_step
                )
        return changed

    def _maybe_start_periodic_target_ratio_calibration(self):
        """Open a fresh calibration window once its sample cadence is due."""

        if self._continuous_calibration:
            return False
        due_sample = self._next_target_ratio_recalibration_sample
        if (
            due_sample is None
            or self._target_ratio_calibration_pending()
            or self.regularization_samples_seen < due_sample
        ):
            return False

        self._target_ratio_calibration_cycle += 1
        self._target_ratio_measurements = []
        self._target_ratio_frozen = self.target_ratio is None
        self._extra_ratio_measurements = {
            name: [] for name in self._extra_target_ratios
        }
        self._extra_ratio_frozen = set()
        self._unusable_calibration_batches = {}
        # Keep the nominal sample boundary as the window origin. The first
        # batch can only observe that boundary after overshooting it, but
        # anchoring the next window to ``due_sample + interval`` prevents this
        # unavoidable rounding from accumulating once per cycle.
        self._start_target_ratio_probe_schedule(
            due_sample,
            self.regularization_samples_seen,
        )
        self._next_target_ratio_recalibration_sample = None
        logger.info(
            f"Starting target-ratio calibration cycle "
            f"{self._target_ratio_calibration_cycle} at "
            f"{self.regularization_samples_seen} regularized samples seen"
        )
        return True

    def _complete_target_ratio_calibration_cycle(self):
        """Schedule the next window after every configured ratio has settled."""

        self._target_ratio_calibrations_completed += 1
        window_start = (
            self.regularization_samples_seen
            if self._target_ratio_cycle_start_sample is None
            else self._target_ratio_cycle_start_sample
        )
        self._next_target_ratio_recalibration_sample = max(
            window_start + self.target_ratio_recalibration_interval_samples,
            self.regularization_samples_seen,
        )
        self.args.regularizer_target_ratio_calibrations_completed = (
            self._target_ratio_calibrations_completed
        )
        self.args.regularizer_target_ratio_next_recalibration_sample = (
            self._next_target_ratio_recalibration_sample
        )
        logger.info(
            f"Target-ratio calibration cycle "
            f"{self._target_ratio_calibration_cycle} complete; next window starts "
            f"at {self._next_target_ratio_recalibration_sample} regularized "
            "samples seen"
        )
        if getattr(self.args, "log_dir", None) is not None:
            write_run_config(self.args, self.ssl_config)

    def _count_unusable_calibration_batch(
        self,
        label,
        recorded,
        supervised_norm,
        component_norm,
    ):
        """Give up on a term whose gradient never rises above the noise floor.

        Without this the run continues with the configured probe weight, or
        worse, a weight inverted from a denormal ratio, and only fails once the
        model has already gone non-finite -- several epochs from the cause.
        """

        if recorded:
            self._unusable_calibration_batches[label] = 0
            return
        seen = self._unusable_calibration_batches.get(label, 0) + 1
        self._unusable_calibration_batches[label] = seen
        budget = max(MAXIMUM_UNUSABLE_CALIBRATION_BATCHES, self.target_ratio_batches)
        if seen < budget:
            return
        ratio = component_norm / supervised_norm if supervised_norm > 0 else 0.0
        raise FloatingPointError(
            f"{label} produced no usable gradient in {seen} consecutive calibration "
            f"batches (last component/supervised gradient norm ratio {ratio:g}, "
            f"floor {MINIMUM_CALIBRATION_NORM_RATIO:g}). Calibrating a weight from "
            "this would invert a denormal ratio into an enormous weight. The term "
            "is not reaching the model: check that its loss is not saturated, e.g. "
            "NTXentLoss below temperature ~0.05 is exactly zero on mined pairs."
        )

    def _target_ratio_calibration_pending(self):
        """Whether any weight still has measurements to collect."""

        if self._continuous_calibration:
            # A windowless schedule never finishes: there is no window to fill.
            return True
        if not self._target_ratio_frozen:
            return True
        return any(
            name not in self._extra_ratio_frozen for name in self._extra_target_ratios
        )

    def _set_regularizer_weight(self, weight):
        self.regularizer.regularizer_weight = weight

    def _continuous_estimator_span(self):
        """Finite buffer size, or the EMA reciprocal-alpha reporting horizon."""

        if self.target_ratio_probe_memory is not None:
            return int(self.target_ratio_probe_memory)
        return max(1, round(1.0 / self.target_ratio_ema_alpha))

    def _record_calibration_measurement(
        self,
        measurements,
        supervised_norm,
        component_norm,
        target_ratio,
        apply_weight,
        on_frozen,
        key=None,
    ):
        """Fold one scheduled probe into the running estimate and apply its weight.

        The estimate is the configured statistic over ``measurements`` -- the
        whole window, or the last ``probe_memory`` probes -- unless an EMA alpha
        is configured, in which case it is an exponential moving average and
        ``measurements`` only carries the last raw ratio. ``key`` names the
        component so each keeps its own average.

        Returns whether the measurement was usable; a component gradient far
        below the supervised one is rejected rather than inverted into a weight.
        """

        if component_norm < MINIMUM_CALIBRATION_NORM_RATIO * supervised_norm:
            return False
        ratio = supervised_norm / component_norm * self._statistic_anchor_scale
        measurements.append(ratio)
        if self._ema_calibration:
            previous = self._ratio_ema.get(key)
            alpha = self.target_ratio_ema_alpha
            estimate = (
                ratio
                if previous is None
                else alpha * ratio + (1.0 - alpha) * previous
            )
            self._ratio_ema[key] = estimate
        elif self.target_ratio_aggregation == "mean":
            estimate = sum(measurements) / len(measurements)
        else:
            ordered = sorted(measurements)
            estimate = ordered[len(ordered) // 2]
            if len(ordered) % 2 == 0:
                estimate = (estimate + ordered[len(ordered) // 2 - 1]) / 2.0
        apply_weight(target_ratio * estimate)
        if not self._continuous_calibration and len(measurements) >= self.target_ratio_batches:
            on_frozen()
        return True

    def _freeze_calibrated_component_weight(self, name):
        self._extra_ratio_frozen.add(name)
        weight = self.regularizer.calibrated_component_weight(name)
        logger.info(
            f"Calibrated {name} weight={weight:g} for target gradient ratio "
            f"{self._extra_target_ratios[name]:g} from "
            f"{len(self._extra_ratio_measurements[name])} gradient probes in cycle "
            f"{self._target_ratio_calibration_cycle}"
        )

    def _maybe_log_sliding_calibration(self):
        """Report a windowless weight once per finite or nominal horizon.

        The windowed schedule logs at each cycle completion. Sliding and EMA have
        no such boundary, so they report every ``probe_memory`` -- or reciprocal-
        alpha horizon -- probes instead, in the same format, and keep the same
        ``args`` fields fed for run_config.json.
        """

        if self._target_ratio_frozen or self.target_ratio is None:
            return
        memory = self._continuous_estimator_span()
        if self._target_ratio_measurement_events % memory:
            return
        weight = self.regularizer.regularizer_weight
        self._target_ratio_calibrations_completed += 1
        logger.info(
            f"Calibrated regularizer_weight={weight:g} for target gradient ratio "
            f"{self.target_ratio:g} from "
            f"{min(self._target_ratio_measurement_events, memory)} gradient probes "
            f"(supervised/regularizer gradient norm ratio "
            f"{'ema' if self._ema_calibration else self.target_ratio_aggregation}="
            f"{weight / self.target_ratio:g}, "
            f"{'moving average' if self._ema_calibration else 'sliding window'}"
            f", probes={self._target_ratio_measurement_events})"
        )
        self.args.regularizer_weight_calibrated = weight
        self.args.regularizer_target_ratio_resolved = self.target_ratio
        self.args.regularizer_target_ratio_calibrations_completed = (
            self._target_ratio_calibrations_completed
        )
        resolved = getattr(self.args, "ssl_config_resolved", None)
        if isinstance(resolved, dict) and isinstance(resolved.get("method_params"), dict):
            resolved["method_params"]["regularizer_weight"] = weight

    def _freeze_calibrated_regularizer_weight(self):
        """Finish this window and record its regularizer weight."""

        self._target_ratio_frozen = True
        weight = self.regularizer.regularizer_weight
        logger.info(
            f"Calibrated regularizer_weight={weight:g} for target gradient ratio "
            f"{self.target_ratio:g} from "
            f"{len(self._target_ratio_measurements)} gradient probes "
            f"(supervised/regularizer gradient norm ratio "
            f"{self.target_ratio_aggregation}="
            f"{weight / self.target_ratio:g}, cycle="
            f"{self._target_ratio_calibration_cycle})"
        )
        self.args.regularizer_weight_calibrated = weight
        self.args.regularizer_target_ratio_resolved = self.target_ratio
        resolved = getattr(self.args, "ssl_config_resolved", None)
        if isinstance(resolved, dict) and isinstance(resolved.get("method_params"), dict):
            # run_config.json is written before training starts, so the resolved
            # copy is what carries the calibrated weight into reporting.
            resolved["method_params"]["regularizer_weight"] = weight

    def _run_separate_regularizer_optimizer_steps(self, regularizer_batch, timer):
        """Consume ordered regularizer losses, stepping between fresh forwards."""

        losses = iter(
            self.regularizer.separate_optimizer_step_losses(
                student_model=self.model,
                state=self.regularizer_state,
                batch=regularizer_batch,
                device=self.args.device,
                timings=timer.totals if self.batch_timing_enabled else None,
            )
        )
        raw_detached_losses = []
        weighted_detached_losses = []
        step_diagnostics = {}
        component_names = set()
        while True:
            started = timer.start()
            with torch.autocast(
                device_type=self.device_type,
                dtype=torch.bfloat16,
                enabled=self.train_amp_enabled,
            ):
                try:
                    component_name, raw_loss = next(losses)
                except StopIteration:
                    timer.stop("regularization_loss", started)
                    break
                weighted_loss = self.regularizer.regularizer_weight * raw_loss
            timer.stop("regularization_loss", started)

            component_name = str(component_name)
            if component_name in component_names:
                raise RuntimeError(
                    f"{self.regularizer.name} yielded duplicate ordered loss "
                    f"{component_name!r}"
                )
            component_names.add(component_name)
            if not torch.is_tensor(raw_loss) or raw_loss.numel() != 1:
                raise TypeError(
                    f"{self.regularizer.name} ordered loss {component_name!r} "
                    "must be a scalar tensor"
                )

            started = timer.start()
            weighted_loss.backward()
            timer.stop("backward", started)
            if self.batch_diagnostics_enabled:
                step_diagnostics[
                    f"train/{self.regularizer.name}/gradient_norm/{component_name}_update"
                ] = utils.gradient_l2_norm(self.model.parameters())

            raw_detached_losses.append(raw_loss.detach())
            weighted_detached_losses.append(weighted_loss.detach())
            started = timer.start()
            self.model_optimizer.step()
            self.model_optimizer.zero_grad(set_to_none=True)
            timer.stop("optimizer_step", started)
            # Do not retain a completed branch's autograd graph while asking a
            # generator to build the next fresh forward.
            del raw_loss, weighted_loss

        if not raw_detached_losses:
            raise RuntimeError(
                f"{self.regularizer.name} enabled separate optimizer steps but "
                "yielded no losses"
            )
        return (
            torch.stack(raw_detached_losses).sum(),
            torch.stack(weighted_detached_losses).sum(),
            step_diagnostics,
        )

    def _should_log_gradient_contributions(self, phase):
        return (
            self.gradient_contribution_log_interval > 0
            and phase.regularization_active
            and self.regularization_step % self.gradient_contribution_log_interval == 0
        )

    def _active_optimizers(self, phase):
        active_optimizers = [self.model_optimizer]
        if phase.objective.is_classification:
            active_optimizers.append(phase.objective.optimizer)
        unique_optimizers = []
        optimizer_ids = set()
        for optimizer in active_optimizers:
            optimizer_id = id(optimizer)
            if optimizer_id in optimizer_ids:
                continue
            optimizer_ids.add(optimizer_id)
            unique_optimizers.append(optimizer)
        return unique_optimizers

    def _active_learning_rates(self, phase):
        learning_rates = utils.optimizer_learning_rates(self.model_optimizer, "model")
        if phase.objective.is_classification:
            learning_rates.update(
                utils.optimizer_learning_rates(phase.objective.optimizer, "criterion")
            )
        return learning_rates

    def _step_batch_lr_schedulers(self, phase, batch_fraction):
        if not self._lr_scheduler_uses_batch_steps:
            return
        for optimizer in self._active_optimizers(phase):
            optimizer_id = id(optimizer)
            scheduler = self._lr_schedulers.get(optimizer_id)
            if scheduler is not None:
                scheduler.step(
                    self._lr_scheduler_active_epochs[optimizer_id] + batch_fraction
                )

    def _finish_lr_scheduler_epoch(self, phase):
        for optimizer in self._active_optimizers(phase):
            optimizer_id = id(optimizer)
            scheduler = self._lr_schedulers.get(optimizer_id)
            if scheduler is None:
                continue
            if self._lr_scheduler_uses_batch_steps:
                self._lr_scheduler_active_epochs[optimizer_id] += 1
            else:
                scheduler.step()

    @staticmethod
    def _summarize_epoch_learning_rates(epoch_learning_rates):
        diagnostics = {}
        for name, values in epoch_learning_rates.items():
            mean_value = sum(values) / len(values)
            diagnostics[name] = mean_value
            summary_name = name.replace(
                "train/learning_rate/",
                "train/learning_rate_summary/",
                1,
            )
            diagnostics.update(
                {
                    f"{summary_name}/start": values[0],
                    f"{summary_name}/mean": mean_value,
                    f"{summary_name}/min": min(values),
                    f"{summary_name}/max": max(values),
                    f"{summary_name}/end": values[-1],
                }
            )
        return diagnostics

    def _iterate_epoch_batches(self, train_loader, epoch, refresh, batches_per_epoch):
        """Yield the epoch's batches, rebuilding the loader when a refresh is due.

        The epoch keeps the batch budget its first loader reports, so a
        sample-scoped rebuild changes only which graph the remaining steps train
        against, never how long the epoch is. Each rebuilt loader starts a fresh
        sampler pass, of which only the next interval's worth is consumed.
        """

        loader = train_loader
        produced = 0
        while produced < batches_per_epoch:
            produced_at_loader_start = produced
            for batch in loader:
                yield batch
                produced += 1
                refresh.count_step()
                if produced >= batches_per_epoch or refresh.due:
                    break
            if produced >= batches_per_epoch:
                return
            if produced == produced_at_loader_start:
                raise RuntimeError(
                    "sample-scoped SSL refresh produced a training loader with no batches"
                )
            loader = refresh.rebuild_loader(epoch)

    def train_epoch(self, train_loader, epoch, refresh=None):
        phase = self._resolve_phase(epoch)
        logger.info(f"Epoch {epoch}: training with {phase.description}")
        self.teacher_model = initialize_stml_teacher_for_phase(
            teacher_model=self.teacher_model,
            student_model=self.model,
            active_criterion=phase.objective.criterion,
            device=self.args.device,
        )
        if phase.regularization_active and self.regularizer_state is None:
            self.regularizer_state = self.regularizer.initialize_state(self.model, self.args.device)
        if phase.regularization_active:
            # This epoch's loader was built before train_epoch, so any refresh
            # reset it queued is collected here, before the first step.
            self._apply_pending_regularizer_optimizer_resets()

        self.model.train()
        if self.teacher_model is not None:
            self.teacher_model.eval()
        epoch_losses = []
        num_batches = 0
        epoch_miner_totals = defaultdict(float)
        epoch_learning_rates = defaultdict(list)
        if not self._lr_scheduler_uses_batch_steps:
            for name, value in self._active_learning_rates(phase).items():
                epoch_learning_rates[name].append(value)
        timer = _BatchTimer(self.args.device, self.batch_timing_enabled)
        last_timing_log_batch = 0
        last_timing_totals = {}
        next_batch_wait_start = timer.start()
        batches_per_epoch = len(train_loader)
        batch_source = train_loader
        if refresh is not None:
            refresh.current_loader = train_loader
            batch_source = self._iterate_epoch_batches(
                train_loader,
                epoch,
                refresh,
                batches_per_epoch,
            )
        self.progress_bar = tqdm(batch_source, total=batches_per_epoch)

        try:
            for batch_index, batch in enumerate(self.progress_bar):
                data_ready_time = timer.start()
                timer.stop("data_wait", next_batch_wait_start, data_ready_time)
                self._target_ratio_probe_diagnostics = {}
                separate_optimizer_steps = bool(
                    phase.regularization_active
                    and self.regularizer.uses_separate_optimizer_steps
                )

                if self._lr_scheduler_uses_batch_steps:
                    for name, value in self._active_learning_rates(phase).items():
                        epoch_learning_rates[name].append(value)
                loss, supervised_loss, regularization_loss, miner_outputs = self._compute_batch_loss(
                    batch,
                    phase,
                    timer,
                    data_ready_time,
                )
                if phase.regularization_active:
                    self._maybe_start_periodic_target_ratio_calibration()
                if self._target_ratio_calibration_pending():
                    started = timer.start()
                    if self._calibrate_regularizer_weight(
                        phase,
                        supervised_loss,
                        regularization_loss,
                    ):
                        # Recombine so this batch already trains with the weight it
                        # was measured for, instead of the probe weight.
                        loss = self.regularizer.combine_losses(
                            supervised_loss,
                            regularization_loss,
                        )
                    timer.stop("regularizer_weight_calibration", started)
                detached_loss = loss.detach()
                if id(phase.objective.criterion) in self._compiled_forward_loss:
                    # CUDA-graph outputs reuse static storage on the next replay.
                    # Keep the epoch accumulator independent without inserting a
                    # host synchronization; the clone stays queued on-device.
                    detached_loss = detached_loss.clone()
                epoch_losses.append(detached_loss)

                gradient_contribution_norms = None
                if self._should_log_gradient_contributions(phase):
                    started = timer.start()
                    gradient_contribution_norms = measure_gradient_component_norms(
                        supervised_loss,
                        regularization_loss,
                        self._shared_trunk_parameters(),
                        supervised_weight=self.regularizer.supervised_weight,
                        regularizer_weight=self.regularizer.regularizer_weight,
                        extra_components=self.regularizer.extra_loss_components(),
                    )
                    timer.stop("gradient_contribution_diagnostics", started)

                if self.grad_norm_weights is not None:
                    # After the diagnostic, so the logged ratio describes the
                    # weights this batch actually trained with, and before the
                    # backward, which frees the graph the measurement needs.
                    started = timer.start()
                    self._update_grad_norm_weights(
                        phase,
                        supervised_loss,
                        regularization_loss,
                    )
                    timer.stop("grad_norm_update", started)

                started = timer.start()
                surgery_diagnostics = {}
                if (
                    self.gradient_surgery is not None
                    and phase.regularization_active
                    and regularization_loss is not None
                    and regularization_loss.requires_grad
                ):
                    surgery_diagnostics = self._surgical_backward(
                        phase,
                        supervised_loss,
                        regularization_loss,
                    )
                else:
                    loss.backward()
                timer.stop("backward", started)

                if surgery_diagnostics:
                    self._target_ratio_probe_diagnostics.update(surgery_diagnostics)
                diagnostics_enabled = (
                    self.batch_diagnostics_enabled
                    or gradient_contribution_norms is not None
                    or bool(self._target_ratio_probe_diagnostics)
                )
                started = timer.start() if diagnostics_enabled else None
                miner_diagnostics = {}
                batch_diagnostics = None
                if self.batch_diagnostics_enabled:
                    miner_diagnostics = utils.summarize_miner_outputs(miner_outputs)
                    batch_diagnostics = self._make_batch_diagnostics(
                        phase,
                        detached_loss,
                        supervised_loss,
                        regularization_loss,
                        miner_diagnostics,
                    )
                gradient_contribution_diagnostics = None
                if gradient_contribution_norms is not None:
                    combined_gradient_norm = (
                        None
                        if batch_diagnostics is None
                        else batch_diagnostics.get("train/gradient_norm/model")
                    )
                    if combined_gradient_norm is None:
                        combined_gradient_norm = utils.gradient_l2_norm(
                            self.model.parameters()
                        )
                    gradient_contribution_diagnostics = (
                        make_gradient_contribution_diagnostics(
                            gradient_contribution_norms,
                            combined_gradient_norm,
                            regularizer_weight=self.regularizer.regularizer_weight,
                        )
                    )
                if gradient_contribution_diagnostics:
                    batch_diagnostics = {
                        **(batch_diagnostics or {}),
                        **gradient_contribution_diagnostics,
                    }
                if self._target_ratio_probe_diagnostics:
                    batch_diagnostics = {
                        **(batch_diagnostics or {}),
                        **self._target_ratio_probe_diagnostics,
                    }
                if batch_diagnostics is not None and self.grad_norm_weights is not None:
                    # Learned weights only mean something next to the gradient
                    # contributions they were computed to balance.
                    batch_diagnostics.update(self.grad_norm_weights.diagnostics())
                timer.stop("diagnostics", started)

                started = timer.start()
                self.model_optimizer.step()
                self.model_optimizer.zero_grad(set_to_none=True)
                if phase.objective.is_classification:
                    phase.objective.optimizer.step()
                    phase.objective.optimizer.zero_grad(set_to_none=True)
                timer.stop("optimizer_step", started)
                if separate_optimizer_steps:
                    (
                        regularization_loss,
                        weighted_regularization_loss,
                        ordered_step_diagnostics,
                    ) = self._run_separate_regularizer_optimizer_steps(
                        batch[1],
                        timer,
                    )
                    # These components were evaluated sequentially at three
                    # parameter points, so this is a diagnostic aggregate rather
                    # than one scalar objective whose gradient was taken jointly.
                    detached_loss = detached_loss + weighted_regularization_loss
                    epoch_losses[-1] = detached_loss
                    if batch_diagnostics is not None:
                        batch_diagnostics["train/zero_loss_batch"] = float(
                            detached_loss == 0.0
                        )
                        batch_diagnostics["train/regularization_loss"] = float(
                            regularization_loss
                        )
                        supervised_gradient_norm = batch_diagnostics.get(
                            "train/gradient_norm/model"
                        )
                        if supervised_gradient_norm is not None:
                            batch_diagnostics[
                                f"train/{self.regularizer.name}/gradient_norm/"
                                "supervised_update"
                            ] = supervised_gradient_norm
                        batch_diagnostics.update(
                            self.regularizer.batch_diagnostics()
                        )
                        batch_diagnostics.update(ordered_step_diagnostics)
                        batch_diagnostics[
                            f"train/{self.regularizer.name}/sequential_loss_aggregate"
                        ] = 1.0
                started = timer.start()
                self._step_batch_lr_schedulers(
                    phase,
                    (batch_index + 1) / max(1, batches_per_epoch),
                )
                timer.stop("optimizer_step", started)

                started = timer.start()
                if phase.standalone_stml:
                    update_ema_teacher(
                        self.teacher_model,
                        self.model,
                        momentum=phase.objective.criterion.teacher_momentum,
                        excluded_parameter_prefixes=("fc.",),
                        only_trainable_parameters=True,
                    )
                if phase.regularization_active:
                    self.regularizer.after_optimizer_step(self.model, self.regularizer_state)
                timer.stop("post_step_hooks", started)

                batch_loss_log_due = (
                    self.batch_loss_log_interval > 0
                    and (
                        self.global_step % self.batch_loss_log_interval == 0
                        or batch_index + 1 == batches_per_epoch
                    )
                )
                should_log_batch = batch_diagnostics is not None or batch_loss_log_due
                loss_value = None
                started = timer.start()
                if should_log_batch:
                    # Synchronize only after backward, optimizer updates, and
                    # post-step hooks have already been queued on the device.
                    loss_value = detached_loss.item()
                timer.stop("loss_item", started)

                started = timer.start()
                if should_log_batch:
                    self.metrics_logger.log_train_batch(
                        loss_value,
                        epoch,
                        self.global_step,
                        diagnostics=batch_diagnostics,
                    )
                timer.stop("metrics_logging", started)

                num_batches += 1
                for name, value in miner_diagnostics.items():
                    epoch_miner_totals[name] += value
                if phase.regularization_active:
                    self.regularization_step += 1
                    self.regularization_samples_seen += (
                        self._regularized_samples_per_step
                    )
                self.global_step += 1
                if loss_value is not None:
                    self.progress_bar.set_description(f"loss = {loss_value:.5f}", refresh=False)

                if self.batch_timing_enabled and (batch_index + 1) % self.batch_timing_interval == 0:
                    current_batch_count = batch_index + 1
                    interval_batches = current_batch_count - last_timing_log_batch
                    interval_timings = {
                        name: total - last_timing_totals.get(name, 0.0)
                        for name, total in timer.totals.items()
                    }
                    logger.info(
                        "Batch timing interval "
                        f"epoch={epoch} batch={current_batch_count}/{len(train_loader)} "
                        f"last_batches={interval_batches} phase={phase.description}: "
                        + ", ".join(
                            f"{name}={total / interval_batches:.4f}s"
                            for name, total in sorted(interval_timings.items())
                        )
                    )
                    last_timing_log_batch = current_batch_count
                    last_timing_totals = dict(timer.totals)
                next_batch_wait_start = timer.start()
        finally:
            self.close_progress_bar()

        if num_batches == 0:
            return None
        # Transfer the small detached loss scalars together, preserving the
        # previous Python-float epoch mean without synchronizing every batch.
        epoch_loss_values = torch.stack(epoch_losses).to(
            device="cpu",
            dtype=torch.float64,
        ).tolist()
        mean_loss = sum(epoch_loss_values) / num_batches
        zero_loss_batches = sum(loss_value == 0.0 for loss_value in epoch_loss_values)
        epoch_diagnostics = {
            "train/zero_loss_batches": zero_loss_batches,
            "train/zero_loss_fraction": zero_loss_batches / num_batches,
            **self._summarize_epoch_learning_rates(epoch_learning_rates),
        }
        for name, total in epoch_miner_totals.items():
            epoch_diagnostics[f"{name}_total"] = total
            epoch_diagnostics[f"{name}_mean_per_batch"] = total / num_batches
        if self.grad_norm_weights is not None and self.grad_norm_weights.updates:
            # The weights move all run long, so the epoch series is the record of
            # what the objective actually was.
            epoch_diagnostics.update(self.grad_norm_weights.diagnostics())
            learned = ", ".join(
                [
                    f"supervised={self.grad_norm_weights.supervised_weight:g}",
                    f"regularizer={self.grad_norm_weights.regularizer_weight:g}",
                    *(
                        f"{name}={weight:g}"
                        for name, weight in self.grad_norm_weights.extra_weights().items()
                    ),
                ]
            )
            logger.info(
                f"Epoch {epoch} GradNorm weights: {learned} "
                f"({self.grad_norm_weights.updates} updates)"
            )
        self.metrics_logger.log_train_epoch(
            mean_loss,
            epoch,
            self.global_step,
            diagnostics=epoch_diagnostics,
        )
        # Epoch schedulers advance here. Warm-restart schedulers already moved
        # after each batch; only their active-epoch counters need advancing.
        self._finish_lr_scheduler_epoch(phase)
        return mean_loss

    def close_progress_bar(self):
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None


def run_training(
    args,
    ssl_config,
    optuna_trial=None,
    optuna_metric=None,
    cv_fold=None,
    frozen_feature_cache=None,
):
    """Build the data/model, train one fold, and return its best metrics.

    Training can use labeled warmup, static or periodically rebuilt pseudo
    labels, STML neighbor batches, or supervised batches paired with a
    regularizer. Validation selects the checkpoint and can trigger early
    stopping; a final HPO fit instead trains for a fixed duration.
    """

    # Record the resolved dataclass, including defaults and HPO overrides, on
    # the namespace that will later be serialized into run_config.json.
    args.ssl_config_resolved = ssl_config.to_dict()
    # Keep this concrete in run metadata even when the CLI omitted it and SSL
    # simply follows the training device.
    args.ssl_device = get_ssl_device(args)
    # Keep the legacy configured value only for auditing. No training component
    # may consume it before the complete fold pool has been constructed below.
    if not hasattr(args, "configured_length_before_new_iter"):
        args.configured_length_before_new_iter = getattr(
            args,
            "length_before_new_iter",
            None,
        )
    args.length_before_new_iter = None
    final_full_train = bool(getattr(args, "final_full_train", False))
    if final_full_train and cv_fold is not None:
        raise ValueError("A final full-development fit must train one model, not an individual CV fold")

    # Validate before creating a model or downloading/loading a dataset so
    # configuration mistakes fail quickly and cheaply.
    validate_run_args(args, ssl_config)
    utils.seed_everything(args.seed, device=args.device)
    batch_diagnostics_enabled = bool(getattr(args, "log_batch_diagnostics", False))
    loss_driven_ssl = ssl_config.method in semi_supervised.LOSS_DRIVEN_METHODS
    # Pseudo-label methods may own run-local state (Algorithm 1 accumulates
    # promoted examples), so each fold gets an isolated registry instance.
    ssl_method = semi_supervised.create_method(ssl_config)
    regularized_ssl = ssl_method is not None and ssl_method.is_regularization_method
    regularizer = ssl_method.make_regularizer(ssl_config) if regularized_ssl else None
    slade_teacher_student = bool(
        regularizer is not None
        and getattr(regularizer, "within_fold_teacher_student", False)
    )
    # Wanted at the ``warmup_epochs`` boundary itself, because a teacher
    # checkpoint is not a candidate student whatever its score.
    restart_selection_at_boundary = slade_teacher_student
    # Wanted once the regularizer's steady-state terms actually reach the shared
    # trunk, which ``warmup_epochs`` only bounds from below. Latched in the loop.
    restart_selection_when_ready = bool(ssl_config.restart_selection_after_warmup)
    # Per-step artifacts read the same batch diagnostics the loss reports, and
    # the SERAPH trace additionally needs the labeled loader's own indices to
    # name the labeled endpoint of every scored pair.
    step_visualization_enabled = step_visualization.step_visualization_enabled(
        args,
        regularizer,
    )
    if regularizer is not None:
        regularizer.set_ssl_device(args.ssl_device)
        regularizer.set_batch_diagnostics_enabled(
            batch_diagnostics_enabled or step_visualization_enabled
        )
    stml_params = dict(getattr(args, "loss_params", {})) if loss_driven_ssl else {}
    supervised_mode = is_supervised_mode(args)
    precompute_frozen_features = should_precompute_frozen_features(args, ssl_config)
    ssl_frozen_feature_precompute = bool(
        precompute_frozen_features and ssl_config.enabled
    )
    requested_frozen_feature_train_views = get_frozen_feature_train_views(args)
    frozen_feature_train_views = (
        1 if ssl_frozen_feature_precompute else requested_frozen_feature_train_views
    )
    augmented_frozen_feature_precompute = (
        precompute_frozen_features
        and not ssl_frozen_feature_precompute
        and frozen_feature_train_views > 1
    )
    model_use_cache = bool(args.use_cache)
    pin_memory = torch.device(args.device).type == "cuda"

    args.torch_sharing_strategy_resolved = utils.configure_torch_sharing_strategy()
    utils.initialize_logger(args)
    logger.info(f"Torch multiprocessing sharing strategy: {args.torch_sharing_strategy_resolved}")
    if ssl_frozen_feature_precompute and requested_frozen_feature_train_views != 1:
        logger.warning(
            "SSL frozen feature precompute uses one deterministic view per sample; "
            f"ignoring frozen_feature_train_views={requested_frozen_feature_train_views}"
        )
    # DinoWrapper supplies a pretrained DINO backbone and, when feat_dim is
    # provided, a projection layer that becomes the learned embedding space.
    model = make_run_model(args, ssl_config, regularizer=regularizer).to(args.device)
    args.backbone_cache_dir = None if model.cache_dir is None else model.cache_dir
    args.model_uses_backbone_cache = model_use_cache
    args.frozen_feature_precompute = precompute_frozen_features
    args.frozen_feature_batch_size_resolved = get_frozen_feature_batch_size(args)
    args.frozen_feature_train_views_resolved = frozen_feature_train_views
    pseudo_label_diagnostics = semi_supervised.make_pseudo_label_diagnostics_tracker(
        args.log_dir,
        ssl_config,
        mode=getattr(args, "pseudo_label_diagnostics_mode", None),
    )

    dataset_bundle, ssl_split = _prepare_datasets(
        args,
        ssl_config,
        cv_fold,
        precompute_frozen_features=precompute_frozen_features,
        augmented_frozen_feature_precompute=augmented_frozen_feature_precompute,
        frozen_feature_train_views=frozen_feature_train_views,
    )
    sampler_epoch_length = override_length_before_new_iter_from_fold(
        args,
        dataset_bundle.train_dataset,
        ssl_split,
    )
    if regularizer is not None:
        # Epoch-denominated regularizer settings can only be resolved here: the
        # sampler epoch is this fold's training pool divided by batch_size.
        regularizer.set_steps_per_epoch(
            max(1, int(sampler_epoch_length) // int(args.batch_size))
        )
        # Some methods add trainable training-only heads whose class and bank
        # sizes are known only after the actual train split has been built.
        regularizer.configure_model(
            student_model=model,
            train_dataset=dataset_bundle.train_dataset,
            split=ssl_split,
            train_labels_mapper=dataset_bundle.train_labels_mapper,
            device=args.device,
        )
    teacher_head_state_path = getattr(args, "slade_teacher_head_state_path", None)
    if teacher_head_state_path is not None:
        if ssl_config.method != "slade":
            raise ValueError(
                "slade_teacher_head_state_path is only valid for method='slade'"
            )
        teacher_head_state_path = Path(teacher_head_state_path)
        if not teacher_head_state_path.is_file():
            raise FileNotFoundError(
                "The previous SLADE student's head state is unavailable: "
                f"{teacher_head_state_path}"
            )
        restore_slade_teacher_projection_head(
            model,
            load_head_state(teacher_head_state_path),
        )
        args.slade_teacher_head_state_path = teacher_head_state_path
    args.model_total_parameters = sum(parameter.numel() for parameter in model.parameters())
    args.model_trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    frozen_head_only_checkpoint = bool(
        getattr(args, "frozen_head_only_checkpoint", False)
    )
    logger.info(
        "Model parameters: "
        f"{args.model_trainable_parameters:,} trainable / {args.model_total_parameters:,} total. "
        f"Backbone tuning: {args.backbone_tuning}. Cache: {model_use_cache}. "
        f"Frozen feature precompute: {precompute_frozen_features}."
    )
    if frozen_head_only_checkpoint and not final_full_train:
        logger.info(
            "Best-model checkpoint: non-backbone state in CPU memory "
            "(frozen DINO backbone omitted)"
        )
    write_run_config(args, ssl_config)
    # Persist both subset-relative positions and source-dataset indices before
    # training so the exact experiment split is recoverable. HPO trials repeat
    # one split across every trial and fold, so they stay opt-in.
    if should_write_split_manifest(args):
        write_split_manifest(args.log_dir, dataset_bundle, ssl_config, ssl_split)
    precomputed_train_samples = None
    # Per-step contact sheets show the images a traced pair compared. Once the
    # batches hold feature rows, the only way back to those images is the image
    # dataset the rows were computed from, so keep it before it is replaced.
    step_visualization_image_dataset = None
    if ssl_frozen_feature_precompute:
        if step_visualization_enabled:
            step_visualization_image_dataset = dataset_bundle.train_dataset
        dataset_bundle.train_dataset = _precompute_backbone_features(
            args,
            model,
            dataset_bundle.train_dataset,
            f"precompute frozen {ssl_config.method} train features",
            pin_memory,
            require_feature_transform=True,
            use_feature_transform=True,
            frozen_feature_cache=frozen_feature_cache,
        )
        precomputed_train_samples = len(dataset_bundle.train_dataset)
    # Build loaders that are known before training.  For every-epoch SSL, the
    # loader is deliberately deferred until the current model can pseudo-label.
    static_train_loader = None
    warmup_train_loader = None
    # Kept so per-step artifacts can map a supervised batch index back to the
    # train-dataset position it was drawn from.
    supervised_train_dataset = None
    ssl_training_dataset_update_epoch = None
    if supervised_mode and not ssl_config.enabled:
        # The supervised comparison trains permanently on true-labeled samples
        # only.  Its loader never changes between epochs.
        train_dataset = semi_supervised.build_labeled_training_dataset(
            train_dataset=dataset_bundle.train_dataset,
            train_labels_mapper=dataset_bundle.train_labels_mapper,
            split=ssl_split,
        )
        if precompute_frozen_features:
            train_dataset = _precompute_backbone_features(
                args,
                model,
                train_dataset,
                "precompute frozen train features",
                pin_memory,
                require_feature_transform=not augmented_frozen_feature_precompute,
                use_feature_transform=not augmented_frozen_feature_precompute,
                num_views=frozen_feature_train_views,
                frozen_feature_cache=frozen_feature_cache,
            )
        static_train_loader = _make_train_loader(
            args,
            train_dataset,
            args.seed,
            pin_memory,
        )
    elif ssl_config.enabled and (ssl_config.warmup_epochs > 0 or regularized_ssl):
        # Regularization methods keep this labeled loader active after warm-up
        # and pair it with each unlabeled regularizer batch.
        supervised_source_dataset = (
            dataset_bundle.train_dataset
            if regularizer is None
            else regularizer.make_supervised_source_dataset(dataset_bundle.train_dataset)
        )
        warmup_train_dataset = semi_supervised.build_labeled_training_dataset(
            train_dataset=supervised_source_dataset,
            train_labels_mapper=dataset_bundle.train_labels_mapper,
            split=ssl_split,
            return_indices=bool(
                regularizer is not None
                and regularizer.regularizer_weight > 0
                and (regularizer.requires_labeled_indices or step_visualization_enabled)
            ),
        )
        supervised_train_dataset = warmup_train_dataset
        warmup_train_loader = _make_train_loader(
            args,
            warmup_train_dataset,
            args.seed,
            pin_memory,
        )
    if loss_driven_ssl:
        loss_driven_train_dataset = semi_supervised.build_loss_driven_training_dataset(
            dataset_bundle.train_dataset,
            ssl_split,
            num_views=int(stml_params.get("num_views", 2)),
        )
    elif regularized_ssl:
        regularizer.build_dataset(
            dataset_bundle.train_dataset,
            ssl_split,
            use_cache=model_use_cache,
        )
    elif static_train_loader is None and ssl_config.update_mode == "once" and ssl_config.warmup_epochs == 0:
        # With no warmup, "once" means pseudo-label immediately using the
        # off-the-shelf model and reuse those predictions for every epoch.
        train_dataset = semi_supervised.build_ssl_training_dataset(
            model=model,
            train_dataset=dataset_bundle.train_dataset,
            train_labels_mapper=dataset_bundle.train_labels_mapper,
            device=args.ssl_device,
            config=ssl_config,
            split=ssl_split,
            start_method=args.dataloader_start_method,
            diagnostics_tracker=pseudo_label_diagnostics,
            log_dir=args.log_dir,
            method=ssl_method,
            **_pseudo_label_capacity_filter_kwargs(args, ssl_config),
        )
        ssl_training_dataset_update_epoch = 0
        static_train_loader = _make_train_loader(
            args,
            train_dataset,
            args.seed,
            pin_memory,
            labeled_batch_size=_pseudo_label_labeled_batch_size(ssl_config),
            class_overlap=ssl_config.class_overlap,
        )
    # Evaluation loaders never shuffle because metric computation needs only a
    # deterministic pass over all embeddings and labels.
    valid_loader = None
    if not final_full_train:
        valid_loader = _make_eval_loader(
            args,
            model,
            dataset_bundle.valid_dataset,
            "precompute frozen valid features",
            pin_memory,
            precompute_features=precompute_frozen_features,
            frozen_feature_cache=frozen_feature_cache,
        )
    evaluate_test = bool(getattr(args, "evaluate_test", False))
    test_loader = None
    if evaluate_test:
        # Test evaluation is opt-in so HPO and intermediate runs cannot select
        # models based on held-out test performance.
        test_loader = _make_eval_loader(
            args,
            model,
            dataset_bundle.test_dataset,
            "precompute frozen test features",
            pin_memory,
            precompute_features=precompute_frozen_features,
            frozen_feature_cache=frozen_feature_cache,
        )
    (
        validation_retrieval_device,
        test_retrieval_device,
        validation_retrieval_backend,
    ) = resolve_run_evaluation_retrieval_devices(
        args,
        valid_loader,
        test_loader,
    )
    test_retrieval_backend = (
        None
        if test_retrieval_device is None
        else _retrieval_backend_name(test_retrieval_device)
    )
    evaluation_embedding_residency = get_evaluation_embedding_residency(args)
    args.validation_retrieval_backend_resolved = validation_retrieval_backend
    args.test_retrieval_backend_resolved = test_retrieval_backend
    # The first config snapshot is intentionally written before expensive data
    # setup. Refresh it now with the hardware policy resolved from validation.
    write_run_config(args, ssl_config)
    logger.info(
        "Evaluation retrieval backends: "
        f"validation={validation_retrieval_backend}, "
        f"test={test_retrieval_backend or 'disabled'}, "
        f"embedding_residency={evaluation_embedding_residency}"
    )
    # Source datasets can use sparse/non-zero-based class IDs.  Training losses
    # receive this dense mapping, while datasets continue returning original IDs.
    train_labels_mapper = dataset_bundle.train_labels_mapper
    train_label_lookup = make_label_lookup_tensor(train_labels_mapper, args.device)

    # Frozen backbone parameters are deliberately omitted from optimizer state.
    trainable_model_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optim = make_optimizer(args, trainable_model_parameters, lr=args.lr)
    num_train_classes = len(train_labels_mapper)
    main_objective = make_training_loss_components(
        args=args,
        loss_name=args.loss,
        loss_params=getattr(args, "loss_params", {}),
        miner_name=args.miner,
        miner_params=getattr(args, "miner_params", {}),
        num_classes=num_train_classes,
        embedding_size=model.feat_dim,
    )
    warmup_objective = None
    if uses_ssl_warmup_objective(ssl_config):
        warmup_objective = make_warmup_loss_components(
            args=args,
            ssl_config=ssl_config,
            criterion=main_objective.criterion,
            is_classification=main_objective.is_classification,
            miner=main_objective.miner,
            classifier_optim=main_objective.optimizer,
            num_classes=num_train_classes,
            embedding_size=model.feat_dim,
        )
    # MetricsLogger mirrors values to TensorBoard and a CSV in the run folder.
    metrics_logger = utils.MetricsLogger(args.log_dir, args)
    step_visualizer = step_visualization.make_step_visualizer(
        args,
        regularizer,
        train_dataset=dataset_bundle.train_dataset,
        split=ssl_split,
        model=model,
        device=args.device,
        supervised_dataset=supervised_train_dataset,
        image_dataset=step_visualization_image_dataset,
    )
    epoch_trainer = _EpochTrainer(
        args=args,
        ssl_config=ssl_config,
        model=model,
        model_optimizer=optim,
        main_objective=main_objective,
        warmup_objective=warmup_objective,
        regularizer=regularizer,
        metrics_logger=metrics_logger,
        train_labels_mapper=train_labels_mapper,
        train_label_lookup=train_label_lookup,
        model_use_cache=model_use_cache,
        step_visualizer=step_visualizer,
    )
    # The checkpoint is temporary: it is used to restore the selected epoch and
    # removed after final evaluation because the run currently returns metrics.
    best_model_path = args.log_dir / "best_model.pth"
    best_model_checkpoint = _BestModelCheckpoint(
        model,
        best_model_path,
        non_backbone_in_memory=frozen_head_only_checkpoint,
    )
    # Measurement columns follow the CLI list. P@1 and MAP@R are still computed
    # internally because model selection and legacy result fields require them.
    requested_measurements = utils.normalize_measurements(
        getattr(args, "measurements", None)
    )
    recall_at_k = utils.normalize_recall_at_k(getattr(args, "recall_at_k", ()) or ())
    # D_test is measured once per run, so it can report a wider table than the
    # per-epoch validation without costing anything per epoch.
    test_measurements_requested = utils.resolve_test_measurements(args)
    test_recall_at_k_values = utils.resolve_test_recall_at_k(args)
    final_train_loss = None
    test_precision = None
    test_map = None
    test_recall = {}
    epoch0_test_recall = {}
    best_recall_at_k = {}
    test_measurement_values = {}
    epoch0_test_measurement_values = {}
    best_measurement_values = {}
    test_pacmap_coordinates = None
    test_pacmap_plot = None
    test_tsne_coordinates = None
    test_tsne_plot = None
    test_embeddings_path = None
    test_embedding_set = None
    head_state_path = None
    epoch0_test_precision = None
    epoch0_test_map = None
    best_precision = None
    best_map = None
    selected_epoch = -1
    # A restore that rolls weights back needs the optimizer state that belongs
    # to them, so warm-up checkpoints carry one. Only while the warm-up is
    # running, and only for the mode that can actually roll back.
    track_warmup_optimization_state = (
        ssl_config.enabled
        and ssl_config.warmup_epochs > 0
        and ssl_config.warmup_checkpoint_mode == "restore_best"
        and not final_full_train
    )
    warmup_optimization_snapshot = None
    # Filled at the warm-up boundary, so they stay None for runs that have no
    # warm-up phase and for the final full-development fit.
    warmup_best_precision = None
    warmup_best_map = None
    warmup_selected_epoch = None
    warmup_restored_epoch = None
    train_loader = None
    loss_driven_train_loader = None
    loss_driven_sampling_rebuild_epoch = None

    try:
        last_epoch = -1
        if not final_full_train:
            # Epoch -1 measures the pretrained/off-the-shelf embedding before
            # task-specific updates. It is also a valid initial checkpoint.
            valid_outcome = utils.evaluate_split(
                model,
                valid_loader,
                "valid",
                device=args.device,
                return_per_class=False,
                return_diagnostics=True,
                retrieval_device=validation_retrieval_device,
                allow_cpu_fallback=False,
                embedding_residency=evaluation_embedding_residency,
                recall_at_k=recall_at_k,
                measurements=requested_measurements,
            )
            valid_precision = valid_outcome.precision_at_1
            valid_map = valid_outcome.mean_average_precision_at_r
            valid_per_class = None
            metrics_logger.log_eval(
                "valid",
                valid_precision,
                valid_map,
                step=0,
                epoch=-1,
                per_class_metrics=valid_per_class,
                diagnostics=valid_outcome.diagnostics,
                recall_at_k=valid_outcome.recall_at_k,
                measurements=valid_outcome.measurements,
            )

            # Track both metrics for reporting, but only selection_metric decides
            # which model state is checkpointed and when patience is reset.
            patience = args.patience
            best_precision = valid_precision
            best_map = valid_map
            best_recall_at_k = dict(valid_outcome.recall_at_k)
            best_measurement_values = dict(valid_outcome.measurements)
            best_selection_value = get_selection_metric_value(args.selection_metric, valid_precision, valid_map)
            epochs_no_improve = 0
            best_model_checkpoint.save()
            if track_warmup_optimization_state:
                warmup_optimization_snapshot = epoch_trainer.capture_optimization_state()
            logger.info(
                f"Model selection metric: {args.selection_metric}. "
                f"Initial selected value: {best_selection_value:.6f}"
            )
        else:
            logger.info(f"Training final full-development model for exactly {args.epochs} epochs")

        if evaluate_test and final_full_train:
            epoch0_outcome = utils.evaluate_split(
                model,
                test_loader,
                "test epoch 0 - no optimization",
                device=args.device,
                return_per_class=False,
                retrieval_device=test_retrieval_device,
                allow_cpu_fallback=False,
                embedding_residency=evaluation_embedding_residency,
                recall_at_k=test_recall_at_k_values,
                measurements=test_measurements_requested,
            )
            epoch0_test_precision = epoch0_outcome.precision_at_1
            epoch0_test_map = epoch0_outcome.mean_average_precision_at_r
            epoch0_test_recall = dict(epoch0_outcome.recall_at_k)
            epoch0_test_measurement_values = dict(epoch0_outcome.measurements)
            epoch0_test_per_class = None
            metrics_logger.log_eval(
                "epoch0_test",
                epoch0_test_precision,
                epoch0_test_map,
                step=0,
                epoch=0,
                per_class_metrics=epoch0_test_per_class,
                recall_at_k=epoch0_outcome.recall_at_k,
                measurements=epoch0_outcome.measurements,
            )

        def build_regularized_train_loader(num_epoch, seed, rebuild_graph):
            """Pair the labeled stream with the regularizer's unlabeled stream."""

            if rebuild_graph:
                regularizer.request_refresh()
            # The regularizer's persistent loss tensors remain on the training
            # device, while refresh-time embeddings and graph construction use
            # the independently selected SSL device.
            with semi_supervised.ssl_compute_device(args.ssl_device):
                return regularizer.make_loader(
                    model=model,
                    train_dataset=dataset_bundle.train_dataset,
                    supervised_loader=warmup_train_loader,
                    device=args.device,
                    config=ssl_config,
                    batch_size=args.batch_size,
                    seed=seed,
                    num_workers=args.num_workers,
                    start_method=args.dataloader_start_method,
                    epoch=num_epoch,
                    log_dir=args.log_dir,
                )

        def build_pseudo_label_train_loader(num_epoch, seed):
            """Re-embed, pseudo-label, and build the loader those labels imply."""

            nonlocal train_dataset, static_train_loader
            nonlocal ssl_training_dataset_update_epoch
            if static_train_loader is not None:
                utils.shutdown_dataloaders(static_train_loader)
            train_dataset = semi_supervised.build_ssl_training_dataset(
                model=model,
                train_dataset=dataset_bundle.train_dataset,
                train_labels_mapper=train_labels_mapper,
                device=args.ssl_device,
                config=ssl_config,
                split=ssl_split,
                epoch=num_epoch,
                start_method=args.dataloader_start_method,
                diagnostics_tracker=pseudo_label_diagnostics,
                log_dir=args.log_dir,
                method=ssl_method,
                **_pseudo_label_capacity_filter_kwargs(args, ssl_config),
            )
            loader = _make_train_loader(
                args,
                train_dataset,
                seed,
                pin_memory,
                persistent_workers=ssl_config.update_mode != "every_epoch",
                labeled_batch_size=_pseudo_label_labeled_batch_size(ssl_config),
                class_overlap=ssl_config.class_overlap,
            )
            ssl_training_dataset_update_epoch = num_epoch
            if ssl_config.update_mode == "every_epoch":
                static_train_loader = None
            else:
                static_train_loader = loader
            return loader

        def rebuild_train_loader_mid_epoch(num_epoch, rebuild_ordinal, previous_loader):
            """Refresh the graph between two steps of the same epoch."""

            # Mid-epoch rebuilds need seeds of their own so the resampled stream
            # does not replay the batches this epoch already trained on; placing
            # them past args.epochs leaves every per-epoch seed untouched.
            seed = args.seed + args.epochs + rebuild_ordinal
            if regularized_ssl:
                # A no-op while the regularizer's own loader keeps persistent
                # workers, which is what make_loader configures.
                shutdown_epoch_train_loader(
                    previous_loader,
                    warmup_train_loader,
                    static_train_loader,
                    loss_driven_train_loader,
                )
                rebuilt_loader = build_regularized_train_loader(
                    num_epoch,
                    seed=seed,
                    rebuild_graph=True,
                )
                # A mid-epoch rebuild can also open a self-training iteration,
                # and the next step happens immediately, so any parameter the
                # refresh re-initialized loses its stale moments here rather
                # than at the next epoch boundary.
                epoch_trainer._apply_pending_regularizer_optimizer_resets()
                return rebuilt_loader
            # The pseudo-label builder shuts down the loader it replaces itself.
            return build_pseudo_label_train_loader(num_epoch, seed=seed)

        sample_refresh_interval_steps = resolve_sample_scoped_refresh_interval_steps(
            args,
            ssl_config,
        )
        sample_refresh = (
            None
            if sample_refresh_interval_steps is None
            else _SampleScopedRefresh(
                sample_refresh_interval_steps,
                rebuild_train_loader_mid_epoch,
            )
        )

        # Latch for the readiness-triggered restart: ``steady_state_active`` is
        # allowed to go back to False, and a baseline that followed it would
        # restart repeatedly instead of once.
        selection_restart_latched = False

        def restart_model_selection(reason):
            """Discard the validation history so far and rebaseline on what follows."""

            nonlocal best_precision, best_map, best_recall_at_k
            nonlocal best_measurement_values
            nonlocal best_selection_value, selected_epoch, epochs_no_improve

            if best_selection_value > float("-inf"):
                logger.info(
                    f"Discarded peak ({args.selection_metric}): "
                    f"{best_selection_value:.6f} at epoch {selected_epoch}. "
                    "It is not a selection candidate and is excluded from "
                    "best_valid_*; metrics.csv keeps the full curve."
                )
            # Clearing the high-water mark is also what hands the phase that
            # follows its own patience budget: its first epoch becomes the new
            # best, so ``epochs_no_improve`` only starts counting once that phase
            # itself stops improving.
            best_precision = float("-inf")
            best_map = float("-inf")
            best_recall_at_k = {}
            best_measurement_values = {}
            best_selection_value = float("-inf")
            selected_epoch = -1
            epochs_no_improve = 0
            logger.info(reason)

        for num_epoch in range(args.epochs):
            last_epoch = num_epoch

            if (
                ssl_config.enabled
                and ssl_config.warmup_epochs > 0
                and num_epoch == ssl_config.warmup_epochs
                and not final_full_train
            ):
                # Record what the warm-up reached on its own before any
                # selection restart discards it. This is the run's supervised
                # baseline under identical hyperparameters, and otherwise it
                # survives only in metrics.csv.
                warmup_best_precision = best_precision
                warmup_best_map = best_map
                warmup_selected_epoch = selected_epoch
                logger.info(
                    f"Warm-up finished after {ssl_config.warmup_epochs} epochs. "
                    f"Best warm-up {args.selection_metric}="
                    f"{best_selection_value:.6f} at epoch {selected_epoch} "
                    f"(P@1={best_precision:.6f}, MAP@R={best_map:.6f})"
                )
                if ssl_config.warmup_checkpoint_mode == "restore_best":
                    if selected_epoch == num_epoch - 1:
                        # The warm-up's best epoch is the one it ended on, so
                        # there is nothing to roll back and nothing stale about
                        # the optimizer state. Doing the restore and the reset
                        # anyway would change the trajectory of a run this mode
                        # is supposed to leave alone -- and would make every such
                        # run irreproducible against one recorded under
                        # 'keep_last' for no reason at all.
                        logger.info(
                            "warmup_checkpoint_mode='restore_best': the warm-up's "
                            f"best epoch is also its last ({selected_epoch}), so "
                            "the SSL phase starts where the warm-up ended and "
                            "this run is identical to 'keep_last'."
                        )
                    else:
                        # Before this epoch's loader is built, so a graph method
                        # also builds its graph from the restored embedding
                        # rather than from the last warm-up epoch's.
                        best_model_checkpoint.restore(consume=False)
                        warmup_restored_epoch = selected_epoch
                        reason = (
                            "the SSL phase starts from the best warm-up "
                            f"checkpoint (epoch {selected_epoch})"
                        )
                        if warmup_optimization_snapshot is not None:
                            epoch_trainer.restore_optimization_state(
                                warmup_optimization_snapshot,
                                reason,
                            )
                        else:
                            # No snapshot should be missing here, but a cold
                            # optimizer is still better than moments belonging
                            # to weights that were just discarded.
                            epoch_trainer.reset_optimization_state(reason)
                        logger.info(
                            "warmup_checkpoint_mode='restore_best': the SSL phase "
                            f"starts from epoch {selected_epoch}'s weights instead "
                            f"of epoch {num_epoch - 1}'s."
                        )
                    if selected_epoch < 0:
                        logger.warning(
                            "The restored warm-up checkpoint is the epoch -1 "
                            "baseline: no warm-up epoch beat the untrained head "
                            f"on {args.selection_metric}, so the entire warm-up "
                            "has been discarded."
                        )
                else:
                    logger.info(
                        "warmup_checkpoint_mode='keep_last': the SSL phase "
                        f"continues from epoch {num_epoch - 1}'s weights, not "
                        f"from the warm-up's best epoch ({selected_epoch})."
                    )
                # Past the boundary nothing reads it again, and it is the size
                # of the optimizer's own state.
                warmup_optimization_snapshot = None
            # Choose the data source for this epoch. During warmup only true
            # labels are used; afterward pseudo-label or regularization methods
            # own the loader lifecycle.
            if ssl_config.enabled and num_epoch < ssl_config.warmup_epochs:
                # Warmup loader contains only ground-truth labeled examples.
                train_loader = warmup_train_loader
            elif loss_driven_ssl:
                if semi_supervised.should_rebuild_on_epoch(
                    ssl_config.update_mode,
                    ssl_config.update_interval_epochs,
                    num_epoch,
                    loss_driven_sampling_rebuild_epoch,
                ):
                    if loss_driven_train_loader is not None:
                        utils.shutdown_dataloaders(loss_driven_train_loader)
                    sampling_embeddings = semi_supervised.extract_embeddings(
                        model=model,
                        dataset=dataset_bundle.train_dataset,
                        positions=loss_driven_train_dataset.positions,
                        device=args.ssl_device,
                        batch_size=ssl_config.embedding_batch_size,
                        num_workers=ssl_config.embedding_num_workers,
                        seed=args.seed + num_epoch,
                        start_method=args.dataloader_start_method,
                        desc=f"STML sampling embeddings - epoch {num_epoch}",
                        embedding_kind="stml_g",
                    )
                    loss_driven_train_loader = utils.make_stml_train_loader(
                        train_dataset=loss_driven_train_dataset,
                        sampling_embeddings=sampling_embeddings,
                        batch_size=args.batch_size,
                        neighbors_per_query=main_objective.criterion.num_neighbors,
                        seed=args.seed + num_epoch,
                        num_workers=args.num_workers,
                        start_method=args.dataloader_start_method,
                        pin_memory=pin_memory,
                        graph_device=args.ssl_device,
                    )
                    loss_driven_sampling_rebuild_epoch = num_epoch
                train_loader = loss_driven_train_loader
            elif regularized_ssl:
                if regularizer.regularizer_weight == 0:
                    # This is the exact supervised-baseline sanity path: do not
                    # even iterate the unlabeled stream, since its augmentations
                    # could otherwise perturb process RNG state at num_workers=0.
                    train_loader = warmup_train_loader
                else:
                    rebuild_graph = sample_refresh is not None and sample_refresh.due
                    train_loader = build_regularized_train_loader(
                        num_epoch,
                        seed=args.seed + num_epoch,
                        rebuild_graph=rebuild_graph,
                    )
                    if rebuild_graph:
                        sample_refresh.note_rebuild(train_loader)
            elif should_rebuild_pseudo_label_training_dataset(
                ssl_config,
                num_epoch,
                ssl_training_dataset_update_epoch,
                refresh_due=sample_refresh is not None and sample_refresh.due,
            ):
                # Re-embed and pseudo-label with the latest model.  A new loader
                # is required because accepted pseudo-labels may have changed.
                train_loader = build_pseudo_label_train_loader(
                    num_epoch,
                    seed=args.seed + num_epoch,
                )
                if sample_refresh is not None:
                    sample_refresh.note_rebuild(train_loader)
            else:
                if static_train_loader is None:
                    raise RuntimeError("SSL training loader was not built before reuse")
                train_loader = static_train_loader

            if slade_teacher_student and num_epoch == ssl_config.warmup_epochs:
                # ``make_loader`` immediately above has just embedded the pool
                # with the fully warmed-up teacher and frozen its cluster IDs.
                # The live module now assumes the student role from identical
                # weights, while its optimization history starts from scratch.
                if warmup_restored_epoch is not None:
                    logger.info(
                        "SLADE's student stage resets optimization by design, so "
                        f"the optimizer state restored with epoch {warmup_restored_epoch} "
                        "is discarded again here; only its weights carry over."
                    )
                epoch_trainer.begin_slade_student_stage()

            if (
                restart_selection_at_boundary
                and num_epoch == ssl_config.warmup_epochs
                and not final_full_train
            ):
                # A teacher checkpoint is useful as a logged baseline but must
                # never win selection for a run whose output promises a trained
                # student (or be handed to the next fold). This one cannot wait
                # for steady state: it is about what the checkpoint *is*, not
                # about how far the objective has started up.
                restart_model_selection(
                    "SLADE student model selection starts now; teacher/warm-up "
                    "validation checkpoints are excluded"
                )

            # A sample-scoped schedule only applies to an epoch that actually
            # trains against the graph: warm-up epochs and the zero-weight
            # supervised-baseline path both fall back to the labeled loader.
            epoch_refresh = (
                sample_refresh
                if sample_refresh is not None and train_loader is not warmup_train_loader
                else None
            )
            epoch_train_loss = epoch_trainer.train_epoch(
                train_loader,
                num_epoch,
                refresh=epoch_refresh,
            )
            if epoch_train_loss is not None:
                final_train_loss = epoch_train_loss
            shutdown_epoch_train_loader(
                train_loader if epoch_refresh is None else epoch_refresh.current_loader,
                warmup_train_loader,
                static_train_loader,
                loss_driven_train_loader,
            )
            if final_full_train:
                # No validation or early stopping is permitted in the final
                # fit; the HPO-selected duration determines the resulting model.
                selected_epoch = num_epoch
                continue

            # Validation runs after every epoch and supplies both early-stopping
            # decisions and intermediate values for Optuna pruning.
            cur_outcome = utils.evaluate_split(
                model,
                valid_loader,
                f"valid - epoch {num_epoch:>2}",
                device=args.device,
                return_per_class=False,
                return_diagnostics=True,
                retrieval_device=validation_retrieval_device,
                allow_cpu_fallback=False,
                embedding_residency=evaluation_embedding_residency,
                recall_at_k=recall_at_k,
                measurements=requested_measurements,
            )
            cur_precision = cur_outcome.precision_at_1
            cur_map = cur_outcome.mean_average_precision_at_r
            cur_per_class = None
            metrics_logger.log_eval(
                "valid",
                cur_precision,
                cur_map,
                step=epoch_trainer.global_step,
                epoch=num_epoch,
                per_class_metrics=cur_per_class,
                diagnostics=cur_outcome.diagnostics,
                recall_at_k=cur_outcome.recall_at_k,
                measurements=cur_outcome.measurements,
            )
            # Warmup epochs do not consume early-stopping patience because the
            # SSL method has not started using pseudo-labels yet.
            is_after_warmup = num_epoch >= ssl_config.warmup_epochs

            # Asked after the epoch has trained, so the regularizer's own step
            # counters and gates describe work this model has actually had done
            # to it. This epoch is the first whose training the steady-state
            # terms reached, so it is the epoch the new baseline is taken from.
            if restart_selection_when_ready and not selection_restart_latched:
                if selection_restart_is_due(ssl_config, regularizer, num_epoch):
                    selection_restart_latched = True
                    restart_model_selection(
                        f"SSL-phase model selection starts at epoch {num_epoch}: the "
                        "regularizer's steady-state terms now reach the shared "
                        "parameters. Earlier validation checkpoints are excluded."
                    )
                elif num_epoch == ssl_config.warmup_epochs + patience:
                    # Early stopping is suspended until the restart fires, so a
                    # regularizer that never reaches steady state trains out the
                    # whole epoch budget. Say so once, while there is still time
                    # to kill the run, rather than only in the summary below.
                    logger.warning(
                        f"{patience} epochs past warmup_epochs and "
                        f"{regularizer.name if regularizer is not None else ssl_config.method} "
                        "still reports no steady state, so restart_selection_after_warmup "
                        "has not rebaselined and early stopping stays suspended. This run "
                        f"will train to epoch {args.epochs - 1} unless the regularizer "
                        "starts up."
                    )

            # Report the running best rather than only the current epoch when
            # the HPO objective is a "best_valid_*" metric. Epochs whose scores a
            # pending restart is about to discard are withheld from pruning: an
            # explicit SLADE teacher is a training input rather than a candidate
            # student, and a start-up stage has not produced a score the SSL
            # phase will be judged on, so neither should steer Optuna.
            selection_restart_pending = (
                restart_selection_at_boundary and not is_after_warmup
            ) or (restart_selection_when_ready and not selection_restart_latched)
            if not selection_restart_pending:
                best_precision_for_report = max(best_precision, cur_precision)
                best_map_for_report = max(best_map, cur_map)
                maybe_report_to_optuna(
                    optuna_trial=optuna_trial,
                    metric=optuna_metric,
                    epoch=num_epoch,
                    train_loss=final_train_loss,
                    best_precision=best_precision_for_report,
                    best_map=best_map_for_report,
                )

            cur_selection_value = get_selection_metric_value(args.selection_metric, cur_precision, cur_map)
            # Keep independent maxima for the final report even when the chosen
            # checkpoint is selected by only one of these metrics.
            if cur_map > best_map:
                best_map = cur_map
            if cur_precision > best_precision:
                best_precision = cur_precision
            # Each Recall@K keeps its own maximum, matching how the two metrics
            # above are reported independently of the selected checkpoint.
            for k, value in cur_outcome.recall_at_k.items():
                if value > best_recall_at_k.get(k, float("-inf")):
                    best_recall_at_k[k] = value
            for measurement, value in cur_outcome.measurements.items():
                if value > best_measurement_values.get(measurement, float("-inf")):
                    best_measurement_values[measurement] = value

            if cur_selection_value > best_selection_value:
                # Strict improvement replaces the selected checkpoint and
                # restarts patience.
                best_selection_value = cur_selection_value
                selected_epoch = num_epoch
                epochs_no_improve = 0
                best_model_checkpoint.save()
                if track_warmup_optimization_state and num_epoch < ssl_config.warmup_epochs:
                    warmup_optimization_snapshot = (
                        epoch_trainer.capture_optimization_state()
                    )
            elif is_after_warmup and not selection_restart_pending:
                # Equal or worse selected metric consumes one patience unit.
                # A pending restart is exempt for the same reason warm-up epochs
                # are: the phase whose progress patience measures has not started
                # yet, and burning the budget against the stale baseline would
                # end the run inside the start-up stage -- leaving selection
                # spanning the warm-up after all, which is what the restart was
                # asked to prevent.
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

        if (
            restart_selection_when_ready
            and not selection_restart_latched
            and not final_full_train
        ):
            # best_valid_* still spans the warm-up, so this run is not comparable
            # with the ones that did rebaseline. Loud, because the number reads
            # like every other trial's.
            logger.warning(
                "restart_selection_after_warmup never fired: "
                f"{regularizer.name if regularizer is not None else ssl_config.method} "
                "reported no steady state before the run ended, so best_valid_* "
                "still covers the warm-up and this run's objective is not "
                "comparable with trials whose SSL phase was rebaselined."
            )

        if not final_full_train:
            # Evaluate/report only the checkpoint selected on validation, never
            # the last epoch's potentially overfit model.
            best_model_checkpoint.restore()
        if getattr(args, "save_head_state", False):
            # Written before test evaluation so an evaluation that fails still
            # leaves behind the model it was supposed to measure.
            head_state_path = save_head_state(
                model,
                args,
                ssl_config,
                selected_epoch=selected_epoch,
                cv_fold=cv_fold,
                output_dir=args.log_dir,
            )
        if evaluate_test:
            keep_test_embeddings_on_device = (
                test_retrieval_device.type == "cuda"
                and evaluation_embedding_residency
                == utils.EVALUATION_EMBEDDING_RESIDENCY_GPU
            )
            test_embeddings, test_labels = utils.extract_eval_embeddings(
                model,
                test_loader,
                "test",
                device=args.device,
                keep_on_device=keep_test_embeddings_on_device,
                output_device=(
                    test_retrieval_device
                    if keep_test_embeddings_on_device
                    else None
                ),
            )
            test_outcome = utils.unpack_evaluation_result(
                utils.evaluate_embeddings(
                    test_embeddings,
                    test_labels,
                    name="test",
                    return_per_class=False,
                    dataset=dataset_bundle.test_dataset,
                    device=test_retrieval_device,
                    allow_cpu_fallback=False,
                    recall_at_k=test_recall_at_k_values,
                    measurements=test_measurements_requested,
                    return_measurements=True,
                ),
                recall_at_k=test_recall_at_k_values,
                measurements=test_measurements_requested,
                return_measurements=True,
            )
            test_precision = test_outcome.precision_at_1
            test_map = test_outcome.mean_average_precision_at_r
            test_recall = dict(test_outcome.recall_at_k)
            test_measurement_values = dict(test_outcome.measurements)
            test_per_class = None
            metrics_logger.log_eval(
                "test",
                test_precision,
                test_map,
                step=epoch_trainer.global_step,
                epoch=last_epoch,
                per_class_metrics=test_per_class,
                recall_at_k=test_outcome.recall_at_k,
                measurements=test_outcome.measurements,
            )
            if getattr(args, "save_test_embeddings", False):
                # Retained so sibling runs (the folds of one cross-validation)
                # can be evaluated together on the same test samples later.
                embedding_set = build_test_embedding_set(
                    test_embeddings,
                    test_labels,
                    dataset=dataset_bundle.test_dataset,
                )
                if (
                    get_fold_test_embedding_storage(args)
                    == FOLD_TEST_EMBEDDING_STORAGE_MEMORY
                ):
                    # The folds of one cross-validation run in this process, so
                    # the joint evaluation can read the arrays out of the
                    # returned result and no file has to be written at all.
                    test_embedding_set = embedding_set
                    logger.info(
                        f"Keeping {len(embedding_set['embeddings'])} test embeddings of "
                        f"dimension {embedding_set['embeddings'].shape[1]} in memory; no "
                        "test_embeddings.npz is written"
                    )
                else:
                    test_embeddings_path = save_test_embedding_set(
                        embedding_set,
                        args.log_dir,
                    )
            # Every requested projection is drawn from the same test embeddings,
            # so asking for both costs one extra projection and no extra run.
            final_test_visualizations = normalize_final_test_visualization(
                getattr(args, "final_test_visualization", FINAL_TEST_VISUALIZATION_NONE)
            )
            if FINAL_TEST_VISUALIZATION_PACMAP in final_test_visualizations:
                pacmap_artifacts = utils.write_pacmap_visualization(
                    test_embeddings,
                    test_labels,
                    output_dir=args.log_dir,
                    stem="test_pacmap",
                    title=f"{args.dataset} final test embeddings - PacMAP",
                    dataset=dataset_bundle.test_dataset,
                    dataset_name=args.dataset,
                )
                test_pacmap_coordinates = pacmap_artifacts["coordinates"]
                test_pacmap_plot = pacmap_artifacts["plot"]
                logger.info(f"PacMAP final test visualization written to {test_pacmap_plot}")
            if FINAL_TEST_VISUALIZATION_TSNE in final_test_visualizations:
                tsne_artifacts = utils.write_tsne_visualization(
                    test_embeddings,
                    test_labels,
                    output_dir=args.log_dir,
                    stem="test_tsne",
                    title=f"{args.dataset} final test embeddings - t-SNE",
                    dataset=dataset_bundle.test_dataset,
                    dataset_name=args.dataset,
                    seed=args.seed,
                )
                test_tsne_coordinates = tsne_artifacts["coordinates"]
                test_tsne_plot = tsne_artifacts["plot"]
                logger.info(f"t-SNE final test visualization written to {test_tsne_plot}")
        else:
            logger.info("Skipping test evaluation for this run")
    finally:
        # Ensure file handles and TensorBoard writers close even when training,
        # evaluation, or an Optuna pruning decision raises an exception.
        epoch_trainer.close_progress_bar()
        if step_visualizer is not None:
            # Drawing runs behind the training loop, so the run waits here for
            # the last figures rather than exiting with them half-written.
            step_visualizer.close()
        shutdown_epoch_train_loader(
            train_loader,
            warmup_train_loader,
            static_train_loader,
            loss_driven_train_loader,
        )
        utils.shutdown_dataloaders(
            train_loader,
            static_train_loader,
            loss_driven_train_loader,
            warmup_train_loader,
            valid_loader,
            test_loader,
            getattr(regularizer, "_regularizer_loader", None),
        )
        gc.collect()
        metrics_logger.close()
        if precompute_frozen_features:
            feature_stats = {
                "enabled": True,
                "batch_size": get_frozen_feature_batch_size(args),
                "train_views": frozen_feature_train_views,
                "train_samples": (
                    precomputed_train_samples
                    if precomputed_train_samples is not None
                    else gpu_feature_loader.loader_sample_count(static_train_loader)
                ),
                "valid_samples": gpu_feature_loader.loader_sample_count(valid_loader),
                "test_samples": gpu_feature_loader.loader_sample_count(test_loader),
                "backbone": f"dinov2_vit{args.dino_size}14",
                "residency": get_frozen_feature_residency(args),
                "resident_views": sorted(
                    name
                    for name, loader in (
                        ("train", static_train_loader),
                        ("valid", valid_loader),
                        ("test", test_loader),
                    )
                    if gpu_feature_loader.loader_features_are_resident(loader)
                ),
                "persistent_cache_enabled": bool(args.use_cache),
                "persistent_cache_dir": None if model.cache_dir is None else str(model.cache_dir),
                "shared_hpo_cache": (
                    None
                    if frozen_feature_cache is None
                    else frozen_feature_cache.stats()
                ),
            }
            write_json(args.log_dir / "frozen_feature_precompute_stats.json", feature_stats)
            logger.info(f"Frozen feature precompute stats: {feature_stats}")
        if args.use_cache:
            cache_stats = model.cache_stats()
            write_json(args.log_dir / "backbone_cache_stats.json", cache_stats)
            logger.info(f"Backbone cache stats: {cache_stats}")
        best_model_checkpoint.cleanup()

    return TrainingResult(
        log_dir=args.log_dir,
        metrics_csv=args.log_dir / "metrics.csv",
        best_valid_precision_at_1=None if best_precision is None else float(best_precision),
        best_valid_mean_average_precision_at_r=None if best_map is None else float(best_map),
        test_precision_at_1=None if test_precision is None else float(test_precision),
        test_mean_average_precision_at_r=None if test_map is None else float(test_map),
        final_train_loss=None if final_train_loss is None else float(final_train_loss),
        last_epoch=last_epoch,
        selected_epoch=selected_epoch,
        global_step=epoch_trainer.global_step,
        epoch0_test_precision_at_1=None if epoch0_test_precision is None else float(epoch0_test_precision),
        epoch0_test_mean_average_precision_at_r=None if epoch0_test_map is None else float(epoch0_test_map),
        best_valid_recall_at_k=dict(best_recall_at_k) or None,
        test_recall_at_k=dict(test_recall) or None,
        epoch0_test_recall_at_k=dict(epoch0_test_recall) or None,
        best_valid_measurements=dict(best_measurement_values) or None,
        test_measurements=dict(test_measurement_values) or None,
        epoch0_test_measurements=dict(epoch0_test_measurement_values) or None,
        cv_k=args.cv_k if cv_fold is not None else 1,
        cv_mode=args.cv_mode if cv_fold is not None else None,
        cv_fold=cv_fold,
        validation_retrieval_backend=validation_retrieval_backend,
        test_retrieval_backend=test_retrieval_backend,
        test_pacmap_coordinates=test_pacmap_coordinates,
        test_pacmap_plot=test_pacmap_plot,
        test_tsne_coordinates=test_tsne_coordinates,
        test_tsne_plot=test_tsne_plot,
        test_embeddings_path=test_embeddings_path,
        test_embedding_set=test_embedding_set,
        head_state_path=head_state_path,
        warmup_best_valid_precision_at_1=None
        if warmup_best_precision is None
        else float(warmup_best_precision),
        warmup_best_valid_mean_average_precision_at_r=None
        if warmup_best_map is None
        else float(warmup_best_map),
        warmup_selected_epoch=warmup_selected_epoch,
        warmup_checkpoint_mode=(
            ssl_config.warmup_checkpoint_mode
            if ssl_config.enabled and ssl_config.warmup_epochs > 0
            else None
        ),
        warmup_restored_epoch=warmup_restored_epoch,
    )


def build_test_embedding_set(test_embeddings, test_labels, dataset):
    """Return one run's D_test embeddings in the form a joint evaluation needs.

    The query/gallery partition travels with the embeddings because it is the
    only thing the evaluator needs the dataset object for; carrying it here
    means a combined evaluation never has to rebuild the test dataset. The same
    dictionary is what a saved ``test_embeddings.npz`` is read back into, so
    keeping the arrays in memory and reloading them are interchangeable.
    """

    embeddings = to_numpy_embeddings(test_embeddings)
    labels = np.asarray(to_numpy_embeddings(test_labels)).reshape(-1)
    query_gallery_indices = utils.get_query_gallery_indices(dataset, len(embeddings))
    query_indices = None
    gallery_indices = None
    if query_gallery_indices is not None:
        query_indices = np.asarray(query_gallery_indices[0], dtype=np.int64)
        gallery_indices = np.asarray(query_gallery_indices[1], dtype=np.int64)
    return {
        "embeddings": embeddings,
        "labels": labels,
        "query_indices": query_indices,
        "gallery_indices": gallery_indices,
    }


def save_test_embedding_set(embedding_set, output_dir):
    """Persist one run's D_test embeddings for later joint evaluation."""

    embeddings = embedding_set["embeddings"]
    arrays = {"embeddings": embeddings, "labels": embedding_set["labels"]}
    if embedding_set.get("query_indices") is not None:
        arrays["query_indices"] = embedding_set["query_indices"]
        arrays["gallery_indices"] = embedding_set["gallery_indices"]
    output_path = Path(output_dir) / "test_embeddings.npz"
    np.savez(output_path, **arrays)
    logger.info(
        f"Saved {len(embeddings)} test embeddings of dimension "
        f"{embeddings.shape[1]} to {output_path}"
    )
    return output_path


def save_test_embeddings(test_embeddings, test_labels, dataset, output_dir):
    """Build and persist one run's D_test embeddings in one step."""

    return save_test_embedding_set(
        build_test_embedding_set(test_embeddings, test_labels, dataset),
        output_dir,
    )


def to_numpy_embeddings(values):
    """Return a host-side numpy view of possibly device-resident embeddings."""

    if torch.is_tensor(values):
        return values.detach().to("cpu").numpy()
    return np.asarray(values)


def concatenate_fold_test_embeddings(fold_results):
    """Join every fold's saved test embeddings into one matrix per test sample.

    This is the "Concatenated" protocol from A Metric Learning Reality Check:
    each fold contributes its own L2 normalized embedding of the same test
    sample, the pieces are joined in fold order into one cv_k * feat_dim vector,
    and that vector is L2 normalized.
    """

    embedding_sets = []
    for result in fold_results:
        # A fold that kept its embeddings in memory needs no file; one that
        # wrote them is read back into the same dictionary shape.
        embedding_set = getattr(result, "test_embedding_set", None)
        if embedding_set is not None:
            embedding_sets.append(embedding_set)
            continue
        path = getattr(result, "test_embeddings_path", None)
        if path is None:
            raise ValueError(
                "Concatenated fold evaluation requires every fold's test embeddings, "
                "either held in memory or saved to test_embeddings.npz"
            )
        with np.load(path) as arrays:
            embedding_sets.append(
                {
                    "embeddings": np.asarray(arrays["embeddings"], dtype=np.float32),
                    "labels": np.asarray(arrays["labels"]).reshape(-1),
                    "query_indices": (
                        arrays["query_indices"] if "query_indices" in arrays else None
                    ),
                    "gallery_indices": (
                        arrays["gallery_indices"] if "gallery_indices" in arrays else None
                    ),
                }
            )
    return concatenate_test_embedding_sets(embedding_sets)


def concatenate_test_embedding_sets(embedding_sets):
    """Join per-fold embeddings of the same test samples, in fold order."""

    fold_embeddings = []
    labels = None
    query_indices = None
    gallery_indices = None
    for fold_index, embedding_set in enumerate(embedding_sets):
        embeddings = np.asarray(embedding_set["embeddings"], dtype=np.float32)
        fold_labels = np.asarray(embedding_set["labels"]).reshape(-1)
        fold_query = embedding_set.get("query_indices")
        fold_gallery = embedding_set.get("gallery_indices")
        if labels is None:
            labels = fold_labels
            query_indices = fold_query
            gallery_indices = fold_gallery
        elif not np.array_equal(labels, fold_labels):
            # Concatenation is positional, so folds that embedded a different
            # test set (or the same one in a different order) cannot be joined.
            raise ValueError(
                f"Fold {fold_index} evaluated a different test set than fold 0; "
                "concatenated evaluation requires identical test samples in the same order"
            )
        # Each fold is renormalized before joining so one fold with larger norms
        # cannot dominate the concatenated vector.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        fold_embeddings.append(embeddings / np.maximum(norms, np.finfo(np.float32).tiny))

    concatenated = np.concatenate(fold_embeddings, axis=1)
    norms = np.linalg.norm(concatenated, axis=1, keepdims=True)
    concatenated = concatenated / np.maximum(norms, np.finfo(np.float32).tiny)
    return concatenated, labels, query_indices, gallery_indices


def evaluate_concatenated_fold_test_embeddings(
    fold_results,
    device,
    recall_at_k=(),
    measurements=None,
):
    """Evaluate D_test once on every fold's embeddings joined per test sample."""

    recall_at_k = utils.normalize_recall_at_k(recall_at_k)
    measurements = utils.normalize_measurements(measurements)
    concatenated, labels, query_indices, gallery_indices = concatenate_fold_test_embeddings(
        fold_results
    )
    evaluation_dataset = ConcatenatedEvaluationDataset(query_indices, gallery_indices)
    outcome = utils.unpack_evaluation_result(
        utils.evaluate_embeddings(
            concatenated,
            labels,
            name="concatenated fold test",
            return_per_class=False,
            dataset=evaluation_dataset,
            device=device,
            allow_cpu_fallback=False,
            recall_at_k=recall_at_k,
            measurements=measurements,
            return_measurements=True,
        ),
        recall_at_k=recall_at_k,
        measurements=measurements,
        return_measurements=True,
    )
    return {
        "concatenated_test_precision_at_1": float(outcome.precision_at_1),
        "concatenated_test_mean_average_precision_at_r": float(
            outcome.mean_average_precision_at_r
        ),
        "concatenated_test_embedding_dim": int(concatenated.shape[1]),
        "concatenated_test_recall_at_k": dict(outcome.recall_at_k) or None,
        "concatenated_test_measurements": dict(outcome.measurements) or None,
    }


class ConcatenatedEvaluationDataset:
    """Carries only the query/gallery partition the evaluator reads."""

    def __init__(self, query_indices, gallery_indices):
        self.query_indices = query_indices
        self.gallery_indices = gallery_indices


def run_cross_validation(
    args,
    ssl_config,
    optuna_trial=None,
    optuna_metric=None,
    frozen_feature_cache=None,
):
    """Train and aggregate CV folds, independently unless SLADE opts into handoff."""

    validate_run_args(args, ssl_config)
    slade_teacher_handoff = slade_cross_fold_teacher_handoff_enabled(ssl_config)
    if slade_teacher_handoff:
        logger.warning(
            "SLADE paper-style cross-fold teacher handoff is enabled: fold models are "
            "sequential and validation scores are order-dependent, so their mean is not an "
            "independent cross-validation estimate"
        )
    # A restarted replay continues the CV run it was interrupted in, keeping
    # the folds it already trained; every other run starts its own directory.
    resumable_run = resume.find_resumable_cross_validation_run(args)
    fold_results = [] if resumable_run is None else list(resumable_run.fold_results)
    # All fold directories live below one timestamped CV directory so their
    # partial and final aggregate summaries can be updated in place.
    cv_run_name = (
        f"cv_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        if resumable_run is None
        else resumable_run.cv_dir.name
    )
    cv_relative_dir = Path(args.save_dir) / cv_run_name
    cv_dir = Path("logs") / cv_relative_dir
    cv_dir.mkdir(parents=True, exist_ok=True)
    if fold_results:
        logger.info(
            f"Resuming cross-validation run {cv_dir} at fold {len(fold_results)} of {args.cv_k}: "
            f"reusing {len(fold_results)} completed fold(s) from the interrupted attempt"
        )

    validation_retrieval_backend = getattr(
        args,
        "validation_retrieval_backend",
        None,
    )
    if validation_retrieval_backend is None and fold_results:
        # Reused folds already fixed the backend the remaining folds must use.
        validation_retrieval_backend = fold_results[0].validation_retrieval_backend
    teacher_head_state_path = None
    if slade_teacher_handoff and fold_results:
        teacher_head_state_path = fold_results[-1].head_state_path
        if teacher_head_state_path is None or not Path(teacher_head_state_path).is_file():
            raise FileNotFoundError(
                "Cannot resume paper-style SLADE cross-fold handoff because the last "
                "completed fold has no saved student head. Restart the CV run so every "
                "fold is recorded with cross_fold_teacher_handoff enabled."
            )
    for fold_index in range(len(fold_results), args.cv_k):
        # Each fold gets a fresh namespace, model, basis, Gaussian statistics,
        # and optimizer. In the opt-in SLADE paper lifecycle, only the selected
        # retrieval projection from the preceding student crosses this boundary
        # and becomes the next fold's teacher initialization.
        fold_args = copy.deepcopy(args)
        fold_args.cv_fold = fold_index
        fold_args.save_dir = cv_relative_dir / f"fold_{fold_index:02d}"
        if slade_teacher_handoff:
            fold_args.save_head_state = True
            fold_args.slade_teacher_head_state_path = teacher_head_state_path
        if validation_retrieval_backend is not None:
            fold_args.validation_retrieval_backend = validation_retrieval_backend
        result = run_training(
            fold_args,
            ssl_config,
            optuna_trial=None,
            optuna_metric=None,
            cv_fold=fold_index,
            frozen_feature_cache=frozen_feature_cache,
        )
        fold_results.append(result)
        if slade_teacher_handoff:
            teacher_head_state_path = result.head_state_path
            if teacher_head_state_path is None or not Path(teacher_head_state_path).is_file():
                raise RuntimeError(
                    "SLADE cross-fold teacher handoff requires each fold to save its "
                    "validation-selected student projection head"
                )
        if validation_retrieval_backend is None:
            validation_retrieval_backend = result.validation_retrieval_backend
        # Rewrite the summary after every fold so interrupted CV still leaves
        # useful completed-fold results.
        write_cross_validation_summary(cv_dir, args, fold_results)
        # Optuna sees the aggregate of completed folds as an intermediate value
        # and may prune the trial before remaining folds are trained.
        maybe_report_cv_to_optuna(optuna_trial, optuna_metric, fold_results, fold_index)

    # The returned TrainingResult uses mean metrics, total optimization steps,
    # and the maximum last epoch reached among folds.
    aggregate = aggregate_cross_validation_result(cv_dir, args, fold_results)
    evaluated_concatenated_folds = should_evaluate_concatenated_folds(args, fold_results)
    if evaluated_concatenated_folds:
        # One more retrieval evaluation, no more training: the folds' models are
        # already selected and their test embeddings already computed.
        concatenated = evaluate_concatenated_fold_test_embeddings(
            fold_results,
            device=resolve_concatenated_fold_test_device(args, fold_results),
            recall_at_k=utils.resolve_test_recall_at_k(args),
            measurements=utils.resolve_test_measurements(args),
        )
        aggregate = replace(aggregate, **concatenated)
        logger.info(
            f"Concatenated fold test evaluation ({concatenated['concatenated_test_embedding_dim']}-dim "
            f"from {len(fold_results)} folds): "
            f"precision_at_1={concatenated['concatenated_test_precision_at_1']:.6f}, "
            f"mean_average_precision_at_r="
            f"{concatenated['concatenated_test_mean_average_precision_at_r']:.6f}"
            + "".join(
                f", recall_at_{k}={value:.6f}"
                for k, value in (concatenated["concatenated_test_recall_at_k"] or {}).items()
            )
            + "".join(
                f", {name}={value:.6f}"
                for name, value in (
                    concatenated["concatenated_test_measurements"] or {}
                ).items()
                if name not in utils.BASE_RETRIEVAL_METRICS
            )
        )
    if get_fold_test_embedding_storage(args) == FOLD_TEST_EMBEDDING_STORAGE_TEMPORARY:
        # The files existed only to get the folds' embeddings into one joint
        # evaluation, which has now happened.
        fold_results = discard_fold_test_embedding_files(
            fold_results,
            evaluated=evaluated_concatenated_folds,
        )
    write_cross_validation_summary(cv_dir, args, fold_results, aggregate)
    return aggregate


def should_evaluate_concatenated_folds(args, fold_results):
    """Return whether the folds can and should be evaluated jointly."""

    if not getattr(args, "concatenated_fold_test", False):
        return False
    if not getattr(args, "save_test_embeddings", False):
        return False
    missing = [
        result.cv_fold
        for result in fold_results
        if getattr(result, "test_embedding_set", None) is None
        and getattr(result, "test_embeddings_path", None) is None
    ]
    if missing:
        logger.warning(
            f"Skipping the concatenated fold test evaluation: fold(s) {missing} have no "
            "test embeddings, which happens when a reused fold predates this setting or "
            "kept its embeddings in the memory of an interrupted process"
        )
        return False
    return True


def discard_fold_test_embedding_files(fold_results, evaluated):
    """Delete the per-fold ``test_embeddings.npz`` files this CV run wrote.

    The recorded path is cleared along with the file so the summaries this run
    writes afterwards do not point at something that no longer exists.
    """

    remaining = []
    removed = 0
    for result in fold_results:
        path = getattr(result, "test_embeddings_path", None)
        if path is None:
            remaining.append(result)
            continue
        Path(path).unlink(missing_ok=True)
        removed += 1
        remaining.append(replace(result, test_embeddings_path=None))
    if removed:
        reason = (
            "after the concatenated fold evaluation read them"
            if evaluated
            else "without a concatenated fold evaluation to read them"
        )
        logger.info(
            f"fold_test_embedding_storage='temporary': deleted {removed} per-fold "
            f"test_embeddings.npz file(s) {reason}"
        )
    return remaining


def resolve_concatenated_fold_test_device(args, fold_results):
    """Evaluate the joined embeddings on the backend the folds already used."""

    backend = common_optional_value(fold_results, "test_retrieval_backend")
    if backend is None:
        return torch.device(utils.normalize_device_name(args.device))
    return torch.device(backend)


def aggregate_cross_validation_result(cv_dir, args, fold_results):
    # Keep full fold dictionaries inside the aggregate result for JSON metadata
    # while exposing arithmetic means through the normal TrainingResult fields.
    fold_dicts = [result_to_dict(result) for result in fold_results]
    return TrainingResult(
        log_dir=cv_dir,
        metrics_csv=cv_dir / "cv_results.csv",
        best_valid_precision_at_1=mean_metric(fold_results, "best_valid_precision_at_1"),
        best_valid_mean_average_precision_at_r=mean_metric(
            fold_results,
            "best_valid_mean_average_precision_at_r",
        ),
        test_precision_at_1=mean_optional_metric(fold_results, "test_precision_at_1"),
        test_mean_average_precision_at_r=mean_optional_metric(fold_results, "test_mean_average_precision_at_r"),
        # The spread of the per-fold test scores is the honest error bar on
        # their mean, so it is reported next to it rather than derived later.
        test_precision_at_1_std=std_optional_metric(fold_results, "test_precision_at_1"),
        test_mean_average_precision_at_r_std=std_optional_metric(
            fold_results,
            "test_mean_average_precision_at_r",
        ),
        best_valid_recall_at_k=mean_optional_metric_mapping(
            fold_results,
            "best_valid_recall_at_k",
        ),
        test_recall_at_k=mean_optional_metric_mapping(fold_results, "test_recall_at_k"),
        epoch0_test_recall_at_k=mean_optional_metric_mapping(
            fold_results,
            "epoch0_test_recall_at_k",
        ),
        best_valid_measurements=mean_optional_named_metric_mapping(
            fold_results,
            "best_valid_measurements",
        ),
        test_measurements=mean_optional_named_metric_mapping(
            fold_results,
            "test_measurements",
        ),
        epoch0_test_measurements=mean_optional_named_metric_mapping(
            fold_results,
            "epoch0_test_measurements",
        ),
        test_measurements_std=std_optional_named_metric_mapping(
            fold_results,
            "test_measurements",
        ),
        warmup_best_valid_precision_at_1=mean_optional_metric(
            fold_results,
            "warmup_best_valid_precision_at_1",
        ),
        warmup_best_valid_mean_average_precision_at_r=mean_optional_metric(
            fold_results,
            "warmup_best_valid_mean_average_precision_at_r",
        ),
        warmup_selected_epoch=mean_optional_epoch(
            fold_results,
            "warmup_selected_epoch",
        ),
        warmup_checkpoint_mode=common_optional_value(
            fold_results,
            "warmup_checkpoint_mode",
        ),
        warmup_restored_epoch=mean_optional_epoch(
            fold_results,
            "warmup_restored_epoch",
        ),
        final_train_loss=mean_optional_metric(fold_results, "final_train_loss"),
        last_epoch=max(result.last_epoch for result in fold_results),
        selected_epoch=round(mean_metric(fold_results, "selected_epoch")),
        global_step=sum(result.global_step for result in fold_results),
        cv_k=args.cv_k,
        cv_mode=args.cv_mode,
        fold_results=fold_dicts,
        validation_retrieval_backend=common_optional_value(
            fold_results,
            "validation_retrieval_backend",
        ),
        test_retrieval_backend=common_optional_value(
            fold_results,
            "test_retrieval_backend",
        ),
    )


def mean_metric(results, attr):
    return float(sum(getattr(result, attr) for result in results) / len(results))


def mean_optional_metric(results, attr):
    values = [getattr(result, attr) for result in results if getattr(result, attr) is not None]
    if not values:
        return None
    return float(sum(values) / len(values))


def mean_optional_epoch(results, attr):
    """Mean of an optional per-fold epoch index, rounded back to an integer."""

    value = mean_optional_metric(results, attr)
    return None if value is None else round(value)


def std_optional_metric(results, attr):
    """Sample standard deviation across folds, or None when it is undefined."""

    values = [getattr(result, attr) for result in results if getattr(result, attr) is not None]
    if len(values) < 2:
        return None
    return float(np.std(np.asarray(values, dtype=np.float64), ddof=1))


def mean_optional_metric_mapping(results, attr):
    """Average one per-K metric mapping across folds, K by K.

    Folds that never computed Recall@K contribute nothing, and a K missing from
    some folds is averaged over the folds that do have it rather than dropped.
    """

    mappings = [getattr(result, attr, None) or {} for result in results]
    keys = sorted({int(key) for mapping in mappings for key in mapping})
    if not keys:
        return None
    averaged = {}
    for key in keys:
        values = [
            float(mapping[key])
            for mapping in mappings
            if mapping.get(key) is not None
        ]
        if values:
            averaged[key] = float(np.mean(values))
    return averaged or None


def mean_optional_named_metric_mapping(results, attr):
    """Average string-keyed measurement mappings across folds."""

    mappings = [getattr(result, attr, None) or {} for result in results]
    keys = sorted({str(key) for mapping in mappings for key in mapping})
    averaged = {}
    for key in keys:
        values = [
            float(mapping[key])
            for mapping in mappings
            if mapping.get(key) is not None
        ]
        if values:
            averaged[key] = float(np.mean(values))
    return averaged or None


def std_optional_named_metric_mapping(results, attr):
    """Sample standard deviations for string-keyed fold measurements."""

    mappings = [getattr(result, attr, None) or {} for result in results]
    keys = sorted({str(key) for mapping in mappings for key in mapping})
    spread = {}
    for key in keys:
        values = [
            float(mapping[key])
            for mapping in mappings
            if mapping.get(key) is not None
        ]
        if len(values) >= 2:
            spread[key] = float(np.std(np.asarray(values, dtype=np.float64), ddof=1))
    return spread or None


def common_optional_value(results, attr):
    values = {
        getattr(result, attr)
        for result in results
        if getattr(result, attr) is not None
    }
    if len(values) > 1:
        raise ValueError(f"Cross-validation folds used mixed {attr} values: {values}")
    return None if not values else next(iter(values))


def write_cross_validation_summary(cv_dir, args, fold_results, aggregate=None):
    # During execution only completed folds are written. The final call adds a
    # synthetic "mean" row once all folds have completed.
    recall_at_k = utils.resolve_validation_recall_at_k(args)
    measurements = utils.resolve_validation_measurements(args)
    additional_measurements = utils.additional_measurements(measurements)
    # The test columns follow the test scope, which --report_test_metrics and
    # --test_recall_at_k can widen beyond what validation reported.
    test_recall_at_k = utils.resolve_test_recall_at_k(args)
    test_measurements = utils.resolve_test_measurements(args)
    additional_test_measurements = utils.additional_measurements(test_measurements)
    rows = [
        make_cv_summary_row(
            result,
            recall_at_k=recall_at_k,
            measurements=measurements,
            test_recall_at_k=test_recall_at_k,
            test_measurements=test_measurements,
        )
        for result in fold_results
    ]
    if aggregate is not None:
        rows.append(
            make_cv_summary_row(
                aggregate,
                fold="mean",
                recall_at_k=recall_at_k,
                measurements=measurements,
                test_recall_at_k=test_recall_at_k,
                test_measurements=test_measurements,
            )
        )

    csv_path = cv_dir / "cv_results.csv"
    fieldnames = [
        "fold",
        "cv_k",
        "cv_mode",
        "log_dir",
        "metrics_csv",
        "best_valid_precision_at_1",
        "best_valid_mean_average_precision_at_r",
        *(f"best_valid_{name}" for name in additional_measurements),
        *(f"best_valid_recall_at_{k}" for k in recall_at_k),
        "test_precision_at_1",
        "test_mean_average_precision_at_r",
        *(f"test_{name}" for name in additional_test_measurements),
        *(f"test_recall_at_{k}" for k in test_recall_at_k),
        "test_precision_at_1_std",
        "test_mean_average_precision_at_r_std",
        *(f"test_{name}_std" for name in additional_test_measurements),
        "concatenated_test_precision_at_1",
        "concatenated_test_mean_average_precision_at_r",
        *(f"concatenated_test_{name}" for name in additional_test_measurements),
        *(f"concatenated_test_recall_at_{k}" for k in test_recall_at_k),
        "concatenated_test_embedding_dim",
        "test_embeddings_path",
        "head_state_path",
        "validation_retrieval_backend",
        "test_retrieval_backend",
        "test_pacmap_coordinates",
        "test_pacmap_plot",
        "test_tsne_coordinates",
        "test_tsne_plot",
        "final_train_loss",
        "last_epoch",
        "selected_epoch",
        "warmup_best_valid_precision_at_1",
        "warmup_best_valid_mean_average_precision_at_r",
        "warmup_selected_epoch",
        "warmup_restored_epoch",
        "warmup_checkpoint_mode",
        "global_step",
    ]
    # Rewriting is intentional: the file always represents the latest complete
    # view instead of requiring readers to deduplicate appended rows.
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        cv_dir / "cv_summary.json",
        {
            "args": namespace_to_dict(args),
            "completed_folds": len(fold_results),
            "cv_k": args.cv_k,
            "cv_mode": args.cv_mode,
            "folds": [result_to_dict(result) for result in fold_results],
            "aggregate": None if aggregate is None else result_to_dict(aggregate),
        },
    )


def optional_csv_value(value):
    return "" if value is None else (str(value) if isinstance(value, Path) else value)


def make_cv_summary_row(
    result,
    fold=None,
    recall_at_k=(),
    measurements=None,
    test_recall_at_k=None,
    test_measurements=None,
):
    test_recall = getattr(result, "test_recall_at_k", None) or {}
    best_valid_recall = getattr(result, "best_valid_recall_at_k", None) or {}
    # Unset test scopes mean the test evaluation reported what validation did.
    if test_recall_at_k is None:
        test_recall_at_k = recall_at_k
    if test_measurements is None:
        test_measurements = measurements
    additional = utils.additional_measurements(measurements)
    additional_test = utils.additional_measurements(test_measurements)
    best_valid_measurements = getattr(result, "best_valid_measurements", None) or {}
    test_measurement_values = getattr(result, "test_measurements", None) or {}
    test_measurements_std = getattr(result, "test_measurements_std", None) or {}
    concatenated_measurements = (
        getattr(result, "concatenated_test_measurements", None) or {}
    )
    concatenated_recall = getattr(result, "concatenated_test_recall_at_k", None) or {}
    return {
        **{
            f"best_valid_{name}": optional_csv_value(
                best_valid_measurements.get(name)
            )
            for name in additional
        },
        **{
            f"test_{name}": optional_csv_value(test_measurement_values.get(name))
            for name in additional_test
        },
        **{
            f"test_{name}_std": optional_csv_value(
                test_measurements_std.get(name)
            )
            for name in additional_test
        },
        **{
            f"concatenated_test_{name}": optional_csv_value(
                concatenated_measurements.get(name)
            )
            for name in additional_test
        },
        **{
            f"best_valid_recall_at_{k}": optional_csv_value(best_valid_recall.get(k))
            for k in recall_at_k
        },
        **{
            f"test_recall_at_{k}": optional_csv_value(test_recall.get(k))
            for k in test_recall_at_k
        },
        **{
            f"concatenated_test_recall_at_{k}": optional_csv_value(
                concatenated_recall.get(k)
            )
            for k in test_recall_at_k
        },
        "fold": result.cv_fold if fold is None else fold,
        "cv_k": result.cv_k,
        "cv_mode": "" if result.cv_mode is None else result.cv_mode,
        "log_dir": str(result.log_dir),
        "metrics_csv": str(result.metrics_csv),
        "best_valid_precision_at_1": result.best_valid_precision_at_1,
        "best_valid_mean_average_precision_at_r": result.best_valid_mean_average_precision_at_r,
        "test_precision_at_1": "" if result.test_precision_at_1 is None else result.test_precision_at_1,
        "test_mean_average_precision_at_r": ""
        if result.test_mean_average_precision_at_r is None
        else result.test_mean_average_precision_at_r,
        # Only the synthetic mean row carries a spread and a joint evaluation;
        # a single fold has neither.
        "test_precision_at_1_std": optional_csv_value(
            getattr(result, "test_precision_at_1_std", None)
        ),
        "test_mean_average_precision_at_r_std": optional_csv_value(
            getattr(result, "test_mean_average_precision_at_r_std", None)
        ),
        "concatenated_test_precision_at_1": optional_csv_value(
            getattr(result, "concatenated_test_precision_at_1", None)
        ),
        "concatenated_test_mean_average_precision_at_r": optional_csv_value(
            getattr(result, "concatenated_test_mean_average_precision_at_r", None)
        ),
        "concatenated_test_embedding_dim": optional_csv_value(
            getattr(result, "concatenated_test_embedding_dim", None)
        ),
        "test_embeddings_path": optional_csv_value(
            getattr(result, "test_embeddings_path", None)
        ),
        "head_state_path": optional_csv_value(
            getattr(result, "head_state_path", None)
        ),
        "validation_retrieval_backend": result.validation_retrieval_backend or "",
        "test_retrieval_backend": result.test_retrieval_backend or "",
        "test_pacmap_coordinates": ""
        if result.test_pacmap_coordinates is None
        else str(result.test_pacmap_coordinates),
        "test_pacmap_plot": "" if result.test_pacmap_plot is None else str(result.test_pacmap_plot),
        "test_tsne_coordinates": ""
        if result.test_tsne_coordinates is None
        else str(result.test_tsne_coordinates),
        "test_tsne_plot": "" if result.test_tsne_plot is None else str(result.test_tsne_plot),
        "final_train_loss": "" if result.final_train_loss is None else result.final_train_loss,
        "last_epoch": result.last_epoch,
        "selected_epoch": result.selected_epoch,
        "warmup_best_valid_precision_at_1": optional_csv_value(
            getattr(result, "warmup_best_valid_precision_at_1", None)
        ),
        "warmup_best_valid_mean_average_precision_at_r": optional_csv_value(
            getattr(result, "warmup_best_valid_mean_average_precision_at_r", None)
        ),
        "warmup_selected_epoch": optional_csv_value(
            getattr(result, "warmup_selected_epoch", None)
        ),
        "warmup_restored_epoch": optional_csv_value(
            getattr(result, "warmup_restored_epoch", None)
        ),
        "warmup_checkpoint_mode": optional_csv_value(
            getattr(result, "warmup_checkpoint_mode", None)
        ),
        "global_step": result.global_step,
    }


def maybe_report_cv_to_optuna(optuna_trial, metric, fold_results, fold_index):
    if optuna_trial is None or metric is None:
        return
    # The partial fold mean becomes more representative as CV progresses.
    value = mean_optional_metric(fold_results, metric)
    if value is None:
        # Optional metrics such as test performance may be unavailable when
        # test evaluation is disabled during HPO.
        return
    optuna_trial.report(float(value), step=fold_index)
    if optuna_trial.should_prune():
        import optuna

        raise optuna.TrialPruned()


def validate_run_args(args, ssl_config):
    """Fail early on invalid combinations before allocating model resources."""

    normalize_backbone_tuning_args(args)
    if args.dataset not in DATASETS:
        raise ValueError(f"dataset must be one of {DATASETS}: {args.dataset}")
    utils.validate_dataset_protocol(args.dataset, args.dataset_protocol)
    utils.validate_cifar_imbalance_factor(args.dataset, args.cifar_imbalance_factor)
    utils.validate_cifar_balanced_fraction_protocol(
        dataset_name=args.dataset,
        dataset_protocol=args.dataset_protocol,
        train_fraction=args.cifar_train_fraction,
        test_fraction=args.cifar_test_fraction,
        imbalance_factor=args.cifar_imbalance_factor,
    )
    utils.validate_semi_aves_ood_fraction(
        dataset_protocol=args.dataset_protocol,
        ood_fraction=getattr(
            args,
            "semi_aves_ood_fraction",
            utils.SEMI_AVES_DEFAULT_OOD_FRACTION,
        ),
    )
    utils.validate_semi_inat_ood_fraction(
        dataset_protocol=args.dataset_protocol,
        ood_fraction=getattr(
            args,
            "semi_inat_ood_fraction",
            utils.SEMI_INAT_DEFAULT_OOD_FRACTION,
        ),
    )
    if args.dino_size not in {"s", "b", "l", "g"}:
        raise ValueError(f"dino_size must be one of ['s', 'b', 'l', 'g']: {args.dino_size}")
    if args.loss not in ALL_LOSSES:
        raise ValueError(f"loss must be one of {ALL_LOSSES}: {args.loss}")
    if args.miner not in ALL_MINERS:
        raise ValueError(f"miner must be one of {ALL_MINERS}: {args.miner}")
    for name in ("loss_params", "miner_params", "warmup_loss_params", "warmup_miner_params"):
        if not isinstance(getattr(args, name, {}), dict):
            raise ValueError(f"{name} must be a JSON object")
    loss_driven_ssl = ssl_config.method in semi_supervised.LOSS_DRIVEN_METHODS
    ssl_method = semi_supervised.get_method(ssl_config)
    regularized_ssl = ssl_method is not None and ssl_method.is_regularization_method
    regularizer = ssl_method.make_regularizer(ssl_config) if regularized_ssl else None
    if ssl_config.restart_selection_after_warmup and int(args.epochs) <= int(
        ssl_config.warmup_epochs
    ):
        raise ValueError(
            "restart_selection_after_warmup requires epochs > warmup_epochs so at "
            "least one post-warm-up epoch can be selected; otherwise the run "
            "discards the warm-up's checkpoint and has nothing to replace it with"
        )
    validate_effective_miner_params(args.loss, args.miner, args.miner_params, "miner")
    if uses_ssl_warmup_objective(ssl_config):
        validate_warmup_loss_args(args)
        if uses_shared_mixed_lp_proxy_warmup(args, ssl_config) and args.warmup_loss != args.loss:
            raise ValueError(
                "mixed_label_propagation proxy warmup must name "
                "warmup_loss='MixedLabelPropagationProxyLoss'; the same proxy module is reused"
            )
    unlabeled_source = getattr(args, "unlabeled_source", "split")
    external_unlabeled_dir = getattr(args, "external_unlabeled_dir", None)
    external_unlabeled_filter = getattr(args, "external_unlabeled_filter", utils.EXTERNAL_UNLABELED_FILTER_NONE)
    if external_unlabeled_filter not in utils.EXTERNAL_UNLABELED_FILTERS:
        raise ValueError(
            "external_unlabeled_filter must be one of "
            f"{utils.EXTERNAL_UNLABELED_FILTERS}: {external_unlabeled_filter}"
        )
    if getattr(args, "compcars_min_model_images", 100) <= 0:
        raise ValueError("compcars_min_model_images must be positive")
    compcars_paper_threshold_calibration = getattr(
        args,
        "compcars_paper_threshold_calibration",
        utils.COMPCARS_PAPER_THRESHOLD_CALIBRATION_AUTO,
    )
    if compcars_paper_threshold_calibration not in utils.COMPCARS_PAPER_THRESHOLD_CALIBRATION_MODES:
        raise ValueError(
            "compcars_paper_threshold_calibration must be one of "
            f"{utils.COMPCARS_PAPER_THRESHOLD_CALIBRATION_MODES}: "
            f"{compcars_paper_threshold_calibration}"
        )
    if (
        getattr(args, "compcars_strict_paper_counts", False)
        and external_unlabeled_filter not in utils.EXTERNAL_UNLABELED_FILTERS_COMPCARS_PAPER
    ):
        raise ValueError(
            "compcars_strict_paper_counts requires external_unlabeled_filter in "
            f"{utils.EXTERNAL_UNLABELED_FILTERS_COMPCARS_PAPER}"
        )
    if unlabeled_source not in EXTERNAL_UNLABELED_SOURCES and external_unlabeled_dir is not None:
        raise ValueError(
            "external_unlabeled_dir requires unlabeled_source='external' or 'split_and_external'"
        )
    if (
        unlabeled_source not in EXTERNAL_UNLABELED_SOURCES
        and external_unlabeled_filter != utils.EXTERNAL_UNLABELED_FILTER_NONE
    ):
        raise ValueError("external_unlabeled_filter requires unlabeled_source='external' or 'split_and_external'")
    if unlabeled_source == "labeled" and not is_supervised_mode(args) and not ssl_config.enabled:
        raise ValueError(
            "unlabeled_source='labeled' is an ablation of the unlabeled objective and "
            "requires --mode ssl with an enabled SSL config"
        )
    if unlabeled_source in EXTERNAL_UNLABELED_SOURCES and not is_supervised_mode(args):
        if external_unlabeled_dir is None:
            raise ValueError(f"unlabeled_source={unlabeled_source!r} requires external_unlabeled_dir")
        if not Path(external_unlabeled_dir).is_dir():
            raise ValueError(f"external_unlabeled_dir does not exist: {external_unlabeled_dir}")
        if not ssl_config.enabled:
            raise ValueError("External unlabeled data requires --mode ssl with an enabled SSL config")
    if args.loss == "STMLLoss":
        if not loss_driven_ssl:
            raise ValueError("STMLLoss requires an SSL config with method='stml'")
        if args.miner != "no_miner":
            raise ValueError("STMLLoss requires miner='no_miner' because it does not consume labels")
        if args.batch_size < 2:
            raise ValueError("STMLLoss requires batch_size >= 2")
        if args.stml_g_dim is not None and args.stml_g_dim <= 0:
            raise ValueError("stml_g_dim must be positive when set")
        try:
            stml_loss = metric_losses.STMLLoss(**args.loss_params)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid parameters for loss STMLLoss: {args.loss_params}") from exc
        if args.batch_size % stml_loss.num_neighbors != 0:
            raise ValueError("STMLLoss requires batch_size to be divisible by loss_params.num_neighbors")
    elif loss_driven_ssl:
        raise ValueError(f"SSL method {ssl_config.method!r} requires loss='STMLLoss'")
    if regularizer is not None:
        if args.loss == "STMLLoss":
            raise ValueError("regularized mode requires a supervised loss, not STMLLoss")
        regularizer.validate_run_args(args)
        if getattr(regularizer, "within_fold_teacher_student", False):
            if ssl_config.update_mode != "once":
                raise ValueError(
                    "slade within_fold_teacher_student requires update_mode='once': "
                    "the fold-local teacher clusters once and the student trains against "
                    "those fixed pseudo-labels"
                )
            if int(ssl_config.warmup_epochs) <= 0:
                raise ValueError(
                    "slade within_fold_teacher_student requires warmup_epochs > 0 so the "
                    "teacher is trained on labeled data before it generates pseudo-labels"
                )
            if int(args.epochs) <= int(ssl_config.warmup_epochs):
                raise ValueError(
                    "slade within_fold_teacher_student requires epochs > warmup_epochs so "
                    "at least one student epoch is trained"
                )
            if regularizer.regularizer_weight <= 0:
                raise ValueError(
                    "slade within_fold_teacher_student requires regularizer_weight > 0"
                )
        refresh_resets = [
            name
            for name in ("reset_basis_on_refresh", "reset_basis_warmup_on_refresh")
            if getattr(regularizer, name, False)
        ]
        if refresh_resets and ssl_config.update_mode == "once":
            raise ValueError(
                f"slade {', '.join(refresh_resets)} only take effect on a re-clustering, "
                "but update_mode='once' clusters a single time. Use "
                "update_mode='every_n_epochs' with an update_interval_epochs shorter than "
                "the run so the self-training loop actually iterates. Note that "
                "within_fold_teacher_student pins update_mode='once', so its single "
                "explicit teacher/student boundary cannot currently be combined with "
                "per-iteration resets."
            )
    try:
        utils.normalize_device_name(args.device)
    except ValueError as exc:
        raise ValueError(f"{exc}: {args.device}") from exc
    try:
        args.ssl_device = get_ssl_device(args)
    except ValueError as exc:
        raise ValueError(f"Invalid SSL device: {exc}") from exc
    if args.optim not in {"adamw", "adam", "rmsprop"}:
        raise ValueError(f"optim must be 'adamw', 'adam', or 'rmsprop': {args.optim}")
    scheduler_name = getattr(args, "lr_scheduler", LR_SCHEDULER_NONE)
    if scheduler_name not in LR_SCHEDULERS:
        raise ValueError(f"lr_scheduler must be one of {LR_SCHEDULERS}: {scheduler_name}")
    args.lr_scheduler_params_resolved = resolve_lr_scheduler_params(args)
    if args.mode not in {"supervised", "ssl"}:
        raise ValueError("mode must be 'supervised' or 'ssl'")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if getattr(args, "frozen_feature_batch_size", None) is not None and args.frozen_feature_batch_size <= 0:
        raise ValueError("frozen_feature_batch_size must be positive when set")
    if get_frozen_feature_train_views(args) <= 0:
        raise ValueError("frozen_feature_train_views must be positive")
    residency = get_frozen_feature_residency(args)
    if residency not in utils.FEATURE_RESIDENCIES:
        raise ValueError(
            f"frozen_feature_residency must be one of {utils.FEATURE_RESIDENCIES}: {residency}"
        )
    residency_max_gb = getattr(args, "frozen_feature_residency_max_gb", None)
    if residency_max_gb is not None and float(residency_max_gb) <= 0:
        raise ValueError("frozen_feature_residency_max_gb must be positive when set")
    evaluation_embedding_residency = get_evaluation_embedding_residency(args)
    if evaluation_embedding_residency not in utils.EVALUATION_EMBEDDING_RESIDENCIES:
        raise ValueError(
            "evaluation_embedding_residency must be one of "
            f"{utils.EVALUATION_EMBEDDING_RESIDENCIES}: "
            f"{evaluation_embedding_residency!r}"
        )
    faiss_temp_memory_mb = getattr(args, "faiss_temp_memory_mb", None)
    if faiss_temp_memory_mb is not None and float(faiss_temp_memory_mb) < 0:
        raise ValueError("faiss_temp_memory_mb must be non-negative when set")
    cache_max_gb = getattr(args, "frozen_feature_cache_max_gb", None)
    if cache_max_gb is not None and float(cache_max_gb) <= 0:
        raise ValueError("frozen_feature_cache_max_gb must be positive when set")
    if getattr(args, "debug_batch_timing_interval", 5) <= 0:
        raise ValueError("debug_batch_timing_interval must be positive")
    if getattr(args, "batch_loss_log_interval", 0) < 0:
        raise ValueError("batch_loss_log_interval must be non-negative")
    if (getattr(args, "ssl_gradient_contribution_log_interval", 0) or 0) < 0:
        raise ValueError("ssl_gradient_contribution_log_interval must be non-negative")
    if args.lr <= 0:
        raise ValueError("lr must be positive")
    if args.classifier_lr <= 0:
        raise ValueError("classifier_lr must be positive")
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    if not np.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if args.sampler_m <= 0:
        raise ValueError("sampler_m must be positive")
    if ssl_config.labeled_batch_size is not None:
        labeled_batch_size = int(ssl_config.labeled_batch_size)
        unlabeled_batch_size = args.batch_size - labeled_batch_size
        if labeled_batch_size <= 0 or labeled_batch_size >= args.batch_size:
            raise ValueError(
                "Two-stream labeled_batch_size must be greater than zero and smaller than "
                "batch_size so both streams are non-empty"
            )
        if labeled_batch_size % args.sampler_m != 0:
            raise ValueError(
                "Two-stream labeled_batch_size must be divisible by sampler_m: "
                f"labeled_batch_size={labeled_batch_size}, sampler_m={args.sampler_m}"
            )
        if unlabeled_batch_size % args.sampler_m != 0:
            raise ValueError(
                "Two-stream unlabeled batch size (batch_size - labeled_batch_size) must be "
                f"divisible by sampler_m: unlabeled_batch_size={unlabeled_batch_size}, "
                f"sampler_m={args.sampler_m}"
            )
    if args.epochs < 0 or (args.epochs == 0 and not getattr(args, "final_full_train", False)):
        raise ValueError("epochs must be positive, except a final full-train run may use zero selected epochs")
    if args.patience <= 0:
        raise ValueError("patience must be positive")
    if args.cv_k <= 0:
        raise ValueError("cv_k must be positive")
    if getattr(args, "final_full_train", False) and args.cv_k != 1:
        raise ValueError("A final full-development fit requires cv_k=1")
    if args.cv_mode not in utils.CV_MODES:
        raise ValueError(f"cv_mode must be one of {utils.CV_MODES}: {args.cv_mode}")
    if args.cv_mode in utils.SUPERCLASS_AWARE_CV_MODES and args.dataset != "CIFAR100":
        raise ValueError(f"cv_mode={args.cv_mode!r} is only supported for CIFAR100")
    if args.val_mode not in utils.VAL_MODES:
        raise ValueError(f"val_mode must be one of {utils.VAL_MODES}: {args.val_mode}")
    holdout_val_ratio = get_holdout_val_ratio(args)
    if holdout_val_ratio is not None and getattr(args, "final_full_train", False):
        raise ValueError(
            "holdout_val_ratio sizes a validation slice, but a final full-development fit "
            "keeps no validation data; use final_fit_mode='early_stop_holdout' instead"
        )
    if holdout_val_ratio is not None and args.cv_k > 1:
        # Not an error: the same request usually carries this flag for the
        # single-holdout final fit that follows the folds.
        logger.info(
            f"holdout_val_ratio={holdout_val_ratio} sizes a single validation holdout and "
            f"does not apply to this cv_k={args.cv_k} run, whose folds define the split"
        )
    if args.selection_metric not in SELECTION_METRICS:
        raise ValueError(f"selection_metric must be one of {SELECTION_METRICS}: {args.selection_metric}")
    normalize_final_test_visualization(
        getattr(args, "final_test_visualization", FINAL_TEST_VISUALIZATION_NONE)
    )
    validation_retrieval_backend = getattr(
        args,
        "validation_retrieval_backend",
        None,
    )
    if (
        validation_retrieval_backend is not None
        and str(validation_retrieval_backend).lower()
        not in utils.EVALUATION_RETRIEVAL_BACKENDS
    ):
        raise ValueError(
            "validation_retrieval_backend must be one of "
            f"{utils.EVALUATION_RETRIEVAL_BACKENDS}: "
            f"{validation_retrieval_backend!r}"
        )
    if args.feat_dim is not None and args.feat_dim <= 0:
        raise ValueError("feat_dim must be positive when set")
    regularizer_provides_projection = (
        loss_driven_ssl
        or (regularizer is not None and regularizer.provides_trainable_projection_without_feat_dim)
    )
    if args.backbone_tuning == BACKBONE_TUNING_FROZEN and args.feat_dim is None and not regularizer_provides_projection:
        raise ValueError("backbone_tuning='frozen' requires feat_dim so a trainable projection head remains")
    projection_layers = get_projection_layers(args)
    if projection_layers < 1:
        raise ValueError("projection_layers must be at least 1")
    if projection_layers > 1 and args.feat_dim is None and not regularizer_provides_projection:
        raise ValueError(
            "projection_layers > 1 requires feat_dim so the model has a trainable projection head"
        )
    projection_hidden_dim = getattr(args, "projection_hidden_dim", None)
    if projection_hidden_dim is not None and int(projection_hidden_dim) <= 0:
        raise ValueError("projection_hidden_dim must be positive when set")
    if (
        getattr(args, "frozen_head_only_checkpoint", False)
        and args.backbone_tuning != BACKBONE_TUNING_FROZEN
    ):
        raise ValueError(
            "frozen_head_only_checkpoint requires backbone_tuning='frozen'"
        )
    if (
        getattr(args, "save_head_state", False)
        and args.backbone_tuning != BACKBONE_TUNING_FROZEN
    ):
        raise ValueError(
            "save_head_state requires backbone_tuning='frozen' because a tuned "
            "backbone is part of the fitted model and is not saved with the head"
        )
    if args.use_cache and args.backbone_tuning != BACKBONE_TUNING_FROZEN:
        raise ValueError(
            "use_cache requires backbone_tuning='frozen' because tuned backbone features are not stable"
        )
    utils.validate_dataloader_settings(
        device=args.device,
        num_workers=args.num_workers,
        ssl_embedding_num_workers=ssl_config.embedding_num_workers if ssl_config.enabled else 0,
        start_method=args.dataloader_start_method,
    )
    if args.ssl_device != args.device:
        utils.validate_dataloader_settings(
            device=args.ssl_device,
            num_workers=0,
            ssl_embedding_num_workers=(
                ssl_config.embedding_num_workers if ssl_config.enabled else 0
            ),
            start_method=args.dataloader_start_method,
        )


def validate_effective_miner_params(loss_name, miner_name, miner_params, param_name):
    """Validate miner params only when the training loop will actually use the miner."""

    if miner_name == "no_miner" or loss_name in CLASSIFICATION_LOSSES or loss_name == "STMLLoss":
        return
    try:
        validate_named_miner_params(miner_name, miner_params)
    except ValueError as exc:
        raise ValueError(f"Invalid parameters for {param_name} {miner_name}: {exc}") from exc


def validate_warmup_loss_args(args):
    """Validate the supervised objective used during labeled-only SSL warmup."""

    if args.warmup_loss == WARMUP_LOSS_SAME_AS_LOSS:
        # Resolved into a concrete objective by resolve_warmup_objective before
        # any component is built; --loss and --miner are validated on their own.
        return
    if args.warmup_loss not in ALL_LOSSES or args.warmup_loss == "STMLLoss":
        raise ValueError("warmup_loss must be a standard supervised loss")
    if args.warmup_miner not in ALL_MINERS:
        raise ValueError(f"warmup_miner must be one of {ALL_MINERS}: {args.warmup_miner}")
    if args.warmup_loss in CLASSIFICATION_LOSSES and args.warmup_miner != "no_miner":
        raise ValueError("classification warmup_loss requires warmup_miner='no_miner'")
    validate_effective_miner_params(
        args.warmup_loss,
        args.warmup_miner,
        args.warmup_miner_params,
        "warmup_miner",
    )


def validate_named_miner_params(name, params=None):
    """Return validated/normalized miner constructor params."""

    params = dict(params or {})
    if name == "BatchEasyHardMiner":
        return validate_batch_easy_hard_miner_params(params)
    return params


def validate_batch_easy_hard_miner_params(params):
    params = dict(params or {})
    pos_strategy = validate_batch_easy_hard_strategy(
        params.get("pos_strategy", BATCH_EASY_HARD_DEFAULT_POS_STRATEGY),
        "pos_strategy",
    )
    neg_strategy = validate_batch_easy_hard_strategy(
        params.get("neg_strategy", BATCH_EASY_HARD_DEFAULT_NEG_STRATEGY),
        "neg_strategy",
    )

    if pos_strategy == "semihard" and neg_strategy == "semihard":
        raise ValueError("pos_strategy and neg_strategy cannot both be 'semihard'")
    if pos_strategy == "semihard" and neg_strategy == "all":
        raise ValueError("neg_strategy cannot be 'all' when pos_strategy is 'semihard'")
    if pos_strategy == "all" and neg_strategy == "semihard":
        raise ValueError("pos_strategy cannot be 'all' when neg_strategy is 'semihard'")

    for range_name in BATCH_EASY_HARD_RANGE_PARAMS:
        if range_name in params:
            params[range_name] = validate_batch_easy_hard_allowed_range(params[range_name], range_name)
    return params


def validate_batch_easy_hard_strategy(value, name):
    if value not in BATCH_EASY_HARD_MINER_STRATEGIES:
        allowed = sorted(BATCH_EASY_HARD_MINER_STRATEGIES)
        raise ValueError(f"{name} must be one of {allowed}: {value!r}")
    return value


def validate_batch_easy_hard_allowed_range(value, name):
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be null or a two-value range")
    lower, upper = value
    if (
        isinstance(lower, bool)
        or isinstance(upper, bool)
        or not isinstance(lower, Real)
        or not isinstance(upper, Real)
    ):
        raise ValueError(f"{name} bounds must be numeric")
    if lower > upper:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    return (lower, upper)


def make_named_loss(name, params=None, num_classes=None, embedding_size=None):
    """Construct a selected loss name with explicit constructor parameters."""

    params = dict(params or {})
    loss_class = get_loss_class(name)
    logger.info(f"Loss: {name}, params={params}")
    try:
        if name in CLASSIFICATION_LOSSES:
            if num_classes is None or embedding_size is None:
                raise ValueError(f"{name} requires num_classes and embedding_size")
            return loss_class(num_classes, embedding_size, **params)
        return loss_class(**params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid parameters for loss {name}: {params}") from exc


def make_training_loss_components(
    args,
    loss_name,
    loss_params,
    miner_name,
    miner_params,
    num_classes,
    embedding_size,
):
    """Build a criterion, optional miner, and optional criterion optimizer."""

    criterion = make_named_loss(
        name=loss_name,
        params=loss_params,
        num_classes=num_classes,
        embedding_size=embedding_size,
    ).to(args.device)
    is_classification = loss_name in CLASSIFICATION_LOSSES
    classifier_optim = None
    if is_classification:
        classifier_optim = make_optimizer(
            args,
            criterion.parameters(),
            lr=args.classifier_lr,
        )
        miner = None
    else:
        miner = make_named_miner(miner_name, miner_params)
    return TrainingLossComponents(criterion, is_classification, miner, classifier_optim)


def make_warmup_loss_components(
    args,
    ssl_config,
    criterion,
    is_classification,
    miner,
    classifier_optim,
    num_classes,
    embedding_size,
):
    """Build warmup components, reusing the main objective when it matches."""

    if uses_main_objective_for_warmup(args, ssl_config):
        # Alias the objects so trainable criterion state (for example class
        # proxies) and optimizer moments continue after labeled-only warmup.
        return TrainingLossComponents(criterion, is_classification, miner, classifier_optim)
    return make_training_loss_components(
        args=args,
        loss_name=args.warmup_loss,
        loss_params=args.warmup_loss_params,
        miner_name=args.warmup_miner,
        miner_params=args.warmup_miner_params,
        num_classes=num_classes,
        embedding_size=embedding_size,
    )


def make_optimizer(args, parameters, lr):
    weight_decay = float(getattr(args, "weight_decay", 0.0))
    if args.optim == "adamw":
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    if args.optim == "adam":
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if args.optim == "rmsprop":
        return torch.optim.RMSprop(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {args.optim}")


def resolve_lr_scheduler_params(args):
    """Return validated constructor parameters for the selected LR scheduler."""

    scheduler_name = getattr(args, "lr_scheduler", LR_SCHEDULER_NONE)
    params = getattr(args, "lr_scheduler_params", {})
    if not isinstance(params, dict):
        raise ValueError("lr_scheduler_params must be a JSON object")
    if scheduler_name == LR_SCHEDULER_NONE:
        return {}
    if scheduler_name == LR_SCHEDULER_STEP:
        defaults = {"step_size": 10, "gamma": 0.1}
    elif scheduler_name == LR_SCHEDULER_COSINE:
        defaults = {
            "T_max": max(1, int(getattr(args, "epochs", 1))),
            "eta_min": 0.0,
        }
    elif scheduler_name == LR_SCHEDULER_COSINE_WARM_RESTARTS:
        defaults = {"T_0": 10, "T_mult": 1, "eta_min": 0.0}
    else:
        raise ValueError(f"lr_scheduler must be one of {LR_SCHEDULERS}: {scheduler_name}")

    unknown_params = sorted(set(params) - set(defaults))
    if unknown_params:
        raise ValueError(
            f"Unknown parameters for lr_scheduler={scheduler_name!r}: {unknown_params}. "
            f"Supported parameters are {sorted(defaults)}"
        )
    resolved = {**defaults, **params}

    if scheduler_name == LR_SCHEDULER_STEP:
        step_size = resolved["step_size"]
        gamma = resolved["gamma"]
        if isinstance(step_size, bool) or not isinstance(step_size, Integral) or step_size <= 0:
            raise ValueError("lr_scheduler_params.step_size must be a positive integer")
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, Real)
            or not np.isfinite(gamma)
            or gamma <= 0
        ):
            raise ValueError("lr_scheduler_params.gamma must be finite and positive")
        resolved["step_size"] = int(step_size)
        resolved["gamma"] = float(gamma)
    elif scheduler_name == LR_SCHEDULER_COSINE:
        t_max = resolved["T_max"]
        eta_min = resolved["eta_min"]
        if isinstance(t_max, bool) or not isinstance(t_max, Integral) or t_max <= 0:
            raise ValueError("lr_scheduler_params.T_max must be a positive integer")
        if (
            isinstance(eta_min, bool)
            or not isinstance(eta_min, Real)
            or not np.isfinite(eta_min)
            or eta_min < 0
        ):
            raise ValueError("lr_scheduler_params.eta_min must be finite and non-negative")
        resolved["T_max"] = int(t_max)
        resolved["eta_min"] = float(eta_min)
    else:
        t_0 = resolved["T_0"]
        t_mult = resolved["T_mult"]
        eta_min = resolved["eta_min"]
        if isinstance(t_0, bool) or not isinstance(t_0, Integral) or t_0 <= 0:
            raise ValueError("lr_scheduler_params.T_0 must be a positive integer")
        if isinstance(t_mult, bool) or not isinstance(t_mult, Integral) or t_mult < 1:
            raise ValueError("lr_scheduler_params.T_mult must be an integer >= 1")
        if (
            isinstance(eta_min, bool)
            or not isinstance(eta_min, Real)
            or not np.isfinite(eta_min)
            or eta_min < 0
        ):
            raise ValueError("lr_scheduler_params.eta_min must be finite and non-negative")
        resolved["T_0"] = int(t_0)
        resolved["T_mult"] = int(t_mult)
        resolved["eta_min"] = float(eta_min)
    return resolved


def make_lr_scheduler(args, optimizer):
    """Construct the selected learning-rate scheduler, or None when disabled."""

    scheduler_name = getattr(args, "lr_scheduler", LR_SCHEDULER_NONE)
    params = resolve_lr_scheduler_params(args)
    if scheduler_name == LR_SCHEDULER_NONE:
        return None
    if scheduler_name == LR_SCHEDULER_STEP:
        return torch.optim.lr_scheduler.StepLR(optimizer, **params)
    if scheduler_name == LR_SCHEDULER_COSINE:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **params)
    if scheduler_name == LR_SCHEDULER_COSINE_WARM_RESTARTS:
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **params)
    raise ValueError(f"lr_scheduler must be one of {LR_SCHEDULERS}: {scheduler_name}")


def make_stml_teacher(student_model):
    """Create STML's teacher with the student's current backbone and a fresh g head."""

    teacher_model = copy.deepcopy(student_model)
    torch.nn.init.orthogonal_(teacher_model.embedding_g.weight)
    torch.nn.init.zeros_(teacher_model.embedding_g.bias)
    teacher_model.requires_grad_(False)
    teacher_model.eval()
    return teacher_model


def initialize_stml_teacher_for_phase(teacher_model, student_model, active_criterion, device):
    """Create the EMA teacher when, and only when, the STML phase begins."""

    if teacher_model is not None or not getattr(active_criterion, "requires_stml_embeddings", False):
        return teacher_model
    teacher_model = make_stml_teacher(student_model).to(device)
    logger.info("Initialized EMA teacher from the fully warmed-up student")
    return teacher_model


@torch.no_grad()
def update_ema_teacher(
    teacher_model,
    student_model,
    momentum,
    excluded_parameter_prefixes=(),
    only_trainable_parameters=False,
):
    """Update teacher state by EMA, copying only non-floating counters."""

    if not 0 <= momentum < 1:
        raise ValueError("teacher momentum must be in [0, 1)")
    excluded_parameter_prefixes = tuple(excluded_parameter_prefixes)
    teacher_parameters = dict(teacher_model.named_parameters())
    teacher_updates = []
    student_updates = []
    for name, student_parameter in student_model.named_parameters():
        if name.startswith(excluded_parameter_prefixes) or (
            only_trainable_parameters and not student_parameter.requires_grad
        ):
            continue
        teacher_updates.append(teacher_parameters[name])
        student_updates.append(student_parameter.detach())
    if teacher_updates:
        torch._foreach_lerp_(teacher_updates, student_updates, 1 - momentum)

    teacher_buffers = dict(teacher_model.named_buffers())
    floating_teacher_buffers = []
    floating_student_buffers = []
    for name, student_buffer in student_model.named_buffers():
        teacher_buffer = teacher_buffers[name]
        if torch.is_floating_point(teacher_buffer):
            floating_teacher_buffers.append(teacher_buffer)
            floating_student_buffers.append(student_buffer.detach())
        else:
            teacher_buffer.copy_(student_buffer.detach())
    if floating_teacher_buffers:
        torch._foreach_lerp_(floating_teacher_buffers, floating_student_buffers, 1 - momentum)


def get_loss_class(name):
    """Return a project-local or pytorch-metric-learning loss class."""

    if name in metric_losses.LOSS_REGISTRY:
        return metric_losses.LOSS_REGISTRY[name]
    return getattr(losses, name)


def unpack_training_batch(batch):
    """Normalize training batches to images, labels, confidence, and index."""

    if len(batch) == 2:
        images, labels = batch
        return images, labels, torch.ones(len(labels), dtype=torch.float32), None
    if len(batch) in {3, 4}:
        images, labels, sample_weights, *optional_index = batch
        index = optional_index[0] if optional_index else None
        return images, labels, sample_weights.to(dtype=torch.float32), index
    raise ValueError(f"Training batches must contain 2 to 4 items; got {len(batch)}")


def make_named_miner(name, params=None):
    """Construct a selected miner name with explicit constructor parameters."""

    if name == "no_miner":
        return None
    raw_params = dict(params or {})
    params = validate_named_miner_params(name, raw_params)
    logger.info(f"Miner: {name}, params={params}")
    try:
        return getattr(miners, name)(**params)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid parameters for miner {name}: {raw_params}") from exc


def make_supervised_split_config(ssl_config):
    # Preserve label-selection settings and seed so the supervised baseline
    # sees the same labeled subset, while removing all pseudo-label behavior.
    config_dict = ssl_config.to_dict()
    config_dict.update(
        {
            "method": "none",
            "update_mode": "once",
            "update_interval_epochs": 1,
            "warmup_epochs": 0,
            # No SSL phase means no warm-up boundary to restart selection at,
            # and leaving it set would discard the epoch -1 baseline at epoch 0.
            "restart_selection_after_warmup": False,
            "confidence_threshold": 0.0,
            "pseudo_label_rescue_confidence_floor": 0.0,
            "pseudo_label_rescue_top_k": None,
            "labeled_batch_size": None,
            "class_overlap": "independent",
            "graph_batch_mode": "global",
            "graph_labeled_batch_size": None,
            "graph_unlabeled_batch_size": None,
            "method_params": {},
        }
    )
    config = semi_supervised.SemiSupervisedConfig(**config_dict)
    semi_supervised.validate_ssl_config(config)
    return config


def maybe_report_to_optuna(
    optuna_trial,
    metric,
    epoch,
    train_loss,
    best_precision,
    best_map,
):
    """Report an intermediate objective value and honor Optuna pruning."""

    if optuna_trial is None or metric is None:
        # Normal non-HPO training uses this same path but has nothing to report
        # and cannot be pruned.
        return
    value_by_metric = {
        "best_valid_precision_at_1": best_precision,
        "best_valid_mean_average_precision_at_r": best_map,
        "final_train_loss": train_loss,
    }
    # Test metrics are unavailable until training finishes, so only metrics
    # meaningful during epochs appear in this intermediate mapping.
    value = value_by_metric.get(metric)
    if value is None:
        return
    # Optuna compares this step/value pair with other trials according to the
    # configured pruning algorithm.
    optuna_trial.report(float(value), step=epoch)
    if optuna_trial.should_prune():
        import optuna

        raise optuna.TrialPruned()


def write_run_config(args, ssl_config):
    write_json(
        args.log_dir / "run_config.json",
        {
            "args": namespace_to_dict(args),
            "ssl_config": ssl_config.to_dict(),
        },
    )
