"""CUDA graph capture for the frozen-backbone training step.

A frozen-backbone step is a handful of microsecond-scale kernels, so per-kernel
launch overhead is a large share of the step. Capturing forward, loss, backward,
and the optimizer update into one CUDA graph replaces those launches with a
single replay.

Capture freezes tensor shapes. Anything whose output shape depends on the data --
a miner returning a variable number of pairs, most notably -- would silently
replay at the shape it happened to have during capture, which is a correctness
bug rather than a slow path. Such steps are refused up front, and every accepted
graph is checked numerically against an eager step before it is used.
"""

import logging

import torch

logger = logging.getLogger(__name__)

# Adam-family optimizers keep their step counter on the host unless capturable is
# set, which would bake a constant into the graph.
_CAPTURABLE_OPTIMIZERS = (torch.optim.Adam, torch.optim.AdamW)


def describe_capture_blockers(phase, model_use_cache, device):
    """Return the reasons this phase cannot be captured, empty if it can."""

    blockers = []
    if torch.device(device).type != "cuda":
        blockers.append(f"device is {device}")
    if phase.standalone_stml:
        blockers.append("STML phases run a teacher forward with its own cache")
    if phase.regularization_active:
        blockers.append("regularized phases build batches from two streams")
    if getattr(phase.objective, "miner", None) is not None:
        blockers.append("miners emit a data-dependent number of pairs")
    if getattr(phase.objective, "is_classification", False):
        blockers.append("classification objectives step a second optimizer")
    if not model_use_cache:
        blockers.append("graph capture targets the cached frozen-feature path")
    return blockers


def _clone_state(model, optimizers):
    return (
        {key: value.detach().clone() for key, value in model.state_dict().items()},
        [
            {
                id(parameter): {
                    key: value.detach().clone() if torch.is_tensor(value) else value
                    for key, value in state.items()
                }
                for parameter, state in optimizer.state.items()
            }
            for optimizer in optimizers
        ],
    )


def _restore_state(model, optimizers, snapshot):
    model_state, optimizer_states = snapshot
    with torch.no_grad():
        for key, value in model.state_dict().items():
            value.copy_(model_state[key])
    for optimizer, saved in zip(optimizers, optimizer_states):
        for parameter, state in optimizer.state.items():
            for key, value in saved.get(id(parameter), {}).items():
                if torch.is_tensor(value) and torch.is_tensor(state.get(key)):
                    state[key].copy_(value)
                else:
                    state[key] = value


class CudaGraphTrainStep:
    """Replay a captured forward/backward/optimizer step on static buffers.

    Call :meth:`build` to attempt capture. It returns ``None`` when the step
    cannot be captured safely, and the caller keeps the eager path.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device,
        batch_size,
        feature_dim,
        autocast_dtype,
        label_lookup,
        graph,
        static_features,
        static_labels,
        static_weights,
        static_loss,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.feature_dim = int(feature_dim)
        self.autocast_dtype = autocast_dtype
        self.label_lookup = label_lookup
        self.graph = graph
        self.static_features = static_features
        self.static_labels = static_labels
        self.static_weights = static_weights
        self.static_loss = static_loss
        self.replays = 0

    def accepts(self, features, labels):
        """Whether this batch matches the shapes frozen into the graph."""

        return (
            features.ndim == 2
            and features.shape[0] == self.batch_size
            and features.shape[1] == self.feature_dim
            and labels.shape[0] == self.batch_size
        )

    def run(self, features, labels, sample_weights=None):
        """Copy a batch into the static buffers and replay the captured step."""

        self.static_features.copy_(features, non_blocking=True)
        self.static_labels.copy_(labels, non_blocking=True)
        if sample_weights is not None:
            self.static_weights.copy_(sample_weights, non_blocking=True)
        self.graph.replay()
        self.replays += 1
        return self.static_loss


def _make_eager_step(model, optimizer, criterion, label_lookup, autocast_dtype, device_type):
    def step(features, labels):
        optimizer.zero_grad(set_to_none=False)
        with torch.autocast(device_type=device_type, dtype=autocast_dtype, cache_enabled=False):
            embeddings = model.project_features(features)
            mapped = labels if label_lookup is None else label_lookup[labels]
            loss = criterion(embeddings, mapped)
        loss.backward()
        optimizer.step()
        return loss

    return step


def build(
    model,
    optimizer,
    criterion,
    device,
    batch_size,
    feature_dim,
    label_lookup=None,
    autocast_dtype=torch.bfloat16,
    sample_batches=None,
    warmup_steps=3,
    tolerance=1e-3,
):
    """Capture the training step, returning a :class:`CudaGraphTrainStep` or None.

    ``sample_batches`` supplies real ``(features, labels)`` pairs for warmup and
    for the numerical check. Two distinct batches are needed so the check can
    tell a correct replay from a graph that froze the first batch's values.
    """

    device = torch.device(device)
    if device.type != "cuda":
        logger.info("CUDA graph step skipped: not a CUDA device")
        return None
    if not hasattr(model, "project_features"):
        logger.info("CUDA graph step skipped: model has no project_features path")
        return None

    if isinstance(optimizer, _CAPTURABLE_OPTIMIZERS):
        for group in optimizer.param_groups:
            if not group.get("capturable", False):
                group["capturable"] = True

    if sample_batches is None or len(sample_batches) < 2:
        logger.warning("CUDA graph step skipped: need two sample batches to verify the capture")
        return None

    eager_step = _make_eager_step(
        model,
        optimizer,
        criterion,
        label_lookup,
        autocast_dtype,
        device.type,
    )
    snapshot = _clone_state(model, [optimizer])

    features_a, labels_a = sample_batches[0]
    features_b, labels_b = sample_batches[1]

    try:
        # Reference values from the eager path, taken from the same starting
        # state the graph will be replayed from.
        for _ in range(warmup_steps):
            eager_step(features_a, labels_a)
        warm_snapshot = _clone_state(model, [optimizer])
        eager_a = float(eager_step(features_a, labels_a).detach())
        eager_b = float(eager_step(features_b, labels_b).detach())
        _restore_state(model, [optimizer], warm_snapshot)

        static_features = torch.zeros(
            (batch_size, feature_dim), device=device, dtype=features_a.dtype
        )
        static_labels = torch.zeros((batch_size,), device=device, dtype=labels_a.dtype)
        static_weights = torch.ones((batch_size,), device=device, dtype=torch.float32)
        static_features.copy_(features_a)
        static_labels.copy_(labels_a)

        # Warmup on a side stream is required before capture so cuBLAS and the
        # allocator have already done their lazy first-call work.
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(warmup_steps):
                eager_step(static_features, static_labels)
        torch.cuda.current_stream().wait_stream(side_stream)
        _restore_state(model, [optimizer], warm_snapshot)

        graph = torch.cuda.CUDAGraph()
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.graph(graph):
            optimizer.zero_grad(set_to_none=False)
            with torch.autocast(
                device_type=device.type, dtype=autocast_dtype, cache_enabled=False
            ):
                embeddings = model.project_features(static_features)
                mapped = (
                    static_labels if label_lookup is None else label_lookup[static_labels]
                )
                static_loss = criterion(embeddings, mapped)
            static_loss.backward()
            optimizer.step()
    except Exception as error:  # noqa: BLE001 - capture failures must not kill the run
        _restore_state(model, [optimizer], snapshot)
        logger.warning(f"CUDA graph capture failed, using the eager step instead: {error}")
        return None

    step = CudaGraphTrainStep(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        batch_size=batch_size,
        feature_dim=feature_dim,
        autocast_dtype=autocast_dtype,
        label_lookup=label_lookup,
        graph=graph,
        static_features=static_features,
        static_labels=static_labels,
        static_weights=static_weights,
        static_loss=static_loss,
    )

    # Replay from the same state the eager reference started from. A graph that
    # captured data-dependent shapes reproduces batch A but not batch B, so both
    # are checked.
    _restore_state(model, [optimizer], warm_snapshot)
    graph_a = float(step.run(features_a, labels_a).detach())
    graph_b = float(step.run(features_b, labels_b).detach())
    # Capture, warmup, and verification all mutate parameters and optimizer
    # state. Training must resume from exactly where it was, so the entry
    # snapshot is restored either way. copy_ keeps the storages the graph
    # captured pointers to, so the graph stays valid.
    _restore_state(model, [optimizer], snapshot)

    drift_a = abs(graph_a - eager_a)
    drift_b = abs(graph_b - eager_b)
    scale = max(1.0, abs(eager_a), abs(eager_b))
    if drift_a > tolerance * scale or drift_b > tolerance * scale:
        logger.warning(
            "CUDA graph capture rejected: replayed loss disagrees with the eager step "
            f"(batch A {graph_a:.6f} vs {eager_a:.6f}, batch B {graph_b:.6f} vs {eager_b:.6f}). "
            "This usually means a data-dependent shape was frozen into the graph."
        )
        _restore_state(model, [optimizer], snapshot)
        return None

    logger.info(
        f"CUDA graph step captured: batch={batch_size}, feat_dim={feature_dim}, "
        f"verified against the eager step (max drift {max(drift_a, drift_b):.2e})"
    )
    return step
