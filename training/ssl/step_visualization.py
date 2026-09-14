"""Per-step regularizer artifacts for ordinary training runs.

``scripts/ssl_loss_debugger`` draws one set of figures per training step so a
regularizer's batch can be read pair by pair.  Those figures answer the same
questions during a real run, but a real run has thousands of steps, so here they
are opt-in and throttled: ``--visualization_interval`` is the number of active
regularization steps between two sets of artifacts, and ``null`` draws none.

The drawing code is not duplicated.  It is imported from the debugger the first
time a visualizer is built, which keeps matplotlib and the debugger package out
of the import graph of every run that does not ask for artifacts.
"""

import copy
import queue
import threading
import traceback
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from loguru import logger

import utils


# Regularizers the debugger knows how to draw a single step for.  Any other
# regularizer trains normally and simply produces no per-step artifacts.
VISUALIZED_REGULARIZERS = ("seraph", "simmatch_v2", "stml")

# Regularizers whose artifacts name the true class behind an unlabeled sample.
# They are debugger-style diagnostics: the regularizer itself never reads them.
LABEL_READING_REGULARIZERS = ("seraph", "simmatch_v2")

VISUALIZATION_LAYOUTS = ("pca", "pacmap", "tsne")
DEFAULT_VISUALIZATION_LAYOUT = "tsne"
DEFAULT_SERAPH_TRACE_PAIRS = 12

# How many drawings may be waiting on the worker thread. Rendering a contact
# sheet costs far more than the step that produced it, so the queue is short: a
# run that keeps hitting this bound wants a larger interval, not a deeper
# backlog of figures describing steps it has long since left behind.
MAX_PENDING_DRAWINGS = 2


def get_visualization_interval(args):
    """Return the per-step visualization interval, or None when it is disabled."""

    interval = getattr(args, "visualization_interval", None)
    if interval is None:
        return None
    interval = int(interval)
    return interval if interval > 0 else None


def step_visualization_enabled(args, regularizer):
    """Return whether this run draws per-step artifacts for its regularizer."""

    if get_visualization_interval(args) is None or regularizer is None:
        return False
    return regularizer.name in VISUALIZED_REGULARIZERS


def describe_unsupported_step_visualization(args, regularizer):
    """Return why a requested interval draws nothing, or None when it will draw."""

    if get_visualization_interval(args) is None:
        return None
    if step_visualization_enabled(args, regularizer):
        return None
    if regularizer is None:
        return "this run trains without an SSL regularizer"
    return (
        f"regularizer {regularizer.name!r} has no per-step visualization; "
        f"the drawable regularizers are {list(VISUALIZED_REGULARIZERS)}"
    )


class _VisualizationContext:
    """The attribute surface the debugger's drawing helpers read.

    ``DebugContext`` carries a whole debugger scenario.  The drawing helpers only
    ever touch these five members, so a real run supplies its own equivalents
    rather than reconstructing a scenario it does not have.
    """

    def __init__(self, args, dataset, split, model, device):
        self.args = args
        self.dataset = dataset
        self.split = split
        self.model = model
        self.device = device


class _TrainDatasetView:
    """The train dataset as the drawing helpers see it.

    They read the true class behind a batch's dataset positions and, on a run
    whose batches hold precomputed features, the images those features were
    computed from.  Wrapping keeps both on one object instead of attaching
    diagnostic state to the dataset the training loop is feeding from.
    """

    def __init__(self, dataset, display_image_dataset=None):
        self._dataset = dataset
        # Only the artifacts that name a true class need these; a regularizer
        # whose figures are pure geometry runs without them.
        self.labels = getattr(dataset, "labels", None)
        self.display_image_dataset = display_image_dataset

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]


def _make_display_image_dataset(image_dataset):
    """Return ``image_dataset`` read through its deterministic feature transform.

    A frozen-backbone run precomputes one feature row per position from exactly
    this transform, so these are the images the traced pairs actually compared.
    The copy exists because the training loader is still reading the original
    with its augmentations attached.
    """

    if image_dataset is None:
        return None
    feature_transform = utils.get_nested_feature_transform(image_dataset)
    if feature_transform is None:
        return None
    display_dataset = copy.deepcopy(image_dataset)
    utils.set_nested_transform(display_dataset, feature_transform)
    return display_dataset


def _snapshot(value):
    """Copy tensors onto the CPU so a later step cannot change what gets drawn."""

    if torch.is_tensor(value):
        return value.detach().to("cpu", copy=True)
    if isinstance(value, (list, tuple)):
        return type(value)(_snapshot(item) for item in value)
    return value


class _DrawingThread:
    """Run the figure drawing off the training thread.

    Everything queued here is already a CPU-side reading of the step: matplotlib
    rendering, 2-D projection, and image decoding are pure functions of that
    reading, so they can lag arbitrarily far behind training without changing
    it. What may *not* lag is taking the reading, which is why the model reads
    stay with the caller.
    """

    def __init__(self, name, on_failure):
        self._queue = queue.Queue(maxsize=MAX_PENDING_DRAWINGS)
        self._on_failure = on_failure
        self._name = name
        self._drawn = 0
        self._skipped = 0
        self._thread = threading.Thread(
            target=self._run,
            name=f"{name}-step-visualization",
            daemon=True,
        )
        self._thread.start()

    def submit(self, description, draw):
        """Queue a drawing, or skip it when the thread is already behind."""

        try:
            self._queue.put_nowait((description, draw))
            return True
        except queue.Full:
            self._skipped += 1
            if self._skipped == 1:
                # Said once. Drawing is far slower than a step, so on a real run
                # most due steps are skipped by design and the count at the end
                # is the useful number, not one line per step.
                logger.info(
                    f"The {self._name} visualization thread is drawing more slowly "
                    "than steps come due, so due steps it is still busy for are "
                    "skipped rather than queued. Training is never held up; raise "
                    "--visualization_interval to ask for fewer."
                )
            return False

    def _run(self):
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                description, draw = item
                try:
                    draw()
                    # Only this thread writes _drawn, only the training thread
                    # writes _skipped, and close() reads both after the join.
                    self._drawn += 1
                except Exception:
                    self._on_failure(f"{description} failed:\n{traceback.format_exc()}")
            finally:
                self._queue.task_done()

    def close(self):
        """Finish the queued drawings and stop the thread."""

        self._queue.put(None)
        self._thread.join()
        logger.info(
            f"Per-step {self._name} visualization drew {self._drawn} step(s)"
            + (
                ""
                if not self._skipped
                else f" and skipped {self._skipped} due step(s) it was too busy for"
            )
        )


def _run_output_root(args):
    """Return the directory this particular run's artifacts belong in.

    The default sits inside the run's own log directory, which is already unique.
    An explicit ``--visualization_dir`` is shared by every fold and every run that
    names it, so the run's identity has to be put back: the cross-validation
    directory names the run and the fold directory names the fold, the same way
    the log tree separates them. Without this, each fold would overwrite the
    previous one's figures.
    """

    requested = getattr(args, "visualization_dir", None)
    log_dir = Path(args.log_dir)
    if requested is None:
        return log_dir / "step_visualizations"
    run_parts = [
        part
        for part in log_dir.parts
        if part.startswith("cv_") or part.startswith("fold_")
    ]
    # A single-fold run has neither, and its log directory's own timestamp (or
    # HPO trial name) is what separates it from the run before it.
    return Path(requested).joinpath(*(run_parts or [log_dir.name]))


def _make_context_args(args):
    """Translate run arguments into the namespace the helpers expect."""

    return Namespace(
        # The helpers gate on this themselves; a visualizer only exists when the
        # interval already asked for artifacts.
        visualize=True,
        visualization_dir=_run_output_root(args),
        visualization_layout=(
            getattr(args, "visualization_layout", None) or DEFAULT_VISUALIZATION_LAYOUT
        ),
        seed=int(args.seed),
        dataset=str(args.dataset),
        # Names the scenario directory, exactly as the debugger's
        # --supervised-loss does.
        supervised_loss=str(args.loss),
    )


class StepVisualizer:
    """Draw the debugger's per-step artifacts on an interval during training."""

    def __init__(
        self,
        *,
        args,
        regularizer,
        train_dataset,
        split,
        model,
        device,
        labeled_positions=None,
        image_dataset=None,
    ):
        # Imported here rather than at module scope: the drawing code lives with
        # the debugger and pulls in matplotlib, and an ordinary run never builds
        # a visualizer at all.
        from scripts.ssl_loss_debugger.utils import visualization

        self._visualization = visualization
        self.interval = get_visualization_interval(args)
        self.regularizer = regularizer
        self.name = regularizer.name
        self.max_trace_pairs = int(
            getattr(args, "visualization_seraph_trace_pairs", DEFAULT_SERAPH_TRACE_PAIRS)
        )
        # Positions of the supervised loader's samples within the train dataset.
        # Only available when that loader returns its own indices.
        self.labeled_positions = (
            None
            if labeled_positions is None
            else torch.as_tensor(np.asarray(labeled_positions), dtype=torch.long)
        )
        self._disabled = False
        self._drawings = None
        if self.name in LABEL_READING_REGULARIZERS and not hasattr(train_dataset, "labels"):
            self._disable(
                f"the {self.name} artifacts report the true class behind each "
                "unlabeled sample, which this training dataset does not expose"
            )
            return
        self.context = _VisualizationContext(
            args=_make_context_args(args),
            dataset=_TrainDatasetView(
                train_dataset,
                display_image_dataset=_make_display_image_dataset(image_dataset),
            ),
            split=split,
            model=model,
            device=torch.device(device),
        )
        self.output_dir = visualization.visualization_scenario_dir(
            self.context,
            self.name,
        )
        self._drawings = _DrawingThread(self.name, self._disable)
        logger.info(
            f"Per-step {self.name} visualization enabled: every {self.interval} active "
            f"regularization step(s), drawn on a background thread, writing to "
            f"{self.output_dir}"
            + (
                ""
                if self.context.dataset.display_image_dataset is None
                else "; contact sheets read their images back through the "
                "deterministic feature transform"
            )
        )

    def is_due(self, regularization_step):
        """Return whether this active regularization step draws its artifacts."""

        return not self._disabled and regularization_step % self.interval == 0

    def close(self):
        """Finish the drawings still in flight."""

        if self._drawings is not None:
            self._drawings.close()
            self._drawings = None

    def _disable(self, reason):
        self._disabled = True
        logger.warning(f"Per-step {self.name} visualization disabled: {reason}")

    def _submit(self, description, draw):
        self._drawings.submit(description, draw)

    @contextmanager
    def _guarded(self, description):
        """Keep a failed diagnostic from ending a training run."""

        try:
            yield
        except Exception:
            # One failure is almost always structural rather than incidental, so
            # stop drawing instead of repeating the same traceback every N steps.
            self._disable(f"{description} failed:\n{traceback.format_exc()}")

    def before_regularizer_loss(
        self,
        *,
        step,
        state,
        supervised_inputs,
        regularizer_batch,
    ):
        """Draw what this step reads before its own update overwrites it."""

        if self.name != "simmatch_v2":
            return
        with self._guarded("the labeled-memory propagation graph"):
            reading = self._visualization.read_simmatch_batch_graph(
                self.context,
                self.regularizer,
                state,
                supervised_inputs,
                regularizer_batch,
            )
            self._submit(
                "labeled-memory propagation graph",
                lambda: self._visualization.save_simmatch_batch_graph_from_reading(
                    self.context,
                    self.regularizer,
                    reading,
                    step=step,
                ),
            )

    def after_regularizer_loss(
        self,
        *,
        step,
        state,
        diagnostics,
        supervised_inputs,
        supervised_embeddings,
        supervised_labels,
        supervised_indices,
        regularizer_batch,
        regularizer_embeddings,
    ):
        """Draw what the step's own loss computed, from that loss's diagnostics."""

        if self.name == "seraph":
            with self._guarded("the SERAPH comparison trace"):
                self._save_seraph_comparisons(
                    step=step,
                    diagnostics=diagnostics,
                    supervised_inputs=supervised_inputs,
                    supervised_embeddings=supervised_embeddings,
                    supervised_labels=supervised_labels,
                    supervised_indices=supervised_indices,
                    regularizer_batch=regularizer_batch,
                    regularizer_embeddings=regularizer_embeddings,
                )
        elif self.name == "stml":
            with self._guarded("the STML batch embedding spaces"):
                self._save_stml_batch_spaces(
                    step=step,
                    state=state,
                    regularizer_batch=regularizer_batch,
                )

    def _save_seraph_comparisons(
        self,
        *,
        step,
        diagnostics,
        supervised_inputs,
        supervised_embeddings,
        supervised_labels,
        supervised_indices,
        regularizer_batch,
        regularizer_embeddings,
    ):
        from scripts.ssl_loss_debugger.training.seraph_trace import (
            build_seraph_comparison,
            debug_true_labels,
        )

        if regularizer_embeddings is None:
            raise RuntimeError(
                "the SERAPH trace needs the joint forward's unlabeled embeddings"
            )
        unlabeled_positions = regularizer_batch[2]
        # Without the supervised loader's indices the pair table still holds; it
        # just cannot say which dataset sample the labeled endpoint came from,
        # and the labeled-unlabeled block loses its same-label sheet.
        supervised_positions = None
        if supervised_indices is not None and self.labeled_positions is not None:
            supervised_positions = self.labeled_positions[
                supervised_indices.detach().cpu().long()
            ]
        dataset_labels = self.context.dataset.labels
        comparisons = build_seraph_comparison(
            regularizer=self.regularizer,
            head=self.context.model.seraph_posterior,
            diagnostics=diagnostics,
            supervised_embeddings=supervised_embeddings,
            regularizer_embeddings=regularizer_embeddings,
            supervised_positions=supervised_positions,
            supervised_labels=supervised_labels,
            unlabeled_positions=unlabeled_positions,
            max_pairs_per_block=self.max_trace_pairs,
            true_labels={
                "unlabeled": debug_true_labels(dataset_labels, unlabeled_positions),
                "labeled": debug_true_labels(dataset_labels, supervised_positions),
            },
        )
        # ``comparisons`` is already plain Python numbers. The two input batches
        # are not, and the step that produced them is about to reuse their
        # storage, so the sheet gets its own copy.
        supervised_images = _snapshot(supervised_inputs)
        regularizer_images = _snapshot(regularizer_batch[0])
        self._submit(
            "SERAPH comparison trace",
            lambda: self._visualization.save_seraph_comparison_artifacts(
                context=self.context,
                comparisons=comparisons,
                supervised_inputs=supervised_images,
                regularizer_inputs=regularizer_images,
                step=step,
            ),
        )

    def _save_stml_batch_spaces(self, *, step, state, regularizer_batch):
        views, _, instance_ids = regularizer_batch
        inputs = torch.cat(list(views), dim=0)
        visualization_ids = instance_ids.repeat(self.regularizer.num_views).to(
            self.context.device
        )
        with torch.no_grad():
            student_g, student_f = self.context.model.forward_stml_cached(
                inputs,
                self.context.device,
            )
            teacher_g = state.forward_stml_teacher_cached(inputs, self.context.device)
        # The student and teacher both move on after this step, so the plot gets
        # the three spaces as they were when the loss read them.
        student_f, student_g, teacher_g, visualization_ids = _snapshot(
            (student_f, student_g, teacher_g, visualization_ids)
        )
        self._submit(
            "STML batch embedding spaces",
            lambda: self._visualization.save_stml_batch_visualizations(
                context=self.context,
                scenario=self.name,
                criterion=self.regularizer.criterion,
                student_f=student_f,
                student_g=student_g,
                teacher_g=teacher_g,
                instance_ids=visualization_ids,
                output_stem=f"step_{step:03d}_batch_embedding_spaces",
            ),
        )


def make_step_visualizer(
    args,
    regularizer,
    *,
    train_dataset,
    split,
    model,
    device,
    supervised_dataset=None,
    image_dataset=None,
):
    """Build the run's visualizer, or return None when no artifacts were asked for."""

    unsupported = describe_unsupported_step_visualization(args, regularizer)
    if unsupported is not None:
        logger.warning(
            f"Ignoring visualization_interval={get_visualization_interval(args)}: {unsupported}"
        )
        return None
    if not step_visualization_enabled(args, regularizer):
        return None
    return StepVisualizer(
        args=args,
        regularizer=regularizer,
        train_dataset=train_dataset,
        split=split,
        model=model,
        device=device,
        labeled_positions=getattr(supervised_dataset, "positions", None),
        image_dataset=image_dataset,
    )
