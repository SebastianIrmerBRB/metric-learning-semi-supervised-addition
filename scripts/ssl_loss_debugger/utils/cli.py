"""Command-line parsing and validation for the debugger harness."""

from __future__ import annotations

import argparse
from pathlib import Path

import utils as project_utils
from models.retrieval_model import (
    BACKBONE_TUNING_FROZEN,
    BACKBONE_TUNING_FULL,
    normalize_backbone_tuning,
)

from scripts.ssl_loss_debugger.utils.constants import (
    ALL_SCENARIOS,
    BACKBONE_FEATURE_SOURCES,
    BACKBONE_FEATURES_CACHED,
    BACKBONE_FEATURES_PIXELS,
    BREAK_PHASES,
    DEBUG_DATASETS,
    SCENARIO_ALIASES,
)
from training.ssl.seraph import (
    KAPPA_MODES,
    SERAPH_METRICS,
    SERAPH_OBJECTIVES,
    THRESHOLD_MODES,
)


def normalize_scenario(value: str) -> str:
    normalized = str(value).strip().lower()
    return SCENARIO_ALIASES.get(normalized, normalized)


def normalize_debug_dataset(value: str) -> str:
    aliases = {
        "synthetic": "synthetic",
        "tiny": "synthetic",
        "cars": "Cars196",
        "cars196": "Cars196",
        "cub": "CUB",
    }
    normalized = str(value).strip().lower().replace("-", "").replace("_", "")
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"dataset must be one of {DEBUG_DATASETS} (aliases: cars, cub)"
        ) from exc


def _add_seraph_arguments(parser: argparse.ArgumentParser) -> None:
    """Expose the SERAPH regularizer_params documented in docs/seraph.md.

    Defaults mirror ``SeraphRegularizer``'s own so a debug run starts from the
    same operating point a production config with an empty ``regularizer_params``
    would, and the choices are imported from the implementation rather than
    restated here.
    """

    group = parser.add_argument_group(
        "seraph",
        "SERAPH regularizer_params; see docs/seraph.md for measured effects.",
    )
    group.add_argument(
        "--seraph-objective",
        choices=SERAPH_OBJECTIVES,
        default="entropy",
        help="'em' runs Eq 9's E-step then a cross-entropy M-step; 'entropy' descends Eq 7.",
    )
    group.add_argument(
        "--seraph-em-sharpening",
        type=float,
        default=2.0,
        help=(
            "E-step exponent 1 + mu; ignored when --seraph-objective=entropy. "
            "1.0 is accepted but makes the gradient identically zero."
        ),
    )
    group.add_argument(
        "--seraph-metric",
        choices=SERAPH_METRICS,
        default="identity",
        help=(
            "Parameterization of A = B^T B. Learnable metrics absorb most of the "
            "regularizer gradient into B instead of the embedding."
        ),
    )
    group.add_argument(
        "--seraph-metric-dim",
        type=int,
        default=None,
        help="Rows of B; only valid with --seraph-metric=lowrank.",
    )
    group.add_argument(
        "--seraph-kappa",
        type=float,
        default=1.0,
        help="Inverse temperature of the pair posterior; drives the term's magnitude.",
    )
    group.add_argument(
        "--seraph-kappa-mode",
        choices=KAPPA_MODES,
        default="fixed",
        help=(
            "'fixed' uses kappa as-is; 'batch_std' uses kappa / std(d^2). The two "
            "modes are not on the same scale."
        ),
    )
    group.add_argument(
        "--seraph-threshold",
        type=float,
        default=1.0,
        help="The fixed-mode eta; only the seed of the running buffers in quantile mode.",
    )
    group.add_argument(
        "--seraph-threshold-mode",
        choices=THRESHOLD_MODES,
        default="fixed",
        help=(
            "'quantile' places eta at the positive_pair_prior quantile of the batch's "
            "distances; 'fixed' pins it and silently deactivates the two knobs below."
        ),
    )
    group.add_argument(
        "--seraph-threshold-momentum",
        type=float,
        default=0.1,
        help="EMA rate of the running quantile eta, in (0, 1) exclusive.",
    )
    group.add_argument(
        "--seraph-positive-pair-prior",
        type=float,
        default=None,
        help=(
            "Quantile eta sits at, in (0, 1) exclusive. Unset derives 1 / num_classes. "
            "Its effect is U-shaped, not monotonic."
        ),
    )
    group.add_argument(
        "--seraph-unlabeled-batch-size",
        type=int,
        default=None,
        help=(
            "Absolute unlabeled batch drawn per step. Unset falls back to "
            "--seraph-unlabeled-ratio. An inverse strength knob: the gradient shrinks "
            "roughly as 1/sqrt(batch)."
        ),
    )
    group.add_argument(
        "--seraph-unlabeled-ratio",
        type=float,
        default=2.0,
        help=(
            "Unlabeled batch as a multiple of the labeled batch. Ignored once "
            "--seraph-unlabeled-batch-size is set."
        ),
    )
    group.add_argument(
        "--seraph-trace-pairs",
        type=int,
        default=12,
        help=(
            "Pairs per block listed in the comparison trace and drawn on the contact "
            "sheets, spread evenly across the block's distance range. The same count "
            "is listed again for the pairs that share a debugger-only true label, on "
            "their own contact sheet. 0 lists none; block aggregates always cover "
            "every scored pair."
        ),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run real SSL loss/regularizer paths on a bounded synthetic or production-dataset "
            "debug workload."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        type=normalize_scenario,
        choices=("all", *ALL_SCENARIOS),
        default="seraph",
        help=(
            "Objective or regularizer to exercise. Accepts the pre-rename alias "
            f"{sorted(SCENARIO_ALIASES)}."
        ),
    )
    parser.add_argument(
        "--supervised-loss",
        choices=("proxy_anchor",),
        default="proxy_anchor",
        help="Supervised objective combined with regularizer scenarios.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help=(
            "Training epochs for regularizer scenarios. Values above one rebuild "
            "LRML/SLRML graphs each epoch and report edge-count changes."
        ),
    )
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--dataset",
        type=normalize_debug_dataset,
        choices=DEBUG_DATASETS,
        default="Cars196",
        help=(
            "Image source; real datasets use the project's production loader and transforms."
        ),
    )
    parser.add_argument(
        "--dataset-protocol",
        choices=project_utils.DATASET_PROTOCOLS,
        default=project_utils.DATASET_PROTOCOL_OFFICIAL,
        help="Production dataset protocol; ignored for the synthetic dataset.",
    )
    parser.add_argument(
        "--download-dataset",
        action="store_true",
        default=True,
        help="Allow the production loader to download a missing real dataset.",
    )
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--labeled-per-class", type=int, default=16)
    parser.add_argument("--unlabeled-per-class", type=int, default=3000000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sampler-m", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=16)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument(
        "--backbone-tuning",
        "--backbone_tuning",
        dest="backbone_tuning",
        type=normalize_backbone_tuning,
        choices=(BACKBONE_TUNING_FULL, BACKBONE_TUNING_FROZEN),
        default=BACKBONE_TUNING_FROZEN,
        help=(
            "Tiny-model backbone policy. 'frozen' trains only the projection and "
            "objective-specific heads."
        ),
    )
    parser.add_argument(
        "--backbone-features",
        choices=BACKBONE_FEATURE_SOURCES,
        default=BACKBONE_FEATURES_PIXELS,
        help=(
            "'pixels' feeds transformed images to the tiny model. 'cached' feeds the "
            "run cache's frozen DINOv2 features instead, which removes image decoding "
            "from every step and puts the losses on the geometry a frozen-backbone run "
            "actually sees. Cached features are one deterministic view per sample, so "
            "scenarios that need augmented views are rejected. Real datasets only."
        ),
    )
    parser.add_argument(
        "--backbone-feature-cache",
        type=Path,
        default=None,
        help=(
            "Explicit .features.npy for --backbone-features=cached. Unset picks the "
            "matrix under data/<dataset>/backbone_cache whose row count matches the "
            "dataset split."
        ),
    )
    parser.add_argument("--graph-neighbors", type=int, default=5)
    parser.add_argument("--graph-batch-size", type=int, default=5)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument(
        "--supervised-weight",
        type=float,
        default=1.0,
        help="method_params.supervised_weight for regularizer scenarios.",
    )
    parser.add_argument(
        "--regularizer-weight",
        type=float,
        default=1.0,
        help=(
            "method_params.regularizer_weight for regularizer scenarios. Scales the "
            "regularizer gradient linearly."
        ),
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=1,
        help=(
            "Epoch cadence for per-step batch dumps, comparison traces and saved "
            "artifacts. The first and last epoch always report; other epochs print a "
            "one-line summary. Raise it for long runs."
        ),
    )
    _add_seraph_arguments(parser)
    parser.add_argument(
        "--visualize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save 2-D embedding-space plots and, for graph-based scenarios, "
            "the graph overlay and sampled-edge CSV."
        ),
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default="debug_ssl_losses",
        help="Root directory for --visualize artifacts.",
    )
    parser.add_argument(
        "--visualization-layout",
        choices=("pca", "pacmap", "tsne"),
        default="tsne",
        help="2-D projection used by plots; nonlinear layouts fall back to PCA on failure.",
    )
    parser.add_argument(
        "--include-supervised-graph",
        action="store_true",
        help="Add SLRML's labeled same-class graph component.",
    )
    parser.add_argument(
        "--include-labeled-unlabeled",
        action="store_true",
        default=True,
        help="Include SERAPH labeled-unlabeled candidate pairs.",
    )
    parser.add_argument(
        "--break-at",
        choices=BREAK_PHASES,
        default="none",
        help="Call breakpoint() at this stable hand-off phase.",
    )
    parser.add_argument(
        "--anomaly-detection",
        action="store_true",
        help="Enable torch autograd anomaly detection.",
    )
    parser.add_argument(
        "--log-level",
        choices=("TRACE", "DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    args = parser.parse_args(argv)
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "epochs",
        "steps",
        "threads",
        "num_classes",
        "labeled_per_class",
        "unlabeled_per_class",
        "batch_size",
        "sampler_m",
        "image_size",
        "embedding_dim",
        "graph_neighbors",
        "graph_batch_size",
        "report_every",
    )
    for name in positive_integer_names:
        if int(getattr(args, name)) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.seraph_trace_pairs < 0:
        raise SystemExit("--seraph-trace-pairs must be zero or positive")
    if args.seraph_metric_dim is not None and args.seraph_metric_dim <= 0:
        raise SystemExit("--seraph-metric-dim must be positive when set")
    if args.seraph_unlabeled_batch_size is not None and args.seraph_unlabeled_batch_size < 2:
        # SERAPH raises the same way, but only after the dataset has been built.
        raise SystemExit("--seraph-unlabeled-batch-size must be at least two")
    if args.num_classes < 2:
        raise SystemExit("--num-classes must be at least 2")
    if args.batch_size % args.sampler_m != 0:
        raise SystemExit("--batch-size must be divisible by --sampler-m")
    if (
        args.dataset == "synthetic"
        and args.batch_size // args.sampler_m > args.num_classes
    ):
        raise SystemExit(
            "The M-per-class batch needs more classes than the synthetic dataset provides; "
            "reduce --batch-size, increase --sampler-m, or increase --num-classes."
        )
    if args.sampler_m > args.labeled_per_class:
        # MPerClassSampler can sample with replacement, but avoiding replacement
        # makes local batch inspection substantially easier to reason about.
        raise SystemExit(
            "--sampler-m cannot exceed --labeled-per-class in this debug harness"
        )
    if not 0.0 <= args.confidence_threshold <= 1.0:
        raise SystemExit("--confidence-threshold must be in [0, 1]")
    if args.backbone_features == BACKBONE_FEATURES_CACHED:
        if args.dataset == "synthetic":
            raise SystemExit(
                "--backbone-features=cached needs a real dataset with a populated "
                "backbone_cache; the synthetic dataset has none"
            )
    elif args.backbone_feature_cache is not None:
        raise SystemExit(
            "--backbone-feature-cache only applies to "
            f"--backbone-features={BACKBONE_FEATURES_CACHED}"
        )
    if args.dataset != "synthetic":
        try:
            project_utils.validate_dataset_protocol(args.dataset, args.dataset_protocol)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
