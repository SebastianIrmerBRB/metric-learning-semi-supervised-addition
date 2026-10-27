"""Measure whether LRML's Laplacian term collapses the embedding, and what stops it.

``LRMLRegularizer``'s docstring and ``docs/lrml.md`` section 8.1 both assert that
the graph term is attraction only and is minimized by mapping every node to one
point, and that only the supervised loss or the optional eq. (3) repulsion
prevents that. Those are claims about an optimum; this benchmark turns them into
measurements of an actual optimization run, so the practical questions get
numbers instead of an argument:

1. does the regularizer alone (``supervised_weight=0``) actually collapse, or is
   the untrained-graph/normalized-Laplacian geometry enough to stop it;
2. at which ``regularizer_weight`` does the *combined* objective start collapsing
   while the supervised loss is still present; and
3. does the ``contrastive`` repulsion buy anything the supervised loss does not
   already provide - i.e. is the repulsive branch needed in the regime the runs
   use. It rides at the same ``regularizer_weight`` as the Laplacian branch,
   exactly as the single lambda of Weston et al. (2012) does, so the arms toggle
   it rather than weighting it.

Why the constant map is the exact optimum
-----------------------------------------
Model outputs are L2-normalized (``DinoWrapper.project_features``), so with the
unnormalized Laplacian the per-edge energy is ``2 - 2 z_i.z_j``, minimized at
``0`` when every ``z_i`` is the same unit vector. With the normalized Laplacian
the energy is ``sum_e (1/d_i + 1/d_j) - 2 sum_e z_i.z_j / sqrt(d_i d_j)``, whose
first sum is ``N`` regardless of ``z``; the second is again maximized when every
``z_i`` coincides. Degree scaling therefore changes the optimum's *value*, not
its location: collapse is the global minimizer in both settings, and the only
question a benchmark can answer is whether an optimizer gets there and how much
supervision it takes to stop it.

What is measured
----------------
Per epoch, from the embeddings of the whole graph pool:

``centroid_norm``
    ``|| mean_i z_i ||`` for unit-norm ``z``: ``1.0`` is exact collapse to a
    single point, ``0`` is a centered cloud. This is the headline number.
``rms_pair_dist``
    ``sqrt(2 (1 - centroid_norm^2))``, the exact RMS pairwise distance of unit
    vectors, on the same scale as ``contrastive_margin``.
``rankme``
    ``exp(entropy(singular value distribution))`` of the embedding matrix
    (Garrido et al., 2023): dimensional collapse that a centroid cannot see.
``uniformity``
    ``log E exp(-2 d^2)`` of Wang & Isola (2020); ``0`` at full collapse,
    decreasing as the sphere fills.
``pre_norm_scale``
    Mean ``|| fc(x) ||`` *before* normalization. A frozen backbone with a linear
    head can also approach collapse by shrinking the head toward zero, which the
    normalized metrics alone would not distinguish from a healthy embedding.
``knn_purity``
    Fraction of the current kNN graph's edges joining same-class nodes. It is
    both the graph's quality and the mechanism behind collapse's self-
    reinforcement: once the embedding contracts, the rebuilt graph stops carrying
    class information, so the term keeps pulling on noise.
``recall_at_1``
    Held-out retrieval on the dataset's other split, which is what collapse
    ultimately costs.

Setup
-----
Real cached frozen DINOv2 features, a trainable ``Linear + L2 normalize`` head
mirroring ``DinoWrapper.fc``/``project_features``, an m-per-class labeled batch
from a k-shot split, and the repository's own LRML pieces:
``build_lrml_knn_graph`` for the symmetric binary kNN graph,
``_reduce_sampled_graph_energy`` for the unbiased edge-batch scaling, and
``_sample_repulsion_pairs``/``_repulsion_energy`` for eq. (3). The per-edge
energy mirrors ``LRMLRegularizer.compute_loss``'s ``graph_batch_mode='global'``
branch, including the degree scaling, the anchoring of each repulsion pair on a
sampled edge, the uniform partner pool appended behind the edge endpoints, and
the ``W_in = 0`` restriction on which of those partners a pair may use.

The frozen backbone is the deliberate limitation: a linear head over fixed
features cannot collapse quite as freely as a fine-tuned DINOv2 can, so the
collapse thresholds this reports are, if anything, conservative. Confirm a
chosen operating point with a real run (see ``--print-run-command``).

Examples::

    python scripts/bench_lrml_collapse.py
    python scripts/bench_lrml_collapse.py --dataset CUB --epochs 40
    python scripts/bench_lrml_collapse.py --weights 1 10 100 --contrastive off on
    python scripts/bench_lrml_collapse.py --csv scripts/reports/lrml_collapse/cars.csv
"""

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from scipy import sparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from training import engine  # noqa: E402
from training.semi_supervised import (  # noqa: E402
    _partner_connectivity_test,
    _reduce_sampled_graph_energy,
    _repulsion_energy,
    _sample_repulsion_pairs,
)
from training.ssl.algorithms import build_lrml_knn_graph  # noqa: E402
from training.ssl.data import graph_upper_triangle_edges  # noqa: E402

# The SERAPH benchmark already sets up exactly the harness this one needs -
# cached DINOv2 features aligned to labels, a k-shot split, an m-per-class batch
# stream, and DinoWrapper's Linear + L2-normalize head - so it is imported rather
# than copied.
from bench_seraph_gradient_contribution import (  # noqa: E402
    BatchStream,
    FrozenFeatureModel,
    load_feature_source,
    make_k_shot_split,
)

# ``configs/ssl_regularized_lrml.json`` plus the graph defaults of
# ``LRMLRegularizer.DEFAULT_PARAMS``.
DEFAULT_N_NEIGHBORS = 6
DEFAULT_NORMALIZED_LAPLACIAN = True
DEFAULT_REDUCTION = "mean"
DEFAULT_CONTRASTIVE_MARGIN = 1.0

# ``centroid_norm`` above this is treated as collapsed: with unit-norm
# embeddings it puts the RMS pairwise distance below 0.2, i.e. every pair inside
# ~11 degrees, which no retrieval task survives.
COLLAPSE_CENTROID_NORM = 0.98
# Reported alongside it: a head whose pre-normalization outputs have shrunk this
# far below their initial scale is collapsing through the normalization floor
# rather than through the embedding geometry.
SHRINK_RATIO = 0.1

LEGEND = """
columns
  centroid   || mean z ||, 1.000 = every embedding on one point (collapse)
  rms_dist   RMS pairwise distance, same scale as contrastive_margin (<= 2)
  rankme     exp(entropy of singular values); 1.0 = one direction left
  unif       log E exp(-2 d^2) of Wang & Isola; 0.0 = collapsed
  scale      mean pre-normalization || fc(x) || relative to its value at init
  purity     fraction of kNN graph edges joining same-class nodes
  R@1        retrieval precision@1 on the held-out split
""".strip()


@dataclass
class Arm:
    """One objective mix: ``alpha * L_sup + lambda * (R_G + repulsion)``.

    The repulsion has no weight of its own: Algorithm 1 spends the same lambda
    on both branches of eq. (3), so an arm only decides whether it is present.
    """

    label: str
    supervised_weight: float
    regularizer_weight: float
    contrastive: bool


@dataclass
class LrmlGraph:
    """One epoch's frozen graph snapshot, as ``LRMLRegularizer`` keeps it."""

    node_positions: np.ndarray  # dataset positions of the graph's nodes
    adjacency: sparse.csr_matrix  # W, which the repulsion draw excludes partners by
    node_scale: torch.Tensor  # 1/sqrt(deg), or ones for the unnormalized Laplacian
    edge_left: torch.Tensor  # upper-triangle edges as local node ids
    edge_right: torch.Tensor
    edge_count: int
    weight_sum: float
    purity: float

    @property
    def num_nodes(self):
        return len(self.node_positions)


# --------------------------------------------------------------------------- #
# Collapse diagnostics
# --------------------------------------------------------------------------- #


def rankme(embeddings, max_rows=8192):
    """Effective rank ``exp(entropy(sigma / sum sigma))`` (Garrido et al., 2023).

    A centroid norm sees collapse to a point; this sees collapse to a subspace,
    which a Laplacian term can also produce on its way down.
    """

    rows = embeddings
    if len(rows) > max_rows:
        rows = rows[torch.randperm(len(rows), device=rows.device)[:max_rows]]
    singular_values = torch.linalg.svdvals(rows.float())
    total = singular_values.sum()
    if not torch.isfinite(total) or float(total) <= 0.0:
        return float("nan")
    distribution = singular_values / total
    entropy = -(distribution * torch.log(distribution.clamp_min(1e-12))).sum()
    return float(torch.exp(entropy))


def uniformity(embeddings, num_pairs=100_000, generator=None):
    """``log E exp(-2 ||z_i - z_j||^2)`` of Wang & Isola (2020).

    Zero at full collapse and decreasing as the embedding spreads, so it
    separates "contracted but not yet a point" from "already a point" long after
    the centroid norm has saturated at 1.
    """

    count = len(embeddings)
    if count < 2:
        return float("nan")
    left = torch.randint(count, (num_pairs,), device=embeddings.device, generator=generator)
    right = torch.randint(count, (num_pairs,), device=embeddings.device, generator=generator)
    keep = left != right
    if not bool(keep.any()):
        return float("nan")
    differences = embeddings[left[keep]] - embeddings[right[keep]]
    squared = (differences * differences).sum(dim=1)
    return float(torch.log(torch.exp(-2.0 * squared).mean()))


def collapse_metrics(embeddings, generator=None):
    """Every scale-free collapse statistic of one embedding matrix."""

    centroid_norm = float(embeddings.mean(dim=0).norm())
    # Exact for unit-norm rows: E ||z_i - z_j||^2 = 2 (1 - ||mean z||^2).
    rms_pair_dist = math.sqrt(max(0.0, 2.0 * (1.0 - centroid_norm**2)))
    return {
        "centroid_norm": centroid_norm,
        "rms_pair_dist": rms_pair_dist,
        "rankme": rankme(embeddings),
        "uniformity": uniformity(embeddings, generator=generator),
        "dim_std_mean": float(embeddings.std(dim=0).mean()),
    }


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


# --------------------------------------------------------------------------- #
# The LRML objective, as ``LRMLRegularizer`` evaluates it
# --------------------------------------------------------------------------- #


@torch.no_grad()
def embed_positions(model, features, positions, batch_size=4096):
    """Embed dataset positions with the deterministic no-grad pass a rebuild uses."""

    model.eval()
    chunks = []
    for start in range(0, len(positions), batch_size):
        rows = torch.as_tensor(positions[start : start + batch_size], device=features.device)
        chunks.append(model(features[rows]))
    model.train()
    return torch.cat(chunks) if chunks else torch.empty(0, device=features.device)


def build_graph(model, source, node_positions, args):
    """Rebuild the frozen graph snapshot from the model's current embeddings.

    Mirrors ``LRMLRegularizer.build_graph``: deterministic no-grad embeddings, an
    exact FAISS kNN search, the symmetric binary union, and degrees that stay
    fixed for the whole epoch while the head keeps moving.
    """

    embeddings = embed_positions(model, source.features, node_positions)
    features = np.ascontiguousarray(embeddings.detach().float().cpu().numpy(), dtype=np.float32)
    _, adjacency, degrees = build_lrml_knn_graph(features, n_neighbors=args.n_neighbors)
    edge_rows, edge_cols, edge_weights = graph_upper_triangle_edges(adjacency)

    node_labels = source.labels.cpu().numpy()[node_positions]
    purity = float(np.mean(node_labels[edge_rows] == node_labels[edge_cols]))
    if args.normalized_laplacian:
        scale = 1.0 / np.sqrt(np.maximum(degrees, 1.0))
    else:
        scale = np.ones_like(degrees)
    return LrmlGraph(
        node_positions=node_positions,
        adjacency=adjacency,
        node_scale=torch.as_tensor(scale, dtype=torch.float32, device=source.features.device),
        edge_left=torch.as_tensor(edge_rows, dtype=torch.long),
        edge_right=torch.as_tensor(edge_cols, dtype=torch.long),
        edge_count=len(edge_rows),
        weight_sum=float(edge_weights.sum()),
        purity=purity,
    )


class EdgeStream:
    """``GraphEdgeBatchSampler``: shuffled passes over the upper-triangle edges."""

    def __init__(self, edge_count, batch_size, seed):
        self.edge_count = int(edge_count)
        self.batch_size = int(batch_size)
        self.generator = np.random.default_rng(seed)
        self.order = self.generator.permutation(self.edge_count)
        self.cursor = 0

    def next_edges(self):
        if self.cursor >= len(self.order):
            self.order = self.generator.permutation(self.edge_count)
            self.cursor = 0
        selected = self.order[self.cursor : self.cursor + self.batch_size]
        self.cursor += self.batch_size
        return selected


def lrml_loss(model, source, graph, edge_stream, arm, args, pair_generator):
    """One step of the graph term, mirroring ``compute_loss``'s global branch."""

    edges = edge_stream.next_edges()
    left_nodes = graph.edge_left[edges]
    right_nodes = graph.edge_right[edges]

    # collate_graph_edge_batch loads each endpoint once and carries local edge
    # indices; a shared endpoint therefore gets one forward and accumulates the
    # gradient of every incident edge.
    endpoints, local = torch.unique(torch.cat([left_nodes, right_nodes]), return_inverse=True)
    local_left, local_right = local[: len(edges)], local[len(edges) :]

    node_ids = endpoints
    if arm.contrastive:
        # The uniform partner pool is appended *behind* the endpoints and is
        # deliberately not deduplicated against them.
        requested_pairs = args.contrastive_pairs or len(edges)
        pool = torch.randint(
            graph.num_nodes, (requested_pairs,), generator=pair_generator, dtype=torch.long
        )
        node_ids = torch.cat([endpoints, pool])

    rows = torch.as_tensor(graph.node_positions[node_ids.numpy()], device=source.features.device)
    embeddings = model(source.features[rows])

    # Fold the per-node 1/sqrt(deg) factor in: the normalized-Laplacian energy is
    # the unnormalized energy of degree-scaled embeddings.
    scaled = embeddings * graph.node_scale[node_ids.to(source.features.device)][:, None]
    differences = scaled[local_left.to(scaled.device)] - scaled[local_right.to(scaled.device)]
    energy = (differences * differences).sum(dim=1).sum()
    laplacian = _reduce_sampled_graph_energy(
        energy=energy,
        sampled_edges=len(edges),
        graph_edges=graph.edge_count,
        graph_weight=graph.weight_sum,
        reduction=args.reduction,
    )
    if not arm.contrastive:
        return laplacian, float(laplacian.detach()), 0.0

    pairs = _sample_repulsion_pairs(
        len(node_ids),
        edge_left=local_left,
        edge_right=local_right,
        partner_start=len(endpoints),
        num_pairs=args.contrastive_pairs or len(edges),
        generator=pair_generator,
        is_connected=_partner_connectivity_test(graph.adjacency, node_ids),
    )
    if pairs is None:
        return laplacian, float(laplacian.detach()), 0.0
    pair_left, pair_right = pairs
    # The repulsion reads the plain embeddings: the degree scaling belongs to the
    # normalized Laplacian, the margin to the embedding space itself.
    repulsion_energy, _ = _repulsion_energy(
        embeddings,
        pair_left.to(embeddings.device),
        pair_right.to(embeddings.device),
        args.contrastive_margin,
    )
    repulsion = _reduce_sampled_graph_energy(
        energy=repulsion_energy,
        sampled_edges=len(pair_left),
        graph_edges=graph.edge_count,
        graph_weight=graph.weight_sum,
        reduction=args.reduction,
    )
    total = laplacian + repulsion
    return total, float(laplacian.detach()), float(repulsion.detach())


# --------------------------------------------------------------------------- #
# One arm
# --------------------------------------------------------------------------- #


def run_arm(arm, source, eval_source, graph_positions, stream, label_lookup, num_classes, args):
    """Train one objective mix from the shared initialization and log its epochs."""

    device = args.device
    # Every arm starts from the same head, so a difference between two rows is
    # the objective and not the draw.
    torch.manual_seed(args.seed)
    model = FrozenFeatureModel(source.backbone_dim, args.feat_dim).to(device)
    criterion = engine.make_named_loss(
        name=args.loss,
        params=args.loss_params,
        num_classes=num_classes,
        embedding_size=args.feat_dim,
    ).to(device)
    miner = engine.make_named_miner(args.miner, args.miner_params)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion_parameters = list(criterion.parameters())
    criterion_optimizer = (
        torch.optim.Adam(criterion_parameters, lr=args.classifier_lr)
        if criterion_parameters
        else None
    )
    pair_generator = torch.Generator().manual_seed(args.seed)
    metric_generator = torch.Generator(device=device).manual_seed(args.seed)

    initial_scale = float(
        model.fc(source.features[torch.as_tensor(graph_positions[:2048], device=device)])
        .norm(dim=1)
        .mean()
    )

    graph = None
    edge_stream = None
    rows = []
    global_step = 0
    for epoch in range(args.epochs):
        # Warmup is labeled-only training, so an arm without a supervised loss has
        # nothing to warm up with and starts its graph term immediately - from a
        # random-init head, whose linear map still preserves enough of the frozen
        # features' neighborhoods for the graph to be meaningful (see knn_purity).
        regularized = arm.regularizer_weight > 0 and (
            epoch >= args.warmup_epochs or arm.supervised_weight == 0
        )
        if regularized and (graph is None or (args.graph_refresh and epoch % args.graph_refresh == 0)):
            graph = build_graph(model, source, graph_positions, args)
            edge_stream = EdgeStream(graph.edge_count, args.graph_batch_size, args.seed + epoch)

        epoch_supervised, epoch_laplacian, epoch_repulsion = [], [], []
        for _ in range(args.steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            if criterion_optimizer is not None:
                criterion_optimizer.zero_grad(set_to_none=True)
            total = torch.zeros((), device=device)

            if arm.supervised_weight > 0:
                indices = torch.as_tensor(stream.labeled_batch(global_step), device=device)
                embeddings = model(source.features[indices])
                labels = label_lookup[source.labels[indices]]
                mined = miner(embeddings, labels) if miner is not None else None
                supervised = (
                    criterion(embeddings, labels, mined)
                    if mined is not None
                    else criterion(embeddings, labels)
                )
                total = total + arm.supervised_weight * supervised
                epoch_supervised.append(float(supervised.detach()))

            if regularized:
                regularizer, laplacian_value, repulsion_value = lrml_loss(
                    model, source, graph, edge_stream, arm, args, pair_generator
                )
                total = total + arm.regularizer_weight * regularizer
                epoch_laplacian.append(laplacian_value)
                epoch_repulsion.append(repulsion_value)

            if not total.requires_grad:
                raise RuntimeError(
                    f"arm {arm.label} produced no trainable term at epoch {epoch}"
                )
            total.backward()
            optimizer.step()
            if criterion_optimizer is not None:
                criterion_optimizer.step()
            global_step += 1

        with torch.no_grad():
            embeddings = embed_positions(model, source.features, graph_positions)
            metrics = collapse_metrics(embeddings, generator=metric_generator)
            scale = float(
                model.fc(source.features[torch.as_tensor(graph_positions[:2048], device=device)])
                .norm(dim=1)
                .mean()
            )
            evaluation = embed_positions(model, eval_source.features, args.eval_positions)
            metrics["recall_at_1"] = recall_at_1(evaluation, eval_source.labels[args.eval_positions])
        metrics.update(
            {
                "arm": arm.label,
                "supervised_weight": arm.supervised_weight,
                "regularizer_weight": arm.regularizer_weight,
                "contrastive": int(arm.contrastive),
                "epoch": epoch,
                "scale_ratio": scale / initial_scale if initial_scale > 0 else float("nan"),
                "knn_purity": graph.purity if graph is not None else float("nan"),
                "supervised_loss": float(np.mean(epoch_supervised)) if epoch_supervised else float("nan"),
                "laplacian": float(np.mean(epoch_laplacian)) if epoch_laplacian else float("nan"),
                "repulsion": float(np.mean(epoch_repulsion)) if epoch_repulsion else float("nan"),
            }
        )
        rows.append(metrics)
        if args.verbose:
            print(f"    {format_row(metrics)}")
    return rows


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def format_row(row, prefix=""):
    return (
        f"{prefix}{row['epoch']:>5}  "
        f"{row['centroid_norm']:>8.4f}  {row['rms_pair_dist']:>8.3f}  "
        f"{row['rankme']:>7.1f}  {row['uniformity']:>7.3f}  "
        f"{row['scale_ratio']:>6.3f}  {row['knn_purity']:>6.3f}  {row['recall_at_1']:>6.3f}"
    )


def collapse_epoch(rows):
    """First epoch whose centroid norm crosses the collapse threshold."""

    for row in rows:
        if row["centroid_norm"] >= COLLAPSE_CENTROID_NORM:
            return row["epoch"]
    return None


def summarize(arm_rows):
    """One line per arm: where it ended and whether it collapsed on the way."""

    header = (
        f"{'arm':<26} {'centroid':>9} {'rms_dist':>9} {'rankme':>7} {'unif':>7} "
        f"{'scale':>6} {'purity':>7} {'R@1':>7}  verdict"
    )
    print(header)
    print("-" * len(header))
    for arm, rows in arm_rows:
        final = rows[-1]
        epoch = collapse_epoch(rows)
        if epoch is not None:
            verdict = f"COLLAPSED (epoch {epoch})"
        elif final["scale_ratio"] < SHRINK_RATIO:
            verdict = f"head shrinking ({final['scale_ratio']:.3f} of init scale)"
        else:
            verdict = "no collapse"
        print(
            f"{arm.label:<26} {final['centroid_norm']:>9.4f} {final['rms_pair_dist']:>9.3f} "
            f"{final['rankme']:>7.1f} {final['uniformity']:>7.3f} {final['scale_ratio']:>6.3f} "
            f"{final['knn_purity']:>7.3f} {final['recall_at_1']:>7.3f}  {verdict}"
        )


def write_csv(path, rows):
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nper-epoch rows: {path}")


def build_arms(args):
    """Supervised-only and regularizer-only controls, then the weight grid."""

    arms = []
    settings = [choice == "on" for choice in args.contrastive]
    if not args.skip_controls:
        arms.append(Arm("sup_only", 1.0, 0.0, False))
        for contrastive in settings:
            label = "lrml_only+rep" if contrastive else "lrml_only"
            arms.append(Arm(label, 0.0, 1.0, contrastive))
    for weight in args.weights:
        for contrastive in settings:
            suffix = "+rep" if contrastive else ""
            arms.append(Arm(f"sup+lrml{weight:g}{suffix}", 1.0, weight, contrastive))
    return arms


def run_command_hint(args):
    """The real-pipeline confirmation run for whatever this benchmark points at.

    The weights live in the SSL config rather than on the command line, so the
    confirmation is a copied config plus one ``--ssl_config`` pointing at it.
    """

    return (
        "  1. copy configs/ssl_regularized_lrml.json and set method_params\n"
        "     supervised_weight / regularizer_weight, and\n"
        "     method_params.regularizer_params.contrastive\n"
        f"  2. python main.py --dataset {args.dataset} --mode ssl "
        f"--ssl_config <copy>.json --loss {args.loss} --miner {args.miner} "
        "--backbone_tuning frozen --use_cache"
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="Cars196", help="dataset whose backbone cache is read")
    parser.add_argument("--split", default="train", help="split the graph and labeled batches come from")
    parser.add_argument("--eval-split", default="test", help="split held-out retrieval is measured on")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=2, help="labeled-only epochs before the graph term")
    parser.add_argument("--graph-refresh", type=int, default=1, help="rebuild the graph every N epochs; 0 keeps the first")
    parser.add_argument("--feat-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--classifier-lr", type=float, default=1e-2)

    parser.add_argument("--loss", default="MultiSimilarityLoss")
    parser.add_argument("--loss-params", default={}, type=json.loads)
    parser.add_argument("--miner", default="MultiSimilarityMiner")
    parser.add_argument("--miner-params", default={}, type=json.loads)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sampler-m", type=int, default=4)
    parser.add_argument("--labeled-per-class", type=int, default=8)

    parser.add_argument("--n-neighbors", type=int, default=DEFAULT_N_NEIGHBORS)
    parser.add_argument("--graph-batch-size", type=int, default=128, help="sampled graph edges per step")
    parser.add_argument("--reduction", default=DEFAULT_REDUCTION, choices=["mean", "sum"])
    parser.add_argument(
        "--unnormalized-laplacian",
        dest="normalized_laplacian",
        action="store_false",
        default=DEFAULT_NORMALIZED_LAPLACIAN,
    )
    parser.add_argument("--contrastive-margin", type=float, default=DEFAULT_CONTRASTIVE_MARGIN)
    parser.add_argument("--contrastive-pairs", type=int, default=None)

    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=[0.1, 1.0, 10.0, 100.0],
        help="regularizer_weight grid combined with the supervised loss",
    )
    parser.add_argument(
        "--contrastive",
        nargs="+",
        choices=["off", "on"],
        default=["off", "on"],
        help="eq. (3) repulsion settings each regularizer_weight is run at",
    )
    parser.add_argument("--skip-controls", action="store_true", help="drop the sup-only and lrml-only arms")
    parser.add_argument("--eval-samples", type=int, default=4096, help="held-out rows R@1 is measured on")
    parser.add_argument("--csv", default=None, help="write every epoch of every arm here")
    parser.add_argument("--verbose", action="store_true", help="print each epoch as it is measured")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    if not args.verbose:
        # Every arm rebuilds the loss and the miner, each logging at INFO.
        logger.remove()
        logger.add(sys.stderr, level="WARNING")

    source = load_feature_source(args.dataset, args.split, args.data_root, args.device)
    eval_source = load_feature_source(args.dataset, args.eval_split, args.data_root, args.device)
    labeled_positions, unlabeled_positions = make_k_shot_split(
        source.labels, args.labeled_per_class, args.seed
    )
    # graph_on="all": the graph spans the labeled and unlabeled pools alike.
    graph_positions = np.sort(np.concatenate([labeled_positions, unlabeled_positions]))

    unique_labels = torch.unique(source.labels)
    num_classes = int(len(unique_labels))
    label_lookup = torch.full(
        (int(unique_labels.max()) + 1,), -1, dtype=torch.long, device=args.device
    )
    label_lookup[unique_labels] = torch.arange(num_classes, device=args.device)

    evaluation_rows = np.arange(len(eval_source.labels))
    if len(evaluation_rows) > args.eval_samples:
        evaluation_rows = np.sort(
            np.random.default_rng(args.seed).choice(
                len(evaluation_rows), size=args.eval_samples, replace=False
            )
        )
    args.eval_positions = evaluation_rows

    print(f"features:  {source.name} {tuple(source.features.shape)} from {source.cache_path}")
    print(
        f"alignment: 1-NN label agreement {source.nn_label_agreement:.3f} "
        f"over {num_classes} classes (chance {1.0 / num_classes:.3f})"
    )
    if source.nn_label_agreement < 5.0 / num_classes:
        raise RuntimeError(
            "cached features look unaligned with the dataset labels; refusing to "
            "report collapse measured against the wrong supervised gradient"
        )
    print(
        f"split:     {len(labeled_positions)} labeled ({args.labeled_per_class}/class), "
        f"{len(unlabeled_positions)} unlabeled, {len(graph_positions)} graph nodes"
    )
    print(
        f"graph:     k={args.n_neighbors}, "
        f"{'normalized' if args.normalized_laplacian else 'unnormalized'} Laplacian, "
        f"reduction={args.reduction}, {args.graph_batch_size} edges/step, "
        f"refresh every {args.graph_refresh or args.epochs} epoch(s)"
    )
    print(
        f"objective: {args.loss}{args.loss_params} + {args.miner}, "
        f"batch {args.batch_size}:{args.sampler_m}, {args.epochs}x{args.steps_per_epoch} steps, "
        f"device {args.device}"
    )
    print(f"held-out:  {len(evaluation_rows)} rows of {eval_source.name}\n")
    print(LEGEND)
    print()

    stream = BatchStream(
        source.labels,
        labeled_positions,
        unlabeled_positions,
        args.batch_size,
        args.sampler_m,
        args.seed,
    )

    arms = build_arms(args)
    arm_rows = []
    all_rows = []
    for arm in arms:
        print(
            f"[{arm.label}] alpha={arm.supervised_weight:g} lambda={arm.regularizer_weight:g} "
            f"repulsion={'on' if arm.contrastive else 'off'}"
        )
        rows = run_arm(
            arm, source, eval_source, graph_positions, stream, label_lookup, num_classes, args
        )
        arm_rows.append((arm, rows))
        all_rows.extend(rows)
    print()
    summarize(arm_rows)
    print(f"\ncollapse threshold: centroid_norm >= {COLLAPSE_CENTROID_NORM}")
    print(f"confirm a chosen point with a real run:\n{run_command_hint(args)}")
    if args.csv:
        write_csv(args.csv, all_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
