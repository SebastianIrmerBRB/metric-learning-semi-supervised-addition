"""Configuration builders for regularizer scenarios."""

from __future__ import annotations

from typing import Any

from scripts.ssl_loss_debugger.training.context import DebugContext
from scripts.ssl_loss_debugger.utils.constants import METHOD_NAMED_SCENARIOS
from scripts.ssl_loss_debugger.utils.visualization import graph_diagnostics_config_kwargs
from training.ssl.config import SemiSupervisedConfig


def regularizer_params(name: str, context: DebugContext) -> dict[str, Any]:
    args = context.args
    graph_neighbors = min(args.graph_neighbors, len(context.dataset) - 1)
    if name == "lrml":
        return {
            "n_neighbors": graph_neighbors,
            "normalized_laplacian": True,
            "graph_on": "all",
            "reduction": "mean",
            "graph_batch_size": 256,
        }
    if name == "slrml":
        return {
            "n_neighbors": graph_neighbors,
            "reduction": "mean",
            "include_supervised_graph": args.include_supervised_graph,
            "graph_batch_size": args.graph_batch_size,
        }
    if name == "seraph":
        params = {
            "objective": args.seraph_objective,
            "em_sharpening": args.seraph_em_sharpening,
            "metric": args.seraph_metric,
            "metric_dim": args.seraph_metric_dim,
            "kappa": args.seraph_kappa,
            "kappa_mode": args.seraph_kappa_mode,
            "threshold": args.seraph_threshold,
            "threshold_mode": args.seraph_threshold_mode,
            "threshold_momentum": args.seraph_threshold_momentum,
            "positive_pair_prior": args.seraph_positive_pair_prior,
            "include_labeled_unlabeled": args.include_labeled_unlabeled,
        }
        # SERAPH accepts both sizings but warns and ignores the ratio when they
        # arrive together, so only the one that is actually in force is passed.
        if args.seraph_unlabeled_batch_size is None:
            params["unlabeled_ratio"] = args.seraph_unlabeled_ratio
        else:
            params["unlabeled_batch_size"] = args.seraph_unlabeled_batch_size
            params["unlabeled_ratio"] = None
        return params
    if name == "hoffer_entropy":
        return {
            "distance_scale": 1.0,
            "reference_sets": 1,
            "unlabeled_batch_size": min(
                args.batch_size,
                len(context.split.unlabeled_positions),
            ),
        }
    raise ValueError(f"Unknown regularizer: {name}")


def make_regularizer_config(context: DebugContext, name: str) -> SemiSupervisedConfig:
    return SemiSupervisedConfig(
        method=name if name in METHOD_NAMED_SCENARIOS else "regularized",
        update_mode="every_epoch" if context.args.epochs > 1 else "once",
        seed=context.args.seed,
        support_seed=context.args.seed,
        confidence_threshold=context.args.confidence_threshold,
        pseudo_label_diagnostics_mode="off",
        embedding_batch_size=min(8, len(context.dataset)),
        embedding_num_workers=0,
        **graph_diagnostics_config_kwargs(context),
        method_params={
            "regularizer": name,
            "supervised_weight": context.args.supervised_weight,
            "regularizer_weight": context.args.regularizer_weight,
            "regularizer_params": regularizer_params(name, context),
        },
    )
