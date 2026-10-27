"""Embedding-space and graph visualizations for debugger scenarios."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import torch
from torch import nn

from scripts.ssl_loss_debugger.training.seraph_trace import (
    DISTANCE_SPREAD_SELECTION,
    SAME_LABEL_SELECTION,
)
from scripts.ssl_loss_debugger.utils.constants import REGULARIZER_SCENARIOS
from training.ssl.embeddings import extract_embeddings
from training.ssl.graph_diagnostics import project_graph_embeddings_2d

if TYPE_CHECKING:
    from scripts.ssl_loss_debugger.training.context import DebugContext


def visualization_scenario_name(context: DebugContext, scenario: str) -> str:
    if scenario in REGULARIZER_SCENARIOS:
        scenario = f"{context.args.supervised_loss}_{scenario}"
    if context.args.dataset == "synthetic":
        return scenario
    return f"{context.args.dataset.lower()}_{scenario}"


def visualization_scenario_dir(context: DebugContext, scenario: str) -> Path:
    output_dir = Path(
        context.args.visualization_dir
    ).expanduser() / visualization_scenario_name(context, scenario)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def graph_diagnostics_config_kwargs(context: DebugContext) -> dict[str, Any]:
    num_nodes = len(context.dataset)
    return {
        "graph_diagnostics_mode": "save" if context.args.visualize else "off",
        "graph_diagnostics_max_nodes": max(1, num_nodes),
        "graph_diagnostics_max_edges": max(1, num_nodes * (num_nodes - 1) // 2),
        "graph_diagnostics_max_labels": num_nodes,
        "graph_diagnostics_layout": context.args.visualization_layout,
    }


def extract_debug_embeddings(
    context: DebugContext,
    model: nn.Module | None = None,
    description: str = "debug visualization embeddings",
) -> np.ndarray:
    positions = np.arange(len(context.dataset), dtype=np.int64)
    return extract_embeddings(
        model=context.model if model is None else model,
        dataset=context.dataset,
        positions=positions,
        device=context.device,
        batch_size=min(8, len(positions)),
        num_workers=0,
        seed=context.args.seed,
        start_method="default",
        desc=description,
    )


def criterion_proxy_embeddings(criterion: nn.Module | None) -> np.ndarray | None:
    proxies = None if criterion is None else getattr(criterion, "proxies", None)
    if proxies is None:
        return None
    values = proxies.detach().float().cpu().numpy()
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, np.finfo(np.float32).eps)


def save_embedding_spaces(
    context: DebugContext,
    scenario: str,
    spaces: dict[str, np.ndarray | torch.Tensor],
    labels: np.ndarray | torch.Tensor,
    point_ids: Iterable[Any],
    point_kinds: Iterable[str],
    *,
    proxies_by_space: dict[str, np.ndarray] | None = None,
    connect_groups: np.ndarray | torch.Tensor | None = None,
    label_caption: str = "class",
    output_stem: str = "embedding_spaces",
    title: str | None = None,
) -> Path | None:
    """Save aligned 2-D views of one or more debug embedding tensors."""

    if not context.args.visualize:
        return None
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise RuntimeError("--visualize requires matplotlib") from exc

    labels = np.asarray(
        labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels,
        dtype=np.int64,
    )
    point_ids = np.asarray(list(point_ids), dtype=object)
    point_kinds = np.asarray(list(point_kinds), dtype=object)
    connect_groups = (
        None
        if connect_groups is None
        else np.asarray(
            connect_groups.detach().cpu().numpy()
            if torch.is_tensor(connect_groups)
            else connect_groups
        )
    )
    proxies_by_space = {} if proxies_by_space is None else proxies_by_space

    sample_count = len(labels)
    if len(point_ids) != sample_count or len(point_kinds) != sample_count:
        raise ValueError("embedding plot labels, point IDs, and point kinds must align")
    if connect_groups is not None and len(connect_groups) != sample_count:
        raise ValueError("embedding plot connection groups must align with samples")
    if not spaces:
        raise ValueError("embedding plot needs at least one space")

    arrays: dict[str, np.ndarray] = {}
    all_label_values = labels.tolist()
    for name, embeddings in spaces.items():
        array = (
            embeddings.detach().float().cpu().numpy()
            if torch.is_tensor(embeddings)
            else np.asarray(embeddings, dtype=np.float32)
        )
        if array.ndim != 2 or len(array) != sample_count:
            raise ValueError(
                f"embedding space {name!r} must be a matrix aligned with samples"
            )
        arrays[name] = array
        proxies = proxies_by_space.get(name)
        if proxies is not None:
            proxies = np.asarray(proxies, dtype=np.float32)
            if proxies.ndim != 2 or proxies.shape[1] != array.shape[1]:
                raise ValueError(
                    f"proxy embeddings for {name!r} do not match its dimension"
                )
            all_label_values.extend(range(len(proxies)))

    unique_labels = sorted({int(value) for value in all_label_values})
    label_to_color_index = {value: index for index, value in enumerate(unique_labels)}
    base_cmap = plt.get_cmap("tab10" if len(unique_labels) <= 10 else "tab20")
    palette = [base_cmap(index % base_cmap.N) for index in range(len(unique_labels))]
    cmap = ListedColormap(palette)
    norm = BoundaryNorm(np.arange(len(unique_labels) + 1) - 0.5, cmap.N)
    sample_color_indices = np.asarray(
        [label_to_color_index[int(value)] for value in labels],
        dtype=np.int64,
    )

    panel_count = len(arrays)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(5.2 * panel_count, 4.8),
        squeeze=False,
    )
    axes = axes[0]
    marker_styles = {
        "labeled": ("D", 42.0, 0.95),
        "unlabeled": ("o", 38.0, 0.58),
        "view 1": ("o", 42.0, 0.82),
        "view 2": ("^", 46.0, 0.82),
        "sample": ("o", 40.0, 0.8),
    }
    for ax, (space_name, embeddings) in zip(axes, arrays.items()):
        proxies = proxies_by_space.get(space_name)
        projection_input = embeddings
        if proxies is not None:
            projection_input = np.concatenate(
                [embeddings, np.asarray(proxies, dtype=np.float32)],
                axis=0,
            )
        coordinates, projection_name = project_graph_embeddings_2d(
            projection_input,
            layout=context.args.visualization_layout,
            seed=context.args.seed,
        )
        sample_coordinates = coordinates[:sample_count]

        if connect_groups is not None:
            for group in np.unique(connect_groups):
                members = np.flatnonzero(connect_groups == group)
                if len(members) > 1:
                    ax.plot(
                        sample_coordinates[members, 0],
                        sample_coordinates[members, 1],
                        color="#7f8792",
                        linewidth=0.7,
                        alpha=0.45,
                        zorder=1,
                    )

        for kind_index, kind in enumerate(dict.fromkeys(point_kinds.tolist())):
            mask = point_kinds == kind
            marker, size, alpha = marker_styles.get(
                str(kind),
                (("o", "^", "s", "v")[kind_index % 4], 40.0, 0.8),
            )
            edgecolors = "#171717" if kind == "labeled" else "none"
            ax.scatter(
                sample_coordinates[mask, 0],
                sample_coordinates[mask, 1],
                c=sample_color_indices[mask],
                cmap=cmap,
                norm=norm,
                marker=marker,
                s=size,
                alpha=alpha,
                edgecolors=edgecolors,
                linewidths=0.45 if kind == "labeled" else 0.0,
                zorder=3,
            )

        if sample_count <= 80:
            label_offsets = (
                (4, 4),
                (-10, 4),
                (4, -11),
                (-10, -11),
                (10, 0),
                (-14, 0),
                (0, 10),
                (0, -14),
            )
            for point_index, (coordinate, point_id) in enumerate(
                zip(sample_coordinates, point_ids)
            ):
                ax.annotate(
                    str(point_id),
                    xy=(coordinate[0], coordinate[1]),
                    xytext=label_offsets[point_index % len(label_offsets)],
                    textcoords="offset points",
                    fontsize=7,
                    alpha=0.78,
                    zorder=5,
                )

        if proxies is not None:
            proxy_coordinates = coordinates[sample_count:]
            proxy_labels = np.arange(len(proxy_coordinates), dtype=np.int64)
            proxy_color_indices = np.asarray(
                [label_to_color_index[int(value)] for value in proxy_labels],
                dtype=np.int64,
            )
            ax.scatter(
                proxy_coordinates[:, 0],
                proxy_coordinates[:, 1],
                c=proxy_color_indices,
                cmap=cmap,
                norm=norm,
                marker="*",
                s=150,
                alpha=0.95,
                edgecolors="#171717",
                linewidths=0.65,
                zorder=4,
            )
            for proxy_label, coordinate in enumerate(proxy_coordinates):
                ax.annotate(
                    f"P{proxy_label}",
                    xy=(coordinate[0], coordinate[1]),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=8,
                    fontweight="bold",
                    zorder=5,
                )

        ax.set_title(space_name)
        ax.set_xlabel(f"{projection_name} 1")
        ax.set_ylabel(f"{projection_name} 2")
        ax.grid(True, linewidth=0.35, alpha=0.22)

    kind_handles = []
    for kind_index, kind in enumerate(dict.fromkeys(point_kinds.tolist())):
        marker, _, _ = marker_styles.get(
            str(kind),
            (("o", "^", "s", "v")[kind_index % 4], 40.0, 0.8),
        )
        kind_handles.append(
            Line2D(
                [],
                [],
                marker=marker,
                linestyle="none",
                markerfacecolor="#8b919a",
                markeredgecolor="#171717" if kind == "labeled" else "none",
                markersize=7,
                label=str(kind),
            )
        )
    if any(value is not None for value in proxies_by_space.values()):
        kind_handles.append(
            Line2D(
                [],
                [],
                marker="*",
                linestyle="none",
                markerfacecolor="#8b919a",
                markeredgecolor="#171717",
                markersize=10,
                label="class proxy",
            )
        )
    class_handles = []
    if len(unique_labels) <= 12:
        class_handles = [
            Patch(facecolor=palette[index], label=f"{label_caption} {value}")
            for index, value in enumerate(unique_labels)
        ]
    legend_handles = [*kind_handles, *class_handles]
    if legend_handles:
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(6, len(legend_handles)),
            frameon=False,
        )

    fig.suptitle(
        title or f"{visualization_scenario_name(context, scenario)} embedding spaces"
    )
    bottom = 0.14 if legend_handles else 0.04
    fig.tight_layout(rect=(0.0, bottom, 1.0, 0.94))
    output_path = visualization_scenario_dir(context, scenario) / f"{output_stem}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"embedding visualization: {output_path}")
    return output_path


def visualize_model_embedding_spaces(
    context: DebugContext,
    scenario: str,
    *,
    criterion: nn.Module | None = None,
    regularizer: Any | None = None,
    state: Any | None = None,
) -> Path | None:
    if not context.args.visualize:
        return None

    retrieval = extract_debug_embeddings(
        context,
        description=f"{scenario} retrieval-space visualization",
    )
    spaces: dict[str, np.ndarray] = {"retrieval f": retrieval}
    proxies_by_space: dict[str, np.ndarray] = {}
    proxies = criterion_proxy_embeddings(criterion)
    if proxies is not None:
        proxies_by_space["retrieval f"] = proxies

    regularizer_name = None if regularizer is None else regularizer.name
    if regularizer_name == "seraph" and regularizer.metric != "identity":
        # With a learnable metric the posterior scores B @ z, not z, and most of
        # the regularizer gradient lands on B; plot the space it actually reads.
        with torch.no_grad():
            projected = context.model.seraph_posterior.project(
                torch.as_tensor(retrieval, dtype=torch.float32, device=context.device)
            )
        spaces[f"seraph metric {regularizer.metric}"] = projected.cpu().numpy()

    positions = np.arange(len(context.dataset), dtype=np.int64)
    known_mask = np.isin(positions, context.split.labeled_positions)
    return save_embedding_spaces(
        context=context,
        scenario=scenario,
        spaces=spaces,
        labels=np.asarray(context.dataset.labels, dtype=np.int64),
        point_ids=positions,
        point_kinds=np.where(known_mask, "labeled", "unlabeled"),
        proxies_by_space=proxies_by_space,
        title=(
            f"{visualization_scenario_name(context, scenario)}: "
            "model spaces at the end of the debug run"
        ),
    )


def save_seraph_comparison_artifacts(
    context: DebugContext,
    comparisons: dict[str, Any],
    supervised_inputs: torch.Tensor,
    regularizer_inputs: torch.Tensor,
    *,
    step: int,
) -> list[Path]:
    """Save an exact pair CSV and paged side-by-side input contact sheets."""

    if not context.args.visualize or not comparisons:
        return []

    output_dir = visualization_scenario_dir(context, "seraph")
    lookup = {
        (sample["role"], int(sample["batch_index"])): sample
        for samples in comparisons["samples"].values()
        for sample in samples
    }

    def endpoint(pair: dict[str, Any], side: str) -> dict[str, Any]:
        return lookup[(pair[f"{side}_role"], int(pair[f"{side}_index"]))]

    def true_label(sample: dict[str, Any]) -> int | None:
        position = sample.get("source_position")
        return None if position is None else int(context.dataset.labels[int(position)])

    # Constant for the whole step, but repeated on every row so a CSV collected
    # over several steps or runs still says which posterior produced each pair.
    quantile_threshold = comparisons["threshold_mode"] == "quantile"
    head_settings = {
        "kappa_mode": comparisons["kappa_mode"],
        "configured_kappa": comparisons["kappa"],
        "threshold_mode": comparisons["threshold_mode"],
        # Both are inert outside quantile mode, where eta is simply the
        # configured threshold.
        "positive_pair_prior": (
            comparisons["positive_pair_prior"] if quantile_threshold else None
        ),
        "threshold_momentum": (
            comparisons["threshold_momentum"] if quantile_threshold else None
        ),
    }

    rows = []
    visual_pairs = []
    for pair in comparisons["loss_pairs"]:
        left = endpoint(pair, "left")
        right = endpoint(pair, "right")
        left_true = true_label(left)
        right_true = true_label(right)
        debug_relation = (
            None
            if left_true is None or right_true is None
            else ("same" if left_true == right_true else "different")
        )
        rows.append(
            {
                "pair_kind": pair["kind"],
                "selection": "+".join(pair["selection"]),
                "left_role": left["role"],
                "left_batch_index": left["batch_index"],
                "left_source_position": left["source_position"],
                "left_mapped_label": left["mapped_label"],
                "left_debug_true_label": left_true,
                "right_role": right["role"],
                "right_batch_index": right["batch_index"],
                "right_source_position": right["source_position"],
                "right_mapped_label": right["mapped_label"],
                "right_debug_true_label": right_true,
                "debug_true_relation": debug_relation,
                "distance_rank": pair["distance_rank"],
                "block_pair_count": pair["block_pair_count"],
                "same_label_rank": pair["same_label_rank"],
                "same_label_pair_count": pair["same_label_pair_count"],
                "eta": pair["eta"],
                "kappa": pair["kappa"],
                "squared_distance": pair["squared_distance"],
                "logit": pair["logit"],
                "same_probability": pair["same_probability"],
                "em_target": pair["target"],
                "entropy": pair["entropy"],
                "pair_term": pair["pair_term"],
                "loss_contribution": pair["loss_contribution"],
                **head_settings,
            }
        )
        visual_pairs.append((pair, left, right, debug_relation))

    csv_path = output_dir / f"step_{step:03d}_seraph_comparisons.csv"
    fieldnames = list(rows[0]) if rows else []
    if fieldnames:
        with csv_path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"SERAPH comparison CSV: {csv_path}")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("--visualize requires matplotlib") from exc

    image_streams = {
        "labeled": supervised_inputs.detach().float().cpu(),
        "unlabeled": regularizer_inputs.detach().float().cpu(),
    }
    # A cached-feature run compares 768-d rows, so the batch holds no pixels. The
    # images behind those rows are still what the pairs mean, and only the traced
    # endpoints have to be decoded to show them.
    display_images = getattr(context.dataset, "display_image_dataset", None)
    batch_holds_images = all(stream.ndim == 4 for stream in image_streams.values())

    def display_image(sample: dict[str, Any]) -> np.ndarray:
        if batch_holds_images:
            image = image_streams[sample["role"]][int(sample["batch_index"])]
        elif display_images is None or sample["source_position"] is None:
            raise ValueError(
                "SERAPH comparison visualization needs either image batches or a "
                "dataset position to read the compared image back from"
            )
        else:
            image = display_images[int(sample["source_position"])][0].float()
        if image.ndim != 3:
            raise ValueError(
                "SERAPH comparison visualization expects CHW image tensors"
            )
        array = image.permute(1, 2, 0).numpy()
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)
        minimum = float(array.min())
        maximum = float(array.max())
        if minimum < 0.0 or maximum > 1.0:
            scale = maximum - minimum
            array = (array - minimum) / scale if scale > 0 else np.zeros_like(array)
        return np.clip(array, 0.0, 1.0)

    def short_endpoint(sample: dict[str, Any]) -> str:
        prefix = "L" if sample["role"] == "labeled" else "U"
        return f"{prefix}{sample['batch_index']} (pos {sample['source_position']})"

    outputs = [csv_path] if fieldnames else []
    # One sheet per selection reason, because mixing them would hide what makes
    # the same-label pairs worth looking at: a distance-spread sheet is a random
    # sample of the block, where true positives are roughly 1 in num_classes.
    inputs = (
        "actual augmented comparison inputs"
        if batch_holds_images
        else "images behind the compared cached features"
    )
    sheets = (
        ("", inputs, DISTANCE_SPREAD_SELECTION),
        (
            "_same_label",
            f"{inputs}, restricted to pairs sharing a debug true label",
            SAME_LABEL_SELECTION,
        ),
    )
    # The posterior settings the per-pair eta and kappa above came out of. In
    # fixed-threshold mode the prior and momentum are never consulted, so naming
    # them would suggest they explain an eta they had no part in.
    posterior_settings = (
        f"kappa={head_settings['configured_kappa']:.4g} "
        f"({head_settings['kappa_mode']}), "
        f"threshold={comparisons['threshold_mode']}"
        + (
            ""
            if not quantile_threshold
            else (
                f" (prior={head_settings['positive_pair_prior']:.4g}, "
                f"momentum={head_settings['threshold_momentum']:.4g})"
            )
        )
    )
    pairs_per_page = 12
    for name_suffix, description, selection in sheets:
        sheet_pairs = sorted(
            (entry for entry in visual_pairs if selection in entry[0]["selection"]),
            key=lambda entry: entry[0]["distance_rank"],
        )
        for page_start in range(0, len(sheet_pairs), pairs_per_page):
            page = sheet_pairs[page_start : page_start + pairs_per_page]
            columns = min(3, len(page))
            row_count = int(np.ceil(len(page) / columns))
            fig, axes = plt.subplots(
                row_count,
                columns,
                figsize=(5.1 * columns, 3.6 * row_count),
                squeeze=False,
            )
            for ax in axes.reshape(-1):
                ax.axis("off")
            for ax, (pair, left, right, debug_relation) in zip(
                axes.reshape(-1),
                page,
            ):
                left_image = display_image(left)
                right_image = display_image(right)
                separator = np.ones((left_image.shape[0], 3, left_image.shape[2]))
                ax.imshow(np.concatenate([left_image, separator, right_image], axis=1))
                same_label_rank = (
                    ""
                    if pair["same_label_rank"] is None
                    else (
                        f", same-label rank {pair['same_label_rank']}"
                        f"/{pair['same_label_pair_count']}"
                    )
                )
                ax.set_title(
                    f"{pair['kind'].upper()} pair, d² rank "
                    f"{pair['distance_rank']}/{pair['block_pair_count']}"
                    f"{same_label_rank}: "
                    f"{short_endpoint(left)} ↔ {short_endpoint(right)}\n"
                    # Split across lines rather than run on: a page holding one
                    # or two pairs is only as wide as its images, and a single
                    # line of this would overrun into its neighbour.
                    f"debug truth={debug_relation} | "
                    # Both halves of the logit kappa * (eta - d²): eta says which
                    # side of the boundary the pair is on, kappa how hard.
                    f"d²={pair['squared_distance']:.5g}  "
                    f"eta={pair['eta']:.4g}  kappa={pair['kappa']:.4g}\n"
                    f"p(same)={pair['same_probability']:.4f}  "
                    f"H={pair['entropy']:.4f}",
                    fontsize=8,
                )
            page_number = page_start // pairs_per_page + 1
            page_count = int(np.ceil(len(sheet_pairs) / pairs_per_page))
            fig.suptitle(
                f"SERAPH step {step}: {description} (page {page_number}/{page_count})"
                f"\n{posterior_settings}"
            )
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
            page_suffix = "" if page_count == 1 else f"_page_{page_number:02d}"
            image_path = output_dir / (
                f"step_{step:03d}_seraph_comparisons{name_suffix}{page_suffix}.png"
            )
            fig.savefig(image_path, dpi=180, bbox_inches="tight")
            plt.close(fig)
            outputs.append(image_path)
            print(f"SERAPH comparison contact sheet: {image_path}")
    return outputs
