#!/usr/bin/env python3
"""Reproduce pooled Phase-0 thesis plots from validation trials only.

Four methods per dataset/loss: supervised, noncontrastive LRML, Hoffer, Iscen.
Cars LRML reads lrml__initial_target_ratio: the sorted lrml__initial campaign
is contrastive and must not silently substitute for the requested baseline.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

LOSSES = {
    "arcface": ("ArcFaceLoss", "ArcFace"),
    "circle": ("CircleLoss", "Circle"),
    "multi_similarity": ("MultiSimilarityLoss", "Multi-Similarity"),
    "ntxent": ("NTXentLoss", "NT-Xent"),
    "proxy_anchor": ("ProxyAnchorLoss", "Proxy-Anchor"),
}
DATASETS = {"cars196": ("Cars196", "#2864a0", "o"),
            "in_shop": ("In-Shop", "#c17624", "D")}
METHODS = ("supervised", "lrml", "hoffer", "iscen")
# Phase-1 bounds as documented in kapitel/anhang.tex, tab:hyperparameter-spaces.
# Proxy-Anchor m keeps its bounds (sampling uniform -> log). Circle m keeps them
# too, but its top bin holds no trial within 1 PP, so it is no clean example.
PANELS = [
    ("arcface", "scale", "ArcFace: Skalierung $s$", (0.1, 80),
     "[10; 250] → [0,1; 80]", True),
    ("ntxent", "temperature", r"NT-Xent: Temperatur $\tau$", (0.01, 0.4),
     "[0,001; 0,5] → [0,01; 0,4]", True),
    ("proxy_anchor", "margin", "Proxy-Anchor: Margin $m$", (0.01, 1.0),
     "[0,01; 1] → unveränderte Grenzen", False),
    ("multi_similarity", "beta", r"Multi-Similarity: $\beta$", (1, 280),
     "[20; 160] → [1; 280]", True),
]
XLABELS = {"scale": "$s$ (logarithmisch)", "temperature": r"$\tau$ (logarithmisch)",
           "margin": "$m$ (linear wie in Phase 0)", "beta": r"$\beta$ (logarithmisch)"}


def extract(workspace, output):
    records, manifest = [], []
    archive = workspace / "sorted_logs/phase_0"
    for ds in DATASETS:
        for loss, (cls, _) in LOSSES.items():
            for method in METHODS:
                if method == "supervised":
                    directory = archive / "supervised" / ds / loss
                elif method == "lrml" and ds == "cars196":
                    directory = archive / "ssl" / ds / "lrml__initial_target_ratio" / loss
                else:
                    directory = archive / "ssl" / ds / (method + "__initial") / loss
                source = directory / "trials.jsonl"
                config_path = directory / "study_config.json"
                config = json.loads(config_path.read_text())
                args, hpo = config["base_args"], config["hparam_config"]
                assert hpo["sampler"] == "random" and hpo["direction"] == "maximize"
                assert hpo["metric"] == "best_valid_mean_average_precision_at_r"
                assert args["loss"] == cls and args["cv_k"] == 4 and args["seed"] == 7
                trials = [json.loads(line) for line in source.read_text().splitlines() if line]
                assert len({t["number"] for t in trials}) == len(trials)
                complete = [t for t in trials if t["state"] == "COMPLETE"
                            and t["value"] is not None and np.isfinite(t["value"])]
                best = max(t["value"] for t in complete)
                identity = f"{ds}/{method}/{loss}"
                manifest.append({
                    "study": identity, "source": str(source),
                    "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                    "study_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                    "dataset": ds, "method": method, "loss": loss,
                    "k_shot_grid": args.get("k_shot_grid"),
                    "states": dict(Counter(t["state"] for t in trials)),
                    "trials": len(trials), "complete": len(complete),
                    "best": best, "spaces": hpo["spaces"],
                })
                for trial in trials:
                    attrs = trial["user_attrs"]
                    ssl = attrs.get("resolved_ssl_config")
                    if ssl is None:
                        run = next(directory.glob(f"trial_{trial['number']:04d}/cv_*/fold_00/run_config.json"), None)
                        ssl = json.loads(run.read_text()).get("ssl_config") if run else None
                    if method != "supervised" and trial in complete:
                        assert ssl, (identity, trial["number"], "missing method evidence")
                        params = ssl.get("method_params", {})
                        if method == "lrml":
                            assert params.get("regularizer") == "lrml"
                            assert not params.get("regularizer_params", {}).get("contrastive", False)
                        elif method == "hoffer":
                            assert params.get("regularizer") == "hoffer_entropy"
                        else:
                            assert ssl["method"] == "iscen_label_spreading"
                    folds = attrs.get("fold_results") or []
                    if trial in complete:
                        fold_values = [f["best_valid_mean_average_precision_at_r"] for f in folds]
                        assert len(fold_values) == 4, (identity, trial["number"])
                        assert np.isclose(np.mean(fold_values), trial["value"], atol=1e-10, rtol=0)
                    records.append({
                        "study": identity, "dataset": ds, "method": method, "loss": loss,
                        "number": trial["number"], "state": trial["state"], "value": trial["value"],
                        "study_best": best,
                        "shortfall_pp": 100 * (trial["value"] - best) if trial in complete else None,
                        **trial["params"],
                    })
    frame = pd.DataFrame(records)
    assert not frame.duplicated(["study", "number"]).any()
    assert len(manifest) == 40
    frame.to_csv(output / "data/trials.csv", index=False)
    (output / "data/sources.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return frame, manifest


def above_axes(ax, points):
    """Data x, axes top plus a fixed offset in points, independent of the panel height."""
    return ax.get_xaxis_transform() + ScaledTranslation(0, points / 72, ax.figure.dpi_scale_trans)


def plot_panel(ax, data, param, searched, retained, log, title, subtitle):
    """All finite completed trials; values below -6 PP remain visible as triangles."""
    complete = data[data.shortfall_pp.notna()]
    assert complete[param].notna().all()
    low, high = searched
    xmin, xmax = min(low, retained[0]), max(high, retained[1])
    axis = np.log10 if log else np.asarray
    inverse = (lambda x: 10 ** x) if log else (lambda x: x)
    amin, amax = axis(np.array([xmin, xmax]))
    padding = (amax - amin) * 0.025
    ax.set_xlim(inverse(amin - padding), inverse(amax + padding))
    if log:
        ax.set_xscale("log")
    ax.set_ylim(-6.35, 0.45)
    ax.set_yticks([-6, -4, -2, -1, 0])
    ax.axhspan(-1, 0, color="#edbc55", alpha=0.12, zorder=0)
    ax.axhline(-1, color="#af831d", linestyle=(0, (4, 3)), linewidth=0.9, zorder=1)
    ax.axhline(0, color="#60727c", linewidth=0.7, zorder=1)
    # Original sampling extent remains explicit; extensions contain no observations.
    for left, right in [(xmin, low), (high, xmax)]:
        if right > left:
            ax.axvspan(left, right, facecolor="#f0f1f2", edgecolor="#d4d9dc",
                       hatch="///", linewidth=0, zorder=0)
    for edge in searched:
        ax.axvline(edge, color="#c0c8cc", linestyle=":", linewidth=0.8, zorder=0)
    for ds, (_, color, marker) in DATASETS.items():
        group = complete[complete.dataset == ds]
        clipped = group.shortfall_pp < -6
        ax.scatter(group.loc[~clipped, param], group.loc[~clipped, "shortfall_pp"],
                   color=color, marker=marker, s=11, alpha=0.6, edgecolors="none", zorder=3)
        ax.scatter(group.loc[clipped, param], np.full(clipped.sum(), -6.13),
                   color=color, marker="v", s=12, alpha=0.5, edgecolors="none", zorder=3)
    # Ten bins on Phase-0 sampling scale, as specified in the thesis.
    edges = np.linspace(*axis(np.array(searched)), 11)
    x = axis(complete[param].to_numpy())
    indices = np.clip(np.searchsorted(edges, x, side="right") - 1, 0, 9)
    medians, diagnostics = [], []
    for i in range(10):
        group = complete.iloc[np.flatnonzero(indices == i)]
        median = float(group.shortfall_pp.median()) if len(group) else None
        medians.append(max(-6, median) if median is not None else np.nan)
        diagnostics.append({"bin": i, "low": float(inverse(edges[i])),
                            "high": float(inverse(edges[i+1])), "n": len(group),
                            "in_band": int((group.shortfall_pp >= -1).sum()),
                            "median_shortfall_pp": median})
    ax.plot(inverse((edges[:-1] + edges[1:]) / 2), medians, color="#3e4b52",
            marker=".", markersize=4, linewidth=1.0, alpha=0.9, zorder=4)
    # Edge criterion from kapitel/methodik.tex: >= 3 of a study's top 10 in an outer bin.
    edge_studies = {"low": 0, "high": 0}
    for _, study in complete.assign(bin=indices).groupby("study"):
        top = study.nlargest(10, "value")
        edge_studies["low"] += int((top.bin == 0).sum() >= 3)
        edge_studies["high"] += int((top.bin == 9).sum() >= 3)
    # Range bars and counts sit above the axes, clear of the points and median line.
    for interval, offset, color in [(searched, 3.5, "#7b858c"), (retained, 8.5, "#168777")]:
        ax.plot(interval, [1, 1], transform=above_axes(ax, offset),
                color=color, linewidth=2.4, solid_capstyle="butt", clip_on=False)
    clipped_count = int((complete.shortfall_pp < -6).sum())
    ax.annotate(f"n = {len(complete)} · ▼ {clipped_count} unter −6 PP", xy=(0, 1),
                xycoords="axes fraction", xytext=(0, 13), textcoords="offset points",
                ha="left", va="bottom", fontsize=6.4, color="#5b686e", annotation_clip=False)
    ax.set_title(title + "\n" + subtitle, loc="left", fontsize=8.4, pad=24, linespacing=1.45)
    ax.grid(axis="y", alpha=0.16, linewidth=0.7)
    return {"parameter": param, "n": len(complete), "clipped": clipped_count,
            "n_by_dataset": {ds: int((complete.dataset == ds).sum()) for ds in DATASETS},
            "studies": int(complete.study.nunique()), "top10_edge_studies": edge_studies,
            "phase0": list(searched), "phase1": list(retained), "bins": diagnostics}


def legend_handles():
    handles = [Line2D([], [], marker=marker, color=color, linestyle="none", markersize=4,
                      label=name) for name, color, marker in DATASETS.values()]
    handles += [Line2D([], [], marker="v", color="#7b858c", linestyle="none", markersize=4,
                       label="unter −6 PP (am Rand)"),
                Line2D([], [], color="#3e4b52", linewidth=1, label="Median je Bin"),
                Line2D([], [], color="#af831d", linestyle="--", label="−1 PP"),
                Line2D([], [], color="#7b858c", linewidth=2.4, label="Suchraum Phase 0"),
                Line2D([], [], color="#168777", linewidth=2.4, label="Suchraum Phase 1"),
                Patch(facecolor="#f0f1f2", hatch="///", edgecolor="#d4d9dc", label="nicht untersucht")]
    return handles


def render(frame, manifest, output):
    plt.rcParams.update({"font.size": 7.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.8,
                         "ytick.labelsize": 6.8, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": "#9aa5aa", "text.color": "#26353c",
                         "axes.labelcolor": "#26353c"})
    audit = {}
    # 6.3 in is the thesis text width (A4, 2.5 cm margins), so fonts print at true size.
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.6))
    for i, (ax, (loss, suffix, title, retained, subtitle, log)) in enumerate(zip(axes.flat, PANELS)):
        param = f"loss.{LOSSES[loss][0]}.{suffix}"
        spaces = [m["spaces"][param] for m in manifest if m["loss"] == loss]
        assert all(s == spaces[0] for s in spaces), (loss, spaces)
        space = spaces[0]
        assert space["log"] == log, (param, space)
        audit[param] = plot_panel(ax, frame[frame.loss == loss], param,
                                 (space["low"], space["high"]), retained, log,
                                 f"({chr(97+i)}) {title}", subtitle)
        ax.set_xlabel(XLABELS[suffix])
    for ax in axes[:, 0]:
        ax.set_ylabel("Shortfall Validierungs-MAP@R [PP]")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.87, bottom=0.17, hspace=0.78, wspace=0.26)
    fig.legend(handles=legend_handles(), loc="lower center", ncol=4, frameon=False,
               fontsize=6.8, bbox_to_anchor=(0.54, 0.0), columnspacing=1.2)
    save(fig, output, "phase0_lossparameter")

    fig, axes = plt.subplots(3, 2, figsize=(6.3, 7.2))
    for i, (ax, (loss, (_, title))) in enumerate(zip(axes.flat, LOSSES.items())):
        spaces = [m["spaces"]["lr"] for m in manifest if m["loss"] == loss]
        assert all(s == spaces[0] for s in spaces)
        assert (spaces[0]["low"], spaces[0]["high"]) == (1e-5, 10), spaces[0]
        retained = (1e-5, 10) if loss in ("arcface", "proxy_anchor") else (1e-5, 5e-4)
        subtitle = "Phase 1: [10⁻⁵; 10]" if retained[1] == 10 else "Phase 1: [10⁻⁵; 5 × 10⁻⁴]"
        audit[loss + "/lr"] = plot_panel(ax, frame[frame.loss == loss], "lr", (1e-5, 10),
                                         retained, True, f"({chr(97+i)}) {title}", subtitle)
        ax.set_xlabel("Lernrate (logarithmisch)")
        ax.set_xticks([1e-5, 1e-3, 1e-1, 10])
    for ax in axes[:, 0]:
        ax.set_ylabel("Shortfall [PP]")
    axes[2, 1].axis("off")
    axes[2, 1].legend(handles=legend_handles(), loc="center left", frameon=False,
                      fontsize=7.2, borderaxespad=0, handlelength=2.2, labelspacing=0.75)
    fig.subplots_adjust(left=0.1, right=0.98, top=0.9, bottom=0.07, hspace=0.8, wspace=0.26)
    save(fig, output, "phase0_lernraten")
    (output / "data/plot_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(frame.groupby(["dataset", "loss"]).agg(archived=("number", "size"), plotted=("shortfall_pp", "count")))


def save(fig, output, name):
    fig.savefig(output / "figures" / (name + ".pdf"), bbox_inches="tight")
    fig.savefig(output / "figures" / (name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for subdir in ["data", "figures"]:
        (args.output / subdir).mkdir(parents=True, exist_ok=True)
    frame, manifest = extract(args.workspace, args.output)
    render(frame, manifest, args.output)


if __name__ == "__main__":
    main()
