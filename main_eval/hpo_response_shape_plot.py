#!/usr/bin/env python3
"""Thesis figure: pooled response shape of the SSL hyperparameters (Phase 1).

Reads ``response_shape.csv`` written by ``hpo_sensitivity.py``: for every study and
numeric parameter, the median within-study rank quantile of the random trials per
fifth of the search range. This script takes the median over the studies of each
method and draws one panel per parameter, one line per method.

    python main_eval/hpo_response_shape_plot.py --out /mnt/f/.../bilder/phase2_hpo_verlauf.pdf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Fixed hue per method (validated categorical slots 1-4), plus a marker so identity never rests on colour.
METHODS = {
    "lp": ("LP", "#2a78d6", "o"),
    "lrml": ("LRML", "#eb6834", "s"),
    "hoffer": ("Hoffer", "#1baf7a", "^"),
    "slade": ("SLADE", "#eda100", "D"),
}
PANELS = [
    ("warmup_epochs", "Warm-up-Epochen", ["lp", "lrml", "hoffer", "slade"]),
    ("regularizer_target_ratio", r"Zielgradientenverhältnis $\eta$", ["lrml", "hoffer", "slade"]),
    ("n_neighbors", "Nachbarzahl $k$", ["lp", "lrml"]),
    ("distance_scale", "Distanzskala $s$ (Hoffer)", ["hoffer"]),
    ("confidence_threshold", "Konfidenzschwelle (LP)", ["lp"]),
    ("alpha", r"Diffusionsparameter $\alpha$ (LP)", ["lp"]),
    ("gamma", r"Affinitätsexponent $\gamma$ (LP)", ["lp"]),
]
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#dcdbd5"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shape", type=Path, default=Path("main_eval_out/hpo/response_shape.csv"))
    parser.add_argument("--phase", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    shape = pd.read_csv(args.shape)
    shape = shape[shape.phase == args.phase]
    # Fifths of a log range and of a linear range are different things, so a curve pools only the studies
    # that sample the parameter on its majority scale (this drops Hoffer's linear s on In-Shop 10 %).
    scales = pd.read_csv(args.shape.with_name("param_stats.csv"))[["study_id", "param", "scale"]]
    shape = shape.merge(scales, on=["study_id", "param"])
    majority = shape.groupby("short").scale.agg(lambda s: s.mode().iloc[0])
    dropped = shape[shape.scale != shape.short.map(majority)]
    for (short, method), rows in dropped.groupby(["short", "method"]):
        print(f"excluded {rows.study_id.nunique()} {method} studies from {short}: scale differs from {majority[short]}")
    shape = shape[shape.scale == shape.short.map(majority)]
    pooled = shape.groupby(["short", "method", "bin"]).median_quantile.median()
    studies = shape.groupby(["short", "method"]).study_id.nunique()

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 4, figsize=(6.3, 3.5), sharey=True)
    for ax, (param, title, methods) in zip(axes.flat, PANELS):
        ax.axhline(0.5, color=GRID, lw=1.0, zorder=0)
        for method in methods:
            name, color, marker = METHODS[method]
            y = pooled.loc[(param, method)].reindex(range(5))
            ax.plot(range(5), y.values, color=color, lw=1.6, marker=marker, ms=4.5,
                    mec="white", mew=0.6, label=f"{name} ({studies.loc[(param, method)]})")
        ax.set_title(title, fontsize=7.5, color=INK)
        ax.set_xticks(range(5), ["1", "2", "3", "4", "5"])
        ax.set_ylim(0.1, 0.8)
        ax.set_yticks([0.2, 0.35, 0.5, 0.65, 0.8], ["0,2", "0,35", "0,5", "0,65", "0,8"])
        ax.grid(axis="y", color=GRID, lw=0.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    legend_ax = axes.flat[len(PANELS)]
    legend_ax.axis("off")
    handles = [plt.Line2D([], [], color=c, marker=m, lw=1.6, ms=4.5, mec="white", mew=0.6, label=n)
               for n, c, m in METHODS.values()]
    legend_ax.legend(handles=handles, loc="center", frameon=False, fontsize=7.5, title="Verfahren", title_fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Median Rangquantil")
    for ax in axes[1]:
        ax.set_xlabel("Fünftel des Suchraums")
    fig.tight_layout(w_pad=1.2, h_pad=1.0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
