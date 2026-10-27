#!/usr/bin/env python3
"""Hyperparameter sensitivity from the Phase-0/1 trials, without fANOVA.

fANOVA marginalizes a surrogate uniformly over the whole search space, but TPE
samples densely where it expects good values, so outside that region the
surrogate extrapolates (Moosbauer et al. 2021 show the same bias for partial
dependence on BO samples). This script therefore separates what each part of
the data can support:

* **Random trials** (all of Phase 0, the first ``tpe_startup_trials`` of every
  TPE study) are an unbiased sample of the space: Spearman correlation and
  binned medians are computed on them only. The spread of the bin medians is
  compared with a permutation floor (scores shuffled among the same trials),
  because with ~8 trials per bin it is positive even for an inert parameter.
* **All trials** say where good configurations were found: the top-gamma
  trials against the rest (the l(x)/g(x) split TPE itself uses), the best trial
  per bin (Phase-0 shortfall logic), and how far TPE narrowed each parameter
  after the random phase.
* **Plateau size**: how many trials are within 1 PP / one standard error of the
  best, i.e. whether the exact selected values mean anything at all.
* **Consistency (A4)**: the selected value of every parameter across losses,
  methods and datasets, with the top-5 range as the plateau width.
* ``--ped-anova`` adds Optuna's PED-ANOVA importances, once on the random trials
  and once on all trials, so the two rankings can be compared.

Outputs below ``--out/hpo``: trials.csv, param_stats.csv, study_stats.csv,
best_params.csv, response_shape.csv (random-trial score quantile per bin, for
pooling the shape of a response over studies),
figures/<study>/{shortfall,trajectory}.png, consistency_*.png.

Examples::

    python main_eval/hpo_sensitivity.py                      # Phase 0 + Phase 1, everything
    python main_eval/hpo_sensitivity.py --phase 1 --dataset Cars196 --method lp
    python main_eval/hpo_sensitivity.py --roots sorted_logs/phase_0/ssl/cars196 --no-figures
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

PP = 100.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", nargs="+", type=Path, default=[common.PHASE0_ROOT, common.PHASE1_ROOT])
    parser.add_argument("--study-list", type=Path, default=None,
                        help="text file with one study dir per line; replaces the --roots walk (filters still apply)")
    parser.add_argument("--phase", type=int, choices=(0, 1), default=None)
    parser.add_argument("--dataset", default=None, help="dataset name as recorded, e.g. Cars196, CUB, DeepFashionInShop")
    parser.add_argument("--method", nargs="+", default=None, help="sup lp lrml hoffer slade ...")
    parser.add_argument("--loss", nargs="+", default=None, help="full loss names, e.g. MultiSimilarityLoss")
    parser.add_argument("--path-contains", default=None, help="keep only studies whose path contains this text")
    parser.add_argument("--min-trials", type=int, default=30, help="skip studies with fewer COMPLETE trials")
    parser.add_argument("--gamma", type=float, default=0.2, help="top share of trials treated as 'good'")
    parser.add_argument("--bins", type=int, default=5, help="bins per parameter on the sampling scale")
    parser.add_argument("--late-tpe", type=int, default=20, help="last N TPE trials used for the concentration measure")
    parser.add_argument("--permutations", type=int, default=500,
                        help="shuffles for the noise floor of the bin-median range (0 disables it)")
    parser.add_argument("--consistency-phase", type=int, default=1, help="which phase the consistency figure uses")
    parser.add_argument("--ped-anova", action="store_true", help="also compute PED-ANOVA importances (needs optuna_study.db)")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--out", type=Path, default=common.DEFAULT_OUT)
    return parser.parse_args()


def load_listed_studies(list_path, roots, min_trials):
    """Load the studies named in ``list_path``; the root is the --roots entry each one lies under."""
    studies = []
    for line in Path(list_path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        path = Path(line)
        # outside every root: key the study by its full path, so such studies stay distinct
        root = next((Path(r) for r in roots if path.is_relative_to(Path(r))), Path(path.anchor))
        study = common.load_study(root, path)
        if len(study.completed()) >= min_trials:
            studies.append(study)
        else:
            print(f"[skip] {path}: {len(study.completed())} complete trials < {min_trials}", file=sys.stderr)
    return studies


def keep(study, args):
    if args.phase is not None and study.phase != args.phase:
        return False
    if args.dataset and study.dataset != args.dataset:
        return False
    if args.method and study.method not in args.method:
        return False
    if args.loss and study.loss not in args.loss:
        return False
    if args.path_contains and args.path_contains not in str(study.path):
        return False
    return True


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")[:150]


def is_numeric_space(space):
    return space is not None and space.get("type") in ("float", "int")


def iqr(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    q75, q25 = np.percentile(values, [75, 25])
    return float(q75 - q25)


def bin_median_range(scores, bins, n_bins):
    """Largest minus smallest per-bin median in PP, over bins with at least two trials."""
    medians = [np.median(scores[bins == b]) for b in range(n_bins) if (bins == b).sum() >= 2]
    return float((max(medians) - min(medians)) * PP) if len(medians) >= 2 else float("nan")


def trial_rows(study):
    rows = []
    for trial in study.completed():
        folds = common.fold_values(trial)
        attrs = trial.get("user_attrs") or {}
        epochs = [f.get("selected_epoch") for f in attrs.get("fold_results") or [] if f.get("selected_epoch") is not None]
        row = {
            "study_id": study.study_id,
            "label": study.label,
            "phase": study.phase,
            "dataset": study.dataset,
            "method": study.method,
            "loss": study.loss,
            "number": int(trial["number"]),
            "is_random": study.is_random(trial),
            "value": float(trial["value"]),
            "fold_std": float(np.std(folds, ddof=1)) if len(folds) > 1 else float("nan"),
            "n_folds": len(folds),
            "selected_epoch_mean": float(np.mean(epochs)) if epochs else float("nan"),
        }
        for name, value in (trial.get("params") or {}).items():
            row[f"p:{name}"] = value
        rows.append(row)
    return rows


def numeric_param_stats(study, name, space, rows, args):
    values = np.array([common.to_unit(r.get(f"p:{name}"), space) for r in rows], dtype=float)
    scores = np.array([r["value"] for r in rows], dtype=float)
    random_mask = np.array([r["is_random"] for r in rows], dtype=bool)
    numbers = np.array([r["number"] for r in rows])
    present = np.isfinite(values)
    best = scores[present].max()
    out = {
        "study_id": study.study_id,
        "label": study.label,
        "phase": study.phase,
        "dataset": study.dataset,
        "method": study.method,
        "loss": study.loss,
        "param": name,
        "short": common.short_param(name),
        "kind": "numeric",
        "scale": common.space_scale(space),
        "low": space.get("low"),
        "high": space.get("high"),
        "n": int(present.sum()),
        "n_random": int((present & random_mask).sum()),
    }

    # Unbiased part: random trials only.
    r_mask = present & random_mask
    if r_mask.sum() >= 5 and np.ptp(values[r_mask]) > 0:
        rho, p_value = stats.spearmanr(values[r_mask], scores[r_mask])
        out["spearman_random"] = float(rho)
        out["spearman_random_p"] = float(p_value)
        edges = np.linspace(0, 1, args.bins + 1)
        bins = np.clip(np.digitize(values[r_mask], edges[1:-1]), 0, args.bins - 1)
        observed = bin_median_range(scores[r_mask], bins, args.bins)
        if np.isfinite(observed):
            out["random_bin_median_range_pp"] = observed
            # The range is positive even without any effect, because each bin holds only a few trials whose
            # spread comes from all other parameters. Shuffling the scores keeps that spread and the binning
            # but breaks the link to this parameter; the seed makes each row reproducible on its own.
            if args.permutations > 0:
                rng = np.random.default_rng(0)
                null = np.array([bin_median_range(rng.permutation(scores[r_mask]), bins, args.bins)
                                 for _ in range(args.permutations)])
                null = null[np.isfinite(null)]
                out["random_bin_median_range_null_pp"] = float(np.median(null))
                # exact Monte-Carlo p-value (b + 1) / (m + 1), Phipson & Smyth (2010)
                out["random_bin_median_range_perm_p"] = float((np.sum(null >= observed) + 1) / (len(null) + 1))

    # Where good trials were found: all trials.
    edges = np.linspace(0, 1, args.bins + 1)
    bins = np.clip(np.digitize(values[present], edges[1:-1]), 0, args.bins - 1)
    bin_best = [scores[present][bins == b].max() for b in range(args.bins) if (bins == b).any()]
    out["bins_covered"] = len(bin_best)
    out["worst_bin_best_shortfall_pp"] = float((min(bin_best) - best) * PP) if bin_best else float("nan")

    threshold = np.quantile(scores[present], 1 - args.gamma)
    top = present & (scores >= threshold)
    rest = present & ~top
    out["top_n"] = int(top.sum())
    out["top_median_unit"] = float(np.median(values[top]))
    out["top_iqr_ratio"] = iqr(values[top]) / max(iqr(values[present]), 1e-12)
    if top.sum() >= 3 and rest.sum() >= 3:
        ks = stats.ks_2samp(values[top], values[rest])
        out["ks_top_vs_rest"] = float(ks.statistic)
        out["ks_top_vs_rest_p"] = float(ks.pvalue)

    tpe = present & ~random_mask
    if tpe.sum() >= 5:
        late = tpe & (numbers >= np.sort(numbers[tpe])[-min(args.late_tpe, tpe.sum())])
        out["tpe_late_iqr_unit"] = iqr(values[late])
        out["random_iqr_unit"] = iqr(values[present & random_mask])
        out["tpe_concentration"] = out["tpe_late_iqr_unit"] / max(out["random_iqr_unit"], 1e-12)

    order = np.argsort(-scores[present])
    top5 = values[present][order[:5]]
    best_unit = float(values[present][order[0]])
    out["best_unit"] = best_unit
    out["best_raw"] = rows[int(np.flatnonzero(present)[order[0]])].get(f"p:{name}")
    out["top5_min_unit"] = float(np.min(top5))
    out["top5_max_unit"] = float(np.max(top5))
    out["best_at_boundary"] = bool(best_unit < 0.1 or best_unit > 0.9)
    return out


def response_shape_rows(study, name, space, rows, args):
    """Median within-study score quantile of the random trials per bin, so studies can be pooled per parameter."""
    random_rows = [r for r in rows if r["is_random"]]
    if len(random_rows) < 5:
        return []
    values = np.array([common.to_unit(r.get(f"p:{name}"), space) for r in random_rows], dtype=float)
    scores = np.array([r["value"] for r in random_rows], dtype=float)
    quantiles = (stats.rankdata(scores) - 0.5) / len(scores)
    present = np.isfinite(values)
    bins = np.clip(np.digitize(values, np.linspace(0, 1, args.bins + 1)[1:-1]), 0, args.bins - 1)
    return [
        {"study_id": study.study_id, "phase": study.phase, "dataset": study.dataset, "method": study.method,
         "loss": study.loss, "param": name, "short": common.short_param(name), "bin": b,
         "n": int((present & (bins == b)).sum()), "median_quantile": float(np.median(quantiles[present & (bins == b)]))}
        for b in range(args.bins) if (present & (bins == b)).sum() >= 2
    ]


def categorical_param_stats(study, name, space, rows, args):
    raw = [r.get(f"p:{name}") for r in rows]
    scores = np.array([r["value"] for r in rows], dtype=float)
    random_mask = np.array([r["is_random"] for r in rows], dtype=bool)
    present = np.array([v is not None for v in raw])
    choices = sorted({str(v) for v in raw if v is not None})
    best = scores[present].max()
    threshold = np.quantile(scores[present], 1 - args.gamma)
    out = {
        "study_id": study.study_id,
        "label": study.label,
        "phase": study.phase,
        "dataset": study.dataset,
        "method": study.method,
        "loss": study.loss,
        "param": name,
        "short": common.short_param(name),
        "kind": "categorical",
        "scale": "categorical",
        "n": int(present.sum()),
        "n_random": int((present & random_mask).sum()),
    }
    groups = []
    detail = {}
    for choice in choices:
        mask = present & np.array([str(v) == choice for v in raw])
        r_mask = mask & random_mask
        if r_mask.sum() >= 2:
            groups.append(scores[r_mask])
        top_share = float(np.mean(scores[mask] >= threshold)) if mask.any() else float("nan")
        detail[choice] = {
            "n": int(mask.sum()),
            "median_random_pp": float(np.median(scores[r_mask]) * PP) if r_mask.any() else None,
            "best_shortfall_pp": float((scores[mask].max() - best) * PP) if mask.any() else None,
            "share_in_top": top_share,
        }
    if len(groups) >= 2:
        out["kruskal_random_p"] = float(stats.kruskal(*groups).pvalue)
    shortfalls = [d["best_shortfall_pp"] for d in detail.values() if d["best_shortfall_pp"] is not None]
    out["worst_bin_best_shortfall_pp"] = float(min(shortfalls)) if shortfalls else float("nan")
    order = np.argsort(-scores[present])
    out["best_raw"] = [v for v in raw if v is not None][order[0]]
    out["top5_values"] = sorted({str([v for v in raw if v is not None][i]) for i in order[:5]})
    out["per_choice"] = detail
    return out


def study_stats(study, rows):
    scores = np.array([r["value"] for r in rows])
    order = np.argsort(-scores)
    best_row = rows[order[0]]
    se = best_row["fold_std"] / math.sqrt(best_row["n_folds"]) if best_row["n_folds"] > 1 else float("nan")
    return {
        "study_id": study.study_id,
        "label": study.label,
        "phase": study.phase,
        "dataset": study.dataset,
        "method": study.method,
        "loss": study.loss,
        "sampler": study.sampler,
        "n_complete": len(rows),
        "n_random": int(sum(r["is_random"] for r in rows)),
        "best_number": best_row["number"],
        "best_is_random": best_row["is_random"],
        "best_value_pct": best_row["value"] * PP,
        "best_fold_std_pp": best_row["fold_std"] * PP,
        "best_fold_se_pp": se * PP,
        "second_gap_pp": float((scores[order[0]] - scores[order[1]]) * PP) if len(scores) > 1 else float("nan"),
        "n_within_1pp": int((scores >= scores[order[0]] - 0.01).sum()),
        "n_within_1se": int((scores >= scores[order[0]] - se).sum()) if np.isfinite(se) else None,
        "random_best_shortfall_pp": float(
            (max((r["value"] for r in rows if r["is_random"]), default=float("nan")) - scores[order[0]]) * PP
        ),
        "path": str(study.path),
    }


def draw_study(study, rows, params, out_dir, args):
    pyplot = common.plt()
    names = [n for n in params if any(r.get(f"p:{n}") is not None for r in rows)]
    if not names:
        return
    scores = np.array([r["value"] for r in rows])
    best = scores.max()
    threshold = np.quantile(scores, 1 - args.gamma)
    numbers = np.array([r["number"] for r in rows])
    random_mask = np.array([r["is_random"] for r in rows])
    cols = min(4, len(names))
    nrows = math.ceil(len(names) / cols)

    fig, axes = pyplot.subplots(nrows, cols, figsize=(3.3 * cols, 2.9 * nrows), squeeze=False)
    fig.subplots_adjust(hspace=0.6, wspace=0.3)
    for ax, name in zip(axes.flat, names):
        space = study.spaces.get(name)
        raw = [r.get(f"p:{name}") for r in rows]
        mask = np.array([v is not None for v in raw])
        shortfall = (scores - best) * PP
        if is_numeric_space(space):
            x = np.array([float(v) if v is not None else np.nan for v in raw])
            if space.get("log"):
                ax.set_xscale("log")
            ax.axvspan(space["low"], space["high"], color="0.95", zorder=0)
        else:
            choices = sorted({str(v) for v in raw if v is not None})
            jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(raw))
            x = np.array([choices.index(str(v)) + j if v is not None else np.nan for v, j in zip(raw, jitter)])
            ax.set_xticks(range(len(choices)), choices, rotation=30, fontsize=7)
        ax.scatter(x[mask & random_mask], shortfall[mask & random_mask], s=12, c="0.6", label="random")
        tpe = mask & ~random_mask
        if tpe.any():
            ax.scatter(x[tpe], shortfall[tpe], s=12, c=numbers[tpe], cmap="viridis", label="TPE")
        top = mask & (scores >= threshold)
        ax.scatter(x[top], shortfall[top], s=40, facecolors="none", edgecolors="tab:red", linewidths=0.8)
        best_index = int(np.argmax(scores))
        ax.scatter([x[best_index]], [0], marker="*", s=120, c="tab:red", zorder=5)
        ax.axhline(-1.0, color="0.4", lw=0.7, ls="--")
        ax.set_title(common.short_param(name), fontsize=8)
        ax.set_ylim(max(shortfall.min() - 0.5, -15), 0.8)
    for ax in axes.flat[len(names):]:
        ax.axis("off")
    axes.flat[0].set_ylabel("MAP@R − best [PP]")
    fig.suptitle(f"{study.label} · {study.study_id}\nred ring: top {args.gamma:.0%} · dashed: −1 PP", fontsize=8)
    common.savefig(fig, out_dir / "shortfall.png")

    numeric = [n for n in names if is_numeric_space(study.spaces.get(n))]
    if not numeric:
        return
    cols = min(4, len(numeric))
    nrows = math.ceil(len(numeric) / cols)
    fig, axes = pyplot.subplots(nrows, cols, figsize=(3.3 * cols, 2.4 * nrows), squeeze=False)
    for ax, name in zip(axes.flat, numeric):
        unit = np.array([common.to_unit(r.get(f"p:{name}"), study.spaces.get(name)) for r in rows])
        sc = ax.scatter(numbers, unit, c=scores * PP, cmap="viridis", s=12)
        if study.sampler == "tpe":
            ax.axvline(study.n_startup - 0.5, color="0.4", ls="--", lw=0.7)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(common.short_param(name), fontsize=8)
    for ax in axes.flat[len(numeric):]:
        ax.axis("off")
    fig.colorbar(sc, ax=axes, shrink=0.6, label="MAP@R [%]")
    fig.suptitle(f"{study.label}: sampled value (search-space position) over trials", fontsize=8)
    common.savefig(fig, out_dir / "trajectory.png")


def draw_consistency(best_rows, out_dir, phase):
    pyplot = common.plt()
    rows = [r for r in best_rows if r["phase"] == phase and r["kind"] == "numeric"]
    if not rows:
        return
    by_param = {}
    for row in rows:
        by_param.setdefault(row["short"], []).append(row)
    for short, items in sorted(by_param.items()):
        items.sort(key=lambda r: (r["dataset"], common.METHOD_ORDER.index(r["method"]) if r["method"] in common.METHOD_ORDER else 99, r["loss"]))
        fig, ax = pyplot.subplots(figsize=(max(4, 0.32 * len(items)), 2.8))
        x = np.arange(len(items))
        best = np.array([r["best_unit"] for r in items])
        low = np.array([r["top5_min_unit"] for r in items])
        high = np.array([r["top5_max_unit"] for r in items])
        colors = [f"C{common.METHOD_ORDER.index(r['method']) % 10}" if r["method"] in common.METHOD_ORDER else "k" for r in items]
        ax.vlines(x, low, high, colors=colors, lw=2, alpha=0.4)
        ax.scatter(x, best, c=colors, s=25, zorder=3)
        ax.set_xticks(x, [r["label"] for r in items], rotation=90, fontsize=6)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("position in search space")
        ax.set_title(f"{short}: best trial (dot) and top-5 range (bar), phase {phase}", fontsize=8)
        common.savefig(fig, out_dir / f"consistency_{slug(short)}.png")


def ped_anova(study):
    """PED-ANOVA importances on the random trials and on all trials, or {} when unavailable."""

    try:
        import optuna
        from optuna.importance import PedAnovaImportanceEvaluator, get_param_importances
    except ImportError:
        return {}
    db = study.path / "optuna_study.db"
    if not db.exists():
        return {}
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import warnings

    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    name = study.config.get("resolved_study_name")
    try:
        stored = optuna.load_study(study_name=name, storage=f"sqlite:///{db}")
    except Exception:
        return {}
    complete = [t for t in stored.get_trials(deepcopy=False) if t.state == optuna.trial.TrialState.COMPLETE]
    subsets = {"random": [t for t in complete if study.sampler != "tpe" or t.number < study.n_startup], "all": complete}
    out = {}
    for subset, trials in subsets.items():
        if len(trials) < 10:
            continue
        memory = optuna.create_study(direction=stored.direction)
        memory.add_trials(trials)
        try:
            importances = get_param_importances(memory, evaluator=PedAnovaImportanceEvaluator())
        except Exception as exc:
            print(f"[warn] PED-ANOVA failed for {study.study_id} ({subset}): {exc}", file=sys.stderr)
            continue
        for param, value in importances.items():
            out.setdefault(param, {})[f"ped_anova_{subset}"] = float(value)
    return out


def main():
    args = parse_args()
    out = args.out / "hpo"
    if args.study_list:
        studies = [s for s in load_listed_studies(args.study_list, args.roots, args.min_trials) if keep(s, args)]
    else:
        studies = [s for s in common.load_studies(args.roots, min_trials=args.min_trials) if keep(s, args)]
    print(f"{len(studies)} studies with >= {args.min_trials} complete trials")
    all_trials, param_rows, study_rows, shape_rows = [], [], [], []
    for study in studies:
        rows = trial_rows(study)
        all_trials.extend(rows)
        study_rows.append(study_stats(study, rows))
        params = sorted({k[2:] for r in rows for k in r if k.startswith("p:")})
        importances = ped_anova(study) if args.ped_anova else {}
        for name in params:
            space = study.spaces.get(name)
            if space is None:
                # Phase-0 studies written before spaces were recorded: fall back to observed range.
                observed = [r.get(f"p:{name}") for r in rows if r.get(f"p:{name}") is not None]
                if observed and all(isinstance(v, (int, float)) for v in observed):
                    positive = min(observed) > 0
                    ratio = max(observed) / min(observed) if positive else 1
                    space = {"type": "float", "low": min(observed), "high": max(observed), "log": bool(positive and ratio > 50)}
                else:
                    space = {"type": "categorical"}
                study.spaces[name] = space
            if is_numeric_space(space):
                stats_row = numeric_param_stats(study, name, space, rows, args)
                shape_rows.extend(response_shape_rows(study, name, space, rows, args))
            else:
                stats_row = categorical_param_stats(study, name, space, rows, args)
            stats_row.update(importances.get(name, {}))
            param_rows.append(stats_row)
        if not args.no_figures:
            draw_study(study, rows, params, out / "figures" / slug(study.study_id), args)

    common.write_rows(out / "trials.csv", all_trials)
    common.write_rows(out / "param_stats.csv", param_rows)
    common.write_rows(out / "study_stats.csv", study_rows)
    common.write_rows(out / "response_shape.csv", shape_rows)
    common.write_rows(
        out / "best_params.csv",
        [
            {k: r.get(k) for k in ("study_id", "label", "phase", "dataset", "method", "loss", "param", "short", "kind",
                                   "best_raw", "best_unit", "top5_min_unit", "top5_max_unit", "top5_values", "best_at_boundary")}
            for r in param_rows
        ],
    )
    if not args.no_figures:
        draw_consistency(param_rows, out, args.consistency_phase)

    # Terminal digest: the parameters that look influential on the unbiased random trials.
    ranked = sorted(
        (r for r in param_rows if r["kind"] == "numeric" and r.get("spearman_random") is not None),
        key=lambda r: -abs(r["spearman_random"]),
    )
    print("\nstrongest random-trial rank correlations:")
    for r in ranked[:15]:
        print(f"  {r['label']:<24} {r['short']:<28} rho={r['spearman_random']:+.2f} "
              f"(n={r['n_random']}), top-IQR ratio={r['top_iqr_ratio']:.2f}, worst-bin-best={r['worst_bin_best_shortfall_pp']:.2f} PP")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
