"""Read what the finished HPO studies already say about ``n_neighbors``.

Several studies searched k past 50 (75 or 100) before the phase-1 spaces settled
on a 50 cap, so the question "did a larger k ever pay off" has evidence lying in
the Optuna databases. This reads it out honestly.

Reporting the best trial per k bin - the obvious thing to do - is not honest.
Every trial redraws lr, gamma, alpha, the loss parameters and the batch sampler,
so a bin's maximum is largely max-of-n luck over 5 to 20 confounded draws, and
the bin with the most trials wins by construction. This instead reports, per
study:

``rank correlation``
    Spearman rho between k and the objective over completed trials, with a
    permutation p-value. Negative means larger k scored worse.
``k > 50 vs k <= 50``
    A Mann-Whitney U test between the two populations, plus the median
    difference. This is the direct form of the question being asked.
``top-decile composition``
    Which k values appear among the study's best trials, against how often they
    were drawn. A k that is drawn 20% of the time and holds 20% of the top decile
    is doing nothing.

The confound remains in all three: this is observational data, not an ablation.
Use it to decide whether a controlled sweep is worth running, not as the answer.

Examples::

    python scripts/analyze_n_neighbors_trials.py
    python scripts/analyze_n_neighbors_trials.py --filter iscen --min-trials 40
"""

import argparse
import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# The study directory encodes the label budget, which is how the datasets are
# told apart: Cars196 runs 16-shot, DeepFashionInShop 2-shot, CIFAR100 24-shot.
K_SHOT_DATASET = {"2": "in_shop", "16": "cars", "24": "cifar", "99999": "full"}

TRIAL_QUERY = """
SELECT t.trial_id, t.state, tv.value, tp.param_name, tp.param_value, tp.distribution_json
FROM trials t
LEFT JOIN trial_values tv ON tv.trial_id = t.trial_id
LEFT JOIN trial_params tp ON tp.trial_id = t.trial_id
"""


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=["logs", "exporter"],
        help="Directories to scan for optuna_study.db files",
    )
    parser.add_argument(
        "--filter",
        default=None,
        help="Only report studies whose path contains this substring",
    )
    parser.add_argument("--min-trials", type=int, default=20)
    parser.add_argument("--split-at", type=int, default=50, help="k boundary to test across")
    parser.add_argument("--permutations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "scripts/reports/bench_graph_neighborhood_size/hpo_trials.json"),
    )
    return parser.parse_args()


def read_study(path):
    """Return ``(k, objective)`` for every completed trial that searched k."""

    try:
        connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        rows = connection.execute(TRIAL_QUERY).fetchall()
        connection.close()
    except sqlite3.Error:
        return []

    trials = defaultdict(dict)
    for trial_id, state, value, name, param_value, distribution in rows:
        trial = trials[trial_id]
        trial["state"] = state
        trial["value"] = value
        if name is not None and "n_neighbors" in name:
            trial["k"] = decode_categorical(param_value, distribution)

    return [
        (trial["k"], float(trial["value"]))
        for trial in trials.values()
        if trial.get("k") is not None
        and trial.get("state") == "COMPLETE"
        and trial.get("value") is not None
    ]


def decode_categorical(param_value, distribution_json):
    """Optuna stores a categorical as an index into its choices list."""

    try:
        choices = json.loads(distribution_json)["attributes"]["choices"]
    except (TypeError, ValueError, KeyError):
        return float(param_value)
    return choices[int(param_value)]


def permutation_spearman(k_values, objectives, permutations, seed):
    """Spearman rho with a permutation p-value.

    The k ladder is coarse and heavily tied, which is exactly where the analytic
    Spearman p-value is least trustworthy, so the null is sampled instead.
    """

    rho = stats.spearmanr(k_values, objectives).statistic
    if not np.isfinite(rho):
        return float("nan"), float("nan")
    generator = np.random.default_rng(seed)
    shuffled = np.array(objectives, dtype=float)
    null = np.empty(permutations)
    for index in range(permutations):
        generator.shuffle(shuffled)
        null[index] = stats.spearmanr(k_values, shuffled).statistic
    return float(rho), float((np.abs(null) >= abs(rho)).mean())


def summarize(path, trials, args):
    k_values = np.array([k for k, _ in trials], dtype=float)
    objectives = np.array([value for _, value in trials], dtype=float)
    rho, p_value = permutation_spearman(k_values, objectives, args.permutations, args.seed)

    above = objectives[k_values > args.split_at]
    below = objectives[k_values <= args.split_at]
    split = {"n_above": int(len(above)), "n_below": int(len(below))}
    if len(above) >= 3 and len(below) >= 3:
        test = stats.mannwhitneyu(above, below, alternative="two-sided")
        split.update(
            {
                "median_above": float(np.median(above)),
                "median_below": float(np.median(below)),
                "median_delta": float(np.median(above) - np.median(below)),
                "mannwhitney_p": float(test.pvalue),
            }
        )

    decile = max(3, int(round(0.1 * len(objectives))))
    top_indices = np.argsort(objectives)[-decile:]
    top_k, top_counts = np.unique(k_values[top_indices], return_counts=True)
    all_k, all_counts = np.unique(k_values, return_counts=True)
    drawn = dict(zip(all_k.tolist(), all_counts.tolist()))
    composition = {
        str(int(value)): {
            "top_share": float(count / decile),
            "drawn_share": float(drawn[value] / len(k_values)),
        }
        for value, count in zip(top_k.tolist(), top_counts.tolist())
    }

    return {
        "path": str(path),
        "trials": int(len(objectives)),
        "k_values": sorted(int(value) for value in drawn),
        "spearman_rho": rho,
        "spearman_p": p_value,
        "split": split,
        "top_decile": composition,
    }


def study_label(path):
    """Name a study by the run group it belongs to, not just its leaf directory.

    Several run groups reuse the same leaf name (``iscen_arc`` exists under
    ``ssl/in_shop``, ``ssl/test_lr`` and ``ssl/in_shop_all_gpu``), so the leaf
    alone would collapse distinct studies into one row.
    """

    text = str(path)
    parts = Path(text).parts
    group = "/".join(parts[max(0, len(parts) - 6):len(parts) - 4]) or text
    for token, name in K_SHOT_DATASET.items():
        if f"_k_{token}_" in text:
            return group, name
    return group, "unknown"


def main():
    args = parse_args()
    paths = sorted(
        {
            path
            for root in args.roots
            for path in Path(root).rglob("optuna_study.db")
        }
    )
    reports = []
    for path in paths:
        if args.filter and args.filter not in str(path):
            continue
        trials = read_study(path)
        if len(trials) < args.min_trials:
            continue
        if len({k for k, _ in trials}) < 2:
            continue
        report = summarize(path, trials, args)
        name, dataset = study_label(path)
        report["study"] = name
        report["dataset"] = dataset
        reports.append(report)

    header = (
        f"{"study":<34} {'data':<8} {'n':>4} {'k range':>12} "
        f"{'rho':>7} {'p':>7} {'n>50':>5} {'med>50':>8} {'med<=50':>8} {'delta':>8} {'MW p':>7}"
    )
    print(header)
    print("-" * len(header))
    for report in sorted(reports, key=lambda item: (item["dataset"], item["study"])):
        split = report["split"]
        k_range = f"{report['k_values'][0]}-{report['k_values'][-1]}"
        print(
            f"{report["study"][-34:]:<34} {report['dataset']:<8} {report['trials']:>4} "
            f"{k_range:>12} {report['spearman_rho']:>7.3f} {report['spearman_p']:>7.4f} "
            f"{split['n_above']:>5} "
            f"{split.get('median_above', float('nan')):>8.4f} "
            f"{split.get('median_below', float('nan')):>8.4f} "
            f"{split.get('median_delta', float('nan')):>8.4f} "
            f"{split.get('mannwhitney_p', float('nan')):>7.4f}"
        )

    with_split = [item for item in reports if "mannwhitney_p" in item["split"]]
    if with_split:
        deltas = np.array([item["split"]["median_delta"] for item in with_split])
        print(
            f"\nStudies that actually drew k > {args.split_at}: {len(with_split)}; "
            f"median delta {np.median(deltas):+.4f}, "
            f"{int((deltas > 0).sum())} positive / {int((deltas < 0).sum())} negative, "
            f"{int(sum(item['split']['mannwhitney_p'] < 0.05 for item in with_split))} "
            "with p < 0.05"
        )
    rhos = np.array([item["spearman_rho"] for item in reports])
    print(
        f"All {len(reports)} studies: median Spearman rho {np.median(rhos):+.3f}, "
        f"{int((rhos < 0).sum())} negative / {int((rhos > 0).sum())} positive, "
        f"{int(sum(item['spearman_p'] < 0.05 for item in reports))} with permutation p < 0.05"
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(reports, indent=2))
    print(f"\nWrote {output_path}")


if __name__ == "__main__":
    main()
