"""Summarize --log_head_dynamics runs: does the head's effective step shrink, and does weight decay change that?

For every run directory under the given roots (one per ablation variant), reads each fold's
``diagnostics.csv`` (category ``head_dynamics``) and ``metrics.csv`` (validation MAP@R per epoch)
and reports, averaged over folds:

- ``||W||`` at the first epoch, at the best validation epoch, and at the last epoch;
- the effective step ``||dW||/||W||`` per step, averaged over the first, middle and last thirds of
  training, and its last/first ratio (how far it shrank);
- the best epoch, the stopping epoch, the epoch at which validation first reached 99 % of its best
  (the plateau onset), and the best validation MAP@R;
- test MAP@R (fold mean and concatenated embedding) when the CV summary records it.

``--curves`` writes the per-epoch series of every fold to one long CSV for plotting.

    python scripts/analyze_head_dynamics.py logs/phase_2/weight_decay/cars --curves curves.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def fold_frame(fold_dir):
    diagnostics = pd.read_csv(fold_dir / "diagnostics.csv")
    head = diagnostics[diagnostics["category"] == "head_dynamics"]
    if head.empty:
        return None
    wide = head.pivot_table(index="epoch", columns="name", values="value", aggfunc="last")
    wide.columns = [column.replace("head_dynamics/", "") for column in wide.columns]
    metrics = pd.read_csv(fold_dir / "metrics.csv")
    valid = metrics[metrics["split"] == "valid"].groupby("epoch")["mean_average_precision_at_r"].last()
    frame = wide[["weight_norm", "relative_update", "relative_update_per_step"]].join(
        valid.rename("valid_map_at_r"), how="outer"
    )
    return frame.sort_index()


def thirds_mean(series):
    values = series.dropna().to_numpy()
    if len(values) < 3:
        return [float("nan")] * 3
    return [float(chunk.mean()) for chunk in np.array_split(values, 3)]


def summarize_fold(frame):
    valid = frame["valid_map_at_r"].dropna()
    best_epoch = int(valid.idxmax())
    best = float(valid.max())
    plateau_epoch = int(valid[valid >= 0.99 * best].index.min())
    weight_norm = frame["weight_norm"].dropna()
    step = frame["relative_update_per_step"].dropna()
    early, middle, late = thirds_mean(step)
    return {
        "best_valid": best,
        "best_epoch": best_epoch,
        "plateau_epoch_99pct": plateau_epoch,
        "last_epoch": int(frame.index.max()),
        "W_first": float(weight_norm.iloc[0]),
        "W_at_best": float(weight_norm.loc[:best_epoch].iloc[-1]),
        "W_last": float(weight_norm.iloc[-1]),
        "step_early": early,
        "step_middle": middle,
        "step_late": late,
        "step_late_over_early": late / early if early > 0 else float("nan"),
    }


def test_scores(cv_dir):
    """Fold-mean and concatenated-embedding test MAP@R from the CV summary, when it has them."""

    summary_path = cv_dir / "cv_summary.json"
    if not summary_path.exists():
        return None, None
    aggregate = json.loads(summary_path.read_text()).get("aggregate") or {}
    return (
        aggregate.get("test_mean_average_precision_at_r"),
        aggregate.get("concatenated_test_mean_average_precision_at_r"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="directories holding one run directory per variant")
    parser.add_argument("--curves", type=Path, default=None, help="write per-epoch series to this CSV")
    args = parser.parse_args()

    rows, curves = [], []
    for root in args.roots:
        for run_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("_")):
            for cv_dir in sorted(run_dir.glob("**/cv_*")):
                fold_rows = []
                for diagnostics_path in sorted(cv_dir.glob("fold_*/**/diagnostics.csv")):
                    frame = fold_frame(diagnostics_path.parent)
                    if frame is None or frame["valid_map_at_r"].dropna().empty:
                        continue
                    fold = diagnostics_path.relative_to(cv_dir).parts[0]
                    fold_rows.append(summarize_fold(frame))
                    curves.append(frame.assign(variant=run_dir.name, fold=fold).reset_index())
                if not fold_rows:
                    continue
                summary = pd.DataFrame(fold_rows).mean(numeric_only=True).to_dict()
                test, concatenated = test_scores(cv_dir)
                summary.update(
                    variant=run_dir.name, folds=len(fold_rows), test_map_at_r=test, test_concat_map_at_r=concatenated
                )
                rows.append(summary)

    if not rows:
        raise SystemExit("no head_dynamics diagnostics found")
    columns = [
        "variant", "folds", "best_valid", "test_map_at_r", "test_concat_map_at_r", "best_epoch", "plateau_epoch_99pct", "last_epoch",
        "W_first", "W_at_best", "W_last", "step_early", "step_middle", "step_late", "step_late_over_early",
    ]
    table = pd.DataFrame(rows)[columns]
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.4g}".format):
        print(table.to_string(index=False))
    if args.curves is not None:
        pd.concat(curves).to_csv(args.curves, index=False)
        print(f"curves -> {args.curves}")


if __name__ == "__main__":
    main()
