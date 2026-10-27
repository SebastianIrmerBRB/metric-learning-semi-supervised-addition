"""Collect diagnostic runs and create standalone CSV/PNG artifacts."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "logs/diagnostics/inat_lrml"
OUT = ROOT / "scripts/reports/inat_lrml_diagnosis"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    for path in sorted(RUNS.rglob("metrics.csv")):
        args = json.loads((path.parent / "run_config.json").read_text())["args"]
        valid = pd.read_csv(path).query("split == 'valid'")
        if len(valid) < 2:
            continue
        relative = path.relative_to(RUNS)
        tag, dataset, loss, seed, fold, arm = relative.parts[:6]
        geometry = json.loads((path.parent / "lrml_geometry.json").read_text())
        diagnostics = pd.read_csv(path.parent / "diagnostics.csv")
        gc = diagnostics[diagnostics.name.str.startswith("train/gradient_contribution/")].groupby("name").value.mean()
        record = dict(tag=tag, dataset=dataset, loss=loss, seed=seed, fold=fold, arm=arm,
                      lr=args["lr"], k_shot=args["ssl_config_resolved"]["labeled_per_class"],
                      epochs=len(valid) - 1, steps=int(valid.step.iloc[-1]),
                      initial_map=float(valid.mean_average_precision_at_r.iloc[0]),
                      final_map=float(valid.mean_average_precision_at_r.iloc[-1]),
                      best_map=float(valid.mean_average_precision_at_r.max()),
                      initial_p1=float(valid.precision_at_1.iloc[0]),
                      final_p1=float(valid.precision_at_1.iloc[-1]),
                      mean_gradient_ratio=gc.get("train/gradient_contribution/regularizer_to_supervised_ratio"),
                      mean_gradient_cosine=gc.get("train/gradient_contribution/cosine_similarity"),
                      initial_graph_purity=geometry[0].get("graph_purity"),
                      final_graph_purity=geometry[-1].get("graph_purity"),
                      final_centroid_norm=geometry[-1].get("centroid_norm") if dataset != "Cars196" else None,
                      final_centered_rank=geometry[-1].get("centered_effective_rank") if dataset != "Cars196" else None,
                      log_dir=str(path.parent.relative_to(ROOT)))
        rows.append(record)
        curves[tag, dataset, arm] = (valid, geometry)
    pd.DataFrame(rows).to_csv(OUT / "runs.csv", index=False)

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.3), constrained_layout=True)
    styles = {"baseline": ("LRML, target 1", "#d1495b"),
              "supervised": ("Supervised", "#197278"),
              "weak": ("LRML, target 0.1", "#4c78a8"),
              "oracle": ("LRML, same-species oracle", "#b279a2")}
    for index, tag in enumerate(("initial", "faster_lr")):
        for arm, (label, color) in styles.items():
            data = curves.get((tag, "iNat", arm))
            if data is None:
                continue
            valid, _ = data
            axes[index].plot(valid.epoch + 1, 100 * valid.mean_average_precision_at_r,
                             label=label, color=color, linewidth=2, marker="o", markersize=3)
        axes[index].set(title=f"iNat: learning rate {'1e-5' if index == 0 else '1e-4'}",
                        xlabel="Completed epochs", ylabel="Validation mAP@R (%)", ylim=(25, 52))
        axes[index].grid(alpha=0.18)
        axes[index].legend(fontsize=8, frameon=False, loc="lower left")
    for arm in ("baseline", "supervised", "oracle"):
        data = curves.get(("faster_lr", "iNat", arm))
        if data is None:
            continue
        _, geometry = data
        label, color = styles[arm]
        axes[2].plot([g["epoch"] + 1 for g in geometry],
                     [g["centered_effective_rank"] for g in geometry],
                     label=label, color=color, linewidth=2, marker="o", markersize=3)
    axes[2].set(title="iNat: loss of embedding dimensions (1e-4)",
                xlabel="Completed epochs", ylabel="Centered effective rank")
    axes[2].grid(alpha=0.18)
    axes[2].legend(fontsize=8, frameon=False)
    figure.savefig(OUT / "curves.png", dpi=180)
    figure.savefig(OUT / "curves.pdf")
    print(pd.DataFrame(rows)[["tag", "dataset", "arm", "epochs", "initial_map", "final_map"]].to_string(index=False))


if __name__ == "__main__":
    main()
