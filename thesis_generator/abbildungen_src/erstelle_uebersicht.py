"""Write an Excel-friendly summary of all 450 Phase-1 experiment settings."""
from pathlib import Path
import pandas as pd
import datensammlung as data


def overview():
    all_results = pd.read_csv(data.DATA / "results.csv")
    all_folds = pd.read_csv(data.DATA / "folds.csv", low_memory=False)
    rows = []
    for ds, r in all_results.groupby("dataset_key", sort=False):
        b = "budget_percent" if ds == "inshop10" else "k"
        r["loss"] = r.method.str.rsplit("_", n=1).str[-1]
        base = r[r.method.str.startswith("sup_")].set_index([b, "seed", "scope", "loss"])
        for (method, budget), x in r.groupby(["method", b], sort=True):
            mean, concat = x[x.scope == "mean"], x[x.scope == "concat"]
            f = all_folds[all_folds.run_id.isin(x.run_id)]
            loss = method.rsplit("_", 1)[-1]
            gain = {}
            for scope in ("mean", "concat"):
                g = x[x.scope == scope].set_index([b, "seed", "scope", "loss"])
                gain[scope] = 100 * (g.mean_average_precision_at_r - base.mean_average_precision_at_r.reindex(g.index)).mean()
            rows.append(dict(dataset=ds, method=method.rsplit("_", 1)[0], loss=loss,
                budget_type="percent_classes" if ds == "inshop10" else "images_per_class", budget=int(budget),
                seeds=mean.seed.nunique(), folds=len(f),
                mean_map_percent=100 * mean.mean_average_precision_at_r.mean(),
                mean_map_seed_sd_pp=100 * mean.mean_average_precision_at_r.std(ddof=1),
                concat_map_percent=100 * concat.mean_average_precision_at_r.mean(),
                concat_map_seed_sd_pp=100 * concat.mean_average_precision_at_r.std(ddof=1),
                paired_mean_gain_pp=gain["mean"], paired_concat_gain_pp=gain["concat"],
                selected_epoch_median=f.selected_epoch.median(),
                trained_epochs_median=f.epochs_trained.median(),
                trained_epochs_min=f.epochs_trained.min(), trained_epochs_max=f.epochs_trained.max(),
                config_ids=" | ".join(sorted(x.config_id.unique())),
                source_table_ids=" | ".join(sorted(x.source_table_id.unique()))))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p = Path(data.output_dir("")) / "uebersicht.csv"
    overview().to_csv(p, index=False, float_format="%.8g")
    print(p)
