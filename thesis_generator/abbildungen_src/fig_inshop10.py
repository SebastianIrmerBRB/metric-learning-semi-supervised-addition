"""In-Shop mit vollständig gelabelten Klassen: gepaarter MAP@R-Gewinn ggü. Sup über den Anteil gelabelter Klassen."""
import datensammlung as sammlung
import pandas as pd
import matplotlib.pyplot as plt
import stil

stil.apply()
R = ""
pdf = sammlung.read_csv(R + "inshop10_ergebnisse/data/phase1_paired_differences.csv")
pdf = pdf[pdf.metric == "mean_average_precision_at_r"].copy()
pdf["fam"] = pdf.method.str.split("_").str[0]
pdf["loss"] = pdf.method.str.split("_").str[1]
# beste Einzelkonfiguration der Serie: größter über die vier Anteile gemittelter Gewinn der Einzelmodelle (Mean)
BEST = pdf[pdf.scope == "mean"].groupby("method")["mean"].mean().idxmax()
BEST_FAM, BEST_LOSS = BEST.split("_")
fig, axes = plt.subplots(1, 2, figsize=(stil.TEXTWIDTH, 2.9), sharey=True)
ps = [5, 10, 25, 50]
for ax, letter, scope, title in zip(axes, "ab", ["mean", "concat"], ["Mean (Einzelmodell)", "Concat (Ensemble)"]):
    for fi, fam in enumerate(["iscen", "lrml", "hoffer", "slade"]):
        d = pdf[(pdf.scope == scope) & (pdf.fam == fam)]
        # pooled over all paired comparisons (weights = seeds per loss), as in tab:e:klassen; bars: range over losses
        w = d.assign(gew=d["mean"] * d["n_seeds"]).groupby("budget_percent")
        g = pd.DataFrame({"mean": w.gew.sum() / w.n_seeds.sum(), "min": w["mean"].min(), "max": w["mean"].max()}).reindex(ps)
        x = [p * (2 ** ((fi - 1.5) * 0.06)) for p in ps]
        ax.vlines(x, g["min"], g["max"], color=stil.METHOD_COLORS[fam], lw=0.8)
        ax.plot(x, g["mean"], color=stil.METHOD_COLORS[fam], marker=stil.METHOD_MARKERS[fam],
                label=stil.METHOD_LABELS[fam])
    d = pdf[(pdf.scope == scope) & (pdf.method == BEST)].set_index("budget_percent").reindex(ps)
    ax.plot(ps, d["mean"], color=stil.METHOD_COLORS[BEST_FAM], ls=(0, (1, 1.5)), marker="none",
            label=f"{stil.METHOD_LABELS[BEST_FAM]} mit {stil.LOSS_LABELS[BEST_LOSS]}")
    ax.axhline(0, color=stil.BLACK, lw=0.8, ls=(0, (4, 2)))
    ax.set_xscale("log")
    ax.set_xticks(ps, [str(p) for p in ps])
    ax.minorticks_off()
    ax.set_xlim(4.3, 58)
    stil.panel(ax, letter, title)
axes[0].set_ylabel("Differenz in Prozentpunkten")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=5)
fig.supxlabel("Vollständig gelabelte Trainingsklassen in %", fontsize=10)
stil.save(fig, "inshop10_gewinn")
