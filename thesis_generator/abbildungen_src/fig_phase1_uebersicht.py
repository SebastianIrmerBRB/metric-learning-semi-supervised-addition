"""Phase 1: gepaarter MAP@R-Gewinn gegenüber ÜBl. über das Label-Budget, Mittel über die fünf Verluste
(Linie) und Spanne über die Verluste (senkrechter Strich). 2x2 Kästen, je Kasten ein Datensatz mit
Mean (Einzelmodell) links und Concat (Ensemble) rechts auf derselben y-Achse."""
import datensammlung as sammlung
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox, ScaledTranslation
from matplotlib.ticker import MultipleLocator
import stil

stil.apply()
R = ""
DATA = [("Cars196", "cars", [4, 8, 16, 32], 2), ("CUB", "cub", [4, 8, 16, 32], 0.5),
        ("In-Shop", "inshop", [2, 4], 1), ("Semi-iNat", "semiinat", [2, 4, 16, 100], 0.5)]
FAMS = ["iscen", "lrml", "hoffer", "slade"]
# ein Raster für alle acht Felder (gleiche Breiten, bündige Kanten), mittlere Spalte als Abstand zwischen den Kästen
fig, grid = plt.subplots(2, 5, figsize=stil.SIZES["full2"], gridspec_kw={"width_ratios": [1, 1, 0.12, 1, 1]})
for ax in grid[:, 2]:
    ax.axis("off")


def ueber_paar(a0, a1, pt):
    """Koordinaten: Rahmen um beide Felder eines Kastens, um pt Punkte nach oben verschoben."""
    def f(renderer):
        b = Bbox.union([a0.get_window_extent(renderer), a1.get_window_extent(renderer)])
        return b.translated(0, renderer.points_to_pixels(pt))
    return f


letters = iter("abcdefgh")
for (name, key, ks, step), (r, c) in zip(DATA, [(0, 0), (0, 3), (1, 0), (1, 3)]):
    rand = max(0.13 * np.log2(ks[-1] / ks[0]), 0.4)  # Rand in log2-Einheiten, Platz für den äußersten Versatz
    lo, hi = ks[0] / 2 ** rand, ks[-1] * 2 ** rand
    dodge = 0.03 * np.log2(hi / lo)  # gleicher sichtbarer Versatz in jedem Kasten
    pair = grid[r, c], grid[r, c + 1]
    pair[1].sharey(pair[0])
    pair[1].tick_params(labelleft=False)
    for ax, scope, title in zip(pair, ["mean", "concat"], ["Mean", "Concat"]):
        pdf = sammlung.read_csv(R + f"{key}_ergebnisse/data/phase1_paired_differences.csv")
        pdf = pdf[(pdf.metric == "mean_average_precision_at_r") & (pdf.scope == scope)].copy()
        pdf["fam"] = pdf.method.str.split("_").str[0]
        g = pdf.groupby(["fam", "k"])["mean"].agg(["mean", "min", "max"]).reset_index()
        for fi, fam in enumerate(FAMS):
            d = g[g.fam == fam].sort_values("k")
            x = d.k * (2 ** ((fi - 1.5) * dodge))
            ax.vlines(x, d["min"], d["max"], color=stil.METHOD_COLORS[fam], lw=0.8)
            ax.plot(x, d["mean"], color=stil.METHOD_COLORS[fam], marker=stil.METHOD_MARKERS[fam],
                    label=stil.METHOD_LABELS[fam])
        ax.axhline(0, color=stil.BLACK, lw=0.8, ls=(0, (4, 2)), zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(ks, [str(k) for k in ks])
        ax.minorticks_off()
        ax.set_xlim(lo, hi)
        ax.yaxis.set_major_locator(MultipleLocator(step))
        stil.panel(ax, next(letters), title)
    # Kopf des Kastens über beiden Feldern, darunter eine Linie wie \cmidrule. Beide liegen außerhalb des Layouts,
    # sonst schiebt die Breite über zwei Felder das rechte Feld weg; die Höhe hält ein unsichtbares Rechteck frei.
    for a in (pair[0].annotate("", xy=(0, 1), xycoords=ueber_paar(*pair, 17), xytext=(1, 1),
                               textcoords=ueber_paar(*pair, 17),
                               arrowprops=dict(arrowstyle="-", lw=0.6, color=stil.BLACK, shrinkA=0, shrinkB=0)),
              pair[0].annotate(name, xy=(0.5, 1), xycoords=ueber_paar(*pair, 19), ha="center", va="bottom",
                               fontsize=10)):
        a.set_in_layout(False)
    pair[0].add_patch(Rectangle((0, 1), 0.5, 0, alpha=0, clip_on=False,
                                transform=pair[0].transAxes + ScaledTranslation(0, 30 / 72, fig.dpi_scale_trans)))
h, l = grid[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=4)
fig.supylabel("Differenz in Prozentpunkten", fontsize=10)
fig.supxlabel("Gelabelte Bilder je Klasse $k$", fontsize=10)
stil.save(fig, "phase1_uebersicht")
