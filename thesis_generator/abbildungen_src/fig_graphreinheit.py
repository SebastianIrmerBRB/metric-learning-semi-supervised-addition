"""Reinheit des kNN-Graphen von LP und LRML vor dem Training und am per Validierung gewählten Stand
(Geometrieläufe, Seed 2, Mittel über vier Folds) bei den Budgets der Phase 2 (Cars196 k=16, In-Shop k=2) und auf
Semi-iNat-2021 bei k=2. Reinheit: Anteil der Kanten zwischen Bildern derselben Klasse, gemessen mit Nachbarzahl und
Gewichtung des jeweiligen Mechanismus (Abschnitt sec:m:geometrie). Daten: Sichten geometrie/epochen_*.csv und
geometrie/laeufe_*.csv des Datenpakets. Ausgabe: bilder/neu/graphreinheit.pdf und generiert/geometrie/graphreinheit.csv.
Abbildung fig:e:reinheit, seit 25.10.2026 in der Arbeit (review/ergebnisse_notizen_2026-10-25)."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
import datensammlung as sammlung
import geodaten
import stil

stil.apply()
PANELS = [("cars", 16, r"Cars196 ($k=\mathrm{16}$)"), ("inshop", 2, r"In-Shop ($k=\mathrm{2}$)"),
          ("semiinat", 2, r"Semi-iNat-2021 ($k=\mathrm{2}$)")]
METHODS = [("sup", "ÜBl."), ("iscen", "LP"), ("lrml", "LRML")]  # ÜBl.: ein Graph mit zehn Nachbarn, Mittel der Verlustfunktionen
LOSSES = stil.LOSS_ORDER


def purity(ds, k):
    E, R, _ = geodaten.load(ds)
    E, R = E[E.k == k], R[R.k == k]
    sel = geodaten.at_selected(E, R)
    head = E[E.epoch == -1]
    keys = ["method", "loss", "fold"]
    m = sel[keys + ["graph_edge_purity", "graph_n_neighbors", "selected_epoch"]].merge(
        head[keys + ["graph_edge_purity"]], on=keys, suffixes=("_sel", "_head"), validate="one_to_one")
    g = m.groupby(["method", "loss"]).agg(head=("graph_edge_purity_head", "mean"), sel=("graph_edge_purity_sel", "mean"),
                                          nn=("graph_n_neighbors", "first"), folds=("fold", "nunique"))
    g[["head", "sel"]] *= 100
    return g.reset_index().assign(dataset=ds, k=k)


T = pd.concat([purity(ds, k) for ds, k, _ in PANELS], ignore_index=True)
assert (T.folds == 4).all(), T[T.folds != 4]
sammlung.write_csv(T, "geometrie/graphreinheit.csv", index=False)
SPREAD = T[T.method == "sup"].groupby("dataset").sel.agg(lambda v: v.max() - v.min())  # Spanne von ÜBl. über die Verlustfunktionen

# Zeilen von oben nach unten: ÜBl. (Mittel), LP und LRML (je fünf Verlustfunktionen), Gruppen durch Lücken getrennt
ypos, labels = {}, []
y = 0.0
for m, name in METHODS:
    labels.append((y, name, True))
    y += 1
    for loss in (["mittel"] if m == "sup" else LOSSES):
        ypos[(m, loss)] = y
        labels.append((y, "Mittel" if loss == "mittel" else stil.LOSS_LABELS[loss], False))
        y += 1
    y += 0.4

fig, axes = plt.subplots(1, 3, figsize=(stil.TEXTWIDTH, 3.75), sharey=True)
for ax, (ds, k, title), letter in zip(axes, PANELS, "abc"):
    t = T[T.dataset == ds]
    for (m, loss), yy in ypos.items():
        if loss == "mittel":
            r = t[t.method == m][["head", "sel", "nn"]].mean()
            col, mk = stil.NEUTRAL, "o"
        else:
            r = t[(t.method == m) & (t.loss == loss)].iloc[0]
            col, mk = stil.LOSS_COLORS[loss], stil.LOSS_MARKERS[loss]
        ax.plot([r["head"], r["sel"]], [yy, yy], color=col, lw=1.2, zorder=1)
        ax.plot(r["head"], yy, marker=mk, ms=5, mfc="white", mec=col, mew=1.0, ls="none", zorder=2)
        ax.plot(r["sel"], yy, marker=mk, ms=5, color=col, mec=stil.BLACK, mew=0.5, ls="none", zorder=3)
        ax.text(106, yy, f"{int(r['nn'])}", ha="center", va="center", fontsize=7.5)
    ax.text(106, -0.15, r"$k_{\mathrm{NN}}$", ha="center", va="bottom", fontsize=7.5)
    ax.axvline(100, color=stil.LIGHT, lw=0.8, zorder=0)
    ax.set_xlim(-3, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("Kanten innerhalb\neiner Klasse in %")
    stil.panel(ax, letter, title)
axes[0].set_ylim(y - 0.4 + 0.3, -0.9)
axes[0].set_yticks([p for p, _, _ in labels], [n for _, n, _ in labels])
for tick, (_, _, head) in zip(axes[0].get_yticklabels(), labels):
    if head:
        tick.set_fontstyle("italic")
for ax in axes:
    ax.tick_params(axis="y", length=0)
fig.legend([Line2D([], [], marker="o", ms=5, mfc="white", mec=stil.NEUTRAL, ls="none"),
            Line2D([], [], marker="o", ms=5, color=stil.NEUTRAL, mec=stil.BLACK, mew=0.5, ls="none")],
           ["vor dem Training", "nach dem Training"], loc="outside upper center", ncol=2)
stil.save(fig, "graphreinheit")
plt.close(fig)
print(T[T.method.isin(["sup", "iscen", "lrml"])].round(1).to_string())
print("Spanne ÜBl. am gewählten Stand:", SPREAD.round(1).to_dict())
