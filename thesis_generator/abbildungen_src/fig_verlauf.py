"""Trainingsverlauf bei k=4 (Hauptteil) und k=32 (Anhang), Cars196, Seed 2, alle fünf Verlustfunktionen.

verlauf_k4.pdf, verlauf_k32.pdf  Mittel über fünf Verlustfunktionen und vier Folds (20 Läufe je Verfahren), jeder Lauf
    ausgerichtet am Ende seines Warm-ups: x = Epoche - Warm-up, x = 0 ist die erste Epoche mit Mechanismus (Ü: Epoche).
    Das Warm-up dauert je nach Verlustfunktion 0 bis 30 Epochen. Gezeigt nur x, an denen alle 20 Läufe einen Wert haben
    (keine wechselnde Zusammensetzung des Mittels): von -min(Warm-up) bis min(letzte Epoche - Warm-up), ohne Epoche -1.
    (a) Validierungs-MAP@R, (b) bis (d) pi_intra der gelabelten und ungelabelten Bilder und der Validierungsklassen,
    (e) pi_inter der ungelabelten Bilder, (f) Anteil der ungelabelten Bilder mit richtigem nächstem Klassenprototyp.
verlauf_verluste.pdf  Ü und LP bei k=4 je Verlustfunktion, Epoche ohne Ausrichtung (Anhang)
verlauf_rho.pdf       spektraler Abfall des Trainingspools, ausgerichtet wie oben, k=4 und k=32 (Anhang)
pi_intra: mittlerer Kosinusabstand zwischen Bildern derselben Klasse (Referenzcode von Roth et al. 2020).
Eingabe: Sichten geometrie/epochen_cars.csv, validierung_cars.csv, laeufe_cars.csv des Datenpakets (Geometrieläufe, jede
Epoche, alle Bilder).
LP/Circle k=4 Fold 0 hat keine Geometrie in Epoche 53 bis 56 (Lücke, x = 33 bis 36 fällt für die Geometrie-Panels weg).
"""
import datensammlung as sammlung
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import stil

stil.apply()
RD = "geometrie/"
E = sammlung.read_csv(RD + "epochen_cars.csv", low_memory=False)
R = sammlung.read_csv(RD + "laeufe_cars.csv")
V = sammlung.read_csv(RD + "validierung_cars.csv")
M = stil.METHOD_ORDER
KEY = ["method", "loss", "k", "fold"]
R["warm"] = R.warmup.fillna(0).astype(int)
E = E.merge(R[KEY + ["warm", "last_epoch"]], on=KEY)
V = V.merge(R[KEY + ["warm", "last_epoch"]], on=KEY)


def curve(x, col):
    """Mittel über die Folds an den Epochen, an denen alle vier Folds einen Wert haben (ohne Ausrichtung)."""
    x = x.dropna(subset=[col])
    n = x.groupby("epoch").fold.nunique()
    g = x.groupby("epoch")[col].mean()
    return g[n == 4]


def aligned(x, col):
    """Mittel über alle Läufe (Verlustfunktion x Fold) eines Verfahrens, ausgerichtet am Ende des Warm-ups."""
    x = x[x.epoch >= 0].dropna(subset=[col]).assign(rel=lambda d: d.epoch - d.warm)
    runs = x.groupby(["loss", "fold"]).ngroups
    lo, hi = -x.warm.min(), (x.last_epoch - x.warm).min()
    x = x[(x.rel >= lo) & (x.rel <= hi)]
    n = x.drop_duplicates(["rel", "loss", "fold"]).groupby("rel").size()  # Läufe mit einem Wert je Epoche
    g = x.groupby("rel")[col].mean()
    return g[n == runs]


def draw(K, name):
    d = E[E.k == K]
    v_all = V[V.k == K]
    assert (R[R.k == K].groupby("method").fold.count() == 20).all()
    fig, axes = plt.subplots(2, 3, figsize=stil.SIZES["full2"], sharex=True)
    (a, b, c), (dd, e, f) = axes
    panels = [(b, "labeled_pi_intra_roth", "b", "Gelabelte Bilder", r"$\pi_{\mathrm{intra}}$", 1, (0.0, 0.7)),
              (c, "unlabeled_pi_intra_roth", "c", "Ungelabelte Bilder", r"$\pi_{\mathrm{intra}}$", 1, (0.0, 0.7)),
              (dd, "validation_pi_intra_roth", "d", "Validierungsklassen", r"$\pi_{\mathrm{intra}}$", 1, (0.0, 0.7)),
              (e, "unlabeled_pi_inter_roth", "e", "Ungelabelte Bilder", r"$\pi_{\mathrm{inter}}$", 1, (0.6, 1.05)),
              (f, "prototype_unlabeled_accuracy", "f", "Ungelabelte Bilder", "Richtiger Prototyp in %", 100, (35, 80))]
    left, right = 0, 0
    seen = {col: [] for _, col, *_ in panels}
    for m in M:
        kw = dict(color=stil.METHOD_COLORS[m], label=stil.METHOD_LABELS[m], marker=stil.METHOD_MARKERS[m])
        v = aligned(v_all[v_all.method == m], "valid_map") * 100
        a.plot(v.index, v.values, markevery=[len(v) - 1], **kw)
        right, left = max(right, v.index.max()), min(left, v.index.min())
        for ax, col, _, _, _, scale, _ in panels:
            g = aligned(d[d.method == m], col) * scale
            ax.plot(g.index, g.values, markevery=[len(g) - 1], **kw)
            seen[col] += [g.min(), g.max()]
    for ax in axes.flat:
        ax.axvspan(left - 1, -0.5, color=stil.LIGHT, lw=0)
    stil.panel(a, "a", "Validierung")
    a.set_ylabel("MAP@R in %")
    for ax, col, letter, title, ylab, _, lim in panels:
        stil.panel(ax, letter, title)
        ax.set_ylabel(ylab)
        # feste Grenzen für gleiche Achsen bei k=4 und k=32, aber nie enger als die Kurven (k=32: Prototyp über 80 %)
        step = 5 if lim[1] > 2 else 0.05
        lo = min(lim[0], step * np.floor(min(seen[col]) / step))
        hi = max(lim[1], step * np.ceil(max(seen[col]) / step))
        ax.set_ylim(lo, hi)
    for ax in axes[1]:
        ax.set_xlabel("Epoche nach dem Warm-up")
    right = 10 * (right // 10) + 10
    for ax in axes.flat:
        ax.set_xlim(left - 1, right)
    h, l = a.get_legend_handles_labels()
    fig.legend(h[:len(M)], l[:len(M)], loc="outside upper center", ncol=len(M))
    stil.save(fig, name)

    # -------------------------------------------------------------- Zahlen für den Text
    print(f"===== k={K}")
    for m in M:
        v = aligned(v_all[v_all.method == m], "valid_map") * 100
        out = {"x": (int(v.index.min()), int(v.index.max())), "valid max": (round(v.max(), 1), int(v.idxmax())),
               "valid end": round(v.iloc[-1], 1), "valid -1/0": (round(v.get(-1, np.nan), 1), round(v.get(0, np.nan), 1))}
        for col in ["labeled_pi_intra_roth", "unlabeled_pi_intra_roth", "validation_pi_intra_roth", "unlabeled_pi_inter_roth",
                    "validation_pi_inter_roth", "prototype_unlabeled_accuracy", "train_rho_roth"]:
            g = aligned(d[d.method == m], col)
            pts = sorted(set([int(g.index.min()), -1, 0, 1, 2, 5, 10, 20, int(g.index.max())]) & set(g.index.astype(int)))
            out[col.replace("_roth", "")] = {x: round(g.loc[x], 3) for x in pts}
            out[col.replace("_roth", "") + " min/max"] = (round(g.min(), 3), int(g.idxmin()), round(g.max(), 3), int(g.idxmax()))
        print(m, out)


def draw_losses(K=4, name="verlauf_verluste"):
    """Ü (oben) und LP (unten) mit allen fünf Verlustfunktionen: Validierungs-MAP@R, pi_intra der ungelabelten Bilder
    und der Validierungsklassen. Marker: Ende der Kurve."""
    fig, axes = plt.subplots(2, 3, figsize=stil.SIZES["full2"])
    letters = iter("abcdef")
    cols = [("valid_map", "Validierungs-MAP@R in %"), ("unlabeled_pi_intra_roth", r"$\pi_{\mathrm{intra}}$ ungelabelte Bilder"),
            ("validation_pi_intra_roth", r"$\pi_{\mathrm{intra}}$ Validierungsklassen")]
    for i, m in enumerate(["sup", "iscen"]):
        right = 0
        for j, (col, ylab) in enumerate(cols):
            ax = axes[i, j]
            werte = []
            for loss in stil.LOSS_ORDER:
                src = V if col == "valid_map" else E
                x = src[(src.method == m) & (src.loss == loss) & (src.k == K)]
                g = curve(x, col) * (100 if col == "valid_map" else 1)
                werte += list(g.values)
                w = int(R[(R.method == m) & (R.loss == loss)].warmup.iloc[0])
                ax.plot(g.index, g.values, color=stil.LOSS_COLORS[loss], marker=stil.LOSS_MARKERS[loss], markevery=[len(g) - 1],
                        label=stil.LOSS_LABELS[loss])
                right = max(right, g.index.max())
            stil.panel(ax, next(letters), stil.METHOD_LABELS[m])
            ax.set_ylabel(ylab)
            if j == 0:
                ax.set_ylim(*stil.grenzen(28, 66, werte, 2))
            else:
                ax.set_ylim(*stil.grenzen(0.1, 0.7, werte, 0.05))
            ax.set_xlabel("Epoche")
        for ax in axes[i]:
            ax.set_xlim(-4, 10 * (right // 10) + 10)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=5)
    stil.save(fig, name)


def draw_rho(name="verlauf_rho"):
    """Spektraler Abfall rho des Trainingspools, ausgerichtet am Ende des Warm-ups, k=4 und k=32, Referenz DINOv2."""
    fig, axes = plt.subplots(1, 2, figsize=stil.SIZES["full"], sharey=True)
    ref = R.bb_train_rho_roth.mean()
    for ax, letter, K in ((axes[0], "a", 4), (axes[1], "b", 32)):
        d = E[E.k == K]
        left, right = 0, 0
        for m in M:
            g = aligned(d[d.method == m], "train_rho_roth")
            ax.plot(g.index, g.values, color=stil.METHOD_COLORS[m], marker=stil.METHOD_MARKERS[m], markevery=[len(g) - 1],
                    label=stil.METHOD_LABELS[m])
            right, left = max(right, g.index.max()), min(left, g.index.min())
        ax.axhline(ref, color=stil.BLACK, lw=0.8, ls=(0, (4, 2)))
        ax.axvspan(left - 1, -0.5, color=stil.LIGHT, lw=0)
        ax.set_xlim(left - 1, 10 * (right // 10) + 10)
        ax.set_xlabel("Epoche nach dem Warm-up")
        stil.panel(ax, letter, f"$k={K}$")
    axes[0].set_ylabel(r"$\rho$ des Trainingspools")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside upper center", ncol=5)
    stil.save(fig, name)


draw(4, "verlauf_k4")
draw(32, "verlauf_k32")
draw_losses()
draw_rho()
