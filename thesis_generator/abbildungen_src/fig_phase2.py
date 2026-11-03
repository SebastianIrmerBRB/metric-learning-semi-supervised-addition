"""Phase 2: Ablationen (Cars196 k=16 und In-Shop k=2, je mit allen fünf Verlustfunktionen, je ein Seed).
In der Arbeit: phase2_2x2_verluste, phase2_lp_qualitaet, phase2_sweeps_verluste und phase2_sweeps_verluste_inshop.
Nicht in der Arbeit (ergänzend): phase2_2x2 und phase2_sweeps (nur Circle) sowie phase2_graph (Graph-Kontrollen; die
Werte stehen in Tabelle tab:e:ablationen:verluste).
Werte: Mean-MAP@R. Daten: Sichten phase2/variants.csv, phase2/tuned.csv und phase2_lp_diagnostik.csv des Datenpakets
(gesammelt von sammle_phase2.py)."""
import datensammlung as sammlung
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import stil

stil.apply()
V = sammlung.read_csv("phase2/variants.csv")
V["variant"] = V.variant.fillna("defaults").replace("", "defaults")
TUNED = sammlung.read_csv("phase2/tuned.csv")
DS = {"Cars196": r"Cars196 ($k=\mathrm{16}$)", "DeepFashionInShop": r"In-Shop ($k=\mathrm{2}$)"}
DSKEY = {"Cars196": "cars", "DeepFashionInShop": "inshop"}
DSCOL = {k: stil.DATASET_COLORS[v] for k, v in DSKEY.items()}
DSMARK = {k: stil.DATASET_MARKERS[v] for k, v in DSKEY.items()}
REFLINE = dict(color=stil.BLACK, lw=0.8, ls=(0, (4, 2)))
CUT = "$\\downarrow$"  # Marke für Werte unterhalb der Achse (keine Verlust- oder Verfahrensmarke)
LOSSES = stil.LOSS_ORDER


def val(ds, method, variant, loss="cir"):
    r = V[(V.dataset == ds) & (V.loss == loss) & (V.method == method) & (V.variant == variant)]
    return float(r.mean_value.iloc[0]) if len(r) else np.nan


def sup(ds, loss="cir"):
    return val(ds, "sup", "defaults", loss)


def tuned(ds, method, param, loss="cir"):
    r = TUNED[(TUNED.dataset == ds) & (TUNED.loss == loss) & (TUNED.method == method) & (TUNED.param == param)]
    return float(r.value.iloc[0]) if len(r) else np.nan


# ---------- A: 2x2-Zerlegung (Circle) ----------
cells = [("sup_all_shared", "ÜBl.-Hyperparameter, mit ungelabelten Bildern", stil.TOL["cyan"]),
         ("pool_labeled", "SSL-Hyperparameter, ohne ungelabelte Bilder", stil.NEUTRAL_FILL),
         ("defaults", "SSL-Hyperparameter, mit ungelabelten Bildern", stil.TOL["blue"]),
         ("zero_weight", "SSL-Hyperparameter, SSL-Term abgeschaltet", "white")]
groups = [("Cars196", "lp"), ("Cars196", "lrml"), ("Cars196", "hoffer"), ("Cars196", "slade"),
          ("DeepFashionInShop", "lp"), ("DeepFashionInShop", "lrml"), ("DeepFashionInShop", "hoffer"),
          ("DeepFashionInShop", "slade")]
fig, ax = plt.subplots(figsize=(stil.TEXTWIDTH, 3.5))
w = 0.2
alle = []
for gi, (ds, m) in enumerate(groups):
    s = sup(ds)
    for ci, (var, lab, col) in enumerate(cells):
        y = val(ds, m, var) - s
        if np.isnan(y):
            continue
        x = gi + (ci - 1.5) * (w + 0.01)
        alle.append(y)
        ax.bar(x, y, w, color=col, edgecolor=stil.BLACK, lw=0.5, label=lab if gi == 0 else None)
        ax.text(x, y + (0.08 if y >= 0 else -0.08), stil.de(y, 1), ha="center",
                va="bottom" if y >= 0 else "top", fontsize=6.5)
ax.axhline(0, color=stil.BLACK, lw=0.6)
ax.axvline(3.5, color=stil.LIGHT, lw=0.8)
ax.set_xticks(range(len(groups)), [stil.METHOD_LABELS[m] for _, m in groups])
ax.tick_params(axis="x", length=0)
ax.text(1.5, -0.2, DS["Cars196"], transform=ax.get_xaxis_transform(), ha="center", va="top")
ax.text(5.5, -0.2, DS["DeepFashionInShop"], transform=ax.get_xaxis_transform(), ha="center", va="top")
ax.set_ylabel("Differenz in Prozentpunkten")
ax.set_ylim(*stil.grenzen(-1.9, 4.0, alle, 0.1, rand=0.35))  # Platz für die Beschriftung unter den Balken
fig.legend(loc="outside upper center", ncol=1)
stil.save(fig, "phase2_2x2")
plt.close(fig)

# ---------- A2: 2x2-Zerlegung mit allen Verlustfunktionen (Zeilen: Mechanismus, Spalten: Datensatz) ----------
fig, axes = plt.subplots(4, 2, figsize=(stil.TEXTWIDTH, 7.0), sharey=True)
w = 0.2
alle = []
for ri, m in enumerate(["lp", "lrml", "hoffer", "slade"]):
    for ci_ds, ds in enumerate(["Cars196", "DeepFashionInShop"]):
        ax = axes[ri, ci_ds]
        for li, loss in enumerate(LOSSES):
            s = sup(ds, loss)
            for ci, (var, lab, col) in enumerate(cells):
                y = val(ds, m, var, loss) - s
                if np.isnan(y):
                    continue
                alle.append(y)
                ax.bar(li + (ci - 1.5) * (w + 0.01), y, w, color=col, edgecolor=stil.BLACK, lw=0.5)
        ax.axhline(0, color=stil.BLACK, lw=0.6)
        ax.set_xticks(range(len(LOSSES)), [stil.LOSS_LABELS[l].replace("Proxy-", "Proxy-\n") for l in LOSSES]
                      if ri == 3 else [], fontsize=8)
        ax.tick_params(axis="x", length=0)
        stil.panel(ax, "abcdefgh"[2 * ri + ci_ds], f"{stil.METHOD_LABELS[m]}, {DS[ds]}")
axes[0, 0].set_ylim(*stil.grenzen(-11.5, 5.0, alle, 0.5))  # gemeinsame Achse; kleinster Wert derzeit -10,7
for ax in axes[:, 0]:
    ax.set_ylabel("Differenz zur ÜBl.\nin Prozentpunkten")
fig.legend([Patch(facecolor=c, edgecolor=stil.BLACK, lw=0.5) for _, _, c in cells], [l for _, l, _ in cells],
           loc="outside upper center", ncol=2, columnspacing=1.0, handlelength=1.2, handletextpad=0.4)  # Standardabstände schnitten den rechten Eintrag ab
stil.save(fig, "phase2_2x2_verluste")
plt.close(fig)

# ---------- B: LP-Oracle-Rauschen, alle Verlustfunktionen (links Cars196, rechts In-Shop) ----------
# Seit 30.10. ohne die Panels zu Abdeckung und Genauigkeit über die Epochen, deren Werte in 5.7.2 im Text stehen.
D = sammlung.read_csv("phase2_lp_diagnostik.csv")
NOISE = [0, 10, 20, 40, 60]
fig, axs = plt.subplots(1, 2, figsize=(stil.TEXTWIDTH, 3.0), sharey=True)


def oracle(ax, ds, loss, col, marker, g, i):
    ys = [val(ds, "lp", f"oracle_noise_{p}", loss) - sup(ds, loss) for p in NOISE]
    ax.plot(stil.versatz(NOISE, i, len(LOSSES), 0.9, log=False), ys, color=col, marker=marker, **stil.PUNKT)
    err = 100 * (1 - g.acc.mean())  # mittlere Fehlerrate der übernommenen Pseudo-Labels
    ax.plot([err], [val(ds, "lp", "defaults", loss) - sup(ds, loss)], marker="*", ms=9, color=col, mec=stil.BLACK,
            mew=0.5, ls="none")
    return ys, [val(ds, "lp", "defaults", loss) - sup(ds, loss)]


alle = []
for ax, letter, (ds, key, name) in zip(axs, "ab", (("Cars196", "cars", "Cars196"), ("DeepFashionInShop", "in_shop", "In-Shop"))):
    reihen = []
    for i, loss in enumerate(LOSSES):
        g = D[(D.ds == key) & (D.loss == loss)]
        ys, reg = oracle(ax, ds, loss, stil.LOSS_COLORS[loss], stil.LOSS_MARKERS[loss], g, i)
        alle += ys + reg
        reihen.append((NOISE, ys))
    stil.trend(ax, reihen)
    ax.axhline(0, **REFLINE)
    ax.set_xlabel("Falsche Labels in %")
    stil.panel(ax, letter, name)
axs[0].set_ylabel("Differenz zur ÜBl.\nin Prozentpunkten")
axs[0].set_ylim(*stil.grenzen(-10, 8, alle, 2))
fig.legend([Line2D([], [], color=stil.LOSS_COLORS[l], marker=stil.LOSS_MARKERS[l], **stil.PUNKT) for l in LOSSES] +
           [Line2D([], [], color=stil.NEUTRAL, marker="*", ms=9, ls="none", mec=stil.BLACK, mew=0.5)],
           [stil.LOSS_LABELS[l] for l in LOSSES] + ["reguläres LP"], loc="outside lower center", ncol=6)
fig.legends[-1].remove()
fig.legend([Line2D([], [], color=stil.LOSS_COLORS[l], marker=stil.LOSS_MARKERS[l], **stil.PUNKT) for l in LOSSES] +
           [Line2D([], [], color=stil.NEUTRAL, marker="*", ms=9, ls="none", mec=stil.BLACK, mew=0.5),
            Line2D([], [], **{k: v for k, v in stil.TREND.items() if k != "zorder"})],
           [stil.LOSS_LABELS[l] for l in LOSSES] + ["reguläres LP", "Trend"], loc="outside lower center", ncol=7,
           columnspacing=1.0, handlelength=1.6)
stil.save(fig, "phase2_lp_qualitaet")
plt.close(fig)

# ---------- C: Graph-Kontrollen, alle Verlustfunktionen (Zeilen: LP, LRML, Spalten: Datensatz) ----------
ctrl = [("graph_rewired", "Graph umverdrahtet"),
        ("pl_once|graph_once", "nach Warm-up eingefroren"),
        ("no_warmup", "ohne Warm-up"),
        ("graph_random_pruned", "Kanten zufällig entfernt"),
        ("graph_oracle_pruned", "Kanten zwischen Klassen\nentfernt (Oracle)")]
XMIN = -16
fig, axes = plt.subplots(2, 2, figsize=(stil.TEXTWIDTH, 4.8), sharex=True, sharey=True)
rechts = []
for ri, m in enumerate(["lp", "lrml"]):
    for ci, ds in enumerate(["Cars196", "DeepFashionInShop"]):
        ax = axes[ri, ci]
        for row, (vs, lab) in enumerate(ctrl):
            y0 = len(ctrl) - 1 - row
            for li, loss in enumerate(LOSSES):
                r0 = val(ds, m, "defaults", loss)
                v = np.nan
                for vv in vs.split("|"):
                    if not np.isnan(val(ds, m, vv, loss)):
                        v = val(ds, m, vv, loss)
                if np.isnan(v) or np.isnan(r0):
                    continue
                d = v - r0
                rechts.append(d)
                y = y0 + (2 - li) * 0.15
                col = stil.LOSS_COLORS[loss]
                ax.plot([0, max(d, XMIN + 0.7)], [y, y], color=col, lw=0.8)
                if d < XMIN + 0.7:  # abgeschnitten: Dreieck am linken Rand
                    ax.plot(XMIN + 0.7, y, marker="<", ms=4, color=col, ls="none")
                else:
                    ax.plot(d, y, marker=stil.LOSS_MARKERS[loss], ms=3.5, color=col, ls="none")
        ax.axvline(0, color=stil.BLACK, lw=0.6)
        ax.set_yticks(range(len(ctrl)), [l for _, l in ctrl][::-1])
        ax.tick_params(axis="y", length=0)
        # links bewusst abgeschnitten (Dreieck am Rand): derzeit LP/ArcFace auf In-Shop, zufällige Kanten, -22,6
        ax.set_xlim(XMIN, stil.grenzen(XMIN, 3, rechts, 1)[1])
        stil.panel(ax, "abcd"[2 * ri + ci], f"{stil.METHOD_LABELS[m]}, {DS[ds]}")
fig.supxlabel("Differenz zur regulären Konfiguration in Prozentpunkten", fontsize=10)
fig.legend([Line2D([], [], color=stil.LOSS_COLORS[l], marker=stil.LOSS_MARKERS[l], ms=4) for l in LOSSES],
           [stil.LOSS_LABELS[l] for l in LOSSES], loc="outside upper center", ncol=5)
stil.save(fig, "phase2_graph")
plt.close(fig)

# ---------- D: Sweeps ----------
SW = [("lp", "n_neighbors", "LP: $k_{\\mathrm{NN}}$", True),
      ("lp", "confidence_threshold", "LP: Konfidenzschwelle", False),
      ("lrml", "n_neighbors", "LRML: $k_{\\mathrm{NN}}$", True),
      ("lrml", "regularizer_target_ratio", "LRML: $\\eta$", True),
      ("hoffer", "distance_scale", "Hoffer: $\\gamma_{\\mathrm{H}}$", True),
      ("hoffer", "regularizer_target_ratio", "Hoffer: $\\eta$", True)]  # seit 17.10.2026


def sweep(ax, ds, m, param, loss, col, marker, ms=None, floor=None, i=0, n=1, logx=True):
    """floor: untere Achsengrenze; Punkte darunter erscheinen als Pfeil am unteren Rand. i, n: Serie i von n, leicht
    versetzt (logarithmische Achse um 0,02 Dekaden, lineare um 0,008 je Serie)."""
    pre = param + "_"
    r = V[(V.dataset == ds) & (V.loss == loss) & (V.method == m) & V.variant.str.startswith(pre)].copy()
    if r.empty:
        return []
    r["x"] = r.variant.str[len(pre):].str.replace("p", ".").astype(float)
    base = r[["x", "mean_value"]].values.tolist()
    dv, t = val(ds, m, "defaults", loss), tuned(ds, m, param, loss)
    if not np.isnan(dv):
        base.append([t, dv])
    b = np.array(sorted(base))
    y = b[:, 1] - sup(ds, loss)
    x = stil.versatz(b[:, 0], i, n, 0.02 if logx else 0.008, log=logx)
    # Keine Trendlinie: Die Sweeps haben Optima und Abbrüche (k_NN = 400, Schwelle 0,9), die eine quadratische Anpassung
    # verschiebt oder erfindet (geprüft 01.11.2026, z. B. LRML k_NN mit MS).
    inside = y >= floor if floor is not None else np.ones(len(y), bool)
    ax.plot(x[inside], y[inside], color=col, marker=marker, **stil.PUNKT)
    # Pfeil statt Dreieck: "v" ist schon die Marke von NT-Xent
    ax.plot(x[~inside], np.full((~inside).sum(), (floor or 0) + 0.8), color=col, marker=CUT, ms=7, ls="none")
    if not np.isnan(dv):
        ax.plot(stil.versatz([t], i, n, 0.02 if logx else 0.008, log=logx), [dv - sup(ds, loss)], marker="*", ms=9,
                color=col, mec=stil.BLACK, mew=0.5, ls="none")
    return b[:, 0], y


def sweep_axes(axes, logx):
    for ax, lx in zip(axes, logx):
        ax.axhline(0, **REFLINE)
        if lx:
            ax.set_xscale("log")
            ax.minorticks_off()
            ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, p: stil.smart(v, 1)))


fig, axes = plt.subplots(2, 3, figsize=stil.SIZES["full2"])
for ax, letter, (m, param, title, logx) in zip(axes.flat, "abcdef", SW):
    for i, ds in enumerate(("Cars196", "DeepFashionInShop")):
        sweep(ax, ds, m, param, "cir", DSCOL[ds], DSMARK[ds], i=i, n=2, logx=logx)
    stil.panel(ax, letter, title)
sweep_axes(axes.flat, [s[3] for s in SW])
fig.supylabel("Differenz in Prozentpunkten", fontsize=10)
fig.legend([Line2D([], [], color=DSCOL[d], marker=DSMARK[d], **stil.PUNKT) for d in DS] +
           [Line2D([], [], color=stil.NEUTRAL, marker="*", ms=9, ls="none", mec=stil.BLACK, mew=0.5)],
           list(DS.values()) + ["Wert aus Phase 1"], loc="outside lower center", ncol=3)
stil.save(fig, "phase2_sweeps")
plt.close(fig)

# ---------- E: Sweeps auf Cars196 mit allen Verlustfunktionen (Anhang) ----------
fig, axes = plt.subplots(2, 3, figsize=stil.SIZES["full2"])
for ax, letter, (m, param, title, logx) in zip(axes.flat, "abcdef", SW):
    reihen = []
    for i, loss in enumerate(LOSSES):
        reihen.append(sweep(ax, "Cars196", m, param, loss, stil.LOSS_COLORS[loss], stil.LOSS_MARKERS[loss], i=i,
                            n=len(LOSSES), logx=logx))
    stil.trend(ax, [r for r in reihen if len(r)], log=logx)
    stil.panel(ax, letter, title)
sweep_axes(axes.flat, [s[3] for s in SW])
fig.supylabel("Differenz zur ÜBl. in Prozentpunkten", fontsize=10)
fig.legend([Line2D([], [], color=stil.LOSS_COLORS[l], marker=stil.LOSS_MARKERS[l], **stil.PUNKT) for l in LOSSES] +
           [Line2D([], [], color=stil.NEUTRAL, marker="*", ms=9, ls="none", mec=stil.BLACK, mew=0.5)],
           [stil.LOSS_LABELS[l] for l in LOSSES] + ["Wert aus Phase 1"], loc="outside lower center", ncol=6)
fig.legends[-1].remove()
fig.legend([Line2D([], [], color=stil.LOSS_COLORS[l], marker=stil.LOSS_MARKERS[l], **stil.PUNKT) for l in LOSSES] +
           [Line2D([], [], color=stil.NEUTRAL, marker="*", ms=9, ls="none", mec=stil.BLACK, mew=0.5),
            Line2D([], [], **{k: v for k, v in stil.TREND.items() if k != "zorder"})],
           [stil.LOSS_LABELS[l] for l in LOSSES] + ["Wert aus Phase 1", "Trend"], loc="outside lower center", ncol=4,
           columnspacing=1.0, handlelength=1.6)
stil.save(fig, "phase2_sweeps_verluste")
plt.close(fig)

# ---------- E2: Sweeps auf In-Shop mit allen Verlustfunktionen (Anhang), y-Achse unten bei FLOOR abgeschnitten ----------
FLOOR = -20
fig, axes = plt.subplots(2, 3, figsize=stil.SIZES["full2"])
for ax, letter, (m, param, title, logx) in zip(axes.flat, "abcdef", SW):
    ys = []
    reihen = []
    for i, loss in enumerate(LOSSES):
        r = sweep(ax, "DeepFashionInShop", m, param, loss, stil.LOSS_COLORS[loss], stil.LOSS_MARKERS[loss],
                  floor=FLOOR, i=i, n=len(LOSSES), logx=logx)
        if len(r):
            reihen.append(r)
            ys += list(r[1])
    stil.trend(ax, reihen, log=logx)
    ax.set_ylim(FLOOR, stil.grenzen(FLOOR, 7, [y for y in ys if y >= FLOOR], 1)[1])  # unten bewusst bei FLOOR
    stil.panel(ax, letter, title)
sweep_axes(axes.flat, [s[3] for s in SW])
fig.supylabel("Differenz zur ÜBl. in Prozentpunkten", fontsize=10)
fig.legend([Line2D([], [], color=stil.LOSS_COLORS[l], marker=stil.LOSS_MARKERS[l], **stil.PUNKT) for l in LOSSES] +
           [Line2D([], [], color=stil.NEUTRAL, marker="*", ms=9, ls="none", mec=stil.BLACK, mew=0.5),
            Line2D([], [], color=stil.NEUTRAL, marker=CUT, ms=7, ls="none")],
           [stil.LOSS_LABELS[l] for l in LOSSES] + ["Wert aus Phase 1", f"unter ${FLOOR}$"],
           loc="outside lower center", ncol=4)
fig.legends[-1].remove()
fig.legend([Line2D([], [], color=stil.LOSS_COLORS[l], marker=stil.LOSS_MARKERS[l], **stil.PUNKT) for l in LOSSES] +
           [Line2D([], [], color=stil.NEUTRAL, marker="*", ms=9, ls="none", mec=stil.BLACK, mew=0.5),
            Line2D([], [], color=stil.NEUTRAL, marker=CUT, ms=7, ls="none"),
            Line2D([], [], **{k: v for k, v in stil.TREND.items() if k != "zorder"})],
           [stil.LOSS_LABELS[l] for l in LOSSES] + ["Wert aus Phase 1", f"unter ${FLOOR}$", "Trend"],
           loc="outside lower center", ncol=4)
stil.save(fig, "phase2_sweeps_verluste_inshop")
plt.close(fig)
