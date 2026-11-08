"""Pseudo-Labels von LP in den Geometrieläufen (Phase-1-Konfiguration, Seed 2) über das Label-Budget, alle fünf
Verlustfunktionen, Mittel über die vier Folds. In der Arbeit: lp_pseudolabels (Anhang C, zu Abschnitt 5.6.2).

Zeilen: Abdeckung und Genauigkeit der Pseudo-Labels (offen: erste, gefüllt: letzte Aktualisierung des Folds), Verbleib
der anfangs falschen Pseudo-Labels bei der letzten Aktualisierung (gefüllt: richtig, offen: dieselbe Klasse) und auf
Cars196 der Anteil der falschen, die ein Modell desselben Herstellers nennen (grau: bei zufälliger Verwechslung).
Spalten: Cars196, In-Shop, Semi-iNat-2021. Ein grauer gestrichelter Trend je Feld über die gefüllten Punkte.
Daten: Sichten geometrie/lp_verfolgung.csv (sammle_lp_verfolgung.py) und phase2/lp_hersteller.csv (lp_hersteller.py)
des Datenpakets, beide aus den gespeicherten Pseudo-Labels der Geometrieläufe auf dem ganzen ungelabelten Pool."""
import datensammlung as sammlung
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
import stil

stil.apply()
V = sammlung.read_csv("geometrie/lp_verfolgung.csv")
H = sammlung.read_csv("phase2/lp_hersteller.csv")
LOSSES = stil.LOSS_ORDER
DATA = [("cars", "Cars196", [4, 8, 16, 32]), ("inshop", "In-Shop", [2, 4]), ("semiinat", "Semi-iNat-2021", [2, 4, 16, 100])]
OFFEN = dict(mfc="white", mew=0.9)


def mittel(df, ds, loss, spalte, ks):
    """Mittel über die Folds in Prozent je Budget."""
    g = df[(df.ds == ds) & (df.loss == loss)].groupby("k")[spalte].mean()
    return np.array([100 * g.get(k, np.nan) for k in ks])


BREITE = [1.1, 0.65, 1.5]  # Spaltenbreiten etwa nach der Spanne der Budgets (log2), In-Shop hat nur zwei


def achse(ax, ks, c):
    rand = max(0.13 * np.log2(ks[-1] / ks[0]), 0.4)  # wie fig_phase1_uebersicht: Platz für den äußersten Versatz
    lo, hi = ks[0] / 2 ** rand, ks[-1] * 2 ** rand
    ax.set_xscale("log", base=2)
    ax.set_xticks(ks, [str(k) for k in ks])
    ax.minorticks_off()
    ax.set_xlim(lo, hi)
    breite = 5.3 * BREITE[c] / sum(BREITE)  # ungefähre Breite des Feldes in Zoll
    return 0.05 * np.log2(hi / lo) / breite  # Versatz je Serie in log2-Einheiten, sichtbar etwa 0,05 Zoll in jedem Feld


def feld(ax, ks, paare, dodge):
    """paare: je Verlustfunktion (gefüllt, offen) als Werte je Budget. Trend über die gefüllten Punkte."""
    reihen = []
    for i, loss in enumerate(LOSSES):
        voll, offen = paare[loss]
        x = np.array(ks, dtype=float) * 2 ** ((i - 2) * dodge)
        col, mk = stil.LOSS_COLORS[loss], stil.LOSS_MARKERS[loss]
        ax.plot(x, offen, color=col, marker=mk, mec=col, zorder=3, **OFFEN, **stil.PUNKT)
        ax.plot(x, voll, color=col, marker=mk, zorder=4, **stil.PUNKT)
        reihen.append((ks, voll))
    stil.trend(ax, reihen, log=True)


fig = plt.figure(figsize=(stil.TEXTWIDTH, 6.5))
gs = fig.add_gridspec(4, 3, width_ratios=BREITE)
zeilen = [("cov1", "cov0", "Abdeckung in %", (0, 100), 25),
          ("acc1", "acc0", "Genauigkeit in %", (20, 100), 20),
          ("wrong_fixed", "wrong_keep", "Anfangs falsche in %", (0, 100), 25)]
letters = iter("abcdefghij")
erste = {}
for r, (voll, offen, ylabel, (lo, hi), step) in enumerate(zeilen):
    werte, achsen = [], []
    for c, (ds, name, ks) in enumerate(DATA):
        ax = fig.add_subplot(gs[r, c], sharey=erste.get(r))
        erste.setdefault(r, ax)
        dodge = achse(ax, ks, c)
        paare = {l: (mittel(V, ds, l, voll, ks), mittel(V, ds, l, offen, ks)) for l in LOSSES}
        werte += [v for p in paare.values() for v in np.concatenate(p)]
        feld(ax, ks, paare, dodge)
        stil.panel(ax, next(letters), name)
        if c:
            ax.tick_params(labelleft=False)
        else:
            ax.set_ylabel(ylabel)
        if r == 2 and c:  # unterste Felder von In-Shop (schmal, nur $k$) und Semi-iNat-2021
            ax.set_xlabel("$k$" if c == 1 else "Gelabelte Bilder je Klasse $k$")
        achsen.append(ax)
    lo, hi = stil.grenzen(lo, hi, werte, step)
    achsen[0].set_ylim(lo - 3, hi + 3)  # Rand, damit Punkte bei 0 und 100 % nicht angeschnitten werden
    achsen[0].yaxis.set_major_locator(MultipleLocator(step))

# Hersteller (nur Cars196): Anteil der falschen Pseudo-Labels mit Modell desselben Herstellers
ax = fig.add_subplot(gs[3, 0])
ks = DATA[0][2]
dodge = achse(ax, ks, 0)
Hs = H.assign(ds="cars")
reihen, werte = [], []
for i, loss in enumerate(LOSSES):
    x = np.array(ks, dtype=float) * 2 ** ((i - 2) * dodge)
    col, mk = stil.LOSS_COLORS[loss], stil.LOSS_MARKERS[loss]
    for stage, stil_kw in (("erste", OFFEN), ("letzte", {})):
        d = Hs[Hs.stage == stage]
        y, zufall = mittel(d, "cars", loss, "same_make", ks), mittel(d, "cars", loss, "same_make_random", ks)
        ax.plot(x, y, color=col, marker=mk, mec=col, zorder=4 if not stil_kw else 3, **stil_kw, **stil.PUNKT)
        ax.plot(x, zufall, color=stil.NEUTRAL, marker=mk, mec=stil.NEUTRAL, zorder=3, **stil_kw,
                **dict(stil.PUNKT, ms=3.8))
        werte += list(y) + list(zufall)
        if stage == "letzte":
            reihen.append((ks, y))
stil.trend(ax, reihen, log=True)
lo, hi = stil.grenzen(0, 100, werte, 25)
ax.set_ylim(lo - 3, hi + 3)
ax.yaxis.set_major_locator(MultipleLocator(25))
ax.set_ylabel("Gleicher Hersteller in %")
ax.set_xlabel("Gelabelte Bilder je Klasse $k$")
stil.panel(ax, next(letters), "Cars196")

# Legende im freien Platz neben (j)
lax = fig.add_subplot(gs[3, 1:])
lax.axis("off")
grau = dict(color=stil.NEUTRAL, marker="o", mec=stil.NEUTRAL, **stil.PUNKT)
h = [Line2D([], [], color=stil.LOSS_COLORS[l], marker=stil.LOSS_MARKERS[l], **stil.PUNKT) for l in LOSSES]
l = [stil.LOSS_LABELS[x] for x in LOSSES]
h += [Line2D([], [], **grau, **OFFEN), Line2D([], [], **grau),
      Line2D([], [], **dict(grau, ms=3.8)), Line2D([], [], **{k: v for k, v in stil.TREND.items() if k != "zorder"})]
l += ["offen: erste Aktualisierung, in (g–i) dieselbe Klasse", "gefüllt: letzte Aktualisierung, in (g–i) richtig",
      "klein, grau: zufällige Verwechslung in (j)", "Trend"]
lax.legend(h, l, loc="center", ncol=1, handlelength=1.6)
stil.save(fig, "lp_pseudolabels")
plt.close(fig)
