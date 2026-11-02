"""Phase 1: gepaarter MAP@R-Gewinn der SSL-Verfahren ggü. Ü je Budget als Heatmap mit Werten (Mean oben, Concat unten).
Mittel über die fünf Verlustfunktionen der Seed-Mittel, gleiche Werte wie die Anhangtabelle gewinne.tex (tab_hauptteil.py)."""
import datensammlung as sammlung
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import stil

stil.apply()
R = ""
cmap = plt.get_cmap(stil.DIV_CMAP)
DATA = [("Cars196", "cars", [4, 8, 16, 32]), ("CUB", "cub", [4, 8, 16, 32]), ("In-Shop", "inshop", [2, 4]),
        ("Semi-iNat", "semiinat", [2, 4, 16, 100])]
FAMS = ["iscen", "lrml", "hoffer", "slade"]


def txt(v):
    s = stil.de(v, 1)
    return "+" + s if round(v, 1) > 0 else s


# eine Spalte je Datensatz, Breite proportional zur Zahl der Budgets (gleich große Zellen)
fig, axes = plt.subplots(2, 4, figsize=(stil.TEXTWIDTH, 2.75), gridspec_kw={"width_ratios": [len(d[2]) for d in DATA]})
for j, (name, key, ks) in enumerate(DATA):
    pdf = sammlung.read_csv(R + f"{key}_ergebnisse/data/phase1_paired_differences.csv")
    pdf = pdf[pdf.metric == "mean_average_precision_at_r"].copy()
    pdf["fam"] = pdf.method.str.split("_").str[0]
    g = pdf.groupby(["scope", "fam", "k"])["mean"].agg(["mean", "size"])
    assert (g["size"] == 5).all(), key  # alle fünf Verlustfunktionen
    vmax = g["mean"].abs().max()  # Farbskala je Datensatz, Mean und Concat gemeinsam
    norm = TwoSlopeNorm(0, -vmax, vmax)
    for i, (scope, scope_name) in enumerate([("mean", "Mean"), ("concat", "Concat")]):
        ax = axes[i, j]
        M = np.array([[g.loc[(scope, fam, k), "mean"] for k in ks] for fam in FAMS])
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for bi in range(len(ks)):
            top = max(round(v, 1) for v in M[:, bi])  # fett: größter Gewinn je Budget, sofern positiv (wie fett())
            for fi in range(len(FAMS)):
                v = M[fi, bi]
                # weiße Schrift auf dunklen Zellen
                ax.text(bi, fi, txt(v), ha="center", va="center", fontsize=7,
                        fontweight="bold" if top > 0 and round(v, 1) == top else "normal",
                        color="white" if abs(norm(v) - 0.5) > 0.32 else stil.BLACK)
        ax.set_xticks(range(len(ks)), [str(k) for k in ks])
        ax.set_yticks(range(len(FAMS)), [stil.METHOD_LABELS[f] for f in FAMS] if j == 0 else [])
        ax.tick_params(length=0, top=False, right=False)
        if i == 0:
            ax.set_title(name)  # Datensatz als Spaltenkopf
        if j == 0:
            ax.set_ylabel(scope_name)
fig.supxlabel("Gelabelte Bilder je Klasse $k$", fontsize=10)
stil.save(fig, "phase1_gewinne")
