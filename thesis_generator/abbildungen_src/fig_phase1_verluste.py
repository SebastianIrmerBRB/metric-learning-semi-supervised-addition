"""Phase 1: gepaarter MAP@R-Gewinn je Verlust und Verfahren beim kleinsten Budget (Mean oben, Concat unten)."""
import datensammlung as sammlung
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import stil

stil.apply()
R = ""
cmap = plt.get_cmap(stil.DIV_CMAP)
DATA = [("Cars196", "cars", 4), ("CUB", "cub", 4), ("In-Shop", "inshop", 2), ("Semi-iNat", "semiinat", 2)]
FAMS = ["iscen", "lrml", "hoffer", "slade"]
fig, axes = plt.subplots(2, 4, figsize=(stil.TEXTWIDTH, 4.4))
letters = iter("abcdefgh")
for i, (scope, scope_name) in enumerate([("mean", "M"), ("concat", "C")]):
    for j, (name, key, k) in enumerate(DATA):
        pdf = sammlung.read_csv(R + f"{key}_ergebnisse/data/phase1_paired_differences.csv")
        pdf = pdf[(pdf.metric == "mean_average_precision_at_r") & (pdf.k == k)]
        vmax = pdf["mean"].abs().max()
        norm = TwoSlopeNorm(0, -vmax, vmax)
        ax = axes[i, j]
        M = np.full((5, 4), np.nan)
        P = np.zeros((5, 4))
        for li, loss in enumerate(stil.LOSS_ORDER):
            for fi, fam in enumerate(FAMS):
                r = pdf[(pdf.scope == scope) & (pdf.method == f"{fam}_{loss}")]
                if len(r):
                    M[li, fi] = r["mean"].iloc[0]
                    P[li, fi] = r.positive_seeds.iloc[0]
        ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for li in range(5):
            for fi in range(4):
                v = M[li, fi]
                # weiße Schrift auf dunklen Zellen
                dark = abs(norm(v) - 0.5) > 0.32
                ax.text(fi, li, stil.de(v, 1), ha="center", va="center", fontsize=7,
                        fontweight="bold" if P[li, fi] in (0, 5) else "normal",
                        color="white" if dark else stil.BLACK)
        ax.set_xticks(range(4), [stil.METHOD_LABELS[f] for f in FAMS])
        ax.set_yticks(range(5), [stil.LOSS_LABELS[l] for l in stil.LOSS_ORDER] if j == 0 else [])
        ax.tick_params(length=0, top=False, right=False)
        ax.tick_params(axis="x", labelsize=7, rotation=90)
        stil.panel(ax, next(letters), f"{name}, {scope_name}")
stil.save(fig, "phase1_verluste")
