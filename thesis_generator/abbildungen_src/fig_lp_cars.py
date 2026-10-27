"""Label Propagation in einem realen Trainingslauf: Status der Pseudo-Labels im t-SNE des Cars196-Trainingspools
(LP mit Circle, k=16, Seed 2, Fold 0; 4000 verfolgte Bilder) bei der ersten und der letzten Aktualisierung."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import datensammlung as sammlung
import stil, tracking

stil.apply()
# Konfidenzschwelle der Phase-1-Konfiguration des gezeigten Laufs (LP mit Circle, k=16, Seed 2)
_S = sammlung.read_csv("cars_ergebnisse/data/phase1_settings.csv")
THR = _S[(_S.method == "iscen_cir") & (_S.k == 16) & (_S.seed == 2)]["ssl_config.confidence_threshold"].item()
OK, BAD, REJ, LAB = stil.TOL["blue"], stil.TOL["red"], stil.LIGHT, stil.BLACK
d = tracking.run_dir("iscen_cir")
eps = [20, tracking.epochs(d)[-1]]
fig, axes = plt.subplots(1, 2, figsize=stil.SIZES["full"])
for ax, ep, letter in zip(axes, eps, "ab"):
    f = tracking.frame(d, ep)
    x, y, L, kind, pred, cert = f["x"], f["y"], f["label"], f["kind"], f["lp_pred"], f["lp_cert"]
    U = kind == "unlabeled"
    acc = U & (cert >= THR)
    good, bad = acc & (pred == L), acc & (pred != L)
    ax.scatter(x[U & ~acc], y[U & ~acc], s=2, color=REJ, rasterized=True)
    ax.scatter(x[good], y[good], s=2, color=OK, rasterized=True)
    ax.scatter(x[~U], y[~U], s=2, color=LAB, rasterized=True)
    ax.scatter(x[bad], y[bad], s=6, color=BAD, marker="x", rasterized=True)
    cov, a = acc.sum() / U.sum(), good.sum() / acc.sum()
    stil.panel(ax, letter, f"Epoche {ep}")
    ax.set_xticks([]); ax.set_yticks([])
    print(ep, "U", U.sum(), "L", (~U).sum(), "cov %.3f acc %.3f all %.3f" % (cov, a, (pred[U] == L[U]).mean()))
hh = [Line2D([], [], ls="none", marker="o", color=LAB), Line2D([], [], ls="none", marker="o", color=OK),
      Line2D([], [], ls="none", marker="x", color=BAD), Line2D([], [], ls="none", marker="o", color=REJ)]
fig.legend(hh, ["gelabelt", "Pseudo-Label richtig", "Pseudo-Label falsch", "unter der Schwelle"],
           loc="outside upper center", ncol=4)
stil.save(fig, "lp_cars")
