"""Label Propagation im Graphlauf auf Cars196: Status der Pseudo-Labels im t-SNE des Trainingspools (LP mit Circle, k=16,
Seed 2, Fold 0; 4000 verfolgte Bilder), nach Epoche 20 und 135. Gezeigt sind die Pseudo-Labels, die das Training aus dem gezeigten Stand für die folgende Epoche übernommen hat
(pseudo_labels/*.npz der Läufe), nicht eine Neuberechnung auf den verfolgten Bildern (seit 25.10.2026)."""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import stil, tracking

stil.apply()
OK, BAD, REJ, LAB = stil.TOL["cyan"], stil.TOL["red"], "#AAAAAA", stil.BLACK  # kontrastreich auch ohne Zoom
fig, axes = plt.subplots(1, 2, figsize=(stil.TEXTWIDTH, 2.75))  # Felder so groß wie zuvor, damit die Marker ohne Zoom lesbar sind
frames = tracking.adopted("cars")
for ax, ep, letter in zip(axes, sorted(frames), "ab"):
    f = frames[ep]
    x, y, L, pred = f.x.values, f.y.values, f.label.values, f.adopted.values
    U = f.kind.values == "unlabeled"
    acc = U & (pred >= 0)
    good, bad = acc & (pred == L), acc & (pred != L)
    ax.scatter(x[U & ~acc], y[U & ~acc], s=7, color=REJ, lw=0, rasterized=True)
    ax.scatter(x[good], y[good], s=7, color=OK, lw=0, rasterized=True)
    ax.scatter(x[~U], y[~U], s=5, color=LAB, lw=0, rasterized=True)
    ax.scatter(x[bad], y[bad], s=16, color=BAD, marker="x", lw=0.8, rasterized=True)  # Kreuz groß genug, um es als Kreuz zu erkennen
    stil.panel(ax, letter, f"Cars196, nach Epoche {ep}")
    ax.set_xticks([]); ax.set_yticks([])
    print("cars", ep, "U", U.sum(), "L", (~U).sum(), "Klassen", len(set(L)),
          "cov %.3f acc %.3f" % (acc.sum() / U.sum(), good.sum() / acc.sum()))
hh = [Line2D([], [], ls="none", marker="o", ms=5, color=LAB), Line2D([], [], ls="none", marker="o", ms=5, color=OK),
      Line2D([], [], ls="none", marker="x", ms=6, mew=1.2, color=BAD), Line2D([], [], ls="none", marker="o", ms=5, color=REJ)]
fig.legend(hh, ["gelabelt", "Pseudo-Label richtig", "Pseudo-Label falsch", "unter der Schwelle"],
           loc="outside upper center", ncol=4)
stil.save(fig, "lp_cars")
