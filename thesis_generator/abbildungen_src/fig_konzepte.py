"""Konzeptabbildungen mit synthetischen Daten bzw. aus Formeln (keine Versuchsergebnisse).

ssl_annahmen.pdf      Smoothness-, Mannigfaltigkeits- und Low-Density-Annahme nach van Engelen und Hoos an 2-D-Daten
lp_toy.pdf           Label Propagation nach Zhou et al. auf einem kNN-Graphen synthetischer 2-D-Daten
hoffer_skala.pdf      punktweise obere Schranke der Zuordnungswahrscheinlichkeit bei Hoffer und Ailon über s_H
geometrie_konzept.pdf Streuung / Abstand (euklidisches pi_ratio) und spektraler Abfall rho an synthetischen Embeddings
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Ellipse
import stil
import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 0

stil.apply()
CAT = stil.PALETTE  # zwei Klassen: blau, rot


# ------------------------------------------------------------------ SSL-Annahmen
def ssl_annahmen():
    """Drei Teilbilder, je ein gelabelter Punkt pro Klasse (groß, schwarz umrandet). Ungelabelte Punkte sind farbig,
    wenn sie nach der Annahme dem gelabelten Punkt derselben Farbe ähneln (van Engelen und Hoos, Abschn. 2.2: nah im
    Eingaberaum, auf derselben Mannigfaltigkeit, im selben Bereich hoher Dichte), sonst grau. Nicht gezeigt ist die
    transitive Anwendung der Smoothness-Annahme über ungelabelte Zwischenpunkte.
    (a) und (c): zwei langgestreckte Normalverteilungen mit gleicher Kovarianz (je 100 Punkte), vgl. van Engelen und
        Hoos, Abb. 1 und 2a. (a) färbt die Punkte in einer Umgebung jedes gelabelten Punkts, (c) die Punkte in den
        Bereichen hoher Dichte, hier innerhalb der Ellipse mit Mahalanobis-Abstand 2 um den Mittelpunkt einer der
        erzeugenden Verteilungen (Konturen wie in Abb. 1 dort). Punkte dazwischen, im Bereich geringer Dichte, bleiben
        grau. Durchgezogen: x = 0, die Bayes-Grenze der beiden erzeugenden Verteilungen, eine der Grenzen im Bereich
        geringer Dichte.
        Gestrichelt: Mittelsenkrechte der gelabelten Punkte, die Grenze allein aus den Labels.
    (b): zwei konzentrische Ringe wie sklearn.datasets.make_circles (factor 0,5, je 100 Punkte), vgl. Abb. 2b dort
        (dort zwei getrennte Ringe).
    Gelabelt ist jeweils die Stichprobe, die einer festen Position am nächsten liegt, keine Seed-Auswahl (Seed 0)."""
    rng = np.random.default_rng(0)
    n, sx, sy, abst = 100, 0.32, 1.0, 1.0          # Streuung quer und längs, halber Abstand der Mittelpunkte
    B = np.r_[rng.normal(size=(n, 2)) * [sx, sy] + [-abst, 0], rng.normal(size=(n, 2)) * [sx, sy] + [abst, 0]]
    yb = np.r_[np.zeros(n, int), np.ones(n, int)]
    lab_b = np.array([np.argmin(((B - p) ** 2).sum(1)) for p in ([-abst, 1.3], [abst, -1.3])])
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    R = np.r_[0.5 * np.c_[np.cos(t), np.sin(t)], np.c_[np.cos(t), np.sin(t)]] + 0.05 * rng.normal(size=(2 * n, 2))
    yr = np.r_[np.zeros(n, int), np.ones(n, int)]
    lab_r = np.array([np.argmin(((R - p) ** 2).sum(1)) for p in ([-0.5, 0], [1, 0])])  # gegenüberliegende Seiten
    radius, klein, gross = 0.7, 6.0, 42.0
    lim_x, lim_y = (-2.3, 2.3), (-3.0, 3.0)

    def achsen(ax):
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_aspect("equal")

    def punkte(ax, X, farbe, lab, y):
        unl = np.setdiff1d(np.arange(len(X)), lab)
        g = unl[farbe[unl] < 0]
        ax.scatter(X[g, 0], X[g, 1], s=klein, color=stil.NEUTRAL_FILL, lw=0, zorder=2)
        for c in range(2):
            m = unl[farbe[unl] == c]
            ax.scatter(X[m, 0], X[m, 1], s=klein, color=CAT[c], lw=0, zorder=2, marker=stil.CAT_MARKERS[c])
            m = lab[y[lab] == c]
            ax.scatter(X[m, 0], X[m, 1], s=gross, color=CAT[c], edgecolor=stil.BLACK, lw=0.8, zorder=4,
                       marker=stil.CAT_MARKERS[c])

    fig, axes = plt.subplots(1, 3, figsize=(stil.TEXTWIDTH, 1.9))
    ax = axes[0]
    farbe = np.full(len(B), -1)
    for i in lab_b:
        farbe[np.sqrt(((B - B[i]) ** 2).sum(1)) <= radius] = yb[i]
        ax.add_patch(plt.Circle(B[i], radius, fill=False, color=stil.NEUTRAL, lw=0.7, zorder=3))
    punkte(ax, B, farbe, lab_b, yb)
    n_nah = int((farbe >= 0).sum()) - len(lab_b)
    stil.panel(ax, "a", "Smoothness")
    ax = axes[1]
    punkte(ax, R, yr.copy(), lab_r, yr)
    stil.panel(ax, "b", "Mannigfaltigkeit")
    ax = axes[2]
    mahal = np.stack([np.sqrt((((B - [m, 0]) / [sx, sy]) ** 2).sum(1)) for m in (-abst, abst)], 1)
    farbe = np.where(mahal.min(1) <= 2, mahal.argmin(1), -1)
    for m in (-abst, abst):
        ax.add_patch(Ellipse((m, 0), 4 * sx, 4 * sy, fill=False, color=stil.NEUTRAL, lw=0.7, zorder=3))
    punkte(ax, B, farbe, lab_b, yb)
    ax.plot([0, 0], lim_y, color=stil.BLACK, lw=1.0, zorder=3)
    p, q = B[lab_b[0]], B[lab_b[1]]
    mitte, v = (p + q) / 2, q - p
    ys = np.array(lim_y)
    ax.plot(mitte[0] - (ys - mitte[1]) * v[1] / v[0], ys, color=stil.BLACK, lw=1.0, ls=(0, (4, 2)), zorder=3)
    stil.panel(ax, "c", "Low Density")
    for ax in axes:
        achsen(ax)
    for ax in (axes[0], axes[2]):
        ax.set_xlim(*lim_x); ax.set_ylim(*lim_y)
    stil.save(fig, "ssl_annahmen")
    print("SSL-Annahmen: ungelabelte Punkte in den Umgebungen", n_nah, "in den Ellipsen", int((farbe >= 0).sum()) - 2,
          "davon andere Klasse als erzeugt", int(((farbe >= 0) & (farbe != yb)).sum()),
          "Mahalanobis der gelabelten", np.round(mahal[lab_b].min(1), 2))


ssl_annahmen()

# ------------------------------------------------------------------ LP-Spielbeispiel
rng = np.random.default_rng(SEED)
# Zwei Monde wie sklearn.datasets.make_moons: je 100 gleichmäßig verteilte Punkte auf zwei ineinandergreifenden
# Halbkreisen plus Gaußsches Rauschen. Je Mond zwei Labels, bei einem Viertel und der Hälfte des Bogens vom äußeren
# Ende aus, sodass die inneren Enden, die in den anderen Mond hineinragen, am weitesten von den eigenen Labels liegen.
n, RAUSCHEN = 100, 0.12
t = np.linspace(0, np.pi, n)
X = np.r_[np.c_[np.cos(t), np.sin(t)], np.c_[1 - np.cos(t), 0.5 - np.sin(t)]] + RAUSCHEN * rng.normal(size=(2 * n, 2))
y = np.r_[np.zeros(n, int), np.ones(n, int)]
N, C = len(X), 2
q = n - 1 - np.array([n // 4, n // 2])
lab = np.r_[q, n + q]
kn = 7
D2 = ((X[:, None] - X[None]) ** 2).sum(-1)
np.fill_diagonal(D2, np.inf)
nn = np.argsort(D2, axis=1)[:, :kn]
sig2 = np.median(D2[np.arange(N)[:, None], nn])
W = np.zeros((N, N))
for i in range(N):
    W[i, nn[i]] = np.exp(-D2[i, nn[i]] / sig2)
W = np.maximum(W, W.T)
d = W.sum(1)
S = W / np.sqrt(d[:, None] * d[None])
Y = np.zeros((N, C))
Y[lab, y[lab]] = 1
alpha = 0.99
F = np.linalg.solve(np.eye(N) - alpha * S, Y)
P = F / F.sum(1, keepdims=True)
pred = P.argmax(1)
H = -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)
omega = 1 - H / np.log(C)  # Konfidenz nach Iscen et al., Gl. 11
omega = omega / omega.max()  # wie im Training auf das Maximum skaliert
SCHWELLE = 0.7  # einer der Werte der Konfidenzschwelle in Phase 2
GROSS, KLEIN = 26.0, 3.0  # Markerflächen über und unter der Schwelle
fig, axes = plt.subplots(1, 2, figsize=(stil.TEXTWIDTH, 2.45))
for ax in axes:
    ii, jj = np.nonzero(np.triu(W))
    for i, j in zip(ii, jj):
        ax.plot(X[[i, j], 0], X[[i, j], 1], color=stil.LIGHT, lw=0.5, zorder=1)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_aspect("equal")
ax = axes[0]
unl = np.setdiff1d(np.arange(N), lab)
ax.scatter(X[unl, 0], X[unl, 1], s=9, color=stil.NEUTRAL_FILL, zorder=2, lw=0)
for c in range(C):
    m = lab[y[lab] == c]
    ax.scatter(X[m, 0], X[m, 1], s=42, color=CAT[c], edgecolor=stil.BLACK, lw=0.8, zorder=3, marker=stil.CAT_MARKERS[c])
stil.panel(ax, "a", "$k$NN-Graph mit zwei Labels je Klasse")
ax = axes[1]
acc = (pred == y).mean()
for c in range(C):
    for m, s in (((pred == c) & (omega >= SCHWELLE), GROSS), ((pred == c) & (omega < SCHWELLE), KLEIN)):
        ax.scatter(X[m, 0], X[m, 1], s=s, color=CAT[c], zorder=2, lw=0, marker=stil.CAT_MARKERS[c])
wrong = pred != y
# Ringe statt Kreuze: Kreuze verdecken die kleinen Marker mit niedriger Konfidenz
ax.scatter(X[wrong, 0], X[wrong, 1], s=30, marker="o", facecolor="none", edgecolor=stil.BLACK, lw=0.6, zorder=4)
for c in range(C):
    m = lab[y[lab] == c]
    ax.scatter(X[m, 0], X[m, 1], s=42, color=CAT[c], edgecolor=stil.BLACK, lw=0.8, zorder=3, marker=stil.CAT_MARKERS[c])
stil.panel(ax, "b", "Propagierte Pseudo-Labels")
leg = [Line2D([], [], marker="o", ls="", color=stil.NEUTRAL, ms=np.sqrt(GROSS), label=r"Konfidenz $\tilde\omega\geq\mathrm{0{,}7}$"),
       Line2D([], [], marker="o", ls="", color=stil.NEUTRAL, ms=np.sqrt(KLEIN), label=r"Konfidenz $\tilde\omega<\mathrm{0{,}7}$"),
       Line2D([], [], marker="o", ls="", mfc="none", mec=stil.BLACK, mew=0.6, ms=np.sqrt(30), label="falsches Pseudo-Label")]
fig.legend(handles=leg, loc="outside lower center", ncol=3)
stil.save(fig, "lp_toy")
print("LP toy: acc", round(acc, 3), "wrong", int(wrong.sum()), "unter Schwelle", int((omega < SCHWELLE).sum()),
      "davon falsch", int((wrong & (omega < SCHWELLE)).sum()))

# ------------------------------------------------------------------ Hoffer-Skala
s = np.logspace(np.log10(0.25), np.log10(32), 300)
fig, ax = plt.subplots(figsize=stil.SIZES["narrow"])
for Cn, col, name in ((73, stil.TOL["blue"], "73 Klassen"),
                      (3000, stil.TOL["red"], "3000 Klassen")):
    dist = 4.0  # punktweise Schranke: eigene Referenz gleich, übrige antipodal
    p = 1 / (1 + (Cn - 1) * np.exp(-s * dist))
    ax.plot(s, p, color=col, label=name)
    ax.axhline(1 / Cn, color=col, lw=0.8, ls=(0, (4, 2)))
    s90 = np.log(9 * (Cn - 1)) / 4
    ax.plot([s90], [0.9], marker="o", color=col)
    print(Cn, "s fuer p=0,9:", round(s90, 2), "p bei s=1:", round(1 / (1 + (Cn - 1) * np.exp(-dist)), 4))
ax.set_xscale("log", base=2)
ax.set_xticks([0.25, 0.5, 1, 2, 4, 8, 16, 32], ["0,25", "0,5", "1", "2", "4", "8", "16", "32"])
ax.minorticks_off()
ax.set_xlabel("Distanzskala $\\gamma_{\\mathrm{H}}$")
ax.set_ylabel("Obere Schranke für $p^{(l)}_{ic}$")
ax.legend(loc="lower right", bbox_to_anchor=(1, 0.08))
stil.comma_axis(ax, "y", 1)
stil.save(fig, "hoffer_skala")

# ------------------------------------------------------------------ Geometrie-Konzept
rng = np.random.default_rng(1)


def rho(E):
    sv = np.linalg.svd(E, compute_uv=False)[: E.shape[1] - 1][1:]  # verwendeter Roth-Pfad: höchstens d-1 Werte, dann größten entfernen
    p = sv / sv.sum()
    u = np.full_like(p, 1 / len(p))
    return float(np.sum(u * np.log(u / p)))


def mittlerer_paarabstand(q):
    d = np.linalg.norm(q[:, None] - q[None], axis=-1)
    return d[np.triu_indices(len(q), 1)].mean()


def zwei(v):
    return f"{v:.2f}".replace(".", ",")


# (a), (b): drei Klassen mit denselben Zentren und denselben Abweichungen, in (b) nur stärker skaliert. Kreisradius =
# mittlerer euklidischer Paarabstand der Klasse, gestrichelt = alle Abstände zwischen den Zentren. "Streuung" ist der
# Mittelwert der Radien, "Abstand" der Mittelwert der gestrichelten Längen (pi_ratio wie bei Roth et al., hier euklidisch in 2-D).
fig = plt.figure(figsize=(stil.TEXTWIDTH, 1.9))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.3], wspace=0.15)
N = 12
cent = np.array([[0.0, 0.0], [3.2, 0.2], [1.5, 2.8]])
paare = ((0, 1), (0, 2), (1, 2))
abw = rng.normal(size=(len(cent), N, 2))
for j, (spread, title) in enumerate(((0.35, "Kompakte Klassen"), (0.85, "Gestreute Klassen"))):
    ax = fig.add_subplot(gs[0, j])
    radien = []
    for c, m in enumerate(cent):
        q = m + spread * abw[c]
        r = mittlerer_paarabstand(q)
        radien.append(r)
        ax.add_patch(Circle(m, r, facecolor=CAT[c], alpha=0.15, edgecolor="none"))
        ax.add_patch(Circle(m, r, facecolor="none", edgecolor=CAT[c], lw=0.8))
        ax.scatter(q[:, 0], q[:, 1], s=8, color=CAT[c], marker=stil.CAT_MARKERS[c], lw=0, zorder=3)
    for a, b in paare:
        ax.plot(cent[[a, b], 0], cent[[a, b], 1], color=stil.BLACK, lw=0.7, ls=(0, (4, 2)), zorder=4)
    ax.scatter(cent[:, 0], cent[:, 1], s=30, marker="+", color=stil.BLACK, lw=1.0, zorder=5)
    intra = np.mean(radien)
    inter = np.mean([np.linalg.norm(cent[a] - cent[b]) for a, b in paare])
    m, r = cent[0], radien[0]
    spitze = m + r * np.array([-np.sqrt(0.5), -np.sqrt(0.5)])
    ax.annotate("", xy=spitze, xytext=m, arrowprops=dict(arrowstyle="-|>", lw=0.8, color=stil.BLACK,
                                                        shrinkA=0, shrinkB=0, mutation_scale=6), zorder=6)
    ax.text(spitze[0] - 0.1, spitze[1] - 0.05, "Streuung", ha="right", va="top", fontsize=8, zorder=6)
    mitte = (cent[0] + cent[2]) / 2
    ax.text(mitte[0] - 0.3, mitte[1] + 0.1, "Abstand", ha="right", va="center", fontsize=8, zorder=6)
    ax.text(1.0, -0.09, r"$\pi_{\mathrm{ratio}}^{\mathrm{Eucl}}=\frac{\mathrm{Streuung}}{\mathrm{Abstand}}\approx$ " + zwei(intra / inter),
            transform=ax.transAxes, ha="right", va="bottom", zorder=7)
    stil.panel(ax, "ab"[j], title)
    ax.set_xlim(-2.3, 5.1); ax.set_ylim(-2.5, 4.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    print("geometrie_konzept", title, "Streuung", round(intra, 3), "Abstand", round(inter, 3), "Verhältnis", round(intra / inter, 3))
# (c): normierte Singulärwerte so, wie sie in rho eingehen (höchstens d-1 Werte, ohne den größten), dazu die
# Gleichverteilung, gegen die rho die KL-Divergenz misst
ax = fig.add_subplot(gs[0, 2])
dim = 64
kurven = {}
for decay, col, name in ((0.075, stil.TOL["red"], "wenige Richtungen"),
                         (0.02, stil.TOL["blue"], "viele Richtungen")):
    E = rng.normal(size=(2000, dim)) * np.exp(-decay * np.arange(dim))
    sv = np.linalg.svd(E, compute_uv=False)[: dim - 1][1:]
    p = sv / sv.sum()
    ax.plot(np.arange(2, 2 + len(p)), p, color=col)
    kurven[name] = (col, rho(E))
    print("geometrie_konzept", name, "rho", round(rho(E), 4))
ax.axhline(1 / len(p), color=stil.NEUTRAL, lw=0.9, ls=":")
for name, (x, y) in (("wenige Richtungen", (9, 0.064)), ("viele Richtungen", (30, 0.034))):
    ax.text(x, y, name + "\n" + r"$\rho\approx$ " + zwei(kurven[name][1]), color=kurven[name][0], ha="left", va="center")
ax.text(64, 0.0175, r"gleichverteilt, $\rho=0$", color=stil.NEUTRAL, ha="right", va="bottom")
ax.set_xlabel("Index des Singulärwerts")
ax.set_ylabel("Anteil am Spektrum")
ax.set_xlim(0, 65)
ax.set_ylim(0, 0.08)
ax.set_yticks([0, 0.02, 0.04, 0.06, 0.08])
stil.comma_axis(ax, "y", 2)
stil.panel(ax, "c", "Singulärwertspektrum")
stil.save(fig, "geometrie_konzept")
print("ok")
