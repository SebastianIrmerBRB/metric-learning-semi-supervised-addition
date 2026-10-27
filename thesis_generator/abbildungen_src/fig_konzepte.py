"""Konzeptabbildungen mit synthetischen Daten bzw. aus Formeln (keine Versuchsergebnisse).

lp_toy.pdf            Label Propagation nach Zhou et al. auf einem kNN-Graphen synthetischer 2-D-Daten
hoffer_skala.pdf      punktweise obere Schranke der Zuordnungswahrscheinlichkeit bei Hoffer und Ailon über s_H
geometrie_konzept.pdf pi_intra / pi_inter und spektraler Abfall rho an synthetischen Embeddings
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import stil
import sys
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2

stil.apply()
CAT = stil.PALETTE  # drei Klassen: blau, rot, grün

# ------------------------------------------------------------------ LP-Spielbeispiel
rng = np.random.default_rng(SEED)
n = 70
t = rng.uniform(0, np.pi, n)
A = np.c_[np.cos(t), np.sin(t)] + 0.09 * rng.normal(size=(n, 2))
t = rng.uniform(0, np.pi, n)
B = np.c_[1 - np.cos(t), 0.45 - np.sin(t)] + 0.09 * rng.normal(size=(n, 2))
Cc = np.c_[2.25, 0.62] + 0.2 * rng.normal(size=(40, 2))
X = np.r_[A, B, Cc]
y = np.r_[np.zeros(n, int), np.ones(n, int), 2 * np.ones(40, int)]
N, C = len(X), 3
lab = np.r_[rng.choice(np.where(y == 0)[0], 2, replace=False), rng.choice(np.where(y == 1)[0], 2, replace=False),
            rng.choice(np.where(y == 2)[0], 2, replace=False)]
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
omega = 1 - H / np.log(C)
omega = omega / omega.max()
thr = 0.5
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
    m = (pred == c) & (omega >= thr)
    ax.scatter(X[m, 0], X[m, 1], s=11, color=CAT[c], zorder=2, lw=0, marker=stil.CAT_MARKERS[c])
    m = (pred == c) & (omega < thr)
    ax.scatter(X[m, 0], X[m, 1], s=11, facecolor="white", edgecolor=CAT[c], zorder=2, lw=0.7, marker=stil.CAT_MARKERS[c])
wrong = pred != y
ax.scatter(X[wrong, 0], X[wrong, 1], s=34, marker="x", color=stil.BLACK, lw=0.8, zorder=4)
for c in range(C):
    m = lab[y[lab] == c]
    ax.scatter(X[m, 0], X[m, 1], s=42, color=CAT[c], edgecolor=stil.BLACK, lw=0.8, zorder=3, marker=stil.CAT_MARKERS[c])
stil.panel(ax, "b", "Propagierte Pseudo-Labels")
leg = [Line2D([], [], marker="o", ls="", color=stil.NEUTRAL, label=r"Konfidenz $\tilde\omega\geq\mathrm{0{,}5}$"),
       Line2D([], [], marker="o", ls="", mfc="white", mec=stil.NEUTRAL, label=r"Konfidenz $\tilde\omega<\mathrm{0{,}5}$"),
       Line2D([], [], marker="x", ls="", color=stil.BLACK, ms=5, label="falsches Pseudo-Label")]
fig.legend(handles=leg, loc="outside lower center", ncol=3)
stil.save(fig, "lp_toy")
print("LP toy: acc", round(acc, 3), "wrong", int(wrong.sum()), "low conf", int((omega < thr).sum()),
      "wrong&low", int((wrong & (omega < thr)).sum()))

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
ax.set_xlabel("Skala $s_{\\mathrm{H}}$")
ax.set_ylabel("Obere Schranke für $p^{(r)}_{ic}$")
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


def pis(E, y):
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    cs = np.unique(y)
    intra = np.mean([np.mean(1 - (E[y == c] @ E[y == c].T)[np.triu_indices((y == c).sum(), 1)]) for c in cs])
    mu = np.stack([E[y == c].mean(0) for c in cs]); mu = mu / np.linalg.norm(mu, axis=1, keepdims=True)
    inter = np.mean(1 - (mu @ mu.T)[np.triu_indices(len(cs), 1)])
    return intra, inter, intra / inter


fig = plt.figure(figsize=(stil.TEXTWIDTH, 2.7))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.25])
cent = np.array([[0, 0], [3.0, 0.4], [1.2, 2.6], [3.6, 2.9]])
for j, (spread, title) in enumerate(((0.35, "Kompakte Klassen"), (0.85, "Gestreute Klassen"))):
    ax = fig.add_subplot(gs[0, j])
    pts, ys = [], []
    for c, m in enumerate(cent):
        q = m + spread * rng.normal(size=(18, 2))
        pts.append(q); ys += [c] * 18
        ax.scatter(q[:, 0], q[:, 1], s=9, color=CAT[c], marker=stil.CAT_MARKERS[c], lw=0)
    pts = np.vstack(pts); ys = np.array(ys)
    q = pts[ys == 0]
    for a in range(0, 6):
        ax.plot([q[a, 0], q[a + 6, 0]], [q[a, 1], q[a + 6, 1]], color=CAT[0], lw=0.6)
    for a, b in ((0, 1), (0, 2), (1, 3)):
        ax.plot(cent[[a, b], 0], cent[[a, b], 1], color=stil.BLACK, lw=0.6, ls=(0, (4, 2)))
    ax.scatter(cent[:, 0], cent[:, 1], s=24, marker="+", color=stil.BLACK, lw=0.9)
    # Kennzahlen im selben Sinn wie Roth et al. (hier euklidisch in 2-D)
    intra = np.mean([np.mean(np.linalg.norm(pts[ys == c][:, None] - pts[ys == c][None], axis=-1)[np.triu_indices(18, 1)]) for c in range(4)])
    inter = np.mean(np.linalg.norm(cent[:, None] - cent[None], axis=-1)[np.triu_indices(4, 1)])
    stil.panel(ax, "ab"[j], title)
    ax.text(0.0, -0.02, r"$\pi_{\mathrm{ratio}}^{\mathrm{Eucl}}\approx$ " + stil.smart(intra / inter, 2), transform=ax.transAxes, va="top")
    ax.set_xlim(-1.6, 5.2); ax.set_ylim(-1.6, 4.4)
    ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
ax = fig.add_subplot(gs[0, 2])
dim = 64
for decay, col, name in ((0.075, stil.TOL["red"], ""),
                         (0.02, stil.TOL["blue"], "")):
    scales = np.exp(-decay * np.arange(dim))
    E = rng.normal(size=(2000, dim)) * scales
    sv = np.linalg.svd(E, compute_uv=False)
    ax.plot(np.arange(1, dim + 1), sv / sv.sum(), color=col, label=r"$\rho=\mathrm{" + stil.smart(rho(E), 2).replace(",", "{,}") + "}$")
ax.set_yscale("log")
ax.set_xlabel("Index des Singulärwerts")
ax.set_ylabel("Anteil am Spektrum")
stil.panel(ax, "c", "Singulärwertspektrum")
ax.legend(loc="lower left")
stil.save(fig, "geometrie_konzept")
print("ok")
