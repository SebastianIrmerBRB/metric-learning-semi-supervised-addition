"""Retrieval-Beispiele auf Cars196: Anfrage + 6 nächste Nachbarn für DINOv2 / Ü / LP (Circle, k=16, Seed 2, Concat)."""
import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
import stil, daten

stil.apply()
paths, L = daten.cars_test()
EMB = {key: key for key in ("dino", "sup", "lp")}
ROWLAB = {"dino": "DINOv2", "sup": "ÜBl.", "lp": "LP"}
GREEN, RED = stil.TOL["green"], stil.TOL["red"]


def thumb(p, size=(200, 130)):
    im = daten.image(p)
    im.thumbnail((size[0] * 2, size[1] * 2))
    w, h = im.size
    tr = size[0] / size[1]
    if w / h > tr:
        nw = int(h * tr); im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = int(w / tr); im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    return im.resize(size)


def model_name(p, short=False):
    n = p.split("/")[-2]
    if short:
        n = n.replace("Mercedes-Benz", "Mercedes")
        w = n.split(" ")
        n = " ".join(w[:3] if w[0] in ("Land", "Aston", "Rolls-Royce", "AM") else w[:2])
    return n


def figure(queries, rows, fname, n=6, width=stil.TEXTWIDTH):
    nr = len(queries) * len(rows)
    # feste Platzierung in Zoll: Bildseitenverhältnis 200:130 bleibt erhalten
    left, qgap, gap = 0.64, 0.14, 0.05
    cw = (width - left - qgap - (n - 1) * gap) / (n + 1)
    ch = cw * 130 / 200
    title, lab, block = 0.2, 0.3, 0.12
    H = title + nr * (ch + lab) + (len(queries) - 1) * block
    fig = plt.figure(figsize=(width, H), layout="none")
    axes = np.empty((nr, n + 1), dtype=object)
    for rr in range(nr):
        y = H - title - rr * (ch + lab) - (rr // len(rows)) * block - ch
        for cc in range(n + 1):
            x = left + cc * cw + (qgap + (cc - 1) * gap if cc else 0)
            axes[rr, cc] = fig.add_axes([x / width, y / H, cw / width, ch / H])
    r = 0
    for q in queries:
        for key in rows:
            nb = daten.neighbours(EMB[key], q, n)
            ax = axes[r, 0]
            ax.imshow(thumb(paths[q]), aspect="auto"); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(stil.BLACK); s.set_linewidth(1.6)
            if key == rows[0] and r == 0:
                ax.set_title("Anfrage", loc="center", fontsize=9, pad=2)
            ax.set_ylabel(ROWLAB[key], rotation=0, ha="right", va="center", labelpad=4)
            hits = 0
            for j, i in enumerate(nb):
                a = axes[r, j + 1]
                a.imshow(thumb(paths[i]), aspect="auto"); a.set_xticks([]); a.set_yticks([])
                ok = L[i] == L[q]
                hits += ok
                for s in a.spines.values():
                    s.set_visible(True); s.set_color(GREEN if ok else RED); s.set_linewidth(2.2)
                a.grid(False)
                if not ok:
                    a.text(0.5, -0.05, model_name(paths[i], short=True).replace(" ", "\n", 1),
                           transform=a.transAxes, ha="center", va="top", fontsize=8, linespacing=1.0)
                if key == rows[0] and r == 0:
                    a.set_title(f"Rang {j + 1}", loc="center", fontsize=9, pad=2)
            ax.grid(False)
            r += 1
    fig.set_layout_engine("none")  # feste Platzierung: ohne Layout-Engine speichern
    fig.savefig(stil.OUT + fname + ".pdf")
    if stil.PREVIEW:
        os.makedirs(stil.PREVIEW, exist_ok=True)  # auch wenn dieses Skript allein läuft
        with warnings.catch_warnings():  # Hinweis von matplotlib zu frei platzierten Achsen beim PNG, ohne Wirkung
            warnings.filterwarnings("ignore", message="There are no gridspecs with layoutgrids")
            fig.savefig(stil.PREVIEW + "/" + fname + ".png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    # ohne Argumente die Abbildung der Arbeit (erzeuge_aus_daten.py übergibt dieselben Werte)
    args = sys.argv[1:] or ["758,5348", "dino,sup,lp", "retrieval_ergebnisse"]
    qs = [int(x) for x in args[0].split(",")]
    rows = args[1].split(",")
    figure(qs, rows, args[2])
    for q in qs:
        print(q, model_name(paths[q]))
