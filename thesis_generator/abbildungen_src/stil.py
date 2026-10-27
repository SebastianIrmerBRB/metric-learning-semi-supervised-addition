"""Gemeinsamer Stil für alle Abbildungen der Arbeit.

Schrift: TeX Gyre Pagella (= Palatino, identisch mit newpxtext im Fließtext).
Farben: Paul Tol "bright" (farbenblind-sicher), Referenz dunkelgrau.
Maße: \\textwidth = 150 mm = 5.906 in (A4, 3 cm Rand auf allen Seiten).
Regel: Abbildungen werden in genau dieser Größe erzeugt und in LaTeX
OHNE Skalierung eingebunden: \\includegraphics{datei} (kein width=).
"""
import datensammlung as sammlung
import glob
import os

import matplotlib as mpl
from cycler import cycler
from matplotlib import font_manager
from matplotlib.ticker import ScalarFormatter
import numpy as np

# ------------------------------------------------------------------ Schrift
_FONTDIRS = [  # nur falls die Schrift im Datenpaket fehlt
    "/usr/share/texmf/fonts/opentype/public/tex-gyre/",
    "/usr/share/texlive/texmf-dist/fonts/opentype/public/tex-gyre/",
]
FONT = "TeX Gyre Pagella"
_found = False
for f in sammlung.font_files():
    font_manager.fontManager.addfont(f)
    _found = True
for d in ([] if _found else _FONTDIRS):
    for f in glob.glob(os.path.join(d, "texgyrepagella-*.otf")):
        font_manager.fontManager.addfont(f)
        _found = True
if not _found:
    raise RuntimeError("TeX Gyre Pagella nicht gefunden - _FONTDIRS anpassen.")

# ------------------------------------------------------------------ Maße (Zoll)
TEXTWIDTH = 150 / 25.4           # 5.906 in = 150 mm
TEXTHEIGHT = 237 / 25.4          # 9.331 in
SIZES = {
    "full":    (TEXTWIDTH, 2.60),   # 1 Zeile Diagramme, volle Breite
    "full2":   (TEXTWIDTH, 4.60),   # 2 Zeilen Diagramme (z. B. 2x2)
    "full3":   (TEXTWIDTH, 6.60),   # 3 Zeilen Diagramme
    "narrow":  (0.62 * TEXTWIDTH, 2.60),  # Einzeldiagramm, zentriert
}

# ------------------------------------------------------------------ Farben
BLACK = "#000000"
NEUTRAL = "#4D4D4D"        # überwachte Referenz (Linien, Punkte)
NEUTRAL_FILL = "#BBBBBB"   # überwachte Referenz (Balken, Flächen)
LIGHT = "#DDDDDD"          # Referenzlinien, ausgegraute Bereiche

TOL = {
    "blue": "#4477AA", "red": "#EE6677", "green": "#228833",
    "purple": "#AA3377", "yellow": "#CCBB44", "cyan": "#66CCEE",
}
PALETTE = [TOL["blue"], TOL["red"], TOL["green"], TOL["purple"], TOL["yellow"], TOL["cyan"]]
MARKERS = ["s", "^", "D", "v", "P", "X"]

# Kompatibel zu den bisherigen Namen
INK = BLACK
MUTED = NEUTRAL
GRID = LIGHT

METHOD_ORDER = ["sup", "iscen", "lrml", "hoffer", "slade"]
METHOD_COLORS = {
    "sup": NEUTRAL,
    "iscen": TOL["blue"], "lp": TOL["blue"],
    "lrml": TOL["red"],
    "hoffer": TOL["green"],
    "slade": TOL["purple"],
}
METHOD_MARKERS = {"sup": "o", "iscen": "s", "lp": "s", "lrml": "^", "hoffer": "D", "slade": "v"}
METHOD_LABELS = {"sup": "ÜBl.", "iscen": "LP", "lp": "LP", "lrml": "LRML", "hoffer": "Hoffer", "slade": "SLADE"}

LOSS_ORDER = ["arc", "cir", "ms", "nt", "prx"]
LOSS_LABELS = {"arc": "ArcFace", "cir": "Circle", "ms": "MS", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
LOSS_COLORS = dict(zip(LOSS_ORDER, PALETTE[:5]))
LOSS_MARKERS = dict(zip(LOSS_ORDER, MARKERS[:5]))

DATASET_COLORS = {"cars": TOL["blue"], "cub": TOL["red"], "inshop": TOL["green"]}
DATASET_MARKERS = {"cars": "o", "cub": "s", "inshop": "^"}

# Klassen in Streudiagrammen (bis 9): Paul Tol "muted"
CAT = ["#CC6677", "#332288", "#DDCC77", "#117733", "#88CCEE",
       "#882255", "#44AA99", "#999933", "#AA4499"]
CAT_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*", "h"]

# Divergierend (Differenzen, Gewinn/Verlust): Farbskala "RdBu" mit Mittelpunkt 0
DIV_CMAP = "RdBu"          # negativ = rot, positiv = blau
DIV_NEG = "#B2182B"
DIV_POS = "#2166AC"
DIV_MID = "#F7F7F7"
SEQ_CMAP = "viridis"       # sequenziell (Heatmaps ohne Vorzeichen)


# ------------------------------------------------------------------ rcParams
RC = {
    # Schrift
    "font.family": "serif",
    "font.serif": [FONT],
    "mathtext.fontset": "custom",
    "mathtext.rm": FONT,
    "mathtext.it": f"{FONT}:italic",
    "mathtext.bf": f"{FONT}:bold",
    "mathtext.cal": f"{FONT}:italic",
    "mathtext.sf": FONT,
    "mathtext.default": "it",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlepad": 4,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.title_fontsize": 9,
    "figure.titlesize": 10,
    # Farben: alles Schwarz
    "text.color": BLACK,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.prop_cycle": cycler(color=PALETTE),
    "image.cmap": SEQ_CMAP,
    # Rahmen und Striche: geschlossener Rahmen, Striche innen, keine Nebenstriche
    "axes.linewidth": 0.6,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "xtick.minor.size": 1.5,
    "ytick.minor.size": 1.5,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.pad": 3,
    "ytick.major.pad": 3,
    # Gitter: aus
    "axes.grid": False,
    "grid.color": LIGHT,
    "grid.linewidth": 0.4,
    "axes.axisbelow": True,
    # Linien und Marker
    "lines.linewidth": 1.25,
    "lines.markersize": 4.0,
    "lines.markeredgewidth": 0.6,
    "lines.solid_capstyle": "butt",
    "patch.linewidth": 0.5,
    "hatch.linewidth": 0.5,
    "errorbar.capsize": 2.0,
    "scatter.edgecolors": "face",
    "boxplot.flierprops.markersize": 3.0,
    # Legende: ohne Rahmen
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.5,
    "legend.borderaxespad": 0.3,
    "legend.columnspacing": 1.2,
    "legend.labelspacing": 0.3,
    "legend.markerscale": 1.0,
    # Zahlen
    "axes.unicode_minus": True,
    "axes.formatter.use_mathtext": True,
    "axes.formatter.limits": (-4, 5),
    "axes.xmargin": 0.02,
    "axes.ymargin": 0.04,
    # Layout und Export
    "figure.constrained_layout.use": True,
    "figure.constrained_layout.h_pad": 0.03,
    "figure.constrained_layout.w_pad": 0.03,
    "figure.constrained_layout.hspace": 0.04,
    "figure.constrained_layout.wspace": 0.04,
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.format": "pdf",
    "savefig.bbox": None,
    "savefig.pad_inches": 0.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}


def apply():
    """Stil global setzen (einmal am Anfang jedes Skripts aufrufen)."""
    mpl.rcdefaults()
    mpl.rcParams.update(RC)


# ------------------------------------------------------------------ Zahlen
def de(x, nd=1):
    """Zahl mit deutschem Dezimalkomma und echtem Minuszeichen."""
    if round(x, nd) == 0:
        x = 0.0  # no "−0,0" for values that round to zero
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def smart(v, nd_max=2):
    """Deutsches Komma, nur so viele Nachkommastellen wie nötig (höchstens nd_max)."""
    s = f"{v:.{nd_max}f}".rstrip("0").rstrip(".")
    s = "0" if s in ("-0", "") else s
    return s.replace(".", ",").replace("-", "−")


class CommaFormatter(ScalarFormatter):
    """ScalarFormatter mit Dezimalkomma (behält Offset/Skalierung bei)."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")

    def get_offset(self):
        return super().get_offset().replace(".", ",")


def comma_axis(ax, axis="both", nd=None):
    """Dezimalkomma für lineare Achsen. nd=None: automatisch; nd=k: feste Logik wie bisher."""
    from matplotlib.ticker import FuncFormatter
    axes = {"x": [ax.xaxis], "y": [ax.yaxis], "both": [ax.xaxis, ax.yaxis]}[axis]
    for a in axes:
        if a.get_scale() != "linear":
            continue
        if nd is None:
            a.set_major_formatter(CommaFormatter(useMathText=True))
        else:
            a.set_major_formatter(FuncFormatter(lambda v, p, k=nd: smart(v, max(k, 2))))


def panel(ax, letter, text=""):
    """Teilbildtitel links oben: '(a) Cars196'."""
    ax.set_title(f"({letter})" + (f" {text}" if text else ""))


def finish(fig, path):
    """Dezimalkomma auf allen linearen Achsen ohne eigenen Formatter, dann in exakter Größe speichern."""
    from matplotlib.ticker import ScalarFormatter as _SF
    for ax in fig.get_axes():
        for a, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            f = a.get_major_formatter()
            if type(f) is _SF:
                comma_axis(ax, name)
    fig.savefig(path)


# ------------------------------------------------------------------ Speichern
OUT = sammlung.output_dir("bilder/neu")
PREVIEW = os.environ.get("STIL_PREVIEW")  # optional: Ordner für PNG-Vorschauen


def grenzen(lo, hi, werte, raster, rand=0.0):
    """Feste Achsengrenzen (gleiche Achsen über mehrere Teilbilder), aber nie enger als die gezeigten Werte (plus rand):
    wo nötig nach außen auf ein Vielfaches von raster erweitert. Mit den Daten der Arbeit bleiben die festen Grenzen."""
    w = np.asarray(werte, dtype=float).ravel()
    w = w[np.isfinite(w)]
    if w.size:
        lo = min(lo, raster * np.floor((w.min() - rand) / raster))
        hi = max(hi, raster * np.ceil((w.max() + rand) / raster))
    return lo, hi


def save(fig, name):
    """Als bilder/neu/<name>.pdf in exakter Größe speichern (LaTeX: \\includegraphics{neu/<name>})."""
    finish(fig, OUT + name + ".pdf")
    if PREVIEW:
        os.makedirs(PREVIEW, exist_ok=True)
        fig.savefig(os.path.join(PREVIEW, name + ".png"), dpi=110)
