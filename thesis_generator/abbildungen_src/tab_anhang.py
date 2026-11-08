"""Erzeugt die Ergebnistabellen des Anhangs (MAP@R Mean / Concat je Budget). Phase 2: tab_phase02.py.

Jede Zelle zeigt Mittel ± Standardabweichung über fünf Seeds (ddof=1). Die Verlustfunktionen stehen in den Spalten,
die Budgets in aufeinanderfolgenden Zeilenblöcken mit Mean / Concat je Verfahren. So passen alle Tabellen ins Hochformat.
"""
import datensammlung as sammlung
import numpy as np
import pandas as pd
import stil

R = ""
OUT = sammlung.output_dir("kapitel/anhang_tabellen")
import os
os.makedirs(OUT, exist_ok=True)
FAM = {"sup": "\\UEBL", "iscen": "LP", "lp": "LP", "lrml": "LRML", "hoffer": "Hoffer", "slade": "SLADE"}
LOSS = {"arc": "ArcFace", "cir": "Circle", "ms": "MS", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
DATA = [("cars", "Cars196", "k", [4, 8, 16, 32], "$k={}$"), ("cub", "CUB", "k", [4, 8, 16, 32], "$k={}$"),
        ("inshop", "In-Shop", "k", [2, 4], "$k={}$"), ("inshop10", "In-Shop mit gelabelten Klassen", "budget_percent", [5, 10, 25, 50], "{}\\,\\%"),
        ("semiinat", "Semi-iNat-2021", "k", [2, 4, 16, 100], "$k={}$")]


FORMAT = "table-format=2.2(3), retain-zero-uncertainty"


def f(x, sd):
    """Zahl mit Unsicherheit für eine siunitx-S-Spalte (Eingabe mit Punkt, Ausgabe mit Komma und ±)."""
    return "{--}" if np.isnan(x) else f"{x:.2f} +- {sd:.2f}"


def markiert(s, x, je_verlust, gesamt):
    """Fett: bestes Verfahren je Verlust, Budget und Modelltyp. Unterstrichen: bester Wert je Budget und Modelltyp.
    Die Linie zieht \\uline (ulem) mit \\ULdepth=1.5pt dicht unter der Grundlinie, \\underline läge unter der ganzen Zeile. Die Zahl steht
    dort in \\mbox{\\tablenum} (für ulem eine Einheit) mit dem Spaltenformat, damit sie wie die übrigen Zellen ausgerichtet bleibt."""
    if np.isnan(x) or abs(x - je_verlust) >= 1e-9:
        return s
    if abs(x - gesamt) < 1e-9:
        return f"{{\\bfseries\\ULdepth=1.5pt\\uline{{\\mbox{{\\tablenum[{FORMAT}]{{{s}}}}}}}}}"
    return "\\bfseries " + s


for key, name, kc, ks, fmt in DATA:
    r = sammlung.read_csv(R + f"{key}_ergebnisse/data/phase1_runs.csv")
    r["map"] = 100 * r.mean_average_precision_at_r
    g = r.groupby(["method", kc, "scope"])["map"].mean()
    gs = r.groupby(["method", kc, "scope"])["map"].std()
    best = {(k, sc): g.xs((k, sc), level=[1, 2]).max() for k in ks for sc in ("mean", "concat")}
    # bester Wert je Verlustfunktion: Methoden heißen <Verfahren>_<Verlust>
    loss_of = g.index.get_level_values(0).str.rsplit("_", n=1).str[1]
    best_loss = g.groupby([loss_of, g.index.get_level_values(1), g.index.get_level_values(2)]).max()
    lines = []
    lines.append("\\begin{table}[p]\n\\centering")
    extra = ""  # seit 15.10.2026 alle Mean-Werte exakt aus cv_summary.json, keine gerundeten Fold-Werte mehr
    sdnote = ("" if key != "inshop10" else
              " Die großen Standardabweichungen bei LP mit ArcFace bei 25 und 50\\,\\% entstehen, weil dort einzelne Fold-Modelle deutlich schlechter waren.")
    lines.append(f"\\caption[Test-MAP@R {name}]{{Test-MAP@R in \\% auf {name}, Mittel $\\pm$ Standardabweichung über fünf Seeds. M: einzelne Fold-Modelle (Mean), C: Ensemble (Concat). Fett: bester Mittelwert je Verlustfunktion, Budget und Modelltyp; zusätzlich unterstrichen: bester Mittelwert je Budget und Modelltyp.{sdnote}{extra}}}")
    lines.append(f"\\label{{tab:anhang:{key}}}\n\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}")
    lines.append("\\begin{tabular}{@{}ll" + f"S[{FORMAT}]" * len(LOSS) + "@{}}\n\\toprule")
    lines.append("Verfahren & Modell & " + " & ".join("{" + ln + "}" for ln in LOSS.values()) + " \\\\\n\\midrule")
    for ki, k in enumerate(ks):
        if ki:
            lines.append("\\midrule")
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{{fmt.format(k)}}} \\\\\n\\addlinespace[2pt]")
        first_method = True
        for fk, fn in FAM.items():
            if not any((f"{fk}_{lk}", k, "mean") in g.index for lk in LOSS):
                continue
            if not first_method:
                lines.append("\\addlinespace[1pt]")
            first_method = False
            for sc, model in (("mean", "M"), ("concat", "C")):
                cells = []
                for lk in LOSS:
                    m = f"{fk}_{lk}"
                    x = g.get((m, k, sc), np.nan)
                    value = f(x, gs.get((m, k, sc), np.nan))
                    cells.append(markiert(value, x, best_loss.get((lk, k, sc), np.nan), best[(k, sc)]))
                lines.append(f"{fn if sc == 'mean' else ''} & {model} & " + " & ".join(cells) + " \\\\")
    lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    sd = r.groupby(["method", kc, "scope"])["map"].std().max()
    print(key, "max SD", round(sd, 2))
    open(OUT + f"map_{key}.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

# Phase 2: Die Variantentabelle (phase2_varianten.tex) erzeugt seit 29.09.2026 tab_phase02.py.
