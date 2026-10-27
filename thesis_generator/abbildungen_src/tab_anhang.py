"""Erzeugt die Ergebnistabellen des Anhangs (MAP@R Mean / Concat je Budget). Phase 2: tab_phase02.py.

Seit 18.10.2026 jede Zelle als Mittel ± Standardabweichung über die fünf Seeds (ddof=1); Tabellen mit vier Budgets quer
(sidewaystable), weil acht Spalten mit Unsicherheit nicht in die Textbreite passen."""
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


def f(x, sd):
    """Zahl mit Unsicherheit für eine siunitx-S-Spalte (Eingabe mit Punkt, Ausgabe mit Komma und ±)."""
    return "{--}" if np.isnan(x) else f"{x:.2f} +- {sd:.2f}"


for key, name, kc, ks, fmt in DATA:
    r = sammlung.read_csv(R + f"{key}_ergebnisse/data/phase1_runs.csv")
    r["map"] = 100 * r.mean_average_precision_at_r
    g = r.groupby(["method", kc, "scope"])["map"].mean()
    gs = r.groupby(["method", kc, "scope"])["map"].std()
    best = {(k, sc): g.xs((k, sc), level=[1, 2]).max() for k in ks for sc in ("mean", "concat")}
    lines = []
    quer = len(ks) > 2
    lines.append("\\begin{sidewaystable}\n\\centering" if quer else "\\begin{table}[p]\n\\centering")
    extra = ""  # seit 15.10.2026 alle Mean-Werte exakt aus cv_summary.json, keine gerundeten Fold-Werte mehr
    sdnote = ("" if key != "inshop10" else
              " Die großen Standardabweichungen bei LP mit ArcFace bei 25 und 50\\,\\% entstehen, weil dort einzelne Fold-Modelle deutlich schlechter sind.")
    lines.append(f"\\caption[Test-MAP@R {name}]{{Test-MAP@R in \\% auf {name}, Mittel $\\pm$ Standardabweichung über fünf Seeds. M: einzelne Fold-Modelle (Mean), C: Ensemble (Concat). Fett: bester Mittelwert je Spalte.{sdnote}{extra}}}")
    lines.append(f"\\label{{tab:anhang:{key}}}\n\\footnotesize\\setlength{{\\tabcolsep}}{{4pt}}")
    lines.append("\\begin{tabular}{@{}ll" + "S[table-format=2.2(3), retain-zero-uncertainty]" * (2 * len(ks)) + "@{}}\n\\toprule")
    lines.append(" & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{fmt.format(k)}}}" for k in ks) + " \\\\")
    lines.append("".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(ks))))
    lines.append("Verlust & Verfahren & " + " & ".join("{M} & {C}" for _ in ks) + " \\\\\n\\midrule")
    for li, (lk, ln) in enumerate(LOSS.items()):
        for fk, fn in FAM.items():
            m = f"{fk}_{lk}"
            if (m, ks[0], "mean") not in g.index:
                continue
            cells = []
            for k in ks:
                a, b = g.get((m, k, "mean"), np.nan), g.get((m, k, "concat"), np.nan)
                sa = f(a, gs.get((m, k, "mean"), np.nan)); sb = f(b, gs.get((m, k, "concat"), np.nan))
                if abs(a - best[(k, "mean")]) < 1e-9: sa = "\\bfseries " + sa
                if abs(b - best[(k, "concat")]) < 1e-9: sb = "\\bfseries " + sb
                cells.append(f"{sa} & {sb}")
            lines.append(f"{ln if fk == 'sup' else ''} & {fn} & " + " & ".join(cells) + " \\\\")
        if li < 4:
            lines.append("\\addlinespace[2pt]")
    lines.append("\\bottomrule\n\\end{tabular}\n" + ("\\end{sidewaystable}" if quer else "\\end{table}"))
    sd = r.groupby(["method", kc, "scope"])["map"].std().max()
    print(key, "max SD", round(sd, 2))
    open(OUT + f"map_{key}.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

# Phase 2: Die Variantentabelle (phase2_varianten.tex) erzeugt seit 29.09.2026 tab_phase02.py.
