"""Tabelle der 2x2-Zerlegung für LRML in der In-Shop-Serie mit 10 % gelabelten Klassen (Abschnitt sec:e:klassen).
Gestaltung wie die Ablationstabellen aus tab_phase02.py. Daten: Sicht phase2_klassen/variants.csv (sammle_phase2_klassen.py),
\\UEBL und Rauschschwellen aus den fünf Seeds der Phase 1 bei 10 %. Gibt die Kennzahlen für den Text aus.

    python3 tab_klassen_2x2.py
"""
import datensammlung as sammlung
import numpy as np

TAB = sammlung.output_dir("kapitel/ergebnisse_tabellen")
LOSSES = ["arc", "cir", "ms", "nt", "prx"]
LOSS_NAME = {"arc": "ArcFace", "cir": "Circle", "ms": "\\acs{MS}", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
SCOPES = [("mean", "Einzelmodell (Mean)"), ("concat", "Ensemble (Concat)")]

V = sammlung.read_csv("phase2_klassen/variants.csv").set_index(["loss", "variant"])
R = sammlung.read_csv("inshop10_ergebnisse/data/phase1_runs.csv")
R = R[R.budget_percent == 10]


def seeds(method, loss, scope):
    g = R[(R.method == f"{method}_{loss}") & (R.scope == scope)]
    assert sorted(g.seed) == [2, 8, 9, 10, 11], (method, loss, scope)
    return 100 * g.set_index("seed").mean_average_precision_at_r


def cell(x):
    return f"{0.0 if abs(x) < 0.05 else x:+.1f}"


rows = ["\\begin{table}[tbp]\n\\centering",
        "\\caption[Zerlegung des Gewinns von \\acs{LRML} bei Klassen ohne Labels]{Zerlegung des Gewinns von \\acs{LRML} "
        "in der In-Shop-Serie mit \\qty{10}{\\percent} vollständig gelabelten Klassen (Seed~2): Änderung der Test-MAP@R "
        "gegenüber der regulären Konfiguration in Prozentpunkten, \\UEBL und regulär absolut in \\%. Varianten wie in "
        "Tabelle~\\ref{tab:m:2x2}. Rauschschwelle "
        "$2\\sqrt{2}\\,\\sigma$ aus den fünf Seeds der Phase~1 bei \\qty{10}{\\percent} (Abschnitt~\\ref{sec:m:phase2}).}",
        "\\label{tab:e:klassen:2x2}",
        "\\footnotesize\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{0.95}\n"
        "\\begin{tabular}{@{}l *{5}{S[table-format=+2.1, retain-explicit-plus]}@{}}\n\\toprule",
        "{Variante} & " + " & ".join(f"{{{LOSS_NAME[l]}}}" for l in LOSSES) + " \\\\\n\\midrule"]
facts = []
for n, (scope, title) in enumerate(SCOPES):
    sup = {l: seeds("sup", l, scope)[2] for l in LOSSES}
    sd = {l: seeds("lrml", l, scope).std(ddof=1) for l in LOSSES}
    sd_sup = {l: seeds("sup", l, scope).std(ddof=1) for l in LOSSES}
    reg = {l: V.loc[(l, "defaults"), f"{scope}_value"] for l in LOSSES}
    d = {(l, v): V.loc[(l, v), f"{scope}_value"] - reg[l] for l in LOSSES for v in ("sup_all_shared", "pool_labeled")}
    thr = {l: 2 * np.sqrt(2) * sd[l] for l in LOSSES}
    thr_sup = {l: 2 * np.sqrt(sd[l] ** 2 + sd_sup[l] ** 2) for l in LOSSES}
    rows.append(("\\addlinespace\n" if n else "") + f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\")
    rows.append("\\quad \\UEBL (MAP@R) & " + " & ".join(f"{sup[l]:.1f}" for l in LOSSES) + " \\\\")
    rows.append("\\quad regulär (MAP@R) & " + " & ".join(f"{reg[l]:.1f}" for l in LOSSES) + " \\\\")
    rows.append("\\quad Rauschschwelle & " + " & ".join(f"{thr[l]:.1f}" for l in LOSSES) + " \\\\")
    rows.append("\\quad Hyperparameter der \\UEBL & " + " & ".join(cell(d[(l, 'sup_all_shared')]) for l in LOSSES) + " \\\\")
    rows.append("\\quad ohne ungelabelte Bilder & " + " & ".join(cell(d[(l, 'pool_labeled')]) for l in LOSSES) + " \\\\")
    for l in LOSSES:
        facts.append(f"{scope:6s} {l:3s}  Ü {sup[l]:5.2f}  regulär {reg[l]:5.2f}  regulär−Ü {reg[l] - sup[l]:+5.2f} "
                     f"(Schwelle {thr_sup[l]:.2f}{', über' if abs(reg[l] - sup[l]) > thr_sup[l] else ''})  "
                     f"Ü-HP−regulär {d[(l, 'sup_all_shared')]:+6.2f}  ohne−regulär {d[(l, 'pool_labeled')]:+5.2f} "
                     f"(Schwelle {thr[l]:.2f}{', über' if abs(d[(l, 'pool_labeled')]) > thr[l] else ''})")
rows.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(TAB + "klassen_2x2.tex", "w", encoding="utf-8").write("\n".join(rows) + "\n")
print("\n".join(facts))
