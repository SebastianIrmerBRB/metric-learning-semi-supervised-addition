"""LRML auf Cars196 bei k=4: 2x2-Zerlegung mit abgeschaltetem SSL-Term (zero_weight), alle fünf Verlustfunktionen, Seeds 2 8 9 10 11 (15.10.2026).
Erklärt, worauf der größere Concat-Gewinn von LRML beruht (Abschnitt sec:e:phase2:concat).

Eingabe: Sicht phase2_k4_ensemble/runs.csv des Datenpakets (daten/ablations.csv, table_key k4_ensemble). Ursprung der Werte:
  die cv_summary.json der Läufe cars_k4_ensemble/lrml_<loss>_<variant> (Phase 2) und der Phase-1-Läufe von Ü auf Cars196.
  Geteilte Fehler: b7_ensemble/b7_pilot.py auf denselben Läufen und auf Ü (im Datenpaket
  phase2_k4_ensemble/geteilte_fehler.jsonl).
Seit 20.10.2026: LRML/MS ohne ungelabelte Bilder (Seed 10) und mit abgeschaltetem SSL-Term (Seed 8) aus der Wiederholung mit
  verzögerter Patience, weil deren Validierungs-MAP@R höher ist (verzoegerung.py). Ihre Zeilen in geteilte_fehler.jsonl
  stammen aus b7_pilot.py auf diesen Wiederholungen (Feld quelle).
Ausgabe: kapitel/ergebnisse_tabellen/k4_ensemble.tex, im Protokoll die gepaarten Differenzen für den Text."""
import glob
import json
import os

import numpy as np
import pandas as pd

import datensammlung as sammlung

TAB = sammlung.output_dir("kapitel/ergebnisse_tabellen") + "k4_ensemble.tex"
A = sammlung.read_csv("phase2_k4_ensemble/runs.csv")
# Gap is derived from the two editable performance columns.
A["gap"] = A.concat - A["mean"]

ORDER = [("ue", "\\UEBL"), ("zero_weight", "SSL-Term abgeschaltet"), ("pool_labeled", "ohne ungelabelte Bilder"),
         ("sup_all_shared", "Hyperparameter von \\UEBL"), ("defaults", "regulär")]
M = A.groupby("variant").mean(numeric_only=True)
L = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Ensemble-Gewinn von \\acs{LRML} bei $k=4$]{$2\\times2$-Zerlegung von \\ac{LRML} auf Cars196 bei $k=4$ mit der Kontrolle "
     "mit abgeschaltetem \\ac{SSL}-Term: Test-MAP@R der Einzelmodelle (Mean) und des Ensembles (Concat) in \\%, ihr Abstand in Prozentpunkten und "
     "der Anteil der falschen Nachbarn unter den ersten $R$ Treffern eines Fold-Modells, die ein anderes Fold-Modell ebenfalls "
     "zurückgibt, in \\%. Mittel über fünf Verlustfunktionen und die fünf Seeds der Phase~1. Bezeichnungen wie in "
     "Tabelle~\\ref{tab:e:ablationen:verluste}.}",
     "\\label{tab:e:k4ensemble}",
     "\\small\n\\begin{tabular}{@{}l *{3}{S[table-format=2.1]} S[table-format=2.1]@{}}\n\\toprule",
     "{Konfiguration} & {Mean} & {Concat} & {Concat $-$ Mean} & {geteilte Fehler} \\\\\n\\midrule"]
for v, lab in ORDER:
    r = M.loc[v]
    fp = f"{100 * r.fp_also_in_other_fold_topR:.1f}" if "fp_also_in_other_fold_topR" in M and not np.isnan(r.fp_also_in_other_fold_topR) else "{--}"
    L.append(f"{lab} & {r['mean']:.1f} & {r.concat:.1f} & {r.gap:.1f} & {fp} \\\\")
L.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(TAB, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))

# Zahlen für den Text: seed-gepaarte Differenzen je Verlustfunktion
def pair(a, b, col):
    X = A[A.variant == a].set_index(["loss", "seed"])[col]
    Y = A[A.variant == b].set_index(["loss", "seed"])[col]
    d = (X - Y).dropna()
    per = d.groupby("loss").mean()
    return d.mean(), int((d > 0).sum()), d.size, per.min(), per.max()


for a, b, lab in (("defaults", "ue", "SSL-Konfiguration vs Ü"), ("zero_weight", "ue", "Hyperparameter (SSL-Term abgeschaltet vs Ü)"),
                  ("pool_labeled", "zero_weight", "Term auf gelabelten Bildern"),
                  ("defaults", "pool_labeled", "ungelabelte Bilder"), ("sup_all_shared", "ue", "Term+Bilder mit Ü-HP"),
                  ("defaults", "sup_all_shared", "gemeinsam abgestimmte HP")):
    for col in ["mean", "concat", "gap"] + (["fp_also_in_other_fold_topR"] if "fp_also_in_other_fold_topR" in A else []):
        m, pos, n, lo, hi = pair(a, b, col)
        f = 100 if col.startswith("fp") else 1
        print(f"{lab:36s} {col:28s} {f * m:+6.2f}  positiv {pos}/{n}  je Verlust {f * lo:+.2f} bis {f * hi:+.2f}")
