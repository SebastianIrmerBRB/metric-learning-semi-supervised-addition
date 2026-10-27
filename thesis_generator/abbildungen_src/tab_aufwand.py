"""Rechenaufwand je Epoche und je Lauf (Messung bench_epoch_cost.py, Batchgrößen-Sweep vom 01.10.2026).

Eingabe: Sicht timings/faelle.csv des Datenpakets, eine Zeile je gemessenem Fall mit Raten, Abschnitten und Trainingsgerät.
Ursprung der Messungen: Ordner Timings und timings_new/batch_sweep des Trainingsprojekts (cases.csv,
cases/<Fall>/{epochs,sections}.csv, result.json, worker.log). Je Fall ein Fold
der besten Konfiguration einer Studie (Budget der HPO: Cars196 k=16, In-Shop k=2, In-Shop 10 %), allein auf einer H100 NVL,
Warm-up, eine erste SSL-Epoche und 20 SSL-Epochen (SLADE ganze Refresh-Zyklen). Je Epoche gemessen: gap (Refresh der
SSL-Verfahren zwischen Validierung und Training), train, valid. Raten wie in bench_epoch_cost.summarize_case: Warm-up- bzw.
Ü-Epochen ab Epoche 1, SSL-Epochen nach der ersten SSL-Epoche, Refresh-Abschnitte je SSL-Epoche (sections.csv, Teil gap).

Ausgaben (die Arbeit bindet aufwand.tex nicht ein, die Zahlen im Text stehen im Protokoll dieses Skripts)
kapitel/ergebnisse_tabellen/aufwand.tex In-Shop k=2: SSL-Epoche relativ zur Ü-Epoche bei gleicher Verlustfunktion und
                                        Batchgröße, Refresh je Epoche, Epochen bis zum Abbruch (Phase 1), rechnerische Dauer
                                        eines Laufs je Fold, Gewinn (Einzelmodell).
Rechnerische Dauer je Fold: gemessene Raten der gewählten Konfiguration (eigene Batchgröße) mal den Epochen der 100
Phase-1-Folds bei k=2 (fünf Seeds, vier Folds, fünf Verlustfunktionen):
    min(E, W) t_Warm-up + [E > W] t_erste SSL-Epoche + max(0, E - W - 1) t_SSL   (Ü: E t_Epoche)
Keine Messung der ganzen Läufe, keine Hochrechnung auf andere Budgets oder die HPO.

Seit 15.10.2026 aus timings_new/batch_sweep (dieselben Fälle wie Timings vom 01.10. für
In-Shop, dazu Ü auf Cars196 auf der GPU, Hoffer/ArcFace auf In-Shop 10 % und alle Kombinationen auf Semi-iNat). Verwendet werden
nur Fälle mit Training auf der GPU (device='cuda'; die älteren Ü-Fälle auf Cars196 trainierten auf der CPU) und, wo für eine
Kombination mehrere Trials gemessen wurden (Semi-iNat: erste und zweite Suche), der Trial, den die Arbeit berichtet
(phase1_configs.json). LRML auf Cars196 und In-Shop ist mit den Trials der univariaten Suche gemessen und wird nicht neu
gemessen (Entscheidung 15.10.2026); die rechnerische Dauer je Fold nimmt deshalb für jedes Verfahren die Raten des gemessenen
Falls mit der Batchgröße des Phase-1-Folds und dessen eigene Warm-up-Länge.
"""
import datensammlung as sammlung
import glob
import json
import os
import re
import numpy as np
import pandas as pd

TAB = sammlung.output_dir("kapitel/ergebnisse_tabellen")
IN = "inshop_ergebnisse/data/"
SHORT = {"supervised": "sup", "iscen_label_spreading": "iscen", "lrml": "lrml", "hoffer_entropy": "hoffer", "slade": "slade"}
LOSS = {"ArcFaceLoss": "arc", "CircleLoss": "cir", "MultiSimilarityLoss": "ms", "NTXentLoss": "nt", "ProxyAnchorLoss": "prx"}
M = ["sup", "iscen", "lrml", "hoffer", "slade"]
LB = {"sup": "\\UEBL", "iscen": "LP", "lrml": "\\ac{LRML}", "hoffer": "Hoffer", "slade": "SLADE"}

# Rates are collected once; figure/table generation uses the portable data.
F = sammlung.read_csv("timings/faelle.csv")

# ------------------------------------------------------------------ In-Shop k=2
X = F[F.dataset == "DeepFashionInShop"]
assert (X.device == "cuda").all() and len(X) == 125
sup = X[X.method == "sup"].set_index(["loss", "batch_size"]).epoch_s
ratio, refresh = {}, {}
for m in M[1:]:
    x = X[X.method == m].set_index(["loss", "batch_size"])
    q = (x.ssl_s / sup.reindex(x.index)).dropna()
    assert len(q) == 25
    ratio[m] = q
    refresh[m] = x.ssl_gap_s
# Warm-up-Epochen der SSL-Verfahren als Kontrolle gegen die Ü-Epoche
for m in ["iscen", "lrml"]:
    x = X[X.method == m].set_index(["loss", "batch_size"])
    print("Warm-up/Ü", m, round((x.warm_s / sup.reindex(x.index)).median(), 2))

P = sammlung.read_csv(IN + "phase1_folds_enriched.csv")
P = P[P.k == 2].copy()
P["m"], P["loss"] = P.method.str.split("_").str[0], P.method.str.split("_").str[1]
P["E"] = P.last_epoch + 1
byb = X.set_index(["method", "loss", "batch_size"])


def fold_seconds(r):
    b = byb.loc[(r.m, r.loss, r.batch_size)]  # gemessener Fall mit der Batchgröße des Folds
    if r.m == "sup":
        return r.E * b.warm_s
    W = int(r.warmup_epochs)  # Warm-up des Phase-1-Folds
    w = 0.0 if np.isnan(b.warm_s) else b.warm_s
    return min(r.E, W) * w + (r.E > W) * b.first_s + max(0, r.E - W - 1) * b.ssl_s


P["fold_min"] = P.apply(fold_seconds, axis=1) / 60
assert (P.groupby("m").size() == 100).all()
G = sammlung.read_csv(IN + "phase1_paired_differences.csv")
G = G[(G.k == 2) & (G.scope == "mean") & (G.metric == "mean_average_precision_at_r")]
G["m"] = G.method.str.split("_").str[0]
gain = G.groupby("m")["mean"].mean()

T = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Rechenaufwand auf In-Shop]{Rechenaufwand auf In-Shop bei $k=2$, gemessen je Epoche für einen Fold allein auf einer GPU. "
     "Epoche: Dauer einer Epoche nach dem Warm-up relativ zu \\UEBL mit derselben Verlustfunktion und Batchgröße, Median und Spanne über "
     "fünf Verlustfunktionen und fünf Batchgrößen. Refresh: Neuberechnung von Graph, Pseudo-Labels oder Clustern je Epoche, Median, Hoffer hat keinen. "
     "Epochen: trainierte Epochen bis zum Abbruch in Phase~1, Median über 100 Folds. Lauf: rechnerische Dauer eines Folds aus den "
     "gemessenen Raten bei der Batchgröße und dem Warm-up der Phase-1-Folds und deren Epochen, Median. Gewinn: Test-MAP@R der Einzelmodelle gegenüber \\UEBL "
     "in Prozentpunkten (Tabelle~\\ref{tab:e:gewinne}).}",
     "\\label{tab:e:aufwand}",
     "\\begin{tabular}{@{}l c S[table-format=1.2] S[table-format=3.0] S[table-format=2.1] S[table-format=+1.1, retain-explicit-plus]@{}}",
     "\\toprule",
     "{Verfahren} & {Epoche rel.\\ zu \\UEBL} & {Refresh in s} & {Epochen} & {Lauf in min} & {Gewinn} \\\\",
     "\\midrule"]
for m in M:
    ep = P[P.m == m].E.median()
    lauf = P[P.m == m].fold_min.median()
    if m == "sup":
        T.append(f"{LB[m]} & 1 & {{--}} & {ep:.0f} & {lauf:.1f} & {{--}} \\\\")
        continue
    q = ratio[m]
    rel = f"{q.median():.1f} ({q.min():.1f}--{q.max():.1f})".replace(".", ",")
    ref = f"{refresh[m].median():.2f}" if refresh[m].median() >= 0.005 else "{--}"
    T.append(f"{LB[m]} & {rel} & {ref} & {ep:.0f} & {lauf:.1f} & {gain[m]:+.1f} \\\\")
T.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(TAB + "aufwand.tex", "w", encoding="utf-8").write("\n".join(T) + "\n")
print("\n".join(T))

# ------------------------------------------------------------------ Zahlen für den Text
print("\nIn-Shop: Ü-Epoche s (Median über 25)", round(sup.median(), 2), "Spanne", round(sup.min(), 2), round(sup.max(), 2))
for m in M[1:]:
    x = X[X.method == m]
    secs = x.filter(like="gap_").median().round(3)
    print(m, "Refresh-Abschnitte", secs[secs > 0.01].to_dict(), "ms/Schritt", round(x.train_ms_step.min(), 1), round(x.train_ms_step.max(), 1))
print("Ü ms/Schritt In-Shop", round(X[X.method == "sup"].train_ms_step.min(), 1), round(X[X.method == "sup"].train_ms_step.max(), 1))
print("Lauf je Fold min/max über Verluste:", P.groupby(["m", "loss"]).fold_min.median().groupby("m").agg(["min", "max"]).round(1).to_dict())
for ds in ["Cars196", "DeepFashionInShop_10pct"]:
    x = F[(F.dataset == ds) & (F.method != "sup")]
    print(ds, "Refresh Median", x.groupby("method").ssl_gap_s.median().round(2).to_dict())
    print(ds, "Abschnitte", x.groupby("method")[[c for c in x if c.startswith("gap_")]].median().round(3)
          .apply(lambda r: r[r > 0.01].to_dict(), axis=1).to_dict())
    print(ds, "ms/Schritt", x.groupby("method").train_ms_step.agg(["min", "max"]).round(1).to_dict())
x = F[F.dataset == "DeepFashionInShop_10pct"]
s10 = x[x.method == "sup"].set_index(["loss", "batch_size"]).epoch_s
for m in ["iscen", "hoffer", "slade"]:
    y = x[x.method == m].set_index(["loss", "batch_size"])
    q = (y.ssl_s / s10.reindex(y.index)).dropna()
    print("In-Shop 10 %", m, "rel.", round(q.median(), 1), round(q.min(), 1), round(q.max(), 1), "n", len(q))
c = F[(F.dataset == "Cars196") & (F.method == "sup")]
print("Cars Ü Gerät", c.device.unique(), "ms/Schritt je bs", c.groupby("batch_size").train_ms_step.median().round(1).to_dict())
c = F[(F.dataset == "Cars196") & (F.method == "iscen")]
print("Cars LP (cuda) ms/Schritt je bs", c.groupby("batch_size").train_ms_step.median().round(1).to_dict())
print("Semi-iNat Fälle:", (F.dataset == "iNat").sum(), "im Datenpaket")
