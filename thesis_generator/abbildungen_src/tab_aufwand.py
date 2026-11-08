"""Rechenzeit je Schritt, je Epoche und je Fold (Messung bench_epoch_cost.py, Batchgrößen-Sweep).

Eingabe: Sicht timings/faelle.csv des Datenpakets, eine Zeile je gemessenem Fall mit Raten, Abschnitten und Trainingsgerät,
dazu die Phase-1-Folds (trainierte Epochen, Batchgröße, Warm-up) und für Semi-iNat das Warm-up aus phase1_settings.csv.
Ursprung der Messungen: timings_new/batch_sweep des Trainingsprojekts (cases.csv, cases/<Fall>/{epochs,sections}.csv,
result.json, worker.log). Je Fall ein Fold der besten Konfiguration einer Studie beim Budget der HPO (Cars196 k=16,
Semi-iNat k=16, In-Shop k=2, In-Shop-Serie mit gelabelten Klassen 10 %), allein auf einer H100 NVL, Batchgrößen 32 bis 512
(Cars196 32 bis 256 wie der dortige Suchraum), Warm-up, eine erste SSL-Epoche und 20 SSL-Epochen (SLADE ganze
Refresh-Zyklen). Raten wie in bench_epoch_cost.summarize_case: Warm-up- bzw. Ü-Epochen ab Epoche 1, SSL-Epochen nach der
ersten SSL-Epoche, Refresh-Abschnitte je SSL-Epoche (sections.csv, Teil gap).

Abdeckung: nur Fälle mit Training auf der GPU (device='cuda'). LRML auf Cars196 und In-Shop mit dem Trial der univariaten
Suche (reported_trial=False), auf Semi-iNat mit dem berichteten Trial; in der In-Shop-Serie keine LRML-Fälle. Auf Semi-iNat
fehlen Hoffer/NT-Xent bei 128 und SLADE/NT-Xent bei 32 und 64 (keine Fallordner); die Phase-1-Konfigurationen beider nutzen
512, die Dauer je Fold ist deshalb vollständig, Schritt und Epoche bei 128 für Hoffer nur über vier Verlustfunktionen.
CUB wurde nicht gemessen.

Ausgabe: kapitel/anhang_tabellen/aufwand.tex (Tabelle tab:anhang:aufwand, Anhang zu Phase 1) und das Protokoll der Zahlen
im Text (Ergebnisse, Abschnitt sec:e:aufwand).
Schritt und Epoche: Median über die Verlustfunktionen bei Batchgröße 128. Rechnerische Dauer je Fold: gemessene Raten bei der
Batchgröße des Phase-1-Folds mal dessen Epochen, Median über die 100 Folds (fünf Seeds, vier Folds, fünf Verlustfunktionen):
    min(E, W) t_Warm-up + [E > W] t_erste SSL-Epoche + max(0, E - W - 1) t_SSL   (Ü: E t_Epoche)
mit W dem Warm-up des Phase-1-Folds. Keine Messung der ganzen Läufe, keine Hochrechnung auf andere Budgets oder die HPO.
"""
from decimal import Decimal, ROUND_HALF_UP

import datensammlung as sammlung
import numpy as np
import pandas as pd

TAB = sammlung.output_dir("kapitel/anhang_tabellen")
M = ["sup", "iscen", "lrml", "hoffer", "slade"]
LB = {"sup": "\\UEBL", "iscen": "\\acs{LP}", "lrml": "\\acs{LRML}", "hoffer": "Hoffer", "slade": "\\acs{SLADE}"}
# Reihenfolge der Datensätze wie in den übrigen Ergebnistabellen
SETTINGS = [
    ("Cars196", "Cars196 ($k=16$)", "cars_ergebnisse/data/phase1_folds.csv", "k", 16),
    ("iNat", "Semi-iNat-2021 ($k=16$)", "semiinat_ergebnisse/data/phase1_folds_enriched.csv", "k", 16),
    ("DeepFashionInShop", "In-Shop ($k=2$)", "inshop_ergebnisse/data/phase1_folds_enriched.csv", "k", 2),
    ("DeepFashionInShop_10pct", "In-Shop-Serie mit gelabelten Klassen (\\qty{10}{\\percent})",
     "inshop10_ergebnisse/data/phase1_folds_enriched.csv", "budget_percent", 10),
]


def rnd(x, digits):
    """Kaufmännisch runden (nicht half-to-even wie format)."""
    q = Decimal(1).scaleb(-digits)
    return f"{Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP)}"


F = sammlung.read_csv("timings/faelle.csv")
assert (F.device == "cuda").all()


def folds(ds, view, budget, value):
    X = F[F.dataset == ds]
    P = sammlung.read_csv(view)
    P = P[P[budget] == value].copy()
    P["m"], P["loss"] = P.method.str.split("_").str[0], P.method.str.split("_").str[1]
    P["E"] = P.last_epoch + 1
    if "warmup_epochs" not in P:  # Semi-iNat: Warm-up aus den aufgelösten Einstellungen der Läufe
        S = sammlung.read_csv(view.replace("phase1_folds_enriched", "phase1_settings"))
        S = S[S[budget] == value]
        w = S.groupby("method")["ssl_config.warmup_epochs"].agg(lambda v: set(v.fillna(0).astype(int)))
        assert (w.map(len) == 1).all()
        P["warmup_epochs"] = P.method.map(w.map(lambda v: v.pop()))
    byb = X.set_index(["method", "loss", "batch_size"])
    assert byb.index.is_unique

    def seconds(r):
        key = (r.m, r.loss, r.batch_size)
        if key not in byb.index:
            return np.nan
        b = byb.loc[key]  # gemessener Fall mit der Batchgröße des Folds
        if r.m == "sup":
            return r.E * b.warm_s
        W = int(r.warmup_epochs)  # Warm-up des Phase-1-Folds
        w = 0.0 if np.isnan(b.warm_s) else b.warm_s
        return min(r.E, W) * w + (r.E > W) * b.first_s + max(0, r.E - W - 1) * b.ssl_s

    P["fold_min"] = P.apply(seconds, axis=1) / 60
    return X, P


R = {}
for ds, label, view, budget, value in SETTINGS:
    X, P = folds(ds, view, budget, value)
    assert (P.groupby("m").size() == 100).all()
    x128 = X[X.batch_size == 128].groupby("method")
    rows = {}
    for m in M:
        p = P[P.m == m]
        n = p.fold_min.notna().sum()
        assert n in (0, 100), (ds, m, n)  # ein Verfahren ist entweder ganz oder gar nicht gemessen
        if n == 0:  # nicht gemessen: nur die trainierten Epochen der Phase-1-Folds
            rows[m] = dict(step=np.nan, epoch=np.nan, n128=0, E=p.E.median(), fold=np.nan, fold_loss=None)
            continue
        g = x128.get_group(m)
        rows[m] = dict(step=g.train_ms_step.median(), epoch=g.epoch_s.median(), n128=len(g),
                       E=p.E.median(), fold=p.fold_min.median(),
                       fold_loss=p.groupby("loss").fold_min.median())
    R[ds] = dict(label=label, rows=rows, steps=int(X[X.batch_size == 128].steps.iloc[0]), P=P, X=X)

# ------------------------------------------------------------------ Tabelle
# Epoche je Batchgröße: die Neuberechnung fällt je Epoche an, das Training je Schritt, deshalb hängt die Rangfolge je Epoche
# von der Batchgröße ab. Schritt bei 128 (die Rangfolge je Schritt ist bei jeder Batchgröße dieselbe).
BS = [32, 64, 128, 256, 512]
T = ["\\begin{table}[tbp]",
     "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\caption[Rechenzeit der Verfahren]{Rechenzeit der Verfahren beim Budget der Hyperparametersuche (Abschnitt "
     "\\ref{sec:m:aufwand}), Median über die Verlustfunktionen. Schritt und Epoche nach dem Warm-up, Neuberechnung je Epoche über "
     "alle Batchgrößen $B$. Epochen und Fold: trainierte Epochen der Phase-1-Folds und daraus "
     "berechnete Dauer, Median über 100 Folds. "
     "Auf Semi-iNat-2021 fehlt \\acs{NT-Xent} bei Hoffer für $B=128$ und bei \\acs{SLADE} für $B=32$ und 64.}",
     "\\label{tab:anhang:aufwand}",
     "\\begin{tabular}{@{}l *{5}{c}@{}}",  # zentriert: die Zeilen sind verschiedene Größen
     "\\toprule",
     " & " + " & ".join(LB[m] for m in M) + " \\\\",
     "\\midrule"]


def cell(x, digits):
    return "--" if x is None or np.isnan(x) else "\\num{" + rnd(x, digits) + "}"


for i, (ds, *_rest) in enumerate(SETTINGS):
    r, X = R[ds], R[ds]["X"]
    if i:
        T.append("\\midrule")
    T.append("\\multicolumn{6}{@{}l}{\\textbf{" + r["label"] + "}} \\\\*")
    T.append("\\quad Schritt in ms bei $B=128$ & " + " & ".join(cell(r["rows"][m]["step"], 1) for m in M) + " \\\\")
    ep = X.pivot_table(index="batch_size", columns="method", values="epoch_s", aggfunc="median")
    for j, b in enumerate(b for b in BS if b in ep.index):
        name = "\\quad Epoche in s bei $B=" if j == 0 else "\\quad \\phantom{Epoche in s bei }$B="
        T.append(name + str(b) + "$ & " + " & ".join(cell(ep.loc[b, m] if m in ep else np.nan, 2) for m in M) + " \\\\")
    gap = {m: (np.nan if m == "sup" else X[X.method == m].ssl_gap_s.median()) for m in M}
    T.append("\\quad Neuberechnung in s & " + " & ".join(cell(gap[m], 2) for m in M) + " \\\\")
    T.append("\\quad Epochen & " + " & ".join(cell(r["rows"][m]["E"], 0) for m in M) + " \\\\")
    T.append("\\quad Fold in min & " + " & ".join(cell(r["rows"][m]["fold"], 1) for m in M) + " \\\\")
T += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(TAB + "aufwand.tex", "w", encoding="utf-8").write("\n".join(T) + "\n")
print("\n".join(T))

# ------------------------------------------------------------------ Protokoll der Zahlen im Text
print()
for ds, label, *_ in SETTINGS:
    r = R[ds]
    ok = {m: v for m, v in r["rows"].items() if v["n128"]}
    print(f"== {ds}: Schritte je Epoche bei 128 = {r['steps']}")
    for key, name, d in [("step", "ms/Schritt", 1), ("epoch", "s/Epoche", 2), ("E", "Epochen", 1), ("fold", "min/Fold", 2)]:
        order = sorted(ok, key=lambda m: ok[m][key])
        print(f"   {name:11s}", ", ".join(f"{m} {rnd(ok[m][key], d)}" for m in order))
    print("   n bei 128:", {m: v["n128"] for m, v in ok.items()})
    print("   min/Fold je Verlust (Median), Spanne:",
          {m: (rnd(v["fold_loss"].min(), 1), rnd(v["fold_loss"].max(), 1)) for m, v in ok.items()})
    sup = ok["sup"]
    print("   Faktor zur ÜBl. je Schritt / Epoche / Fold:",
          {m: (rnd(v["step"] / sup["step"], 1), rnd(v["epoch"] / sup["epoch"], 1), rnd(v["fold"] / sup["fold"], 1))
           for m, v in ok.items() if m != "sup"})
slow = {ds: max((v["fold"], m) for m, v in R[ds]["rows"].items() if v["n128"]) for ds, *_ in SETTINGS}
print("Langsamstes Verfahren je Fold:", {ds: (m, rnd(f, 1)) for ds, (f, m) in slow.items()})
for key in ["step", "epoch", "fold"]:
    print("Langsamstes je", key, {ds: max((v[key], m) for m, v in R[ds]["rows"].items() if v["n128"])[1]
                                  for ds, *_ in SETTINGS})

# ------------------------------------------------------------------ Protokoll: Abhängigkeit von der Batchgröße (nur Text, Tabelle unverändert)
# Die Neuberechnung (ssl_gap_s) fällt je Epoche an, das Training je Schritt. Die Rangfolge je Epoche gilt deshalb nur je Batchgröße.
print("\nAbhängigkeit von der Batchgröße (Median über die Verlustfunktionen):")
konf = {"kleiner": 0, "gleich": 0, "groesser": 0}
for ds, label, *_ in SETTINGS:
    X = R[ds]["X"]
    print(f"== {ds}")
    t = X.pivot_table(index="batch_size", columns="method", values="epoch_s", aggfunc="median")
    s = X.pivot_table(index="batch_size", columns="method", values="train_ms_step", aggfunc="median")
    g = X[X.method != "sup"].pivot_table(index="batch_size", columns="method", values="ssl_gap_s", aggfunc="median")
    tr = X[X.method != "sup"].pivot_table(index="batch_size", columns="method", values="ssl_train_s", aggfunc="median")
    for b in t.index:
        print(f"   B={b:3d} Schritte {int(X[X.batch_size == b].steps.median()):4d} | längster Schritt {s.loc[b].idxmax()},"
              f" längste Epoche {t.loc[b].idxmax()}, kürzeste Epoche {t.loc[b].idxmin()} | Epoche in s:",
              ", ".join(f"{m} {rnd(t.loc[b, m], 2)}" for m in M if m in t and not np.isnan(t.loc[b, m])))
        print("         Faktor zur ÜBl.:", ", ".join(f"{m} {rnd(t.loc[b, m] / t.loc[b, 'sup'], 1)}" for m in M if m != "sup" and m in t),
              "| Training je Epoche in s:", ", ".join(f"{m} {rnd(tr.loc[b, m], 2)}" for m in M if m in tr))
    print("   Neuberechnung je Epoche in s, kleinster und größter Median über die Batchgrößen:",
          {m: (rnd(g[m].min(), 2), rnd(g[m].max(), 2)) for m in M if m in g})
    B = R[ds]["P"].groupby(["m", "loss"]).batch_size.agg(lambda v: v.iloc[0] if v.nunique() == 1 else np.nan)
    assert B.notna().all() and len(B) == 25
    n = {"kleiner": int((B < 128).sum()), "gleich": int((B == 128).sum()), "groesser": int((B > 128).sum())}
    print("   Batchgröße der 25 Phase-1-Konfigurationen gegenüber 128:", n)
    for k in konf:
        konf[k] += n[k]
print("Alle vier Settings gegenüber 128:", konf)
