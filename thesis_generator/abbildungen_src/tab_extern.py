"""Tabellen zu externen ungelabelten Bildern (nach sammle_extern.py): Haupttext extern.tex, Anhang extern_{cars,inshop}.tex
und extern_sweeps.tex.

Seed 2. Gewinn = Mechanismus minus Ü mit denselben Labels (alle Trainingslabels, keine externen Bilder), gepaart je
Verlustfunktion und dann über die fünf Verlustfunktionen gemittelt. Spalte "ohne" (seit 12.10.2026): Kontrolle mit der
SSL-Konfiguration ohne externe Bilder, bei LP und LRML mit leerem Pool, bei Hoffer und SLADE mit abgeschaltetem SSL-Term.
Schwelle wie in Phase 2 für den Vergleich mit Ü: 2*sqrt(sigma_SSL^2 + sigma_Ü^2) der Mean-MAP@R aus den fünf Seeds der
Phase 1 (Cars196 k=16, In-Shop k=2). Sweeps (nur Cars196): Nachbarzahl für LP und LRML, Clusterzahl für SLADE.
"""
import datensammlung as sammlung
import numpy as np
import pandas as pd

R = sammlung.read_csv("extern/laeufe.csv")
R = R[R.seed == 2]
ET = sammlung.output_dir("kapitel/ergebnisse_tabellen")
AN = sammlung.output_dir("kapitel/anhang_tabellen")
LOSS = {"arc": "ArcFace", "cir": "Circle", "ms": "\\ac{MS}", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
METH = {"iscen": "LP", "lrml": "\\ac{LRML}", "hoffer": "Hoffer", "slade": "SLADE"}
PCT = {"Cars196": [0, 10, 25, 50, 100], "In-Shop": [0, 5, 10, 25, 50]}


def n(x, d=1, plus=False):
    return ("{--}" if np.isnan(x) else (f"{x:+.{d}f}" if plus else f"{x:.{d}f}"))


sup = R[R.kind == "referenz"].set_index(["dataset", "loss"])
D = R[R.kind.isin(["regulär", "kontrolle", "sweep"])].copy()
D["gain_m"] = [r.mean_map - sup.loc[(r.dataset, r.loss), "mean_map"] for r in D.itertuples()]
D["gain_c"] = [r.concat_map - sup.loc[(r.dataset, r.loss), "concat_map"] for r in D.itertuples()]

# Schwellen aus Phase 1 (Mean, fünf Seeds)
THR = {}
for ds, path, kk in (("Cars196", "cars_ergebnisse", 16), ("In-Shop", "inshop_ergebnisse", 2)):
    # LRML seit 16.10.2026 mit der multivariaten Suche (cars_full_lrml_mult, in_shop_full_lrml_mult), Schwellen daher
    # aus den aktuellen Phase-1-Läufen (vorher aus archiv_20261015_lrml_univariat)
    P = sammlung.read_csv(f"{path}/data/phase1_runs.csv")
    P = P[(P.scope == "mean") & (P.k == kk)]
    for m, g in P.groupby("method"):
        THR[(ds, m)] = 100 * g.mean_average_precision_at_r.std(ddof=1)
# Vergleich mit Ü: 2 * Wurzel aus der Summe der Seed-Varianzen beider Konfigurationen (Abschnitt sec:m:phase2)
D["thr"] = [2 * np.hypot(THR[(r.dataset, f"{r.method}_{r.loss}")], THR[(r.dataset, f"sup_{r.loss}")]) for r in D.itertuples()]
sammlung.write_csv(D, "extern/gewinne.csv", index=False)
M = D[D.kind.isin(["regulär", "kontrolle"])]  # Haupttabelle: Kontrolle (Anteil 0) und reguläre Konfiguration

# ---------------------------------------------------------------- Haupttext
cols = [(ds, p) for ds in PCT for p in PCT[ds]]
nx = {(ds, p): int(M[(M.dataset == ds) & (M.pct == p)].n_external.iloc[0]) for ds, p in cols}
T = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Externe ungelabelte Bilder]{Externe ungelabelte Bilder bei vollständig gelabelten Trainingsklassen "
     "(Seed~2): gepaarter Gewinn der Test-MAP@R gegenüber \\UEBL mit denselben Labels und ohne externe Bilder in "
     "Prozentpunkten, gemittelt über die fünf Verlustfunktionen. Anteil: verwendeter Anteil des externen Pools "
     "(CompCars für Cars196, Fashion200k für In-Shop), darunter die Zahl der Bilder. Ohne: Kontrolle mit der "
     "\\ac{SSL}-Konfiguration ohne externe Bilder, bei Hoffer und SLADE mit abgeschaltetem \\ac{SSL}-Term. \\UEBL: Test-MAP@R in \\%. "
     "M: Mean, C: Concat. Positiv: Zahl der Verlustfunktionen mit positivem Gewinn (Mean) von fünf.}",
     "\\label{tab:e:extern}",
     "\\setlength{\\tabcolsep}{2.9pt}\n\\small\n\\begin{tabular}{@{}ll *{10}{S[table-format=+2.1, retain-explicit-plus]}@{}}\n\\toprule",
     " & & \\multicolumn{5}{c}{Cars196 + CompCars} & \\multicolumn{5}{c}{In-Shop + Fashion200k} \\\\",
     "\\cmidrule(lr){3-7}\\cmidrule(lr){8-12}",
     "\\multicolumn{2}{@{}l}{Anteil} & " + " & ".join("{ohne}" if p == 0 else f"{{\\qty{{{p}}}{{\\percent}}}}" for _, p in cols) + " \\\\",
     "\\multicolumn{2}{@{}l}{Bilder} & " + " & ".join(f"{{\\num{{{nx[c]}}}}}" for c in cols) + " \\\\\n\\midrule"]
us = {ds: (sup.loc[ds].mean_map.mean(), sup.loc[ds].concat_map.mean()) for ds in PCT}
T.append("\\UEBL & M & \\multicolumn{5}{c}{\\num{" + n(us["Cars196"][0]) + "}} & \\multicolumn{5}{c}{\\num{" + n(us["In-Shop"][0]) + "}} \\\\")
T.append(" & C & \\multicolumn{5}{c}{\\num{" + n(us["Cars196"][1]) + "}} & \\multicolumn{5}{c}{\\num{" + n(us["In-Shop"][1]) + "}} \\\\\n\\addlinespace")
for scope, col, label in (("M", "gain_m", "Gewinn (M)"), ("C", "gain_c", "Gewinn (C)")):
    T.append(f"\\multicolumn{{2}}{{@{{}}l}}{{\\emph{{{label}}}}}" + " &" * 10 + " \\\\")
    for m, mn in METH.items():
        cells = [n(M[(M.dataset == ds) & (M.pct == p) & (M.method == m)][col].mean(), plus=True) for ds, p in cols]
        T.append(f"{mn} & & " + " & ".join(cells) + " \\\\")
    T.append("\\addlinespace")
T.append("\\multicolumn{2}{@{}l}{\\emph{Positiv (M)}}" + " &" * 10 + " \\\\")
for m, mn in METH.items():
    cells = [str(int((M[(M.dataset == ds) & (M.pct == p) & (M.method == m)].gain_m > 0).sum())) for ds, p in cols]
    T.append(f"{mn} & & " + " & ".join(f"{{{c}}}" for c in cells) + " \\\\")
T.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(ET + "extern.tex", "w", encoding="utf-8").write("\n".join(T) + "\n")

# ---------------------------------------------------------------- Anhang je Datensatz
for ds, key, ext in (("Cars196", "cars", "CompCars"), ("In-Shop", "inshop", "Fashion200k")):
    ps = PCT[ds]
    L = ["\\begin{table}[p]\n\\centering",
         f"\\caption[Externe ungelabelte Bilder auf {ds}]{{Test-MAP@R in \\% auf {ds} mit allen Trainingslabels und "
         f"einem Anteil von {ext} als externem ungelabeltem Pool (Seed~2). Ohne: Kontrolle mit der \\ac{{SSL}}-Konfiguration "
         f"ohne externe Bilder, bei Hoffer und SLADE mit abgeschaltetem \\ac{{SSL}}-Term. M: einzelne Fold-Modelle (Mean), C: Ensemble "
         f"(Concat). Die Zeile \\UEBL nennt die Referenz mit denselben Labels ohne externe Bilder.}}",
         f"\\label{{tab:anhang:extern:{key}}}\n\\small\\setlength{{\\tabcolsep}}{{2.6pt}}",
         "\\begin{tabular}{@{}ll" + "S[table-format=2.2]" * (2 * len(ps)) + "@{}}\n\\toprule",
         " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{{'ohne' if p == 0 else chr(92) + 'qty{' + str(p) + '}{' + chr(92) + 'percent}'}}}" for p in ps) + " \\\\",
         "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(ps))),
         "Verlust & Verfahren & " + " & ".join("{M} & {C}" for _ in ps) + " \\\\\n\\midrule"]
    for li, (lk, ln) in enumerate(LOSS.items()):
        s = sup.loc[(ds, lk)]
        L.append(f"{ln} & \\UEBL & \\multicolumn{{{2 * len(ps)}}}{{l}}{{M {s.mean_map:.2f}, C {s.concat_map:.2f}}} \\\\".replace(".", "{,}"))
        for m, mn in METH.items():
            cells = []
            for p in ps:
                x = M[(M.dataset == ds) & (M.method == m) & (M.loss == lk) & (M.pct == p)].iloc[0]
                cells.append(f"{x.mean_map:.2f} & {x.concat_map:.2f}")
            L.append(f" & {mn} & " + " & ".join(cells) + " \\\\")
        if li < 4:
            L.append("\\addlinespace[2pt]")
    L.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    open(AN + f"extern_{key}.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- Anhang: Sweeps auf Cars196
S = D[(D.dataset == "Cars196") & (D.kind.isin(["regulär", "sweep"])) & (D.method.isin(["iscen", "lrml", "slade"]))]
ps = [10, 25, 50, 100]
L = ["\\begin{table}[p]\n\\centering",
     "\\caption[Nachbar- und Clusterzahl bei externen Bildern auf Cars196]{Externe ungelabelte Bilder auf Cars196 (CompCars, "
     "Seed~2) mit veränderter Nachbarzahl (LP, \\ac{LRML}) bzw.\\ fester Clusterzahl (SLADE): gepaarter Gewinn der Test-MAP@R "
     "gegenüber \\UEBL in Prozentpunkten, gemittelt über die fünf Verlustfunktionen, übrige Werte wie in Phase~1. Phase~1: "
     "Nachbarzahl bzw.\\ Clusterverhältnis aus Phase~1 (Tabelle~\\ref{tab:e:extern}). Positiv: Zahl der Läufe mit positivem "
     "Gewinn (Mean) von 20 (vier Anteile, fünf Verlustfunktionen). M: Mean, C: Concat.}",
     "\\label{tab:anhang:extern:sweeps}\n\\small\\setlength{\\tabcolsep}{3.2pt}",
     "\\begin{tabular}{@{}ll" + "S[table-format=+1.1, retain-explicit-plus]" * (2 * len(ps)) + "S[table-format=1.0]@{}}\n\\toprule",
     " & & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{\\qty{{{p}}}{{\\percent}}}}" for p in ps) + " & \\\\",
     "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(len(ps))),
     "Verfahren & Wert & " + " & ".join("{M} & {C}" for _ in ps) + " & {Positiv} \\\\\n\\midrule"]
for mi, (m, mn, what) in enumerate((("iscen", "LP", "Nachbarn"), ("lrml", "\\ac{LRML}", "Nachbarn"), ("slade", "SLADE", "Cluster"))):
    g = S[S.method == m]
    vals = [np.nan] + sorted(g[g.kind == "sweep"].sweep_value.unique())
    for vi, v in enumerate(vals):
        h = g[g.kind == "regulär"] if np.isnan(v) else g[np.isclose(g.sweep_value, v)]
        cells = []
        for p in ps:
            x = h[h.pct == p]
            cells.append(f"{n(x.gain_m.mean(), plus=True)} & {n(x.gain_c.mean(), plus=True)}")
        label = "Phase~1" if np.isnan(v) else f"{int(v)} {what}"
        L.append(f"{mn if vi == 0 else ''} & {label} & " + " & ".join(cells) + f" & {int((h.gain_m > 0).sum())} \\\\")
    if mi < 2:
        L.append("\\addlinespace")
L.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(AN + "extern_sweeps.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# Kennzahlen für den Text
for ds in PCT:
    d = D[(D.dataset == ds) & (D.kind == "regulär")]
    print(ds, "Ü", [round(x, 2) for x in us[ds]], "positiv", int((d.gain_m > 0).sum()), "von", len(d),
          "| unter -Schwelle", int((d.gain_m < -d.thr).sum()), "| über Schwelle", int((d.gain_m > d.thr).sum()))
    k = D[(D.dataset == ds) & (D.kind == "kontrolle")]
    print("  Kontrolle: über Schwelle", int((k.gain_m > k.thr).sum()), "unter -Schwelle", int((k.gain_m < -k.thr).sum()),
          k.groupby("method").gain_m.agg(["mean", "min", "max"]).round(1).to_dict("index"))
sw = D[D.kind == "sweep"]
print("Sweeps: positiv", int((sw.gain_m > 0).sum()), "von", len(sw), "| über Schwelle", int((sw.gain_m > sw.thr).sum()),
      sw[sw.gain_m > sw.thr][["method", "loss", "pct", "sweep_value", "gain_m", "gain_c"]].round(2).to_string())
