"""Geometrie des Embedding-Raums auf Semi-iNat-2021 (k=2, 4, 16), In-Shop (k=2, 4) und in der In-Shop-Serie mit vollständig
gelabelten Klassen (5, 10, 25, 50 %): Phase-2-Geometrieläufe (seit 09.10.2026 in der Arbeit), Seed 2, vier Folds, alle fünf
Verlustfunktionen, Maße nach jeder Epoche auf allen Bildern (sammle_geometrie.py semiinat inshop inshop10).

anhang_tabellen/geometrie_weitere_budgets.tex  Maße am gewählten Stand für alle Budgets (Anhang)
generiert/geometrie/korrelation_weitere.csv     Korrelationen je Datensatz, gelesen von fig_geometrie.py (tab:e:korrelation)
generiert/geometrie/ausgeschlossen_weitere.csv  ausgeschlossene Kombinationen mit Grund

pi aus den Spalten *_multi (nur Klassen mit mindestens zwei Bildern, Methodik). Ausgeschlossen werden Folds ohne Testwert,
Folds, deren Verfolgung vor der gewählten Epoche endet, und Kombinationen, in denen ein Fold den Head vor dem Training
(Epoche -1) wählt (derzeit keine) oder weniger als vier gültige Folds hat (Liste in ausgeschlossen_weitere.csv). Mittel je Budget
nur über die Verlustfunktionen, die für alle fünf Verfahren gültig sind (gepaart), Korrelation über alle gültigen Kombinationen.
"""
import datensammlung as sammlung
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import stil
import geodaten

RD = "geometrie/"
ANH = sammlung.output_dir("kapitel/anhang_tabellen")
M, LB, LOSSES = stil.METHOD_ORDER, stil.METHOD_LABELS, stil.LOSS_ORDER
DS = ["semiinat", "inshop", "inshop10"]
NAME = {"semiinat": "Semi-iNat-2021", "inshop": "In-Shop", "inshop10": "In-Shop-Serie mit gelabelten Klassen"}
P = "_multi"  # pi nur über Klassen mit mindestens zwei Bildern


def budget_label(ds, k):
    return f"\\qty{{{k}}}{{\\percent}}" if ds == "inshop10" else f"$k={k}$"


def load(ds):
    E, R, V = geodaten.load(ds)
    R = R.copy()
    bad = []
    R["valid"] = R.test_map.notna() & (R.geo_last_epoch >= R.selected_epoch)
    for _, r in R[~R.valid].iterrows():
        bad.append((ds, r.method, r.loss, r.k, r.fold, "kein Testwert" if pd.isna(r.test_map) else
                    f"Verfolgung endet bei Epoche {r.geo_last_epoch:.0f}, gewählt {r.selected_epoch:.0f}"))
    S = geodaten.at_selected(E, R[R.valid])
    nf = S.groupby(["method", "loss", "k"]).fold.transform("count")
    untrained = S.groupby(["method", "loss", "k"]).selected_epoch.transform("min") < 0
    for (m, l, k), g in S[untrained].groupby(["method", "loss", "k"]):
        bad.append((ds, m, l, k, -1, f"{(g.selected_epoch < 0).sum()} Folds wählen Epoche -1"))
    for (m, l, k), g in S[(nf < 4) & ~untrained].groupby(["method", "loss", "k"]):
        bad.append((ds, m, l, k, -1, f"nur {len(g)} gültige Folds"))
    S = S[(nf == 4) & ~untrained]
    return E, R, S, bad


D, BAD = {}, []
for ds in DS:
    E, R, S, bad = load(ds)
    D[ds] = dict(E=E, R=R, S=S)
    BAD += bad
sammlung.write_csv(pd.DataFrame(BAD, columns=["ds", "method", "loss", "k", "fold", "grund"]), RD + "ausgeschlossen_weitere.csv", index=False)
print(pd.DataFrame(BAD, columns=["ds", "method", "loss", "k", "fold", "grund"]).to_string())

COLS = dict(rho="train_rho_roth", pin="train_pi_intra" + P, pie="train_pi_inter" + P, pr="train_pi_ratio" + P,
            unl="unlabeled_pi_intra" + P, lab="labeled_pi_intra" + P, vpr="validation_pi_ratio" + P,
            vin="validation_pi_intra" + P, vie="validation_pi_inter" + P, er="train_embedding_effective_rank",
            proto="prototype_unlabeled_accuracy", ep="selected_epoch", m="test_map")


def paired_means(ds, k):
    """Mittel je Verfahren über die Verlustfunktionen, die bei diesem Budget für alle fünf Verfahren gültig sind."""
    S = D[ds]["S"]
    x = S[S.k == k]
    ok = [l for l in LOSSES if all(((x.method == m) & (x.loss == l)).any() for m in M)]
    x = x[x.loss.isin(ok)]
    g = x.groupby("method").agg(**{c: (col, "mean") for c, col in COLS.items()}).loc[M]
    g["proto"] *= 100
    return g, ok


def n(x, nd=2):
    return "{--}" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


# ------------------------------------------------------------------ Anhang: alle Budgets
# als longtable (passte als Gleitobjekt nicht auf eine Seite)
AHEAD = ("{Budget} & {Modell} & {$n$} & {$\\rho$} & {$\\pi_{\\mathrm{ratio}}$} & {$\\pi^{U}_{\\mathrm{intra}}$} & "
         "{$\\pi^{V}_{\\mathrm{ratio}}$} & {Prototyp} & {Epoche} & {MAP@R} \\\\\n\\midrule")
A = ["\\begingroup\\small\\setstretch{1}\\setlength{\\tabcolsep}{4pt}",
     "\\begin{longtable}{@{}ll S[table-format=1.0] S[table-format=1.2] S[table-format=1.2] S[table-format=1.2] S[table-format=1.2]"
     " S[table-format=2.1] S[table-format=3.0] S[table-format=2.1]@{}}",
     "\\caption[Geometrie auf Semi-iNat-2021 und In-Shop für alle Budgets]{Geometrie am per Validierung gewählten Stand für alle "
     "Budgets von Semi-iNat-2021, In-Shop und der In-Shop-Serie mit gelabelten Klassen (Seed~2, Mittel über vier Folds "
     "und die Verlustfunktionen, die bei diesem Budget für alle fünf Verfahren vollständig vorliegen, Spalte $n$). "
     "$\\rho$ und $\\pi_{\\mathrm{ratio}}$ auf dem Trainingspool, $\\pi^{U}_{\\mathrm{intra}}$ auf seinen ungelabelten Bildern, "
     "Prototyp: Anteil der ungelabelten Bilder mit richtigem nächstem Klassenprototyp in \\%, übrige Spalten wie in "
     "Tabelle~\\ref{tab:e:geometrie}, "
     "$\\pi^{V}_{\\mathrm{ratio}}$ auf den Validierungsklassen.}\\label{tab:a:geometrie:weitere}\\\\",
     "\\toprule", AHEAD, "\\endfirsthead",
     "\\multicolumn{10}{@{}l}{\\small Tabelle~\\thetable{} (Fortsetzung)}\\\\", "\\toprule", AHEAD, "\\endhead",
     "\\bottomrule", "\\endlastfoot"]
ALL = []
for i, ds in enumerate(DS):
    if i:
        A.append("\\addlinespace")
    A.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\emph{{{NAME[ds]}}}}} \\\\")
    for j, k in enumerate(sorted(D[ds]["S"].k.unique())):
        g, ok = paired_means(ds, k)
        g = g.assign(ds=ds, k=k, nloss=len(ok), losses=",".join(ok))
        ALL.append(g.reset_index())
        for m in M:
            x = g.loc[m]
            first = budget_label(ds, k) if m == "sup" else ""
            pr = "{--}" if ds == "inshop10" else n(x.proto, 1)
            name = LB[m].replace("LRML", "\\ac{LRML}").replace("ÜBl.", "\\UEBL")
            A.append(f"{first} & {name} & {len(ok) if m == 'sup' else ''} & {n(x.rho)} & {n(x.pr)} & {n(x.unl)} & {n(x.vpr)} & "
                     f"{pr} & {x.ep:.0f} & {n(x.m, 1)} \\\\")
A.append("\\end{longtable}\n\\endgroup")
open(ANH + "geometrie_weitere_budgets.tex", "w", encoding="utf-8").write("\n".join(A) + "\n")
ALL = pd.concat(ALL, ignore_index=True)
sammlung.write_csv(ALL, RD + "mittel_weitere.csv", index=False)
pd.set_option("display.width", 250)
print(ALL.round(3).to_string())

# ------------------------------------------------------------------ Korrelationen je Datensatz (für tab:e:korrelation)
ROWS = ["train_pi_intra", "train_pi_inter", "train_pi_ratio", "train_rho_roth", "validation_pi_intra",
        "validation_pi_inter", "validation_pi_ratio", "prototype_unlabeled_accuracy"]
K = []
for ds in DS:
    S = D[ds]["S"]
    kk = S.groupby(["method", "loss", "k"]).mean(numeric_only=True).reset_index()
    for row in ROWS:
        col = row + P if row.endswith(("intra", "inter", "ratio")) else row
        if ds == "inshop10" and row == "prototype_unlabeled_accuracy":
            K.append(dict(ds=ds, row=row, n=len(kk), pearson=np.nan, spearman=np.nan, within=""))
            continue
        w = [round(pearsonr(g[col], g.test_map)[0], 2) for _, g in kk.groupby("k")]
        K.append(dict(ds=ds, row=row, n=len(kk), pearson=pearsonr(kk[col], kk.test_map)[0],
                      spearman=spearmanr(kk[col], kk.test_map)[0], within=" ".join(map(str, w))))
    se = [round(spearmanr(g.selected_epoch, g.test_map)[0], 2) for _, g in kk.groupby("k")]
    print(ds, "Kombinationen", len(kk), "gewählte Epoche vs. MAP@R (Spearman je Budget)", se)
K = pd.DataFrame(K)
sammlung.write_csv(K, RD + "korrelation_weitere.csv", index=False)
print(K.round(2).to_string())

# ------------------------------------------------------------------ Klassen mit einem Bild (Methodik)
for ds in DS:
    E = D[ds]["E"]
    h = E[E.epoch == -1]
    print(ds, {sp: (int(h[f"{sp}_pi_single_class_count"].max()), int(h[f"{sp}_pi_class_count"].max()))
               for sp in ["train", "labeled", "unlabeled", "validation", "test"] if f"{sp}_pi_single_class_count" in h})
