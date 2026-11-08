"""Rangkorrelation nach Spearman (seit 6.11.2026, vorher Pearson) der Geometrie am gewählten Stand mit der Test-MAP@R, je
Datensatz gesamt und innerhalb jedes Budgets (tab:e:korrelation, seit 25.10.2026 in der Arbeit, review/ergebnisse_notizen_2026-10-25).

Gegenüber korrelation.tex (fig_geometrie.py) stehen Datensätze und Budgets in den Zeilen und die Maße in den Spalten, sodass
auch Semi-iNat-2021 und beide In-Shop-Serien je Budget erscheinen. Cars196 wird wie in fig_geometrie.py aus den
Geometrieläufen berechnet, die übrigen Datensätze kommen aus generiert/geometrie/korrelation_weitere.csv, mittel_weitere.csv
und ausgeschlossen_weitere.csv (fig_geometrie_weitere.py, fehlen die Dateien oder sind sie veraltet, schreibt
datensammlung.read_csv sie zuerst). Ausgabe: kapitel/ergebnisse_tabellen/korrelation_budgets.tex
"""
import datensammlung as sammlung
import pandas as pd
from scipy.stats import spearmanr
import geodaten

TAB = sammlung.output_dir("kapitel/ergebnisse_tabellen")
COLS = [("train_pi_intra", "train_pi_intra_roth"), ("train_pi_inter", "train_pi_inter_roth"),
        ("train_pi_ratio", "train_pi_ratio_roth"), ("train_rho_roth", "train_rho_roth"),
        ("validation_pi_intra", "validation_pi_intra_roth"), ("validation_pi_inter", "validation_pi_inter_roth"),
        ("validation_pi_ratio", "validation_pi_ratio_roth"), ("prototype_unlabeled_accuracy", "prototype_unlabeled_accuracy")]
NAME = {"cars": "Cars196", "semiinat": "Semi-iNat-2021", "inshop": "In-Shop", "inshop10": "In-Shop mit gelabelten Klassen"}


def budget(ds, k):
    return f"\\qty{{{k}}}{{\\percent}}" if ds == "inshop10" else f"$k={k}$"


def cell(v):
    return "{--}" if pd.isna(v) else f"{v:.2f}"


# Cars196 wie in fig_geometrie.py: Mittel je Kombination über die vier Folds, Korrelation über die Kombinationen
E, R, _ = geodaten.load("cars")
kk = geodaten.at_selected(E, R).groupby(["method", "loss", "k"]).mean(numeric_only=True).reset_index()
rows = [("cars", "gesamt", len(kk), [spearmanr(kk[c], kk.test_map)[0] for _, c in COLS])]
for k, g in kk.groupby("k"):
    rows.append(("cars", budget("cars", k), len(g), [spearmanr(g[c], g.test_map)[0] for _, c in COLS]))

# Übrige Datensätze aus fig_geometrie_weitere.py (Spalte within_spearman: Werte je Budget in aufsteigender Reihenfolge, gerundet)
KW = sammlung.read_csv("geometrie/korrelation_weitere.csv").set_index(["ds", "row"])
MW = sammlung.read_csv("geometrie/mittel_weitere.csv")
AUS = sammlung.read_csv("geometrie/ausgeschlossen_weitere.csv")
AUS = AUS[AUS.fold == -1]  # ganze Kombinationen, die fig_geometrie_weitere.py ausschließt
for ds in ["semiinat", "inshop", "inshop10"]:
    ks = sorted(MW[MW.ds == ds].k.unique())
    rows.append((ds, "gesamt", int(KW.loc[ds].n.iloc[0]), [KW.loc[(ds, r)].spearman for r, _ in COLS]))
    within = {r: str(KW.loc[(ds, r)].within_spearman).split() for r, _ in COLS}
    for i, k in enumerate(ks):
        vals = [float(within[r][i]) if within[r] and within[r][0] != "nan" else float("nan") for r, _ in COLS]
        rows.append((ds, budget(ds, k), 25 - int(((AUS.ds == ds) & (AUS.k == k)).sum()), vals))

T = ["\\begin{table}[tbp]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\caption[Korrelation der Geometrie mit der Retrieval-Leistung]{Rangkorrelation nach Spearman zwischen Maßen nach dem Training und der Test-MAP@R (Mean), Seed 2, Mittel über vier Folds. Gesamt: alle gültigen Kombinationen aus Verfahren, Verlustfunktion und Budget eines Datensatzes, darunter je Budget, Anzahl: Zahl dieser Kombinationen. In der In-Shop-Serie mit gelabelten Klassen steht der Anteil gelabelter Klassen für das Budget. Prototyp: Anteil der ungelabelten Bilder mit richtigem nächstem Klassenprototyp, in dieser Serie nicht definiert (--).}",
     "\\label{tab:e:korrelation}",
     "\\begin{tabular}{@{}l r S[table-format=-1.2] S[table-format=-1.2] S[table-format=-1.2] S[table-format=-1.2] "
     "@{\\hspace{12pt}} S[table-format=-1.2] S[table-format=-1.2] S[table-format=-1.2] "
     "@{\\hspace{12pt}} S[table-format=-1.2]@{}}",
     "\\toprule",
     "& & \\multicolumn{4}{c}{\\textbf{Trainingspool}} & \\multicolumn{3}{c}{\\textbf{Validierungsklassen}} & {\\textbf{Ungelabelt}} \\\\",
     "\\cmidrule(lr){3-6}\\cmidrule(lr){7-9}\\cmidrule(l){10-10}",
     "& {Anzahl} & {$\\pi_{\\mathrm{intra}}$} & {$\\pi_{\\mathrm{inter}}$} & {$\\pi_{\\mathrm{ratio}}$} & {$\\rho$} & "
     "{$\\pi_{\\mathrm{intra}}$} & {$\\pi_{\\mathrm{inter}}$} & {$\\pi_{\\mathrm{ratio}}$} & {Prototyp} \\\\",
     "\\midrule"]
last = None
for ds, label, n, vals in rows:
    if ds != last:
        if last is not None:
            T.append("\\midrule")
        T.append(f"\\multicolumn{{10}}{{@{{}}l}}{{\\textbf{{{NAME[ds]}}}}} \\\\*")
        last = ds
    name = "\\textit{gesamt}" if label == "gesamt" else label
    T.append(f"\\quad {name} & {n} & " + " & ".join(cell(v) for v in vals) + " \\\\")
    if label == "gesamt":
        T.append("\\cmidrule{1-10}")
T.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(TAB + "korrelation_budgets.tex", "w", encoding="utf-8").write("\n".join(T) + "\n")
print("\n".join(T))
