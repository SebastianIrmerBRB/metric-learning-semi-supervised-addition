"""Geometrie des Embedding-Raums am per Validierung gewählten Stand (Cars196, Phase-2-Geometrieläufe, Seed 2, vier Folds).

geometrie_budget.pdf       pi_ratio, rho und Test-MAP@R (Mean) am gewählten Stand über k (Mittel über alle Verlustfunktionen). Referenz gestrichelt: Head vor
                           dem ersten Optimierungsschritt (Epoche -1, Seed 2), gepunktet in (b): DINOv2-Features
geometrie_korrelation.pdf  pi_ratio des Trainingspools gegen Test-MAP@R, je Budget ein Teilbild, alle 100 Kombinationen
                           (vgl. Roth et al. 2020, Abb. 5)
korrelation.tex            Korrelationen aller Maße mit der Test-MAP@R, gesamt und innerhalb der Budgets, dazu je eine Spalte
                           für Semi-iNat, In-Shop und die In-Shop-Serie mit gelabelten Klassen über alle gültigen Kombinationen
                           (generiert/geometrie/korrelation_weitere.csv aus fig_geometrie_weitere.py, vorher ausführen)
geometrie.tex              Tabelle der Maße am gewählten Stand (Mittel über alle Verlustfunktionen, k=4 und k=32)
anhang_tabellen/geometrie_verluste.tex  pi_ratio der gelabelten und ungelabelten Bilder und Test-MAP@R je Verlustfunktion
Cars196 seit 01.10.: Geometrieläufe, Maße nach jeder Epoche auf allen Bildern des Trainingspools (Teilmenge train, bzw.
labeled und unlabeled), wie bei Roth et al. Gewählter Stand: die per Validierung gewählte Epoche selbst (at_selected nimmt
die letzte verfolgte Epoche <= gewählte, bei Verfolgung jeder Epoche also die gewählte). Test-MAP@R: Mean der einzelnen
Fold-Modelle (cv_results.csv). Semi-iNat und In-Shop (Spalten der Korrelationstabelle): Geometrieläufe seit 09.10.2026.
Eingabe: Sichten geometrie/epochen_cars.csv und laeufe_cars.csv des Datenpakets (gesammelt von sammle_geometrie.py cars).
"""
import datensammlung as sammlung
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import stil
import geodaten

stil.apply()
RD = "geometrie/"
TAB = sammlung.output_dir("kapitel/ergebnisse_tabellen")
ANH = sammlung.output_dir("kapitel/anhang_tabellen")
M = stil.METHOD_ORDER
C, MK, LB = stil.METHOD_COLORS, stil.METHOD_MARKERS, stil.METHOD_LABELS
KS = [4, 8, 16, 32]
REF = dict(color=stil.BLACK, lw=0.8, ls=(0, (4, 2)))
# Test-MAP@R der DINOv2-Features auf Cars196 (Tabelle referenz) und des Heads vor dem Training mit Seed 2 in %
_BASE = next(r for r in sammlung.read_json("ausgangsleistung/untrained_baseline.json")["records"] if r["dataset"] == "Cars196")
DINO_MAP = 100 * _BASE["pretrained_dinov2_test"]["mean_average_precision_at_r"]
HEAD_MAP = 100 * next(s for s in _BASE["initial_head_per_seed"] if s["seed"] == 2)["test"]["mean_average_precision_at_r"]

E = sammlung.read_csv(RD + "epochen_cars.csv", low_memory=False)
R = sammlung.read_csv(RD + "laeufe_cars.csv")
assert len(R) == 400 and R.test_map.notna().all()


def at_selected():
    rows = []
    for _, r in R.iterrows():
        e = E[(E.method == r.method) & (E.loss == r.loss) & (E.k == r.k) & (E.fold == r.fold)]
        s = e[e.epoch <= r.selected_epoch]
        x = s.loc[s.epoch.idxmax()].copy()
        x["selected_epoch"], x["test_map"] = r.selected_epoch, 100 * r.test_map
        rows.append(x)
    return pd.DataFrame(rows)


S = at_selected()
bb = R  # DINOv2-Features auf denselben Bildern, je Fold gemessen
h0 = E[E.epoch == -1]  # Head vor dem ersten Optimierungsschritt
# Mittel über alle fünf Verlustfunktionen und vier Folds (seit 01.10.2026, vorher nur Circle)
cir = S.groupby(["method", "k"]).agg(
    rho=("train_rho_roth", "mean"), pr=("train_pi_ratio_roth", "mean"), m=("test_map", "mean")).reset_index()

# ------------------------------------------------------------------ Budget (alle Verlustfunktionen)
fig, axes = plt.subplots(1, 3, figsize=stil.SIZES["full"])
DOT = dict(color=stil.BLACK, lw=0.8, ls=(0, (1, 2)))
for ax, letter, col, lab, ref in ((axes[0], "a", "pr", r"$\pi_{\mathrm{ratio}}$", h0.train_pi_ratio_roth.mean()),
                                  (axes[1], "b", "rho", r"$\rho$", h0.train_rho_roth.mean()),
                                  (axes[2], "c", "m", "MAP@R in %", HEAD_MAP)):
    for m in M:
        d = cir[cir.method == m].sort_values("k")
        ax.plot(d.k, d[col], color=C[m], marker=MK[m], label=LB[m])
    ax.axhline(ref, **REF)
    ax.set_xscale("log", base=2)
    ax.set_xticks(KS, [str(k) for k in KS])
    ax.minorticks_off()
    stil.panel(ax, letter, {"pr": r"$\pi_{\mathrm{ratio}}$", "rho": r"Spektraler Abfall $\rho$", "m": "Test-MAP@R (Mean)"}[col])
    ax.set_ylabel(lab + (" des Trainingspools" if col != "m" else ""))
axes[1].axhline(bb.bb_train_rho_roth.mean(), **DOT)
axes[2].set_ylim(*stil.grenzen(15, 48, list(cir.m) + [HEAD_MAP], 1))
fig.supxlabel("Gelabelte Bilder je Klasse $k$", fontsize=10)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=5)
stil.save(fig, "geometrie_budget")

# ------------------------------------------------------------------ Korrelation (alle 100 Kombinationen)
r = S.groupby(["method", "loss", "k"]).agg(n=("test_map", "count"), rho=("train_rho_roth", "mean"),
                                           pr=("train_pi_ratio_roth", "mean"), prv=("validation_pi_ratio_roth", "mean"),
                                           prt=("test_pi_ratio_roth", "mean"), m=("test_map", "mean"),
                                           se=("selected_epoch", "mean"),
                                           proto=("prototype_unlabeled_accuracy", "mean")).reset_index()
assert (r.n == 4).all() and len(r) == 100
fig, axes = plt.subplots(1, 4, figsize=stil.SIZES["full"], sharey=True, sharex=True)
for j, (ax, k) in enumerate(zip(axes, KS)):
    g = r[r.k == k]
    for m in M:
        d = g[g.method == m]
        ax.scatter(d.pr, d.m, s=12, color=C[m], marker=MK[m], label=LB[m])
    stil.panel(ax, "abcd"[j], f"$k={k}$, $r={stil.de(pearsonr(g.pr, g.m)[0], 2)}$")
    if j == 0:
        ax.set_ylabel("Test-MAP@R in %")
fig.supxlabel(r"$\pi_{\mathrm{ratio}}$ des Trainingspools", fontsize=10)
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="outside upper center", ncol=5)
stil.save(fig, "geometrie_korrelation")
for col in ["pr", "prv", "prt", "rho", "proto"]:
    print(col, "gesamt", round(pearsonr(r[col], r.m)[0], 3), round(spearmanr(r[col], r.m)[0], 3),
          "innerhalb k", [round(pearsonr(g[col], g.m)[0], 2) for _, g in r.groupby("k")])
print("gewählte Epoche vs. Test-MAP@R, Spearman innerhalb k", [round(spearmanr(g.se, g.m)[0], 2) for _, g in r.groupby("k")])

# ------------------------------------------------------------------ Tabelle der Korrelationen (Hauptteil)
kk = S.groupby(["method", "loss", "k"]).mean(numeric_only=True).reset_index()
# Layout der Arbeit (30.09.): Zeilen nach Teilmenge gruppiert
ROWS = [("Trainingspool", [("$\\pi_{\\mathrm{intra}}$", "train_pi_intra_roth"), ("$\\pi_{\\mathrm{inter}}$", "train_pi_inter_roth"),
                           ("$\\pi_{\\mathrm{ratio}}$", "train_pi_ratio_roth"), ("$\\rho$", "train_rho_roth")]),
        ("Validierungsklassen", [("$\\pi_{\\mathrm{intra}}$", "validation_pi_intra_roth"),
                                 ("$\\pi_{\\mathrm{inter}}$", "validation_pi_inter_roth"),
                                 ("$\\pi_{\\mathrm{ratio}}$", "validation_pi_ratio_roth")]),
        ("Ungelabelte Bilder", [("Prototyp", "prototype_unlabeled_accuracy")])]
# Semi-iNat, In-Shop und In-Shop-Serie mit gelabelten Klassen (seit 09.10.2026): je eine Spalte über alle gültigen
# Kombinationen, aus generiert/geometrie/korrelation_weitere.csv (fig_geometrie_weitere.py, pi aus *_multi)
KW = sammlung.read_csv(RD + "korrelation_weitere.csv").set_index(["ds", "row"])
WDS = ["semiinat", "inshop", "inshop10"]
NW = {ds: int(KW.loc[ds].n.iloc[0]) for ds in WDS}


K = ["\\begin{table}[tbp]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\caption[Korrelation der Geometrie mit der Retrieval-Leistung]{Korrelation nach Pearson zwischen Maßen am gewählten Stand und der Test-MAP@R der einzelnen Fold-Modelle (Seed~2, Mittel über vier Folds), auf Cars196 über alle 100 Kombinationen aus Verfahren, Verlustfunktion und Budget und innerhalb jedes Budgets, auf Semi-iNat-2021, In-Shop und der In-Shop-Serie mit gelabelten Klassen über alle gültigen Kombinationen und Budgets ("
     + f"{NW['semiinat']}, {NW['inshop']} und {NW['inshop10']}" + "). Prototyp: Anteil der ungelabelten Bilder mit richtigem nächstem Klassenprototyp, in der Serie mit gelabelten Klassen nicht definiert.}",
     "\\label{tab:e:korrelation}",
     "\\begin{tabular}{@{}l S[table-format=-1.2] S[table-format=-1.2] S[table-format=-1.2] S[table-format=-1.2] S[table-format=-1.2] S[table-format=-1.2] S[table-format=-1.2] S[table-format=-1.2]@{}}",
     "\\toprule",
     "& \\multicolumn{5}{c}{Cars196} & {Semi-iNat} & \\multicolumn{2}{c}{In-Shop} \\\\",
     "\\cmidrule(lr){2-6}\\cmidrule(l){8-9}",
     "{Maß} & {alle} & {$k=4$} & {$k=8$} & {$k=16$} & {$k=32$} & {alle} & {$k$, alle} & {Klassen, alle} \\\\",
     "\\midrule"]
for i, (sub, rows) in enumerate(ROWS):
    if i:
        K.append("\\addlinespace")
    K.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{sub}}}}} \\\\*")
    for name, col in rows:
        vals = [pearsonr(kk[col], kk.test_map)[0]] + [pearsonr(g[col], g.test_map)[0] for _, g in kk.groupby("k")]
        row = col.replace("_roth", "") if col != "train_rho_roth" else col
        wv = [KW.loc[(ds, row)].pearson for ds in WDS]
        K.append(f"\\quad {name} & " + " & ".join(f"{v:.2f}" for v in vals) + " & "
                 + " & ".join("{--}" if pd.isna(v) else f"{v:.2f}" for v in wv) + " \\\\")
K.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(TAB + "korrelation.tex", "w", encoding="utf-8").write("\n".join(K) + "\n")


# ------------------------------------------------------------------ Tabelle im Hauptteil (alle Verlustfunktionen)
def n(x, nd=2):
    return f"{x:.{nd}f}"


full = S.groupby(["method", "k"]).agg(
    rho=("train_rho_roth", "mean"), pin=("train_pi_intra_roth", "mean"), pie=("train_pi_inter_roth", "mean"),
    pr=("train_pi_ratio_roth", "mean"), uni=("train_embedding_uniformity", "mean"),
    er=("train_embedding_effective_rank", "mean"), m=("test_map", "mean"), ep=("selected_epoch", "mean")).reset_index()
T = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Geometrie des Embedding-Raums nach dem Training]{Geometrie des Trainingspools am per Validierung gewählten Stand (Cars196, Seed~2, Mittel über fünf Verlustfunktionen und vier Folds). Head ($-1$): Head vor dem ersten Optimierungsschritt. $\\rho$: spektraler Abfall, $\\pi_{\\mathrm{intra}}$ und $\\pi_{\\mathrm{inter}}$: mittlere Kosinusabstände innerhalb der Klassen und zwischen den Klassenmittelpunkten, $\\mathcal{L}_{\\mathrm{unif}}$: Uniformity, $r_{\\mathrm{eff}}$: effektiver Rang, Epoche: Mittel der gewählten Epochen, MAP@R: Test-MAP@R der einzelnen Fold-Modelle in \\%.}",
     "\\label{tab:e:geometrie}",
     "\\begin{tabular}{@{}ll S[table-format=1.2] S[table-format=1.2] S[table-format=1.2] S[table-format=1.2] S[table-format=-1.2] S[table-format=2.1] S[table-format=3.0] S[table-format=2.1]@{}}\n\\toprule",
     "{$k$} & {Modell} & {$\\rho$} & {$\\pi_{\\mathrm{intra}}$} & {$\\pi_{\\mathrm{inter}}$} & {$\\pi_{\\mathrm{ratio}}$} & {$\\mathcal{L}_{\\mathrm{unif}}$} & {$r_{\\mathrm{eff}}$} & {Epoche} & {MAP@R} \\\\\n\\midrule"]
T.append(f"{{--}} & DINOv2 & {n(bb.bb_train_rho_roth.mean())} & {n(bb.bb_train_pi_intra_roth.mean())} & "
         f"{n(bb.bb_train_pi_inter_roth.mean())} & {n(bb.bb_train_pi_ratio_roth.mean())} & "
         f"{n(bb.bb_train_embedding_uniformity.mean())} & {n(bb.bb_train_embedding_effective_rank.mean(), 1)} & {{--}} & "
         f"{DINO_MAP:.1f} \\\\")
T.append(f"{{--}} & Head ($-1$) & {n(h0.train_rho_roth.mean())} & {n(h0.train_pi_intra_roth.mean())} & "
         f"{n(h0.train_pi_inter_roth.mean())} & {n(h0.train_pi_ratio_roth.mean())} & "
         f"{n(h0.train_embedding_uniformity.mean())} & {n(h0.train_embedding_effective_rank.mean(), 1)} & {{--}} & "
         f"{HEAD_MAP:.1f} \\\\\n\\addlinespace")
for k in [4, 32]:
    for m in M:
        x = full[(full.method == m) & (full.k == k)].iloc[0]
        first = f"{k}" if m == "sup" else ""
        name = LB[m].replace("LRML", "\\ac{LRML}").replace("ÜBl.", "\\UEBL")
        T.append(f"{first} & {name} & {n(x.rho)} & {n(x.pin)} & {n(x.pie)} & {n(x.pr)} & {n(x.uni)} & {n(x.er, 1)} & "
                 f"{x.ep:.0f} & {n(x.m, 1)} \\\\")
    if k == 4:
        T.append("\\addlinespace")
T.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(TAB + "geometrie.tex", "w", encoding="utf-8").write("\n".join(T) + "\n")
print(full.round(3).to_string())

# ------------------------------------------------------------------ Anhang: alle Verlustfunktionen, k=4 und k=32
g = S.groupby(["k", "loss", "method"]).agg(lab=("labeled_pi_intra_roth", "mean"), unl=("unlabeled_pi_intra_roth", "mean"),
                                           m=("test_map", "mean")).reset_index()
A = ["\\begin{table}[tbp]\n\\centering\\small\\setlength{\\tabcolsep}{3.2pt}",
     "\\caption[Geometrie nach dem Training je Verlustfunktion]{Mittlerer Kosinusabstand innerhalb der Klassen $\\pi_{\\mathrm{intra}}$ für die ungelabelten und, in Klammern, die gelabelten Bilder des Trainingspools und Test-MAP@R der einzelnen Fold-Modelle am per Validierung gewählten Stand für alle Verlustfunktionen (Cars196, Seed~2, Mittel über vier Folds). Fett: kleinstes $\\pi_{\\mathrm{intra}}$ der ungelabelten Bilder und größte MAP@R je Zeile, bei gleichen gerundeten Werten nach dem ungerundeten Wert.}",
     "\\label{tab:a:geometrie}",
     "\\begin{tabular}{@{}l S[table-format=1.2] S[table-format=1.2] S[table-format=1.2] S[table-format=1.2] S[table-format=1.2]"
     " S[table-format=2.1] S[table-format=2.1] S[table-format=2.1] S[table-format=2.1] S[table-format=2.1]@{}}",
     "\\toprule",
     "& \\multicolumn{5}{c}{$\\pi_{\\mathrm{intra}}$ ungelabelt (gelabelt)} & \\multicolumn{5}{c}{Test-MAP@R in \\%} \\\\",
     "\\cmidrule(lr){2-6}\\cmidrule(l){7-11}",
     "{Verlust} & " + " & ".join("{" + LB[m].replace("LRML", "\\ac{LRML}").replace("ÜBl.", "\\UEBL") + "}" for m in M) + " & "
     + " & ".join("{" + LB[m].replace("LRML", "\\ac{LRML}").replace("ÜBl.", "\\UEBL") + "}" for m in M) + " \\\\",
     "\\midrule"]


def bold(v, is_best, fmt):
    s = fmt.format(v)
    return f"\\bfseries {s}" if is_best else s


for k in [4, 32]:
    if k == 32:
        A.append("\\addlinespace")
    A.append(f"\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{$k={k}$}}}} \\\\")
    for loss in stil.LOSS_ORDER:
        x = g[(g.k == k) & (g.loss == loss)].set_index("method").loc[M]
        umin, mmax = x.unl.min(), x.m.max()
        cells = [stil.LOSS_LABELS[loss]]
        cells += [bold(x.loc[m, "unl"], x.loc[m, "unl"] == umin, "{:.2f}") for m in M]
        cells += [bold(x.loc[m, "m"], x.loc[m, "m"] == mmax, "{:.1f}") for m in M]
        A.append(" & ".join(cells) + " \\\\")
        # gelabelte Teilmenge in einer eigenen Zeile darunter, in Klammern
        A.append(" & " + " & ".join("{(" + f"{x.loc[m, 'lab']:.2f}".replace(".", ",") + ")}" for m in M)
                 + " & & & & & \\\\")
A.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(ANH + "geometrie_verluste.tex", "w", encoding="utf-8").write("\n".join(A) + "\n")
print(g[g.k.isin([4, 32])].pivot_table(index=["k", "loss"], columns="method", values=["unl", "lab", "m"]).round(3).to_string())

# ------------------------------------------------------------------ Anhang: Reinheit des kNN-Graphen von LRML
# Anteil gleichklassiger Kanten des kNN-Graphen über den ganzen Trainingspool (Geometrieläufe, Zeile "train"), mit
# Nachbarzahl und binärer Gewichtung der LRML-Konfiguration; Head (-1): vor dem ersten Optimierungsschritt.
L = S[S.method == "lrml"]
nn = L.groupby("loss").graph_n_neighbors.first().astype(int)
pur = (100 * L.groupby(["loss", "k"]).graph_edge_purity.mean()).unstack()
pre = 100 * E[(E.method == "lrml") & (E.epoch == -1)].groupby("loss").graph_edge_purity.mean()
P = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Reinheit des $k$NN-Graphen von LRML]{Reinheit des $k$NN-Graphen von \\ac{LRML} über den Trainingspool am per "
     "Validierung gewählten Stand: Anteil der Kanten zwischen Bildern derselben Klasse in \\%, mit der Nachbarzahl der "
     "Phase-1-Konfiguration (Cars196, Seed~2, Mittel über vier Folds). Head ($-1$): vor dem ersten Optimierungsschritt. "
     "Vergleichbar nur innerhalb einer Zeile, weil sich die Nachbarzahlen unterscheiden.}",
     "\\label{tab:a:reinheit:lrml}",
     "\\begin{tabular}{@{}l S[table-format=2.0] S[table-format=2.1] S[table-format=2.1] S[table-format=2.1] S[table-format=2.1] S[table-format=2.1]@{}}",
     "\\toprule",
     "{Verlust} & {Nachbarn} & {Head ($-1$)} & {$k=4$} & {$k=8$} & {$k=16$} & {$k=32$} \\\\",
     "\\midrule"]
for loss in stil.LOSS_ORDER:
    P.append(f"{stil.LOSS_LABELS[loss].replace('MS', chr(92) + 'ac{MS}')} & {nn[loss]} & {pre[loss]:.1f} & "
             + " & ".join(f"{pur.loc[loss, k]:.1f}" for k in KS) + " \\\\")
P.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(ANH + "geometrie_reinheit_lrml.tex", "w", encoding="utf-8").write("\n".join(P) + "\n")
print("\n".join(P))
