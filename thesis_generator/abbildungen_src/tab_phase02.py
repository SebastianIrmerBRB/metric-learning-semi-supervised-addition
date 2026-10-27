"""Tabellen für Phase 2 (Ablationen im Hauptteil und alle Varianten im Anhang).
Gestaltung nach review/Designrichtlinien.md (booktabs, S-Spalten, eine Nachkommastelle, keine Farbe)."""
import datensammlung as sammlung
import numpy as np
import pandas as pd

TAB = sammlung.output_dir("kapitel/ergebnisse_tabellen")

# ------------------------------------------------------------------ Phase 2
V = sammlung.read_csv("phase2/variants.csv")
V["variant"] = V.variant.fillna("defaults").replace("", "defaults")
LOSS2 = ["arc", "cir", "ms", "nt", "prx"]
LOSS_NAME = {"arc": "ArcFace", "cir": "Circle", "ms": "\\ac{MS}", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
METHOD_NAME = {"lp": "LP", "lrml": "\\ac{LRML}", "hoffer": "Hoffer", "slade": "SLADE"}


def val2(ds, m, var, loss="cir"):
    g = V[(V.dataset == ds) & (V.loss == loss) & (V.method == m) & (V.variant.str.fullmatch(var))]
    return g.mean_value.iloc[0] if len(g) else np.nan


def cell(x):
    if np.isnan(x):
        return "{--}"
    return f"{0.0 if abs(x) < 0.05 else x:+.1f}"


# Schwelle 2*sqrt(2)*sigma aus den fünf Seeds der Phase 1 (Cars196 k=16, In-Shop k=2), Abschnitt sec:m:phase2
PREFIX = {"sup": "sup", "lp": "iscen", "lrml": "lrml", "hoffer": "hoffer", "slade": "slade"}
THR = {}
for ds, path, k in (("Cars196", "cars_ergebnisse", 16), ("DeepFashionInShop", "inshop_ergebnisse", 2)):
    R = sammlung.read_csv(f"{path}/data/phase1_runs.csv")
    # LRML: Phase 2 beruht auf den Wiederholungen der multivariaten Suche, die Schwellen kommen wie bei den übrigen
    # Verfahren aus den Phase-1-Läufen.
    R = R[(R.scope == "mean") & (R.k == k)]
    for (method, _), g in R.groupby(["method", "k"]):
        pre, loss = method.rsplit("_", 1)
        m = {v: k2 for k2, v in PREFIX.items()}[pre]
        THR[(ds, loss, m)] = 2 * np.sqrt(2) * 100 * g.mean_average_precision_at_r.std(ddof=1)

ROWS2 = [("zero_weight", "SSL-Term abgeschaltet"),
         ("pool_labeled", "ohne ungelabelte Bilder"),
         ("sup_all_shared", "Hyperparameter von \\UEBL"),
         ("no_warmup", "ohne Warm-up"),
         ("pl_once|graph_once", "Labels bzw.\\ Graph eingefroren"),
         ("graph_rewired", "Graph umverdrahtet"),
         ("graph_oracle_pruned", "Oracle-Kanten entfernt"),
         ("graph_random_pruned", "zufällige Kanten entfernt"),
         ("oracle_noise_0", "Oracle-Labels, \\qty{0}{\\percent} Rauschen"),
         ("oracle_noise_10", "Oracle-Labels, \\qty{10}{\\percent} Rauschen"),
         ("oracle_noise_20", "Oracle-Labels, \\qty{20}{\\percent} Rauschen"),
         ("oracle_noise_40", "Oracle-Labels, \\qty{40}{\\percent} Rauschen"),
         ("oracle_noise_60", "Oracle-Labels, \\qty{60}{\\percent} Rauschen")]

def delta(ds, m, key, loss="cir"):
    return val2(ds, m, key, loss) - val2(ds, m, "defaults", loss)


# Ablationen je Datensatz mit allen fünf Verlustfunktionen (ohne Oracle-Labels und Sweeps, siehe Abbildungen und Anhang)
BLOCKS = [("lp", ROWS2[:8]), ("lrml", ROWS2[:8]), ("hoffer", ROWS2[:4]), ("slade", ROWS2[:4])]


def loss_table(ds, caption, label, path):
    B = ["\\begin{table}[tbp]\n\\centering", caption, f"\\label{{{label}}}",
         "\\footnotesize\\setlength{\\tabcolsep}{4pt}\\renewcommand{\\arraystretch}{0.95}\n\\begin{tabular}{@{}l *{5}{S[table-format=+2.1, retain-explicit-plus]}@{}}\n\\toprule",
         "{Variante} & " + " & ".join(f"{{{LOSS_NAME[l]}}}" for l in LOSS2) + " \\\\\n\\midrule",
         "\\UEBL (MAP@R) & " + " & ".join(f"{val2(ds, 'sup', 'defaults', l):.1f}" for l in LOSS2) + " \\\\"]
    for m, rows in BLOCKS:
        B.append(f"\\addlinespace\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{METHOD_NAME[m]}}}}} \\\\")
        B.append("\\quad regulär (MAP@R) & " + " & ".join(f"{val2(ds, m, 'defaults', l):.1f}" for l in LOSS2) + " \\\\")
        B.append("\\quad Schwelle & " + " & ".join(f"{THR[(ds, l, m)]:.1f}" for l in LOSS2) + " \\\\")
        for key, lab in rows:
            B.append(f"\\quad {lab} & " + " & ".join(cell(delta(ds, m, key, l)) for l in LOSS2) + " \\\\")
    B.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    open(path, "w", encoding="utf-8").write("\n".join(B) + "\n")


AN = sammlung.output_dir("kapitel/anhang_tabellen")
loss_table("Cars196", '\\caption[Ablationen der Phase~2 auf Cars196]{Ablationen der Phase~2 auf Cars196 ($k=16$, Seed~2) mit allen fünf Verlustfunktionen: Änderung der Test-MAP@R (Mean) gegenüber der regulären Konfiguration desselben Mechanismus und derselben Verlustfunktion in Prozentpunkten. \\UEBL und regulär: absolute Werte in \\%. Schwelle: $2\\sqrt{2}\\,\\sigma$ der jeweiligen Kombination, ab der ein Unterschied als Befund gilt (Abschnitt~\\ref{sec:m:phase2}). SSL-Term abgeschaltet: \\ac{SSL}-Konfiguration ohne \\ac{SSL}-Term, bei LP ohne Pseudo-Labels. Ohne ungelabelte Bilder: \\ac{SSL}-Hyperparameter, der Mechanismus erhält die gelabelten Bilder. Hyperparameter von \\UEBL: mit ungelabelten Bildern. Ohne Warm-up: wo Phase~1 kein Warm-up gewählt hat, dieselbe Einstellung wie die reguläre Konfiguration, Abweichungen zeigen dann nur die Streuung wiederholter Läufe. Eingefroren: Pseudo-Labels (LP) bzw.\\ Graph (\\ac{LRML}) nach dem Warm-up nicht mehr erneuert. Oracle-Kanten entfernt: alle Kanten zwischen verschiedenen Klassen entfernt. Oracle-Varianten verwenden verdeckte Labels und dienen nur der Diagnose. SLADE wurde außer in der $2\\times2$-Zerlegung nur mit abgeschaltetem \\ac{SSL}-Term, ohne Warm-up und in Sweeps untersucht (Tabelle~\\ref{tab:anhang:phase2:slade}). Die Oracle-Labels zeigt Abbildung~\\ref{fig:e:lpqualitaet}, alle Einzelwerte stehen in Anhang~\\ref{chap:anhang:phase2}.}',
           "tab:e:ablationen:verluste", TAB + "ablationen_verluste.tex")
loss_table("DeepFashionInShop", '\\caption[Ablationen der Phase~2 auf In-Shop]{Ablationen der Phase~2 auf In-Shop ($k=2$, Seed~2) mit allen fünf Verlustfunktionen, aufgebaut wie Tabelle~\\ref{tab:e:ablationen:verluste}. Die Oracle-Labels zeigt Abbildung~\\ref{fig:e:lpqualitaet}, die Sweeps Abbildung~\\ref{fig:e:sweeps:inshop}, alle Einzelwerte Tabelle~\\ref{tab:anhang:phase2:inshop}.}',
           "tab:e:ablationen:inshop", TAB + "ablationen_inshop.tex")

# Anhang: alle Varianten als absolute Test-Mean-MAP@R, je Datensatz eine Tabelle mit allen fünf Verlustfunktionen
NAMES = {"defaults": "reguläre Konfiguration", "diag": "Wiederholung mit Diagnose", "zero_weight": "SSL-Term abgeschaltet",
         "pool_labeled": "ohne ungelabelte Bilder", "sup_all_shared": "Hyperparameter von \\UEBL", "no_warmup": "ohne Warm-up",
         "pl_once": "Pseudo-Labels eingefroren", "graph_once": "Graph eingefroren", "graph_rewired": "Graph umverdrahtet",
         "graph_oracle_pruned": "Oracle-Kanten entfernt", "graph_random_pruned": "zufällige Kanten entfernt"}
SWEEP = {"n_neighbors": "Nachbarzahl", "confidence_threshold": "Konfidenzschwelle", "regularizer_target_ratio": "$\\eta$",
         "distance_scale": "Distanzskala $s$", "warmup_epochs": "Warm-up"}


def order(v):
    if v in NAMES:
        return (0, list(NAMES).index(v), 0.0)
    if v.startswith("oracle_noise_"):
        return (1, 0, float(v.rsplit("_", 1)[1]))
    p, x = v.rsplit("_", 1)
    return (2, list(SWEEP).index(p), float(x.replace("p", ".")))


def label(v):
    if v in NAMES:
        return NAMES[v]
    if v.startswith("oracle_noise_"):
        return f"Oracle-Labels, {v.rsplit('_', 1)[1]}\\,\\% Rauschen"
    p, x = v.rsplit("_", 1)
    return f"{SWEEP[p]} $= {x.replace('p', '{,}')}$"


def f2(x):
    return "{--}" if np.isnan(x) else f"{x:.2f}"


def all_variants(ds, caption, path):
    head = "\\toprule\nVariante & " + " & ".join(f"{{{LOSS_NAME[l]}}}" for l in LOSS2) + " \\\\\n\\midrule"
    L = ["\\begingroup\\small\\setstretch{1}\\setlength{\\tabcolsep}{4pt}",
         "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash\\hangindent=2em\\hangafter=1}p{5.1cm} *{5}{S[table-format=2.2]}@{}}",
         caption + "\\\\", head + "\n\\endfirsthead",
         "\\multicolumn{6}{@{}l}{\\small Tabelle~\\thetable{} (Fortsetzung)}\\\\\n" + head + "\n\\endhead",
         "\\bottomrule\n\\endlastfoot",
         "\\UEBL & " + " & ".join(f2(val2(ds, "sup", "defaults", l)) for l in LOSS2) + " \\\\"]
    n = 0
    for m in ["lp", "lrml", "hoffer", "slade"]:
        L.append(f"\\addlinespace\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{METHOD_NAME[m]}}}}} \\\\*")
        L.append("\\quad Schwelle & " + " & ".join(f"{THR[(ds, l, m)]:.1f}" for l in LOSS2) + " \\\\")
        for v in sorted(V[(V.dataset == ds) & (V.method == m)].variant.unique(), key=order):
            L.append(f"\\quad {label(v)} & " + " & ".join(f2(val2(ds, m, v, l)) for l in LOSS2) + " \\\\")
            n += 1
    L.append("\\end{longtable}\\endgroup")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(path, n, "Zeilen")


all_variants("Cars196", "\\caption[Alle Varianten der Phase~2 auf Cars196]{Alle Varianten der Phase~2 auf Cars196 ($k=16$, Seed~2) mit allen fünf Verlustfunktionen: Test-Mean-MAP@R in \\%. Die Differenzen zur regulären Konfiguration und zu \\UEBL ergeben sich aus den Zeilen \\enquote{reguläre Konfiguration} und \\UEBL. Schwelle: $2\\sqrt{2}\\,\\sigma$ der jeweiligen Kombination in Prozentpunkten (Abschnitt~\\ref{sec:m:phase2}). Bezeichnungen der Varianten wie in Tabelle~\\ref{tab:e:ablationen:verluste}. Bei \\ac{LRML} sind \\enquote{ohne Warm-up} und \\enquote{Warm-up $= 0$} dieselbe Einstellung. --: Lauf nicht vorhanden.}\\label{tab:anhang:phase2}",
             AN + "phase2_varianten.tex")
all_variants("DeepFashionInShop", "\\caption[Alle Varianten der Phase~2 auf In-Shop]{Alle Varianten der Phase~2 auf In-Shop ($k=2$, Seed~2) mit allen fünf Verlustfunktionen, aufgebaut wie Tabelle~\\ref{tab:anhang:phase2}: Test-Mean-MAP@R in \\%. Die $2\\times2$-Zerlegung besteht aus \\UEBL und den Zeilen \\enquote{reguläre Konfiguration}, \\enquote{ohne ungelabelte Bilder} und \\enquote{Hyperparameter von \\UEBL}. SLADE wurde außer in der $2\\times2$-Zerlegung nur mit abgeschaltetem \\ac{SSL}-Term, ohne Warm-up und in Sweeps untersucht (Tabelle~\\ref{tab:anhang:phase2:slade}). --: Lauf nicht vorhanden.}\\label{tab:anhang:phase2:inshop}",
             AN + "phase2_varianten_inshop.tex")
