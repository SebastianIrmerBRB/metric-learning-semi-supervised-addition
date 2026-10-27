"""Anhangtabellen der Phase 2 mit der Concat-MAP@R (Ensemble der Fold-Heads), Gegenstück zu phase2_varianten.tex und
phase2_varianten_inshop.tex aus tab_phase02.py (seit 07.10. je Datensatz eine Tabelle mit allen fünf Verlustfunktionen;
vorher Cars196 mit In-Shop/Circle in einer Tabelle und die In-Shop-2x2-Zerlegung getrennt). Schwelle: 2*sqrt(2)*sigma der Concat-MAP@R aus den fünf Seeds der Phase 1.
Seit 09.10.2026 fehlt kein Concat-Wert mehr, seit 22.10.2026 stammen alle aus cv_summary.json (Wiederholungen, sammle_cvsum.py).
Die Legende "--" steht nur noch in der Bildunterschrift, wenn eine Zelle leer ist."""
import datensammlung as sammlung
import numpy as np
import pandas as pd

AN = sammlung.output_dir("kapitel/anhang_tabellen")
V = sammlung.read_csv("phase2/variants.csv")
V["variant"] = V.variant.fillna("defaults").replace("", "defaults")
LOSS2 = ["arc", "cir", "ms", "nt", "prx"]
LOSS_NAME = {"arc": "ArcFace", "cir": "Circle", "ms": "\\ac{MS}", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
METHOD_NAME = {"lp": "LP", "lrml": "\\ac{LRML}", "hoffer": "Hoffer", "slade": "SLADE"}
IS = "DeepFashionInShop"


def val(ds, m, var, loss="cir"):
    g = V[(V.dataset == ds) & (V.loss == loss) & (V.method == m) & (V.variant.str.fullmatch(var))]
    return g.concat_value.iloc[0] if len(g) else np.nan


PREFIX = {"sup": "sup", "lp": "iscen", "lrml": "lrml", "hoffer": "hoffer", "slade": "slade"}
THR = {}
for ds, path, k in (("Cars196", "cars_ergebnisse", 16), (IS, "inshop_ergebnisse", 2)):
    R = sammlung.read_csv(f"{path}/data/phase1_runs.csv")
    # LRML: Phase 2 beruht auf den Wiederholungen der multivariaten Suche, die Schwellen kommen wie bei den übrigen
    # Verfahren aus den Phase-1-Läufen.
    R = R[(R.scope == "concat") & (R.k == k)]
    for method, g in R.groupby("method"):
        pre, loss = method.rsplit("_", 1)
        m = {v: k2 for k2, v in PREFIX.items()}[pre]
        THR[(ds, loss, m)] = 2 * np.sqrt(2) * 100 * g.mean_average_precision_at_r.std(ddof=1)

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


def thr(ds, loss, m):
    x = THR.get((ds, loss, m), np.nan)
    return "{--}" if np.isnan(x) else f"{x:.1f}"


# Alle Varianten je Datensatz mit allen fünf Verlustfunktionen, Aufbau wie tab:anhang:phase2 bzw. tab:anhang:phase2:inshop
def all_variants(ds, caption, path):
    head = "\\toprule\nVariante & " + " & ".join(f"{{{LOSS_NAME[l]}}}" for l in LOSS2) + " \\\\\n\\midrule"
    L = ["\\begingroup\\small\\setstretch{1}\\setlength{\\tabcolsep}{4pt}",
         "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash\\hangindent=2em\\hangafter=1}p{5.1cm} *{5}{S[table-format=2.2]}@{}}",
         caption + "\\\\", head + "\n\\endfirsthead",
         "\\multicolumn{6}{@{}l}{\\small Tabelle~\\thetable{} (Fortsetzung)}\\\\\n" + head + "\n\\endhead",
         "\\bottomrule\n\\endlastfoot",
         "\\UEBL & " + " & ".join(f2(val(ds, "sup", "defaults", l)) for l in LOSS2) + " \\\\",
         "Schwelle \\UEBL & " + " & ".join(thr(ds, l, "sup") for l in LOSS2) + " \\\\"]
    n = 0
    for m in ["lp", "lrml", "hoffer", "slade"]:
        L.append(f"\\addlinespace\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{METHOD_NAME[m]}}}}} \\\\*")
        L.append("\\quad Schwelle & " + " & ".join(thr(ds, l, m) for l in LOSS2) + " \\\\")
        for v in sorted(V[(V.dataset == ds) & (V.method == m)].variant.unique(), key=order):
            L.append(f"\\quad {label(v)} & " + " & ".join(f2(val(ds, m, v, l)) for l in LOSS2) + " \\\\")
            n += 1
    L.append("\\end{longtable}\\endgroup")
    if not any("{--}" in x for x in L[5:]):
        L[2] = L[2].replace(" --: Lauf oder Concat-Auswertung nicht vorhanden.", "")
    open(AN + path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(path, n, "Zeilen")


SCHWELLE = "Schwelle: $2\\sqrt{2}\\,\\sigma$ in Prozentpunkten, mit $\\sigma$ der Seed-Standardabweichung der Concat-MAP@R in Phase~1 (Abschnitt~\\ref{sec:m:phase2})."
all_variants("Cars196", "\\caption[Alle Varianten der Phase~2 auf Cars196 als Ensemble]{Alle Varianten der Phase~2 auf Cars196 wie in Tabelle~\\ref{tab:anhang:phase2}, aber als Test-Concat-MAP@R in \\%, also für das Ensemble der Fold-Heads (Seed~2, $k=16$). " + SCHWELLE + " Bezeichnungen der Varianten wie in Tabelle~\\ref{tab:e:ablationen:verluste}. --: Lauf oder Concat-Auswertung nicht vorhanden.}\\label{tab:anhang:phase2:concat}",
             "phase2_varianten_concat.tex")
all_variants(IS, "\\caption[Alle Varianten der Phase~2 auf In-Shop als Ensemble]{Alle Varianten der Phase~2 auf In-Shop wie in Tabelle~\\ref{tab:anhang:phase2:inshop}, aber als Test-Concat-MAP@R in \\%, also für das Ensemble der Fold-Heads (Seed~2, $k=2$). " + SCHWELLE + " Bezeichnungen der Varianten wie in Tabelle~\\ref{tab:e:ablationen:verluste}. --: Lauf oder Concat-Auswertung nicht vorhanden.}\\label{tab:anhang:phase2:inshop:concat}",
             "phase2_varianten_inshop_concat.tex")

miss = V[V.concat_value.isna()]
print(len(miss), "von", len(V), "Läufen ohne Concat")
