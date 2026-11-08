"""Tabellen für Phase 2 (Ablationen im Hauptteil und alle Varianten im Anhang).
Gestaltung nach review/Designrichtlinien.md (booktabs, S-Spalten, eine Nachkommastelle, keine Farbe)."""
import datensammlung as sammlung
import numpy as np
import pandas as pd

TAB = sammlung.output_dir("kapitel/ergebnisse_tabellen")

# ------------------------------------------------------------------ Phase 2
V = sammlung.read_csv("phase2/variants.csv")
V["variant"] = V.variant.fillna("defaults").replace("", "defaults")
# Bei LP deckt die Kontrolle mit abgeschaltetem SSL-Mechanismus die Variante ohne ungelabelte Bilder ab (6.11.2026).
V = V[~((V.method == "lp") & (V.variant == "pool_labeled"))]
LOSS2 = ["arc", "cir", "ms", "nt", "prx"]
LOSS_NAME = {"arc": "ArcFace", "cir": "Circle", "ms": "\\acs{MS}", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
METHOD_NAME = {"lp": "LP", "lrml": "\\acs{LRML}", "hoffer": "Hoffer", "slade": "SLADE"}


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

ROWS2 = [("zero_weight", "SSL-Mechanismus abgeschaltet"),
         ("pool_labeled", "ohne ungelabelte Bilder"),
         ("sup_all_shared", "Hyperparameter der \\UEBL"),
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
BLOCKS = [("lp", [r for r in ROWS2[:8] if r[0] != "pool_labeled"]), ("lrml", ROWS2[:8]), ("hoffer", ROWS2[:4]), ("slade", ROWS2[:4])]


def marke(x, thr):
    """Fett, wenn die Änderung über der Rauschschwelle liegt (ungerundet verglichen, Abschnitt sec:m:phase2)."""
    return ("\\bfseries " if abs(x) > thr else "") + cell(x)


def loss_table(caption, label, path):
    """Cars196 und In-Shop nebeneinander, eine Spalte je Verlustfunktion und Datensatz, Zeilen wie in BLOCKS."""
    def zeile(name, f):
        return name + " & " + " & ".join(f(ds, l) for ds in ("Cars196", "DeepFashionInShop") for l in LOSS2) + " \\\\"
    S = "*{5}{S[table-format=+2.1, retain-explicit-plus]}"
    B = ["\\begin{table}[tbp]\n\\centering", caption, f"\\label{{{label}}}",
         "\\footnotesize\\setlength{\\tabcolsep}{3.2pt}\\setlength{\\hfuzz}{1.2pt}\\renewcommand{\\arraystretch}{0.95}\n\\begin{tabular}{@{}l " + S
         + " @{\\hspace{12pt}} " + S + "@{}}\n\\toprule",
         "& \\multicolumn{5}{c}{\\textbf{Cars196} ($k=16$)} & \\multicolumn{5}{c}{\\textbf{In-Shop} ($k=2$)} \\\\\n\\cmidrule(lr){2-6}\\cmidrule(l){7-11}",
         "{Variante} & " + " & ".join(["{Arc}", "{Cir}", "{\\acs{MS}}", "{NT}", "{PA}"] * 2) + " \\\\\n\\midrule",
         zeile("\\textit{\\UEBL (MAP@R)}", lambda ds, l: f"{val2(ds, 'sup', 'defaults', l):.1f}")]
    for m, rows in BLOCKS:
        B.append(f"\\midrule\n\\multicolumn{{11}}{{@{{}}l}}{{\\textbf{{{METHOD_NAME[m]}}}}} \\\\")
        B.append(zeile("\\quad\\textit{regulär (MAP@R)}", lambda ds, l: f"{val2(ds, m, 'defaults', l):.1f}"))
        B.append(zeile("\\quad\\textit{Rauschschwelle}", lambda ds, l: f"{THR[(ds, l, m)]:.1f}"))
        B.append("\\cmidrule{1-11}")
        for key, lab in rows:
            if key.startswith("pl_once"):
                B.append("\\addlinespace[2pt]")  # Eingriffe in Pseudo-Labels und Graph als eigene Gruppe
            B.append(zeile("\\quad " + lab.replace("Labels bzw.\\ Graph eingefroren", "eingefroren"),
                           lambda ds, l: marke(delta(ds, m, key, l), THR[(ds, l, m)])))
    B.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
    open(path, "w", encoding="utf-8").write("\n".join(B) + "\n")


AN = sammlung.output_dir("kapitel/anhang_tabellen")
loss_table('\\caption[Ablationen der Phase 2 auf Cars196 und In-Shop]{Ablationen der Phase 2 auf Cars196 und In-Shop (Seed 2): Änderung der Test-MAP@R (Mean) gegenüber der regulären Konfiguration in Prozentpunkten, \\UEBL und regulär absolut in \\%. Eingriffe wie in Tabelle \\ref{tab:phase2:eingriffe} und Abschnitt \\ref{sec:m:phase2}, eingefroren: Pseudo-Labels bzw.\\ Graph. Fett: Änderung größer als die Rauschschwelle. Ohne Warm-up: gleich der regulären Konfiguration, wo Phase 1 kein Warm-up gewählt hat. Arc: ArcFace, Cir: Circle, NT: NT-Xent, PA: Proxy-Anchor.}',
           "tab:e:ablationen:verluste", TAB + "ablationen_verluste.tex")

# Anhang: alle Varianten als absolute Test-Mean-MAP@R, je Datensatz eine Tabelle mit allen fünf Verlustfunktionen
NAMES = {"defaults": "reguläre Konfiguration", "diag": "Wiederholung mit Diagnose", "zero_weight": "SSL-Mechanismus abgeschaltet",
         "pool_labeled": "ohne ungelabelte Bilder", "sup_all_shared": "Hyperparameter der \\UEBL", "no_warmup": "ohne Warm-up",
         "pl_once": "Pseudo-Labels eingefroren", "graph_once": "Graph eingefroren", "graph_rewired": "Graph umverdrahtet",
         "graph_oracle_pruned": "Oracle-Kanten entfernt", "graph_random_pruned": "zufällige Kanten entfernt"}
SWEEP = {"n_neighbors": "Nachbarzahl", "confidence_threshold": "Konfidenzschwelle", "regularizer_target_ratio": "$\\eta$",
         "distance_scale": "Distanzskala $\\gamma_{\\mathrm{H}}$", "warmup_epochs": "Warm-up"}


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
         "\\multicolumn{6}{@{}l}{\\small Tabelle \\thetable{} (Fortsetzung)}\\\\\n" + head + "\n\\endhead",
         "\\midrule\n\\multicolumn{6}{r@{}}{\\emph{Fortsetzung auf der nächsten Seite}}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot",
         "\\UEBL & " + " & ".join(f2(val2(ds, "sup", "defaults", l)) for l in LOSS2) + " \\\\"]
    n = 0
    for m in ["lp", "lrml", "hoffer", "slade"]:
        L.append(f"\\addlinespace\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{METHOD_NAME[m]}}}}} \\\\*")
        L.append("\\quad Rauschschwelle & " + " & ".join(f"{THR[(ds, l, m)]:.1f}" for l in LOSS2) + " \\\\")
        for v in sorted(V[(V.dataset == ds) & (V.method == m)].variant.unique(), key=order):
            L.append(f"\\quad {label(v)} & " + " & ".join(f2(val2(ds, m, v, l)) for l in LOSS2) + " \\\\")
            n += 1
    L.append("\\end{longtable}\\endgroup")
    open(path, "w", encoding="utf-8").write("\n".join(L) + "\n")
    print(path, n, "Zeilen")


all_variants("Cars196", "\\caption[Alle Varianten der Phase 2 auf Cars196]{Alle Varianten der Phase 2 auf Cars196 ($k=16$, Seed 2): Test-MAP@R (Mean) in \\%. Bezeichnungen wie in Tabelle \\ref{tab:phase2:eingriffe} und Abschnitt \\ref{sec:m:phase2}, Rauschschwelle $2\\sqrt{2}\\,\\sigma$ in Prozentpunkten (Abschnitt \\ref{sec:m:phase2}). Bei \\acs{LRML} sind \\enquote{ohne Warm-up} und \\enquote{Warm-up $= 0$} dieselbe Einstellung. --: Lauf nicht vorhanden.}\\label{tab:anhang:phase2}",
             AN + "phase2_varianten.tex")
all_variants("DeepFashionInShop", "\\caption[Alle Varianten der Phase 2 auf In-Shop]{Alle Varianten der Phase 2 auf In-Shop ($k=2$, Seed 2), aufgebaut wie Tabelle \\ref{tab:anhang:phase2}. Fehlende SLADE-Werte: nicht untersucht, Sweeps in Tabelle \\ref{tab:anhang:phase2:slade}. --: Lauf nicht vorhanden.}\\label{tab:anhang:phase2:inshop}",
             AN + "phase2_varianten_inshop.tex")
