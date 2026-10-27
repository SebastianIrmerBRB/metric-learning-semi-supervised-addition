"""Anhang: alle Varianten der Phase 2 auf Cars196 mit k=8 (nach sammle_phase2_k8.py), aufgebaut wie tab:anhang:phase2.

Test-Mean-MAP@R in %, Seed 2, alle fünf Verlustfunktionen, auch die SLADE-Sweeps (bei k=16 in tab:anhang:phase2:slade).
Schwelle 2*sqrt(2)*sigma aus den fünf Seeds der Phase 1 bei k=8 (Cars196). Dazu die Kennzahlen des Vergleichs mit k=16 für
den Text (Abschnitt sec:e:phase2:k8, mit den SLADE-Sweeps aus phase2/slade_sweeps.csv): Befund je Variante (über, innerhalb oder unter der Schwelle) gegenüber der regulären
Konfiguration und der regulären Konfiguration gegenüber Ü (Schwelle 2*sqrt(sigma_SSL^2 + sigma_Ü^2)), jeweils mit der
Schwelle des eigenen Budgets.
Ausgabe: kapitel/anhang_tabellen/phase2_varianten_k8.tex
"""
import datensammlung as sammlung
import numpy as np
import pandas as pd

RD = ""
OUT = sammlung.output_dir("kapitel/anhang_tabellen") + "phase2_varianten_k8.tex"
LOSS2 = ["arc", "cir", "ms", "nt", "prx"]
LOSS_NAME = {"arc": "ArcFace", "cir": "Circle", "ms": "\\ac{MS}", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
METHOD_NAME = {"lp": "LP", "lrml": "\\ac{LRML}", "hoffer": "Hoffer", "slade": "SLADE"}
PRE = {"sup": "sup", "iscen": "lp", "lrml": "lrml", "hoffer": "hoffer", "slade": "slade"}
NAMES = {"defaults": "reguläre Konfiguration", "diag": "Wiederholung mit Diagnose", "zero_weight": "SSL-Term abgeschaltet",
         "pool_labeled": "ohne ungelabelte Bilder", "sup_all_shared": "Hyperparameter von \\UEBL", "no_warmup": "ohne Warm-up",
         "pl_once": "Pseudo-Labels eingefroren", "graph_once": "Graph eingefroren", "graph_rewired": "Graph umverdrahtet",
         "graph_oracle_pruned": "Oracle-Kanten entfernt", "graph_random_pruned": "zufällige Kanten entfernt"}
SWEEP = {"n_neighbors": "Nachbarzahl", "confidence_threshold": "Konfidenzschwelle", "regularizer_target_ratio": "$\\eta$",
         "distance_scale": "Distanzskala $s$", "warmup_epochs": "Warm-up", "basis_target_ratio": "Basis-Zielgradientenverhältnis",
         "cluster_ratio": "Clusterverhältnis", "update_interval_epochs": "Teacher-Aktualisierung alle", "unlabeled_batch_size":
         "ungelabelte Batchgröße", "sd_margin": "Margin von $\\mathcal{L}_{\\mathrm{SD}}$"}


def load(path):
    V = sammlung.read_csv(path)
    V["variant"] = V.variant.fillna("defaults").replace("", "defaults")
    return V[V.dataset == "Cars196"]


def sigma(k):
    P = sammlung.read_csv(RD + "cars_ergebnisse/data/phase1_runs.csv")
    P = P[(P.scope == "mean") & (P.k == k)]
    out = {}
    for meth, g in P.groupby("method"):
        pre, loss = meth.rsplit("_", 1)
        out[(loss, PRE[pre])] = 100 * g.mean_average_precision_at_r.std(ddof=1)
    return out


V8, V16 = load(RD + "phase2_k8/variants.csv"), load(RD + "phase2/variants.csv")
S8, S16 = sigma(8), sigma(16)


def val(V, m, v, l):
    g = V[(V.method == m) & (V.variant == v) & (V.loss == l)]
    return g.mean_value.iloc[0] if len(g) else np.nan


def order(v):
    if v in NAMES:
        return (0, list(NAMES).index(v), 0.0)
    if v.startswith("oracle_noise_"):
        return (1, 0, float(v.rsplit("_", 1)[1]))
    p = next(p for p in SWEEP if v.startswith(p + "_"))
    return (2, list(SWEEP).index(p), float(v[len(p) + 1:].replace("p", ".")))


def label(v):
    if v in NAMES:
        return NAMES[v]
    if v.startswith("oracle_noise_"):
        return f"Oracle-Labels, {v.rsplit('_', 1)[1]}\\,\\% Rauschen"
    p = next(p for p in SWEEP if v.startswith(p + "_"))
    x = v[len(p) + 1:].replace("p", "{,}")
    return f"{SWEEP[p]} {x} Epochen" if p == "update_interval_epochs" else f"{SWEEP[p]} $= {x}$"


def f2(x):
    return "{--}" if np.isnan(x) else f"{x:.2f}"


cap = ("\\caption[Alle Varianten der Phase~2 auf Cars196 mit $k=8$]{Alle Varianten der Phase~2 auf Cars196 mit $k=8$ "
       "(Seed~2, Konfigurationen der Phase~1, verzögerte Patience für jeden Lauf, Abschnitt~\\ref{sec:m:phase2}) mit allen fünf "
       "Verlustfunktionen: Test-Mean-MAP@R in \\%, aufgebaut wie Tabelle~\\ref{tab:anhang:phase2}, hier mit den Sweeps von SLADE. "
       "Schwelle: $2\\sqrt{2}\\,\\sigma$ der jeweiligen Kombination aus den fünf Seeds der Phase~1 bei $k=8$ in Prozentpunkten. "
       "--: Lauf nicht vorhanden.}\\label{tab:anhang:phase2:k8}")
head = "\\toprule\nVariante & " + " & ".join(f"{{{LOSS_NAME[l]}}}" for l in LOSS2) + " \\\\\n\\midrule"
L = ["\\begingroup\\small\\setstretch{1}\\setlength{\\tabcolsep}{4pt}",
     "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash\\hangindent=2em\\hangafter=1}p{5.1cm} *{5}{S[table-format=2.2]}@{}}",
     cap + "\\\\", head + "\n\\endfirsthead",
     "\\multicolumn{6}{@{}l}{\\small Tabelle~\\thetable{} (Fortsetzung)}\\\\\n" + head + "\n\\endhead",
     "\\bottomrule\n\\endlastfoot",
     "\\UEBL & " + " & ".join(f2(val(V8, "sup", "defaults", l)) for l in LOSS2) + " \\\\"]
n = 0
for m in ["lp", "lrml", "hoffer", "slade"]:
    L.append(f"\\addlinespace\n\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{METHOD_NAME[m]}}}}} \\\\*")
    L.append("\\quad Schwelle & " + " & ".join(f"{2 * np.sqrt(2) * S8[(l, m)]:.1f}" for l in LOSS2) + " \\\\")
    for v in sorted(V8[V8.method == m].variant.unique(), key=order):
        L.append(f"\\quad {label(v)} & " + " & ".join(f2(val(V8, m, v, l)) for l in LOSS2) + " \\\\")
        n += 1
L.append("\\end{longtable}\\endgroup")
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print(OUT, n, "Zeilen")

# ---------------------------------------------------------------- Kennzahlen für den Text (k=8 gegenüber k=16)
rows = []
for k, V, S in ((16, V16, S16), (8, V8, S8)):
    for _, r in V[V.method != "sup"].iterrows():
        reg, u = val(V, r.method, "defaults", r.loss), val(V, "sup", "defaults", r.loss)
        t = 2 * np.sqrt(2) * S[(r.loss, r.method)]
        tu = 2 * np.sqrt(S[(r.loss, r.method)] ** 2 + S[(r.loss, "sup")] ** 2)
        d, du = r.mean_value - reg, r.mean_value - u
        rows.append(dict(k=k, m=r.method, l=r.loss, v=r.variant, d=d, f=np.sign(d) * (abs(d) > t), du=du, fu=np.sign(du) * (abs(du) > tu)))
# SLADE-Sweeps bei k=16 stehen in phase2/slade_sweeps.csv (ohne Wiederholungen der regulären Konfiguration, is_tuned)
SWP = sammlung.read_csv(RD + "phase2/slade_sweeps.csv")
SWP = SWP[(SWP.dataset == "Cars196") & ~SWP.is_tuned]
k8_names = set(V8[V8.method == "slade"].variant)
for r in SWP.itertuples():
    cands = [v for v in k8_names if v.startswith(r.param + "_") and abs(float(v[len(r.param) + 1:].replace("p", ".")) - r.value) < 1e-9]
    assert len(cands) == 1, (r.param, r.value, cands)
    rows.append(dict(k=16, m="slade", l=r.loss, v=cands[0], d=r.diff_mean,
                     f=np.sign(r.diff_mean) * (abs(r.diff_mean) > r.threshold_mean), du=np.nan, fu=np.nan))
D = pd.DataFrame(rows)
W = D.pivot_table(index=["m", "l", "v"], columns="k", values=["d", "f"]).dropna()
WU = D.pivot_table(index=["m", "l", "v"], columns="k", values=["du", "fu"]).dropna()
W = W.join(WU, how="left")
reg = W.xs("defaults", level="v")
print("regulär über Ü (über der Schwelle), k=16:", int((reg[("fu", 16)] > 0).sum()), " k=8:", int((reg[("fu", 8)] > 0).sum()), "von", len(reg))
print("  nicht über Ü bei k=8:", reg[reg[("fu", 8)] <= 0].index.tolist())
for m in ["lp", "lrml", "hoffer", "slade"]:
    x = reg.loc[m]
    print(f"  {m}: regulär − Ü k=16 {x[('du', 16)].min():.2f} bis {x[('du', 16)].max():.2f}, k=8 {x[('du', 8)].min():.2f} bis {x[('du', 8)].max():.2f}")
for v in ("pool_labeled", "zero_weight", "sup_all_shared"):
    x = W.xs(v, level="v")
    print(f"{v}: unter regulär über der Schwelle k=16 {int((x[('f', 16)] < 0).sum())}, k=8 {int((x[('f', 8)] < 0).sum())} von {len(x)}"
          f" | gegenüber Ü über/unter der Schwelle k=16 {int((x[('fu', 16)] > 0).sum())}/{int((x[('fu', 16)] < 0).sum())},"
          f" k=8 {int((x[('fu', 8)] > 0).sum())}/{int((x[('fu', 8)] < 0).sum())}")
P = W.drop(index="defaults", level="v")
same = int((P[("f", 16)] == P[("f", 8)]).sum())
flip = P[P[("f", 16)] * P[("f", 8)] < 0]
print(f"Varianten bei beiden Budgets: {len(P)}, gleicher Befund {same}, Wechsel zwischen Befund und Schwelle {len(P) - same - len(flip)},"
      f" umgekehrt {len(flip)}: {flip.index.tolist()}")
print("gleiches Vorzeichen der Änderung:", int((np.sign(P[("d", 16)]) == np.sign(P[("d", 8)])).sum()), "von", len(P))
print("Median |Änderung| k=16 / k=8:", round(P[("d", 16)].abs().median(), 2), round(P[("d", 8)].abs().median(), 2))
