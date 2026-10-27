"""Erzeugt die Ergebnistabellen des Hauptteils aus den Phase-1-Rohdaten (Gestaltung nach review/Designrichtlinien.md:
booktabs, siunitx-S-Spalten, keine Farbe, Kopfzellen nicht fett, eine Nachkommastelle).

referenz.tex   Ausgangsleistung (DINOv2) und supervised Referenz Ü je Budget, Mittel und Spanne über die Verlustfunktionen
gewinne.tex    gepaarter MAP@R-Gewinn der SSL-Verfahren ggü. Ü, Mittel über fünf Seeds und fünf Verlustfunktionen
               ± Standardabweichung der 25 gepaarten Differenzen (seit 18.10.2026)
klassen.tex    In-Shop-Serie mit vollständig gelabelten Klassen (fünf Verlustfunktionen), Gewinne wie gewinne.tex mit ±
../anhang_tabellen/negativ.tex  Zahl der gepaarten Vergleiche mit negativer Differenz je Zelle von gewinne.tex (seit 23.10.2026)
"""
import datensammlung as sammlung
import pandas as pd

R = ""
OUT = sammlung.output_dir("kapitel/ergebnisse_tabellen")
FAM = ["iscen", "lrml", "hoffer", "slade"]
FAMN = {"sup": "\\UEBL", "iscen": "LP", "lrml": "\\ac{LRML}", "hoffer": "Hoffer", "slade": "SLADE"}
DATA = [("cars", "Cars196", [4, 8, 16, 32]), ("cub", "\\ac{CUB}", [4, 8, 16, 32]), ("inshop", "In-Shop", [2, 4]),
        ("semiinat", "Semi-iNat-2021", [2, 4, 16, 100])]
METRIC = "mean_average_precision_at_r"
# Testleistung der DINOv2-Features vor jedem Training (Sichten ausgangsleistung/*untrained_baseline.json des Datenpakets)
BASELINE = {"cars": ("ausgangsleistung/untrained_baseline.json", "Cars196"),
            "cub": ("ausgangsleistung/cub_2026-09-25/untrained_baseline.json", "CUB"),
            "inshop": ("ausgangsleistung/untrained_baseline.json", "DeepFashionInShop"),
            "semiinat": ("ausgangsleistung/semiinat_2026-10-18/untrained_baseline.json", "iNat")}


def dino(key):
    """MAP@R, RP und P@1 der DINOv2-Features in %."""
    path, name = BASELINE[key]
    t = next(r for r in sammlung.read_json(path)["records"] if r["dataset"] == name)["pretrained_dinov2_test"]
    return tuple(100 * t[m] for m in (METRIC, "r_precision", "precision_at_1"))


DINO = {key: dino(key) for key in BASELINE}


def f1(x):
    return "{--}" if x != x else f"{x:.1f}"


def s1(x):
    if x != x:
        return "{--}"
    x = 0.0 if abs(x) < 0.05 else x
    return f"{x:+.1f}"


def spm(x, sd):
    """Gewinn ± Standardabweichung für eine siunitx-S-Spalte mit Unsicherheit."""
    return "{--}" if x != x else f"{s1(x)} +- {sd:.1f}"


def runs(key):
    r = sammlung.read_csv(R + f"{key}_ergebnisse/data/phase1_runs.csv")
    kc = "budget_percent" if "budget_percent" in r.columns else "k"
    r = r.rename(columns={kc: "b"})
    r["fam"] = r.method.str.split("_").str[0]
    r["loss"] = r.method.str.split("_").str[1]
    for m in (METRIC, "precision_at_1", "r_precision"):
        r[m] = 100 * r[m]
    return r


def paired(r, metric=METRIC):
    sup = r[r.fam == "sup"].set_index(["loss", "b", "seed", "scope"])[metric]
    x = r[r.fam != "sup"].copy()
    x["diff"] = x[metric].values - sup.reindex(pd.MultiIndex.from_frame(x[["loss", "b", "seed", "scope"]])).values
    return x


# ---------------------------------------------------------------- Referenz
ALLE = sammlung.read_csv(R + "ausgangsleistung/alle_labels.csv")
L = ["\\begin{table}[tbp]\n\\centering\\setlength{\\tabcolsep}{4.5pt}",
     "\\caption[Ausgangsleistung und supervised Referenz]{Testleistung der unveränderten DINOv2-Features und der supervised Referenz \\UEBL je Label-Budget in \\%. Für \\UEBL ist das Mittel über die fünf Verlustfunktionen angegeben, jeweils gemittelt über fünf Seeds. Die Spanne reicht von der Mean-MAP@R des schwächsten zu der des stärksten Verlusts. Mean: einzelne Fold-Modelle, Concat: Ensemble der Fold-Modelle. \\UEBL, alle Labels: alle Bilder der Trainingsklassen gelabelt, ohne ungelabelten Pool, Konfiguration der Phase~1, auf Cars196 und In-Shop nur Seed~2 (Referenz des Versuchs mit externen Bildern).}",
     "\\label{tab:e:referenz}",
     "\\begin{tabular}{@{}ll S[table-format=2.1] l S[table-format=2.1] S[table-format=2.1] S[table-format=2.1]@{}}\n\\toprule",
     " & & \\multicolumn{2}{c}{MAP@R (Mean)} & {MAP@R} & {RP} & {P@1} \\\\\n\\cmidrule(lr){3-4}",
     "{Datensatz} & {Modell} & {Mittel} & {Spanne} & {(Concat)} & {(Concat)} & {(Concat)} \\\\\n\\midrule"]
for di, (key, name, ks) in enumerate(DATA):
    r = runs(key)
    m, rp, p1 = DINO[key]
    short = "Semi-iNat" if key == "semiinat" else name  # Spaltenbreite
    L.append(f"{short} & DINOv2 & {f1(m)} & & {f1(m)} & {f1(rp)} & {f1(p1)} \\\\")
    for k in ks:
        s = r[(r.fam == "sup") & (r.b == k)]
        mean_l = s[s.scope == "mean"].groupby("loss")[METRIC].mean()
        cc = [s[s.scope == "concat"].groupby("loss")[met].mean().mean() for met in (METRIC, "r_precision", "precision_at_1")]
        rng = f"\\numrange{{{mean_l.min():.1f}}}{{{mean_l.max():.1f}}}"
        L.append(f" & \\UEBL, $k={k}$ & {f1(mean_l.mean())} & {rng} & {f1(cc[0])} & {f1(cc[1])} & {f1(cc[2])} \\\\")
    # seit 17.10.2026: Ü mit allen Labels (sammle_alle_labels.py), Mittel über Seeds je Verlustfunktion, dann über die Verluste
    a = ALLE[ALLE.key == key].groupby("loss")[["mean_map", "concat_map", "concat_rp", "concat_p1"]].mean()
    rng = f"\\numrange{{{a.mean_map.min():.1f}}}{{{a.mean_map.max():.1f}}}"
    L.append(f" & \\UEBL, alle Labels & {f1(a.mean_map.mean())} & {rng} & {f1(a.concat_map.mean())} & {f1(a.concat_rp.mean())} & {f1(a.concat_p1.mean())} \\\\")
    if di < len(DATA) - 1:
        L.append("\\addlinespace")
L.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(OUT + "referenz.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- Gewinne
G = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Gewinn der SSL-Verfahren gegenüber der supervised Referenz]{Gepaarter Gewinn der Test-MAP@R der vier \\ac{SSL}-Verfahren gegenüber der supervised Referenz \\UEBL mit derselben Verlustfunktion, demselben Budget und demselben Seed in Prozentpunkten, gemittelt über fünf Seeds und die fünf Verlustfunktionen. M: einzelne Fold-Modelle (Mean), C: Ensemble (Concat). Werte je Verlustfunktion stehen in Anhang~\\ref{chap:anhang:ergebnisse}.}",
     "\\label{tab:e:gewinne}",
     "\\footnotesize\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}l *{8}{S[table-format=+1.1(2), retain-explicit-plus, retain-zero-uncertainty]}@{}}\n\\toprule"]
SEED_SD = []
NEG = []  # Zeilen der Anhangtabelle negativ.tex, gleiche Gliederung wie gewinne.tex
for di, (key, name, ks) in enumerate(DATA):
    r = paired(runs(key))
    pad = "".join(" & {} & {}" for _ in range(4 - len(ks)))
    if di:
        G.append("\\midrule")
    # Name in einer eigenen Zeile, mit ± ist die erste Spalte zu schmal
    G.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\")
    G.append("".join(f" & \\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in ks) + pad + " \\\\")
    G.append("".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ks))))
    G.append("".join(" & {M} & {C}" for _ in ks) + pad + " \\\\")
    if di:
        NEG.append("\\midrule")
    NEG += G[-4:]  # Name, Budgets, Linien, M/C wie in gewinne.tex
    for fam in FAM:
        gc = []
        nc = []
        for k in ks:
            for scope in ("mean", "concat"):
                x = r[(r.fam == fam) & (r.b == k) & (r.scope == scope)]
                gc.append(spm(x.groupby("loss")["diff"].mean().mean(), x["diff"].std()))  # erst je Verlust, dann über die Verluste (gleich, wenn alle Läufe vorliegen); SD der gepaarten Differenzen
                SEED_SD.append(x.groupby("seed")["diff"].mean().std())  # Seed-SD des über die Verluste gemittelten Gewinns
                assert x["diff"].notna().sum() == 25, (key, fam, k, scope)
                nc.append(str(int((x["diff"] < 0).sum())))
        G.append(f"{FAMN[fam]} & " + " & ".join(gc) + pad + " \\\\")
        NEG.append(f"{FAMN[fam]} & " + " & ".join(nc) + pad + " \\\\")
G[1] = G[1].replace(  # Seed-SD erst nach der Schleife bekannt
    "gemittelt über fünf Seeds und die fünf Verlustfunktionen.",
    "gemittelt über fünf Seeds und die fünf Verlustfunktionen, $\\pm$ Standardabweichung der 25 gepaarten Differenzen "
    "(fünf Verlustfunktionen mit je fünf Seeds). Sie gibt vor allem die Unterschiede zwischen den Verlustfunktionen wieder, "
    "die Standardabweichung des gemittelten Gewinns über die Seeds liegt bei höchstens "
    + f"{max(SEED_SD):.1f}".replace(".", ",") + "~Prozentpunkten.")
G.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(OUT + "gewinne.tex", "w", encoding="utf-8").write("\n".join(G) + "\n")

# ---------------------------------------------------------------- Negative Vergleiche (Anhang)
N = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Negative Vergleiche gegenüber der supervised Referenz]{Zahl der gepaarten Vergleiche von 25 (fünf "
     "Verlustfunktionen mit je fünf Seeds), in denen ein \\ac{SSL}-Verfahren eine niedrigere Test-MAP@R erreicht als "
     "\\UEBL mit derselben Verlustfunktion, demselben Budget und demselben Seed, zu den Gewinnen in "
     "Tabelle~\\ref{tab:e:gewinne}. M: einzelne Fold-Modelle (Mean), C: Ensemble (Concat).}",
     "\\label{tab:anhang:negativ}",
     "\\small\n\\begin{tabular}{@{}l *{8}{S[table-format=2.0]}@{}}\n\\toprule"]
N += NEG
N.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(sammlung.output_dir("kapitel/anhang_tabellen") + "negativ.tex", "w", encoding="utf-8").write("\n".join(N) + "\n")

# ---------------------------------------------------------------- In-Shop mit gelabelten Klassen
r = runs("inshop10")
p = paired(r)
ks = sorted(r.b.unique())
K = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[In-Shop mit vollständig gelabelten Klassen]{In-Shop mit vollständig gelabelten Klassen: Test-MAP@R der supervised Referenz \\UEBL in \\% und gepaarter Gewinn der \\ac{SSL}-Verfahren in Prozentpunkten, gemittelt über fünf Seeds und die fünf Verlustfunktionen, beim Gewinn $\\pm$ Standardabweichung der 25 gepaarten Differenzen. Negativ: Zahl der Vergleiche mit negativem Gewinn von 25. Anteil: Anteil der vollständig gelabelten Trainingsklassen. M: Mean, C: Concat.}",
     "\\label{tab:e:klassen}",
     "\\footnotesize\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}l *{8}{S[table-format=+1.1(2), retain-explicit-plus, retain-zero-uncertainty]}@{}}\n\\toprule",
     "{Anteil}" + "".join(f" & \\multicolumn{{2}}{{c}}{{\\qty{{{k}}}{{\\percent}}}}" for k in ks) + " \\\\",
     "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ks))),
     "".join(" & {M} & {C}" for _ in ks) + " \\\\\n\\midrule"]
cells = ["{" + f1(r[(r.fam == "sup") & (r.b == k) & (r.scope == scope)][METRIC].mean()).replace(".", ",") + "}"
         for k in ks for scope in ("mean", "concat")]  # Textzellen: die S-Spalten sind für die Gewinne mit ± bemessen
K.append("\\multicolumn{9}{@{}l}{\\emph{Test-MAP@R von \\UEBL in \\%}} \\\\")
K.append("\\UEBL & " + " & ".join(cells) + " \\\\\n\\addlinespace")
K.append("\\multicolumn{9}{@{}l}{\\emph{Gewinn}} \\\\")
for fam in ["iscen", "lrml", "hoffer", "slade"]:
    cells = [spm(p[(p.fam == fam) & (p.b == k) & (p.scope == scope)]["diff"].mean(),
                 p[(p.fam == fam) & (p.b == k) & (p.scope == scope)]["diff"].std()) for k in ks for scope in ("mean", "concat")]
    K.append(f"{FAMN[fam]} & " + " & ".join(cells) + " \\\\")
K.append("\\addlinespace\n\\multicolumn{9}{@{}l}{\\emph{Negativ}} \\\\")
full = {}
for fam in ["iscen", "lrml", "hoffer", "slade"]:
    cells = []
    for k in ks:
        for scope in ("mean", "concat"):
            x = p[(p.fam == fam) & (p.b == k) & (p.scope == scope)]["diff"].dropna()
            neg = int((x < 0).sum())
            cells.append(str(neg) if len(x) == full.get(fam, 25) else f"{neg}/{len(x)}")
    K.append(f"{FAMN[fam]} & " + " & ".join(f"{{{c}}}" for c in cells) + " \\\\")
K.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(OUT + "klassen.tex", "w", encoding="utf-8").write("\n".join(K) + "\n")
print(open(OUT + "referenz.tex", encoding="utf-8").read())
