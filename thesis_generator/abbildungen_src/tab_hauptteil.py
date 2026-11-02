"""Erzeugt die Ergebnistabellen des Hauptteils aus den Phase-1-Rohdaten (Gestaltung nach review/Designrichtlinien.md:
booktabs, siunitx-S-Spalten, keine Farbe, Kopfzellen nicht fett, eine Nachkommastelle).

referenz.tex   Ausgangsleistung (DINOv2) und supervised Referenz Ü je Budget, Mittel und Spanne über die Verlustfunktionen
../anhang_tabellen/gewinne.tex  gepaarter MAP@R-Gewinn der SSL-Verfahren ggü. Ü, Mittel über fünf Seeds und fünf
               Verlustfunktionen ± Standardabweichung der 25 gepaarten Differenzen (seit 18.10.2026); eine Zeile je
               Datensatz und Budget, Spalten Verfahren je Mean/Concat; seit 30.10.2026 im Anhang, im Hauptteil
               dieselben Gewinne als Abbildung (fig_phase1_gewinne.py)
../anhang_tabellen/klassen.tex  seit 30.10.2026 im Anhang, im Hauptteil Abbildung fig_inshop10.py: In-Shop-Serie mit vollständig gelabelten Klassen (fünf Verlustfunktionen), Gewinne wie gewinne.tex mit ±
../anhang_tabellen/negativ.tex  Zahl der gepaarten Vergleiche mit negativer Differenz je Zelle von gewinne.tex (seit 23.10.2026)
"""
import datensammlung as sammlung
import pandas as pd

R = ""
OUT = sammlung.output_dir("kapitel/ergebnisse_tabellen")
FAM = ["iscen", "lrml", "hoffer", "slade"]
FAMN = {"sup": "\\UEBL", "iscen": "LP", "lrml": "\\acs{LRML}", "hoffer": "Hoffer", "slade": "SLADE"}
DATA = [("cars", "Cars196", [4, 8, 16, 32]), ("cub", "\\acs{CUB}", [4, 8, 16, 32]), ("inshop", "In-Shop", [2, 4]),
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


def fett(zeilen):
    """Fett: größter Gewinn je Spalte, sofern er auf eine Nachkommastelle gerundet positiv ist. Gleichstände nach dem
    Runden werden alle fett. zeilen: {Verfahren: (Zellen, Werte)}, gibt {Verfahren: Zellen} zurück."""
    for j in range(len(next(iter(zeilen.values()))[1])):
        top = max((round(v[j], 1) for _, v in zeilen.values() if v[j] == v[j]), default=0)
        if top > 0:
            for txt, v in zeilen.values():
                if v[j] == v[j] and round(v[j], 1) == top:
                    txt[j] = "\\bfseries " + txt[j]
    return {k: txt for k, (txt, _) in zeilen.items()}

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
# Seit 30.10.2026 kompakt und nur noch MAP@R (RP und P@1 im digitalen Anhang): je Datensatz ein Block aus einer Kopfzeile mit
# seinen Budgets und je einer Zeile für Mean und Concat, In-Shop zuletzt, weil dort Spalten leer bleiben. Die Spanne über die
# Verlustfunktionen ergibt sich aus den Tabellen map_*.tex.
L = ["\\begin{table}[tbp]\n\\centering\\small\\setlength{\\tabcolsep}{4pt}",
     "\\caption[Ausgangsleistung und supervised Referenz]{Test-MAP@R (Mean, Concat) der unveränderten DINOv2-Merkmale und der \\UEBL in \\%, bei der \\UEBL Mittel $\\pm$ Standardabweichung der 25 Läufe aus fünf Verlustfunktionen und fünf Seeds, die vor allem die Unterschiede zwischen den Verlustfunktionen wiedergibt. Alle Labels: alle Bilder der Trainingsklassen gelabelt, auf Cars196 und In-Shop nur Seed~2 (fünf Läufe).}",
     "\\label{tab:e:referenz}",
     "\\begin{tabular}{@{}l S[table-format=2.1] *{5}{S[table-format=2.1(2), retain-zero-uncertainty]}@{}}\n\\toprule",
     " & {DINOv2} & \\multicolumn{4}{c}{\\UEBL je Label-Budget} & {\\UEBL, alle Labels} \\\\"]
for di, (key, name, ks) in enumerate(sorted(DATA, key=lambda d: d[0] == "inshop")):  # In-Shop zuletzt
    r = runs(key)
    m = DINO[key][0]
    kopf = [f"{{$k={k}$}}" for k in ks] + [""] * (4 - len(ks))
    L.append(f"\\midrule\n\\textbf{{{name}}} & & " + " & ".join(kopf) + " & \\\\")
    werte = []
    for k in ks:
        s = r[(r.fam == "sup") & (r.b == k)]
        werte.append([s[s.scope == scope][METRIC] for scope in ("mean", "concat")])  # je 25 Läufe
    # seit 17.10.2026: Ü mit allen Labels (sammle_alle_labels.py), Mittel über Seeds je Verlustfunktion, dann über die Verluste
    a = ALLE[ALLE.key == key]
    for i, (lab, x) in enumerate((("Mean", a.mean_map), ("Concat", a.concat_map))):
        zellen = [f"{w[i].mean():.1f} +- {w[i].std():.1f}" for w in werte] + [""] * (4 - len(ks))
        L.append(f"\\quad {lab} & {f1(m)} & " + " & ".join(zellen) + f" & {x.mean():.1f} +- {x.std():.1f} \\\\")
L.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(OUT + "referenz.tex", "w", encoding="utf-8").write("\n".join(L) + "\n")

# ---------------------------------------------------------------- Gewinne (Anhang)
# Seit 30.10.2026 kompakt und im Anhang: eine Zeile je Datensatz und Budget, Spalten die Verfahren getrennt nach Mean
# und Concat. Im Hauptteil zeigt fig_phase1_gewinne.py dieselben Gewinne ohne Standardabweichung.
# negativ.tex behält die Gliederung mit einem Block je Datensatz und den Verfahren als Zeilen.
SC = "S[table-format=+1.1(2), retain-explicit-plus, retain-zero-uncertainty]"
G = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Gewinn der SSL-Verfahren mit Standardabweichung]{Gepaarter Gewinn der Test-MAP@R gegenüber der \\UEBL in Prozentpunkten, gemittelt über die fünf Seeds und die fünf Verlustfunktionen. Fett: größter gerundeter Gewinn je Zeile, getrennt nach Mean und Concat, sofern er positiv ist.}",
     "\\label{tab:anhang:gewinne}",
     f"\\footnotesize\\setlength{{\\tabcolsep}}{{2pt}}\n\\begin{{tabular}}{{@{{}}l r@{{\\hspace{{6pt}}}} *{{8}}{{{SC}}}@{{}}}}\n\\toprule",
     " & & \\multicolumn{4}{c}{Mean} & \\multicolumn{4}{c}{Concat} \\\\\n\\cmidrule(lr){3-6}\\cmidrule(l){7-10}",
     "{Datensatz} & {$k$}" + "".join(f" & {{{FAMN[f]}}}" for f in FAM) * 2 + " \\\\\n\\midrule"]
SEED_SD = []
NEG = []  # Zeilen der Anhangtabelle negativ.tex
for di, (key, name, ks) in enumerate(DATA):
    r = paired(runs(key))
    pad = "".join(" & {} & {}" for _ in range(4 - len(ks)))
    if di:
        G.append("\\addlinespace")
        NEG.append("\\midrule")
    # negativ.tex: Name in einer eigenen Zeile, darunter Budgets und M/C
    NEG.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\")
    NEG.append("".join(f" & \\multicolumn{{2}}{{c}}{{$k={k}$}}" for k in ks) + pad + " \\\\")
    NEG.append("".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ks))))
    NEG.append("".join(" & {M} & {C}" for _ in ks) + pad + " \\\\")
    zellen = {}  # (Budget, Auswertung) -> {Verfahren: ([Zelle], [Gewinn])} für fett()
    for fam in FAM:
        nc = []
        for k in ks:
            for scope in ("mean", "concat"):
                x = r[(r.fam == fam) & (r.b == k) & (r.scope == scope)]
                g = x.groupby("loss")["diff"].mean().mean()  # erst je Verlust, dann über die Verluste (gleich, wenn alle Läufe vorliegen)
                zellen.setdefault((k, scope), {})[fam] = ([spm(g, x["diff"].std())], [g])  # SD der gepaarten Differenzen
                SEED_SD.append(x.groupby("seed")["diff"].mean().std())  # Seed-SD des über die Verluste gemittelten Gewinns
                assert x["diff"].notna().sum() == 25, (key, fam, k, scope)
                nc.append(str(int((x["diff"] < 0).sum())))
        NEG.append(f"{FAMN[fam]} & " + " & ".join(nc) + pad + " \\\\")
    short = "Semi-iNat" if key == "semiinat" else name  # Spaltenbreite, wie in referenz.tex
    for i, k in enumerate(ks):
        cells = [c[0] for scope in ("mean", "concat") for c in fett(zellen[k, scope]).values()]
        G.append(f"{short if i == 0 else ''} & {k} & " + " & ".join(cells) + " \\\\")
G[1] = G[1].replace(  # Seed-SD erst nach der Schleife bekannt
    "gemittelt über die fünf Seeds und die fünf Verlustfunktionen.",
    "gemittelt über die fünf Seeds und die fünf Verlustfunktionen, $\\pm$ Standardabweichung der 25 gepaarten Differenzen. "
    "Sie gibt vor allem die Unterschiede zwischen den Verlustfunktionen wieder, "
    "die Seed-Standardabweichung des Mittels liegt bei höchstens "
    + f"{max(SEED_SD):.1f}".replace(".", ",") + "~Prozentpunkten.")
G.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(sammlung.output_dir("kapitel/anhang_tabellen") + "gewinne.tex", "w", encoding="utf-8").write("\n".join(G) + "\n")

# ---------------------------------------------------------------- Negative Vergleiche (Anhang)
N = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Negative Vergleiche gegenüber der supervised Referenz]{Zahl der gepaarten Vergleiche (von 25), in denen ein \\acs{SSL}-Verfahren eine niedrigere Test-MAP@R erreicht als die \\UEBL, zu den Gewinnen in Tabelle~\\ref{tab:anhang:gewinne}. M: Mean, C: Concat.}",
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
     "\\caption[Gewinn bei vollständig gelabelten Klassen mit Standardabweichung]{In-Shop mit vollständig gelabelten Klassen: Test-MAP@R der \\UEBL in \\% und gepaarter Gewinn der \\acs{SSL}-Verfahren in Prozentpunkten, Mittel über die fünf Seeds und die fünf Verlustfunktionen, beim Gewinn $\\pm$ Standardabweichung der 25 gepaarten Differenzen. Anteil: vollständig gelabelte Trainingsklassen. Fett: größter Gewinn je Spalte, sofern er positiv ist. M: Mean, C: Concat.}",
     "\\label{tab:anhang:klassen}",
     "\\footnotesize\\setlength{\\tabcolsep}{2pt}\n\\begin{tabular}{@{}l *{8}{S[table-format=+1.1(2), retain-explicit-plus, retain-zero-uncertainty]}@{}}\n\\toprule",
     "{Anteil}" + "".join(f" & \\multicolumn{{2}}{{c}}{{\\qty{{{k}}}{{\\percent}}}}" for k in ks) + " \\\\",
     "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(ks))),
     "".join(" & {M} & {C}" for _ in ks) + " \\\\\n\\midrule"]
cells = ["{" + f1(r[(r.fam == "sup") & (r.b == k) & (r.scope == scope)][METRIC].mean()).replace(".", ",") + "}"
         for k in ks for scope in ("mean", "concat")]  # Textzellen: die S-Spalten sind für die Gewinne mit ± bemessen
K.append("\\multicolumn{9}{@{}l}{\\emph{Test-MAP@R der \\UEBL in \\%}} \\\\")
K.append("\\UEBL & " + " & ".join(cells) + " \\\\\n\\addlinespace")
K.append("\\multicolumn{9}{@{}l}{\\emph{Gewinn}} \\\\")
zeilen = {}
for fam in ["iscen", "lrml", "hoffer", "slade"]:
    d = [p[(p.fam == fam) & (p.b == k) & (p.scope == scope)]["diff"] for k in ks for scope in ("mean", "concat")]
    zeilen[fam] = ([spm(x.mean(), x.std()) for x in d], [x.mean() for x in d])
for fam, cells in fett(zeilen).items():
    K.append(f"{FAMN[fam]} & " + " & ".join(cells) + " \\\\")
K.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(sammlung.output_dir("kapitel/anhang_tabellen") + "klassen.tex", "w", encoding="utf-8").write("\n".join(K) + "\n")
print(open(OUT + "referenz.tex", encoding="utf-8").read())
