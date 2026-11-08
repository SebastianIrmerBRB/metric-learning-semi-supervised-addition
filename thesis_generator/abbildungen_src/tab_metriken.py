"""Gepaarte Gewinne der SSL-Verfahren in MAP@R, RP und P@1 (Phase 1, fünf Seeds, Mittel über die fünf Verlustfunktionen).
In der Arbeit: anhang_tabellen/metriken.tex (tab:anhang:metriken), dazu die Zählungen für Abschnitt sec:e:metriken."""
import numpy as np
import pandas as pd
import datensammlung as sammlung

OUT = sammlung.output_dir("kapitel/anhang_tabellen")
DATA = [("cars", "Cars196", "k", [4, 8, 16, 32], "{}"), ("cub", "\\acs{CUB}", "k", [4, 8, 16, 32], "{}"), ("inshop", "In-Shop", "k", [2, 4], "{}"),
        ("inshop10", "In-Shop-Serie", "budget_percent", [5, 10, 25, 50], "{}\\,\\%"), ("semiinat", "Semi-iNat", "k", [2, 4, 16, 100], "{}")]
VERFAHREN = [("iscen", "LP"), ("lrml", "\\acs{LRML}"), ("hoffer", "Hoffer"), ("slade", "SLADE")]
METRIKEN = {"map": "mean_average_precision_at_r", "rp": "r_precision", "p1": "precision_at_1"}


def gewinne(key, spalte):
    R = sammlung.read_csv(f"{key}_ergebnisse/data/phase1_runs.csv")
    R[["fam", "loss"]] = R.method.str.rsplit("_", n=1, expand=True)
    R = R.rename(columns={spalte: "b"})
    idx = ["b", "loss", "seed", "scope"]
    sup = R[R.fam == "sup"].set_index(idx)[list(METRIKEN.values())]
    D = R[R.fam != "sup"].join(sup, on=idx, rsuffix="_sup")
    for k, c in METRIKEN.items():
        D[k] = 100 * (D[c] - D[c + "_sup"])
    assert D[list(METRIKEN)].notna().all().all()
    return D.assign(ds=key)


def zelle(x):
    return f"{0.0 if abs(x) < 0.05 else x:+.1f}"


def zeile(key, k, scope):
    """Eine Zeile der Tabelle. Fett wie in gewinne.tex: größter gerundeter Gewinn je Metrik, sofern er positiv ist, Gleichstände alle."""
    werte = {(f, m): float(zelle(MITTEL.loc[(key, k, f, scope), m])) for f, _ in VERFAHREN for m in METRIKEN}
    beste = {m: max(werte[f, m] for f, _ in VERFAHREN) for m in METRIKEN}
    return "".join(" & " + ("\\bfseries " if werte[f, m] == beste[m] and beste[m] > 0 else "") + zelle(MITTEL.loc[(key, k, f, scope), m])
                   for f, _ in VERFAHREN for m in METRIKEN)


ALLE = pd.concat([gewinne(key, sp) for key, _, sp, _, _ in DATA])
assert len(ALLE) == 3600
MITTEL = ALLE.groupby(["ds", "b", "fam", "scope"])[list(METRIKEN)].mean()
SC = "S[table-format=+1.1, retain-explicit-plus]"
T = ["\\begin{table}[tbp]\n\\centering",
     "\\caption[Gewinn der \\acs{SSL}-Verfahren in \\acs{MAP@R}, \\acs{RP} und \\acs{P@1}]{Gepaarter Gewinn der \\acs{SSL}-Verfahren gegenüber der \\UEBL in \\acs{MAP@R}, \\acs{RP} und \\acs{P@1} in Prozentpunkten, gemittelt über die fünf Seeds und die fünf Verlustfunktionen. M: einzelne Fold-Modelle (Mean), C: Ensemble (Concat). In-Shop-Serie: Anteil vollständig gelabelter Klassen. Fett: größter gerundeter Gewinn je Zeile und Metrik, sofern er positiv ist.}",
     "\\label{tab:anhang:metriken}",
     f"\\footnotesize\\setlength{{\\tabcolsep}}{{2pt}}\n\\begin{{tabular}}{{@{{}}l r c@{{\\hspace{{4pt}}}} *{{12}}{{{SC}}}@{{}}}}\n\\toprule",
     " & & " + "".join(f" & \\multicolumn{{3}}{{c}}{{{n}}}" for _, n in VERFAHREN) + " \\\\",
     "\\cmidrule(lr){4-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\\cmidrule(l){13-15}",
     "{Datensatz} & {Budget} & " + "".join(" & {\\acs{MAP@R}} & {\\acs{RP}} & {\\acs{P@1}}" for _ in VERFAHREN) + " \\\\\n\\midrule"]
for di, (key, name, _, ks, fmt) in enumerate(DATA):
    if di:
        T.append("\\addlinespace")
    for ki, k in enumerate(ks):
        for scope, kurz in (("mean", "M"), ("concat", "C")):
            links = (name if ki == 0 and scope == "mean" else "") + " & " + (fmt.format(k) if scope == "mean" else "") + f" & {kurz}"
            T.append(links + zeile(key, k, scope) + " \\\\")
T.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(OUT + "metriken.tex", "w", encoding="utf-8").write("\n".join(T) + "\n")

# Zählungen für den Text: je Vergleich aus Datensatz, Budget, Verlustfunktion und Verfahren
g = ALLE.groupby(["scope", "ds", "b", "loss", "fam"])
A = g[list(METRIKEN)].mean()
E = g[list(METRIKEN)].agg(lambda x: (x > 0).all() or (x < 0).all())
for scope in ("mean", "concat"):
    a, e = A.loc[scope], E.loc[scope]
    for o in ("rp", "p1"):
        beide = e["map"] & e[o]
        gegen = beide & (np.sign(a["map"]) != np.sign(a[o]))
        print(f"{scope} {o}: {int(gegen.sum())} von {int(beide.sum())} widersprechen sich, davon MAP@R + und {o} -: {int((gegen & (a['map'] > 0)).sum())}")
    plus = e["map"] & (a["map"] > 0)
    print(f"{scope}: MAP@R in allen Seeds positiv {int(plus.sum())}, Gewinn in P@1 kleiner {int((a.p1[plus] < a['map'][plus]).sum())}")
