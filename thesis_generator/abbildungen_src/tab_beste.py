"""Tabelle der besten Kombination aus Verfahren und Verlustfunktion je Datensatz und Budget (Abschnitt sec:e:beste).

Je Datensatz, Budget und Modelltyp: die höchste Test-MAP@R unter den 20 Kombinationen aus den vier SSL-Verfahren und
den fünf Verlustfunktionen und die höchste unter den fünf Varianten der \\UEBL (Mittel über die fünf Seeds). Gestaltung
wie referenz.tex (tab_hauptteil.py): booktabs, siunitx, fette Datensatzköpfe, keine fetten Zahlen. Gibt die Kennzahlen
für den Text aus (beste Verlustfunktion, Spanne über die Verlustfunktionen, Gewinne beim kleinsten und größten Budget).

    python3 tab_beste.py
"""
import datensammlung as sammlung

OUT = sammlung.output_dir("kapitel/ergebnisse_tabellen")
METRIC = "mean_average_precision_at_r"
FAM = ["iscen", "lrml", "hoffer", "slade"]
METHOD = {"iscen": "\\acs{LP}", "lrml": "\\acs{LRML}", "hoffer": "Hoffer", "slade": "\\acs{SLADE}"}
LOSS = {"arc": "ArcFace", "cir": "Circle", "ms": "\\acs{MS}", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
# Reihenfolge wie Abschnitt 5.3. Die In-Shop-Serie mit gelabelten Klassen steht in Tabelle tab:anhang:inshop10.
DATA = [("cars", "Cars196", [4, 8, 16, 32]), ("cub", "\\acs{CUB}", [4, 8, 16, 32]), ("inshop", "In-Shop", [2, 4]),
        ("semiinat", "Semi-iNat-2021", [2, 4, 16, 100])]
HPO = {"cars": 16, "cub": 16, "inshop": 2, "semiinat": 16}
SCOPES = ["mean", "concat"]


def runs(key):
    r = sammlung.read_csv(f"{key}_ergebnisse/data/phase1_runs.csv")
    r = r.copy()
    r["budget"] = r["k"]
    r["mech"] = r.method.str.split("_").str[0]
    r["loss"] = r.method.str.split("_").str[1]
    r["m"] = 100 * r[METRIC]
    return r


def cells(r, budget, scope):
    """Mittel über die fünf Seeds je (Verfahren, Verlust)."""
    g = r[(r.budget == budget) & (r.scope == scope)]
    n = g.groupby(["mech", "loss"]).seed.nunique()
    assert len(n) == 25 and (n == 5).all(), (budget, scope, n.to_dict())
    return g.groupby(["mech", "loss"]).m.mean()


def budget_label(key, b):
    return f"$k={b}$"


lines = ["\\begin{table}[tbp]",
         "\\centering\\small\\setlength{\\tabcolsep}{4pt}",
         "\\caption[Beste Kombination aus Verfahren und Verlustfunktion]{Höchste Test-MAP@R in \\% je Datensatz, Budget "
         "und Modelltyp, Mittel über die fünf Seeds. \\acs{SSL}: beste der 20 Kombinationen aus den vier "
         "\\acs{SSL}-Mechanismen und den fünf Verlustfunktionen (Spalten Verfahren und Verlust), \\UEBL: mit der besten "
         "der fünf Verlustfunktionen.}",
         "\\label{tab:e:beste}",
         "\\begin{tabular}{@{}l *{2}{l l S[table-format=2.1] S[table-format=2.1]}@{}}",
         "\\toprule",
         " & \\multicolumn{4}{c}{Einzelmodell (Mean)} & \\multicolumn{4}{c}{Ensemble (Concat)} \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}",
         "Budget & Verfahren & Verlust & {\\acs{SSL}} & {\\UEBL} & Verfahren & Verlust & {\\acs{SSL}} & {\\UEBL} \\\\"]
facts = []
stats = {"cells": 0, "ms_best_loss": 0, "ms_best_sup": 0, "lp_ms_top_overall": 0, "sup_top": []}
exceptions = []
for i, (key, name, budgets) in enumerate(DATA):
    r = runs(key)
    lines.append("\\midrule")
    lines.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textbf{{{name}}}}} \\\\")
    for b in budgets:
        row = [f"\\quad {budget_label(key, b)}"]
        for scope in SCOPES:
            c = cells(r, b, scope)
            ssl = c.drop("sup", level="mech")
            (mech, loss), best = ssl.idxmax(), ssl.max()
            sup = c.xs("sup", level="mech")
            row += [METHOD[mech], LOSS[loss], f"{best:.1f}", f"{sup.max():.1f}"]
            # Kennzahlen für den Text
            lo = c.groupby("loss").mean()
            stats["cells"] += 1
            stats["ms_best_loss"] += lo.idxmax() == "ms"
            stats["ms_best_sup"] += sup.idxmax() == "ms"
            stats["lp_ms_top_overall"] += c.idxmax() == ("iscen", "ms")
            if c.idxmax()[0] == "sup":
                stats["sup_top"].append((key, b, scope, c.idxmax()[1]))
            if lo.idxmax() != "ms":
                exceptions.append((key, b, scope, lo.idxmax(), round(lo.max() - lo["ms"], 3)))
            facts.append(f"{key:9s} {b:>4} {scope:6s} SSL {mech}_{loss} {best:6.3f} | Ü {sup.idxmax()} {sup.max():6.3f} "
                         f"| Δ {best - sup.max():+5.2f} | Verluste: bester {lo.idxmax()}, schwächster {lo.idxmin()}, "
                         f"Spanne {lo.max() - lo.min():.2f}")
        lines.append(" & ".join(row) + " \\\\")
lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(OUT + "beste.tex", "w", encoding="utf-8").write("\n".join(lines) + "\n")

print("\n".join(facts))
print()
print(f"Zellen {stats['cells']}, MS beste Verlustfunktion im Mittel über die Verfahren {stats['ms_best_loss']}, "
      f"für die Ü allein {stats['ms_best_sup']}, LP mit MS beste der 25 Kombinationen {stats['lp_ms_top_overall']}")
print("Ü selbst am besten:", stats["sup_top"])
print("Ausnahmen (beste Verlustfunktion nicht MS, Vorsprung vor MS):", exceptions)
for key, _, _ in DATA:
    r = runs(key)
    spans = []
    for scope in SCOPES:
        lo = cells(r, HPO[key], scope).groupby("loss").mean()
        spans.append(round(lo.max() - lo.min(), 2))
    print(f"Spanne der Verlustfunktionen beim Budget der Suche {key} ({HPO[key]}): {spans}")
