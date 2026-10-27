"""Datensätze, Aufteilung und Label-Budgets: kapitel/methodik_tabellen/datensaetze.tex.

Klassen- und Bildzahlen aus datensaetze/statistik.json (Datensatzcode des Trainingsprojekts), Budgets aus den
Phase-1-Ergebnissen, Budget der Suche aus studies.json.
"""
import datensammlung as sammlung

STATS = sammlung.read_json("datensaetze/statistik.json")
STUDIES = [s for s in sammlung.read_json("studies.json") if s["phase"] == 1]
# dataset_key, Zeile, Statistik
ROWS = [("cars", "Cars196", "cars"), ("cub", "\\ac{CUB}", "cub"), ("inshop", "In-Shop", "inshop"),
        ("inshop10", "In-Shop (Klassen)", "inshop"), ("semiinat", "Semi-iNat-2021", "semiinat")]


def unique(values, what):
    values = set(values)
    if len(values) != 1:
        raise ValueError(f"{what} not unique: {sorted(values)}")
    return values.pop()


lines = [
    "\\begin{table}[tbp]", "  \\centering",
    "  \\caption[Datensätze und Label-Budgets]{Datensätze, Aufteilung und untersuchte Label-Budgets. $k$: gelabelte Bilder je Trainingsklasse, die übrigen Bilder der Trainingsklassen bilden den ungelabelten Pool. In-Shop-Test: Queries / Gallery. In-Shop (Klassen): Anteil vollständig gelabelter Trainingsklassen. \\ac{HPO}: Budget, bei dem die Hyperparameter gesucht werden. Klassen- und Bildzahlen für Cars196 und \\ac{CUB}~\\cite{song_deep_2016} und für In-Shop~\\cite{wang_multi-similarity_2019}, Semi-iNat-2021 in eigener Aufteilung (Abschnitt~\\ref{sec:daten}).}",
    "  \\label{tab:datensaetze}", "  \\setlength{\\tabcolsep}{3pt}",
    "  \\begin{tabular}{@{}lS[table-format=4.0]S[table-format=5.0]S[table-format=4.0]c>{\\raggedright\\arraybackslash}p{2.35cm}>{\\raggedright\\arraybackslash}p{1.25cm}@{}}",
    "    \\toprule", "    & \\multicolumn{2}{c}{Training} & \\multicolumn{2}{c}{Test} & & \\\\",
    "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
    "    Datensatz & {Klassen} & {Bilder} & {Klassen} & Bilder & Budgets $k$ & \\acs{HPO} \\\\", "    \\midrule"]
for key, label, stat in ROWS:
    s = STATS[stat]
    test = (f"\\num{{{s['test_queries']}}}\\,/\\,\\num{{{s['test_gallery']}}}" if "test_queries" in s
            else f"\\num{{{s['test_images']}}}")
    runs = sammlung.read_csv(f"{key}_ergebnisse/data/phase1_runs.csv")
    args = [x["base_args"] for x in STUDIES if x["dataset_key"] == key]
    if key == "inshop10":
        budgets = ", ".join(str(b) for b in sorted(runs.budget_percent.unique())) + "\\,\\%"
        hpo = f"\\SI{{{round(100 * unique((a['label_budget_grid'][0] for a in args), 'HPO budget'))}}}{{\\percent}}"
    else:
        budgets = ", ".join(str(k) for k in sorted(runs.k.unique()))
        hpo = str(unique((a["k_shot_grid"][0] for a in args), "HPO budget"))
    lines.append(f"    {label} & {s['train_classes']} & {s['train_images']} & {s['test_classes']} & {test} & {budgets} & {hpo} \\\\")
lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
out = sammlung.output_dir("kapitel/methodik_tabellen") + "datensaetze.tex"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(out)
