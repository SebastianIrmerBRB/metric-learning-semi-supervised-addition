"""Zahlen im Fließtext der Ergebnisse: kapitel/cars_tabellen/zahlen.tex (Makro \\CarsStat).

Validierung gegen Test auf Cars196: je Verfahren, Verlust und Budget die über die Seeds gemittelte, nach Seed gepaarte
Differenz der MAP@R zu \\UEBL, in der Validierung (beste Validierungs-MAP@R je Fold, Mittel über die Folds) und im Test
(Mean-MAP@R).
"""
import datensammlung as sammlung


def n(x):
    return f"{x:.2f}".replace(".", "{,}")


runs = sammlung.read_csv("cars_ergebnisse/data/phase1_runs.csv")
folds = sammlung.read_csv("cars_ergebnisse/data/phase1_folds.csv")
valid = folds.groupby(["method", "k", "seed"]).best_valid_map.mean().rename("valid").reset_index()
x = valid.merge(runs[runs.scope == "mean"][["method", "k", "seed", "mean_average_precision_at_r"]],
                on=["method", "k", "seed"], validate="one_to_one").rename(columns={"mean_average_precision_at_r": "test"})
x["fam"], x["loss"] = x.method.str.rsplit("_", n=1).str[0], x.method.str.rsplit("_", n=1).str[1]
base = x[x.fam == "sup"][["loss", "k", "seed", "valid", "test"]]
y = x[x.fam != "sup"].merge(base, on=["loss", "k", "seed"], suffixes=("", "_sup"), validate="many_to_one")
y["dv"], y["dt"] = 100 * (y.valid - y.valid_sup), 100 * (y.test - y.test_sup)
z = y.groupby(["fam", "loss", "k"])[["dv", "dt"]].mean()
stats = {"vt-cells": str(len(z)), "vt-larger": str(int((z.dv > z.dt).sum())),
         "vt-dv": f"${n(z.dv.mean())}$", "vt-dt": f"${n(z.dt.mean())}$"}
lines = ["\\newcommand{\\CarsStat}[1]{\\csname cars-stat-#1\\endcsname}"]
lines += [f"\\expandafter\\def\\csname cars-stat-{k}\\endcsname{{{v}}}" for k, v in stats.items()]
out = sammlung.output_dir("kapitel/cars_tabellen") + "zahlen.tex"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(out, stats)
