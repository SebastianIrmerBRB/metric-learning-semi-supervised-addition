"""Anhangtabelle der SLADE-Sweeps (nach sammle_slade_sweeps.py): kapitel/anhang_tabellen/phase2_slade_sweeps.tex.

Zellen: Änderung der Test-MAP@R (Mean) gegenüber der regulären SLADE-Konfiguration in Prozentpunkten. Fett: Betrag über der
Schwelle 2*sqrt(2)*sigma. Kursiv: der in Phase 1 gewählte Wert (Wiederholung der regulären Konfiguration).
"""
import json

import datensammlung as sammlung
import numpy as np
import pandas as pd

S = sammlung.read_csv("phase2/slade_sweeps.csv")
OUT = sammlung.output_dir("kapitel/anhang_tabellen") + "phase2_slade_sweeps.tex"
DS = ["Cars196", "DeepFashionInShop"]
LOSS = ["arc", "cir", "ms", "nt", "prx"]
LOSS_NAME = {"arc": "Arc", "cir": "Cir", "ms": "\\acs{MS}", "nt": "NT", "prx": "PA"}
PARAM = [("regularizer_target_ratio", "Zielgradientenverhältnis $\\eta$"),
         ("basis_target_ratio", "Basis: Zielgradientenverhältnis"),
         ("cluster_ratio", "Clusterverhältnis"),
         ("update_interval_epochs", "Teacher-Aktualisierung alle $n$ Epochen"),
         ("unlabeled_batch_size", "Ungelabelte Batchgröße"),
         ("sd_margin", "$\\mathcal{L}_{\\mathrm{SD}}$: Margin")]


def de(x, d=1, plus=False):
    s = f"{x:+.{d}f}" if plus else f"{x:.{d}f}"
    return s.replace(".", "{,}").replace("-", "\\textminus ")


def val(v):
    return de(v, 0) if float(v).is_integer() else de(v, 2 if abs(v * 10 - round(v * 10)) > 1e-9 else 1)


def liste(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " und " + xs[-1]


def selbst():
    """Teilsatz der Bildunterschrift: in Phase 1 gewählte Werte ohne eigenen Sweep-Lauf, deren kursive Zelle die reguläre
    Konfiguration selbst ist (die Tabelle setzt dort +0,0)."""
    satz = {"regularizer_target_ratio": "$\\eta$ = {}", "basis_target_ratio": "dem Basis-Zielgradientenverhältnis {}",
            "cluster_ratio": "dem Clusterverhältnis {}", "update_interval_epochs": "der Teacher-Aktualisierung alle {}~Epochen",
            "unlabeled_batch_size": "der ungelabelten Batchgröße {}", "sd_margin": "der Margin {}"}
    teile = []
    for p, _ in PARAM:
        g = S[S.param == p]
        werte = sorted({v for v in g.value.unique() for d in DS for l in LOSS
                        if g[(g.dataset == d) & (g.loss == l) & np.isclose(g.value, v)].empty
                        and np.isclose(g[(g.dataset == d) & (g.loss == l)].tuned_value, v).any()})
        if werte:
            teile.append(satz[p].format(liste([val(v).replace("{,}", ",") for v in werte])))
    return ", bei " + liste(teile) + " die reguläre Konfiguration selbst" if teile else ""


KURZ = {"regularizer_target_ratio": "$\\eta$ = {}", "basis_target_ratio": "Basis-Zielgradientenverhältnis {}",
        "update_interval_epochs": "Teacher-Aktualisierung alle {} Epochen", "cluster_ratio": "Clusterverhältnis {}",
        "unlabeled_batch_size": "ungelabelte Batchgröße {}", "sd_margin": "Margin von $\\mathcal{{L}}_{{\\mathrm{{SD}}}}$ {}"}


def ausserhalb_werte():
    """Sweep-Werte außerhalb der Suchräume der Phase 1 (studies.json, SLADE, Budget der HPO): {(param, wert): Datensätze}."""
    studies = [x for x in sammlung.read_json("studies.json")
               if x["phase"] == 1 and x["method"] == "slade" and x["dataset_key"] in ("cars", "inshop")]
    space = {}
    for x in studies:
        for key, sp in x["hparam_config"]["spaces"].items():
            space.setdefault((x["dataset_key"], key), set()).add(json.dumps(sp, sort_keys=True))
    aus = {}
    for p in KURZ:
        for d, key in (("Cars196", "cars"), ("DeepFashionInShop", "inshop")):
            sp = [json.loads(v) for k, v in space.items() for v in v if k[0] == key and k[1].split(".")[-1] == p]
            assert len(sp) == 1, (key, p, sp)  # ein Suchraum je Datensatz und Parameter
            sp = sp[0]
            inside = (lambda v: any(np.isclose(v, c) for c in sp["choices"])) if "choices" in sp else \
                (lambda v: sp["low"] - 1e-12 <= v <= sp["high"] + 1e-12)
            for v in S[(S.dataset == d) & (S.param == p)].value.unique():
                if not inside(v):
                    aus.setdefault((p, float(v)), set()).add(d)
    return aus


AUS = ausserhalb_werte()


def marke(p, v):
    """Kennzeichen am Zeilenwert: außerhalb der Suchräume der Phase 1 (auf mindestens einem Datensatz)."""
    return "$^\\dagger$" if any(q == p and np.isclose(w, v) for q, w in AUS) else ""


def ausserhalb():
    """Schlüssel der Bildunterschrift zu marke(), mit den Werten, die nur auf einem Datensatz außerhalb liegen."""
    if not AUS:
        return ""
    nur = [KURZ[p].format(val(v).replace("{,}", ",")) + " nur auf " + ("Cars196" if ds == {"Cars196"} else "In-Shop")
           for (p, v), ds in sorted(AUS.items()) if len(ds) == 1]
    return "$^\\dagger$: außerhalb der Suchräume der Phase~1" + (" (" + liste(nur) + ")" if nur else "") + ". "


L = ["\\begin{table}[p]\n\\centering",
     "\\caption[Sweeps von SLADE in Phase~2]{Sweeps einzelner Hyperparameter von SLADE (Phase~2, Seed~2, Cars196 mit $k=16$, "
     "In-Shop mit $k=2$): Änderung der Test-MAP@R (Mean) gegenüber der regulären SLADE-Konfiguration in "
     "Prozentpunkten. Fett: Betrag über der Rauschschwelle $2\\sqrt{2}\\,\\sigma$, "
     "kursiv: in Phase~1 gewählter Wert (Wiederholung der regulären Konfiguration" + selbst() + "). "
     + ausserhalb() + "Arc: ArcFace, Cir: Circle, "
     "NT: NT-Xent, PA: Proxy-Anchor.}",
     "\\label{tab:anhang:phase2:slade}",
     "\\small\\setlength{\\tabcolsep}{3.2pt}",
     "\\begin{tabular}{@{}l" + "r" * 10 + "@{}}\n\\toprule",
     " & \\multicolumn{5}{c}{Cars196} & \\multicolumn{5}{c}{In-Shop} \\\\",
     "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
     "Wert & " + " & ".join(LOSS_NAME[l] for _ in DS for l in LOSS) + " \\\\\n\\midrule"]
reg = S.drop_duplicates(["dataset", "loss"]).set_index(["dataset", "loss"])
L.append("reguläre Konfiguration [\\%] & " + " & ".join(de(reg.loc[(d, l), "regular_mean"]) for d in DS for l in LOSS) + " \\\\")
L.append("Rauschschwelle & " + " & ".join(de(reg.loc[(d, l), "threshold_mean"]) for d in DS for l in LOSS) + " \\\\")
for p, name in PARAM:
    g = S[S.param == p]
    L.append(f"\\addlinespace\n\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{{name}}}}} \\\\")
    for v in sorted(g.value.unique()):
        cells = []
        for d in DS:
            for l in LOSS:
                x = g[(g.dataset == d) & (g.loss == l) & np.isclose(g.value, v)]
                if not len(x):
                    # gewählter Wert ohne eigenen Sweep-Lauf (Teacher-Aktualisierung alle 5 Epochen): die reguläre
                    # Konfiguration selbst, wie alle Wiederholungen der übrigen Zeilen mit Abweichung 0,0
                    tv = g[(g.dataset == d) & (g.loss == l)].tuned_value
                    cells.append("\\textit{" + de(0.0, plus=True) + "}" if len(tv) and np.isclose(tv.iloc[0], v) else "--")
                    continue
                x = x.iloc[0]
                c = de(x.diff_mean, plus=True)
                if x.is_tuned:
                    c = "\\textit{" + c + "}"
                elif abs(x.diff_mean) > x.threshold_mean:
                    c = "\\textbf{" + c + "}"
                cells.append(c)
        L.append(f"\\quad {val(v)}{marke(p, v)} & " + " & ".join(cells) + " \\\\")
L.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
open(OUT, "w", encoding="utf-8").write("\n".join(L) + "\n")
print("geschrieben", OUT, len(L), "Zeilen")
