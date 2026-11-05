"""Condensed reading view of the geometry, pseudo-label and neighborhood data of the thesis (5.11.2026).

Reads only the views of the data bundle and writes, into --output (default ../datenansicht/diagnostics next to daten/):
  geometry.csv, pseudo_labels.csv, threshold_sweeps.csv, neighborhood.csv, graph_interventions.csv   one row per setting
  README.md     the same results as short tables, readable directly on GitHub
  tables.json   the same tables for the Diagnostics page of the data viewer

Sources per measurement, as in the thesis:
  MAP@R of the Phase-2 runs                                 cv_summary.json (phase2/variants.csv, phase2_k8/variants.csv)
  coverage and accuracy with the selected configuration     geometry runs (geometrie/lp_verfolgung.csv)
  coverage and accuracy at other confidence thresholds      diagnostics of the sweep runs (phase2/lp_schwellen.csv)
  geometry and graph purity                                 geometry runs (geometrie/epochen_*.csv, laeufe_*.csv)
Values at the selected epoch follow geodaten.at_selected and the exclusions of fig_geometrie_weitere.py.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import datensammlung as sammlung
import geodaten

DS = {"cars": "Cars196", "inshop": "In-Shop", "inshop10": "In-Shop, labeled classes", "semiinat": "Semi-iNat-2021"}
METHODS = {"sup": "ÜBl.", "iscen": "LP", "lrml": "LRML", "hoffer": "Hoffer", "slade": "SLADE"}
LOSSES = {"arc": "ArcFace", "cir": "Circle", "ms": "MS", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
NAH = {"cars": "Cars196", "inshop": "DeepFashionInShop"}
GEO = {"rho": "train_rho_roth", "pi_ratio": "train_pi_ratio", "pi_intra": "train_pi_intra", "pi_inter": "train_pi_inter",
       "pi_intra_labeled": "labeled_pi_intra", "pi_intra_unlabeled": "unlabeled_pi_intra", "val_pi_ratio": "validation_pi_ratio",
       "val_pi_intra": "validation_pi_intra", "val_pi_inter": "validation_pi_inter",
       "effective_rank": "train_embedding_effective_rank", "prototype_accuracy": "prototype_unlabeled_accuracy"}
THRESHOLDS = [0.1, 0.3, 0.5, 0.7, 0.9]
INTERVENTIONS = [("frozen", "Pseudo-labels or graph frozen after warm-up"), ("graph_rewired", "Graph rewired, node degrees kept"),
                 ("graph_oracle_pruned", "Edges between classes removed (oracle)"), ("graph_random_pruned", "Same number of random edges removed")]


def budget(ds, k):
    return f"{k} %" if ds == "inshop10" else f"k = {k}"


def spalte(E, name):
    """pi over classes with at least two images (Methodik), on Cars196 every class has two."""
    return name if name.endswith("_roth") or name in E.columns else name + ("_multi" if name + "_multi" in E.columns else "_roth")


def geometrie(ds):
    """Selected epoch per fold and head before training, with the exclusions of fig_geometrie_weitere.py."""
    E, R, _ = geodaten.load(ds)
    R = R.copy()
    letzte = R.geo_last_epoch if "geo_last_epoch" in R else R.last_epoch
    bad = [dict(dataset=DS[ds], budget=budget(ds, r.k), method=METHODS[r.method], loss=LOSSES[r.loss], fold=r.fold,
                reason="no test value" if pd.isna(r.test_map) else "tracking ended before the selected epoch")
           for r, ok in zip(R.itertuples(), R.test_map.notna() & (letzte >= R.selected_epoch)) if not ok]
    S = geodaten.at_selected(E, R[R.test_map.notna() & (letzte >= R.selected_epoch)])
    n = S.groupby(["method", "loss", "k"]).fold.transform("count")
    weg = S[n < 4].groupby(["method", "loss", "k"]).fold.count().reset_index()
    bad += [dict(dataset=DS[ds], budget=budget(ds, r.k), method=METHODS[r.method], loss=LOSSES[r.loss], fold="all",
                 reason=f"only {r.fold} valid folds") for r in weg.itertuples()]
    return E, S[n == 4], bad


def tabelle_geometrie():
    rows, bad, heads = [], [], []
    for ds in DS:
        E, S, b = geometrie(ds)
        bad += b
        cols = {neu: spalte(E, alt) for neu, alt in GEO.items()}
        for (m, l, k), g in S.groupby(["method", "loss", "k"]):
            x = dict(dataset=ds, budget=k, method=m, loss=l, test_map=g.test_map.mean(), selected_epoch=g.selected_epoch.mean())
            x.update({neu: g[alt].astype(float).mean() for neu, alt in cols.items()})
            rows.append(x)
        H = E[(E.epoch == -1) & (E.method == "sup")]
        for k, g in H.groupby("k"):
            x = dict(dataset=ds, budget=k, method="head", loss="", selected_epoch=-1,
                     test_map=100 * g.test_mean_average_precision_at_r.astype(float).mean())
            x.update({neu: g[alt].astype(float).mean() for neu, alt in cols.items()})
            heads.append(x)
    G = pd.DataFrame(rows)
    G["prototype_accuracy"] *= 100
    H = pd.DataFrame(heads)
    H["prototype_accuracy"] *= 100
    return G, H, pd.DataFrame(bad)


def tabelle_reinheit():
    rows = []
    for ds in DS:
        E, S, _ = geometrie(ds)
        S = S[S.method.isin(["sup", "iscen", "lrml"])]
        H = E[E.epoch == -1]
        keys = ["method", "loss", "k", "fold"]
        m = S[keys + ["graph_edge_purity", "graph_n_neighbors"]].merge(H[keys + ["graph_edge_purity"]], on=keys,
                                                                     suffixes=("", "_head"), validate="one_to_one")
        for (me, l, k), g in m.groupby(["method", "loss", "k"]):
            nach, vor = 100 * g.graph_edge_purity.mean(), 100 * g.graph_edge_purity_head.mean()
            rows.append(dict(dataset=ds, budget=k, method=me, loss=l, neighbors=int(g.graph_n_neighbors.iloc[0]),
                             purity_before=vor, purity_after=nach, purity_gain=nach - vor))
    return pd.DataFrame(rows)


def tabelle_pseudolabels():
    V = sammlung.read_csv("geometrie/lp_verfolgung.csv")
    H = sammlung.read_csv("phase2/lp_hersteller.csv")
    spalten = dict(cov0="coverage_first", cov1="coverage_last", acc0="accuracy_first", acc1="accuracy_last", right_stay="right_stays_right",
                   wrong_fixed="wrong_corrected", wrong_keep="wrong_same_class", changed="names_other_class")
    P = (V.groupby(["ds", "k", "loss"])[list(spalten)].mean() * 100).rename(columns=spalten).reset_index()
    H = H.groupby(["stage", "k", "loss"])[["same_make", "same_make_random"]].mean() * 100
    for stage, name in (("erste", "first"), ("letzte", "last")):
        h = H.loc[stage].reset_index().rename(columns=dict(same_make=f"same_make_{name}", same_make_random=f"same_make_random_{name}"))
        P = P.merge(h.assign(ds="cars"), on=["ds", "k", "loss"], how="left")
    return P.rename(columns=dict(ds="dataset", k="budget"))


def varianten():
    a = sammlung.read_csv("phase2/variants.csv")
    b = sammlung.read_csv("phase2_k8/variants.csv")
    V = pd.concat([a, b[b.dataset == "Cars196"]])
    V["variant"] = V.variant.fillna("defaults").replace("", "defaults")
    return V


def tabelle_schwellen():
    D = sammlung.read_csv("phase2/lp_schwellen.csv")
    D = D[D.grund.isna() | (D.grund == "")].groupby(["dataset", "loss", "threshold"])[["coverage", "accuracy"]].mean() * 100
    V = varianten()
    T = sammlung.read_csv("phase2/tuned.csv")
    T = T[(T.method == "lp") & (T.param == "confidence_threshold")].set_index(["dataset", "loss"]).value
    Z = sammlung.read_csv("geometrie/lp_verfolgung.csv")
    Z = Z.groupby(["ds", "k", "loss"])[["cov0", "acc0"]].mean() * 100
    rows = []
    for ds, k in (("inshop", 2), ("cars", 16)):
        lang = NAH[ds]
        W = V[(V.dataset == lang) & (V.k == k) & (V.method == "lp")]
        for l in LOSSES:
            x = W[W.loss == l].set_index("variant")
            reg, sup = x.mean_value["defaults"], x.mean_value["defaults"] - x.mean_diff_supervised["defaults"]
            rows.append(dict(dataset=ds, budget=k, loss=l, threshold=round(float(T[(lang, l)]), 2), setting="selected",
                             coverage_first=Z.cov0[(ds, k, l)], accuracy_first=Z.acc0[(ds, k, l)], coverage_source="geometry run",
                             test_map=reg, diff_regular=0.0, diff_supervised=reg - sup))
            for t in THRESHOLDS:
                wert = x.mean_value[f"confidence_threshold_{str(t).replace('.', 'p')}"]
                key = ("In-Shop", l, t)
                rows.append(dict(dataset=ds, budget=k, loss=l, threshold=t, setting="sweep",
                                 coverage_first=D.coverage.get(key, np.nan) if ds == "inshop" else np.nan,
                                 accuracy_first=D.accuracy.get(key, np.nan) if ds == "inshop" else np.nan,
                                 coverage_source="sweep diagnostics" if ds == "inshop" else "not recorded",
                                 test_map=wert, diff_regular=wert - reg, diff_supervised=wert - sup))
    return pd.DataFrame(rows)


def schwelle(ds, k):
    """Noise threshold 2*sqrt(2)*sigma from the five Phase-1 seeds, as in the thesis."""
    R = sammlung.read_csv({"cars": "cars_ergebnisse", "inshop": "inshop_ergebnisse"}[ds] + "/data/phase1_runs.csv")
    R = R[(R.scope == "mean") & (R.k == k) & R.method.str.startswith(("iscen_", "lrml_"))]
    out = {}
    for method, g in R.groupby("method"):
        pre, loss = method.rsplit("_", 1)
        out[("lp" if pre == "iscen" else pre, loss)] = 2 * np.sqrt(2) * 100 * g.mean_average_precision_at_r.std(ddof=1)
    return out


def tabelle_eingriffe():
    V = varianten()
    rows = []
    for ds, k in (("cars", 16), ("inshop", 2), ("cars", 8)):
        thr = schwelle(ds, k)
        W = V[(V.dataset == NAH[ds]) & (V.k == k)]
        for m in ("lp", "lrml"):
            for l in LOSSES:
                x = W[(W.method == m) & (W.loss == l)].set_index("variant").mean_value
                reg = x["defaults"]
                y = dict(dataset=ds, budget=k, method="iscen" if m == "lp" else m, loss=l, regular=reg, noise_threshold=thr[(m, l)])
                for key, _ in INTERVENTIONS:
                    v = x["pl_once" if m == "lp" else "graph_once"] if key == "frozen" else x[key]
                    y["diff_" + key] = v - reg
                rows.append(y)
    return pd.DataFrame(rows)


def pruefe(G, N, P, S, I):
    """The tables must reproduce values printed in the thesis, otherwise nothing is written."""
    def nah(a, b, tol=0.051):
        assert abs(a - b) <= tol, (a, b)
    # Pseudo-labels (5.6.2)
    letzte = P[P.dataset == "cars"]
    nah(letzte.same_make_last.min(), 70.0, 0.5), nah(letzte.same_make_last.max(), 82.0, 0.5)
    nah(P[(P.dataset == "semiinat") & (P.budget == 100)].accuracy_last.min(), 80.0, 0.5)
    nah(P[(P.dataset == "semiinat") & (P.budget == 100)].accuracy_last.max(), 96.0, 0.5)
    # Sweep of the confidence threshold (5.6.4)
    a = S[(S.dataset == "inshop") & (S.loss == "arc")].set_index("threshold")
    nah(a.diff_regular[0.9], -29.4), nah(a.coverage_first[0.7], 0.2), nah(a.coverage_first[0.58], 5.3)
    # Graph purity (5.6.3): gain per budget on Cars196 for LP and LRML
    c = N[(N.dataset == "cars") & N.method.isin(["iscen", "lrml"])]
    for k, lo, hi in ((16, 29.5, 44.4), (4, 22.6, 37.0), (8, 28.2, 42.2), (32, 30.0, 47.7)):
        g = c[c.budget == k].purity_gain
        nah(g.min(), lo), nah(g.max(), hi)
    # Table of the LRML graphs in the appendix
    l = N[(N.dataset == "cars") & (N.method == "lrml") & (N.loss == "arc")].set_index("budget")
    nah(l.purity_after[4], 56.3), nah(l.purity_after[32], 78.5), nah(l.purity_before[4], 30.8)
    assert l.neighbors.unique().tolist() == [49]
    # Interventions (5.6.3)
    c16 = I[(I.dataset.isin(["cars", "inshop"])) & (I.budget.isin([16, 2]))]
    assert (c16.diff_graph_rewired < 0).all() and len(c16) == 20
    nah(c16[c16.method == "iscen"].diff_graph_rewired.max(), -6.3)
    assert (c16.diff_graph_oracle_pruned > 0).sum() == 18
    nah(c16[(c16.dataset == "inshop") & (c16.method == "iscen") & (c16.loss == "arc")].diff_graph_oracle_pruned.iloc[0], -8.2)
    nah(c16[(c16.dataset == "inshop") & (c16.method == "iscen") & (c16.loss == "arc")].diff_frozen.iloc[0], -11.5)
    assert (c16.diff_graph_random_pruned.abs() <= c16.noise_threshold).sum() == 18
    # Geometry: means of the appendix and the correlation table
    m = sammlung.read_csv("geometrie/mittel_weitere.csv")
    for r in m.itertuples():
        g = G[(G.dataset == r.ds) & (G.budget == r.k) & (G.method == r.method) & G.loss.isin(r.losses.split(","))]
        for spalte_, wert in (("pi_ratio", r.pr), ("pi_intra", r.pin), ("pi_inter", r.pie), ("rho", r.rho), ("pi_intra_unlabeled", r.unl),
                              ("pi_intra_labeled", r.lab), ("val_pi_ratio", r.vpr), ("prototype_accuracy", r.proto), ("test_map", r.m)):
            nah(g[spalte_].mean(), wert, 1e-6)
    return len(m)


def zahl(x, d=1):
    return "–" if x is None or pd.isna(x) else f"{x:.{d}f}"


def seite(G, H, N, P, S, I, bad):
    sections = []
    # Geometry
    tabellen = []
    for ds, name in DS.items():
        rows = []
        for k in sorted(G[G.dataset == ds].budget.unique()):
            h = H[(H.dataset == ds) & (H.budget == k)].iloc[0]
            rows.append([budget(ds, k), "untrained head", zahl(h.test_map), zahl(h.pi_ratio, 2), zahl(h.pi_intra, 2), zahl(h.pi_inter, 2), zahl(h.rho, 2),
                         zahl(h.pi_intra_labeled, 2), zahl(h.pi_intra_unlabeled, 2), zahl(h.prototype_accuracy), "–"])
            g = G[(G.dataset == ds) & (G.budget == k)]
            gemeinsam = set.intersection(*[set(x.loss) for _, x in g.groupby("method")])
            for m, mn in METHODS.items():
                x = g[(g.method == m) & g.loss.isin(gemeinsam)]
                rows.append([budget(ds, k), mn, zahl(x.test_map.mean()), zahl(x.pi_ratio.mean(), 2), zahl(x.pi_intra.mean(), 2), zahl(x.pi_inter.mean(), 2),
                             zahl(x.rho.mean(), 2), zahl(x.pi_intra_labeled.mean(), 2), zahl(x.pi_intra_unlabeled.mean(), 2),
                             zahl(x.prototype_accuracy.mean()), zahl(x.selected_epoch.mean(), 0)])
        tabellen.append(dict(title=name, columns=["Budget", "Method", "Test MAP@R %", "π ratio", "π intra", "π inter", "ρ", "π intra labeled",
                                                  "π intra unlabeled", "Prototype acc. %", "Epoch"], rows=rows))
    sections.append(dict(
        id="geometry", title="Geometry of the embedding space", file="geometry.csv",
        intro=("Section 5.7 of the thesis. Means over the five losses of four fold models at the epoch selected by validation, whole training pool. "
               "π intra is the spread within classes, π inter the distance between class centers, π ratio = π intra / π inter (smaller means more compact classes), "
               "ρ the spectral decay. Prototype accuracy is the share of unlabeled images whose nearest class prototype is correct. "
               "geometry.csv has every loss and the values on the validation classes."),
        tables=tabellen))
    # Pseudo-labels
    tabellen = []
    for ds, name in (("cars", "Cars196"), ("inshop", "In-Shop"), ("semiinat", "Semi-iNat-2021")):
        x = P[P.dataset == ds]
        cols = ["k", "Loss", "Coverage first %", "Coverage last %", "Accuracy first %", "Accuracy last %", "Right stays right %",
                "Wrong corrected %", "Wrong, same class %"] + (["Same make first %", "Same make last %"] if ds == "cars" else [])
        rows = []
        for k in sorted(x.budget.unique()):
            for l, ln in LOSSES.items():
                r = x[(x.budget == k) & (x.loss == l)].iloc[0]
                z = [str(k), ln, zahl(r.coverage_first), zahl(r.coverage_last), zahl(r.accuracy_first), zahl(r.accuracy_last), zahl(r.right_stays_right),
                     zahl(r.wrong_corrected), zahl(r.wrong_same_class)]
                rows.append(z + ([zahl(r.same_make_first), zahl(r.same_make_last)] if ds == "cars" else []))
        tabellen.append(dict(title=name, columns=cols, rows=rows))
    rows = []
    w = S[S.dataset == "inshop"]
    for l, ln in LOSSES.items():
        for r in w[w.loss == l].sort_values("threshold").itertuples():
            t = f"**{r.threshold:.2f} (selected)**" if r.setting == "selected" else f"{r.threshold:.1f}"
            rows.append([ln, t, zahl(r.coverage_first), zahl(r.accuracy_first), zahl(r.test_map), "–" if r.setting == "selected" else f"{r.diff_regular:+.1f}",
                         f"{r.diff_supervised:+.1f}"])
    tabellen.append(dict(title="Confidence threshold on In-Shop, k = 2", columns=["Loss", "Threshold", "Coverage first %", "Accuracy first %", "Test MAP@R %",
                                                                                  "Δ regular", "Δ ÜBl."], rows=rows,
                         note="Selected threshold from the geometry run, other thresholds from the diagnostics of the separate sweep runs, MAP@R from cv_summary.json."))
    z = P[P.dataset == "cars"]
    sections.append(dict(
        id="pseudo-labels", title="Quality of the LP pseudo-labels", file="pseudo_labels.csv",
        intro=("Section 5.6.2 of the thesis. Geometry runs with the selected configuration, seed 2, means over four folds, whole unlabeled pool. "
               "Coverage is the share of unlabeled images with an accepted pseudo-label, accuracy the share of those that are right according to the hidden labels, "
               "first and last meaning the first and the last update of a fold. Right stays right and wrong corrected compare only these two updates for images that "
               "had a pseudo-label at both. Wrong, same class is the share of initially wrong labels that still name the same class. "
               f"Same make is the share of wrong labels naming a car of the same manufacturer, about {z.same_make_random_last.mean():.0f} % under random confusion."),
        tables=tabellen))
    # Neighborhood
    tabellen = []
    for ds, name in DS.items():
        x = N[N.dataset == ds]
        rows = []
        for m in ("iscen", "lrml", "sup"):
            y = x[x.method == m]
            nn = y.groupby("loss").neighbors.agg(lambda s: "/".join(map(str, sorted(s.unique()))))
            rows.append(["neighbors", METHODS[m]] + [nn.get(l, "–") for l in LOSSES])
            for k in sorted(y.budget.unique()):
                cell = lambda r: f"{r.purity_before:.0f} → {r.purity_after:.0f}"
                rows.append([budget(ds, k), METHODS[m]] + [
                    (cell(y[(y.budget == k) & (y.loss == l)].iloc[0]) if len(y[(y.budget == k) & (y.loss == l)]) else "–") for l in LOSSES])
        tabellen.append(dict(title=f"Graph purity, {name}", columns=["Budget", "Method"] + list(LOSSES.values()), rows=rows,
                             note="Cells: purity in % before training → at the selected epoch. The neighbors rows give the neighbor count of each graph."))
    for ds, k, name in (("cars", 16, "Cars196, k = 16"), ("inshop", 2, "In-Shop, k = 2"), ("cars", 8, "Cars196, k = 8")):
        x = I[(I.dataset == ds) & (I.budget == k)]
        rows = []
        for m in ("iscen", "lrml"):
            y = x[x.method == m].set_index("loss")
            rows.append([METHODS[m], "Regular, MAP@R %"] + [zahl(y.regular[l]) for l in LOSSES])
            rows.append([METHODS[m], "Noise threshold"] + [zahl(y.noise_threshold[l]) for l in LOSSES])
            for key, label in INTERVENTIONS:
                z_ = []
                for l in LOSSES:
                    v = y["diff_" + key][l]
                    s_ = f"{v:+.1f}"
                    z_.append(f"**{s_}**" if abs(v) > y.noise_threshold[l] else s_)
                rows.append([METHODS[m], label] + z_)
        tabellen.append(dict(title=f"Graph interventions, {name}", columns=["Method", "Row"] + list(LOSSES.values()), rows=rows,
                             note="Change of test MAP@R in percentage points against the regular configuration, one seed. Bold: larger than the noise threshold 2√2σ from the five Phase-1 seeds."))
    sections.append(dict(
        id="neighborhood", title="Neighborhood structure", file="neighborhood.csv, graph_interventions.csv",
        intro=("Section 5.6.3 of the thesis. Graph purity is the share of kNN edges between images of the same class in the training pool, measured with the neighbor count and "
               "weighting of the mechanism, before training and at the selected epoch (geometry runs, seed 2, means over four folds). ÜBl. uses one graph with ten neighbors for comparison. "
               "The interventions change the graph of LP and LRML in the Phase-2 runs."),
        tables=tabellen))
    wichtig = [
        "Geometry runs retrain the Phase-1 configurations with seed 2 and measure after every epoch. All values use this one seed and four folds. Five-seed results are in ergebnisuebersicht.csv.",
        "CUB was not tracked. In-Shop, labeled classes is the series with fully labeled classes, its budget is the share of labeled classes.",
        "LRML in the geometry runs on Cars196, In-Shop and the In-Shop series uses the configuration of a search with univariate TPE, so its neighbor counts and MAP@R differ from Phase 1 and from the interventions (up to 5.1 percentage points on Cars196, 1.9 on In-Shop, 3.4 on the In-Shop series).",
    ]
    if len(bad):
        fehlend = ", ".join(sorted({f"{r.method} with {r.loss} at {r.budget}" for r in bad.itertuples() if r.fold == "all"}))
        wichtig.append(f"Left out of the means because a fold has no value at the selected epoch (In-Shop, labeled classes): {fehlend}.")
    return dict(title="Geometry, pseudo-labels and neighborhoods", notes=wichtig, sections=sections)


def markdown(doc):
    aus = [f"# {doc['title']}", "",
           "Condensed reading view of the diagnostic measurements behind Chapters 5 and 6 of the thesis. The same tables are online on the "
           "[Diagnostics page of the data viewer](https://sebastianirmerbrb.github.io/metric-learning-semi-supervised-addition/#/diagnostics). "
           "Generated by `python thesis_generator/export_diagnostics.py` from the compact data in [`../../daten`](../../daten/README.md).", "",
           "Keep in mind:", ""]
    aus += [f"- {n}" for n in doc["notes"]]
    aus += ["", "ÜBl. is the supervised baseline. LP is label propagation."]
    for s in doc["sections"]:
        aus += ["", f"## {s['title']}", "", s["intro"], "", "Files: " + ", ".join(f"[`{f}`]({f})" for f in s["file"].split(", "))]
        for t in s["tables"]:
            aus += ["", f"### {t['title']}", ""]
            aus.append("| " + " | ".join(t["columns"]) + " |")
            aus.append("|" + "|".join("---" for _ in t["columns"]) + "|")
            aus += ["| " + " | ".join(map(str, r)) + " |" for r in t["rows"]]
            if t.get("note"):
                aus += ["", t["note"]]
    return "\n".join(aus) + "\n"


def csv_schreiben(df, pfad):
    df = df.copy()
    for spalte_, namen in (("dataset", DS), ("method", {**METHODS, "head": "untrained head"}), ("loss", LOSSES)):
        if spalte_ in df:
            df[spalte_] = df[spalte_].map(lambda v: namen.get(v, v))
    df.to_csv(pfad, index=False, float_format="%.4f", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, default=sammlung.DATA.parent / "datenansicht" / "diagnostics")
    a = p.parse_args()
    G, H, bad = tabelle_geometrie()
    N, P, S, I = tabelle_reinheit(), tabelle_pseudolabels(), tabelle_schwellen(), tabelle_eingriffe()
    n = pruefe(G, N, P, S, I)
    a.output.mkdir(parents=True, exist_ok=True)
    csv_schreiben(pd.concat([G, H]).sort_values(["dataset", "budget", "method", "loss"]), a.output / "geometry.csv")
    csv_schreiben(N, a.output / "neighborhood.csv")
    csv_schreiben(P, a.output / "pseudo_labels.csv")
    csv_schreiben(S, a.output / "threshold_sweeps.csv")
    csv_schreiben(I, a.output / "graph_interventions.csv")
    doc = seite(G, H, N, P, S, I, bad)
    (a.output / "tables.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (a.output / "README.md").write_text(markdown(doc), encoding="utf-8")
    print(f"{a.output}: geometry {len(G)}+{len(H)}, neighborhood {len(N)}, pseudo-labels {len(P)}, sweeps {len(S)}, interventions {len(I)}; "
          f"printed values reproduced, geometry means compared with {n} rows")


if __name__ == "__main__":
    main()
