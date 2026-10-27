"""Anhang zu den Hyperparametersuchen: kapitel/anhang_tabellen/{suchraum_phase0,hpo_protokoll,suchraum_phase1,
feste_einstellungen}.tex.

Suchräume, Budgets, Folds, Trials, Sampler, Seeds und Epochengrenzen aus studies.json (study_config.json der ausgewerteten
Studien), fest gesetzte SSL-Einstellungen und Obergrenzen des Gradientenverhältnisses aus den aufgelösten Einstellungen der
Phase-1-Wiederholungen (Sicht phase1_settings). Jeder gesuchte Parameter muss in einer Zeile vorkommen, sonst bricht das
Skript ab. Ausgenommen sind nur Suchräume, die das Verfahren nicht liest (INERT).
"""
import json
import math

import pandas as pd

import datensammlung as sammlung

OUT = sammlung.output_dir("kapitel/anhang_tabellen")
STUDIES = sammlung.read_json("studies.json")
DATASETS = ["cars", "cub", "inshop", "inshop10", "semiinat"]
METHODS = ["sup", "iscen", "lrml", "hoffer", "slade"]
LOSSES = ["arc", "cir", "ms", "nt", "prx"]
# LRML liest nur regularizer_params.contrastive_margin; der Suchraum auf Ebene method_params bleibt ohne Wirkung.
INERT = {("lrml", "ssl_config.method_params.contrastive_margin")}
MP, RP = "ssl_config.method_params.", "ssl_config.method_params.regularizer_params."
WARMUP = "ssl_config.warmup_epochs"


def flat(space, prefix=""):
    if isinstance(space, dict) and "type" in space and ("low" in space or "choices" in space):
        return {prefix: space}
    out = {}
    for k, v in (space or {}).items():
        out.update(flat(v, f"{prefix}.{k}" if prefix else k))
    return out


for s in STUDIES:
    s["flat"] = {k: v for k, v in flat(s["hparam_config"]["spaces"]).items() if (s["method"], k) not in INERT}


def canon(space):
    def norm(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return v
        return float(v)
    return json.dumps({k: [norm(c) for c in v] if k == "choices" else norm(v) for k, v in space.items() if k != "type"},
                      sort_keys=True)


def unique(values, what):
    values = list(values)
    keys = {canon(v) if isinstance(v, dict) else json.dumps(v) for v in values}
    if len(keys) != 1:
        raise ValueError(f"{what} not unique: {sorted(keys)}")
    return values[0]


def num(x):
    x = float(x)
    if 0 < abs(x) < 1e-3:
        m, e = f"{x:.0e}".split("e")
        return (f"{m}{{\\times}}" if m != "1" else "") + f"10^{{{int(e)}}}"
    return f"{x:g}".replace(".", "{,}")


def interval(sp):
    lo, hi = num(sp["low"]), num(sp["high"])
    brk = "\\allowbreak" if "\\times" in lo + hi else ""  # außerhalb des f-Strings, sonst erst ab Python 3.12 gültig
    return f"$[{lo},{brk}\\,{hi}]$"


def stepped(sp):
    step = sp.get("step") or 1
    return [sp["low"] + i * step for i in range(int(round((sp["high"] - sp["low"]) / step)) + 1)]


def plain(v):
    return f"{float(v):g}"


# ---------------------------------------------------------------- Phase 0
P0_SETS = [("cars", "Cars"), ("inshop", "In-Shop"), ("cub", "CUB"), ("semiinat", "Semi-iNat")]
P0_ALIAS = {"inshop10": "inshop"}  # In-Shop mit gelabelten Klassen zählt in den Vorexperimenten zu In-Shop.
P0_BLOCKS = [
    ("sup", "Supervised Basiskonfiguration", [
        ("classifier_lr", "Klassifikator-Lernrate"), ("lr", "Lernrate"), ("batch_sampler", "Batchstruktur $B{:}\\mathrm{SPC}$"),
        ("loss.ArcFaceLoss.margin", "ArcFace $m_{\\mathrm{deg}}$ [Grad]"), ("loss.ArcFaceLoss.scale", "ArcFace $s$"),
        ("loss.CircleLoss.gamma", "Circle $\\gamma$"), ("loss.CircleLoss.m", "Circle $m$"),
        ("loss.MultiSimilarityLoss.alpha", "Multi-Similarity $\\alpha$"), ("loss.MultiSimilarityLoss.base", "Multi-Similarity $\\lambda$"),
        ("loss.MultiSimilarityLoss.beta", "Multi-Similarity $\\beta$"), ("miner.MultiSimilarityMiner.epsilon", "MS-Mining $\\epsilon$"),
        ("loss.NTXentLoss.temperature", "NT-Xent $\\tau$"), ("loss.ProxyAnchorLoss.alpha", "Proxy-Anchor $\\alpha$"),
        ("loss.ProxyAnchorLoss.margin", "Proxy-Anchor $\\delta$")]),
    ("lrml", "LRML", [
        (RP + "graph_batch_size", "Kanten-Batchgröße"), (RP + "n_neighbors", "Nachbarn $k_{\\mathrm{NN}}$ (Regularisierer)"),
        (MP + "regularizer_target_ratio", "Zielgradienten\\-verhältnis $\\eta$"), (WARMUP, "Warm-up [Epochen]")]),
    ("hoffer", "Hoffer", [
        (MP + "regularizer_target_ratio", "Zielgradienten\\-verhältnis $\\eta$"), (WARMUP, "Warm-up [Epochen]"),
        (RP + "distance_scale", "Distanzskala $s_{\\mathrm{H}}$"), (RP + "reference_sets", "Referenzmengen $K_{\\mathrm{ref}}$"),
        (RP + "unlabeled_batch_size", "Ungelabelte Batchgröße")]),
    ("iscen", "Label Propagation (\\texttt{iscen})", [
        (WARMUP, "Warm-up [Epochen]"), ("ssl_config.confidence_threshold", "Konfidenzschwelle"),
        ("labeled_batch_fraction", "Gelabelter Batchanteil"), (MP + "alpha", "Diffusionsparameter $\\alpha$"),
        (MP + "gamma", "Affinitätsexponent $\\gamma$"), (MP + "n_neighbors", "Nachbarn $k_{\\mathrm{NN}}$ (Propagation)")]),
    ("slade", "SLADE", [
        ("loss.NTXentLoss.temperature", "NT-Xent $\\tau$"), (WARMUP, "Warm-up [Epochen]"),
        (RP + "unlabeled_batch_size", "Ungelabelte Batchgröße"), (RP + "beta", "SLADE $\\beta$"),
        (RP + "basis_warmup_epochs", "Basis-Warm-up [Epochen]"), (RP + "sd_margin", "SD-Margin (\\texttt{sd\\_margin})"),
        (RP + "sd_variance_weight", "SD-Varianzgewicht $\\lambda_v$"), (RP + "contrastive_pos_margin", "Positive kontrastive Margin"),
        (RP + "contrastive_neg_margin", "Negative kontrastive Margin"), ("ssl_config.update_interval_epochs", "Aktualisierung [Epochen]")]),
]


def p0_values(method, param):
    """Suchraum je Datensatz der Vorexperimente (über Verluste und beide In-Shop-Serien eindeutig)."""
    found = {}
    for s in STUDIES:
        if s["phase"] == 0 and s["method"] == method and param in s["flat"]:
            found.setdefault(P0_ALIAS.get(s["dataset_key"], s["dataset_key"]), []).append(s["flat"][param])
    return {d: unique(v, f"Phase 0 {method} {param} {d}") for d, v in found.items()}


def p0_cell(sp):
    if "choices" in sp:
        items = [c.replace(":", "{:}") if isinstance(c, str) else plain(c).replace(".", "{,}") for c in sp["choices"]]
        dist = "fest" if len(items) == 1 else "kat."
    elif "step" in sp:
        items = [plain(v).replace(".", "{,}") for v in stepped(sp)]
        items = items if len(items) <= 7 else items[:3] + ["\\dots", items[-1]]
        dist = "fest" if len(items) == 1 else "diskr."
    else:
        return interval(sp), "log." if sp.get("log") else "unif."
    return "$\\{" + ",\\,\\allowbreak ".join(items) + "\\}$", dist


def phase0():
    ran = {m: {P0_ALIAS.get(s["dataset_key"], s["dataset_key"]) for s in STUDIES if s["phase"] == 0 and s["method"] == m}
           for m in METHODS}
    lines = []
    for method, title, rows in P0_BLOCKS:
        lines += ["\\addlinespace", f"\\multicolumn{{4}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\*", "\\midrule"]
        searched = {p for s in STUDIES if s["phase"] == 0 and s["method"] == method for p in s["flat"]}
        listed = set()
        for param, label in rows:
            values = p0_values(method, param)
            if method != "sup":
                base = p0_values("sup", param)
                values = {d: v for d, v in values.items() if d not in base or canon(base[d]) != canon(v)}
            if not values:
                raise ValueError(f"Phase 0 {method} {param}: no deviation from the base configuration")
            listed.add(param)
            groups = {}
            for d, _ in P0_SETS:
                if d in values:
                    groups.setdefault(canon(values[d]), []).append(d)
            for members in groups.values():
                names = "alle" if set(members) == ran[method] else ", ".join(n for d, n in P0_SETS if d in members)
                cell, dist = p0_cell(values[members[0]])
                lines.append(f"{label} & {names} & {cell} & {dist} \\\\")
        rest = {p for p in searched - listed
                if method == "sup" or any(canon(v) != canon(p0_values("sup", p).get(d, {"x": v})) for d, v in p0_values(method, p).items())}
        if rest:
            raise ValueError(f"Phase 0 {method}: searched but not listed: {sorted(rest)}")
    return "\n".join(PHASE0_HEAD + lines + ["\\end{longtable}", "\\endgroup"]) + "\n"


PHASE0_HEAD = [
    "\\begingroup", "\\small\\setstretch{1}", "\\setlength{\\tabcolsep}{4pt}", "\\renewcommand{\\arraystretch}{1.12}",
    "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash\\hangindent=1em\\hangafter=1}p{0.30\\textwidth}>{\\raggedright\\arraybackslash}p{0.16\\textwidth}>{\\raggedright\\arraybackslash}p{\\dimexpr0.42\\textwidth-6\\tabcolsep\\relax}>{\\raggedright\\arraybackslash}p{0.12\\textwidth}@{}}",
    "\\caption[Hyperparametersuchräume der Vorexperimente]{Hyperparametersuchräume der Vorexperimente: Basiskonfiguration und verfahrensspezifische Ergänzungen bzw. Überschreibungen.}\\label{tab:first-run-hyperparameter-spaces}\\\\",
    "\\toprule", "Hyperparameter & Datensatz & Suchraum & Verteilung \\\\", "\\midrule", "\\endfirsthead",
    "\\multicolumn{4}{l}{\\tablename~\\thetable: Vorexperimente (Fortsetzung)}\\\\", "\\toprule",
    "Hyperparameter & Datensatz & Suchraum & Verteilung \\\\", "\\midrule", "\\endhead", "\\midrule",
    "\\multicolumn{4}{r}{\\emph{Fortsetzung auf der nächsten Seite}}\\\\", "\\endfoot", "\\bottomrule", "\\endlastfoot"]


# ---------------------------------------------------------------- Phase 1
GENERAL = ["sup", "iscen", "lrml", "hoffer"]
LOSS_PARAMS = [  # Verlust, Parameter, Zeile
    ("arc", "loss.ArcFaceLoss.scale", "ArcFace $s$"), ("arc", "loss.ArcFaceLoss.margin", "ArcFace $m_{\\mathrm{deg}}$ [Grad]"),
    ("cir", "loss.CircleLoss.gamma", "Circle $\\gamma$ (\\texttt{gamma})"), ("cir", "loss.CircleLoss.m", "Circle $m$"),
    ("ms", "loss.MultiSimilarityLoss.alpha", "Multi-Similarity $\\alpha$"), ("ms", "loss.MultiSimilarityLoss.beta", "Multi-Similarity $\\beta$"),
    ("ms", "loss.MultiSimilarityLoss.base", "Multi-Similarity $\\lambda$ (\\texttt{base})"),
    ("ms", "miner.MultiSimilarityMiner.epsilon", "MS-Mining $\\epsilon$"), ("nt", "loss.NTXentLoss.temperature", "NT-Xent $\\tau$"),
    ("prx", "loss.ProxyAnchorLoss.alpha", "Proxy-Anchor $\\alpha$ (\\texttt{alpha})"), ("prx", "loss.ProxyAnchorLoss.margin", "Proxy-Anchor $\\delta$")]
# Reihenfolge der SLADE-Abweichungen in der Tabelle; welche abweichen, ergibt sich aus den Daten.
SLADE_DEVIATIONS = ["loss.MultiSimilarityLoss.base", "loss.NTXentLoss.temperature", "loss.ArcFaceLoss.margin",
                    "loss.ProxyAnchorLoss.alpha", "loss.ProxyAnchorLoss.margin"]
ETA = (MP + "regularizer_target_ratio", "Zielgradienten\\-verhältnis $\\eta$")
WARM = (WARMUP, "Warm-up [Epochen]")
P1_BLOCKS = [  # Titel, Verfahren, [(Parameter, Zeile, Verluste)]
    ("Alle Verfahren außer SLADE: Optimierung und Batch", GENERAL, [
        ("batch_sampler", "Batchstruktur $B{:}\\mathrm{SPC}$", LOSSES), ("lr", "Lernrate (Circle, MS, NT-Xent)", ["cir", "ms", "nt"]),
        ("lr", "Lernrate (ArcFace, Proxy-Anchor)", ["arc", "prx"]),
        ("classifier_lr", "Klassifikator-Lernrate (ArcFace, Proxy-Anchor)", ["arc", "prx"])]),
    ("Verlustfunktionen (alle Verfahren, Abweichungen bei SLADE siehe unten)", GENERAL,
     [(p, label, [l]) for l, p, label in LOSS_PARAMS]),
    ("Label Propagation (\\texttt{iscen})", ["iscen"], [
        WARM + (LOSSES,), (MP + "n_neighbors", "Nachbarzahl $k_{\\mathrm{NN}}$", LOSSES),
        (MP + "alpha", "Diffusions\\-parameter $\\alpha$", LOSSES), (MP + "gamma", "Affinitätsexponent $\\gamma$", LOSSES),
        ("ssl_config.confidence_threshold", "Konfidenzschwelle", LOSSES), ("labeled_batch_fraction", "Gelabelter Batchanteil", LOSSES)]),
    ("LRML", ["lrml"], [
        WARM + (LOSSES,), ETA + (LOSSES,), (RP + "n_neighbors", "Nachbarzahl $k_{\\mathrm{NN}}$", LOSSES),
        (RP + "graph_batch_size", "Kanten-Batchgröße", LOSSES)]),
    ("Hoffer", ["hoffer"], [
        WARM + (LOSSES,), ETA + (LOSSES,), (RP + "distance_scale", "Distanzskala $s_{\\mathrm{H}}$", LOSSES),
        (RP + "reference_sets", "Referenzmengen $K_{\\mathrm{ref}}$", LOSSES),
        (RP + "unlabeled_batch_size", "Ungelabelte Batchgröße", LOSSES)]),
    ("SLADE", ["slade"], [
        ("batch_sampler", "Batchstruktur $B{:}\\mathrm{SPC}$", LOSSES), ("lr", "Lernrate (alle Verluste)", LOSSES),
        ("classifier_lr", "Klassifikator-Lernrate (ArcFace, Proxy-Anchor)", ["arc", "prx"]), "deviations",
        WARM + (LOSSES,), ("ssl_config.update_interval_epochs", "Teacher-Aktualisierung alle $n$ Epochen", LOSSES),
        ETA + (LOSSES,), (RP + "cluster_ratio", "Clusterverhältnis (Cluster je gelabelter Klasse)", LOSSES),
        (RP + "unlabeled_batch_size", "Ungelabelte Batchgröße", LOSSES),
        (RP + "basis_target_ratio", "Basis: Zielgradienten\\-verhältnis", LOSSES),
        (RP + "basis_warmup_epochs", "Basis: Warm-up [Epochen]", LOSSES), (RP + "beta", "Basis: Aktualisierungsrate $\\beta$", LOSSES),
        (RP + "sd_margin", "$\\mathcal{L}_{\\mathrm{SD}}$: Margin (\\texttt{sd\\_margin})", LOSSES),
        (RP + "sd_variance_weight", "$\\mathcal{L}_{\\mathrm{SD}}$: Varianzgewicht $\\lambda_v$", LOSSES),
        (RP + "contrastive_pos_margin", "Kontrastive Margin positiv (ArcFace, Proxy-Anchor)", ["arc", "prx"]),
        (RP + "contrastive_neg_margin", "Kontrastive Margin negativ (ArcFace, Proxy-Anchor)", ["arc", "prx"])]),
]


def p1_space(dataset, methods, losses, param, covered):
    found = [s for s in STUDIES if s["phase"] == 1 and s["dataset_key"] == dataset and s["method"] in methods
             and s["loss"] in losses and param in s["flat"]]
    if not found:
        raise ValueError(f"Phase 1 {dataset} {methods} {losses} {param}: no search space")
    covered.update((s["dataset_key"], s["method"], s["loss"], param) for s in found)
    return unique((s["flat"][param] for s in found), f"Phase 1 {dataset} {methods} {losses} {param}")


def p1_cell(sp):
    if "choices" in sp:
        c = sp["choices"]
        if all(isinstance(x, str) and ":" in x for x in c):
            spc = unique([x.split(":")[1] for x in c], "samples per class")
            return "\\{" + ", ".join(x.split(":")[0] for x in c) + "\\}:" + spc
        if len(c) == 1:
            return f"\\{{{plain(c[0])}\\}} (fest)"
        if any(float(x) != int(float(x)) for x in c):
            return "\\{" + "; ".join(plain(x).replace(".", ",") for x in c) + "\\}"
        return "\\{" + ", ".join(plain(x) for x in c) + "\\}"
    if sp["type"] == "int":
        v = stepped(sp)
        return "\\{" + ", ".join(map(str, v if len(v) <= 3 else [v[0], v[1], "\\dots", v[-1]])) + "\\}"
    return interval(sp) + (" log." if sp.get("log") else "")


def p1_row(label, cells):
    if len(set(cells)) == 1:
        return f"{label} & \\spFuenf{{{cells[0]}}} \\\\"
    if len(set(cells[:4])) == 1:
        return f"{label} & \\spVier{{{cells[0]}}} & {cells[4]} \\\\"
    return f"{label} & " + " & ".join(cells) + " \\\\"


def phase1():
    covered, lines = set(), []
    for title, methods, rows in P1_BLOCKS:
        lines += ["\\addlinespace", f"\\multicolumn{{6}}{{@{{}}l}}{{\\emph{{{title}}}}}\\\\*", "\\midrule"]
        for row in rows:
            if row == "deviations":
                deviating = []
                for loss, param, label in LOSS_PARAMS:
                    general = [p1_space(d, GENERAL, [loss], param, set()) for d in DATASETS]
                    slade = [p1_space(d, ["slade"], [loss], param, covered) for d in DATASETS]
                    if any(canon(g) != canon(s) for g, s in zip(general, slade)):
                        deviating.append((param, label, [("wie oben" if canon(g) == canon(s) else p1_cell(s))
                                                         for g, s in zip(general, slade)]))
                if {p for p, _, _ in deviating} != set(SLADE_DEVIATIONS):
                    raise ValueError(f"SLADE deviations changed: {[p for p, _, _ in deviating]}")
                for param in SLADE_DEVIATIONS:
                    _, label, cells = next(x for x in deviating if x[0] == param)
                    lines.append(p1_row(label + " (abweichend)", cells))
                continue
            param, label, losses = row
            lines.append(p1_row(label, [p1_cell(p1_space(d, methods, losses, param, covered)) for d in DATASETS]))
    searched = {(s["dataset_key"], s["method"], s["loss"], p) for s in STUDIES if s["phase"] == 1 for p in s["flat"]}
    if searched - covered:
        raise ValueError(f"Phase 1: searched but not listed: {sorted(searched - covered)[:10]}")
    return "\n".join(PHASE1_HEAD + lines + ["\\end{longtable}", "\\endgroup"]) + "\n"


PHASE1_HEAD = [
    "\\begingroup",
    "\\newcommand*{\\spVier}[1]{\\multicolumn{4}{>{\\centering\\arraybackslash}p{\\dimexpr0.62\\textwidth-20.8pt+6\\tabcolsep\\relax}}{#1}}",
    "\\newcommand*{\\spFuenf}[1]{\\multicolumn{5}{>{\\centering\\arraybackslash}p{\\dimexpr0.775\\textwidth-26pt+8\\tabcolsep\\relax}}{#1}}",
    "\\small\\setstretch{1}", "\\setlength{\\tabcolsep}{2.5pt}", "\\renewcommand{\\arraystretch}{1.15}",
    "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash\\hangindent=1em\\hangafter=1}p{0.22\\textwidth}*{5}{>{\\centering\\arraybackslash}p{\\dimexpr0.155\\textwidth-5.2pt\\relax}}@{}}",
    "\\caption[Hyperparametersuchräume der Phase~1]{Hyperparametersuchräume der Phase~1, gelesen aus den Konfigurationen der ausgewerteten Studien.}\\label{tab:hyperparameter-spaces}\\\\",
    "\\toprule", "Hyperparameter & Cars196 & CUB & In-Shop & In-Shop 10\\,\\% & Semi-iNat \\\\", "\\midrule", "\\endfirsthead",
    "\\multicolumn{6}{l}{\\tablename~\\thetable: Phase~1 (Fortsetzung)}\\\\", "\\toprule",
    "Hyperparameter & Cars196 & CUB & In-Shop & In-Shop 10\\,\\% & Semi-iNat \\\\", "\\midrule", "\\endhead", "\\midrule",
    "\\multicolumn{6}{r}{\\emph{Fortsetzung auf der nächsten Seite}}\\\\", "\\endfoot", "\\bottomrule", "\\endlastfoot"]


# ---------------------------------------------------------------- Protokoll der Phase-1-HPO
WIDTHS = [1.9, 1.9, 1.9, 2.3, 2.0]  # cm, Spalten Cars196 bis Semi-iNat
LABEL = {"sup": "\\UEBL", "iscen": "\\acs{LP}", "lrml": "\\acs{LRML}", "hoffer": "Hoffer", "slade": "\\acs{SLADE}"}
TARGET = {("best_valid_mean_average_precision_at_r", "maximize", "mean"):
          "Validierungs-\\acs{MAP@R}: beste Epoche je Fold, Mittel über die Folds"}
SETTINGS = pd.concat([sammlung.read_csv(f"{d}_ergebnisse/data/phase1_settings.csv").assign(dataset_key=d) for d in DATASETS],
                     ignore_index=True)
SETTINGS["family"] = SETTINGS.method.str.rsplit("_", n=1).str[0]
SETTINGS["loss"] = SETTINGS.method.str.rsplit("_", n=1).str[1]


def span(cells):
    def multi(n, text):
        width = sum(WIDTHS[:n]) if n < 5 else sum(WIDTHS)
        return f"\\multicolumn{{{n}}}{{>{{\\centering\\arraybackslash}}p{{\\dimexpr{width:.1f}cm+{2 * (n - 1)}\\tabcolsep\\relax}}}}{{{text}}}"
    if len(set(cells)) == 1:
        return multi(5, cells[0])
    if len(set(cells[:4])) == 1:
        return multi(4, cells[0]) + " & " + cells[4]
    return " & ".join(cells)


def setting(families, param, dataset=None):
    g = SETTINGS[SETTINGS.family.isin(families) & ((SETTINGS.dataset_key == dataset) if dataset else True)]
    values = [None if isinstance(v, float) and math.isnan(v) else v for v in g[param]] if param in g else [None] * len(g)
    return unique(values, f"setting {families} {param} {dataset}")


def cap(family, dataset):
    v = setting([family], MP + "regularizer_max_ratio", dataset)
    return "keine" if v is None else plain(v)


def protocol():
    p1 = [s for s in STUDIES if s["phase"] == 1]
    by = {d: [s for s in p1 if s["dataset_key"] == d] for d in DATASETS}

    def per(f):
        return [f(by[d], d) for d in DATASETS]

    def one(studies, key, part="base_args"):
        return unique((s[part][key] for s in studies), key)

    def budget(st, d):
        if d == "inshop10":
            return f"{round(100 * one(st, 'label_budget_grid')[0])}\\,\\% der Klassen"
        return f"$k={one(st, 'k_shot_grid')[0]}$"

    def trials(st, d):
        n, start = one(st, "n_trials", "hparam_config"), one(st, "tpe_startup_trials", "hparam_config")
        if one(st, "sampler", "hparam_config") != "tpe":
            raise ValueError("sampler")
        return f"{n}: zuerst {start} zufällige Startup-Trials, danach {n - start} \\acs{{TPE}}-Trials"

    def sampler(st, d):
        params = one(st, "sampler_params", "hparam_config")
        mode = {(True, True): ", multivariat und gruppiert", (False, False): ", univariat"}[
            (bool(params.get("multivariate")), bool(params.get("group")))]
        return "Optuna-\\acs{TPE}" + mode

    def seeds(st, d):
        seed = unique([one(st, k) for k in ("hparam_seed", "seed", "support_seed", "data_split_seed")], "seeds")
        return f"{seed} für \\acs{{HPO}}, Training und Auswahl der gelabelten Beispiele"

    def target(st, d):
        key = (one(st, "metric", "hparam_config"), one(st, "direction", "hparam_config"), "mean")
        return TARGET[key]

    def epochs(st, d):
        return f"höchstens {max(s['base_args']['epochs'] for s in st)} ({one(st, 'patience')})"

    def caps(st, d):
        return ", ".join(f"{LABEL[f]}: {cap(f, d)}" for f in ("lrml", "hoffer", "slade"))

    def counts(st, d):
        parts = []
        for m in METHODS:
            n = [len(s["flat"]) for s in st if s["method"] == m]
            parts.append(f"{LABEL[m]}: {min(n)}" + (f"--{max(n)}" if max(n) > min(n) else ""))
        return ", ".join(parts)

    pruning = per(lambda st, d: {"none": "keines"}[one(st, "pruner", "hparam_config")])
    rows = [("Label-Budget der \\acs{HPO}", " & ".join(per(budget))),
            ("Folds (klassendisjunkt)", " & ".join(str(c) for c in per(lambda st, d: one(st, "cv_k")))),
            ("Trials je Studie", span(per(trials))), ("Sampler", span(per(sampler))), ("Pruning", span(pruning)),
            ("Seeds", span(per(seeds))), ("Zielgröße", span(per(target))), ("Max.\\ Epochen (Patience)", span(per(epochs))),
            ("Obergrenze des Gradienten\\-verhältnisses", span(per(caps))),
            ("Gesuchte Hyperparameter je Studie", span(per(counts)))]
    body = []
    for i, (label, cells) in enumerate(rows):
        if i:
            body.append("\t\t\\addlinespace")
        body.append(f"\t\t{label} & {cells} \\\\")
    return "\n".join(PROTOCOL_HEAD + body + ["\t\t\\bottomrule", "\t\\end{tabular}", "\\end{table}"]) + "\n"


PROTOCOL_HEAD = [
    "\\begin{table}[htbp]", "\t\\centering",
    "\t\\caption[Protokoll der Phase-1-\\acs{HPO}]{Protokoll der Phase-1-\\acs{HPO} je Datensatz. In-Shop 10\\,\\%: 10\\,\\% der Trainingsklassen vollständig gelabelt, übrige Klassen ungelabelt. \\UEBL: supervised Referenz. Obergrenze: Begrenzung des Gradientenverhältnisses (Abschnitt~\\ref{sec:m:gewicht}).}",
    "\t\\label{tab:phase1:hpo-protokoll}", "\t\\setlength{\\tabcolsep}{4pt}",
    "\t\\begin{tabular}{@{}>{\\raggedright\\arraybackslash}p{3.2cm}" + "".join(f">{{\\centering\\arraybackslash}}p{{{w}cm}}" for w in WIDTHS) + "@{}}",
    "\t\t\\toprule", "\t\tMerkmal & Cars196 & CUB & In-Shop & In-Shop 10\\,\\% & Semi-iNat \\\\", "\t\t\\midrule"]


# ---------------------------------------------------------------- Fest gesetzte Einstellungen
SSL = ["iscen", "lrml", "hoffer", "slade"]
WEIGHTED = ["lrml", "hoffer", "slade"]


def tt(s):
    return "\\texttt{" + s.replace("_", "\\_") + "}"


def fixed():
    if (setting(SSL, "ssl_config.unlabeled_class_scope"), setting(SSL, "ssl_config.max_unlabeled_samples"),
            setting(SSL, "ssl_config.unlabeled_fraction")) != ("all", None, None):
        raise ValueError("unlabeled pool")
    selection = {True: "nach dem Warm-up neu begonnen"}[setting(SSL, "ssl_config.restart_selection_after_warmup")]
    weighting = (f"über das Zielgradienten\\-verhältnis $\\eta$ (Abschnitt~\\ref{{sec:m:gewicht}}), Gradientenstatistik "
                 f"{tt(setting(WEIGHTED, MP + 'regularizer_target_ratio_statistic'))}, EMA mit "
                 f"$\\zeta = {num(setting(WEIGHTED, MP + 'regularizer_target_ratio_ema_alpha'))}$, Neuberechnung alle "
                 f"\\num{{{plain(setting(WEIGHTED, MP + 'regularizer_target_ratio_update_interval_samples'))}}} Trainingsbeispiele, "
                 f"Startgewicht ${num(setting(WEIGHTED, MP + 'regularizer_weight'))}$")
    lp_update = {("every_epoch", 1): "in jeder Epoche neu berechnet"}[
        (setting(["iscen"], "ssl_config.update_mode"), setting(["iscen"], "ssl_config.update_interval_epochs"))]
    solver = setting(["iscen"], MP + "linear_solver")
    propagation = ({"torch_block_cg": "konjugierte Gradienten"}[solver] + f" ({tt(solver)}), relative Toleranz "
                   f"${num(setting(['iscen'], MP + 'cg_rtol'))}$, höchstens \\num{{{plain(setting(['iscen'], MP + 'cg_max_iter'))}}} Iterationen")
    graph = ", ".join([{"all": "über gelabelte und ungelabelte Bilder"}[setting(["lrml"], RP + "graph_on")],
                       {"binary": "binäre Kantengewichte"}[setting(["lrml"], RP + "edge_weighting")],
                       {True: "normierter Laplace-Operator"}[setting(["lrml"], RP + "normalized_laplacian")],
                       {("every_n_epochs", 1): "in jeder Epoche neu aufgebaut"}[
                           (setting(["lrml"], "ssl_config.update_mode"), setting(["lrml"], "ssl_config.update_interval_epochs"))]])
    ranking =SETTINGS[SETTINGS.family == "slade"].groupby("loss")[RP + "unlabeled_ranking_loss"].unique()
    if not ranking.map(lambda v: len(v) == 1 and v[0] in ("contrastive", "supervised")).all():
        raise ValueError("SLADE unlabeled loss")
    contrastive = [l for l in LOSSES if ranking[l][0] == "contrastive"]
    names = {"arc": "ArcFace", "cir": "Circle", "ms": "Multi-Similarity", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
    unlabeled = (f"kontrastiver Verlust bei {' und '.join(names[l] for l in contrastive)}, "
                 "sonst derselbe Verlust wie für die gelabelten Daten")
    clusters = {None: "aus dem Clusterverhältnis relativ zur Zahl der gelabelten Klassen"}[setting(["slade"], RP + "num_clusters")]
    rows = [
        ("Alle SSL", "Ungelabelter Pool", "alle ungelabelten Bilder der Trainingsklassen des Folds"),
        ("", "Modellauswahl", selection), None,
        ("LRML, Hoffer, \\acs{SLADE}", "Gewichtung des SSL-Terms", weighting),
        ("", "Gewicht des supervised Verlusts", plain(setting(WEIGHTED, MP + "supervised_weight"))), None,
        ("LP", "Pseudo-Labels", lp_update), ("", "Lösung der Propagation", propagation), None,
        ("LRML", "Graph", graph), ("", "Reduktion des Regularisierers", {"torch_sum": "Summe"}[setting(["lrml"], RP + "reduction")]),
        ("", "Obergrenze des Gradienten\\-verhältnisses", cap("lrml", None)), None,
        ("Hoffer", "Obergrenze des Gradienten\\-verhältnisses", cap("hoffer", None)), None,
        ("SLADE", "Obergrenze des Gradienten\\-verhältnisses", cap("slade", None)),
        ("", "Verlust der ungelabelten Paare", unlabeled),
        ("", "Basisverlust", f"Startgewicht $\\lambda_{{\\mathrm{{basis}}}} = {num(setting(['slade'], RP + 'lambda_basis'))}$, "
                             "kalibriert über das Basis-Zielgradienten\\-verhältnis"),
        ("", "Gradient von $\\mathcal{L}_{\\mathrm{SD}}$",
         {"damped": "gedämpft mit dem Faktor $(1-\\beta)$ (\\texttt{damped})"}[setting(["slade"], RP + "sd_gradient_mode")]),
        ("", "Clusterzahl", clusters)]
    body = ["\\addlinespace" if r is None else (f"{r[0]} & {r[1]} & {r[2]} \\\\" if r[0] else f" & {r[1]} & {r[2]} \\\\")
            for r in rows]
    return "\n".join(FIXED_HEAD + body + ["\\end{longtable}", "\\endgroup"]) + "\n"


FIXED_HEAD = [
    "\\begingroup", "\\small\\setstretch{1}", "\\setlength{\\tabcolsep}{4pt}", "\\renewcommand{\\arraystretch}{1.12}",
    "\\begin{longtable}{@{}>{\\raggedright\\arraybackslash}p{0.13\\textwidth}>{\\raggedright\\arraybackslash}p{0.30\\textwidth}>{\\raggedright\\arraybackslash}p{\\dimexpr0.57\\textwidth-4\\tabcolsep\\relax}@{}}",
    "\\caption[Fest gesetzte Einstellungen der SSL-Verfahren]{Fest gesetzte Einstellungen der SSL-Verfahren in den Phase-1-Studien (nicht gesucht, auf allen Datensätzen gleich, sofern nicht anders angegeben).}\\label{tab:phase1:feste-einstellungen}\\\\",
    "\\toprule", "Verfahren & Einstellung & Wert \\\\", "\\midrule", "\\endfirsthead",
    "\\multicolumn{3}{l}{\\tablename~\\thetable: Fortsetzung}\\\\", "\\toprule", "Verfahren & Einstellung & Wert \\\\",
    "\\midrule", "\\endhead", "\\bottomrule", "\\endlastfoot"]


for name, make in (("suchraum_phase0", phase0), ("hpo_protokoll", protocol), ("suchraum_phase1", phase1),
                   ("feste_einstellungen", fixed)):
    with open(OUT + name + ".tex", "w", encoding="utf-8") as f:
        f.write(make())
    print(OUT + name + ".tex")
