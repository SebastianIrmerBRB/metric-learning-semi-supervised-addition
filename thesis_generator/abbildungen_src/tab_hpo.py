"""Übernommene HPO-Konfigurationen der Phase 1 je Datensatz: kapitel/<ds>_tabellen/phase1_hpo.tex.

Trial, Lernrate, Batchgröße, Warm-up, Epochengrenze, Patience und verzögerte Patience aus den aufgelösten Einstellungen
der Wiederholungen (Sicht phase1_settings), Budget und Seed der Suche aus studies.json.
"""
import pandas as pd

import datensammlung as sammlung

FAM = {"sup": "\\UEBL", "iscen": "LP", "lrml": "LRML", "hoffer": "Hoffer", "slade": "SLADE"}
LOSS = {"arc": "ArcFace", "cir": "Circle", "ms": "Multi-Similarity", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
# dataset_key, Name in der Kurzbeschriftung, Spalte E/P statt Satz in der Bildunterschrift, \small
DATA = [("cars", "Cars196", False, False), ("cub", "CUB", False, False), ("inshop", "In-Shop", False, False),
        ("inshop10", "In-Shop mit gelabelten Klassen", False, True), ("semiinat", "Semi-iNat-2021", True, False)]
COLUMNS = {"trial": "hparam_final_trial_number", "lr": "lr", "b": "batch_size", "tw": "ssl_config.warmup_epochs",
           "epochs": "epochs", "patience": "patience"}

STUDIES = [s for s in sammlung.read_json("studies.json") if s["phase"] == 1]


def unique(values, what):
    values = set(values)
    if len(values) != 1:
        raise ValueError(f"{what} not unique: {sorted(values)}")
    return values.pop()


def search(key):
    args = [s["base_args"] for s in STUDIES if s["dataset_key"] == key]
    seed = unique((a["hparam_seed"] for a in args), f"HPO seed {key}")
    if key == "inshop10":
        share = unique((tuple(a["label_budget_grid"]) for a in args), f"HPO budget {key}")
        return f"{round(100 * share[0])}\\% vollständig gelabelten Klassen", seed
    k = unique((tuple(a["k_shot_grid"]) for a in args), f"HPO budget {key}")
    return f"$k={k[0]}$", seed


def delayed(settings):
    """Kombinationen, die mit verzögerter Patience berichtet werden, aus den Einstellungen der berichteten Läufe."""
    if "patience_delay_epochs" not in settings:
        return []
    used = settings[pd.to_numeric(settings.patience_delay_epochs, errors="coerce").fillna(0) > 0]
    names = []
    for method, g in used.groupby("method", sort=False):
        fam, loss = method.rsplit("_", 1)
        name = f"{'die ' if fam == 'sup' else ''}{FAM[fam]} mit {LOSS[loss]}"
        if "budget_percent" in g:
            shares = sorted(g.budget_percent.dropna().astype(int).unique())
            if shares:
                name += f" bei {shares[0]} bis {shares[-1]}\\% gelabelter Klassen"
        names.append(name)
    return names


for key, name, ep_column, small in DATA:
    settings = sammlung.read_csv(f"{key}_ergebnisse/data/phase1_settings.csv")
    budget, seed = search(key)
    legend = "Versuch: Nummer innerhalb der Suche, $B$: supervised bzw. gemischte Batchgröße, Warm-up in Epochen"
    if ep_column:
        legend += ", Ep./Patience: Epochenlimit und Early-Stopping-Patience der Wiederholungen."
    else:
        patience = unique(settings.patience, f"Patience {key}")
        legend += (f". Wiederholungen mit höchstens \\num{{{settings.epochs.max()}}} Epochen und einer Patience "
                   f"von {patience} Epochen")
        names = delayed(settings)
        if names:
            legend += ". Verzögerte Patience (Abschnitt~\\ref{sec:training}) für " + " und ".join(names)
        legend += "."
    lines = ["\\begin{table}[htbp]", "\\centering",
             f"\\caption[Übernommene Konfigurationen {name}]{{Für die Wiederholungen übernommene HPO-Konfigurationen "
             f"(Optimierung bei {budget}, Seed~{seed}). "
             f"{legend} Übrige Loss-, Miner- und SSL-Parameter im digitalen Anhang.}}",
             f"\\label{{tab:{key}:hpo}}"]
    if small:
        lines.append("\\small")
    lines += ["\\begin{tabular}{@{}llrrrr" + ("r" if ep_column else "") + "@{}}", "\\toprule",
              "Verlust & Verf. & Versuch & Lernrate & $B$ & Warm-up" + (" & Ep./Patience" if ep_column else "") + " \\\\",
              "\\midrule"]
    for i, (lk, ln) in enumerate(LOSS.items()):
        if i:
            lines.append("\\addlinespace[3pt]")
        for fk, fn in FAM.items():
            g = settings[settings.method == f"{fk}_{lk}"]
            # Epochengrenze und Patience je Verfahren nur, wo die Spalte E/P sie zeigt (In-Shop: Ü mit 600 Epochen bis auf
            # drei Wiederholungen mit 1000 an der Grenze, die Bildunterschrift nennt die höchste Grenze)
            v = {c: unique(g[col], f"{c} {key} {fk}_{lk}") for c, col in COLUMNS.items()
                 if ep_column or c not in ("epochs", "patience")}
            row = f"{ln} & {fn} & {v['trial']} & \\num{{{v['lr']:.3e}}} & {v['b']} & {v['tw']}"
            lines.append(row + (f" & {v['epochs']}/{v['patience']}" if ep_column else "") + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    out = sammlung.output_dir(f"kapitel/{key}_tabellen") + "phase1_hpo.tex"
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(out)
