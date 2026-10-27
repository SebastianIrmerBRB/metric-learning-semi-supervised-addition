"""Übernommene HPO-Konfigurationen der Phase 1 je Datensatz: kapitel/<ds>_tabellen/phase1_hpo.tex.

Trial, Lernrate, Batchgröße, Warm-up, Epochengrenze und Patience aus den aufgelösten Einstellungen der Wiederholungen
(Sicht phase1_settings), Budget und Seed der Suche aus studies.json, verzögerte Patience aus patience_delay/.
"""
import re

import datensammlung as sammlung

FAM = {"sup": "\\UEBL", "iscen": "LP", "lrml": "LRML", "hoffer": "Hoffer", "slade": "SLADE"}
LOSS = {"arc": "ArcFace", "cir": "Circle", "ms": "Multi-Similarity", "nt": "NT-Xent", "prx": "Proxy-Anchor"}
# dataset_key, Name in der Kurzbeschriftung, Spalte E/P statt Satz in der Bildunterschrift, \small
DATA = [("cars", "Cars196", False, False), ("cub", "CUB", False, False), ("inshop", "In-Shop", False, False),
        ("inshop10", "In-Shop mit gelabelten Klassen", False, True), ("semiinat", "Semi-iNat-2021", True, False)]
COLUMNS = {"trial": "hparam_final_trial_number", "lr": "lr", "b": "batch_size", "tw": "ssl_config.warmup_epochs",
           "epochs": "epochs", "patience": "patience"}

STUDIES = [s for s in sammlung.read_json("studies.json") if s["phase"] == 1]
CHOSEN = sammlung.read_csv("patience_delay/auswahl.csv")
DELAY_ONLY = sammlung.read_csv("patience_delay/ersatz_p40.csv")


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


def delayed(key, settings):
    """Kombinationen, deren Läufe alle nur mit verzögerter Patience eingehen, und ob weitere Läufe sie verwenden."""
    runs = settings.groupby("method").size()
    named, partial = [], False
    for group, g in DELAY_ONLY.groupby("gruppe"):
        m = re.fullmatch(rf"phase1_{key}_([a-z]+)_([a-z]+)", group)
        if m is None:
            partial |= group.startswith(f"phase1_{key}_")
        elif len(g) == runs.get(f"{m[1]}_{m[2]}", -1):
            named.append((m[1], m[2]))
        else:
            partial = True
    chosen = CHOSEN[(CHOSEN.quelle == f"phase1_{key}") & CHOSEN.wiederholung_gewaehlt].shape[0] > 0
    names = [f"{FAM[f]} mit {LOSS[l]}" for f, l in sorted(named, key=lambda x: (list(FAM).index(x[0]), list(LOSS).index(x[1])))]
    if chosen or partial:
        names.append("Läufe mit zu frühem Abbruch")
    return names


for key, name, ep_column, small in DATA:
    settings = sammlung.read_csv(f"{key}_ergebnisse/data/phase1_settings.csv")
    budget, seed = search(key)
    legend = "$B$: supervised bzw. gemischte Batchgröße, $T_{\\mathrm{W}}$: Warm-up-Epochen"
    if ep_column:
        legend += ", $E/P$: Epochenlimit und Early-Stopping-Patience der Wiederholungen."
    else:
        patience = unique(settings.patience, f"Patience {key}")
        legend += (f". Die Wiederholungen laufen mit höchstens \\num{{{settings.epochs.max()}}} Epochen und einer Patience "
                   f"von {patience} Epochen")
        names = delayed(key, settings)
        if names:
            legend += ", " + " und ".join(names) + " mit verzögerter Patience (Abschnitt~\\ref{sec:training})"
        legend += "."
    lines = ["\\begin{table}[htbp]", "\\centering",
             f"\\caption[Übernommene Konfigurationen {name}]{{Für die Wiederholungen übernommene HPO-Konfigurationen "
             f"(Optimierung bei {budget}, Seed~{seed}). Die Trial-Nummer ist nur innerhalb der jeweiligen Studie eindeutig. "
             f"{legend} Sämtliche Loss-, Miner- und SSL-Parameter sind im digitalen Anhang gespeichert.}}",
             f"\\label{{tab:{key}:hpo}}"]
    if small:
        lines.append("\\small")
    lines += ["\\begin{tabular}{@{}llrrrr" + ("r" if ep_column else "") + "@{}}", "\\toprule",
              "Verlust & Verf. & Trial & Lernrate & $B$ & $T_{\\mathrm{W}}$" + (" & $E/P$" if ep_column else "") + " \\\\",
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
