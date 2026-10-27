#!/usr/bin/env python3
"""Evaluate the Phase-2 ablations: every variant against its baseline, and the 2x2 table.

Reads the per-seed ``cv_summary.json`` of every replay below ``--roots`` (the
layout ``make_phase2_ablations.py`` writes: ``<root>/<combo>[_<variant>]/...``)
and pairs each variant with the unmodified replay of the same dataset, budget,
loss and method, seed by seed. A baseline is the plain replay ``<combo>`` or the
explicit ``<combo>_defaults`` variant; if both exist, ``defaults`` wins because it
went through the same ablation launcher as the variants.

Every variant gets a readable name from ``ABLATIONS`` (German for the thesis,
English for the app), the 2x2 cell it fills if any, and whether it uses hidden
labels (oracle variants are diagnostics, never SSL results).

For each (dataset, k, loss, SSL method) with all four cells the 2x2 table is
decomposed into:

* data effect with SSL HPs      = SSL baseline      - pool_labeled
* data effect with supervised HPs = sup_all_shared  - supervised baseline
* HP effect without unlabeled data = pool_labeled   - supervised baseline
* HP effect with unlabeled data    = SSL baseline   - sup_all_shared
* interaction = difference of the two data effects

Differences are paired over the seeds both sides share. With one seed there is
no spread, so p-values are left empty rather than pretending.

Outputs below ``--out/ablations``: variants.csv, two_by_two.csv, and (with
``--app``) ``public/ablations.json`` for the Trial Atlas page ``/phase2``.

Examples::

    python main_eval/ablation_report.py
    python main_eval/ablation_report.py --roots logs/final/phase_2/in_shop --metric map_at_r
    python main_eval/ablation_report.py --app react_trial_visualizer
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

BASELINE_VARIANTS = ("defaults", "")
PHASE2_ROOT = common.REPO / "logs" / "final" / "phase_2"
METHOD_LABEL = {"sup": "Überwacht", "lp": "LP (Iscen)", "lrml": "LRML", "hoffer": "Hoffer", "slade": "SLADE"}
LOSS_LABEL = {"ms": "MS", "cir": "Circle", "nt": "NT-Xent", "arc": "ArcFace", "prx": "ProxyAnchor"}
DATASET_LABEL = {"Cars196": "Cars196", "DeepFashionInShop": "In-Shop", "CUB": "CUB"}

# variant key -> (German label, English label, group, 2x2 cell or None, uses hidden labels, description)
ABLATIONS = {
    "baseline": ("SSL-HPs mit ungelabelten Daten", "SSL HPs, with unlabeled data", "2x2", "ssl_hp/with_data", False,
                 "Unveränderte Wiederholung der getunten SSL-Konfiguration."),
    "sup_all_shared": ("Überwachte HPs mit ungelabelten Daten", "Supervised HPs, with unlabeled data", "2x2",
                       "sup_hp/with_data", False,
                       "SSL-Verfahren, alle mit dem überwachten Lauf geteilten HPs auf dessen bestem Trial."),
    "pool_labeled": ("SSL-HPs ohne ungelabelte Daten", "SSL HPs, no unlabeled data", "2x2", "ssl_hp/no_data", False,
                     "Ungelabelter Term sieht nur die gelabelten Samples (Labels verborgen); gleiche Gewichtung "
                     "und gleiche Updates pro Epoche."),
    "zero_weight": ("Gewicht des ungelabelten Terms = 0", "Unlabeled weight 0", "Kontrolle", None, False,
                    "Ungelabelte Daten vorhanden, ihr Beitrag abgeschaltet (LP: keine Pseudo-Labels)."),
    "no_warmup": ("Ohne Warm-up", "No warm-up", "Kontrolle", None, False, "warmup_epochs = 0."),
    "pl_once": ("Pseudo-Labels einmalig", "Pseudo-labels frozen", "Kontrolle", None, False,
                "Pseudo-Labels einmal nach dem Warm-up berechnet, danach eingefroren."),
    "graph_once": ("Graph einmalig", "Graph frozen", "Kontrolle", None, False,
                   "LRML-Graph einmal nach dem Warm-up berechnet, danach eingefroren."),
    "graph_rewired": ("Graph gradtreu umverdrahtet", "Degree-preserving rewiring", "Graph-Kontrolle", None, False,
                      "Kanten zufällig neu verbunden, Knotengrade, Kantenzahl und Gewichte bleiben."),
    "graph_oracle_pruned": ("Graph: nur gleichklassige Kanten", "Graph: oracle-pruned", "Oracle", None, True,
                            "Verschiedenklassige Kanten mit verborgenen Labels entfernt (Diagnose)."),
    "graph_random_pruned": ("Graph: zufällig ausgedünnt", "Graph: random-pruned", "Oracle", None, True,
                            "Gleich viele Kanten wie beim Oracle zufällig entfernt (Dichtekontrolle)."),
    "diag": ("Diagnose (unverändert)", "Diagnostics only", "Kontrolle", None, False,
             "Wie die Baseline, nur mit Pseudo-Label-Diagnosen."),
    "clusters_once": ("Cluster einmalig", "Clusters frozen", "Kontrolle", None, False,
                      "SLADE-Cluster einmal nach dem Warm-up berechnet, danach eingefroren."),
    # SLADE paper ablations (Duan et al. 2021, Sec. 4.5), make_phase2_ablations.py paper.
    "t4_pseudo_label": ("Nur Pseudo-Label-Paare", "Pseudo-label pairs only", "Paper-Ablation", None, False,
                        "Tab. 4: Ranking-Loss auf allen Paaren der Cluster, Basis nicht trainiert."),
    "t4_pseudo_label_basis": ("Pseudo-Label-Paare + Basis", "Pseudo-label pairs + basis", "Paper-Ablation", None,
                              False, "Tab. 4: Basis trainiert, wählt aber keine Paare aus."),
    "t4_matched_count": ("Pseudo-Label-Paare, Anzahl wie Mining", "Pseudo-label pairs at mining count",
                         "Paper-Ablation", None, False,
                         "So viele zufällige Cluster-Paare, wie das Basis-Mining im selben Schritt auswählt."),
    "t5_local_ce": ("Basis: lokale Kreuzentropie", "Basis: local CE", "Paper-Ablation", None, False,
                    "Tab. 5: Kreuzentropie je Paar statt Similarity-Distribution-Loss."),
    "t5_global_ce": ("Basis: globale Kreuzentropie", "Basis: global CE", "Paper-Ablation", None, False,
                     "Tab. 5: Kreuzentropie auf den globalen Mittelwerten statt Similarity-Distribution-Loss."),
    "t5_ce_only": ("Basis: nur L_CE", "Basis: CE only", "Paper-Ablation", None, False,
                   "Basis nur mit der Kreuzentropie auf gelabelten Daten trainiert."),
    "paper_beta": ("β = 0,99 (Paper)", "beta = 0.99 (paper)", "Paper-Ablation", None, False,
                   "Aktualisierungsrate der Gaußverteilungen wie im Paper."),
    "paper_unlabeled_batch": ("Ungelabelter Batch 32 (Paper)", "Unlabeled batch 32 (paper)", "Paper-Ablation", None,
                              False, "Größe des ungelabelten Batches wie im Paper."),
    "paper_weights": ("Feste Gewichte λ1 = 1, λ2 = 0,25 (Paper)", "Fixed weights (paper)", "Paper-Ablation", None,
                      False, "Gewichte wie im Paper fest statt über Gradientenverhältnisse kalibriert."),
    "paper_no_basis_warmup": ("Ohne Basis-Warm-up", "No basis warm-up", "Paper-Ablation", None, False,
                              "Basis und Student von Anfang an gemeinsam trainiert."),
    "paper_teacher_student": ("Lehrer-Schüler-Grenze", "Teacher/student boundary", "Paper-Ablation", None, False,
                              "Lehrer clustert einmal, Schüler startet mit neuem Optimierer."),
    "paper_new_student_per_iteration": ("Neue Basis je Iteration", "New basis per iteration", "Paper-Ablation", None,
                                        False, "Basis, Gaußverteilungen und Basis-Warm-up bei jedem Clustern neu."),
}
ORACLE_NOISE = re.compile(r"^oracle_noise_(\d+)$")
SUP_PARAM = re.compile(r"^sup_(.+)$")
# Table 6 of the SLADE paper, a cluster_ratio sweep under the paper's table number.
PAPER_TABLE_SWEEP = re.compile(r"^t\d+_(?P<param>.+)_(?P<value>m?\d+(?:p\d+)?)$")


def describe(variant: str, method: str = "") -> dict:
    key = "baseline" if variant in BASELINE_VARIANTS else variant
    if key == "baseline" and method == "sup":
        return {"key": key, "label_de": "Überwachte Referenz (ohne ungelabelte Daten)",
                "label_en": "Supervised reference (no unlabeled data)", "group": "2x2", "cell": "sup_hp/no_data",
                "uses_hidden_labels": False, "description": "Unveränderte Wiederholung des getunten überwachten Laufs."}
    if key in ABLATIONS:
        de, en, group, cell, oracle, text = ABLATIONS[key]
    elif (match := ORACLE_NOISE.match(key)) and method == "slade":
        p = int(match.group(1))
        de, en, group, cell, oracle = f"Oracle-Cluster, {p} % Rauschen", f"Oracle clusters, {p}% noise", "Oracle", None, True
        text = f"Wahre Klassen als Cluster, {p} % der Bilder fest pro Seed einer falschen Klasse zugeordnet."
    elif match := ORACLE_NOISE.match(key):
        p = int(match.group(1))
        de, en, group, cell, oracle = f"Oracle-Labels, {p} % Rauschen", f"Oracle labels, {p}% noise", "Oracle", None, True
        text = f"LP-Abdeckung mit wahren Labels; {p} % davon fest pro Seed durch falsche ersetzt."
    elif match := PAPER_TABLE_SWEEP.match(key):
        param, value = match.group("param"), match.group("value").replace("p", ",")
        de, en, group, cell, oracle = f"{param} = {value} (Paper)", f"{param} = {value.replace(',', '.')} (paper)", \
            "Sweep", None, False
        text = "Einzelparameter-Variante aus einer Tabelle des Papers."
    elif match := SUP_PARAM.match(key):
        param = match.group(1)
        de, en, group, cell, oracle = f"Überwachter Wert: {param}", f"Supervised value: {param}", "HP-Pfad", None, False
        text = f"Nur {param} auf den Wert des besten überwachten Trials gesetzt."
    else:
        de, en, group, cell, oracle, text = key, key, "Sweep", None, False, "Einzelparameter-Variante."
    return {"key": key, "label_de": de, "label_en": en, "group": group, "cell": cell,
            "uses_hidden_labels": oracle, "description": text}


SWEEP_VALUE = re.compile(r"^(?P<param>.+)_(?P<value>m?\d+(?:p\d+)?)$")


def sweep_point(variant: str, baseline_args: dict | None) -> dict | None:
    """Parameter, value and tuned value of a one-at-a-time sweep variant such as ``distance_scale_0p5``."""
    match = SWEEP_VALUE.match(variant or "")
    if not match:
        return None
    raw = match.group("value").replace("p", ".")
    number = -float(raw[1:]) if raw.startswith("m") else float(raw)

    def find(node, key):
        if isinstance(node, dict):
            if key in node and isinstance(node[key], (int, float)) and not isinstance(node[key], bool):
                return float(node[key])
            for child in node.values():
                found = find(child, key)
                if found is not None:
                    return found
        return None

    # A paper-table sweep such as t6_cluster_ratio_0p5 moves cluster_ratio.
    param = re.sub(r"^t\d+_", "", match.group("param"))
    return {"param": param, "value": number, "tuned": find(baseline_args or {}, param)}


def first_and_last_line(path: Path) -> list[bytes]:
    """The first and the last non-empty line of a JSONL file without reading the middle (files hold ~300 generations)."""
    with open(path, "rb") as handle:
        first = handle.readline().strip()
        handle.seek(0, 2)
        end = handle.tell()
        chunk, position, tail = 1 << 16, end, b""
        while position > 0:
            position = max(0, position - chunk)
            handle.seek(position)
            tail = handle.read(end - position)
            lines = [line for line in tail.split(b"\n") if line.strip()]
            if len(lines) >= 2 or position == 0:
                break
        last = lines[-1].strip() if lines else first
    return [line for line in (first, last) if line]


def pseudo_label_quality(run) -> dict | None:
    """Mean over folds of LP's accepted pseudo-label accuracy, first and last generation (hidden labels, diagnostic)."""
    first, last, share, precision = [], [], [], []
    for fold_dir in run.fold_dirs():
        files = sorted(Path(fold_dir).glob("pseudo_label_diagnostics.jsonl"))
        if not files:
            continue
        rows = []
        for line in first_and_last_line(files[-1]):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        rows = [r for r in rows if isinstance(r.get("accepted"), dict) and r["accepted"].get("audit_accuracy") is not None]
        if not rows:
            continue
        first.append(rows[0]["accepted"]["audit_accuracy"])
        last.append(rows[-1]["accepted"]["audit_accuracy"])
        raw_count = (rows[-1].get("raw") or {}).get("count")
        if raw_count:
            share.append(rows[-1]["accepted"].get("count", 0) / raw_count)
        pair = (rows[-1]["accepted"].get("pairs") or {}).get("pair_precision")
        if pair is not None:
            precision.append(pair)
    if not last:
        return None
    mean = lambda values: float(np.mean(values)) if values else None  # noqa: E731
    return {"folds": len(last), "accuracy_first": mean(first), "accuracy_last": mean(last),
            "accepted_share_last": mean(share), "pair_precision_last": mean(precision)}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--roots", type=Path, nargs="+", default=None,
                        help="replay roots; default: every dataset folder below logs/final/phase_2")
    parser.add_argument("--metric", default="map_at_r", help="map_at_r, p_at_1, rp, ... (see common.PRIMARY_METRICS)")
    parser.add_argument("--out", type=Path, default=common.DEFAULT_OUT)
    parser.add_argument("--app", type=Path, default=None, help="Trial Atlas folder; writes public/ablations.json")
    parser.add_argument("--stdout", action="store_true",
                        help="print the app snapshot as JSON to stdout and write no files (the dev server's live view)")
    return parser.parse_args()


def default_roots():
    return sorted(p for p in PHASE2_ROOT.glob("*") if p.is_dir() and not p.name.startswith(("_", "."))
                  and not p.name.endswith("_geometry"))


def value(run, metric, aggregation):
    measured = run.metric(metric, aggregation)
    if isinstance(measured, dict):
        measured = measured.get("mean")
    return None if measured is None else 100.0 * float(measured)


def paired(a: dict, b: dict) -> dict:
    """Paired difference a - b over shared seeds."""
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return {"n": 0}
    diff = np.array([a[s] - b[s] for s in seeds])
    out = {"n": len(seeds), "mean": float(diff.mean()), "seeds": seeds,
           "positive": int((diff > 0).sum())}
    if len(diff) > 1:
        out["sd"] = float(diff.std(ddof=1))
        out["se"] = out["sd"] / math.sqrt(len(diff))
        out["p"] = float(stats.ttest_1samp(diff, 0.0).pvalue) if out["sd"] > 0 else None
    return out


def summary(values: dict) -> dict:
    arr = np.array(list(values.values()), dtype=float)
    out = {"n": len(arr), "mean": float(arr.mean()), "per_seed": {str(k): v for k, v in sorted(values.items())}}
    if len(arr) > 1:
        out["sd"] = float(arr.std(ddof=1))
    return out


def main():
    args = parse_args()
    real_stdout = sys.stdout
    if args.stdout:
        sys.stdout = sys.stderr  # progress goes to stderr, only the JSON to stdout
    roots = args.roots or default_roots()
    runs = common.load_seeded_runs(roots)
    print(f"{len(runs)} complete seed runs below {', '.join(str(r) for r in roots)}")

    # (dataset, k, loss, method, variant) -> {aggregation: {seed: value}}
    table = defaultdict(lambda: defaultdict(dict))
    quality = defaultdict(list)  # (dataset, k, loss, method, variant) -> pseudo-label quality per seed
    run_args = {}  # (dataset, k, loss, method, variant) -> resolved run_config of one fold (tuned values live there)
    for run in runs:
        key = (run.dataset, run.k, run.loss, run.method, run.variant)
        if key not in run_args:
            config = {}
            for fold_dir in run.fold_dirs()[:1]:
                path = Path(fold_dir) / "run_config.json"
                if path.exists():
                    try:
                        config = json.loads(path.read_text())
                    except json.JSONDecodeError:
                        config = {}
            run_args[key] = config
        if run.method == "lp":
            q = pseudo_label_quality(run)
            if q:
                quality[key].append(q)
        for aggregation in ("mean", "concat"):
            v = value(run, args.metric, aggregation)
            if v is not None:
                table[(run.dataset, run.k, run.loss, run.method, run.variant)][aggregation][run.seed] = v

    def baseline_of(dataset, k, loss, method):
        for name in BASELINE_VARIANTS:
            if (dataset, k, loss, method, name) in table:
                return table[(dataset, k, loss, method, name)], name
        return None, None

    variant_rows, groups = [], defaultdict(list)
    for (dataset, k, loss, method, variant), values in sorted(table.items()):
        base, base_name = baseline_of(dataset, k, loss, method)
        sup, _ = baseline_of(dataset, k, loss, "sup")
        info = describe(variant, method)
        entry = {"dataset": dataset, "k": k, "loss": loss, "method": method, "variant": variant or "", **info}
        if info["group"] == "Sweep":
            point = sweep_point(variant, run_args.get((dataset, k, loss, method, base_name)))
            if point and point["tuned"] is None:
                # No baseline run here: any other variant of this combination that leaves the parameter alone.
                for (d2, k2, l2, m2, v2), config in run_args.items():
                    if (d2, k2, l2, m2) == (dataset, k, loss, method) and not (v2 or "").startswith(point["param"]):
                        fallback = sweep_point(variant, config)
                        if fallback and fallback["tuned"] is not None:
                            point["tuned"] = fallback["tuned"]
                            break
            if point:
                entry["sweep"] = point
        if quality.get((dataset, k, loss, method, variant)):
            seeds = quality[(dataset, k, loss, method, variant)]
            entry["pseudo_label_quality"] = {
                name: (float(np.mean([q[name] for q in seeds if q.get(name) is not None]))
                       if any(q.get(name) is not None for q in seeds) else None)
                for name in ("accuracy_first", "accuracy_last", "accepted_share_last", "pair_precision_last")}
            entry["pseudo_label_quality"]["seeds"] = len(seeds)
        for aggregation in ("mean", "concat"):
            if not values[aggregation]:
                continue
            entry[aggregation] = summary(values[aggregation])
            if base is not None and variant != base_name:
                entry[f"{aggregation}_vs_baseline"] = paired(values[aggregation], base[aggregation])
            if sup is not None and method != "sup":
                entry[f"{aggregation}_vs_supervised"] = paired(values[aggregation], sup[aggregation])
        variant_rows.append(entry)
        groups[(dataset, k, loss, method)].append(entry)

    two_by_two = []
    for (dataset, k, loss, method), entries in sorted(groups.items()):
        if method == "sup":
            continue
        sup_base, _ = baseline_of(dataset, k, loss, "sup")
        cells = {e["cell"]: e for e in entries if e["cell"]}
        if sup_base is None or not {"ssl_hp/with_data", "sup_hp/with_data", "ssl_hp/no_data"} <= set(cells):
            continue
        block = {"dataset": dataset, "k": k, "loss": loss, "method": method}
        for aggregation in ("mean", "concat"):
            ssl = table[(dataset, k, loss, method, cells["ssl_hp/with_data"]["variant"])][aggregation]
            sup_hp = table[(dataset, k, loss, method, "sup_all_shared")][aggregation]
            no_data = table[(dataset, k, loss, method, "pool_labeled")][aggregation]
            supervised = sup_base[aggregation]
            if not (ssl and sup_hp and no_data and supervised):
                continue
            data_ssl, data_sup = paired(ssl, no_data), paired(sup_hp, supervised)
            interaction = {}
            seeds = sorted(set(ssl) & set(no_data) & set(sup_hp) & set(supervised))
            if seeds:
                inter = np.array([(ssl[s] - no_data[s]) - (sup_hp[s] - supervised[s]) for s in seeds])
                interaction = {"n": len(seeds), "mean": float(inter.mean())}
                if len(seeds) > 1 and inter.std(ddof=1) > 0:
                    interaction["p"] = float(stats.ttest_1samp(inter, 0.0).pvalue)
            block[aggregation] = {
                "cells": {"sup_hp/no_data": summary(supervised), "sup_hp/with_data": summary(sup_hp),
                          "ssl_hp/no_data": summary(no_data), "ssl_hp/with_data": summary(ssl)},
                "total": paired(ssl, supervised),
                "data_effect_ssl_hp": data_ssl,
                "data_effect_sup_hp": data_sup,
                "hp_effect_no_data": paired(no_data, supervised),
                "hp_effect_with_data": paired(ssl, sup_hp),
                "interaction": interaction,
            }
        if "mean" in block:
            two_by_two.append(block)

    payload = {
        "generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "metric": args.metric,
        "metricLabel": {"map_at_r": "MAP@R", "p_at_1": "P@1", "rp": "R-Precision"}.get(args.metric, args.metric),
        "roots": [str(r) for r in roots],
        "labels": {"method": METHOD_LABEL, "loss": LOSS_LABEL, "dataset": DATASET_LABEL},
        "variants": variant_rows,
        "twoByTwo": two_by_two,
    }
    if args.stdout:
        real_stdout.write(json.dumps(payload, default=common._json_default))
        real_stdout.flush()
        return

    out = args.out / "ablations"
    flat = []
    for e in variant_rows:
        row = {k: e[k] for k in ("dataset", "k", "loss", "method", "variant", "label_de", "group", "cell",
                                  "uses_hidden_labels")}
        for aggregation in ("mean", "concat"):
            if aggregation in e:
                row[f"{aggregation}_value"] = e[aggregation]["mean"]
                row[f"{aggregation}_sd"] = e[aggregation].get("sd")
                row[f"{aggregation}_seeds"] = e[aggregation]["n"]
            for ref in ("baseline", "supervised"):
                d = e.get(f"{aggregation}_vs_{ref}") or {}
                row[f"{aggregation}_diff_{ref}"] = d.get("mean")
                row[f"{aggregation}_p_{ref}"] = d.get("p")
        flat.append(row)
    common.write_rows(out / "variants.csv", flat)
    common.write_rows(out / "two_by_two.csv", [
        {"dataset": b["dataset"], "k": b["k"], "loss": b["loss"], "method": b["method"],
         **{f"{name}": (b["mean"][name].get("mean") if name != "cells" else None) for name in
            ("total", "data_effect_ssl_hp", "data_effect_sup_hp", "hp_effect_no_data", "hp_effect_with_data",
             "interaction")},
         **{f"cell_{cell}": b["mean"]["cells"][cell]["mean"] for cell in b["mean"]["cells"]}}
        for b in two_by_two])

    print(f"\n{len(variant_rows)} variant rows, {len(two_by_two)} complete 2x2 tables ({args.metric}, fold mean)")
    for b in two_by_two:
        m = b["mean"]
        print(f"  {DATASET_LABEL.get(b['dataset'], b['dataset']):8s} k={b['k']:<3} {b['loss']:4s} {b['method']:7s} "
              f"total {m['total'].get('mean', float('nan')):+.2f}  data|SSL-HP {m['data_effect_ssl_hp'].get('mean', float('nan')):+.2f}  "
              f"data|sup-HP {m['data_effect_sup_hp'].get('mean', float('nan')):+.2f}  "
              f"HP|no data {m['hp_effect_no_data'].get('mean', float('nan')):+.2f}  (seeds {m['total'].get('n', 0)})")

    if args.app:
        target = args.app / "public" / "ablations.json"
        common.write_json(target, payload)
        print(f"wrote {target}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
