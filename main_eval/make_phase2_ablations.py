#!/usr/bin/env python3
"""Write Phase-2 ablation configs and a run manifest for the existing replay machinery.

Every file this writes is an ordinary ``--ablation_config`` (see
docs/ablation_config.md) plus a run manifest for ``training/run_scheduler.py``.
The launcher expands each run into ``<save_dir_root>/<combo>_<variant>/``, which
is exactly the layout ``seeded_runs.py`` pairs against the unmodified replay
(pass both roots via ``--seeded-roots``). Nothing is launched here.

Three kinds of ablation, all replaying a tuned Phase-1 study:

``sweep``     One-at-a-time sensitivity (plan A5). Each chosen parameter moves
              over a ladder while everything else keeps the tuned value.
              ``--ladder range`` spans the Phase-1 search space, ``--ladder
              local`` stays near the optimum (x1/4 … x4 for log parameters,
              ±10/25 % of the range otherwise).

``hp-path``   Supervised config -> SSL config (plans A6/B1). Replays the SSL
              study but sets one shared parameter (lr, classifier_lr,
              batch_sampler, loss.*, miner.*) at a time to the *supervised*
              study's best value, plus ``sup_all_shared`` that sets all of them:
              the "supervised hyperparameters + SSL" cell of the 2x2 table. The
              "SSL hyperparameters, no unlabeled data" cell is ``controls``'
              ``pool_labeled``.

``controls``  Mechanism controls (plan B): ``pool_labeled`` (unlabeled objective
              sees only labeled samples, same weight and updates per epoch),
              ``zero_weight`` (unlabeled pool kept, its loss weight 0; for LP
              no pseudo-labels at all), ``no_warmup``, for LP ``pl_once``, for
              LRML ``graph_once`` and for SLADE ``clusters_once``
              (pseudo-labels / graph / k-means clusters computed once after the
              warm-up, then frozen), for LP ``oracle_noise_<p>`` (LP's coverage
              with true labels, p % of them wrong) and for SLADE the same name
              for clusters that are the true classes, p % of the images moved
              to a wrong one (levels via ``--oracle-noise``), and for LP/LRML
              the graph controls ``graph_rewired`` (degree-preserving random
              rewiring) and the hidden-label oracles ``graph_oracle_pruned`` /
              ``graph_random_pruned``. Oracle variants are diagnostic only,
              never a valid SSL result. Every cell carries the Phase-2 cell
              settings, as ``2x2`` does.

``paper``     SLADE only: the ablations of Duan et al. (CVPR 2021, Sec. 4.5)
              rebuilt on the tuned study. Table 4 (``t4_pseudo_label``: ranking
              loss on pseudo-label pairs, no basis; ``t4_pseudo_label_basis``:
              + Basis; the replay itself is + Basis + Mining, and controls'
              ``zero_weight`` is the teacher), plan B5's pair-count control
              (``t4_matched_count``: pseudo-label pairs at Eq 7's count), Table 5
              (``t5_local_ce``, ``t5_global_ce``, ``t5_ce_only`` in place of the
              similarity distribution loss), Table 6 (``t6_cluster_ratio_<r>``,
              ``--cluster-ratios``), the paper's fixed settings where the study
              tuned its own (``paper_beta``, ``paper_unlabeled_batch``,
              ``paper_weights``), and Sec. 3.3.3's structure
              (``paper_no_basis_warmup``, ``paper_teacher_student``,
              ``paper_new_student_per_iteration``).

``external``  External unlabeled data, the pairing SLADE (Duan et al., CVPR 2021,
              Sec. 4.1) uses -- Cars-196 + CompCars filtered to 16,537 images /
              145 models, CUB + NABirds, In-Shop + Fashion200k
              (docs/slade_implementation_vs_paper.md, 2.3) -- at several shares of
              that pool (``--fractions``, like Semi-iNat's out-of-class fraction;
              nested, so 25 % holds the 10 % images). An SSL combo gets one
              ``external_<pct>`` variant per share (``--source split_and_external``
              adds the pool to the in-split one instead); a supervised combo gets
              the unmodified ``defaults`` cell at the same label budget, the
              reference an external variant has to beat. ``--k 99999`` labels every
              training image, SLADE's protocol, which leaves no in-split pool.
              ``--with-defaults`` also adds ``defaults`` to SSL combos (k-shot only).

``2x2``       The four cells of the HP x data table for every combo given: the
              supervised ``defaults`` and, per SSL combo, ``defaults`` (tuned SSL),
              ``sup_all_shared`` (hp-path: supervised values for every shared
              parameter) and ``pool_labeled`` (controls: no genuinely unlabeled
              data). Every cell carries the Phase-2 cell settings (GPU eval
              residency, LRML uncapped, SLADE 600 epochs / patience 20). Each cell
              is its own manifest run (``ablation_variants``), disabled when its
              result already exists below ``logs/<save_dir_root>``; one combined
              manifest plus one per method (``<name>_<method>.json``).
              ``--label-budget 0.1`` replays class-subset studies (In-Shop 10 %).

``baseline``  Unmodified replays (``--combo sup_ms iscen_ms ...``) written to
              ``<save_dir_root>/<combo>/``: the 2x2's two corner cells, run on
              the same code and machine as the variants they are paired with.

``--variants`` keeps only the named variants, ``--combo`` takes several combos
(one manifest run each), and ``--append`` merges into an existing manifest, so
one manifest can mix kinds -- e.g. the two new cells of the 2x2 table::

    python main_eval/make_phase2_ablations.py hp-path --dataset cars --combo iscen_ms \\
        --variants sup_all_shared --name cars_ms_2x2
    python main_eval/make_phase2_ablations.py controls --dataset cars --combo iscen_ms lrml_ms \\
        --variants pool_labeled --name cars_ms_2x2 --append

Ablation configs go to ``configs/ablations/phase2/<name>/<combo>.json``.

``--diagnostics`` switches pseudo-label diagnostics on in every variant and adds
a ``diag`` variant that changes nothing else, so the tuned baseline also gets
the pseudo-label accuracy the Phase-1 runs did not log. ``--track`` adds epoch
tracking (``--track-artifacts``, default graph and embeddings; ``geometry`` for B6) so the embedding space can be inspected over
training with evaluate/report.py and react_trial_visualizer. Every replay keeps
its fold ``test_embeddings.npz`` for the geometry / diversity / per-query scripts.

Examples::

    python main_eval/make_phase2_ablations.py sweep --dataset cars --combo iscen_ms \\
        --params ssl_config.method_params.n_neighbors ssl_config.confidence_threshold lr
    python main_eval/make_phase2_ablations.py hp-path --dataset cars --combo iscen_ms
    python main_eval/make_phase2_ablations.py controls --dataset cars --combo lrml_ms --diagnostics --track
    python main_eval/make_phase2_ablations.py paper --dataset cars --combo slade_cir --seeds 2
    python main_eval/make_phase2_ablations.py sweep --study /mnt/f/server_logs/final/cub/.../hpo/in-shop --params lr --k 4 16

Then::

    python -m training.run_scheduler --manifest configs/run_manifests/phase2/<name>.json --dry-run
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from main_eval import common  # noqa: E402

REPO = common.REPO
TEMPLATE = REPO / "configs" / "final_manifests" / "cars_all.json"
PHASE1_DATASET_DIR = {"cars": "cars", "cub": "cub", "in_shop": "in_shop", "in_shop_10": "in_shop_10"}
NOT_ABLATABLE = {"labeled_batch_fraction"}  # exists only inside a search space (docs/ablation_config.md)

# The unlabeled term at weight 0 while the pool, sampler budget and everything else stay. The target
# ratio must be positive, so the calibration is switched off with all its companions (each would
# otherwise be rejected as orphaned), and the max-ratio cap goes too: at weight 0 it has nothing to cap.
_TARGET_RATIO_KEYS = ["regularizer_target_ratio", *(f"regularizer_target_ratio_{s}" for s in (
    "batches", "recalibration_interval_samples", "update_interval_samples", "probe_memory", "ema_alpha",
    "aggregation", "statistic", "statistic_anchor", "diagnostics")), "regularizer_max_ratio"]
ZERO_WEIGHT = {"ssl_config.method_params.regularizer_weight": 0.0,
               **{f"ssl_config.method_params.{key}": None for key in _TARGET_RATIO_KEYS}}
# Validated against the tuned Cars SSL configs. LP has no such weight: its pseudo-labels enter the batch.
ZERO_WEIGHT_METHODS = {"lrml", "hoffer", "slade"}

# The external unlabeled data SLADE pairs with each benchmark (Sec. 4.1). Paths are relative to the
# repository, beside data/Cars196. The first run that finds CompCars or Fashion200k missing
# downloads it (utils/external_pool_download.py); NABirds must be placed by hand.
EXTERNAL_POOLS = {
    "cars": {"external_unlabeled_dir": "data/CompCars", "external_unlabeled_filter": "compcars_slade_paper",
             "compcars_strict_paper_counts": True},
    "cub": {"external_unlabeled_dir": "data/NABirds", "external_unlabeled_filter": "nabirds"},
    "in_shop": {"external_unlabeled_dir": "data/Fashion200k"},
}
EXTERNAL_POOLS["in_shop_10"] = EXTERNAL_POOLS["in_shop"]
# Default --fractions. Fashion200k stops at a quarter: label propagation holds dense float64
# pool x class matrices, and ~20k labeled In-Shop images + 25 % of ~200k Fashion200k images against
# ~3k fold classes is already ~1.7 GB per matrix.
EXTERNAL_FRACTIONS = {"cars": [0.1, 0.25, 0.5, 1.0], "cub": [0.1, 0.25, 0.5, 1.0],
                      "in_shop": [0.05, 0.1, 0.25], "in_shop_10": [0.05, 0.1, 0.25]}
# What every Phase-2 cell of a method sets besides its variant, so a new cell pairs with the existing
# <combo>_defaults baseline: LRML runs uncapped, SLADE for 600 epochs / patience 20.
CELL_DEFAULTS = {"evaluation_embedding_residency": "gpu"}
METHOD_CELL_DEFAULTS = {
    "lrml": {"ssl_config.method_params.regularizer_max_ratio": None},
    "slade": {"epochs": 600, "patience": 20},
}

# SLADE's own settings, below ssl_config.method_params.
SLADE_PARAMS = "ssl_config.method_params.regularizer_params."
# Table 6 of the paper tries k = 100 ... 500 for NABirds' 400 species, 0.25 to 1.25 times the true class
# count. The unlabeled pools here hold the labeled classes, so cluster_ratio 1 is that count; the ladder
# covers the paper's side and the over-clustering the Phase-1 searches chose (Cars: 2).
PAPER_CLUSTER_RATIOS = [0.25, 0.5, 1.0, 2.0, 4.0]
# Sec. 4.3 and 3.3.3: batch 32 for both streams, beta = 0.99, lambda1 = 1, lambda2 = 0.25.
PAPER_SETTINGS = {"beta": 0.99, "unlabeled_batch_size": 32, "lambda1": 1.0, "lambda2": 0.25}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="kind", required=True)
    for name in ("sweep", "hp-path", "controls", "paper", "external", "2x2", "baseline"):
        p = sub.add_parser(name)
        p.add_argument("--study", type=Path, nargs="+",
                       help="TPE study dir(s) (with trials.jsonl); or give --dataset/--combo")
        p.add_argument("--dataset", help="cars / cub / in_shop")
        p.add_argument("--combo", nargs="+", help="e.g. iscen_ms, lrml_nt, hoffer_cir; several give one run each")
        p.add_argument("--phase1-root", type=Path, default=common.PHASE1_ROOT)
        p.add_argument("--seeds", type=int, nargs="+", default=[2, 8, 9, 10, 11],
                       help="replay seeds; keep the Phase-1 seeds so variants pair with the existing runs")
        p.add_argument("--k", type=int, nargs="+", default=[16])
        p.add_argument("--save-dir-root", default=None, help="default: final/phase_2/<dataset>")
        p.add_argument("--study-prefix", default=None, metavar="OLD=NEW",
                       help="rewrite the study path in the manifest, e.g. /mnt/f/server_logs/final=final/phase_1")
        p.add_argument("--template", type=Path, default=TEMPLATE, help="manifest whose scheduler settings are copied")
        p.add_argument("--diagnostics", action="store_true", help="pseudo-label diagnostics on + a 'diag' variant")
        p.add_argument("--track", action="store_true", help="epoch tracking (see --track-artifacts)")
        p.add_argument("--track-artifacts", nargs="+", default=["graph", "embeddings"],
                       help="what --track records: graph, embeddings, test, geometry (Roth/Wang-Isola geometry "
                            "per split plus the backbone reference)")
        p.add_argument("--track-interval", type=int, default=5)
        p.add_argument("--name", default=None, help="file stem of the written configs and manifest")
        p.add_argument("--variants", nargs="+", default=None, metavar="NAME",
                       help="keep only these variants; a name a combo does not generate is skipped for that combo")
        p.add_argument("--append", action="store_true",
                       help="merge into the existing manifest --name instead of overwriting it, so one manifest "
                            "can hold variants of several kinds (e.g. hp-path's sup_all_shared + controls' "
                            "pool_labeled)")
        if name == "sweep":
            p.add_argument("--params", nargs="+", required=True, help="full parameter names as in the search space")
            p.add_argument("--points", type=int, default=5)
            p.add_argument("--ladder", choices=("range", "local"), default="range")
        if name == "controls":
            p.add_argument("--oracle-noise", type=int, nargs="+", default=[0, 10, 20, 40, 60], metavar="PERCENT",
                           help="LP oracle pseudo-label / SLADE oracle cluster noise levels, one "
                                "oracle_noise_<p> variant each")
        if name == "paper":
            p.add_argument("--cluster-ratios", type=float, nargs="+", default=PAPER_CLUSTER_RATIOS,
                           help="Table 6 ladder of cluster_ratio (k over the labeled class count); the tuned "
                                "value is skipped")
        if name in ("hp-path", "2x2"):
            p.add_argument("--sup-study", type=Path, help="supervised study; default: sup_<loss> of the same dataset "
                                                          "(2x2: the sup study of the same loss among --study)")
        if name == "2x2":
            p.add_argument("--label-budget", type=float, default=None,
                           help="class-subset label budget (e.g. 0.1) instead of --k; for In-Shop 10 %% studies")
        if name == "external":
            p.add_argument("--external-dir", default=None, help="default: the dataset's SLADE pool, e.g. data/CompCars")
            p.add_argument("--fractions", type=float, nargs="+", default=None,
                           help="shares of the external pool, one variant each (default per dataset)")
            p.add_argument("--source", choices=("external", "split_and_external"), default="external",
                           help="replace the in-split unlabeled pool, or add to it")
            p.add_argument("--with-defaults", action="store_true",
                           help="also add the unmodified 'defaults' cell to SSL combos")
    return parser.parse_args()


# ------------------------------------------------------------------ studies


def resolve_study(root: Path, dataset: str, combo: str) -> Path:
    prefix, loss = combo.split("_")[:2]
    folder = f"supervised_{loss}" if prefix == "sup" else combo
    ds = PHASE1_DATASET_DIR.get(dataset, dataset)
    candidates = sorted(p.parent for p in (root / ds).glob(f"*/{folder}/*/hpo/*/trials.jsonl"))
    if not candidates:
        raise SystemExit(f"no Phase-1 study for {dataset}/{combo} below {root / ds}")
    if len(candidates) > 1:
        sizes = {c: len(common.read_jsonl(c / "trials.jsonl")) for c in candidates}
        chosen = max(candidates, key=sizes.get)
        print(f"[warn] {len(candidates)} studies for {combo}; using the largest: {chosen}", file=sys.stderr)
        return chosen
    return candidates[0]


def studies_from_args(args):
    if args.study:
        return [common.load_study(path.parent, path) for path in args.study]
    if not (args.dataset and args.combo):
        raise SystemExit("give --study, or --dataset and --combo")
    paths = [resolve_study(args.phase1_root, args.dataset, combo) for combo in args.combo]
    return [common.load_study(path.parent, path) for path in paths]


def study_dataset(study) -> str:
    return common.DATASET_SHORT.get(study.dataset, study.dataset)


def combo_name(study) -> str:
    prefix = {"lp": "iscen", "sup": "sup"}.get(study.method, study.method)
    return f"{prefix}_{common.LOSS_SHORT.get(study.loss, study.loss)}"


def best_params(study) -> dict:
    best = study.best()
    if best is None:
        raise SystemExit(f"{study.path} has no COMPLETE trial")
    print(f"  {study.label}: best trial {best['number']} (MAP@R {best['value'] * 100:.2f} %)")
    return dict(best.get("params") or {})


# ------------------------------------------------------------------- values


def clean(value):
    if isinstance(value, (bool, str)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    value = float(value)
    return float(f"{value:.4g}")


def ladder(space: dict, best, points: int, mode: str) -> list:
    kind = space.get("type")
    if kind == "categorical":
        return [c for c in space["choices"] if c != best]
    low, high = float(space["low"]), float(space["high"])
    is_int = kind == "int"
    if mode == "range":
        values = np.geomspace(low, high, points) if space.get("log") else np.linspace(low, high, points)
    elif space.get("log"):
        values = [best * f for f in (0.25, 0.5, 2.0, 4.0)]
    else:
        span = high - low
        values = [best + d * span for d in (-0.25, -0.1, 0.1, 0.25)]
    step = space.get("step")
    out = []
    for value in values:
        value = min(max(value, low), high)
        if is_int:
            value = int(round(value if not step else low + round((value - low) / step) * step))
        value = clean(value)
        if value not in out and value != clean(best):
            out.append(value)
    return out


# ------------------------------------------------------------------ builders


def diagnostics_overrides(study, enabled):
    if not enabled or study.method == "sup":
        return {}
    return {"ssl_config.pseudo_label_diagnostics_mode": "save"} if study.method == "lp" else {}


def build_sweep(study, args):
    params = best_params(study)
    variants = {}
    for name in args.params:
        if name in NOT_ABLATABLE:
            print(f"[skip] {name} cannot be ablated (search-space-only parameter)", file=sys.stderr)
            continue
        space = study.spaces.get(name)
        if space is None:
            raise SystemExit(f"{name} is not in the study's search space; known: {sorted(study.spaces)}")
        best = params.get(name)
        values = ladder(space, best, args.points, args.ladder)
        for value in values:
            label = f"{common.short_param(name).replace(':', '_')}_{str(value).replace('.', 'p').replace('-', 'm').replace(':', '-')}"
            variants[label] = {name: value}
        print(f"  {name}: tuned {clean(best)} -> {values}")
    description = (f"One-at-a-time {args.ladder} sweep around the tuned {study.label} configuration "
                   f"(best trial {study.best()['number']}); every other parameter keeps its tuned value.")
    return variants, description


def build_hp_path(study, args):
    if study.method == "sup":
        raise SystemExit("hp-path replays an SSL study; give the SSL combo")
    sup_path = args.sup_study or resolve_study(args.phase1_root, study_dataset(study),
                                               f"sup_{common.LOSS_SHORT.get(study.loss, study.loss)}")
    sup = common.load_study(Path(sup_path).parent, Path(sup_path))
    ssl_params, sup_params = best_params(study), best_params(sup)
    shared = sorted(
        k for k in set(ssl_params) & set(sup_params)
        if not k.startswith(("ssl_config.", "ssl.")) and k not in NOT_ABLATABLE
    )
    variants, all_shared = {}, {}
    for name in shared:
        if clean(ssl_params[name]) == clean(sup_params[name]):
            continue
        value = clean(sup_params[name])
        variants[f"sup_{common.short_param(name).replace(':', '_')}"] = {name: value}
        all_shared[name] = value
        print(f"  {name}: SSL {clean(ssl_params[name])} -> supervised {value}")
    if all_shared:
        variants["sup_all_shared"] = dict(all_shared)
    only_ssl = sorted(k for k in set(ssl_params) - set(sup_params) if k not in NOT_ABLATABLE)
    print(f"  kept at SSL values (no supervised counterpart): {only_ssl}")
    description = (f"Ablation path from the supervised to the SSL configuration of {study.label}: each variant sets "
                   f"shared parameters to the supervised study's best trial ({sup.path}).")
    return variants, description


def build_controls(study, args):
    params = best_params(study)
    variants = {"pool_labeled": {"unlabeled_source": "labeled"}}
    if study.method in ZERO_WEIGHT_METHODS:
        variants["zero_weight"] = dict(ZERO_WEIGHT)
    elif study.method == "lp":
        # LP's pseudo-labels enter the batch rather than a weighted term: weight zero is no pseudo-labels.
        variants["zero_weight"] = {"ssl_config.pseudo_label_source": "none"}
    else:
        print(f"  [skip] zero_weight: {study.method} has no unlabeled loss weight to zero", file=sys.stderr)
    if int(params.get("ssl_config.warmup_epochs", 0) or 0) > 0:
        variants["no_warmup"] = {"ssl_config.warmup_epochs": 0}
    if study.method == "lp":
        variants["pl_once"] = {"ssl_config.update_mode": "once"}
    if study.method in ("lp", "slade"):
        # LP: its coverage with hidden true labels. SLADE: the hidden classes as clusters in place of
        # k-means, so k becomes the pool's class count. A seeded, nested share of them made wrong.
        for percent in args.oracle_noise:
            variants[f"oracle_noise_{percent}"] = {"ssl_config.pseudo_label_source": "oracle",
                                                   "ssl_config.pseudo_label_oracle_noise": percent / 100}
    if study.method == "lrml":
        # LRML builds its graph when the SSL phase starts, i.e. after the warm-up.
        variants["graph_once"] = {"ssl_config.update_mode": "once"}
    if study.method == "slade":
        # SLADE clusters when the SSL phase starts, too; the teacher's clusters then never change.
        variants["clusters_once"] = {"ssl_config.update_mode": "once"}
    if study.method in ("lp", "lrml"):
        variants["graph_oracle_pruned"] = {"ssl_config.graph_edge_ablation": "oracle_pruned"}
        variants["graph_random_pruned"] = {"ssl_config.graph_edge_ablation": "random_pruned"}
        # Same pool, edge count, weights and node degrees; which nodes an edge joins is random.
        variants["graph_rewired"] = {"ssl_config.graph_edge_ablation": "degree_rewired"}
    cell = {**CELL_DEFAULTS, **METHOD_CELL_DEFAULTS.get(study.method, {})}
    variants = {name: {**change, **cell} for name, change in variants.items()}
    description = (f"Mechanism controls for {study.label}. graph_*_pruned and oracle_noise_* use hidden labels and "
                   "are diagnostic only; never report them as SSL results or use them for selection.")
    return variants, description


def build_paper(study, args):
    """The ablations of the SLADE paper (Duan et al., CVPR 2021, Sec. 4.5) on a tuned SLADE study."""
    if study.method != "slade":
        raise SystemExit(f"paper rebuilds the SLADE paper's ablations; {combo_name(study)} is not a SLADE combo")
    params = best_params(study)
    p = SLADE_PARAMS
    variants = {
        # Table 4. "Student (pseudo label)": the ranking loss on every pseudo-label pair, nothing trains the
        # basis. "+ Basis": the basis is trained and shapes the embedding through L_Basis but selects no pairs.
        "t4_pseudo_label": {f"{p}pair_source": "pseudo_label", f"{p}lambda_basis": 0.0,
                            f"{p}basis_target_ratio": None},
        "t4_pseudo_label_basis": {f"{p}pair_source": "pseudo_label"},
        # Plan B5: Eq 7 keeps fewer and easier pairs than t4_pseudo_label_basis. The same number of
        # pseudo-label pairs drawn at random separates "fewer pairs" from "better pairs".
        "t4_matched_count": {f"{p}pair_source": "pseudo_label_matched"},
        # Table 5: the unlabeled half of L_Basis.
        "t5_local_ce": {f"{p}basis_unlabeled_loss": "local_ce"},
        "t5_global_ce": {f"{p}basis_unlabeled_loss": "global_ce"},
        "t5_ce_only": {f"{p}basis_unlabeled_loss": "none"},
    }
    # Table 6: the number of clusters. An absolute num_clusters would win over the ratio.
    tuned_ratio = params.get(f"{p}cluster_ratio")
    for ratio in args.cluster_ratios:
        if tuned_ratio is not None and float(ratio) == float(tuned_ratio):
            continue
        variants[f"t6_cluster_ratio_{f'{ratio:g}'.replace('.', 'p')}"] = {
            f"{p}cluster_ratio": float(ratio), f"{p}num_clusters": None}
    # The paper's fixed settings (Sec. 4.3), one at a time, where the study searched its own.
    if clean(params.get(f"{p}beta")) != PAPER_SETTINGS["beta"]:
        variants["paper_beta"] = {f"{p}beta": PAPER_SETTINGS["beta"]}
    if params.get(f"{p}unlabeled_batch_size") != PAPER_SETTINGS["unlabeled_batch_size"]:
        variants["paper_unlabeled_batch"] = {f"{p}unlabeled_batch_size": PAPER_SETTINGS["unlabeled_batch_size"],
                                             f"{p}unlabeled_ratio": None}
    # lambda1 and lambda2 fixed, as the paper sets them, instead of calibrated to gradient ratios.
    variants["paper_weights"] = {**ZERO_WEIGHT,
                                 "ssl_config.method_params.regularizer_weight": PAPER_SETTINGS["lambda1"],
                                 f"{p}lambda_basis": PAPER_SETTINGS["lambda2"], f"{p}basis_target_ratio": None}
    # Sec. 3.3.3: "We first train the basis vectors for a few iterations".
    variants["paper_no_basis_warmup"] = {f"{p}basis_warmup_epochs": 0.0, f"{p}basis_warmup_steps": None}
    # Sec. 3.3.3 and 4.3: a trained teacher clusters once, a fresh student trains against those clusters.
    if int(params.get("ssl_config.warmup_epochs", 0) or 0) > 0:
        variants["paper_teacher_student"] = {f"{p}within_fold_teacher_student": True,
                                             "ssl_config.update_mode": "once"}
    else:
        print("  [skip] paper_teacher_student: the teacher needs warmup_epochs > 0", file=sys.stderr)
    # Every re-clustering starts a new student: fresh basis, fresh Gaussians, basis warm-up again.
    variants["paper_new_student_per_iteration"] = {f"{p}reset_basis_on_refresh": True,
                                                   f"{p}reset_basis_warmup_on_refresh": True}
    cell = {**CELL_DEFAULTS, **METHOD_CELL_DEFAULTS.get(study.method, {})}
    variants = {name: {**change, **cell} for name, change in variants.items()}
    description = (f"SLADE paper ablations (Duan et al., CVPR 2021, Tables 4-6, Sec. 3.3.3 and 4.3) on the tuned "
                   f"{study.label} configuration (best trial {study.best()['number']}). The replay itself is Table "
                   "4's '+ Basis + Mining', controls' zero_weight its teacher. t4_matched_count is plan B5's "
                   "pair-count control.")
    return variants, description


def fraction_token(fraction: float) -> str:
    return f"{fraction * 100:g}".replace(".", "p")


def build_external(study, args):
    dataset = study_dataset(study)
    if dataset not in EXTERNAL_POOLS:
        raise SystemExit(f"no external unlabeled pool is set up for {dataset}; known: {sorted(EXTERNAL_POOLS)}")
    best_params(study)
    cell = {**CELL_DEFAULTS, **METHOD_CELL_DEFAULTS.get(study.method, {})}
    if study.method == "sup":
        # The supervised run at the same label budget is what an external pool has to improve on.
        return {"defaults": dict(cell)}, f"Unmodified supervised replay of {study.label}: the external-data reference."
    pool = dict(EXTERNAL_POOLS[dataset])
    if args.external_dir:
        pool["external_unlabeled_dir"] = args.external_dir
    fractions = args.fractions or EXTERNAL_FRACTIONS[dataset]
    if any(not 0.0 < fraction <= 1.0 for fraction in fractions):
        raise SystemExit(f"--fractions must lie in (0, 1]: {fractions}")
    variants = {}
    for fraction in fractions:
        share = {} if fraction == 1.0 else {"external_unlabeled_fraction": float(fraction)}
        variants[f"{args.source}_{fraction_token(fraction)}"] = {
            "unlabeled_source": args.source, **pool, **share, **cell}
    if args.with_defaults:
        variants["defaults"] = dict(cell)
    description = (f"External unlabeled data for {study.label}: {pool['external_unlabeled_dir']} "
                   f"({pool.get('external_unlabeled_filter', 'unfiltered')}) at "
                   f"{', '.join(fraction_token(f) + ' %' for f in fractions)} of the pool, "
                   + ("replacing the in-split pool." if args.source == "external" else "added to the in-split pool."))
    return variants, description


def build_2x2(study, args):
    cell = {**CELL_DEFAULTS, **METHOD_CELL_DEFAULTS.get(study.method, {})}
    if study.method == "sup":
        best_params(study)
        return {"defaults": dict(cell)}, f"Supervised corner of the 2x2 for {study.label}."
    sup_path = args.sup_study or getattr(args, "sup_by_loss", {}).get(study.loss)
    hp_args = argparse.Namespace(**{**vars(args), "sup_study": sup_path})
    path_variants, _ = build_hp_path(study, hp_args)
    variants = {"defaults": dict(cell)}
    if "sup_all_shared" in path_variants:
        variants["sup_all_shared"] = {**path_variants["sup_all_shared"], **cell}
    else:
        print(f"  [skip] {combo_name(study)}: SSL and supervised best trials share every value; "
              "sup_all_shared would be the SSL baseline", file=sys.stderr)
    variants["pool_labeled"] = {"unlabeled_source": "labeled", **cell}
    description = (f"2x2 of {study.label}: tuned SSL (defaults), supervised hyperparameters + SSL "
                   f"(sup_all_shared, from {sup_path or 'the sup study of the same loss'}), SSL hyperparameters "
                   "without genuinely unlabeled data (pool_labeled).")
    return variants, description


def build_baseline(study, args):
    best_params(study)
    # No ablation at all: a plain replay, written to <save_dir_root>/<combo>/, the layout
    # seeded_runs.py pairs the <combo>_<variant>/ directories against.
    return None, f"Unmodified replay of {study.label}."


# ------------------------------------------------------------------- writing


def with_diagnostics(study, variants, args):
    extra = diagnostics_overrides(study, args.diagnostics)
    if not extra:
        return variants
    variants = {name: {**change, **extra} for name, change in variants.items()}
    variants["diag"] = dict(extra)
    return variants


def select_variants(entries, wanted):
    """Keep only the named variants; a name is an error only if no combo generates it."""
    if not wanted or any(variants is None for _, variants, _ in entries):
        return entries
    known = {name for _, variants, _ in entries for name in variants}
    unknown = sorted(set(wanted) - known)
    if unknown:
        raise SystemExit(f"--variants {unknown} are generated by no combo; available: {sorted(known)}")
    kept = []
    for study, variants, description in entries:
        chosen = {name: change for name, change in variants.items() if name in wanted}
        skipped = sorted(set(wanted) - set(chosen))
        if skipped:
            print(f"  [skip] {combo_name(study)} does not generate {skipped}", file=sys.stderr)
        if chosen:
            kept.append((study, chosen, description))
    return kept


def merge_variants(path: Path, variants: dict, description: str, name: str) -> dict:
    """Add variants to an existing ablation config; a name may only repeat with identical changes."""
    if not path.exists():
        return {"name": name, "description": description, "variants": variants}
    config = common.read_json(path)
    merged = dict(config.get("variants") or {})
    for variant, change in variants.items():
        if variant in merged and merged[variant] != change:
            raise SystemExit(f"{path}: variant {variant!r} already exists with different changes "
                             f"({merged[variant]} vs {change}); drop --append or pick another --name")
        merged[variant] = change
    old = config.get("description") or ""
    config["description"] = old if description in old else f"{old} | {description}".strip(" |")
    config["variants"] = merged
    return config


def write_outputs(entries, args):
    if not entries:
        raise SystemExit("nothing to ablate")
    datasets = sorted({study_dataset(study) for study, _, _ in entries})
    if len(datasets) > 1:
        raise SystemExit(f"one manifest per dataset; got {datasets}")
    dataset = datasets[0]
    kind = args.kind.replace("-", "_")
    if args.name:
        name = args.name
    elif args.append:
        raise SystemExit("--append needs --name: the manifest to merge into")
    elif len(entries) == 1:
        name = f"{dataset}_{combo_name(entries[0][0])}_{kind}"
    else:
        name = f"{dataset}_{kind}"
    manifest_path = REPO / "configs" / "run_manifests" / "phase2" / f"{name}.json"
    save_root = args.save_dir_root or f"final/phase_2/{dataset}"

    if args.append and manifest_path.exists():
        manifest = common.read_json(manifest_path)
        replay = manifest.setdefault("study_replay", {})
        # Seeds and budgets are manifest-wide; a mismatch would silently re-scope the runs already in it.
        for key, value in (("comparison_seeds", list(args.seeds)), ("k_shot_grid", list(args.k))):
            if replay.get(key) != value:
                raise SystemExit(f"{manifest_path.name} has {key}={replay.get(key)}, this call {value}; "
                                 f"pass the same --seeds/--k or write a separate manifest")
        if manifest.get("save_dir_root") != save_root:
            raise SystemExit(f"{manifest_path.name} writes to {manifest.get('save_dir_root')}, this call to {save_root}")
        print(f"  appending to {manifest_path.relative_to(REPO)}")
    else:
        template = common.read_json(args.template)
        manifest = {k: copy.deepcopy(v) for k, v in template.items() if k != "runs"}
        manifest["save_dir_root"] = save_root
        manifest["scheduler_log_dir"] = f"{save_root}/_scheduler/{name}"
        manifest["runs"] = []
        replay = manifest.setdefault("study_replay", {})
        replay["comparison_seeds"] = list(args.seeds)
        replay["k_shot_grid"] = list(args.k)
        # Keep every fold's test_embeddings.npz: geometry, fold diversity and per-query gains read them.
        replay["fold_test_embedding_storage"] = "file"
        # Off by default: true / an interval > 0 keeps the selected head / a head every N epochs.
        replay.setdefault("save_head_state", False)
        replay.setdefault("save_epoch_head_interval", 0)
        replay.setdefault("epoch_head_dtype", "bfloat16")
    if args.track:
        # Manifest-wide: every run in this manifest is tracked, not only this call's.
        # "test" tracking additionally needs --evaluate_test; graph + embeddings are enough to look at the space.
        replay["track_epoch_artifacts"] = list(args.track_artifacts)
        replay["track_epoch_interval"] = args.track_interval
        replay["track_epoch_layout"] = "tsne"

    # A plain replay and an ablation run of the same combo may share a name: the
    # scheduler checks uniqueness after expanding the ablation into <combo>_<variant>.
    runs = {(run["name"], "ablation_config" in run): run for run in manifest["runs"]}
    for study, variants, description in entries:
        combo = combo_name(study)
        study_dir = str(study.path)
        if args.study_prefix:
            old, new = args.study_prefix.split("=", 1)
            study_dir = study_dir.replace(old, new, 1)
        if variants is None:
            run = runs.setdefault((combo, False), {"name": combo, "study_dir": study_dir})
            if run["study_dir"] != study_dir:
                raise SystemExit(f"{manifest_path.name}: replay {combo!r} is of {run['study_dir']}, not {study_dir}")
            continue
        run = runs.get((combo, True))
        if run is not None and run["study_dir"] != study_dir:
            raise SystemExit(f"{manifest_path.name}: run {combo!r} replays {run['study_dir']}, not {study_dir}")
        ablation_path = (REPO / run["ablation_config"]) if run is not None else (
            REPO / "configs" / "ablations" / "phase2" / name / f"{combo}.json")
        config = (merge_variants(ablation_path, variants, description, f"{name}_{combo}") if args.append
                  else {"name": f"{name}_{combo}", "description": description, "variants": variants})
        common.write_json(ablation_path, config)
        if run is None:
            runs[(combo, True)] = {"name": combo, "study_dir": study_dir,
                                   "ablation_config": str(ablation_path.relative_to(REPO))}
    manifest["runs"] = list(runs.values())
    common.write_json(manifest_path, manifest)

    seeds, budgets = replay["comparison_seeds"], replay["k_shot_grid"]
    total = 0
    print()
    for run in manifest["runs"]:
        if "ablation_config" not in run:
            total += 1
            print(f"  {run['name']:<12} unmodified replay")
            continue
        names = list(common.read_json(REPO / run["ablation_config"])["variants"])
        total += len(names)
        print(f"  {run['name']:<12} {len(names)} variants: {', '.join(names)}")
        print(f"               {run['ablation_config']}")
    print(f"{total} variants x {len(seeds)} seeds x {len(budgets)} budgets = {total * len(seeds) * len(budgets)} CV runs"
          + ("  (epoch tracking on)" if replay.get("track_epoch_artifacts") else ""))
    print(f"  manifest: {manifest_path.relative_to(REPO)}")
    print(f"  children: logs/{save_root}/<combo>_<variant>/")
    print(f"  check:    python -m training.run_scheduler --manifest {manifest_path.relative_to(REPO)} --dry-run")
    print(f"  evaluate: python main_eval/seeded_runs.py --include-variants --seeded-roots "
          f"{' '.join(str(r) for r in common.SEEDED_ROOTS)} logs/{save_root}")


def budget_dir(args, seed: int) -> str:
    if args.label_budget is not None:
        return f"class_subset_label_{f'{args.label_budget:g}'.replace('.', 'p')}_comparison_seed_{seed}"
    return f"class_subset_k_shot_label_1_k_{args.k[0]}_comparison_seed_{seed}"


def cell_complete(save_root: str, child: str, args) -> bool:
    """A seed-2-style replay of this cell already reported a test MAP@R below logs/<save_root>/<child>."""
    summary = REPO / "logs" / save_root / child / "study_replay_grid" / "grid_summary.json"
    if not summary.is_file():
        return False
    wanted = {budget_dir(args, seed) for seed in args.seeds}
    found = set()
    for run in common.read_json(summary).get("runs", []):
        value = run.get("test_mean_average_precision_at_r")
        if value in (None, ""):
            continue
        found.update(part for part in Path(str(run.get("log_dir", ""))).parts if part in wanted)
    return wanted <= found


def write_2x2_outputs(entries, args):
    if len(args.k) != 1 and args.label_budget is None:
        raise SystemExit("2x2 replays one label budget; give one --k or --label-budget")
    name = args.name or f"{study_dataset(entries[0][0])}_2x2"
    save_root = args.save_dir_root or f"final/phase_2/{study_dataset(entries[0][0])}"
    template = common.read_json(args.template)
    base = {k: copy.deepcopy(v) for k, v in template.items() if k != "runs"}
    replay = base.setdefault("study_replay", {})
    replay["comparison_seeds"] = list(args.seeds)
    replay["fold_test_embedding_storage"] = "file"
    if args.label_budget is not None:
        replay.pop("k_shot_grid", None)
        replay["ssl_label_sampling_modes"] = ["class_subset"]
        replay["label_budget_grid"] = [args.label_budget]
    else:
        replay.pop("label_budget_grid", None)
        replay.pop("ssl_label_sampling_modes", None)
        replay["k_shot_grid"] = list(args.k)
    for key in ("track_epoch_artifacts", "track_epoch_interval", "track_epoch_layout"):
        replay.pop(key, None)
    base["save_dir_root"] = save_root

    runs, done = [], []
    for study, variants, description in entries:
        combo = combo_name(study)
        ablation_path = REPO / "configs" / "ablations" / "phase2" / name / f"{combo}.json"
        common.write_json(ablation_path, {"name": f"{name}_{combo}", "description": description, "variants": variants})
        for variant in variants:
            children = [f"{combo}_{variant}"] + ([combo] if variant == "defaults" else [])
            complete = any(cell_complete(save_root, child, args) for child in children)
            if complete:
                done.append(f"{combo}_{variant}")
            runs.append({"name": combo, "study_dir": str(study.path),
                         "ablation_config": str(ablation_path.relative_to(REPO)),
                         "ablation_variants": [variant], "enabled": not complete})

    def write_manifest(path, stem, selected):
        manifest = copy.deepcopy(base)
        manifest["scheduler_log_dir"] = f"{save_root}/_scheduler/{stem}"
        manifest["runs"] = selected
        common.write_json(path, manifest)
        enabled = sum(run["enabled"] for run in selected)
        print(f"  {path.relative_to(REPO)}: {len(selected)} cells, {enabled} to run")

    manifests = REPO / "configs" / "run_manifests" / "phase2"
    print()
    write_manifest(manifests / f"{name}.json", name, runs)
    for method in dict.fromkeys(run["name"].split("_")[0] for run in runs):
        selected = [run for run in runs if run["name"].split("_")[0] == method]
        write_manifest(manifests / f"{name}_{method}.json", f"{name}_{method}", selected)
    print(f"  already complete ({len(done)}): {', '.join(done) or '-'}")
    print(f"  children: logs/{save_root}/<combo>_<variant>/")


def main():
    args = parse_args()
    builders = {"sweep": build_sweep, "hp-path": build_hp_path, "controls": build_controls,
                "paper": build_paper, "external": build_external, "2x2": build_2x2, "baseline": build_baseline}
    entries = []
    studies = studies_from_args(args)
    args.sup_by_loss = {study.loss: study.path for study in studies if study.method == "sup"}
    for study in studies:
        variants, description = builders[args.kind](study, args)
        if variants == {}:
            continue
        if variants is not None:
            variants = with_diagnostics(study, variants, args)
        entries.append((study, variants, description))
    if args.kind == "2x2":
        write_2x2_outputs(select_variants(entries, args.variants), args)
        return
    write_outputs(select_variants(entries, args.variants), args)


if __name__ == "__main__":
    main()
