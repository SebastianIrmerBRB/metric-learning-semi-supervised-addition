"""Shared discovery, parsing and metric helpers for the Phase-2 evaluation scripts.

Nothing in main_eval/ trains or writes into a run directory. Every script reads
the artifacts training already left behind (``trials.jsonl``, ``cv_summary.json``,
``pseudo_label_diagnostics.jsonl``, ``test_embeddings.npz``, ``split/``) and
writes its own tables and figures below ``--out``.
"""

from __future__ import annotations

import csv
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_OUT = REPO / "main_eval_out"

# Random-search studies (Phase 0) and the TPE studies the Phase-1 seed runs were
# replayed from. Both are only defaults; every script takes --roots.
PHASE0_ROOT = REPO / "sorted_logs" / "phase_0"
PHASE1_ROOT = Path("/mnt/f/server_logs/final")

# Later roots win for the same (dataset, combo, variant, k, seed): the thesis
# reports LRML/Hoffer/SLADE on Cars196 from seeded_runs/cars (1000/40), the
# supervised and LP rows from seeded_runs_cars.
SEEDED_ROOTS = [
    Path("/mnt/f/server_logs/seeded_runs_cars"),
    Path("/mnt/f/server_logs/seeded_runs_cub"),
    Path("/mnt/f/server_logs/seeded_runs_in_shop"),
    Path("/mnt/f/server_logs/seeded_runs/cars"),
]

LOSS_SHORT = {
    "ArcFaceLoss": "arc",
    "CircleLoss": "cir",
    "MultiSimilarityLoss": "ms",
    "NTXentLoss": "nt",
    "ProxyAnchorLoss": "prx",
    "TripletMarginLoss": "tri",
    "ThresholdConsistentMarginLoss": "tcm",
}
DATASET_SHORT = {
    "Cars196": "cars",
    "CUB": "cub",
    "DeepFashionInShop": "in_shop",
    "StanfordOnlineProducts": "sop",
    "iNat": "semi_inat",
}
# Directory prefix of a seeded-run combination -> method label used everywhere here.
COMBO_METHOD = {"sup": "sup", "iscen": "lp", "lrml": "lrml", "hoffer": "hoffer", "slade": "slade"}
METHOD_ORDER = ["sup", "lp", "lrml", "hoffer", "slade", "ismlp", "seraph", "other"]
PRIMARY_METRICS = {
    "map_at_r": "mean_average_precision_at_r",
    "rp": "r_precision",
    "p_at_1": "precision_at_1",
}


# --------------------------------------------------------------------------- io


def read_json(path):
    with open(path) as handle:
        return json.load(handle)


def read_jsonl(path) -> list[dict]:
    rows = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(payload, handle, indent=2, default=_json_default)
    return path


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def write_rows(path, rows: list[dict]):
    """Write dict rows as CSV; the header is the union of keys in first-seen order."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header: list[str] = []
    for row in rows:
        for key in row:
            if key not in header:
                header.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in header})
    return path


def _csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=_json_default)
    return value


def plt():
    """matplotlib.pyplot with a headless backend, imported only when a figure is drawn."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    pyplot.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9})
    return pyplot


def savefig(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    import matplotlib.pyplot as pyplot

    pyplot.close(fig)
    return path


# ---------------------------------------------------------------- method labels


def method_from_ssl_config(ssl_config: dict | None, mode: str | None = None) -> str:
    """Map a recorded ``ssl_config`` to sup / lp / lrml / hoffer / slade / ..."""

    if mode == "supervised" or not ssl_config:
        return "sup"
    method = ssl_config.get("method") or "none"
    if method == "none":
        return "sup"
    if method == "iscen_label_spreading":
        return "lp"
    if method == "regularized":
        regularizer = (ssl_config.get("method_params") or {}).get("regularizer")
        return {"lrml": "lrml", "hoffer_entropy": "hoffer"}.get(regularizer, str(regularizer))
    return {"slade": "slade", "ismlp": "ismlp", "seraph": "seraph"}.get(method, method)


def short_param(name: str) -> str:
    """``ssl_config.method_params.regularizer_params.n_neighbors`` -> ``n_neighbors``.

    Loss and miner parameters keep a prefix, because MS-loss ``alpha`` and label
    propagation's ``alpha`` are different parameters of the same study.
    """

    last = name.split(".")[-1]
    if name.startswith("loss."):
        return f"loss:{last}"
    if name.startswith("miner."):
        return f"miner:{last}"
    return last


# ------------------------------------------------------------------ HPO studies


@dataclass
class Study:
    path: Path
    root: Path
    config: dict
    trials: list[dict]
    phase: int
    sampler: str
    n_startup: int
    dataset: str
    loss: str
    method: str
    spaces: dict = field(default_factory=dict)

    @property
    def study_id(self) -> str:
        try:
            relative = self.path.relative_to(self.root)
        except ValueError:
            relative = self.path
        return f"p{self.phase}:{relative}"

    @property
    def label(self) -> str:
        return f"{DATASET_SHORT.get(self.dataset, self.dataset)}/{self.method}/{LOSS_SHORT.get(self.loss, self.loss)}"

    def completed(self) -> list[dict]:
        return [t for t in self.trials if t.get("state") == "COMPLETE" and t.get("value") is not None]

    def is_random(self, trial: dict) -> bool:
        return self.sampler != "tpe" or int(trial["number"]) < self.n_startup

    def best(self) -> dict | None:
        done = self.completed()
        return max(done, key=lambda t: t["value"]) if done else None


def iter_study_dirs(roots: Iterable[Path]) -> Iterator[tuple[Path, Path]]:
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"[warn] study root does not exist: {root}", file=sys.stderr)
            continue
        for trials in sorted(root.rglob("trials.jsonl")):
            if (trials.parent / "study_config.json").exists():
                yield root, trials.parent


def load_study(root: Path, path: Path) -> Study:
    config = read_json(path / "study_config.json")
    trials = read_jsonl(path / "trials.jsonl")
    hparam = config.get("hparam_config") or {}
    base = config.get("base_args") or {}
    sampler = str(hparam.get("sampler", "random"))
    if sampler == "tpe":
        n_startup = (
            config.get("resolved_tpe_startup_trials")
            or config.get("effective_tpe_startup_trials")
            or hparam.get("tpe_startup_trials")
            or 10  # Optuna's own default
        )
    else:
        n_startup = 10**9
    method = "sup"
    for trial in trials:
        ssl = (trial.get("user_attrs") or {}).get("resolved_ssl_config")
        if ssl is not None:
            method = method_from_ssl_config(ssl, base.get("mode"))
            break
    else:
        method = "sup" if base.get("mode") == "supervised" else "unknown"
    return Study(
        path=path,
        root=root,
        config=config,
        trials=trials,
        phase=1 if sampler == "tpe" else 0,
        sampler=sampler,
        n_startup=int(n_startup),
        dataset=str(base.get("dataset")),
        loss=str(base.get("loss")),
        method=method,
        spaces=dict(hparam.get("spaces") or {}),
    )


def load_studies(roots: Iterable[Path], min_trials: int = 1) -> list[Study]:
    studies = []
    for root, path in iter_study_dirs(roots):
        try:
            study = load_study(root, path)
        except Exception as exc:  # a half-written study must not stop the sweep
            print(f"[warn] skipping {path}: {exc}", file=sys.stderr)
            continue
        if len(study.completed()) >= min_trials:
            studies.append(study)
    return studies


def fold_values(trial: dict, key: str = "best_valid_mean_average_precision_at_r") -> list[float]:
    folds = (trial.get("user_attrs") or {}).get("fold_results") or []
    return [float(f[key]) for f in folds if f.get(key) is not None]


def space_scale(space: dict | None) -> str:
    if not space:
        return "linear"
    if space.get("type") == "categorical":
        return "categorical"
    return "log" if space.get("log") else "linear"


def to_unit(value, space: dict | None):
    """Position of ``value`` in its search space on the space's sampling scale, in [0, 1]."""

    if space is None or space.get("type") == "categorical" or value is None:
        return float("nan")
    low, high = float(space["low"]), float(space["high"])
    value = float(value)
    if space.get("log"):
        low, high, value = math.log(low), math.log(high), math.log(max(value, 1e-300))
    if high == low:
        return 0.5
    return (value - low) / (high - low)


def local_trial_dir(study_path: Path, recorded_log_dir: str) -> Path:
    """Map a fold ``log_dir`` recorded on the training machine below the local study copy."""

    parts = Path(recorded_log_dir).parts
    for index, part in enumerate(parts):
        if re.fullmatch(r"trial_\d+", part):
            return study_path.joinpath(*parts[index:])
    return study_path / recorded_log_dir


# ---------------------------------------------------------------- seeded runs

SEEDED_DIR = re.compile(r"class_subset_k_shot_label_[\d.]+_k_(?P<k>\d+)_comparison_seed_(?P<seed>\d+)")
COMBO = re.compile(r"^(?P<prefix>sup|iscen|lrml|hoffer|slade)_(?P<loss>arc|cir|ms|nt|prx)(?:_(?P<variant>.+))?$")


@dataclass
class SeededRun:
    root: Path
    combo: str  # directory name, e.g. hoffer_ms_pool_labeled
    method: str  # sup / lp / lrml / hoffer / slade
    loss: str  # arc / cir / ms / nt / prx
    variant: str  # "" for the unmodified replay
    k: int
    seed: int
    dataset: str
    cv_summary_path: Path
    summary: dict

    @property
    def key(self):
        return (self.dataset, self.combo, self.k, self.seed)

    @property
    def base_combo(self) -> str:
        return self.combo if not self.variant else self.combo[: -(len(self.variant) + 1)]

    def fold_dirs(self) -> list[Path]:
        """Every fold directory of this CV run, remapped below the local cv_* directory."""

        cv_dir = self.cv_summary_path.parent
        dirs = []
        for fold in sorted(self.summary.get("folds") or [], key=lambda f: f.get("cv_fold", 0)):
            recorded = Path(fold["log_dir"]).parts
            if cv_dir.name in recorded:
                local = cv_dir.joinpath(*recorded[recorded.index(cv_dir.name) + 1 :])
            else:
                local = cv_dir / f"fold_{int(fold.get('cv_fold', 0)):02d}"
            dirs.append(local)
        return dirs

    def metric(self, name: str, aggregation: str = "mean"):
        aggregate = self.summary.get("aggregate") or {}
        key = "concatenated_test_measurements" if aggregation == "concat" else "test_measurements"
        measurements = aggregate.get(key) or {}
        return measurements.get(PRIMARY_METRICS.get(name, name))


def _latest_complete_cv_summary(mode_dir: Path) -> Path | None:
    candidates = []
    for path in mode_dir.glob("cv_*/cv_summary.json"):
        try:
            summary = read_json(path)
        except Exception:
            continue
        completed = summary.get("completed_folds")
        if isinstance(completed, (list, tuple)):
            completed = len(completed)
        complete = completed is None or int(completed) >= int(summary.get("cv_k") or 0)
        if complete and (summary.get("aggregate") or {}).get("test_measurements"):
            candidates.append(path)
    return sorted(candidates)[-1] if candidates else None


def load_seeded_runs(roots: Iterable[Path] = SEEDED_ROOTS) -> list[SeededRun]:
    """Every complete seeded CV run below ``roots``; later roots override earlier ones."""

    runs: dict[tuple, SeededRun] = {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            print(f"[warn] seeded root does not exist: {root}", file=sys.stderr)
            continue
        for combo_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            match = COMBO.match(combo_dir.name)
            if not match:
                continue
            for run_dir in sorted(combo_dir.iterdir()):
                parsed = SEEDED_DIR.fullmatch(run_dir.name)
                if not parsed:
                    continue
                for mode_dir in sorted((run_dir / "final").glob("*")):
                    summary_path = _latest_complete_cv_summary(mode_dir)
                    if summary_path is None:
                        continue
                    summary = read_json(summary_path)
                    dataset = (summary.get("args") or {}).get("dataset") or "unknown"
                    run = SeededRun(
                        root=root,
                        combo=combo_dir.name,
                        method=COMBO_METHOD[match["prefix"]],
                        loss=match["loss"],
                        variant=match["variant"] or "",
                        k=int(parsed["k"]),
                        seed=int(parsed["seed"]),
                        dataset=str(dataset),
                        cv_summary_path=summary_path,
                        summary=summary,
                    )
                    runs[run.key] = run
    return sorted(runs.values(), key=lambda r: (r.dataset, r.combo, r.k, r.seed))


def select_runs(runs, *, dataset=None, k=None, combos=None, methods=None, losses=None, include_variants=False):
    out = []
    for run in runs:
        if dataset and DATASET_SHORT.get(run.dataset, run.dataset) != dataset and run.dataset != dataset:
            continue
        if k is not None and run.k not in (k if isinstance(k, (list, tuple, set)) else [k]):
            continue
        if combos and run.combo not in combos:
            continue
        if methods and run.method not in methods:
            continue
        if losses and run.loss not in losses:
            continue
        if run.variant and not include_variants and not (combos and run.combo in combos):
            continue
        out.append(run)
    return out


def load_test_embeddings(fold_dir: Path):
    path = Path(fold_dir) / "test_embeddings.npz"
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=False)
    return np.asarray(data["embeddings"], dtype=np.float32), np.asarray(data["labels"], dtype=np.int64)


def load_run_test_embeddings(run: SeededRun):
    """All fold embeddings of one run, checked to share one label order."""

    embeddings, labels = [], None
    for fold_dir in run.fold_dirs():
        fold_embeddings, fold_labels = load_test_embeddings(fold_dir)
        if fold_embeddings is None:
            return None, None
        if labels is not None and not np.array_equal(labels, fold_labels):
            raise ValueError(f"fold test label order differs in {fold_dir}")
        labels = fold_labels
        embeddings.append(fold_embeddings)
    return embeddings, labels


def l2_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)


def concat_embeddings(fold_embeddings: list[np.ndarray]) -> np.ndarray:
    """The thesis' Concat: normalized fold embeddings stacked and scaled by F^-1/2."""

    parts = [l2_normalize(e) for e in fold_embeddings]
    return np.concatenate(parts, axis=1) / math.sqrt(len(parts))


# ----------------------------------------------------------- retrieval metrics


def per_query_retrieval(embeddings, labels, *, chunk: int = 1024, device: str | None = None) -> dict:
    """Same-set retrieval (query excluded from its own ranking), per query.

    Returns arrays ``ap_at_r``, ``r_precision``, ``p_at_1`` and ``r`` (number of
    relevant items). Queries without any relevant item get NaN. This is the
    Cars196/CUB protocol; In-Shop's query/gallery split is not handled here.
    """

    import torch

    labels = np.asarray(labels, dtype=np.int64)
    n = len(labels)
    _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
    relevant = counts[inverse] - 1
    max_r = int(relevant.max())
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(l2_normalize(embeddings)).to(device)
    y = torch.from_numpy(labels).to(device)
    ap = np.full(n, np.nan)
    rp = np.full(n, np.nan)
    p1 = np.full(n, np.nan)
    ranks = torch.arange(1, max_r + 1, device=device, dtype=torch.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = x[start:stop] @ x.T
        sims[torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)] = -float("inf")
        top = torch.topk(sims, k=max_r, dim=1).indices
        hits = (y[top] == y[start:stop, None]).float()
        r = torch.from_numpy(relevant[start:stop]).to(device)
        within_r = (ranks[None, :] <= r[:, None]).float()
        hits_r = hits * within_r
        precision_at_i = torch.cumsum(hits_r, dim=1) / ranks[None, :]
        r_safe = torch.clamp(r, min=1).float()
        ap_chunk = (precision_at_i * hits_r).sum(dim=1) / r_safe
        rp_chunk = hits_r.sum(dim=1) / r_safe
        valid = (r > 0).cpu().numpy()
        ap[start:stop] = np.where(valid, ap_chunk.cpu().numpy(), np.nan)
        rp[start:stop] = np.where(valid, rp_chunk.cpu().numpy(), np.nan)
        p1[start:stop] = np.where(valid, hits[:, 0].cpu().numpy(), np.nan)
    return {"ap_at_r": ap, "r_precision": rp, "p_at_1": p1, "r": relevant}


def topk_neighbors(embeddings, k: int, *, chunk: int = 2048, device: str | None = None) -> np.ndarray:
    """Indices of each row's k nearest cosine neighbours, self excluded."""

    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.from_numpy(l2_normalize(embeddings)).to(device)
    n = x.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        sims = x[start:stop] @ x.T
        sims[torch.arange(stop - start, device=device), torch.arange(start, stop, device=device)] = -float("inf")
        out[start:stop] = torch.topk(sims, k=k, dim=1).indices.cpu().numpy()
    return out


# ------------------------------------------------------------ representation


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA (Kornblith et al. 2019) between two representations of the same samples."""

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(y.T @ x, ord="fro") ** 2
    return float(cross / (np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")))


def singular_spectrum(embeddings) -> np.ndarray:
    """Singular values of the centered, normalized embeddings (shared with epoch tracking)."""
    from training.epoch_tracking import singular_spectrum as _singular_spectrum

    return _singular_spectrum(embeddings)


def spectral_decay(singular_values, *, drop_first: bool = False) -> float:
    """Roth et al. (2020)'s rho; one implementation with epoch tracking's ``geometry``."""
    from training.epoch_tracking import spectral_decay as _spectral_decay

    return _spectral_decay(singular_values, drop_first=drop_first)


def class_geometry(embeddings, labels) -> dict:
    """Intra-class spread and centroid separation on the unit sphere (shared with epoch tracking)."""
    from training.epoch_tracking import class_geometry as _class_geometry

    return _class_geometry(embeddings, labels)


# ------------------------------------------------------------------ statistics


def paired_summary(differences: Iterable[float]) -> dict:
    values = np.asarray([d for d in differences if d is not None and np.isfinite(d)], dtype=np.float64)
    if len(values) == 0:
        return {"n": 0}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "sd": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        "min": float(values.min()),
        "max": float(values.max()),
        "n_positive": int((values > 0).sum()),
    }


def bootstrap_ci(values, *, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def add_common_run_args(parser):
    parser.add_argument("--seeded-roots", nargs="+", type=Path, default=SEEDED_ROOTS,
                        help="seeded-run roots; later roots win for the same run (default: the thesis series)")
    parser.add_argument("--dataset", default=None, help="cars / cub / in_shop (or the full dataset name)")
    parser.add_argument("--k", type=int, nargs="+", default=None, help="label budgets k to include")
    parser.add_argument("--losses", nargs="+", default=None, help="arc cir ms nt prx")
    parser.add_argument("--methods", nargs="+", default=None, help="sup lp lrml hoffer slade")
    parser.add_argument("--combos", nargs="+", default=None,
                        help="explicit combination directories, e.g. iscen_ms sup_ms hoffer_ms_pool_labeled")
    parser.add_argument("--include-variants", action="store_true", help="also include ablation variants")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser


def runs_from_args(args) -> list[SeededRun]:
    runs = load_seeded_runs(args.seeded_roots)
    return select_runs(
        runs,
        dataset=args.dataset,
        k=args.k,
        combos=args.combos,
        methods=args.methods,
        losses=args.losses,
        include_variants=args.include_variants,
    )
