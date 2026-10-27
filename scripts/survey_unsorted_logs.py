"""Reports which studies in logs/ and server_logs/ are missing from sorted_logs/.

Read-only. Identity is (base_args.save_dir, resolved_study_name), which together
name the campaign, the run and the Optuna study, and which survive the move into
sorted_logs untouched because study_config.json is copied verbatim.

Usage:
    python scripts/survey_unsorted_logs.py [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DB_NAME = "optuna_study.db"

DATASET_DIRS = {
    "Cars196": "cars196",
    "CUB": "cub",
    "DeepFashionInShop": "in_shop",
    "StanfordOnlineProducts": "sop",
    "iNat": "semi_inat2021",
}

LOSS_DIRS = {
    "ArcFaceLoss": "arcface",
    "CircleLoss": "circle",
    "MultiSimilarityLoss": "multi_similarity",
    "NTXentLoss": "ntxent",
    "ProxyAnchorLoss": "proxy_anchor",
    "TripletMarginLoss": "triplet",
    "ThresholdConsistentMarginLoss": "tcm",
}

# method / regularizer as recorded in the resolved SSL config -> folder name
METHOD_DIRS = {
    "iscen_label_spreading": "iscen",
    "seraph": "seraph",
    "slade": "slade",
    "ismlp": "ismlp",
}
REGULARIZER_DIRS = {
    "lrml": "lrml",
    "hoffer_entropy": "hoffer",
}

UNFINISHED_TRIALS = 30


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def count_trials(directory: Path) -> int:
    """Completed-or-not trial records, from whichever file the run left behind."""
    jsonl = directory / "trials.jsonl"
    if jsonl.exists():
        try:
            return sum(1 for line in jsonl.read_text().splitlines() if line.strip())
        except Exception:
            pass
    csv_path = directory / "trials.csv"
    if csv_path.exists():
        try:
            return max(0, sum(1 for _ in csv_path.read_text().splitlines()) - 1)
        except Exception:
            pass
    try:
        import sqlite3

        with sqlite3.connect(f"file:{directory / DB_NAME}?mode=ro", uri=True) as db:
            return db.execute("SELECT count(*) FROM trials").fetchone()[0]
    except Exception:
        return 0


def ssl_method(directory: Path, base_args: dict) -> str | None:
    """The SSL method, read from the generated per-run config, not the folder."""
    config_path = base_args.get("ssl_config")
    candidates = []
    if isinstance(config_path, str) and config_path:
        candidates.append(REPO / config_path)
        candidates.append(Path(config_path))
    # The run directory keeps a resolved copy next to each fold.
    for run_config in sorted(directory.glob("trial_*/cv_*/fold_*/run_config.json"))[:1]:
        candidates.append(run_config)

    for candidate in candidates:
        data = read_json(candidate)
        if not isinstance(data, dict):
            continue
        ssl = data.get("ssl_config") if "ssl_config" in data else data
        if not isinstance(ssl, dict):
            continue
        method = str(ssl.get("method", ""))
        if method in METHOD_DIRS:
            return METHOD_DIRS[method]
        if method == "regularized":
            params = ssl.get("method_params") or {}
            regularizer = str(params.get("regularizer", ""))
            if regularizer in REGULARIZER_DIRS:
                return REGULARIZER_DIRS[regularizer]
            return regularizer or None
        if method and method != "none":
            return METHOD_DIRS.get(method, method)
    return None


@dataclass
class Study:
    directory: Path
    root: str
    config: dict = field(repr=False)

    @property
    def base_args(self) -> dict:
        return self.config.get("base_args") or {}

    @property
    def hparam(self) -> dict:
        return self.config.get("hparam_config") or {}

    @property
    def sampler(self) -> str:
        return str(self.hparam.get("sampler", "")).lower()

    @property
    def study_name(self) -> str:
        return str(self.config.get("resolved_study_name") or self.hparam.get("study_name") or "")

    @property
    def save_dir(self) -> str:
        return str(self.base_args.get("save_dir") or "")

    @property
    def identity(self) -> tuple[str, str]:
        return (self.save_dir, self.study_name)

    @property
    def mode(self) -> str:
        return str(self.base_args.get("mode", "")).lower()

    @property
    def dataset_dir(self) -> str:
        dataset = str(self.base_args.get("dataset", ""))
        return DATASET_DIRS.get(dataset, dataset.lower())

    @property
    def loss_dir(self) -> str:
        loss = str(self.base_args.get("loss", ""))
        return LOSS_DIRS.get(loss, loss.lower())

    def target_path(self, method: str | None) -> str:
        phase = "phase_0" if self.sampler == "random" else "phase_1"
        parts = [phase, self.mode, self.dataset_dir]
        if self.mode == "ssl":
            parts.append(method or "unknown")
        parts.append(self.loss_dir)
        return "/".join(parts)


def collect(root: Path, label: str) -> list[Study]:
    studies: list[Study] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        if DB_NAME not in filenames:
            continue
        directory = Path(dirpath)
        config = read_json(directory / "study_config.json")
        if config is None:
            config = {}
        studies.append(Study(directory=directory, root=label, config=config))
    return studies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, default=None)
    arguments = parser.parse_args()

    sorted_root = REPO / "sorted_logs"
    known: dict[tuple[str, str], tuple[Path, int]] = {}
    for study in collect(sorted_root, "sorted_logs"):
        known[study.identity] = (study.directory, count_trials(study.directory))

    print(f"sorted_logs: {len(known)} studies indexed")

    rows = []
    for label in ("logs", "server_logs"):
        root = REPO / label
        if not root.exists():
            continue
        for study in collect(root, label):
            trials = count_trials(study.directory)
            method = ssl_method(study.directory, study.base_args) if study.mode == "ssl" else None
            existing = known.get(study.identity)
            rows.append(
                {
                    "root": label,
                    "path": str(study.directory.relative_to(REPO)),
                    "sampler": study.sampler,
                    "n_trials_cfg": study.hparam.get("n_trials"),
                    "trials": trials,
                    "mode": study.mode,
                    "dataset": study.base_args.get("dataset"),
                    "loss": study.base_args.get("loss"),
                    "miner": study.base_args.get("miner"),
                    "method": method,
                    "study_name": study.study_name,
                    "save_dir": study.save_dir,
                    "target": study.target_path(method),
                    "existing": str(existing[0].relative_to(REPO)) if existing else None,
                    "existing_trials": existing[1] if existing else None,
                }
            )

    phase0 = [row for row in rows if row["sampler"] == "random"]
    new = [row for row in phase0 if row["existing"] is None]
    grew = [
        row
        for row in phase0
        if row["existing"] is not None and row["trials"] > (row["existing_trials"] or 0)
    ]
    same = [
        row
        for row in phase0
        if row["existing"] is not None and row["trials"] <= (row["existing_trials"] or 0)
    ]

    print(f"\nlogs + server_logs: {len(rows)} studies")
    print(f"  phase 0 (random): {len(phase0)}")
    print(f"  phase 1 (tpe) or other, ignored: {len(rows) - len(phase0)}")
    print(f"\n  new to sorted_logs:            {len(new)}")
    print(f"  already there, more trials now: {len(grew)}")
    print(f"  already there, no new trials:   {len(same)}")

    if new:
        print("\nNEW")
        for row in sorted(new, key=lambda item: item["target"]):
            flag = " [unfinished]" if row["trials"] < UNFINISHED_TRIALS else ""
            print(
                f"  {row['trials']:>3} trials  {row['target']}{flag}\n"
                f"        from {row['path']}"
            )
    if grew:
        print("\nGREW")
        for row in sorted(grew, key=lambda item: item["target"]):
            print(
                f"  {row['existing_trials']} -> {row['trials']} trials  {row['existing']}\n"
                f"        from {row['path']}"
            )

    if arguments.json:
        arguments.json.write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {arguments.json}")


if __name__ == "__main__":
    main()
