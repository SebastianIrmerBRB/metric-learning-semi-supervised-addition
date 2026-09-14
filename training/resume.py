"""Reuse of finished work when an interrupted study replay is restarted.

A replay writes its own resume state as a side effect of normal reporting:
every completed validation or final run leaves a summary in the study
directory, and every cross-validation run rewrites ``cv_summary.json`` after
each fold. This module reads those artifacts back, so replays started before
this module existed resume just as well as new ones.

Two levels of reuse exist because the two artifacts have different granularity:

* A summary in the study directory means one whole (trial, seed) replay run
  finished, so the run is skipped and its recorded metrics are reused.
* A ``cv_*`` directory whose summary lists fewer completed folds than ``cv_k``
  means one run was interrupted mid-cross-validation. Training continues in
  that directory at the first unfinished fold; the interrupted fold itself is
  retrained, because a fold keeps no resumable state of its own.

Reuse only ever applies to a run whose artifacts sit exactly where the current
request would write them and whose identity fields still match, so replaying
the same study into a different ``save_dir`` (or with changed settings) trains
again instead of adopting the old numbers.
"""

from dataclasses import dataclass
from pathlib import Path

from loguru import logger

import utils
from .io import read_json, result_from_dict, to_jsonable
from .types import TrainingResult

# Fold-level resume is opt-in per run rather than global: only a study replay
# marks its runs, so ordinary training and HPO trials keep starting a fresh
# cross-validation directory even though they parse the same CLI flag.
RESUME_CROSS_VALIDATION_ATTRIBUTE = "resume_cross_validation_folds"

# Fields that make one cross-validation run the same piece of work as another.
# The directory being scanned already pins the scenario, role, and label
# budget, so these separate different trials, seeds, and durations within it.
CV_RESUME_IDENTITY_FIELDS = (
    "cv_k",
    "cv_mode",
    "val_mode",
    "holdout_val_ratio",
    "epochs",
    "seed",
    "data_split_seed",
    "support_seed",
    "loss",
    "miner",
    "batch_size",
    "final_full_train",
    # A fold trained without test evaluation, or without keeping its test
    # embeddings, cannot stand in for one that a per-fold final run needs.
    "evaluate_test",
    "save_test_embeddings",
    "ssl_config",
    "hparam_params",
    "hparam_replay_trial_number",
    # Variants of one ablation can share every field above -- the same tuned
    # params, batch size and SSL config path -- while training something else.
    "ablation_config",
    "ablation_variant",
)


@dataclass(frozen=True)
class ResumableCrossValidationRun:
    """One interrupted ``cv_*`` directory and the folds it already finished."""

    cv_dir: Path
    fold_results: tuple[TrainingResult, ...]

    @property
    def completed_folds(self) -> int:
        return len(self.fold_results)


def resume_enabled(args):
    """Return whether this run may adopt a previous attempt's finished work."""

    return bool(getattr(args, "resume_interrupted_runs", True))


def mark_cross_validation_resume(args, enabled):
    """Allow (or forbid) continuing an interrupted CV run for these args."""

    setattr(args, RESUME_CROSS_VALIDATION_ATTRIBUTE, bool(enabled))
    return args


def load_reusable_replay_result(
    summary_path,
    result_key,
    expected_log_root,
    study_result=None,
    require_metrics=(),
):
    """Return the result a previous attempt recorded, or ``None`` to run again.

    ``expected_log_root`` is the directory the current request would train
    into; a summary pointing anywhere else describes a different replay and is
    ignored rather than reused.
    """

    summary_path = Path(summary_path)
    if not summary_path.is_file():
        return None
    try:
        summary = read_json(summary_path)
    except (OSError, ValueError):
        logger.warning(f"Ignoring unreadable replay summary for resume: {summary_path}")
        return None
    if not isinstance(summary, dict):
        return None
    values = summary.get(result_key)
    if not isinstance(values, dict) or not values.get("log_dir"):
        return None
    if not _matches_recorded_study(summary.get("study"), study_result):
        return None
    log_dir = Path(values["log_dir"])
    if not _is_within(log_dir, Path(expected_log_root)) or not log_dir.is_dir():
        return None
    if any(values.get(name) is None for name in require_metrics):
        return None
    return result_from_dict(values)


def find_resumable_cross_validation_run(args):
    """Return the interrupted CV directory this run should continue, if any."""

    if not getattr(args, RESUME_CROSS_VALIDATION_ATTRIBUTE, False) or not resume_enabled(args):
        return None
    run_root = Path("logs") / args.save_dir
    if not run_root.is_dir():
        return None
    candidates = []
    for cv_dir in sorted(run_root.glob("cv_*")):
        summary_path = cv_dir / "cv_summary.json"
        if not summary_path.is_file():
            # Interrupted before the first fold finished: nothing to reuse.
            continue
        try:
            summary = read_json(summary_path)
        except (OSError, ValueError):
            logger.warning(f"Ignoring unreadable cross-validation summary for resume: {summary_path}")
            continue
        if not isinstance(summary, dict) or not _matches_cv_identity(summary.get("args"), args):
            # A directory left by different settings stays untouched.
            continue
        fold_results = reusable_fold_results(summary, args.cv_k, args=args)
        if not fold_results:
            continue
        candidates.append(ResumableCrossValidationRun(cv_dir, tuple(fold_results)))
    if not candidates:
        return None
    # Prefer the run that got furthest; the newest attempt breaks ties because
    # sorted() ordered the timestamped directory names oldest first.
    return max(candidates, key=lambda candidate: (candidate.completed_folds, candidate.cv_dir.name))


def reusable_fold_results(summary, cv_k, args=None):
    """Return the leading folds of a CV summary that need no retraining.

    With ``args`` from a run that evaluates the test set, a fold must also have
    measured the test table that run reports; reusing one that did not would
    leave its columns blank in the reported results.
    """

    folds = summary.get("folds")
    if not isinstance(folds, list):
        return []
    reusable = []
    for fold_index, fold in enumerate(folds):
        if fold_index >= cv_k:
            break
        if not isinstance(fold, dict) or fold.get("cv_fold") != fold_index:
            break
        if fold.get("best_valid_precision_at_1") is None:
            break
        if fold.get("best_valid_mean_average_precision_at_r") is None:
            break
        log_dir = fold.get("log_dir")
        if not log_dir or not Path(log_dir).is_dir():
            # A fold whose artifacts were removed is treated as unfinished.
            break
        if args is not None and getattr(args, "evaluate_test", False):
            missing = missing_test_metrics(fold, args)
            if missing:
                logger.info(
                    f"Retraining from fold {fold_index}: its recorded test evaluation lacks "
                    f"{list(missing)}, which this run reports"
                )
                break
        reusable.append(result_from_dict(fold))
    return reusable


def missing_test_metrics(recorded, args):
    """Return the D_test metrics this run reports that a recorded result lacks.

    ``recorded`` is a result or its JSON dictionary, and the answer comes from
    what it actually measured rather than from the arguments it recorded. P@1
    and MAP@R are always computed, so only optional measurements and Recall@K
    rungs can be missing.
    """

    def recorded_value(name):
        if isinstance(recorded, dict):
            return recorded.get(name)
        return getattr(recorded, name, None)

    measurements = recorded_value("test_measurements") or {}
    # A JSON summary keys Recall@K by string, a loaded result by int.
    recall_at_k = {int(k) for k in (recorded_value("test_recall_at_k") or {})}
    requested_measurements = utils.additional_measurements(
        utils.resolve_test_measurements(args)
    )
    return (
        *(name for name in requested_measurements if name not in measurements),
        *(
            utils.recall_at_k_metric_name(k)
            for k in utils.resolve_test_recall_at_k(args)
            if k not in recall_at_k
        ),
    )


def _matches_cv_identity(recorded_args, args):
    if not isinstance(recorded_args, dict):
        return False
    for name in CV_RESUME_IDENTITY_FIELDS:
        current = to_jsonable(getattr(args, name, None))
        if recorded_args.get(name) != current:
            return False
    # Old summaries can safely stand in for today's unchanged default pair.
    # Any optional request, meaning anything beyond P@1 and MAP@R, cannot reuse
    # folds that never computed it.
    current_measurements = list(
        getattr(args, "measurements", utils.BASE_RETRIEVAL_METRICS)
    )
    if any(name not in utils.BASE_RETRIEVAL_METRICS for name in current_measurements):
        if recorded_args.get("measurements") != current_measurements:
            return False
    return True


def _matches_recorded_study(recorded_study, study_result):
    if study_result is None:
        return True
    if not isinstance(recorded_study, dict):
        return False
    return (
        recorded_study.get("best_trial_number") == study_result.best_trial_number
        and recorded_study.get("best_params") == study_result.best_params
    )


def _is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
    except ValueError:
        return False
    return True
