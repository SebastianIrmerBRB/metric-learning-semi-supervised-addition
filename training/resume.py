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
  that directory with the folds it does not list; an interrupted fold itself is
  retrained, because a fold keeps no resumable state of its own. A directory
  whose summary already holds the aggregate of all ``cv_k`` folds is a
  finished run, and that recorded aggregate is reused as it stands.

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


# The joint evaluation of every fold's test embeddings, which only a CV
# aggregate carries. It cannot be recomputed from the summary alone.
CONCATENATED_TEST_FIELDS = (
    "concatenated_test_precision_at_1",
    "concatenated_test_mean_average_precision_at_r",
    "concatenated_test_embedding_dim",
    "concatenated_test_recall_at_k",
    "concatenated_test_measurements",
)


@dataclass(frozen=True)
class ResumableCrossValidationRun:
    """One ``cv_*`` directory and the folds it already finished."""

    cv_dir: Path
    fold_results: tuple[TrainingResult, ...]
    # The aggregate the summary recorded over exactly these folds; set only
    # when all cv_k of them are reused, i.e. the run had already finished.
    recorded_aggregate: TrainingResult | None = None

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
    """Return the interrupted or finished CV directory this run should continue, if any."""

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
        candidates.append(
            ResumableCrossValidationRun(
                cv_dir,
                tuple(fold_results),
                recorded_aggregate=recorded_cross_validation_aggregate(
                    summary,
                    fold_results,
                    args.cv_k,
                ),
            )
        )
    if not candidates:
        return None
    # Prefer the run that got furthest; the newest attempt breaks ties because
    # sorted() ordered the timestamped directory names oldest first.
    return max(candidates, key=lambda candidate: (candidate.completed_folds, candidate.cv_dir.name))


# Where --save_epoch_head_interval keeps a run's heads, and the one file
# epoch_head_pack.py packs them into once the run ends. Defined here rather than
# in the engine, which imports this module.
EPOCH_HEADS_DIRNAME = "epoch_heads"
PACKED_EPOCH_HEADS_FILENAME = "epoch_heads.npz"


def lacks_requested_epoch_heads(log_dir, args):
    """Whether ``args`` asks for epoch heads the run in ``log_dir`` did not keep."""

    if args is None or not getattr(args, "save_epoch_head_interval", None):
        return False
    if not log_dir:
        return True
    if (Path(log_dir) / PACKED_EPOCH_HEADS_FILENAME).is_file():
        return False
    return not any((Path(log_dir) / EPOCH_HEADS_DIRNAME).glob("*.pt"))


def reusable_fold_results(summary, cv_k, args=None):
    """Return the folds of a CV summary that need no retraining, in fold order.

    Folds trained in parallel finish in any order, so a summary can record fold
    2 without fold 1; each recorded fold is judged on its own. With ``args`` from
    a run that evaluates the test set, a fold must also have measured the test
    table that run reports; reusing one that did not would leave its columns
    blank in the reported results.
    """

    folds = summary.get("folds")
    if not isinstance(folds, list):
        return []
    reusable = {}
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        fold_index = fold.get("cv_fold")
        if not isinstance(fold_index, int) or not 0 <= fold_index < cv_k or fold_index in reusable:
            continue
        if fold.get("best_valid_precision_at_1") is None:
            continue
        if fold.get("best_valid_mean_average_precision_at_r") is None:
            continue
        log_dir = fold.get("log_dir")
        if not log_dir or not Path(log_dir).is_dir():
            # A fold whose artifacts were removed is treated as unfinished.
            continue
        if args is not None and getattr(args, "save_head_state", False):
            head_path = fold.get("head_state_path")
            if not head_path or not Path(head_path).is_file():
                logger.info(
                    f"Retraining fold {fold_index}: this run requests a saved head and "
                    "the recorded fold has none"
                )
                continue
        if lacks_requested_epoch_heads(log_dir, args):
            logger.info(
                f"Retraining fold {fold_index}: this run saves epoch heads and the "
                "recorded fold has none"
            )
            continue
        if args is not None and getattr(args, "evaluate_test", False):
            missing = missing_test_metrics(fold, args)
            if missing:
                logger.info(
                    f"Retraining fold {fold_index}: its recorded test evaluation lacks "
                    f"{list(missing)}, which this run reports"
                )
                continue
        reusable[fold_index] = result_from_dict(fold)
    return [reusable[index] for index in sorted(reusable)]


def recorded_cross_validation_aggregate(summary, fold_results, cv_k):
    """The aggregate a CV summary recorded, if it covers exactly ``fold_results``.

    The aggregate is written once, after the last fold, so it describes the
    reused folds only while every one of the ``cv_k`` recorded folds is reused.
    """

    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, dict) or summary.get("completed_folds") != cv_k:
        return None
    if len(fold_results) != cv_k:
        # A fold dropped for retraining changes the mean the aggregate holds.
        return None
    try:
        return result_from_dict(aggregate)
    except (KeyError, TypeError, ValueError):
        logger.warning(f"Ignoring an unreadable recorded cross-validation aggregate: {aggregate.get('log_dir')}")
        return None


def recorded_concatenated_test_metrics(aggregate):
    """The concatenated fold test fields of a recorded aggregate, or ``None`` without them."""

    if aggregate is None:
        return None
    if aggregate.concatenated_test_precision_at_1 is None:
        return None
    if aggregate.concatenated_test_mean_average_precision_at_r is None:
        return None
    return {name: getattr(aggregate, name) for name in CONCATENATED_TEST_FIELDS}


def missing_concatenated_test_metrics(aggregate, args):
    """Return the concatenated test metrics this run reports that ``aggregate`` lacks.

    The joint evaluation reports the same table as each fold's test run, so it
    is held to the same check as a fold, Recall@K rungs included.
    """

    concatenated = recorded_concatenated_test_metrics(aggregate)
    if concatenated is None:
        return ("concatenated_test_precision_at_1", "concatenated_test_mean_average_precision_at_r")
    return missing_test_metrics(
        {
            "test_measurements": concatenated["concatenated_test_measurements"],
            "test_recall_at_k": concatenated["concatenated_test_recall_at_k"],
        },
        args,
    )


def leading_fold_results(fold_results):
    """The folds ``0..n-1`` without a gap, for work that must run in fold order."""

    leading = []
    for fold_index, result in enumerate(sorted(fold_results, key=lambda result: result.cv_fold)):
        if result.cv_fold != fold_index:
            break
        leading.append(result)
    return leading


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
