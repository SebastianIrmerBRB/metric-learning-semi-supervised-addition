"""Decide how target-ratio gradient probes should be aggregated, from the logs.

The trainer calibrates ``regularizer_weight`` so that
``||grad(w L_reg)|| / ||grad(w_sup L_sup)||`` hits a configured ratio. It cannot
read that ratio off one batch without committing to whatever that batch happened
to contain, so it summarizes a set of probes ``q_b = ||grad(w_sup L_sup)|| /
||grad(L_reg)||`` with a statistic. Four rules are implemented:

``median`` / ``mean`` over a window that resets every
``recalibration_interval_samples``; the same two over the last
``probe_memory`` probes with no reset; and an exponential moving average with
``ema_alpha``. This script measures which of them the recorded runs support,
and answers four separate questions that are easy to conflate:

``outliers``
    Do wildly atypical probes actually occur? The median is only worth its
    variance penalty if they do. Measured on the probe stream itself and on the
    per-step reconstruction, per run and pooled.
``replay``
    Held fixed the recorded gradient stream, what target error would each rule
    have produced? Paired over runs, with a t-test and a Wilcoxon test, because
    the differences turn out to be small enough that "eight of ten runs" is not
    by itself evidence.
``decomposition``
    How much of the residual target error is the estimator's, and how much is
    simply that ``q`` drifts between probes? This is what decides whether a
    denser probe cadence could help at all.
``stress``
    What each rule does when a contaminated probe *is* present: a single
    injected outlier, and a sweep over contamination rates. This is the regime
    the default is chosen for, and it cannot be read off clean runs.

``endtoend`` compares the run groups that were trained under different rules,
and ``hpo`` repeats the outlier question over a tuned study, where the searched
regularizer parameters produce gradient scales the fixed diagnostic
configuration never visits.

The per-step reconstruction underpinning all of it uses that
``train/gradient_contribution/*`` and the calibration probe measure the same
norms over the same parameters, and that ``regularizer_weight`` is piecewise
constant between probes:

    q_t = supervised_norm_t * w_t / regularizer_norm_t

``--verify`` checks this against the probe rows, where both are logged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy import stats as scipy_stats
except ImportError:  # pragma: no cover - reported instead of failing the run
    scipy_stats = None

REPO = Path(__file__).resolve().parent.parent
TR = "train/target_ratio/"
GC = "train/gradient_contribution/"

# The groups of complete runs recorded for this question. Each holds one grid of
# five DML losses at two batch sizes, seed 7, trained under the named rule.
RUN_GROUPS = {
    "windowed median": "target_ratio_windowed_rerun",
    "rolling median": "target_ratio_sliding",
    "ema a=0.1": "target_ratio_ema",
    "windowed mean": "target_ratio_mean_windowed",
    "rolling mean": "target_ratio_mean_sliding",
    "per-step k=1": "target_ratio_perstep_k1",
    "per-step matched": "target_ratio_perstep_match",
    "rolling median k=3": "target_ratio_k3",
    "rolling median k=5": "target_ratio_k5",
    "rolling median k=20": "target_ratio_k20",
    "rolling median k=30": "target_ratio_k30",
    "hoffer c1000 k10": "hoffer_cadence_c1000_k10",
    "hoffer c500 k20": "hoffer_cadence_c500_k20",
    "hoffer c250 k40": "hoffer_cadence_c250_k40",
    "hoffer c250 k10": "hoffer_cadence_c250_k10",
    "lrml bs32": "lrml_control_bs32",
    "lrml lr1e-4": "lrml_control_bs64_lr1e4",
    "cadence samples1000": "cadence_unit_samples1000",
    "cadence steps10": "cadence_unit_steps10",
    "surgery off": "surgery_off",
    "surgery supervised_priority": "surgery_supervised_priority",
    "surgery pcgrad": "surgery_pcgrad",
}
# The configuration those runs share; see configs/test_bed/lrml_target_ratio_*.
TARGET_RATIO = 0.5
PROBE_INTERVAL_SAMPLES = 1000
WINDOW_SAMPLES = 10_000
PROBE_MEMORY = 10


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _search_roots(root):
    """Where a run group may live; ``moved_target`` is an archived copy of logs."""

    return [REPO / "moved_target", REPO / "logs"] if root is None else [Path(root)]


def find_runs(group, root=None):
    """Yield ``(batch_size_dir, run_name, timestamp_dir)`` for one run group."""

    for base in _search_roots(root):
        group_dir = base / group
        if not group_dir.is_dir():
            continue
        found = []
        for bs_dir in sorted(group_dir.iterdir()):
            if not bs_dir.is_dir():
                continue
            for run_dir in sorted(bs_dir.iterdir()):
                if run_dir.name == "experiment_grid" or not run_dir.is_dir():
                    continue
                for stamp in sorted(run_dir.iterdir()):
                    if (stamp / "diagnostics.csv").exists():
                        found.append((bs_dir.name, run_dir.name, stamp))
        if found:
            return found
    return []


def load_diagnostics(stamp):
    frame = pd.read_csv(stamp / "diagnostics.csv")
    frame = frame[frame["category"] == "train_batch"]
    return frame.pivot_table(index="step", columns="name", values="value")


def probe_series(wide):
    """The probe stream exactly as the calibrator saw it."""

    unit = TR + "unit_regularizer_gradient_norm"
    if unit not in wide.columns:
        return None
    columns = [
        TR + "supervised_gradient_norm",
        unit,
        TR + "regularizer_weight",
        TR + "applied_ratio",
        TR + "regularized_samples_seen",
    ]
    probes = wide[[c for c in columns if c in wide.columns]].dropna(subset=[unit]).copy()
    probes["q"] = probes[TR + "supervised_gradient_norm"] / probes[unit]
    return probes


def step_series(wide):
    """Per-step ``q`` reconstructed from the weighted contribution log."""

    needed = [GC + "supervised_norm", GC + "regularizer_norm"]
    if any(c not in wide.columns for c in needed):
        return None
    steps = wide[needed].dropna().copy()
    weight = wide.get(TR + "regularizer_weight")
    if weight is None:
        return None
    steps["weight"] = weight.reindex(wide.index).ffill().reindex(steps.index)
    steps = steps.dropna(subset=["weight"])
    steps["q"] = steps[GC + "supervised_norm"] * steps["weight"] / steps[GC + "regularizer_norm"]
    # The ratio this step actually trained with; the target error is measured on it.
    steps["achieved"] = steps[GC + "regularizer_norm"] / steps[GC + "supervised_norm"]
    return steps


def run_label(run_name):
    return run_name.split("_k_16_")[1].replace("_seed_7", "")


def collect(group, root=None):
    """``(batch_size, loss, probes, steps)`` for every run in a group."""

    out = []
    for bs, run, stamp in find_runs(group, root):
        wide = load_diagnostics(stamp)
        probes = probe_series(wide)
        steps = step_series(wide)
        if probes is None or steps is None or probes.empty:
            continue
        out.append((bs, run_label(run), probes, steps, stamp))
    return out


# --------------------------------------------------------------------------- #
# estimators
# --------------------------------------------------------------------------- #
def aggregate(values, statistic):
    return float(np.median(values)) if statistic == "median" else float(np.mean(values))


def replay_weights(q, samples, kind, statistic=None, param=None, target=TARGET_RATIO,
                   with_buffer_size=False):
    """The weight each rule leaves in place after every probe.

    ``kind`` is ``window`` (reset every ``WINDOW_SAMPLES``), ``roll`` (last
    ``param`` probes) or ``ema`` (folding factor ``param``). ``with_buffer_size``
    also returns how many probes each estimate was formed from, which is what
    separates a reset window's opening probes -- where a median has no majority
    to protect it -- from its steady state.
    """

    weights = np.empty(len(q))
    sizes = np.empty(len(q), dtype=int)
    if kind == "ema":
        estimate = None
        for index, value in enumerate(q):
            estimate = value if estimate is None else param * value + (1.0 - param) * estimate
            weights[index] = target * estimate
            sizes[index] = index + 1
        return (weights, sizes) if with_buffer_size else weights
    buffer = []
    window_start = samples[0]
    for index, value in enumerate(q):
        if kind == "window" and samples[index] - window_start >= WINDOW_SAMPLES:
            buffer = []
            window_start = samples[index]
        buffer.append(value)
        if kind == "roll" and len(buffer) > param:
            buffer = buffer[-param:]
        weights[index] = target * aggregate(buffer, statistic)
        sizes[index] = len(buffer)
    return (weights, sizes) if with_buffer_size else weights


ESTIMATORS = {
    "windowed median": ("window", "median", None),
    "windowed mean": ("window", "mean", None),
    "rolling median": ("roll", "median", PROBE_MEMORY),
    "rolling mean": ("roll", "mean", PROBE_MEMORY),
    "ema a=0.01": ("ema", None, 0.01),
    "ema a=0.05": ("ema", None, 0.05),
    "ema a=0.1": ("ema", None, 0.1),
    "ema a=0.2": ("ema", None, 0.2),
    "ema a=0.3": ("ema", None, 0.3),
    "ema a=0.5": ("ema", None, 0.5),
    "ema a=0.9": ("ema", None, 0.9),
}


def rolling(values, k, statistic):
    out = np.empty(len(values))
    buffer = []
    for index, value in enumerate(values):
        buffer.append(value)
        if len(buffer) > k:
            buffer = buffer[-k:]
        out[index] = aggregate(buffer, statistic)
    return out


def weight_in_effect(probe_steps, weights, step_index, causal=False):
    """Map a per-probe weight onto every training step.

    The engine recombines the loss on the probe's own batch, so by default the
    probe step trains with the weight measured on it. ``causal`` scores the same
    schedule with the weight from strictly earlier probes instead, which is what
    a cadence comparison needs: otherwise per-step probing wins by fitting the
    weight to the very batch it is applied to.
    """

    side = "left" if causal else "right"
    slot = np.searchsorted(probe_steps, step_index, side=side) - 1
    usable = slot >= 0
    applied = np.full(len(step_index), np.nan)
    applied[usable] = weights[slot[usable]]
    return applied, usable


# --------------------------------------------------------------------------- #
# sections
# --------------------------------------------------------------------------- #
def section_verify(runs):
    rows = []
    for bs, loss, probes, steps, _ in runs:
        common = probes.index.intersection(steps.index)
        error = (steps.loc[common, "q"] / probes.loc[common, "q"] - 1.0).abs()
        recorded = probes[TR + "applied_ratio"].to_numpy()
        implied = probes[TR + "regularizer_weight"].to_numpy() / probes["q"].to_numpy()
        rows.append(
            dict(
                batch_size=bs,
                loss=loss,
                probes=len(probes),
                steps=len(steps),
                max_rel_error_q=float(error.max()),
                max_rel_error_applied=float(np.abs(recorded / implied - 1.0).max()),
            )
        )
    return pd.DataFrame(rows)


def section_shape(runs, window=21, k=PROBE_MEMORY, draws=200_000, seed=0):
    """Is the probe noise the light-tailed, symmetric case the mean is best for?

    ``q`` drifts with the projection head, so the shape has to be read off
    residuals around a local level rather than off the raw series. A centred
    rolling median over ``window`` steps supplies that level; what is left is the
    batch-to-batch variation a ten-probe estimator has to average out. The
    bootstrap then asks the question the choice actually turns on: resampling
    ``k`` of this run's own residuals, is the median or the mean the tighter
    estimate of the level? Under a normal distribution the ratio is about 1.37
    in the mean's favour.
    """

    rng = np.random.default_rng(seed)
    rows = []
    for bs, loss, _, steps, _ in runs:
        q = steps["q"].to_numpy()
        level = pd.Series(q).rolling(window, center=True, min_periods=window).median().to_numpy()
        log_residual = np.log(q) - np.log(level)
        log_residual = log_residual[np.isfinite(log_residual)]
        multiplier = (q / level)[np.isfinite(q / level)]
        sample = rng.choice(multiplier, size=(draws, k))
        row = dict(
            batch_size=bs,
            loss=loss,
            n=len(log_residual),
            sd_log=float(log_residual.std()),
            robust_sd_log=float(
                (np.quantile(log_residual, 0.75) - np.quantile(log_residual, 0.25)) / 1.349
            ),
            var_median_over_var_mean=float(
                np.var(np.median(sample, axis=1)) / np.var(sample.mean(axis=1))
            ),
        )
        if scipy_stats is not None:
            row["skew_log"] = float(scipy_stats.skew(log_residual))
            row["excess_kurtosis_log"] = float(scipy_stats.kurtosis(log_residual))
        rows.append(row)
    return pd.DataFrame(rows)


def section_outliers(runs, factor=2.0, k=PROBE_MEMORY):
    """How often a probe deviates from the median of its predecessors."""

    rows = []
    for bs, loss, probes, steps, _ in runs:
        for name, values in (("probe", probes["q"].to_numpy()), ("step", steps["q"].to_numpy())):
            if len(values) <= k:
                continue
            local = np.array([np.median(values[i - k : i]) for i in range(k, len(values))])
            ratio = values[k:] / local
            deviation = np.maximum(ratio, 1.0 / ratio)
            rows.append(
                dict(
                    batch_size=bs,
                    loss=loss,
                    stream=name,
                    n=len(ratio),
                    beyond_factor=int((deviation > factor).sum()),
                    up=int((ratio > factor).sum()),
                    down=int((ratio < 1.0 / factor).sum()),
                    p99_deviation=float(np.quantile(deviation, 0.99)),
                    max_deviation=float(deviation.max()),
                )
            )
    return pd.DataFrame(rows)


def _paired_report(pivot, baseline):
    rows = []
    for column in pivot.columns:
        if column == baseline:
            continue
        difference = pivot[column] - pivot[baseline]
        n = len(difference)
        half = 1.96 * difference.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
        row = dict(
            estimator=column,
            mean_difference_pp=difference.mean(),
            ci_low_pp=difference.mean() - half,
            ci_high_pp=difference.mean() + half,
            wins=int((difference < 0).sum()),
            n=n,
        )
        if scipy_stats is not None and n > 2:
            row["t_p"] = float(scipy_stats.ttest_rel(pivot[column], pivot[baseline]).pvalue)
            row["wilcoxon_p"] = float(scipy_stats.wilcoxon(pivot[column], pivot[baseline]).pvalue)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_difference_pp")


def section_replay(runs):
    rows = []
    for bs, loss, probes, steps, _ in runs:
        q_probe = probes["q"].to_numpy()
        samples = probes[TR + "regularized_samples_seen"].to_numpy()
        probe_steps = probes.index.to_numpy()
        q_step = steps["q"].to_numpy()
        step_index = steps.index.to_numpy()
        for name, (kind, statistic, param) in ESTIMATORS.items():
            weights = replay_weights(q_probe, samples, kind, statistic, param)
            applied, usable = weight_in_effect(probe_steps, weights, step_index)
            error = np.abs(applied[usable] / (q_step[usable] * TARGET_RATIO) - 1.0)
            rows.append(
                dict(
                    batch_size=bs,
                    loss=loss,
                    estimator=name,
                    mae_pct=error.mean() * 100,
                    p95_pct=np.quantile(error, 0.95) * 100,
                    p99_pct=np.quantile(error, 0.99) * 100,
                    max_pct=error.max() * 100,
                )
            )
    return pd.DataFrame(rows)


def section_decomposition(runs):
    """Estimator error against drift, and what a denser cadence would buy."""

    rows = []
    for bs, loss, probes, steps, _ in runs:
        batch_size = int(bs.split("bs")[-1])
        q_probe = probes["q"].to_numpy()
        samples = probes[TR + "regularized_samples_seen"].to_numpy()
        probe_steps = probes.index.to_numpy()
        q_step = steps["q"].to_numpy()
        step_index = steps.index.to_numpy()
        n = len(q_step)

        estimate = rolling(q_probe, PROBE_MEMORY, "median")
        weights = TARGET_RATIO * estimate
        applied, usable = weight_in_effect(probe_steps, weights, step_index)
        causal_applied, causal_usable = weight_in_effect(
            probe_steps, weights, step_index, causal=True
        )
        slot = np.searchsorted(probe_steps, step_index, side="right") - 1

        def score(values, mask):
            return float(np.abs(values[mask] / (q_step[mask] * TARGET_RATIO) - 1.0).mean() * 100)

        # the estimator's own error, judged on the probe batch it just absorbed
        estimator_error = np.abs(estimate[PROBE_MEMORY:] / q_probe[PROBE_MEMORY:] - 1.0).mean() * 100
        # how far q has moved from the probe whose weight is still in effect
        drift = np.abs(q_step[usable] / q_probe[slot[usable]] - 1.0).mean() * 100

        # per-step probing at the same buffer length, and at a matched sample lookback
        matched = max(1, round(WINDOW_SAMPLES / batch_size))
        per_step = TARGET_RATIO * rolling(q_step, PROBE_MEMORY, "median")
        per_step_matched = TARGET_RATIO * rolling(q_step, matched, "median")
        shift = lambda a: np.concatenate([[np.nan], a[:-1]])
        rows.append(
            dict(
                batch_size=bs,
                loss=loss,
                estimator_error_pct=estimator_error,
                drift_pct=drift,
                cadence_1000_pct=score(applied, usable),
                cadence_1000_causal_pct=score(causal_applied, causal_usable),
                per_step_k10_pct=score(per_step, np.arange(n) >= PROBE_MEMORY),
                per_step_k10_causal_pct=score(shift(per_step), np.arange(n) >= PROBE_MEMORY + 1),
                per_step_matched_pct=score(per_step_matched, np.arange(n) >= matched),
                per_step_matched_causal_pct=score(
                    shift(per_step_matched), np.arange(n) >= matched + 1
                ),
                per_step_k1_causal_pct=score(shift(TARGET_RATIO * q_step), np.arange(n) >= 1),
                matched_buffer=matched,
            )
        )
    return pd.DataFrame(rows)


def section_memory(runs, memories=(1, 2, 3, 5, 8, 10, 15, 20, 30, 50, 80, 120),
                   magnitude=100.0, contamination=0.02, repeats=10, seed=2000):
    """How long the sliding buffer should be.

    ``k`` sets two things at once and they pull in opposite directions. Longer
    buffers average away more of the per-batch noise, but they also lag a
    drifting ``q`` by roughly ``k/2`` probes, and ``q`` does drift -- it tracks
    the projection head. Longer buffers also raise the number of contaminated
    probes a window can absorb, since a median needs a clean majority.

    Reported per ``k``: the clean target error, the same error under a fixed
    contamination rate, and the lag and noise terms separately. Lag is measured
    against an oracle that knows the current probe; noise is the spread of the
    estimate around a centred local level.
    """

    rng = np.random.default_rng(seed)
    rows = []
    for k in memories:
        clean, dirty, lag, noise = [], [], [], []
        for bs, loss, probes, steps, _ in runs:
            q = probes["q"].to_numpy()
            samples = probes[TR + "regularized_samples_seen"].to_numpy()
            probe_steps = probes.index.to_numpy()
            q_step = steps["q"].to_numpy()
            step_index = steps.index.to_numpy()

            weights = replay_weights(q, samples, "roll", "median", k)
            applied, usable = weight_in_effect(probe_steps, weights, step_index)
            clean.append(np.abs(applied[usable] / (q_step[usable] * TARGET_RATIO) - 1.0).mean() * 100)

            # lag: the estimator against the probe it has just seen, on a smoothed
            # q, so only the systematic offset from the trend is left
            level = pd.Series(q).rolling(5, center=True, min_periods=5).median().to_numpy()
            estimate = rolling(q, k, "median")
            finite = np.isfinite(level)
            finite[:k] = False
            lag.append(float(np.abs(estimate[finite] / level[finite] - 1.0).mean() * 100))
            noise.append(float(np.std(np.log(estimate[finite] / level[finite])) * 100))

            for repeat in range(repeats):
                local = np.random.default_rng(rng.integers(1 << 30))
                contaminated = q.copy()
                contaminated[local.random(len(q)) < contamination] *= magnitude
                weights = replay_weights(contaminated, samples, "roll", "median", k)
                applied, usable = weight_in_effect(probe_steps, weights, step_index)
                dirty.append(
                    np.abs(applied[usable] / (q_step[usable] * TARGET_RATIO) - 1.0).mean() * 100
                )
        rows.append(
            dict(
                k=k,
                lookback_samples=k * PROBE_INTERVAL_SAMPLES,
                clean_mae_pct=float(np.mean(clean)),
                contaminated_mae_pct=float(np.mean(dirty)),
                lag_pct=float(np.mean(lag)),
                noise_pct=float(np.mean(noise)),
            )
        )
    return pd.DataFrame(rows)


def section_stress(runs, magnitude=100.0, repeats=20, seed=1000):
    """What each rule does when a probe is contaminated.

    ``magnitude`` multiplies ``q``: a regularizer gradient ``magnitude`` times
    smaller than usual, which is the direction that inflates the weight and the
    one the ``1e-6`` rejection floor is too low to catch.
    """

    prepared = []
    for bs, loss, probes, steps, _ in runs:
        prepared.append(
            (
                bs,
                loss,
                probes["q"].to_numpy(),
                probes[TR + "regularized_samples_seen"].to_numpy(),
                probes.index.to_numpy(),
                steps["q"].to_numpy(),
                steps.index.to_numpy(),
            )
        )

    sweep = []
    for rate in (0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.40):
        totals = {name: [] for name in ESTIMATORS}
        for repeat in range(repeats):
            rng = np.random.default_rng(seed + repeat)
            for _, _, q, samples, probe_steps, q_step, step_index in prepared:
                contaminated = q.copy()
                contaminated[rng.random(len(q)) < rate] *= magnitude
                for name, (kind, statistic, param) in ESTIMATORS.items():
                    weights = replay_weights(contaminated, samples, kind, statistic, param)
                    applied, usable = weight_in_effect(probe_steps, weights, step_index)
                    error = np.abs(applied[usable] / (q_step[usable] * TARGET_RATIO) - 1.0)
                    totals[name].append(error.mean() * 100)
        sweep.append(dict(contamination_pct=rate * 100, **{k: float(np.mean(v)) for k, v in totals.items()}))

    # Where in a window the damage happens. A reset window spends its opening
    # probes with one or two values in the buffer, where the median is the
    # contaminated probe; the sliding window never returns to that state.
    positions = []
    for rate in (0.005, 0.02):
        for name in ("windowed median", "rolling median"):
            kind, statistic, param = ESTIMATORS[name]
            everywhere, mature = [], []
            for repeat in range(repeats):
                rng = np.random.default_rng(seed + 7000 + repeat)
                for _, _, q, samples, probe_steps, q_step, step_index in prepared:
                    contaminated = q.copy()
                    contaminated[rng.random(len(q)) < rate] *= magnitude
                    weights, sizes = replay_weights(
                        contaminated, samples, kind, statistic, param, with_buffer_size=True
                    )
                    applied, usable = weight_in_effect(probe_steps, weights, step_index)
                    error = np.abs(applied[usable] / (q_step[usable] * TARGET_RATIO) - 1.0)
                    slot = np.searchsorted(probe_steps, step_index, side="right") - 1
                    grown = sizes[slot[usable]] >= 3
                    everywhere.append(error.mean() * 100)
                    mature.append(error[grown].mean() * 100)
            positions.append(
                dict(
                    contamination_pct=rate * 100,
                    estimator=name,
                    mae_pct=float(np.mean(everywhere)),
                    mae_buffer_ge3_pct=float(np.mean(mature)),
                )
            )

    transient = []
    for name, (kind, statistic, param) in ESTIMATORS.items():
        peaks, durations = [], []
        for _, _, q, samples, _, _, _ in prepared:
            clean = replay_weights(q, samples, kind, statistic, param)
            hit = len(q) // 2
            contaminated = q.copy()
            contaminated[hit] *= magnitude
            dirty = replay_weights(contaminated, samples, kind, statistic, param)
            relative = np.abs(dirty[hit:] / clean[hit:] - 1.0)
            peaks.append(relative.max() * 100)
            beyond = np.where(relative > 0.10)[0]
            durations.append(int(beyond.max() + 1) if len(beyond) else 0)
        transient.append(
            dict(
                estimator=name,
                peak_weight_error_pct=float(np.mean(peaks)),
                probes_above_10pct=float(np.mean(durations)),
            )
        )
    return pd.DataFrame(sweep), pd.DataFrame(transient), pd.DataFrame(positions)


def section_endtoend(root=None):
    """Runs actually trained under each rule: target error and validation mAP@R."""

    rows = []
    for label, group in RUN_GROUPS.items():
        for bs, run, stamp in find_runs(group, root):
            wide = load_diagnostics(stamp)
            steps = step_series(wide)
            metrics_path = stamp / "metrics.csv"
            if steps is None or not metrics_path.exists():
                continue
            error = np.abs(steps["achieved"].to_numpy() / TARGET_RATIO - 1.0)
            metrics = pd.read_csv(metrics_path)
            valid = metrics[metrics["split"] == "valid"]
            if valid.empty:
                continue
            rows.append(
                dict(
                    estimator=label,
                    batch_size=bs,
                    loss=run_label(run),
                    mae_pct=error.mean() * 100,
                    p95_pct=np.quantile(error, 0.95) * 100,
                    best_map_at_r=valid["mean_average_precision_at_r"].max(),
                    best_precision_at_1=valid["precision_at_1"].max(),
                    final_map_at_r=valid["mean_average_precision_at_r"].iloc[-1],
                )
            )
    return pd.DataFrame(rows)


def section_hpo(study_root=None, factor=2.0):
    """The same outlier question over a tuned study.

    Those runs log only ``gradient_contribution``, but the achieved ratio there
    is ``w_t / q_t`` and ``w_t`` is the rolling median of the last ten probes,
    so ``ratio / target`` *is* the deviation of this batch's ``q`` from its local
    median -- the same statistic ``outliers`` computes, over configurations the
    fixed diagnostic runs never visit.
    """

    root = Path(study_root) if study_root else REPO / "logs/ssl/cars/lrml_target_ratio"
    column = GC + "regularizer_to_supervised_ratio"
    rows = []
    for path in sorted(root.rglob("diagnostics.csv")):
        config_path = path.parent / "run_config.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        params = config.get("ssl_config", {}).get("method_params", {})
        target = params.get("regularizer_target_ratio")
        if target is None:
            continue
        frame = pd.read_csv(path)
        frame = frame[frame["category"] == "train_batch"]
        wide = frame.pivot_table(index="step", columns="name", values="value")
        if column not in wide.columns:
            continue
        ratio = wide[column].dropna()
        # drop the opening steps, where the configured probe weight may still stand
        ratio = ratio[ratio.index >= 64]
        if len(ratio) < 50:
            continue
        deviation = ratio.to_numpy() / target  # = local median of q / this batch's q
        regularizer = params.get("regularizer_params", {})
        rows.append(
            dict(
                loss=config.get("args", {}).get("loss"),
                target_ratio=target,
                n=len(deviation),
                graph_batch_size=regularizer.get("graph_batch_size"),
                n_neighbors=regularizer.get("n_neighbors"),
                q_above_local_median=100 * float(np.mean(deviation < 1.0 / factor)),
                q_above_local_median_10x=100 * float(np.mean(deviation < 0.1)),
                q_below_local_median=100 * float(np.mean(deviation > factor)),
                worst_q_over_local_median=float(1.0 / deviation.min()),
                mae_pct=float(np.abs(deviation - 1.0).mean() * 100),
            )
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def _show(title, frame, index=False):
    print(f"\n=== {title} ===")
    if frame is None or len(frame) == 0:
        print("(no data)")
        return
    print(frame.to_string(index=index, float_format=lambda value: f"{value:.4g}"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--section",
        default="all",
        choices=["all", "verify", "shape", "outliers", "replay", "memory", "decomposition", "stress", "endtoend", "hpo"],
    )
    parser.add_argument("--group", default="target_ratio_sliding", help="run group the replay reads")
    parser.add_argument("--root", default=None, help="override the log root")
    parser.add_argument("--study-root", default=None, help="tuned study root for --section hpo")
    parser.add_argument("--out", default=None, help="write the tables as CSV into this directory")
    arguments = parser.parse_args()

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)

    outputs = {}
    wanted = arguments.section
    runs = []
    if wanted in ("all", "verify", "shape", "outliers", "replay", "memory", "decomposition", "stress"):
        runs = collect(arguments.group, arguments.root)
        if not runs:
            raise SystemExit(f"no runs with diagnostics found for group {arguments.group!r}")
        print(f"{len(runs)} runs from {arguments.group}: "
              f"{sorted({bs for bs, *_ in runs})}, {sorted({loss for _, loss, *_ in runs})}")

    if wanted in ("all", "verify"):
        frame = section_verify(runs)
        outputs["verify"] = frame
        _show("per-step reconstruction against the logged probes", frame)

    if wanted in ("all", "shape"):
        frame = section_shape(runs)
        outputs["shape"] = frame
        _show("shape of the probe noise around a local level", frame)
        _show("means", frame.drop(columns=["batch_size", "loss"]).mean().to_frame("mean"), index=True)

    if wanted in ("all", "outliers"):
        frame = section_outliers(runs)
        outputs["outliers"] = frame
        _show("probes deviating by more than a factor of two from their local median", frame)
        pooled = frame.groupby("stream")[["n", "beyond_factor", "up", "down"]].sum()
        _show("pooled", pooled, index=True)

    if wanted in ("all", "replay"):
        frame = section_replay(runs)
        outputs["replay"] = frame
        summary = frame.groupby("estimator")[["mae_pct", "p95_pct", "p99_pct", "max_pct"]].mean()
        _show("offline replay on the recorded gradient stream", summary.sort_values("mae_pct"), index=True)
        pivot = frame.pivot_table(index=["batch_size", "loss"], columns="estimator", values="mae_pct")
        _show("per-run mean absolute target error [%]", pivot, index=True)
        paired = _paired_report(pivot, "rolling median")
        outputs["replay_paired"] = paired
        _show("paired against the rolling median", paired)

    if wanted in ("all", "memory"):
        frame = section_memory(runs)
        outputs["memory"] = frame
        _show("how long the sliding buffer should be", frame)

    if wanted in ("all", "decomposition"):
        frame = section_decomposition(runs)
        outputs["decomposition"] = frame
        _show("target error by source and by probe cadence [%]", frame)
        _show("means", frame.drop(columns=["batch_size", "loss"]).mean().to_frame("mean"), index=True)

    if wanted in ("all", "stress"):
        sweep, transient, positions = section_stress(runs)
        outputs["stress_sweep"] = sweep
        outputs["stress_transient"] = transient
        outputs["stress_window_position"] = positions
        _show("contamination sweep: mean absolute target error [%]", sweep)
        _show("one injected outlier: weight excursion and how long it lasts", transient)
        _show("where a contaminated window does its damage", positions)

    if wanted in ("all", "endtoend"):
        frame = section_endtoend(arguments.root)
        outputs["endtoend"] = frame
        if len(frame):
            summary = frame.groupby("estimator")[
                ["mae_pct", "p95_pct", "best_map_at_r", "best_precision_at_1", "final_map_at_r"]
            ].agg(["mean", "count"])
            _show("runs trained under each rule", summary, index=True)
            pivot = frame.pivot_table(index=["batch_size", "loss"], columns="estimator", values="best_map_at_r")
            _show("best validation mAP@R per run", pivot, index=True)
        else:
            _show("runs trained under each rule", frame)

    if wanted in ("all", "hpo"):
        frame = section_hpo(arguments.study_root)
        outputs["hpo"] = frame
        if len(frame):
            _show("tuned study: how far a batch's q strays from its local median",
                  frame.groupby("loss")[
                      ["n", "q_above_local_median", "q_above_local_median_10x",
                       "q_below_local_median", "worst_q_over_local_median", "mae_pct"]
                  ].mean(), index=True)
            print(
                f"\npooled over {len(frame)} folds: "
                f"{frame.n.sum()} logged steps, "
                f"worst q/local median {frame.worst_q_over_local_median.max():.4g}"
            )
        else:
            _show("tuned study", frame)

    if arguments.out:
        directory = Path(arguments.out)
        directory.mkdir(parents=True, exist_ok=True)
        for name, frame in outputs.items():
            frame.to_csv(directory / f"{name}.csv", index=False)
        print(f"\nwrote {len(outputs)} tables to {directory}")


if __name__ == "__main__":
    main()
