/**
 * Summarises the per-fold gradient-contribution diagnostics of one study.
 *
 * Each fold of each trial writes a `diagnostics.csv` of
 *
 *     step,epoch,category,name,value,details
 *
 * carrying two families of series. `train/gradient_contribution/*` measures what
 * the regularizer actually did to the gradient — its norm against the
 * supervised norm, their ratio, their cosine similarity — and
 * `train/grad_norm/*` carries the controller state, including
 * `target_regularizer`, the ratio a GradNorm-style controller was asked to hold.
 *
 * There are ~10 GB of these across the archive, so the result is cached next to
 * the study as `gradient_summary.json` and only recomputed when a fold has
 * written something newer.
 */
import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";

const CACHE_NAME = "gradient_summary.json";
const CACHE_VERSION = 3;

/** The series worth carrying, mapped to the short key they get in the output. */
const CONTRIBUTION_SERIES = new Map([
  ["train/gradient_contribution/supervised_norm", "supervised"],
  ["train/gradient_contribution/regularizer_norm", "regularizer"],
  ["train/gradient_contribution/combined_norm", "combined"],
  ["train/gradient_contribution/regularizer_to_supervised_ratio", "ratio"],
  ["train/gradient_contribution/regularizer_fraction", "fraction"],
  ["train/gradient_contribution/cosine_similarity", "cosine"],
  ["train/gradient_contribution/regularizer_weight", "weight"],
  ["train/gradient_contribution/unit_regularizer_to_supervised_ratio", "unitRatio"],
  ["train/gradient_contribution/slade_basis_to_supervised_ratio", "sladeRatio"],
  ["train/gradient_contribution/slade_basis_norm", "sladeNorm"],
  ["train/grad_norm/target_regularizer", "target"],
  ["train/grad_norm/regularizer_weight", "controllerWeight"],
  ["train/grad_norm/supervised_weight", "supervisedWeight"],
  ["train/grad_norm/relative_rate_regularizer", "relativeRate"],
  ["train/grad_norm/target_slade_basis", "sladeTarget"],
]);

/** Series whose spread is only meaningful on a log axis. */
const LOG_SERIES = new Set(["ratio", "unitRatio", "sladeRatio", "supervised", "regularizer"]);

function ascending(values) {
  return [...values].sort((left, right) => left - right);
}

function quantile(sorted, probability) {
  if (sorted.length === 0) return null;
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function round(value) {
  if (value == null || !Number.isFinite(value)) return null;
  return Number(Number(value).toPrecision(4));
}

/**
 * Median, inner-decile spread and drift for one series. Ratios span orders of
 * magnitude, so their spread is reported in decades — a ratio that wanders
 * between 0.1 and 10 has a spread of 2, whichever end you start from.
 */
function describe(values, key) {
  const finite = values.filter((value) => Number.isFinite(value));
  if (finite.length === 0) return null;
  const sorted = ascending(finite);
  const logScale = LOG_SERIES.has(key) && sorted[0] > 0;
  // The inner deciles, not the extremes: with a few dozen samples per fold the
  // min and max are the two noisiest numbers in the series and they were a
  // quarter of the payload.
  const summary = {
    n: finite.length,
    median: round(quantile(sorted, 0.5)),
    p10: round(quantile(sorted, 0.1)),
    p90: round(quantile(sorted, 0.9)),
  };
  if (logScale) {
    const low = quantile(sorted, 0.1);
    const high = quantile(sorted, 0.9);
    summary.decades = low > 0 && high > 0 ? round(Math.log10(high / low)) : null;
  }
  // Steady state: the last quarter of the run, where a controller has settled.
  const tail = finite.slice(Math.floor(finite.length * 0.75));
  summary.late = tail.length > 0 ? round(quantile(ascending(tail), 0.5)) : null;
  return summary;
}

/** Parses one diagnostics.csv into { seriesKey: Map<step, value> }. */
function readDiagnostics(filePath) {
  let text;
  try {
    text = readFileSync(filePath, "utf8");
  } catch {
    return null;
  }
  const series = new Map();
  let cursor = text.indexOf("\n") + 1; // skip the header
  if (cursor <= 0) return null;

  while (cursor < text.length) {
    let end = text.indexOf("\n", cursor);
    if (end === -1) end = text.length;
    const line = text.slice(cursor, end);
    cursor = end + 1;
    // step,epoch,category,name,value,details — name and value are fields 4 and 5.
    const first = line.indexOf(",");
    if (first === -1) continue;
    const second = line.indexOf(",", first + 1);
    if (second === -1) continue;
    const third = line.indexOf(",", second + 1);
    if (third === -1) continue;
    const fourth = line.indexOf(",", third + 1);
    if (fourth === -1) continue;
    const name = line.slice(third + 1, fourth);
    const key = CONTRIBUTION_SERIES.get(name);
    if (key === undefined) continue;
    const fifth = line.indexOf(",", fourth + 1);
    const raw = fifth === -1 ? line.slice(fourth + 1) : line.slice(fourth + 1, fifth);
    const value = Number(raw);
    if (!Number.isFinite(value)) continue;
    const step = Number(line.slice(0, first));
    let bucket = series.get(key);
    if (!bucket) {
      bucket = new Map();
      series.set(key, bucket);
    }
    bucket.set(step, value);
  }
  return series;
}

/**
 * How far the achieved ratio sat from the ratio the controller was targeting,
 * joined step by step. Reported in decades, because a controller that holds
 * 0.5x the target is exactly as wrong as one holding 2x.
 */
function targetDeviation(series) {
  const ratio = series.get("ratio");
  const target = series.get("target");
  if (!ratio || !target) return null;
  const deviations = [];
  for (const [step, achieved] of ratio) {
    const wanted = target.get(step);
    if (wanted == null || !(wanted > 0) || !(achieved > 0)) continue;
    deviations.push(Math.log10(achieved / wanted));
  }
  if (deviations.length === 0) return null;
  const sorted = ascending(deviations);
  const tail = deviations.slice(Math.floor(deviations.length * 0.75));
  return {
    n: deviations.length,
    median: round(quantile(sorted, 0.5)),
    p10: round(quantile(sorted, 0.1)),
    p90: round(quantile(sorted, 0.9)),
    absMedian: round(quantile(ascending(deviations.map(Math.abs)), 0.5)),
    late: tail.length > 0 ? round(quantile(ascending(tail), 0.5)) : null,
  };
}

function summariseFold(filePath) {
  const series = readDiagnostics(filePath);
  if (!series || series.size === 0) return null;
  const summary = {};
  for (const [key, bucket] of series) {
    const described = describe([...bucket.values()], key);
    if (described) summary[key] = described;
  }
  const deviation = targetDeviation(series);
  if (deviation) summary.targetDeviation = deviation;
  return Object.keys(summary).length > 0 ? summary : null;
}

function foldNumber(name) {
  const match = /fold_(\d+)/.exec(name);
  return match ? Number(match[1]) : null;
}

function trialNumber(name) {
  const match = /trial_(\d+)/.exec(name);
  return match ? Number(match[1]) : null;
}

function listDirectories(directory) {
  try {
    return readdirSync(directory, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .map((entry) => entry.name);
  } catch {
    return [];
  }
}

/** Every fold diagnostics file of a study, with the trial and fold it belongs to. */
function findFoldFiles(studyDirectory) {
  const found = [];
  for (const trialName of listDirectories(studyDirectory)) {
    const trial = trialNumber(trialName);
    if (trial == null) continue;
    const trialPath = path.join(studyDirectory, trialName);
    for (const cvName of listDirectories(trialPath)) {
      if (!cvName.startsWith("cv_")) continue;
      const cvPath = path.join(trialPath, cvName);
      for (const foldName of listDirectories(cvPath)) {
        const fold = foldNumber(foldName);
        if (fold == null) continue;
        const filePath = path.join(cvPath, foldName, "diagnostics.csv");
        if (existsSync(filePath)) found.push({ trial, fold, filePath });
      }
    }
  }
  return found;
}

function newestMtime(files) {
  let newest = 0;
  for (const entry of files) {
    try {
      newest = Math.max(newest, statSync(entry.filePath).mtimeMs);
    } catch {
      // A file that vanished mid-scan simply does not contribute.
    }
  }
  return newest;
}

/**
 * Gradient summaries for one study, as { trialNumber: { foldNumber: summary } }.
 * Returns null when the study logged no gradient diagnostics at all.
 */
export function gradientSummary(studyDirectory) {
  const files = findFoldFiles(studyDirectory);
  if (files.length === 0) return null;

  const cachePath = path.join(studyDirectory, CACHE_NAME);
  const newest = newestMtime(files);
  if (existsSync(cachePath)) {
    try {
      const cached = JSON.parse(readFileSync(cachePath, "utf8"));
      if (
        cached?.version === CACHE_VERSION &&
        cached?.files === files.length &&
        Number(cached?.newest) >= newest
      ) {
        return cached.trials ?? null;
      }
    } catch {
      // A damaged cache is simply rebuilt.
    }
  }

  const trials = {};
  for (const { trial, fold, filePath } of files) {
    const summary = summariseFold(filePath);
    if (!summary) continue;
    if (!trials[trial]) trials[trial] = {};
    trials[trial][fold] = summary;
  }

  try {
    writeFileSync(
      cachePath,
      JSON.stringify({ version: CACHE_VERSION, files: files.length, newest, trials }),
    );
  } catch {
    // A read-only archive just means recomputing next time.
  }
  return Object.keys(trials).length > 0 ? trials : null;
}

export { CACHE_NAME as GRADIENT_CACHE_NAME };
