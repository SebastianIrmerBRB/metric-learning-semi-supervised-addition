/**
 * Phase 0 range refinement.
 *
 * A random-search sweep is not read for its best value — it is read for the
 * *shape* of the response, and the shape decides whether the Phase 1 range
 * should be narrowed, widened or left alone. This module turns that reading
 * into arithmetic so the same rule is applied to every parameter.
 *
 * Three signals, following Bischl et al.'s iterative range refinement:
 *
 *   1. Hard failures — trials that never produced an objective, never improved
 *      on their first epoch, or landed in the study's failure tail. Regions
 *      made of these are dead weight for any sampler and can be cut.
 *   2. A tolerance band around the best — everything within epsilon of the best
 *      observed value is "as good as the best", because below the noise floor
 *      the ranking is not real. The good region is the parameter interval that
 *      band spans.
 *   3. A boundary hit — if the good region reaches an edge of the searched
 *      range, or the response is still flat there, the range was wrong and has
 *      to be extended in that direction before anything else is concluded.
 *
 * Everything is computed in *shortfall* space, `value - best(study)`, so trials
 * from studies that score at different levels can be pooled without pretending
 * their absolute objectives are comparable.
 */

import type {
  ArchiveParamValue,
  ArchiveStudy,
  ArchiveTrial,
  GradientFold,
  SeriesSummary,
} from "@/lib/sorted-studies";

// ------------------------------------------------------------------- settings

export type EpsilonMode = "absolute" | "relative" | "noise";
export type DeadMode = "fence" | "absolute" | "off";

export interface RangeSettings {
  epsilonMode: EpsilonMode;
  /** Tolerance in objective units, for `epsilonMode: "absolute"`. */
  epsilonAbsolute: number;
  /** Tolerance as a percentage of the study's best, for `"relative"`. */
  epsilonPercent: number;
  /** Trials at or above this quantile define the measured noise floor. */
  noiseQuantile: number;
  /** Standard errors of the fold mean that the noise band spans. */
  noiseMultiplier: number;
  deadMode: DeadMode;
  /** Tukey multiplier for the lower fence, for `deadMode: "fence"`. */
  deadFence: number;
  /** Objective at or below which a trial counts as collapsed, for `"absolute"`. */
  deadFloor: number;
  /** Share of the searched span that still counts as sitting "at the edge". */
  boundaryTolerance: number;
  /** Margin below the good region: decades on a log axis, else a share of it. */
  marginLow: number;
  /** Margin above the good region, same units. */
  marginHigh: number;
  /** Bins used for the trend line and the edge-flatness test. */
  bins: number;
  /** A good region this small a share of the searched span may be narrowed. */
  narrowShare: number;
  /** Snap recommended bounds outward to readable numbers. */
  roundBounds: boolean;
}

export const DEFAULT_RANGE_SETTINGS: RangeSettings = {
  epsilonMode: "noise",
  epsilonAbsolute: 0.01,
  epsilonPercent: 1.5,
  noiseQuantile: 0.9,
  noiseMultiplier: 2,
  deadMode: "fence",
  deadFence: 1.5,
  deadFloor: 0.1,
  boundaryTolerance: 0.1,
  marginLow: 0.5,
  marginHigh: 0.5,
  bins: 8,
  narrowShare: 0.6,
  roundBounds: true,
};

/** A bin needs this many trials before its median is allowed to decide anything. */
const MINIMUM_BIN_TRIALS = 2;
/** A choice sampled fewer times than this cannot be declared dead. */
const MINIMUM_CHOICE_TRIALS = 3;
/** Best epoch at or beyond this share of the run means it was still improving. */
const TRUNCATION_SHARE = 0.8;

// ------------------------------------------------------------------ statistics

export function ascending(values: number[]): number[] {
  return [...values].sort((left, right) => left - right);
}

export function quantileSorted(sorted: number[], probability: number): number {
  if (sorted.length === 0) return Number.NaN;
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

export function medianOf(values: number[]): number {
  return quantileSorted(ascending(values), 0.5);
}

export function meanOf(values: number[]): number {
  return values.length === 0
    ? Number.NaN
    : values.reduce((total, value) => total + value, 0) / values.length;
}

export function standardDeviationOf(values: number[]): number {
  if (values.length < 2) return Number.NaN;
  const average = meanOf(values);
  return Math.sqrt(
    values.reduce((total, value) => total + (value - average) ** 2, 0) /
      (values.length - 1),
  );
}

export interface NoiseModel {
  /** Mean of the per-trial fold standard deviations, before any correction. */
  foldSigma: number | null;
  /** Residual spread once the shared fold effect is removed. */
  residualSigma: number | null;
  /** Standard error of one trial's fold mean. */
  standardError: number | null;
  /** Standard error of the difference between two trials' means. */
  comparisonError: number | null;
  /** Trials and folds the decomposition was fitted on. */
  trials: number;
  folds: number;
}

/**
 * Every trial of a study is scored on the *same* cross-validation folds, so a
 * fold that is simply harder shifts every trial together and cancels when two
 * trials are compared. Treating the raw fold spread as noise would therefore
 * charge that shared difficulty to each trial twice over — on this archive it
 * inflates the tolerance by roughly an order of magnitude.
 *
 * The two-way decomposition below removes it: with x[t][f] the score of trial t
 * on fold f, the residual
 *
 *     e[t][f] = x[t][f] - mean_f(x[t]) - mean_t(x[f]) + mean(x)
 *
 * carries only the trial x fold interaction, which is the part that genuinely
 * varies between repeats. Its standard deviation over (T-1)(F-1) degrees of
 * freedom gives the standard error of one trial's mean, sigma/sqrt(F), and of
 * the difference between two trial means, sigma * sqrt(2/F) — the latter being
 * the scale a "within epsilon of the best" band is really measured on.
 */
export function fitNoiseModel(rows: number[][]): NoiseModel {
  const perTrialSigma = rows
    .map((row) => standardDeviationOf(row))
    .filter((value) => Number.isFinite(value));
  const foldSigma = perTrialSigma.length > 0 ? meanOf(perTrialSigma) : null;

  // Only trials measured on the same number of folds can share a decomposition.
  const widths = new Map<number, number>();
  for (const row of rows) widths.set(row.length, (widths.get(row.length) ?? 0) + 1);
  let folds = 0;
  let best = 0;
  for (const [width, count] of widths) {
    if (width > 1 && count > best) {
      best = count;
      folds = width;
    }
  }
  const matrix = rows.filter((row) => row.length === folds);
  const trials = matrix.length;
  if (trials < 2 || folds < 2) {
    const fallback =
      foldSigma != null && folds > 1 ? foldSigma / Math.sqrt(folds) : null;
    return {
      foldSigma,
      residualSigma: null,
      standardError: fallback,
      comparisonError: fallback == null ? null : fallback * Math.SQRT2,
      trials,
      folds,
    };
  }

  const rowMeans = matrix.map((row) => meanOf(row));
  const columnMeans = Array.from({ length: folds }, (_, fold) =>
    meanOf(matrix.map((row) => row[fold])),
  );
  const grandMean = meanOf(rowMeans);
  let sumSquares = 0;
  for (const [trial, row] of matrix.entries()) {
    for (const [fold, value] of row.entries()) {
      const residual = value - rowMeans[trial] - columnMeans[fold] + grandMean;
      sumSquares += residual * residual;
    }
  }
  const degreesOfFreedom = (trials - 1) * (folds - 1);
  const residualSigma = Math.sqrt(sumSquares / degreesOfFreedom);
  return {
    foldSigma,
    residualSigma,
    standardError: residualSigma / Math.sqrt(folds),
    comparisonError: residualSigma * Math.sqrt(2 / folds),
    trials,
    folds,
  };
}

/**
 * The objective values of a trial's folds.
 *
 * Deliberately local rather than imported: this module is also run directly by
 * scripts/verify-phase0.mts under Node's type stripping, which does not resolve
 * the `@/` path alias, so phase0 keeps every non-type import to itself.
 */
function foldValues(trial: ArchiveTrial): number[] {
  return (trial.folds ?? [])
    .map((fold) => fold.value)
    .filter((value) => Number.isFinite(value));
}

function numeric(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

// ------------------------------------------------------------ trial assessment

export type TrialHealth =
  /** Produced a usable objective inside the study's live range. */
  | "alive"
  /** In the study's failure tail, or under the absolute collapse floor. */
  | "dead"
  /** Best validation epoch was the first one — the run never improved. */
  | "stalled"
  /** No objective at all: Optuna recorded FAIL, PRUNED or never finished. */
  | "failed";

export interface TrialAssessment {
  trial: ArchiveTrial;
  /** Objective, flipped where needed so larger is always better. */
  value: number | null;
  /** `value - best(study)`; zero for the study's best trial, negative below it. */
  shortfall: number | null;
  health: TrialHealth;
  /** Standard deviation across cross-validation folds, where recorded. */
  spread: number | null;
  /** Best epoch as a share of the last epoch; near 1 means still improving. */
  epochShare: number | null;
  /** Inside the tolerance band around the study's best. */
  inBand: boolean;
}

export interface StudyAssessment {
  study: ArchiveStudy;
  trials: TrialAssessment[];
  alive: TrialAssessment[];
  keep: TrialAssessment[];
  best: number | null;
  bestTrial: TrialAssessment | null;
  epsilon: number;
  /** Plain-language account of where epsilon came from, for the caption. */
  epsilonSource: string;
  deadFloor: number | null;
  /** The paired fold decomposition fitted on the study's strongest trials. */
  noise: NoiseModel;
  /** Standard error of the difference between two trials — the band's natural unit. */
  noiseFloor: number | null;
  /** Best minus median over the live trials: how much this sweep actually moved. */
  effectSize: number | null;
  /**
   * Effect size in units of epsilon. At or below 1 the sweep cannot tell its
   * own best config from its median one, so no range conclusion is supported.
   */
  resolution: number | null;
  /** Share of in-band trials whose best epoch sat at the very end of the run. */
  truncatedShare: number | null;
  counts: Record<TrialHealth, number>;
}

function orientedValue(study: ArchiveStudy, trial: ArchiveTrial): number | null {
  if (trial.state !== "COMPLETE" || typeof trial.value !== "number") return null;
  return study.direction === "minimize" ? -trial.value : trial.value;
}

function foldSpread(trial: ArchiveTrial): number | null {
  const values = foldValues(trial);
  if (values.length < 2) return null;
  const spread = standardDeviationOf(values);
  return Number.isFinite(spread) ? spread : null;
}

function epochShare(trial: ArchiveTrial): number | null {
  const selected = numeric(trial.metrics.selected_epoch);
  const last = numeric(trial.metrics.last_epoch);
  if (selected == null || last == null || last <= 0) return null;
  return selected / last;
}

/**
 * The objective below which a trial is treated as a hard failure rather than a
 * weak result. The Tukey fence is the default because it adapts to a study's
 * own scale; an absolute floor is there for the case where the failure mode is
 * known (a collapsed embedding sits at chance level, not at a relative
 * distance from the rest of the sweep).
 */
function deadFloorFor(values: number[], settings: RangeSettings): number | null {
  if (settings.deadMode === "off") return null;
  if (settings.deadMode === "absolute") return settings.deadFloor;
  if (values.length < 8) return null;
  const sorted = ascending(values);
  const firstQuartile = quantileSorted(sorted, 0.25);
  const interQuartile = quantileSorted(sorted, 0.75) - firstQuartile;
  if (!(interQuartile > 0)) return null;
  return firstQuartile - settings.deadFence * interQuartile;
}

export function assessStudy(
  study: ArchiveStudy,
  settings: RangeSettings,
): StudyAssessment {
  const scoredValues = study.trials
    .map((trial) => orientedValue(study, trial))
    .filter((value): value is number => value != null);
  const deadFloor = deadFloorFor(scoredValues, settings);

  const trials: TrialAssessment[] = study.trials.map((trial) => {
    const value = orientedValue(study, trial);
    const spread = foldSpread(trial);
    const share = epochShare(trial);
    let health: TrialHealth;
    if (value == null) health = "failed";
    else if (deadFloor != null && value <= deadFloor) health = "dead";
    else if (numeric(trial.metrics.selected_epoch) === 0) health = "stalled";
    else health = "alive";
    return {
      trial,
      value,
      shortfall: null,
      health,
      spread,
      epochShare: share,
      inBand: false,
    };
  });

  const alive = trials.filter(
    (assessment) => assessment.health === "alive" && assessment.value != null,
  );
  const best =
    alive.length > 0 ? Math.max(...alive.map((assessment) => assessment.value as number)) : null;

  // The noise floor is the cross-validation spread of the trials that are
  // actually in contention. It is fold-to-fold, not seed-to-seed: every study
  // in this archive ran a single data split seed, so it measures how much the
  // metric moves between folds of one split, which is the closest available
  // stand-in and generally the smaller of the two.
  const liveValues = alive.map((assessment) => assessment.value as number);
  const contenderCut =
    liveValues.length > 0
      ? quantileSorted(ascending(liveValues), settings.noiseQuantile)
      : null;
  const contenders = alive.filter(
    (assessment) => contenderCut != null && (assessment.value as number) >= contenderCut,
  );
  const noise = fitNoiseModel(
    contenders
      .map((assessment) => foldValues(assessment.trial))
      .filter((values) => values.length > 1),
  );
  const noiseFloor = noise.comparisonError;

  let epsilon: number;
  let epsilonSource: string;
  if (settings.epsilonMode === "absolute") {
    epsilon = settings.epsilonAbsolute;
    epsilonSource = `fixed ${settings.epsilonAbsolute} in objective units`;
  } else if (settings.epsilonMode === "relative") {
    epsilon = best != null ? (settings.epsilonPercent / 100) * best : settings.epsilonAbsolute;
    epsilonSource = `${settings.epsilonPercent}% of this study's best`;
  } else if (noiseFloor != null) {
    epsilon = settings.noiseMultiplier * noiseFloor;
    epsilonSource = `${settings.noiseMultiplier}x the paired standard error between two trials (${noise.trials} contenders, ${noise.folds} folds)`;
  } else {
    epsilon = settings.epsilonAbsolute;
    epsilonSource = `fixed ${settings.epsilonAbsolute} (no fold spread recorded)`;
  }

  for (const assessment of trials) {
    if (assessment.value == null || best == null) continue;
    assessment.shortfall = assessment.value - best;
    assessment.inBand = assessment.health === "alive" && assessment.shortfall >= -epsilon;
  }

  const keep = trials.filter((assessment) => assessment.inBand);
  const truncatedCandidates = keep
    .map((assessment) => assessment.epochShare)
    .filter((value): value is number => value != null);
  const truncatedShare =
    truncatedCandidates.length > 0
      ? truncatedCandidates.filter((share) => share >= TRUNCATION_SHARE).length /
        truncatedCandidates.length
      : null;

  const counts: Record<TrialHealth, number> = {
    alive: 0,
    dead: 0,
    stalled: 0,
    failed: 0,
  };
  for (const assessment of trials) counts[assessment.health] += 1;

  const effectSize =
    best != null && liveValues.length > 1 ? best - medianOf(liveValues) : null;

  return {
    study,
    trials,
    alive,
    keep,
    best,
    bestTrial:
      best == null
        ? null
        : (alive.find((assessment) => assessment.value === best) ?? null),
    epsilon,
    epsilonSource,
    deadFloor,
    noise,
    noiseFloor,
    effectSize,
    resolution: effectSize != null && epsilon > 0 ? effectSize / epsilon : null,
    truncatedShare,
    counts,
  };
}

// ------------------------------------------------------- parameter transforms

/** Log-scaled parameters are reasoned about in decades throughout. */
/**
 * Mantissas a rounded bound is allowed to land on. `0.000035714…` is not a
 * range anybody types into a config; `3e-5` is, and it says exactly as much.
 */
const NICE_MANTISSAS = [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];

/**
 * Snaps a bound to a readable number, always outward.
 *
 * Rounding a recommendation inward would quietly exclude trials the tolerance
 * band had accepted, so the low bound only ever moves down and the high bound
 * only ever moves up. The range therefore stays a superset of the exact one.
 */
export function niceBound(value: number, direction: "down" | "up"): number {
  if (!Number.isFinite(value) || value === 0) return value;
  if (value < 0) return -niceBound(-value, direction === "down" ? "up" : "down");

  const exponent = Math.floor(Math.log10(value));
  const scale = 10 ** exponent;
  const mantissa = value / scale;
  const candidates = direction === "down"
    ? [...NICE_MANTISSAS].reverse().filter((step) => step <= mantissa + 1e-12)
    : NICE_MANTISSAS.filter((step) => step >= mantissa - 1e-12);
  const snapped = candidates.length > 0
    ? candidates[0]
    : direction === "down"
      ? NICE_MANTISSAS[0]
      : NICE_MANTISSAS[NICE_MANTISSAS.length - 1];
  return Number((snapped * scale).toPrecision(3));
}

export function toAxis(value: number, log: boolean): number {
  return log ? Math.log10(Math.max(value, Number.MIN_VALUE)) : value;
}

export function fromAxis(value: number, log: boolean): number {
  return log ? 10 ** value : value;
}

// ------------------------------------------------------------------- findings

export type Verdict = "extend-low" | "extend-high" | "extend-both" | "narrow" | "keep";

export const VERDICT_LABELS: Record<Verdict, string> = {
  "extend-low": "Extend downward",
  "extend-high": "Extend upward",
  "extend-both": "Extend both ways",
  narrow: "Narrow",
  keep: "Keep as is",
};

export interface RangeBin {
  /** Bin edges in axis space (log10 for log parameters). */
  from: number;
  to: number;
  count: number;
  deadCount: number;
  /** Median shortfall of the bin's live trials, or null when it holds none. */
  median: number | null;
  best: number | null;
}

export interface NumericFinding {
  kind: "numeric";
  name: string;
  log: boolean;
  isInt: boolean;
  /** The range Phase 0 actually searched, in natural units. */
  searched: [number, number] | null;
  /** The interval the in-band trials span, in natural units. */
  good: [number, number] | null;
  bestValue: number | null;
  /** The Phase 1 range this rule recommends, in natural units. */
  recommended: [number, number] | null;
  verdict: Verdict;
  reasons: string[];
  touchesLow: boolean;
  touchesHigh: boolean;
  flatLow: boolean;
  flatHigh: boolean;
  bestAtEdge: "low" | "high" | null;
  /** Good span over searched span, in axis space. */
  goodShare: number | null;
  /** Share of bins outside the good region that are clearly degraded or dead. */
  degradedOutside: number | null;
  /** True when the recommendation leaves the range Phase 0 actually searched. */
  beyondSearched: boolean;
  /** Spread of the binned trend: how much moving this parameter alone moved the metric. */
  effectSize: number | null;
  /**
   * False when that spread is inside the tolerance band. The response is then
   * flat because the sweep cannot see the parameter, not because the parameter
   * does nothing — a distinction that needs seed repeats, not a narrower range.
   */
  resolvable: boolean;
  keepCount: number;
  sampleCount: number;
  deadCount: number;
  clampedLow: boolean;
  bins: RangeBin[];
  studyCount: number;
}

export interface ChoiceFinding {
  value: ArchiveParamValue;
  label: string;
  sampleCount: number;
  keepCount: number;
  deadCount: number;
  bestShortfall: number | null;
  /** Never reached the band despite being sampled often enough to judge. */
  drop: boolean;
}

export interface CategoricalFinding {
  kind: "categorical";
  name: string;
  choices: ChoiceFinding[];
  surviving: ArchiveParamValue[];
  dropped: ArchiveParamValue[];
  verdict: Verdict;
  reasons: string[];
  keepCount: number;
  sampleCount: number;
  effectSize: number | null;
  resolvable: boolean;
  studyCount: number;
}

export type RangeFinding = NumericFinding | CategoricalFinding;

export interface ParameterSample {
  x: number;
  /** NaN for a trial that never produced an objective. */
  shortfall: number;
  health: TrialHealth;
  inBand: boolean;
  /** Standard error of this trial's own fold mean, for its error bar. */
  error: number | null;
  study: ArchiveStudy;
  trialNumber: number;
}

/**
 * Every trial that sampled one parameter, across the given studies, in the
 * shortfall space the plots and the range rules both work in.
 */
export function collectSamples(
  name: string,
  assessments: StudyAssessment[],
  log: boolean,
): ParameterSample[] {
  const samples: ParameterSample[] = [];
  for (const assessment of assessments) {
    const searched = assessment.study.parameters.some(
      (candidate) => candidate.name === name && candidate.searched,
    );
    if (!searched) continue;
    for (const trial of assessment.trials) {
      const x = numeric(trial.trial.params[name]);
      if (x == null) continue;
      if (log && x <= 0) continue;
      const folds = foldValues(trial.trial).length;
      samples.push({
        x,
        // A failed trial still marks its region as hostile; it just has no height.
        shortfall: trial.shortfall ?? Number.NaN,
        health: trial.health,
        inBand: trial.inBand,
        error: trial.spread != null && folds > 1 ? trial.spread / Math.sqrt(folds) : null,
        study: assessment.study,
        trialNumber: trial.trial.number,
      });
    }
  }
  return samples;
}

type Sample = ParameterSample;

interface CategoricalSample {
  value: ArchiveParamValue;
  shortfall: number;
  health: TrialHealth;
  inBand: boolean;
}

/** Merges what each study declared about one parameter into a single spec. */
function parameterSpec(assessments: StudyAssessment[], name: string) {
  let low: number | null = null;
  let high: number | null = null;
  let logVotes = 0;
  let intVotes = 0;
  let categoricalVotes = 0;
  let numericVotes = 0;
  let studyCount = 0;
  const choices: ArchiveParamValue[] = [];
  const seenChoices = new Set<string>();

  for (const assessment of assessments) {
    const parameter = assessment.study.parameters.find(
      (candidate) => candidate.name === name && candidate.searched,
    );
    if (!parameter) continue;
    studyCount += 1;
    if (parameter.kind === "categorical") {
      categoricalVotes += 1;
      for (const choice of parameter.choices ?? []) {
        const key = JSON.stringify(choice);
        if (!seenChoices.has(key)) {
          seenChoices.add(key);
          choices.push(choice);
        }
      }
      continue;
    }
    numericVotes += 1;
    if (parameter.log) logVotes += 1;
    if (parameter.distribution === "int") intVotes += 1;
    if (parameter.low != null) low = low == null ? parameter.low : Math.min(low, parameter.low);
    if (parameter.high != null) {
      high = high == null ? parameter.high : Math.max(high, parameter.high);
    }
  }

  return {
    studyCount,
    categorical: categoricalVotes > numericVotes,
    log: logVotes > numericVotes / 2,
    isInt: intVotes > numericVotes / 2,
    searched: low != null && high != null && high > low ? ([low, high] as [number, number]) : null,
    choices,
  };
}

function binSamples(
  samples: Sample[],
  domain: [number, number],
  log: boolean,
  binCount: number,
): RangeBin[] {
  const [low, high] = domain;
  const width = (high - low) / binCount;
  if (!(width > 0)) return [];
  const bins: RangeBin[] = Array.from({ length: binCount }, (_, index) => ({
    from: low + index * width,
    to: low + (index + 1) * width,
    count: 0,
    deadCount: 0,
    median: null,
    best: null,
  }));
  const collected: number[][] = bins.map(() => []);

  for (const sample of samples) {
    const axis = toAxis(sample.x, log);
    const index = Math.min(binCount - 1, Math.max(0, Math.floor((axis - low) / width)));
    bins[index].count += 1;
    if (sample.health === "alive") collected[index].push(sample.shortfall);
    else bins[index].deadCount += 1;
  }

  for (const [index, values] of collected.entries()) {
    if (values.length === 0) continue;
    bins[index].median = medianOf(values);
    bins[index].best = Math.max(...values);
  }
  return bins;
}

function findNumeric(
  name: string,
  assessments: StudyAssessment[],
  settings: RangeSettings,
  spec: ReturnType<typeof parameterSpec>,
): NumericFinding {
  const { log, isInt } = spec;
  const samples = collectSamples(name, assessments, log);
  let bestShortfall = -Infinity;
  let bestValue: number | null = null;
  for (const sample of samples) {
    if (sample.inBand && sample.shortfall > bestShortfall) {
      bestShortfall = sample.shortfall;
      bestValue = sample.x;
    }
  }

  const keep = samples.filter((sample) => sample.inBand);
  const observed = samples.map((sample) => sample.x);
  const searched =
    spec.searched ??
    (observed.length > 0
      ? ([Math.min(...observed), Math.max(...observed)] as [number, number])
      : null);

  const good: [number, number] | null =
    keep.length > 0
      ? [Math.min(...keep.map((s) => s.x)), Math.max(...keep.map((s) => s.x))]
      : null;

  const axisSearched: [number, number] | null = searched
    ? [toAxis(searched[0], log), toAxis(searched[1], log)]
    : null;
  const searchedSpan = axisSearched ? axisSearched[1] - axisSearched[0] : null;
  const axisGood: [number, number] | null = good
    ? [toAxis(good[0], log), toAxis(good[1], log)]
    : null;
  const goodSpan = axisGood ? axisGood[1] - axisGood[0] : null;

  const bins =
    axisSearched && searchedSpan && searchedSpan > 0
      ? binSamples(samples, axisSearched, log, Math.max(3, settings.bins))
      : [];

  const tolerance =
    searchedSpan != null ? settings.boundaryTolerance * searchedSpan : null;
  const touchesLow =
    axisGood != null && axisSearched != null && tolerance != null
      ? axisGood[0] - axisSearched[0] <= tolerance
      : false;
  const touchesHigh =
    axisGood != null && axisSearched != null && tolerance != null
      ? axisSearched[1] - axisGood[1] <= tolerance
      : false;

  const epsilon = meanOf(assessments.map((assessment) => assessment.epsilon));
  const edgeIsFlat = (bin: RangeBin | undefined) =>
    bin != null &&
    bin.count - bin.deadCount >= MINIMUM_BIN_TRIALS &&
    bin.median != null &&
    bin.median >= -epsilon;
  const flatLow = edgeIsFlat(bins[0]);
  const flatHigh = edgeIsFlat(bins[bins.length - 1]);

  let bestAtEdge: "low" | "high" | null = null;
  if (bestValue != null && axisSearched != null && tolerance != null) {
    const axisBest = toAxis(bestValue, log);
    if (axisBest - axisSearched[0] <= tolerance) bestAtEdge = "low";
    else if (axisSearched[1] - axisBest <= tolerance) bestAtEdge = "high";
  }

  const outside = bins.filter(
    (bin) => axisGood == null || bin.to <= axisGood[0] || bin.from >= axisGood[1],
  );
  const judgedOutside = outside.filter(
    (bin) => bin.count >= MINIMUM_BIN_TRIALS,
  );
  const degradedOutside =
    judgedOutside.length > 0
      ? judgedOutside.filter(
          (bin) => bin.median == null || bin.median < -2 * epsilon,
        ).length / judgedOutside.length
      : null;

  // How far the binned trend travels is the parameter's own effect. Comparing
  // it with epsilon separates "flat because it does not matter" from "flat
  // because the noise floor is wider than the effect".
  const binMedians = bins
    .filter((bin) => bin.count - bin.deadCount >= MINIMUM_BIN_TRIALS && bin.median != null)
    .map((bin) => bin.median as number);
  const effectSize =
    binMedians.length >= 2 ? Math.max(...binMedians) - Math.min(...binMedians) : null;
  const resolvable = effectSize != null && effectSize > epsilon;

  const reasons: string[] = [];
  if (good == null) {
    reasons.push(
      keep.length === 0 && samples.length > 0
        ? "no trial reached the tolerance band — the study has no live best to measure against"
        : "no trial in the selection sampled this parameter",
    );
  }
  if (effectSize != null && !resolvable) {
    reasons.push(
      `the whole response spans ${effectSize.toFixed(4)}, inside the ${epsilon.toFixed(4)} tolerance — this sweep cannot resolve the parameter`,
    );
  }
  const extendLow = touchesLow || flatLow || bestAtEdge === "low";
  const extendHigh = touchesHigh || flatHigh || bestAtEdge === "high";
  if (bestAtEdge === "low") reasons.push("the best trial sits at the lower edge");
  if (bestAtEdge === "high") reasons.push("the best trial sits at the upper edge");
  if (touchesLow && bestAtEdge !== "low") reasons.push("the band reaches the lower edge");
  if (touchesHigh && bestAtEdge !== "high") reasons.push("the band reaches the upper edge");
  if (flatLow) reasons.push("the response is still flat at the lower edge");
  if (flatHigh) reasons.push("the response is still flat at the upper edge");

  const goodShare = goodSpan != null && searchedSpan ? goodSpan / searchedSpan : null;
  let verdict: Verdict;
  if (extendLow && extendHigh) verdict = "extend-both";
  else if (extendLow) verdict = "extend-low";
  else if (extendHigh) verdict = "extend-high";
  else if (
    goodShare != null &&
    goodShare <= settings.narrowShare &&
    degradedOutside != null &&
    degradedOutside >= 0.6
  ) {
    verdict = "narrow";
    reasons.push(
      `the band covers ${Math.round(goodShare * 100)}% of the searched span and the rest is degraded or dead`,
    );
  } else {
    verdict = "keep";
    reasons.push("the band is bounded by degradation on both sides");
  }

  // The margin is applied to the good region, not the searched range, so an
  // edge-touching parameter naturally lands outside what Phase 0 covered —
  // which is the whole point of the extension rule.
  let recommended: [number, number] | null = null;
  let clampedLow = false;
  if (axisGood != null) {
    const base = log ? 1 : goodSpan && goodSpan > 0 ? goodSpan : (searchedSpan ?? 1) * 0.1;
    let low = fromAxis(axisGood[0] - settings.marginLow * base, log);
    let high = fromAxis(axisGood[1] + settings.marginHigh * base, log);
    if (!log && searched != null && searched[0] >= 0 && low < 0) {
      low = 0;
      clampedLow = true;
    }
    if (settings.roundBounds) {
      low = niceBound(low, "down");
      high = niceBound(high, "up");
      if (!log && searched != null && searched[0] >= 0 && low < 0) {
        low = 0;
        clampedLow = true;
      }
    }
    if (isInt) {
      low = Math.floor(low);
      high = Math.ceil(high);
      if (searched != null && searched[0] >= 1 && low < 1) {
        low = 1;
        clampedLow = true;
      }
    }
    recommended = [low, high];
  }

  const beyondSearched =
    recommended != null &&
    searched != null &&
    (recommended[0] < searched[0] || recommended[1] > searched[1]);

  return {
    kind: "numeric",
    name,
    log,
    isInt,
    searched,
    good,
    bestValue,
    recommended,
    verdict,
    reasons,
    touchesLow,
    touchesHigh,
    flatLow,
    flatHigh,
    bestAtEdge,
    goodShare,
    degradedOutside,
    beyondSearched,
    keepCount: keep.length,
    sampleCount: samples.length,
    deadCount: samples.filter((sample) => sample.health !== "alive").length,
    effectSize,
    resolvable,
    clampedLow,
    bins,
    studyCount: spec.studyCount,
  };
}

function findCategorical(
  name: string,
  assessments: StudyAssessment[],
  spec: ReturnType<typeof parameterSpec>,
): CategoricalFinding {
  const samples: CategoricalSample[] = [];
  for (const assessment of assessments) {
    const searched = assessment.study.parameters.some(
      (candidate) => candidate.name === name && candidate.searched,
    );
    if (!searched) continue;
    for (const trial of assessment.trials) {
      const value = trial.trial.params[name];
      if (value === undefined || value === null) continue;
      samples.push({
        value,
        shortfall: trial.shortfall ?? Number.NaN,
        health: trial.health,
        inBand: trial.inBand,
      });
    }
  }

  const byChoice = new Map<string, ChoiceFinding>();
  const order: string[] = [];
  const register = (value: ArchiveParamValue) => {
    const key = JSON.stringify(value);
    if (!byChoice.has(key)) {
      byChoice.set(key, {
        value,
        label: value === null ? "null" : String(value),
        sampleCount: 0,
        keepCount: 0,
        deadCount: 0,
        bestShortfall: null,
        drop: false,
      });
      order.push(key);
    }
    return byChoice.get(key) as ChoiceFinding;
  };
  for (const choice of spec.choices) register(choice);

  for (const sample of samples) {
    const entry = register(sample.value);
    entry.sampleCount += 1;
    if (sample.inBand) entry.keepCount += 1;
    if (sample.health !== "alive") entry.deadCount += 1;
    if (Number.isFinite(sample.shortfall)) {
      entry.bestShortfall =
        entry.bestShortfall == null
          ? sample.shortfall
          : Math.max(entry.bestShortfall, sample.shortfall);
    }
  }

  const choices = order.map((key) => byChoice.get(key) as ChoiceFinding);
  const choiceBest = choices
    .map((choice) => choice.bestShortfall)
    .filter((value): value is number => value != null);
  const effectSize =
    choiceBest.length >= 2 ? Math.max(...choiceBest) - Math.min(...choiceBest) : null;
  const epsilon = meanOf(assessments.map((assessment) => assessment.epsilon));
  const resolvable = effectSize != null && effectSize > epsilon;

  // Dropping a choice is a claim that it is worse than the survivors. That
  // claim needs the sweep to be able to tell choices apart in the first place.
  for (const choice of choices) {
    choice.drop =
      resolvable && choice.sampleCount >= MINIMUM_CHOICE_TRIALS && choice.keepCount === 0;
  }
  const dropped = choices.filter((choice) => choice.drop).map((choice) => choice.value);
  const surviving = choices
    .filter((choice) => !choice.drop && choice.sampleCount > 0)
    .map((choice) => choice.value);

  const reasons: string[] = [];
  if (effectSize != null && !resolvable) {
    reasons.push(
      `the best and worst choice differ by ${effectSize.toFixed(4)}, inside the ${epsilon.toFixed(4)} tolerance — keep every choice`,
    );
  } else if (dropped.length > 0) {
    reasons.push(
      `${dropped.length} choice${dropped.length === 1 ? "" : "s"} never reached the band despite being sampled`,
    );
  } else {
    reasons.push("every sampled choice reached the band at least once");
  }
  const unsampled = choices.filter((choice) => choice.sampleCount === 0);
  if (unsampled.length > 0) {
    reasons.push(`${unsampled.length} declared choice(s) were never sampled`);
  }

  return {
    kind: "categorical",
    name,
    choices,
    surviving,
    dropped,
    verdict: dropped.length > 0 ? "narrow" : "keep",
    reasons,
    keepCount: samples.filter((sample) => sample.inBand).length,
    sampleCount: samples.length,
    effectSize,
    resolvable,
    studyCount: spec.studyCount,
  };
}

export function findRange(
  name: string,
  assessments: StudyAssessment[],
  settings: RangeSettings,
): RangeFinding {
  const spec = parameterSpec(assessments, name);
  return spec.categorical
    ? findCategorical(name, assessments, spec)
    : findNumeric(name, assessments, settings, spec);
}

/**
 * The `spaces` block for a Phase 1 config, in the exact shape the repository's
 * `configs/runs/*.json` files use, so a finding can be pasted straight in.
 */
export function toSpaces(findings: RangeFinding[]): Record<string, unknown> {
  const spaces: Record<string, unknown> = {};
  for (const finding of findings) {
    if (finding.kind === "categorical") {
      if (finding.surviving.length === 0) continue;
      spaces[finding.name] = { type: "categorical", choices: finding.surviving };
      continue;
    }
    if (!finding.recommended) continue;
    const [low, high] = finding.recommended;
    // `recommended` is already snapped when rounding is on; the toPrecision
    // here only trims float noise from an unrounded bound.
    spaces[finding.name] = {
      type: finding.isInt ? "int" : "float",
      low: finding.isInt ? Math.round(low) : Number(low.toPrecision(3)),
      high: finding.isInt ? Math.round(high) : Number(high.toPrecision(3)),
      log: finding.log,
    };
  }
  return spaces;
}

// ------------------------------------------------------------ budget planning

export interface StartupPoint {
  n: number;
  /** P(at least one trial inside the band), drawing n of the N observed trials. */
  probability: number;
}

export interface StartupEstimate {
  study: ArchiveStudy | null;
  /** Trials inside the tolerance band. */
  hits: number;
  /** Trials the sweep spent, failures included — they consume startup budget too. */
  trials: number;
  /** Observed hit rate. */
  p: number;
  /** ceil(ln(1 - confidence) / ln(1 - p)), for n independent draws. */
  analytic: number | null;
  /** Smallest n whose subset of the observed trials clears the confidence. */
  empirical: number | null;
  /** The larger of the two, raised to the density floor. */
  recommended: number | null;
  curve: StartupPoint[];
}

export interface StartupSummary {
  confidence: number;
  /** Startup trials below this leave TPE's better group with a single point. */
  floor: number;
  perStudy: StartupEstimate[];
  /** The median study's recommendation — the headline number. */
  median: number | null;
  /** What the hardest study in the selection would need. */
  worst: number | null;
  /** Median hit rate across the selection. */
  p: number | null;
  studies: number;
}

/**
 * TPE's better group is the top ceil(0.1 N) trials, so ten startup trials build
 * the "good" density from a single point plus the prior. Twenty to thirty gives
 * two or three, which is a density estimate rather than a prior with a dot on
 * it, so a p-based answer below twenty is raised rather than taken literally.
 */
export const STARTUP_FLOOR = 20;
const STARTUP_CONFIDENCE = 0.9;
const MAXIMUM_STARTUP = 200;

/**
 * P(a sample of n drawn from N observed trials contains at least one of the K
 * inside the band) = 1 - C(N-K, n)/C(N, n), computed as a running product so it
 * never overflows. This is the exact value of the bootstrap-over-subsets
 * experiment, so there is nothing to resample.
 */
function subsetHitProbability(total: number, hits: number, draws: number): number {
  if (hits <= 0 || draws <= 0) return 0;
  if (draws > total) return 1;
  let miss = 1;
  for (let index = 0; index < draws; index += 1) {
    const remainingMisses = total - hits - index;
    if (remainingMisses <= 0) return 1;
    miss *= remainingMisses / (total - index);
  }
  return 1 - miss;
}

function estimateOne(assessment: StudyAssessment): StartupEstimate {
  // Every trial the sweep spent is a draw, failures included: a startup trial
  // that diverges still costs one of the n.
  const trials = assessment.trials.length;
  const hits = assessment.keep.length;
  const p = trials > 0 ? hits / trials : 0;

  const analytic =
    p > 0 && p < 1
      ? Math.ceil(Math.log(1 - STARTUP_CONFIDENCE) / Math.log(1 - p))
      : p >= 1
        ? 1
        : null;

  const curve: StartupPoint[] = [];
  let empirical: number | null = null;
  const ceiling = Math.min(MAXIMUM_STARTUP, Math.max(trials, 1));
  for (let n = 1; n <= ceiling; n += 1) {
    const probability = subsetHitProbability(trials, hits, n);
    curve.push({ n, probability });
    if (empirical === null && probability >= STARTUP_CONFIDENCE) empirical = n;
  }

  const both = [analytic, empirical].filter((value): value is number => value != null);
  const recommended =
    both.length > 0 ? Math.max(STARTUP_FLOOR, Math.max(...both)) : null;

  return {
    study: assessment.study,
    hits,
    trials,
    p,
    analytic,
    empirical,
    recommended,
    curve,
  };
}

export function estimateStartupTrials(assessments: StudyAssessment[]): StartupSummary {
  const perStudy = assessments
    .filter((assessment) => assessment.trials.length > 0 && assessment.best != null)
    .map(estimateOne);
  const recommendations = perStudy
    .map((estimate) => estimate.recommended)
    .filter((value): value is number => value != null);
  const rates = perStudy.map((estimate) => estimate.p);

  return {
    confidence: STARTUP_CONFIDENCE,
    floor: STARTUP_FLOOR,
    perStudy,
    median: recommendations.length > 0 ? Math.ceil(medianOf(recommendations)) : null,
    worst: recommendations.length > 0 ? Math.max(...recommendations) : null,
    p: rates.length > 0 ? medianOf(rates) : null,
    studies: perStudy.length,
  };
}

/**
 * True when the selection mixes studies with different fold counts, so the
 * higher fold indices are not comparable with the lower ones.
 */
export function mixedFoldCounts(assessments: StudyAssessment[]): number[] {
  const counts = new Set<number>();
  for (const assessment of assessments) {
    for (const trial of assessment.trials) {
      const folds = trial.trial.folds?.length ?? 0;
      if (folds > 0) counts.add(folds);
    }
  }
  return [...counts].sort((left, right) => left - right);
}

export interface EpochBudget {
  /** The configured epoch budget, when the selection agrees on one. */
  budget: number | null;
  budgets: number[];
  folds: number;
  /** Folds whose last epoch reached the budget instead of stopping early. */
  atBudget: number;
  share: number | null;
  medianLast: number | null;
  medianSelected: number | null;
  /** Studies contributing at least one fold that ran out of budget. */
  studiesAtBudget: number;
  studies: number;
}

/**
 * Whether the epoch budget was ever the binding constraint.
 *
 * Early stopping fires `patience` epochs after the best one, so a fold that
 * reaches the final epoch did not stop — it ran out of room while still within
 * patience of an improvement. Those folds are the ones whose scores would move
 * if the budget went up.
 */
export function summariseEpochs(assessments: StudyAssessment[]): EpochBudget {
  const budgets = new Set<number>();
  const lastEpochs: number[] = [];
  const selectedEpochs: number[] = [];
  let folds = 0;
  let atBudget = 0;
  let studiesAtBudget = 0;

  for (const assessment of assessments) {
    const budget = assessment.study.epochs;
    if (budget != null && Number.isFinite(budget)) budgets.add(budget);
    let studyHit = false;
    for (const trial of assessment.trials) {
      for (const fold of trial.trial.folds ?? []) {
        if (fold.lastEpoch == null) continue;
        folds += 1;
        lastEpochs.push(fold.lastEpoch);
        if (fold.selectedEpoch != null) selectedEpochs.push(fold.selectedEpoch);
        // Epochs are zero-indexed, so a 500-epoch budget ends at 499.
        if (budget != null && fold.lastEpoch >= budget - 1) {
          atBudget += 1;
          studyHit = true;
        }
      }
    }
    if (studyHit) studiesAtBudget += 1;
  }

  const sortedBudgets = [...budgets].sort((left, right) => left - right);
  return {
    budget: sortedBudgets.length === 1 ? sortedBudgets[0] : null,
    budgets: sortedBudgets,
    folds,
    atBudget,
    share: folds > 0 ? atBudget / folds : null,
    medianLast: lastEpochs.length > 0 ? medianOf(lastEpochs) : null,
    medianSelected: selectedEpochs.length > 0 ? medianOf(selectedEpochs) : null,
    studiesAtBudget,
    studies: assessments.length,
  };
}

// ------------------------------------------------------------- fold structure

export interface FoldRow {
  fold: number;
  trials: number;
  medianValue: number | null;
  medianPrecisionAt1: number | null;
  /** Median of this fold's score minus the trial's mean over its folds. */
  effect: number | null;
  effectP10: number | null;
  effectP90: number | null;
  medianLastEpoch: number | null;
  medianSelectedEpoch: number | null;
  atBudget: number;
  /** Studies that ran this fold index at all. */
  studies: number;
}

/**
 * The per-fold breakdown, with each fold's systematic offset separated from the
 * trials that ran on it. `effect` is the fold main effect of the same
 * decomposition the noise model uses: how much harder or easier this fold is
 * than the trial's own average, which is the part that cancels when two trials
 * are compared and so must not be read as noise.
 *
 * Offsets only sum to zero over a population that ran the same number of folds.
 * This archive mixes 4-fold and 5-fold studies, so a high fold index is drawn
 * from a different, smaller set of studies than fold 0 — which is why each row
 * carries the number of studies behind it, and why `mixedFoldCounts` exists.
 */
export function foldBreakdown(assessments: StudyAssessment[]): FoldRow[] {
  const byFold = new Map<
    number,
    {
      values: number[];
      precision: number[];
      effects: number[];
      lastEpochs: number[];
      selectedEpochs: number[];
      atBudget: number;
      studies: Set<string>;
    }
  >();

  for (const assessment of assessments) {
    const budget = assessment.study.epochs;
    for (const trial of assessment.trials) {
      const folds = trial.trial.folds ?? [];
      if (folds.length === 0) continue;
      const values = folds.map((fold) => fold.value).filter(Number.isFinite);
      const trialMean = values.length > 0 ? meanOf(values) : null;
      for (const fold of folds) {
        let bucket = byFold.get(fold.fold);
        if (!bucket) {
          bucket = {
            values: [],
            precision: [],
            effects: [],
            lastEpochs: [],
            selectedEpochs: [],
            atBudget: 0,
            studies: new Set<string>(),
          };
          byFold.set(fold.fold, bucket);
        }
        bucket.studies.add(assessment.study.id);
        if (Number.isFinite(fold.value)) {
          bucket.values.push(fold.value);
          if (trialMean != null) bucket.effects.push(fold.value - trialMean);
        }
        if (fold.precisionAt1 != null) bucket.precision.push(fold.precisionAt1);
        if (fold.lastEpoch != null) {
          bucket.lastEpochs.push(fold.lastEpoch);
          if (budget != null && fold.lastEpoch >= budget - 1) bucket.atBudget += 1;
        }
        if (fold.selectedEpoch != null) bucket.selectedEpochs.push(fold.selectedEpoch);
      }
    }
  }

  return [...byFold.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([fold, bucket]) => {
      const sortedEffects = ascending(bucket.effects);
      return {
        fold,
        trials: bucket.values.length,
        medianValue: bucket.values.length > 0 ? medianOf(bucket.values) : null,
        medianPrecisionAt1:
          bucket.precision.length > 0 ? medianOf(bucket.precision) : null,
        effect: sortedEffects.length > 0 ? quantileSorted(sortedEffects, 0.5) : null,
        effectP10: sortedEffects.length > 0 ? quantileSorted(sortedEffects, 0.1) : null,
        effectP90: sortedEffects.length > 0 ? quantileSorted(sortedEffects, 0.9) : null,
        medianLastEpoch:
          bucket.lastEpochs.length > 0 ? medianOf(bucket.lastEpochs) : null,
        medianSelectedEpoch:
          bucket.selectedEpochs.length > 0 ? medianOf(bucket.selectedEpochs) : null,
        atBudget: bucket.atBudget,
        studies: bucket.studies.size,
      };
    });
}

// --------------------------------------------------------- gradient contribution

/** The diagnostic series the gradient view can plot. */
export const GRADIENT_SERIES = [
  "ratio",
  "fraction",
  "cosine",
  "weight",
  "supervised",
  "regularizer",
  "unitRatio",
  "controllerWeight",
  "relativeRate",
  "sladeRatio",
] as const;

export type GradientSeriesKey = (typeof GRADIENT_SERIES)[number];

export const GRADIENT_SERIES_LABELS: Record<GradientSeriesKey, string> = {
  ratio: "Regularizer / supervised norm",
  fraction: "Regularizer share of the combined norm",
  cosine: "Cosine(regularizer, supervised)",
  weight: "Regularizer weight",
  supervised: "Supervised gradient norm",
  regularizer: "Regularizer gradient norm",
  unitRatio: "Ratio at unit weight",
  controllerWeight: "Controller regularizer weight",
  relativeRate: "Relative training rate",
  sladeRatio: "SLADE basis / supervised norm",
};

/** Series that are ratios of norms, so they belong on a log axis. */
export const GRADIENT_LOG_SERIES = new Set<GradientSeriesKey>([
  "ratio",
  "unitRatio",
  "sladeRatio",
  "supervised",
  "regularizer",
  "weight",
  "controllerWeight",
]);

/** Which statistic of a fold's series to read: its median, or its steady state. */
export type GradientStatistic = "median" | "late";

/**
 * Gradient diagnostics, indexed study -> trial -> fold, exactly as the separate
 * snapshot ships them. Passing the index around rather than grafting it onto
 * the trials keeps the main snapshot usable before it has been fetched.
 */
export type GradientIndex = Record<
  string,
  Record<string, Record<string, GradientFold>>
>;

/** The folds of one trial, or null when the study logged no diagnostics. */
function foldsFor(
  index: GradientIndex,
  study: ArchiveStudy,
  trialNumber: number,
): Record<string, GradientFold> | null {
  return index[study.id]?.[String(trialNumber)] ?? null;
}

export interface GradientPoint {
  study: ArchiveStudy;
  trialNumber: number;
  /** The fold this came from, or null for the trial's average over folds. */
  fold: number | null;
  value: number;
  /** Inner-decile spread of the series within the fold, in decades. */
  decades: number | null;
  /** Objective shortfall of the owning trial, for correlating with performance. */
  shortfall: number | null;
  inBand: boolean;
}

function readSeries(
  fold: GradientFold,
  key: GradientSeriesKey,
  statistic: GradientStatistic,
): SeriesSummary | null {
  const summary = fold[key];
  if (!summary) return null;
  const value = statistic === "late" ? summary.late : summary.median;
  return value != null && Number.isFinite(value) ? summary : null;
}

function seriesValue(summary: SeriesSummary, statistic: GradientStatistic): number | null {
  const value = statistic === "late" ? summary.late : summary.median;
  return value != null && Number.isFinite(value) ? value : null;
}

/**
 * One point per fold, plus one per trial averaging over its folds.
 *
 * The gradient diagnostics are written per fold, and a controller can hold its
 * target on one fold and lose it on another, so the fold-level points are the
 * primary evidence and the trial average is the summary — not the other way
 * round.
 */
export function collectGradientPoints(
  assessments: StudyAssessment[],
  index: GradientIndex,
  key: GradientSeriesKey,
  statistic: GradientStatistic,
): { folds: GradientPoint[]; trials: GradientPoint[] } {
  const folds: GradientPoint[] = [];
  const trials: GradientPoint[] = [];

  for (const assessment of assessments) {
    for (const trial of assessment.trials) {
      const gradients = foldsFor(index, assessment.study, trial.trial.number);
      if (!gradients) continue;
      const perFold: number[] = [];
      const perFoldDecades: number[] = [];
      for (const [foldKey, fold] of Object.entries(gradients)) {
        const summary = readSeries(fold, key, statistic);
        if (!summary) continue;
        const value = seriesValue(summary, statistic);
        if (value == null) continue;
        perFold.push(value);
        if (summary.decades != null) perFoldDecades.push(summary.decades);
        folds.push({
          study: assessment.study,
          trialNumber: trial.trial.number,
          fold: Number(foldKey),
          value,
          decades: summary.decades ?? null,
          shortfall: trial.shortfall,
          inBand: trial.inBand,
        });
      }
      if (perFold.length > 0) {
        trials.push({
          study: assessment.study,
          trialNumber: trial.trial.number,
          fold: null,
          value: meanOf(perFold),
          decades: perFoldDecades.length > 0 ? meanOf(perFoldDecades) : null,
          shortfall: trial.shortfall,
          inBand: trial.inBand,
        });
      }
    }
  }
  return { folds, trials };
}

export interface GradientFoldRow {
  fold: number;
  samples: number;
  median: number | null;
  p10: number | null;
  p90: number | null;
  /** Median within-fold spread of the series, in decades. */
  decades: number | null;
  /** Median signed deviation from the controller's target, in decades. */
  targetDeviation: number | null;
  targetAbsDeviation: number | null;
  cosine: number | null;
}

/** Per-fold rows plus the aggregate row, for the fold table of the gradient view. */
export function gradientFoldRows(
  assessments: StudyAssessment[],
  index: GradientIndex,
  key: GradientSeriesKey,
  statistic: GradientStatistic,
): { rows: GradientFoldRow[]; aggregate: GradientFoldRow | null } {
  interface Bucket {
    values: number[];
    decades: number[];
    deviation: number[];
    absolute: number[];
    cosine: number[];
  }
  const emptyBucket = (): Bucket => ({
    values: [],
    decades: [],
    deviation: [],
    absolute: [],
    cosine: [],
  });
  const buckets = new Map<number, Bucket>();
  // The aggregate row is filled alongside the per-fold ones rather than
  // reconstructed from them afterwards.
  const combined = emptyBucket();
  const bucketFor = (fold: number) => {
    let bucket = buckets.get(fold);
    if (!bucket) {
      bucket = emptyBucket();
      buckets.set(fold, bucket);
    }
    return bucket;
  };

  for (const assessment of assessments) {
    for (const trial of assessment.trials) {
      const gradients = foldsFor(index, assessment.study, trial.trial.number);
      if (!gradients) continue;
      for (const [foldKey, fold] of Object.entries(gradients)) {
        const bucket = bucketFor(Number(foldKey));
        const summary = fold[key];
        const value = summary ? seriesValue(summary, statistic) : null;
        if (value != null) {
          bucket.values.push(value);
          combined.values.push(value);
        }
        if (summary?.decades != null) {
          bucket.decades.push(summary.decades);
          combined.decades.push(summary.decades);
        }
        const deviation = fold.targetDeviation;
        if (deviation) {
          const signed = statistic === "late" ? deviation.late : deviation.median;
          if (signed != null) {
            bucket.deviation.push(signed);
            combined.deviation.push(signed);
          }
          if (deviation.absMedian != null) {
            bucket.absolute.push(deviation.absMedian);
            combined.absolute.push(deviation.absMedian);
          }
        }
        const cosine = fold.cosine ? seriesValue(fold.cosine, statistic) : null;
        if (cosine != null) {
          bucket.cosine.push(cosine);
          combined.cosine.push(cosine);
        }
      }
    }
  }

  const summarise = (fold: number, bucket: Bucket): GradientFoldRow => {
    const sorted = ascending(bucket.values);
    return {
      fold,
      samples: bucket.values.length,
      median: sorted.length > 0 ? quantileSorted(sorted, 0.5) : null,
      p10: sorted.length > 0 ? quantileSorted(sorted, 0.1) : null,
      p90: sorted.length > 0 ? quantileSorted(sorted, 0.9) : null,
      decades: bucket.decades.length > 0 ? medianOf(bucket.decades) : null,
      targetDeviation: bucket.deviation.length > 0 ? medianOf(bucket.deviation) : null,
      targetAbsDeviation: bucket.absolute.length > 0 ? medianOf(bucket.absolute) : null,
      cosine: bucket.cosine.length > 0 ? medianOf(bucket.cosine) : null,
    };
  };

  const rows = [...buckets.entries()]
    .sort((left, right) => left[0] - right[0])
    .map(([fold, bucket]) => summarise(fold, bucket));

  return {
    rows,
    aggregate: rows.length > 0 ? summarise(-1, combined) : null,
  };
}

/** Which gradient series the current selection actually recorded. */
export function availableGradientSeries(
  assessments: StudyAssessment[],
  index: GradientIndex,
): { series: GradientSeriesKey[]; hasTarget: boolean; trials: number } {
  const present = new Set<GradientSeriesKey>();
  let hasTarget = false;
  let trials = 0;

  for (const assessment of assessments) {
    for (const trial of assessment.trials) {
      const gradients = foldsFor(index, assessment.study, trial.trial.number);
      if (!gradients) continue;
      trials += 1;
      for (const fold of Object.values(gradients)) {
        for (const key of GRADIENT_SERIES) {
          if (fold[key]) present.add(key);
        }
        if (fold.targetDeviation || fold.target) hasTarget = true;
      }
    }
  }
  return {
    series: GRADIENT_SERIES.filter((key) => present.has(key)),
    hasTarget,
    trials,
  };
}
