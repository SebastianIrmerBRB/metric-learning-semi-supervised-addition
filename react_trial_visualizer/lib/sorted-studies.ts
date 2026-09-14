/**
 * Types for the sorted HPO archive snapshot that
 * scripts/export-sorted-studies.mjs writes to public/sorted-studies.json.
 *
 * One record per Optuna study directory under sorted_logs/, with the facets the
 * archive layout encodes (phase, mode, dataset, SSL method, DML loss, campaign
 * variant) already resolved so the browser only filters and aggregates.
 */

export type ArchiveParamValue = string | number | boolean | null;

/**
 * One cross-validation fold of one trial.
 *
 * Validation only. Nothing in this archive is tuned against the test split, so
 * the exporter does not carry test metrics at all and there is no field here
 * that could be mistaken for one.
 */
export interface ArchiveFold {
  fold: number;
  /** The study's objective metric on this fold's validation split. */
  value: number;
  precisionAt1?: number;
  /** The last epoch this fold ran — equal to the budget when it never stopped early. */
  lastEpoch?: number;
  /** The epoch whose validation score was selected. */
  selectedEpoch?: number;
  trainLoss?: number;
  globalStep?: number;
}

/** Median, spread and steady-state value of one diagnostic series over a fold. */
export interface SeriesSummary {
  n: number;
  median: number | null;
  p10: number | null;
  p90: number | null;
  /** Inner-decile spread in orders of magnitude, for ratio-like series. */
  decades?: number | null;
  /** Median over the last quarter of the run. */
  late: number | null;
}

/** How far the achieved gradient ratio sat from the controller's target, in decades. */
export interface TargetDeviation {
  n: number;
  median: number | null;
  p10: number | null;
  p90: number | null;
  absMedian: number | null;
  late: number | null;
}

/**
 * The gradient-contribution diagnostics of a single fold. `ratio` is the
 * regularizer gradient norm over the supervised one, `target` the ratio a
 * GradNorm-style controller was asked to hold, and `cosine` the alignment
 * between the two gradients — a negative value means the regularizer is pulling
 * against the supervised loss rather than with it.
 */
export interface GradientFold {
  supervised?: SeriesSummary;
  regularizer?: SeriesSummary;
  combined?: SeriesSummary;
  fraction?: SeriesSummary;
  cosine?: SeriesSummary;
  weight?: SeriesSummary;
  unitRatio?: SeriesSummary;
  ratio?: SeriesSummary;
  sladeRatio?: SeriesSummary;
  sladeNorm?: SeriesSummary;
  target?: SeriesSummary;
  sladeTarget?: SeriesSummary;
  controllerWeight?: SeriesSummary;
  supervisedWeight?: SeriesSummary;
  relativeRate?: SeriesSummary;
  targetDeviation?: TargetDeviation;
}

export interface ArchiveTrial {
  number: number;
  /** Optuna trial state: COMPLETE, FAIL, PRUNED, RUNNING or WAITING. */
  state: string;
  /** The objective, or null when the trial never produced one. */
  value: number | null;
  seconds?: number;
  params: Record<string, ArchiveParamValue>;
  metrics: Record<string, ArchiveParamValue>;
  /** The cross-validation folds behind the aggregate, when the trial ran them. */
  folds?: ArchiveFold[];
}

/**
 * Gradient diagnostics, indexed study -> trial -> fold.
 *
 * These live in their own file because they are three times the size of
 * everything else put together, and only the gradient view needs them. The
 * explorer fetches it the first time that view is opened.
 */
export interface GradientSnapshot {
  schemaVersion: 1;
  generatedAt: string;
  studies: Record<string, Record<string, Record<string, GradientFold>>>;
}

export function isGradientSnapshot(value: unknown): value is GradientSnapshot {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<GradientSnapshot>;
  return candidate.schemaVersion === 1 && typeof candidate.studies === "object";
}

export interface ArchiveParameter {
  name: string;
  kind: "numeric" | "categorical";
  distribution: "float" | "int" | "categorical";
  log: boolean;
  low?: number;
  high?: number;
  step?: number;
  choices?: ArchiveParamValue[];
  /** fANOVA importance, only where optuna_param_importance.py has run. */
  importance?: number;
  importanceRank?: number;
  /** False for a space the config declares that no trial ever sampled. */
  searched: boolean;
}

export interface ArchiveStudy {
  id: string;
  path: string;
  /** "complete" for phase_0/phase_1, "unfinished" for the < 30 trial bucket. */
  bucket: "complete" | "unfinished";
  /** phase_0 = random search, phase_1 = TPE — kept for unfinished studies too. */
  phase: string;
  mode: string;
  dataset: string;
  datasetLabel: string;
  method: string;
  methodLabel: string;
  /** The `__suffix` campaign tag that disambiguates repeated method sweeps. */
  variant: string | null;
  loss: string;
  lossLabel: string;
  lossClass: string | null;
  miner: string;
  labelBudget: string;
  sampler: string;
  direction: "maximize" | "minimize";
  metric: string;
  plannedTrials: number | null;
  studyName: string;
  runName: string | null;
  campaign: string | null;
  sourcePath: string | null;
  experimentConfig: string | null;
  hparamConfigPath: string | null;
  seed: number | null;
  cvK: number | null;
  cvMode: string | null;
  epochs: number | null;
  patience: number | null;
  earlyStoppingMetric: string | null;
  backboneTuning: string | null;
  trialCount: number;
  completeTrials: number;
  best: { number: number; value: number } | null;
  /** SSL settings that stayed constant across the study's trials. */
  sslConfig: Record<string, ArchiveParamValue>;
  parameters: ArchiveParameter[];
  trials: ArchiveTrial[];
}

export interface ArchiveSnapshot {
  schemaVersion: 1;
  generatedAt: string;
  source: string;
  metricKeys?: string[];
  studyDirectories?: number;
  mappedStudies?: number;
  /** Studies sorting_map.csv lists that never landed on disk. */
  missingFromDisk?: string[];
  /** Every distinct objective metric in the snapshot; all validation metrics. */
  objectiveMetrics?: string[];
  studies: ArchiveStudy[];
  warning?: string;
  failures?: Array<{ study: string; error: string }>;
}

export function isArchiveSnapshot(value: unknown): value is ArchiveSnapshot {
  if (!value || typeof value !== "object") return false;
  const candidate = value as Partial<ArchiveSnapshot>;
  return candidate.schemaVersion === 1 && Array.isArray(candidate.studies);
}

/** The completed trials that actually carry an objective. */
export function scoredTrials(study: ArchiveStudy): ArchiveTrial[] {
  return study.trials.filter(
    (trial) => trial.state === "COMPLETE" && typeof trial.value === "number",
  );
}

/** Just the fold objective values, for the statistics that only need those. */
export function foldValues(trial: ArchiveTrial): number[] {
  return (trial.folds ?? [])
    .map((fold) => fold.value)
    .filter((value) => Number.isFinite(value));
}

export function isBetter(
  direction: ArchiveStudy["direction"],
  candidate: number,
  incumbent: number,
): boolean {
  return direction === "minimize" ? candidate < incumbent : candidate > incumbent;
}
