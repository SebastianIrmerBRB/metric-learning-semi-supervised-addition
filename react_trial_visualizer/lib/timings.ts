/**
 * Types and derived values for the epoch-cost snapshots written by
 * `scripts/bench_epoch_cost.py` (`timings.json`). A snapshot holds one row per
 * HPO study (epoch counts from Optuna, cost estimate) and one summary per
 * benchmarked case (clean per-epoch rates measured alone on one GPU).
 */

export interface PhaseRate {
  n: number;
  total_s: number;
  train_s: number;
  nontrain_s: number;
  valid_s: number;
  gap_s: number;
  other_s: number;
  steps: number;
  total_median_s: number | null;
  total_p90_s: number | null;
}

export interface CurvePoint {
  epoch: number;
  phase: "sup" | "warmup" | "ssl";
  gap_s: number;
  train_s: number;
  valid_s: number;
  other_s: number;
}

export interface TimingCase {
  case_id: string;
  study_key: string;
  name: string | null;
  dataset: string;
  method: string;
  loss: string;
  trial: number;
  batch_size: number | null;
  repeats: number;
  ssl_enabled: boolean;
  warmup_epochs: number;
  update_mode: string | null;
  update_interval_epochs: number | null;
  epochs_per_fold: number;
  folds: number;
  steps_per_epoch: number | null;
  warm: PhaseRate | null;
  ssl_first: PhaseRate | null;
  ssl: PhaseRate | null;
  fold_fixed_s: number | null;
  study_once_s: number | null;
  trial_overhead_s: number | null;
  fold_setup_s: number | null;
  baseline_valid_s: number | null;
  contaminated: boolean;
  gpu_process_visibility: boolean;
  gpu_utilization_mean: number | null;
  load_average_max: number | null;
  measured_at: string;
  curve: CurvePoint[];
}

export interface TimingStudy {
  study_key: string;
  name: string | null;
  dataset: string;
  dataset_label: string;
  method: string;
  method_label: string;
  loss: string;
  loss_label: string;
  trials: number;
  trials_estimated: number;
  epochs_per_fold_mean: number | null;
  ssl_epochs_per_fold_mean: number | null;
  warmup_epochs_mean: number | null;
  estimated_gpu_hours: number | null;
  estimated_gpu_hours_imputed: number | null;
  observed_hours: number | null;
  trial_minutes_median: number | null;
  time_models: Record<string, number>;
  selected_trial: number | null;
  selected_batch_size: number | null;
  selected_warmup_epochs: number | null;
  selected_warm_epoch_s: number | null;
  selected_ssl_first_epoch_s: number | null;
  selected_ssl_epoch_s: number | null;
  selected_fold_fixed_s: number | null;
  cases: string[];
}

export interface TimingRun {
  settings?: { select: string; window: number; repeats: number };
  host?: string;
  gpu?: { index?: string; name?: string; uuid?: string; driver?: string; memory_total_mib?: string };
  device_name?: string;
  torch?: string;
  cuda?: string;
  started?: string;
  finished?: string;
}

export interface TimingSnapshot {
  version: number;
  generated: string;
  bench_dir: string;
  run: TimingRun;
  labels: { datasets: Record<string, string>; methods: Record<string, string>; losses: Record<string, string> };
  studies: TimingStudy[];
  cases: TimingCase[];
}

export interface TimingSnapshotEntry {
  /** Folder of the snapshot relative to the timings root, e.g. "phase1_2026-10-01". */
  id: string;
  path: string;
  snapshot: TimingSnapshot;
}

/* --------------------------------------------------------------- dimensions */

export type Dimension = "method" | "dataset" | "loss";

export const DIMENSION_LABEL: Record<Dimension, string> = {
  method: "Verfahren",
  dataset: "Datensatz",
  loss: "Verlust",
};

/** Thesis order; anything unknown sorts after these, alphabetically. */
const ORDER: Record<Dimension, string[]> = {
  method: ["supervised", "hoffer_entropy", "slade", "lrml", "iscen_label_spreading"],
  dataset: ["Cars196", "CUB", "DeepFashionInShop", "DeepFashionInShop_10pct"],
  loss: ["NTXentLoss", "MultiSimilarityLoss", "CircleLoss", "ProxyAnchorLoss", "ArcFaceLoss"],
};

export function dimensionValue(study: TimingStudy, dimension: Dimension): string {
  return study[dimension];
}

export function dimensionLabel(study: TimingStudy, dimension: Dimension): string {
  return dimension === "method" ? study.method_label : dimension === "dataset" ? study.dataset_label : study.loss_label;
}

export function orderedValues(studies: TimingStudy[], dimension: Dimension): string[] {
  const values = Array.from(new Set(studies.map((study) => dimensionValue(study, dimension))));
  const order = ORDER[dimension];
  return values.sort((a, b) => {
    const ia = order.indexOf(a);
    const ib = order.indexOf(b);
    if (ia !== ib) return (ia < 0 ? order.length : ia) - (ib < 0 ? order.length : ib);
    return a.localeCompare(b);
  });
}

/* ------------------------------------------------------------------ metrics */

export type Aggregation = "sum" | "median" | "mean";

export const AGGREGATION_LABEL: Record<Aggregation, string> = { sum: "Summe", median: "Median", mean: "Mittel" };

export interface MetricContext {
  /** Supervised (Ü) study of the same dataset and loss, the reference for ratios. */
  supervisedOf: (study: TimingStudy) => TimingStudy | undefined;
}

export interface MetricDef {
  key: MetricKey;
  label: string;
  unit: string;
  digits: number;
  aggregation: Aggregation;
  description: string;
  value: (study: TimingStudy, context: MetricContext) => number | null;
}

export type MetricKey =
  | "epoch"
  | "warmEpoch"
  | "firstSslEpoch"
  | "epochsPerFold"
  | "gpuHours"
  | "trialMinutes"
  | "observedHours"
  | "epochVsSup"
  | "gpuHoursVsSup";

/** Seconds per training epoch of the study's own phase: supervised for Ü, SSL phase otherwise. */
export function mainEpochSeconds(study: TimingStudy): number | null {
  return study.method === "supervised" ? study.selected_warm_epoch_s : study.selected_ssl_epoch_s;
}

function ratio(value: number | null, reference: number | null | undefined): number | null {
  return value == null || reference == null || reference <= 0 ? null : value / reference;
}

export const METRICS: MetricDef[] = [
  {
    key: "epoch",
    label: "Epoche",
    unit: "s",
    digits: 2,
    aggregation: "median",
    description:
      "Sekunden je Epoche der ausgewählten Konfiguration, allein auf der GPU gemessen: bei Ü eine überwachte Epoche, sonst eine Epoche der SSL-Phase (Mittel, schließt periodische Refreshes anteilig ein).",
    value: mainEpochSeconds,
  },
  {
    key: "warmEpoch",
    label: "Warm-up-Epoche",
    unit: "s",
    digits: 2,
    aggregation: "median",
    description: "Sekunden je Epoche, die nur mit gelabelten Bildern trainiert (Warm-up; bei Ü jede Epoche).",
    value: (study) => study.selected_warm_epoch_s,
  },
  {
    key: "firstSslEpoch",
    label: "Erste SSL-Epoche",
    unit: "s",
    digits: 2,
    aggregation: "median",
    description:
      "Die Epoche am Ende des Warm-ups mit dem einmaligen Wechsel: erster Refresh (Pseudo-Labels, Graph, k-Means) und ggf. Wiederherstellen des besten Warm-up-Stands.",
    value: (study) => study.selected_ssl_first_epoch_s,
  },
  {
    key: "epochsPerFold",
    label: "Epochen je Fold",
    unit: "",
    digits: 1,
    aggregation: "mean",
    description: "Tatsächlich trainierte Epochen je Fold, gemittelt über alle Trials der Studie (inklusive Patience-Epochen).",
    value: (study) => study.epochs_per_fold_mean,
  },
  {
    key: "gpuHours",
    label: "HPO-Kosten",
    unit: "GPU-h",
    digits: 2,
    aggregation: "sum",
    description:
      "Geschätzte GPU-Stunden der ganzen Studie: gemessene Zeit je Epoche × protokollierte Epochen jedes Trials und Folds, plus Fixkosten je Fold. Trials ohne Epochenprotokoll mit dem Median ergänzt.",
    value: (study) => study.estimated_gpu_hours_imputed,
  },
  {
    key: "trialMinutes",
    label: "Trial-Dauer (Median)",
    unit: "min",
    digits: 1,
    aggregation: "median",
    description: "Geschätzte Dauer eines Trials (alle Folds) allein auf der GPU, Median über die Trials der Studie.",
    value: (study) => study.trial_minutes_median,
  },
  {
    key: "observedHours",
    label: "Beobachtete Laufzeit",
    unit: "h",
    digits: 1,
    aggregation: "sum",
    description:
      "Summe der von Optuna protokollierten Trial-Dauern. Die Läufe teilten sich die GPUs mit wechselnder Last; nicht zwischen Studien vergleichbar, nur zur Einordnung.",
    value: (study) => study.observed_hours,
  },
  {
    key: "epochVsSup",
    label: "Epoche relativ zu Ü",
    unit: "×",
    digits: 2,
    aggregation: "median",
    description: "Zeit je Epoche geteilt durch die überwachte Epoche von Ü mit demselben Datensatz und Verlust.",
    value: (study, context) => {
      const reference = context.supervisedOf(study);
      return reference ? ratio(mainEpochSeconds(study), mainEpochSeconds(reference)) : null;
    },
  },
  {
    key: "gpuHoursVsSup",
    label: "HPO-Kosten relativ zu Ü",
    unit: "×",
    digits: 2,
    aggregation: "median",
    description: "Geschätzte HPO-Kosten geteilt durch die von Ü mit demselben Datensatz und Verlust.",
    value: (study, context) =>
      ratio(study.estimated_gpu_hours_imputed, context.supervisedOf(study)?.estimated_gpu_hours_imputed),
  },
];

export function metricByKey(key: MetricKey): MetricDef {
  return METRICS.find((metric) => metric.key === key) ?? METRICS[0];
}

export function aggregate(values: (number | null)[], aggregation: Aggregation): number | null {
  const present = values.filter((value): value is number => value != null && Number.isFinite(value));
  if (present.length === 0) return null;
  if (aggregation === "sum") return present.reduce((sum, value) => sum + value, 0);
  if (aggregation === "mean") return present.reduce((sum, value) => sum + value, 0) / present.length;
  const sorted = [...present].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function makeContext(studies: TimingStudy[]): MetricContext {
  const supervised = new Map<string, TimingStudy>();
  for (const study of studies) {
    if (study.method === "supervised") supervised.set(`${study.dataset}|${study.loss}`, study);
  }
  return { supervisedOf: (study) => supervised.get(`${study.dataset}|${study.loss}`) };
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return "–";
  return value.toLocaleString("de-DE", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function formatMetric(metric: MetricDef, value: number | null | undefined): string {
  const text = formatNumber(value, metric.digits);
  return text === "–" || !metric.unit ? text : `${text} ${metric.unit}`;
}

/* ------------------------------------------------------------ time models */

export const TIME_MODEL_LABEL: Record<string, string> = {
  measured: "gemessen (gleiche Batchgröße)",
  scaled_batch_size: "skaliert (andere Batchgröße)",
  same_combination: "andere Studie, gleiche Kombination",
  same_combination_scaled: "andere Studie, skaliert",
  no_measurement: "keine Messung",
  incomplete_rates: "Raten unvollständig",
  no_epoch_record: "kein Epochenprotokoll",
};

export const PHASE_LABEL: Record<CurvePoint["phase"], string> = {
  sup: "Überwacht",
  warmup: "Warm-up",
  ssl: "SSL-Phase",
};
