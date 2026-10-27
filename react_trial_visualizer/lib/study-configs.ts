/**
 * Run configurations under F:\Studien, exported by scripts/export-study-configs.mjs
 * (npm run sync-configs), and the consistency checks the /configs/ page runs on them.
 */

export type ConfigPhase = "phase0" | "phase1" | "seeded" | "phase2";
export type Archive = "main" | "weitere";
export type Phase2Kind =
  | "replay"
  | "2x2"
  | "controls"
  | "sweeps"
  | "hp_path"
  | "geometry"
  | "external"
  | "ood"
  | "weight_decay"
  | "other";

export type SettingValue = string | number | boolean | null | SettingValue[];
export type Settings = Record<string, SettingValue>;

export interface SpaceParam {
  type: string;
  low?: number;
  high?: number;
  log?: boolean;
  step?: number;
  choices?: SettingValue[];
}

export interface HpoInfo {
  sampler: string | null;
  planned: number | null;
  complete: number;
  failed: number;
  pruned: number;
  running: number;
  waiting: number;
  studyName: string | null;
  best: { number: number; value: number; params: Record<string, SettingValue> } | null;
}

export type RunStatus = "complete" | "partial" | "started" | "empty";

export interface RunRecord {
  name: string;
  budget: string;
  seed: number | null;
  status: RunStatus;
  folds: number;
  cvK: number | null;
  attempts: number;
  seconds: number | null;
  lastEpoch: number | null;
  capHits: number | null;
}

export interface Spread {
  median: number;
  min: number;
  max: number;
}

/**
 * How long a study or series ran and how far its folds trained. HPO: from the
 * trial timestamps; replays: from each run's cv_<timestamp> folder to the time
 * its cv_summary.json was written, so a summary rewritten later stretches it.
 */
export interface Runtime {
  start: string | null;
  end: string | null;
  spanSeconds: number | null;
  computeSeconds: number | null;
  unitSeconds: number | null;
  units: number;
  folds: number;
  lastEpoch: Spread | null;
  selectedEpoch: Spread | null;
  cap: number | null;
  capHits: number | null;
}

export interface VaryingValue {
  value: SettingValue;
  count: number;
  runs: string[];
}

export interface ReplaySource {
  study: string | null;
  trial: number | null;
  params: Record<string, SettingValue> | null;
  count: number;
}

export interface AblationInfo {
  name: string | null;
  variant: string | null;
  changes: Record<string, SettingValue>;
  count: number;
}

export interface ReplayInfo {
  planned: string[] | null;
  runs: RunRecord[];
  complete: number;
  partial: number;
  started: number;
  empty: number;
  varying: Record<string, VaryingValue[]>;
  sources: ReplaySource[];
  ablations: AblationInfo[];
}

export interface ConfigEntry {
  id: string;
  phase: ConfigPhase;
  archive: Archive;
  dataset: string;
  method: string;
  loss: string;
  variant: string;
  /** Phase 2 only: the save root below Phase_2 (cars, cars_geometry, …) and what kind of replay the folder is. */
  root?: string;
  kind?: Phase2Kind;
  target: string | null;
  /** Which file the settings came from; null when the folder has none. */
  config: string | null;
  lossClass?: string | null;
  settings: Settings;
  /** HPO only: setting keys the search space overrides per trial. */
  tuned?: string[];
  space?: Record<string, SpaceParam>;
  hpo?: HpoInfo | null;
  replay?: ReplayInfo;
  runtime?: Runtime | null;
  notes: string[];
}

export interface PlannedFolder {
  id: string;
  root: string;
  dataset: string;
  method: string | null;
  loss: string | null;
  variant: string;
  kind: Phase2Kind;
  manifests: string[];
  ablations: string[];
  runs: string[];
  enabled: boolean;
}

export interface ConfigSnapshot {
  generatedAt: string;
  root: string;
  manifestRoot: string;
  entries: ConfigEntry[];
  phase2Plan: PlannedFolder[];
  manifests: { id: string; root: string; runs: number }[];
  skipped: { path: string; reason: string }[];
}

/** public/study-configs.json as written: settings are [key index, value] pairs against `settingKeys`. */
interface EncodedSnapshot extends Omit<ConfigSnapshot, "entries"> {
  version: number;
  settingKeys: string[];
  entries: (Omit<ConfigEntry, "settings"> & { settings: [number, SettingValue][] })[];
}

export function decodeSnapshot(raw: EncodedSnapshot): ConfigSnapshot {
  const keys = raw.settingKeys;
  return {
    ...raw,
    entries: raw.entries.map((entry) => {
      // A replay's tuned settings are the ones its trial's parameters set.
      const params = entry.replay?.sources[0]?.params;
      return {
        ...entry,
        tuned: entry.tuned ?? (params ? [...new Set(Object.keys(params).flatMap(settingKeysOfParam))] : undefined),
        settings: Object.fromEntries(entry.settings.map(([index, value]) => [keys[index], value])),
      };
    }),
  };
}

/** Where a search-space parameter lands among the settings (the exporter maps them the same way). */
export function settingKeysOfParam(param: string): string[] {
  if (param === "batch_sampler") return ["batch_size", "sampler_m"];
  if (param.startsWith("ssl_config.")) return [`ssl.${param.slice("ssl_config.".length)}`];
  const loss = /^loss\.[^.]+\.(.+)$/.exec(param);
  if (loss) return [`loss_params.${loss[1]}`];
  const miner = /^miner\.[^.]+\.(.+)$/.exec(param);
  if (miner) return [`miner_params.${miner[1]}`];
  return [param];
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "–";
  if (seconds < 90) return `${Math.round(seconds)} s`;
  const minutes = seconds / 60;
  if (minutes < 90) return `${Math.round(minutes)} min`;
  const hours = minutes / 60;
  if (hours < 48) return `${hours < 10 ? hours.toFixed(1) : Math.round(hours)} h`;
  const days = hours / 24;
  return `${days < 10 ? days.toFixed(1) : Math.round(days)} d`;
}

export function formatDate(iso: string | null | undefined): string {
  if (!iso) return "–";
  const date = new Date(iso);
  return `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

/**
 * Measured rows shown next to the settings: how long it ran and how far it trained.
 * They are results, not configuration, so they are never counted as deviations.
 */
export const RUNTIME_ROWS: { key: string; label: string; hint: string; value: (runtime: Runtime, entry: ConfigEntry) => string; sort?: (runtime: Runtime) => number | null }[] = [
  {
    key: "runtime.span",
    label: "Wall time",
    hint: "First start to last finish, idle gaps included",
    value: (runtime) => formatDuration(runtime.spanSeconds),
    sort: (runtime) => runtime.spanSeconds,
  },
  {
    key: "runtime.compute",
    label: "Summed run time",
    hint: "Sum over finished trials / runs (parallel runs overlap)",
    value: (runtime) => formatDuration(runtime.computeSeconds),
    sort: (runtime) => runtime.computeSeconds,
  },
  {
    key: "runtime.unit",
    label: "Median per trial / run",
    hint: "Median duration of one finished trial (all folds) or one seeded run",
    value: (runtime) => `${formatDuration(runtime.unitSeconds)}${runtime.units ? ` · n=${runtime.units}` : ""}`,
    sort: (runtime) => runtime.unitSeconds,
  },
  {
    key: "runtime.last",
    label: "Epochs trained",
    hint: "Last epoch per fold: median (min–max)",
    value: (runtime) => (runtime.lastEpoch ? `${formatNumber(runtime.lastEpoch.median)} (${runtime.lastEpoch.min}–${runtime.lastEpoch.max})` : "–"),
    sort: (runtime) => runtime.lastEpoch?.median ?? null,
  },
  {
    key: "runtime.selected",
    label: "Selected epoch",
    hint: "Epoch early stopping kept per fold: median (min–max)",
    value: (runtime) =>
      runtime.selectedEpoch ? `${formatNumber(runtime.selectedEpoch.median)} (${runtime.selectedEpoch.min}–${runtime.selectedEpoch.max})` : "–",
    sort: (runtime) => runtime.selectedEpoch?.median ?? null,
  },
  {
    key: "runtime.cap",
    label: "Folds at the epoch cap",
    hint: "Folds whose last epoch reached epochs − 1: they never stopped early and ran out of budget",
    value: (runtime) => (runtime.capHits == null || !runtime.folds ? "–" : `${runtime.capHits} of ${runtime.folds}`),
    sort: (runtime) => runtime.capHits,
  },
  {
    key: "runtime.dates",
    label: "Ran",
    hint: "Date of the first start and the last finish",
    value: (runtime) => (runtime.start ? `${formatDate(runtime.start)} → ${formatDate(runtime.end)}` : "–"),
  },
];

// ------------------------------------------------------------------ labels & order

export const PHASES: ConfigPhase[] = ["phase0", "phase1", "seeded", "phase2"];
export const PHASE_LABEL: Record<ConfigPhase, string> = {
  phase0: "Phase 0",
  phase1: "Phase 1",
  seeded: "Seeded runs",
  phase2: "Phase 2",
};
export const PHASE_HINT: Record<ConfigPhase, string> = {
  phase0: "Random search",
  phase1: "TPE",
  seeded: "Best trial, several seeds",
  phase2: "Ablations & geometry",
};

const DATASET_ORDER = ["Cars196", "CUB", "InShop", "InShop_10", "SemiINat", "SOP", "iNat", "iNat2018", "iNat2019", "iNat2021Mini", "NABirds", "CIFAR100"];
export const DATASET_LABEL: Record<string, string> = {
  Cars196: "Cars196",
  CUB: "CUB-200",
  InShop: "In-Shop",
  InShop_10: "In-Shop 10 %",
  SemiINat: "Semi-iNat",
  SOP: "SOP",
  iNat: "iNat",
  iNat2018: "iNat 2018",
  iNat2019: "iNat 2019",
  iNat2021Mini: "iNat 2021 mini",
  NABirds: "NABirds",
  CIFAR100: "CIFAR-100",
};

const METHOD_ORDER = ["Ueberwacht", "LP", "LRML", "Hoffer", "SLADE", "ISMLP", "SERAPH", "SSMLPC"];
export const METHOD_LABEL: Record<string, string> = {
  Ueberwacht: "Supervised",
  LP: "LP",
  LRML: "LRML",
  Hoffer: "Hoffer",
  SLADE: "SLADE",
  ISMLP: "ISMLP",
  SERAPH: "SERAPH",
  SSMLPC: "SSMLPC",
};
/** Identity dots only; every dot sits next to the method's name. Same hues as the Phase-2 page. */
export const METHOD_COLOR: Record<string, string> = {
  Ueberwacht: "#8895aa",
  LP: "#4fd5e7",
  LRML: "#9b8cff",
  Hoffer: "#f4c86f",
  SLADE: "#b8e56e",
  ISMLP: "#f49ac1",
  SERAPH: "#7fb8ff",
  SSMLPC: "#e6a57e",
};

const LOSS_ORDER = ["ArcFace", "Circle", "MultiSimilarity", "NTXent", "ProxyAnchor", "TripletMargin"];
export const LOSS_LABEL: Record<string, string> = {
  ArcFace: "ArcFace",
  Circle: "Circle",
  MultiSimilarity: "Multi-Sim.",
  NTXent: "NT-Xent",
  ProxyAnchor: "Proxy Anchor",
  TripletMargin: "Triplet",
};
export const LOSS_SHORT: Record<string, string> = {
  ArcFace: "Arc",
  Circle: "Cir",
  MultiSimilarity: "MS",
  NTXent: "NT",
  ProxyAnchor: "Prx",
  TripletMargin: "Tri",
};

export const KIND_ORDER: Phase2Kind[] = ["2x2", "replay", "controls", "sweeps", "hp_path", "geometry", "external", "ood", "weight_decay", "other"];
export const KIND_LABEL: Record<Phase2Kind, string> = {
  replay: "Plain replay",
  "2x2": "2×2 (HP × data)",
  controls: "Mechanism controls",
  sweeps: "Sensitivity sweeps",
  hp_path: "HP path",
  geometry: "Geometry (B6)",
  external: "External data",
  ood: "OOD share",
  weight_decay: "Weight decay",
  other: "Other",
};
/** Ablations change the method to learn something; geometry replays the tuned config with tracking on. */
export const KIND_GROUP: Record<Phase2Kind, "ablation" | "geometry"> = {
  replay: "ablation",
  "2x2": "ablation",
  controls: "ablation",
  sweeps: "ablation",
  hp_path: "ablation",
  geometry: "geometry",
  external: "ablation",
  ood: "ablation",
  weight_decay: "ablation",
  other: "ablation",
};

function rank(order: string[], value: string): number {
  const index = order.indexOf(value);
  return index < 0 ? order.length : index;
}

export function sortDatasets(values: Iterable<string>): string[] {
  return [...new Set(values)].sort((a, b) => rank(DATASET_ORDER, a) - rank(DATASET_ORDER, b) || a.localeCompare(b));
}
export function sortMethods(values: Iterable<string>): string[] {
  return [...new Set(values)].sort((a, b) => rank(METHOD_ORDER, a) - rank(METHOD_ORDER, b) || a.localeCompare(b));
}
export function sortLosses(values: Iterable<string>): string[] {
  return [...new Set(values)].sort((a, b) => rank(LOSS_ORDER, a) - rank(LOSS_ORDER, b) || a.localeCompare(b));
}

export function compareEntries(a: ConfigEntry, b: ConfigEntry): number {
  return (
    rank(PHASES, a.phase) - rank(PHASES, b.phase) ||
    rank(DATASET_ORDER, a.dataset) - rank(DATASET_ORDER, b.dataset) ||
    rank(METHOD_ORDER, a.method) - rank(METHOD_ORDER, b.method) ||
    rank(LOSS_ORDER, a.loss) - rank(LOSS_ORDER, b.loss) ||
    (a.root ?? "").localeCompare(b.root ?? "") ||
    a.variant.localeCompare(b.variant)
  );
}

export const datasetLabel = (value: string) => DATASET_LABEL[value] ?? value;
export const methodLabel = (value: string) => METHOD_LABEL[value] ?? value;
export const lossLabel = (value: string) => LOSS_LABEL[value] ?? value;

/** "LRML · Circle · gekappt", without the dataset. */
export function entryName(entry: ConfigEntry, { dataset = false }: { dataset?: boolean } = {}): string {
  const parts = [dataset ? datasetLabel(entry.dataset) : null, methodLabel(entry.method), lossLabel(entry.loss)];
  if (entry.phase === "phase2") parts.push(entry.variant || "replay");
  else if (entry.variant) parts.push(entry.variant);
  return parts.filter(Boolean).join(" · ");
}

// ------------------------------------------------------------------ setting sections

export type SectionId = "protocol" | "data" | "optim" | "loss" | "ssl" | "hpo" | "seeds" | "technical";

export const SECTIONS: { id: SectionId; label: string; hint: string }[] = [
  { id: "protocol", label: "Training protocol", hint: "Epochs, early stopping, folds, validation and model selection" },
  { id: "data", label: "Data & labels", hint: "Dataset protocol, label budget and the unlabeled pool" },
  { id: "optim", label: "Model & optimisation", hint: "Backbone, head, optimiser, learning rates, batches" },
  { id: "loss", label: "Loss & miner", hint: "Metric-learning loss, miner and their parameters" },
  { id: "ssl", label: "SSL method", hint: "Everything in the resolved SSL config" },
  { id: "hpo", label: "HPO", hint: "Sampler, budget and objective of the study" },
  { id: "seeds", label: "Seeds", hint: "Split, HPO, run and support seeds" },
  { id: "technical", label: "Technical", hint: "Precision, compilation and other arguments that should not change results" },
];

const PROTOCOL_KEYS = [
  "epochs",
  "patience",
  "cv_k",
  "cv_mode",
  "val_mode",
  "validation_retrieval_mode",
  "validation_gallery_fraction",
  "holdout_val_ratio",
  "selection_metric",
  "final_fit_mode",
  "final_epoch_aggregation",
  "study_dir_mode",
  "concatenated_fold_test",
  "measurements",
  "evaluate_test",
  "skip_test_during_hpo",
  "final_test_after_hpo",
  "final_test_top_n",
  "ssl.warmup_epochs",
  "ssl.restart_selection_after_warmup",
  "ssl.warmup_checkpoint_mode",
];
const DATA_KEYS = [
  "dataset",
  "dataset_protocol",
  "k_shot_grid",
  "label_budget_grid",
  "ssl_label_sampling_modes",
  "ssl.label_sampling_mode",
  "ssl.labeled_per_class",
  "ssl.labeled_fraction",
  "ssl.class_overlap",
  "unlabeled_source",
  "unlabeled_class_scope",
  "unlabeled_fraction",
  "ssl.unlabeled_class_scope",
  "ssl.unlabeled_fraction",
  "max_unlabeled_samples",
  "ssl.max_unlabeled_samples",
  "external_unlabeled_filter",
  "semi_inat_ood_fraction",
  "semi_inat_ood_seed",
  "image_resize_mode",
];
const OPTIM_KEYS = [
  "backbone_tuning",
  "dino_size",
  "feat_dim",
  "projection_layers",
  "projection_hidden_dim",
  "graph_projection",
  "optim",
  "lr",
  "classifier_lr",
  "weight_decay",
  "lr_scheduler",
  "batch_size",
  "sampler_m",
  "labeled_batch_fraction",
  "ssl.labeled_batch_size",
  "frozen_feature_train_views",
  "frozen_head_only_checkpoint",
];
const LOSS_KEYS = ["loss", "miner", "warmup_loss", "warmup_miner"];
const SEED_KEYS = ["seed", "data_split_seed", "hparam_seed", "support_seed", "comparison_seeds", "comparison_seed_targets"];

const KEY_SECTION = new Map<string, SectionId>([
  ...PROTOCOL_KEYS.map((key) => [key, "protocol"] as [string, SectionId]),
  ...DATA_KEYS.map((key) => [key, "data"] as [string, SectionId]),
  ...OPTIM_KEYS.map((key) => [key, "optim"] as [string, SectionId]),
  ...LOSS_KEYS.map((key) => [key, "loss"] as [string, SectionId]),
  ...SEED_KEYS.map((key) => [key, "seeds"] as [string, SectionId]),
]);
const KEY_ORDER = new Map<string, number>(
  [...PROTOCOL_KEYS, ...DATA_KEYS, ...OPTIM_KEYS, ...LOSS_KEYS, ...SEED_KEYS].map((key, index) => [key, index]),
);

export function sectionOf(key: string): SectionId {
  const known = KEY_SECTION.get(key);
  if (known) return known;
  if (key.startsWith("hpo.")) return "hpo";
  if (/^(loss_params|miner_params|warmup_loss_params|warmup_miner_params)\./.test(key)) return "loss";
  if (key.startsWith("lr_scheduler_params")) return "optim";
  if (key.startsWith("ssl.")) return "ssl";
  if (/^(cifar_|semi_aves_|semi_inat_)/.test(key)) return "data";
  return "technical";
}

export function compareKeys(a: string, b: string): number {
  const left = KEY_ORDER.get(a);
  const right = KEY_ORDER.get(b);
  if (left != null && right != null) return left - right;
  if (left != null) return -1;
  if (right != null) return 1;
  // SSL: the method first, then its parameters grouped by their dotted path.
  if (a === "ssl.method") return -1;
  if (b === "ssl.method") return 1;
  return a.localeCompare(b);
}

const KEY_LABEL: Record<string, string> = {
  epochs: "Epochs",
  patience: "Patience",
  cv_k: "Folds (k)",
  cv_mode: "CV mode",
  val_mode: "Validation mode",
  validation_retrieval_mode: "Validation retrieval",
  validation_gallery_fraction: "Validation gallery share",
  holdout_val_ratio: "Hold-out share",
  selection_metric: "Selection metric",
  final_fit_mode: "Final fit",
  final_epoch_aggregation: "Final epoch aggregation",
  study_dir_mode: "Study dir mode",
  concatenated_fold_test: "Concatenated fold test",
  measurements: "Validation measurements",
  "ssl.warmup_epochs": "SSL warm-up epochs",
  "ssl.restart_selection_after_warmup": "Restart selection after warm-up",
  "ssl.warmup_checkpoint_mode": "Warm-up checkpoint",
  dataset: "Dataset",
  dataset_protocol: "Dataset protocol",
  k_shot_grid: "k-shot grid",
  label_budget_grid: "Label-budget grid",
  "ssl.label_sampling_mode": "Label sampling",
  "ssl.labeled_per_class": "Labeled per class (k)",
  "ssl.labeled_fraction": "Labeled share",
  "ssl.class_overlap": "Class overlap",
  unlabeled_source: "Unlabeled source",
  image_resize_mode: "Image resize",
  backbone_tuning: "Backbone",
  dino_size: "DINO size",
  feat_dim: "Feature dim",
  projection_layers: "Projection layers",
  optim: "Optimiser",
  lr: "Learning rate",
  classifier_lr: "Classifier LR",
  weight_decay: "Weight decay",
  lr_scheduler: "LR schedule",
  batch_size: "Batch size",
  sampler_m: "Samples per class (m)",
  loss: "Loss",
  miner: "Miner",
  warmup_loss: "Warm-up loss",
  "ssl.method": "SSL method",
  "ssl.update_mode": "Update mode",
  "ssl.update_interval_epochs": "Update interval (epochs)",
  "ssl.confidence_threshold": "Confidence threshold",
  "ssl.method_params.regularizer_max_ratio": "Regularizer max ratio (cap)",
  "ssl.method_params.regularizer_target_ratio": "Regularizer target ratio",
  "ssl.method_params.regularizer_weight": "Regularizer weight",
  "ssl.method_params.n_neighbors": "Neighbors (n)",
  "hpo.sampler": "Sampler",
  "hpo.n_trials": "Trials planned",
  "hpo.startup_trials": "Startup trials",
  "hpo.metric": "Objective",
  seed: "Run seed",
  data_split_seed: "Split seed",
  hparam_seed: "HPO seed",
  support_seed: "Support seed",
  comparison_seeds: "Comparison seeds",
  train_amp: "Train AMP",
  compile_train_step: "Compiled train step",
};

export function keyLabel(key: string): string | null {
  return KEY_LABEL[key] ?? null;
}

// ------------------------------------------------------------------ values

export function valueKey(value: SettingValue | undefined): string {
  return JSON.stringify(value ?? null);
}

export function formatNumber(value: number): string {
  if (!Number.isFinite(value)) return String(value);
  if (Number.isInteger(value)) return Math.abs(value) >= 1e6 ? value.toExponential(0) : String(value);
  const abs = Math.abs(value);
  if (abs !== 0 && (abs < 1e-3 || abs >= 1e5)) return value.toExponential(2).replace(/\.?0+e/, "e");
  return String(Number(value.toPrecision(4)));
}

export function formatValue(value: SettingValue | undefined): string {
  if (value === undefined) return "–";
  if (value === null) return "null";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return `[${value.map((item) => formatValue(item)).join(", ")}]`;
  return value;
}

/** Most common value (ties: the first seen), and how many entries hold it. */
export function modeOf(values: (SettingValue | undefined)[]): { key: string; value: SettingValue | undefined; count: number; distinct: number } {
  const counts = new Map<string, { value: SettingValue | undefined; count: number }>();
  for (const value of values) {
    const key = valueKey(value);
    const current = counts.get(key);
    if (current) current.count += 1;
    else counts.set(key, { value, count: 1 });
  }
  let best: { key: string; value: SettingValue | undefined; count: number } = { key: "null", value: undefined, count: 0 };
  for (const [key, item] of counts) if (item.count > best.count) best = { key, ...item };
  return { ...best, distinct: counts.size };
}

export function isTuned(entry: ConfigEntry, key: string): boolean {
  return entry.tuned?.includes(key) ?? false;
}

/** The search-space parameter that sets a tuned setting (batch_size and sampler_m share batch_sampler). */
export function paramOfSetting(entry: ConfigEntry, key: string): string | null {
  if (!entry.space) return null;
  for (const param of Object.keys(entry.space)) {
    if (param === "batch_sampler" && (key === "batch_size" || key === "sampler_m")) return param;
    if (param.startsWith("ssl_config.") && key === `ssl.${param.slice("ssl_config.".length)}`) return param;
    const loss = /^loss\.[^.]+\.(.+)$/.exec(param);
    if (loss && key === `loss_params.${loss[1]}`) return param;
    const miner = /^miner\.[^.]+\.(.+)$/.exec(param);
    if (miner && key === `miner_params.${miner[1]}`) return param;
    if (param === key) return param;
  }
  return null;
}

export function formatSpace(spec: SpaceParam | undefined): string {
  if (!spec) return "–";
  if (spec.choices) return `{${spec.choices.map((choice) => formatValue(choice)).join(", ")}}`;
  const low = spec.low != null ? formatNumber(spec.low) : "?";
  const high = spec.high != null ? formatNumber(spec.high) : "?";
  return `[${low}, ${high}]${spec.log ? " log" : ""}${spec.step != null ? ` step ${formatNumber(spec.step)}` : ""}`;
}

export function spaceKey(spec: SpaceParam | undefined): string {
  return JSON.stringify(spec ?? null);
}

// ------------------------------------------------------------------ progress

export function isHpo(entry: ConfigEntry): boolean {
  return entry.phase === "phase0" || entry.phase === "phase1";
}

/** Done / expected: trials for a study, complete runs for a replay folder. */
export function progressOf(entry: ConfigEntry): { done: number; expected: number | null; unit: string } {
  if (isHpo(entry)) {
    return { done: entry.hpo?.complete ?? 0, expected: entry.hpo?.planned ?? null, unit: "trials" };
  }
  const replay = entry.replay;
  const planned = replay?.planned?.length ?? null;
  return { done: replay?.complete ?? 0, expected: planned ?? replay?.runs.length ?? null, unit: "runs" };
}

// ------------------------------------------------------------------ checks

export type IssueType = "deviation" | "series" | "replay" | "cap" | "incomplete" | "missing";

export interface Issue {
  id: string;
  type: IssueType;
  phase: ConfigPhase;
  dataset: string;
  title: string;
  detail: string;
  entries: string[];
  /** A setting the issue is about; the page jumps to it in the compare view. */
  key?: string;
}

export const ISSUE_LABEL: Record<IssueType, string> = {
  deviation: "Protocol deviation",
  series: "Mixed within a series",
  replay: "Replay ≠ Phase-1 best",
  cap: "Hit the epoch cap",
  incomplete: "Incomplete",
  missing: "Missing",
};

/**
 * Protocol settings every study of one phase and dataset should share. A
 * setting a study tunes is left out for that study, and the SSL cap only counts
 * among the methods that have one.
 */
export const PROTOCOL_CHECK_KEYS = [
  "epochs",
  "patience",
  "cv_k",
  "cv_mode",
  "val_mode",
  "validation_retrieval_mode",
  "validation_gallery_fraction",
  "selection_metric",
  "final_fit_mode",
  "final_epoch_aggregation",
  "dataset_protocol",
  "image_resize_mode",
  "backbone_tuning",
  "dino_size",
  "feat_dim",
  "optim",
  "lr_scheduler",
  "weight_decay",
  "k_shot_grid",
  "label_budget_grid",
  "ssl.label_sampling_mode",
  "ssl.labeled_per_class",
  "ssl.labeled_fraction",
  "hpo.sampler",
  "hpo.n_trials",
  "hpo.startup_trials",
  "hpo.metric",
  "data_split_seed",
  "hparam_seed",
  "ssl.method_params.regularizer_max_ratio",
];

export const CAP_SHARE_HPO = 0.02;

const LABEL_BUDGET_KEYS = new Set(["k_shot_grid", "label_budget_grid", "ssl.labeled_per_class", "ssl.labeled_fraction"]);

/** Settings an entry's ablation config changes: a Phase-2 variant is allowed to differ there on purpose. */
export function ablationKeys(entry: ConfigEntry): Set<string> {
  const keys = new Set<string>();
  for (const ablation of entry.replay?.ablations ?? []) {
    for (const change of Object.keys(ablation.changes)) {
      keys.add(change.startsWith("ssl_config.") ? `ssl.${change.slice("ssl_config.".length)}` : change);
    }
  }
  return keys;
}

function groupBy<T>(items: T[], key: (item: T) => string): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const item of items) {
    const id = key(item);
    const group = groups.get(id);
    if (group) group.push(item);
    else groups.set(id, [item]);
  }
  return groups;
}

function paramsEqual(left: Record<string, SettingValue> | null, right: Record<string, SettingValue> | null): string[] {
  if (!left || !right) return [];
  const differing: string[] = [];
  for (const key of new Set([...Object.keys(left), ...Object.keys(right)])) {
    const a = left[key];
    const b = right[key];
    if (typeof a === "number" && typeof b === "number") {
      if (Math.abs(a - b) > 1e-4 * Math.max(Math.abs(a), Math.abs(b), 1e-12)) differing.push(key);
    } else if (valueKey(a) !== valueKey(b)) {
      differing.push(key);
    }
  }
  return differing;
}

/** Phase-1 studies of the same dataset, method and loss, the linked one (no variant, not _weitere) first. */
function phase1Studies(studies: ConfigEntry[], entry: ConfigEntry): ConfigEntry[] {
  return studies
    .filter(
      (candidate) =>
        candidate.phase === "phase1" &&
        candidate.dataset === entry.dataset &&
        candidate.method === entry.method &&
        candidate.loss === entry.loss &&
        candidate.hpo?.best,
    )
    .sort(
      (a, b) =>
        Number(isLinkedStudy(b)) - Number(isLinkedStudy(a)) ||
        // The same study is often linked twice (a Phase_1 variant and a _weitere copy); name the Phase_1 one.
        Number(b.archive === "main") - Number(a.archive === "main"),
    );
}

function isLinkedStudy(entry: ConfigEntry): boolean {
  return entry.archive === "main" && entry.variant === "";
}

/** "Phase_1/InShop_10/Hoffer/ArcFace__Seeded_Runs" → "Phase_1/InShop_10/Hoffer/*__Seeded_Runs". */
export function studyPattern(entry: ConfigEntry): string {
  return entry.id.replace(/\/[^/]*$/, `/*${entry.variant ? `__${entry.variant}` : ""}`);
}

export interface ReplayCheck {
  /**
   * match: the linked Phase-1 study's best trial. other: the best trial of a
   * different Phase-1 study (a variant or one in _weitere). mismatch: not the
   * best trial of any Phase-1 study on F:. mixed: the runs disagree.
   */
  status: "match" | "other" | "mismatch" | "mixed" | "unknown";
  linked: ConfigEntry | null;
  source: ConfigEntry | null;
  text: string;
  differing: string[];
}

/** Which Phase-1 study's best trial a seeded or Phase-2 folder replayed. `studies` must include _weitere. */
export function replayCheck(studies: ConfigEntry[], entry: ConfigEntry): ReplayCheck | null {
  if (!entry.replay || entry.replay.sources.length === 0) return null;
  const candidates = phase1Studies(studies, entry);
  const linked = candidates.find(isLinkedStudy) ?? null;
  const sources = entry.replay.sources;
  if (sources.length > 1) {
    return {
      status: "mixed",
      linked,
      source: null,
      differing: [],
      text: `Runs replay ${sources.length} different trials: ${sources.map((source) => `#${source.trial ?? "?"} (${source.count})`).join(", ")}`,
    };
  }
  const replayed = sources[0];
  const matched = candidates.find(
    (candidate) => candidate.hpo?.best?.number === replayed.trial && paramsEqual(replayed.params, candidate.hpo.best.params).length === 0,
  );
  if (matched && matched === linked) {
    return { status: "match", linked, source: matched, differing: [], text: `Replays best trial #${replayed.trial} of ${matched.id}` };
  }
  if (matched) {
    return {
      status: "other",
      linked,
      source: matched,
      differing: [],
      text: `Replays best trial #${replayed.trial} of ${matched.id}${linked ? `, not of ${linked.id} (best #${linked.hpo?.best?.number})` : ""}`,
    };
  }
  if (!linked?.hpo?.best) {
    return { status: "unknown", linked, source: null, differing: [], text: `Replays trial #${replayed.trial ?? "?"} of ${replayed.study ?? "?"}; no Phase-1 study to compare with` };
  }
  const best = linked.hpo.best;
  const differing = paramsEqual(replayed.params, best.params);
  return {
    status: "mismatch",
    linked,
    source: null,
    differing,
    text: `Replays trial #${replayed.trial ?? "?"} of "${replayed.study ?? "?"}", which is no Phase-1 study's best on F:; ${linked.id} has best #${best.number}${
      differing.length ? ` (${differing.length} parameter${differing.length === 1 ? "" : "s"} differ)` : ""
    }`,
  };
}

export function computeIssues(snapshot: ConfigSnapshot, entries: ConfigEntry[]): Issue[] {
  const issues: Issue[] = [];

  // 1. Protocol settings that differ within one phase and dataset.
  const scoped = entries.filter((entry) => entry.config);
  const byScope = groupBy(scoped, (entry) => `${entry.phase}|${entry.dataset}|${entry.phase === "phase2" ? (entry.kind === "geometry" ? "geometry" : "ablation") : ""}`);
  for (const [scope, group] of byScope) {
    if (group.length < 2) continue;
    const [phase, dataset, part] = scope.split("|") as [ConfigPhase, string, string];
    for (const key of PROTOCOL_CHECK_KEYS) {
      // A replay folder runs every budget of its grid; which ones is coverage, not protocol.
      if ((phase === "seeded" || phase === "phase2") && LABEL_BUDGET_KEYS.has(key)) continue;
      // Seeded series also set protocol through an ablation config (e.g. increase_epochs); only
      // a Phase-2 variant's overrides are its purpose.
      const relevant = group.filter(
        (entry) => key in entry.settings && !isTuned(entry, key) && !(entry.phase === "phase2" && ablationKeys(entry).has(key)),
      );
      if (relevant.length < 2) continue;
      const mode = modeOf(relevant.map((entry) => entry.settings[key]));
      if (mode.distinct < 2) continue;
      const deviating = relevant.filter((entry) => valueKey(entry.settings[key]) !== mode.key);
      const byValue = groupBy(deviating, (entry) => valueKey(entry.settings[key]));
      const label = keyLabel(key) ?? key;
      issues.push({
        id: `deviation|${scope}|${key}`,
        type: "deviation",
        phase,
        dataset,
        key,
        title: `${label}: ${formatValue(mode.value)} in ${mode.count} of ${relevant.length}${part === "geometry" ? " geometry" : part === "ablation" ? " ablation" : ""} folders, but`,
        detail: [...byValue.values()]
          .map((items) => `${formatValue(items[0].settings[key])} in ${items.length} (${summarizeNames(items)})`)
          .join("; "),
        entries: deviating.map((entry) => entry.id),
      });
    }
  }

  // 2. Seeds of one series that ran with different settings.
  for (const entry of entries) {
    const varying = entry.replay?.varying ?? {};
    const keys = Object.keys(varying).filter((key) => sectionOf(key) !== "technical");
    if (keys.length === 0) continue;
    issues.push({
      id: `series|${entry.id}`,
      type: "series",
      phase: entry.phase,
      dataset: entry.dataset,
      key: keys[0],
      title: `${entryName(entry)}: runs differ in ${keys.map((key) => keyLabel(key) ?? key).join(", ")}`,
      detail: keys
        .slice(0, 4)
        .map((key) => `${keyLabel(key) ?? key}: ${varying[key].map((item) => `${formatValue(item.value)} ×${item.count}`).join(" / ")}`)
        .join("; "),
      entries: [entry.id],
    });
  }

  // 3. Replays that did not take the linked Phase-1 study's best trial, grouped by where they took it from.
  const replayGroups = new Map<string, { entries: ConfigEntry[]; checks: ReplayCheck[] }>();
  for (const entry of entries) {
    if (entry.phase !== "seeded" && entry.phase !== "phase2") continue;
    const check = replayCheck(snapshot.entries, entry);
    if (!check || check.status === "match" || check.status === "unknown") continue;
    const id = `${entry.phase}|${entry.dataset}|${entry.method}|${check.status}|${check.source ? studyPattern(check.source) : ""}`;
    const group = replayGroups.get(id) ?? { entries: [], checks: [] };
    group.entries.push(entry);
    group.checks.push(check);
    replayGroups.set(id, group);
  }
  for (const [id, group] of replayGroups) {
    const [phase, dataset, method, status] = id.split("|") as [ConfigPhase, string, string, ReplayCheck["status"]];
    const count = `${group.entries.length} folder${group.entries.length === 1 ? "" : "s"}`;
    const first = group.checks[0];
    const title =
      status === "other" && first.source
        ? `${methodLabel(method)}: ${count} replay the best trials of ${studyPattern(first.source)}${first.linked ? `, not of ${studyPattern(first.linked)}` : ""}`
        : status === "mixed"
          ? `${methodLabel(method)}: ${count} whose runs replay different trials`
          : `${methodLabel(method)}: ${count} replay a trial that is no Phase-1 study's best`;
    issues.push({
      id: `replay|${id}`,
      type: "replay",
      phase,
      dataset,
      title,
      detail: group.entries
        .slice(0, 6)
        .map((entry, index) => {
          const check = group.checks[index];
          const trial = entry.replay?.sources[0]?.trial;
          const reference = check.linked?.hpo?.best?.number;
          return `${lossLabel(entry.loss)}${entry.variant ? ` · ${entry.variant}` : ""}: #${trial ?? "?"}${reference != null ? ` (linked best #${reference})` : ""}`;
        })
        .join("; ")
        .concat(group.entries.length > 6 ? `; +${group.entries.length - 6} more` : ""),
      entries: group.entries.map((entry) => entry.id),
    });
  }

  // 4. Folds that trained until the last epoch: early stopping never fired, the budget did. In a
  // search a few extreme trials (tiny learning rates) always do, so HPO studies need 2 % of folds;
  // a replay's numbers are the reported ones, so any fold counts there.
  for (const entry of entries) {
    const runtime = entry.runtime;
    if (!runtime?.capHits || !runtime.folds) continue;
    if (isHpo(entry) && runtime.capHits / runtime.folds < CAP_SHARE_HPO) continue;
    issues.push({
      id: `cap|${entry.id}`,
      type: "cap",
      phase: entry.phase,
      dataset: entry.dataset,
      key: "epochs",
      title: `${entryName(entry)}: ${runtime.capHits} of ${runtime.folds} folds reached epoch ${runtime.cap}`,
      detail: `Patience ${formatValue(entry.settings.patience)}; median fold stopped at ${runtime.lastEpoch ? formatNumber(runtime.lastEpoch.median) : "?"}, selected ${runtime.selectedEpoch ? formatNumber(runtime.selectedEpoch.median) : "?"}.`,
      entries: [entry.id],
    });
  }

  // 5. Folders without any config, and studies / series that did not finish.
  for (const entry of entries) {
    if (!entry.config) {
      issues.push({
        id: `missing-config|${entry.id}`,
        type: "missing",
        phase: entry.phase,
        dataset: entry.dataset,
        title: `${entryName(entry)}: no config found`,
        detail: entry.notes.join("; ") || "The folder has no study_config.json / cv_summary.json / run_config.json.",
        entries: [entry.id],
      });
      continue;
    }
    const progress = progressOf(entry);
    if (progress.expected != null && progress.done < progress.expected) {
      issues.push({
        id: `incomplete|${entry.id}`,
        type: "incomplete",
        phase: entry.phase,
        dataset: entry.dataset,
        title: `${entryName(entry)}: ${progress.done} of ${progress.expected} ${progress.unit}`,
        detail: isHpo(entry)
          ? `${entry.hpo?.failed ?? 0} failed, ${entry.hpo?.running ?? 0} running, ${entry.hpo?.pruned ?? 0} pruned`
          : `${entry.replay?.partial ?? 0} partial, ${entry.replay?.started ?? 0} started, ${missingRuns(entry).length} not started`,
        entries: [entry.id],
      });
    }
  }

  // 6. Phase-2 folders the manifests plan but that are not on F:.
  const present = new Set(entries.filter((entry) => entry.phase === "phase2").map((entry) => entry.id));
  const absent = snapshot.phase2Plan.filter((folder) => folder.enabled && !resolvePlanned(folder, present));
  for (const [scope, folders] of groupBy(absent, (folder) => `${folder.dataset}|${KIND_GROUP[folder.kind]}|${folder.root}`)) {
    const [dataset] = scope.split("|");
    issues.push({
      id: `missing-plan|${scope}`,
      type: "missing",
      phase: "phase2",
      dataset,
      title: `Phase_2/${folders[0].root}: ${folders.length} planned folder${folders.length === 1 ? "" : "s"} not found`,
      detail: folders
        .slice(0, 8)
        .map((folder) => folder.id.split("/").pop())
        .join(", ")
        .concat(folders.length > 8 ? `, … (${folders.length - 8} more)` : ""),
      entries: [],
    });
  }

  return issues;
}

/**
 * The folder on F: that fulfils a planned one: the folder itself, or one named
 * the way the server-side generator names variants (`<combo>_controls_<v>`,
 * `<combo>_sweeps_<v>`), or, for a planned `<combo>_defaults`, the unmodified
 * replay `<combo>`.
 */
export function resolvePlanned(folder: PlannedFolder, present: Set<string>): string | null {
  if (present.has(folder.id)) return folder.id;
  const base = folder.id.slice(0, folder.id.length - folder.variant.length);
  for (const prefix of ["controls_", "sweeps_"]) {
    if (folder.variant && present.has(`${base}${prefix}${folder.variant}`)) return `${base}${prefix}${folder.variant}`;
  }
  if (folder.variant === "defaults" && present.has(folder.id.replace(/_defaults$/, ""))) return folder.id.replace(/_defaults$/, "");
  return null;
}

export function missingRuns(entry: ConfigEntry): string[] {
  const planned = entry.replay?.planned;
  if (!planned) return [];
  const present = new Set(entry.replay?.runs.map((run) => run.name));
  return planned.filter((name) => !present.has(name));
}

function summarizeNames(entries: ConfigEntry[]): string {
  const names = entries.map((entry) => entryName(entry));
  return names.length <= 3 ? names.join(", ") : `${names.slice(0, 3).join(", ")} +${names.length - 3}`;
}
