export type TrialRow = Record<string, string | null>;

export interface SeedFile {
  name: string;
  text: string;
}

export interface StudyConfig {
  trial: string;
  metric: string;
  learningRate: string | null;
  state: string | null;
  epoch: string | null;
  selectedEpoch: string | null;
  duration: string | null;
}

export interface Study {
  id: string;
  name: string;
  fileName: string;
  dataset: string;
  source: "bundled" | "uploaded";
  columns: string[];
  rows: TrialRow[];
  config: StudyConfig;
}

export interface StudyStats {
  bestRow: TrialRow | null;
  bestMetric: number | null;
  bestTrial: string | null;
  complete: number;
  failed: number;
  medianDuration: number | null;
}

const METRIC_PRIORITIES = [
  "attr:best_valid_mean_average_precision_at_r",
  "best_valid_mean_average_precision_at_r",
  "valid_mean_average_precision_at_r",
  "mean_average_precision_at_r",
  "map@r",
  "map_at_r",
  "map_r",
  "value",
];

const TRIAL_PRIORITIES = ["number", "trial", "trial_number", "step", "iteration", "index"];
const STATE_PRIORITIES = ["state", "status", "trial_state"];
const EPOCH_PRIORITIES = [
  "epoch_count",
  "epochs_run",
  "num_epochs",
  "n_epochs",
  "attr:last_epoch",
  "last_epoch",
  "epoch",
  "epochs",
];
const SELECTED_EPOCH_PRIORITIES = ["attr:selected_epoch", "selected_epoch", "best_epoch"];
const DURATION_PRIORITIES = ["duration_seconds", "duration", "elapsed_seconds", "runtime_seconds"];

export function normalizeColumn(value: string) {
  return value.trim().toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
}

export function numericValue(value: string | null | undefined): number | null {
  if (value == null || value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function firstMatchingColumn(columns: string[], candidates: string[]): string | null {
  const normalized = new Map(columns.map((column) => [normalizeColumn(column), column]));
  for (const candidate of candidates) {
    const match = normalized.get(normalizeColumn(candidate));
    if (match) return match;
  }
  return null;
}

export function numericColumns(study: Pick<Study, "columns" | "rows">): string[] {
  return study.columns.filter((column) => study.rows.some((row) => numericValue(row[column]) != null));
}

function detectMetric(columns: string[], rows: TrialRow[]): string {
  const exact = firstMatchingColumn(columns, METRIC_PRIORITIES);
  if (exact && rows.some((row) => numericValue(row[exact]) != null)) return exact;

  const scored = columns
    .filter((column) => rows.some((row) => numericValue(row[column]) != null))
    .map((column) => {
      const name = normalizeColumn(column);
      let score = 0;
      if (name.includes("map") || name.includes("mean_average_precision")) score += 10;
      if (name.includes("valid") || name.includes("validation")) score += 4;
      if (name.includes("best")) score += 3;
      if (name.includes("test")) score -= 2;
      return { column, score };
    })
    .filter(({ score }) => score > 0)
    .sort((a, b) => b.score - a.score || a.column.localeCompare(b.column));

  if (scored[0]) return scored[0].column;
  const fallback = columns.find((column) => rows.some((row) => numericValue(row[column]) != null));
  if (!fallback) throw new Error("No numeric metric column could be detected.");
  return fallback;
}

function detectLearningRate(columns: string[], rows: TrialRow[]): string | null {
  const candidates = columns
    .filter((column) => {
      const name = normalizeColumn(column);
      const tokens = name.split("_");
      const looksLikeLearningRate =
        name === "lr" ||
        name.endsWith("_lr") ||
        name.includes("learning_rate") ||
        name.includes("learningrate") ||
        (tokens.includes("lr") && (tokens.includes("param") || tokens.length <= 3));
      return looksLikeLearningRate && rows.some((row) => numericValue(row[column]) != null);
    })
    .sort((a, b) => {
      const aName = normalizeColumn(a);
      const bName = normalizeColumn(b);
      const aPrimary = ["lr", "param_lr", "learning_rate"].includes(aName) ? 0 : 1;
      const bPrimary = ["lr", "param_lr", "learning_rate"].includes(bName) ? 0 : 1;
      return aPrimary - bPrimary || aName.localeCompare(bName);
    });
  return candidates[0] ?? null;
}

function uniqueHeaders(rawHeaders: string[]): string[] {
  const seen = new Map<string, number>();
  return rawHeaders.map((rawHeader, index) => {
    const header = rawHeader.replace(/^\uFEFF/, "").trim() || `column_${index + 1}`;
    const count = seen.get(header) ?? 0;
    seen.set(header, count + 1);
    return count === 0 ? header : `${header}_${count + 1}`;
  });
}

export function parseCsv(text: string): { columns: string[]; rows: TrialRow[] } {
  const records: string[][] = [];
  let record: string[] = [];
  let field = "";
  let quoted = false;

  const pushField = () => {
    record.push(field);
    field = "";
  };
  const pushRecord = () => {
    pushField();
    if (record.some((value) => value.trim() !== "")) records.push(record);
    record = [];
  };

  for (let index = 0; index < text.length; index += 1) {
    const character = text[index];
    if (character === '"') {
      if (quoted && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else {
        quoted = !quoted;
      }
    } else if (character === "," && !quoted) {
      pushField();
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && text[index + 1] === "\n") index += 1;
      pushRecord();
    } else {
      field += character;
    }
  }

  if (field.length > 0 || record.length > 0) pushRecord();
  if (quoted) throw new Error("The CSV contains an unterminated quoted value.");
  if (records.length < 2) throw new Error("The CSV must contain a header and at least one trial row.");

  const columns = uniqueHeaders(records[0]);
  const rows = records.slice(1).map((values) =>
    Object.fromEntries(
      columns.map((column, index) => {
        const value = values[index]?.trim() ?? "";
        return [column, value === "" || value.toLowerCase() === "null" ? null : value];
      }),
    ),
  );

  return { columns, rows };
}

function shortHash(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

function displayName(fileName: string): string {
  const match = fileName.match(/(ProxyAnchorLoss|MultiSimilarityLoss|ArcFaceLoss|CircleLoss|NTXentLoss)/i);
  const labels: Record<string, string> = {
    proxyanchorloss: "Proxy Anchor",
    multisimilarityloss: "Multi-Similarity",
    arcfaceloss: "ArcFace",
    circleloss: "Circle",
    ntxentloss: "NT-Xent",
  };
  const loss = match ? labels[match[1].toLowerCase()] : null;
  const normalizedName = fileName.toLowerCase();
  const method = normalizedName.startsWith("phase1_lrml__")
    ? "LRML"
    : normalizedName.startsWith("supervised__")
      ? "Supervised"
      : null;
  const shotMatch = normalizedName.match(/class_subset_k_shot_label_\d+_k_(\d+)/);
  const shot = shotMatch
    ? Number(shotMatch[1]) >= 99_999
      ? "Full labels"
      : `${Number(shotMatch[1])}-shot`
    : null;
  const descriptiveParts = [method, loss, shot].filter(
    (part): part is string => part != null,
  );
  if (descriptiveParts.length > 1) return descriptiveParts.join(" · ");
  if (loss) return loss;

  return fileName
    .replace(/\.csv$/i, "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .slice(0, 52);
}

function datasetName(fileName: string): string {
  const raw = fileName.split("__")[0] || "dataset";
  const known =
    fileName.match(/(cifar100|cifar10|cifar|cars196|cars|cub|sop|nabirds|in[-_ ]?shop)/i)?.[1] ??
    raw;
  return known
    .replace(/^cars196$/i, "cars")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function createStudy(seed: SeedFile, source: Study["source"] = "uploaded"): Study {
  const parsed = parseCsv(seed.text);
  let { columns, rows } = parsed;
  let trial = firstMatchingColumn(columns, TRIAL_PRIORITIES);

  if (!trial) {
    trial = "_trial_index";
    columns = [trial, ...columns];
    rows = rows.map((row, index) => ({ [trial as string]: String(index), ...row }));
  }

  const metric = detectMetric(columns, rows);
  const id = `${shortHash(seed.name)}-${shortHash(seed.text.slice(0, 2048))}`;

  return {
    id,
    name: displayName(seed.name),
    fileName: seed.name,
    dataset: datasetName(seed.name),
    source,
    columns,
    rows,
    config: {
      trial,
      metric,
      learningRate: detectLearningRate(columns, rows),
      state: firstMatchingColumn(columns, STATE_PRIORITIES),
      epoch: firstMatchingColumn(columns, EPOCH_PRIORITIES),
      selectedEpoch: firstMatchingColumn(columns, SELECTED_EPOCH_PRIORITIES),
      duration: firstMatchingColumn(columns, DURATION_PRIORITIES),
    },
  };
}

export function formatNumber(value: number | null | undefined, digits = 5): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if (value !== 0 && (absolute < 0.0001 || absolute >= 100_000)) return value.toExponential(2);
  return new Intl.NumberFormat("en-US", { maximumSignificantDigits: digits }).format(value);
}

export function formatMetric(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return value.toFixed(4);
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  if (seconds < 60) return `${Math.round(seconds)}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = Math.round(seconds % 60);
  if (minutes < 60) return `${minutes}m ${remainder}s`;
  const hours = Math.floor(minutes / 60);
  return `${hours}h ${minutes % 60}m`;
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const midpoint = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[midpoint - 1] + sorted[midpoint]) / 2 : sorted[midpoint];
}

export function getStudyStats(study: Study): StudyStats {
  const metricRows = study.rows
    .map((row) => ({ row, value: numericValue(row[study.config.metric]) }))
    .filter((entry): entry is { row: TrialRow; value: number } => entry.value != null);
  const best = metricRows.reduce<(typeof metricRows)[number] | null>(
    (current, entry) => (!current || entry.value > current.value ? entry : current),
    null,
  );
  const states = study.config.state
    ? study.rows.map((row) => (row[study.config.state as string] ?? "").toUpperCase())
    : [];
  const durations = study.config.duration
    ? study.rows
        .map((row) => numericValue(row[study.config.duration as string]))
        .filter((value): value is number => value != null)
    : [];

  return {
    bestRow: best?.row ?? null,
    bestMetric: best?.value ?? null,
    bestTrial: best?.row[study.config.trial] ?? null,
    complete: study.config.state ? states.filter((state) => state === "COMPLETE").length : study.rows.length,
    failed: states.filter((state) => state === "FAIL" || state === "FAILED").length,
    medianDuration: median(durations),
  };
}

export function columnLabel(column: string | null): string {
  if (!column) return "Not detected";
  return column
    .replace(/^(attr|param):/i, "")
    .replace(/[_:]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function escapeCsv(value: string | null): string {
  if (value == null) return "";
  return /[",\r\n]/.test(value) ? `"${value.replace(/"/g, '""')}"` : value;
}

export function rowsToCsv(columns: string[], rows: TrialRow[]): string {
  return [columns.map(escapeCsv).join(","), ...rows.map((row) => columns.map((column) => escapeCsv(row[column])).join(","))].join("\r\n");
}

export function slugify(value: string): string {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "") || "trial-diagram";
}
