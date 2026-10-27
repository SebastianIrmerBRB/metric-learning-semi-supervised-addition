export const TEST_METRICS = {
  test_mean_average_precision_at_r: "Test mAP@R",
  test_precision_at_1: "Test P@1",
} as const;

export type TestMetric = keyof typeof TEST_METRICS;
export type TestResultValues = Record<string, string | null>;

export interface TestResult {
  id: string;
  values: TestResultValues;
  dataset: string;
  scenario: string;
  mode: string;
  role: string;
  loss: string;
  miner: string;
  sslMethod: string;
  labelSamplingMode: string;
  labeledFraction: number | null;
  labeledPerClass: number | null;
  seed: string;
  studyName: string;
  trialNumber: string;
  recordedAt: string;
  testPrecisionAt1: number | null;
  testMapAtR: number | null;
  validPrecisionAt1: number | null;
  validMapAtR: number | null;
}

export interface TestResultFile {
  name: string;
  text: string;
}

export interface ConfigurationSummary {
  key: string;
  label: string;
  details: string;
  dataset: string;
  mode: string;
  loss: string;
  miner: string;
  labelBudget: string;
  count: number;
  seedCount: number;
  mean: number;
  standardDeviation: number;
  minimum: number;
  maximum: number;
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

export function parseCsvRecords(text: string): { columns: string[]; rows: TestResultValues[] } {
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
  if (records.length === 0) throw new Error("The CSV is empty.");

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

function numberValue(value: string | null | undefined): number | null {
  if (value == null || value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function stringValue(value: string | null | undefined, fallback = ""): string {
  return value?.trim() || fallback;
}

function shortHash(value: string): string {
  let hash = 2166136261;
  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index);
    hash = Math.imul(hash, 16777619);
  }
  return (hash >>> 0).toString(36);
}

export function parseTestResultsCsv(text: string): TestResult[] {
  const { columns, rows } = parseCsvRecords(text);
  const metricColumns = Object.keys(TEST_METRICS);
  if (!metricColumns.some((column) => columns.includes(column))) {
    throw new Error(
      "This is not a test-results aggregate: no test mAP@R or P@1 column was found.",
    );
  }

  return rows
    .map((values, index): TestResult => {
      const testPrecisionAt1 = numberValue(values.test_precision_at_1);
      const testMapAtR = numberValue(values.test_mean_average_precision_at_r);
      const fallbackId = shortHash(
        [
          values.final_log_dir,
          values.study_name,
          values.trial_number,
          values.seed,
          index,
        ].join("|"),
      );
      return {
        id: stringValue(values.result_id, fallbackId),
        values,
        dataset: stringValue(values.dataset, stringValue(values.log_group, "Unknown dataset")),
        scenario: stringValue(values.scenario, "Unlabeled scenario"),
        mode: stringValue(values.mode, stringValue(values.role, "model")),
        role: stringValue(values.role, stringValue(values.mode, "model")),
        loss: stringValue(values.loss, "Unknown loss"),
        miner: stringValue(values.miner, "no_miner"),
        sslMethod: stringValue(values.ssl_method, "none"),
        labelSamplingMode: stringValue(values.label_sampling_mode, "unspecified"),
        labeledFraction: numberValue(values.labeled_fraction),
        labeledPerClass: numberValue(values.labeled_per_class),
        seed: stringValue(values.seed, "unknown"),
        studyName: stringValue(values.study_name, "Unknown study"),
        trialNumber: stringValue(values.trial_number, "-"),
        recordedAt: stringValue(values.recorded_at_utc),
        testPrecisionAt1,
        testMapAtR,
        validPrecisionAt1: numberValue(values.best_valid_precision_at_1),
        validMapAtR: numberValue(values.best_valid_mean_average_precision_at_r),
      };
    })
    .filter((row) => row.testPrecisionAt1 != null || row.testMapAtR != null);
}

export function resultMetric(result: TestResult, metric: TestMetric): number | null {
  return metric === "test_mean_average_precision_at_r"
    ? result.testMapAtR
    : result.testPrecisionAt1;
}

function cleanLabel(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\bno miner\b/i, "no miner")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

export function labelBudget(result: TestResult): string {
  if (result.labeledPerClass != null) return `${formatCompact(result.labeledPerClass)} / class`;
  if (result.labeledFraction != null) return `${formatCompact(result.labeledFraction * 100)}% labeled`;
  return "Budget unknown";
}

export function testScenarioFamily(value: string): string {
  return value
    .replace(/_tune(?=_seed_|$)/gi, "")
    .replace(/_seed_-?\d+(?=$|_)/gi, "")
    .replace(/__+/g, "_")
    .replace(/^_+|_+$/g, "");
}

export function testConfigurationKey(result: TestResult): string {
  return [
    result.dataset,
    result.values.dataset_protocol,
    testScenarioFamily(result.scenario),
    result.mode,
    result.loss,
    result.miner,
    result.sslMethod,
    result.labelSamplingMode,
    result.labeledFraction,
    result.labeledPerClass,
    result.values.backbone_tuning,
    result.values.feat_dim,
    result.values.final_full_train,
  ]
    .map((value) => value ?? "")
    .join("\u001f");
}

function standardDeviation(values: number[], mean: number): number {
  if (values.length < 2) return 0;
  const sumOfSquares = values.reduce((sum, value) => sum + (value - mean) ** 2, 0);
  return Math.sqrt(sumOfSquares / (values.length - 1));
}

export function summarizeConfigurations(
  results: TestResult[],
  metric: TestMetric,
): ConfigurationSummary[] {
  const groups = new Map<string, TestResult[]>();
  results.forEach((result) => {
    if (resultMetric(result, metric) == null) return;
    const key = testConfigurationKey(result);
    groups.set(key, [...(groups.get(key) ?? []), result]);
  });

  return [...groups.entries()]
    .map(([key, group]) => {
      const first = group[0];
      const values = group
        .map((result) => resultMetric(result, metric))
        .filter((value): value is number => value != null);
      const mean = values.reduce((sum, value) => sum + value, 0) / values.length;
      const budget = labelBudget(first);
      const method =
        first.mode.toLowerCase() === "ssl" && first.sslMethod.toLowerCase() !== "none"
          ? cleanLabel(first.sslMethod)
          : cleanLabel(first.mode);
      const miner = first.miner.toLowerCase() === "no_miner" ? "no miner" : cleanLabel(first.miner);
      const runKind =
        first.values.final_full_train?.toLowerCase() === "true"
          ? "full train"
          : "selected checkpoint";
      return {
        key,
        label: `${cleanLabel(first.loss)} · ${method}`,
        details: `${first.dataset} · ${budget} · ${miner} · ${runKind}`,
        dataset: first.dataset,
        mode: first.mode,
        loss: first.loss,
        miner: first.miner,
        labelBudget: budget,
        count: values.length,
        seedCount: new Set(group.map((result) => result.seed)).size,
        mean,
        standardDeviation: standardDeviation(values, mean),
        minimum: Math.min(...values),
        maximum: Math.max(...values),
      };
    })
    .sort((left, right) => right.mean - left.mean || left.label.localeCompare(right.label));
}

export function formatMetric(value: number | null | undefined): string {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(4);
}

export function formatCompact(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return new Intl.NumberFormat("en-US", { maximumSignificantDigits: 4 }).format(value);
}

export function uniqueSorted(
  results: TestResult[],
  selector: (result: TestResult) => string,
): string[] {
  return [...new Set(results.map(selector).filter(Boolean))].sort((left, right) =>
    left.localeCompare(right),
  );
}
