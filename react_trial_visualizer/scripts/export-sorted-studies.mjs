/**
 * Exports the sorted HPO archive (`sorted_logs/`) into a single snapshot the
 * browser can filter without a server.
 *
 * The archive's directory layout already carries the facets the explorer needs
 *   <bucket>/<mode>/<dataset>/[<ssl_method>[__<variant>]/]<dml_loss>[__<variant>]
 * so the path is the primary source of truth; `sorting_map.csv` and each
 * study's `study_config.json` only fill in what a path cannot express (miner,
 * label budget, search space, campaign origin).
 *
 * Everything else comes from `optuna_study.db`, which is the only file present
 * for every study — the 19 barely-started ones under `unfinished/` never got a
 * trials.csv written.
 */
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  realpathSync,
  writeFileSync,
} from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";
import { gradientSummary } from "./gradient-summary.mjs";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(projectRoot, "..");
const outputPath = path.join(projectRoot, "public", "sorted-studies.json");
// Gradient diagnostics are three times the size of everything else combined and
// only one view needs them, so they get their own lazily fetched file.
const gradientOutputPath = path.join(projectRoot, "public", "sorted-gradients.json");
const databaseFileName = "optuna_study.db";

const archiveRoot = process.env.SORTED_LOGS_DIR?.trim()
  ? path.resolve(process.env.SORTED_LOGS_DIR)
  : path.join(repositoryRoot, "sorted_logs");

/**
 * Trial user attributes worth carrying into the browser.
 *
 * Validation only, deliberately. Nothing is tuned against the test split, so a
 * test number has no business appearing next to an objective where it could be
 * read, sorted or filtered as if it were one. The archive's HPO trials record
 * `test_*` attributes as null anyway, but leaving the keys out means a future
 * run that does populate them still cannot leak them into the explorer.
 */
const METRIC_KEYS = [
  "best_valid_mean_average_precision_at_r",
  "best_valid_precision_at_1",
  "best_valid_recall_at_k",
  "final_train_loss",
  "last_epoch",
  "selected_epoch",
  "global_step",
];

/** Everything else in trial_user_attributes is a path, a plot or a huge blob. */
const METRIC_KEY_SET = new Set([...METRIC_KEYS, "fold_results", "cv_k", "cv_mode"]);

const datasetLabels = new Map([
  ["cars196", "Cars196"],
  ["cub", "CUB-200"],
  ["in_shop", "In-Shop"],
  ["sop", "SOP"],
  ["semi_inat2021", "Semi-iNat 2021"],
  ["semi_aves", "Semi-Aves"],
  ["nabirds", "NABirds"],
  ["cifar", "CIFAR"],
]);

const methodLabels = new Map([
  ["iscen", "ISCEN"],
  ["lrml", "LRML"],
  ["hoffer", "Hoffer"],
  ["seraph", "SERAPH"],
  ["slade", "SLADE"],
  ["stml", "STML"],
  ["simmatch", "SimMatch"],
]);

const lossLabels = new Map([
  ["arcface", "ArcFace"],
  ["circle", "Circle"],
  ["multi_similarity", "Multi-Similarity"],
  ["ntxent", "NT-Xent"],
  ["proxy_anchor", "Proxy Anchor"],
  ["triplet", "Triplet"],
  ["tcm", "TCM"],
]);

const lossClassLabels = new Map([
  ["ArcFaceLoss", "ArcFace"],
  ["CircleLoss", "Circle"],
  ["MultiSimilarityLoss", "Multi-Similarity"],
  ["NTXentLoss", "NT-Xent"],
  ["ProxyAnchorLoss", "Proxy Anchor"],
  ["TripletMarginLoss", "Triplet"],
  ["ThresholdConsistentMarginLoss", "TCM"],
]);

function titleCase(value) {
  return String(value)
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function normalizeRelativePath(value) {
  return String(value).split(/[\\/]+/).filter(Boolean).join("/");
}

function readJson(filePath) {
  if (!existsSync(filePath)) return null;
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

/** Six significant digits keeps every chart honest and roughly halves the file. */
function round(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  if (Number.isInteger(number)) return number;
  return Number(number.toPrecision(6));
}

function finiteOrNull(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

// ---------------------------------------------------------------- CSV parsing

function parseCsv(text) {
  const records = [];
  let record = [];
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
  if (records.length === 0) return [];

  const columns = records[0];
  return records
    .slice(1)
    .map((values) =>
      Object.fromEntries(columns.map((column, index) => [column, values[index] ?? ""])),
    );
}

/** sorting_map.csv, keyed by the archive-relative study path. */
function readSortingMap() {
  const mapPath = path.join(archiveRoot, "sorting_map.csv");
  const byPath = new Map();
  if (!existsSync(mapPath)) return byPath;

  for (const row of parseCsv(readFileSync(mapPath, "utf8"))) {
    const newPath = normalizeRelativePath(row.new_path ?? "");
    if (newPath === "") continue;
    // Rows are written as `sorted_logs/<relative>`; strip the root segment.
    const relative = newPath.startsWith(`${path.basename(archiveRoot)}/`)
      ? newPath.slice(path.basename(archiveRoot).length + 1)
      : newPath;
    byPath.set(relative, row);
  }
  return byPath;
}

// -------------------------------------------------------------- facet parsing

/** Splits `lrml__initial_target_ratio` into its method and its campaign tag. */
function splitVariant(segment) {
  const marker = String(segment ?? "").indexOf("__");
  return marker === -1
    ? { base: String(segment ?? ""), variant: null }
    : {
        base: segment.slice(0, marker),
        variant: segment.slice(marker + 2),
      };
}

/**
 * The archive path is the authoritative facet source: `unfinished/` keeps the
 * phase it would have had, so bucket and phase are two separate facets.
 */
function facetsFromPath(relativePath) {
  const segments = relativePath.split("/");
  const bucket = segments[0] ?? "";
  const unfinished = bucket === "unfinished";
  const phase = unfinished ? (segments[1] ?? "") : bucket;
  const rest = segments.slice(unfinished ? 2 : 1);
  const mode = rest[0] ?? "";
  const dataset = rest[1] ?? "";
  const ssl = mode === "ssl";
  const methodSegment = ssl ? rest[2] : null;
  const lossSegment = ssl ? rest[3] : rest[2];

  const method = splitVariant(methodSegment ?? "");
  const loss = splitVariant(lossSegment ?? "");

  return {
    bucket: unfinished ? "unfinished" : "complete",
    phase,
    mode,
    dataset,
    method: ssl ? method.base : "supervised",
    // Supervised sweeps are flat, so their tie-breaking tag rides on the loss.
    variant: (ssl ? method.variant : loss.variant) ?? null,
    loss: loss.base,
  };
}

function labelBudget(baseArgs, relativePath) {
  const shots = finiteOrNull(baseArgs?.k_shot_grid?.[0]);
  const budget = finiteOrNull(baseArgs?.label_budget_grid?.[0]);
  if (budget != null && budget > 0 && budget < 1) {
    return `${Number((budget * 100).toFixed(2))}% labels`;
  }
  if (shots != null) return shots >= 99_999 ? "Full labels" : `${shots}-shot`;

  const fromPath = relativePath.match(/k_shot_label_[^/]*_k_(\d+)/i);
  if (fromPath) {
    const parsed = Number(fromPath[1]);
    return parsed >= 99_999 ? "Full labels" : `${parsed}-shot`;
  }
  return "Unknown";
}

// ------------------------------------------------------------- database reads

function decodeDistribution(distributionJson) {
  try {
    const parsed = JSON.parse(distributionJson);
    return {
      name: String(parsed?.name ?? ""),
      attributes:
        parsed?.attributes && typeof parsed.attributes === "object"
          ? parsed.attributes
          : {},
    };
  } catch {
    return { name: "", attributes: {} };
  }
}

function decodeParamValue(rawValue, distribution) {
  if (distribution.name === "CategoricalDistribution") {
    const choices = Array.isArray(distribution.attributes.choices)
      ? distribution.attributes.choices
      : [];
    const index = Math.round(Number(rawValue));
    return index >= 0 && index < choices.length ? choices[index] : Number(rawValue);
  }
  return round(rawValue);
}

function parseAttribute(valueJson) {
  try {
    return JSON.parse(valueJson);
  } catch {
    return null;
  }
}

function durationSeconds(start, complete) {
  if (!start || !complete) return null;
  const from = Date.parse(String(start).replace(" ", "T"));
  const to = Date.parse(String(complete).replace(" ", "T"));
  if (!Number.isFinite(from) || !Number.isFinite(to)) return null;
  return round((to - from) / 1000);
}

/**
 * The individual cross-validation folds behind a trial's aggregated objective.
 *
 * The trial-level attributes only keep the aggregate, which hides both the
 * spread the noise model needs and the per-fold epoch counts that say whether a
 * fold ran out of budget. Validation scores only, for the reason above.
 */
function foldRecords(foldResults, metricKey) {
  if (!Array.isArray(foldResults)) return null;
  const records = [];
  for (const [index, fold] of foldResults.entries()) {
    if (!fold || typeof fold !== "object") continue;
    const value = finiteOrNull(fold[metricKey]);
    if (value == null) continue;
    records.push({
      fold: finiteOrNull(fold.cv_fold) ?? index,
      value: round(value),
      ...(finiteOrNull(fold.best_valid_precision_at_1) != null
        ? { precisionAt1: round(fold.best_valid_precision_at_1) }
        : {}),
      ...(finiteOrNull(fold.last_epoch) != null
        ? { lastEpoch: finiteOrNull(fold.last_epoch) }
        : {}),
      ...(finiteOrNull(fold.selected_epoch) != null
        ? { selectedEpoch: finiteOrNull(fold.selected_epoch) }
        : {}),
      ...(finiteOrNull(fold.final_train_loss) != null
        ? { trainLoss: round(fold.final_train_loss) }
        : {}),
      ...(finiteOrNull(fold.global_step) != null
        ? { globalStep: finiteOrNull(fold.global_step) }
        : {}),
    });
  }
  return records.length > 0 ? records : null;
}

/**
 * The scalar SSL settings that stayed identical across every trial — the fixed
 * configuration that separates one campaign of a method from another.
 */
function constantConfig(configs) {
  if (configs.length === 0) return {};
  const flatten = (value, prefix, into) => {
    for (const [key, entry] of Object.entries(value ?? {})) {
      const name = prefix ? `${prefix}.${key}` : key;
      if (entry !== null && typeof entry === "object" && !Array.isArray(entry)) {
        flatten(entry, name, into);
      } else if (entry === null || typeof entry !== "object") {
        into[name] = entry;
      }
    }
    return into;
  };

  const first = flatten(configs[0], "", {});
  const constant = { ...first };
  for (const config of configs.slice(1)) {
    const flat = flatten(config, "", {});
    for (const key of Object.keys(constant)) {
      if (!Object.is(flat[key], constant[key])) delete constant[key];
    }
  }
  for (const [key, value] of Object.entries(constant)) {
    if (typeof value === "number") constant[key] = round(value);
  }
  return constant;
}

function readStudy(studyDirectory, relativePath, mapRow) {
  const config = readJson(path.join(studyDirectory, "study_config.json")) ?? {};
  const baseArgs = config.base_args ?? {};
  const hparamConfig = config.hparam_config ?? {};
  const facets = facetsFromPath(relativePath);
  const db = new DatabaseSync(path.join(studyDirectory, databaseFileName), {
    readOnly: true,
  });

  try {
    const studies = db
      .prepare("SELECT study_id, study_name FROM studies ORDER BY study_id")
      .all();
    const expectedName = config.resolved_study_name ?? mapRow?.study_name ?? null;
    const dbStudy =
      studies.find((study) => study.study_name === expectedName) ??
      (studies.length > 0 ? studies[0] : null);
    if (!dbStudy) throw new Error("no study row in the database");

    const directionRow = db
      .prepare(
        "SELECT direction FROM study_directions WHERE study_id = ? AND objective = 0",
      )
      .get(dbStudy.study_id);
    const direction =
      String(directionRow?.direction ?? hparamConfig.direction ?? "MAXIMIZE")
        .toLowerCase() === "minimize"
        ? "minimize"
        : "maximize";

    const trialRows = db
      .prepare(
        `SELECT trial_id, number, state, datetime_start, datetime_complete
         FROM trials WHERE study_id = ? ORDER BY number`,
      )
      .all(dbStudy.study_id);

    const trialById = new Map();
    for (const row of trialRows) {
      trialById.set(Number(row.trial_id), {
        number: Number(row.number),
        state: String(row.state),
        value: null,
        seconds: durationSeconds(row.datetime_start, row.datetime_complete),
        params: {},
        metrics: {},
        folds: null,
      });
    }
    if (trialById.size === 0) throw new Error("the study has no trials");

    for (const row of db
      .prepare(
        `SELECT tv.trial_id, tv.value FROM trial_values AS tv
         JOIN trials AS t ON t.trial_id = tv.trial_id
         WHERE t.study_id = ? AND tv.objective = 0 AND tv.value_type = 'FINITE'`,
      )
      .all(dbStudy.study_id)) {
      const trial = trialById.get(Number(row.trial_id));
      if (trial) trial.value = round(row.value);
    }

    const distributions = new Map();
    for (const row of db
      .prepare(
        `SELECT p.trial_id, p.param_name, p.param_value, p.distribution_json
         FROM trial_params AS p JOIN trials AS t ON t.trial_id = p.trial_id
         WHERE t.study_id = ? ORDER BY p.trial_id, p.param_id`,
      )
      .all(dbStudy.study_id)) {
      const trial = trialById.get(Number(row.trial_id));
      if (!trial) continue;
      const name = String(row.param_name);
      const distribution = decodeDistribution(String(row.distribution_json ?? ""));
      if (!distributions.has(name)) distributions.set(name, distribution);
      trial.params[name] = decodeParamValue(row.param_value, distribution);
    }

    const metricKey = String(
      hparamConfig.metric ?? "best_valid_mean_average_precision_at_r",
    );
    const sslConfigs = [];
    for (const row of db
      .prepare(
        `SELECT a.trial_id, a.key, a.value_json FROM trial_user_attributes AS a
         JOIN trials AS t ON t.trial_id = a.trial_id
         WHERE t.study_id = ?`,
      )
      .all(dbStudy.study_id)) {
      const key = String(row.key);
      const trial = trialById.get(Number(row.trial_id));
      if (!trial) continue;
      if (key === "resolved_ssl_config") {
        if (sslConfigs.length < 12) {
          const parsed = parseAttribute(String(row.value_json ?? ""));
          if (parsed && typeof parsed === "object") sslConfigs.push(parsed);
        }
        continue;
      }
      if (!METRIC_KEY_SET.has(key)) continue;
      const parsed = parseAttribute(String(row.value_json ?? ""));
      if (key === "fold_results") {
        trial.folds = foldRecords(parsed, metricKey);
        continue;
      }
      if (parsed === null || parsed === undefined) continue;
      trial.metrics[key] =
        typeof parsed === "number" ? round(parsed) : parsed;
    }

    const trials = [...trialById.values()].map((trial) => ({
      number: trial.number,
      state: trial.state,
      value: trial.value,
      ...(trial.seconds != null ? { seconds: trial.seconds } : {}),
      params: trial.params,
      metrics: trial.metrics,
      ...(trial.folds ? { folds: trial.folds } : {}),
    }));

    const complete = trials.filter(
      (trial) => trial.state === "COMPLETE" && trial.value != null,
    );
    const best = complete.reduce((champion, trial) => {
      if (!champion) return trial;
      return direction === "minimize"
        ? trial.value < champion.value
          ? trial
          : champion
        : trial.value > champion.value
          ? trial
          : champion;
    }, null);

    // The searched space, merged from what Optuna recorded and what the study
    // config asked for: a study that died early has spaces but no distributions.
    const configuredSpaces = hparamConfig.spaces ?? {};
    const parameterNames = [
      ...new Set([...distributions.keys(), ...Object.keys(configuredSpaces)]),
    ].sort();
    const importances = readJson(
      path.join(studyDirectory, "parameter_importance", "importances.json"),
    );
    const importanceByName = new Map(
      (Array.isArray(importances?.importances) ? importances.importances : [])
        .filter((row) => typeof row?.param === "string")
        .map((row) => [row.param, row]),
    );

    const parameters = parameterNames.map((name) => {
      const distribution = distributions.get(name) ?? { name: "", attributes: {} };
      const configured = configuredSpaces[name] ?? {};
      const attributes = distribution.attributes;
      const categorical =
        distribution.name === "CategoricalDistribution" ||
        configured.type === "categorical";
      const choices = Array.isArray(attributes.choices)
        ? attributes.choices
        : Array.isArray(configured.choices)
          ? configured.choices
          : undefined;
      const importance = importanceByName.get(name);
      return {
        name,
        kind: categorical ? "categorical" : "numeric",
        distribution: categorical
          ? "categorical"
          : distribution.name === "IntDistribution" || configured.type === "int"
            ? "int"
            : "float",
        log: Boolean(attributes.log ?? configured.log ?? false),
        ...(finiteOrNull(attributes.low ?? configured.low) != null
          ? { low: round(attributes.low ?? configured.low) }
          : {}),
        ...(finiteOrNull(attributes.high ?? configured.high) != null
          ? { high: round(attributes.high ?? configured.high) }
          : {}),
        ...(finiteOrNull(attributes.step ?? configured.step) != null
          ? { step: round(attributes.step ?? configured.step) }
          : {}),
        ...(choices ? { choices } : {}),
        ...(importance
          ? {
              importance: round(importance.importance),
              importanceRank: Number(importance.rank),
            }
          : {}),
        searched: distributions.has(name),
      };
    });

    const lossClass = String(baseArgs.loss ?? mapRow?.loss ?? "").trim();
    const dataset = String(baseArgs.dataset ?? mapRow?.dataset ?? "").trim();
    const datasetLabel =
      datasetLabels.get(facets.dataset) ?? dataset ?? titleCase(facets.dataset);
    const methodLabel =
      facets.mode === "supervised"
        ? "Supervised"
        : (methodLabels.get(facets.method) ?? titleCase(facets.method));
    const lossLabel =
      lossClassLabels.get(lossClass) ??
      lossLabels.get(facets.loss) ??
      titleCase(facets.loss);
    const sslConfig = facets.mode === "ssl" ? constantConfig(sslConfigs) : {};

    return {
      id: relativePath,
      path: relativePath,
      bucket: facets.bucket,
      phase: facets.phase,
      mode: facets.mode,
      dataset: facets.dataset,
      datasetLabel,
      method: facets.method,
      methodLabel,
      variant: facets.variant,
      loss: facets.loss,
      lossLabel,
      lossClass: lossClass || null,
      miner: String(baseArgs.miner ?? mapRow?.miner ?? "no_miner"),
      labelBudget: labelBudget(baseArgs, `${relativePath}/${mapRow?.run_name ?? ""}`),
      sampler: String(hparamConfig.sampler ?? mapRow?.sampler ?? "unknown"),
      direction,
      metric: metricKey,
      plannedTrials: finiteOrNull(hparamConfig.n_trials),
      studyName: String(dbStudy.study_name),
      runName: mapRow?.run_name ?? null,
      campaign: mapRow?.old_campaign_dir ?? null,
      sourcePath: mapRow?.old_path ?? null,
      experimentConfig: String(baseArgs.experiment_config ?? "") || null,
      hparamConfigPath: mapRow?.hparam_config ?? null,
      seed: finiteOrNull(baseArgs.seed),
      cvK: finiteOrNull(baseArgs.cv_k),
      cvMode: String(baseArgs.cv_mode ?? "") || null,
      epochs: finiteOrNull(baseArgs.epochs),
      earlyStoppingMetric: String(baseArgs.early_stopping_metric ?? "") || null,
      patience: finiteOrNull(baseArgs.patience),
      backboneTuning: String(baseArgs.backbone_tuning ?? "") || null,
      // Returned alongside the study rather than inside it: the gradient
      // diagnostics ship in their own file so the main snapshot stays small.
      gradients: gradientSummary(studyDirectory),
      trialCount: trials.length,
      completeTrials: complete.length,
      best: best ? { number: best.number, value: best.value } : null,
      sslConfig,
      parameters,
      trials,
    };
  } finally {
    db.close();
  }
}

// --------------------------------------------------------------------- driver

function findStudyDirectories(root) {
  const found = [];
  const stack = [root];
  // Visit every real directory once.
  const visited = new Set();
  while (stack.length > 0) {
    const directory = stack.pop();
    let real;
    try {
      real = realpathSync(directory);
    } catch {
      continue;
    }
    if (visited.has(real)) continue;
    visited.add(real);
    let entries;
    try {
      entries = readdirSync(directory, { withFileTypes: true });
    } catch {
      continue;
    }
    // A study directory holds only its own trials: stop there instead of walking
    // thousands of trial_* folders, which is slow on the F: drive.
    if (entries.some((entry) => entry.isFile() && entry.name === databaseFileName)) {
      found.push(directory);
      continue;
    }
    for (const entry of entries) {
      // Symlinks point to F: (Phase-0 data moved there); only the local archive is read.
      if (entry.isSymbolicLink()) continue;
      if (entry.isDirectory()) {
        if (!entry.name.startsWith(".")) stack.push(path.join(directory, entry.name));
      } else if (entry.isFile() && entry.name === databaseFileName) {
        found.push(directory);
      }
    }
  }
  return found.sort();
}

function write(snapshot, target = outputPath) {
  mkdirSync(path.dirname(target), { recursive: true });
  writeFileSync(target, JSON.stringify(snapshot));
}

if (!existsSync(archiveRoot)) {
  const reason = `No sorted log archive at ${archiveRoot}.`;
  if (existsSync(outputPath)) {
    console.log(`${reason} Keeping ${path.relative(projectRoot, outputPath)}.`);
  } else {
    write({
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      source: archiveRoot,
      studies: [],
      warning: reason,
    });
  }
  process.exit(0);
}

console.log(`Reading the sorted HPO archive from ${archiveRoot}.`);
const sortingMap = readSortingMap();
const directories = findStudyDirectories(archiveRoot);
const studies = [];
const failures = [];

const gradientsByStudy = {};

for (const directory of directories) {
  const relativePath = normalizeRelativePath(path.relative(archiveRoot, directory));
  try {
    const { gradients, ...study } = readStudy(
      directory,
      relativePath,
      sortingMap.get(relativePath),
    );
    if (gradients && Object.keys(gradients).length > 0) {
      gradientsByStudy[study.id] = gradients;
    }
    studies.push(study);
  } catch (error) {
    failures.push({
      study: relativePath,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

// Studies the map promised but that never landed on disk, so the explorer can
// say so instead of silently showing a smaller archive than SORTING.md claims.
const missing = [...sortingMap.keys()]
  .filter((relativePath) => !existsSync(path.join(archiveRoot, relativePath, databaseFileName)))
  .sort();

studies.sort((left, right) => left.path.localeCompare(right.path));

// Nothing here is tuned against the test split. Anything that optimizes a
// non-validation metric is a bug in the run, not a display choice, so it is
// reported loudly rather than quietly rendered next to the honest studies.
const objectiveMetrics = [...new Set(studies.map((study) => study.metric))].sort();
const testObjectives = studies.filter((study) => !study.metric.startsWith("best_valid_"));

write({
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  source: normalizeRelativePath(path.relative(repositoryRoot, archiveRoot)) || archiveRoot,
  metricKeys: METRIC_KEYS,
  studyDirectories: directories.length,
  mappedStudies: sortingMap.size,
  missingFromDisk: missing,
  objectiveMetrics,
  studies,
  ...(failures.length > 0 ? { failures } : {}),
});

write(
  {
    schemaVersion: 1,
    generatedAt: new Date().toISOString(),
    studies: gradientsByStudy,
  },
  gradientOutputPath,
);

const trialTotal = studies.reduce((total, study) => total + study.trialCount, 0);
const foldTotal = studies.reduce(
  (total, study) =>
    total + study.trials.reduce((count, trial) => count + (trial.folds?.length ?? 0), 0),
  0,
);
const gradientTrials = Object.values(gradientsByStudy).reduce(
  (total, trials) => total + Object.keys(trials).length,
  0,
);
console.log(
  `Exported ${studies.length} archived studies, ${trialTotal} trials and ${foldTotal} fold records to ${path.relative(projectRoot, outputPath)}.`,
);
console.log(
  `Exported gradient diagnostics for ${gradientTrials} trials across ${Object.keys(gradientsByStudy).length} studies to ${path.relative(projectRoot, gradientOutputPath)}.`,
);
console.log(`Objective metrics in the snapshot: ${objectiveMetrics.join(", ")}.`);
if (testObjectives.length > 0) {
  console.warn(
    `${testObjectives.length} studies optimize a metric that is not a validation metric:`,
  );
  for (const study of testObjectives) console.warn(`  ${study.path}: ${study.metric}`);
}
if (failures.length > 0) {
  console.warn(`Skipped ${failures.length} study directories:`);
  for (const failure of failures) console.warn(`  ${failure.study}: ${failure.error}`);
}
if (missing.length > 0) {
  console.warn(`${missing.length} studies in sorting_map.csv are not on disk.`);
}
