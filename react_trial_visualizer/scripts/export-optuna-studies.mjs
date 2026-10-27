import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(projectRoot, "..");
const outputPath = path.join(projectRoot, "public", "optuna-studies.json");
const importanceScriptPath = path.join(
  repositoryRoot,
  "scripts",
  "optuna_param_importance.py",
);

const databaseFileName = "optuna_study.db";
const summaryFileName = "parameter_importance_summary.csv";
const importanceTimeoutMs = 60 * 60 * 1000;
const skipImportance = process.env.OPTUNA_SKIP_IMPORTANCE === "1";
const forceImportance = process.env.OPTUNA_FORCE_IMPORTANCE === "1";

const defaultStudyRoots = [
  path.join(repositoryRoot, "exporter", "optuna-studies"),
  path.join(repositoryRoot, "logs"),
];

function normalizeRelativePath(value) {
  return String(value).split(/[\\/]+/).filter(Boolean).join("/");
}

function splitRootSetting(value) {
  const separators = process.platform === "win32" ? /[;,]+/ : /[:,]+/;
  return value
    .split(separators)
    .map((entry) => entry.trim())
    .filter(Boolean);
}

/**
 * Every configured study tree that exists on disk. Each root keeps its own
 * parameter_importance_summary.csv, so `logs` and the downloaded
 * `exporter/optuna-studies` mirror can be exported side by side.
 */
function resolveStudyRoots() {
  const configured = process.env.OPTUNA_STUDIES_DIR?.trim()
    ? splitRootSetting(process.env.OPTUNA_STUDIES_DIR)
    : defaultStudyRoots;
  const roots = [];
  const seenPaths = new Set();
  const usedKeys = new Set();

  for (const entry of configured) {
    const rootPath = path.resolve(entry);
    if (seenPaths.has(rootPath)) continue;
    seenPaths.add(rootPath);
    if (!existsSync(rootPath)) {
      console.warn(`Skipping missing Optuna study root: ${rootPath}`);
      continue;
    }

    const baseKey = normalizeRelativePath(path.basename(rootPath)) || "studies";
    let key = baseKey;
    for (let suffix = 2; usedKeys.has(key); suffix += 1) key = `${baseKey}-${suffix}`;
    usedKeys.add(key);
    roots.push({ key, path: rootPath });
  }

  return roots;
}

const studyRoots = resolveStudyRoots();
const prefixStudyIds = studyRoots.length > 1;
const sourceLabel =
  studyRoots
    .map((root) => {
      const relative = normalizeRelativePath(path.relative(repositoryRoot, root.path));
      return relative === "" || relative.startsWith("../") ? root.path : relative;
    })
    .join(", ") || "none";

function studyLabel(root, relativeInRoot) {
  return prefixStudyIds ? `${root.key}/${relativeInRoot}` : relativeInRoot;
}

function parseCsv(text, sourcePath) {
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
  if (quoted) throw new Error(`Unterminated quoted value in ${sourcePath}`);
  if (records.length === 0) return [];

  const columns = records[0];
  return records.slice(1).map((values) =>
    Object.fromEntries(columns.map((column, index) => [column, values[index] ?? ""])),
  );
}

function readJson(filePath) {
  if (!existsSync(filePath)) return null;
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function usableImportanceRow(row) {
  return (
    String(row.param ?? "").trim() !== "" &&
    Number.isFinite(Number(row.rank)) &&
    Number.isFinite(Number(row.importance))
  );
}

/**
 * Importance rows a previous batch run recorded for this root, grouped by the
 * study directory they belong to.
 */
function readSummary(root) {
  const summaryPath = path.join(root.path, summaryFileName);
  const rowsByStudy = new Map();
  const noteByStudy = new Map();
  const listedDirectories = new Set();
  if (!existsSync(summaryPath)) {
    return { rowsByStudy, noteByStudy, listedDirectories };
  }

  const rows = parseCsv(readFileSync(summaryPath, "utf8"), summaryPath);

  for (const row of rows) {
    const key = normalizeRelativePath(row.relative_study_dir ?? "");
    if (key === "") continue;
    listedDirectories.add(key);
    if (["ok", "skipped", "cached"].includes(row.status) && usableImportanceRow(row)) {
      const group = rowsByStudy.get(key) ?? [];
      group.push(row);
      rowsByStudy.set(key, group);
      continue;
    }
    // Why the batch run produced no ranking, straight from the summary.
    const note = String(row.skip_reason || row.error || row.status || "").trim();
    if (note !== "" && !noteByStudy.has(key)) noteByStudy.set(key, note);
  }

  return { rowsByStudy, noteByStudy, listedDirectories };
}

function fileMtimeMs(filePath) {
  try {
    return statSync(filePath).mtimeMs;
  } catch {
    return null;
  }
}

/**
 * Whether optuna_param_importance.py still owes this study a ranking, either
 * because it never produced one or because the study collected trials since.
 * Presence in the summary is not the test: a study the batch run skipped for
 * having too few trials stays listed forever and would never be retried once
 * it grows past two trials.
 */
function needsImportanceRefresh(root, relativeInRoot, summary) {
  if (!summary.rowsByStudy.get(relativeInRoot)?.length) return true;

  const studyDirectory = path.join(root.path, ...relativeInRoot.split("/"));
  const rankedAt = fileMtimeMs(
    path.join(studyDirectory, "parameter_importance", "importances.json"),
  );
  if (rankedAt === null) return true;
  const databaseAt = fileMtimeMs(path.join(studyDirectory, databaseFileName));
  return databaseAt !== null && databaseAt > rankedAt;
}

function findStudyDirectories(root) {
  const directories = [];
  const stack = [root.path];

  while (stack.length > 0) {
    const directory = stack.pop();
    let entries;
    try {
      entries = readdirSync(directory, { withFileTypes: true });
    } catch {
      continue;
    }

    for (const entry of entries) {
      if (entry.isSymbolicLink()) continue;
      if (entry.isDirectory()) {
        if (!entry.name.startsWith(".")) stack.push(path.join(directory, entry.name));
      } else if (entry.isFile() && entry.name === databaseFileName) {
        directories.push(directory);
      }
    }
  }

  return directories;
}

let resolvedPython;

function pythonCandidates() {
  const executable =
    process.platform === "win32" ? ["Scripts", "python.exe"] : ["bin", "python"];
  const candidates = [
    process.env.OPTUNA_PYTHON?.trim(),
    path.join(repositoryRoot, "venv", ...executable),
    path.join(repositoryRoot, ".venv", ...executable),
    process.platform === "win32" ? "python" : "python3",
    "python",
  ];
  return candidates.filter(
    (command, index) => command && candidates.indexOf(command) === index,
  );
}

/** The first candidate interpreter that can actually import Optuna. */
function resolvePython() {
  if (resolvedPython !== undefined) return resolvedPython;

  for (const command of pythonCandidates()) {
    if (command.includes(path.sep) && !existsSync(command)) continue;
    const probe = spawnSync(command, ["-c", "import optuna"], {
      stdio: "ignore",
      windowsHide: true,
      timeout: importanceTimeoutMs,
    });
    if (probe.error || probe.status !== 0) continue;
    resolvedPython = command;
    return resolvedPython;
  }

  resolvedPython = null;
  console.warn(
    "Could not find a Python interpreter with Optuna installed. Set OPTUNA_PYTHON to compute missing parameter importances.",
  );
  return resolvedPython;
}

/**
 * Ranks every study below one root with optuna_param_importance.py in batch
 * mode. The script keeps each study's report in <study>/parameter_importance
 * and rewrites <root>/parameter_importance_summary.csv, so --skip-existing
 * makes repeat runs cheap and only new studies are actually analyzed.
 */
function refreshImportanceSummary(root, missingCount) {
  if (!existsSync(importanceScriptPath)) {
    console.warn(
      `Could not refresh parameter importance: missing ${importanceScriptPath}.`,
    );
    return false;
  }
  const python = resolvePython();
  if (!python) return false;

  const args = [importanceScriptPath, "--studies-root", root.path, "--no-plots"];
  if (!forceImportance) args.push("--skip-existing");

  console.log(
    forceImportance
      ? `Re-analyzing every study below ${root.key}...`
      : `Analyzing ${missingCount} unranked study/studies below ${root.key}...`,
  );
  const result = spawnSync(python, args, {
    cwd: repositoryRoot,
    env: process.env,
    stdio: "inherit",
    windowsHide: true,
    timeout: importanceTimeoutMs,
  });

  if (result.error) {
    console.warn(`Parameter importance refresh failed: ${result.error.message}`);
    return false;
  }
  if (result.status !== 0) {
    console.warn(
      `Parameter importance refresh exited with code ${String(result.status)}; exporting every usable result it produced.`,
    );
  }
  return true;
}

function titleCase(value) {
  return value
    .replace(/[_-]+/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

const lossLabels = new Map([
  ["ArcFaceLoss", "ArcFace"],
  ["CircleLoss", "Circle Loss"],
  ["MultiSimilarityLoss", "Multi-Similarity"],
  ["NTXentLoss", "NT-Xent"],
  ["ProxyAnchorLoss", "Proxy Anchor"],
]);

const methodLabels = new Map([
  ["simmatch_v2", "SimMatch v2"],
  ["hoffer", "Hoffer"],
  ["iscen", "ISCEN"],
  ["lrml", "LRML"],
  ["seraph", "SERAPH"],
  ["simmatch", "SimMatch"],
  ["slade", "SLADE"],
  ["slrml", "SLRML"],
  ["slrmml", "SLRMML"],
  ["stml", "STML"],
]);

function detectMethod(relativePath, baseArgs) {
  const mode = String(baseArgs?.mode ?? "").toLowerCase();
  if (mode === "supervised") return "Supervised baseline";
  const normalized = [
    relativePath,
    baseArgs?.save_dir,
    baseArgs?.experiment_config,
    baseArgs?.experiment_config_resolved?.save_dir,
    baseArgs?.experiment_config_resolved?.ssl_config,
  ]
    .filter(Boolean)
    .join("/")
    .toLowerCase();
  for (const [needle, label] of methodLabels) {
    if (new RegExp(`(^|[/_-])${needle}([/_.-]|$)`).test(normalized)) {
      return mode === "ssl" ? `${label} (SSL)` : label;
    }
  }
  return mode === "ssl"
    ? "SSL (method not identified)"
    : String(baseArgs?.mode ?? "HPO").toUpperCase();
}

function detectLabelSetting(relativePath, baseArgs) {
  const shotMatch = relativePath.match(/k_shot_label_[^/]*_k_(\d+)/i);
  if (shotMatch) {
    const shots = Number(shotMatch[1]);
    return shots >= 99_999 ? "Full labels" : `${shots}-shot`;
  }
  const fractionMatch = relativePath.match(/class_subset_label_([0-9]+p[0-9]+)/i);
  if (fractionMatch) {
    const fraction = Number(fractionMatch[1].replace("p", "."));
    if (Number.isFinite(fraction)) return `${Math.round(fraction * 100)}% labels`;
  }
  const configuredShots = Number(baseArgs?.k_shot_grid?.[0]);
  if (Number.isFinite(configuredShots)) {
    return configuredShots >= 99_999 ? "Full labels" : `${configuredShots}-shot`;
  }
  return "Label budget";
}

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
    const choiceIndex = Math.round(Number(rawValue));
    return choiceIndex >= 0 && choiceIndex < choices.length
      ? choices[choiceIndex]
      : rawValue;
  }
  return Number(rawValue);
}

function finiteNumber(value) {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function parameterDescriptor(importanceRow, distribution, configuredSpace) {
  const attributes = distribution.attributes;
  const choices = Array.isArray(attributes.choices)
    ? attributes.choices
    : Array.isArray(configuredSpace?.choices)
      ? configuredSpace.choices
      : undefined;
  const categorical =
    distribution.name === "CategoricalDistribution" ||
    configuredSpace?.type === "categorical";
  const distributionName = categorical
    ? "categorical"
    : distribution.name === "IntDistribution" || configuredSpace?.type === "int"
      ? "int"
      : "float";

  return {
    name: importanceRow.param,
    rank: Number(importanceRow.rank),
    importance: Number(importanceRow.importance),
    kind: categorical ? "categorical" : "numeric",
    distribution: distributionName,
    log: Boolean(attributes.log ?? configuredSpace?.log ?? false),
    ...(finiteNumber(attributes.low ?? configuredSpace?.low) != null
      ? { low: finiteNumber(attributes.low ?? configuredSpace?.low) }
      : {}),
    ...(finiteNumber(attributes.high ?? configuredSpace?.high) != null
      ? { high: finiteNumber(attributes.high ?? configuredSpace?.high) }
      : {}),
    ...(finiteNumber(attributes.step ?? configuredSpace?.step) != null
      ? { step: finiteNumber(attributes.step ?? configuredSpace?.step) }
      : {}),
    ...(choices ? { choices } : {}),
  };
}

function studyConfig(entry) {
  if (entry.config === undefined) {
    entry.config =
      readJson(path.join(entry.studyDirectory, "study_config.json")) ?? {};
  }
  return entry.config;
}

function buildStudy(entry, importanceRows) {
  const { relativeInRoot, databasePath, label } = entry;
  const config = studyConfig(entry);
  const expectedStudyName =
    importanceRows[0]?.study_name ||
    config.resolved_study_name ||
    null;
  const db = new DatabaseSync(databasePath, { readOnly: true });

  try {
    const availableStudies = db
      .prepare("SELECT study_id, study_name FROM studies ORDER BY study_id")
      .all();
    const databaseStudy =
      availableStudies.find((study) => study.study_name === expectedStudyName) ??
      (availableStudies.length === 1 ? availableStudies[0] : null);
    if (!databaseStudy) {
      throw new Error(
        `Could not resolve study ${String(expectedStudyName)} in ${label}`,
      );
    }

    const directionRow = db
      .prepare(
        "SELECT direction FROM study_directions WHERE study_id = ? AND objective = 0",
      )
      .get(databaseStudy.study_id);
    const direction =
      String(directionRow?.direction ?? config.hparam_config?.direction ?? "MAXIMIZE")
        .toLowerCase() === "minimize"
        ? "minimize"
        : "maximize";

    const trialRows = db
      .prepare(
        `SELECT t.trial_id, t.number, tv.value
         FROM trials AS t
         JOIN trial_values AS tv
           ON tv.trial_id = t.trial_id AND tv.objective = 0
         WHERE t.study_id = ?
           AND t.state = 'COMPLETE'
           AND tv.value_type = 'FINITE'
           AND tv.value IS NOT NULL
         ORDER BY t.number`,
      )
      .all(databaseStudy.study_id)
      .filter((trial) => Number.isFinite(Number(trial.value)));

    const trialById = new Map(
      trialRows.map((trial) => [
        Number(trial.trial_id),
        {
          number: Number(trial.number),
          objective: Number(trial.value),
          params: {},
        },
      ]),
    );
    const includedParameterNames = new Set(
      importanceRows.map((row) => row.param),
    );
    const distributionByParameter = new Map();

    const paramRows = db
      .prepare(
        `SELECT p.trial_id, p.param_name, p.param_value, p.distribution_json
         FROM trial_params AS p
         JOIN trials AS t ON t.trial_id = p.trial_id
         WHERE t.study_id = ? AND t.state = 'COMPLETE'
         ORDER BY p.trial_id, p.param_id`,
      )
      .all(databaseStudy.study_id);

    for (const row of paramRows) {
      const parameterName = String(row.param_name);
      if (!includedParameterNames.has(parameterName)) continue;
      const trial = trialById.get(Number(row.trial_id));
      if (!trial) continue;
      const distribution = decodeDistribution(String(row.distribution_json ?? ""));
      if (!distributionByParameter.has(parameterName)) {
        distributionByParameter.set(parameterName, distribution);
      }
      trial.params[parameterName] = decodeParamValue(
        row.param_value,
        distribution,
      );
    }

    const configuredSpaces = config.hparam_config?.spaces ?? {};
    const parameters = [...importanceRows]
      .sort((left, right) => Number(left.rank) - Number(right.rank))
      .map((row) =>
        parameterDescriptor(
          row,
          distributionByParameter.get(row.param) ?? {
            name: "",
            attributes: {},
          },
          configuredSpaces[row.param],
        ),
      )
      .filter(
        (parameter) =>
          Number.isFinite(parameter.rank) &&
          Number.isFinite(parameter.importance),
      );
    const trials = [...trialById.values()].filter((trial) =>
      parameters.some((parameter) =>
        Object.hasOwn(trial.params, parameter.name),
      ),
    );
    const bestTrial = trials.reduce((best, trial) => {
      if (!best) return trial;
      return direction === "minimize"
        ? trial.objective < best.objective
          ? trial
          : best
        : trial.objective > best.objective
          ? trial
          : best;
    }, null);

    const baseArgs = config.base_args ?? {};
    const loss = String(baseArgs.loss ?? "").trim();
    const lossLabel = lossLabels.get(loss) ?? (loss ? titleCase(loss) : "HPO");
    const method = detectMethod(relativeInRoot, baseArgs);
    const labelSetting = detectLabelSetting(relativeInRoot, baseArgs);
    const dataset = String(baseArgs.dataset ?? "Unknown dataset");
    const collection =
      relativeInRoot.startsWith("scheduled/")
        ? "Current"
        : relativeInRoot.startsWith("old_scheduled_maybe/")
          ? "Archive"
          : relativeInRoot.startsWith("ssl_methods/")
            ? "SSL Methods"
            : relativeInRoot.startsWith("ssl/")
              ? "SSL"
              : titleCase(relativeInRoot.split("/")[0] ?? "Studies");

    return {
      id: label,
      relativePath: label,
      studyName: String(databaseStudy.study_name),
      displayName: [dataset, method, lossLabel, labelSetting]
        .filter(Boolean)
        .join(" · "),
      dataset,
      method,
      loss: lossLabel,
      labelSetting,
      collection,
      direction,
      target: String(config.hparam_config?.metric ?? "objective"),
      sampler: String(config.hparam_config?.sampler ?? "unknown"),
      completeTrials: trials.length,
      reportedCompleteTrials: Number(
        importanceRows[0]?.num_complete_trials ?? trials.length,
      ),
      bestTrial: bestTrial
        ? { number: bestTrial.number, objective: bestTrial.objective }
        : null,
      parameters,
      trials,
    };
  } finally {
    db.close();
  }
}

function writeEmptySnapshot(reason) {
  mkdirSync(path.dirname(outputPath), { recursive: true });
  writeFileSync(
    outputPath,
    JSON.stringify({
      schemaVersion: 1,
      generatedAt: new Date().toISOString(),
      source: sourceLabel,
      totalStudyDirectories: 0,
      skippedStudyDirectories: 0,
      studies: [],
      warning: reason,
    }),
  );
}

function keepOrWriteEmpty(reason) {
  if (existsSync(outputPath)) {
    console.log(
      `${reason} Keeping ${path.relative(projectRoot, outputPath)}.`,
    );
  } else {
    writeEmptySnapshot(reason);
  }
  process.exit(0);
}

if (studyRoots.length === 0) {
  keepOrWriteEmpty("No Optuna study root was available.");
}

console.log(`Reading Optuna studies from ${sourceLabel}.`);

/** Every study directory of every root, with the importances already known. */
const entries = [];
const failures = [];
const seenDirectories = new Set();

if (skipImportance) {
  console.log("Skipping Optuna importance refresh (OPTUNA_SKIP_IMPORTANCE=1).");
}

for (const root of studyRoots) {
  const discovered = findStudyDirectories(root).map((directory) =>
    normalizeRelativePath(path.relative(root.path, directory)),
  );
  let summary = readSummary(root);

  // Studies without a current ranking are the only reason to start Python.
  const unranked = discovered.filter((relativeInRoot) =>
    needsImportanceRefresh(root, relativeInRoot, summary),
  );
  if (!skipImportance && (forceImportance || unranked.length > 0)) {
    if (refreshImportanceSummary(root, unranked.length)) summary = readSummary(root);
  }

  const relativeDirectories = [
    ...new Set([...discovered, ...summary.listedDirectories]),
  ].sort();

  for (const relativeInRoot of relativeDirectories) {
    const studyDirectory = path.join(root.path, ...relativeInRoot.split("/"));
    if (seenDirectories.has(studyDirectory)) continue;
    seenDirectories.add(studyDirectory);
    const databasePath = path.join(studyDirectory, databaseFileName);
    entries.push({
      root,
      relativeInRoot,
      studyDirectory,
      databasePath,
      label: studyLabel(root, relativeInRoot),
      importanceRows: summary.rowsByStudy.get(relativeInRoot) ?? null,
      skipReason: existsSync(databasePath)
        ? summary.noteByStudy.get(relativeInRoot)
        : `Missing ${databaseFileName}`,
    });
  }
}

const studies = [];
let skippedDirectories = 0;

for (const entry of entries) {
  if (!entry.importanceRows?.length) {
    skippedDirectories += 1;
    if (entry.skipReason) {
      failures.push({ relativeStudyDir: entry.label, error: entry.skipReason });
    }
    continue;
  }
  try {
    studies.push(buildStudy(entry, entry.importanceRows));
  } catch (error) {
    skippedDirectories += 1;
    failures.push({
      relativeStudyDir: entry.label,
      error: error instanceof Error ? error.message : String(error),
    });
  }
}

if (studies.length === 0) {
  keepOrWriteEmpty("No Optuna study with parameter importances was found.");
}

studies.sort((left, right) => {
  if (left.collection !== right.collection) {
    if (left.collection === "Current") return -1;
    if (right.collection === "Current") return 1;
    return left.collection.localeCompare(right.collection);
  }
  return left.displayName.localeCompare(right.displayName) ||
    left.relativePath.localeCompare(right.relativePath);
});

const snapshot = {
  schemaVersion: 1,
  generatedAt: new Date().toISOString(),
  source: sourceLabel,
  totalStudyDirectories: entries.length,
  skippedStudyDirectories: skippedDirectories,
  studies,
  ...(failures.length > 0 ? { failures } : {}),
};

mkdirSync(path.dirname(outputPath), { recursive: true });
writeFileSync(outputPath, JSON.stringify(snapshot));

const trialCount = studies.reduce(
  (total, study) => total + study.completeTrials,
  0,
);
console.log(
  `Exported ${studies.length} Optuna studies and ${trialCount} completed trials to ${path.relative(projectRoot, outputPath)}.`,
);
if (skippedDirectories > 0) {
  console.warn(
    `Skipped ${skippedDirectories} study directories without usable importances.`,
  );
}
