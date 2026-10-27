/**
 * Exports what every run under F:\Studien (`/mnt/f/Studien`) was actually
 * configured with, for the /configs/ page:
 *
 *   Phase_0/<Datensatz>/<Verfahren>/<Verlust>[__<Variante>]      HPO studies (random search)
 *   Phase_1/<Datensatz>/<Verfahren>/<Verlust>[__<Variante>]      HPO studies (TPE)
 *   _weitere/Phase_{0,1}/…                                      every other study, same layout
 *   Seeded_Runs/<Datensatz>/<Verfahren>/<Verlust>[__<Variante>]  seeded replays of the best trial
 *                                                                (series the thesis does not use → `weitere`)
 *   Phase_2/<root>/<combo>[_<variant>]                          ablation / geometry replays
 *
 * An HPO study's settings come from its `study_config.json` (the arguments every
 * trial started from, plus the search space) and its `optuna_study.db` (trial
 * states, the best trial and that trial's resolved SSL config). A replay
 * folder holds one run per budget and seed; each run's `cv_summary.json` (or,
 * for a run that never finished, its first fold's `run_config.json`) carries
 * the resolved arguments, the resolved SSL config, the replayed trial and any
 * ablation overrides. Settings that differ between the runs of one folder are
 * kept, so a series that mixed protocols shows up as such.
 *
 * What was planned but is missing comes from two places: a replay folder's
 * `study_replay_grid/ssl_configs/*.json` (the grid it was launched with) and
 * the Phase-2 run manifests in `configs/run_manifests/phase2/`, which name
 * every `<save_dir_root>/<run>_<variant>` folder and its budgets and seeds.
 *
 * F: is a Windows drive, so reads are slow; every parsed summary is cached in
 * `.study-configs-cache.json` by path, size and mtime, and a repeat export only
 * reads what changed. Nothing under the studies root is ever written.
 */
import { existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, readlinkSync, statSync, writeFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { DatabaseSync } from "node:sqlite";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const repositoryRoot = path.resolve(projectRoot, "..");
const outputPath = path.join(projectRoot, "public", "study-configs.json");
const cachePath = path.join(projectRoot, ".study-configs-cache.json");
const CACHE_VERSION = 2;

const studiesRoot = path.resolve(process.env.STUDIES_DIR?.trim() || "/mnt/f/Studien");
const manifestRoot = path.resolve(
  process.env.PHASE2_MANIFESTS_DIR?.trim() || path.join(repositoryRoot, "configs", "run_manifests", "phase2"),
);

// ------------------------------------------------------------------ naming

/** Dataset folders the HPO and seeded trees use; anything else is reported as skipped. */
const DATASETS = new Set([
  "Cars196", "CUB", "InShop", "InShop_10", "SemiINat",
  "CIFAR100", "iNat", "iNat2018", "iNat2019", "iNat2021Mini", "NABirds", "SOP",
]);

/** Phase-2 save roots → dataset, and what kind of replay a root holds. */
const PHASE2_ROOT_DATASET = { cars: "Cars196", cub: "CUB", in_shop: "InShop", in_shop_10: "InShop_10", semi_inat: "SemiINat" };
const COMBO_METHOD = { sup: "Ueberwacht", iscen: "LP", lrml: "LRML", hoffer: "Hoffer", slade: "SLADE" };
const COMBO_LOSS = { arc: "ArcFace", cir: "Circle", ms: "MultiSimilarity", nt: "NTXent", prx: "ProxyAnchor" };
// Same pattern main_eval/common.py uses to split a Phase-2 folder name.
const COMBO = /^(sup|iscen|lrml|hoffer|slade)_(arc|cir|ms|nt|prx)(?:_(.+))?$/;

const LOSS_CLASS = {
  ArcFaceLoss: "ArcFace",
  CircleLoss: "Circle",
  MultiSimilarityLoss: "MultiSimilarity",
  NTXentLoss: "NTXent",
  ProxyAnchorLoss: "ProxyAnchor",
  TripletMarginLoss: "TripletMargin",
};

function phase2Root(relativeRoot) {
  // "cars", "cars_geometry", "in_shop_10_geometry", "cars_full", "weight_decay/cars_high_lr"
  const name = relativeRoot.replace(/^Phase_2\//, "");
  if (name.startsWith("weight_decay/")) {
    const base = name.slice("weight_decay/".length).replace(/_(high|huge)_lr$/, "");
    return { dataset: PHASE2_ROOT_DATASET[base] ?? base, rootKind: "weight_decay" };
  }
  const match = /^(.*?)(?:_(geometry|full))?$/.exec(name);
  const base = match?.[1] ?? name;
  const suffix = match?.[2];
  return {
    dataset: PHASE2_ROOT_DATASET[base] ?? base,
    rootKind: suffix === "geometry" ? "geometry" : suffix === "full" ? "external" : null,
  };
}

/** What a Phase-2 folder is: the kind the page groups by. */
function phase2Kind(rootKind, variant) {
  if (rootKind) return rootKind;
  const bare = variant.replace(/^(controls|sweeps)_/, "");
  if (!variant) return "replay";
  if (["defaults", "sup_all_shared", "pool_labeled"].includes(bare)) return "2x2";
  if (/^external_/.test(bare)) return "external";
  if (/^ood_/.test(bare)) return "ood";
  if (/^(lr_|adamw?_wd)/.test(bare)) return "weight_decay";
  if (/^sup_/.test(bare) || bare === "parallel") return "hp_path";
  if (variant.startsWith("sweeps_")) return "sweeps";
  if (variant.startsWith("controls_")) return "controls";
  if (/^(distance_scale|n_neighbors|confidence_threshold|warmup_epochs|regularizer_target_ratio|cluster_ratio|update_interval_epochs)_/.test(bare)) {
    return "sweeps";
  }
  if (/^(zero_weight|no_warmup|pl_once|graph_once|graph_|oracle_noise_|diag$|pseudo_label_)/.test(bare)) return "controls";
  return "other";
}

// ------------------------------------------------------------------ settings

/**
 * Arguments that say nothing about how a run was trained: paths, logging,
 * visualisation, parallelism and memory placement. Everything else is kept and
 * sorted into sections on the page.
 */
const DROPPED_KEYS = new Set([
  "ablation_config", "ablation_config_resolved", "ablation_variant", "backbone_cache_dir", "batch_loss_log_interval",
  "cv_fold", "dataloader_start_method", "device", "evaluate_saved_heads", "experiment_config", "experiment_config_resolved",
  "explicit_cli_args", "final_study_request_overrides", "final_test_study_dir", "fold_test_embedding_storage",
  "hparam_config", "hparam_config_resolved", "hparam_final_epoch_plan", "hparam_final_from_study",
  "hparam_final_trial_number", "hparam_params", "log_dir", "low_memory_mode", "metrics_csv", "model_total_parameters",
  "model_trainable_parameters", "model_uses_backbone_cache", "num_threads", "num_workers", "projection_3d",
  "report_test_metrics", "resume_interrupted_runs", "retry_failed_hpo_trials", "save_dir", "save_final_head_state",
  "save_head_state", "save_test_embeddings", "save_trial_split_data", "ssl_config", "ssl_config_resolved",
  "ssl_device", "tensorboard", "torch_sharing_strategy_resolved", "frozen_feature_precompute",
  "evaluation_embedding_residency", "final_test_visualization", "pseudo_label_diagnostics_mode",
]);
const DROPPED_PREFIXES = [
  "debug_", "visualization_", "tsne_", "track_epoch_", "faiss_", "cv_parallel_", "frozen_feature_cache",
  "frozen_feature_residency", "length_before_new_iter_", "compcars_", "graph_diagnostics_",
];
const DROPPED_SSL_KEYS = new Set([
  "embedding_batch_size", "embedding_num_workers", "pseudo_label_diagnostics_mode", "seed", "support_seed",
]);

function keepArg(key) {
  if (DROPPED_KEYS.has(key)) return false;
  if (DROPPED_PREFIXES.some((prefix) => key.startsWith(prefix))) return false;
  if (/_resolved$|_log_interval$|(^|_)(path|dir)$|_path$|_dir$/.test(key)) return false;
  return true;
}

/** Settings that are supposed to differ between the runs of one replay folder. */
const RUN_AXIS_KEYS = new Set([
  "seed", "support_seed", "comparison_seeds", "comparison_seed_targets", "k_shot_grid", "label_budget_grid",
  "ssl.labeled_per_class", "ssl.labeled_fraction", "length_before_new_iter",
]);

function round(value) {
  if (typeof value !== "number" || !Number.isFinite(value)) return value;
  if (Number.isInteger(value)) return value;
  return Number(value.toPrecision(6));
}

function normalize(value) {
  if (value === undefined) return null;
  if (typeof value === "number") return round(value);
  if (Array.isArray(value)) return value.map(normalize);
  return value;
}

/** Flattens nested objects into dotted keys; arrays and scalars stay values. */
function flatten(object, prefix, output, keep = () => true) {
  if (!object || typeof object !== "object") return output;
  for (const [key, value] of Object.entries(object)) {
    if (!prefix && !keep(key)) continue;
    const name = prefix ? `${prefix}.${key}` : key;
    if (value && typeof value === "object" && !Array.isArray(value)) {
      if (Object.keys(value).length === 0) output[name] = "{}";
      else flatten(value, name, output);
    } else {
      output[name] = normalize(value);
    }
  }
  return output;
}

function settingsFrom(args, sslConfig) {
  const settings = flatten(args ?? {}, "", {}, keepArg);
  if (sslConfig && typeof sslConfig === "object") {
    const ssl = {};
    for (const [key, value] of Object.entries(sslConfig)) {
      if (!DROPPED_SSL_KEYS.has(key) && !key.startsWith("graph_diagnostics_")) ssl[key] = value;
    }
    flatten(ssl, "ssl", settings);
  }
  return settings;
}

/** Where a search-space parameter lands among the settings. */
function settingKeysOfParam(param) {
  if (param === "batch_sampler") return ["batch_size", "sampler_m"];
  if (param.startsWith("ssl_config.")) return [`ssl.${param.slice("ssl_config.".length)}`];
  const loss = /^loss\.[^.]+\.(.+)$/.exec(param);
  if (loss) return [`loss_params.${loss[1]}`];
  const miner = /^miner\.[^.]+\.(.+)$/.exec(param);
  if (miner) return [`miner_params.${miner[1]}`];
  return [param];
}

function settingValuesOfParam(param, value) {
  if (param === "batch_sampler" && typeof value === "string" && value.includes(":")) {
    const [size, m] = value.split(":").map(Number);
    return { batch_size: size, sampler_m: m };
  }
  const keys = settingKeysOfParam(param);
  return Object.fromEntries(keys.map((key) => [key, normalize(value)]));
}

function normalizeSpace(spaces, perLoss, lossClass) {
  const merged = { ...(spaces ?? {}), ...((perLoss ?? {})[lossClass]?.spaces ?? (perLoss ?? {})[lossClass] ?? {}) };
  const output = {};
  for (const [name, spec] of Object.entries(merged)) {
    if (!spec || typeof spec !== "object") continue;
    const type = String(spec.type ?? (Array.isArray(spec.choices) ? "categorical" : "float"));
    output[name] =
      type === "categorical"
        ? { type, choices: (spec.choices ?? []).map(normalize) }
        : {
            type,
            low: normalize(spec.low),
            high: normalize(spec.high),
            ...(spec.log ? { log: true } : {}),
            ...(spec.step != null ? { step: normalize(spec.step) } : {}),
          };
  }
  return output;
}

// ------------------------------------------------------------------ files

function readJson(filePath) {
  try {
    return JSON.parse(readFileSync(filePath, "utf8"));
  } catch {
    return null;
  }
}

function isDirectory(filePath) {
  try {
    return statSync(filePath).isDirectory();
  } catch {
    return false;
  }
}

/** Sub-directories by name, following links (every study folder here is a junction). */
function listDirectories(directory) {
  try {
    return readdirSync(directory, { withFileTypes: true })
      .filter((entry) => entry.isDirectory() || (entry.isSymbolicLink() && isDirectory(path.join(directory, entry.name))))
      .map((entry) => entry.name)
      .sort();
  } catch {
    return [];
  }
}

function listFiles(directory, pattern) {
  try {
    return readdirSync(directory).filter((name) => pattern.test(name)).sort();
  } catch {
    return [];
  }
}

function linkTarget(directory) {
  try {
    return lstatSync(directory).isSymbolicLink() ? readlinkSync(directory) : null;
  } catch {
    return null;
  }
}

function relative(filePath) {
  return path.relative(studiesRoot, filePath).split(path.sep).join("/");
}

function fileStamp(filePath) {
  try {
    const stat = statSync(filePath);
    return `${stat.size}:${Math.round(stat.mtimeMs)}`;
  } catch {
    return null;
  }
}

// ------------------------------------------------------------------ cache

let cache = { version: CACHE_VERSION, entries: {} };
if (existsSync(cachePath)) {
  const stored = readJson(cachePath);
  if (stored?.version === CACHE_VERSION && stored.entries) cache = stored;
}
const usedCacheKeys = new Set();
let cacheHits = 0;
let cacheMisses = 0;

function cached(key, stamp, compute) {
  usedCacheKeys.add(key);
  const hit = cache.entries[key];
  if (stamp && hit && hit.stamp === stamp) {
    cacheHits += 1;
    return hit.value;
  }
  cacheMisses += 1;
  const value = compute();
  if (stamp) cache.entries[key] = { stamp, value };
  return value;
}

// ------------------------------------------------------------------ HPO studies

const OPTUNA_STATES = { COMPLETE: "complete", FAIL: "failed", PRUNED: "pruned", RUNNING: "running", WAITING: "waiting" };

function decodeParamValue(rawValue, distributionJson) {
  try {
    const distribution = JSON.parse(String(distributionJson ?? ""));
    if (distribution?.name === "CategoricalDistribution") {
      const choices = distribution.attributes?.choices ?? [];
      const index = Math.round(Number(rawValue));
      return index >= 0 && index < choices.length ? choices[index] : Number(rawValue);
    }
    if (distribution?.name === "IntDistribution" || distribution?.name === "IntUniformDistribution") {
      return Math.round(Number(rawValue));
    }
  } catch {
    // Unknown distribution: keep the raw number.
  }
  return round(Number(rawValue));
}

function parseTime(value) {
  if (!value) return null;
  const time = Date.parse(String(value).replace(" ", "T"));
  return Number.isFinite(time) ? time : null;
}

function median(values) {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function spread(values) {
  return values.length ? { median: round(median(values)), min: Math.min(...values), max: Math.max(...values) } : null;
}

/**
 * How long a study or series ran and how far its folds trained. `units` are
 * trials or runs with a known duration; a fold hit the epoch cap when its last
 * epoch reached `epochs - 1` (the same rule the HPO archive uses).
 */
function runtimeSummary({ starts, ends, durations, folds, epochs }) {
  const start = starts.length ? Math.min(...starts) : null;
  const end = ends.length ? Math.max(...ends) : null;
  const lasts = folds.map((fold) => fold.last).filter(Number.isFinite);
  const selected = folds.map((fold) => fold.selected).filter(Number.isFinite);
  return {
    start: start != null ? new Date(start).toISOString() : null,
    end: end != null ? new Date(end).toISOString() : null,
    spanSeconds: start != null && end != null && end >= start ? Math.round((end - start) / 1000) : null,
    computeSeconds: durations.length ? Math.round(durations.reduce((sum, value) => sum + value, 0)) : null,
    unitSeconds: durations.length ? Math.round(median(durations)) : null,
    units: durations.length,
    folds: lasts.length,
    lastEpoch: spread(lasts),
    selectedEpoch: spread(selected),
    cap: Number.isFinite(epochs) ? epochs : null,
    capHits: Number.isFinite(epochs) ? lasts.filter((last) => last >= epochs - 1).length : null,
  };
}

function readOptunaDatabase(databasePath, studyName, direction, epochs) {
  const db = new DatabaseSync(databasePath, { readOnly: true });
  try {
    const studies = db.prepare("SELECT study_id, study_name FROM studies ORDER BY study_id").all();
    const study = studies.find((row) => row.study_name === studyName) ?? studies[0];
    if (!study) return { error: "no study in optuna_study.db" };

    const counts = { complete: 0, failed: 0, pruned: 0, running: 0, waiting: 0 };
    for (const row of db.prepare("SELECT state, COUNT(*) AS n FROM trials WHERE study_id = ? GROUP BY state").all(study.study_id)) {
      const state = OPTUNA_STATES[String(row.state)];
      if (state) counts[state] += Number(row.n);
    }

    // Wall time from the trial timestamps; epochs per fold of every finished trial.
    const starts = [];
    const ends = [];
    const durations = [];
    for (const row of db
      .prepare("SELECT state, datetime_start, datetime_complete FROM trials WHERE study_id = ?")
      .all(study.study_id)) {
      const from = parseTime(row.datetime_start);
      const to = parseTime(row.datetime_complete);
      if (from != null) starts.push(from);
      if (to != null) ends.push(to);
      if (row.state === "COMPLETE" && from != null && to != null && to >= from) durations.push((to - from) / 1000);
    }
    let folds = [];
    try {
      folds = db
        .prepare(
          `SELECT json_extract(f.value, '$.last_epoch') AS last, json_extract(f.value, '$.selected_epoch') AS selected
           FROM trial_user_attributes AS a JOIN trials AS t ON t.trial_id = a.trial_id, json_each(a.value_json) AS f
           WHERE t.study_id = ? AND t.state = 'COMPLETE' AND a.key = 'fold_results' AND json_valid(a.value_json)`,
        )
        .all(study.study_id)
        .map((row) => ({ last: Number(row.last ?? NaN), selected: Number(row.selected ?? NaN) }));
    } catch {
      folds = [];
    }
    if (folds.length === 0) {
      // Single-split studies keep no fold list; their trial attributes are the one fold.
      const byTrial = new Map();
      for (const row of db
        .prepare(
          `SELECT a.trial_id, a.key, a.value_json FROM trial_user_attributes AS a JOIN trials AS t ON t.trial_id = a.trial_id
           WHERE t.study_id = ? AND t.state = 'COMPLETE' AND a.key IN ('last_epoch', 'selected_epoch')`,
        )
        .all(study.study_id)) {
        const fold = byTrial.get(row.trial_id) ?? { last: NaN, selected: NaN };
        fold[row.key === "last_epoch" ? "last" : "selected"] = Number(readAttribute(row.value_json) ?? NaN);
        byTrial.set(row.trial_id, fold);
      }
      folds = [...byTrial.values()];
    }
    const runtime = runtimeSummary({ starts, ends, durations, folds, epochs });

    const order = direction === "minimize" ? "ASC" : "DESC";
    const best = db
      .prepare(
        `SELECT t.trial_id, t.number, tv.value FROM trials AS t
         JOIN trial_values AS tv ON tv.trial_id = t.trial_id
         WHERE t.study_id = ? AND t.state = 'COMPLETE' AND tv.objective = 0 AND tv.value_type = 'FINITE'
         ORDER BY tv.value ${order} LIMIT 1`,
      )
      .get(study.study_id);

    let bestParams = null;
    let sslConfig = null;
    if (best) {
      bestParams = {};
      for (const row of db
        .prepare("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id = ? ORDER BY param_id")
        .all(best.trial_id)) {
        bestParams[String(row.param_name)] = normalize(decodeParamValue(row.param_value, row.distribution_json));
      }
    }
    // The best trial's resolved SSL config, or any trial's when the best one did not store it.
    const sslRow =
      (best &&
        db
          .prepare("SELECT value_json FROM trial_user_attributes WHERE trial_id = ? AND key = 'resolved_ssl_config'")
          .get(best.trial_id)) ||
      db
        .prepare(
          `SELECT a.value_json FROM trial_user_attributes AS a JOIN trials AS t ON t.trial_id = a.trial_id
           WHERE t.study_id = ? AND a.key = 'resolved_ssl_config' LIMIT 1`,
        )
        .get(study.study_id);
    if (sslRow) sslConfig = readAttribute(sslRow.value_json);

    return {
      studyName: String(study.study_name),
      counts,
      best: best ? { number: Number(best.number), value: round(Number(best.value)), params: bestParams } : null,
      sslConfig,
      runtime,
    };
  } finally {
    db.close();
  }
}

function readAttribute(valueJson) {
  try {
    return JSON.parse(String(valueJson));
  } catch {
    return null;
  }
}

function readHpoStudy(directory) {
  const configPath = path.join(directory, "study_config.json");
  const databasePath = path.join(directory, "optuna_study.db");
  const stamp = [fileStamp(configPath), fileStamp(databasePath)].join("|");
  return cached(`hpo:${relative(directory)}`, stamp, () => {
    const notes = [];
    const config = readJson(configPath);
    if (!config) notes.push(existsSync(configPath) ? "study_config.json is not valid JSON" : "no study_config.json");
    const baseArgs = config?.base_args ?? null;
    const hparam = config?.hparam_config ?? {};
    const direction = String(hparam.direction ?? "maximize").toLowerCase() === "minimize" ? "minimize" : "maximize";

    let database = null;
    if (existsSync(databasePath)) {
      try {
        database = readOptunaDatabase(
          databasePath,
          config?.resolved_study_name ?? hparam.study_name,
          direction,
          Number(baseArgs?.epochs ?? NaN),
        );
        if (database.error) notes.push(database.error);
      } catch (error) {
        notes.push(`optuna_study.db unreadable: ${error instanceof Error ? error.message : String(error)}`);
      }
    } else {
      notes.push("no optuna_study.db");
    }

    const lossClass = baseArgs?.loss ?? null;
    const space = config ? normalizeSpace(hparam.spaces, hparam.per_loss, lossClass) : {};
    const settings = baseArgs ? settingsFrom(baseArgs, database?.sslConfig) : {};
    const tuned = new Set();
    for (const param of Object.keys(space)) for (const key of settingKeysOfParam(param)) tuned.add(key);
    // A tuned argument's base value is only the placeholder every trial overwrote; show the best trial's instead.
    if (database?.best?.params) {
      for (const [param, value] of Object.entries(database.best.params)) Object.assign(settings, settingValuesOfParam(param, value));
    }
    if (config) {
      Object.assign(settings, {
        "hpo.sampler": hparam.sampler ?? null,
        "hpo.n_trials": hparam.n_trials ?? null,
        "hpo.startup_trials": config.effective_tpe_startup_trials ?? config.resolved_tpe_startup_trials ?? hparam.tpe_startup_trials ?? null,
        "hpo.sampler_params": JSON.stringify(hparam.sampler_params ?? {}),
        "hpo.metric": hparam.metric ?? null,
        "hpo.direction": direction,
        "hpo.pruner": hparam.pruner ?? null,
      });
    }

    return {
      config: config ? "study_config.json" : null,
      lossClass,
      settings,
      tuned: [...tuned].sort(),
      space,
      hpo: config || database
        ? {
            sampler: hparam.sampler ?? null,
            planned: hparam.n_trials ?? null,
            ...(database?.counts ?? { complete: 0, failed: 0, pruned: 0, running: 0, waiting: 0 }),
            studyName: database?.studyName ?? config?.resolved_study_name ?? null,
            best: database?.best ?? null,
          }
        : null,
      runtime: database?.runtime ?? null,
      notes,
    };
  });
}

// ------------------------------------------------------------------ replay runs

function parseRunName(name) {
  const k = /_k_(\d+)_comparison_seed_(\d+)$/.exec(name);
  if (k) return { budget: `k=${k[1]}`, seed: Number(k[2]) };
  const fraction = /_label_(\d+(?:p\d+)?)_comparison_seed_(\d+)$/.exec(name);
  if (fraction) return { budget: `${Math.round(Number(fraction[1].replace("p", ".")) * 1000) / 10}%`, seed: Number(fraction[2]) };
  const seed = /_seed_(\d+)$/.exec(name);
  return { budget: "?", seed: seed ? Number(seed[1]) : null };
}

/** `cv_2026-09-27_16-54-30` → epoch ms, read in this machine's time zone like the file times. */
function attemptStart(name) {
  const match = /^cv_(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})/.exec(name);
  if (!match) return null;
  const [, year, month, day, hour, minute, second] = match.map(Number);
  return new Date(year, month - 1, day, hour, minute, second).getTime();
}

/** The newest run_config.json of a fold: `fold_00/<timestamp>/run_config.json` or `fold_00/run_config.json`. */
function foldRunConfigPath(cvDirectory) {
  const fold = path.join(cvDirectory, "fold_00");
  const direct = path.join(fold, "run_config.json");
  if (existsSync(direct)) return direct;
  for (const stamp of listDirectories(fold).reverse()) {
    const nested = path.join(fold, stamp, "run_config.json");
    if (existsSync(nested)) return nested;
  }
  return null;
}

function summarizeRunArgs(args, sslConfig) {
  const changes = args?.ablation_config_resolved?.changes ?? null;
  return {
    settings: settingsFrom(args, sslConfig),
    replay: {
      study: args?.hparam_final_from_study ?? null,
      trial: args?.hparam_final_trial_number ?? null,
      params: args?.hparam_params ? flatten(args.hparam_params, "", {}) : null,
    },
    ablation: args?.ablation_config_resolved
      ? {
          name: args.ablation_config_resolved.name ?? null,
          variant: args.ablation_config_resolved.variant ?? args.ablation_variant ?? null,
          changes: changes ? flatten(changes, "", {}) : {},
        }
      : null,
  };
}

/** One seed directory: its most complete cv_* attempt, read from cv_summary.json or the first fold's run_config.json. */
function readSeedRun(seedDirectory) {
  const finalDirectory = path.join(seedDirectory, "final");
  const attempts = [];
  for (const mode of listDirectories(finalDirectory)) {
    for (const cv of listDirectories(path.join(finalDirectory, mode)).filter((name) => name.startsWith("cv_"))) {
      attempts.push(path.join(finalDirectory, mode, cv));
    }
  }
  attempts.sort((left, right) => path.basename(right).localeCompare(path.basename(left)));

  let best = null;
  for (const attempt of attempts) {
    const summaryPath = path.join(attempt, "cv_summary.json");
    const stamp = fileStamp(summaryPath);
    if (!stamp) continue;
    const record = cached(`run2:${relative(summaryPath)}`, stamp, () => {
      const summary = readJson(summaryPath);
      if (!summary) return null;
      const args = summary.args ?? {};
      // Older summaries do not carry the resolved SSL config; the first fold's run_config.json does.
      let sslConfig = args.ssl_config_resolved ?? null;
      if (!sslConfig) {
        const configPath = foldRunConfigPath(attempt);
        sslConfig = configPath ? readJson(configPath)?.ssl_config ?? null : null;
      }
      const cvK = Number(summary.cv_k ?? args.cv_k ?? 0) || null;
      const completedFolds = Number(summary.completed_folds ?? 0);
      const foldList = summary.aggregate?.fold_results ?? (Array.isArray(summary.folds) ? summary.folds : []);
      // Started when the cv_<timestamp> folder was created, finished when the summary was last written.
      const start = attemptStart(path.basename(attempt));
      const end = statSync(summaryPath).mtimeMs;
      const seconds = start != null && end >= start && end - start < 14 * 86_400_000 ? Math.round((end - start) / 1000) : null;
      return {
        source: "cv_summary.json",
        status: summary.aggregate && cvK && completedFolds >= cvK ? "complete" : "partial",
        folds: completedFolds,
        cvK,
        start,
        end: seconds != null ? end : null,
        seconds,
        foldEpochs: foldList
          .filter((fold) => fold && typeof fold === "object")
          .map((fold) => ({ last: Number(fold.last_epoch ?? NaN), selected: Number(fold.selected_epoch ?? NaN) })),
        ...summarizeRunArgs(args, sslConfig),
      };
    });
    if (!record) continue;
    if (record.status === "complete") {
      best = record;
      break;
    }
    best ??= record;
  }

  if (!best) {
    // Never summarised: the newest attempt that at least wrote a fold config.
    for (const attempt of attempts) {
      const configPath = foldRunConfigPath(attempt);
      const stamp = configPath ? fileStamp(configPath) : null;
      if (!stamp) continue;
      best = cached(`run2:${relative(configPath)}`, stamp, () => {
        const runConfig = readJson(configPath);
        if (!runConfig) return null;
        return {
          source: "run_config.json",
          status: "started",
          folds: 0,
          cvK: Number(runConfig.args?.cv_k ?? 0) || null,
          start: attemptStart(path.basename(attempt)),
          end: null,
          seconds: null,
          foldEpochs: [],
          ...summarizeRunArgs(runConfig.args ?? {}, runConfig.ssl_config ?? null),
        };
      });
      if (best) break;
    }
  }
  const empty = { source: null, status: "empty", folds: 0, cvK: null, start: null, end: null, seconds: null, foldEpochs: [], settings: {}, replay: null, ablation: null };
  return { ...(best ?? empty), attempts: attempts.length };
}

function valueKey(value) {
  return JSON.stringify(value ?? null);
}

function readReplayGroup(directory) {
  const names = listDirectories(directory).filter((name) => /_comparison_seed_\d+$/.test(name));
  const plannedNames = listFiles(path.join(directory, "study_replay_grid", "ssl_configs"), /\.json$/).map((name) =>
    name.replace(/\.json$/, ""),
  );
  const runs = names.map((name) => ({ name, ...parseRunName(name), ...readSeedRun(path.join(directory, name)) }));
  const withConfig = runs.filter((run) => run.source);

  // Modal value per setting across the runs that recorded it, and every setting on which they
  // disagree. A run that does not record a setting at all (older code) is not a disagreement.
  const keys = new Set(withConfig.flatMap((run) => Object.keys(run.settings)));
  const settings = {};
  const varying = {};
  for (const key of keys) {
    const groups = new Map();
    for (const run of withConfig) {
      if (!(key in run.settings)) continue;
      const value = run.settings[key];
      const id = valueKey(value);
      if (!groups.has(id)) groups.set(id, { value, runs: [] });
      groups.get(id).runs.push(run.name);
    }
    const ranked = [...groups.values()].sort((left, right) => right.runs.length - left.runs.length);
    settings[key] = ranked[0].value;
    if (ranked.length > 1 && !RUN_AXIS_KEYS.has(key)) {
      varying[key] = ranked.map((group) => ({ value: group.value, count: group.runs.length, runs: group.runs.slice(0, 8) }));
    }
  }

  // Which trial the runs replayed; more than one entry means the folder mixes sources.
  const sources = new Map();
  for (const run of withConfig) {
    if (!run.replay) continue;
    const id = valueKey([run.replay.study, run.replay.trial, run.replay.params]);
    if (!sources.has(id)) sources.set(id, { ...run.replay, count: 0 });
    sources.get(id).count += 1;
  }
  const ablations = new Map();
  for (const run of withConfig) {
    if (!run.ablation) continue;
    const id = valueKey(run.ablation);
    if (!ablations.has(id)) ablations.set(id, { ...run.ablation, count: 0 });
    ablations.get(id).count += 1;
  }

  const count = (status) => runs.filter((run) => run.status === status).length;
  const epochs = Number(settings.epochs ?? NaN);
  const finished = runs.filter((run) => run.status === "complete");
  const runtime = runtimeSummary({
    starts: runs.map((run) => run.start).filter((value) => value != null),
    ends: runs.map((run) => run.end).filter((value) => value != null),
    durations: finished.map((run) => run.seconds).filter((value) => value != null),
    folds: runs.flatMap((run) => run.foldEpochs ?? []),
    epochs,
  });
  return {
    config: withConfig.length ? withConfig[0].source : null,
    settings,
    runtime,
    replay: {
      planned: plannedNames.length ? plannedNames : null,
      runs: runs.map((run) => {
        const lasts = (run.foldEpochs ?? []).map((fold) => fold.last).filter(Number.isFinite);
        const runEpochs = Number(run.settings?.epochs ?? epochs);
        return {
          name: run.name,
          budget: run.budget,
          seed: run.seed,
          status: run.status,
          folds: run.folds,
          cvK: run.cvK,
          attempts: run.attempts,
          seconds: run.seconds,
          lastEpoch: lasts.length ? Math.max(...lasts) : null,
          capHits: Number.isFinite(runEpochs) ? lasts.filter((last) => last >= runEpochs - 1).length : null,
        };
      }),
      complete: count("complete"),
      partial: count("partial"),
      started: count("started"),
      empty: count("empty"),
      varying,
      sources: [...sources.values()].sort((left, right) => right.count - left.count),
      ablations: [...ablations.values()].sort((left, right) => right.count - left.count),
    },
    notes: withConfig.length ? [] : ["no run in this folder wrote a config yet"],
  };
}

// ------------------------------------------------------------------ discovery

const entries = [];
const skipped = [];

function splitVariant(name) {
  const index = name.indexOf("__");
  return index < 0 ? { loss: name, variant: "" } : { loss: name.slice(0, index), variant: name.slice(index + 2) };
}

function scanHpoTree(phaseDirectory, phase, archive) {
  for (const dataset of listDirectories(phaseDirectory)) {
    const datasetDirectory = path.join(phaseDirectory, dataset);
    if (!DATASETS.has(dataset)) {
      skipped.push({ path: relative(datasetDirectory), reason: "not a dataset folder (unsorted copy or side folder)" });
      continue;
    }
    for (const method of listDirectories(datasetDirectory)) {
      for (const name of listDirectories(path.join(datasetDirectory, method))) {
        const directory = path.join(datasetDirectory, method, name);
        const { loss, variant } = splitVariant(name);
        const study = readHpoStudy(directory);
        entries.push({
          id: relative(directory),
          phase,
          archive,
          dataset,
          method,
          loss,
          variant,
          target: linkTarget(directory),
          ...study,
        });
      }
    }
  }
}

/**
 * The seeded series the thesis reports, per dataset and method: the folder variant (the part after
 * `__`), or per loss where one loss uses another series. Every other seeded folder is a repeat the
 * thesis does not use and is exported as `weitere`. Source: F:\Studien\Ergaenzung_Arbeit_2026-09-29.md.
 */
const THESIS_SEEDED_VARIANT = {
  Cars196: {
    Ueberwacht: "600ep_Patience20",
    LP: "600ep_Patience20",
    LRML: "600ep_Patience20",
    Hoffer: { default: "Patience20", NTXent: "Patience40" },
    SLADE: "Patience20",
  },
  CUB: { Ueberwacht: "", LP: "", LRML: "ungekappt", Hoffer: "", SLADE: "" },
  InShop: { Ueberwacht: "600ep_Patience20", LP: "600ep_Patience20", LRML: "600ep_Patience20", Hoffer: "", SLADE: "" },
  InShop_10: { Ueberwacht: "", LP: "", LRML: "", Hoffer: "", SLADE: "" },
  SemiINat: { Ueberwacht: "", LP: "", LRML: "", Hoffer: "", SLADE: "" },
};

function seededArchive(dataset, method, loss, variant) {
  const rule = THESIS_SEEDED_VARIANT[dataset]?.[method];
  if (rule === undefined) return "weitere";
  const used = typeof rule === "string" ? rule : (rule[loss] ?? rule.default);
  return variant === used ? "main" : "weitere";
}

function scanSeeded(directory) {
  for (const dataset of listDirectories(directory)) {
    if (dataset.startsWith("_")) continue;
    if (!DATASETS.has(dataset)) {
      skipped.push({ path: relative(path.join(directory, dataset)), reason: "not a dataset folder" });
      continue;
    }
    for (const method of listDirectories(path.join(directory, dataset))) {
      for (const name of listDirectories(path.join(directory, dataset, method))) {
        const folder = path.join(directory, dataset, method, name);
        const { loss, variant } = splitVariant(name);
        entries.push({
          id: relative(folder),
          phase: "seeded",
          archive: seededArchive(dataset, method, loss, variant),
          dataset,
          method,
          loss,
          variant,
          target: linkTarget(folder),
          ...readReplayGroup(folder),
        });
      }
    }
  }
}

function scanPhase2Root(rootDirectory) {
  const rootId = relative(rootDirectory);
  const { dataset, rootKind } = phase2Root(rootId);
  for (const name of listDirectories(rootDirectory)) {
    if (name.startsWith("_") || name.startsWith(".")) continue;
    const match = COMBO.exec(name);
    const folder = path.join(rootDirectory, name);
    if (!match) {
      skipped.push({ path: relative(folder), reason: "not a <method>_<loss>[_<variant>] folder" });
      continue;
    }
    const variant = match[3] ?? "";
    entries.push({
      id: relative(folder),
      phase: "phase2",
      archive: "main",
      dataset,
      method: COMBO_METHOD[match[1]],
      loss: COMBO_LOSS[match[2]],
      variant,
      root: rootId.replace(/^Phase_2\//, ""),
      kind: phase2Kind(rootKind, variant),
      target: linkTarget(folder),
      ...readReplayGroup(folder),
    });
  }
}

function scanPhase2(directory) {
  for (const name of listDirectories(directory)) {
    if (name.startsWith("_") || name.startsWith(".")) {
      skipped.push({ path: relative(path.join(directory, name)), reason: "side folder (deviating copies, partial or scheduler logs)" });
      continue;
    }
    if (name === "weight_decay") {
      for (const sub of listDirectories(path.join(directory, name))) scanPhase2Root(path.join(directory, name, sub));
      continue;
    }
    scanPhase2Root(path.join(directory, name));
  }
}

// ------------------------------------------------------------------ Phase-2 plan

function budgetToken(value) {
  return String(value).replace(".", "p");
}

function walkJson(directory) {
  const files = [];
  for (const entry of (() => {
    try {
      return readdirSync(directory, { withFileTypes: true });
    } catch {
      return [];
    }
  })()) {
    const full = path.join(directory, entry.name);
    if (entry.isDirectory()) files.push(...walkJson(full));
    else if (entry.name.endsWith(".json")) files.push(full);
  }
  return files.sort();
}

/**
 * Every folder the Phase-2 manifests would write, as run_scheduler.py expands
 * them: `<save_dir_root>/<run>` for a plain replay, one `<run>_<variant>` per
 * ablation variant (all of the config's, or `ablation_variants`), and one run
 * per budget and comparison seed inside it. Disabled runs are kept but marked,
 * because the scheduler never launches them.
 */
function readPhase2Plan() {
  const folders = new Map();
  const manifests = [];
  for (const file of walkJson(manifestRoot)) {
    const manifest = readJson(file);
    if (!manifest?.runs || !manifest.save_dir_root) continue;
    const manifestId = path.relative(manifestRoot, file).split(path.sep).join("/");
    const replay = manifest.study_replay ?? {};
    const seeds = replay.comparison_seeds ?? [];
    const runNames = [];
    for (const k of replay.k_shot_grid ?? []) {
      for (const seed of seeds) runNames.push(`class_subset_k_shot_label_1_k_${k}_comparison_seed_${seed}`);
    }
    for (const budget of replay.label_budget_grid ?? []) {
      for (const seed of seeds) runNames.push(`class_subset_label_${budgetToken(budget)}_comparison_seed_${seed}`);
    }
    const root = `Phase_2/${String(manifest.save_dir_root).replace(/^(final\/)?phase_2\//, "")}`;
    manifests.push({ id: manifestId, root: root.replace(/^Phase_2\//, ""), runs: manifest.runs.length });

    for (const run of manifest.runs) {
      if (!run?.name) continue;
      let variants = [""];
      let ablationName = null;
      if (run.ablation_config) {
        const ablation = readJson(path.join(repositoryRoot, run.ablation_config));
        ablationName = String(run.ablation_config).replace(/^configs\/ablations\//, "");
        const all = Object.keys(ablation?.variants ?? {});
        variants = run.ablation_variants ? all.filter((variant) => run.ablation_variants.includes(variant)) : all;
        if (!ablation) variants = run.ablation_variants ?? ["?"];
      }
      for (const variant of variants) {
        const folderName = variant ? `${run.name}_${variant}` : run.name;
        const id = `${root}/${folderName}`;
        if (!folders.has(id)) {
          const match = COMBO.exec(folderName);
          const { dataset, rootKind } = phase2Root(root);
          const folderVariant = match?.[3] ?? variant;
          folders.set(id, {
            id,
            root: root.replace(/^Phase_2\//, ""),
            dataset,
            method: match ? COMBO_METHOD[match[1]] : null,
            loss: match ? COMBO_LOSS[match[2]] : null,
            variant: folderVariant,
            kind: phase2Kind(rootKind, folderVariant),
            manifests: [],
            ablations: [],
            runs: [],
            enabled: false,
          });
        }
        const folder = folders.get(id);
        if (!folder.manifests.includes(manifestId)) folder.manifests.push(manifestId);
        if (ablationName && !folder.ablations.includes(ablationName)) folder.ablations.push(ablationName);
        if (run.enabled !== false) {
          folder.enabled = true;
          for (const name of runNames) if (!folder.runs.includes(name)) folder.runs.push(name);
        }
      }
    }
  }
  return { folders: [...folders.values()].sort((left, right) => left.id.localeCompare(right.id)), manifests };
}

// ------------------------------------------------------------------ run

const started = Date.now();
if (!isDirectory(studiesRoot)) {
  console.error(`[study-configs] ${studiesRoot} is not a directory; set STUDIES_DIR.`);
  process.exit(1);
}

for (const phase of ["Phase_0", "Phase_1"]) {
  scanHpoTree(path.join(studiesRoot, phase), phase === "Phase_0" ? "phase0" : "phase1", "main");
  scanHpoTree(path.join(studiesRoot, "_weitere", phase), phase === "Phase_0" ? "phase0" : "phase1", "weitere");
}
console.log(`[study-configs] HPO studies: ${entries.length} (${((Date.now() - started) / 1000).toFixed(1)} s)`);
scanSeeded(path.join(studiesRoot, "Seeded_Runs"));
console.log(`[study-configs] + seeded series: ${entries.length} (${((Date.now() - started) / 1000).toFixed(1)} s)`);
scanPhase2(path.join(studiesRoot, "Phase_2"));
console.log(`[study-configs] + Phase-2 folders: ${entries.length} (${((Date.now() - started) / 1000).toFixed(1)} s)`);

const plan = readPhase2Plan();

// Setting names repeat in every entry and make up half the file; each entry
// stores [key index, value] pairs against one shared key list instead.
const settingKeys = [...new Set(entries.flatMap((entry) => Object.keys(entry.settings)))].sort();
const keyIndex = new Map(settingKeys.map((key, index) => [key, index]));
for (const entry of entries) {
  entry.settings = Object.entries(entry.settings).map(([key, value]) => [keyIndex.get(key), value]);
}

const snapshot = {
  version: 1,
  generatedAt: new Date().toISOString(),
  root: studiesRoot,
  manifestRoot: path.relative(repositoryRoot, manifestRoot).split(path.sep).join("/"),
  settingKeys,
  entries: entries.sort((left, right) => left.id.localeCompare(right.id)),
  phase2Plan: plan.folders,
  manifests: plan.manifests,
  skipped,
};

mkdirSync(path.dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(snapshot)}\n`);

for (const key of Object.keys(cache.entries)) if (!usedCacheKeys.has(key)) delete cache.entries[key];
writeFileSync(cachePath, JSON.stringify(cache));

const size = statSync(outputPath).size / 1024 / 1024;
console.log(
  `[study-configs] wrote ${entries.length} entries, ${plan.folders.length} planned Phase-2 folders from ${plan.manifests.length} manifests ` +
    `to ${path.relative(projectRoot, outputPath)} (${size.toFixed(1)} MB) in ${((Date.now() - started) / 1000).toFixed(1)} s ` +
    `(cache: ${cacheHits} reused, ${cacheMisses} read)`,
);
