import type { Issue, SpaceSpec, StudioSchema } from "./schema";

/** Classification of one HPO search-space key against the real argument surface. */
export interface SpaceKeyInfo {
  known: boolean;
  kind: "training" | "batch-sampler" | "labeled-batch" | "loss" | "miner" | "ssl" | "unknown";
  description: string;
  /** Set when the key targets a loss/miner class that does not accept it. */
  problem?: string;
}

export function isCategoricalSpec(spec: SpaceSpec): boolean {
  if (Array.isArray(spec)) return true;
  const record = spec as Record<string, unknown>;
  return record.type === "categorical" || (record.type === undefined && "choices" in record);
}

export function specChoices(spec: SpaceSpec): unknown[] {
  if (Array.isArray(spec)) return spec;
  const choices = (spec as { choices?: unknown[] }).choices;
  return Array.isArray(choices) ? choices : [];
}

/** Explain what a search-space key targets, and whether the target exists. */
export function describeSpaceKey(key: string, schema: StudioSchema): SpaceKeyInfo {
  const { constants } = schema;

  if (key === constants.batchSamplerKey) {
    return {
      known: true,
      kind: "batch-sampler",
      description: "Joint batch_size and sampler_m, written as 'batch_size:sampler_m'.",
    };
  }
  if (key === constants.labeledBatchFractionKey) {
    return {
      known: true,
      kind: "labeled-batch",
      description: `Labeled share of each batch, in (0, ${constants.maxLabeledBatchFraction}].`,
    };
  }
  if (constants.labeledBatchSizeKeys.includes(key)) {
    return {
      known: true,
      kind: "labeled-batch",
      description: "Absolute labeled stream size; categorical integer choices only.",
    };
  }

  if (key.startsWith("loss.") || key.startsWith("miner.")) {
    const parts = key.split(".");
    const component = parts[0] as "loss" | "miner";
    if (parts.length === 2) {
      return {
        known: true,
        kind: component,
        description: `Applies to whichever ${component} the run selects.`,
      };
    }
    if (parts.length !== 3 || !parts[2]) {
      return {
        known: false,
        kind: component,
        description: "",
        problem: `Use '${component}.<parameter>' or '${component}.<ClassName>.<parameter>'.`,
      };
    }
    const [, className, parameter] = parts;
    const classes = component === "loss" ? constants.losses : constants.miners;
    if (!classes.includes(className)) {
      return {
        known: false,
        kind: component,
        description: "",
        problem: `Unknown ${component} class '${className}'.`,
      };
    }
    if (component === "miner" && className === "no_miner") {
      return { known: false, kind: "miner", description: "", problem: "A space cannot target no_miner." };
    }
    const params = schema.componentParams[component][className];
    const accepted = params?.some((item) => item.name === parameter);
    return {
      known: true,
      kind: component,
      description: `Constructor argument of ${className}.`,
      problem: params && !accepted
        ? `${className} has no constructor argument '${parameter}'. It accepts: ${params
            .map((item) => item.name)
            .join(", ")}.`
        : undefined,
    };
  }

  if (key.startsWith("ssl_config.") || key.startsWith("ssl.")) {
    const tail = key.replace(/^ssl(_config)?\./, "");
    const field = schema.sslConfigFields.find((item) => item.name === tail);
    return {
      known: true,
      kind: "ssl",
      description: field
        ? `SemiSupervisedConfig.${tail} (default ${JSON.stringify(field.default)}).`
        : "Nested SSL method parameter; created if the base config does not define it.",
    };
  }

  const argument = schema.cliArguments.find((item) => item.dest === key);
  if (argument) {
    return {
      known: true,
      kind: "training",
      description: argument.help || `main.py --${key}`,
    };
  }

  return {
    known: false,
    kind: "unknown",
    description: "",
    problem: `'${key}' is not a main.py argument. HPO would reject it as an unknown training argument.`,
  };
}

function numeric(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/** Structural and range checks for one search dimension. */
export function lintSpace(key: string, spec: SpaceSpec, schema: StudioSchema): Issue[] {
  const issues: Issue[] = [];
  const info = describeSpaceKey(key, schema);
  if (info.problem) {
    issues.push({ level: info.known ? "warning" : "error", message: info.problem, field: key });
  }

  if (key === "loss" || key === "miner") {
    issues.push({
      level: "error",
      message: `'${key}' cannot be searched. Fix it with --${key}, or compare pairs with loss_miner_grid.`,
      field: key,
    });
  }
  if (schema.constants.hpoModeKeys.includes(key)) {
    issues.push({
      level: "error",
      message: `'${key}' cannot be searched; set it in the experiment config instead.`,
      field: key,
    });
  }
  if (schema.constants.comparisonForbiddenKeys.includes(key)) {
    issues.push({
      level: "warning",
      message: `'${key}' changes the data or the comparison itself; searching it makes trials incomparable.`,
      field: key,
    });
  }
  if (schema.constants.labeledBatchFractionAliases.includes(key)) {
    issues.push({
      level: "error",
      message: `Use the bare '${schema.constants.labeledBatchFractionKey}' key; the prefixed form is not an SSL config field.`,
      field: key,
    });
  }

  if (isCategoricalSpec(spec)) {
    const choices = specChoices(spec);
    if (choices.length === 0) {
      issues.push({ level: "error", message: "A categorical space needs at least one choice.", field: key });
    } else if (choices.length === 1) {
      issues.push({
        level: "info",
        message: "One choice pins this value instead of searching it.",
        field: key,
      });
    }
    if (new Set(choices.map((item) => JSON.stringify(item))).size !== choices.length) {
      issues.push({ level: "warning", message: "Duplicate choices are wasted trials.", field: key });
    }
    if (key === schema.constants.batchSamplerKey) {
      for (const choice of choices) {
        if (typeof choice !== "string" || !choice.includes(":")) {
          issues.push({
            level: "error",
            message: `'${String(choice)}' must be written as batch_size:sampler_m.`,
            field: key,
          });
          continue;
        }
        const [rawBatch, rawM] = choice.split(":", 2);
        const batchSize = Number(rawBatch);
        const samplerM = Number(rawM);
        if (!Number.isInteger(batchSize) || !Number.isInteger(samplerM) || batchSize <= 0 || samplerM <= 0) {
          issues.push({ level: "error", message: `'${choice}' needs two positive integers.`, field: key });
        } else if (batchSize % samplerM !== 0) {
          issues.push({
            level: "error",
            message: `'${choice}' is invalid: batch_size must be divisible by sampler_m.`,
            field: key,
          });
        }
      }
    }
    if (key === schema.constants.labeledBatchFractionKey) {
      for (const choice of choices) {
        const value = numeric(choice);
        if (value === null || value <= 0 || value > schema.constants.maxLabeledBatchFraction) {
          issues.push({
            level: "error",
            message: `Fractions must be in (0, ${schema.constants.maxLabeledBatchFraction}]; got ${String(choice)}.`,
            field: key,
          });
        }
      }
    }
    return issues;
  }

  const record = spec as Record<string, unknown>;
  const type = record.type;
  if (type !== "float" && type !== "int") {
    issues.push({ level: "error", message: `Unknown search space type ${JSON.stringify(type)}.`, field: key });
    return issues;
  }

  const low = numeric(record.low);
  const high = numeric(record.high);
  if (low === null || high === null) {
    issues.push({ level: "error", message: "A numeric space needs both low and high.", field: key });
    return issues;
  }
  if (low > high) {
    issues.push({ level: "error", message: `low (${low}) must not exceed high (${high}).`, field: key });
  }
  if (type === "int" && (!Number.isInteger(low) || !Number.isInteger(high))) {
    issues.push({ level: "error", message: "An int space needs integer low and high.", field: key });
  }
  if (record.log === true && low <= 0) {
    issues.push({ level: "error", message: "A log-scale space needs a strictly positive low.", field: key });
  }
  if (record.step != null) {
    const step = numeric(record.step);
    if (step === null || step <= 0) {
      issues.push({ level: "error", message: "step must be positive.", field: key });
    } else if (record.log === true) {
      issues.push({ level: "warning", message: "Optuna rejects step together with log=true.", field: key });
    }
  }
  if (low === high) {
    issues.push({ level: "info", message: "low equals high, so this dimension is fixed.", field: key });
  }
  if (record.log === true && low > 0 && high / low > 1e6) {
    issues.push({
      level: "warning",
      message: `This range spans ${Math.round(Math.log10(high / low))} decades; TPE needs many trials to cover it.`,
      field: key,
    });
  }
  if (record.log !== true && low > 0 && high / low > 1e3) {
    issues.push({
      level: "warning",
      message: "A range this wide is usually searched with log: true.",
      field: key,
    });
  }
  if (key === schema.constants.labeledBatchFractionKey) {
    if (low <= 0 || high > schema.constants.maxLabeledBatchFraction) {
      issues.push({
        level: "error",
        message: `Fractions must stay inside (0, ${schema.constants.maxLabeledBatchFraction}].`,
        field: key,
      });
    }
  }
  if (schema.constants.labeledBatchSizeKeys.includes(key)) {
    issues.push({
      level: "error",
      message: "The labeled batch size must be categorical so it can be paired with batch_sampler.",
      field: key,
    });
  }
  return issues;
}

/** Whole-config checks for an Optuna search configuration. */
export function lintHparamConfig(
  document: Record<string, unknown>,
  schema: StudioSchema,
): Issue[] {
  const issues: Issue[] = [];
  const spaces = (document.spaces ?? {}) as Record<string, SpaceSpec>;
  const keys = Object.keys(spaces);

  if (keys.length === 0 && !document.extends) {
    issues.push({ level: "error", message: "spaces must not be empty unless the config extends a parent." });
  }

  for (const [key, spec] of Object.entries(spaces)) {
    issues.push(...lintSpace(key, spec, schema));
  }

  const batchSampler = schema.constants.batchSamplerKey;
  if (keys.includes(batchSampler) && (keys.includes("batch_size") || keys.includes("sampler_m"))) {
    issues.push({
      level: "error",
      message: `'${batchSampler}' already sets batch_size and sampler_m; remove the separate keys.`,
      field: batchSampler,
    });
  }
  const labeledSizeKeys = keys.filter((key) => schema.constants.labeledBatchSizeKeys.includes(key));
  if (labeledSizeKeys.length > 1) {
    issues.push({
      level: "error",
      message: `${labeledSizeKeys.join(", ")} are aliases for the same setting; keep one.`,
    });
  }
  if (keys.includes(schema.constants.labeledBatchFractionKey) && labeledSizeKeys.length > 0) {
    issues.push({
      level: "error",
      message: "Search either the labeled batch fraction or its absolute size, not both.",
    });
  }

  const trials = document.n_trials;
  if (typeof trials === "number" && trials <= 0) {
    issues.push({ level: "error", message: "n_trials must be positive.", field: "n_trials" });
  }
  const startup = document.tpe_startup_trials;
  if (typeof startup === "number") {
    if (document.sampler !== undefined && document.sampler !== "tpe") {
      issues.push({
        level: "error",
        message: "tpe_startup_trials only applies when the sampler is 'tpe'.",
        field: "tpe_startup_trials",
      });
    }
    if (typeof trials === "number" && startup >= trials) {
      issues.push({
        level: "warning",
        message: `${startup} startup trials out of ${trials} means the search never leaves random sampling.`,
        field: "tpe_startup_trials",
      });
    }
  }
  if (typeof document.n_jobs === "number" && document.n_jobs !== 1) {
    issues.push({
      level: "warning",
      message: "launch_runs.py refuses n_jobs other than 1 unless allow_parallel_trials is set in the manifest.",
      field: "n_jobs",
    });
  }
  if (typeof trials === "number" && keys.length > 0 && trials < keys.length * 8) {
    issues.push({
      level: "info",
      message: `${trials} trials for ${keys.length} dimensions is a thin budget; roughly 10-20 trials per dimension is typical.`,
      field: "n_trials",
    });
  }
  if (document.study_name === undefined || document.study_name === null) {
    issues.push({
      level: "info",
      message: "No study_name here, so it comes from the parent config. Two datasets sharing one name pool their dashboards.",
      field: "study_name",
    });
  }
  return issues;
}

const SSL_METHOD_HINT_WORDS = ["iscen", "lrml", "seraph", "slade", "hoffer", "ismlp", "stml"];

/** Whole-config checks for an experiment (dataset) configuration. */
export function lintExperimentConfig(
  document: Record<string, unknown>,
  schema: StudioSchema,
  knownPaths: Set<string>,
): Issue[] {
  const issues: Issue[] = [];
  const actions = new Map(schema.cliArguments.map((item) => [item.dest, item]));

  for (const key of Object.keys(document)) {
    if (!actions.has(key)) {
      issues.push({
        level: "error",
        message: `'${key}' is not a main.py argument name, so loading this config fails.`,
        field: key,
      });
    }
  }

  const mode = document.mode;
  const sslConfig = document.ssl_config;
  if (mode === "ssl" && !sslConfig) {
    issues.push({
      level: "error",
      message: "mode is 'ssl' but no ssl_config is set, so there is no method to run.",
      field: "ssl_config",
    });
  }
  if (mode !== "ssl" && sslConfig) {
    issues.push({
      level: "warning",
      message: `ssl_config is set but mode is ${JSON.stringify(mode ?? "unset")}; the SSL method will not run.`,
      field: "mode",
    });
  }

  for (const field of ["ssl_config", "hparam_config"] as const) {
    const value = document[field];
    if (typeof value === "string" && value && !knownPaths.has(value)) {
      issues.push({
        level: "error",
        message: `${field} points at ${value}, which does not exist.`,
        field,
      });
    }
  }

  const saveDir = document.save_dir;
  if (typeof saveDir !== "string" || !saveDir) {
    issues.push({
      level: "warning",
      message: "No save_dir; runs land in the default output directory and can overwrite each other.",
      field: "save_dir",
    });
  } else if (typeof sslConfig === "string" && sslConfig) {
    const configured = sslConfig.split("/").pop()?.replace(/\.json$/, "") ?? "";
    const lowered = saveDir.toLowerCase();
    const mismatched = SSL_METHOD_HINT_WORDS.filter(
      (word) => lowered.includes(word) && !configured.toLowerCase().includes(word),
    );
    if (mismatched.length > 0 && !mismatched.some((word) => configured.toLowerCase().includes(word))) {
      issues.push({
        level: "warning",
        message: `save_dir mentions '${mismatched[0]}' but ssl_config is '${configured}'. Results would be written under another method's name.`,
        field: "save_dir",
      });
    }
  }

  const grid = document.loss_miner_grid;
  if (Array.isArray(grid)) {
    for (const entry of grid) {
      if (typeof entry !== "string" || !entry.includes(":")) {
        issues.push({
          level: "error",
          message: `loss_miner_grid entry ${JSON.stringify(entry)} must be written as 'Loss:Miner'.`,
          field: "loss_miner_grid",
        });
        continue;
      }
      const [loss, miner] = entry.split(":", 2);
      if (!schema.constants.losses.includes(loss)) {
        issues.push({ level: "error", message: `Unknown loss '${loss}'.`, field: "loss_miner_grid" });
      }
      if (!schema.constants.miners.includes(miner)) {
        issues.push({ level: "error", message: `Unknown miner '${miner}'.`, field: "loss_miner_grid" });
      }
      if (schema.constants.classificationLosses.includes(loss) && miner !== "no_miner") {
        issues.push({
          level: "warning",
          message: `${loss} is a classification loss, so the miner '${miner}' is ignored.`,
          field: "loss_miner_grid",
        });
      }
    }
  }

  const budgets = document.label_budget_grid;
  if (Array.isArray(budgets)) {
    for (const budget of budgets) {
      if (typeof budget !== "number" || budget <= 0 || budget > 1) {
        issues.push({
          level: "warning",
          message: `label_budget_grid entry ${String(budget)} is outside (0, 1]; it is a fraction of classes.`,
          field: "label_budget_grid",
        });
      }
    }
  }

  const modes = document.ssl_label_sampling_modes;
  const kShots = document.k_shot_grid;
  if (Array.isArray(kShots) && kShots.length > 0) {
    const usesKShot = Array.isArray(modes) && modes.includes("class_subset_k_shot");
    if (!usesKShot) {
      issues.push({
        level: "warning",
        message: "k_shot_grid only varies under the 'class_subset_k_shot' sampling mode; every other mode ignores it.",
        field: "k_shot_grid",
      });
    }
  }
  if (Array.isArray(modes)) {
    for (const item of modes) {
      if (typeof item !== "string" || !schema.constants.labelSamplingModes.includes(item)) {
        issues.push({
          level: "error",
          message: `Unknown label sampling mode ${JSON.stringify(item)}.`,
          field: "ssl_label_sampling_modes",
        });
      }
    }
  }

  const epochs = document.epochs;
  const patience = document.patience;
  if (typeof epochs === "number" && typeof patience === "number" && patience >= epochs) {
    issues.push({
      level: "info",
      message: `patience (${patience}) is not below epochs (${epochs}), so early stopping can never fire.`,
      field: "patience",
    });
  }
  if (document.cv_k === 1) {
    issues.push({ level: "info", message: "cv_k = 1 disables cross-validation.", field: "cv_k" });
  }

  return issues;
}

interface ManifestRun {
  name?: unknown;
  experiment_config?: unknown;
  hparam_config?: unknown;
  study_dir?: unknown;
  study_replay?: unknown;
  save_dir?: unknown;
  args?: unknown;
  enabled?: unknown;
  gpu_memory_mib?: unknown;
  slots?: unknown;
}

/** Whole-config checks for a scheduler run manifest. */
export function lintManifest(
  document: Record<string, unknown>,
  schema: StudioSchema,
  knownPaths: Set<string>,
): Issue[] {
  const issues: Issue[] = [];
  const runs = Array.isArray(document.runs) ? (document.runs as ManifestRun[]) : [];
  const namePattern = new RegExp(schema.manifest.runNamePattern);

  if (runs.length === 0) {
    issues.push({ level: "error", message: "A manifest needs at least one run." });
  }

  const enabledNames = runs
    .filter((run) => run.enabled !== false)
    .map((run) => (typeof run.name === "string" ? run.name : ""));
  const duplicates = enabledNames.filter((name, index) => name && enabledNames.indexOf(name) !== index);
  for (const name of new Set(duplicates)) {
    issues.push({ level: "error", message: `Two enabled runs are both named '${name}'.`, field: "runs" });
  }
  if (enabledNames.length === 0) {
    issues.push({ level: "error", message: "Every run is disabled, so the launcher has nothing to do." });
  }

  runs.forEach((run, index) => {
    const label = typeof run.name === "string" && run.name ? run.name : `runs[${index}]`;
    if (typeof run.name !== "string" || !run.name) {
      issues.push({ level: "error", message: `runs[${index}] needs a name.`, field: label });
    } else if (!namePattern.test(run.name)) {
      issues.push({
        level: "error",
        message: `Run name '${run.name}' must match ${schema.manifest.runNamePattern}. It becomes a directory name.`,
        field: label,
      });
    }

    if (run.study_dir && run.hparam_config) {
      issues.push({
        level: "error",
        message: `${label} sets both study_dir and hparam_config. A replay reuses the saved study config; remove hparam_config.`,
        field: label,
      });
    }
    if (!run.study_dir && run.study_replay && Object.keys(run.study_replay as object).length > 0) {
      issues.push({
        level: "error",
        message: `${label} has study_replay settings but no study_dir to replay.`,
        field: label,
      });
    }
    if (!run.study_dir && !run.experiment_config && !run.hparam_config) {
      issues.push({
        level: "warning",
        message: `${label} names no experiment_config, hparam_config, or study_dir, so it runs main.py defaults.`,
        field: label,
      });
    }

    for (const field of ["experiment_config", "hparam_config"] as const) {
      const value = run[field];
      if (typeof value === "string" && value && !knownPaths.has(value)) {
        issues.push({
          level: "error",
          message: `${label}: ${field} points at ${value}, which does not exist.`,
          field: label,
        });
      }
    }

    const args = Array.isArray(run.args) ? run.args : [];
    for (const token of args) {
      if (typeof token !== "string") {
        issues.push({ level: "error", message: `${label}: every arg must be a string.`, field: label });
        continue;
      }
      const option = token.split("=", 1)[0];
      if (schema.manifest.schedulerOwnedOptions.includes(option)) {
        issues.push({
          level: "error",
          message: `${label}: the scheduler owns ${option}; set it in the manifest fields instead.`,
          field: label,
        });
      }
      if (run.study_dir && schema.manifest.studyReplayOwnedOptions.includes(option)) {
        issues.push({
          level: "error",
          message: `${label}: ${option} belongs in study_replay for a study-directory replay.`,
          field: label,
        });
      }
    }
    if (args.length % 2 === 1 && args.length > 0) {
      issues.push({
        level: "info",
        message: `${label}: an odd number of arg tokens is fine for flags, but check that each option got its value.`,
        field: label,
      });
    }

    if (!run.save_dir && !document.save_dir_root) {
      issues.push({
        level: "warning",
        message: `${label} has no save_dir and the manifest has no save_dir_root, so the run keeps whatever save_dir its experiment config sets.`,
        field: label,
      });
    }
  });

  const device = document.device ?? "cuda";
  if (device === "cuda" && document.gpus === undefined) {
    issues.push({
      level: "info",
      message: "No gpus listed, so the scheduler uses every GPU it discovers.",
      field: "gpus",
    });
  }
  const perGpu = document.max_concurrent_per_gpu;
  const memory = document.default_gpu_memory_mib;
  if (typeof perGpu === "number" && typeof memory === "number" && perGpu * memory > 24_000) {
    issues.push({
      level: "warning",
      message: `${perGpu} concurrent runs x ${memory} MiB reserves ${(perGpu * memory) / 1024} GiB per GPU before the reserve headroom.`,
      field: "max_concurrent_per_gpu",
    });
  }
  const enabledCount = enabledNames.length;
  if (typeof document.max_total_runs === "number" && document.max_total_runs < enabledCount) {
    issues.push({
      level: "info",
      message: `${enabledCount} runs are enabled but at most ${document.max_total_runs} run at a time; the rest queue.`,
      field: "max_total_runs",
    });
  }
  if (document.allow_parallel_trials === true) {
    issues.push({
      level: "warning",
      message: "allow_parallel_trials lets a child run several Optuna trials at once. Only do this deliberately.",
      field: "allow_parallel_trials",
    });
  }
  for (const key of Object.keys(document)) {
    if (!schema.manifest.topLevelKeys.includes(key)) {
      issues.push({ level: "error", message: `'${key}' is not a run-manifest key.`, field: key });
    }
  }
  return issues;
}

export function countIssues(issues: Issue[]): Record<"error" | "warning" | "info", number> {
  return issues.reduce(
    (totals, issue) => ({ ...totals, [issue.level]: totals[issue.level] + 1 }),
    { error: 0, warning: 0, info: 0 },
  );
}
