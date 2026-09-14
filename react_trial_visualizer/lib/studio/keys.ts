import type { StudioSchema } from "./schema";

export interface KeyOption {
  value: string;
  label: string;
  group: string;
}

/** Every search-space key the project can actually apply, ordered by usefulness. */
export function buildSpaceKeyOptions(schema: StudioSchema): KeyOption[] {
  const options: KeyOption[] = [];
  const seen = new Set<string>();

  const push = (value: string, label: string, group: string) => {
    if (seen.has(value)) return;
    seen.add(value);
    options.push({ value, label, group });
  };

  for (const entry of schema.usedSpaceKeys) {
    push(entry.key, `used in ${entry.count} config${entry.count === 1 ? "" : "s"}`, "Already searched here");
  }

  push(schema.constants.batchSamplerKey, "joint batch_size:sampler_m", "Training");
  push(schema.constants.labeledBatchFractionKey, "labeled share of each batch", "Training");

  for (const argument of schema.cliArguments) {
    if (argument.isFlag) continue;
    push(argument.dest, argument.help.slice(0, 90), "Training");
  }

  for (const [component, classes] of Object.entries(schema.componentParams)) {
    for (const [className, params] of Object.entries(classes)) {
      for (const param of params) {
        // Plumbing arguments never belong in a search space.
        if (["collect_stats", "distance", "reducer", "embedding_regularizer", "weight_regularizer", "weight_init_func", "num_classes", "embedding_size"].includes(param.name)) {
          continue;
        }
        push(
          `${component}.${className}.${param.name}`,
          param.hasDefault ? `default ${JSON.stringify(param.default)}` : "required",
          component === "loss" ? "Loss parameters" : "Miner parameters",
        );
      }
    }
  }

  for (const method of schema.sslMethods) {
    for (const entry of method.keys) {
      push(entry.key, `${method.name}: ${JSON.stringify(entry.value)}`, "SSL config");
    }
  }
  for (const field of schema.sslConfigFields) {
    push(field.key, `default ${JSON.stringify(field.default)}`, "SSL config");
  }

  return options;
}

/** Options for an experiment config's `hparam_config`/`ssl_config` pickers. */
export function pathOptions(paths: string[]): { value: string; label?: string }[] {
  return paths.map((path) => ({ value: path, label: path }));
}

/** Relative path from one config file to another, for an `extends` reference. */
export function relativeConfigPath(fromFile: string, toFile: string): string {
  const from = fromFile.split("/").slice(0, -1);
  const to = toFile.split("/");
  const target = to.pop() as string;

  let shared = 0;
  while (shared < from.length && shared < to.length && from[shared] === to[shared]) shared += 1;

  const upwards = new Array(from.length - shared).fill("..");
  const downwards = to.slice(shared);
  const segments = [...upwards, ...downwards, target];
  return segments.length === 1 ? segments[0] : segments.join("/");
}
