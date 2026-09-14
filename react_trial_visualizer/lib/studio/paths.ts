import fs from "node:fs";
import path from "node:path";

/** The metric-learning checkout that owns every config this studio edits. */
export const REPO_ROOT = path.resolve(process.cwd(), "..");

/** The one subtree that may be listed, read, or written. */
export const CONFIG_ROOT = "configs";

export const BACKUP_ROOT = "configs/.studio-backups";
export const STUDIO_LOG_ROOT = "logs/studio";

/** What a config document is, decided by its contents rather than its folder. */
export type ConfigKind = "manifest" | "hparam" | "ssl-method" | "experiment" | "unknown";

export class PathError extends Error {}

/** Resolve a repo-relative path, refusing anything outside `configs/`. */
export function resolveConfigPath(relativePath: string): string {
  if (typeof relativePath !== "string" || relativePath.length === 0) {
    throw new PathError("A repo-relative path is required.");
  }
  if (relativePath.includes("\0")) throw new PathError("Invalid path.");

  const normalized = path.posix.normalize(relativePath.replace(/\\/g, "/"));
  if (normalized.startsWith("..") || path.isAbsolute(normalized)) {
    throw new PathError(`Path escapes the repository: ${relativePath}`);
  }
  if (normalized !== CONFIG_ROOT && !normalized.startsWith(`${CONFIG_ROOT}/`)) {
    throw new PathError(`Only files under ${CONFIG_ROOT}/ can be edited: ${relativePath}`);
  }
  if (normalized.split("/").some((segment) => segment.startsWith("."))) {
    throw new PathError(`Hidden paths are off limits: ${relativePath}`);
  }
  if (!normalized.endsWith(".json")) {
    throw new PathError(`Only .json config files can be edited: ${relativePath}`);
  }
  return path.join(REPO_ROOT, normalized);
}

export function toRepoRelative(absolutePath: string): string {
  return path.relative(REPO_ROOT, absolutePath).split(path.sep).join("/");
}

/** Copy the current file aside before overwriting it, keeping the folder shape. */
export function backupExistingFile(relativePath: string): string | null {
  const absolutePath = path.join(REPO_ROOT, relativePath);
  if (!fs.existsSync(absolutePath)) return null;

  const stamp = new Date().toISOString().replace(/[:.]/g, "-");
  const backupPath = path.join(REPO_ROOT, BACKUP_ROOT, `${relativePath}.${stamp}.json`);
  fs.mkdirSync(path.dirname(backupPath), { recursive: true });
  fs.copyFileSync(absolutePath, backupPath);
  return toRepoRelative(backupPath);
}

const EXPERIMENT_MARKERS = [
  "dataset",
  "save_dir",
  "mode",
  "loss_miner_grid",
  "epochs",
  "cv_k",
  "backbone_tuning",
  "ssl_label_sampling_modes",
];

const HPARAM_MARKERS = ["spaces", "per_loss", "extends", "n_trials", "sampler", "tpe_startup_trials"];

/**
 * Classify a config document.
 *
 * Folders here are not a reliable signal — `configs/runs/experiments` holds
 * experiment configs, and search configs live under both `configs/runs` and
 * `configs/hpo` — so each editor filters on this instead.
 */
export function detectConfigKind(document: unknown): ConfigKind {
  if (!document || typeof document !== "object" || Array.isArray(document)) return "unknown";
  const record = document as Record<string, unknown>;

  if (Array.isArray(record.runs)) return "manifest";
  if (HPARAM_MARKERS.some((marker) => marker in record)) return "hparam";
  if ("method" in record && "method_params" in record) return "ssl-method";
  if (EXPERIMENT_MARKERS.some((marker) => marker in record)) return "experiment";
  return "unknown";
}

export interface ConfigFileEntry {
  path: string;
  name: string;
  directory: string;
  size: number;
  modified: string;
  kind: ConfigKind;
}

/** List every JSON config under `configs/`, tagged with what it is. */
export function listConfigFiles(): ConfigFileEntry[] {
  const entries: ConfigFileEntry[] = [];

  const walk = (directory: string) => {
    let children: fs.Dirent[];
    try {
      children = fs.readdirSync(directory, { withFileTypes: true });
    } catch {
      return;
    }
    for (const child of children) {
      if (child.name.startsWith(".")) continue;
      const childPath = path.join(directory, child.name);
      if (child.isDirectory()) {
        walk(childPath);
        continue;
      }
      if (!child.isFile() || !child.name.endsWith(".json")) continue;

      const stats = fs.statSync(childPath);
      let kind: ConfigKind = "unknown";
      try {
        kind = detectConfigKind(JSON.parse(fs.readFileSync(childPath, "utf8")));
      } catch {
        // A file that does not parse still belongs in the list, flagged unknown.
      }
      const relativePath = toRepoRelative(childPath);
      entries.push({
        path: relativePath,
        name: child.name,
        directory: path.posix.dirname(relativePath),
        size: stats.size,
        modified: stats.mtime.toISOString(),
        kind,
      });
    }
  };

  walk(path.join(REPO_ROOT, CONFIG_ROOT));
  return entries.sort((left, right) => left.path.localeCompare(right.path));
}
