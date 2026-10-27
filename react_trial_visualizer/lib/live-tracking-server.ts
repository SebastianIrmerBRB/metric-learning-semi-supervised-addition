import { spawn } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, type Dirent } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";

import { isAllowedResolvedPath } from "@/lib/data-roots";

import type {
  LiveBase,
  LiveEpochEntry,
  LiveFrame,
  LiveFramesResponse,
  LiveManifest,
  LiveRunEntry,
  LiveStatus,
} from "@/lib/live-tracking";

/**
 * Epoch tracking read straight from the run directories, for `next dev`.
 *
 * The static site gets epoch-tracking views from evaluate/export_embedding_views.py
 * once a fold has finished. The dev server does not wait: it finds every
 * `epoch_tracking/` below the log roots, describes each one with a cheap
 * manifest built from file stats alone, and hands the page one epoch at a time,
 * built by evaluate/live_tracking.py only when that epoch's files changed.
 */

export const REPOSITORY = path.resolve(process.cwd(), "..");
const TRACKING_DIR_NAME = "epoch_tracking";
// Directories a search never needs to enter: run artifacts that cannot hold
// another run, and trees that are large for no benefit.
const SKIPPED_DIRS = new Set([
  TRACKING_DIR_NAME,
  "node_modules",
  ".git",
  "__pycache__",
  "tensorboard",
  "checkpoints",
  "pseudo_labels",
  "slade_clusters",
  "slade_pairs",
  "split",
  "graph_diagnostics",
]);
const MAX_DEPTH = 14;
const RUN_SCAN_TTL_MS = 8_000;
// A run that wrote nothing for this long is not shown as running, whatever its
// status.json says: the process was most likely killed.
const RUNNING_GRACE_MS = 45 * 60_000;
const RECENT_MS = 10 * 60_000;
const PYTHON_CONCURRENCY = 2;
const PYTHON_TIMEOUT_MS = 180_000;
const FRAME_CACHE_LIMIT = 600;
// Files of one epoch directory that shape its frame.
const FRAME_FILES = /^(embeddings_[a-z]+(?:_3d)?\.csv|graph_nodes\.csv|graph_edges\.npz|class_evidence\.npz|label_propagation\.npz)$/;

interface LiveCache {
  runs: { at: number; pending: Promise<LiveRunEntry[]> } | null;
  configs: Map<string, { mtimeMs: number; meta: RunMeta }>;
  bases: Map<string, LiveBase>;
  frames: Map<string, { frame: LiveFrame; extras: number[]; baseKey: string }>;
  running: number;
  queue: (() => void)[];
}

// Next's dev server re-evaluates route modules on every edit; the caches outlive that.
const globalCache = globalThis as typeof globalThis & { __liveTracking?: LiveCache };
const cache: LiveCache = (globalCache.__liveTracking ??= {
  runs: null,
  configs: new Map(),
  bases: new Map(),
  frames: new Map(),
  running: 0,
  queue: [],
});

export class LiveTrackingError extends Error {
  constructor(
    message: string,
    readonly status = 400,
  ) {
    super(message);
  }
}

function roots(): string[] {
  const configured = process.env.LIVE_TRACKING_ROOTS?.trim();
  const list = configured ? configured.split(/[:;,]/).filter(Boolean) : ["logs"];
  return list.map((root) => path.resolve(REPOSITORY, root));
}

function isInsideRepository(candidate: string): boolean {
  const relative = path.relative(REPOSITORY, candidate);
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

export function relativeToRepository(candidate: string): string {
  return path.relative(REPOSITORY, candidate).split(path.sep).join("/");
}

/** Resolve a requested tracking directory, refusing anything outside the repository. */
export async function resolveTrackingDir(requested: string | null): Promise<string> {
  const cleaned = (requested ?? "").trim().replace(/\\/g, "/").replace(/\/+$/, "");
  if (!cleaned) throw new LiveTrackingError("Pass ?dir=<run>/epoch_tracking.");
  const candidate = path.resolve(REPOSITORY, cleaned.endsWith(TRACKING_DIR_NAME) ? cleaned : `${cleaned}/${TRACKING_DIR_NAME}`);
  if (!isInsideRepository(candidate)) throw new LiveTrackingError("Tracking directories must be inside the repository.");
  let real: string;
  try {
    real = await fs.realpath(candidate);
  } catch {
    throw new LiveTrackingError(`No epoch_tracking directory at ${cleaned}.`, 404);
  }
  if (!isAllowedResolvedPath(REPOSITORY, real)) throw new LiveTrackingError("That tracking directory resolves outside the repository and the data roots.");
  return real;
}

// ---------------------------------------------------------------------------
// Finding runs
// ---------------------------------------------------------------------------

interface RunMeta {
  label: string;
  runLabel: string;
  dataset: string | null;
  method: string | null;
  loss: string | null;
  seed: number | null;
  fold: number | null;
  labelBudget: string | null;
}

/** evaluate/export_embedding_views.campaign_name, so live and exported runs read alike. */
function campaignName(saveDir: unknown): [string | null, string | null] {
  if (typeof saveDir !== "string" || !saveDir) return [null, null];
  let parts = saveDir.split(/[\\/]+/).filter(Boolean);
  if (saveDir.startsWith("/")) {
    const anchors = parts.map((part, index) => (part === "logs" || part === "runs" ? index : -1)).filter((index) => index >= 0);
    parts = anchors.length ? parts.slice(anchors[anchors.length - 1] + 1) : parts.slice(-2);
  }
  let stamp = parts.find((part) => part.startsWith("cv_20"))?.slice(3) ?? null;
  const kept = parts.filter(
    (part) => !part.startsWith("fold_") && !part.startsWith("cv_20") && !part.includes("_seed_") && !part.startsWith("20"),
  );
  if (stamp) {
    const [date, time = ""] = stamp.split("_");
    stamp = `${date.slice(5)} ${time.slice(0, 5).replace("-", ":")}`;
  }
  return [kept.slice(0, 2).join("/") || null, stamp];
}

function describeRun(config: { args?: Record<string, unknown>; ssl_config?: Record<string, unknown> }, fallback: string): RunMeta {
  const args = config.args ?? {};
  const ssl = config.ssl_config ?? {};
  const methodParams = (ssl.method_params as Record<string, unknown> | undefined) ?? {};
  const method = (args.mode === "ssl" ? methodParams.regularizer ?? ssl.method : args.mode) as string | null | undefined;
  let budget: string | null = null;
  if (ssl.labeled_per_class != null) budget = `${ssl.labeled_per_class}/class`;
  else if (ssl.labeled_fraction != null && ssl.labeled_fraction !== 1) budget = `${Math.round(Number(ssl.labeled_fraction) * 100)}% labeled`;
  const [campaign, stamp] = campaignName(args.save_dir);
  const fold = typeof args.cv_fold === "number" ? args.cv_fold : null;
  const parts = [
    campaign,
    args.dataset,
    method,
    args.loss,
    budget,
    args.seed == null ? null : `seed ${args.seed}`,
    stamp == null ? null : `cv ${stamp}`,
  ].filter((part) => part != null && part !== "");
  const runLabel = parts.length ? parts.join(" · ") : fallback;
  return {
    label: fold == null ? runLabel : `${runLabel} · fold ${fold}`,
    runLabel,
    dataset: (args.dataset as string) ?? null,
    method: method ?? null,
    loss: (args.loss as string) ?? null,
    seed: typeof args.seed === "number" ? args.seed : null,
    fold,
    labelBudget: budget,
  };
}

async function runMeta(runDir: string): Promise<RunMeta> {
  const fallback = relativeToRepository(runDir).replace(/^logs\//, "");
  const configPath = path.join(runDir, "run_config.json");
  try {
    const stat = await fs.stat(configPath);
    const cached = cache.configs.get(configPath);
    if (cached && cached.mtimeMs === stat.mtimeMs) return cached.meta;
    const meta = describeRun(JSON.parse(await fs.readFile(configPath, "utf8")), fallback);
    cache.configs.set(configPath, { mtimeMs: stat.mtimeMs, meta });
    return meta;
  } catch {
    // Written at the start of training; until then the path is the best name there is.
    return { label: fallback, runLabel: fallback, dataset: null, method: null, loss: null, seed: null, fold: null, labelBudget: null };
  }
}

async function readStatus(trackingDir: string): Promise<LiveStatus | null> {
  try {
    return JSON.parse(await fs.readFile(path.join(trackingDir, "status.json"), "utf8")) as LiveStatus;
  } catch {
    return null;
  }
}

async function findTrackingDirs(): Promise<string[]> {
  const found: string[] = [];
  async function walk(directory: string, depth: number) {
    let entries: Dirent[];
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch {
      return;
    }
    const children: Promise<void>[] = [];
    for (const entry of entries) {
      const child = path.join(directory, entry.name);
      // Follow directory symlinks too: logs/final/phase_2 points to /mnt/f/Studien/phase_2.
      const isDirectory =
        entry.isDirectory() ||
        (entry.isSymbolicLink() && (await fs.stat(child).catch(() => null))?.isDirectory() === true);
      if (!isDirectory) continue;
      if (entry.name === TRACKING_DIR_NAME) found.push(child);
      else if (depth < MAX_DEPTH && !SKIPPED_DIRS.has(entry.name) && !entry.name.startsWith("epoch_")) {
        children.push(walk(child, depth + 1));
      }
    }
    await Promise.all(children);
  }
  await Promise.all(roots().map((root) => walk(root, 0)));
  return found;
}

async function describeTrackingDir(trackingDir: string): Promise<LiveRunEntry | null> {
  let names: string[];
  try {
    names = await fs.readdir(trackingDir);
  } catch {
    return null;
  }
  const epochNames = names
    .filter((name) => epochFromSlug(name) != null)
    .sort((left, right) => epochFromSlug(left)! - epochFromSlug(right)!);
  const stats = await Promise.all(
    [".", "status.json", "epoch_metrics.csv", ...epochNames.slice(-2)].map((name) =>
      fs.stat(path.join(trackingDir, name)).catch(() => null),
    ),
  );
  const modifiedAt = Math.max(0, ...stats.map((stat) => stat?.mtimeMs ?? 0));
  const status = await readStatus(trackingDir);
  const runDir = path.dirname(trackingDir);
  const meta = await runMeta(runDir);
  const now = Date.now();
  const running = status ? status.state === "running" && now - modifiedAt < RUNNING_GRACE_MS : now - modifiedAt < RECENT_MS;
  return {
    id: `live:${relativeToRepository(trackingDir)}`,
    dir: relativeToRepository(trackingDir),
    runDir: relativeToRepository(runDir),
    ...meta,
    epochCount: epochNames.length,
    modifiedAt,
    status,
    running,
  };
}

/** Every tracking directory below the log roots, most recently written first. */
export async function listLiveRuns(force = false): Promise<LiveRunEntry[]> {
  const now = Date.now();
  if (!force && cache.runs && now - cache.runs.at < RUN_SCAN_TTL_MS) return cache.runs.pending;
  const pending = (async () => {
    const directories = await findTrackingDirs();
    const entries = await Promise.all(directories.map(describeTrackingDir));
    return entries
      .filter((entry): entry is LiveRunEntry => entry != null && entry.epochCount > 0)
      .sort((left, right) => right.modifiedAt - left.modifiedAt);
  })();
  cache.runs = { at: now, pending };
  pending.catch(() => {
    if (cache.runs?.pending === pending) cache.runs = null;
  });
  return pending;
}

// ---------------------------------------------------------------------------
// One run's manifest: file stats only
// ---------------------------------------------------------------------------

function epochFromSlug(name: string): number | null {
  const match = /^epoch_(m?)(\d+)$/.exec(name);
  if (!match) return null;
  const value = Number(match[2]);
  return match[1] ? -value : value;
}

async function statLine(file: string): Promise<string | null> {
  try {
    const stat = await fs.stat(file);
    return `${path.basename(file)}:${stat.size}:${Math.round(stat.mtimeMs)}`;
  } catch {
    return null;
  }
}

async function directoryStatLines(directory: string, pattern: RegExp): Promise<string[]> {
  let names: string[];
  try {
    names = await fs.readdir(directory);
  } catch {
    return [];
  }
  const lines = await Promise.all(names.filter((name) => pattern.test(name)).sort().map((name) => statLine(path.join(directory, name))));
  return lines.filter((line): line is string => line != null);
}

function digest(parts: string[]): string {
  return createHash("sha1").update(parts.join("|")).digest("hex").slice(0, 16);
}

function parseMetricsCsv(text: string): Record<string, number | null>[] {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) return [];
  const header = lines[0].split(",");
  return lines.slice(1).map((line) => {
    const cells = line.split(",");
    const row: Record<string, number | null> = {};
    header.forEach((name, index) => {
      const raw = cells[index];
      const value = raw == null || raw === "" ? NaN : raw === "True" ? 1 : raw === "False" ? 0 : Number(raw);
      row[name] = Number.isFinite(value) ? value : null;
    });
    return row;
  });
}

export async function liveManifest(trackingDir: string): Promise<LiveManifest> {
  const runDir = path.dirname(trackingDir);
  const names = (await fs.readdir(trackingDir)).filter((name) => epochFromSlug(name) != null);
  const epochs = names
    .map((name) => ({ name, epoch: epochFromSlug(name)! }))
    .sort((left, right) => left.epoch - right.epoch);

  const [runLevel, statusText, metricsText, configLine] = await Promise.all([
    Promise.all([
      directoryStatLines(path.join(runDir, "pseudo_labels"), /\.npz$/),
      directoryStatLines(path.join(runDir, "slade_clusters"), /\.npz$/),
    ]).then((groups) => groups.flat()),
    fs.readFile(path.join(trackingDir, "status.json"), "utf8").catch(() => null),
    fs.readFile(path.join(trackingDir, "epoch_metrics.csv"), "utf8").catch(() => ""),
    statLine(path.join(runDir, "run_config.json")),
  ]);

  const entries: LiveEpochEntry[] = await Promise.all(
    epochs.map(async ({ name, epoch }) => {
      const directory = path.join(trackingDir, name);
      const lines = await directoryStatLines(directory, FRAME_FILES);
      const pairs = epoch >= 0 ? await statLine(path.join(runDir, "slade_pairs", `epoch_${String(epoch).padStart(4, "0")}.npz`)) : null;
      const has = (pattern: RegExp) => lines.some((line) => pattern.test(line.split(":")[0]));
      return {
        epoch,
        slug: name,
        files: lines,
        pairs,
        has2d: has(/^embeddings_[a-z]+\.csv$/),
        has3d: has(/^embeddings_[a-z]+_3d\.csv$/),
        hasGraph: has(/^graph_(nodes\.csv|edges\.npz)$/),
        hasEvidence: has(/^class_evidence\.npz$/),
        hasPropagation: has(/^label_propagation\.npz$/),
        fingerprint: "",
      } as LiveEpochEntry & { files: string[]; pairs: string | null };
    }),
  );

  // The base comes from the earliest complete layout and the earliest graph_nodes.csv.
  const first2d = entries.find((entry) => entry.has2d) as (LiveEpochEntry & { files: string[] }) | undefined;
  const firstNodes = entries.find((entry) => (entry as LiveEpochEntry & { files: string[] }).files.some((line) => line.startsWith("graph_nodes.csv:"))) as
    | (LiveEpochEntry & { files: string[] })
    | undefined;
  const baseFingerprint = first2d
    ? digest([
        String(first2d.epoch),
        ...first2d.files.filter((line) => /^embeddings_[a-z]+\.csv:/.test(line)),
        firstNodes ? `${firstNodes.epoch}/${firstNodes.files.find((line) => line.startsWith("graph_nodes.csv:"))}` : "no-nodes",
        configLine ?? "no-config",
      ])
    : null;

  // Pseudo-label generations and SLADE refreshes are read as "the latest one at
  // or before this epoch", so only the newest epoch can still gain one; older
  // frames keep their fingerprint once the next epoch has a layout.
  const newest2d = [...entries].reverse().find((entry) => entry.has2d)?.epoch ?? null;
  const cleaned = entries.map((raw) => {
    const entry = raw as LiveEpochEntry & { files: string[]; pairs: string | null };
    const parts = [String(entry.epoch), baseFingerprint ?? "no-base", ...entry.files, entry.pairs ?? "no-pairs"];
    if (newest2d == null || entry.epoch >= newest2d) parts.push(...runLevel);
    const { files: _files, pairs: _pairs, ...rest } = entry;
    return { ...rest, fingerprint: digest(parts) };
  });

  let status: LiveStatus | null = null;
  try {
    status = statusText ? (JSON.parse(statusText) as LiveStatus) : null;
  } catch {
    status = null;
  }
  return {
    dir: relativeToRepository(trackingDir),
    runDir: relativeToRepository(runDir),
    baseFingerprint,
    epochs: cleaned,
    epochMetrics: parseMetricsCsv(metricsText),
    status,
    checkedAt: new Date().toISOString(),
  };
}

// ---------------------------------------------------------------------------
// Base and frames: evaluate/live_tracking.py, a few at a time
// ---------------------------------------------------------------------------

function python(): string {
  const override = process.env.LIVE_TRACKING_PYTHON;
  if (override) return override;
  for (const candidate of [path.join(REPOSITORY, "venv", "bin", "python"), path.join(REPOSITORY, ".venv", "bin", "python")]) {
    if (existsSync(candidate)) return candidate;
  }
  return "python3";
}

async function withSlot<T>(task: () => Promise<T>): Promise<T> {
  if (cache.running >= PYTHON_CONCURRENCY) await new Promise<void>((resolve) => cache.queue.push(resolve));
  cache.running += 1;
  try {
    return await task();
  } finally {
    cache.running -= 1;
    cache.queue.shift()?.();
  }
}

function runLiveTracking(args: string[]): Promise<unknown> {
  return withSlot(
    () =>
      new Promise((resolve, reject) => {
        const child = spawn(python(), ["-m", "evaluate.live_tracking", ...args], {
          cwd: REPOSITORY,
          env: { ...process.env, PYTHONUNBUFFERED: "1" },
        });
        const stdout: Buffer[] = [];
        const stderr: Buffer[] = [];
        const timer = setTimeout(() => child.kill("SIGKILL"), PYTHON_TIMEOUT_MS);
        child.stdout.on("data", (chunk: Buffer) => stdout.push(chunk));
        child.stderr.on("data", (chunk: Buffer) => stderr.push(chunk));
        child.on("error", (error) => {
          clearTimeout(timer);
          reject(error);
        });
        child.on("close", (code) => {
          clearTimeout(timer);
          if (code !== 0) {
            const tail = Buffer.concat(stderr).toString("utf8").trim().split("\n").slice(-3).join(" ");
            reject(new LiveTrackingError(`live_tracking.py exited with ${code}: ${tail}`, 500));
            return;
          }
          try {
            resolve(JSON.parse(Buffer.concat(stdout).toString("utf8")));
          } catch (error) {
            reject(error);
          }
        });
      }),
  );
}

export async function liveBase(trackingDir: string, baseFingerprint: string | null): Promise<LiveBase | { pending: string }> {
  const key = `${trackingDir}|${baseFingerprint}`;
  const cached = cache.bases.get(key);
  if (cached) return cached;
  const result = (await runLiveTracking(["base", trackingDir])) as LiveBase | { pending: string };
  if (!("pending" in result)) cache.bases.set(key, result);
  return result;
}

function isPrefix(shorter: number[], longer: number[]): boolean {
  return shorter.length <= longer.length && shorter.every((value, index) => value === longer[index]);
}

/**
 * The requested epochs' frames. Frames already built for the same files are
 * served from memory; the rest are built in one Python call.
 */
export async function liveFrames(trackingDir: string, requested: number[], extras: number[]): Promise<LiveFramesResponse> {
  const manifest = await liveManifest(trackingDir);
  const byEpoch = new Map(manifest.epochs.map((entry) => [entry.epoch, entry]));
  let known = [...extras];
  let baseKey: string | null = null;
  const frames = new Map<number, LiveFrame>();
  const missing: number[] = [];
  for (const epoch of requested) {
    const entry = byEpoch.get(epoch);
    if (!entry) {
      frames.set(epoch, { epoch, pending: "no such epoch directory" } as LiveFrame);
      continue;
    }
    const hit = cache.frames.get(`${trackingDir}|${epoch}|${entry.fingerprint}`);
    if (hit && (isPrefix(hit.extras, known) || isPrefix(known, hit.extras)) && (baseKey == null || hit.baseKey === baseKey)) {
      if (hit.extras.length > known.length) known = [...hit.extras];
      baseKey = hit.baseKey;
      frames.set(epoch, { ...hit.frame, fingerprint: entry.fingerprint });
    } else {
      missing.push(epoch);
    }
  }

  let extraClassNames: string[] | null = null;
  if (missing.length) {
    const built = (await runLiveTracking([
      "frames",
      trackingDir,
      `--epochs=${missing.join(",")}`,
      `--extra-labels=${known.join(",")}`,
    ])) as LiveFramesResponse & { pending?: string };
    if (built.pending) {
      missing.forEach((epoch) => frames.set(epoch, { epoch, pending: built.pending } as LiveFrame));
    } else {
      if (baseKey != null && built.baseKey !== baseKey) {
        // The base moved between cached and fresh frames; the page refetches it.
        frames.clear();
      }
      baseKey = built.baseKey;
      known = built.extraLabels;
      extraClassNames = built.extraClassNames;
      for (const frame of built.frames) {
        const entry = byEpoch.get(frame.epoch);
        const withPrint = { ...frame, fingerprint: entry?.fingerprint ?? "" };
        frames.set(frame.epoch, withPrint);
        if (entry && !frame.pending) {
          cache.frames.set(`${trackingDir}|${frame.epoch}|${entry.fingerprint}`, { frame, extras: [...known], baseKey: built.baseKey });
        }
      }
      while (cache.frames.size > FRAME_CACHE_LIMIT) cache.frames.delete(cache.frames.keys().next().value!);
    }
  }
  return {
    version: 1,
    baseKey: baseKey ?? "",
    extraLabels: known,
    // Names of appended classes are "class <label>", as the Python side writes them.
    extraClassNames: extraClassNames ?? known.map((label) => `class ${label}`),
    frames: requested.map((epoch) => frames.get(epoch)).filter((frame): frame is LiveFrame => frame != null),
  };
}
