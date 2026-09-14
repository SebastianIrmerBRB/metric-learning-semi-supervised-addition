import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { REPO_ROOT, STUDIO_LOG_ROOT, resolvePythonExecutable } from "./index";

export type LaunchKind = "main" | "scheduler";

export interface LaunchRecord {
  id: string;
  kind: LaunchKind;
  label: string;
  command: string[];
  pid: number;
  startedAt: string;
  logPath: string;
  dryRun: boolean;
  /** Filled in once the studio observes the process is gone. */
  finishedAt?: string;
  exitCode?: number | null;
  stoppedByUser?: boolean;
}

export interface LaunchStatus extends LaunchRecord {
  running: boolean;
  logSize: number;
}

const REGISTRY_PATH = path.join(REPO_ROOT, STUDIO_LOG_ROOT, "registry.json");
const MAX_RECORDS = 200;

function readRegistry(): LaunchRecord[] {
  try {
    const raw = JSON.parse(fs.readFileSync(REGISTRY_PATH, "utf8"));
    return Array.isArray(raw) ? (raw as LaunchRecord[]) : [];
  } catch {
    return [];
  }
}

function writeRegistry(records: LaunchRecord[]): void {
  fs.mkdirSync(path.dirname(REGISTRY_PATH), { recursive: true });
  fs.writeFileSync(REGISTRY_PATH, `${JSON.stringify(records.slice(-MAX_RECORDS), null, 2)}\n`);
}

function isRunning(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function logSizeOf(record: LaunchRecord): number {
  try {
    return fs.statSync(path.join(REPO_ROOT, record.logPath)).size;
  } catch {
    return 0;
  }
}

/** Return every launch this studio knows about, newest first. */
export function listLaunches(): LaunchStatus[] {
  const records = readRegistry();
  let changed = false;

  const statuses = records.map((record) => {
    const running = isRunning(record.pid);
    if (!running && !record.finishedAt) {
      record.finishedAt = new Date().toISOString();
      changed = true;
    }
    return { ...record, running, logSize: logSizeOf(record) };
  });

  if (changed) writeRegistry(records);
  return statuses.reverse();
}

export function findLaunch(id: string): LaunchStatus | undefined {
  return listLaunches().find((record) => record.id === id);
}

/** Start one detached child and record it so a dev-server reload cannot lose it. */
export function startLaunch(options: {
  kind: LaunchKind;
  label: string;
  args: string[];
  dryRun?: boolean;
}): LaunchRecord {
  const id = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  const logDirectory = path.join(REPO_ROOT, STUDIO_LOG_ROOT);
  fs.mkdirSync(logDirectory, { recursive: true });

  const logPath = path.join(logDirectory, `${id}.log`);
  const logHandle = fs.openSync(logPath, "a");
  const python = resolvePythonExecutable();
  const command = [python, ...options.args];

  fs.writeSync(
    logHandle,
    `# ${new Date().toISOString()}\n# ${command.join(" ")}\n\n`,
  );

  const child = spawn(python, options.args, {
    cwd: REPO_ROOT,
    detached: process.platform !== "win32",
    stdio: ["ignore", logHandle, logHandle],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  child.unref();
  fs.closeSync(logHandle);

  if (child.pid === undefined) {
    throw new Error("The process could not be started.");
  }

  const record: LaunchRecord = {
    id,
    kind: options.kind,
    label: options.label,
    command,
    pid: child.pid,
    startedAt: new Date().toISOString(),
    logPath: path.relative(REPO_ROOT, logPath).split(path.sep).join("/"),
    dryRun: Boolean(options.dryRun),
  };

  writeRegistry([...readRegistry(), record]);
  return record;
}

/** Signal a launch's whole process group so its children stop with it. */
export function stopLaunch(id: string, signal: NodeJS.Signals = "SIGTERM"): boolean {
  const records = readRegistry();
  const record = records.find((item) => item.id === id);
  if (!record) return false;

  try {
    if (process.platform === "win32") process.kill(record.pid, signal);
    else process.kill(-record.pid, signal);
  } catch {
    return false;
  }
  record.stoppedByUser = true;
  writeRegistry(records);
  return true;
}

/** Read the tail of a launch log without loading a multi-gigabyte file. */
export function readLaunchLog(id: string, maxBytes = 200_000): { text: string; size: number } {
  const record = readRegistry().find((item) => item.id === id);
  if (!record) throw new Error(`Unknown launch: ${id}`);

  const absolutePath = path.join(REPO_ROOT, record.logPath);
  let size = 0;
  try {
    size = fs.statSync(absolutePath).size;
  } catch {
    return { text: "", size: 0 };
  }

  const start = Math.max(0, size - maxBytes);
  const handle = fs.openSync(absolutePath, "r");
  try {
    const buffer = Buffer.alloc(size - start);
    fs.readSync(handle, buffer, 0, buffer.length, start);
    const text = buffer.toString("utf8");
    return { text: start > 0 ? `…truncated…\n${text}` : text, size };
  } finally {
    fs.closeSync(handle);
  }
}

/** Forget finished launches; running ones are kept so they stay controllable. */
export function clearFinishedLaunches(): number {
  const records = readRegistry();
  const kept = records.filter((record) => isRunning(record.pid));
  writeRegistry(kept);
  return records.length - kept.length;
}
