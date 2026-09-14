import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

import { REPO_ROOT } from "./paths";

const SERVICE_SCRIPT = path.join(process.cwd(), "scripts", "studio_service.py");
const REQUEST_TIMEOUT_MS = 180_000;

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (error: Error) => void;
  timer: NodeJS.Timeout;
}

interface Bridge {
  child: ChildProcessWithoutNullStreams;
  pending: Map<number, PendingRequest>;
  buffer: string;
  stderrTail: string[];
  nextId: number;
}

// Next's dev server re-evaluates route modules on every edit; the interpreter
// takes seconds to import torch, so the warm process lives on globalThis.
const globalBridge = globalThis as typeof globalThis & { __studioBridge?: Bridge | null };

/** Pick the interpreter that already has this project's dependencies. */
export function resolvePythonExecutable(): string {
  const override = process.env.STUDIO_PYTHON;
  if (override) return override;

  const candidates = [
    path.join(REPO_ROOT, "venv", "bin", "python"),
    path.join(REPO_ROOT, "venv", "Scripts", "python.exe"),
    path.join(REPO_ROOT, ".venv", "bin", "python"),
  ];
  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  return process.platform === "win32" ? "python" : "python3";
}

function startBridge(): Bridge {
  const child = spawn(resolvePythonExecutable(), ["-u", SERVICE_SCRIPT], {
    cwd: REPO_ROOT,
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });

  const bridge: Bridge = {
    child,
    pending: new Map(),
    buffer: "",
    stderrTail: [],
    nextId: 1,
  };

  child.stdout.setEncoding("utf8");
  child.stdout.on("data", (chunk: string) => {
    bridge.buffer += chunk;
    let newlineIndex = bridge.buffer.indexOf("\n");
    while (newlineIndex >= 0) {
      const line = bridge.buffer.slice(0, newlineIndex).trim();
      bridge.buffer = bridge.buffer.slice(newlineIndex + 1);
      newlineIndex = bridge.buffer.indexOf("\n");
      if (!line) continue;

      let message: { id?: number; ok?: boolean; result?: unknown; error?: string };
      try {
        message = JSON.parse(line);
      } catch {
        continue;
      }
      const pending = message.id == null ? undefined : bridge.pending.get(message.id);
      if (!pending) continue;
      bridge.pending.delete(message.id as number);
      clearTimeout(pending.timer);
      if (message.ok) pending.resolve(message.result);
      else pending.reject(new Error(message.error ?? "The validation service failed."));
    }
  });

  child.stderr.setEncoding("utf8");
  child.stderr.on("data", (chunk: string) => {
    bridge.stderrTail.push(chunk);
    if (bridge.stderrTail.length > 40) bridge.stderrTail.shift();
  });

  const fail = (reason: string) => {
    for (const [, pending] of bridge.pending) {
      clearTimeout(pending.timer);
      pending.reject(new Error(`${reason}\n${bridge.stderrTail.join("").slice(-2000)}`));
    }
    bridge.pending.clear();
    if (globalBridge.__studioBridge === bridge) globalBridge.__studioBridge = null;
  };

  child.on("error", (error) => fail(`Could not start the Python validation service: ${error.message}`));
  child.on("exit", (code) => fail(`The Python validation service exited (code ${code}).`));

  return bridge;
}

function getBridge(): Bridge {
  if (!globalBridge.__studioBridge || globalBridge.__studioBridge.child.exitCode !== null) {
    globalBridge.__studioBridge = startBridge();
  }
  return globalBridge.__studioBridge;
}

/** Send one request to the warm Python service and await its reply. */
export function callPython<T>(op: string, payload: Record<string, unknown> = {}): Promise<T> {
  const bridge = getBridge();
  const id = bridge.nextId++;

  return new Promise<T>((resolve, reject) => {
    const timer = setTimeout(() => {
      bridge.pending.delete(id);
      reject(new Error(`The Python validation service did not answer '${op}' in time.`));
    }, REQUEST_TIMEOUT_MS);

    bridge.pending.set(id, {
      resolve: resolve as (value: unknown) => void,
      reject,
      timer,
    });

    try {
      bridge.child.stdin.write(`${JSON.stringify({ id, op, ...payload })}\n`);
    } catch (error) {
      bridge.pending.delete(id);
      clearTimeout(timer);
      reject(error instanceof Error ? error : new Error(String(error)));
    }
  });
}
