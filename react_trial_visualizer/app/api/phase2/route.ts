import { execFile } from "node:child_process";
import path from "node:path";
import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

/**
 * Live Phase-2 snapshots for the dev server: runs the same Python exporters the
 * static build reads (main_eval/ablation_report.py, main_eval/geometry_tracking.py)
 * with --stdout, so /phase2 shows runs as they finish without a manual export.
 * Results are cached for a short time and concurrent requests share one run.
 */

const REPO = path.resolve(process.cwd(), "..");
const PYTHON = process.env.PHASE2_PYTHON?.trim() || path.join(REPO, "venv", "bin", "python");
const TTL_MS = 30_000;
const SCRIPTS: Record<string, string[]> = {
  ablations: ["main_eval/ablation_report.py", "--stdout"],
  geometry: ["main_eval/geometry_tracking.py", "--stdout", "--no-figures"],
};

interface Entry {
  at: number;
  body?: string;
  error?: string;
  pending?: Promise<void>;
}
const cache = new Map<string, Entry>();

function run(kind: string): Promise<void> {
  const entry = cache.get(kind) ?? { at: 0 };
  cache.set(kind, entry);
  entry.pending = new Promise<void>((resolve) => {
    execFile(PYTHON, SCRIPTS[kind], { cwd: REPO, maxBuffer: 256 * 1024 * 1024, timeout: 300_000 }, (error, stdout, stderr) => {
      entry.at = Date.now();
      if (error) {
        entry.error = `${error.message}\n${String(stderr).slice(-2000)}`;
      } else {
        entry.body = stdout.trim() || "null";
        entry.error = undefined;
      }
      entry.pending = undefined;
      resolve();
    });
  });
  return entry.pending;
}

export async function GET(request: Request) {
  const params = new URL(request.url).searchParams;
  const kind = params.get("kind") ?? "ablations";
  if (!(kind in SCRIPTS)) {
    return NextResponse.json({ error: `unknown kind ${kind}` }, { status: 400 });
  }
  const entry = cache.get(kind);
  const stale = !entry || Date.now() - entry.at > TTL_MS || params.get("refresh") === "1";
  if (entry?.pending) await entry.pending;
  else if (stale) await run(kind);
  const current = cache.get(kind);
  if (!current?.body) {
    return NextResponse.json({ error: current?.error ?? "no snapshot" }, { status: 500 });
  }
  return new NextResponse(current.body, {
    headers: { "content-type": "application/json", "x-phase2-computed-at": new Date(current.at).toISOString() },
  });
}
