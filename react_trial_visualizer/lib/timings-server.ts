import fs from "node:fs";
import path from "node:path";
import type { TimingSnapshot, TimingSnapshotEntry } from "@/lib/timings";

const SNAPSHOT_NAME = "timings.json";
const MAX_DEPTH = 3;

/**
 * Where `scripts/bench_epoch_cost.py` results are kept. TIMINGS_DIR overrides
 * (several separated by ":" ";" or ","); otherwise F:\Studien\Timings as seen
 * from WSL and from Windows, then public/timings.json.
 */
function roots(): string[] {
  const configured = process.env.TIMINGS_DIR?.trim();
  if (configured) return configured.split(/[;,]|:(?!\\|\/)/).filter(Boolean).map((root) => path.resolve(root));
  return ["/mnt/f/Studien/Timings", "F:/Studien/Timings", path.resolve(process.cwd(), "public")];
}

function findSnapshots(root: string, depth = 0): string[] {
  let entries: fs.Dirent[];
  try {
    entries = fs.readdirSync(root, { withFileTypes: true });
  } catch {
    return [];
  }
  const found: string[] = [];
  for (const entry of entries) {
    const full = path.join(root, entry.name);
    if (entry.isFile() && entry.name === SNAPSHOT_NAME) found.push(full);
    // Case folders hold only CSVs; skip them instead of walking hundreds of directories.
    else if ((entry.isDirectory() || entry.isSymbolicLink()) && depth < MAX_DEPTH && entry.name !== "cases") {
      found.push(...findSnapshots(full, depth + 1));
    }
  }
  return found;
}

/** Every readable snapshot, newest first; the per-trial rows stay on disk. */
export function loadTimingSnapshots(): TimingSnapshotEntry[] {
  const entries: TimingSnapshotEntry[] = [];
  const seen = new Set<string>();
  for (const root of roots()) {
    for (const file of findSnapshots(root)) {
      if (seen.has(file)) continue;
      seen.add(file);
      try {
        const raw = JSON.parse(fs.readFileSync(file, "utf8")) as TimingSnapshot & { trials?: unknown };
        if (!Array.isArray(raw.studies)) continue;
        delete raw.trials;
        const relative = path.relative(root, path.dirname(file)) || path.basename(path.dirname(file));
        entries.push({ id: relative.split(path.sep).join("/"), path: file, snapshot: raw });
      } catch {
        // A snapshot being written or from another version: skip it.
      }
    }
    if (entries.length) break;
  }
  return entries.sort((a, b) => b.snapshot.generated.localeCompare(a.snapshot.generated));
}
