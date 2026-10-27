import fs from "node:fs";
import path from "node:path";
import type { EpochTrackingSnapshot } from "@/lib/epoch-tracking";

/**
 * Read the epoch-tracking snapshot at build time.
 *
 * The published site is a static export, so there is no request-time filesystem
 * to read from: the snapshot has to be baked into the page. `public/` is the
 * canonical location because `evaluate/export_react.py` writes the JSON and the
 * PNGs there together, and the PNGs have to be served from `public/` regardless.
 *
 * Returns null rather than throwing when the export has never been run, so the
 * page can explain how to produce it instead of failing the build.
 */
export function loadEpochTrackingSnapshot(): EpochTrackingSnapshot | null {
  const configuredPath = process.env.EPOCH_TRACKING_JSON?.trim();
  const candidates = [
    configuredPath ? path.resolve(configuredPath) : null,
    path.resolve(process.cwd(), "public", "epoch-tracking.json"),
  ].filter((candidate): candidate is string => candidate != null);

  for (const candidate of candidates) {
    try {
      if (!fs.statSync(candidate).isFile()) continue;
      return JSON.parse(fs.readFileSync(candidate, "utf8")) as EpochTrackingSnapshot;
    } catch {
      // Try the next location; a missing or malformed snapshot is reported by
      // the page as "not exported yet", which is what it almost always is.
    }
  }
  return null;
}
