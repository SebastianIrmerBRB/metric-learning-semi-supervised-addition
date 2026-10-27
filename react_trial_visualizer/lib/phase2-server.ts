import fs from "node:fs";
import path from "node:path";
import type { AblationSnapshot, GeometrySnapshot } from "@/lib/phase2";

/**
 * Read a Phase-2 snapshot from public/ at build time (static export), or null when
 * the exporter has not been run yet, so the page can say how to produce it.
 */
function readSnapshot<T>(envName: string, fileName: string): T | null {
  const configured = process.env[envName]?.trim();
  const candidates = [
    configured ? path.resolve(configured) : null,
    path.resolve(process.cwd(), "public", fileName),
  ].filter((candidate): candidate is string => candidate != null);
  for (const candidate of candidates) {
    try {
      if (!fs.statSync(candidate).isFile()) continue;
      return JSON.parse(fs.readFileSync(candidate, "utf8")) as T;
    } catch {
      // Missing or malformed: try the next location, then report "not exported yet".
    }
  }
  return null;
}

export function loadAblationSnapshot(): AblationSnapshot | null {
  return readSnapshot<AblationSnapshot>("ABLATIONS_JSON", "ablations.json");
}

export function loadGeometrySnapshot(): GeometrySnapshot | null {
  return readSnapshot<GeometrySnapshot>("GEOMETRY_JSON", "geometry.json");
}
