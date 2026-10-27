import fs from "node:fs";
import path from "node:path";
import type { EmbeddingViewIndex } from "@/lib/embedding-views";

/**
 * Read the embedding-view index at build time.
 *
 * The index is small and baked into the page; each view's points are fetched
 * from `public/embedding-views/` only when it is opened, because a single view
 * can hold tens of thousands of points.
 */
export function loadEmbeddingViewIndex(): EmbeddingViewIndex | null {
  const configuredPath = process.env.EMBEDDING_VIEWS_INDEX?.trim();
  const candidates = [
    configuredPath ? path.resolve(configuredPath) : null,
    path.resolve(process.cwd(), "public", "embedding-views", "index.json"),
  ].filter((candidate): candidate is string => candidate != null);

  for (const candidate of candidates) {
    try {
      if (!fs.statSync(candidate).isFile()) continue;
      return JSON.parse(fs.readFileSync(candidate, "utf8")) as EmbeddingViewIndex;
    } catch {
      // A missing or malformed index is reported by the page as "nothing exported yet".
    }
  }
  return null;
}
