const KNOWN_ARTIFACTS = new Set([
  "head_state.pt",
  "run_config.json",
  "test_embeddings.npz",
  "test_tsne.csv",
  "test_tsne_3d.csv",
]);

/**
 * Turn a pasted Linux or Windows WSL path into the Linux path understood by
 * the local Next server. Quotes copied along with the path are harmless.
 */
export function cleanEmbeddingPath(value: string): string {
  let cleaned = value.trim().replace(/^["']+|["']+$/g, "").replace(/\\/g, "/");
  cleaned = cleaned.replace(/^\/\/wsl(?:\.localhost|\$)\/[^/]+(?=\/)/i, "");
  cleaned = cleaned.replace(/\/+/g, "/");
  return cleaned.length > 1 ? cleaned.replace(/\/$/, "") : cleaned;
}

/** A case-insensitive representation used only for matching view metadata. */
export function normalizeEmbeddingLookup(value: string): string {
  return cleanEmbeddingPath(value).toLowerCase();
}

/**
 * Artifact paths identify their containing run. This lets an indexed runDir
 * match a pasted test_embeddings.npz or head_state.pt path exactly.
 */
export function embeddingRunLookup(value: string): string {
  const normalized = normalizeEmbeddingLookup(value);
  const slash = normalized.lastIndexOf("/");
  if (slash >= 0 && KNOWN_ARTIFACTS.has(normalized.slice(slash + 1))) {
    return normalized.slice(0, slash);
  }
  const tracking = normalized.indexOf("/epoch_tracking/");
  return tracking >= 0 ? normalized.slice(0, tracking) : normalized;
}

export function looksLikeEmbeddingPath(value: string): boolean {
  const cleaned = cleanEmbeddingPath(value);
  return cleaned.includes("/") || /\.(?:csv|json|npz|pt)$/i.test(cleaned);
}
