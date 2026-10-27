import path from "node:path";

/**
 * Folders outside the repository that repository symlinks may point into, e.g.
 * logs/final/phase_2 -> /mnt/f/Studien/phase_2 (runs moved to F: for disk space).
 * A requested path must still be named inside the repository; only its resolved
 * target may live here. Override with DATA_ROOTS (separated by ":" ";" or ",").
 */
const DATA_ROOTS = (process.env.DATA_ROOTS?.trim() || "/mnt/f/Studien")
  .split(/[:;,]/)
  .filter(Boolean)
  .map((root) => path.resolve(root));

function isInside(root: string, candidate: string): boolean {
  const relative = path.relative(root, candidate);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

/** True when a symlink-resolved path lies in the repository or in one of the data roots. */
export function isAllowedResolvedPath(repository: string, candidate: string): boolean {
  return isInside(repository, candidate) || DATA_ROOTS.some((root) => isInside(root, candidate));
}
