/**
 * Build-time switches for the hosted thesis site (GitHub Pages). Both are empty for local use, so nothing changes there.
 *
 * NEXT_PUBLIC_BASE_PATH: the path the site is served below, e.g. /<repository>/trial-atlas. next.config.ts passes it to
 *   Next as basePath, which covers links and assets. Plain fetch() URLs do not get it, so they go through withBase().
 * NEXT_PUBLIC_THESIS_SITE=1: hide the workspaces that need local run folders, a dev server or data the compact thesis
 *   bundle does not contain (sorted archive, config studio, epoch tracking, embeddings, Phase-2 explorer, todo, timings,
 *   run configs).
 */
export const BASE_PATH = process.env.NEXT_PUBLIC_BASE_PATH ?? "";
export const THESIS_SITE = process.env.NEXT_PUBLIC_THESIS_SITE === "1";

export function withBase(path: string): string {
  return `${BASE_PATH}${path}`;
}
