/**
 * Build the hosted Phase-2 site (npm run build:share) into share/, ready to
 * upload to Cloudflare Pages (or any static host serving it at the domain root):
 *
 *   /             geometry (B6), snapshot inlined
 *   /embeddings/  the embedding viewer over the exported views in public/embedding-views/
 *
 *   SHARE_TABS=all     also bundle the ablation tabs (default: geometry only)
 *   SHARE_VIEWS=all    publish every exported view; any other value is a regex on the
 *                      view label. Default: everything except the scratch runs below.
 *
 * Published views lose their runDir (an absolute path on this machine, and the
 * key the image route needs, so the hosted viewer shows no images). Views above
 * 24 MiB are split into `<file>.0`, `<file>.1`, … because Cloudflare Pages
 * refuses files over 25 MiB; the viewer joins them again.
 *
 * Next copies all of public/ into the export; it is pruned afterwards. The build also regenerates .next/types and next-env.d.ts
 * for its own routes, which would break `npm run typecheck` for the atlas, so
 * both are restored afterwards.
 */
import { spawnSync } from "node:child_process";
import {
  cpSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  readdirSync,
  renameSync,
  rmSync,
  statSync,
  writeFileSync,
} from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const shareDirectory = path.join(projectRoot, "share");
const viewsSource = path.join(projectRoot, "public", "embedding-views");
// Beside share/, so the final move into share/ is a rename on the same disk.
const viewsStaging = path.join(projectRoot, ".share-embedding-views");
const SCRATCH_RUNS = /^(demo_epoch_tracking|headrun|trackrun)\b/;
const PART_BYTES = 24 * 1024 * 1024;

const mb = (bytes) => `${(bytes / 1024 / 1024).toFixed(1)} MB`;

/** Split UTF-8 text into pieces of at most PART_BYTES, never inside a character. */
function splitUtf8(text) {
  const bytes = Buffer.from(text, "utf8");
  const parts = [];
  let start = 0;
  while (start < bytes.length) {
    let end = Math.min(start + PART_BYTES, bytes.length);
    while (end < bytes.length && (bytes[end] & 0xc0) === 0x80) end -= 1;
    parts.push(bytes.subarray(start, end));
    start = end;
  }
  return parts;
}

function stageViews() {
  rmSync(viewsStaging, { recursive: true, force: true });
  const indexPath = path.join(viewsSource, "index.json");
  if (!existsSync(indexPath)) return null;
  mkdirSync(viewsStaging, { recursive: true });

  const selector = process.env.SHARE_VIEWS?.trim();
  const wanted =
    selector === "all" ? () => true : selector ? (label) => new RegExp(selector).test(label) : (label) => !SCRATCH_RUNS.test(label);
  const index = JSON.parse(readFileSync(indexPath, "utf8"));
  const views = [];
  let total = 0;
  for (const entry of index.views) {
    if (!wanted(entry.label)) continue;
    const view = JSON.parse(readFileSync(path.join(viewsSource, entry.file), "utf8"));
    view.runDir = null;
    const parts = splitUtf8(JSON.stringify(view));
    if (parts.length === 1) {
      writeFileSync(path.join(viewsStaging, entry.file), parts[0]);
    } else {
      parts.forEach((part, number) => writeFileSync(path.join(viewsStaging, `${entry.file}.${number}`), part));
    }
    total += parts.reduce((sum, part) => sum + part.length, 0);
    views.push({ ...entry, runDir: null, ...(parts.length > 1 ? { parts: parts.length } : {}) });
  }
  const publishedIndex = path.join(viewsStaging, "index.json");
  writeFileSync(publishedIndex, JSON.stringify({ ...index, views }));
  console.log(`Embedding views: ${views.length} of ${index.views.length} published, ${mb(total)}.`);
  return publishedIndex;
}

const routeTypes = path.join(projectRoot, ".next", "types");
// Outside .next/, which `next build` empties before it starts.
const routeTypesBackup = path.join(mkdtempSync(path.join(os.tmpdir(), "atlas-types-")), "types");
const nextEnv = path.join(projectRoot, "next-env.d.ts");
const nextEnvBefore = existsSync(nextEnv) ? readFileSync(nextEnv, "utf8") : null;
if (existsSync(routeTypes)) cpSync(routeTypes, routeTypesBackup, { recursive: true });

function restoreAtlasTypes() {
  if (existsSync(routeTypesBackup)) {
    rmSync(routeTypes, { recursive: true, force: true });
    cpSync(routeTypesBackup, routeTypes, { recursive: true });
  }
  rmSync(path.dirname(routeTypesBackup), { recursive: true, force: true });
  if (nextEnvBefore != null) writeFileSync(nextEnv, nextEnvBefore);
}

rmSync(shareDirectory, { recursive: true, force: true });
const publishedIndex = stageViews();
const nextBin = path.join(projectRoot, "node_modules", "next", "dist", "bin", "next");
const build = spawnSync(process.execPath, [nextBin, "build"], {
  cwd: projectRoot,
  stdio: "inherit",
  env: {
    ...process.env,
    TRIAL_ATLAS_SHARE: "1",
    // An index with no views makes the page say so, instead of falling back to all of public/.
    EMBEDDING_VIEWS_INDEX: publishedIndex ?? path.join(os.tmpdir(), "no-embedding-views.json"),
  },
});
restoreAtlasTypes();
if (build.status !== 0) process.exit(build.status ?? 1);
if (!existsSync(path.join(shareDirectory, "index.html"))) {
  throw new Error("The share build did not produce share/index.html.");
}

// Drop what Next copied from public/; keep its own output (pages, RSC payloads, _next/).
for (const name of readdirSync(path.join(projectRoot, "public"))) {
  rmSync(path.join(shareDirectory, name), { recursive: true, force: true });
}
if (publishedIndex) {
  // The viewer reads the index from the page, not from this file.
  rmSync(publishedIndex);
  renameSync(viewsStaging, path.join(shareDirectory, "embedding-views"));
}

function size(target) {
  const stat = statSync(target);
  if (!stat.isDirectory()) return stat.size;
  return readdirSync(target).reduce((sum, name) => sum + size(path.join(target, name)), 0);
}
function largest(target) {
  const stat = statSync(target);
  if (!stat.isDirectory()) return stat.size;
  return Math.max(0, ...readdirSync(target).map((name) => largest(path.join(target, name))));
}
console.log(`\nshare/ ready: ${mb(size(shareDirectory))} total, largest file ${mb(largest(shareDirectory))}.`);
