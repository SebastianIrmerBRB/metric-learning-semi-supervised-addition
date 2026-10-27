import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";

export const dynamic = "force-dynamic";

// The dataset image behind one point of the embedding viewer, read straight from
// the data folder: `?run=<run dir>&set=train|validation|test&sample=<index>`.
// Views carry no image paths. The run's datasets are rebuilt once per run by
// evaluate/embedding_image_index.py (a few seconds, mostly imports) and the file
// list is kept in memory. Only image files inside the repository are served, so a
// crafted request cannot read anything else on this machine. tensorboard_api.py
// serves the same URL for the static build.
const REPOSITORY = path.resolve(process.cwd(), "..");
const CONTENT_TYPES: Record<string, string> = {
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".png": "image/png",
  ".webp": "image/webp",
  ".bmp": "image/bmp",
  ".gif": "image/gif",
};
const IMAGE_SETS = new Set(["train", "validation", "test"]);

type ImageIndex = Record<string, (string | null)[]>;

const indexes = new Map<string, Promise<ImageIndex>>();

function python(): string {
  const venv = path.join(REPOSITORY, "venv", "bin", "python");
  return existsSync(venv) ? venv : "python3";
}

function loadIndex(runDir: string): Promise<ImageIndex> {
  let pending = indexes.get(runDir);
  if (!pending) {
    pending = new Promise<ImageIndex>((resolve, reject) => {
      execFile(
        python(),
        ["-m", "evaluate.embedding_image_index", runDir],
        { cwd: REPOSITORY, maxBuffer: 256 * 1024 * 1024 },
        (error, stdout, stderr) => {
          if (error) {
            reject(new Error(stderr.trim().split("\n").pop() || error.message));
            return;
          }
          resolve(JSON.parse(stdout) as ImageIndex);
        },
      );
    });
    // A failure is not cached, so fixing the data folder and retrying works.
    pending.catch(() => indexes.delete(runDir));
    indexes.set(runDir, pending);
  }
  return pending;
}

export async function GET(request: Request) {
  const query = new URL(request.url).searchParams;
  const run = query.get("run") ?? "";
  const set = query.get("set") ?? "";
  const sample = Number(query.get("sample"));
  if (!run || !IMAGE_SETS.has(set) || !Number.isInteger(sample) || sample < 0) {
    return new Response("Expected run, set (train, validation or test) and a sample index", { status: 400 });
  }
  const runDir = path.resolve(REPOSITORY, run);
  let index: ImageIndex;
  try {
    index = await loadIndex(runDir);
  } catch (error) {
    return new Response(`The run's datasets could not be rebuilt: ${(error as Error).message}`, { status: 404 });
  }
  const relative = index[set]?.[sample];
  if (!relative) return new Response("This sample has no image file", { status: 404 });
  const resolved = path.resolve(REPOSITORY, relative);
  const contentType = CONTENT_TYPES[path.extname(resolved).toLowerCase()];
  if (!contentType || !resolved.startsWith(REPOSITORY + path.sep)) {
    return new Response("Not an image inside the repository", { status: 400 });
  }
  try {
    const body = await fs.readFile(resolved);
    return new Response(new Uint8Array(body), {
      headers: { "Content-Type": contentType, "Cache-Control": "private, max-age=3600" },
    });
  } catch {
    return new Response("Image not found", { status: 404 });
  }
}
