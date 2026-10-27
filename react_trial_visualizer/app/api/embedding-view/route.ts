import fs from "node:fs/promises";
import path from "node:path";

import { NextResponse } from "next/server";

import { isAllowedResolvedPath } from "@/lib/data-roots";
import { cleanEmbeddingPath } from "@/lib/embedding-paths";

export const dynamic = "force-dynamic";

const REPOSITORY = path.resolve(process.cwd(), "..");

function isInsideRepository(candidate: string): boolean {
  const relative = path.relative(REPOSITORY, candidate);
  return relative === "" || (!relative.startsWith(`..${path.sep}`) && relative !== ".." && !path.isAbsolute(relative));
}

function relativeToRepository(candidate: string): string {
  return path.relative(REPOSITORY, candidate).split(path.sep).join("/");
}

async function projectionFor(requested: string): Promise<string> {
  let stats;
  try {
    stats = await fs.stat(requested);
  } catch {
    throw new Error(`No file or directory exists at ${requested}`);
  }

  if (stats.isDirectory()) return path.join(requested, "test_tsne.csv");

  const name = path.basename(requested).toLowerCase();
  if (name === "test_tsne.csv") return requested;
  if (name === "test_tsne_3d.csv") return path.join(path.dirname(requested), "test_tsne.csv");
  if (name === "head_state.pt") {
    throw new Error(
      "This saved head has no lightweight projection to open directly. Export it first with evaluate/export_embedding_views.py --run <run directory>.",
    );
  }
  return path.join(path.dirname(requested), "test_tsne.csv");
}

/**
 * Open a run projection from a pasted local path. In particular,
 * test_embeddings.npz resolves to the test_tsne.csv written beside it.
 */
export async function GET(request: Request) {
  const supplied = new URL(request.url).searchParams.get("path") ?? "";
  const cleaned = cleanEmbeddingPath(supplied);
  if (!cleaned) {
    return NextResponse.json({ ok: false, error: "Paste a run or embedding artifact path." }, { status: 400 });
  }

  const requested = path.resolve(REPOSITORY, cleaned);
  if (!isInsideRepository(requested)) {
    return NextResponse.json(
      { ok: false, error: "Embedding paths must point somewhere inside the metric-learning repository." },
      { status: 400 },
    );
  }

  try {
    const projection = await projectionFor(requested);
    const realProjection = await fs.realpath(projection);
    if (!isAllowedResolvedPath(REPOSITORY, realProjection)) throw new Error("The projection resolves outside the repository and the data roots.");
    const runDir = path.dirname(realProjection);
    const csv = await fs.readFile(realProjection, "utf8");

    const twinPath = realProjection.replace(/\.csv$/i, "_3d.csv");
    let csv3d: string | null = null;
    try {
      const realTwin = await fs.realpath(twinPath);
      if (isAllowedResolvedPath(REPOSITORY, realTwin)) csv3d = await fs.readFile(realTwin, "utf8");
    } catch {
      // A two-dimensional projection is complete on its own.
    }

    return NextResponse.json({
      ok: true,
      sourcePath: relativeToRepository(realProjection),
      runDir: relativeToRepository(runDir),
      label: relativeToRepository(runDir),
      csv,
      csv3d,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    const missingProjection = message.includes("ENOENT")
      ? `No test_tsne.csv was found next to ${relativeToRepository(requested)}.`
      : message;
    return NextResponse.json({ ok: false, error: missingProjection }, { status: 404 });
  }
}
