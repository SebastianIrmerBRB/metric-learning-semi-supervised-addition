import fs from "node:fs";
import path from "node:path";

import { NextResponse } from "next/server";

import { PathError, REPO_ROOT, backupExistingFile, resolveConfigPath } from "@/lib/studio";

export const dynamic = "force-dynamic";

function failure(error: unknown) {
  const status = error instanceof PathError ? 400 : 500;
  return NextResponse.json(
    { ok: false, error: error instanceof Error ? error.message : String(error) },
    { status },
  );
}

/** Read one config file, returning both the parsed document and its raw text. */
export async function GET(request: Request) {
  const relativePath = new URL(request.url).searchParams.get("path") ?? "";
  try {
    const absolutePath = resolveConfigPath(relativePath);
    const raw = fs.readFileSync(absolutePath, "utf8");
    const stats = fs.statSync(absolutePath);
    let data: unknown = null;
    let parseError: string | null = null;
    try {
      data = JSON.parse(raw);
    } catch (error) {
      parseError = error instanceof Error ? error.message : String(error);
    }
    return NextResponse.json({
      ok: true,
      path: relativePath,
      raw,
      data,
      parseError,
      modified: stats.mtime.toISOString(),
    });
  } catch (error) {
    return failure(error);
  }
}

/** Write a config file, keeping a timestamped copy of whatever it replaced. */
export async function PUT(request: Request) {
  try {
    const body = (await request.json()) as { path?: string; data?: unknown };
    const relativePath = body.path ?? "";
    const absolutePath = resolveConfigPath(relativePath);
    if (body.data === undefined) throw new PathError("No document was supplied.");

    const backup = backupExistingFile(relativePath);
    fs.mkdirSync(path.dirname(absolutePath), { recursive: true });
    fs.writeFileSync(absolutePath, `${JSON.stringify(body.data, null, 2)}\n`);

    return NextResponse.json({
      ok: true,
      path: relativePath,
      backup,
      modified: fs.statSync(absolutePath).mtime.toISOString(),
    });
  } catch (error) {
    return failure(error);
  }
}

/** Remove a config file after copying it into the studio backup tree. */
export async function DELETE(request: Request) {
  const relativePath = new URL(request.url).searchParams.get("path") ?? "";
  try {
    const absolutePath = resolveConfigPath(relativePath);
    if (!fs.existsSync(absolutePath)) throw new PathError(`No such config: ${relativePath}`);
    const backup = backupExistingFile(relativePath);
    fs.unlinkSync(absolutePath);
    return NextResponse.json({ ok: true, path: relativePath, backup, repoRoot: REPO_ROOT });
  } catch (error) {
    return failure(error);
  }
}
