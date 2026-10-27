import { NextResponse } from "next/server";

import { clearFinishedLaunches, listLaunches, startLaunch, type LaunchKind } from "@/lib/studio/processes";

export const dynamic = "force-dynamic";

const ENTRYPOINTS: Record<LaunchKind, string> = {
  main: "main.py",
  scheduler: "launch_runs.py",
};

/** Report every launch the studio started, including ones from an earlier reload. */
export async function GET() {
  return NextResponse.json({ ok: true, launches: listLaunches() });
}

/** Start `main.py` or `launch_runs.py` detached, with its output tee'd to a log. */
export async function POST(request: Request) {
  try {
    const body = (await request.json()) as {
      kind?: LaunchKind;
      label?: string;
      args?: unknown;
      dryRun?: boolean;
    };

    const entrypoint = body.kind ? ENTRYPOINTS[body.kind] : undefined;
    if (!entrypoint) {
      return NextResponse.json(
        { ok: false, error: `Unknown launch kind: ${body.kind}` },
        { status: 400 },
      );
    }
    if (!Array.isArray(body.args) || body.args.some((item) => typeof item !== "string")) {
      return NextResponse.json(
        { ok: false, error: "args must be an array of strings." },
        { status: 400 },
      );
    }

    const record = startLaunch({
      kind: body.kind as LaunchKind,
      label: body.label?.trim() || entrypoint,
      args: [entrypoint, ...(body.args as string[])],
      dryRun: body.dryRun,
    });
    return NextResponse.json({ ok: true, launch: record });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : String(error) },
      { status: 500 },
    );
  }
}

/** Drop finished entries from the launch list; running ones stay controllable. */
export async function DELETE() {
  return NextResponse.json({ ok: true, removed: clearFinishedLaunches() });
}
