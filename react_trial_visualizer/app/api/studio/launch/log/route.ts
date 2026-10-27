import { NextResponse } from "next/server";

import { readLaunchLog } from "@/lib/studio/processes";

export const dynamic = "force-dynamic";

/** Return the tail of one launch log for the live console. */
export async function GET(request: Request) {
  const id = new URL(request.url).searchParams.get("id") ?? "";
  try {
    return NextResponse.json({ ok: true, ...readLaunchLog(id) });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : String(error) },
      { status: 404 },
    );
  }
}
