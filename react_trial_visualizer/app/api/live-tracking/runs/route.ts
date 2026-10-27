import { NextResponse } from "next/server";

import { listLiveRuns } from "@/lib/live-tracking-server";

export const dynamic = "force-dynamic";

/** Every epoch_tracking/ below the log roots, most recently written first. */
export async function GET(request: Request) {
  const force = new URL(request.url).searchParams.has("refresh");
  try {
    return NextResponse.json({ runs: await listLiveRuns(force) });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status: 500 });
  }
}
