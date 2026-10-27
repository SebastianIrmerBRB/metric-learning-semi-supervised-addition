import { NextResponse } from "next/server";

import { LiveTrackingError, liveManifest, resolveTrackingDir } from "@/lib/live-tracking-server";

export const dynamic = "force-dynamic";

/** One run's epochs with a fingerprint each, from file stats alone: cheap enough to poll. */
export async function GET(request: Request) {
  try {
    const trackingDir = await resolveTrackingDir(new URL(request.url).searchParams.get("dir"));
    return NextResponse.json(await liveManifest(trackingDir));
  } catch (error) {
    const status = error instanceof LiveTrackingError ? error.status : 500;
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status });
  }
}
