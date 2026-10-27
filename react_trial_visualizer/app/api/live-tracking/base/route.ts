import { NextResponse } from "next/server";

import { LiveTrackingError, liveBase, liveManifest, resolveTrackingDir } from "@/lib/live-tracking-server";

export const dynamic = "force-dynamic";

/** The tracked points, classes and subsets shared by every epoch of a run. */
export async function GET(request: Request) {
  try {
    const trackingDir = await resolveTrackingDir(new URL(request.url).searchParams.get("dir"));
    const manifest = await liveManifest(trackingDir);
    return NextResponse.json(await liveBase(trackingDir, manifest.baseFingerprint));
  } catch (error) {
    const status = error instanceof LiveTrackingError ? error.status : 500;
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status });
  }
}
