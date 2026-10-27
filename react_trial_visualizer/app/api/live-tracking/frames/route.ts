import { NextResponse } from "next/server";

import { LiveTrackingError, liveFrames, resolveTrackingDir } from "@/lib/live-tracking-server";

export const dynamic = "force-dynamic";

function integers(value: string | null): number[] {
  return (value ?? "")
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean)
    .map(Number)
    .filter(Number.isInteger);
}

/**
 * `?dir=&epochs=-1,0,5&extra=12,40`: those epochs' frames. `extra` lists the
 * class labels the page has already appended, so class indices stay stable.
 */
export async function GET(request: Request) {
  const query = new URL(request.url).searchParams;
  try {
    const trackingDir = await resolveTrackingDir(query.get("dir"));
    const epochs = integers(query.get("epochs"));
    if (!epochs.length) throw new LiveTrackingError("Pass ?epochs=-1,0,5.");
    return NextResponse.json(await liveFrames(trackingDir, epochs, integers(query.get("extra"))));
  } catch (error) {
    const status = error instanceof LiveTrackingError ? error.status : 500;
    return NextResponse.json({ error: error instanceof Error ? error.message : String(error) }, { status });
  }
}
