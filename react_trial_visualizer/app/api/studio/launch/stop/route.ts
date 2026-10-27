import { NextResponse } from "next/server";

import { stopLaunch } from "@/lib/studio/processes";

export const dynamic = "force-dynamic";

/** Signal a running launch's process group. */
export async function POST(request: Request) {
  const body = (await request.json()) as { id?: string; signal?: string };
  if (!body.id) {
    return NextResponse.json({ ok: false, error: "A launch id is required." }, { status: 400 });
  }
  const signal = body.signal === "SIGKILL" ? "SIGKILL" : "SIGTERM";
  const stopped = stopLaunch(body.id, signal);
  return NextResponse.json({ ok: stopped, stopped });
}
