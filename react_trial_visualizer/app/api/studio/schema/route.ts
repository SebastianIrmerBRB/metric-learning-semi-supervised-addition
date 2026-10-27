import { NextResponse } from "next/server";

import { callPython } from "@/lib/studio";

export const dynamic = "force-dynamic";

/** The single source of truth for every choice list the studio offers. */
export async function GET() {
  try {
    const schema = await callPython("schema");
    return NextResponse.json({ ok: true, schema });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : String(error) },
      { status: 503 },
    );
  }
}
