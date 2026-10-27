import { NextResponse } from "next/server";

import { CONFIG_ROOT, listConfigFiles } from "@/lib/studio";

export const dynamic = "force-dynamic";

/** List every editable config, tagged with the kind of document it holds. */
export async function GET() {
  return NextResponse.json({ ok: true, root: CONFIG_ROOT, files: listConfigFiles() });
}
