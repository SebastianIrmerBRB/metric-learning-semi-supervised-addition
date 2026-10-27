import { NextResponse } from "next/server";

import { callPython } from "@/lib/studio";

export const dynamic = "force-dynamic";

const OPERATIONS: Record<string, string> = {
  hparam: "validate_hparam",
  experiment: "validate_experiment",
  manifest: "validate_manifest",
  inspect: "inspect",
};

/** Validate an unsaved document with the same code the real run would use. */
export async function POST(request: Request) {
  try {
    const body = (await request.json()) as { kind?: string } & Record<string, unknown>;
    const operation = OPERATIONS[body.kind ?? ""];
    if (!operation) {
      return NextResponse.json(
        { ok: false, error: `Unknown validation kind: ${body.kind}` },
        { status: 400 },
      );
    }
    const { kind, ...payload } = body;
    void kind;
    const result = await callPython(operation, payload);
    return NextResponse.json({ ok: true, result });
  } catch (error) {
    return NextResponse.json(
      { ok: false, error: error instanceof Error ? error.message : String(error) },
      { status: 500 },
    );
  }
}
