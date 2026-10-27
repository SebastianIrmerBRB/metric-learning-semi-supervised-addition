"use client";

import { FileWarning, Loader2 } from "lucide-react";
import { useEffect, useState } from "react";

import { studioApi } from "@/lib/studio/client";

interface InspectResult {
  exists: boolean;
  data?: unknown;
  error?: string;
}

function flatten(value: unknown, prefix = ""): [string, unknown][] {
  if (!value || typeof value !== "object" || Array.isArray(value)) return [[prefix, value]];
  return Object.entries(value as Record<string, unknown>).flatMap(([key, item]) =>
    flatten(item, prefix ? `${prefix}.${key}` : key),
  );
}

/**
 * Read-only view of a config this document points at.
 *
 * A run's behaviour lives as much in the file it references as in the file being
 * edited, so seeing those values inline is what catches "this says hoffer but I
 * meant iscen".
 */
export function ReferencedConfig({ path, title }: { path: string; title: string }) {
  const [state, setState] = useState<InspectResult | null>(null);

  useEffect(() => {
    let cancelled = false;
    setState(null);
    studioApi
      .validate<InspectResult>("inspect", { path })
      .then((result) => {
        if (!cancelled) setState(result);
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({ exists: false, error: error instanceof Error ? error.message : String(error) });
        }
      });
    return () => {
      cancelled = true;
    };
  }, [path]);

  if (state === null) {
    return (
      <p className="flex items-center gap-2 text-xs text-[#8f9bb2]">
        <Loader2 className="size-3.5 animate-spin" />
        Reading {path}…
      </p>
    );
  }

  if (!state.exists) {
    return (
      <p className="flex items-center gap-2 text-xs text-[#ff9a9a]">
        <FileWarning className="size-3.5" />
        {path} does not exist.
      </p>
    );
  }

  const rows = flatten(state.data).filter(([key]) => key);

  return (
    <div>
      <p className="studio-label mb-2">{title}</p>
      <div className="studio-scroll max-h-64 overflow-y-auto">
        <table className="w-full text-left text-[0.7rem]">
          <tbody>
            {rows.map(([key, value]) => (
              <tr key={key} className="border-b border-white/[0.05]">
                <td className="py-1 pr-3 align-top font-mono text-[#8f9bb2]">{key}</td>
                <td className="py-1 align-top font-mono text-[#d9e0ed]">{JSON.stringify(value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
