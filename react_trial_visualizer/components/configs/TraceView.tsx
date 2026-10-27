"use client";

import { useMemo, useState } from "react";

import {
  KIND_LABEL,
  PHASE_LABEL,
  compareEntries,
  datasetLabel,
  formatNumber,
  formatSpace,
  lossLabel,
  methodLabel,
  replayCheck,
  sortDatasets,
  sortLosses,
  sortMethods,
  spaceKey,
  type ConfigEntry,
} from "@/lib/study-configs";
import type { Filters } from "@/components/configs/filters";
import { useTableState } from "@/components/configs/CompareView";
import { ColumnHeader, SettingsTable, TableOptions, type Reference } from "@/components/configs/SettingsTable";
import { Empty, Note, Select, Toggle } from "@/components/configs/ui";

type Phase2Scope = "core" | "all" | "none";

/** The Phase-2 folders that replay the tuned config itself rather than change it. */
function isCorePhase2(entry: ConfigEntry): boolean {
  return entry.kind === "geometry" || entry.kind === "replay" || (entry.kind === "2x2" && entry.variant.endsWith("defaults"));
}

function phaseBand(entry: ConfigEntry): string {
  if (entry.phase === "phase2") return `Phase 2 · ${entry.root}${entry.kind === "geometry" ? "" : ` · ${KIND_LABEL[entry.kind ?? "other"]}`}`;
  return PHASE_LABEL[entry.phase];
}

export function TraceView({
  entries,
  studies,
  filters,
  setFilters,
}: {
  entries: ConfigEntry[];
  studies: ConfigEntry[];
  filters: Filters;
  setFilters: (next: Partial<Filters>) => void;
}) {
  const table = useTableState();
  const [phase2Scope, setPhase2Scope] = useState<Phase2Scope>("core");
  const [referenceState, setReference] = useState<{ scope: string; reference: Reference | null }>({ scope: "", reference: null });

  const datasets = sortDatasets(entries.map((entry) => entry.dataset));
  const dataset = datasets.includes(filters.dataset) ? filters.dataset : datasets[0] ?? "";
  const methods = sortMethods(entries.filter((entry) => entry.dataset === dataset).map((entry) => entry.method));
  const method = methods.includes(filters.method) ? filters.method : methods[0] ?? "";
  const losses = sortLosses(entries.filter((entry) => entry.dataset === dataset && entry.method === method).map((entry) => entry.loss));
  const loss = losses.includes(filters.loss) ? filters.loss : losses[0] ?? "";

  const columns = useMemo(
    () =>
      entries
        .filter((entry) => entry.dataset === dataset && entry.method === method && entry.loss === loss)
        .filter((entry) => entry.phase !== "phase2" || phase2Scope === "all" || (phase2Scope === "core" && isCorePhase2(entry)))
        .sort(compareEntries),
    [entries, dataset, method, loss, phase2Scope],
  );
  const phase2Total = entries.filter((entry) => entry.phase === "phase2" && entry.dataset === dataset && entry.method === method && entry.loss === loss).length;

  const scope = `${dataset}|${method}|${loss}|${phase2Scope}`;
  // The study the replays took their trial from, else the linked Phase-1 study.
  const replayed = columns
    .filter((entry) => entry.phase === "seeded" || entry.phase === "phase2")
    .map((entry) => replayCheck(studies, entry)?.source?.id)
    .find((id) => id && columns.some((entry) => entry.id === id));
  const defaultReference = replayed ?? columns.find((entry) => entry.phase === "phase1" && entry.archive === "main" && !entry.variant)?.id ?? "mode";
  const reference = referenceState.scope === scope && referenceState.reference ? referenceState.reference : defaultReference;

  const hpo = columns.filter((entry) => entry.space && Object.keys(entry.space).length > 0);

  return (
    <div className="mt-6 space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <Select label="Dataset" value={dataset} options={datasets} onChange={(value) => setFilters({ dataset: value })} render={datasetLabel} />
        <Select label="Method" value={method} options={methods} onChange={(value) => setFilters({ method: value })} render={methodLabel} />
        <Select label="Loss" value={loss} options={losses} onChange={(value) => setFilters({ loss: value })} render={lossLabel} />
        <Toggle<Phase2Scope>
          label={`Phase 2 (${phase2Total})`}
          value={phase2Scope}
          onChange={setPhase2Scope}
          options={[
            { value: "core", label: "Defaults & geometry", hint: "Plain replays, the 2×2 defaults cell and geometry replays" },
            { value: "all", label: "All folders" },
            { value: "none", label: "None" },
          ]}
        />
      </div>
      <TableOptions {...table} />
      <Note>
        Everything that ran for {datasetLabel(dataset)} · {methodLabel(method)} · {lossLabel(loss)}, from the first random search to the
        Phase-2 replays. Cells are compared with the{" "}
        {reference === "mode" ? "row's most common value" : "reference column (by default the Phase-1 study the replays should take their trial from)"};
        for the HPO columns, tuned settings show the best trial&apos;s value, so a replay that matches it used that trial.
      </Note>
      <SettingsTable
        entries={columns}
        studies={studies}
        reference={reference}
        onReference={(next) => setReference({ scope, reference: next })}
        onlyDifferences={table.onlyDifferences}
        sections={table.sections}
        showRuntime={table.showRuntime}
        groupOf={phaseBand}
        empty={<Empty>Nothing ran for this combination.</Empty>}
      />
      {hpo.length > 0 && <SpaceTrace entries={hpo} />}
    </div>
  );
}

/** Search space of each HPO study side by side, and where the best trial landed in it. */
function SpaceTrace({ entries }: { entries: ConfigEntry[] }) {
  const params = [...new Set(entries.flatMap((entry) => Object.keys(entry.space ?? {})))].sort();
  return (
    <section className="surface overflow-hidden">
      <div className="border-b border-white/[0.06] px-4 py-3">
        <h2 className="text-sm font-semibold text-white">Search spaces</h2>
        <p className="mt-0.5 text-[11px] text-[#7f8ba2]">Range per study and its best trial&apos;s value; a range that changed from the study before is shaded.</p>
      </div>
      <div className="overflow-x-auto studio-scroll">
        <table className="w-max min-w-full border-separate border-spacing-0 text-left">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 min-w-[15rem] bg-[#111a2d] px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">
                Parameter
              </th>
              {entries.map((entry) => (
                <th key={entry.id} className="min-w-[11rem] border-l border-white/[0.06] bg-[#111a2d] px-2 py-2 align-bottom">
                  <div className="text-[9.5px] font-semibold uppercase tracking-[0.08em] text-[#6f7c92]">{PHASE_LABEL[entry.phase]}</div>
                  <ColumnHeader entry={entry} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {params.map((param) => (
              <tr key={param}>
                <th scope="row" className="sticky left-0 z-10 border-t border-white/[0.04] bg-[#0e1626] px-3 py-1.5 font-mono text-[11px] font-normal text-[#c8d2e4]">
                  {param}
                </th>
                {entries.map((entry, index) => {
                  const spec = entry.space?.[param];
                  const previous = index > 0 ? entries[index - 1].space?.[param] : undefined;
                  const changed = index > 0 && spaceKey(spec) !== spaceKey(previous);
                  const best = entry.hpo?.best?.params[param];
                  return (
                    <td key={entry.id} className={`border-l border-t border-white/[0.04] px-2 py-1.5 ${changed ? "bg-[#f4c86f]/[0.15] shadow-[inset_2px_0_0_#f4c86f]" : ""}`}>
                      {spec ? (
                        <>
                          <div className="font-mono text-[11px] text-[#e4e9f2]">{formatSpace(spec)}</div>
                          {best !== undefined && (
                            <div className="font-mono text-[10px] text-[#8895aa]">best {typeof best === "number" ? formatNumber(best) : String(best)}</div>
                          )}
                        </>
                      ) : (
                        <span className="text-[11px] text-[#56627a]">not searched</span>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
