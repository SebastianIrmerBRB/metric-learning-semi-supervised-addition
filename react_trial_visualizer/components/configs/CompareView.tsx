"use client";

import { useMemo, useState } from "react";

import {
  KIND_LABEL,
  KIND_ORDER,
  PHASES,
  PHASE_HINT,
  PHASE_LABEL,
  compareEntries,
  datasetLabel,
  lossLabel,
  methodLabel,
  sortDatasets,
  sortLosses,
  sortMethods,
  type ConfigEntry,
  type ConfigPhase,
  type Phase2Kind,
  type SectionId,
} from "@/lib/study-configs";
import { ALL, DEFAULT_SECTIONS, matchesPhase2, type Filters, type P2Group } from "@/components/configs/filters";
import { SettingsTable, TableOptions, type Reference } from "@/components/configs/SettingsTable";
import { Empty, Note, Select, Toggle } from "@/components/configs/ui";

const MAX_COLUMNS = 60;

export function useTableState() {
  const [onlyDifferences, setOnlyDifferences] = useState(true);
  const [sections, setSections] = useState<Set<SectionId>>(() => new Set(DEFAULT_SECTIONS));
  const [showRuntime, setShowRuntime] = useState(true);
  return { onlyDifferences, setOnlyDifferences, sections, setSections, showRuntime, setShowRuntime };
}

export function Phase2Filters({
  entries,
  filters,
  setFilters,
  showRoot = true,
}: {
  entries: ConfigEntry[];
  filters: Filters;
  setFilters: (next: Partial<Filters>) => void;
  showRoot?: boolean;
}) {
  const phase2 = entries.filter((entry) => entry.phase === "phase2");
  const roots = [...new Set(phase2.map((entry) => entry.root ?? ""))].sort();
  const kinds = KIND_ORDER.filter((kind) => phase2.some((entry) => entry.kind === kind && matchesPhase2(entry, { ...filters, kind: ALL })));
  return (
    <>
      <Toggle<P2Group>
        label="Phase 2"
        value={filters.p2group}
        onChange={(p2group) => setFilters({ p2group, kind: ALL })}
        options={[
          { value: "all", label: "All" },
          { value: "ablation", label: "Ablations", hint: "2×2, controls, sweeps, HP path, external, OOD, weight decay" },
          { value: "geometry", label: "Geometry", hint: "Replays of the tuned config with geometry tracking (B6)" },
        ]}
      />
      <Select<Phase2Kind | typeof ALL>
        label="Kind"
        value={kinds.includes(filters.kind as Phase2Kind) ? filters.kind : ALL}
        options={[ALL, ...kinds]}
        onChange={(kind) => setFilters({ kind })}
        render={(value) => (value === ALL ? "All kinds" : KIND_LABEL[value])}
      />
      {showRoot && (
        <Select
          label="Save root"
          value={roots.includes(filters.root) ? filters.root : ALL}
          options={[ALL, ...roots]}
          onChange={(root) => setFilters({ root })}
          render={(value) => (value === ALL ? "All roots" : `Phase_2/${value}`)}
        />
      )}
    </>
  );
}

export function CompareView({
  entries,
  studies,
  filters,
  setFilters,
  highlightKey,
}: {
  entries: ConfigEntry[];
  studies: ConfigEntry[];
  filters: Filters;
  setFilters: (next: Partial<Filters>) => void;
  highlightKey: string | null;
}) {
  const table = useTableState();
  const [referenceState, setReference] = useState<{ scope: string; reference: Reference }>({ scope: "", reference: "mode" });

  const inPhase = useMemo(
    () => entries.filter((entry) => entry.phase === filters.phase && matchesPhase2(entry, filters)),
    [entries, filters],
  );
  const datasets = sortDatasets(inPhase.map((entry) => entry.dataset));
  const dataset = datasets.includes(filters.dataset) ? filters.dataset : ALL;
  const inDataset = inPhase.filter((entry) => dataset === ALL || entry.dataset === dataset);
  const methods = sortMethods(inDataset.map((entry) => entry.method));
  const method = methods.includes(filters.method) ? filters.method : ALL;
  const inMethod = inDataset.filter((entry) => method === ALL || entry.method === method);
  const losses = sortLosses(inMethod.map((entry) => entry.loss));
  const loss = losses.includes(filters.loss) ? filters.loss : ALL;
  const columns = useMemo(
    () => inMethod.filter((entry) => loss === ALL || entry.loss === loss).sort(compareEntries),
    [inMethod, loss],
  );
  const shown = columns.slice(0, MAX_COLUMNS);

  const scope = `${filters.phase}|${dataset}|${method}|${loss}|${filters.p2group}|${filters.kind}|${filters.root}`;
  const reference = referenceState.scope === scope ? referenceState.reference : "mode";

  const bandBy = useMemo(() => {
    if (new Set(shown.map((entry) => entry.dataset)).size > 1) return (entry: ConfigEntry) => datasetLabel(entry.dataset);
    if (new Set(shown.map((entry) => entry.method)).size > 1) return (entry: ConfigEntry) => methodLabel(entry.method);
    if (filters.phase === "phase2") return (entry: ConfigEntry) => `${entry.root} · ${KIND_LABEL[entry.kind ?? "other"]}`;
    return undefined;
  }, [shown, filters.phase]);

  return (
    <div className="mt-6 space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <Toggle<ConfigPhase>
          label="Phase"
          value={filters.phase}
          onChange={(phase) => setFilters({ phase })}
          options={PHASES.map((value) => ({ value, label: PHASE_LABEL[value], hint: PHASE_HINT[value] }))}
        />
        <Select label="Dataset" value={dataset} options={[ALL, ...datasets]} onChange={(value) => setFilters({ dataset: value })} render={(value) => (value === ALL ? "All datasets" : datasetLabel(value))} />
        <Select label="Method" value={method} options={[ALL, ...methods]} onChange={(value) => setFilters({ method: value })} render={(value) => (value === ALL ? "All methods" : methodLabel(value))} />
        <Select label="Loss" value={loss} options={[ALL, ...losses]} onChange={(value) => setFilters({ loss: value })} render={(value) => (value === ALL ? "All losses" : lossLabel(value))} />
        {filters.phase === "phase2" && <Phase2Filters entries={entries} filters={filters} setFilters={setFilters} />}
      </div>
      <TableOptions {...table} />
      <Note>
        {columns.length} folder{columns.length === 1 ? "" : "s"}. A shaded cell differs from{" "}
        {reference === "mode" ? "its row's most common value" : "the reference column"}; red shading means the runs inside that folder
        disagree. <span className="text-[#c9c1ff]">Tuned</span> settings show the best trial&apos;s value and are only flagged against a
        chosen reference column.
        {columns.length > MAX_COLUMNS && ` Showing the first ${MAX_COLUMNS}; narrow the selection to see the rest.`}
      </Note>
      <SettingsTable
        entries={shown}
        studies={studies}
        reference={reference}
        onReference={(next) => setReference({ scope, reference: next })}
        onlyDifferences={table.onlyDifferences}
        sections={table.sections}
        showRuntime={table.showRuntime}
        highlightKey={highlightKey}
        groupOf={bandBy}
        showDataset={dataset === ALL}
        empty={<Empty>No folders match this selection.</Empty>}
      />
    </div>
  );
}
