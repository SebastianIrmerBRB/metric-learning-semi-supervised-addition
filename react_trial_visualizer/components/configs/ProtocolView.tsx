"use client";

import { useMemo, useState } from "react";

import {
  PHASES,
  PHASE_HINT,
  PHASE_LABEL,
  datasetLabel,
  formatDuration,
  formatNumber,
  formatValue,
  isHpo,
  isTuned,
  modeOf,
  progressOf,
  sortDatasets,
  sortMethods,
  type ConfigEntry,
  type ConfigPhase,
  type SettingValue,
} from "@/lib/study-configs";
import { ALL, matchesPhase2, type Filters } from "@/components/configs/filters";
import { Phase2Filters } from "@/components/configs/CompareView";
import { Empty, MethodName, Note, Toggle } from "@/components/configs/ui";

interface Field {
  id: string;
  label: string;
  short: string;
  /** Configured fields are compared with the dataset's majority; measured ones are summed or ranged. */
  kind: "config" | "measured";
  get?: (entry: ConfigEntry) => SettingValue | undefined;
}

function setting(key: string) {
  return (entry: ConfigEntry) => (key in entry.settings && !isTuned(entry, key) ? entry.settings[key] : undefined);
}

function labelBudget(entry: ConfigEntry): SettingValue | undefined {
  const k = entry.settings["ssl.labeled_per_class"];
  const fraction = entry.settings["ssl.labeled_fraction"];
  if (entry.phase === "seeded" || entry.phase === "phase2") {
    const budgets = [...new Set(entry.replay?.runs.map((run) => run.budget) ?? [])].sort((a, b) => parseFloat(a.replace(/\D+/, "")) - parseFloat(b.replace(/\D+/, "")));
    if (budgets.length) return budgets.join(" ");
  }
  if (typeof fraction === "number" && fraction < 1) return `${formatNumber(fraction * 100)}%`;
  if (k != null) return `k=${formatValue(k)}`;
  return undefined;
}

const FIELDS: Field[] = [
  { id: "epochs", label: "Epochs", short: "ep", kind: "config", get: setting("epochs") },
  { id: "patience", label: "Patience", short: "pat", kind: "config", get: setting("patience") },
  { id: "cv_k", label: "Folds", short: "folds", kind: "config", get: setting("cv_k") },
  {
    id: "hpo",
    label: "Sampler · trials",
    short: "hpo",
    kind: "config",
    get: (entry) => (isHpo(entry) && entry.settings["hpo.sampler"] != null ? `${entry.settings["hpo.sampler"]} ${formatValue(entry.settings["hpo.n_trials"])}` : undefined),
  },
  { id: "budget", label: "Label budget", short: "labels", kind: "config", get: labelBudget },
  { id: "cap", label: "Regularizer cap", short: "cap", kind: "config", get: setting("ssl.method_params.regularizer_max_ratio") },
  { id: "val_mode", label: "Validation mode", short: "val", kind: "config", get: setting("val_mode") },
  { id: "final_fit_mode", label: "Final fit", short: "fit", kind: "config", get: setting("final_fit_mode") },
  { id: "warmup", label: "SSL warm-up", short: "warm", kind: "config", get: setting("ssl.warmup_epochs") },
  { id: "progress", label: "Finished", short: "done", kind: "measured" },
  { id: "wall", label: "Wall time", short: "wall", kind: "measured" },
  { id: "trained", label: "Epochs trained", short: "trained", kind: "measured" },
  { id: "capHits", label: "Folds at epoch cap", short: "at cap", kind: "measured" },
];
const DEFAULT_FIELDS = ["epochs", "patience", "cv_k", "hpo", "progress", "trained"];

function valueText(value: SettingValue | undefined): string {
  return value === undefined ? "–" : formatValue(value);
}

function Measured({ field, entries }: { field: Field; entries: ConfigEntry[] }) {
  if (field.id === "progress") {
    let done = 0;
    let expected = 0;
    for (const entry of entries) {
      const progress = progressOf(entry);
      done += progress.done;
      expected += progress.expected ?? progress.done;
    }
    const complete = done >= expected;
    return <span className={complete ? "text-[#c7ef85]" : "text-[#7fe3f0]"}>{`${done}/${expected}${complete ? " ✓" : ""}`}</span>;
  }
  const runtimes = entries.map((entry) => entry.runtime).filter((runtime) => runtime != null);
  if (runtimes.length === 0) return <span className="text-[#56627a]">–</span>;
  if (field.id === "wall") {
    const values = runtimes.map((runtime) => runtime.spanSeconds).filter((value): value is number => value != null);
    if (!values.length) return <span className="text-[#56627a]">–</span>;
    const low = Math.min(...values);
    const high = Math.max(...values);
    return <span>{low === high ? formatDuration(low) : `${formatDuration(low)}–${formatDuration(high)}`}</span>;
  }
  if (field.id === "trained") {
    const values = runtimes.map((runtime) => runtime.lastEpoch?.median).filter((value): value is number => value != null);
    if (!values.length) return <span className="text-[#56627a]">–</span>;
    const low = Math.min(...values);
    const high = Math.max(...values);
    return <span title="Median last epoch per study / folder">{low === high ? formatNumber(low) : `${formatNumber(low)}–${formatNumber(high)}`}</span>;
  }
  const hits = runtimes.reduce((sum, runtime) => sum + (runtime.capHits ?? 0), 0);
  const folds = runtimes.reduce((sum, runtime) => sum + runtime.folds, 0);
  return <span className={hits > 0 ? "text-[#f6d488]" : undefined}>{`${hits} of ${folds}`}</span>;
}

function ConfigValues({ field, entries, mode }: { field: Field; entries: ConfigEntry[]; mode: string | null }) {
  const groups = new Map<string, number>();
  for (const entry of entries) {
    const text = valueText(field.get?.(entry));
    groups.set(text, (groups.get(text) ?? 0) + 1);
  }
  const items = [...groups.entries()].sort((left, right) => right[1] - left[1]);
  return (
    <span className="inline-flex flex-wrap gap-x-1.5">
      {items.map(([text, count]) => {
        const deviates = mode != null && text !== "–" && text !== mode;
        return (
          <span key={text} className={deviates ? "rounded bg-[#f4c86f]/15 px-1 text-[#f6d488]" : undefined} title={deviates ? `Most of this dataset: ${mode}` : undefined}>
            {text}
            {items.length > 1 && <span className="text-[#6f7c92]"> ×{count}</span>}
          </span>
        );
      })}
    </span>
  );
}

export function ProtocolView({
  entries,
  filters,
  setFilters,
  onOpen,
}: {
  entries: ConfigEntry[];
  filters: Filters;
  setFilters: (next: Partial<Filters>) => void;
  onOpen: (dataset: string, method: string) => void;
}) {
  const [fieldIds, setFieldIds] = useState<string[]>(DEFAULT_FIELDS);
  const fields = FIELDS.filter((field) => fieldIds.includes(field.id));
  const inPhase = useMemo(
    () => entries.filter((entry) => entry.phase === filters.phase && entry.config && matchesPhase2(entry, { ...filters, root: ALL })),
    [entries, filters],
  );
  const datasets = sortDatasets(inPhase.map((entry) => entry.dataset));
  const methods = sortMethods(inPhase.map((entry) => entry.method));

  /** The dataset's majority per configured field, over every folder of this phase. */
  const modes = useMemo(() => {
    const result = new Map<string, string | null>();
    for (const dataset of datasets) {
      const group = inPhase.filter((entry) => entry.dataset === dataset);
      for (const field of FIELDS) {
        if (field.kind !== "config") continue;
        const values = group.map((entry) => field.get?.(entry)).filter((value) => value !== undefined);
        const mode = modeOf(values);
        result.set(`${dataset}|${field.id}`, values.length ? valueText(mode.value) : null);
      }
    }
    return result;
  }, [datasets, inPhase]);

  return (
    <div className="mt-6 space-y-4">
      <div className="flex flex-wrap items-end gap-3">
        <Toggle<ConfigPhase>
          label="Phase"
          value={filters.phase}
          onChange={(phase) => setFilters({ phase })}
          options={PHASES.map((value) => ({ value, label: PHASE_LABEL[value], hint: PHASE_HINT[value] }))}
        />
        {filters.phase === "phase2" && <Phase2Filters entries={entries} filters={{ ...filters, root: ALL }} setFilters={setFilters} showRoot={false} />}
      </div>
      <div className="flex flex-wrap items-center gap-1 text-[10.5px]" role="group" aria-label="Fields">
        <span className="mr-1 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">Show</span>
        {FIELDS.map((field) => {
          const active = fieldIds.includes(field.id);
          return (
            <button
              key={field.id}
              type="button"
              aria-pressed={active}
              onClick={() => setFieldIds((current) => (active ? current.filter((id) => id !== field.id) : [...current, field.id]))}
              className={`focus-ring rounded-full border px-2 py-0.5 text-[10.5px] font-semibold transition ${
                active
                  ? field.kind === "config"
                    ? "border-[#9b8cff]/35 bg-[#9b8cff]/12 text-[#d8d2ff]"
                    : "border-[#4fd5e7]/30 bg-[#4fd5e7]/10 text-[#9beaf3]"
                  : "border-white/[0.08] text-[#6f7c92] hover:text-[#c8d2e4]"
              }`}
            >
              {field.label}
            </button>
          );
        })}
      </div>
      <Note>
        One cell per dataset and method, summarising all its losses and variants. Configured values (violet chips) that differ from
        what most of that dataset used are shaded; measured values (cyan chips) are ranges or sums. The first column is the dataset&apos;s
        majority, so reading it top to bottom compares datasets. Click a cell to open it in Compare.
      </Note>
      {datasets.length === 0 ? (
        <Empty>No folders in this phase.</Empty>
      ) : (
        <div className="surface overflow-x-auto studio-scroll">
          <table className="w-full min-w-max border-separate border-spacing-0 text-left">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 min-w-[13rem] bg-[#111a2d] px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">
                  Dataset · majority
                </th>
                {methods.map((method) => (
                  <th key={method} className="min-w-[10rem] border-l border-white/[0.06] bg-[#111a2d] px-3 py-2">
                    <MethodName method={method} className="text-xs font-semibold text-white" />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {datasets.map((dataset) => (
                <tr key={dataset}>
                  <th scope="row" className="sticky left-0 z-10 border-t border-white/[0.06] bg-[#0e1626] px-3 py-2.5 align-top font-normal">
                    <div className="text-[12.5px] font-semibold text-white">{datasetLabel(dataset)}</div>
                    <dl className="mt-1 space-y-0.5 text-[11px]">
                      {fields
                        .filter((field) => field.kind === "config")
                        .map((field) => (
                          <div key={field.id} className="flex gap-2">
                            <dt className="w-[4.5rem] shrink-0 text-[#6f7c92]">{field.short}</dt>
                            <dd className="font-mono text-[#c8d2e4]">{modes.get(`${dataset}|${field.id}`) ?? "–"}</dd>
                          </div>
                        ))}
                    </dl>
                  </th>
                  {methods.map((method) => {
                    const cell = inPhase.filter((entry) => entry.dataset === dataset && entry.method === method);
                    if (cell.length === 0) {
                      return (
                        <td key={method} className="border-l border-t border-white/[0.06] px-3 py-2.5 align-top text-[11px] text-[#3f4a5e]">
                          —
                        </td>
                      );
                    }
                    return (
                      <td key={method} className="border-l border-t border-white/[0.06] p-0 align-top">
                        <button
                          type="button"
                          onClick={() => onOpen(dataset, method)}
                          className="focus-ring block w-full px-3 py-2.5 text-left hover:bg-white/[0.03]"
                          title={`Open ${cell.length} folder${cell.length === 1 ? "" : "s"} in Compare`}
                        >
                          <div className="mb-1 text-[10px] text-[#6f7c92]">
                            {cell.length} folder{cell.length === 1 ? "" : "s"}
                          </div>
                          <dl className="space-y-0.5 text-[11px]">
                            {fields.map((field) => (
                              <div key={field.id} className="flex gap-2">
                                <dt className="w-[4.5rem] shrink-0 text-[#6f7c92]">{field.short}</dt>
                                <dd className="min-w-0 font-mono tabular-nums text-[#c8d2e4]">
                                  {field.kind === "config" ? (
                                    <ConfigValues field={field} entries={cell} mode={modes.get(`${dataset}|${field.id}`) ?? null} />
                                  ) : (
                                    <Measured field={field} entries={cell} />
                                  )}
                                </dd>
                              </div>
                            ))}
                          </dl>
                        </button>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
