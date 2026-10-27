"use client";

import { Fragment, useMemo, useState } from "react";

import {
  KIND_GROUP,
  KIND_LABEL,
  KIND_ORDER,
  LOSS_SHORT,
  PHASE_LABEL,
  datasetLabel,
  isHpo,
  lossLabel,
  methodLabel,
  missingRuns,
  progressOf,
  resolvePlanned,
  sortDatasets,
  sortLosses,
  sortMethods,
  type ConfigEntry,
  type ConfigSnapshot,
  type PlannedFolder,
} from "@/lib/study-configs";
import { ALL, type Filters, type P2Group } from "@/components/configs/filters";
import { Empty, MethodName, Note, StatusChip, StatusLegend, Toggle, type Status } from "@/components/configs/ui";

export type CoverageMode = "hpo" | "phase2";
type GridPhase = "phase0" | "phase1" | "seeded";

function entryStatus(entry: ConfigEntry): { status: Status; text: string; title: string } {
  if (!entry.config) return { status: "missing", text: "no config", title: entry.notes.join("; ") || "No config file in this folder" };
  const progress = progressOf(entry);
  const complete = progress.expected != null && progress.done >= progress.expected;
  const missing = missingRuns(entry);
  const title = isHpo(entry)
    ? `${entry.hpo?.complete ?? 0} of ${entry.hpo?.planned ?? "?"} trials complete · ${entry.hpo?.failed ?? 0} failed · ${entry.hpo?.running ?? 0} running`
    : `${entry.replay?.complete ?? 0} complete, ${entry.replay?.partial ?? 0} partial, ${entry.replay?.started ?? 0} started${
        missing.length ? ` · not started: ${missing.slice(0, 6).join(", ")}${missing.length > 6 ? ", …" : ""}` : ""
      }`;
  return {
    status: complete ? "ok" : progress.done > 0 ? "partial" : "warn",
    text: `${progress.done}/${progress.expected ?? "?"}`,
    title: `${entry.id}\n${title}`,
  };
}

export function CoverageView({
  snapshot,
  entries,
  filters,
  setFilters,
  mode,
  setMode,
}: {
  snapshot: ConfigSnapshot;
  entries: ConfigEntry[];
  filters: Filters;
  setFilters: (next: Partial<Filters>) => void;
  mode: CoverageMode;
  setMode: (mode: CoverageMode) => void;
}) {
  return (
    <div className="mt-6 space-y-4">
      <Toggle<CoverageMode>
        value={mode}
        onChange={setMode}
        options={[
          { value: "hpo", label: "HPO studies & seeded runs" },
          { value: "phase2", label: "Phase 2: planned vs. present" },
        ]}
      />
      {mode === "hpo" ? (
        <GridCoverage entries={entries} />
      ) : (
        <Phase2Coverage snapshot={snapshot} entries={entries} filters={filters} setFilters={setFilters} />
      )}
    </div>
  );
}

function GridCoverage({ entries }: { entries: ConfigEntry[] }) {
  const [phase, setPhase] = useState<GridPhase>("phase1");
  const grid = entries.filter((entry) => entry.phase === "phase0" || entry.phase === "phase1" || entry.phase === "seeded");
  const datasets = sortDatasets(grid.map((entry) => entry.dataset));
  const losses = sortLosses(grid.map((entry) => entry.loss));
  const inPhase = grid.filter((entry) => entry.phase === phase);

  return (
    <>
      <div className="flex flex-wrap items-end justify-between gap-3">
        <Toggle<GridPhase>
          label="Phase"
          value={phase}
          onChange={setPhase}
          options={[
            { value: "phase0", label: PHASE_LABEL.phase0 },
            { value: "phase1", label: PHASE_LABEL.phase1 },
            { value: "seeded", label: PHASE_LABEL.seeded },
          ]}
        />
        <StatusLegend
          items={[
            { status: "ok", label: "complete" },
            { status: "partial", label: "unfinished" },
            { status: "missing", label: "no config" },
            { status: "none", label: "no folder" },
          ]}
        />
      </div>
      <Note>
        Rows list every method that ran for a dataset in any of Phase 0, Phase 1 or the seeded runs, so a gap here is a combination another
        phase has. Numbers are finished / planned trials (HPO) or complete / planned runs (seeded; planned from the folder&apos;s{" "}
        <code className="text-[#c8d2e4]">study_replay_grid/ssl_configs</code>). Hover for details.
      </Note>
      <div className="surface overflow-x-auto studio-scroll">
        <table className="w-full min-w-max border-separate border-spacing-0 text-left">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 min-w-[11rem] bg-[#111a2d] px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">
                Method
              </th>
              {losses.map((loss) => (
                <th key={loss} className="min-w-[8.5rem] border-l border-white/[0.06] bg-[#111a2d] px-3 py-2 text-xs font-semibold text-white">
                  {lossLabel(loss)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {datasets.map((dataset) => {
              const methods = sortMethods(grid.filter((entry) => entry.dataset === dataset).map((entry) => entry.method));
              const present = inPhase.filter((entry) => entry.dataset === dataset);
              const cells = methods.length * losses.length;
              const filled = new Set(present.map((entry) => `${entry.method}|${entry.loss}`)).size;
              return (
                <Fragment key={dataset}>
                  <tr>
                    <td colSpan={losses.length + 1} className="border-t border-white/[0.08] bg-[#0c1322] px-3 pb-1.5 pt-3">
                      <span className="text-[12.5px] font-semibold text-white">{datasetLabel(dataset)}</span>
                      <span className="ml-2 text-[11px] text-[#7f8ba2]">
                        {filled} of {cells} method × loss cells have a folder
                      </span>
                    </td>
                  </tr>
                  {methods.map((method) => (
                    <tr key={method}>
                      <th scope="row" className="sticky left-0 z-10 border-t border-white/[0.04] bg-[#0e1626] px-3 py-1.5 font-normal">
                        <MethodName method={method} className="text-[11.5px] font-semibold text-[#e4e9f2]" />
                      </th>
                      {losses.map((loss) => {
                        const cell = present.filter((entry) => entry.method === method && entry.loss === loss);
                        return (
                          <td key={loss} className="border-l border-t border-white/[0.04] px-2 py-1.5 align-top">
                            {cell.length === 0 ? (
                              <StatusChip status="none" title={`No ${PHASE_LABEL[phase]} folder for ${datasetLabel(dataset)} · ${methodLabel(method)} · ${lossLabel(loss)}`}>
                                —
                              </StatusChip>
                            ) : (
                              <div className="flex flex-col items-start gap-1">
                                {cell.map((entry) => {
                                  const status = entryStatus(entry);
                                  return (
                                    <StatusChip key={entry.id} status={status.status} title={status.title}>
                                      {entry.variant && <span className="max-w-[7rem] truncate font-normal opacity-80">{entry.variant}</span>}
                                      {status.text}
                                    </StatusChip>
                                  );
                                })}
                              </div>
                            )}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
    </>
  );
}

interface Phase2Cell {
  status: Status;
  text: string;
  title: string;
}

function comboOf(method: string | null, loss: string | null): string {
  return `${method ?? "?"}|${loss ?? "?"}`;
}

/** `hoffer_cir_controls_no_warmup` → `controls_no_warmup`. */
const COMBO_VARIANT = /^(?:sup|iscen|lrml|hoffer|slade)_(?:arc|cir|ms|nt|prx)_(.+)$/;

function rootGroup(root: string): P2Group {
  return root.endsWith("_geometry") ? "geometry" : "ablation";
}

function Phase2Coverage({
  snapshot,
  entries,
  filters,
  setFilters,
}: {
  snapshot: ConfigSnapshot;
  entries: ConfigEntry[];
  filters: Filters;
  setFilters: (next: Partial<Filters>) => void;
}) {
  const phase2 = useMemo(() => entries.filter((entry) => entry.phase === "phase2"), [entries]);
  const present = useMemo(() => new Set(phase2.map((entry) => entry.id)), [phase2]);

  const roots = useMemo(() => {
    const all = new Set([...phase2.map((entry) => entry.root ?? ""), ...snapshot.phase2Plan.map((folder) => folder.root)]);
    return [...all]
      .filter((root) => filters.p2group === "all" || rootGroup(root) === filters.p2group)
      .sort((a, b) => a.localeCompare(b))
      .map((root) => {
        const planned = snapshot.phase2Plan.filter((folder) => folder.root === root && folder.enabled);
        const found = planned.filter((folder) => resolvePlanned(folder, present));
        return { root, planned: planned.length, found: found.length, folders: phase2.filter((entry) => entry.root === root).length };
      });
  }, [phase2, snapshot.phase2Plan, present, filters.p2group]);

  const root = roots.some((item) => item.root === filters.root) ? filters.root : roots[0]?.root ?? "";
  const plan = snapshot.phase2Plan.filter((folder) => folder.root === root);
  const inRoot = phase2.filter((entry) => entry.root === root);
  // Every folder on F: that fulfils a plan, by its own id (an exact match or a renamed variant).
  const planByPresentId = new Map<string, PlannedFolder>();
  for (const folder of plan) {
    const resolved = folder.enabled ? resolvePlanned(folder, present) : present.has(folder.id) ? folder.id : null;
    if (resolved && !planByPresentId.has(resolved)) planByPresentId.set(resolved, folder);
  }

  const combos = useMemo(() => {
    const pairs = new Map<string, { method: string; loss: string }>();
    for (const item of [...plan, ...inRoot]) if (item.method && item.loss) pairs.set(comboOf(item.method, item.loss), { method: item.method, loss: item.loss });
    const methods = sortMethods([...pairs.values()].map((pair) => pair.method));
    const losses = sortLosses([...pairs.values()].map((pair) => pair.loss));
    return methods.flatMap((method) => losses.filter((loss) => pairs.has(comboOf(method, loss))).map((loss) => ({ method, loss })));
  }, [plan, inRoot]);

  const variants = useMemo(() => {
    const byVariant = new Map<string, { variant: string; kind: (typeof KIND_ORDER)[number] }>();
    for (const item of [...plan, ...inRoot]) {
      const kind = item.kind ?? "other";
      if (!byVariant.has(item.variant)) byVariant.set(item.variant, { variant: item.variant, kind });
    }
    return [...byVariant.values()].sort((a, b) => KIND_ORDER.indexOf(a.kind) - KIND_ORDER.indexOf(b.kind) || a.variant.localeCompare(b.variant, undefined, { numeric: true }));
  }, [plan, inRoot]);

  const cellFor = (variant: string, method: string, loss: string): Phase2Cell | null => {
    const entry = inRoot.find((item) => item.variant === variant && item.method === method && item.loss === loss) ?? null;
    const folder =
      plan.find((item) => item.variant === variant && item.method === method && item.loss === loss) ??
      (entry ? planByPresentId.get(entry.id) ?? null : null);
    if (entry) {
      const expected = new Set([...(folder?.enabled ? folder.runs : []), ...(entry.replay?.planned ?? [])]);
      const runs = entry.replay?.runs ?? [];
      const complete = runs.filter((run) => run.status === "complete" && (expected.size === 0 || expected.has(run.name))).length;
      const total = expected.size || runs.length;
      const notStarted = [...expected].filter((name) => !runs.some((run) => run.name === name));
      const title = [
        entry.id,
        `${complete} of ${total} runs complete${entry.replay?.partial ? `, ${entry.replay.partial} partial` : ""}`,
        notStarted.length ? `not started: ${notStarted.slice(0, 6).join(", ")}${notStarted.length > 6 ? ", …" : ""}` : "",
        folder
          ? `planned by ${folder.manifests.join(", ")}${folder.id !== entry.id ? ` as ${folder.id.split("/").pop()}` : ""}`
          : "in no manifest under configs/run_manifests/phase2",
      ]
        .filter(Boolean)
        .join("\n");
      if (!entry.config) return { status: "missing", text: "no config", title };
      if (!folder) return { status: "extra", text: `${complete}/${total}`, title };
      return { status: complete >= total ? "ok" : complete > 0 || runs.length > 0 ? "partial" : "warn", text: `${complete}/${total}`, title };
    }
    if (!folder) return null;
    const title = `${folder.id}\nplanned by ${folder.manifests.join(", ")}${folder.ablations.length ? `\nablation ${folder.ablations.join(", ")}` : ""}`;
    if (!folder.enabled) return { status: "none", text: "disabled", title: `${title}\nthe run is disabled in its manifest` };
    const resolved = resolvePlanned(folder, present);
    if (resolved) {
      const name = resolved.split("/").pop() ?? resolved;
      const as = COMBO_VARIANT.exec(name)?.[1] ?? "replay";
      return { status: "ok", text: `as ${as}`, title: `${title}\nnot there under this name, but as ${name} (see that row)` };
    }
    return { status: "missing", text: "missing", title: `${title}\n${folder.runs.length} runs planned` };
  };

  const missingFolders: PlannedFolder[] = plan.filter((folder) => folder.enabled && !resolvePlanned(folder, present));
  const unplanned = inRoot.filter((entry) => !planByPresentId.has(entry.id));

  return (
    <>
      <div className="flex flex-wrap items-end gap-3">
        <Toggle<P2Group>
          label="Show"
          value={filters.p2group}
          onChange={(p2group) => setFilters({ p2group, root: ALL })}
          options={[
            { value: "all", label: "All roots" },
            { value: "ablation", label: "Ablations" },
            { value: "geometry", label: "Geometry" },
          ]}
        />
      </div>
      <div className="flex flex-wrap gap-2">
        {roots.map((item) => {
          const active = item.root === root;
          const complete = item.planned > 0 && item.found >= item.planned;
          return (
            <button
              key={item.root}
              type="button"
              aria-pressed={active}
              onClick={() => setFilters({ root: item.root })}
              className={`focus-ring rounded-xl border px-3 py-2 text-left transition ${
                active ? "border-[#9b8cff]/45 bg-[#9b8cff]/10" : "border-white/[0.07] bg-white/[0.015] hover:border-white/[0.15]"
              }`}
            >
              <div className="flex items-center gap-2 text-[12px] font-semibold text-white">
                Phase_2/{item.root}
                <span className="rounded bg-white/[0.06] px-1.5 py-px text-[9.5px] font-semibold uppercase tracking-[0.06em] text-[#aeb9cc]">
                  {rootGroup(item.root) === "geometry" ? "geometry" : "ablations"}
                </span>
              </div>
              <div className={`mt-0.5 text-[11px] tabular-nums ${item.planned === 0 ? "text-[#8895aa]" : complete ? "text-[#c7ef85]" : "text-[#f4b4b4]"}`}>
                {item.planned ? `${item.found} of ${item.planned} planned folders present` : "no manifest"} · {item.folders} on F:
              </div>
            </button>
          );
        })}
      </div>
      <div className="flex flex-wrap items-center justify-between gap-3">
        <Note>
          Planned folders come from <code className="text-[#c8d2e4]">{snapshot.manifestRoot}</code>, expanded like{" "}
          <code className="text-[#c8d2e4]">run_scheduler.py</code> does (<code className="text-[#c8d2e4]">&lt;run&gt;_&lt;variant&gt;</code>, one run per
          budget and seed). Numbers are complete / planned runs.
        </Note>
        <StatusLegend
          items={[
            { status: "ok", label: "complete" },
            { status: "partial", label: "unfinished" },
            { status: "missing", label: "planned, missing" },
            { status: "extra", label: "not in a manifest" },
          ]}
        />
      </div>
      {combos.length === 0 ? (
        <Empty>Nothing planned or present under Phase_2/{root}.</Empty>
      ) : (
        <div className="surface overflow-x-auto studio-scroll">
          <table className="w-max min-w-full border-separate border-spacing-0 text-left">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 min-w-[14rem] bg-[#111a2d] px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">
                  Variant
                </th>
                {combos.map((combo) => (
                  <th key={comboOf(combo.method, combo.loss)} className="min-w-[4.8rem] border-l border-white/[0.06] bg-[#111a2d] px-2 py-2">
                    <MethodName method={combo.method} className="text-[11px] font-semibold text-white" />
                    <div className="text-[10.5px] text-[#aeb9cc]" title={lossLabel(combo.loss)}>
                      {LOSS_SHORT[combo.loss] ?? combo.loss}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {variants.map((row, index) => {
                const newKind = index === 0 || variants[index - 1].kind !== row.kind;
                return (
                  <Fragment key={row.variant}>
                    {newKind && (
                      <tr>
                        <td colSpan={combos.length + 1} className="border-t border-white/[0.08] bg-[#0c1322] px-3 pb-1 pt-2.5">
                          <span className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#b6acff]">{KIND_LABEL[row.kind]}</span>
                          <span className="ml-2 text-[10px] text-[#6f7c92]">{KIND_GROUP[row.kind] === "geometry" ? "geometry" : "ablation"}</span>
                        </td>
                      </tr>
                    )}
                    <tr>
                      <th scope="row" className="sticky left-0 z-10 border-t border-white/[0.04] bg-[#0e1626] px-3 py-1.5 font-mono text-[11px] font-normal text-[#c8d2e4]">
                        {row.variant || "(unmodified replay)"}
                      </th>
                      {combos.map((combo) => {
                        const cell = cellFor(row.variant, combo.method, combo.loss);
                        return (
                          <td key={comboOf(combo.method, combo.loss)} className="border-l border-t border-white/[0.04] px-1.5 py-1">
                            {cell ? (
                              <StatusChip status={cell.status} title={cell.title}>
                                {cell.text}
                              </StatusChip>
                            ) : null}
                          </td>
                        );
                      })}
                    </tr>
                  </Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      {(missingFolders.length > 0 || unplanned.length > 0) && (
        <div className="grid gap-4 lg:grid-cols-2">
          {missingFolders.length > 0 && (
            <FolderList
              title={`Planned but not on F: (${missingFolders.length})`}
              rows={missingFolders.map((folder) => ({ id: folder.id, detail: folder.manifests.join(", ") }))}
            />
          )}
          {unplanned.length > 0 && (
            <FolderList
              title={`On F: but in no manifest (${unplanned.length})`}
              rows={unplanned.map((entry) => ({
                id: entry.id,
                detail: entry.replay?.ablations[0]?.name ? `ablation ${entry.replay.ablations[0].name}` : "no ablation recorded",
              }))}
            />
          )}
        </div>
      )}
    </>
  );
}

function FolderList({ title, rows }: { title: string; rows: { id: string; detail: string }[] }) {
  return (
    <section className="surface-soft p-3">
      <h3 className="text-xs font-semibold text-white">{title}</h3>
      <ul className="mt-2 max-h-72 space-y-1 overflow-auto studio-scroll">
        {rows.map((row) => (
          <li key={row.id} className="text-[11px]">
            <span className="font-mono text-[#e4e9f2]">{row.id}</span>
            <span className="ml-2 text-[#6f7c92]">{row.detail}</span>
          </li>
        ))}
      </ul>
    </section>
  );
}
