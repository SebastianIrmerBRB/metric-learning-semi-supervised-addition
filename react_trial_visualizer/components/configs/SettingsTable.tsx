"use client";

import { Fragment, useMemo, type ReactNode } from "react";
import { Shuffle } from "lucide-react";

import {
  LOSS_SHORT,
  RUNTIME_ROWS,
  SECTIONS,
  compareKeys,
  formatSpace,
  formatValue,
  isHpo,
  isTuned,
  keyLabel,
  lossLabel,
  modeOf,
  paramOfSetting,
  progressOf,
  replayCheck,
  sectionOf,
  valueKey,
  type ConfigEntry,
  type SectionId,
  type SettingValue,
} from "@/lib/study-configs";
import { Checkbox, MethodName, StatusChip } from "@/components/configs/ui";

export type Reference = "mode" | string;

interface Cell {
  present: boolean;
  value: SettingValue | undefined;
  tuned: boolean;
  differs: boolean;
  varying: { value: SettingValue; count: number }[] | null;
}

interface Row {
  key: string;
  section: SectionId;
  cells: Cell[];
  differing: number;
  /** Tuned in some columns, fixed in others: shown as a difference, nothing shaded. */
  mixed: boolean;
}

/**
 * One column per study / series / folder, one row per setting. A cell is
 * flagged when it differs from the reference: the row's most common value by
 * default, or a chosen column. Tuned settings are never flagged against the
 * row's mode (each study found its own value), only when a column is chosen as
 * reference. A setting some columns tune and others fix (often by design, e.g.
 * no SSL warm-up for supervised runs) keeps its row visible without shading.
 */
function buildRows(entries: ConfigEntry[], reference: Reference, sections: Set<SectionId>): Row[] {
  const referenceIndex = reference === "mode" ? -1 : entries.findIndex((entry) => entry.id === reference);
  const keys = new Set<string>();
  for (const entry of entries) for (const key of Object.keys(entry.settings)) if (sections.has(sectionOf(key))) keys.add(key);

  const rows: Row[] = [];
  for (const key of [...keys].sort(compareKeys)) {
    const base = entries.map((entry) => {
      const present = key in entry.settings;
      const varying = entry.replay?.varying[key] ?? null;
      return { present, value: entry.settings[key], tuned: present && isTuned(entry, key), varying };
    });
    const fixed = base.filter((cell) => cell.present && !cell.tuned);
    const anyTuned = base.some((cell) => cell.tuned);
    let cells: Cell[];
    if (referenceIndex >= 0) {
      const ref = base[referenceIndex];
      cells = base.map((cell, index) => ({
        ...cell,
        differs: index !== referenceIndex && cell.present && ref.present && valueKey(cell.value) !== valueKey(ref.value),
      }));
    } else {
      const mode = modeOf(fixed.map((cell) => cell.value));
      cells = base.map((cell) => ({
        ...cell,
        differs: cell.present && !cell.tuned && mode.distinct > 1 && valueKey(cell.value) !== mode.key,
      }));
    }
    cells = cells.map((cell) => (cell.varying ? { ...cell, differs: true } : cell));
    rows.push({
      key,
      section: sectionOf(key),
      cells,
      differing: cells.filter((cell) => cell.differs).length,
      mixed: anyTuned && fixed.length > 0,
    });
  }
  return rows;
}

function ValueText({ value }: { value: SettingValue | undefined }) {
  const text = formatValue(value);
  return (
    <span className="block truncate font-mono text-[11px]" title={text.length > 18 ? text : undefined}>
      {text}
    </span>
  );
}

function SettingCell({ entry, rowKey, cell }: { entry: ConfigEntry; rowKey: string; cell: Cell }) {
  if (!cell.present) {
    return <span className="text-[11px] text-[#3f4a5e]" aria-label="not set">·</span>;
  }
  const param = cell.tuned ? paramOfSetting(entry, rowKey) : null;
  const spec = param ? entry.space?.[param] : undefined;
  const source = entry.replay?.sources[0];
  const ablation = entry.replay?.ablations.find((item) =>
    Object.keys(item.changes).some((change) => change === rowKey || `ssl.${change.replace(/^ssl_config\./, "")}` === rowKey),
  );
  return (
    <div className="flex min-w-0 items-center gap-1.5">
      {cell.tuned && (
        <span
          className="shrink-0 rounded bg-[#9b8cff]/15 px-1 py-px text-[9px] font-bold uppercase tracking-[0.06em] text-[#c9c1ff]"
          title={
            isHpo(entry)
              ? `Tuned: ${param ?? rowKey} ${formatSpace(spec)}. Shown: the best trial's value.`
              : `Set by the replayed trial #${source?.trial ?? "?"} of ${source?.study ?? "?"}.`
          }
        >
          {isHpo(entry) ? "tuned" : "trial"}
        </span>
      )}
      {ablation && (
        <span
          className="shrink-0 rounded bg-[#4fd5e7]/12 px-1 py-px text-[9px] font-bold uppercase tracking-[0.06em] text-[#9beaf3]"
          title={`Set on purpose by ablation ${ablation.name ?? "?"}${ablation.variant ? ` · ${ablation.variant}` : ""}`}
        >
          abl.
        </span>
      )}
      <div className={`min-w-0 flex-1 ${cell.tuned ? "text-[#b3aee6]" : cell.differs ? "text-white" : "text-[#aeb9cc]"}`}>
        <ValueText value={cell.value} />
      </div>
      {cell.varying && (
        <span
          className="shrink-0 text-[#f4b4b4]"
          title={`Runs of this folder disagree: ${cell.varying.map((item) => `${formatValue(item.value)} ×${item.count}`).join(" / ")}`}
        >
          <Shuffle size={11} aria-label="differs between runs" />
        </span>
      )}
    </div>
  );
}

export function ColumnHeader({ entry, showDataset, showPhase }: { entry: ConfigEntry; showDataset?: boolean; showPhase?: boolean }) {
  const variant = entry.phase === "phase2" ? entry.variant || "replay" : entry.variant;
  return (
    <div className="min-w-0 text-left" title={entry.id}>
      {(showDataset || showPhase) && (
        <div className="truncate text-[9.5px] font-semibold uppercase tracking-[0.08em] text-[#6f7c92]">
          {[showPhase ? phaseShort(entry) : null, showDataset ? entry.dataset : null].filter(Boolean).join(" · ")}
        </div>
      )}
      <MethodName method={entry.method} className="text-[11.5px] font-semibold text-white" />
      <div className="truncate text-[11px] text-[#c8d2e4]">
        <span title={lossLabel(entry.loss)}>{LOSS_SHORT[entry.loss] ?? entry.loss}</span>
        {variant && <span className="text-[#8895aa]"> · {variant}</span>}
        {entry.archive === "weitere" && <span className="text-[#6f7c92]"> · _weitere</span>}
      </div>
    </div>
  );
}

function phaseShort(entry: ConfigEntry): string {
  if (entry.phase === "phase0") return "P0";
  if (entry.phase === "phase1") return "P1";
  if (entry.phase === "seeded") return "Seeded";
  return entry.root ? `P2 ${entry.root}` : "P2";
}

function ProgressCell({ entry }: { entry: ConfigEntry }) {
  if (!entry.config) return <StatusChip status="missing">no config</StatusChip>;
  const progress = progressOf(entry);
  const done = progress.expected != null && progress.done >= progress.expected;
  return (
    <StatusChip
      status={done ? "ok" : progress.done > 0 ? "partial" : "warn"}
      title={
        isHpo(entry)
          ? `${entry.hpo?.complete ?? 0} complete, ${entry.hpo?.failed ?? 0} failed, ${entry.hpo?.running ?? 0} running, ${entry.hpo?.pruned ?? 0} pruned`
          : `${entry.replay?.complete ?? 0} complete, ${entry.replay?.partial ?? 0} partial, ${entry.replay?.started ?? 0} started`
      }
    >
      {progress.done}/{progress.expected ?? "?"} {progress.unit}
    </StatusChip>
  );
}

function ReplayCell({ entry, studies }: { entry: ConfigEntry; studies: ConfigEntry[] }) {
  if (isHpo(entry)) {
    const best = entry.hpo?.best;
    return best ? (
      <span className="font-mono text-[11px] text-[#aeb9cc]" title={`Best validation value ${best.value}`}>
        best #{best.number}
      </span>
    ) : (
      <span className="text-[11px] text-[#3f4a5e]">·</span>
    );
  }
  const check = replayCheck(studies, entry);
  if (!check) return <span className="text-[11px] text-[#3f4a5e]">·</span>;
  const trial = entry.replay?.sources[0]?.trial;
  if (check.status === "match") return <StatusChip status="ok" title={check.text}>#{trial} = P1 best</StatusChip>;
  if (check.status === "unknown") return <span className="font-mono text-[11px] text-[#aeb9cc]" title={check.text}>#{trial ?? "?"}</span>;
  if (check.status === "other" && check.source) {
    return (
      <StatusChip status="warn" title={check.text}>
        #{trial} = best of {check.source.archive === "weitere" ? "_weitere" : ""} {check.source.variant || "another study"}
      </StatusChip>
    );
  }
  return (
    <StatusChip status="missing" title={check.differing.length ? `${check.text}: ${check.differing.join(", ")}` : check.text}>
      {check.status === "mixed" ? "mixed trials" : `#${trial ?? "?"} ≠ any P1 best`}
    </StatusChip>
  );
}

export function SettingsTable({
  entries,
  studies,
  reference,
  onReference,
  onlyDifferences,
  sections,
  showRuntime,
  highlightKey,
  groupOf,
  showDataset,
  showPhase,
  empty,
}: {
  entries: ConfigEntry[];
  /** Every entry of the snapshot, _weitere included: replays are matched against all Phase-1 studies. */
  studies: ConfigEntry[];
  reference: Reference;
  onReference: (reference: Reference) => void;
  onlyDifferences: boolean;
  sections: Set<SectionId>;
  showRuntime: boolean;
  highlightKey?: string | null;
  /** Optional band above the column headers (phase in the trace, dataset in compare). */
  groupOf?: (entry: ConfigEntry) => string;
  showDataset?: boolean;
  showPhase?: boolean;
  empty?: ReactNode;
}) {
  const rows = useMemo(() => buildRows(entries, reference, sections), [entries, reference, sections]);
  const visible = onlyDifferences ? rows.filter((row) => row.differing > 0 || row.mixed || row.key === highlightKey) : rows;
  const bands = useMemo(() => {
    if (!groupOf) return null;
    const result: { label: string; span: number }[] = [];
    for (const entry of entries) {
      const label = groupOf(entry);
      const last = result[result.length - 1];
      if (last && last.label === label) last.span += 1;
      else result.push({ label, span: 1 });
    }
    return result.length > 1 ? result : null;
  }, [entries, groupOf]);

  if (entries.length === 0) return <>{empty}</>;

  const columnCount = entries.length + 1;
  const hidden = rows.length - visible.length;

  return (
    <div className="surface overflow-hidden">
      <div className="max-h-[78vh] overflow-auto studio-scroll">
        <table className="w-max min-w-full border-separate border-spacing-0 text-left">
          <thead>
            {bands && (
              <tr>
                <th className="sticky left-0 top-0 z-30 bg-[#111a2d]" />
                {bands.map((band, index) => (
                  <th
                    key={`${band.label}-${index}`}
                    colSpan={band.span}
                    className="sticky top-0 z-20 border-b border-l border-white/[0.06] bg-[#111a2d] px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#8895aa]"
                  >
                    {band.label}
                  </th>
                ))}
              </tr>
            )}
            <tr>
              <th
                className={`sticky left-0 z-30 min-w-[15rem] border-b border-white/[0.08] bg-[#111a2d] px-3 py-2 align-bottom ${bands ? "top-[1.6rem]" : "top-0"}`}
              >
                <div className="flex items-center justify-between gap-2 text-[10px]">
                  <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">Setting</span>
                  <button
                    type="button"
                    onClick={() => onReference("mode")}
                    className={`focus-ring rounded-md px-1.5 py-0.5 text-[10px] font-semibold ${
                      reference === "mode" ? "bg-[#9b8cff]/15 text-[#d8d2ff]" : "text-[#8895aa] hover:text-white"
                    }`}
                    title="Flag cells that differ from the most common value of their row"
                  >
                    vs. majority
                  </button>
                </div>
              </th>
              {entries.map((entry) => {
                const isReference = reference === entry.id;
                return (
                  <th
                    key={entry.id}
                    className={`sticky z-20 w-[9.5rem] min-w-[9.5rem] max-w-[11rem] border-b border-l border-white/[0.06] bg-[#111a2d] px-2 py-2 align-bottom ${
                      bands ? "top-[1.6rem]" : "top-0"
                    } ${isReference ? "shadow-[inset_0_2px_0_#9b8cff]" : ""}`}
                  >
                    <ColumnHeader entry={entry} showDataset={showDataset} showPhase={showPhase} />
                    <div className="text-[9.5px]">
                      <button
                        type="button"
                        onClick={() => onReference(isReference ? "mode" : entry.id)}
                        className={`focus-ring mt-1 rounded px-1 py-px text-[9.5px] font-semibold ${
                          isReference ? "bg-[#9b8cff]/20 text-[#d8d2ff]" : "text-[#6f7c92] hover:text-[#c8d2e4]"
                        }`}
                        title="Compare every column with this one"
                      >
                        {isReference ? "reference" : "set as reference"}
                      </button>
                    </div>
                  </th>
                );
              })}
            </tr>
          </thead>
          <tbody>
            <SectionRow label="Status" hint="Trials or runs finished, and which trial a replay used" span={columnCount} />
            <tr>
              <RowLabel label="Progress" rawKey="finished / planned" />
              {entries.map((entry) => (
                <td key={entry.id} className="border-l border-t border-white/[0.04] px-2 py-1.5">
                  <ProgressCell entry={entry} />
                </td>
              ))}
            </tr>
            <tr>
              <RowLabel label="Trial" rawKey="best trial / replayed trial" />
              {entries.map((entry) => (
                <td key={entry.id} className="border-l border-t border-white/[0.04] px-2 py-1.5">
                  <ReplayCell entry={entry} studies={studies} />
                </td>
              ))}
            </tr>

            {showRuntime && (
              <>
                <SectionRow label="Runtime" hint="Measured, not configured: how long it ran and how far the folds trained" span={columnCount} />
                {RUNTIME_ROWS.map((row) => (
                  <tr key={row.key}>
                    <RowLabel label={row.label} rawKey={row.hint} measured />
                    {entries.map((entry) => {
                      const capped = row.key === "runtime.cap" && (entry.runtime?.capHits ?? 0) > 0;
                      return (
                        <td
                          key={entry.id}
                          className={`border-l border-t border-white/[0.04] px-2 py-1.5 font-mono text-[11px] tabular-nums ${
                            capped ? "bg-[#f4c86f]/[0.15] text-[#f6d488] shadow-[inset_2px_0_0_#f4c86f]" : "text-[#aeb9cc]"
                          }`}
                        >
                          {entry.runtime ? row.value(entry.runtime, entry) : <span className="text-[#3f4a5e]">·</span>}
                        </td>
                      );
                    })}
                  </tr>
                ))}
              </>
            )}

            {SECTIONS.filter((section) => sections.has(section.id)).map((section) => {
              const sectionRows = visible.filter((row) => row.section === section.id);
              const total = rows.filter((row) => row.section === section.id);
              if (total.length === 0) return null;
              const differing = total.filter((row) => row.differing > 0 || row.mixed).length;
              return (
                <Fragment key={section.id}>
                  <SectionRow
                    label={section.label}
                    hint={section.hint}
                    span={columnCount}
                    right={`${differing} of ${total.length} differ`}
                  />
                  {sectionRows.map((row) => (
                    <tr key={row.key} className={row.key === highlightKey ? "outline outline-1 outline-[#9b8cff]/60" : undefined}>
                      <RowLabel label={keyLabel(row.key)} rawKey={row.key} />
                      {row.cells.map((cell, index) => (
                        <td
                          key={entries[index].id}
                          className={`max-w-[11rem] border-l border-t border-white/[0.04] px-2 py-1.5 ${
                            cell.differs ? (cell.varying ? "bg-[#d03b3b]/[0.2] shadow-[inset_2px_0_0_#f07171]" : "bg-[#f4c86f]/[0.15] shadow-[inset_2px_0_0_#f4c86f]") : ""
                          }`}
                        >
                          <SettingCell entry={entries[index]} rowKey={row.key} cell={cell} />
                        </td>
                      ))}
                    </tr>
                  ))}
                  {sectionRows.length === 0 && (
                    <tr>
                      <td className="sticky left-0 bg-[#0e1626] px-3 py-1.5 text-[11px] text-[#56627a]">All identical</td>
                      <td colSpan={entries.length} className="border-t border-white/[0.04]" />
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </table>
      </div>
      {onlyDifferences && hidden > 0 && (
        <p className="border-t border-white/[0.06] px-3 py-2 text-[11px] text-[#7f8ba2]">
          {hidden} identical setting{hidden === 1 ? "" : "s"} hidden.
        </p>
      )}
    </div>
  );
}

export function TableOptions({
  onlyDifferences,
  setOnlyDifferences,
  sections,
  setSections,
  showRuntime,
  setShowRuntime,
}: {
  onlyDifferences: boolean;
  setOnlyDifferences: (value: boolean) => void;
  sections: Set<SectionId>;
  setSections: (value: Set<SectionId>) => void;
  showRuntime: boolean;
  setShowRuntime: (value: boolean) => void;
}) {
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
      <Checkbox checked={onlyDifferences} onChange={setOnlyDifferences}>
        Only settings that differ
      </Checkbox>
      <Checkbox checked={showRuntime} onChange={setShowRuntime}>
        Runtime & epochs
      </Checkbox>
      <div className="flex flex-wrap items-center gap-1 text-[10.5px]" role="group" aria-label="Sections">
        {SECTIONS.map((section) => {
          const active = sections.has(section.id);
          return (
            <button
              key={section.id}
              type="button"
              aria-pressed={active}
              title={section.hint}
              onClick={() => {
                const next = new Set(sections);
                if (active) next.delete(section.id);
                else next.add(section.id);
                setSections(next);
              }}
              className={`focus-ring rounded-full border px-2 py-0.5 text-[10.5px] font-semibold transition ${
                active ? "border-[#9b8cff]/35 bg-[#9b8cff]/12 text-[#d8d2ff]" : "border-white/[0.08] text-[#6f7c92] hover:text-[#c8d2e4]"
              }`}
            >
              {section.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}

function SectionRow({ label, hint, span, right }: { label: string; hint: string; span: number; right?: string }) {
  return (
    <tr>
      <td className="sticky left-0 z-10 border-t border-white/[0.08] bg-[#0c1322] px-3 pb-1.5 pt-3">
        <div className="text-[10px] font-bold uppercase tracking-[0.14em] text-[#b6acff]">{label}</div>
        <div className="text-[10px] text-[#6f7c92]">{hint}</div>
      </td>
      <td colSpan={span - 1} className="border-t border-white/[0.08] bg-[#0c1322] px-3 pb-1.5 pt-3 text-[10px] text-[#6f7c92]">
        {right}
      </td>
    </tr>
  );
}

function RowLabel({ label, rawKey, measured }: { label: string | null; rawKey: string; measured?: boolean }) {
  return (
    <th scope="row" className="sticky left-0 z-10 min-w-[15rem] max-w-[17rem] border-t border-white/[0.04] bg-[#0e1626] px-3 py-1.5 font-normal">
      {label ? (
        <>
          <div className="truncate text-[11.5px] font-semibold text-[#e4e9f2]">{label}</div>
          <div className={`truncate text-[10px] ${measured ? "text-[#6f7c92]" : "font-mono text-[#6f7c92]"}`} title={rawKey}>
            {rawKey}
          </div>
        </>
      ) : (
        <div className="truncate font-mono text-[11px] text-[#c8d2e4]" title={rawKey}>
          {rawKey}
        </div>
      )}
    </th>
  );
}
