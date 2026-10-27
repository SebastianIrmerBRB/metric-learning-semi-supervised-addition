"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  ArrowDown,
  ArrowUp,
  ChevronLeft,
  ChevronRight,
  Columns3,
  Download,
  RotateCcw,
  Search,
  SlidersHorizontal,
  X,
} from "lucide-react";
import { downloadText } from "@/lib/export";
import {
  columnLabel,
  formatMetric,
  formatNumber,
  numericValue,
  rowsToCsv,
  slugify,
  type Study,
  type TrialRow,
} from "@/lib/trials";

const PAGE_SIZE = 10;
const CURVES_SORT_COLUMN = "__selected_curves__";

function defaultTableColumns(study: Study): string[] {
  return [
    study.config.trial,
    study.config.metric,
    study.config.state,
    study.config.learningRate,
    study.config.epoch,
    study.config.selectedEpoch,
    study.config.duration,
  ].filter((column, index, columns): column is string => Boolean(column) && columns.indexOf(column) === index);
}

function valueForSort(row: TrialRow, column: string): string | number {
  const value = row[column];
  const numeric = numericValue(value);
  return numeric ?? value?.toLowerCase() ?? "";
}

function compactValue(value: string | null): string {
  if (!value) return "—";
  const numeric = numericValue(value);
  if (numeric != null) return formatNumber(numeric, 6);
  return value.length > 52 ? `${value.slice(0, 49)}…` : value;
}

function stateStyle(value: string | null) {
  const state = (value ?? "UNKNOWN").toUpperCase();
  if (state === "COMPLETE") return "border-[#b8e56e]/20 bg-[#b8e56e]/10 text-[#c9f18a]";
  if (state === "PRUNED") return "border-[#f4c86f]/20 bg-[#f4c86f]/10 text-[#f4d58b]";
  if (state === "FAIL" || state === "FAILED") return "border-rose-400/20 bg-rose-400/10 text-rose-300";
  return "border-white/10 bg-white/5 text-[#aab5c8]";
}

function StateBadge({ value }: { value: string | null }) {
  return (
    <span className={`inline-flex rounded-full border px-2 py-1 text-[10px] font-semibold tracking-[0.08em] ${stateStyle(value)}`}>
      {(value ?? "UNKNOWN").toUpperCase()}
    </span>
  );
}

interface TrialsTableProps {
  study: Study;
  selectedRow: TrialRow | null;
  curveRows: TrialRow[];
  onSelect: (row: TrialRow | null) => void;
  onToggleCurve: (row: TrialRow) => void;
}

export function TrialsTable({ study, selectedRow, curveRows, onSelect, onToggleCurve }: TrialsTableProps) {
  const [query, setQuery] = useState("");
  const [stateFilter, setStateFilter] = useState("ALL");
  const [sortColumn, setSortColumn] = useState(study.config.metric);
  const [sortDirection, setSortDirection] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(0);
  const defaults = useMemo(() => defaultTableColumns(study), [study]);
  const [visibleColumns, setVisibleColumns] = useState<string[]>(defaults);
  const [columnStorageReady, setColumnStorageReady] = useState(false);
  const [columnsOpen, setColumnsOpen] = useState(false);
  const [columnQuery, setColumnQuery] = useState("");
  const columnsMenuRef = useRef<HTMLDivElement>(null);
  const columnStorageKey = `trial-atlas:table-columns:${study.id}`;
  const defaultsSignature = defaults.join("\u001f");
  const columnsSignature = study.columns.join("\u001f");

  useEffect(() => {
    setSortColumn(study.config.metric);
    setSortDirection("desc");
    setQuery("");
    setStateFilter("ALL");
    setPage(0);
  }, [study.id, study.config.metric]);

  useEffect(() => {
    setColumnStorageReady(false);
    let storedColumns: string[] | null = null;
    try {
      const raw = window.localStorage.getItem(columnStorageKey);
      const parsed = raw ? (JSON.parse(raw) as unknown) : null;
      if (Array.isArray(parsed)) {
        storedColumns = parsed.filter((column): column is string => typeof column === "string" && study.columns.includes(column));
      }
    } catch {
      storedColumns = null;
    }
    setVisibleColumns(storedColumns ?? defaults);
    setColumnsOpen(false);
    setColumnQuery("");
    setColumnStorageReady(true);
  }, [columnStorageKey, columnsSignature, defaultsSignature]);

  useEffect(() => {
    if (!columnStorageReady) return;
    try {
      window.localStorage.setItem(columnStorageKey, JSON.stringify(visibleColumns));
    } catch {
      // Browser storage can be unavailable without preventing column selection in this session.
    }
  }, [columnStorageKey, columnStorageReady, visibleColumns]);

  useEffect(() => {
    if (!columnsOpen) return;
    const closeOnPointerDown = (event: PointerEvent) => {
      if (!columnsMenuRef.current?.contains(event.target as Node)) setColumnsOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setColumnsOpen(false);
    };
    document.addEventListener("pointerdown", closeOnPointerDown);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointerDown);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [columnsOpen]);

  const states = useMemo(() => {
    if (!study.config.state) return [];
    return [...new Set(study.rows.map((row) => row[study.config.state as string]?.toUpperCase()).filter(Boolean) as string[])].sort();
  }, [study]);

  const filteredColumnChoices = useMemo(() => {
    const normalized = columnQuery.trim().toLowerCase();
    if (!normalized) return study.columns;
    return study.columns.filter(
      (column) => column.toLowerCase().includes(normalized) || columnLabel(column).toLowerCase().includes(normalized),
    );
  }, [columnQuery, study.columns]);

  useEffect(() => {
    if (sortColumn === CURVES_SORT_COLUMN || visibleColumns.includes(sortColumn)) return;
    const fallback = visibleColumns.includes(study.config.metric)
      ? study.config.metric
      : visibleColumns[0] ?? CURVES_SORT_COLUMN;
    setSortColumn(fallback);
    setSortDirection(fallback === CURVES_SORT_COLUMN ? "desc" : "asc");
    setPage(0);
  }, [sortColumn, study.config.metric, visibleColumns]);

  const filteredRows = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const rows = study.rows.filter((row) => {
      const matchesState =
        stateFilter === "ALL" ||
        !study.config.state ||
        (row[study.config.state] ?? "").toUpperCase() === stateFilter;
      const matchesQuery =
        normalizedQuery === "" ||
        study.columns.some((column) => (row[column] ?? "").toLowerCase().includes(normalizedQuery));
      return matchesState && matchesQuery;
    });

    return [...rows].sort((a, b) => {
      if (sortColumn === CURVES_SORT_COLUMN) {
        const comparison = Number(curveRows.includes(a)) - Number(curveRows.includes(b));
        return sortDirection === "asc" ? comparison : -comparison;
      }
      const left = valueForSort(a, sortColumn);
      const right = valueForSort(b, sortColumn);
      const comparison = typeof left === "number" && typeof right === "number" ? left - right : String(left).localeCompare(String(right));
      return sortDirection === "asc" ? comparison : -comparison;
    });
  }, [curveRows, query, sortColumn, sortDirection, stateFilter, study]);

  const pageCount = Math.max(1, Math.ceil(filteredRows.length / PAGE_SIZE));
  const safePage = Math.min(page, pageCount - 1);
  const pageRows = filteredRows.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE);

  useEffect(() => {
    setPage(0);
  }, [query, stateFilter]);

  const handleSort = (column: string) => {
    if (sortColumn === column) setSortDirection((current) => (current === "asc" ? "desc" : "asc"));
    else {
      setSortColumn(column);
      setSortDirection(column === CURVES_SORT_COLUMN ? "desc" : "asc");
    }
    setPage(0);
  };

  const exportRows = () => {
    downloadText(rowsToCsv(study.columns, filteredRows), `${slugify(study.name)}-trials.csv`);
  };

  const toggleVisibleColumn = (column: string) => {
    setVisibleColumns((current) =>
      current.includes(column) ? current.filter((candidate) => candidate !== column) : [...current, column],
    );
  };

  return (
    <section className="surface">
      <header className="flex flex-col gap-4 border-b border-white/[0.07] px-5 py-5 xl:flex-row xl:items-center xl:justify-between sm:px-6">
        <div>
          <div className="flex items-center gap-2.5">
            <span className="h-2 w-2 rounded-full bg-[#4fd5e7] shadow-[0_0_16px_rgba(79,213,231,.7)]" />
            <h2 className="text-[15px] font-semibold text-white">Trial explorer</h2>
          </div>
          <p className="mt-1.5 text-xs text-[#8f9bb2]">Choose attributes, search, sort, inspect, and export every row in this study.</p>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
          <label className="relative block min-w-0 sm:w-56">
            <span className="sr-only">Search trials</span>
            <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#66738b]" size={15} />
            <input
              type="search"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="Search any value…"
              className="focus-ring h-10 w-full rounded-xl border border-white/[0.09] bg-black/15 pl-9 pr-3 text-xs text-white placeholder:text-[#657188]"
            />
          </label>
          {states.length > 0 && (
            <label className="relative block">
              <span className="sr-only">Filter by state</span>
              <SlidersHorizontal className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#66738b]" size={14} />
              <select
                value={stateFilter}
                onChange={(event) => setStateFilter(event.target.value)}
                className="focus-ring h-10 appearance-none rounded-xl border border-white/[0.09] bg-[#111a2e] pl-9 pr-8 text-xs text-[#d8dfec]"
              >
                <option value="ALL">All states</option>
                {states.map((state) => <option key={state} value={state}>{state}</option>)}
              </select>
            </label>
          )}
          <div className="relative" ref={columnsMenuRef}>
            <button
              type="button"
              onClick={() => setColumnsOpen((open) => !open)}
              className={`focus-ring flex h-10 items-center justify-center gap-2 rounded-xl border px-3.5 text-xs font-semibold transition ${
                columnsOpen
                  ? "border-[#9b8cff]/35 bg-[#9b8cff]/12 text-[#c8c1ff]"
                  : "border-white/[0.09] bg-white/[0.04] text-[#d8dfec] hover:bg-white/[0.08] hover:text-white"
              }`}
              aria-expanded={columnsOpen}
              aria-haspopup="dialog"
            >
              <Columns3 size={14} aria-hidden="true" /> Columns
              <span className="rounded-full bg-white/[0.07] px-1.5 py-0.5 text-[9px] tabular-nums text-[#9da9bd]">
                {visibleColumns.length}/{study.columns.length}
              </span>
            </button>
            {columnsOpen && (
              <div
                role="dialog"
                aria-label="Choose visible trial attributes"
                className="absolute right-0 top-12 z-50 w-[min(360px,calc(100vw-2rem))] overflow-hidden rounded-2xl border border-white/[0.11] bg-[#111a2e]/98 shadow-[0_24px_80px_rgba(0,0,0,.5)] backdrop-blur-xl"
              >
                <div className="border-b border-white/[0.07] px-4 py-3.5">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-semibold text-white">Visible attributes</p>
                      <p className="mt-1 text-[10px] text-[#77859d]">Choose any columns from this CSV.</p>
                    </div>
                    <span className="text-[10px] tabular-nums text-[#8d99ad]">{visibleColumns.length} selected</span>
                  </div>
                  <label className="relative mt-3 block">
                    <span className="sr-only">Filter attributes</span>
                    <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#65738b]" size={13} />
                    <input
                      type="search"
                      value={columnQuery}
                      onChange={(event) => setColumnQuery(event.target.value)}
                      placeholder="Find an attribute…"
                      className="focus-ring h-9 w-full rounded-xl border border-white/[0.09] bg-black/15 pl-8 pr-3 text-[11px] text-white placeholder:text-[#637087]"
                    />
                  </label>
                </div>
                <div className="max-h-80 overflow-y-auto p-2">
                  {filteredColumnChoices.map((column) => {
                    const checked = visibleColumns.includes(column);
                    return (
                      <label
                        key={column}
                        className={`flex cursor-pointer items-start gap-3 rounded-xl px-3 py-2.5 transition ${
                          checked ? "bg-[#9b8cff]/[0.09]" : "hover:bg-white/[0.035]"
                        }`}
                      >
                        <input
                          type="checkbox"
                          checked={checked}
                          onChange={() => toggleVisibleColumn(column)}
                          className="mt-0.5 h-4 w-4 shrink-0 rounded border-white/20 bg-black/20 accent-[#9b8cff]"
                        />
                        <span className="min-w-0">
                          <span className="block text-[11px] font-semibold text-[#dce3ef]">{columnLabel(column)}</span>
                          <span className="mt-0.5 block truncate text-[9px] text-[#68768e]" title={column}>{column}</span>
                        </span>
                      </label>
                    );
                  })}
                  {filteredColumnChoices.length === 0 && (
                    <p className="px-3 py-8 text-center text-xs text-[#78859b]">No attributes match that filter.</p>
                  )}
                </div>
                <div className="flex flex-wrap items-center justify-between gap-2 border-t border-white/[0.07] px-3 py-3">
                  <button
                    type="button"
                    onClick={() => setVisibleColumns(defaults)}
                    className="focus-ring inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[10px] font-semibold text-[#9ca8bb] transition hover:bg-white/[0.05] hover:text-white"
                  >
                    <RotateCcw size={12} /> Defaults
                  </button>
                  <div className="flex items-center gap-1">
                    <button
                      type="button"
                      onClick={() => setVisibleColumns([])}
                      className="focus-ring h-8 rounded-lg px-2.5 text-[10px] font-semibold text-[#8290a7] transition hover:bg-white/[0.05] hover:text-white"
                    >
                      Clear
                    </button>
                    <button
                      type="button"
                      onClick={() => setVisibleColumns([...study.columns])}
                      className="focus-ring h-8 rounded-lg bg-[#9b8cff]/12 px-2.5 text-[10px] font-semibold text-[#c5bdff] transition hover:bg-[#9b8cff]/20"
                    >
                      Select all
                    </button>
                  </div>
                </div>
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={exportRows}
            className="focus-ring flex h-10 items-center justify-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.04] px-3.5 text-xs font-semibold text-[#d8dfec] transition hover:bg-white/[0.08] hover:text-white"
          >
            <Download size={14} aria-hidden="true" /> Export CSV
          </button>
        </div>
      </header>

      <div className={`grid min-w-0 ${selectedRow ? "xl:grid-cols-[minmax(0,1fr)_340px]" : "grid-cols-1"}`}>
        <div className="min-w-0 overflow-hidden">
          <div className="overflow-x-auto">
            <table
              className="w-full border-collapse text-left text-xs"
              style={{ minWidth: `${Math.max(780, 220 + visibleColumns.length * 150)}px` }}
            >
              <thead>
                <tr className="border-b border-white/[0.07] bg-black/10">
                  <th
                    scope="col"
                    aria-sort={sortColumn === CURVES_SORT_COLUMN ? (sortDirection === "asc" ? "ascending" : "descending") : "none"}
                    className="w-20 px-4 py-3 text-center font-semibold text-[#8996ad] first:pl-6"
                  >
                    <button
                      type="button"
                      onClick={() => handleSort(CURVES_SORT_COLUMN)}
                      className="focus-ring mx-auto flex items-center justify-center gap-1.5 rounded transition hover:text-white"
                      aria-label="Sort by trials selected for learning curves"
                    >
                      Curves
                      {sortColumn === CURVES_SORT_COLUMN && (sortDirection === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                    </button>
                  </th>
                  {visibleColumns.map((column) => (
                    <th key={column} scope="col" className="px-4 py-3 font-semibold text-[#8996ad] first:pl-6">
                      <button
                        type="button"
                        onClick={() => handleSort(column)}
                        className="focus-ring flex items-center gap-1.5 rounded text-left transition hover:text-white"
                      >
                        {column === study.config.metric ? "Metric" : columnLabel(column)}
                        {sortColumn === column && (sortDirection === "asc" ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                      </button>
                    </th>
                  ))}
                  <th scope="col" className="px-4 py-3 text-right font-semibold text-[#8996ad]">Details</th>
                </tr>
              </thead>
              <tbody>
                {pageRows.map((row, rowIndex) => {
                  const selected = row === selectedRow;
                  const includedInCurves = curveRows.includes(row);
                  return (
                    <tr
                      key={`${row[study.config.trial] ?? rowIndex}-${rowIndex}`}
                      onClick={() => onSelect(row)}
                      className={`border-b border-white/[0.055] transition last:border-0 ${
                        selected ? "bg-[#9b8cff]/[0.09]" : "hover:bg-white/[0.025]"
                      }`}
                    >
                      <td className="px-4 py-3.5 text-center first:pl-6">
                        <input
                          type="checkbox"
                          checked={includedInCurves}
                          onClick={(event) => event.stopPropagation()}
                          onChange={() => onToggleCurve(row)}
                          className="h-4 w-4 rounded border-white/20 bg-black/20 accent-[#9b8cff]"
                          aria-label={`${includedInCurves ? "Remove" : "Add"} trial ${row[study.config.trial] ?? rowIndex} ${includedInCurves ? "from" : "to"} learning curves`}
                        />
                      </td>
                      {visibleColumns.map((column) => {
                        const value = row[column];
                        const isTrial = column === study.config.trial;
                        const isMetric = column === study.config.metric;
                        const isState = column === study.config.state;
                        return (
                          <td key={column} className="max-w-56 px-4 py-3.5 text-[#b8c2d5] first:pl-6">
                            {isTrial ? (
                              <button
                                type="button"
                                onClick={(event) => {
                                  event.stopPropagation();
                                  onSelect(row);
                                }}
                                className="focus-ring rounded-md font-semibold text-white transition hover:text-[#b8e56e]"
                              >
                                #{value ?? rowIndex}
                              </button>
                            ) : isState ? (
                              <StateBadge value={value} />
                            ) : isMetric ? (
                              <span className="font-semibold tabular-nums text-[#d8d0ff]">{formatMetric(numericValue(value))}</span>
                            ) : (
                              <span className="block truncate tabular-nums" title={value ?? undefined}>{compactValue(value)}</span>
                            )}
                          </td>
                        );
                      })}
                      <td className="px-4 py-3.5 text-right">
                        <button
                          type="button"
                          onClick={(event) => {
                            event.stopPropagation();
                            onSelect(row);
                          }}
                          className="focus-ring rounded-lg border border-white/[0.08] px-2.5 py-1.5 text-[11px] font-semibold text-[#9da9be] transition hover:border-[#9b8cff]/40 hover:text-white"
                        >
                          Inspect
                        </button>
                      </td>
                    </tr>
                  );
                })}
                {pageRows.length === 0 && (
                  <tr>
                    <td colSpan={visibleColumns.length + 2} className="px-6 py-14 text-center text-sm text-[#7f8ba2]">
                      No trials match the current filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
          <footer className="flex flex-wrap items-center justify-between gap-3 border-t border-white/[0.07] px-5 py-4 sm:px-6">
            <p className="text-xs text-[#7f8ba2]">
              Showing {filteredRows.length === 0 ? 0 : safePage * PAGE_SIZE + 1}–{Math.min((safePage + 1) * PAGE_SIZE, filteredRows.length)} of {filteredRows.length} trials
            </p>
            <div className="flex items-center gap-2">
              <button
                type="button"
                disabled={safePage === 0}
                onClick={() => setPage((current) => Math.max(0, current - 1))}
                className="focus-ring flex h-8 w-8 items-center justify-center rounded-lg border border-white/[0.08] text-[#a5b0c4] transition hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-30"
                aria-label="Previous page"
              >
                <ChevronLeft size={15} />
              </button>
              <span className="min-w-20 text-center text-xs text-[#8f9bb2]">{safePage + 1} / {pageCount}</span>
              <button
                type="button"
                disabled={safePage >= pageCount - 1}
                onClick={() => setPage((current) => Math.min(pageCount - 1, current + 1))}
                className="focus-ring flex h-8 w-8 items-center justify-center rounded-lg border border-white/[0.08] text-[#a5b0c4] transition hover:bg-white/[0.06] disabled:cursor-not-allowed disabled:opacity-30"
                aria-label="Next page"
              >
                <ChevronRight size={15} />
              </button>
            </div>
          </footer>
        </div>

        {selectedRow && (
          <aside className="border-t border-white/[0.07] bg-black/[0.08] xl:border-l xl:border-t-0" aria-label="Selected trial details">
            <div className="flex items-start justify-between border-b border-white/[0.07] px-5 py-5">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[#71809a]">Selected trial</p>
                <h3 className="mt-2 text-xl font-semibold text-white">#{selectedRow[study.config.trial] ?? "—"}</h3>
              </div>
              <button
                type="button"
                onClick={() => onSelect(null)}
                className="focus-ring flex h-8 w-8 items-center justify-center rounded-lg text-[#7f8ba2] transition hover:bg-white/[0.06] hover:text-white"
                aria-label="Close trial details"
              >
                <X size={16} />
              </button>
            </div>
            <div className="grid grid-cols-2 gap-px border-b border-white/[0.07] bg-white/[0.07]">
              <div className="bg-[#111a2e] px-5 py-4">
                <p className="text-[10px] uppercase tracking-[0.12em] text-[#71809a]">Metric</p>
                <p className="mt-1.5 text-lg font-semibold text-[#c9c0ff]">{formatMetric(numericValue(selectedRow[study.config.metric]))}</p>
              </div>
              <div className="bg-[#111a2e] px-5 py-4">
                <p className="text-[10px] uppercase tracking-[0.12em] text-[#71809a]">State</p>
                <div className="mt-2"><StateBadge value={study.config.state ? selectedRow[study.config.state] : "COMPLETE"} /></div>
              </div>
            </div>
            <dl className="max-h-[520px] divide-y divide-white/[0.055] overflow-y-auto px-5">
              {study.columns.map((column) => (
                <div key={column} className="py-3.5">
                  <dt className="text-[10px] font-semibold uppercase tracking-[0.1em] text-[#6f7c94]">{columnLabel(column)}</dt>
                  <dd className="mt-1.5 break-words text-xs leading-5 text-[#bdc6d7]" title={selectedRow[column] ?? undefined}>
                    {compactValue(selectedRow[column])}
                  </dd>
                </div>
              ))}
            </dl>
          </aside>
        )}
      </div>
    </section>
  );
}
