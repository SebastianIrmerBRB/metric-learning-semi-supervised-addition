"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";
import { ArrowLeft, FileCog } from "lucide-react";

import {
  KIND_GROUP,
  computeIssues,
  decodeSnapshot,
  formatDuration,
  type ConfigPhase,
  type ConfigSnapshot,
  type Issue,
  type Phase2Kind,
} from "@/lib/study-configs";
import { ALL, DEFAULT_FILTERS, type Filters, type P2Group } from "@/components/configs/filters";
import { ChecksView, type IssueTarget } from "@/components/configs/ChecksView";
import { CompareView } from "@/components/configs/CompareView";
import { CoverageView, type CoverageMode } from "@/components/configs/CoverageView";
import { ProtocolView } from "@/components/configs/ProtocolView";
import { SearchSpaceView } from "@/components/configs/SearchSpaceView";
import { TraceView } from "@/components/configs/TraceView";
import { Checkbox, Toggle } from "@/components/configs/ui";

type Tab = "checks" | "protocol" | "compare" | "trace" | "spaces" | "coverage";

const TABS: { value: Tab; label: string; hint: string }[] = [
  { value: "checks", label: "Checks", hint: "Everything that ran differently from its peers, or is missing" },
  { value: "protocol", label: "Protocol", hint: "Datasets × methods: epochs, patience, folds, budget per phase" },
  { value: "compare", label: "Compare", hint: "Every setting side by side for any selection" },
  { value: "trace", label: "Phase trace", hint: "One dataset · method · loss through Phase 0, 1, seeded runs and Phase 2" },
  { value: "spaces", label: "Search spaces", hint: "HPO ranges across datasets, Phase 0 vs. Phase 1" },
  { value: "coverage", label: "Coverage", hint: "What exists, what is unfinished, what is missing" },
];

const PHASE_VALUES: ConfigPhase[] = ["phase0", "phase1", "seeded", "phase2"];

function readUrlState(): { tab: Tab; filters: Filters; weitere: boolean; coverage: CoverageMode } {
  const params = new URLSearchParams(window.location.search);
  const tab = TABS.some((item) => item.value === params.get("tab")) ? (params.get("tab") as Tab) : "checks";
  const phase = params.get("phase");
  const p2 = params.get("p2");
  return {
    tab,
    weitere: params.get("weitere") === "1",
    coverage: params.get("cov") === "phase2" ? "phase2" : "hpo",
    filters: {
      phase: PHASE_VALUES.includes(phase as ConfigPhase) ? (phase as ConfigPhase) : DEFAULT_FILTERS.phase,
      dataset: params.get("dataset") ?? ALL,
      method: params.get("method") ?? ALL,
      loss: params.get("loss") ?? ALL,
      p2group: p2 === "ablation" || p2 === "geometry" ? (p2 as P2Group) : "all",
      kind: (params.get("kind") as Phase2Kind | null) ?? ALL,
      root: params.get("root") ?? ALL,
    },
  };
}

function writeUrlState(tab: Tab, filters: Filters, weitere: boolean, coverage: CoverageMode) {
  const params = new URLSearchParams();
  if (tab !== "checks") params.set("tab", tab);
  if (filters.phase !== DEFAULT_FILTERS.phase) params.set("phase", filters.phase);
  for (const key of ["dataset", "method", "loss", "kind", "root"] as const) if (filters[key] !== ALL) params.set(key, filters[key]);
  if (filters.p2group !== "all") params.set("p2", filters.p2group);
  if (weitere) params.set("weitere", "1");
  if (tab === "coverage" && coverage === "phase2") params.set("cov", "phase2");
  const query = params.toString();
  window.history.replaceState(null, "", `${window.location.pathname}${query ? `?${query}` : ""}`);
}

export function ConfigOverview() {
  const [snapshot, setSnapshot] = useState<ConfigSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<Tab>("checks");
  const [filters, setFiltersState] = useState<Filters>(DEFAULT_FILTERS);
  const [includeWeitere, setIncludeWeitere] = useState(false);
  const [highlightKey, setHighlightKey] = useState<string | null>(null);
  const [coverageMode, setCoverageMode] = useState<CoverageMode>("hpo");
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const state = readUrlState();
    setTab(state.tab);
    setFiltersState(state.filters);
    setIncludeWeitere(state.weitere);
    setCoverageMode(state.coverage);
    setReady(true);
    let cancelled = false;
    fetch("/study-configs.json", { cache: "no-store" })
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error(`HTTP ${response.status}`))))
      .then((raw) => {
        if (!cancelled) setSnapshot(decodeSnapshot(raw));
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (ready) writeUrlState(tab, filters, includeWeitere, coverageMode);
  }, [ready, tab, filters, includeWeitere, coverageMode]);

  const setFilters = useCallback((next: Partial<Filters>) => {
    setFiltersState((current) => ({ ...current, ...next }));
    setHighlightKey(null);
  }, []);

  const entries = useMemo(
    () => (snapshot ? snapshot.entries.filter((entry) => includeWeitere || entry.archive === "main") : []),
    [snapshot, includeWeitere],
  );
  const issues = useMemo(() => (snapshot ? computeIssues(snapshot, entries) : []), [snapshot, entries]);

  const stats = useMemo(() => {
    const count = (phase: ConfigPhase) => entries.filter((entry) => entry.phase === phase).length;
    const phase2 = entries.filter((entry) => entry.phase === "phase2");
    const compute = entries.reduce((sum, entry) => sum + (entry.runtime?.computeSeconds ?? 0), 0);
    return {
      phase0: count("phase0"),
      phase1: count("phase1"),
      seeded: count("seeded"),
      ablations: phase2.filter((entry) => KIND_GROUP[entry.kind ?? "other"] === "ablation").length,
      geometry: phase2.filter((entry) => entry.kind === "geometry").length,
      findings: issues.filter((issue) => issue.type !== "incomplete").length,
      compute,
    };
  }, [entries, issues]);

  const openIssue = useCallback(
    (issue: Issue, target: IssueTarget) => {
      const entry = snapshot?.entries.find((item) => item.id === issue.entries[0]);
      if (target === "coverage") {
        const root = issue.id.split("|")[3] ?? ALL;
        setFiltersState((current) => ({ ...current, root, p2group: root.endsWith("_geometry") ? "geometry" : "ablation" }));
        setCoverageMode("phase2");
        setTab("coverage");
      } else if (target === "trace" && entry) {
        setFiltersState((current) => ({ ...current, dataset: entry.dataset, method: entry.method, loss: entry.loss }));
        setTab("trace");
      } else {
        // Deviation ids are deviation|phase|dataset|group|key; the group is "" outside Phase 2.
        const group = issue.type === "deviation" ? issue.id.split("|")[3] : entry?.phase === "phase2" ? KIND_GROUP[entry.kind ?? "other"] : "";
        setFiltersState((current) => ({
          ...current,
          phase: issue.phase,
          dataset: issue.dataset,
          method: issue.type === "deviation" || !entry ? ALL : entry.method,
          loss: ALL,
          p2group: group === "geometry" || group === "ablation" ? group : "all",
          kind: ALL,
          root: ALL,
        }));
        setHighlightKey(issue.key ?? null);
        setTab("compare");
      }
      window.scrollTo({ top: 0, behavior: "smooth" });
    },
    [snapshot],
  );

  return (
    <main className="mx-auto max-w-[1400px] px-4 pb-20 pt-8 sm:px-5">
      <Link href="/" className="focus-ring text-xs font-semibold text-[#8895aa] hover:text-white">
        <ArrowLeft size={14} className="mr-1 inline" />
        Back to Trial Atlas
      </Link>

      <header className="mt-5">
        <div className="flex flex-wrap items-center gap-3">
          <FileCog size={20} className="text-[#9b8cff]" />
          <h1 className="text-2xl font-semibold text-white">Run configs</h1>
          {snapshot && (
            <span className="rounded-full bg-[#9b8cff]/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#c9c1ff]">
              Snapshot {snapshot.generatedAt.slice(0, 16).replace("T", " ")} UTC
            </span>
          )}
        </div>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#98a4ba]">
          What every study and run under <code className="text-[#c8d2e4]">{snapshot?.root ?? "/mnt/f/Studien"}</code> actually ran with, read
          from the files the runs wrote: did each dataset use the same protocol, where do settings differ, how long did it take, and what is
          missing.
        </p>
      </header>

      {error && (
        <div className="mt-8 max-w-3xl">
          <p className="text-sm text-[#98a4ba]">
            No config snapshot yet (<code>public/study-configs.json</code>: {error}). Build it from the study folders, then reload:
          </p>
          <pre className="mt-3 overflow-x-auto rounded-xl border border-white/[0.07] bg-black/40 p-4 text-[11px] leading-relaxed text-[#c8d2e4]">
            cd react_trial_visualizer && npm run sync-configs
          </pre>
        </div>
      )}
      {!error && !snapshot && <p className="mt-8 text-sm text-[#7f8ba2]">Loading the config snapshot…</p>}

      {snapshot && (
        <>
          <div className="mt-6 grid grid-cols-2 gap-2 sm:grid-cols-4 lg:grid-cols-7">
            <Stat label="Phase 0 studies" value={stats.phase0} />
            <Stat label="Phase 1 studies" value={stats.phase1} />
            <Stat label="Seeded series" value={stats.seeded} />
            <Stat label="P2 ablation folders" value={stats.ablations} />
            <Stat label="P2 geometry folders" value={stats.geometry} />
            <Stat label="Summed run time" value={formatDuration(stats.compute)} />
            <Stat label="Findings" value={stats.findings} accent onClick={() => setTab("checks")} />
          </div>

          <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
            <Toggle value={tab} onChange={setTab} options={TABS} />
            <Checkbox checked={includeWeitere} onChange={setIncludeWeitere}>
              Include <code className="font-mono text-[#aeb9cc]">_weitere</code> (studies and seeded series not used in the thesis)
            </Checkbox>
          </div>

          {tab === "checks" && <ChecksView issues={issues} onOpen={openIssue} />}
          {tab === "protocol" && (
            <ProtocolView
              entries={entries}
              filters={filters}
              setFilters={setFilters}
              onOpen={(dataset, method) => {
                setFilters({ dataset, method, loss: ALL });
                setTab("compare");
              }}
            />
          )}
          {tab === "compare" && <CompareView entries={entries} studies={snapshot.entries} filters={filters} setFilters={setFilters} highlightKey={highlightKey} />}
          {tab === "trace" && <TraceView entries={entries} studies={snapshot.entries} filters={filters} setFilters={setFilters} />}
          {tab === "spaces" && <SearchSpaceView entries={entries} filters={filters} setFilters={setFilters} />}
          {tab === "coverage" && (
            <CoverageView
              snapshot={snapshot}
              entries={entries}
              filters={filters}
              setFilters={setFilters}
              mode={coverageMode}
              setMode={setCoverageMode}
            />
          )}

          <footer className="mt-12 space-y-1 border-t border-white/[0.06] pt-4 text-[11px] leading-relaxed text-[#6f7c92]">
            <p>
              Refresh with <code className="text-[#aeb9cc]">npm run sync-configs</code> (reads only files that changed since the last export).
              Phase-2 plans: <code className="text-[#aeb9cc]">{snapshot.manifestRoot}</code> ({snapshot.manifests.length} manifests).
            </p>
            {snapshot.skipped.length > 0 && (
              <p>
                Not scanned:{" "}
                {snapshot.skipped.map((item, index) => (
                  <span key={item.path} title={item.reason}>
                    {index > 0 && ", "}
                    <code className="text-[#8895aa]">{item.path}</code>
                  </span>
                ))}
                .
              </p>
            )}
          </footer>
        </>
      )}
    </main>
  );
}

function Stat({ label, value, accent, onClick }: { label: string; value: number | string; accent?: boolean; onClick?: () => void }) {
  const body = (
    <>
      <div className="text-[10px] font-semibold uppercase tracking-[0.1em] text-[#7f8ba2]">{label}</div>
      <div className={`mt-1 text-xl font-semibold tabular-nums ${accent ? "text-[#f6d488]" : "text-white"}`}>{value}</div>
    </>
  );
  return onClick ? (
    <button type="button" onClick={onClick} className="focus-ring surface-soft px-3 py-2.5 text-left hover:border-white/[0.2]">
      {body}
    </button>
  ) : (
    <div className="surface-soft px-3 py-2.5">{body}</div>
  );
}
