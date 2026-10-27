"use client";

import Link from "next/link";
import {
  ArrowLeft,
  ArrowUpRight,
  BarChart3,
  Check,
  ChevronDown,
  Database,
  FileSpreadsheet,
  Filter,
  FlaskConical,
  Layers3,
  RotateCcw,
  Search,
  Sigma,
  Trophy,
  UploadCloud,
} from "lucide-react";
import {
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
} from "react";
import {
  TEST_METRICS,
  formatCompact,
  formatMetric,
  labelBudget,
  parseTestResultsCsv,
  resultMetric,
  summarizeConfigurations,
  uniqueSorted,
  type TestMetric,
  type TestResult,
  type TestResultFile,
} from "@/lib/test-results";

interface TestResultsExplorerProps {
  initialFile: TestResultFile | null;
}

interface LoadedResults {
  results: TestResult[];
  error: string | null;
}

const selectClass =
  "focus-ring h-10 appearance-none rounded-xl border border-white/[0.09] bg-[#0c1425] pl-3 pr-9 text-xs text-[#d9e0ed]";

function loadResults(file: TestResultFile | null): LoadedResults {
  if (!file) return { results: [], error: null };
  try {
    return { results: parseTestResultsCsv(file.text), error: null };
  } catch (error) {
    return {
      results: [],
      error: error instanceof Error ? error.message : "Could not parse the aggregate CSV.",
    };
  }
}

function humanize(value: string): string {
  return value
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function Kpi({
  label,
  value,
  detail,
  icon,
  tone,
}: {
  label: string;
  value: string;
  detail: string;
  icon: React.ReactNode;
  tone: "violet" | "cyan" | "lime" | "amber";
}) {
  const tones = {
    violet: "bg-[#9b8cff]/10 text-[#b8adff]",
    cyan: "bg-[#4fd5e7]/10 text-[#6ee4ef]",
    lime: "bg-[#b8e56e]/10 text-[#c8ef88]",
    amber: "bg-[#f4b860]/10 text-[#f7c97f]",
  };
  return (
    <article className="surface-soft min-w-0 p-4">
      <div className={`flex h-9 w-9 items-center justify-center rounded-xl ${tones[tone]}`}>
        {icon}
      </div>
      <p className="mt-5 text-[10px] font-semibold uppercase tracking-[0.16em] text-[#75829a]">
        {label}
      </p>
      <p className="mt-1 truncate text-2xl font-semibold tracking-[-0.03em] text-white">{value}</p>
      <p className="mt-1 truncate text-[11px] text-[#77849a]">{detail}</p>
    </article>
  );
}

function EmptyState({
  hasFile,
  onChooseFile,
}: {
  hasFile: boolean;
  onChooseFile: () => void;
}) {
  return (
    <section className="surface mx-auto mt-10 max-w-2xl px-6 py-14 text-center sm:px-12">
      <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-[#9b8cff]/10 text-[#b4a8ff]">
        <FileSpreadsheet size={25} />
      </span>
      <h2 className="mt-6 text-xl font-semibold tracking-[-0.02em] text-white">
        {hasFile ? "No completed test runs in this aggregate" : "Load the server aggregate"}
      </h2>
      <p className="mx-auto mt-3 max-w-lg text-sm leading-6 text-[#8995aa]">
        {hasFile
          ? "The CSV has the expected schema, but it contains no rows with test mAP@R or P@1."
          : "Run the remote export to create test-results.csv, rebuild Trial Atlas, or open the file here for an immediate comparison."}
      </p>
      <button
        type="button"
        onClick={onChooseFile}
        className="focus-ring mt-7 inline-flex h-10 items-center gap-2 rounded-xl bg-[#9b8cff] px-4 text-xs font-semibold text-white transition hover:bg-[#aa9eff]"
      >
        <UploadCloud size={15} />
        Open aggregate CSV
      </button>
    </section>
  );
}

function FilterSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: string[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="min-w-0">
      <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-[0.14em] text-[#67758d]">
        {label}
      </span>
      <span className="relative block">
        <select
          value={value}
          onChange={(event) => onChange(event.target.value)}
          className={`${selectClass} w-full`}
        >
          <option value="all">All</option>
          {options.map((option) => (
            <option key={option} value={option}>
              {humanize(option)}
            </option>
          ))}
        </select>
        <ChevronDown
          size={14}
          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[#66738a]"
        />
      </span>
    </label>
  );
}

export function TestResultsExplorer({ initialFile }: TestResultsExplorerProps) {
  const initial = useMemo(() => loadResults(initialFile), [initialFile]);
  const [results, setResults] = useState(initial.results);
  const [fileName, setFileName] = useState(initialFile?.name ?? null);
  const [error, setError] = useState(initial.error);
  const [metric, setMetric] = useState<TestMetric>("test_mean_average_precision_at_r");
  const [dataset, setDataset] = useState("all");
  const [mode, setMode] = useState("all");
  const [loss, setLoss] = useState("all");
  const [samplingMode, setSamplingMode] = useState("all");
  const [query, setQuery] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  const datasets = useMemo(() => uniqueSorted(results, (result) => result.dataset), [results]);
  const modes = useMemo(() => uniqueSorted(results, (result) => result.mode), [results]);
  const losses = useMemo(() => uniqueSorted(results, (result) => result.loss), [results]);
  const samplingModes = useMemo(
    () => uniqueSorted(results, (result) => result.labelSamplingMode),
    [results],
  );

  const filtered = useMemo(() => {
    const needle = query.trim().toLowerCase();
    return results.filter((result) => {
      if (dataset !== "all" && result.dataset !== dataset) return false;
      if (mode !== "all" && result.mode !== mode) return false;
      if (loss !== "all" && result.loss !== loss) return false;
      if (samplingMode !== "all" && result.labelSamplingMode !== samplingMode) return false;
      if (
        needle &&
        ![
          result.dataset,
          result.scenario,
          result.mode,
          result.role,
          result.loss,
          result.miner,
          result.sslMethod,
          result.studyName,
          result.seed,
        ]
          .join(" ")
          .toLowerCase()
          .includes(needle)
      ) {
        return false;
      }
      return resultMetric(result, metric) != null;
    });
  }, [dataset, loss, metric, mode, query, results, samplingMode]);

  const configurations = useMemo(
    () => summarizeConfigurations(filtered, metric),
    [filtered, metric],
  );
  const sortedRuns = useMemo(
    () =>
      [...filtered].sort(
        (left, right) =>
          (resultMetric(right, metric) ?? -Infinity) -
          (resultMetric(left, metric) ?? -Infinity),
      ),
    [filtered, metric],
  );

  const best = configurations[0] ?? null;
  const runnerUp = configurations[1] ?? null;
  const delta = best && runnerUp ? best.mean - runnerUp.mean : null;
  const seedCount = new Set(filtered.map((result) => result.seed)).size;

  const resetFilters = () => {
    setDataset("all");
    setMode("all");
    setLoss("all");
    setSamplingMode("all");
    setQuery("");
  };

  const acceptFile = async (file: File | null) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Choose the test-results.csv aggregate.");
      return;
    }
    try {
      const parsed = parseTestResultsCsv(await file.text());
      setResults(parsed);
      setFileName(file.name);
      setError(null);
      resetFilters();
    } catch (loadError) {
      setError(
        loadError instanceof Error ? loadError.message : "Could not parse the aggregate CSV.",
      );
    }
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    void acceptFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setDragging(false);
    void acceptFile(event.dataTransfer.files[0] ?? null);
  };

  return (
    <div
      className="min-h-screen"
      onDragEnter={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setDragging(false);
      }}
      onDrop={onDrop}
    >
      <header className="sticky top-0 z-40 border-b border-white/[0.07] bg-[#080d18]/82 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-[1540px] items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
          <div className="flex min-w-0 items-center gap-3">
            <Link
              href="/"
              className="focus-ring flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-white/[0.08] text-[#8995a9] transition hover:border-white/[0.14] hover:text-white"
              aria-label="Back to Trial Atlas"
            >
              <ArrowLeft size={17} />
            </Link>
            <div className="min-w-0">
              <p className="truncate text-xs font-semibold text-[#e7ebf3]">Final test comparison</p>
              <p className="mt-0.5 truncate text-[10px] text-[#68758b]">
                {fileName ?? "No aggregate bundled"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="hidden rounded-xl border border-white/[0.07] bg-white/[0.025] p-1 sm:flex">
              {(Object.keys(TEST_METRICS) as TestMetric[]).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setMetric(option)}
                  className={`focus-ring rounded-lg px-3 py-1.5 text-[11px] font-semibold transition ${
                    metric === option
                      ? "bg-[#9b8cff] text-white shadow-[0_8px_22px_rgba(118,91,255,.2)]"
                      : "text-[#78869c] hover:text-white"
                  }`}
                >
                  {TEST_METRICS[option]}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              className="focus-ring flex h-9 items-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.035] px-3 text-xs font-semibold text-[#d7deea] transition hover:bg-white/[0.07]"
            >
              <UploadCloud size={15} />
              <span className="hidden sm:inline">Open CSV</span>
            </button>
            <input
              ref={fileInput}
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              onChange={onFileChange}
            />
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-[1540px] px-4 py-7 sm:px-6 lg:px-8 lg:py-9">
        <div className="flex flex-col justify-between gap-5 lg:flex-row lg:items-end">
          <div className="max-w-3xl">
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-[#9b8cff]">
              <FlaskConical size={14} />
              Held-out evaluation
            </div>
            <h1 className="mt-3 text-3xl font-semibold tracking-[-0.045em] text-white sm:text-4xl">
              Compare the runs that made it to test.
            </h1>
            <p className="mt-3 max-w-2xl text-sm leading-6 text-[#8995aa]">
              Final runs are grouped by experimental condition and summarized across seeds.
              Validation remains visible for context, but ranking uses only the selected test metric.
            </p>
          </div>
          <div className="flex sm:hidden">
            <label className="w-full">
              <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-[0.14em] text-[#67758d]">
                Ranking metric
              </span>
              <span className="relative block">
                <select
                  value={metric}
                  onChange={(event) => setMetric(event.target.value as TestMetric)}
                  className={`${selectClass} w-full`}
                >
                  {(Object.keys(TEST_METRICS) as TestMetric[]).map((option) => (
                    <option key={option} value={option}>
                      {TEST_METRICS[option]}
                    </option>
                  ))}
                </select>
                <ChevronDown
                  size={14}
                  className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[#66738a]"
                />
              </span>
            </label>
          </div>
        </div>

        {error && (
          <div
            role="alert"
            className="mt-6 rounded-2xl border border-[#ff7b7b]/20 bg-[#ff7b7b]/[0.07] px-4 py-3 text-xs leading-5 text-[#ffaaaa]"
          >
            {error}
          </div>
        )}

        {results.length === 0 ? (
          <EmptyState hasFile={fileName != null} onChooseFile={() => fileInput.current?.click()} />
        ) : (
          <>
            <section className="mt-8 grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
              <Kpi
                label="Final runs"
                value={formatCompact(filtered.length)}
                detail={`${configurations.length} comparable configurations`}
                icon={<Database size={17} />}
                tone="violet"
              />
              <Kpi
                label={`Best mean ${TEST_METRICS[metric]}`}
                value={formatMetric(best?.mean)}
                detail={best?.label ?? "No matching configuration"}
                icon={<Trophy size={17} />}
                tone="lime"
              />
              <Kpi
                label="Lead over #2"
                value={delta == null ? "—" : `+${delta.toFixed(4)}`}
                detail={runnerUp ? `vs. ${runnerUp.label}` : "Add another configuration"}
                icon={<ArrowUpRight size={17} />}
                tone="cyan"
              />
              <Kpi
                label="Seed coverage"
                value={formatCompact(seedCount)}
                detail={`${new Set(filtered.map((result) => result.dataset)).size} datasets in view`}
                icon={<Layers3 size={17} />}
                tone="amber"
              />
            </section>

            <section className="surface mt-5 p-4 sm:p-5">
              <div className="flex flex-col gap-4 xl:flex-row xl:items-end">
                <div className="min-w-0 flex-1">
                  <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-[0.14em] text-[#67758d]">
                    Find a run
                  </span>
                  <label className="relative block">
                    <Search
                      size={15}
                      className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#627087]"
                    />
                    <input
                      value={query}
                      onChange={(event) => setQuery(event.target.value)}
                      placeholder="Scenario, study, loss, seed…"
                      className="focus-ring h-10 w-full rounded-xl border border-white/[0.09] bg-[#0c1425] pl-9 pr-3 text-xs text-[#d9e0ed] placeholder:text-[#526078]"
                    />
                  </label>
                </div>
                <div className="grid flex-[2] grid-cols-2 gap-3 md:grid-cols-4">
                  <FilterSelect
                    label="Dataset"
                    value={dataset}
                    options={datasets}
                    onChange={setDataset}
                  />
                  <FilterSelect label="Mode" value={mode} options={modes} onChange={setMode} />
                  <FilterSelect label="Loss" value={loss} options={losses} onChange={setLoss} />
                  <FilterSelect
                    label="Sampling"
                    value={samplingMode}
                    options={samplingModes}
                    onChange={setSamplingMode}
                  />
                </div>
                <button
                  type="button"
                  onClick={resetFilters}
                  className="focus-ring flex h-10 shrink-0 items-center justify-center gap-2 rounded-xl border border-white/[0.08] px-3 text-xs text-[#8794aa] transition hover:border-white/[0.15] hover:text-white"
                >
                  <RotateCcw size={14} />
                  Reset
                </button>
              </div>
              <div className="mt-4 flex items-center gap-2 border-t border-white/[0.06] pt-3 text-[10px] text-[#657289]">
                <Filter size={12} />
                {filtered.length} of {results.length} final runs · {configurations.length} grouped
                configurations
              </div>
            </section>

            {configurations.length === 0 ? (
              <section className="surface mt-5 px-6 py-12 text-center">
                <p className="text-sm font-semibold text-white">No runs match these filters.</p>
                <button
                  type="button"
                  onClick={resetFilters}
                  className="focus-ring mt-4 text-xs font-semibold text-[#a99cff] hover:text-[#c1b8ff]"
                >
                  Clear filters
                </button>
              </section>
            ) : (
              <div className="mt-5 grid items-start gap-5 xl:grid-cols-[minmax(0,0.95fr)_minmax(620px,1.45fr)]">
                <section className="surface p-5 sm:p-6">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <p className="text-sm font-semibold text-white">Configuration leaders</p>
                      <p className="mt-1 text-[11px] text-[#748198]">
                        Mean {TEST_METRICS[metric]} across matching final runs
                      </p>
                    </div>
                    <span className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#4fd5e7]/10 text-[#66dfeb]">
                      <BarChart3 size={17} />
                    </span>
                  </div>
                  <div className="mt-7 space-y-5">
                    {configurations.slice(0, 10).map((configuration, index) => (
                      <div key={configuration.key}>
                        <div className="mb-2 flex items-end justify-between gap-3">
                          <div className="min-w-0">
                            <p className="truncate text-xs font-semibold text-[#e3e8f1]">
                              <span className="mr-2 text-[#647188]">
                                {String(index + 1).padStart(2, "0")}
                              </span>
                              {configuration.label}
                            </p>
                            <p className="mt-1 truncate text-[10px] text-[#66748a]">
                              {configuration.details}
                            </p>
                          </div>
                          <p className="shrink-0 font-mono text-xs font-semibold text-white">
                            {formatMetric(configuration.mean)}
                          </p>
                        </div>
                        <div className="h-2 overflow-hidden rounded-full bg-white/[0.055]">
                          <div
                            className="h-full rounded-full bg-gradient-to-r from-[#7d6cff] to-[#55d9e8]"
                            style={{
                              width: `${Math.min(100, Math.max(1, configuration.mean * 100))}%`,
                            }}
                            role="img"
                            aria-label={`${configuration.label}: mean ${formatMetric(configuration.mean)}`}
                          />
                        </div>
                        <p className="mt-1.5 text-right font-mono text-[9px] text-[#5f6d83]">
                          ± {formatMetric(configuration.standardDeviation)} · n={configuration.count}
                        </p>
                      </div>
                    ))}
                  </div>
                </section>

                <section className="surface min-w-0 overflow-hidden">
                  <div className="flex items-start justify-between gap-3 border-b border-white/[0.07] px-5 py-5 sm:px-6">
                    <div>
                      <p className="text-sm font-semibold text-white">Seed-aware leaderboard</p>
                      <p className="mt-1 text-[11px] text-[#748198]">
                        Same condition and method are combined; seed channels are excluded from grouping.
                      </p>
                    </div>
                    <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#9b8cff]/10 text-[#b3a7ff]">
                      <Sigma size={17} />
                    </span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[760px] border-collapse text-left">
                      <thead>
                        <tr className="border-b border-white/[0.06] text-[9px] font-semibold uppercase tracking-[0.13em] text-[#68758a]">
                          <th className="px-5 py-3 sm:px-6">Rank</th>
                          <th className="px-3 py-3">Configuration</th>
                          <th className="px-3 py-3 text-right">Runs</th>
                          <th className="px-3 py-3 text-right">Seeds</th>
                          <th className="px-3 py-3 text-right">Mean</th>
                          <th className="px-3 py-3 text-right">Std.</th>
                          <th className="px-5 py-3 text-right sm:px-6">Best</th>
                        </tr>
                      </thead>
                      <tbody>
                        {configurations.map((configuration, index) => (
                          <tr
                            key={configuration.key}
                            className="border-b border-white/[0.045] text-xs last:border-0 hover:bg-white/[0.018]"
                          >
                            <td className="px-5 py-3.5 font-mono text-[#637087] sm:px-6">
                              {index === 0 ? (
                                <span className="inline-flex h-6 w-6 items-center justify-center rounded-lg bg-[#b8e56e]/10 text-[#c5ec82]">
                                  <Check size={13} />
                                </span>
                              ) : (
                                String(index + 1).padStart(2, "0")
                              )}
                            </td>
                            <td className="px-3 py-3.5">
                              <p className="font-semibold text-[#e0e5ef]">{configuration.label}</p>
                              <p className="mt-1 max-w-[290px] truncate text-[10px] text-[#647188]">
                                {configuration.details}
                              </p>
                            </td>
                            <td className="px-3 py-3.5 text-right font-mono text-[#8d99ad]">
                              {configuration.count}
                            </td>
                            <td className="px-3 py-3.5 text-right font-mono text-[#8d99ad]">
                              {configuration.seedCount}
                            </td>
                            <td className="px-3 py-3.5 text-right font-mono font-semibold text-white">
                              {formatMetric(configuration.mean)}
                            </td>
                            <td className="px-3 py-3.5 text-right font-mono text-[#8d99ad]">
                              {formatMetric(configuration.standardDeviation)}
                            </td>
                            <td className="px-5 py-3.5 text-right font-mono text-[#74dce8] sm:px-6">
                              {formatMetric(configuration.maximum)}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              </div>
            )}

            <section className="surface mt-5 min-w-0 overflow-hidden">
              <div className="flex items-start justify-between gap-3 border-b border-white/[0.07] px-5 py-5 sm:px-6">
                <div>
                  <p className="text-sm font-semibold text-white">Individual final runs</p>
                  <p className="mt-1 text-[11px] text-[#748198]">
                    Ranked by {TEST_METRICS[metric]}; validation metrics are context only.
                  </p>
                </div>
                <span className="rounded-lg border border-white/[0.07] px-2 py-1 font-mono text-[10px] text-[#7f8ca1]">
                  {Math.min(sortedRuns.length, 250)} / {sortedRuns.length}
                </span>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[1050px] border-collapse text-left">
                  <thead>
                    <tr className="border-b border-white/[0.06] text-[9px] font-semibold uppercase tracking-[0.13em] text-[#68758a]">
                      <th className="px-5 py-3 sm:px-6">Dataset / condition</th>
                      <th className="px-3 py-3">Method</th>
                      <th className="px-3 py-3">Budget</th>
                      <th className="px-3 py-3 text-right">Seed</th>
                      <th className="px-3 py-3 text-right">Trial</th>
                      <th className="px-3 py-3 text-right">Valid mAP@R</th>
                      <th className="px-3 py-3 text-right">Test P@1</th>
                      <th className="px-5 py-3 text-right sm:px-6">Test mAP@R</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedRuns.slice(0, 250).map((result) => (
                      <tr
                        key={result.id}
                        className="border-b border-white/[0.045] text-xs last:border-0 hover:bg-white/[0.018]"
                      >
                        <td className="max-w-[330px] px-5 py-3.5 sm:px-6">
                          <p className="font-semibold text-[#dce2ed]">{result.dataset}</p>
                          <p className="mt-1 truncate text-[10px] text-[#617087]" title={result.scenario}>
                            {result.scenario}
                          </p>
                        </td>
                        <td className="px-3 py-3.5">
                          <p className="font-semibold text-[#cbd3e1]">{humanize(result.loss)}</p>
                          <p className="mt-1 text-[10px] text-[#647188]">
                            {humanize(result.mode)}
                            {result.sslMethod.toLowerCase() !== "none"
                              ? ` · ${humanize(result.sslMethod)}`
                              : ""}
                          </p>
                        </td>
                        <td className="px-3 py-3.5 text-[#8995aa]">{labelBudget(result)}</td>
                        <td className="px-3 py-3.5 text-right font-mono text-[#8995aa]">
                          {result.seed}
                        </td>
                        <td className="px-3 py-3.5 text-right font-mono text-[#8995aa]">
                          {result.trialNumber}
                        </td>
                        <td className="px-3 py-3.5 text-right font-mono text-[#7e8ba0]">
                          {formatMetric(result.validMapAtR)}
                        </td>
                        <td className="px-3 py-3.5 text-right font-mono text-[#aab4c5]">
                          {formatMetric(result.testPrecisionAt1)}
                        </td>
                        <td className="px-5 py-3.5 text-right font-mono font-semibold text-white sm:px-6">
                          {formatMetric(result.testMapAtR)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              {sortedRuns.length > 250 && (
                <p className="border-t border-white/[0.06] px-6 py-3 text-[10px] text-[#68758b]">
                  Showing the top 250 matching runs. Narrow the filters to inspect the remainder.
                </p>
              )}
            </section>
          </>
        )}
      </main>

      {dragging && (
        <div className="pointer-events-none fixed inset-0 z-[90] flex items-center justify-center bg-[#080d18]/85 p-6 backdrop-blur-sm">
          <div className="flex w-full max-w-lg flex-col items-center rounded-3xl border border-dashed border-[#9b8cff]/50 bg-[#111a2d] px-8 py-16 text-center shadow-2xl">
            <span className="flex h-14 w-14 items-center justify-center rounded-2xl bg-[#9b8cff]/12 text-[#b5aaff]">
              <UploadCloud size={25} />
            </span>
            <p className="mt-5 text-base font-semibold text-white">Drop test-results.csv</p>
            <p className="mt-2 text-xs text-[#7f8ca2]">The current comparison will be replaced.</p>
          </div>
        </div>
      )}
    </div>
  );
}
