"use client";

import Link from "next/link";
import {
  ArrowUpRight,
  BarChart3,
  Check,
  CircleGauge,
  FileSpreadsheet,
  Layers3,
  Medal,
  MousePointer2,
  Rows3,
  Sigma,
  Trophy,
  UploadCloud,
} from "lucide-react";
import {
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
  type ReactNode,
} from "react";
import { ChartBoard } from "@/components/ChartBoard";
import {
  TEST_METRICS,
  formatCompact,
  formatMetric,
  parseTestResultsCsv,
  summarizeConfigurations,
  testConfigurationKey,
  type ConfigurationSummary,
  type TestMetric,
  type TestResult,
  type TestResultFile,
} from "@/lib/test-results";

const STORAGE_KEY = "trial-atlas:final-test-comparison";
const COLORS = [
  "#9b8cff",
  "#4fd5e7",
  "#b8e56e",
  "#f4c86f",
  "#f58ba3",
  "#72a7ff",
  "#d79cff",
  "#64dfb4",
  "#ff9b73",
  "#b7c4e2",
  "#e1ef69",
  "#7fd1ff",
];
const WIDTH = 960;
const GRID = "#27334c";
const AXIS = "#77859f";
const TEXT = "#dce3f1";
const MUTED = "#8f9bb2";

interface LoadedResults {
  results: TestResult[];
  error: string | null;
}

function loadResults(file: TestResultFile | null): LoadedResults {
  if (!file) return { results: [], error: null };
  try {
    return { results: parseTestResultsCsv(file.text), error: null };
  } catch (error) {
    return {
      results: [],
      error:
        error instanceof Error
          ? error.message
          : "The final-test aggregate could not be parsed.",
    };
  }
}

function linearScale(domain: [number, number], range: [number, number]) {
  const span = domain[1] - domain[0] || 1;
  return (value: number) =>
    range[0] + ((value - domain[0]) / span) * (range[1] - range[0]);
}

function ticks(domain: [number, number], count = 5): number[] {
  return Array.from(
    { length: count },
    (_, index) => domain[0] + ((domain[1] - domain[0]) * index) / (count - 1),
  );
}

function shortLabel(value: string, maximum = 34): string {
  return value.length <= maximum ? value : `${value.slice(0, maximum - 1)}…`;
}

function EmptyChart({ label }: { label: string }) {
  return (
    <svg
      data-export-root
      viewBox={`0 0 ${WIDTH} 330`}
      className="block w-full"
      role="img"
      aria-label={label}
    >
      <title>{label}</title>
      <rect
        x="240"
        y="110"
        width="480"
        height="108"
        rx="18"
        fill="rgba(255,255,255,0.025)"
        stroke={GRID}
      />
      <text x="480" y="156" fill={TEXT} fontSize="15" textAnchor="middle">
        Select at least one final configuration
      </text>
      <text x="480" y="183" fill={MUTED} fontSize="12" textAnchor="middle">
        The final-test comparison will update with the selection above.
      </text>
    </svg>
  );
}

function FinalLeaderboardChart({
  configurations,
  metric,
}: {
  configurations: ConfigurationSummary[];
  metric: TestMetric;
}) {
  if (configurations.length === 0) {
    return <EmptyChart label={`${TEST_METRICS[metric]} leaderboard`} />;
  }

  const height = Math.max(320, 92 + configurations.length * 54);
  const left = 260;
  const right = 84;
  const top = 26;
  const bottom = 50;
  const maximum = Math.max(...configurations.map((configuration) => configuration.maximum));
  const domain: [number, number] = [0, Math.max(0.01, maximum * 1.08)];
  const x = linearScale(domain, [left, WIDTH - right]);
  const rowHeight = (height - top - bottom) / configurations.length;

  return (
    <svg
      data-export-root
      viewBox={`0 0 ${WIDTH} ${height}`}
      className="block w-full"
      role="img"
      aria-label={`${TEST_METRICS[metric]} mean and range by final configuration`}
    >
      <title>{TEST_METRICS[metric]} mean and range by final configuration</title>
      {ticks(domain, 6).map((value) => (
        <g key={value} aria-hidden="true">
          <line
            x1={x(value)}
            x2={x(value)}
            y1={top}
            y2={height - bottom}
            stroke={GRID}
          />
          <text
            x={x(value)}
            y={height - 20}
            fill={AXIS}
            fontSize="11"
            textAnchor="middle"
          >
            {value.toFixed(3)}
          </text>
        </g>
      ))}
      {configurations.map((configuration, index) => {
        const centerY = top + rowHeight * index + rowHeight / 2;
        const color = COLORS[index % COLORS.length];
        return (
          <g key={configuration.key}>
            <text
              x={left - 16}
              y={centerY - 3}
              fill={TEXT}
              fontSize="12"
              textAnchor="end"
            >
              {shortLabel(configuration.label)}
            </text>
            <text
              x={left - 16}
              y={centerY + 13}
              fill={MUTED}
              fontSize="9"
              textAnchor="end"
            >
              {shortLabel(configuration.dataset, 24)} · n={configuration.count}
            </text>
            <line
              x1={x(configuration.minimum)}
              x2={x(configuration.maximum)}
              y1={centerY}
              y2={centerY}
              stroke={color}
              strokeWidth="2"
              strokeOpacity="0.72"
            />
            <line
              x1={x(configuration.minimum)}
              x2={x(configuration.minimum)}
              y1={centerY - 6}
              y2={centerY + 6}
              stroke={color}
              strokeWidth="2"
            />
            <line
              x1={x(configuration.maximum)}
              x2={x(configuration.maximum)}
              y1={centerY - 6}
              y2={centerY + 6}
              stroke={color}
              strokeWidth="2"
            />
            <circle
              cx={x(configuration.mean)}
              cy={centerY}
              r={index === 0 ? 8 : 6}
              fill={color}
              stroke="#10182a"
              strokeWidth="3"
            >
              <title>
                {configuration.label} · mean {formatMetric(configuration.mean)} · range{" "}
                {formatMetric(configuration.minimum)}–{formatMetric(configuration.maximum)}
              </title>
            </circle>
            <text
              x={Math.min(WIDTH - 54, x(configuration.maximum) + 10)}
              y={centerY + 4}
              fill={index === 0 ? color : TEXT}
              fontSize="11"
              fontWeight="700"
            >
              {formatMetric(configuration.mean)}
            </text>
          </g>
        );
      })}
    </svg>
  );
}

function StabilityChart({
  configurations,
  metric,
}: {
  configurations: ConfigurationSummary[];
  metric: TestMetric;
}) {
  if (configurations.length === 0) {
    return <EmptyChart label={`${TEST_METRICS[metric]} stability`} />;
  }

  const left = 72;
  const right = 38;
  const top = 30;
  const bottom = 58;
  const height = 360;
  const means = configurations.map((configuration) => configuration.mean);
  const deviations = configurations.map((configuration) => configuration.standardDeviation);
  const minMean = Math.min(...means);
  const maxMean = Math.max(...means);
  const meanPadding = Math.max(0.002, (maxMean - minMean) * 0.12);
  const xDomain: [number, number] =
    minMean === maxMean
      ? [Math.max(0, minMean - 0.02), minMean + 0.02]
      : [Math.max(0, minMean - meanPadding), maxMean + meanPadding];
  const maxDeviation = Math.max(...deviations);
  const yDomain: [number, number] = [0, Math.max(0.005, maxDeviation * 1.18)];
  const x = linearScale(xDomain, [left, WIDTH - right]);
  const y = linearScale(yDomain, [height - bottom, top]);

  return (
    <svg
      data-export-root
      viewBox={`0 0 ${WIDTH} ${height}`}
      className="block w-full"
      role="img"
      aria-label={`${TEST_METRICS[metric]} mean versus standard deviation`}
    >
      <title>{TEST_METRICS[metric]} mean versus standard deviation</title>
      {ticks(xDomain, 6).map((value) => (
        <g key={`x-${value}`} aria-hidden="true">
          <line
            x1={x(value)}
            x2={x(value)}
            y1={top}
            y2={height - bottom}
            stroke={GRID}
          />
          <text
            x={x(value)}
            y={height - 28}
            fill={AXIS}
            fontSize="11"
            textAnchor="middle"
          >
            {value.toFixed(3)}
          </text>
        </g>
      ))}
      {ticks(yDomain, 5).map((value) => (
        <g key={`y-${value}`} aria-hidden="true">
          <line
            x1={left}
            x2={WIDTH - right}
            y1={y(value)}
            y2={y(value)}
            stroke={GRID}
          />
          <text
            x={left - 12}
            y={y(value) + 4}
            fill={AXIS}
            fontSize="11"
            textAnchor="end"
          >
            {value.toFixed(3)}
          </text>
        </g>
      ))}
      {configurations.map((configuration, index) => {
        const color = COLORS[index % COLORS.length];
        const radius = Math.min(14, 5 + Math.sqrt(configuration.count) * 2);
        return (
          <circle
            key={configuration.key}
            cx={x(configuration.mean)}
            cy={y(configuration.standardDeviation)}
            r={radius}
            fill={color}
            fillOpacity="0.72"
            stroke={color}
            strokeWidth="2"
            strokeOpacity="0.95"
          >
            <title>
              {configuration.label} · mean {formatMetric(configuration.mean)} · std.{" "}
              {formatMetric(configuration.standardDeviation)} · {configuration.count} runs
            </title>
          </circle>
        );
      })}
      <text
        x={left + (WIDTH - left - right) / 2}
        y={height - 7}
        fill={MUTED}
        fontSize="12"
        textAnchor="middle"
      >
        Mean {TEST_METRICS[metric]}
      </text>
      <text
        x="17"
        y={top + (height - top - bottom) / 2}
        fill={MUTED}
        fontSize="12"
        textAnchor="middle"
        transform={`rotate(-90 17 ${top + (height - top - bottom) / 2})`}
      >
        Standard deviation
      </text>
    </svg>
  );
}

function MetricCard({
  label,
  value,
  context,
  icon,
}: {
  label: string;
  value: string;
  context: string;
  icon: ReactNode;
}) {
  return (
    <article className="surface relative overflow-hidden px-5 py-5">
      <div className="absolute -right-8 -top-10 h-24 w-24 rounded-full bg-[#4fd5e7]/[0.05] blur-2xl" />
      <div className="relative flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[#77849c]">
            {label}
          </p>
          <p className="mt-3 truncate text-2xl font-semibold tracking-[-0.035em] text-white">
            {value}
          </p>
          <p className="mt-1.5 truncate text-[11px] text-[#7f8ca3]" title={context}>
            {context}
          </p>
        </div>
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#4fd5e7]/10 text-[#70e1ec] ring-1 ring-[#4fd5e7]/15">
          {icon}
        </span>
      </div>
    </article>
  );
}

export function FinalTestComparison({
  initialFile,
}: {
  initialFile: TestResultFile | null;
}) {
  const initial = useMemo(() => loadResults(initialFile), [initialFile]);
  const [results, setResults] = useState(initial.results);
  const [fileName, setFileName] = useState(initialFile?.name ?? null);
  const [error, setError] = useState(initial.error);
  const [metric, setMetric] = useState<TestMetric>(
    "test_mean_average_precision_at_r",
  );
  const [selectedKeys, setSelectedKeys] = useState<string[]>([]);
  const [storageReady, setStorageReady] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);
  const configurations = useMemo(
    () => summarizeConfigurations(results, metric),
    [metric, results],
  );
  const configurationSignature = configurations
    .map((configuration) => configuration.key)
    .join("\u001e");

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(STORAGE_KEY);
      const parsed = raw
        ? (JSON.parse(raw) as { selectedKeys?: unknown; metric?: unknown })
        : null;
      const validKeys = new Set(configurations.map((configuration) => configuration.key));
      const storedKeys = Array.isArray(parsed?.selectedKeys)
        ? parsed.selectedKeys.filter(
            (key): key is string => typeof key === "string" && validKeys.has(key),
          )
        : null;
      setSelectedKeys(storedKeys ?? [...validKeys]);
      if (
        parsed?.metric === "test_mean_average_precision_at_r" ||
        parsed?.metric === "test_precision_at_1"
      ) {
        setMetric(parsed.metric);
      }
    } catch {
      setSelectedKeys(configurations.map((configuration) => configuration.key));
    }
    setStorageReady(true);
    // This component remounts when the comparison source changes.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!storageReady) return;
    const validKeys = new Set(configurations.map((configuration) => configuration.key));
    setSelectedKeys((current) => current.filter((key) => validKeys.has(key)));
  }, [configurationSignature, configurations, storageReady]);

  useEffect(() => {
    if (!storageReady) return;
    try {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify({ selectedKeys, metric }),
      );
    } catch {
      // Keep the active comparison available for this browser session.
    }
  }, [metric, selectedKeys, storageReady]);

  const selectedSet = useMemo(() => new Set(selectedKeys), [selectedKeys]);
  const selectedConfigurations = configurations.filter((configuration) =>
    selectedSet.has(configuration.key),
  );
  const leader = selectedConfigurations[0] ?? null;
  const runnerUp = selectedConfigurations[1] ?? null;
  const advantage =
    leader && runnerUp ? leader.mean - runnerUp.mean : null;
  const selectedRunCount = selectedConfigurations.reduce(
    (sum, configuration) => sum + configuration.count,
    0,
  );
  const selectedSeedCount = new Set(
    results
      .filter((result) => selectedSet.has(testConfigurationKey(result)))
      .map((result) => result.seed),
  ).size;
  const selectedDatasets = new Set(
    selectedConfigurations.map((configuration) => configuration.dataset),
  );

  const toggleConfiguration = (key: string) => {
    setSelectedKeys((current) =>
      current.includes(key)
        ? current.filter((candidate) => candidate !== key)
        : [...current, key],
    );
  };

  const acceptFile = async (file: File | null) => {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Choose the test-results.csv aggregate.");
      return;
    }
    try {
      const parsed = parseTestResultsCsv(await file.text());
      const nextConfigurations = summarizeConfigurations(parsed, metric);
      setResults(parsed);
      setSelectedKeys(nextConfigurations.map((configuration) => configuration.key));
      setFileName(file.name);
      setError(null);
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : "The final-test aggregate could not be parsed.",
      );
    }
  };

  const onFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    void acceptFile(event.target.files?.[0] ?? null);
    event.target.value = "";
  };

  if (results.length === 0) {
    return (
      <section className="surface flex min-h-[480px] items-center justify-center px-6 py-12 text-center">
        <div className="max-w-xl">
          <span className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-[#4fd5e7]/10 text-[#70e1ec]">
            <FileSpreadsheet size={24} />
          </span>
          <h2 className="mt-5 text-xl font-semibold tracking-[-0.025em] text-white">
            {fileName
              ? "No completed final tests in this aggregate"
              : "Load the server’s final-test aggregate"}
          </h2>
          <p className="mx-auto mt-3 text-sm leading-6 text-[#8794aa]">
            Export the server logs to test-results.csv, then rebuild Trial Atlas or
            open the aggregate here. The HPO comparison remains available in the
            other source tab.
          </p>
          {error && (
            <p className="mt-4 rounded-xl border border-[#ff7b7b]/20 bg-[#ff7b7b]/[0.06] px-4 py-3 text-xs text-[#ffaaaa]">
              {error}
            </p>
          )}
          <button
            type="button"
            onClick={() => fileInput.current?.click()}
            className="focus-ring mt-6 inline-flex h-10 items-center gap-2 rounded-xl bg-[#4fd5e7] px-4 text-xs font-semibold text-[#07131d] transition hover:bg-[#6de3ee]"
          >
            <UploadCloud size={15} />
            Open test-results.csv
          </button>
          <input
            ref={fileInput}
            type="file"
            accept=".csv,text/csv"
            className="sr-only"
            onChange={onFileChange}
          />
        </div>
      </section>
    );
  }

  return (
    <div className="fade-up space-y-6">
      <section className="relative overflow-hidden rounded-[1.5rem] border border-[#4fd5e7]/15 bg-[linear-gradient(135deg,rgba(79,213,231,0.1),rgba(155,140,255,0.04)_48%,rgba(16,24,42,0.82))] px-5 py-6 shadow-[0_24px_70px_rgba(0,0,0,.2)] sm:px-7 sm:py-7">
        <div className="absolute -right-20 -top-24 h-64 w-64 rounded-full bg-[#4fd5e7]/10 blur-3xl" />
        <div className="relative flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
          <div>
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.17em] text-[#65dce8]">
              <CircleGauge size={13} /> Held-out comparison
            </div>
            <h1 className="mt-3 text-3xl font-semibold tracking-[-0.045em] text-white sm:text-4xl">
              Compare what generalized.
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-[#909db2]">
              Equivalent final configurations are grouped across runs and seeds.
              Ranking uses only the selected held-out test metric.
            </p>
            <p className="mt-2 text-[10px] text-[#67758c]">
              {fileName} · {results.length} final runs
            </p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1 rounded-2xl border border-white/[0.08] bg-black/15 p-1.5">
              {(Object.keys(TEST_METRICS) as TestMetric[]).map((option) => (
                <button
                  key={option}
                  type="button"
                  onClick={() => setMetric(option)}
                  aria-pressed={metric === option}
                  className={`focus-ring rounded-xl px-3 py-2 text-[11px] font-semibold transition ${
                    metric === option
                      ? "bg-[#4fd5e7] text-[#07131d]"
                      : "text-[#8f9bb0] hover:bg-white/[0.05] hover:text-white"
                  }`}
                >
                  {TEST_METRICS[option]}
                </button>
              ))}
            </div>
            <button
              type="button"
              onClick={() => fileInput.current?.click()}
              className="focus-ring flex h-10 items-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.035] px-3 text-[11px] font-semibold text-[#d7deea] hover:bg-white/[0.07]"
            >
              <UploadCloud size={14} />
              Replace CSV
            </button>
            <Link
              href="/test-results/"
              className="focus-ring flex h-10 items-center gap-2 rounded-xl px-3 text-[11px] font-semibold text-[#78dfe9] hover:bg-[#4fd5e7]/10"
            >
              Full explorer <ArrowUpRight size={13} />
            </Link>
            <input
              ref={fileInput}
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              onChange={onFileChange}
            />
          </div>
        </div>
      </section>

      {error && (
        <div
          role="alert"
          className="rounded-xl border border-[#ff7b7b]/20 bg-[#ff7b7b]/[0.06] px-4 py-3 text-xs text-[#ffaaaa]"
        >
          {error}
        </div>
      )}

      <section className="surface overflow-hidden">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.07] px-5 py-4 sm:px-6">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-white">
                Final configurations in comparison
              </h2>
              <span className="rounded-full bg-[#4fd5e7]/10 px-2 py-0.5 text-[10px] font-semibold text-[#76e1eb]">
                {selectedConfigurations.length} / {configurations.length}
              </span>
            </div>
            <p className="mt-1.5 text-xs text-[#8290a7]">
              Runs with the same dataset, method, label budget, and training mode
              are summarized together.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() =>
                setSelectedKeys(configurations.map((configuration) => configuration.key))
              }
              className="focus-ring rounded-lg px-2.5 py-1.5 text-[11px] font-semibold text-[#8edfe7] hover:bg-white/[0.05] hover:text-white"
            >
              Select all
            </button>
            <span className="h-4 w-px bg-white/[0.09]" />
            <button
              type="button"
              onClick={() => setSelectedKeys([])}
              className="focus-ring rounded-lg px-2.5 py-1.5 text-[11px] font-semibold text-[#7f8da4] hover:bg-white/[0.05] hover:text-white"
            >
              Clear
            </button>
          </div>
        </header>
        <div className="grid gap-2.5 p-4 sm:grid-cols-2 sm:p-5 xl:grid-cols-3">
          {configurations.map((configuration, index) => {
            const selected = selectedSet.has(configuration.key);
            const color = COLORS[index % COLORS.length];
            return (
              <button
                key={configuration.key}
                type="button"
                onClick={() => toggleConfiguration(configuration.key)}
                aria-pressed={selected}
                className={`focus-ring group flex min-w-0 items-center gap-3 rounded-xl border px-3.5 py-3 text-left transition ${
                  selected
                    ? "border-white/[0.12] bg-white/[0.055]"
                    : "border-transparent bg-white/[0.018] opacity-55 hover:opacity-90"
                }`}
              >
                <span
                  className="flex h-8 w-8 shrink-0 items-center justify-center rounded-xl"
                  style={{
                    color,
                    backgroundColor: `${color}18`,
                    boxShadow: selected ? `inset 0 0 0 1px ${color}40` : undefined,
                  }}
                >
                  {selected ? <Check size={15} strokeWidth={2.5} /> : <Layers3 size={14} />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs font-semibold text-[#dce3ee]">
                    {configuration.label}
                  </span>
                  <span className="mt-1 block truncate text-[10px] text-[#6e7c94]">
                    {configuration.details} · mean {formatMetric(configuration.mean)}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      </section>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label={`${TEST_METRICS[metric]} leader`}
          value={formatMetric(leader?.mean)}
          context={leader?.label ?? "Select final configurations"}
          icon={<Trophy size={17} />}
        />
        <MetricCard
          label="Lead over #2"
          value={advantage == null ? "—" : `+${formatMetric(advantage)}`}
          context={runnerUp ? `Ahead of ${runnerUp.label}` : "Choose at least two configurations"}
          icon={<Medal size={17} />}
        />
        <MetricCard
          label="Final runs compared"
          value={formatCompact(selectedRunCount)}
          context={`Across ${selectedConfigurations.length} grouped configurations`}
          icon={<Rows3 size={17} />}
        />
        <MetricCard
          label="Seed coverage"
          value={formatCompact(selectedSeedCount)}
          context={`${selectedDatasets.size} datasets represented`}
          icon={<Sigma size={17} />}
        />
      </div>

      {selectedConfigurations.length === 0 ? (
        <section className="surface flex min-h-72 items-center justify-center px-6 py-10 text-center">
          <div>
            <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-[#4fd5e7]/10 text-[#70e1ec]">
              <MousePointer2 size={21} />
            </span>
            <h2 className="mt-4 text-lg font-semibold text-white">
              Choose final configurations to compare
            </h2>
            <p className="mx-auto mt-2 max-w-md text-xs leading-5 text-[#7f8ca2]">
              Select any configuration above. The held-out charts and scorecard
              update together.
            </p>
          </div>
        </section>
      ) : (
        <>
          {selectedDatasets.size > 1 && (
            <div className="flex items-start gap-3 rounded-xl border border-[#f4c86f]/15 bg-[#f4c86f]/[0.055] px-4 py-3 text-xs text-[#c8b889]">
              <CircleGauge size={16} className="mt-0.5 shrink-0 text-[#f4c86f]" />
              <p className="leading-5">
                Multiple datasets are selected. Their test scores share a scale but
                do not measure identical tasks, so compare within a dataset when
                drawing method conclusions.
              </p>
            </div>
          )}

          <ChartBoard
            storageKey="trial-atlas:charts:final-test-comparison"
            items={[
              {
                id: "final-test-leaderboard",
                title: `${TEST_METRICS[metric]} leaderboard`,
                description:
                  "Dots show configuration means; whiskers span the observed final-run range.",
                exportName: `final-test-${metric}-leaderboard`,
                content: (
                  <FinalLeaderboardChart
                    configurations={selectedConfigurations}
                    metric={metric}
                  />
                ),
              },
              {
                id: "final-test-stability",
                title: "Generalization vs. stability",
                description:
                  "Farther right is stronger; lower is more consistent across final runs and seeds.",
                exportName: `final-test-${metric}-stability`,
                content: (
                  <StabilityChart
                    configurations={selectedConfigurations}
                    metric={metric}
                  />
                ),
              },
            ]}
          />

          <section className="surface overflow-hidden">
            <header className="flex flex-wrap items-start justify-between gap-3 border-b border-white/[0.07] px-5 py-4 sm:px-6">
              <div>
                <div className="flex items-center gap-2">
                  <BarChart3 size={14} className="text-[#4fd5e7]" />
                  <h2 className="text-[15px] font-semibold text-white">
                    Final-test scorecard
                  </h2>
                </div>
                <p className="mt-1.5 text-xs text-[#8290a7]">
                  Ranked by mean {TEST_METRICS[metric]} across matching final runs.
                </p>
              </div>
              <Link
                href="/test-results/"
                className="focus-ring inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 text-[11px] font-semibold text-[#70dce7] hover:bg-[#4fd5e7]/10"
              >
                Inspect individual runs <ArrowUpRight size={13} />
              </Link>
            </header>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[980px] text-left text-xs">
                <thead className="border-b border-white/[0.07] bg-black/10 text-[10px] uppercase tracking-[0.11em] text-[#6f7d95]">
                  <tr>
                    <th className="px-5 py-3.5 text-center font-semibold">Rank</th>
                    <th className="px-4 py-3.5 font-semibold">Configuration</th>
                    <th className="px-4 py-3.5 font-semibold">Runs</th>
                    <th className="px-4 py-3.5 font-semibold">Seeds</th>
                    <th className="px-4 py-3.5 font-semibold">Mean</th>
                    <th className="px-4 py-3.5 font-semibold">Std. dev.</th>
                    <th className="px-4 py-3.5 font-semibold">Minimum</th>
                    <th className="px-5 py-3.5 font-semibold">Maximum</th>
                  </tr>
                </thead>
                <tbody>
                  {selectedConfigurations.map((configuration, index) => (
                    <tr
                      key={configuration.key}
                      className="border-b border-white/[0.055] text-[#aeb8ca] last:border-0 hover:bg-white/[0.025]"
                    >
                      <td className="px-5 py-4 text-center">
                        <span
                          className="inline-flex h-6 min-w-6 items-center justify-center rounded-lg px-1.5 text-[10px] font-bold"
                          style={{
                            color: COLORS[index % COLORS.length],
                            backgroundColor: `${COLORS[index % COLORS.length]}16`,
                          }}
                        >
                          {index + 1}
                        </span>
                      </td>
                      <td className="max-w-96 px-4 py-4">
                        <span className="block truncate font-semibold text-white">
                          {configuration.label}
                        </span>
                        <span className="mt-1 block truncate text-[10px] text-[#6d7a92]">
                          {configuration.details}
                        </span>
                      </td>
                      <td className="px-4 py-4 tabular-nums">{configuration.count}</td>
                      <td className="px-4 py-4 tabular-nums">
                        {configuration.seedCount}
                      </td>
                      <td className="px-4 py-4 font-semibold tabular-nums text-[#80e1eb]">
                        {formatMetric(configuration.mean)}
                      </td>
                      <td className="px-4 py-4 tabular-nums">
                        {formatMetric(configuration.standardDeviation)}
                      </td>
                      <td className="px-4 py-4 tabular-nums">
                        {formatMetric(configuration.minimum)}
                      </td>
                      <td className="px-5 py-4 tabular-nums">
                        {formatMetric(configuration.maximum)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        </>
      )}
    </div>
  );
}
