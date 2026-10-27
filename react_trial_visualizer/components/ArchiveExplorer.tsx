"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  Activity,
  ArrowDownWideNarrow,
  BarChart3,
  Boxes,
  ChevronDown,
  ClipboardCopy,
  Clock3,
  Database,
  Download,
  Filter,
  Grid3x3,
  Hourglass,
  Layers3,
  Rows3,
  RefreshCw,
  Ruler,
  Search,
  Sigma,
  SlidersHorizontal,
  Sparkles,
  Waves,
  Table2,
  Trophy,
  X,
} from "lucide-react";
import { downloadText } from "@/lib/export";
import { formatDuration, formatMetric, formatNumber, slugify } from "@/lib/trials";
import {
  DEFAULT_RANGE_SETTINGS,
  GRADIENT_LOG_SERIES,
  GRADIENT_SERIES_LABELS,
  VERDICT_LABELS,
  assessStudy,
  availableGradientSeries,
  collectGradientPoints,
  collectSamples,
  estimateStartupTrials,
  findRange,
  foldBreakdown,
  gradientFoldRows,
  mixedFoldCounts,
  summariseEpochs,
  toAxis,
  toSpaces,
  type EpochBudget,
  type FoldRow,
  type GradientFoldRow,
  type GradientIndex,
  type GradientSeriesKey,
  type GradientStatistic,
  type NumericFinding,
  type ParameterSample,
  type StartupSummary,
  type RangeFinding,
  type RangeSettings,
  type StudyAssessment,
  type TrialHealth,
  type Verdict,
} from "@/lib/phase0";
import {
  foldValues,
  isArchiveSnapshot,
  isGradientSnapshot,
  scoredTrials,
  type ArchiveParamValue,
  type ArchiveSnapshot,
  type ArchiveStudy,
  type ArchiveTrial,
} from "@/lib/sorted-studies";

const SNAPSHOT_URL = "/sorted-studies.json";
const GRADIENT_URL = "/sorted-gradients.json";
const FILTER_STORAGE_KEY = "trial-atlas:archive-filters";
const STUDY_STORAGE_KEY = "trial-atlas:archive-study";

const inputClass =
  "focus-ring h-9 w-full rounded-xl border border-white/[0.09] bg-[#0c1425] px-3 text-xs text-[#d9e0ed] placeholder:text-[#53617a]";

const CHART = {
  grid: "#27334c",
  axis: "#77859f",
  text: "#dce3f1",
  muted: "#8f9bb2",
  accent: "#9b8cff",
  faint: "rgba(155, 140, 255, 0.22)",
};

/**
 * Categorical identity palette, validated for this dark surface (#10182a):
 * every adjacent pair clears the CVD and normal-vision separation gates. Slots
 * are handed out in this fixed order and never cycled — a ninth group folds
 * into "Other" instead. Series are always direct-labelled as well, so identity
 * never rests on hue alone.
 */
const SERIES_COLORS = [
  "#3987e5",
  "#d95926",
  "#199e70",
  "#c98500",
  "#d55181",
  "#008300",
  "#9085e9",
  "#e66767",
];
const OTHER_COLOR = "#6b7890";

/**
 * Single-hue magnitude ramp. On a dark surface the near-zero end is the step
 * that recedes into the background, so the ramp runs dark → light as the value
 * grows.
 */
const MAGNITUDE_RAMP = [
  "#0d366b",
  "#184f95",
  "#256abf",
  "#3987e5",
  "#6da7ec",
  "#9ec5f4",
  "#cde2fb",
];

const MAX_SERIES = SERIES_COLORS.length;
const MAX_SCATTER_POINTS = 3_500;
const TRIAL_PAGE = 250;

// ------------------------------------------------------------------ utilities

function ascending(values: number[]): number[] {
  return [...values].sort((left, right) => left - right);
}

function quantile(sorted: number[], probability: number): number {
  if (sorted.length === 0) return Number.NaN;
  if (sorted.length === 1) return sorted[0];
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - lower);
}

function median(values: number[]): number {
  return quantile(ascending(values), 0.5);
}

function mean(values: number[]): number {
  return values.length === 0
    ? Number.NaN
    : values.reduce((total, value) => total + value, 0) / values.length;
}

function standardDeviation(values: number[]): number {
  if (values.length < 2) return Number.NaN;
  const average = mean(values);
  const variance =
    values.reduce((total, value) => total + (value - average) ** 2, 0) /
    (values.length - 1);
  return Math.sqrt(variance);
}

function finite(value: unknown): number | null {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function extent(values: number[], padding = 0.06): [number, number] {
  if (values.length === 0) return [0, 1];
  let low = Math.min(...values);
  let high = Math.max(...values);
  if (low === high) {
    const nudge = Math.abs(low) > 1e-9 ? Math.abs(low) * 0.08 : 0.5;
    low -= nudge;
    high += nudge;
  }
  const span = (high - low) * padding;
  return [low - span, high + span];
}

function linearScale(domain: [number, number], range: [number, number]) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0 || 1;
  return (value: number) => r0 + ((value - d0) / span) * (r1 - r0);
}

function ticks(domain: [number, number], count: number): number[] {
  const [low, high] = domain;
  const step = (high - low) / Math.max(1, count - 1);
  return Array.from({ length: count }, (_, index) => low + index * step);
}

function axisNumber(value: number): string {
  if (!Number.isFinite(value)) return "—";
  const absolute = Math.abs(value);
  if (absolute !== 0 && (absolute < 0.001 || absolute >= 100_000)) {
    return value.toExponential(1);
  }
  if (absolute >= 100) return value.toFixed(0);
  if (absolute >= 1) return value.toFixed(2);
  return value.toFixed(3);
}

function signedMetric(value: number | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const text = value.toFixed(4);
  return value > 0 ? `+${text}` : text;
}

function rampColor(position: number): string {
  if (!Number.isFinite(position)) return "#1b2438";
  const clamped = Math.min(1, Math.max(0, position));
  const scaled = clamped * (MAGNITUDE_RAMP.length - 1);
  const index = Math.min(MAGNITUDE_RAMP.length - 1, Math.round(scaled));
  return MAGNITUDE_RAMP[index];
}

/** Readable ink on a magnitude cell: the ramp's light end needs dark text. */
function rampInk(position: number): string {
  return position > 0.62 ? "#0a1120" : "#e6edf9";
}

/**
 * `loss.ArcFaceLoss.margin` → `ArcFace · margin`, and the deep SSL nesting
 * (`ssl_config.method_params.regularizer_params.weight`) down to its tail, so
 * a table header stays readable without losing which family it belongs to.
 */
function parameterLabel(name: string): string {
  const parts = name.split(".");
  if (parts[0] === "loss" && parts.length >= 3) {
    return `${parts[1].replace(/Loss$/, "")} · ${parts.slice(2).join(".")}`;
  }
  if (parts[0] === "ssl_config") {
    const tail = parts.filter(
      (part) => !["ssl_config", "method_params", "regularizer_params"].includes(part),
    );
    return tail.join(".") || parts[parts.length - 1];
  }
  if (parts[0] === "miner" && parts.length >= 3) {
    return `${parts[1]} · ${parts.slice(2).join(".")}`;
  }
  return name;
}

function metricLabel(name: string): string {
  return name
    .replace(/^best_valid_/, "")
    .replace(/^concatenated_test_/, "test ")
    .replace(/^test_/, "test ")
    .replace(/mean_average_precision_at_r/, "mAP@R")
    .replace(/precision_at_1/, "P@1")
    .replace(/recall_at_k/, "R@K")
    .replace(/_/g, " ");
}

function displayValue(value: ArchiveParamValue): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number") return formatNumber(value);
  if (typeof value === "boolean") return value ? "true" : "false";
  return value;
}

/** Row and column labels are joined with a separator no facet value contains. */
function cellKey(row: string, column: string): string {
  return `${row}\u001f${column}`;
}

function shortPath(path: string): string {
  const parts = path.split("/");
  return parts.slice(Math.max(0, parts.length - 3)).join("/");
}

// --------------------------------------------------------------------- facets

interface FacetDefinition {
  key: string;
  label: string;
  hint?: string;
  /** The facet value of a study, and how it reads in the UI. */
  value: (study: ArchiveStudy) => string;
  display?: (value: string) => string;
  /** Facets with many values get a scrollable, searchable list. */
  long?: boolean;
}

const PHASE_LABELS: Record<string, string> = {
  phase_0: "Phase 0 · random",
  phase_1: "Phase 1 · TPE",
};

const BUCKET_LABELS: Record<string, string> = {
  complete: "Finished (≥ 30 trials)",
  unfinished: "Unfinished (< 30)",
};

const FACETS: FacetDefinition[] = [
  {
    key: "phase",
    label: "Phase",
    hint: "The sampler the sweep used",
    value: (study) => study.phase,
    display: (value) => PHASE_LABELS[value] ?? value,
  },
  {
    key: "bucket",
    label: "Completeness",
    value: (study) => study.bucket,
    display: (value) => BUCKET_LABELS[value] ?? value,
  },
  {
    key: "mode",
    label: "Training mode",
    value: (study) => study.mode,
    display: (value) => (value === "ssl" ? "SSL" : "Supervised"),
  },
  { key: "dataset", label: "Dataset", value: (study) => study.datasetLabel },
  { key: "method", label: "Method", value: (study) => study.methodLabel },
  { key: "loss", label: "DML loss", value: (study) => study.lossLabel },
  { key: "labelBudget", label: "Label budget", value: (study) => study.labelBudget },
  {
    key: "variant",
    label: "Campaign",
    hint: "The __suffix tag that separates repeated sweeps",
    value: (study) => study.variant ?? "(base sweep)",
    long: true,
  },
];

const FACET_BY_KEY = new Map(FACETS.map((facet) => [facet.key, facet]));

type FilterState = Record<string, string[]>;

/** Group-by choices reuse the facet definitions plus the study itself. */
const GROUP_KEYS = [
  "method",
  "loss",
  "dataset",
  "phase",
  "mode",
  "labelBudget",
  "variant",
] as const;

// ----------------------------------------------------------------- statistics

type Statistic = "best" | "topFive" | "median";

const STATISTIC_LABELS: Record<Statistic, string> = {
  best: "Best trial",
  topFive: "Median of top 5",
  median: "Median trial",
};

interface StudyStats {
  study: ArchiveStudy;
  scored: ArchiveTrial[];
  values: number[];
  best: number | null;
  topFive: number | null;
  median: number | null;
  iqr: number | null;
  /** Mean cross-validation spread of the study's completed trials. */
  foldSpread: number | null;
  bestTrial: ArchiveTrial | null;
}

function summarize(study: ArchiveStudy): StudyStats {
  const scored = scoredTrials(study);
  const raw = scored.map((trial) => trial.value as number);
  // Every archived study maximizes its retrieval metric; a hypothetical
  // minimizing study is flipped so one comparison direction serves both.
  const values = study.direction === "minimize" ? raw.map((value) => -value) : raw;
  const sorted = ascending(values);
  const foldSpreads = scored
    .map((trial) =>
      foldValues(trial).length > 1 ? standardDeviation(foldValues(trial)) : null,
    )
    .filter((value): value is number => value != null && Number.isFinite(value));
  const bestValue = sorted.length > 0 ? sorted[sorted.length - 1] : null;
  const bestTrial =
    bestValue == null
      ? null
      : (scored.find(
          (trial) =>
            (study.direction === "minimize" ? -(trial.value as number) : (trial.value as number)) ===
            bestValue,
        ) ?? null);

  return {
    study,
    scored,
    values,
    best: bestValue,
    topFive:
      sorted.length === 0
        ? null
        : median(sorted.slice(Math.max(0, sorted.length - 5))),
    median: sorted.length === 0 ? null : quantile(sorted, 0.5),
    iqr:
      sorted.length < 4 ? null : quantile(sorted, 0.75) - quantile(sorted, 0.25),
    foldSpread: foldSpreads.length > 0 ? mean(foldSpreads) : null,
    bestTrial,
  };
}

function statisticOf(stats: StudyStats, statistic: Statistic): number | null {
  const value = stats[statistic];
  return value != null && Number.isFinite(value) ? value : null;
}

/**
 * The supervised sweep a semi-supervised study should be measured against:
 * same dataset, same loss, same label budget and same phase. The comparison
 * only falls back to a different phase when that exact pairing is missing,
 * which is flagged in the UI as an approximate baseline.
 */
interface Baseline {
  value: number;
  exact: boolean;
  study: ArchiveStudy;
}

function buildBaselines(
  all: StudyStats[],
  statistic: Statistic,
): Map<string, Baseline> {
  const baselines = new Map<string, Baseline>();
  const record = (key: string, value: number, exact: boolean, study: ArchiveStudy) => {
    const current = baselines.get(key);
    if (!current || value > current.value) {
      baselines.set(key, { value, exact, study });
    }
  };

  for (const stats of all) {
    if (stats.study.mode !== "supervised") continue;
    const value = statisticOf(stats, statistic);
    if (value == null) continue;
    const { dataset, loss, labelBudget, phase } = stats.study;
    record(`${dataset}|${loss}|${labelBudget}|${phase}`, value, true, stats.study);
    record(`${dataset}|${loss}|${labelBudget}`, value, false, stats.study);
    record(`${dataset}|${loss}`, value, false, stats.study);
  }
  return baselines;
}

function baselineFor(
  baselines: Map<string, Baseline>,
  study: ArchiveStudy,
): Baseline | null {
  const { dataset, loss, labelBudget, phase } = study;
  return (
    baselines.get(`${dataset}|${loss}|${labelBudget}|${phase}`) ??
    baselines.get(`${dataset}|${loss}|${labelBudget}`) ??
    baselines.get(`${dataset}|${loss}`) ??
    null
  );
}

// -------------------------------------------------------------------- atoms

function MetricCard({
  label,
  value,
  context,
  icon,
  accent,
}: {
  label: string;
  value: string;
  context: string;
  icon: ReactNode;
  accent: "violet" | "cyan" | "lime" | "amber";
}) {
  const tone = {
    violet: "bg-[#9b8cff]/10 text-[#b6acff]",
    cyan: "bg-[#4fd5e7]/10 text-[#64dce9]",
    lime: "bg-[#b8e56e]/10 text-[#c7ec86]",
    amber: "bg-[#f4c86f]/10 text-[#f6d488]",
  }[accent];

  return (
    <div className="surface flex items-start gap-3 px-4 py-3.5">
      <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${tone}`}>
        {icon}
      </span>
      <div className="min-w-0">
        <p className="text-[10px] font-semibold uppercase tracking-[0.13em] text-[#79869e]">
          {label}
        </p>
        <p className="mt-1 truncate text-xl font-semibold tracking-[-0.03em] text-white">
          {value}
        </p>
        <p className="mt-0.5 truncate text-[10px] text-[#68768e]" title={context}>
          {context}
        </p>
      </div>
    </div>
  );
}

interface TooltipState {
  x: number;
  y: number;
  lines: string[];
}

function ChartTooltip({ tooltip }: { tooltip: TooltipState | null }) {
  if (!tooltip) return null;
  return (
    <div
      className="pointer-events-none absolute z-20 max-w-[16rem] rounded-lg border border-white/[0.12] bg-[#111a2c] px-2.5 py-1.5 text-[10px] leading-4 text-[#dbe2ef] shadow-xl"
      style={{
        left: `${tooltip.x}px`,
        top: `${tooltip.y}px`,
        transform: "translate(-50%, calc(-100% - 10px))",
      }}
      role="status"
    >
      {tooltip.lines.map((line, index) => (
        <p key={line + index} className={index === 0 ? "font-semibold text-white" : ""}>
          {line}
        </p>
      ))}
    </div>
  );
}

function EmptyChart({ message }: { message: string }) {
  return (
    <div className="flex min-h-[180px] items-center justify-center rounded-xl border border-dashed border-white/[0.08] px-6 text-center text-xs leading-5 text-[#6f7d95]">
      {message}
    </div>
  );
}

function SectionHeading({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="flex flex-wrap items-end justify-between gap-3">
      <div>
        <h3 className="text-sm font-semibold text-white">{title}</h3>
        <p className="mt-1 text-[11px] leading-5 text-[#7e8ba3]">{description}</p>
      </div>
      {action}
    </div>
  );
}

// -------------------------------------------------------------------- charts

interface GroupSummary {
  key: string;
  label: string;
  color: string;
  studies: StudyStats[];
  /** Per-study scores under the active statistic and score mode. */
  scores: number[];
  best: number | null;
  median: number | null;
  trials: number;
}

/**
 * Ranked score across groups, drawn as a dot plot rather than bars: retrieval
 * metrics sit in a narrow band well above zero, so bar lengths would either be
 * indistinguishable (honest zero baseline) or exaggerated (truncated one). A
 * dot encodes position, which stays truthful on a clipped axis. The connector
 * runs from the group's median study to its best, and every row prints its
 * value.
 */
function GroupRankChart({
  groups,
  unit,
  zeroBaseline,
}: {
  groups: GroupSummary[];
  unit: string;
  zeroBaseline: boolean;
}) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const rows = groups.filter((group) => group.best != null);
  if (rows.length === 0) {
    return <EmptyChart message="No group in this selection has a scored trial." />;
  }

  const rowHeight = 28;
  const labelWidth = 172;
  const valueWidth = 78;
  const width = 860;
  const height = rows.length * rowHeight + 26;
  const plotLeft = labelWidth;
  const plotRight = width - valueWidth;
  const points = rows.flatMap((group) => [
    group.best as number,
    group.median ?? (group.best as number),
  ]);
  const domain = extent(zeroBaseline ? [...points, 0] : points, 0.08);
  const x = linearScale(domain, [plotLeft, plotRight]);

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="block w-full"
        role="img"
        aria-label="Score by group"
      >
        {ticks(domain, 6).map((tick) => (
          <line
            key={tick}
            x1={x(tick)}
            x2={x(tick)}
            y1={2}
            y2={height - 22}
            stroke={CHART.grid}
            strokeWidth={1}
          />
        ))}
        {zeroBaseline && (
          <line
            x1={x(0)}
            x2={x(0)}
            y1={2}
            y2={height - 22}
            stroke={CHART.axis}
            strokeWidth={1}
          />
        )}
        {rows.map((group, index) => {
          const y = index * rowHeight + 6;
          const centre = y + (rowHeight - 10) / 2;
          const best = group.best as number;
          const groupMedian = group.median ?? best;
          return (
            <g
              key={group.key}
              onPointerMove={(event) => {
                const box = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
                if (!box) return;
                setTooltip({
                  x: event.clientX - box.left,
                  y: event.clientY - box.top,
                  lines: [
                    group.label,
                    `Best ${zeroBaseline ? signedMetric(best) : formatMetric(best)} ${unit}`,
                    `Median study ${zeroBaseline ? signedMetric(groupMedian) : formatMetric(groupMedian)}`,
                    `${group.studies.length} studies · ${group.trials} scored trials`,
                  ],
                });
              }}
              onPointerLeave={() => setTooltip(null)}
            >
              <rect x={0} y={y - 2} width={width} height={rowHeight - 4} fill="transparent" />
              <text
                x={labelWidth - 12}
                y={centre + 4}
                textAnchor="end"
                fontSize={11}
                fill={CHART.text}
              >
                {group.label.length > 26 ? `${group.label.slice(0, 25)}…` : group.label}
              </text>
              <line
                x1={Math.min(x(groupMedian), x(best))}
                x2={Math.max(x(groupMedian), x(best))}
                y1={centre}
                y2={centre}
                stroke={CHART.accent}
                strokeOpacity={0.45}
                strokeWidth={2}
              />
              <circle
                cx={x(groupMedian)}
                cy={centre}
                r={3.5}
                fill="#0f1729"
                stroke={CHART.accent}
                strokeWidth={1.5}
              />
              <circle
                cx={x(best)}
                cy={centre}
                r={5}
                fill={CHART.accent}
                stroke="#0a1120"
                strokeWidth={1.5}
              />
              <text
                x={plotRight + 10}
                y={centre + 4}
                fontSize={11}
                fill={CHART.text}
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {zeroBaseline ? signedMetric(best) : formatMetric(best)}
              </text>
            </g>
          );
        })}
        {ticks(domain, 6).map((tick) => (
          <text
            key={`label-${tick}`}
            x={x(tick)}
            y={height - 7}
            textAnchor="middle"
            fontSize={9}
            fill={CHART.muted}
          >
            {axisNumber(tick)}
          </text>
        ))}
      </svg>
      <ChartTooltip tooltip={tooltip} />
      <div className="mt-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-[10px] text-[#68768e]">
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-full"
            style={{ background: CHART.accent }}
          />
          best study
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span
            className="inline-block h-2.5 w-2.5 rounded-full border-2"
            style={{ borderColor: CHART.accent }}
          />
          median study
        </span>
        <span>Axis is clipped to the data, so read position, not distance from the edge.</span>
      </div>
    </div>
  );
}

/**
 * Per-group spread of study scores: box from the quartiles, whiskers to the
 * extremes, every study drawn as a dot so a group of two is not mistaken for a
 * distribution.
 */
function GroupBoxPlot({
  groups,
  unit,
  zeroBaseline,
}: {
  groups: GroupSummary[];
  unit: string;
  zeroBaseline: boolean;
}) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const rows = groups.filter((group) => group.scores.length > 0);
  if (rows.length === 0) {
    return <EmptyChart message="No group in this selection has a scored study." />;
  }

  const rowHeight = 34;
  const labelWidth = 168;
  const width = 860;
  const height = rows.length * rowHeight + 28;
  const plotLeft = labelWidth;
  const plotRight = width - 26;
  const all = rows.flatMap((group) => group.scores);
  const domain = extent(zeroBaseline ? [...all, 0] : all, 0.06);
  const x = linearScale(domain, [plotLeft, plotRight]);

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="block w-full"
        role="img"
        aria-label="Score distribution by group"
      >
        {ticks(domain, 6).map((tick) => (
          <line
            key={tick}
            x1={x(tick)}
            x2={x(tick)}
            y1={4}
            y2={height - 22}
            stroke={CHART.grid}
            strokeWidth={1}
          />
        ))}
        {zeroBaseline && (
          <line
            x1={x(0)}
            x2={x(0)}
            y1={4}
            y2={height - 22}
            stroke={CHART.axis}
            strokeDasharray="3 3"
          />
        )}
        {rows.map((group, index) => {
          const y = index * rowHeight + 6;
          const centre = y + (rowHeight - 10) / 2;
          const sorted = ascending(group.scores);
          const q1 = quantile(sorted, 0.25);
          const q2 = quantile(sorted, 0.5);
          const q3 = quantile(sorted, 0.75);
          const low = sorted[0];
          const high = sorted[sorted.length - 1];
          return (
            <g key={group.key}>
              <text
                x={labelWidth - 10}
                y={centre + 4}
                textAnchor="end"
                fontSize={11}
                fill={CHART.text}
              >
                {group.label.length > 26 ? `${group.label.slice(0, 25)}…` : group.label}
              </text>
              <line
                x1={x(low)}
                x2={x(high)}
                y1={centre}
                y2={centre}
                stroke={CHART.axis}
                strokeWidth={1.5}
              />
              {sorted.length >= 4 && (
                <rect
                  x={x(q1)}
                  y={centre - 8}
                  width={Math.max(2, x(q3) - x(q1))}
                  height={16}
                  rx={3}
                  fill={CHART.faint}
                  stroke={CHART.accent}
                  strokeWidth={1}
                />
              )}
              <line
                x1={x(q2)}
                x2={x(q2)}
                y1={centre - 9}
                y2={centre + 9}
                stroke={CHART.accent}
                strokeWidth={2.5}
              />
              {group.studies.map((stats, studyIndex) => {
                const score = group.scores[studyIndex];
                if (score == null || !Number.isFinite(score)) return null;
                return (
                  <circle
                    key={stats.study.id}
                    className="chart-point"
                    cx={x(score)}
                    cy={centre}
                    r={3}
                    fill="#e6edf9"
                    fillOpacity={0.5}
                    stroke="#0a1120"
                    strokeWidth={1}
                    onPointerEnter={(event) => {
                      const box = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
                      if (!box) return;
                      setTooltip({
                        x: event.clientX - box.left,
                        y: event.clientY - box.top,
                        lines: [
                          shortPath(stats.study.path),
                          `${zeroBaseline ? signedMetric(score) : formatMetric(score)} ${unit}`,
                          `${stats.scored.length} scored trials`,
                        ],
                      });
                    }}
                    onPointerLeave={() => setTooltip(null)}
                  />
                );
              })}
              <text
                x={plotRight}
                y={centre - 12}
                textAnchor="end"
                fontSize={9}
                fill={CHART.muted}
              >
                n={group.studies.length}
              </text>
            </g>
          );
        })}
        {ticks(domain, 6).map((tick) => (
          <text
            key={`label-${tick}`}
            x={x(tick)}
            y={height - 7}
            textAnchor="middle"
            fontSize={9}
            fill={CHART.muted}
          >
            {axisNumber(tick)}
          </text>
        ))}
      </svg>
      <ChartTooltip tooltip={tooltip} />
      <p className="mt-1 text-[10px] text-[#68768e]">
        One dot per study ({unit}); the box spans the quartiles once a group holds four or more.
      </p>
    </div>
  );
}

/**
 * Two facets crossed into a grid — the form for "which method works with which
 * loss". Magnitude is a single-hue ramp and every cell also prints its number,
 * so the grid stays readable without relying on the fill.
 */
function MatrixHeatmap({
  rowLabels,
  columnLabels,
  cells,
  unit,
  zeroBaseline,
  onSelect,
}: {
  rowLabels: string[];
  columnLabels: string[];
  cells: Map<string, { value: number; studies: StudyStats[] }>;
  unit: string;
  zeroBaseline: boolean;
  onSelect: (row: string, column: string) => void;
}) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const values = [...cells.values()].map((cell) => cell.value);
  if (values.length === 0) {
    return <EmptyChart message="No cell in this crossing has a scored study." />;
  }
  const low = Math.min(...values);
  const high = Math.max(...values);
  const span = high - low || 1;

  return (
    <div className="relative" data-chart-frame>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[560px] border-separate border-spacing-[2px] text-[11px]">
          <thead>
            <tr>
              <th className="sticky left-0 z-10 bg-[#10182a] px-2 py-1 text-left text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e]">
                <span className="sr-only">Group</span>
              </th>
              {columnLabels.map((column) => (
                <th
                  key={column}
                  className="px-2 py-1 text-center text-[10px] font-semibold text-[#a8b3c7]"
                >
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rowLabels.map((row) => (
              <tr key={row}>
                <th
                  scope="row"
                  className="sticky left-0 z-10 max-w-[170px] truncate bg-[#10182a] px-2 py-1 text-left text-[11px] font-medium text-[#c3ccdc]"
                  title={row}
                >
                  {row}
                </th>
                {columnLabels.map((column) => {
                  const cell = cells.get(cellKey(row, column));
                  if (!cell) {
                    return (
                      <td
                        key={column}
                        className="rounded-md border border-dashed border-white/[0.06] px-2 py-2 text-center text-[10px] text-[#4d5a72]"
                      >
                        —
                      </td>
                    );
                  }
                  const position = (cell.value - low) / span;
                  return (
                    <td key={column} className="p-0">
                      <button
                        type="button"
                        onClick={() => onSelect(row, column)}
                        onPointerEnter={(event) => {
                          const box = event.currentTarget
                            .closest("[data-chart-frame]")
                            ?.getBoundingClientRect();
                          if (!box) return;
                          setTooltip({
                            x: event.clientX - box.left,
                            y: event.clientY - box.top,
                            lines: [
                              `${row} × ${column}`,
                              `${zeroBaseline ? signedMetric(cell.value) : formatMetric(cell.value)} ${unit}`,
                              `${cell.studies.length} stud${cell.studies.length === 1 ? "y" : "ies"} · click to filter`,
                            ],
                          });
                        }}
                        onPointerLeave={() => setTooltip(null)}
                        className="focus-ring w-full rounded-md px-2 py-2 text-center font-medium tabular-nums transition hover:brightness-125"
                        style={{
                          background: rampColor(position),
                          color: rampInk(position),
                        }}
                      >
                        {zeroBaseline ? signedMetric(cell.value) : formatMetric(cell.value)}
                      </button>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <ChartTooltip tooltip={tooltip} />
    </div>
  );
}

/** Legend for the magnitude ramp, with its ends labelled. */
function RampLegend({
  low,
  high,
  zeroBaseline,
}: {
  low: number;
  high: number;
  zeroBaseline: boolean;
}) {
  return (
    <div className="flex items-center gap-2 text-[10px] text-[#79869e]">
      <span className="tabular-nums">
        {zeroBaseline ? signedMetric(low) : formatMetric(low)}
      </span>
      <span className="flex h-2.5 overflow-hidden rounded-full">
        {MAGNITUDE_RAMP.map((color) => (
          <span key={color} className="block h-full w-5" style={{ background: color }} />
        ))}
      </span>
      <span className="tabular-nums">
        {zeroBaseline ? signedMetric(high) : formatMetric(high)}
      </span>
    </div>
  );
}

interface ScatterPoint {
  x: number;
  y: number;
  /** Inside the study's tolerance band — drawn filled rather than faint. */
  inBand: boolean;
  label: string;
  detail: string;
  studyId: string;
  trialNumber: number;
}

interface TrialSelection {
  studyId: string;
  trialNumber: number;
}

/**
 * One panel per group rather than one colour per group: with six methods on
 * screen a colour-coded scatter would need more hues than can be told apart,
 * and small multiples answer "where do the good trials sit for this method"
 * more directly anyway. Panels share both axes so they can be read across.
 */
function ScatterPanel({
  title,
  points,
  xDomain,
  yDomain,
  logX,
  xLabel,
  yLabel,
  color,
  epsilon,
  selectedTrial,
  onSelectTrial,
}: {
  title: string;
  points: ScatterPoint[];
  xDomain: [number, number];
  yDomain: [number, number];
  logX: boolean;
  xLabel: string;
  yLabel: string;
  color: string;
  /** Tolerance band drawn behind the points, so the y span has a yardstick. */
  epsilon: number;
  selectedTrial: TrialSelection | null;
  onSelectTrial: (selection: TrialSelection) => void;
}) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const width = 320;
  const height = 210;
  const padding = { top: 12, right: 12, bottom: 30, left: 44 };
  const project = (value: number) => (logX ? Math.log10(Math.max(value, 1e-12)) : value);
  const x = linearScale(xDomain, [padding.left, width - padding.right]);
  const y = linearScale(yDomain, [height - padding.bottom, padding.top]);

  return (
    <div className="surface-soft relative p-3" data-chart-frame>
      <p className="mb-1 flex items-center gap-1.5 truncate text-[11px] font-semibold text-white">
        <span
          className="inline-block h-2 w-2 shrink-0 rounded-full"
          style={{ background: color }}
          aria-hidden="true"
        />
        {title}
        <span className="ml-auto shrink-0 text-[10px] font-normal text-[#68768e]">
          {points.length}
        </span>
      </p>
      <svg viewBox={`0 0 ${width} ${height}`} className="block w-full" role="img" aria-label={title}>
        {epsilon > 0 && yDomain[1] >= -epsilon && (
          <rect
            x={padding.left}
            y={y(Math.min(yDomain[1], 0))}
            width={width - padding.right - padding.left}
            height={Math.max(
              1,
              y(Math.max(yDomain[0], -epsilon)) - y(Math.min(yDomain[1], 0)),
            )}
            fill="#199e70"
            opacity={0.16}
          />
        )}
        {ticks(yDomain, 4).map((tick) => (
          <g key={`y-${tick}`}>
            <line
              x1={padding.left}
              x2={width - padding.right}
              y1={y(tick)}
              y2={y(tick)}
              stroke={CHART.grid}
              strokeWidth={1}
            />
            <text
              x={padding.left - 5}
              y={y(tick) + 3}
              textAnchor="end"
              fontSize={8}
              fill={CHART.muted}
            >
              {axisNumber(tick)}
            </text>
          </g>
        ))}
        {ticks(xDomain, 4).map((tick) => (
          <text
            key={`x-${tick}`}
            x={x(tick)}
            y={height - padding.bottom + 12}
            textAnchor="middle"
            fontSize={8}
            fill={CHART.muted}
          >
            {axisNumber(logX ? 10 ** tick : tick)}
          </text>
        ))}
        {points.map((point, index) => {
          const selected =
            selectedTrial?.studyId === point.studyId &&
            selectedTrial.trialNumber === point.trialNumber;
          const select = () =>
            onSelectTrial({ studyId: point.studyId, trialNumber: point.trialNumber });
          return (
            <circle
              key={`${point.label}-${index}`}
              className="focus-ring chart-point cursor-pointer"
              cx={x(project(point.x))}
              cy={y(Math.max(point.y, yDomain[0]))}
              r={selected ? 6 : point.inBand ? 4 : 3}
              fill={point.inBand ? color : "#5c6a82"}
              fillOpacity={selected ? 1 : point.inBand ? 0.9 : 0.5}
              stroke={selected ? "#ffffff" : "#0a1120"}
              strokeWidth={selected ? 2 : 1}
              role="button"
              tabIndex={0}
              aria-label={`Show details for ${point.label}`}
              onClick={select}
              onKeyDown={(event) => {
                if (event.key === "Enter" || event.key === " ") {
                  event.preventDefault();
                  select();
                }
              }}
              onPointerEnter={(event) => {
                const box = event.currentTarget
                  .closest("[data-chart-frame]")
                  ?.getBoundingClientRect();
                if (!box) return;
                setTooltip({
                  x: event.clientX - box.left,
                  y: event.clientY - box.top,
                  lines: [
                    point.label,
                    `${xLabel} ${formatNumber(point.x)}`,
                    `${yLabel} ${formatMetric(point.y)}`,
                    point.detail,
                    "Click for trial details",
                  ],
                });
              }}
              onPointerLeave={() => setTooltip(null)}
            />
          );
        })}
        <line
          x1={padding.left}
          x2={width - padding.right}
          y1={height - padding.bottom}
          y2={height - padding.bottom}
          stroke={CHART.axis}
          strokeWidth={1}
        />
      </svg>
      <ChartTooltip tooltip={tooltip} />
    </div>
  );
}

/**
 * Mean fANOVA importance per parameter across the studies that carry a
 * ranking. Magnitude, one hue, value printed on every row.
 */
function ImportanceBars({
  rows,
}: {
  rows: Array<{ name: string; importance: number; studies: number }>;
}) {
  if (rows.length === 0) {
    return (
      <EmptyChart message="No study in this selection has a stored fANOVA ranking. Run scripts/optuna_param_importance.py over sorted_logs to add one." />
    );
  }
  const highest = Math.max(...rows.map((row) => row.importance)) || 1;

  return (
    <ul className="space-y-1.5">
      {rows.map((row) => (
        <li key={row.name} className="flex items-center gap-3">
          <span
            className="w-[42%] shrink-0 truncate text-[11px] text-[#c3ccdc]"
            title={row.name}
          >
            {parameterLabel(row.name)}
          </span>
          <span className="h-2.5 flex-1 overflow-hidden rounded-full bg-white/[0.05]">
            <span
              className="block h-full rounded-full"
              style={{
                width: `${Math.max(2, (row.importance / highest) * 100)}%`,
                background: CHART.accent,
              }}
            />
          </span>
          <span className="w-16 shrink-0 text-right text-[10px] tabular-nums text-[#a8b3c7]">
            {(row.importance * 100).toFixed(1)}%
          </span>
          <span className="w-14 shrink-0 text-right text-[10px] text-[#68768e]">
            {row.studies} st.
          </span>
        </li>
      ))}
    </ul>
  );
}

// --------------------------------------------------------------- facet panel

interface FacetOption {
  value: string;
  label: string;
  count: number;
  selected: boolean;
}

function FacetGroup({
  facet,
  options,
  onToggle,
  onClear,
}: {
  facet: FacetDefinition;
  options: FacetOption[];
  onToggle: (value: string) => void;
  onClear: () => void;
}) {
  const [needle, setNeedle] = useState("");
  const selectedCount = options.filter((option) => option.selected).length;
  const shown = facet.long
    ? options.filter((option) =>
        option.label.toLowerCase().includes(needle.trim().toLowerCase()),
      )
    : options;

  return (
    <section className="border-b border-white/[0.06] px-4 py-3 last:border-b-0">
      <div className="flex items-center justify-between gap-2">
        <div className="min-w-0">
          <h3 className="text-[11px] font-semibold text-[#c3ccdc]">{facet.label}</h3>
          {facet.hint && (
            <p className="mt-0.5 truncate text-[10px] text-[#63718a]" title={facet.hint}>
              {facet.hint}
            </p>
          )}
        </div>
        {selectedCount > 0 && (
          <button
            type="button"
            onClick={onClear}
            className="focus-ring shrink-0 rounded-lg px-1.5 py-0.5 text-[10px] font-semibold text-[#9b8cff] hover:bg-white/[0.06]"
          >
            Clear {selectedCount}
          </button>
        )}
      </div>
      {facet.long && options.length > 8 && (
        <input
          type="search"
          value={needle}
          onChange={(event) => setNeedle(event.target.value)}
          placeholder={`Filter ${options.length} values…`}
          className={`${inputClass} mt-2 h-8`}
        />
      )}
      <div
        className={`mt-2 flex flex-wrap gap-1.5 ${
          facet.long ? "max-h-44 overflow-y-auto pr-1" : ""
        }`}
      >
        {shown.length === 0 ? (
          <p className="py-1 text-[10px] text-[#63718a]">No value matches.</p>
        ) : (
          shown.map((option) => (
            <button
              key={option.value}
              type="button"
              aria-pressed={option.selected}
              onClick={() => onToggle(option.value)}
              disabled={option.count === 0 && !option.selected}
              className={`focus-ring inline-flex max-w-full items-center gap-1.5 rounded-lg border px-2 py-1 text-[10px] font-medium transition ${
                option.selected
                  ? "border-[#9b8cff]/40 bg-[#9b8cff]/[0.14] text-white"
                  : option.count === 0
                    ? "cursor-not-allowed border-white/[0.05] text-[#4b586f]"
                    : "border-white/[0.08] bg-white/[0.03] text-[#a3aec2] hover:bg-white/[0.07] hover:text-white"
              }`}
            >
              <span className="truncate">{option.label}</span>
              <span
                className={`tabular-nums ${option.selected ? "text-[#c9c0ff]" : "text-[#68768e]"}`}
              >
                {option.count}
              </span>
            </button>
          ))
        )}
      </div>
    </section>
  );
}

function Segmented<T extends string>({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: T;
  options: Array<{ value: T; label: string }>;
  onChange: (value: T) => void;
}) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#79869e]">
        {label}
      </span>
      <div className="inline-flex rounded-xl border border-white/[0.08] bg-white/[0.03] p-0.5">
        {options.map((option) => (
          <button
            key={option.value}
            type="button"
            aria-pressed={value === option.value}
            onClick={() => onChange(option.value)}
            className={`focus-ring rounded-[10px] px-2.5 py-1 text-[11px] font-semibold transition ${
              value === option.value
                ? "bg-[#9b8cff]/[0.16] text-white"
                : "text-[#8e9bb1] hover:text-white"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function LabelledSelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: Array<{ value: string; label: string }>;
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex items-center gap-2">
      <span className="whitespace-nowrap text-[10px] font-semibold uppercase tracking-[0.12em] text-[#79869e]">
        {label}
      </span>
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className={`${inputClass} w-auto min-w-[9rem]`}
      >
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

/** A sortable column header that also announces the active sort. */
function SortHeader({
  label,
  columnKey,
  sort,
  onSort,
  align = "left",
  title,
}: {
  label: string;
  columnKey: string;
  sort: { key: string; direction: "asc" | "desc" };
  onSort: (key: string) => void;
  align?: "left" | "right";
  title?: string;
}) {
  const active = sort.key === columnKey;
  return (
    <th
      scope="col"
      className={`whitespace-nowrap px-2.5 py-2 text-[10px] font-semibold uppercase tracking-[0.1em] ${
        align === "right" ? "text-right" : "text-left"
      } ${active ? "text-white" : "text-[#79869e]"}`}
      aria-sort={active ? (sort.direction === "asc" ? "ascending" : "descending") : "none"}
    >
      <button
        type="button"
        onClick={() => onSort(columnKey)}
        title={title ?? label}
        className={`focus-ring inline-flex items-center gap-1 rounded ${
          align === "right" ? "flex-row-reverse" : ""
        }`}
      >
        <span className="truncate">{label}</span>
        {active && (
          <ArrowDownWideNarrow
            size={11}
            className={sort.direction === "asc" ? "rotate-180" : ""}
          />
        )}
      </button>
    </th>
  );
}

// --------------------------------------------------------------- study detail

function DetailRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex items-baseline justify-between gap-3 border-b border-white/[0.05] py-1.5 last:border-b-0">
      <span className="shrink-0 text-[10px] uppercase tracking-[0.1em] text-[#68768e]">
        {label}
      </span>
      <span className="min-w-0 break-words text-right text-[11px] text-[#c9d2e1]">{value}</span>
    </div>
  );
}

function rangeLabel(parameter: ArchiveStudy["parameters"][number]): string {
  if (parameter.kind === "categorical") {
    return (parameter.choices ?? []).map((choice) => displayValue(choice)).join(", ") || "—";
  }
  if (parameter.low == null || parameter.high == null) return "—";
  return `${formatNumber(parameter.low)} … ${formatNumber(parameter.high)}${
    parameter.log ? " (log)" : ""
  }`;
}

function StudyDetail({
  stats,
  baseline,
  onClose,
  onSelectTrial,
}: {
  stats: StudyStats;
  baseline: Baseline | null;
  onClose: () => void;
  onSelectTrial: (trialNumber: number) => void;
}) {
  const [showConfig, setShowConfig] = useState(false);
  const { study } = stats;
  const ranked = [...study.parameters].sort((left, right) => {
    const leftRank = left.importanceRank ?? Number.POSITIVE_INFINITY;
    const rightRank = right.importanceRank ?? Number.POSITIVE_INFINITY;
    if (leftRank !== rightRank) return leftRank - rightRank;
    return left.name.localeCompare(right.name);
  });
  const topTrials = [...stats.scored]
    .sort((left, right) => (right.value as number) - (left.value as number))
    .slice(0, 8);
  const configEntries = Object.entries(study.sslConfig);

  return (
    <section className="surface fade-up mt-4 overflow-hidden">
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-white/[0.07] p-4">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-1.5">
            {[
              PHASE_LABELS[study.phase] ?? study.phase,
              study.mode === "ssl" ? "SSL" : "Supervised",
              study.datasetLabel,
              study.methodLabel,
              study.lossLabel,
              study.labelBudget,
              ...(study.variant ? [study.variant] : []),
              ...(study.bucket === "unfinished" ? ["unfinished"] : []),
            ].map((chip) => (
              <span
                key={chip}
                className="rounded-full border border-white/[0.08] bg-white/[0.035] px-2 py-0.5 text-[10px] text-[#8996ab]"
              >
                {chip}
              </span>
            ))}
          </div>
          <h3 className="mt-2 break-all font-mono text-xs text-[#dbe2ef]">{study.path}</h3>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="focus-ring flex h-8 w-8 items-center justify-center rounded-lg text-[#8290a7] hover:bg-white/[0.07] hover:text-white"
          aria-label="Close study detail"
        >
          <X size={15} />
        </button>
      </header>

      <div className="grid gap-4 p-4 lg:grid-cols-3">
        <div className="surface-soft p-3">
          <h4 className="text-[11px] font-semibold text-white">Sweep</h4>
          <div className="mt-2">
            <DetailRow label="Objective" value={metricLabel(study.metric)} />
            <DetailRow
              label="Best"
              value={
                <span className="tabular-nums">
                  {formatMetric(stats.best)}
                  {stats.bestTrial ? ` · trial #${stats.bestTrial.number}` : ""}
                </span>
              }
            />
            <DetailRow
              label="vs supervised"
              value={
                baseline ? (
                  <span className="tabular-nums">
                    {signedMetric((stats.best ?? 0) - baseline.value)}
                    {baseline.exact ? "" : " (approx.)"}
                  </span>
                ) : (
                  "no baseline"
                )
              }
            />
            <DetailRow
              label="Trials"
              value={`${stats.scored.length} scored of ${study.trialCount}${
                study.plannedTrials ? ` (planned ${study.plannedTrials})` : ""
              }`}
            />
            <DetailRow label="Sampler" value={study.sampler} />
            <DetailRow
              label="CV"
              value={study.cvMode ? `${study.cvMode} · k=${study.cvK ?? "?"}` : "—"}
            />
            <DetailRow
              label="Fold spread"
              value={
                stats.foldSpread != null ? (
                  <span className="tabular-nums">±{stats.foldSpread.toFixed(4)}</span>
                ) : (
                  "—"
                )
              }
            />
            <DetailRow
              label="Schedule"
              value={`${study.epochs ?? "?"} epochs · patience ${study.patience ?? "?"}`}
            />
            <DetailRow label="Backbone" value={study.backboneTuning ?? "—"} />
            <DetailRow label="Seed" value={study.seed != null ? String(study.seed) : "—"} />
          </div>
        </div>

        <div className="surface-soft p-3 lg:col-span-2">
          <h4 className="text-[11px] font-semibold text-white">
            Search space
            <span className="ml-2 font-normal text-[10px] text-[#68768e]">
              {ranked.some((parameter) => parameter.importance != null)
                ? "ranked by stored fANOVA importance"
                : "no fANOVA ranking stored for this study"}
            </span>
          </h4>
          <div className="mt-2 max-h-64 overflow-auto">
            <table className="w-full text-[11px]">
              <thead className="sticky top-0 bg-[#121b2e]">
                <tr>
                  <th className="px-2 py-1 text-left text-[10px] uppercase tracking-[0.1em] text-[#79869e]">
                    Parameter
                  </th>
                  <th className="px-2 py-1 text-left text-[10px] uppercase tracking-[0.1em] text-[#79869e]">
                    Range
                  </th>
                  <th className="px-2 py-1 text-right text-[10px] uppercase tracking-[0.1em] text-[#79869e]">
                    Best trial
                  </th>
                  <th className="px-2 py-1 text-right text-[10px] uppercase tracking-[0.1em] text-[#79869e]">
                    fANOVA
                  </th>
                </tr>
              </thead>
              <tbody>
                {ranked.map((parameter) => (
                  <tr key={parameter.name} className="border-t border-white/[0.05]">
                    <td
                      className="max-w-[16rem] truncate px-2 py-1.5 text-[#c9d2e1]"
                      title={parameter.name}
                    >
                      {parameterLabel(parameter.name)}
                      {!parameter.searched && (
                        <span className="ml-1.5 text-[9px] text-[#8a6d3b]">declared only</span>
                      )}
                    </td>
                    <td className="px-2 py-1.5 text-[#8f9bb2]">{rangeLabel(parameter)}</td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-[#dbe2ef]">
                      {stats.bestTrial
                        ? displayValue(stats.bestTrial.params[parameter.name] ?? null)
                        : "—"}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-[#a8b3c7]">
                      {parameter.importance != null
                        ? `${(parameter.importance * 100).toFixed(1)}%`
                        : "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="surface-soft p-3 lg:col-span-2">
          <h4 className="text-[11px] font-semibold text-white">Top trials</h4>
          <div className="mt-2 overflow-x-auto">
            <table className="w-full text-[11px]">
              <thead>
                <tr>
                  <th className="px-2 py-1 text-left text-[10px] uppercase tracking-[0.1em] text-[#79869e]">
                    #
                  </th>
                  <th className="px-2 py-1 text-right text-[10px] uppercase tracking-[0.1em] text-[#79869e]">
                    {metricLabel(study.metric)}
                  </th>
                  <th className="px-2 py-1 text-right text-[10px] uppercase tracking-[0.1em] text-[#79869e]">
                    P@1
                  </th>
                  <th className="px-2 py-1 text-right text-[10px] uppercase tracking-[0.1em] text-[#79869e]">
                    Fold ±
                  </th>
                  <th className="px-2 py-1 text-right text-[10px] uppercase tracking-[0.1em] text-[#79869e]">
                    Epoch
                  </th>
                  <th className="px-2 py-1 text-right text-[10px] uppercase tracking-[0.1em] text-[#79869e]">
                    Runtime
                  </th>
                </tr>
              </thead>
              <tbody>
                {topTrials.map((trial) => (
                  <tr key={trial.number} className="border-t border-white/[0.05]">
                    <td className="px-2 py-1.5 tabular-nums">
                      <button
                        type="button"
                        onClick={() => onSelectTrial(trial.number)}
                        className="focus-ring rounded px-1 text-[#b6acff] hover:underline"
                        aria-label={`Show details for trial ${trial.number}`}
                      >
                        {trial.number}
                      </button>
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-white">
                      {formatMetric(trial.value)}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-[#a8b3c7]">
                      {formatMetric(finite(trial.metrics.best_valid_precision_at_1))}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-[#a8b3c7]">
                      {foldValues(trial).length > 1
                        ? standardDeviation(foldValues(trial)).toFixed(4)
                        : "—"}
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-[#a8b3c7]">
                      {finite(trial.metrics.selected_epoch) ?? "—"}
                      <span className="text-[#5c6a82]">
                        /{finite(trial.metrics.last_epoch) ?? "?"}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 text-right tabular-nums text-[#a8b3c7]">
                      {formatDuration(trial.seconds)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="surface-soft p-3">
          <h4 className="text-[11px] font-semibold text-white">Provenance</h4>
          <div className="mt-2">
            <DetailRow label="Study name" value={study.studyName} />
            <DetailRow label="Campaign" value={study.campaign ?? "—"} />
            <DetailRow
              label="Config"
              value={
                <span className="break-all font-mono text-[10px]">
                  {study.experimentConfig ?? "—"}
                </span>
              }
            />
            <DetailRow
              label="HPO config"
              value={
                <span className="break-all font-mono text-[10px]">
                  {study.hparamConfigPath ?? "—"}
                </span>
              }
            />
            <DetailRow
              label="Original path"
              value={
                <span className="break-all font-mono text-[10px]">
                  {study.sourcePath ?? "—"}
                </span>
              }
            />
          </div>
          {configEntries.length > 0 && (
            <>
              <button
                type="button"
                onClick={() => setShowConfig((current) => !current)}
                className="focus-ring mt-3 inline-flex items-center gap-1.5 text-[11px] font-semibold text-[#9b8cff]"
              >
                <ChevronDown
                  size={13}
                  className={showConfig ? "" : "-rotate-90"}
                />
                Fixed SSL configuration ({configEntries.length})
              </button>
              {showConfig && (
                <div className="mt-2 max-h-56 overflow-auto rounded-lg bg-black/25 p-2">
                  {configEntries.map(([key, value]) => (
                    <div
                      key={key}
                      className="flex items-baseline justify-between gap-2 py-0.5 font-mono text-[10px]"
                    >
                      <span className="truncate text-[#7e8ba3]" title={key}>
                        {key}
                      </span>
                      <span className="shrink-0 text-[#c9d2e1]">{displayValue(value)}</span>
                    </div>
                  ))}
                </div>
              )}
            </>
          )}
          <p className="mt-3 text-[10px] leading-4 text-[#63718a]">
            Values here stayed identical across every trial of the sweep, so they are what
            separates this campaign from another run of the same method.
          </p>
        </div>
      </div>
    </section>
  );
}

function TrialDetail({
  study,
  trial,
  onClose,
}: {
  study: ArchiveStudy;
  trial: ArchiveTrial;
  onClose: () => void;
}) {
  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [onClose]);

  const selectedEpoch = finite(trial.metrics.selected_epoch);
  const lastEpoch = finite(trial.metrics.last_epoch);
  const precisionAtOne = finite(trial.metrics.best_valid_precision_at_1);
  const parameterEntries = Object.entries(trial.params).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  const metricEntries = Object.entries(trial.metrics)
    .filter(([, value]) => value !== null && value !== undefined)
    .sort(([left], [right]) => left.localeCompare(right));
  const fixedEntries = Object.entries(study.sslConfig).sort(([left], [right]) =>
    left.localeCompare(right),
  );
  const stateTone =
    trial.state === "COMPLETE"
      ? "border-[#b8e56e]/25 bg-[#b8e56e]/10 text-[#c9f18a]"
      : trial.state === "FAIL"
        ? "border-[#e66767]/30 bg-[#e66767]/10 text-[#f08c8c]"
        : "border-[#f4c86f]/25 bg-[#f4c86f]/10 text-[#f6d488]";

  const card = (label: string, value: string, icon: ReactNode) => (
    <div className="rounded-xl border border-white/[0.07] bg-black/15 px-3 py-2.5">
      <div className="flex items-center gap-1.5 text-[9px] font-semibold uppercase tracking-[0.12em] text-[#71809a]">
        {icon}
        {label}
      </div>
      <p className="mt-1.5 truncate text-sm font-semibold tabular-nums text-white" title={value}>
        {value}
      </p>
    </div>
  );

  return (
    <aside
      className="fade-up fixed bottom-4 right-4 top-20 z-50 flex w-[calc(100vw-2rem)] max-w-[440px] flex-col overflow-hidden rounded-[20px] border border-[#9b8cff]/25 bg-[#10192c]/95 shadow-[0_28px_90px_rgba(0,0,0,.55)] backdrop-blur-xl"
      role="dialog"
      aria-modal="false"
      aria-label={`Trial ${trial.number} details`}
    >
      <header className="flex items-start justify-between gap-3 border-b border-white/[0.08] px-4 py-3.5">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <h2 className="text-base font-semibold text-white">Trial #{trial.number}</h2>
            <span
              className={`rounded-full border px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.09em] ${stateTone}`}
            >
              {trial.state}
            </span>
          </div>
          <p className="mt-1 truncate text-[10px] text-[#8794aa]" title={study.path}>
            {study.datasetLabel} · {study.methodLabel} · {study.lossLabel}
          </p>
          <p className="mt-0.5 truncate font-mono text-[9px] text-[#63718a]" title={study.path}>
            {shortPath(study.path)}
          </p>
        </div>
        <button
          type="button"
          onClick={onClose}
          className="focus-ring flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[#8290a7] transition hover:bg-white/[0.07] hover:text-white"
          aria-label="Close trial details"
        >
          <X size={15} />
        </button>
      </header>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        <div className="grid grid-cols-2 gap-2">
          {card(
            `Validation ${metricLabel(study.metric)}`,
            formatMetric(trial.value),
            <BarChart3 size={11} />,
          )}
          {card("Runtime", formatDuration(trial.seconds), <Clock3 size={11} />)}
          {card(
            "Selected epoch",
            selectedEpoch != null ? formatNumber(selectedEpoch, 7) : "—",
            <Trophy size={11} />,
          )}
          {card(
            "Last epoch",
            `${lastEpoch != null ? formatNumber(lastEpoch, 7) : "—"} / ${study.epochs ?? "?"} budget`,
            <Activity size={11} />,
          )}
        </div>

        <section className="surface-soft p-3">
          <h3 className="text-[11px] font-semibold text-white">Run summary</h3>
          <div className="mt-2">
            <DetailRow label="Validation P@1" value={formatMetric(precisionAtOne)} />
            <DetailRow
              label="Cross-validation"
              value={`${trial.folds?.length ?? study.cvK ?? "—"} folds · ${study.cvMode ?? "mode unknown"}`}
            />
            <DetailRow
              label="Schedule"
              value={`${study.epochs ?? "?"} epochs · patience ${study.patience ?? "?"}`}
            />
            <DetailRow label="Sampler" value={study.sampler || "—"} />
            <DetailRow label="Label budget" value={study.labelBudget || "—"} />
          </div>
        </section>

        <section className="surface-soft overflow-hidden">
          <div className="flex items-baseline justify-between gap-3 border-b border-white/[0.06] px-3 py-2.5">
            <h3 className="text-[11px] font-semibold text-white">Hyperparameters</h3>
            <span className="text-[9px] text-[#68768e]">{parameterEntries.length} sampled</span>
          </div>
          {parameterEntries.length === 0 ? (
            <p className="px-3 py-4 text-[10px] text-[#68768e]">No sampled parameters recorded.</p>
          ) : (
            <div className="max-h-64 overflow-auto px-3 py-1.5">
              {parameterEntries.map(([name, value]) => (
                <DetailRow
                  key={name}
                  label={parameterLabel(name)}
                  value={
                    <span className="font-mono text-[10px]" title={name}>
                      {displayValue(value)}
                    </span>
                  }
                />
              ))}
            </div>
          )}
        </section>

        {(trial.folds?.length ?? 0) > 0 && (
          <section className="surface-soft overflow-hidden">
            <div className="border-b border-white/[0.06] px-3 py-2.5">
              <h3 className="text-[11px] font-semibold text-white">Validation folds</h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-[10px]">
                <thead>
                  <tr className="text-[#71809a]">
                    <th className="px-3 py-2 text-left font-semibold">Fold</th>
                    <th className="px-2 py-2 text-right font-semibold">{metricLabel(study.metric)}</th>
                    <th className="px-2 py-2 text-right font-semibold">P@1</th>
                    <th className="px-3 py-2 text-right font-semibold">Epoch</th>
                  </tr>
                </thead>
                <tbody>
                  {trial.folds?.map((fold) => (
                    <tr key={fold.fold} className="border-t border-white/[0.05]">
                      <td className="px-3 py-1.5 text-[#a8b3c7]">{fold.fold}</td>
                      <td className="px-2 py-1.5 text-right tabular-nums text-white">
                        {formatMetric(fold.value)}
                      </td>
                      <td className="px-2 py-1.5 text-right tabular-nums text-[#a8b3c7]">
                        {formatMetric(fold.precisionAt1)}
                      </td>
                      <td className="px-3 py-1.5 text-right tabular-nums text-[#a8b3c7]">
                        {fold.selectedEpoch ?? "—"}/{fold.lastEpoch ?? "—"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>
        )}

        <section className="surface-soft overflow-hidden">
          <div className="flex items-baseline justify-between gap-3 border-b border-white/[0.06] px-3 py-2.5">
            <h3 className="text-[11px] font-semibold text-white">Recorded metrics</h3>
            <span className="text-[9px] text-[#68768e]">{metricEntries.length}</span>
          </div>
          <div className="max-h-52 overflow-auto px-3 py-1.5">
            {metricEntries.map(([name, value]) => (
              <DetailRow
                key={name}
                label={metricLabel(name)}
                value={<span className="font-mono text-[10px]">{displayValue(value)}</span>}
              />
            ))}
          </div>
        </section>

        {fixedEntries.length > 0 && (
          <details className="surface-soft overflow-hidden">
            <summary className="focus-ring cursor-pointer px-3 py-2.5 text-[11px] font-semibold text-[#b6acff]">
              Fixed study configuration ({fixedEntries.length})
            </summary>
            <div className="max-h-52 overflow-auto border-t border-white/[0.06] px-3 py-1.5">
              {fixedEntries.map(([name, value]) => (
                <DetailRow
                  key={name}
                  label={parameterLabel(name)}
                  value={<span className="font-mono text-[10px]">{displayValue(value)}</span>}
                />
              ))}
            </div>
          </details>
        )}
      </div>
    </aside>
  );
}


const HEALTH_LABELS: Record<TrialHealth, string> = {
  alive: "Live trial",
  dead: "Collapsed — inside the study's failure tail",
  stalled: "Stalled — the best epoch was the first one",
  failed: "Failed — Optuna recorded no objective",
};

const VERDICT_TONE: Record<Verdict, string> = {
  "extend-low": "border-[#c98500]/40 bg-[#c98500]/[0.14] text-[#e8b957]",
  "extend-high": "border-[#c98500]/40 bg-[#c98500]/[0.14] text-[#e8b957]",
  "extend-both": "border-[#c98500]/40 bg-[#c98500]/[0.14] text-[#e8b957]",
  narrow: "border-[#3987e5]/40 bg-[#3987e5]/[0.14] text-[#7db4f0]",
  keep: "border-white/[0.12] bg-white/[0.05] text-[#a3aec2]",
};

function VerdictPill({ verdict, muted }: { verdict: Verdict; muted?: boolean }) {
  return (
    <span
      className={`inline-flex whitespace-nowrap rounded-lg border px-2 py-0.5 text-[10px] font-semibold ${
        muted ? "border-white/[0.1] bg-white/[0.04] text-[#7e8ba3]" : VERDICT_TONE[verdict]
      }`}
    >
      {VERDICT_LABELS[verdict]}
    </span>
  );
}

function NumberField({
  label,
  value,
  onChange,
  step = 0.1,
  min,
  title,
  suffix,
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  step?: number;
  min?: number;
  title?: string;
  suffix?: string;
}) {
  return (
    <label className="flex items-center gap-2" title={title}>
      <span className="whitespace-nowrap text-[10px] text-[#79869e]">{label}</span>
      <span className="flex items-center gap-1">
        <input
          type="number"
          value={Number.isFinite(value) ? value : ""}
          step={step}
          min={min}
          onChange={(event) => {
            const parsed = Number(event.target.value);
            if (Number.isFinite(parsed)) onChange(parsed);
          }}
          className={`${inputClass} w-[5.5rem]`}
        />
        {suffix && <span className="text-[10px] text-[#68768e]">{suffix}</span>}
      </span>
    </label>
  );
}

/**
 * Every threshold the range rules depend on, in one place. Nothing here is a
 * property of the data — they are the analyst's choices, so they are editable
 * and the consequences update everywhere at once.
 */
function RangeControls({
  settings,
  onChange,
}: {
  settings: RangeSettings;
  onChange: (settings: RangeSettings) => void;
}) {
  const patch = (update: Partial<RangeSettings>) => onChange({ ...settings, ...update });

  return (
    <div className="surface space-y-3 p-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-white">Decision thresholds</h3>
          <p className="mt-1 text-[11px] leading-5 text-[#7e8ba3]">
            The tolerance band, the failure rule and the safety margin. Every verdict, range
            and plot on this page follows from these four numbers.
          </p>
        </div>
        <button
          type="button"
          onClick={() => onChange(DEFAULT_RANGE_SETTINGS)}
          className="focus-ring inline-flex h-8 items-center gap-1.5 rounded-xl border border-white/[0.09] bg-white/[0.04] px-3 text-[11px] font-semibold text-[#c3ccdc] hover:bg-white/[0.08] hover:text-white"
        >
          <RefreshCw size={12} /> Defaults
        </button>
      </div>

      <div className="grid gap-3 lg:grid-cols-3">
        <div className="surface-soft space-y-2 p-3">
          <p className="text-[11px] font-semibold text-white">Tolerance band ε</p>
          <Segmented
            label=""
            value={settings.epsilonMode}
            options={[
              { value: "noise" as const, label: "Measured noise" },
              { value: "absolute" as const, label: "Absolute" },
              { value: "relative" as const, label: "% of best" },
            ]}
            onChange={(value) => patch({ epsilonMode: value })}
          />
          {settings.epsilonMode === "noise" && (
            <>
              <NumberField
                label="Multiplier"
                value={settings.noiseMultiplier}
                step={0.5}
                min={0.5}
                suffix="× SE"
                title="Standard errors of the trial-to-trial difference the band spans. 1 is one sigma, 2 is roughly a 95% band."
                onChange={(value) => patch({ noiseMultiplier: value })}
              />
              <NumberField
                label="Contenders"
                value={Math.round((1 - settings.noiseQuantile) * 100)}
                step={5}
                min={1}
                suffix="% top"
                title="Share of the strongest trials the noise model is fitted on."
                onChange={(value) =>
                  patch({ noiseQuantile: Math.min(0.99, Math.max(0, 1 - value / 100)) })
                }
              />
            </>
          )}
          {settings.epsilonMode === "absolute" && (
            <NumberField
              label="ε"
              value={settings.epsilonAbsolute}
              step={0.005}
              min={0}
              suffix="absolute"
              onChange={(value) => patch({ epsilonAbsolute: value })}
            />
          )}
          {settings.epsilonMode === "relative" && (
            <NumberField
              label="ε"
              value={settings.epsilonPercent}
              step={0.5}
              min={0}
              suffix="% of best"
              onChange={(value) => patch({ epsilonPercent: value })}
            />
          )}
        </div>

        <div className="surface-soft space-y-2 p-3">
          <p className="text-[11px] font-semibold text-white">Hard failures</p>
          <Segmented
            label=""
            value={settings.deadMode}
            options={[
              { value: "fence" as const, label: "Tukey fence" },
              { value: "absolute" as const, label: "Floor" },
              { value: "off" as const, label: "Off" },
            ]}
            onChange={(value) => patch({ deadMode: value })}
          />
          {settings.deadMode === "fence" && (
            <NumberField
              label="k"
              value={settings.deadFence}
              step={0.5}
              min={0}
              suffix="× IQR below Q1"
              onChange={(value) => patch({ deadFence: value })}
            />
          )}
          {settings.deadMode === "absolute" && (
            <NumberField
              label="Floor"
              value={settings.deadFloor}
              step={0.05}
              min={0}
              suffix="objective"
              title="A collapsed embedding sits at chance level; set this to that level when you know it."
              onChange={(value) => patch({ deadFloor: value })}
            />
          )}
          <p className="text-[10px] leading-4 text-[#68768e]">
            Trials whose best epoch was the first one count as stalled, and Optuna FAIL, PRUNED
            and unfinished trials as failed, independently of this rule.
          </p>
        </div>

        <div className="surface-soft space-y-2 p-3">
          <p className="text-[11px] font-semibold text-white">Boundary and margin</p>
          <NumberField
            label="Edge"
            value={Math.round(settings.boundaryTolerance * 100)}
            step={5}
            min={0}
            suffix="% of span"
            title="How close to an edge the band has to reach before the range counts as hit."
            onChange={(value) => patch({ boundaryTolerance: Math.max(0, value) / 100 })}
          />
          <div className="flex flex-wrap gap-2">
            <NumberField
              label="Margin ↓"
              value={settings.marginLow}
              step={0.25}
              min={0}
              title="Decades on a log axis, fractions of the good span on a linear one."
              onChange={(value) => patch({ marginLow: value })}
            />
            <NumberField
              label="Margin ↑"
              value={settings.marginHigh}
              step={0.25}
              min={0}
              title="Decades on a log axis, fractions of the good span on a linear one."
              onChange={(value) => patch({ marginHigh: value })}
            />
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <NumberField
              label="Bins"
              value={settings.bins}
              step={1}
              min={3}
              title="Bins used for the trend line and the edge-flatness test."
              onChange={(value) => patch({ bins: Math.max(3, Math.round(value)) })}
            />
            <Segmented
              label="Bounds"
              value={settings.roundBounds ? "round" : "exact"}
              options={[
                { value: "round" as const, label: "Rounded" },
                { value: "exact" as const, label: "Exact" },
              ]}
              onChange={(value) => patch({ roundBounds: value === "round" })}
            />
          </div>
          <p className="text-[10px] leading-4 text-[#68768e]">
            Margins are decades on a log axis and fractions of the good span otherwise. Skew the
            downward margin up when the sweep was short — a truncated run flatters high learning
            rates, which win early and lose late. Rounded bounds snap to readable numbers and
            always snap <em>outward</em>, so a rounded range never excludes a trial the exact one
            kept.
          </p>
        </div>
      </div>
    </div>
  );
}

interface Diagnostics {
  epsilon: number | null;
  foldSigma: number | null;
  residualSigma: number | null;
  comparisonError: number | null;
  effectSize: number | null;
  resolution: number | null;
  counts: Record<TrialHealth, number>;
  truncatedShare: number | null;
  studies: number;
  epsilonSource: string;
}

/**
 * The Phase 0 health of whatever is currently selected. It rides above the tab
 * strip rather than inside one tab, because every other number on the page is
 * only meaningful next to the noise floor it was measured against.
 */
function DiagnosticsStrip({
  diagnostics,
  onOpen,
}: {
  diagnostics: Diagnostics;
  onOpen: () => void;
}) {
  const {
    epsilon,
    foldSigma,
    residualSigma,
    comparisonError,
    effectSize,
    resolution,
    counts,
    truncatedShare,
  } = diagnostics;
  const resolvable = resolution != null && resolution > 1;
  const badTrials = counts.dead + counts.stalled + counts.failed;

  const cell = (label: string, value: string, detail: string, tone?: string) => (
    <div className="min-w-0 px-3 py-2">
      <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-[#68768e]">
        {label}
      </p>
      <p className={`mt-0.5 truncate text-sm font-semibold tabular-nums ${tone ?? "text-white"}`}>
        {value}
      </p>
      <p className="mt-0.5 truncate text-[10px] text-[#68768e]" title={detail}>
        {detail}
      </p>
    </div>
  );

  return (
    <div className="surface flex flex-wrap items-stretch divide-x divide-white/[0.06]">
      {cell(
        "Tolerance ε",
        epsilon != null ? epsilon.toFixed(4) : "—",
        diagnostics.epsilonSource,
      )}
      {cell(
        "Noise floor",
        comparisonError != null ? comparisonError.toFixed(4) : "—",
        residualSigma != null && foldSigma != null
          ? `residual σ ${residualSigma.toFixed(4)} · raw fold σ ${foldSigma.toFixed(4)}`
          : "no per-fold scores recorded",
      )}
      {cell(
        "Effect size",
        effectSize != null ? effectSize.toFixed(4) : "—",
        resolution != null
          ? `${resolution.toFixed(1)}× ε — ${resolvable ? "the sweep can resolve this" : "below the noise floor"}`
          : "best minus median live trial",
        resolvable ? "text-[#b8e56e]" : "text-[#e8b957]",
      )}
      {cell(
        "Hard failures",
        badTrials.toLocaleString(),
        `${counts.dead} collapsed · ${counts.stalled} stalled · ${counts.failed} failed of ${(
          counts.alive + badTrials
        ).toLocaleString()}`,
        badTrials > 0 ? "text-[#e8b957]" : "text-white",
      )}
      {cell(
        "Truncation risk",
        truncatedShare != null ? `${Math.round(truncatedShare * 100)}%` : "—",
        "in-band trials whose best epoch was in the last fifth of the run",
        truncatedShare != null && truncatedShare > 0.25 ? "text-[#e8b957]" : "text-white",
      )}
      <button
        type="button"
        onClick={onOpen}
        className="focus-ring ml-auto flex items-center gap-1.5 px-4 text-[11px] font-semibold text-[#b6acff] hover:bg-white/[0.04]"
      >
        <Ruler size={13} /> Range finder
      </button>
    </div>
  );
}

// ------------------------------------------------------- Phase 0 range charts

/** Decade ticks on a log axis, plain linear ticks otherwise. */
function axisTicks(domain: [number, number], log: boolean, count: number): number[] {
  const [low, high] = domain;
  if (log && high - low >= 1.4) {
    const first = Math.ceil(low);
    const last = Math.floor(high);
    const decades: number[] = [];
    const stride = Math.max(1, Math.ceil((last - first + 1) / count));
    for (let value = first; value <= last; value += stride) decades.push(value);
    if (decades.length >= 2) return decades;
  }
  return ticks(domain, count);
}

function tickLabel(axisValue: number, log: boolean): string {
  const natural = log ? 10 ** axisValue : axisValue;
  if (log && Number.isInteger(axisValue)) {
    if (axisValue === 0) return "1";
    return `1e${axisValue}`;
  }
  return axisNumber(natural);
}

/**
 * The Phase 0 reading, in one figure: every trial against the parameter, the
 * tolerance band, the regions where runs die, the binned trend and the range
 * the rules recommend for Phase 1.
 *
 * The vertical axis is *shortfall* — each trial minus its own study's best — so
 * trials from several studies can share the plot, and, more importantly, so the
 * axis has a meaningful zero. Plotted raw, a sweep with one collapsed run gives
 * the collapse most of the axis and squeezes a 3% spread between the real
 * candidates into a couple of pixels. Here the failure tail is folded onto the
 * bottom edge as triangles instead of being allowed to set the scale.
 */
function ResponseCurve({
  finding,
  samples,
  epsilon,
  absoluteBest,
  metric,
  selectedTrial,
  onSelectTrial,
}: {
  finding: NumericFinding;
  samples: ParameterSample[];
  epsilon: number;
  /** Set when a single study is in view, so ticks can read in real units. */
  absoluteBest: number | null;
  metric: string;
  selectedTrial: TrialSelection | null;
  onSelectTrial: (selection: TrialSelection) => void;
}) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const { log } = finding;

  const width = 900;
  const height = 384;
  const padding = { top: 18, right: 20, bottom: 58, left: 66 };
  const plotLeft = padding.left;
  const plotRight = width - padding.right;
  const plotTop = padding.top;
  const plotBottom = height - padding.bottom;

  const alive = samples.filter(
    (sample) => sample.health === "alive" && Number.isFinite(sample.shortfall),
  );
  if (alive.length === 0) {
    return <EmptyChart message="No live trial sampled this parameter in the current selection." />;
  }

  // x spans whatever Phase 0 searched plus whatever the rule now recommends,
  // so an extension is visible as an extension rather than falling off-screen.
  const xCandidates = samples.map((sample) => toAxis(sample.x, log));
  if (finding.searched) {
    xCandidates.push(toAxis(finding.searched[0], log), toAxis(finding.searched[1], log));
  }
  if (finding.recommended) {
    xCandidates.push(toAxis(finding.recommended[0], log), toAxis(finding.recommended[1], log));
  }
  const xDomain = extent(xCandidates, 0.04);
  const x = linearScale(xDomain, [plotLeft, plotRight]);

  // The failure tail is clipped off the objective axis rather than allowed to
  // set the scale; clipped trials are pinned to the bottom edge, never dropped.
  const shortfalls = alive.map((sample) => sample.shortfall);
  const sorted = ascending(shortfalls);
  const firstQuartile = quantile(sorted, 0.25);
  const interQuartile = quantile(sorted, 0.75) - firstQuartile;
  const fence = interQuartile > 0 ? firstQuartile - 1.5 * interQuartile : -Infinity;
  // The floor never cuts closer than a few tolerance bands, so a sweep whose
  // trials really are all within epsilon does not get magnified into drama.
  const yLow = Math.max(Math.min(...shortfalls), Math.max(fence, -6 * epsilon));
  const clipped = alive.filter((sample) => sample.shortfall < yLow);
  const yHigh = Math.max(0, epsilon * 0.35);
  const yDomain: [number, number] =
    yHigh - yLow > 0 ? [yLow, yHigh] : [yLow - 0.01, yHigh + 0.01];
  const y = linearScale(yDomain, [plotBottom, plotTop]);
  const pin = (value: number) => Math.max(value, yDomain[0]);

  const binWidth =
    finding.bins.length > 0 ? finding.bins[0].to - finding.bins[0].from : 0;
  const deadBins = finding.bins.filter(
    (bin) => bin.count >= 2 && bin.deadCount / bin.count >= 0.5,
  );
  const trend = finding.bins
    .filter((bin) => bin.median != null)
    .map((bin) => ({ x: x((bin.from + bin.to) / 2), y: y(pin(bin.median as number)) }));

  const showErrorBars = samples.length <= 400;

  return (
    <div className="relative" data-chart-frame>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="block w-full"
        role="img"
        aria-label={`${parameterLabel(finding.name)} response with the derived Phase 1 range`}
      >
        <defs>
          <pattern
            id="dead-hatch"
            width="7"
            height="7"
            patternUnits="userSpaceOnUse"
            patternTransform="rotate(45)"
          >
            <line x1="0" y1="0" x2="0" y2="7" stroke="#e66767" strokeWidth="2" opacity="0.45" />
          </pattern>
        </defs>

        {deadBins.map((bin) => (
          <rect
            key={`dead-${bin.from}`}
            x={x(bin.from)}
            y={plotTop}
            width={Math.max(1, x(bin.to) - x(bin.from))}
            height={plotBottom - plotTop}
            fill="url(#dead-hatch)"
          />
        ))}

        {finding.good && (
          <rect
            x={x(toAxis(finding.good[0], log))}
            y={plotTop}
            width={Math.max(
              2,
              x(toAxis(finding.good[1], log)) - x(toAxis(finding.good[0], log)),
            )}
            height={plotBottom - plotTop}
            fill={CHART.accent}
            opacity={0.07}
          />
        )}

        <rect
          x={plotLeft}
          y={y(0)}
          width={plotRight - plotLeft}
          height={Math.max(1, y(-epsilon) - y(0))}
          fill="#199e70"
          opacity={0.16}
        />
        <line x1={plotLeft} x2={plotRight} y1={y(0)} y2={y(0)} stroke="#199e70" strokeWidth={1.5} />
        <line
          x1={plotLeft}
          x2={plotRight}
          y1={y(-epsilon)}
          y2={y(-epsilon)}
          stroke="#199e70"
          strokeWidth={1}
          strokeDasharray="4 3"
        />

        {ticks(yDomain, 5).map((tick) => (
          <g key={`y-${tick}`}>
            <line
              x1={plotLeft}
              x2={plotRight}
              y1={y(tick)}
              y2={y(tick)}
              stroke={CHART.grid}
              strokeWidth={1}
            />
            <text
              x={plotLeft - 8}
              y={y(tick) + 3}
              textAnchor="end"
              fontSize={9}
              fill={CHART.muted}
              style={{ fontVariantNumeric: "tabular-nums" }}
            >
              {absoluteBest != null
                ? (absoluteBest + tick).toFixed(3)
                : tick.toFixed(3)}
            </text>
          </g>
        ))}

        {finding.searched &&
          ([finding.searched[0], finding.searched[1]] as const).map((edge, index) => (
            <g key={`searched-${index}`}>
              <line
                x1={x(toAxis(edge, log))}
                x2={x(toAxis(edge, log))}
                y1={plotTop}
                y2={plotBottom}
                stroke={CHART.axis}
                strokeWidth={1}
                strokeDasharray="5 4"
              />
              <text
                x={x(toAxis(edge, log)) + (index === 0 ? 4 : -4)}
                y={plotTop + 10}
                textAnchor={index === 0 ? "start" : "end"}
                fontSize={9}
                fill={CHART.muted}
              >
                Phase 0 {index === 0 ? "low" : "high"}
              </text>
            </g>
          ))}

        {alive.map((sample, index) => {
          const isClipped = sample.shortfall < yLow;
          const isSelected =
            selectedTrial?.studyId === sample.study.id &&
            selectedTrial.trialNumber === sample.trialNumber;
          const cx = x(toAxis(sample.x, log));
          const cy = y(pin(sample.shortfall));
          const select = () =>
            onSelectTrial({ studyId: sample.study.id, trialNumber: sample.trialNumber });
          const selectWithKeyboard = (event: { key: string; preventDefault: () => void }) => {
            if (event.key === "Enter" || event.key === " ") {
              event.preventDefault();
              select();
            }
          };
          const tip = (event: { clientX: number; clientY: number; currentTarget: Element }) => {
            const box = event.currentTarget
              .closest("[data-chart-frame]")
              ?.getBoundingClientRect();
            if (!box) return;
            setTooltip({
              x: event.clientX - box.left,
              y: event.clientY - box.top,
              lines: [
                `${shortPath(sample.study.path)} · trial #${sample.trialNumber}`,
                `${parameterLabel(finding.name)} = ${formatNumber(sample.x)}`,
                `${sample.shortfall === 0 ? "best of its study" : `${sample.shortfall.toFixed(4)} below its study's best`}`,
                sample.error != null ? `±${sample.error.toFixed(4)} across folds` : "",
                sample.inBand ? "inside the tolerance band" : "",
                "Click for trial details",
              ].filter(Boolean),
            });
          };
          if (isClipped) {
            return (
              <path
                key={`clip-${index}`}
                className="focus-ring cursor-pointer"
                d={`M ${cx} ${plotBottom - 2} l 4 7 l -8 0 z`}
                fill={CHART.muted}
                stroke={isSelected ? "#ffffff" : "none"}
                strokeWidth={isSelected ? 2 : 0}
                role="button"
                tabIndex={0}
                aria-label={`Show details for trial #${sample.trialNumber}`}
                onClick={select}
                onKeyDown={selectWithKeyboard}
                onPointerEnter={tip}
                onPointerLeave={() => setTooltip(null)}
              />
            );
          }
          return (
            <g key={`point-${index}`}>
              {showErrorBars && sample.error != null && (
                <line
                  x1={cx}
                  x2={cx}
                  y1={y(pin(sample.shortfall - sample.error))}
                  y2={y(Math.min(yDomain[1], sample.shortfall + sample.error))}
                  stroke={sample.inBand ? "#3987e5" : CHART.axis}
                  strokeWidth={1}
                  opacity={0.55}
                />
              )}
              <circle
                className="focus-ring chart-point cursor-pointer"
                cx={cx}
                cy={cy}
                r={isSelected ? 6 : sample.inBand ? 4 : 3}
                fill={sample.inBand ? "#3987e5" : "#5c6a82"}
                fillOpacity={isSelected ? 1 : sample.inBand ? 0.95 : 0.6}
                stroke={isSelected ? "#ffffff" : "#0a1120"}
                strokeWidth={isSelected ? 2 : 1}
                role="button"
                tabIndex={0}
                aria-label={`Show details for trial #${sample.trialNumber}`}
                onClick={select}
                onKeyDown={selectWithKeyboard}
                onPointerEnter={tip}
                onPointerLeave={() => setTooltip(null)}
              />
            </g>
          );
        })}

        {samples
          .filter((sample) => sample.health !== "alive")
          .map((sample, index) => {
            const isSelected =
              selectedTrial?.studyId === sample.study.id &&
              selectedTrial.trialNumber === sample.trialNumber;
            const select = () =>
              onSelectTrial({ studyId: sample.study.id, trialNumber: sample.trialNumber });
            return (
              <path
                key={`bad-${index}`}
                className="focus-ring cursor-pointer"
                d={`M ${x(toAxis(sample.x, log))} ${plotBottom + 6} l 4 7 l -8 0 z`}
                fill="#e66767"
                opacity={isSelected ? 1 : 0.8}
                stroke={isSelected ? "#ffffff" : "none"}
                strokeWidth={isSelected ? 2 : 0}
                role="button"
                tabIndex={0}
                aria-label={`Show details for trial #${sample.trialNumber}`}
                onClick={select}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    select();
                  }
                }}
                onPointerEnter={(event) => {
                  const box = event.currentTarget
                    .closest("[data-chart-frame]")
                    ?.getBoundingClientRect();
                  if (!box) return;
                  setTooltip({
                    x: event.clientX - box.left,
                    y: event.clientY - box.top,
                    lines: [
                      `${shortPath(sample.study.path)} · trial #${sample.trialNumber}`,
                      `${parameterLabel(finding.name)} = ${formatNumber(sample.x)}`,
                      HEALTH_LABELS[sample.health],
                      "Click for trial details",
                    ],
                  });
                }}
                onPointerLeave={() => setTooltip(null)}
              />
            );
          })}

        {trend.length >= 2 && (
          <polyline
            points={trend.map((point) => `${point.x},${point.y}`).join(" ")}
            fill="none"
            stroke={CHART.accent}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}

        {finding.recommended && (
          <g>
            <line
              x1={x(toAxis(finding.recommended[0], log))}
              x2={x(toAxis(finding.recommended[1], log))}
              y1={height - 30}
              y2={height - 30}
              stroke="#c98500"
              strokeWidth={2.5}
            />
            {([finding.recommended[0], finding.recommended[1]] as const).map((edge, index) => (
              <line
                key={`rec-${index}`}
                x1={x(toAxis(edge, log))}
                x2={x(toAxis(edge, log))}
                y1={height - 36}
                y2={height - 24}
                stroke="#c98500"
                strokeWidth={2.5}
              />
            ))}
            <text
              x={(x(toAxis(finding.recommended[0], log)) + x(toAxis(finding.recommended[1], log))) / 2}
              y={height - 36}
              textAnchor="middle"
              fontSize={9}
              fill="#e0a53a"
            >
              recommended Phase 1 range
            </text>
          </g>
        )}

        <line
          x1={plotLeft}
          x2={plotRight}
          y1={plotBottom}
          y2={plotBottom}
          stroke={CHART.axis}
          strokeWidth={1}
        />
        {axisTicks(xDomain, log, 7).map((tick) => (
          <text
            key={`x-${tick}`}
            x={x(tick)}
            y={plotBottom + 14}
            textAnchor="middle"
            fontSize={9}
            fill={CHART.muted}
          >
            {tickLabel(tick, log)}
          </text>
        ))}
      </svg>
      <ChartTooltip tooltip={tooltip} />
      <div className="mt-2 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-[10px] text-[#79869e]">
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-[#3987e5]" /> in band
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-[#5c6a82]" /> live, outside band
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 bg-[#e66767]" style={{ clipPath: "polygon(50% 0, 100% 100%, 0 100%)" }} />
          failed, stalled or collapsed
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-4 bg-[#199e70]/30 ring-1 ring-[#199e70]" /> within ε of best
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4 bg-[#9b8cff]" /> binned median
        </span>
        <span className="inline-flex items-center gap-1.5">
          <span className="inline-block h-0.5 w-4 bg-[#c98500]" /> recommendation
        </span>
      </div>
      <p className="mt-1 text-[10px] leading-4 text-[#68768e]">
        Click any marker for its trial details. y is each trial minus its own study&rsquo;s best{absoluteBest != null ? `, labelled in ${metricLabel(metric)}` : ""}
        ; the axis spans {(yDomain[1] - yDomain[0]).toFixed(4)}, so the ε band is{" "}
        {Math.round((epsilon / (yDomain[1] - yDomain[0])) * 100)}% of its height.
        {clipped.length > 0
          ? ` ${clipped.length} trial${clipped.length === 1 ? "" : "s"} below ${yLow.toFixed(3)} folded onto the bottom edge.`
          : ""}
      </p>
    </div>
  );
}

function rangeText(range: [number, number] | null, isInt: boolean): string {
  if (!range) return "—";
  const format = (value: number) =>
    isInt ? String(Math.round(value)) : formatNumber(value, 3);
  return `${format(range[0])} … ${format(range[1])}`;
}

/**
 * One row per searched parameter: what Phase 0 covered, what the tolerance band
 * actually spans, and the range the rules put forward for Phase 1.
 */
function FindingsTable({
  findings,
  selected,
  onSelect,
}: {
  findings: RangeFinding[];
  selected: string | null;
  onSelect: (name: string) => void;
}) {
  if (findings.length === 0) {
    return <EmptyChart message="No study in this selection searched a hyperparameter." />;
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[980px] text-[11px]">
        <thead>
          <tr className="border-b border-white/[0.07]">
            {[
              "Parameter",
              "Phase 0 range",
              "Inside ε of best",
              "Verdict",
              "Recommended Phase 1",
              "Effect / ε",
              "In band",
            ].map((label, index) => (
              <th
                key={label}
                className={`px-2.5 py-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e] ${
                  index >= 5 ? "text-right" : "text-left"
                }`}
              >
                {label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {findings.map((finding) => {
            const isSelected = finding.name === selected;
            const unresolved = !finding.resolvable;
            return (
              <tr
                key={finding.name}
                onClick={() => onSelect(finding.name)}
                className={`cursor-pointer border-b border-white/[0.04] transition ${
                  isSelected ? "bg-[#9b8cff]/[0.11]" : "hover:bg-white/[0.035]"
                }`}
              >
                <td className="max-w-[18rem] px-2.5 py-2" title={finding.name}>
                  <span className="block truncate text-[#dbe2ef]">
                    {parameterLabel(finding.name)}
                  </span>
                  <span className="mt-0.5 block truncate text-[10px] text-[#68768e]">
                    {finding.kind === "numeric"
                      ? `${finding.log ? "log" : "linear"} ${finding.isInt ? "int" : "float"}`
                      : "categorical"}
                    {" · "}
                    {finding.studyCount} stud{finding.studyCount === 1 ? "y" : "ies"}
                  </span>
                </td>
                {finding.kind === "numeric" ? (
                  <>
                    <td className="px-2.5 py-2 tabular-nums text-[#8f9bb2]">
                      {rangeText(finding.searched, finding.isInt)}
                    </td>
                    <td className="px-2.5 py-2 tabular-nums text-[#c9d2e1]">
                      {rangeText(finding.good, finding.isInt)}
                    </td>
                    <td className="px-2.5 py-2">
                      <VerdictPill verdict={finding.verdict} muted={unresolved} />
                      {unresolved && (
                        <span className="ml-1.5 whitespace-nowrap text-[10px] text-[#e8b957]">
                          unresolved
                        </span>
                      )}
                    </td>
                    <td className="px-2.5 py-2 tabular-nums text-white">
                      {rangeText(finding.recommended, finding.isInt)}
                      {finding.clampedLow && (
                        <span className="ml-1 text-[10px] text-[#68768e]">clamped</span>
                      )}
                      {finding.beyondSearched && (
                        <span
                          className="ml-1 text-[10px] text-[#e8b957]"
                          title="This range goes outside what Phase 0 searched. Check that the extended bound is valid for the parameter — the rule knows the sweep, not the parameter's domain."
                        >
                          beyond Phase 0
                        </span>
                      )}
                    </td>
                  </>
                ) : (
                  <>
                    <td className="px-2.5 py-2 text-[#8f9bb2]">
                      {finding.choices.map((choice) => choice.label).join(", ") || "—"}
                    </td>
                    <td className="px-2.5 py-2 text-[#c9d2e1]">
                      {finding.surviving.map(String).join(", ") || "—"}
                    </td>
                    <td className="px-2.5 py-2">
                      <VerdictPill verdict={finding.verdict} muted={unresolved} />
                      {unresolved && (
                        <span className="ml-1.5 whitespace-nowrap text-[10px] text-[#e8b957]">
                          unresolved
                        </span>
                      )}
                    </td>
                    <td className="px-2.5 py-2 text-white">
                      {finding.dropped.length > 0
                        ? `drop ${finding.dropped.map(String).join(", ")}`
                        : "keep every choice"}
                    </td>
                  </>
                )}
                <td className="px-2.5 py-2 text-right tabular-nums text-[#8f9bb2]">
                  {finding.effectSize != null ? finding.effectSize.toFixed(4) : "—"}
                </td>
                <td className="px-2.5 py-2 text-right tabular-nums text-[#8f9bb2]">
                  {finding.keepCount}/{finding.sampleCount}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}


// ------------------------------------------------------------ budget planning

/**
 * P(at least one startup trial lands inside the tolerance band) against the
 * number of startup trials, with the confidence threshold drawn in. The curve
 * is the exact subset probability over the trials this sweep actually ran, so
 * it needs no resampling and it already carries whatever dimensionality cost
 * the search space imposed — a good region that shrank in higher dimensions
 * shows up as a lower hit rate and a curve that crosses later.
 */
function StartupCurve({
  curve,
  crossing,
  confidence,
}: {
  curve: Array<{ n: number; probability: number }>;
  crossing: number | null;
  confidence: number;
}) {
  if (curve.length < 2) {
    return <div className="h-[74px]" />;
  }
  const width = 260;
  const height = 74;
  const padding = { top: 8, right: 8, bottom: 16, left: 24 };
  const x = linearScale(
    [curve[0].n, curve[curve.length - 1].n],
    [padding.left, width - padding.right],
  );
  const y = linearScale([0, 1], [height - padding.bottom, padding.top]);
  const line = curve.map((point) => `${x(point.n)},${y(point.probability)}`).join(" ");

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="block w-full" role="img" aria-label="Chance of hitting the good region against the number of startup trials">
      <line
        x1={padding.left}
        x2={width - padding.right}
        y1={y(confidence)}
        y2={y(confidence)}
        stroke="#199e70"
        strokeWidth={1}
        strokeDasharray="4 3"
      />
      <text x={padding.left - 4} y={y(confidence) + 3} textAnchor="end" fontSize={8} fill="#199e70">
        {Math.round(confidence * 100)}%
      </text>
      <polyline
        points={line}
        fill="none"
        stroke={CHART.accent}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      {crossing != null && (
        <>
          <line
            x1={x(crossing)}
            x2={x(crossing)}
            y1={padding.top}
            y2={height - padding.bottom}
            stroke="#c98500"
            strokeWidth={1.5}
          />
          <text
            x={x(crossing) + 4}
            y={padding.top + 8}
            fontSize={8}
            fill="#e0a53a"
          >
            n={crossing}
          </text>
        </>
      )}
      <line
        x1={padding.left}
        x2={width - padding.right}
        y1={height - padding.bottom}
        y2={height - padding.bottom}
        stroke={CHART.axis}
        strokeWidth={1}
      />
      {[curve[0].n, curve[curve.length - 1].n].map((tick, index) => (
        <text
          key={tick}
          x={x(tick)}
          y={height - 4}
          textAnchor={index === 0 ? "start" : "end"}
          fontSize={8}
          fill={CHART.muted}
        >
          {tick}
        </text>
      ))}
    </svg>
  );
}

/**
 * The two budget questions a Phase 0 sweep answers about Phase 1: how many
 * startup trials TPE needs before its better group is worth modelling, and
 * whether the epoch budget was ever the binding constraint.
 */
function BudgetHeader({
  startup,
  epochs,
  epsilon,
  onOpenRanges,
}: {
  startup: StartupSummary;
  epochs: EpochBudget;
  epsilon: number | null;
  onOpenRanges: () => void;
}) {
  const median = startup.perStudy.length > 0
    ? [...startup.perStudy].sort(
        (left, right) => (left.recommended ?? 0) - (right.recommended ?? 0),
      )[Math.floor(startup.perStudy.length / 2)]
    : null;
  const budgetShare = epochs.share ?? 0;
  const budgetBinding = budgetShare >= 0.05;

  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <section className="surface p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-[0.13em] text-[#79869e]">
              TPE startup trials
            </p>
            <p className="mt-1 flex items-baseline gap-2">
              <span className="text-3xl font-semibold tracking-[-0.03em] text-white">
                {startup.median ?? "—"}
              </span>
              <span className="text-[11px] text-[#8f9bb2]">
                per study, median of {startup.studies}
              </span>
            </p>
            <p className="mt-1 text-[11px] leading-5 text-[#8f9bb2]">
              {startup.p != null ? (
                <>
                  {(startup.p * 100).toFixed(1)}% of trials land within ε
                  {epsilon != null ? ` = ${epsilon.toFixed(4)}` : ""} of their study&rsquo;s best,
                  so n ≥ ln(1−{startup.confidence})/ln(1−p) gives{" "}
                  {median?.analytic ?? "—"} and the exact subset draw gives{" "}
                  {median?.empirical ?? "—"}.
                </>
              ) : (
                "No scored study in the current selection."
              )}
            </p>
          </div>
          <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#9b8cff]/10 text-[#b6acff]">
            <Sigma size={17} />
          </span>
        </div>
        <div className="mt-2">
          <StartupCurve
            curve={median?.curve ?? []}
            crossing={median?.empirical ?? null}
            confidence={startup.confidence}
          />
        </div>
        <p className="mt-1 text-[10px] leading-4 text-[#68768e]">
          Raised to a floor of {startup.floor}: TPE&rsquo;s better group is the top ⌈0.1·N⌉ trials,
          so ten startup trials build the good density from a single point plus the prior.
          {startup.worst != null && startup.median != null && startup.worst > startup.median
            ? ` The hardest study in this selection needs ${startup.worst}.`
            : ""}{" "}
          <button
            type="button"
            onClick={onOpenRanges}
            className="focus-ring rounded font-semibold text-[#9b8cff] hover:underline"
          >
            ε is set in the Range finder
          </button>
          .
        </p>
      </section>

      <section className="surface p-4">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <p className="text-[10px] font-semibold uppercase tracking-[0.13em] text-[#79869e]">
              Epoch budget
            </p>
            <p className="mt-1 flex items-baseline gap-2">
              <span
                className={`text-3xl font-semibold tracking-[-0.03em] ${
                  budgetBinding ? "text-[#e8b957]" : "text-white"
                }`}
              >
                {epochs.folds > 0 ? `${(budgetShare * 100).toFixed(1)}%` : "—"}
              </span>
              <span className="text-[11px] text-[#8f9bb2]">
                of {epochs.folds.toLocaleString()} folds ran out of epochs
              </span>
            </p>
            <p className="mt-1 text-[11px] leading-5 text-[#8f9bb2]">
              {epochs.budget != null
                ? `Budget ${epochs.budget} epochs, so a fold that reaches epoch ${
                    epochs.budget - 1
                  } never early-stopped.`
                : epochs.budgets.length > 1
                  ? `Mixed budgets in view: ${epochs.budgets.join(", ")} epochs.`
                  : "No epoch budget recorded."}{" "}
              Median fold stops at {epochs.medianLast ?? "—"} with its best epoch at{" "}
              {epochs.medianSelected ?? "—"}.
            </p>
          </div>
          <span
            className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ${
              budgetBinding
                ? "bg-[#c98500]/10 text-[#e8b957]"
                : "bg-[#4fd5e7]/10 text-[#64dce9]"
            }`}
          >
            <Hourglass size={17} />
          </span>
        </div>
        <div className="mt-3 h-2.5 overflow-hidden rounded-full bg-white/[0.05]">
          <div
            className="h-full rounded-full"
            style={{
              width: `${Math.max(budgetShare * 100, budgetShare > 0 ? 1.5 : 0)}%`,
              background: budgetBinding ? "#c98500" : "#199e70",
            }}
          />
        </div>
        <p className="mt-2 text-[10px] leading-4 text-[#68768e]">
          {epochs.folds === 0
            ? "No fold recorded its last epoch."
            : budgetBinding
              ? `${epochs.atBudget.toLocaleString()} folds across ${epochs.studiesAtBudget} of ${epochs.studies} studies hit the ceiling while still inside their patience window. Their scores would move if the budget went up.`
              : `${epochs.atBudget.toLocaleString()} folds hit the ceiling. Early stopping is doing the work, so the budget is not what is limiting these runs.`}
        </p>
      </section>
    </div>
  );
}

// ------------------------------------------------------------- per-fold views

/** Blue below, red above, neutral in the middle — polarity around a baseline. */
const DIVERGING_LOW = ["#0d366b", "#256abf", "#6da7ec", "#b7d3f6"];
const DIVERGING_HIGH = ["#f6cccc", "#e88f8f", "#d64c4c", "#a02020"];
const DIVERGING_MID = "#2a3244";

function divergingColor(value: number, scale: number): string {
  if (!Number.isFinite(value) || scale <= 0) return DIVERGING_MID;
  const position = Math.min(1, Math.abs(value) / scale);
  if (position < 0.08) return DIVERGING_MID;
  const ramp = value < 0 ? DIVERGING_LOW : DIVERGING_HIGH;
  const index = Math.min(ramp.length - 1, Math.floor(position * ramp.length));
  return ramp[value < 0 ? ramp.length - 1 - index : index];
}

function divergingInk(value: number, scale: number): string {
  const position = Math.min(1, Math.abs(value) / scale);
  return position > 0.6 ? "#f4f7fd" : "#dbe2ef";
}

/**
 * How much harder or easier each fold is than the trial it belongs to.
 *
 * This is the fold main effect: the same shared offset the noise model removes
 * before measuring anything. It is not noise — every trial sees it — so a fold
 * sitting well away from zero here is a property of the split, not of the
 * hyperparameters, and reading it as trial-to-trial variation is what makes a
 * sweep look noisier than it is.
 */
function FoldEffectChart({ rows }: { rows: FoldRow[] }) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const usable = rows.filter((row) => row.effect != null);
  if (usable.length === 0) {
    return <EmptyChart message="No trial in this selection recorded per-fold scores." />;
  }

  const rowHeight = 30;
  const labelWidth = 82;
  const width = 780;
  const height = usable.length * rowHeight + 26;
  const plotLeft = labelWidth;
  const plotRight = width - 70;
  const spread = usable.flatMap((row) =>
    [row.effect, row.effectP10, row.effectP90].filter(
      (value): value is number => value != null,
    ),
  );
  const domain = extent([...spread, 0], 0.08);
  const x = linearScale(domain, [plotLeft, plotRight]);

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="block w-full"
        role="img"
        aria-label="Per-fold offset from the trial mean"
      >
        {ticks(domain, 6).map((tick) => (
          <line
            key={tick}
            x1={x(tick)}
            x2={x(tick)}
            y1={4}
            y2={height - 22}
            stroke={CHART.grid}
            strokeWidth={1}
          />
        ))}
        <line
          x1={x(0)}
          x2={x(0)}
          y1={4}
          y2={height - 22}
          stroke={CHART.axis}
          strokeWidth={1.5}
        />
        {usable.map((row, index) => {
          const y = index * rowHeight + 6;
          const centre = y + (rowHeight - 10) / 2;
          const effect = row.effect as number;
          return (
            <g
              key={row.fold}
              onPointerMove={(event) => {
                const box = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
                if (!box) return;
                setTooltip({
                  x: event.clientX - box.left,
                  y: event.clientY - box.top,
                  lines: [
                    `Fold ${row.fold}`,
                    `${effect >= 0 ? "+" : ""}${effect.toFixed(4)} against the trial mean`,
                    `median ${formatMetric(row.medianValue)} · P@1 ${formatMetric(row.medianPrecisionAt1)}`,
                    `${row.trials} folds · best epoch ${row.medianSelectedEpoch ?? "—"} of ${row.medianLastEpoch ?? "—"}`,
                  ],
                });
              }}
              onPointerLeave={() => setTooltip(null)}
            >
              <rect x={0} y={y - 2} width={width} height={rowHeight - 4} fill="transparent" />
              <text
                x={labelWidth - 12}
                y={centre + 4}
                textAnchor="end"
                fontSize={11}
                fill={CHART.text}
              >
                Fold {row.fold}
              </text>
              {row.effectP10 != null && row.effectP90 != null && (
                <line
                  x1={x(row.effectP10)}
                  x2={x(row.effectP90)}
                  y1={centre}
                  y2={centre}
                  stroke={CHART.accent}
                  strokeOpacity={0.45}
                  strokeWidth={2}
                />
              )}
              <circle
                cx={x(effect)}
                cy={centre}
                r={5}
                fill={CHART.accent}
                stroke="#0a1120"
                strokeWidth={1.5}
              />
              <text
                x={plotRight + 8}
                y={centre + 4}
                fontSize={11}
                fill={CHART.text}
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {effect >= 0 ? "+" : ""}
                {effect.toFixed(4)}
              </text>
            </g>
          );
        })}
        {ticks(domain, 6).map((tick) => (
          <text
            key={`label-${tick}`}
            x={x(tick)}
            y={height - 7}
            textAnchor="middle"
            fontSize={9}
            fill={CHART.muted}
          >
            {axisNumber(tick)}
          </text>
        ))}
      </svg>
      <ChartTooltip tooltip={tooltip} />
      <p className="mt-1 text-[10px] text-[#68768e]">
        Dot is the median offset, the bar its inner-decile spread across trials.
      </p>
    </div>
  );
}

/**
 * Trial by fold, each cell coloured by how far that fold sat from the trial's
 * own mean. Vertical stripes are a fold that is systematically hard; scattered
 * colour is genuine trial-by-fold interaction, which is the part that actually
 * limits how finely two trials can be told apart.
 */
function FoldMatrix({
  assessment,
  selectedTrial,
  onSelectTrial,
}: {
  assessment: StudyAssessment;
  selectedTrial: TrialSelection | null;
  onSelectTrial?: (trialNumber: number) => void;
}) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const rows = assessment.trials
    .filter((trial) => (trial.trial.folds?.length ?? 0) > 0)
    .map((trial) => {
      const folds = trial.trial.folds ?? [];
      const values = folds.map((fold) => fold.value).filter(Number.isFinite);
      const mean = values.length > 0 ? values.reduce((a, b) => a + b, 0) / values.length : null;
      return { trial, folds, mean };
    });
  if (rows.length === 0) {
    return <EmptyChart message="This study recorded no per-fold scores." />;
  }

  const foldNumbers = [
    ...new Set(rows.flatMap((row) => row.folds.map((fold) => fold.fold))),
  ].sort((left, right) => left - right);
  const deviations = rows.flatMap((row) =>
    row.mean == null ? [] : row.folds.map((fold) => fold.value - (row.mean as number)),
  );
  const scale =
    deviations.length > 0
      ? Math.max(...deviations.map((value) => Math.abs(value))) || 1
      : 1;

  return (
    <div className="relative" data-chart-frame>
      <div className="max-h-[52vh] overflow-auto">
        <table className="w-full min-w-[520px] border-separate border-spacing-[2px] text-[10px]">
          <thead className="sticky top-0 z-10 bg-[#10182a]">
            <tr>
              <th className="px-2 py-1 text-left text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e]">
                Trial
              </th>
              {foldNumbers.map((fold) => (
                <th key={fold} className="px-2 py-1 text-center text-[10px] text-[#a8b3c7]">
                  F{fold}
                </th>
              ))}
              <th className="px-2 py-1 text-right text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e]">
                Mean
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map(({ trial, folds, mean }) => {
              const highlighted =
                selectedTrial?.studyId === assessment.study.id &&
                selectedTrial.trialNumber === trial.trial.number;
              return (
              <tr
                key={trial.trial.number}
                className={highlighted ? "ring-1 ring-inset ring-[#9b8cff]/50" : ""}
              >
                <th
                  scope="row"
                  className={`px-2 py-1 text-left font-medium ${
                    highlighted
                      ? "text-white"
                      : trial.inBand
                        ? "text-[#7db4f0]"
                        : "text-[#8f9bb2]"
                  }`}
                >
                  #{trial.trial.number}
                  {trial.inBand && <span className="ml-1 text-[9px]">in band</span>}
                </th>
                {foldNumbers.map((foldNumber) => {
                  const fold = folds.find((entry) => entry.fold === foldNumber);
                  if (!fold || mean == null) {
                    return (
                      <td
                        key={foldNumber}
                        className="rounded border border-dashed border-white/[0.06] px-2 py-1 text-center text-[#4d5a72]"
                      >
                        —
                      </td>
                    );
                  }
                  const deviation = fold.value - mean;
                  return (
                    <td key={foldNumber} className="p-0">
                      <button
                        type="button"
                        onClick={() => onSelectTrial?.(trial.trial.number)}
                        onPointerEnter={(event) => {
                          const box = event.currentTarget
                            .closest("[data-chart-frame]")
                            ?.getBoundingClientRect();
                          if (!box) return;
                          setTooltip({
                            x: event.clientX - box.left,
                            y: event.clientY - box.top,
                            lines: [
                              `Trial #${trial.trial.number}, fold ${fold.fold}`,
                              `${formatMetric(fold.value)} (${deviation >= 0 ? "+" : ""}${deviation.toFixed(4)} vs its trial mean)`,
                              `P@1 ${formatMetric(fold.precisionAt1)}`,
                              `best epoch ${fold.selectedEpoch ?? "—"} of ${fold.lastEpoch ?? "—"}`,
                            ],
                          });
                        }}
                        onPointerLeave={() => setTooltip(null)}
                        className="focus-ring w-full rounded px-2 py-1 text-center tabular-nums transition hover:brightness-125"
                        style={{
                          background: divergingColor(deviation, scale),
                          color: divergingInk(deviation, scale),
                        }}
                      >
                        {fold.value.toFixed(3)}
                      </button>
                    </td>
                  );
                })}
                <td className="px-2 py-1 text-right tabular-nums text-white">
                  {mean != null ? mean.toFixed(4) : "—"}
                </td>
              </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <ChartTooltip tooltip={tooltip} />
      <div className="mt-2 flex items-center gap-2 text-[10px] text-[#79869e]">
        <span>below trial mean</span>
        <span className="flex h-2.5 overflow-hidden rounded-full">
          {[...DIVERGING_LOW].reverse().concat(DIVERGING_MID, DIVERGING_HIGH).map((color, index) => (
            <span key={`${color}-${index}`} className="block h-full w-4" style={{ background: color }} />
          ))}
        </span>
        <span>above</span>
        <span className="ml-2 tabular-nums">±{scale.toFixed(3)}</span>
      </div>
    </div>
  );
}

// ------------------------------------------------------------ gradient views

function GradientFoldTable({
  rows,
  aggregate,
  seriesLabel,
  hasTarget,
}: {
  rows: GradientFoldRow[];
  aggregate: GradientFoldRow | null;
  seriesLabel: string;
  hasTarget: boolean;
}) {
  if (rows.length === 0) {
    return <EmptyChart message="No fold in this selection logged gradient diagnostics." />;
  }

  const cells = (row: GradientFoldRow, label: string, emphasis: boolean) => (
    <tr
      key={label}
      className={
        emphasis
          ? "border-b border-white/[0.1] bg-white/[0.03] font-semibold"
          : "border-b border-white/[0.04]"
      }
    >
      <td className="px-3 py-2 text-[#c9d2e1]">{label}</td>
      <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">{row.samples}</td>
      <td className="px-3 py-2 text-right tabular-nums text-white">
        {row.median != null ? formatNumber(row.median, 4) : "—"}
      </td>
      <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
        {row.p10 != null && row.p90 != null
          ? `${formatNumber(row.p10, 3)} – ${formatNumber(row.p90, 3)}`
          : "—"}
      </td>
      <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
        {row.decades != null ? `${row.decades.toFixed(2)} dec` : "—"}
      </td>
      <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
        {row.cosine != null ? row.cosine.toFixed(3) : "—"}
      </td>
      {hasTarget && (
        <td className="px-3 py-2 text-right tabular-nums">
          {row.targetDeviation != null ? (
            <span
              className={
                Math.abs(row.targetDeviation) > 0.3 ? "text-[#e8b957]" : "text-[#8f9bb2]"
              }
            >
              {row.targetDeviation >= 0 ? "+" : ""}
              {row.targetDeviation.toFixed(2)} dec
            </span>
          ) : (
            "—"
          )}
        </td>
      )}
    </tr>
  );

  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[680px] text-[11px]">
        <thead>
          <tr className="border-b border-white/[0.07]">
            {[
              { label: "Fold", align: "left" },
              { label: "Folds seen", align: "right" },
              { label: seriesLabel, align: "right" },
              { label: "p10 – p90", align: "right" },
              { label: "Within-fold spread", align: "right" },
              { label: "Cosine", align: "right" },
              ...(hasTarget ? [{ label: "Off target", align: "right" as const }] : []),
            ].map((column) => (
              <th
                key={column.label}
                className={`px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e] ${
                  column.align === "right" ? "text-right" : "text-left"
                }`}
              >
                {column.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {aggregate && cells(aggregate, "All folds", true)}
          {rows.map((row) => cells(row, `Fold ${row.fold}`, false))}
        </tbody>
      </table>
    </div>
  );
}

/**
 * How far each fold's achieved regularizer/supervised ratio sat from the ratio
 * its controller was told to hold, in decades. Zero is on target, +1 means the
 * regularizer ended up ten times stronger than asked. A controller that misses
 * by more than a couple of tenths is not regulating anything, and the weight it
 * reports is then a number the run wrote down rather than one it acted on.
 */
function TargetDeviationChart({ rows }: { rows: GradientFoldRow[] }) {
  const [tooltip, setTooltip] = useState<TooltipState | null>(null);
  const usable = rows.filter((row) => row.targetDeviation != null);
  if (usable.length === 0) {
    return (
      <EmptyChart message="No study in this selection ran a controller with a target ratio." />
    );
  }

  const rowHeight = 30;
  const labelWidth = 82;
  const width = 780;
  const height = usable.length * rowHeight + 26;
  const plotLeft = labelWidth;
  const plotRight = width - 78;
  const values = usable.flatMap((row) =>
    [row.targetDeviation, row.targetAbsDeviation].filter(
      (value): value is number => value != null,
    ),
  );
  const domain = extent([...values, 0], 0.1);
  const x = linearScale(domain, [plotLeft, plotRight]);

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="block w-full"
        role="img"
        aria-label="Deviation from the controller's target ratio, per fold"
      >
        {ticks(domain, 6).map((tick) => (
          <line
            key={tick}
            x1={x(tick)}
            x2={x(tick)}
            y1={4}
            y2={height - 22}
            stroke={CHART.grid}
            strokeWidth={1}
          />
        ))}
        <line x1={x(0)} x2={x(0)} y1={4} y2={height - 22} stroke="#199e70" strokeWidth={2} />
        {usable.map((row, index) => {
          const y = index * rowHeight + 6;
          const centre = y + (rowHeight - 10) / 2;
          const deviation = row.targetDeviation as number;
          const off = Math.abs(deviation) > 0.3;
          return (
            <g
              key={row.fold}
              onPointerMove={(event) => {
                const box = event.currentTarget.ownerSVGElement?.getBoundingClientRect();
                if (!box) return;
                setTooltip({
                  x: event.clientX - box.left,
                  y: event.clientY - box.top,
                  lines: [
                    `Fold ${row.fold}`,
                    `${deviation >= 0 ? "+" : ""}${deviation.toFixed(3)} decades off target`,
                    `achieved ratio is ${(10 ** deviation).toPrecision(3)}x the target`,
                    `${row.samples} folds`,
                  ],
                });
              }}
              onPointerLeave={() => setTooltip(null)}
            >
              <rect x={0} y={y - 2} width={width} height={rowHeight - 4} fill="transparent" />
              <text
                x={labelWidth - 12}
                y={centre + 4}
                textAnchor="end"
                fontSize={11}
                fill={CHART.text}
              >
                Fold {row.fold}
              </text>
              <line
                x1={x(0)}
                x2={x(deviation)}
                y1={centre}
                y2={centre}
                stroke={off ? "#c98500" : "#199e70"}
                strokeWidth={2}
                strokeOpacity={0.5}
              />
              <circle
                cx={x(deviation)}
                cy={centre}
                r={5}
                fill={off ? "#c98500" : "#199e70"}
                stroke="#0a1120"
                strokeWidth={1.5}
              />
              <text
                x={plotRight + 8}
                y={centre + 4}
                fontSize={11}
                fill={off ? "#e8b957" : CHART.text}
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {(10 ** deviation).toPrecision(3)}×
              </text>
            </g>
          );
        })}
        {ticks(domain, 6).map((tick) => (
          <text
            key={`label-${tick}`}
            x={x(tick)}
            y={height - 7}
            textAnchor="middle"
            fontSize={9}
            fill={CHART.muted}
          >
            {tick.toFixed(1)}
          </text>
        ))}
      </svg>
      <ChartTooltip tooltip={tooltip} />
      <p className="mt-1 text-[10px] text-[#68768e]">
        x is log10(achieved / target); the right-hand number is the same thing as a multiple.
        Amber marks folds more than 0.3 decades (2x) off.
      </p>
    </div>
  );
}

// ----------------------------------------------------------------- the screen

type Tab =
  | "compare"
  | "studies"
  | "folds"
  | "gradients"
  | "hyperparameters"
  | "ranges"
  | "trials";

const TABS: Array<{ value: Tab; label: string; icon: ReactNode }> = [
  { value: "compare", label: "Compare", icon: <Grid3x3 size={14} /> },
  { value: "studies", label: "Studies", icon: <Layers3 size={14} /> },
  { value: "folds", label: "Folds", icon: <Rows3 size={14} /> },
  { value: "gradients", label: "Gradients", icon: <Waves size={14} /> },
  { value: "hyperparameters", label: "Hyperparameters", icon: <SlidersHorizontal size={14} /> },
  { value: "ranges", label: "Range finder", icon: <Ruler size={14} /> },
  { value: "trials", label: "Trials", icon: <Table2 size={14} /> },
];

interface PersistedFilters {
  filters?: FilterState;
  query?: string;
  minTrials?: string;
  tab?: Tab;
  groupKey?: string;
  rowFacet?: string;
  columnFacet?: string;
  statistic?: Statistic;
  scoreMode?: "absolute" | "delta";
  rangeSettings?: Partial<RangeSettings>;
}

function studyMatches(
  study: ArchiveStudy,
  filters: FilterState,
  skipKey?: string,
): boolean {
  for (const facet of FACETS) {
    if (facet.key === skipKey) continue;
    const selected = filters[facet.key];
    if (!selected || selected.length === 0) continue;
    if (!selected.includes(facet.value(study))) return false;
  }
  return true;
}

function textMatches(study: ArchiveStudy, needle: string): boolean {
  if (needle === "") return true;
  return [
    study.path,
    study.studyName,
    study.campaign ?? "",
    study.runName ?? "",
    study.experimentConfig ?? "",
    study.hparamConfigPath ?? "",
    study.datasetLabel,
    study.methodLabel,
    study.lossLabel,
    study.variant ?? "",
  ]
    .join(" ")
    .toLowerCase()
    .includes(needle);
}

export function ArchiveExplorer() {
  const [snapshot, setSnapshot] = useState<ArchiveSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [storageReady, setStorageReady] = useState(false);

  const [filters, setFilters] = useState<FilterState>({});
  const [query, setQuery] = useState("");
  const [minTrials, setMinTrials] = useState("");
  const [filtersOpen, setFiltersOpen] = useState(true);

  const [tab, setTab] = useState<Tab>("compare");
  const [groupKey, setGroupKey] = useState("method");
  const [rowFacet, setRowFacet] = useState("method");
  const [columnFacet, setColumnFacet] = useState("loss");
  const [statistic, setStatistic] = useState<Statistic>("best");
  const [scoreMode, setScoreMode] = useState<"absolute" | "delta">("absolute");

  const [rangeSettings, setRangeSettings] = useState<RangeSettings>(DEFAULT_RANGE_SETTINGS);
  const [selectedStudyId, setSelectedStudyId] = useState<string | null>(null);
  // One trial highlighted across every chart at once, so a point picked out of
  // the response panels can be found again in the fold and gradient views.
  const [selectedTrial, setSelectedTrial] = useState<TrialSelection | null>(null);
  const [parameterName, setParameterName] = useState<string | null>(null);
  const [copiedSpaces, setCopiedSpaces] = useState(false);
  const [gradientIndex, setGradientIndex] = useState<GradientIndex | null>(null);
  const [gradientError, setGradientError] = useState<string | null>(null);
  const [gradientSeries, setGradientSeries] = useState<GradientSeriesKey>("ratio");
  const [gradientStatistic, setGradientStatistic] = useState<GradientStatistic>("median");
  const [studySort, setStudySort] = useState<{ key: string; direction: "asc" | "desc" }>({
    key: "score",
    direction: "desc",
  });
  const [trialSort, setTrialSort] = useState<{ key: string; direction: "asc" | "desc" }>({
    key: "value",
    direction: "desc",
  });
  const [visibleTrials, setVisibleTrials] = useState(TRIAL_PAGE);

  // Restore the saved view before the first paint that could overwrite it.
  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(FILTER_STORAGE_KEY);
      const stored: PersistedFilters | null = raw ? JSON.parse(raw) : null;
      if (stored && typeof stored === "object") {
        if (stored.filters && typeof stored.filters === "object") {
          const restored: FilterState = {};
          for (const facet of FACETS) {
            const values = stored.filters[facet.key];
            if (Array.isArray(values)) {
              restored[facet.key] = values.filter((value) => typeof value === "string");
            }
          }
          setFilters(restored);
        }
        if (typeof stored.query === "string") setQuery(stored.query);
        if (typeof stored.minTrials === "string") setMinTrials(stored.minTrials);
        if (TABS.some((entry) => entry.value === stored.tab)) setTab(stored.tab as Tab);
        if (stored.groupKey && FACET_BY_KEY.has(stored.groupKey)) setGroupKey(stored.groupKey);
        if (stored.rowFacet && FACET_BY_KEY.has(stored.rowFacet)) setRowFacet(stored.rowFacet);
        if (stored.columnFacet && FACET_BY_KEY.has(stored.columnFacet)) {
          setColumnFacet(stored.columnFacet);
        }
        if (stored.statistic && stored.statistic in STATISTIC_LABELS) {
          setStatistic(stored.statistic);
        }
        if (stored.scoreMode === "delta" || stored.scoreMode === "absolute") {
          setScoreMode(stored.scoreMode);
        }
        if (stored.rangeSettings && typeof stored.rangeSettings === "object") {
          // Merge rather than replace, so a snapshot saved before a new
          // threshold existed still restores every other choice.
          setRangeSettings({ ...DEFAULT_RANGE_SETTINGS, ...stored.rangeSettings });
        }
      }
    } catch {
      // A corrupt entry just means the default view.
    }
    setStorageReady(true);
  }, []);

  useEffect(() => {
    if (!storageReady) return;
    try {
      const payload: PersistedFilters = {
        filters,
        query,
        minTrials,
        tab,
        groupKey,
        rowFacet,
        columnFacet,
        statistic,
        scoreMode,
        rangeSettings,
      };
      window.localStorage.setItem(FILTER_STORAGE_KEY, JSON.stringify(payload));
    } catch {
      // The view still holds for this session.
    }
  }, [
    columnFacet,
    filters,
    groupKey,
    minTrials,
    query,
    rangeSettings,
    rowFacet,
    scoreMode,
    statistic,
    storageReady,
    tab,
  ]);

  useEffect(() => {
    const controller = new AbortController();
    setError(null);
    fetch(`${SNAPSHOT_URL}?snapshot=${reloadKey}`, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Snapshot request failed (${response.status})`);
        const payload: unknown = await response.json();
        if (!isArchiveSnapshot(payload)) {
          throw new Error("The sorted-logs snapshot format is not supported.");
        }
        return payload;
      })
      .then(setSnapshot)
      .catch((loadError: unknown) => {
        if (controller.signal.aborted) return;
        setError(
          loadError instanceof Error
            ? loadError.message
            : "The sorted-logs snapshot could not be loaded.",
        );
      });
    return () => controller.abort();
  }, [reloadKey]);

  // 34 MB of diagnostics that only one view needs, so they are fetched the
  // first time that view is opened rather than on every page load.
  useEffect(() => {
    if (tab !== "gradients" || gradientIndex || gradientError) return;
    const controller = new AbortController();
    fetch(GRADIENT_URL, { signal: controller.signal, cache: "force-cache" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Gradient request failed (${response.status})`);
        const payload: unknown = await response.json();
        if (!isGradientSnapshot(payload)) {
          throw new Error("The gradient snapshot format is not supported.");
        }
        return payload;
      })
      .then((payload) => setGradientIndex(payload.studies))
      .catch((loadError: unknown) => {
        if (controller.signal.aborted) return;
        setGradientError(
          loadError instanceof Error
            ? loadError.message
            : "The gradient diagnostics could not be loaded.",
        );
      });
    return () => controller.abort();
  }, [gradientError, gradientIndex, tab]);

  useEffect(() => {
    if (!selectedStudyId) return;
    try {
      window.localStorage.setItem(STUDY_STORAGE_KEY, selectedStudyId);
    } catch {
      // Selection survives the session either way.
    }
  }, [selectedStudyId]);

  const allStats = useMemo(
    () => (snapshot ? snapshot.studies.map(summarize) : []),
    [snapshot],
  );

  const normalizedQuery = query.trim().toLowerCase();
  const trialFloor = minTrials.trim() === "" ? null : Number(minTrials);

  /** Everything the free-text box and the trial floor already exclude. */
  const searchable = useMemo(
    () =>
      allStats.filter(
        (stats) =>
          textMatches(stats.study, normalizedQuery) &&
          (trialFloor == null ||
            !Number.isFinite(trialFloor) ||
            stats.scored.length >= trialFloor),
      ),
    [allStats, normalizedQuery, trialFloor],
  );

  const filteredStats = useMemo(
    () => searchable.filter((stats) => studyMatches(stats.study, filters)),
    [filters, searchable],
  );

  // Counts inside a facet ignore that facet's own selection, so a chip always
  // says how many studies it would add rather than dropping to zero.
  const facetOptions = useMemo(() => {
    const result = new Map<string, FacetOption[]>();
    for (const facet of FACETS) {
      const pool = searchable.filter((stats) =>
        studyMatches(stats.study, filters, facet.key),
      );
      const counts = new Map<string, number>();
      for (const stats of allStats) {
        const value = facet.value(stats.study);
        if (!counts.has(value)) counts.set(value, 0);
      }
      for (const stats of pool) {
        const value = facet.value(stats.study);
        counts.set(value, (counts.get(value) ?? 0) + 1);
      }
      const selected = new Set(filters[facet.key] ?? []);
      result.set(
        facet.key,
        [...counts.entries()]
          .map(([value, count]) => ({
            value,
            label: facet.display ? facet.display(value) : value,
            count,
            selected: selected.has(value),
          }))
          .sort((left, right) => right.count - left.count || left.label.localeCompare(right.label)),
      );
    }
    return result;
  }, [allStats, filters, searchable]);

  const baselines = useMemo(() => buildBaselines(allStats, statistic), [allStats, statistic]);

  const scoreOf = useMemo(() => {
    return (stats: StudyStats): number | null => {
      const value = statisticOf(stats, statistic);
      if (value == null) return null;
      if (scoreMode === "absolute") return value;
      const baseline = baselineFor(baselines, stats.study);
      return baseline ? value - baseline.value : null;
    };
  }, [baselines, scoreMode, statistic]);

  const zeroBaseline = scoreMode === "delta";
  const scoreUnit = zeroBaseline ? "vs supervised" : metricLabel(
    filteredStats[0]?.study.metric ?? "best_valid_mean_average_precision_at_r",
  );

  const groups = useMemo<GroupSummary[]>(() => {
    const facet = FACET_BY_KEY.get(groupKey);
    if (!facet) return [];
    const byValue = new Map<string, StudyStats[]>();
    for (const stats of filteredStats) {
      const key = facet.value(stats.study);
      const bucket = byValue.get(key);
      if (bucket) bucket.push(stats);
      else byValue.set(key, [stats]);
    }
    return [...byValue.entries()]
      .map(([key, studies]) => {
        const scored = studies
          .map((stats) => ({ stats, score: scoreOf(stats) }))
          .filter((entry): entry is { stats: StudyStats; score: number } => entry.score != null);
        const scores = scored.map((entry) => entry.score);
        return {
          key,
          label: facet.display ? facet.display(key) : key,
          color: OTHER_COLOR,
          studies: scored.map((entry) => entry.stats),
          scores,
          best: scores.length > 0 ? Math.max(...scores) : null,
          median: scores.length > 0 ? median(scores) : null,
          trials: studies.reduce((total, stats) => total + stats.scored.length, 0),
        };
      })
      .sort((left, right) => (right.best ?? -Infinity) - (left.best ?? -Infinity))
      .map((group, index) => ({
        ...group,
        color: index < MAX_SERIES ? SERIES_COLORS[index] : OTHER_COLOR,
      }));
  }, [filteredStats, groupKey, scoreOf]);

  const matrix = useMemo(() => {
    const rows = FACET_BY_KEY.get(rowFacet);
    const columns = FACET_BY_KEY.get(columnFacet);
    if (!rows || !columns) {
      return { rowLabels: [], columnLabels: [], cells: new Map(), low: 0, high: 0 };
    }
    const cells = new Map<string, { value: number; studies: StudyStats[] }>();
    const rowLabels = new Set<string>();
    const columnLabels = new Set<string>();

    for (const stats of filteredStats) {
      const rowValue = rows.value(stats.study);
      const columnValue = columns.value(stats.study);
      const rowLabel = rows.display ? rows.display(rowValue) : rowValue;
      const columnLabel = columns.display ? columns.display(columnValue) : columnValue;
      rowLabels.add(rowLabel);
      columnLabels.add(columnLabel);
      const score = scoreOf(stats);
      if (score == null) continue;
      const key = cellKey(rowLabel, columnLabel);
      const cell = cells.get(key);
      if (!cell) {
        cells.set(key, { value: score, studies: [stats] });
      } else {
        cell.studies.push(stats);
        if (score > cell.value) cell.value = score;
      }
    }
    const values = [...cells.values()].map((cell) => cell.value);
    return {
      rowLabels: [...rowLabels].sort(),
      columnLabels: [...columnLabels].sort(),
      cells,
      low: values.length > 0 ? Math.min(...values) : 0,
      high: values.length > 0 ? Math.max(...values) : 0,
    };
  }, [columnFacet, filteredStats, rowFacet, scoreOf]);

  const parameterCatalog = useMemo(() => {
    const catalog = new Map<
      string,
      { name: string; studies: number; numeric: number; log: number }
    >();
    for (const stats of filteredStats) {
      for (const parameter of stats.study.parameters) {
        if (!parameter.searched) continue;
        const entry = catalog.get(parameter.name) ?? {
          name: parameter.name,
          studies: 0,
          numeric: 0,
          log: 0,
        };
        entry.studies += 1;
        if (parameter.kind === "numeric") entry.numeric += 1;
        if (parameter.log) entry.log += 1;
        catalog.set(parameter.name, entry);
      }
    }
    return [...catalog.values()].sort(
      (left, right) => right.studies - left.studies || left.name.localeCompare(right.name),
    );
  }, [filteredStats]);

  const activeParameter =
    parameterCatalog.find((entry) => entry.name === parameterName) ??
    parameterCatalog.find((entry) => entry.numeric >= entry.studies / 2) ??
    parameterCatalog[0] ??
    null;
  const parameterIsNumeric = activeParameter
    ? activeParameter.numeric >= activeParameter.studies / 2
    : false;
  const parameterIsLog = activeParameter
    ? activeParameter.log > activeParameter.studies / 2
    : false;

  /** The Phase 0 reading of every study currently in view. */
  const assessments = useMemo(
    () => filteredStats.map((stats) => assessStudy(stats.study, rangeSettings)),
    [filteredStats, rangeSettings],
  );

  const assessmentById = useMemo(
    () => new Map(assessments.map((assessment) => [assessment.study.id, assessment])),
    [assessments],
  );

  const diagnostics = useMemo<Diagnostics>(() => {
    const counts: Record<TrialHealth, number> = { alive: 0, dead: 0, stalled: 0, failed: 0 };
    let truncatedNumerator = 0;
    let truncatedDenominator = 0;
    const epsilons: number[] = [];
    const foldSigmas: number[] = [];
    const residualSigmas: number[] = [];
    const comparisonErrors: number[] = [];
    const effects: number[] = [];
    const resolutions: number[] = [];

    for (const assessment of assessments) {
      for (const health of Object.keys(counts) as TrialHealth[]) {
        counts[health] += assessment.counts[health];
      }
      epsilons.push(assessment.epsilon);
      if (assessment.noise.foldSigma != null) foldSigmas.push(assessment.noise.foldSigma);
      if (assessment.noise.residualSigma != null) {
        residualSigmas.push(assessment.noise.residualSigma);
      }
      if (assessment.noise.comparisonError != null) {
        comparisonErrors.push(assessment.noise.comparisonError);
      }
      if (assessment.effectSize != null) effects.push(assessment.effectSize);
      if (assessment.resolution != null) resolutions.push(assessment.resolution);
      if (assessment.truncatedShare != null) {
        truncatedNumerator += assessment.truncatedShare * assessment.keep.length;
        truncatedDenominator += assessment.keep.length;
      }
    }

    // Medians, not means: one pathological study should not move the headline.
    const mid = (values: number[]) => (values.length > 0 ? median(values) : null);
    return {
      epsilon: mid(epsilons),
      foldSigma: mid(foldSigmas),
      residualSigma: mid(residualSigmas),
      comparisonError: mid(comparisonErrors),
      effectSize: mid(effects),
      resolution: mid(resolutions),
      counts,
      truncatedShare:
        truncatedDenominator > 0 ? truncatedNumerator / truncatedDenominator : null,
      studies: assessments.length,
      epsilonSource:
        assessments.length === 0
          ? "no study in view"
          : assessments.length === 1
            ? assessments[0].epsilonSource
            : `${assessments[0].epsilonSource}, median over ${assessments.length} studies`,
    };
  }, [assessments]);

  const startup = useMemo<StartupSummary>(
    () => estimateStartupTrials(assessments),
    [assessments],
  );

  const epochBudget = useMemo<EpochBudget>(
    () => summariseEpochs(assessments),
    [assessments],
  );

  const foldRows = useMemo<FoldRow[]>(() => foldBreakdown(assessments), [assessments]);
  const foldCounts = useMemo(() => mixedFoldCounts(assessments), [assessments]);

  const gradientCatalog = useMemo(
    () => availableGradientSeries(assessments, gradientIndex ?? {}),
    [assessments, gradientIndex],
  );

  const activeGradientSeries: GradientSeriesKey = gradientCatalog.series.includes(gradientSeries)
    ? gradientSeries
    : (gradientCatalog.series[0] ?? "ratio");

  const gradientRows = useMemo(
    () =>
      gradientFoldRows(
        assessments,
        gradientIndex ?? {},
        activeGradientSeries,
        gradientStatistic,
      ),
    [activeGradientSeries, assessments, gradientIndex, gradientStatistic],
  );

  const gradientScatter = useMemo(() => {
    if (!gradientIndex) return null;
    const { folds } = collectGradientPoints(
      assessments,
      gradientIndex,
      activeGradientSeries,
      gradientStatistic,
    );
    if (folds.length === 0) return null;
    const facet = FACET_BY_KEY.get(groupKey);
    const byGroup = new Map<string, ScatterPoint[]>();
    for (const point of folds) {
      if (point.shortfall == null || !Number.isFinite(point.shortfall)) continue;
      const key = facet ? facet.value(point.study) : "All";
      const label = facet?.display ? facet.display(key) : key;
      const bucket = byGroup.get(label) ?? [];
      bucket.push({
        x: point.value,
        y: point.shortfall,
        inBand: point.inBand,
        label: `${shortPath(point.study.path)} · trial #${point.trialNumber} · fold ${point.fold}`,
        detail: `${GRADIENT_SERIES_LABELS[activeGradientSeries]} ${formatNumber(point.value)}`,
        studyId: point.study.id,
        trialNumber: point.trialNumber,
      });
      byGroup.set(label, bucket);
    }
    const logX = GRADIENT_LOG_SERIES.has(activeGradientSeries);
    const panels = [...byGroup.entries()]
      .sort((left, right) => right[1].length - left[1].length)
      .slice(0, 12)
      .map(([label, points], index) => ({
        label,
        points: logX ? points.filter((point) => point.x > 0) : points,
        color: index < MAX_SERIES ? SERIES_COLORS[index] : OTHER_COLOR,
      }));
    const xs = panels.flatMap((panel) =>
      panel.points.map((point) => (logX ? Math.log10(Math.max(point.x, 1e-12)) : point.x)),
    );
    const ys = panels.flatMap((panel) => panel.points.map((point) => point.y));
    if (xs.length === 0) return null;
    const sorted = ascending(ys);
    const interQuartile = quantile(sorted, 0.75) - quantile(sorted, 0.25);
    const fence =
      sorted.length >= 8 && interQuartile > 0
        ? quantile(sorted, 0.25) - 1.5 * interQuartile
        : -Infinity;
    const epsilon = diagnostics.epsilon ?? 0.01;
    const floor = Math.max(sorted[0], Math.max(fence, -6 * epsilon));
    const visible = ys.filter((value) => value >= floor);
    return {
      panels,
      logX,
      xDomain: extent(xs, 0.05),
      yDomain: extent(visible.length > 0 ? visible : ys, 0.05),
      epsilon,
      total: folds.length,
    };
  }, [
    activeGradientSeries,
    assessments,
    diagnostics.epsilon,
    gradientIndex,
    gradientStatistic,
    groupKey,
  ]);

  const findings = useMemo(
    () =>
      parameterCatalog.map((entry) => findRange(entry.name, assessments, rangeSettings)),
    [assessments, parameterCatalog, rangeSettings],
  );

  const activeFinding =
    findings.find((finding) => finding.name === activeParameter?.name) ?? null;

  const activeSamples = useMemo<ParameterSample[]>(
    () =>
      activeFinding && activeFinding.kind === "numeric"
        ? collectSamples(activeFinding.name, assessments, activeFinding.log)
        : [],
    [activeFinding, assessments],
  );

  const importanceRows = useMemo(() => {
    const totals = new Map<string, { total: number; studies: number }>();
    for (const stats of filteredStats) {
      for (const parameter of stats.study.parameters) {
        if (parameter.importance == null) continue;
        const entry = totals.get(parameter.name) ?? { total: 0, studies: 0 };
        entry.total += parameter.importance;
        entry.studies += 1;
        totals.set(parameter.name, entry);
      }
    }
    return [...totals.entries()]
      .map(([name, entry]) => ({
        name,
        importance: entry.total / entry.studies,
        studies: entry.studies,
      }))
      .sort((left, right) => right.importance - left.importance)
      .slice(0, 14);
  }, [filteredStats]);

  const scatter = useMemo(() => {
    if (!activeParameter || !parameterIsNumeric) return null;
    const name = activeParameter.name;
    const panels = groups.slice(0, 12).map((group) => {
      const points: ScatterPoint[] = [];
      for (const stats of group.studies) {
        const assessment = assessmentById.get(stats.study.id);
        if (!assessment) continue;
        for (const trial of assessment.trials) {
          const x = finite(trial.trial.params[name]);
          if (x == null || (parameterIsLog && x <= 0)) continue;
          // Shortfall, not the raw objective: panels then share a zero that
          // means "this study's best", and a study scoring 0.30 no longer
          // stretches the axis until a 3% gap elsewhere disappears.
          if (trial.health !== "alive" || trial.shortfall == null) continue;
          points.push({
            x,
            y: trial.shortfall,
            inBand: trial.inBand,
            label: `${shortPath(stats.study.path)} · trial #${trial.trial.number}`,
            detail: `${stats.study.datasetLabel} · ${stats.study.lossLabel} · ${formatMetric(
              trial.value,
            )} absolute`,
            studyId: stats.study.id,
            trialNumber: trial.trial.number,
          });
        }
      }
      return { group, points };
    });

    const total = panels.reduce((sum, panel) => sum + panel.points.length, 0);
    // Thinning keeps the panels responsive without biasing which trials show.
    const stride = total > MAX_SCATTER_POINTS ? Math.ceil(total / MAX_SCATTER_POINTS) : 1;
    const thinned = panels.map((panel) => ({
      ...panel,
      points: stride === 1 ? panel.points : panel.points.filter((_, index) => index % stride === 0),
    }));

    const xs = thinned.flatMap((panel) =>
      panel.points.map((point) => (parameterIsLog ? Math.log10(Math.max(point.x, 1e-12)) : point.x)),
    );
    const ys = thinned.flatMap((panel) => panel.points.map((point) => point.y));
    // The failure tail is cut off the shared axis rather than allowed to set
    // it; anything below the fence is pinned to the floor inside each panel.
    const sorted = ascending(ys);
    const interQuartile = quantile(sorted, 0.75) - quantile(sorted, 0.25);
    const fence =
      sorted.length >= 8 && interQuartile > 0
        ? quantile(sorted, 0.25) - 1.5 * interQuartile
        : -Infinity;
    const epsilon = diagnostics.epsilon ?? 0;
    const floor = Math.max(
      sorted.length > 0 ? sorted[0] : -1,
      Math.max(fence, -6 * (epsilon || 0.01)),
    );
    const visible = ys.filter((value) => value >= floor);
    return {
      panels: thinned,
      xDomain: extent(xs, 0.05),
      yDomain: extent(visible.length > 0 ? visible : ys, 0.05),
      clippedCount: ys.length - visible.length,
      epsilon,
      thinned: stride > 1,
      total,
    };
  }, [
    activeParameter,
    assessmentById,
    diagnostics.epsilon,
    groups,
    parameterIsLog,
    parameterIsNumeric,
  ]);

  /**
   * Where the good trials actually sit: the top quartile is taken per study, so
   * studies with different objective levels contribute on equal terms.
   */
  const parameterInsight = useMemo(() => {
    if (!activeParameter) return [];
    const name = activeParameter.name;
    return groups.map((group) => {
      const top: ArchiveParamValue[] = [];
      const all: ArchiveParamValue[] = [];
      for (const stats of group.studies) {
        const withParameter = stats.scored.filter(
          (trial) => trial.params[name] !== undefined && trial.params[name] !== null,
        );
        if (withParameter.length === 0) continue;
        const ordered = [...withParameter].sort(
          (left, right) => (right.value as number) - (left.value as number),
        );
        const cut = Math.max(1, Math.ceil(ordered.length * 0.25));
        ordered.slice(0, cut).forEach((trial) => top.push(trial.params[name]));
        withParameter.forEach((trial) => all.push(trial.params[name]));
      }
      return { group, top, all };
    });
  }, [activeParameter, groups]);

  const trialRows = useMemo(
    () =>
      filteredStats.flatMap((stats) =>
        stats.scored.map((trial) => ({
          stats,
          trial,
          delta: (() => {
            const baseline = baselineFor(baselines, stats.study);
            return baseline ? (trial.value as number) - baseline.value : null;
          })(),
        })),
      ),
    [baselines, filteredStats],
  );

  const trialParameterColumns = useMemo(() => {
    const counts = new Map<string, number>();
    for (const stats of filteredStats) {
      for (const parameter of stats.study.parameters) {
        if (!parameter.searched) continue;
        counts.set(parameter.name, (counts.get(parameter.name) ?? 0) + 1);
      }
    }
    return [...counts.entries()]
      .sort((left, right) => right[1] - left[1] || left[0].localeCompare(right[0]))
      .map(([name]) => name);
  }, [filteredStats]);

  const shownParameterColumns = trialParameterColumns.slice(0, 9);

  const sortedTrialRows = useMemo(() => {
    const direction = trialSort.direction === "asc" ? 1 : -1;
    const readValue = (row: (typeof trialRows)[number]): number | string | null => {
      switch (trialSort.key) {
        case "study":
          return row.stats.study.path;
        case "number":
          return row.trial.number;
        case "value":
          return row.trial.value;
        case "delta":
          return row.delta;
        case "p1":
          return finite(row.trial.metrics.best_valid_precision_at_1);
        case "fold":
          return foldValues(row.trial).length > 1
            ? standardDeviation(foldValues(row.trial))
            : null;
        case "epoch":
          return finite(row.trial.metrics.selected_epoch);
        case "seconds":
          return row.trial.seconds ?? null;
        default:
          if (trialSort.key.startsWith("param:")) {
            const value = row.trial.params[trialSort.key.slice(6)];
            return typeof value === "number" ? value : (value == null ? null : String(value));
          }
          return null;
      }
    };
    return [...trialRows].sort((left, right) => {
      const leftValue = readValue(left);
      const rightValue = readValue(right);
      if (leftValue == null && rightValue == null) return 0;
      if (leftValue == null) return 1;
      if (rightValue == null) return -1;
      if (typeof leftValue === "string" || typeof rightValue === "string") {
        return String(leftValue).localeCompare(String(rightValue)) * direction;
      }
      return (leftValue - rightValue) * direction;
    });
  }, [trialRows, trialSort]);

  const sortedStudyRows = useMemo(() => {
    const direction = studySort.direction === "asc" ? 1 : -1;
    const readValue = (stats: StudyStats): number | string | null => {
      switch (studySort.key) {
        case "path":
          return stats.study.path;
        case "dataset":
          return stats.study.datasetLabel;
        case "method":
          return stats.study.methodLabel;
        case "loss":
          return stats.study.lossLabel;
        case "phase":
          return stats.study.phase;
        case "trials":
          return stats.scored.length;
        case "spread":
          return stats.iqr;
        case "fold":
          return stats.foldSpread;
        case "score":
        default:
          return scoreOf(stats);
      }
    };
    return [...filteredStats].sort((left, right) => {
      const leftValue = readValue(left);
      const rightValue = readValue(right);
      if (leftValue == null && rightValue == null) return 0;
      if (leftValue == null) return 1;
      if (rightValue == null) return -1;
      if (typeof leftValue === "string" || typeof rightValue === "string") {
        return String(leftValue).localeCompare(String(rightValue)) * direction;
      }
      return (leftValue - rightValue) * direction;
    });
  }, [filteredStats, scoreOf, studySort]);

  useEffect(() => {
    setVisibleTrials(TRIAL_PAGE);
  }, [filters, normalizedQuery, trialFloor, trialSort]);

  const selectedStats =
    filteredStats.find((stats) => stats.study.id === selectedStudyId) ??
    allStats.find((stats) => stats.study.id === selectedStudyId) ??
    null;
  const selectedTrialStudy = selectedTrial
    ? (allStats.find((stats) => stats.study.id === selectedTrial.studyId)?.study ?? null)
    : null;
  const selectedTrialRecord =
    selectedTrialStudy && selectedTrial
      ? (selectedTrialStudy.trials.find(
          (trial) => trial.number === selectedTrial.trialNumber,
        ) ?? null)
      : null;

  const selectTrial = (selection: TrialSelection) => {
    setSelectedTrial((current) =>
      current?.studyId === selection.studyId && current.trialNumber === selection.trialNumber
        ? null
        : selection,
    );
    setSelectedStudyId(selection.studyId);
  };

  const toggleFacet = (facetKey: string, value: string) => {
    setFilters((current) => {
      const selected = current[facetKey] ?? [];
      const next = selected.includes(value)
        ? selected.filter((entry) => entry !== value)
        : [...selected, value];
      const updated = { ...current };
      if (next.length === 0) delete updated[facetKey];
      else updated[facetKey] = next;
      return updated;
    });
  };

  const clearFacet = (facetKey: string) => {
    setFilters((current) => {
      const updated = { ...current };
      delete updated[facetKey];
      return updated;
    });
  };

  const resetFilters = () => {
    setFilters({});
    setQuery("");
    setMinTrials("");
  };

  /** Clicking a heatmap cell narrows the selection to that crossing. */
  const selectMatrixCell = (rowLabel: string, columnLabel: string) => {
    const rows = FACET_BY_KEY.get(rowFacet);
    const columns = FACET_BY_KEY.get(columnFacet);
    if (!rows || !columns) return;
    const rowValue = [...new Set(filteredStats.map((stats) => rows.value(stats.study)))].find(
      (value) => (rows.display ? rows.display(value) : value) === rowLabel,
    );
    const columnValue = [
      ...new Set(filteredStats.map((stats) => columns.value(stats.study))),
    ].find((value) => (columns.display ? columns.display(value) : value) === columnLabel);
    if (rowValue === undefined || columnValue === undefined) return;
    setFilters((current) => ({
      ...current,
      [rowFacet]: [rowValue],
      [columnFacet]: [columnValue],
    }));
    setTab("studies");
  };

  const exportTrials = () => {
    const parameterNames = trialParameterColumns;
    const header = [
      "study_path",
      "phase",
      "mode",
      "dataset",
      "method",
      "variant",
      "loss",
      "label_budget",
      "trial",
      "objective_metric",
      "objective",
      "delta_vs_supervised",
      "best_valid_precision_at_1",
      "fold_std",
      "selected_epoch",
      "last_epoch",
      "duration_seconds",
      ...parameterNames.map((name) => `param:${name}`),
    ];
    const lines = [header.join(",")];
    const escape = (value: unknown) => {
      const text = value === null || value === undefined ? "" : String(value);
      return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    };
    for (const row of sortedTrialRows) {
      lines.push(
        [
          row.stats.study.path,
          row.stats.study.phase,
          row.stats.study.mode,
          row.stats.study.datasetLabel,
          row.stats.study.methodLabel,
          row.stats.study.variant ?? "",
          row.stats.study.lossLabel,
          row.stats.study.labelBudget,
          row.trial.number,
          row.stats.study.metric,
          row.trial.value ?? "",
          row.delta ?? "",
          row.trial.metrics.best_valid_precision_at_1 ?? "",
          foldValues(row.trial).length > 1
            ? standardDeviation(foldValues(row.trial))
            : "",
          row.trial.metrics.selected_epoch ?? "",
          row.trial.metrics.last_epoch ?? "",
          row.trial.seconds ?? "",
          ...parameterNames.map((name) => row.trial.params[name] ?? ""),
        ]
          .map(escape)
          .join(","),
      );
    }
    downloadText(lines.join("\n"), `${slugify("sorted-logs-trials")}.csv`);
  };

  const activeFilterCount = Object.values(filters).reduce(
    (total, values) => total + values.length,
    0,
  );

  if (!snapshot && !error) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="text-center">
          <span className="mx-auto block h-6 w-6 animate-spin rounded-full border-2 border-[#9b8cff]/25 border-t-[#9b8cff]" />
          <p className="mt-3 text-xs text-[#7f8ca2]">Loading the sorted HPO archive…</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="surface w-full max-w-xl px-8 py-9 text-center">
          <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-rose-400/10 text-rose-300 ring-1 ring-rose-300/15">
            <Activity size={22} />
          </span>
          <h1 className="mt-5 text-xl font-semibold text-white">Archive snapshot unavailable</h1>
          <p className="mt-2 text-sm leading-6 text-[#8794aa]">{error}</p>
          <p className="mt-2 text-xs leading-5 text-[#68768e]">
            Run <code className="font-mono text-[#b6acff]">npm run sync-sorted</code> to rebuild{" "}
            <code className="font-mono">public/sorted-studies.json</code> from{" "}
            <code className="font-mono">sorted_logs/</code>.
          </p>
          <button
            type="button"
            onClick={() => setReloadKey((key) => key + 1)}
            className="focus-ring mt-6 inline-flex h-10 items-center gap-2 rounded-xl bg-[#9b8cff] px-4 text-xs font-semibold text-white"
          >
            <RefreshCw size={14} /> Try again
          </button>
        </div>
      </div>
    );
  }

  if (!snapshot || snapshot.studies.length === 0) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <div className="surface w-full max-w-xl px-8 py-9 text-center">
          <Database size={28} className="mx-auto text-[#9b8cff]" aria-hidden="true" />
          <h1 className="mt-4 text-xl font-semibold text-white">The archive is empty</h1>
          <p className="mt-2 text-sm leading-6 text-[#8794aa]">
            {snapshot?.warning ??
              "No study directory under sorted_logs/ carried an optuna_study.db."}
          </p>
        </div>
      </div>
    );
  }

  const snapshotDate = new Date(snapshot.generatedAt);
  const snapshotLabel = Number.isNaN(snapshotDate.getTime())
    ? "Snapshot ready"
    : `Synced ${snapshotDate.toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" })}`;
  const selectionTrials = filteredStats.reduce((total, stats) => total + stats.scored.length, 0);
  const bestInSelection = filteredStats.reduce<StudyStats | null>((champion, stats) => {
    const value = statisticOf(stats, "best");
    if (value == null) return champion;
    const currentBest = champion ? statisticOf(champion, "best") : null;
    return currentBest == null || value > currentBest ? stats : champion;
  }, null);
  const datasetsInSelection = new Set(filteredStats.map((stats) => stats.study.datasetLabel));
  const metricsInSelection = new Set(filteredStats.map((stats) => stats.study.metric));

  return (
    <div className="fade-up space-y-5">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.17em] text-[#78869e]">
            <Sparkles size={12} className="text-[#b8e56e]" /> Sorted HPO archive
          </div>
          <h1 className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-white sm:text-4xl">
            Every sweep, side by side.
          </h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-[#8794aa]">
            {snapshot.studies.length} studies from{" "}
            <code className="font-mono text-[11px] text-[#a9b4c8]">{snapshot.source}</code>, faceted
            by phase, dataset, method, loss and campaign — with their hyperparameters and every
            trial they scored.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 self-start xl:self-auto">
          <span className="rounded-full border border-white/[0.08] bg-white/[0.035] px-3 py-1.5 text-[10px] text-[#8290a7]">
            {snapshotLabel}
          </span>
          <button
            type="button"
            onClick={() => setReloadKey((key) => key + 1)}
            className="focus-ring inline-flex h-9 items-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.04] px-3.5 text-xs font-semibold text-[#c3ccdc] transition hover:bg-white/[0.08] hover:text-white"
          >
            <RefreshCw size={14} /> Reload
          </button>
        </div>
      </div>

      <BudgetHeader
        startup={startup}
        epochs={epochBudget}
        epsilon={diagnostics.epsilon}
        onOpenRanges={() => setTab("ranges")}
      />

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Studies in view"
          value={filteredStats.length.toLocaleString()}
          context={`of ${snapshot.studies.length} in the archive`}
          icon={<Layers3 size={17} />}
          accent="violet"
        />
        <MetricCard
          label="Scored trials"
          value={selectionTrials.toLocaleString()}
          context={`${datasetsInSelection.size} dataset${datasetsInSelection.size === 1 ? "" : "s"} · ${groups.length} group${groups.length === 1 ? "" : "s"}`}
          icon={<Database size={17} />}
          accent="cyan"
        />
        <MetricCard
          label="Best in view"
          value={formatMetric(bestInSelection ? statisticOf(bestInSelection, "best") : null)}
          context={
            bestInSelection
              ? `${bestInSelection.study.methodLabel} · ${bestInSelection.study.lossLabel} · ${bestInSelection.study.datasetLabel}`
              : "No scored trial in this selection"
          }
          icon={<Trophy size={17} />}
          accent="lime"
        />
        <MetricCard
          label="Active filters"
          value={String(activeFilterCount)}
          context={
            metricsInSelection.size > 1
              ? `${metricsInSelection.size} different objectives in view`
              : metricLabel([...metricsInSelection][0] ?? "objective")
          }
          icon={<Filter size={17} />}
          accent="amber"
        />
      </div>

      {metricsInSelection.size > 1 && (
        <p className="surface-soft px-3 py-2 text-[11px] leading-5 text-[#f4c86f]">
          This selection mixes {metricsInSelection.size} different objective metrics. Scores are not
          comparable across them — narrow the filters, or switch the score to &ldquo;Δ vs
          supervised&rdquo;.
        </p>
      )}

      <DiagnosticsStrip diagnostics={diagnostics} onOpen={() => setTab("ranges")} />

      <div className="grid gap-4 xl:grid-cols-[298px_minmax(0,1fr)]">
        <aside className="surface h-fit overflow-hidden xl:sticky xl:top-20">
          <header className="flex items-center justify-between gap-2 border-b border-white/[0.07] px-4 py-3">
            <div className="flex items-center gap-2">
              <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-[#9b8cff]/10 text-[#b6acff]">
                <Filter size={14} />
              </span>
              <h2 className="text-sm font-semibold text-white">Filters</h2>
            </div>
            <div className="flex items-center gap-1">
              {activeFilterCount + (query ? 1 : 0) + (minTrials ? 1 : 0) > 0 && (
                <button
                  type="button"
                  onClick={resetFilters}
                  className="focus-ring rounded-lg px-2 py-1 text-[10px] font-semibold text-[#9b8cff] hover:bg-white/[0.06]"
                >
                  Reset
                </button>
              )}
              <button
                type="button"
                onClick={() => setFiltersOpen((current) => !current)}
                aria-expanded={filtersOpen}
                className="focus-ring flex h-7 w-7 items-center justify-center rounded-lg text-[#8290a7] hover:bg-white/[0.07] hover:text-white xl:hidden"
              >
                <ChevronDown size={15} className={filtersOpen ? "" : "-rotate-90"} />
              </button>
            </div>
          </header>
          <div className={filtersOpen ? "block" : "hidden xl:block"}>
            <div className="space-y-2 border-b border-white/[0.06] px-4 py-3">
              <label className="relative block">
                <Search
                  size={13}
                  className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#5f6e87]"
                />
                <span className="sr-only">Search studies</span>
                <input
                  type="search"
                  value={query}
                  onChange={(event) => setQuery(event.target.value)}
                  placeholder="Path, campaign, config…"
                  className={`${inputClass} pl-8`}
                />
              </label>
              <label className="flex items-center gap-2">
                <span className="whitespace-nowrap text-[10px] text-[#79869e]">Min. trials</span>
                <input
                  type="number"
                  inputMode="numeric"
                  min={0}
                  value={minTrials}
                  onChange={(event) => setMinTrials(event.target.value)}
                  placeholder="0"
                  className={inputClass}
                />
              </label>
            </div>
            {FACETS.map((facet) => (
              <FacetGroup
                key={facet.key}
                facet={facet}
                options={facetOptions.get(facet.key) ?? []}
                onToggle={(value) => toggleFacet(facet.key, value)}
                onClear={() => clearFacet(facet.key)}
              />
            ))}
          </div>
        </aside>

        <div className="min-w-0 space-y-4">
          <div className="surface flex flex-wrap items-center gap-x-4 gap-y-3 px-4 py-3">
            <div className="flex flex-wrap gap-1">
              {TABS.map((entry) => (
                <button
                  key={entry.value}
                  type="button"
                  aria-pressed={tab === entry.value}
                  onClick={() => setTab(entry.value)}
                  className={`focus-ring inline-flex items-center gap-1.5 rounded-xl px-3 py-1.5 text-xs font-semibold transition ${
                    tab === entry.value
                      ? "bg-[#9b8cff]/[0.14] text-white ring-1 ring-inset ring-[#9b8cff]/25"
                      : "text-[#8e9bb1] hover:bg-white/[0.05] hover:text-white"
                  }`}
                >
                  {entry.icon}
                  {entry.label}
                </button>
              ))}
            </div>
            <div className="ml-auto flex flex-wrap items-center gap-x-4 gap-y-2">
              <LabelledSelect
                label="Score"
                value={statistic}
                options={(Object.keys(STATISTIC_LABELS) as Statistic[]).map((key) => ({
                  value: key,
                  label: STATISTIC_LABELS[key],
                }))}
                onChange={(value) => setStatistic(value as Statistic)}
              />
              <Segmented
                label="As"
                value={scoreMode}
                options={[
                  { value: "absolute" as const, label: "Absolute" },
                  { value: "delta" as const, label: "Δ vs supervised" },
                ]}
                onChange={setScoreMode}
              />
            </div>
          </div>

          {scoreMode === "delta" && (
            <p className="surface-soft px-3 py-2 text-[11px] leading-5 text-[#8f9bb2]">
              Each study is measured against the best supervised sweep with the same dataset, loss,
              label budget and phase; where that exact pairing is missing the nearest supervised
              sweep on the same dataset and loss is used. Studies with no supervised counterpart
              drop out of the comparison.
            </p>
          )}

          {tab === "compare" && (
            <>
              <section className="surface space-y-4 p-4 sm:p-5">
                <SectionHeading
                  title={`${STATISTIC_LABELS[statistic]} by ${FACET_BY_KEY.get(groupKey)?.label.toLowerCase() ?? "group"}`}
                  description="Ranked across the studies currently in view."
                  action={
                    <LabelledSelect
                      label="Group by"
                      value={groupKey}
                      options={GROUP_KEYS.map((key) => ({
                        value: key,
                        label: FACET_BY_KEY.get(key)?.label ?? key,
                      }))}
                      onChange={setGroupKey}
                    />
                  }
                />
                <GroupRankChart groups={groups} unit={scoreUnit} zeroBaseline={zeroBaseline} />
              </section>

              <section className="surface space-y-4 p-4 sm:p-5">
                <SectionHeading
                  title="How consistent is each group?"
                  description="A group that wins on one loss but collapses on another has a wide box — the leaderboard alone would hide that."
                />
                <GroupBoxPlot groups={groups} unit={scoreUnit} zeroBaseline={zeroBaseline} />
              </section>

              <section className="surface space-y-4 p-4 sm:p-5">
                <SectionHeading
                  title="Crossing two facets"
                  description="Each cell is the best study at that crossing. Click one to filter the selection down to it."
                  action={
                    <div className="flex flex-wrap items-center gap-3">
                      <LabelledSelect
                        label="Rows"
                        value={rowFacet}
                        options={GROUP_KEYS.map((key) => ({
                          value: key,
                          label: FACET_BY_KEY.get(key)?.label ?? key,
                        }))}
                        onChange={setRowFacet}
                      />
                      <LabelledSelect
                        label="Columns"
                        value={columnFacet}
                        options={GROUP_KEYS.map((key) => ({
                          value: key,
                          label: FACET_BY_KEY.get(key)?.label ?? key,
                        }))}
                        onChange={setColumnFacet}
                      />
                    </div>
                  }
                />
                <MatrixHeatmap
                  rowLabels={matrix.rowLabels}
                  columnLabels={matrix.columnLabels}
                  cells={matrix.cells}
                  unit={scoreUnit}
                  zeroBaseline={zeroBaseline}
                  onSelect={selectMatrixCell}
                />
                {matrix.cells.size > 0 && (
                  <RampLegend low={matrix.low} high={matrix.high} zeroBaseline={zeroBaseline} />
                )}
              </section>
            </>
          )}

          {tab === "studies" && (
            <section className="surface overflow-hidden">
              <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.07] px-4 py-3">
                <div>
                  <h3 className="text-sm font-semibold text-white">
                    {filteredStats.length} stud{filteredStats.length === 1 ? "y" : "ies"}
                  </h3>
                  <p className="mt-0.5 text-[10px] text-[#718098]">
                    Select a row to open its search space, provenance and top trials.
                  </p>
                </div>
              </header>
              <div className="max-h-[62vh] overflow-auto">
                <table className="w-full min-w-[900px] text-[11px]">
                  <thead className="sticky top-0 z-10 bg-[#111a2c]">
                    <tr className="border-b border-white/[0.07]">
                      <SortHeader label="Study" columnKey="path" sort={studySort} onSort={(key) => setStudySort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }))} />
                      <SortHeader label="Phase" columnKey="phase" sort={studySort} onSort={(key) => setStudySort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }))} />
                      <SortHeader label="Dataset" columnKey="dataset" sort={studySort} onSort={(key) => setStudySort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }))} />
                      <SortHeader label="Method" columnKey="method" sort={studySort} onSort={(key) => setStudySort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }))} />
                      <SortHeader label="Loss" columnKey="loss" sort={studySort} onSort={(key) => setStudySort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }))} />
                      <SortHeader label="Trials" columnKey="trials" align="right" sort={studySort} onSort={(key) => setStudySort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }))} />
                      <SortHeader label={STATISTIC_LABELS[statistic]} columnKey="score" align="right" sort={studySort} onSort={(key) => setStudySort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }))} />
                      <SortHeader label="IQR" columnKey="spread" align="right" title="Interquartile range of the study's trial objectives" sort={studySort} onSort={(key) => setStudySort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }))} />
                      <SortHeader label="Fold ±" columnKey="fold" align="right" title="Mean cross-validation spread across the study's trials" sort={studySort} onSort={(key) => setStudySort((current) => ({ key, direction: current.key === key && current.direction === "desc" ? "asc" : "desc" }))} />
                    </tr>
                  </thead>
                  <tbody>
                    {sortedStudyRows.length === 0 ? (
                      <tr>
                        <td colSpan={9} className="px-4 py-10 text-center text-xs text-[#6f7d95]">
                          No study matches these filters.
                        </td>
                      </tr>
                    ) : (
                      sortedStudyRows.map((stats) => {
                        const score = scoreOf(stats);
                        const selected = stats.study.id === selectedStudyId;
                        return (
                          <tr
                            key={stats.study.id}
                            onClick={() =>
                              setSelectedStudyId((current) =>
                                current === stats.study.id ? null : stats.study.id,
                              )
                            }
                            className={`cursor-pointer border-b border-white/[0.04] transition ${
                              selected ? "bg-[#9b8cff]/[0.11]" : "hover:bg-white/[0.035]"
                            }`}
                          >
                            <td className="max-w-[22rem] px-2.5 py-2" title={stats.study.path}>
                              <span className="block truncate font-mono text-[10px] text-[#c9d2e1]">
                                {stats.study.path}
                              </span>
                              <span className="mt-0.5 block truncate text-[10px] text-[#68768e]">
                                {stats.study.labelBudget}
                                {stats.study.variant ? ` · ${stats.study.variant}` : ""}
                                {stats.study.bucket === "unfinished" ? " · unfinished" : ""}
                              </span>
                            </td>
                            <td className="whitespace-nowrap px-2.5 py-2 text-[#8f9bb2]">
                              {stats.study.phase === "phase_0" ? "0 · random" : "1 · TPE"}
                            </td>
                            <td className="whitespace-nowrap px-2.5 py-2 text-[#c9d2e1]">
                              {stats.study.datasetLabel}
                            </td>
                            <td className="whitespace-nowrap px-2.5 py-2 text-[#c9d2e1]">
                              {stats.study.methodLabel}
                            </td>
                            <td className="whitespace-nowrap px-2.5 py-2 text-[#c9d2e1]">
                              {stats.study.lossLabel}
                            </td>
                            <td className="px-2.5 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {stats.scored.length}
                            </td>
                            <td className="px-2.5 py-2 text-right tabular-nums text-white">
                              {score == null
                                ? "—"
                                : zeroBaseline
                                  ? signedMetric(score)
                                  : formatMetric(score)}
                            </td>
                            <td className="px-2.5 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {stats.iqr != null ? stats.iqr.toFixed(4) : "—"}
                            </td>
                            <td className="px-2.5 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {stats.foldSpread != null ? stats.foldSpread.toFixed(4) : "—"}
                            </td>
                          </tr>
                        );
                      })
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          {tab === "folds" && (
            <>
              <section className="surface space-y-4 p-4 sm:p-5">
                <SectionHeading
                  title="What each fold contributes"
                  description="Validation scores only — nothing here is tuned against the test split, and the exporter carries no test metric at all."
                />
                {foldCounts.length > 1 && (
                  <p className="surface-soft px-3 py-2 text-[11px] leading-5 text-[#e8b957]">
                    This selection mixes studies with {foldCounts.join(" and ")} folds, so the
                    higher fold indices come from a smaller and different set of studies than fold
                    0. Their offsets are not comparable with the rest and will not sum to zero —
                    filter to one fold count before reading them against each other.
                  </p>
                )}
                <FoldEffectChart rows={foldRows} />
              </section>

              <section className="surface overflow-hidden">
                <header className="border-b border-white/[0.07] px-4 py-3">
                  <h3 className="text-sm font-semibold text-white">Aggregate and per fold</h3>
                  <p className="mt-1 text-[11px] leading-5 text-[#7e8ba3]">
                    Across {assessments.length} stud{assessments.length === 1 ? "y" : "ies"} and{" "}
                    {foldRows.reduce((total, row) => total + row.trials, 0).toLocaleString()} fold
                    runs.
                  </p>
                </header>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[760px] text-[11px]">
                    <thead>
                      <tr className="border-b border-white/[0.07]">
                        {[
                          { label: "Fold", align: "left" as const },
                          { label: "Runs", align: "right" as const },
                          { label: "Studies", align: "right" as const },
                          { label: "Median mAP@R", align: "right" as const },
                          { label: "Median P@1", align: "right" as const },
                          { label: "Offset vs trial", align: "right" as const },
                          { label: "Best epoch", align: "right" as const },
                          { label: "Last epoch", align: "right" as const },
                          { label: "Out of epochs", align: "right" as const },
                        ].map((column) => (
                          <th
                            key={column.label}
                            className={`px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e] ${
                              column.align === "right" ? "text-right" : "text-left"
                            }`}
                          >
                            {column.label}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(() => {
                        const totals = foldRows.reduce(
                          (accumulator, row) => ({
                            trials: accumulator.trials + row.trials,
                            atBudget: accumulator.atBudget + row.atBudget,
                          }),
                          { trials: 0, atBudget: 0 },
                        );
                        return (
                          <tr className="border-b border-white/[0.1] bg-white/[0.03] font-semibold">
                            <td className="px-3 py-2 text-[#c9d2e1]">All folds</td>
                            <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {totals.trials.toLocaleString()}
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {assessments.length}
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-white">
                              {formatMetric(
                                foldRows.length > 0
                                  ? median(
                                      foldRows
                                        .map((row) => row.medianValue)
                                        .filter((value): value is number => value != null),
                                    )
                                  : null,
                              )}
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {formatMetric(
                                foldRows.length > 0
                                  ? median(
                                      foldRows
                                        .map((row) => row.medianPrecisionAt1)
                                        .filter((value): value is number => value != null),
                                    )
                                  : null,
                              )}
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-[#5c6a82]">
                              0 by construction
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {epochBudget.medianSelected ?? "—"}
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {epochBudget.medianLast ?? "—"}
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {totals.atBudget.toLocaleString()}
                            </td>
                          </tr>
                        );
                      })()}
                      {foldRows.map((row) => (
                        <tr key={row.fold} className="border-b border-white/[0.04]">
                          <td className="px-3 py-2 text-[#c9d2e1]">Fold {row.fold}</td>
                          <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                            {row.trials.toLocaleString()}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                            {row.studies}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-white">
                            {formatMetric(row.medianValue)}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                            {formatMetric(row.medianPrecisionAt1)}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums">
                            <span
                              className={
                                row.effect != null && Math.abs(row.effect) > 0.01
                                  ? "text-[#e8b957]"
                                  : "text-[#8f9bb2]"
                              }
                            >
                              {row.effect != null
                                ? `${row.effect >= 0 ? "+" : ""}${row.effect.toFixed(4)}`
                                : "—"}
                            </span>
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                            {row.medianSelectedEpoch ?? "—"}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                            {row.medianLastEpoch ?? "—"}
                          </td>
                          <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                            {row.atBudget.toLocaleString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>

              <section className="surface space-y-4 p-4 sm:p-5">
                <SectionHeading
                  title="Trial by fold"
                  description={
                    selectedStats
                      ? "Each cell against its own trial's mean. A vertical stripe is a fold that is simply harder; scattered colour is the trial-by-fold interaction the noise floor is built from."
                      : "Select a study in the Studies tab to see its trials fold by fold."
                  }
                />
                {selectedStats && assessmentById.get(selectedStats.study.id) ? (
                  <FoldMatrix
                    assessment={
                      assessmentById.get(selectedStats.study.id) as StudyAssessment
                    }
                    selectedTrial={selectedTrial}
                    onSelectTrial={(trialNumber) =>
                      selectTrial({ studyId: selectedStats.study.id, trialNumber })
                    }
                  />
                ) : (
                  <EmptyChart message="No study selected. Pick one in the Studies tab, or click a study anywhere else on this page." />
                )}
              </section>
            </>
          )}

          {tab === "gradients" && (
            <>
              {gradientError ? (
                <p className="surface-soft px-3 py-2 text-[11px] leading-5 text-[#e8b957]">
                  {gradientError} Run <code className="font-mono">npm run sync-sorted</code> to
                  rebuild <code className="font-mono">public/sorted-gradients.json</code>.
                </p>
              ) : !gradientIndex ? (
                <div className="surface flex min-h-[220px] items-center justify-center">
                  <div className="text-center">
                    <span className="mx-auto block h-6 w-6 animate-spin rounded-full border-2 border-[#9b8cff]/25 border-t-[#9b8cff]" />
                    <p className="mt-3 text-xs text-[#7f8ca2]">
                      Loading gradient diagnostics…
                    </p>
                    <p className="mt-1 text-[10px] text-[#68768e]">
                      34 MB, fetched once and only for this view.
                    </p>
                  </div>
                </div>
              ) : gradientCatalog.trials === 0 ? (
                <EmptyChart message="No study in this selection logged gradient diagnostics. They are written per fold by SSL runs; supervised sweeps have none." />
              ) : (
                <>
                  <div className="surface flex flex-wrap items-center gap-x-4 gap-y-3 px-4 py-3">
                    <LabelledSelect
                      label="Series"
                      value={activeGradientSeries}
                      options={gradientCatalog.series.map((key) => ({
                        value: key,
                        label: GRADIENT_SERIES_LABELS[key],
                      }))}
                      onChange={(value) => setGradientSeries(value as GradientSeriesKey)}
                    />
                    <Segmented
                      label="Read"
                      value={gradientStatistic}
                      options={[
                        { value: "median" as const, label: "Whole run" },
                        { value: "late" as const, label: "Steady state" },
                      ]}
                      onChange={setGradientStatistic}
                    />
                    <span className="ml-auto text-[10px] text-[#68768e]">
                      {gradientCatalog.trials.toLocaleString()} trials with diagnostics
                      {gradientCatalog.hasTarget ? " · controller targets present" : ""}
                    </span>
                  </div>

                  <section className="surface overflow-hidden">
                    <header className="border-b border-white/[0.07] px-4 py-3">
                      <h3 className="text-sm font-semibold text-white">
                        {GRADIENT_SERIES_LABELS[activeGradientSeries]}, aggregate and per fold
                      </h3>
                      <p className="mt-1 text-[11px] leading-5 text-[#7e8ba3]">
                        &ldquo;Within-fold spread&rdquo; is how far the series wandered during a
                        single fold, in decades; the p10–p90 column is how much folds differ from
                        each other. A cosine below zero means the regularizer gradient points
                        against the supervised one.
                      </p>
                    </header>
                    <GradientFoldTable
                      rows={gradientRows.rows}
                      aggregate={gradientRows.aggregate}
                      seriesLabel={GRADIENT_SERIES_LABELS[activeGradientSeries]}
                      hasTarget={gradientCatalog.hasTarget}
                    />
                  </section>

                  {gradientCatalog.hasTarget && (
                    <section className="surface space-y-4 p-4 sm:p-5">
                      <SectionHeading
                        title="Did the controller hold its target?"
                        description="Achieved ratio against the ratio the controller was asked to hold, joined step by step and reported per fold."
                      />
                      <TargetDeviationChart rows={gradientRows.rows} />
                    </section>
                  )}

                  <section className="surface space-y-4 p-4 sm:p-5">
                    <SectionHeading
                      title="Does it move the objective?"
                      description="One point per fold, against that trial's shortfall from its own study's best. Click a point for its trial details; a vertical smear means the gradient balance is not what separates the good trials from the rest."
                      action={
                        <LabelledSelect
                          label="Panels"
                          value={groupKey}
                          options={GROUP_KEYS.map((key) => ({
                            value: key,
                            label: FACET_BY_KEY.get(key)?.label ?? key,
                          }))}
                          onChange={setGroupKey}
                        />
                      }
                    />
                    {gradientScatter ? (
                      <>
                        <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                          {gradientScatter.panels.map((panel) => (
                            <ScatterPanel
                              key={panel.label}
                              title={panel.label}
                              points={panel.points}
                              xDomain={gradientScatter.xDomain}
                              yDomain={gradientScatter.yDomain}
                              logX={gradientScatter.logX}
                              xLabel={GRADIENT_SERIES_LABELS[activeGradientSeries]}
                              yLabel="shortfall"
                              color={panel.color}
                              epsilon={gradientScatter.epsilon}
                              selectedTrial={selectedTrial}
                              onSelectTrial={selectTrial}
                            />
                          ))}
                        </div>
                        <p className="text-[10px] leading-4 text-[#68768e]">
                          x: {GRADIENT_SERIES_LABELS[activeGradientSeries]}
                          {gradientScatter.logX ? " (log)" : ""} · y: each trial minus its own
                          study&rsquo;s best, with the ε ={" "}
                          {gradientScatter.epsilon.toFixed(4)} band shaded.{" "}
                          {gradientScatter.total.toLocaleString()} fold points.
                        </p>
                      </>
                    ) : (
                      <EmptyChart message="No fold in this selection has both a diagnostic value and a scored trial." />
                    )}
                  </section>
                </>
              )}
            </>
          )}

          {tab === "hyperparameters" && (
            <>
              <section className="surface space-y-4 p-4 sm:p-5">
                <SectionHeading
                  title="Which parameter mattered"
                  description={`Mean stored fANOVA importance across the ${filteredStats.filter((stats) => stats.study.parameters.some((parameter) => parameter.importance != null)).length} studies in view that carry a ranking.`}
                />
                <ImportanceBars rows={importanceRows} />
              </section>

              <section className="surface space-y-4 p-4 sm:p-5">
                <SectionHeading
                  title="Response to one parameter"
                  description="One panel per group, sharing both axes — click a point for its trial details."
                  action={
                    <div className="flex flex-wrap items-center gap-3">
                      <LabelledSelect
                        label="Parameter"
                        value={activeParameter?.name ?? ""}
                        options={parameterCatalog.map((entry) => ({
                          value: entry.name,
                          label: `${parameterLabel(entry.name)} (${entry.studies})`,
                        }))}
                        onChange={setParameterName}
                      />
                      <LabelledSelect
                        label="Panels"
                        value={groupKey}
                        options={GROUP_KEYS.map((key) => ({
                          value: key,
                          label: FACET_BY_KEY.get(key)?.label ?? key,
                        }))}
                        onChange={setGroupKey}
                      />
                    </div>
                  }
                />
                {!activeParameter ? (
                  <EmptyChart message="No study in this selection searched a hyperparameter." />
                ) : parameterIsNumeric && scatter ? (
                  <>
                    <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
                      {scatter.panels.map((panel) => (
                        <ScatterPanel
                          key={panel.group.key}
                          title={panel.group.label}
                          points={panel.points}
                          xDomain={scatter.xDomain}
                          yDomain={scatter.yDomain}
                          logX={parameterIsLog}
                          xLabel={parameterLabel(activeParameter.name)}
                          yLabel="shortfall"
                          color={panel.group.color}
                          epsilon={scatter.epsilon}
                          selectedTrial={selectedTrial}
                          onSelectTrial={selectTrial}
                        />
                      ))}
                    </div>
                    <p className="text-[10px] leading-4 text-[#68768e]">
                      x: {parameterLabel(activeParameter.name)}
                      {parameterIsLog ? " (log)" : ""} · y: each trial minus its own study&rsquo;s
                      best, so 0 is that study&rsquo;s optimum and the shaded strip is the ε ={" "}
                      {scatter.epsilon.toFixed(4)} tolerance band. Panels share both axes, which
                      span {(scatter.yDomain[1] - scatter.yDomain[0]).toFixed(4)}.{" "}
                      {scatter.total.toLocaleString()} trials
                      {scatter.thinned ? ", thinned for drawing" : ""}
                      {scatter.clippedCount > 0
                        ? `, ${scatter.clippedCount} in the failure tail pinned to the floor`
                        : ""}
                      {groups.length > 12
                        ? ` · showing the 12 strongest of ${groups.length} groups`
                        : ""}
                    </p>
                  </>
                ) : (
                  <p className="text-[11px] text-[#8f9bb2]">
                    {parameterLabel(activeParameter.name)} is categorical — see the table below for
                    which choice the best trials landed on.
                  </p>
                )}
              </section>

              <section className="surface overflow-hidden">
                <header className="border-b border-white/[0.07] px-4 py-3">
                  <h3 className="text-sm font-semibold text-white">
                    Where the good trials sit
                  </h3>
                  <p className="mt-1 text-[11px] leading-5 text-[#7e8ba3]">
                    The top quartile is taken <em>per study</em> and then pooled, so studies scoring
                    at different levels contribute on equal terms.
                  </p>
                </header>
                <div className="overflow-x-auto">
                  <table className="w-full min-w-[620px] text-[11px]">
                    <thead>
                      <tr className="border-b border-white/[0.07]">
                        <th className="px-3 py-2 text-left text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e]">
                          Group
                        </th>
                        <th className="px-3 py-2 text-right text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e]">
                          {parameterIsNumeric ? "Median, top 25%" : "Best-trial favourite"}
                        </th>
                        <th className="px-3 py-2 text-right text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e]">
                          {parameterIsNumeric ? "Median, all trials" : "Share of top trials"}
                        </th>
                        <th className="px-3 py-2 text-right text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e]">
                          Trials
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {parameterInsight.filter((entry) => entry.all.length > 0).length === 0 ? (
                        <tr>
                          <td colSpan={4} className="px-4 py-8 text-center text-xs text-[#6f7d95]">
                            No group in this selection searched that parameter.
                          </td>
                        </tr>
                      ) : (
                        parameterInsight
                          .filter((entry) => entry.all.length > 0)
                          .map((entry) => {
                            const numericTop = entry.top
                              .map((value) => finite(value))
                              .filter((value): value is number => value != null);
                            const numericAll = entry.all
                              .map((value) => finite(value))
                              .filter((value): value is number => value != null);
                            let favourite = "—";
                            let share = "—";
                            if (!parameterIsNumeric) {
                              const counts = new Map<string, number>();
                              for (const value of entry.top) {
                                const key = displayValue(value);
                                counts.set(key, (counts.get(key) ?? 0) + 1);
                              }
                              const ranked = [...counts.entries()].sort((a, b) => b[1] - a[1]);
                              if (ranked.length > 0) {
                                favourite = ranked[0][0];
                                share = `${Math.round((ranked[0][1] / entry.top.length) * 100)}%`;
                              }
                            }
                            return (
                              <tr key={entry.group.key} className="border-b border-white/[0.04]">
                                <td className="px-3 py-2">
                                  <span className="flex items-center gap-2 text-[#c9d2e1]">
                                    <span
                                      className="inline-block h-2 w-2 shrink-0 rounded-full"
                                      style={{ background: entry.group.color }}
                                      aria-hidden="true"
                                    />
                                    {entry.group.label}
                                  </span>
                                </td>
                                <td className="px-3 py-2 text-right tabular-nums text-white">
                                  {parameterIsNumeric
                                    ? numericTop.length > 0
                                      ? formatNumber(median(numericTop))
                                      : "—"
                                    : favourite}
                                </td>
                                <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                                  {parameterIsNumeric
                                    ? numericAll.length > 0
                                      ? formatNumber(median(numericAll))
                                      : "—"
                                    : share}
                                </td>
                                <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                                  {entry.top.length} / {entry.all.length}
                                </td>
                              </tr>
                            );
                          })
                      )}
                    </tbody>
                  </table>
                </div>
              </section>
            </>
          )}

          {tab === "ranges" && (
            <>
              <RangeControls settings={rangeSettings} onChange={setRangeSettings} />

              {diagnostics.resolution != null && diagnostics.resolution <= 1 && (
                <p className="surface-soft px-3 py-2 text-[11px] leading-5 text-[#e8b957]">
                  The median study in this selection moves its objective by{" "}
                  {diagnostics.effectSize?.toFixed(4)} between its best and median trial, which is
                  inside the {diagnostics.epsilon?.toFixed(4)} tolerance. That is a statement about
                  the noise floor, not about the hyperparameters: no narrowing is justified from
                  this sweep until repeats bring the floor below the effect.
                </p>
              )}
              {diagnostics.truncatedShare != null && diagnostics.truncatedShare > 0.25 && (
                <p className="surface-soft px-3 py-2 text-[11px] leading-5 text-[#e8b957]">
                  {Math.round(diagnostics.truncatedShare * 100)}% of in-band trials had their best
                  epoch in the last fifth of the run, so these sweeps were cut short. A short run
                  flatters high learning rates — they win early and lose late — so skew the
                  downward margin wider before trusting the upper edge of an lr recommendation.
                </p>
              )}

              <section className="surface overflow-hidden">
                <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.07] px-4 py-3">
                  <div className="min-w-0">
                    <h3 className="text-sm font-semibold text-white">
                      Derived Phase 1 ranges
                    </h3>
                    <p className="mt-0.5 text-[11px] leading-5 text-[#7e8ba3]">
                      Across {assessments.length} stud{assessments.length === 1 ? "y" : "ies"} in
                      view. The good region is the union over them, so the recommendation covers
                      every optimum in the selection rather than an average of them. A range marked{" "}
                      <span className="text-[#e8b957]">beyond Phase 0</span> leaves what was
                      searched — check that the new bound is valid for the parameter, because the
                      rule knows the sweep and not the parameter&rsquo;s domain.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => {
                      const payload = JSON.stringify({ spaces: toSpaces(findings) }, null, 2);
                      void navigator.clipboard
                        ?.writeText(payload)
                        .then(() => {
                          setCopiedSpaces(true);
                          window.setTimeout(() => setCopiedSpaces(false), 2_000);
                        })
                        .catch(() => {
                          downloadText(payload, "phase-1-spaces.json", "application/json");
                        });
                    }}
                    disabled={findings.length === 0}
                    className="focus-ring inline-flex h-9 shrink-0 items-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.04] px-3.5 text-xs font-semibold text-[#c3ccdc] transition hover:bg-white/[0.08] hover:text-white disabled:opacity-40"
                  >
                    <ClipboardCopy size={14} />
                    {copiedSpaces ? "Copied" : "Copy spaces JSON"}
                  </button>
                </header>
                <FindingsTable
                  findings={findings}
                  selected={activeFinding?.name ?? null}
                  onSelect={setParameterName}
                />
              </section>

              <section className="surface space-y-4 p-4 sm:p-5">
                <SectionHeading
                  title={
                    activeFinding
                      ? `${parameterLabel(activeFinding.name)} — response and derived range`
                      : "Response and derived range"
                  }
                  description="Every trial that sampled the parameter, the tolerance band, the regions where runs die, and the range the rules put forward."
                  action={
                    <LabelledSelect
                      label="Parameter"
                      value={activeFinding?.name ?? ""}
                      options={findings.map((finding) => ({
                        value: finding.name,
                        label: `${parameterLabel(finding.name)} — ${VERDICT_LABELS[finding.verdict]}`,
                      }))}
                      onChange={setParameterName}
                    />
                  }
                />
                {activeFinding == null ? (
                  <EmptyChart message="No study in this selection searched a hyperparameter." />
                ) : activeFinding.kind === "categorical" ? (
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[520px] text-[11px]">
                      <thead>
                        <tr className="border-b border-white/[0.07]">
                          {["Choice", "Sampled", "In band", "Best shortfall", "Verdict"].map(
                            (label, index) => (
                              <th
                                key={label}
                                className={`px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e] ${
                                  index === 0 ? "text-left" : "text-right"
                                }`}
                              >
                                {label}
                              </th>
                            ),
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {activeFinding.choices.map((choice) => (
                          <tr key={choice.label} className="border-b border-white/[0.04]">
                            <td className="px-3 py-2 text-[#c9d2e1]">{choice.label}</td>
                            <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {choice.sampleCount}
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-white">
                              {choice.keepCount}
                            </td>
                            <td className="px-3 py-2 text-right tabular-nums text-[#8f9bb2]">
                              {choice.bestShortfall != null
                                ? choice.bestShortfall.toFixed(4)
                                : "—"}
                            </td>
                            <td className="px-3 py-2 text-right">
                              <span
                                className={
                                  choice.drop ? "text-[#e8b957]" : "text-[#8f9bb2]"
                                }
                              >
                                {choice.drop ? "drop" : "keep"}
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <ResponseCurve
                    finding={activeFinding}
                    samples={activeSamples}
                    epsilon={diagnostics.epsilon ?? rangeSettings.epsilonAbsolute}
                    absoluteBest={
                      assessments.length === 1 ? (assessments[0].best ?? null) : null
                    }
                    metric={assessments[0]?.study.metric ?? "objective"}
                    selectedTrial={selectedTrial}
                    onSelectTrial={selectTrial}
                  />
                )}

                {activeFinding && (
                  <div className="surface-soft space-y-1.5 p-3">
                    <div className="flex flex-wrap items-center gap-2">
                      <VerdictPill
                        verdict={activeFinding.verdict}
                        muted={!activeFinding.resolvable}
                      />
                      {!activeFinding.resolvable && (
                        <span className="text-[11px] font-semibold text-[#e8b957]">
                          below the noise floor
                        </span>
                      )}
                    </div>
                    <ul className="space-y-1 text-[11px] leading-5 text-[#a3aec2]">
                      {activeFinding.reasons.map((reason) => (
                        <li key={reason} className="flex gap-2">
                          <span className="text-[#5c6a82]">·</span>
                          <span>{reason}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </section>

              {activeFinding && activeFinding.kind === "numeric" && assessments.length > 1 && (
                <section className="surface overflow-hidden">
                  <header className="border-b border-white/[0.07] px-4 py-3">
                    <h3 className="text-sm font-semibold text-white">
                      Where each study puts the optimum
                    </h3>
                    <p className="mt-1 text-[11px] leading-5 text-[#7e8ba3]">
                      The pooled recommendation covers all of these. Wide disagreement here is a
                      sign the parameter should be set per dataset or per loss rather than shared.
                    </p>
                  </header>
                  <div className="max-h-[42vh] overflow-auto">
                    <table className="w-full min-w-[760px] text-[11px]">
                      <thead className="sticky top-0 z-10 bg-[#111a2c]">
                        <tr className="border-b border-white/[0.07]">
                          {["Study", "ε", "Best at", "Good region", "Verdict"].map(
                            (label, index) => (
                              <th
                                key={label}
                                className={`px-2.5 py-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#79869e] ${
                                  index === 0 ? "text-left" : index === 4 ? "text-left" : "text-right"
                                }`}
                              >
                                {label}
                              </th>
                            ),
                          )}
                        </tr>
                      </thead>
                      <tbody>
                        {assessments
                          .map((assessment) => ({
                            assessment,
                            finding: findRange(activeFinding.name, [assessment], rangeSettings),
                          }))
                          .filter((row) => row.finding.studyCount > 0)
                          .map(({ assessment, finding }) => (
                            <tr
                              key={assessment.study.id}
                              onClick={() => setSelectedStudyId(assessment.study.id)}
                              className="cursor-pointer border-b border-white/[0.04] hover:bg-white/[0.035]"
                            >
                              <td
                                className="max-w-[24rem] px-2.5 py-2"
                                title={assessment.study.path}
                              >
                                <span className="block truncate font-mono text-[10px] text-[#c9d2e1]">
                                  {assessment.study.path}
                                </span>
                              </td>
                              <td className="px-2.5 py-2 text-right tabular-nums text-[#8f9bb2]">
                                {assessment.epsilon.toFixed(4)}
                              </td>
                              <td className="px-2.5 py-2 text-right tabular-nums text-white">
                                {finding.kind === "numeric" && finding.bestValue != null
                                  ? formatNumber(finding.bestValue, 3)
                                  : "—"}
                              </td>
                              <td className="px-2.5 py-2 text-right tabular-nums text-[#8f9bb2]">
                                {finding.kind === "numeric"
                                  ? rangeText(finding.good, finding.isInt)
                                  : "—"}
                              </td>
                              <td className="px-2.5 py-2">
                                <VerdictPill
                                  verdict={finding.verdict}
                                  muted={!finding.resolvable}
                                />
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </section>
              )}
            </>
          )}

          {tab === "trials" && (
            <section className="surface overflow-hidden">
              <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.07] px-4 py-3">
                <div>
                  <h3 className="text-sm font-semibold text-white">
                    {trialRows.length.toLocaleString()} scored trials
                  </h3>
                  <p className="mt-0.5 text-[10px] text-[#718098]">
                    Showing the {Math.min(visibleTrials, sortedTrialRows.length).toLocaleString()}{" "}
                    highest under the current sort
                    {trialParameterColumns.length > shownParameterColumns.length
                      ? ` · ${trialParameterColumns.length - shownParameterColumns.length} more hyperparameters are in the CSV`
                      : ""}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={exportTrials}
                  disabled={trialRows.length === 0}
                  className="focus-ring inline-flex h-9 items-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.04] px-3.5 text-xs font-semibold text-[#c3ccdc] transition hover:bg-white/[0.08] hover:text-white disabled:opacity-40"
                >
                  <Download size={14} /> Export CSV
                </button>
              </header>
              <div className="max-h-[62vh] overflow-auto">
                <table className="w-full min-w-[1100px] text-[11px]">
                  <thead className="sticky top-0 z-10 bg-[#111a2c]">
                    <tr className="border-b border-white/[0.07]">
                      {[
                        { key: "study", label: "Study", align: "left" as const },
                        { key: "number", label: "#", align: "right" as const },
                        { key: "value", label: "Objective", align: "right" as const },
                        { key: "delta", label: "Δ sup.", align: "right" as const },
                        { key: "p1", label: "P@1", align: "right" as const },
                        { key: "fold", label: "Fold ±", align: "right" as const },
                        { key: "epoch", label: "Epoch", align: "right" as const },
                        { key: "seconds", label: "Runtime", align: "right" as const },
                      ].map((column) => (
                        <SortHeader
                          key={column.key}
                          label={column.label}
                          columnKey={column.key}
                          align={column.align}
                          sort={trialSort}
                          onSort={(key) =>
                            setTrialSort((current) => ({
                              key,
                              direction:
                                current.key === key && current.direction === "desc" ? "asc" : "desc",
                            }))
                          }
                        />
                      ))}
                      {shownParameterColumns.map((name) => (
                        <SortHeader
                          key={name}
                          label={parameterLabel(name)}
                          columnKey={`param:${name}`}
                          align="right"
                          title={name}
                          sort={trialSort}
                          onSort={(key) =>
                            setTrialSort((current) => ({
                              key,
                              direction:
                                current.key === key && current.direction === "desc" ? "asc" : "desc",
                            }))
                          }
                        />
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sortedTrialRows.length === 0 ? (
                      <tr>
                        <td
                          colSpan={8 + shownParameterColumns.length}
                          className="px-4 py-10 text-center text-xs text-[#6f7d95]"
                        >
                          No scored trial matches these filters.
                        </td>
                      </tr>
                    ) : (
                      sortedTrialRows.slice(0, visibleTrials).map((row) => (
                        <tr
                          key={`${row.stats.study.id}#${row.trial.number}`}
                          className={`border-b border-white/[0.04] ${
                            selectedTrial?.studyId === row.stats.study.id &&
                            selectedTrial.trialNumber === row.trial.number
                              ? "bg-[#9b8cff]/[0.11]"
                              : "hover:bg-white/[0.03]"
                          }`}
                        >
                          <td
                            className="max-w-[18rem] px-2.5 py-1.5"
                            title={row.stats.study.path}
                          >
                            <button
                              type="button"
                              onClick={() => {
                                setSelectedStudyId(row.stats.study.id);
                                setTab("studies");
                              }}
                              className="focus-ring block max-w-full truncate text-left font-mono text-[10px] text-[#b6acff] hover:underline"
                            >
                              {shortPath(row.stats.study.path)}
                            </button>
                          </td>
                          <td className="px-2.5 py-1.5 text-right tabular-nums">
                            <button
                              type="button"
                              onClick={() =>
                                selectTrial({
                                  studyId: row.stats.study.id,
                                  trialNumber: row.trial.number,
                                })
                              }
                              className="focus-ring rounded px-1 text-[#b6acff] hover:underline"
                              aria-label={`Show details for trial ${row.trial.number}`}
                            >
                              {row.trial.number}
                            </button>
                          </td>
                          <td className="px-2.5 py-1.5 text-right tabular-nums text-white">
                            {formatMetric(row.trial.value)}
                          </td>
                          <td className="px-2.5 py-1.5 text-right tabular-nums text-[#a8b3c7]">
                            {signedMetric(row.delta)}
                          </td>
                          <td className="px-2.5 py-1.5 text-right tabular-nums text-[#8f9bb2]">
                            {formatMetric(finite(row.trial.metrics.best_valid_precision_at_1))}
                          </td>
                          <td className="px-2.5 py-1.5 text-right tabular-nums text-[#8f9bb2]">
                            {foldValues(row.trial).length > 1
                              ? standardDeviation(foldValues(row.trial)).toFixed(4)
                              : "—"}
                          </td>
                          <td className="px-2.5 py-1.5 text-right tabular-nums text-[#8f9bb2]">
                            {finite(row.trial.metrics.selected_epoch) ?? "—"}
                          </td>
                          <td className="px-2.5 py-1.5 text-right tabular-nums text-[#8f9bb2]">
                            {formatDuration(row.trial.seconds)}
                          </td>
                          {shownParameterColumns.map((name) => (
                            <td
                              key={name}
                              className="px-2.5 py-1.5 text-right tabular-nums text-[#a8b3c7]"
                            >
                              {displayValue(row.trial.params[name] ?? null)}
                            </td>
                          ))}
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
              </div>
              {visibleTrials < sortedTrialRows.length && (
                <div className="border-t border-white/[0.07] p-3 text-center">
                  <button
                    type="button"
                    onClick={() => setVisibleTrials((current) => current + TRIAL_PAGE)}
                    className="focus-ring inline-flex h-9 items-center gap-2 rounded-xl border border-white/[0.09] bg-white/[0.04] px-4 text-xs font-semibold text-[#c3ccdc] hover:bg-white/[0.08] hover:text-white"
                  >
                    <Boxes size={14} /> Show {TRIAL_PAGE} more of{" "}
                    {(sortedTrialRows.length - visibleTrials).toLocaleString()}
                  </button>
                </div>
              )}
            </section>
          )}

          {selectedStats && (
            <StudyDetail
              stats={selectedStats}
              baseline={baselineFor(baselines, selectedStats.study)}
              onClose={() => setSelectedStudyId(null)}
              onSelectTrial={(trialNumber) =>
                selectTrial({ studyId: selectedStats.study.id, trialNumber })
              }
            />
          )}

          {(snapshot.missingFromDisk?.length ?? 0) > 0 && (
            <p className="text-[10px] leading-4 text-[#68768e]">
              <BarChart3 size={11} className="mr-1 inline align-[-1px]" />
              {snapshot.missingFromDisk?.length} studies listed in sorting_map.csv have no directory
              on disk and are not in this view.
            </p>
          )}
        </div>
      </div>
      {selectedTrialStudy && selectedTrialRecord && (
        <TrialDetail
          study={selectedTrialStudy}
          trial={selectedTrialRecord}
          onClose={() => setSelectedTrial(null)}
        />
      )}
    </div>
  );
}
