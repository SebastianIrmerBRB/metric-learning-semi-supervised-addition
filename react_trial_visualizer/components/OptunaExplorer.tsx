"use client";

import {
  useEffect,
  useMemo,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type ReactNode,
} from "react";
import {
  Activity,
  BarChart3,
  Database,
  Eye,
  EyeOff,
  Gauge,
  Layers3,
  RefreshCw,
  Search,
  SlidersHorizontal,
  Sparkles,
  Target,
  Trophy,
} from "lucide-react";
import { DiagramCard } from "@/components/Charts";
import {
  contourLayers,
  contourLevels,
  fitResponseSurface,
  sampleGrid,
  type PlotBox,
  type StationaryPoint,
  type SurfaceGrid,
  type SurfaceModel,
  type SurfaceSample,
} from "@/lib/response-surface";
import {
  isOptunaSnapshot,
  type OptunaParamValue,
  type OptunaParameterSnapshot,
  type OptunaSnapshot,
  type OptunaStudySnapshot,
  type OptunaTrialSnapshot,
} from "@/lib/optuna";
import { formatMetric, formatNumber, slugify } from "@/lib/trials";

const SNAPSHOT_URL = "/optuna-studies.json";
const STUDY_STORAGE_KEY = "trial-atlas:optuna-study";
const PARAMETER_STORAGE_PREFIX = "trial-atlas:optuna-parameter:";
const HIDDEN_STUDIES_STORAGE_KEY = "trial-atlas:optuna-hidden-studies";
const inputClass =
  "focus-ring h-10 w-full rounded-xl border border-white/[0.09] bg-[#0c1425] px-3 text-xs text-[#d9e0ed] placeholder:text-[#53617a]";

const CHART = {
  grid: "#27334c",
  axis: "#77859f",
  text: "#dce3f1",
  muted: "#8f9bb2",
  violet: "#9b8cff",
  cyan: "#4fd5e7",
  lime: "#b8e56e",
  amber: "#f4c86f",
};

/** Tukey multiplier used to detect the failure tail on the objective axis. */
const OUTLIER_FENCE = 1.5;
/** Below this many trials the quartiles are too noisy to trust. */
const MINIMUM_TRIALS_FOR_CLIPPING = 8;
/** Never fold away more than this share of a study's trials. */
const MAXIMUM_CLIPPED_SHARE = 0.2;
/** Only fold the tail when it owns at least this share of the full range. */
const MINIMUM_RECLAIMED_SHARE = 0.15;

interface ParameterPoint {
  trial: OptunaTrialSnapshot;
  value: OptunaParamValue;
  numericValue: number | null;
  objective: number;
}

interface BehaviorGroup {
  key: string;
  label: string;
  chartLabelLines: string[];
  center: number | null;
  count: number;
  median: number;
  best: number;
  minimum: number;
  maximum: number;
}

function MetricCard({
  label,
  value,
  context,
  icon,
  accent = "violet",
}: {
  label: string;
  value: string;
  context: string;
  icon: ReactNode;
  accent?: "violet" | "cyan" | "lime" | "amber";
}) {
  const styles = {
    violet: "bg-[#9b8cff]/10 text-[#b5aaff] ring-[#9b8cff]/15",
    cyan: "bg-[#4fd5e7]/10 text-[#6fe1ee] ring-[#4fd5e7]/15",
    lime: "bg-[#b8e56e]/10 text-[#c9f18a] ring-[#b8e56e]/15",
    amber: "bg-[#f4c86f]/10 text-[#f6d488] ring-[#f4c86f]/15",
  }[accent];

  return (
    <article className="surface relative overflow-hidden px-5 py-5">
      <div className="absolute -right-10 -top-12 h-28 w-28 rounded-full bg-white/[0.025] blur-xl" />
      <div className="relative flex items-start justify-between gap-4">
        <div className="min-w-0">
          <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-[#77849c]">
            {label}
          </p>
          <p
            className="mt-3 truncate text-2xl font-semibold tracking-[-0.035em] text-white"
            title={value}
          >
            {value}
          </p>
          <p className="mt-1.5 truncate text-[11px] text-[#7f8ca3]" title={context}>
            {context}
          </p>
        </div>
        <span className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-xl ring-1 ${styles}`}>
          {icon}
        </span>
      </div>
    </article>
  );
}

function parameterLabel(name: string): string {
  if (name === "__joint_hparam__.two_stream_batch_sampler") {
    return "Two-stream batch configuration";
  }
  if (name === "lr") return "Learning rate";
  if (name === "classifier_lr") return "Classifier learning rate";
  if (name === "batch_sampler") return "Batch sampler";

  const parts = name.split(".");
  const leaf = parts.at(-1) ?? name;
  const parent = parts.at(-2) ?? "";
  const parentLabel = parent
    .replace(/Loss$/i, "")
    .replace(/Miner$/i, "")
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2");
  const leafLabel = leaf
    .replace(/_/g, " ")
    .replace(/([a-z])([A-Z])/g, "$1 $2");

  return [parentLabel, leafLabel]
    .filter(Boolean)
    .map((part) => part.replace(/\b\w/g, (letter) => letter.toUpperCase()))
    .join(" · ");
}

function objectiveLabel(name: string): string {
  const normalized = name.toLowerCase();
  if (normalized === "best_valid_mean_average_precision_at_r") {
    return "Validation mAP@R";
  }
  if (normalized === "best_valid_precision_at_1") return "Validation P@1";
  if (normalized === "mean_average_precision_at_r") return "mAP@R";
  if (normalized === "precision_at_1") return "P@1";
  return name
    .replace(/^best_/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function parameterAxisLabel(parameter: OptunaParameterSnapshot): string {
  if (parameter.name === "__joint_hparam__.two_stream_batch_sampler") {
    return "Batch sampler / labeled batch size";
  }
  return parameterLabel(parameter.name);
}

function compactParameterLabel(name: string, maximum = 32): string {
  const label = parameterLabel(name);
  return label.length <= maximum ? label : `${label.slice(0, maximum - 1)}…`;
}

function valueKey(value: OptunaParamValue): string {
  return JSON.stringify([typeof value, value]);
}

interface StructuredParamField {
  label: string;
  shortLabel: string;
  value: string;
}

function structuredFieldLabels(name: string): Pick<StructuredParamField, "label" | "shortLabel"> {
  if (name === "batch_sampler") {
    return { label: "Batch sampler", shortLabel: "Sampler" };
  }
  if (name === "ssl_config.labeled_batch_size") {
    return { label: "Labeled batch size", shortLabel: "Labeled" };
  }

  const label = parameterLabel(name);
  const shortLabel = label.split(" · ").at(-1) ?? label;
  return { label, shortLabel };
}

function structuredParamFields(
  value: OptunaParamValue | undefined,
): StructuredParamField[] | null {
  if (typeof value !== "string" || !value.trimStart().startsWith("{")) return null;

  try {
    const parsed: unknown = JSON.parse(value);
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;

    const fields = Object.entries(parsed as Record<string, unknown>).flatMap(
      ([name, fieldValue]) => {
        if (
          fieldValue !== null &&
          typeof fieldValue !== "string" &&
          typeof fieldValue !== "number" &&
          typeof fieldValue !== "boolean"
        ) {
          return [];
        }
        const labels = structuredFieldLabels(name);
        const formattedValue =
          fieldValue === null
            ? "None"
            : typeof fieldValue === "boolean"
              ? fieldValue
                ? "True"
                : "False"
              : String(fieldValue);
        return [{ ...labels, value: formattedValue }];
      },
    );
    return fields.length > 0 ? fields : null;
  } catch {
    return null;
  }
}

function chartParamValueLines(
  value: OptunaParamValue | undefined,
  parameter?: OptunaParameterSnapshot,
): string[] {
  const fields = structuredParamFields(value);
  if (fields) {
    const lines = fields.slice(0, 2).map((field) => `${field.shortLabel} ${field.value}`);
    if (fields.length > 2) lines[1] = `${lines[1]} +${fields.length - 2}`;
    return lines;
  }

  const label = displayParamValue(value, parameter);
  if (label.length <= 18) return [label];
  return [`${label.slice(0, 9)}…${label.slice(-8)}`];
}

function displayParamValue(
  value: OptunaParamValue | undefined,
  parameter?: OptunaParameterSnapshot,
): string {
  if (value === undefined || value === null) return "—";
  const fields = structuredParamFields(value);
  if (fields) {
    return fields.map((field) => `${field.label} ${field.value}`).join(" / ");
  }
  if (typeof value === "number") {
    if (parameter?.distribution === "int") return Math.round(value).toLocaleString("en-US");
    return formatNumber(value, 5);
  }
  if (typeof value === "boolean") return value ? "True" : "False";
  return String(value);
}

function axisNumber(value: number): string {
  const absolute = Math.abs(value);
  if (value !== 0 && (absolute < 0.001 || absolute >= 10_000)) {
    return value.toExponential(1);
  }
  if (absolute >= 100) return Math.round(value).toLocaleString("en-US");
  if (absolute >= 1) return Number(value.toFixed(2)).toString();
  return Number(value.toFixed(3)).toString();
}

function extent(values: number[], padding = 0.08): [number, number] {
  if (values.length === 0) return [0, 1];
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (minimum === maximum) {
    const delta = Math.abs(minimum) * 0.05 || 1;
    return [minimum - delta, maximum + delta];
  }
  const delta = (maximum - minimum) * padding;
  return [minimum - delta, maximum + delta];
}

function linearScale(domain: [number, number], range: [number, number]) {
  const span = domain[1] - domain[0] || 1;
  return (value: number) =>
    range[0] + ((value - domain[0]) / span) * (range[1] - range[0]);
}

function ticks(domain: [number, number], count: number): number[] {
  return Array.from(
    { length: count },
    (_, index) => domain[0] + ((domain[1] - domain[0]) * index) / (count - 1),
  );
}

function quantile(values: number[], probability: number): number {
  if (values.length === 0) return Number.NaN;
  const sorted = [...values].sort((left, right) => left - right);
  const position = (sorted.length - 1) * probability;
  const lower = Math.floor(position);
  const upper = Math.ceil(position);
  if (lower === upper) return sorted[lower];
  const weight = position - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

function median(values: number[]): number {
  return quantile(values, 0.5);
}

interface ObjectiveScale {
  domain: [number, number];
  /** Side of the axis that outlier trials were folded onto, if any. */
  clipSide: "low" | "high" | null;
  /** Number of trials pinned to the axis edge instead of stretching the domain. */
  clippedCount: number;
  /** Worst objective that fell outside the domain, for labelling. */
  clippedExtreme: number | null;
  /** True when this value sits outside the plotted domain. */
  isClipped: (value: number) => boolean;
  /** Value to draw at: clipped points are pinned just inside the axis edge. */
  plot: (value: number) => number;
}

/**
 * Objective axis that ignores the failure tail. A handful of collapsed trials
 * would otherwise own most of the range and flatten every surviving trial into
 * a single band, which reads as "this parameter does nothing". Only the tail
 * away from the study's optimum is trimmed (Tukey fence), so the best trials
 * always keep their real position; trimmed trials are pinned to the axis edge
 * rather than dropped.
 */
function objectiveScale(
  values: number[],
  direction: OptunaStudySnapshot["direction"],
  padding = 0.12,
): ObjectiveScale {
  const finite = values.filter((value) => Number.isFinite(value));
  const plain = (domain: [number, number]): ObjectiveScale => ({
    domain,
    clipSide: null,
    clippedCount: 0,
    clippedExtreme: null,
    isClipped: () => false,
    plot: (value) => value,
  });
  const fullDomain = extent(finite, padding);
  if (finite.length < MINIMUM_TRIALS_FOR_CLIPPING) return plain(fullDomain);

  const firstQuartile = quantile(finite, 0.25);
  const thirdQuartile = quantile(finite, 0.75);
  const interQuartile = thirdQuartile - firstQuartile;
  if (!(interQuartile > 0)) return plain(fullDomain);

  const clipSide = direction === "minimize" ? "high" : "low";
  const fence =
    clipSide === "low"
      ? firstQuartile - OUTLIER_FENCE * interQuartile
      : thirdQuartile + OUTLIER_FENCE * interQuartile;
  const kept = finite.filter((value) =>
    clipSide === "low" ? value >= fence : value <= fence,
  );
  const clipped = finite.filter((value) =>
    clipSide === "low" ? value < fence : value > fence,
  );
  if (clipped.length === 0) return plain(fullDomain);
  if (clipped.length > finite.length * MAXIMUM_CLIPPED_SHARE) {
    return plain(fullDomain);
  }

  const domain = extent(kept, padding);
  const fullSpan = fullDomain[1] - fullDomain[0];
  const reclaimed =
    clipSide === "low" ? domain[0] - fullDomain[0] : fullDomain[1] - domain[1];
  // Not worth an off-scale marker if the tail was barely stretching the axis.
  if (!(fullSpan > 0) || reclaimed / fullSpan < MINIMUM_RECLAIMED_SHARE) {
    return plain(fullDomain);
  }

  const inset = (domain[1] - domain[0]) * 0.035;
  return {
    domain,
    clipSide,
    clippedCount: clipped.length,
    clippedExtreme:
      clipSide === "low" ? Math.min(...clipped) : Math.max(...clipped),
    isClipped: (value) => value < domain[0] || value > domain[1],
    plot: (value) =>
      value < domain[0]
        ? domain[0] + inset
        : value > domain[1]
          ? domain[1] - inset
          : value,
  };
}

/** Marker for one trial; folded outliers become a triangle at the axis edge. */
function TrialMarker({
  x,
  y,
  radius,
  fill,
  fillOpacity,
  clipSide,
  children,
}: {
  x: number;
  y: number;
  radius: number;
  fill: string;
  fillOpacity: number;
  clipSide: "low" | "high" | null;
  children: ReactNode;
}) {
  const size = radius * 1.3;
  const tip = clipSide === "low" ? y + size : y - size;
  const base = clipSide === "low" ? y - size * 0.65 : y + size * 0.65;
  return clipSide ? (
    <polygon
      className="chart-point"
      points={`${x},${tip} ${x - size},${base} ${x + size},${base}`}
      fill={fill}
      fillOpacity={fillOpacity}
      stroke="#10182a"
      strokeWidth="1.75"
    >
      {children}
    </polygon>
  ) : (
    <circle
      className="chart-point"
      cx={x}
      cy={y}
      r={radius}
      fill={fill}
      fillOpacity={fillOpacity}
      stroke="#10182a"
      strokeWidth="1.75"
    >
      {children}
    </circle>
  );
}

/** Axis note explaining the folded tail, drawn top-right of the plot area. */
function ClippedNote({
  scale,
  x,
  y,
}: {
  scale: ObjectiveScale;
  x: number;
  y: number;
}) {
  if (scale.clippedCount === 0 || scale.clippedExtreme == null) return null;
  return (
    <text x={x} y={y} fill={CHART.muted} fontSize="11" textAnchor="end">
      {scale.clippedCount} outlier trial{scale.clippedCount === 1 ? "" : "s"} at
      the axis edge (to {formatMetric(scale.clippedExtreme)})
    </text>
  );
}

function bestValue(
  values: number[],
  direction: OptunaStudySnapshot["direction"],
): number {
  return direction === "minimize" ? Math.min(...values) : Math.max(...values);
}

function pointsFor(
  study: OptunaStudySnapshot,
  parameter: OptunaParameterSnapshot,
): ParameterPoint[] {
  return study.trials.flatMap((trial) => {
    if (!Object.hasOwn(trial.params, parameter.name)) return [];
    const value = trial.params[parameter.name];
    const numericValue =
      typeof value === "number" && Number.isFinite(value) ? value : null;
    if (parameter.kind === "numeric" && numericValue == null) return [];
    return [
      {
        trial,
        value,
        numericValue,
        objective: trial.objective,
      },
    ];
  });
}

function rankValues(values: number[]): number[] {
  const indexed = values
    .map((value, index) => ({ value, index }))
    .sort((left, right) => left.value - right.value);
  const ranks = Array(values.length).fill(0) as number[];

  for (let start = 0; start < indexed.length; ) {
    let end = start + 1;
    while (end < indexed.length && indexed[end].value === indexed[start].value) {
      end += 1;
    }
    const averageRank = (start + end - 1) / 2 + 1;
    for (let index = start; index < end; index += 1) {
      ranks[indexed[index].index] = averageRank;
    }
    start = end;
  }
  return ranks;
}

function pearson(left: number[], right: number[]): number | null {
  if (left.length !== right.length || left.length < 3) return null;
  const leftMean = left.reduce((sum, value) => sum + value, 0) / left.length;
  const rightMean = right.reduce((sum, value) => sum + value, 0) / right.length;
  let numerator = 0;
  let leftSquare = 0;
  let rightSquare = 0;
  for (let index = 0; index < left.length; index += 1) {
    const leftDelta = left[index] - leftMean;
    const rightDelta = right[index] - rightMean;
    numerator += leftDelta * rightDelta;
    leftSquare += leftDelta ** 2;
    rightSquare += rightDelta ** 2;
  }
  const denominator = Math.sqrt(leftSquare * rightSquare);
  return denominator === 0 ? null : numerator / denominator;
}

function relationshipSummary(
  points: ParameterPoint[],
  parameter: OptunaParameterSnapshot,
): { value: string; context: string } {
  if (parameter.kind === "numeric") {
    const numericPoints = points.filter(
      (point): point is ParameterPoint & { numericValue: number } =>
        point.numericValue != null,
    );
    const coefficient = pearson(
      rankValues(numericPoints.map((point) => point.numericValue)),
      rankValues(numericPoints.map((point) => point.objective)),
    );
    if (coefficient == null) return { value: "—", context: "Not enough variation" };
    const strength =
      Math.abs(coefficient) >= 0.6
        ? "Strong"
        : Math.abs(coefficient) >= 0.3
          ? "Moderate"
          : Math.abs(coefficient) >= 0.1
            ? "Weak"
            : "Flat";
    const direction =
      Math.abs(coefficient) < 0.1
        ? "rank relationship"
        : coefficient > 0
          ? "positive relationship"
          : "negative relationship";
    return {
      value: `${coefficient >= 0 ? "+" : ""}${coefficient.toFixed(2)} ρ`,
      context: `${strength} ${direction}`,
    };
  }

  const medians = categoricalGroups(points, { direction: "maximize" }).map(
    (group) => group.median,
  );
  if (medians.length < 2) return { value: "—", context: "Only one observed choice" };
  const spread = Math.max(...medians) - Math.min(...medians);
  return {
    value: `Δ ${formatMetric(spread)}`,
    context: "Median objective spread by choice",
  };
}

function categoricalGroups(
  points: ParameterPoint[],
  study: Pick<OptunaStudySnapshot, "direction">,
  parameter?: OptunaParameterSnapshot,
): BehaviorGroup[] {
  const observed = new Map<string, { value: OptunaParamValue; objectives: number[] }>();
  for (const point of points) {
    const key = valueKey(point.value);
    const entry = observed.get(key) ?? { value: point.value, objectives: [] };
    entry.objectives.push(point.objective);
    observed.set(key, entry);
  }

  const orderedKeys = [
    ...(parameter?.choices ?? []).map(valueKey),
    ...observed.keys(),
  ].filter((key, index, values) => values.indexOf(key) === index);

  return orderedKeys.flatMap((key) => {
    const entry = observed.get(key);
    if (!entry || entry.objectives.length === 0) return [];
    return [
      {
        key,
        label: displayParamValue(entry.value, parameter),
        chartLabelLines: chartParamValueLines(entry.value, parameter),
        center: null,
        count: entry.objectives.length,
        median: median(entry.objectives),
        best: bestValue(entry.objectives, study.direction),
        minimum: Math.min(...entry.objectives),
        maximum: Math.max(...entry.objectives),
      },
    ];
  });
}

function numericGroups(
  points: ParameterPoint[],
  study: Pick<OptunaStudySnapshot, "direction">,
  parameter: OptunaParameterSnapshot,
): BehaviorGroup[] {
  const numericPoints = points.filter(
    (point): point is ParameterPoint & { numericValue: number } =>
      point.numericValue != null,
  );
  if (numericPoints.length === 0) return [];

  const uniqueValues = [...new Set(numericPoints.map((point) => point.numericValue))].sort(
    (left, right) => left - right,
  );
  if (parameter.distribution === "int" && uniqueValues.length <= 12) {
    return uniqueValues.map((value) => {
      const objectives = numericPoints
        .filter((point) => point.numericValue === value)
        .map((point) => point.objective);
      return {
        key: String(value),
        label: displayParamValue(value, parameter),
        chartLabelLines: chartParamValueLines(value, parameter),
        center: value,
        count: objectives.length,
        median: median(objectives),
        best: bestValue(objectives, study.direction),
        minimum: Math.min(...objectives),
        maximum: Math.max(...objectives),
      };
    });
  }

  const useLog =
    parameter.log && numericPoints.every((point) => point.numericValue > 0);
  const transform = (value: number) => (useLog ? Math.log10(value) : value);
  const invert = (value: number) => (useLog ? 10 ** value : value);
  const transformed = numericPoints.map((point) => transform(point.numericValue));
  const minimum = Math.min(...transformed);
  const maximum = Math.max(...transformed);
  if (minimum === maximum) {
    const objectives = numericPoints.map((point) => point.objective);
    return [
      {
        key: String(uniqueValues[0]),
        label: displayParamValue(uniqueValues[0], parameter),
        chartLabelLines: chartParamValueLines(uniqueValues[0], parameter),
        center: uniqueValues[0],
        count: objectives.length,
        median: median(objectives),
        best: bestValue(objectives, study.direction),
        minimum: Math.min(...objectives),
        maximum: Math.max(...objectives),
      },
    ];
  }

  const binCount = Math.min(8, Math.max(4, Math.round(Math.sqrt(points.length))));
  const span = maximum - minimum;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    index,
    low: minimum + (span * index) / binCount,
    high: minimum + (span * (index + 1)) / binCount,
    points: [] as Array<ParameterPoint & { numericValue: number }>,
  }));
  for (const point of numericPoints) {
    const transformedValue = transform(point.numericValue);
    const index = Math.min(
      binCount - 1,
      Math.max(0, Math.floor(((transformedValue - minimum) / span) * binCount)),
    );
    bins[index].points.push(point);
  }

  return bins.flatMap((bin) => {
    if (bin.points.length === 0) return [];
    const objectives = bin.points.map((point) => point.objective);
    const center = median(bin.points.map((point) => point.numericValue));
    const label = `${displayParamValue(invert(bin.low), parameter)} – ${displayParamValue(
      invert(bin.high),
      parameter,
    )}`;
    return [
      {
        key: String(bin.index),
        label,
        chartLabelLines: [label],
        center,
        count: objectives.length,
        median: median(objectives),
        best: bestValue(objectives, study.direction),
        minimum: Math.min(...objectives),
        maximum: Math.max(...objectives),
      },
    ];
  });
}

function ImportanceChart({
  study,
  selectedParameter,
  onSelect,
}: {
  study: OptunaStudySnapshot;
  selectedParameter: string;
  onSelect: (name: string) => void;
}) {
  const width = 820;
  const left = 276;
  const right = 72;
  const top = 40;
  const bottom = 50;
  const rowHeight = 38;
  const height = Math.max(310, top + bottom + study.parameters.length * rowHeight);
  const maximum = Math.max(
    ...study.parameters.map((parameter) => parameter.importance),
    0.01,
  );
  const x = linearScale([0, maximum * 1.08], [left, width - right]);
  const chartTicks = ticks([0, maximum], 5);

  const handleKey = (
    event: ReactKeyboardEvent<SVGGElement>,
    parameterName: string,
  ) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      onSelect(parameterName);
    }
  };

  return (
    <svg
      data-export-root
      viewBox={`0 0 ${width} ${height}`}
      className="block w-full"
      role="img"
      aria-label={`Parameter importances for ${study.displayName}`}
    >
      <title>fANOVA parameter importance for {study.displayName}</title>
      {chartTicks.map((value) => (
        <g key={value} aria-hidden="true">
          <line
            x1={x(value)}
            x2={x(value)}
            y1={top - 12}
            y2={height - bottom}
            stroke={CHART.grid}
          />
          <text
            x={x(value)}
            y={height - 20}
            textAnchor="middle"
            fill={CHART.axis}
            fontSize="11"
          >
            {(value * 100).toFixed(value >= 0.1 ? 0 : 1)}%
          </text>
        </g>
      ))}
      {study.parameters.map((parameter, index) => {
        const centerY = top + index * rowHeight + rowHeight / 2;
        const selected = parameter.name === selectedParameter;
        return (
          <g
            key={parameter.name}
            role="button"
            tabIndex={0}
            aria-label={`Inspect ${parameterLabel(parameter.name)}, ${(parameter.importance * 100).toFixed(1)} percent importance`}
            className="cursor-pointer outline-none"
            onClick={() => onSelect(parameter.name)}
            onKeyDown={(event) => handleKey(event, parameter.name)}
          >
            {selected && (
              <rect
                x="8"
                y={centerY - rowHeight / 2 + 3}
                width={width - 16}
                height={rowHeight - 6}
                rx="9"
                fill="rgba(79,213,231,0.07)"
                stroke="rgba(79,213,231,0.18)"
              />
            )}
            <text
              x="22"
              y={centerY + 4}
              fill={selected ? CHART.cyan : CHART.axis}
              fontSize="10"
              fontWeight="700"
            >
              {parameter.rank}
            </text>
            <text
              x="46"
              y={centerY + 4}
              fill={selected ? CHART.text : "#aab4c6"}
              fontSize="11"
              fontWeight={selected ? "650" : "500"}
            >
              {compactParameterLabel(parameter.name)}
            </text>
            <rect
              x={left}
              y={centerY - 9}
              width={Math.max(2, x(parameter.importance) - x(0))}
              height="18"
              rx="9"
              fill={selected ? CHART.cyan : index === 0 ? CHART.lime : CHART.violet}
              fillOpacity={selected ? 0.95 : 0.78}
            >
              <title>
                {parameter.name} · {(parameter.importance * 100).toFixed(2)}%
              </title>
            </rect>
            <text
              x={Math.min(width - 12, x(parameter.importance) + 9)}
              y={centerY + 4}
              fill={selected ? CHART.cyan : CHART.text}
              fontSize="11"
              fontWeight="650"
            >
              {(parameter.importance * 100).toFixed(1)}%
            </text>
          </g>
        );
      })}
      <text
        x={left + (width - right - left) / 2}
        y={height - 5}
        fill={CHART.muted}
        fontSize="12"
        textAnchor="middle"
      >
        Relative importance
      </text>
    </svg>
  );
}

function NumericBehaviorChart({
  study,
  parameter,
  points,
}: {
  study: OptunaStudySnapshot;
  parameter: OptunaParameterSnapshot;
  points: ParameterPoint[];
}) {
  const width = 960;
  const height = 420;
  const margin = { top: 30, right: 30, bottom: 76, left: 82 };
  const numericPoints = points.filter(
    (point): point is ParameterPoint & { numericValue: number } =>
      point.numericValue != null,
  );
  if (numericPoints.length === 0) return null;

  const axis = numericAxis(
    parameter,
    numericPoints.map((point) => point.numericValue),
  );
  const objectiveValues = numericPoints.map((point) => point.objective);
  const scale = objectiveScale(objectiveValues, study.direction, 0.12);
  const yDomain = scale.domain;
  const xTransformed = linearScale(axis.domain, [margin.left, width - margin.right]);
  const x = (value: number) => xTransformed(axis.transform(value));
  const y = linearScale(yDomain, [height - margin.bottom, margin.top]);
  const yPlot = (value: number) => y(scale.plot(value));
  const xTicks = ticks(axis.domain, 6).map(axis.invert);
  const yTicks = ticks(yDomain, 5);
  const bestThreshold =
    study.direction === "minimize"
      ? quantile(objectiveValues, 0.25)
      : quantile(objectiveValues, 0.75);
  const bestPoint = numericPoints.reduce((best, point) =>
    study.direction === "minimize"
      ? point.objective < best.objective
        ? point
        : best
      : point.objective > best.objective
        ? point
        : best,
  );
  const trend = numericGroups(numericPoints, study, parameter)
    .filter(
      (group): group is BehaviorGroup & { center: number } =>
        group.center != null,
    )
    .sort((left, right) => left.center - right.center);
  const trendPath = trend
    .map(
      (group, index) =>
        `${index === 0 ? "M" : "L"}${x(group.center)},${yPlot(group.median)}`,
    )
    .join(" ");

  return (
    <svg
      data-export-root
      viewBox={`0 0 ${width} ${height}`}
      className="block w-full"
      role="img"
      aria-label={`${parameterLabel(parameter.name)} response for ${study.displayName}`}
    >
      <title>
        {objectiveLabel(study.target)} by {parameterLabel(parameter.name)} for {study.displayName}
      </title>
      {yTicks.map((value) => (
        <g key={`y-${value}`} aria-hidden="true">
          <line
            x1={margin.left}
            x2={width - margin.right}
            y1={y(value)}
            y2={y(value)}
            stroke={CHART.grid}
          />
          <text
            x={margin.left - 12}
            y={y(value) + 4}
            fill={CHART.axis}
            fontSize="11"
            textAnchor="end"
          >
            {axisNumber(value)}
          </text>
        </g>
      ))}
      {xTicks.map((value) => (
        <g key={`x-${value}`} aria-hidden="true">
          <line
            x1={x(value)}
            x2={x(value)}
            y1={margin.top}
            y2={height - margin.bottom}
            stroke={CHART.grid}
          />
          <text
            x={x(value)}
            y={height - margin.bottom + 22}
            fill={CHART.axis}
            fontSize="11"
            textAnchor="middle"
          >
            {axisNumber(value)}
          </text>
        </g>
      ))}
      {trend.length > 1 && (
        <path
          d={trendPath}
          fill="none"
          stroke={CHART.cyan}
          strokeWidth="2.5"
          strokeDasharray="7 6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
      )}
      {numericPoints.map((point) => {
        const inBestQuartile =
          study.direction === "minimize"
            ? point.objective <= bestThreshold
            : point.objective >= bestThreshold;
        const isBest = point === bestPoint;
        const clipped = scale.isClipped(point.objective);
        const pointX = x(point.numericValue);
        const pointY = yPlot(point.objective);
        return (
          <g key={point.trial.number}>
            {isBest && (
              <circle
                cx={pointX}
                cy={pointY}
                r="10"
                fill="none"
                stroke={CHART.amber}
                strokeWidth="2"
              />
            )}
            <TrialMarker
              x={pointX}
              y={pointY}
              radius={isBest ? 6.5 : 5}
              fill={isBest ? CHART.amber : inBestQuartile ? CHART.lime : CHART.violet}
              fillOpacity={clipped ? 0.55 : inBestQuartile || isBest ? 0.94 : 0.68}
              clipSide={clipped ? scale.clipSide : null}
            >
              <title>
                Trial #{point.trial.number} · {parameterLabel(parameter.name)}{" "}
                {displayParamValue(point.numericValue, parameter)} · {objectiveLabel(study.target)}{" "}
                {formatMetric(point.objective)}
                {clipped ? " · off scale" : ""}
              </title>
            </TrialMarker>
          </g>
        );
      })}
      <ClippedNote scale={scale} x={width - margin.right} y={margin.top + 11} />
      <text
        x={margin.left + (width - margin.right - margin.left) / 2}
        y={height - 13}
        fill={CHART.muted}
        fontSize="12"
        textAnchor="middle"
      >
        {parameterLabel(parameter.name)}
        {axis.useLog ? " · log scale" : ""}
      </text>
      <text
        x="18"
        y={margin.top + (height - margin.bottom - margin.top) / 2}
        fill={CHART.muted}
        fontSize="12"
        textAnchor="middle"
        transform={`rotate(-90 18 ${margin.top + (height - margin.bottom - margin.top) / 2})`}
      >
        {objectiveLabel(study.target)}
      </text>
      <g transform={`translate(${margin.left + 8} ${margin.top + 7})`} aria-hidden="true">
        <line
          x1="0"
          x2="22"
          y1="0"
          y2="0"
          stroke={CHART.cyan}
          strokeWidth="2.5"
          strokeDasharray="6 5"
        />
        <text x="29" y="4" fill={CHART.text} fontSize="11">
          Binned median
        </text>
        <circle cx="127" cy="0" r="4" fill={CHART.lime} />
        <text x="137" y="4" fill={CHART.text} fontSize="11">
          Best quartile
        </text>
      </g>
    </svg>
  );
}

function CategoricalBehaviorChart({
  study,
  parameter,
  points,
}: {
  study: OptunaStudySnapshot;
  parameter: OptunaParameterSnapshot;
  points: ParameterPoint[];
}) {
  const height = 440;
  const margin = { top: 30, right: 30, bottom: 108, left: 82 };
  const groups = categoricalGroups(points, study, parameter);
  if (groups.length === 0) return null;

  const width = Math.max(
    960,
    margin.left + margin.right + groups.length * 86,
  );
  const categories = groups.map((group) => group.key);
  const innerWidth = width - margin.left - margin.right;
  const spacing = innerWidth / Math.max(1, categories.length);
  const xForKey = (key: string) =>
    margin.left + spacing * (categories.indexOf(key) + 0.5);
  const objectiveValues = points.map((point) => point.objective);
  const scale = objectiveScale(objectiveValues, study.direction, 0.12);
  const yDomain = scale.domain;
  const y = linearScale(yDomain, [height - margin.bottom, margin.top]);
  const yPlot = (value: number) => y(scale.plot(value));
  const yTicks = ticks(yDomain, 5);
  const bestThreshold =
    study.direction === "minimize"
      ? quantile(objectiveValues, 0.25)
      : quantile(objectiveValues, 0.75);
  const bestObjective = bestValue(objectiveValues, study.direction);

  return (
    <div
      className="overflow-x-auto pb-1"
      role="region"
      tabIndex={0}
      aria-label={`${parameterLabel(parameter.name)} response chart. Scroll horizontally to compare every configuration.`}
    >
      <svg
        data-export-root
        viewBox={`0 0 ${width} ${height}`}
        className="block h-auto w-full"
        style={{ minWidth: `${width}px` }}
        role="img"
        aria-label={`${parameterLabel(parameter.name)} response for ${study.displayName}`}
      >
        <title>
          {objectiveLabel(study.target)} by {parameterLabel(parameter.name)} for {study.displayName}
        </title>
        <desc>
          Each column is one parameter choice. The box shows the middle half of scores,
          and green points are trials in the best quarter.
        </desc>
        {yTicks.map((value) => (
          <g key={value} aria-hidden="true">
            <line
              x1={margin.left}
              x2={width - margin.right}
              y1={y(value)}
              y2={y(value)}
              stroke={CHART.grid}
            />
            <text
              x={margin.left - 12}
              y={y(value) + 4}
              fill={CHART.axis}
              fontSize="11"
              textAnchor="end"
            >
              {axisNumber(value)}
            </text>
          </g>
        ))}
        {groups.map((group) => {
          const centerX = xForKey(group.key);
          const groupPoints = points.filter((point) => valueKey(point.value) === group.key);
          const objectives = groupPoints.map((point) => point.objective);
          const firstQuartile = quantile(objectives, 0.25);
          const thirdQuartile = quantile(objectives, 0.75);
          const boxWidth = Math.min(54, spacing * 0.44);
          return (
            <g key={group.key}>
              <line
                x1={centerX}
                x2={centerX}
                y1={yPlot(group.minimum)}
                y2={yPlot(group.maximum)}
                stroke={CHART.cyan}
                strokeOpacity="0.5"
              />
              <rect
                x={centerX - boxWidth / 2}
                y={yPlot(thirdQuartile)}
                width={boxWidth}
                height={Math.max(2, yPlot(firstQuartile) - yPlot(thirdQuartile))}
                rx="6"
                fill="rgba(79,213,231,0.12)"
                stroke={CHART.cyan}
                strokeOpacity="0.72"
              />
              <line
                x1={centerX - boxWidth / 2}
                x2={centerX + boxWidth / 2}
                y1={yPlot(group.median)}
                y2={yPlot(group.median)}
                stroke={CHART.cyan}
                strokeWidth="2.5"
              />
              {groupPoints.map((point) => {
                const jitterSeed = (point.trial.number * 9301 + 49297) % 233;
                const jitter = (jitterSeed / 232 - 0.5) * Math.min(42, spacing * 0.58);
                const inBestQuartile =
                  study.direction === "minimize"
                    ? point.objective <= bestThreshold
                    : point.objective >= bestThreshold;
                const isBest = point.objective === bestObjective;
                const clipped = scale.isClipped(point.objective);
                return (
                  <TrialMarker
                    key={point.trial.number}
                    x={centerX + jitter}
                    y={yPlot(point.objective)}
                    radius={isBest ? 6.5 : 5}
                    fill={isBest ? CHART.amber : inBestQuartile ? CHART.lime : CHART.violet}
                    fillOpacity={clipped ? 0.55 : inBestQuartile || isBest ? 0.94 : 0.7}
                    clipSide={clipped ? scale.clipSide : null}
                  >
                    <title>
                      Trial #{point.trial.number} · {parameterLabel(parameter.name)}{" "}
                      {displayParamValue(point.value, parameter)} · {objectiveLabel(study.target)}{" "}
                      {formatMetric(point.objective)}
                      {clipped ? " · off scale" : ""}
                    </title>
                  </TrialMarker>
                );
              })}
              <text
                x={centerX}
                y={height - margin.bottom + 24}
                fill={CHART.axis}
                fontSize="11"
                textAnchor="middle"
              >
                <title>{group.label}</title>
                {group.chartLabelLines.map((line, index) => (
                  <tspan key={line} x={centerX} dy={index === 0 ? 0 : 15}>
                    {line}
                  </tspan>
                ))}
              </text>
            </g>
          );
        })}
        <ClippedNote scale={scale} x={width - margin.right} y={margin.top + 11} />
        <text
          x={margin.left + innerWidth / 2}
          y={height - 12}
          fill={CHART.muted}
          fontSize="12"
          textAnchor="middle"
        >
          {parameterAxisLabel(parameter)}
        </text>
        <text
          x="18"
          y={margin.top + (height - margin.bottom - margin.top) / 2}
          fill={CHART.muted}
          fontSize="12"
          textAnchor="middle"
          transform={`rotate(-90 18 ${margin.top + (height - margin.bottom - margin.top) / 2})`}
        >
          {objectiveLabel(study.target)}
        </text>
        <g transform={`translate(${margin.left + 8} ${margin.top + 7})`} aria-hidden="true">
          <rect
            x="0"
            y="-5"
            width="16"
            height="10"
            rx="3"
            fill="rgba(79,213,231,0.15)"
            stroke={CHART.cyan}
          />
          <text x="24" y="4" fill={CHART.text} fontSize="11">
            Middle 50%
          </text>
          <circle cx="125" cy="0" r="4" fill={CHART.lime} />
          <text x="135" y="4" fill={CHART.text} fontSize="11">
            Best quartile
          </text>
        </g>
      </svg>
    </div>
  );
}

function BehaviorChart({
  study,
  parameter,
  points,
}: {
  study: OptunaStudySnapshot;
  parameter: OptunaParameterSnapshot;
  points: ParameterPoint[];
}) {
  if (parameter.kind === "categorical") {
    return (
      <CategoricalBehaviorChart
        study={study}
        parameter={parameter}
        points={points}
      />
    );
  }
  return (
    <NumericBehaviorChart study={study} parameter={parameter} points={points} />
  );
}

/** Colour ramp for the fitted mean, darkest first. */
const MEAN_RAMP = ["#0d1a2b", "#164a5e", "#1f8377", "#69c774", "#dcf2a4"];
/** Colour ramp for the standard error: bright means "the model is guessing". */
const ERROR_RAMP = ["#101426", "#252559", "#453b93", "#7b67ca", "#cdbeff"];
/** Sampling resolution of the fitted surface; a quadratic needs no more. */
const SURFACE_GRID = 48;
/** Filled colour bands per surface, so `SURFACE_BANDS - 1` iso-lines. */
const SURFACE_BANDS = 12;
/**
 * A quadratic over two parameters has six terms. Eight trials would already
 * fit one, but its residual variance would rest on two degrees of freedom, so
 * it reports a near-perfect R2 and a meaningless standard error. Six spare
 * observations is the floor worth plotting; it costs 15 of 434 studies.
 */
const MINIMUM_SURFACE_TRIALS = 12;
const SURFACE_STORAGE_PREFIX = "trial-atlas:optuna-surface:";

function hexChannels(hex: string): [number, number, number] {
  const value = Number.parseInt(hex.slice(1), 16);
  return [(value >> 16) & 255, (value >> 8) & 255, value & 255];
}

/** Linear sRGB interpolation along a ramp; `position` is clamped to [0, 1]. */
function rampColor(ramp: string[], position: number): string {
  const scaled = Math.min(1, Math.max(0, position)) * (ramp.length - 1);
  const index = Math.min(ramp.length - 2, Math.floor(scaled));
  const weight = scaled - index;
  const from = hexChannels(ramp[index]);
  const to = hexChannels(ramp[index + 1]);
  const channels = from.map((channel, slot) =>
    Math.round(channel + (to[slot] - channel) * weight),
  );
  return `rgb(${channels[0]},${channels[1]},${channels[2]})`;
}

interface NumericAxis {
  useLog: boolean;
  /** Plotted domain, in transformed (log or linear) space. */
  domain: [number, number];
  transform: (value: number) => number;
  invert: (value: number) => number;
  /** Where a raw parameter value sits in [0, 1] across the plotted domain. */
  toUnit: (value: number) => number;
  /** Raw parameter value at a [0, 1] position across the plotted domain. */
  fromUnit: (unit: number) => number;
}

/**
 * Plotted domain for one numeric parameter: the configured search range when
 * the study declares one, widened to cover anything the sampler actually
 * visited, and log-transformed when the parameter was searched that way.
 */
function numericAxis(
  parameter: OptunaParameterSnapshot,
  observed: number[],
): NumericAxis {
  const useLog = parameter.log && observed.every((value) => value > 0);
  const transform = (value: number) => (useLog ? Math.log10(value) : value);
  const invert = (value: number) => (useLog ? 10 ** value : value);
  const observedMinimum = Math.min(...observed);
  const observedMaximum = Math.max(...observed);
  const low =
    parameter.low != null && (!useLog || parameter.low > 0)
      ? Math.min(parameter.low, observedMinimum)
      : observedMinimum;
  const high =
    parameter.high != null
      ? Math.max(parameter.high, observedMaximum)
      : observedMaximum;
  let domain: [number, number] = [transform(low), transform(high)];
  if (domain[0] === domain[1]) {
    const delta = Math.abs(domain[0] * 0.05) || 1;
    domain = [domain[0] - delta, domain[1] + delta];
  }
  const span = domain[1] - domain[0];
  return {
    useLog,
    domain,
    transform,
    invert,
    toUnit: (value) => (transform(value) - domain[0]) / span,
    fromUnit: (unit) => invert(domain[0] + unit * span),
  };
}

interface SurfaceMarker {
  trialNumber: number;
  /** Position in the plotted box, both axes normalized to [0, 1]. */
  x: number;
  y: number;
  best: boolean;
  /** True when the objective was winsorized before the fit saw it. */
  clipped: boolean;
  title: string;
}

/**
 * One fitted field drawn as filled contour bands.
 *
 * Both panels share this: only the sampled grid and the ramp differ, so the
 * mean and the standard error stay pixel-comparable.
 */
function ContourPlot({
  grid,
  ramp,
  markers,
  xAxis,
  yAxis,
  xParameter,
  yParameter,
  stationary,
  scaleTitle,
  ariaLabel,
}: {
  grid: SurfaceGrid;
  ramp: string[];
  markers: SurfaceMarker[];
  xAxis: NumericAxis;
  yAxis: NumericAxis;
  xParameter: OptunaParameterSnapshot;
  yParameter: OptunaParameterSnapshot;
  stationary: StationaryPoint | null;
  scaleTitle: string;
  ariaLabel: string;
}) {
  const width = 560;
  const height = 452;
  const margin = { top: 26, right: 112, bottom: 78, left: 78 };
  const box: PlotBox = {
    left: margin.left,
    top: margin.top,
    width: width - margin.left - margin.right,
    height: height - margin.top - margin.bottom,
  };
  const levels = contourLevels(grid.minimum, grid.maximum, SURFACE_BANDS);
  const layers = contourLayers(grid.values, levels, box);
  const bandColor = (band: number) => rampColor(ramp, (band + 0.5) / SURFACE_BANDS);
  const unitX = (unit: number) => box.left + unit * box.width;
  const unitY = (unit: number) => box.top + (1 - unit) * box.height;
  const span = grid.maximum - grid.minimum;
  const barX = width - margin.right + 36;
  const barWidth = 12;
  const barY = (value: number) =>
    box.top + box.height * (span > 0 ? 1 - (value - grid.minimum) / span : 0.5);
  const barTicks = span > 0 ? ticks([grid.minimum, grid.maximum], 5) : [grid.minimum];
  const unitTicks = ticks([0, 1], 5);
  const insideBox =
    stationary != null &&
    stationary.x >= 0 &&
    stationary.x <= 1 &&
    stationary.y >= 0 &&
    stationary.y <= 1;
  // Best-trial marker last so it is never buried under a neighbour.
  const orderedMarkers = [...markers].sort(
    (left, right) => Number(left.best) - Number(right.best),
  );

  return (
    <svg
      data-export-root
      viewBox={`0 0 ${width} ${height}`}
      className="block w-full"
      role="img"
      aria-label={ariaLabel}
    >
      <title>{ariaLabel}</title>
      <desc>
        Filled contours of a quadratic fitted to the completed trials, over{" "}
        {parameterLabel(xParameter.name)} horizontally and{" "}
        {parameterLabel(yParameter.name)} vertically. Squares mark the trials the
        fit was built from.
      </desc>
      <rect
        x={box.left}
        y={box.top}
        width={box.width}
        height={box.height}
        fill={bandColor(0)}
      />
      {layers.map((layer, index) => (
        <path key={layer.level} d={layer.fill} fill={bandColor(index + 1)} />
      ))}
      {layers.map((layer) => (
        <path
          key={`line-${layer.level}`}
          d={layer.line}
          fill="none"
          stroke="rgba(8,14,26,0.45)"
          strokeWidth="0.9"
        />
      ))}
      <rect
        x={box.left}
        y={box.top}
        width={box.width}
        height={box.height}
        fill="none"
        stroke={CHART.grid}
      />
      {unitTicks.map((unit) => (
        <g key={`x-${unit}`} aria-hidden="true">
          <line
            x1={unitX(unit)}
            x2={unitX(unit)}
            y1={box.top + box.height}
            y2={box.top + box.height + 6}
            stroke={CHART.axis}
          />
          <text
            x={unitX(unit)}
            y={box.top + box.height + 20}
            fill={CHART.axis}
            fontSize="11"
            textAnchor="middle"
          >
            {axisNumber(xAxis.fromUnit(unit))}
          </text>
        </g>
      ))}
      {unitTicks.map((unit) => (
        <g key={`y-${unit}`} aria-hidden="true">
          <line
            x1={box.left - 6}
            x2={box.left}
            y1={unitY(unit)}
            y2={unitY(unit)}
            stroke={CHART.axis}
          />
          <text
            x={box.left - 11}
            y={unitY(unit) + 4}
            fill={CHART.axis}
            fontSize="11"
            textAnchor="end"
          >
            {axisNumber(yAxis.fromUnit(unit))}
          </text>
        </g>
      ))}
      {insideBox && stationary && (
        <g>
          <circle
            cx={unitX(stationary.x)}
            cy={unitY(stationary.y)}
            r="9"
            fill="none"
            stroke="#ffffff"
            strokeWidth="1.6"
            strokeDasharray="3 3"
          />
          <line
            x1={unitX(stationary.x) - 5}
            x2={unitX(stationary.x) + 5}
            y1={unitY(stationary.y)}
            y2={unitY(stationary.y)}
            stroke="#ffffff"
            strokeWidth="1.6"
          />
          <line
            x1={unitX(stationary.x)}
            x2={unitX(stationary.x)}
            y1={unitY(stationary.y) - 5}
            y2={unitY(stationary.y) + 5}
            stroke="#ffffff"
            strokeWidth="1.6"
          />
          <title>
            Fitted {stationary.kind} at {parameterLabel(xParameter.name)}{" "}
            {displayParamValue(xAxis.fromUnit(stationary.x), xParameter)} ·{" "}
            {parameterLabel(yParameter.name)}{" "}
            {displayParamValue(yAxis.fromUnit(stationary.y), yParameter)}
          </title>
        </g>
      )}
      {orderedMarkers.map((marker) => {
        const size = marker.best ? 6 : 4.4;
        return (
          <rect
            key={marker.trialNumber}
            x={unitX(marker.x) - size}
            y={unitY(marker.y) - size}
            width={size * 2}
            height={size * 2}
            fill={marker.clipped ? "none" : marker.best ? CHART.amber : "#e2e9f7"}
            fillOpacity={marker.best ? 0.95 : 0.82}
            stroke={marker.best ? "#10182a" : "rgba(10,17,32,0.85)"}
            strokeWidth="1.4"
          >
            <title>{marker.title}</title>
          </rect>
        );
      })}
      {Array.from({ length: SURFACE_BANDS }, (_, band) => (
        <rect
          key={`band-${band}`}
          x={barX}
          y={box.top + (box.height * (SURFACE_BANDS - 1 - band)) / SURFACE_BANDS}
          width={barWidth}
          height={box.height / SURFACE_BANDS + 0.5}
          fill={bandColor(band)}
        />
      ))}
      <rect
        x={barX}
        y={box.top}
        width={barWidth}
        height={box.height}
        fill="none"
        stroke={CHART.grid}
      />
      {barTicks.map((value) => (
        <text
          key={`bar-${value}`}
          x={barX + barWidth + 6}
          y={barY(value) + 4}
          fill={CHART.axis}
          fontSize="10"
        >
          {formatMetric(value)}
        </text>
      ))}
      <text
        x={barX + barWidth / 2}
        y={box.top - 10}
        fill={CHART.muted}
        fontSize="10"
        textAnchor="middle"
      >
        {scaleTitle}
      </text>
      <text
        x={box.left + box.width / 2}
        y={box.top + box.height + 44}
        fill={CHART.muted}
        fontSize="12"
        textAnchor="middle"
      >
        {parameterLabel(xParameter.name)}
        {xAxis.useLog ? " · log scale" : ""}
      </text>
      <text
        x="16"
        y={box.top + box.height / 2}
        fill={CHART.muted}
        fontSize="12"
        textAnchor="middle"
        transform={`rotate(-90 16 ${box.top + box.height / 2})`}
      >
        {parameterLabel(yParameter.name)}
        {yAxis.useLog ? " · log scale" : ""}
      </text>
      <g transform={`translate(${box.left} ${box.top + box.height + 66})`} aria-hidden="true">
        <rect x="0" y="-4.5" width="9" height="9" fill="#e2e9f7" fillOpacity="0.82" />
        <text x="16" y="4" fill={CHART.text} fontSize="11">
          In-sample trial
        </text>
        <rect x="112" y="-4.5" width="9" height="9" fill={CHART.amber} />
        <text x="128" y="4" fill={CHART.text} fontSize="11">
          Best trial
        </text>
        {insideBox && (
          <>
            <circle
              cx="204"
              cy="0"
              r="5.5"
              fill="none"
              stroke="#ffffff"
              strokeWidth="1.4"
              strokeDasharray="3 3"
            />
            <text x="216" y="4" fill={CHART.text} fontSize="11">
              Fitted {stationary?.kind}
            </text>
          </>
        )}
      </g>
    </svg>
  );
}

interface FittedSurface {
  xParameter: OptunaParameterSnapshot;
  yParameter: OptunaParameterSnapshot;
  xAxis: NumericAxis;
  yAxis: NumericAxis;
  model: SurfaceModel;
  meanGrid: SurfaceGrid;
  errorGrid: SurfaceGrid;
  markers: SurfaceMarker[];
  /** Trials whose objective was winsorized before the fit saw it. */
  clippedCount: number;
}

/**
 * Fits the surface for one pair of parameters.
 *
 * The objective is winsorized on the same Tukey fence the response plots use:
 * a couple of collapsed trials would otherwise dominate a least-squares fit
 * and tilt the whole surface towards the failure, which says nothing about
 * where the optimum is.
 */
function fitSurfaceFor(
  study: OptunaStudySnapshot,
  xParameter: OptunaParameterSnapshot,
  yParameter: OptunaParameterSnapshot,
): { surface: FittedSurface | null; observed: number } {
  const rows = study.trials.flatMap((trial) => {
    const xValue = trial.params[xParameter.name];
    const yValue = trial.params[yParameter.name];
    if (typeof xValue !== "number" || !Number.isFinite(xValue)) return [];
    if (typeof yValue !== "number" || !Number.isFinite(yValue)) return [];
    if (!Number.isFinite(trial.objective)) return [];
    return [{ trial, xValue, yValue }];
  });
  if (rows.length < MINIMUM_SURFACE_TRIALS) {
    return { surface: null, observed: rows.length };
  }

  const xAxis = numericAxis(xParameter, rows.map((row) => row.xValue));
  const yAxis = numericAxis(yParameter, rows.map((row) => row.yValue));
  const scale = objectiveScale(
    rows.map((row) => row.trial.objective),
    study.direction,
  );
  const samples: SurfaceSample[] = rows.map((row) => ({
    x: xAxis.toUnit(row.xValue),
    y: yAxis.toUnit(row.yValue),
    value: scale.plot(row.trial.objective),
  }));
  const model = fitResponseSurface(samples);
  if (!model) return { surface: null, observed: rows.length };

  const bestObjective = bestValue(
    rows.map((row) => row.trial.objective),
    study.direction,
  );
  const markers: SurfaceMarker[] = rows.map((row) => {
    const clipped = scale.isClipped(row.trial.objective);
    return {
      trialNumber: row.trial.number,
      x: xAxis.toUnit(row.xValue),
      y: yAxis.toUnit(row.yValue),
      best: row.trial.objective === bestObjective,
      clipped,
      title:
        `Trial #${row.trial.number} · ${parameterLabel(xParameter.name)} ` +
        `${displayParamValue(row.xValue, xParameter)} · ${parameterLabel(yParameter.name)} ` +
        `${displayParamValue(row.yValue, yParameter)} · ${objectiveLabel(study.target)} ` +
        `${formatMetric(row.trial.objective)}${clipped ? " · winsorized for the fit" : ""}`,
    };
  });

  return {
    surface: {
      xParameter,
      yParameter,
      xAxis,
      yAxis,
      model,
      meanGrid: sampleGrid(model.predict, SURFACE_GRID, SURFACE_GRID),
      errorGrid: sampleGrid(model.standardError, SURFACE_GRID, SURFACE_GRID),
      markers,
      clippedCount: markers.filter((marker) => marker.clipped).length,
    },
    observed: rows.length,
  };
}

/** Plain-language reading of where the fitted quadratic turns over. */
function stationarySummary(
  study: OptunaStudySnapshot,
  surface: FittedSurface,
): string {
  const { model, xAxis, yAxis, xParameter, yParameter } = surface;
  const wanted = study.direction === "minimize" ? "minimum" : "maximum";
  if (model.kind === "linear") {
    return "Only a plane could be fitted, so the surface has no turning point: the trials do not pin down any curvature yet.";
  }
  const point = model.stationaryPoint;
  if (!point) return "The fit is flat enough that no turning point can be located.";
  if (point.kind === "saddle") {
    return "The fit is a saddle, not a peak: these two parameters trade off against each other rather than having one joint best setting.";
  }
  if (point.kind !== wanted) {
    return `The fit curves away from a ${wanted}, so within this box the best setting sits on an edge rather than in the middle.`;
  }
  const location =
    `${parameterLabel(xParameter.name)} ${displayParamValue(
      xAxis.fromUnit(point.x),
      xParameter,
    )} · ${parameterLabel(yParameter.name)} ${displayParamValue(
      yAxis.fromUnit(point.y),
      yParameter,
    )}`;
  return point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1
    ? `Predicted ${wanted} at ${location}.`
    : `Predicted ${wanted} falls outside the searched box, at ${location}.`;
}

/** Stored axis pair, defensively parsed: localStorage can hold anything. */
function readSurfacePair(payload: unknown): [string, string] | null {
  if (!Array.isArray(payload) || payload.length !== 2) return null;
  const [x, y] = payload;
  if (typeof x !== "string" || typeof y !== "string") return null;
  return [x, y];
}

/**
 * Two-parameter view of the search space.
 *
 * fANOVA importance ranks parameters one at a time, which cannot show a pair
 * that only works at the right ratio. Fitting a second-order surface over two
 * parameters recovers that: the mean panel is where the model expects the
 * objective to be, and the standard-error panel is where it is extrapolating
 * because no trial was ever sampled nearby.
 */
function ResponseSurfaceSection({ study }: { study: OptunaStudySnapshot }) {
  const numericParameters = useMemo(
    () => study.parameters.filter((parameter) => parameter.kind === "numeric"),
    [study],
  );
  const [selection, setSelection] = useState<{ x: string; y: string } | null>(null);

  useEffect(() => {
    let stored: [string, string] | null = null;
    try {
      const raw = window.localStorage.getItem(`${SURFACE_STORAGE_PREFIX}${study.id}`);
      stored = raw ? readSurfacePair(JSON.parse(raw) as unknown) : null;
    } catch {
      stored = null;
    }
    setSelection(stored ? { x: stored[0], y: stored[1] } : null);
  }, [study.id]);

  // Resolved during render rather than stored, so a selection left over from
  // the previous study never reaches the fit: the two most important numeric
  // parameters are the fallback, and the axes can never collapse onto one.
  const resolved = useMemo(() => {
    if (numericParameters.length < 2) return null;
    const x =
      numericParameters.find((parameter) => parameter.name === selection?.x) ??
      numericParameters[0];
    const candidate = numericParameters.find(
      (parameter) => parameter.name === selection?.y,
    );
    const y =
      candidate && candidate.name !== x.name
        ? candidate
        : (numericParameters.find((parameter) => parameter.name !== x.name) ??
          numericParameters[1]);
    return { x, y };
  }, [numericParameters, selection]);

  const fitted = useMemo(
    () => (resolved ? fitSurfaceFor(study, resolved.x, resolved.y) : null),
    [resolved, study],
  );

  if (!resolved) return null;

  const chooseAxis = (axis: "x" | "y", name: string) => {
    // Picking the other axis' parameter swaps them instead of collapsing the
    // surface onto a line, which has no identifiable fit.
    const next =
      axis === "x"
        ? { x: name, y: resolved.y.name === name ? resolved.x.name : resolved.y.name }
        : { x: resolved.x.name === name ? resolved.y.name : resolved.x.name, y: name };
    setSelection(next);
    try {
      window.localStorage.setItem(
        `${SURFACE_STORAGE_PREFIX}${study.id}`,
        JSON.stringify([next.x, next.y]),
      );
    } catch {
      // The pair remains selected for the current session.
    }
  };

  const surface = fitted?.surface ?? null;
  const axisSelect = (axis: "x" | "y") => (
    <label className="block">
      <span className="mb-2 block text-[10px] font-semibold uppercase tracking-[0.12em] text-[#79869e]">
        {axis === "x" ? "Horizontal parameter" : "Vertical parameter"}
      </span>
      <select
        value={axis === "x" ? resolved.x.name : resolved.y.name}
        onChange={(event) => chooseAxis(axis, event.target.value)}
        className={inputClass}
      >
        {numericParameters.map((parameter) => (
          <option key={parameter.name} value={parameter.name}>
            #{parameter.rank} · {parameterLabel(parameter.name)} ·{" "}
            {(parameter.importance * 100).toFixed(1)}%
          </option>
        ))}
      </select>
    </label>
  );

  return (
    <>
      <section className="surface px-5 py-5 sm:px-6">
        <div className="flex flex-col gap-5 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0">
            <h2 className="text-[15px] font-semibold text-white">
              Two-parameter response surface
            </h2>
            <p className="mt-1.5 max-w-3xl text-xs leading-5 text-[#8290a7]">
              A second-order fit over both parameters at once, which importance
              rankings cannot show. The mean panel is the objective the model
              expects; the standard-error panel is where it is extrapolating.
            </p>
            {surface ? (
              <>
                <div className="mt-3 flex flex-wrap items-center gap-2">
                  {[
                    surface.model.kind === "quadratic" ? "Quadratic fit" : "Plane fit",
                    `${surface.model.observations} trials`,
                    `R² ${surface.model.rSquared.toFixed(2)}`,
                    `σ ${formatMetric(surface.model.sigma)}`,
                    ...(surface.clippedCount > 0
                      ? [
                          `${surface.clippedCount} outlier${
                            surface.clippedCount === 1 ? "" : "s"
                          } winsorized`,
                        ]
                      : []),
                  ].map((chip) => (
                    <span
                      key={chip}
                      className="rounded-full border border-white/[0.08] bg-white/[0.035] px-2.5 py-1 text-[10px] text-[#8996ab]"
                    >
                      {chip}
                    </span>
                  ))}
                </div>
                <p className="mt-3 max-w-3xl text-xs leading-5 text-[#a7b2c6]">
                  {stationarySummary(study, surface)}
                </p>
              </>
            ) : (
              <p className="mt-3 max-w-3xl text-xs leading-5 text-[#f4c86f]">
                {(fitted?.observed ?? 0) < MINIMUM_SURFACE_TRIALS
                  ? `Only ${fitted?.observed ?? 0} trials varied both parameters; a two-parameter fit needs at least ${MINIMUM_SURFACE_TRIALS}.`
                  : "These trials do not pin down a surface: the sampled points are collinear, so no fit is identifiable."}
              </p>
            )}
          </div>
          <div className="grid w-full shrink-0 gap-3 sm:grid-cols-2 xl:w-[520px]">
            {axisSelect("x")}
            {axisSelect("y")}
          </div>
        </div>
      </section>

      {surface && (
        <div className="grid gap-6 2xl:grid-cols-2">
          <DiagramCard
            title="Predicted mean"
            description="Where the fitted surface expects the objective to land. Squares are the trials it was fitted to."
            exportName={`${slugify(study.displayName)}-${slugify(
              `${resolved.x.name}-${resolved.y.name}`,
            )}-surface-mean`}
          >
            <ContourPlot
              grid={surface.meanGrid}
              ramp={MEAN_RAMP}
              markers={surface.markers}
              xAxis={surface.xAxis}
              yAxis={surface.yAxis}
              xParameter={surface.xParameter}
              yParameter={surface.yParameter}
              stationary={surface.model.stationaryPoint}
              scaleTitle="Mean"
              ariaLabel={`Predicted ${objectiveLabel(study.target)} over ${parameterLabel(
                surface.xParameter.name,
              )} and ${parameterLabel(surface.yParameter.name)} for ${study.displayName}`}
            />
          </DiagramCard>
          <DiagramCard
            title="Standard error"
            description="Uncertainty of that prediction. Bright regions were never sampled, so the mean there is extrapolation."
            exportName={`${slugify(study.displayName)}-${slugify(
              `${resolved.x.name}-${resolved.y.name}`,
            )}-surface-error`}
          >
            <ContourPlot
              grid={surface.errorGrid}
              ramp={ERROR_RAMP}
              markers={surface.markers}
              xAxis={surface.xAxis}
              yAxis={surface.yAxis}
              xParameter={surface.xParameter}
              yParameter={surface.yParameter}
              stationary={null}
              scaleTitle="Std. error"
              ariaLabel={`Standard error of the predicted ${objectiveLabel(
                study.target,
              )} over ${parameterLabel(surface.xParameter.name)} and ${parameterLabel(
                surface.yParameter.name,
              )} for ${study.displayName}`}
            />
          </DiagramCard>
        </div>
      )}
    </>
  );
}

function BehaviorSummaryTable({
  study,
  parameter,
  points,
}: {
  study: OptunaStudySnapshot;
  parameter: OptunaParameterSnapshot;
  points: ParameterPoint[];
}) {
  const groups =
    parameter.kind === "categorical"
      ? categoricalGroups(points, study, parameter)
      : numericGroups(points, study, parameter);
  const sortedGroups =
    parameter.kind === "categorical"
      ? groups
      : [...groups].sort((left, right) => (left.center ?? 0) - (right.center ?? 0));
  const bestGroup = sortedGroups.reduce<BehaviorGroup | null>((best, group) => {
    if (!best) return group;
    return study.direction === "minimize"
      ? group.median < best.median
        ? group
        : best
      : group.median > best.median
        ? group
        : best;
  }, null);

  return (
    <section className="surface overflow-hidden">
      <header className="flex flex-wrap items-start justify-between gap-4 border-b border-white/[0.07] px-5 py-4 sm:px-6">
        <div>
          <h2 className="text-[15px] font-semibold text-white">Response summary</h2>
          <p className="mt-1.5 text-xs text-[#8290a7]">
            {parameter.kind === "categorical"
              ? "Observed objective distribution for every categorical choice."
              : "Observed trials grouped across the search range; medians reduce outlier noise."}
          </p>
        </div>
        <span className="rounded-full border border-[#4fd5e7]/20 bg-[#4fd5e7]/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#70ddeb]">
          {study.direction === "minimize" ? "Lower is better" : "Higher is better"}
        </span>
      </header>
      <div className="overflow-x-auto">
        <table className="w-full min-w-[680px] text-left text-xs">
          <thead className="border-b border-white/[0.07] bg-black/10 text-[10px] uppercase tracking-[0.11em] text-[#6f7d95]">
            <tr>
              <th className="px-6 py-3.5 font-semibold">
                {parameter.name === "__joint_hparam__.two_stream_batch_sampler"
                  ? "Sampler / labeled batch"
                  : parameter.kind === "categorical"
                    ? "Choice"
                    : "Search interval"}
              </th>
              <th className="px-4 py-3.5 text-right font-semibold">Trials</th>
              <th className="px-4 py-3.5 text-right font-semibold">Median objective</th>
              <th className="px-4 py-3.5 text-right font-semibold">Best objective</th>
              <th className="px-6 py-3.5 text-right font-semibold">Observed range</th>
            </tr>
          </thead>
          <tbody>
            {sortedGroups.map((group) => {
              const leading = group === bestGroup;
              return (
                <tr
                  key={group.key}
                  className={`border-b border-white/[0.055] last:border-0 ${
                    leading ? "bg-[#b8e56e]/[0.045]" : "hover:bg-white/[0.02]"
                  }`}
                >
                  <td className="px-6 py-4 font-semibold text-white">
                    <span className="flex flex-wrap items-center gap-2">
                      {group.label}
                      {leading && (
                        <span className="rounded-full bg-[#b8e56e]/10 px-2 py-0.5 text-[9px] uppercase tracking-[0.08em] text-[#c7ee86]">
                          Leading median
                        </span>
                      )}
                    </span>
                  </td>
                  <td className="px-4 py-4 text-right tabular-nums text-[#aeb8ca]">
                    {group.count}
                  </td>
                  <td className="px-4 py-4 text-right font-semibold tabular-nums text-[#74ddea]">
                    {formatMetric(group.median)}
                  </td>
                  <td className="px-4 py-4 text-right font-semibold tabular-nums text-[#c9f18a]">
                    {formatMetric(group.best)}
                  </td>
                  <td className="px-6 py-4 text-right tabular-nums text-[#7f8ca3]">
                    {formatMetric(group.minimum)} – {formatMetric(group.maximum)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

/**
 * Hidden study ids, defensively parsed: the payload comes back from
 * localStorage, so a stale or hand-edited entry must not break the page.
 */
function readHiddenStudies(payload: unknown): string[] {
  if (!Array.isArray(payload)) return [];
  return [
    ...new Set(payload.filter((value): value is string => typeof value === "string")),
  ];
}

/** Empty input means "unbounded"; anything unparseable is treated the same. */
function parseBound(value: string): number | null {
  if (value.trim() === "") return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function LoadingState() {
  return (
    <div className="flex min-h-[66vh] items-center justify-center">
      <div className="surface w-full max-w-md px-8 py-9 text-center">
        <span className="mx-auto block h-7 w-7 animate-spin rounded-full border-2 border-[#9b8cff]/25 border-t-[#9b8cff]" />
        <p className="mt-4 text-sm font-semibold text-white">Loading Optuna studies</p>
        <p className="mt-2 text-xs leading-5 text-[#7f8ca2]">
          Preparing parameter importances and trial responses.
        </p>
      </div>
    </div>
  );
}

export function OptunaExplorer() {
  const [snapshot, setSnapshot] = useState<OptunaSnapshot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [reloadKey, setReloadKey] = useState(0);
  const [selectedStudyId, setSelectedStudyId] = useState<string | null>(null);
  const [selectedParameterName, setSelectedParameterName] = useState<string | null>(
    null,
  );
  const [query, setQuery] = useState("");
  const [collection, setCollection] = useState("all");
  const [hiddenStudyIds, setHiddenStudyIds] = useState<string[]>([]);
  const [showHidden, setShowHidden] = useState(false);
  const [minTrials, setMinTrials] = useState("");
  const [maxTrials, setMaxTrials] = useState("");
  const [storageReady, setStorageReady] = useState(false);

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(HIDDEN_STUDIES_STORAGE_KEY);
      const parsed: unknown = raw ? JSON.parse(raw) : null;
      setHiddenStudyIds(readHiddenStudies(parsed));
    } catch {
      setHiddenStudyIds([]);
    }
    setStorageReady(true);
  }, []);

  useEffect(() => {
    if (!storageReady) return;
    try {
      window.localStorage.setItem(
        HIDDEN_STUDIES_STORAGE_KEY,
        JSON.stringify(hiddenStudyIds),
      );
    } catch {
      // Hidden studies stay hidden for the current session.
    }
  }, [hiddenStudyIds, storageReady]);

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
        if (!isOptunaSnapshot(payload)) throw new Error("The snapshot format is not supported.");
        return payload;
      })
      .then((payload) => {
        setSnapshot(payload);
        let storedStudyId: string | null = null;
        try {
          storedStudyId = window.localStorage.getItem(STUDY_STORAGE_KEY);
        } catch {
          storedStudyId = null;
        }
        const initialStudy =
          payload.studies.find((study) => study.id === storedStudyId) ??
          payload.studies[0] ??
          null;
        setSelectedStudyId((current) =>
          payload.studies.some((study) => study.id === current)
            ? current
            : initialStudy?.id ?? null,
        );
      })
      .catch((loadError: unknown) => {
        if (controller.signal.aborted) return;
        setError(
          loadError instanceof Error
            ? loadError.message
            : "The Optuna snapshot could not be loaded.",
        );
      });
    return () => controller.abort();
  }, [reloadKey]);

  const collections = useMemo(
    () =>
      snapshot
        ? [...new Set(snapshot.studies.map((study) => study.collection))].sort()
        : [],
    [snapshot],
  );
  const hiddenSet = useMemo(() => new Set(hiddenStudyIds), [hiddenStudyIds]);
  // Only studies actually present in this snapshot count towards the badge, so
  // stale ids left over from an earlier sync do not inflate it.
  const hiddenCount = useMemo(
    () =>
      snapshot
        ? snapshot.studies.filter((study) => hiddenSet.has(study.id)).length
        : 0,
    [hiddenSet, snapshot],
  );
  const trialBounds = useMemo(() => {
    const counts = snapshot?.studies.map((study) => study.completeTrials) ?? [];
    return counts.length > 0
      ? { minimum: Math.min(...counts), maximum: Math.max(...counts) }
      : { minimum: 0, maximum: 0 };
  }, [snapshot]);
  const filteredStudies = useMemo(() => {
    if (!snapshot) return [];
    const normalizedQuery = query.trim().toLowerCase();
    const lowerBound = parseBound(minTrials);
    const upperBound = parseBound(maxTrials);
    return snapshot.studies.filter((study) => {
      if (hiddenSet.has(study.id) && !showHidden) return false;
      if (collection !== "all" && study.collection !== collection) return false;
      if (lowerBound != null && study.completeTrials < lowerBound) return false;
      if (upperBound != null && study.completeTrials > upperBound) return false;
      if (!normalizedQuery) return true;
      return [
        study.displayName,
        study.relativePath,
        study.dataset,
        study.method,
        study.loss,
        study.labelSetting,
        study.studyName,
      ].some((value) => value.toLowerCase().includes(normalizedQuery));
    });
  }, [
    collection,
    hiddenSet,
    maxTrials,
    minTrials,
    query,
    showHidden,
    snapshot,
  ]);
  const selectedStudy =
    snapshot?.studies.find((study) => study.id === selectedStudyId) ?? null;

  // Hiding the inspected study moves the view onto the next visible one instead
  // of leaving a detail pane for a study that is no longer in the catalog. The
  // search and range filters deliberately do not move the selection.
  useEffect(() => {
    if (!selectedStudyId || showHidden) return;
    if (!hiddenSet.has(selectedStudyId)) return;
    setSelectedStudyId(filteredStudies[0]?.id ?? null);
  }, [filteredStudies, hiddenSet, selectedStudyId, showHidden]);

  const setStudyHidden = (studyId: string, hidden: boolean) => {
    setHiddenStudyIds((current) =>
      hidden
        ? [...new Set([...current, studyId])]
        : current.filter((id) => id !== studyId),
    );
  };

  const showAllStudies = () => {
    if (!snapshot) return;
    const visible = new Set(snapshot.studies.map((study) => study.id));
    setHiddenStudyIds((current) => current.filter((id) => !visible.has(id)));
    setShowHidden(false);
  };

  useEffect(() => {
    if (!selectedStudy) return;
    let storedParameter: string | null = null;
    try {
      storedParameter = window.localStorage.getItem(
        `${PARAMETER_STORAGE_PREFIX}${selectedStudy.id}`,
      );
    } catch {
      storedParameter = null;
    }
    setSelectedParameterName((current) => {
      const candidate =
        current && selectedStudy.parameters.some((parameter) => parameter.name === current)
          ? current
          : storedParameter;
      return selectedStudy.parameters.some((parameter) => parameter.name === candidate)
        ? candidate
        : selectedStudy.parameters[0]?.name ?? null;
    });
    try {
      window.localStorage.setItem(STUDY_STORAGE_KEY, selectedStudy.id);
    } catch {
      // The selection remains available for the current session.
    }
  }, [selectedStudy]);

  const selectedParameter =
    selectedStudy?.parameters.find(
      (parameter) => parameter.name === selectedParameterName,
    ) ?? selectedStudy?.parameters[0] ?? null;
  const parameterPoints = useMemo(
    () =>
      selectedStudy && selectedParameter
        ? pointsFor(selectedStudy, selectedParameter)
        : [],
    [selectedParameter, selectedStudy],
  );

  const chooseParameter = (name: string) => {
    setSelectedParameterName(name);
    if (!selectedStudy) return;
    try {
      window.localStorage.setItem(
        `${PARAMETER_STORAGE_PREFIX}${selectedStudy.id}`,
        name,
      );
    } catch {
      // The selection remains available for the current session.
    }
  };

  if (!snapshot && !error) return <LoadingState />;

  if (error) {
    return (
      <div className="flex min-h-[66vh] items-center justify-center">
        <div className="surface w-full max-w-xl px-8 py-9 text-center">
          <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-rose-400/10 text-rose-300 ring-1 ring-rose-300/15">
            <Activity size={22} />
          </span>
          <h1 className="mt-5 text-xl font-semibold text-white">
            Optuna snapshot unavailable
          </h1>
          <p className="mt-2 text-sm leading-6 text-[#8794aa]">{error}</p>
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
      <div className="flex min-h-[66vh] items-center justify-center">
        <div className="surface w-full max-w-xl px-8 py-9 text-center">
          <Database
            size={28}
            className="mx-auto text-[#9b8cff]"
            aria-hidden="true"
          />
          <h1 className="mt-4 text-xl font-semibold text-white">
            No analyzable studies found
          </h1>
          <p className="mt-2 text-sm leading-6 text-[#8794aa]">
            {snapshot?.warning ??
              "The parameter-importance snapshot does not contain any completed studies yet."}
          </p>
        </div>
      </div>
    );
  }

  const totalTrials = snapshot.studies.reduce(
    (total, study) => total + study.completeTrials,
    0,
  );
  const leadingParameter = selectedStudy?.parameters[0] ?? null;
  const bestTrialWithParameter =
    selectedStudy && selectedParameter
      ? parameterPoints.reduce<ParameterPoint | null>((best, point) => {
          if (!best) return point;
          return selectedStudy.direction === "minimize"
            ? point.objective < best.objective
              ? point
              : best
            : point.objective > best.objective
              ? point
              : best;
        }, null)
      : null;
  const relationship =
    selectedParameter && parameterPoints.length > 0
      ? relationshipSummary(parameterPoints, selectedParameter)
      : { value: "—", context: "No plottable trials" };
  const snapshotDate = new Date(snapshot.generatedAt);
  const snapshotLabel = Number.isNaN(snapshotDate.getTime())
    ? "Snapshot ready"
    : `Synced ${snapshotDate.toLocaleString("en-GB", {
        dateStyle: "medium",
        timeStyle: "short",
      })}`;

  return (
    <div className="fade-up space-y-6">
      <div className="flex flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
        <div>
          <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.17em] text-[#78869e]">
            <Sparkles size={12} className="text-[#b8e56e]" /> Optuna parameter lab
          </div>
          <h1 className="mt-2 text-3xl font-semibold tracking-[-0.04em] text-white sm:text-4xl">
            See what moves the objective.
          </h1>
          <p className="mt-2 max-w-3xl text-sm leading-6 text-[#8794aa]">
            Ranked fANOVA importance plus trial-level response plots for every
            analyzable study under optuna-studies.
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
            <RefreshCw size={14} /> Reload snapshot
          </button>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Analyzable studies"
          value={snapshot.studies.length.toLocaleString()}
          context={`${snapshot.skippedStudyDirectories} skipped without importances`}
          icon={<Layers3 size={17} />}
          accent="violet"
        />
        <MetricCard
          label="Completed trials"
          value={totalTrials.toLocaleString()}
          context="Finite objectives in the snapshot"
          icon={<Database size={17} />}
          accent="cyan"
        />
        <MetricCard
          label="Selected best"
          value={formatMetric(selectedStudy?.bestTrial?.objective)}
          context={
            selectedStudy?.bestTrial
              ? `Trial #${selectedStudy.bestTrial.number} · ${objectiveLabel(selectedStudy.target)}`
              : "Choose a study"
          }
          icon={<Trophy size={17} />}
          accent="lime"
        />
        <MetricCard
          label="Leading parameter"
          value={
            leadingParameter
              ? `${(leadingParameter.importance * 100).toFixed(1)}%`
              : "—"
          }
          context={leadingParameter ? parameterLabel(leadingParameter.name) : "Choose a study"}
          icon={<Gauge size={17} />}
          accent="amber"
        />
      </div>

      <section className="surface grid min-h-[430px] overflow-hidden xl:grid-cols-[360px_minmax(0,1fr)]">
        <div className="flex min-h-0 flex-col border-b border-white/[0.07] xl:border-b-0 xl:border-r">
          <header className="space-y-3 border-b border-white/[0.07] p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h2 className="text-sm font-semibold text-white">Study catalog</h2>
                <p className="mt-1 text-[10px] text-[#718098]">
                  {filteredStudies.length} of {snapshot.studies.length} studies
                  {hiddenCount > 0 ? ` · ${hiddenCount} hidden` : ""}
                </p>
              </div>
              <span className="flex h-8 w-8 items-center justify-center rounded-xl bg-[#9b8cff]/10 text-[#b6acff]">
                <BarChart3 size={15} />
              </span>
            </div>
            <label className="relative block">
              <Search
                size={14}
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#5f6e87]"
              />
              <span className="sr-only">Search Optuna studies</span>
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Method, loss, dataset, path…"
                className={`${inputClass} pl-9`}
              />
            </label>
            <label className="block">
              <span className="sr-only">Filter study collection</span>
              <select
                value={collection}
                onChange={(event) => setCollection(event.target.value)}
                className={inputClass}
              >
                <option value="all">All collections</option>
                {collections.map((value) => (
                  <option key={value} value={value}>
                    {value}
                  </option>
                ))}
              </select>
            </label>
            <div>
              <span className="mb-1.5 block text-[10px] font-semibold uppercase tracking-[0.12em] text-[#79869e]">
                Completed trials
              </span>
              <div className="flex items-center gap-2">
                <label className="min-w-0 flex-1">
                  <span className="sr-only">Minimum completed trials</span>
                  <input
                    type="number"
                    inputMode="numeric"
                    min={0}
                    value={minTrials}
                    onChange={(event) => setMinTrials(event.target.value)}
                    placeholder={`${trialBounds.minimum}`}
                    className={inputClass}
                  />
                </label>
                <span className="text-xs text-[#5f6e87]" aria-hidden="true">
                  –
                </span>
                <label className="min-w-0 flex-1">
                  <span className="sr-only">Maximum completed trials</span>
                  <input
                    type="number"
                    inputMode="numeric"
                    min={0}
                    value={maxTrials}
                    onChange={(event) => setMaxTrials(event.target.value)}
                    placeholder={`${trialBounds.maximum}`}
                    className={inputClass}
                  />
                </label>
                {(minTrials !== "" || maxTrials !== "") && (
                  <button
                    type="button"
                    onClick={() => {
                      setMinTrials("");
                      setMaxTrials("");
                    }}
                    className="focus-ring inline-flex h-8 shrink-0 items-center rounded-lg border border-white/[0.09] bg-white/[0.04] px-2 text-[10px] font-semibold text-[#c3ccdc] transition hover:bg-white/[0.08] hover:text-white"
                  >
                    Clear
                  </button>
                )}
              </div>
            </div>
            {hiddenCount > 0 && (
              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowHidden((current) => !current)}
                  aria-pressed={showHidden}
                  className={`focus-ring inline-flex h-8 flex-1 items-center justify-center gap-1.5 rounded-lg border px-2.5 text-[11px] font-semibold transition ${
                    showHidden
                      ? "border-[#f4c86f]/25 bg-[#f4c86f]/10 text-[#f4c86f]"
                      : "border-white/[0.09] bg-white/[0.04] text-[#c3ccdc] hover:bg-white/[0.08] hover:text-white"
                  }`}
                >
                  {showHidden ? <Eye size={12} /> : <EyeOff size={12} />}
                  {showHidden ? "Hiding off" : `Show ${hiddenCount} hidden`}
                </button>
                <button
                  type="button"
                  onClick={showAllStudies}
                  className="focus-ring inline-flex h-8 items-center gap-1.5 rounded-lg border border-white/[0.09] bg-white/[0.04] px-2.5 text-[11px] font-semibold text-[#c3ccdc] transition hover:bg-white/[0.08] hover:text-white"
                >
                  Unhide all
                </button>
              </div>
            )}
          </header>
          <div
            className="max-h-[420px] flex-1 space-y-1 overflow-y-auto p-2 xl:max-h-[560px]"
            role="listbox"
            aria-label="Optuna studies"
          >
            {filteredStudies.length === 0 ? (
              <p className="px-4 py-8 text-center text-xs leading-5 text-[#6f7d95]">
                No studies match this filter.
              </p>
            ) : (
              filteredStudies.map((study) => {
                const selected = study.id === selectedStudy?.id;
                const hidden = hiddenSet.has(study.id);
                return (
                  <div key={study.id} role="none" className="group relative">
                    <button
                      type="button"
                      role="option"
                      aria-selected={selected}
                      onClick={() => setSelectedStudyId(study.id)}
                      className={`focus-ring w-full rounded-xl py-3 pl-3.5 pr-11 text-left transition ${
                        selected
                          ? "bg-[#9b8cff]/[0.11] text-white ring-1 ring-inset ring-[#9b8cff]/20"
                          : "text-[#94a0b5] hover:bg-white/[0.035] hover:text-white"
                      } ${hidden ? "opacity-45" : ""}`}
                    >
                      <span className="flex items-start justify-between gap-3">
                        <span className="min-w-0">
                          <span className="block truncate text-xs font-semibold">
                            {study.displayName}
                          </span>
                          <span className="mt-1.5 block truncate text-[10px] text-[#68768e]">
                            {study.dataset} · {study.completeTrials} trials ·{" "}
                            {study.parameters.length} params
                          </span>
                        </span>
                        <span
                          className={`mt-0.5 h-2 w-2 shrink-0 rounded-full ${
                            study.collection === "Current"
                              ? "bg-[#b8e56e]"
                              : "bg-[#65748c]"
                          }`}
                          aria-hidden="true"
                        />
                      </span>
                    </button>
                    <button
                      type="button"
                      onClick={() => setStudyHidden(study.id, !hidden)}
                      aria-label={`${hidden ? "Unhide" : "Hide"} ${study.displayName}`}
                      title={hidden ? "Unhide this study" : "Hide this study"}
                      className={`focus-ring absolute right-2 top-1/2 flex h-7 w-7 -translate-y-1/2 items-center justify-center rounded-lg text-[#8290a7] transition hover:bg-white/[0.08] hover:text-white focus-visible:opacity-100 group-hover:opacity-100 ${
                        hidden ? "text-[#f4c86f] opacity-100" : "opacity-0"
                      }`}
                    >
                      {hidden ? <Eye size={13} /> : <EyeOff size={13} />}
                    </button>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {selectedStudy && selectedParameter ? (
          <div className="min-w-0 p-5 sm:p-6">
            <div className="flex flex-col gap-5 2xl:flex-row 2xl:items-start 2xl:justify-between">
              <div className="min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  {[selectedStudy.collection, selectedStudy.dataset, selectedStudy.sampler].map(
                    (label) => (
                      <span
                        key={label}
                        className="rounded-full border border-white/[0.08] bg-white/[0.035] px-2.5 py-1 text-[10px] text-[#8996ab]"
                      >
                        {label}
                      </span>
                    ),
                  )}
                </div>
                <h2 className="mt-3 text-2xl font-semibold tracking-[-0.035em] text-white">
                  {selectedStudy.displayName}
                </h2>
                <p
                  className="mt-2 max-w-4xl break-words font-mono text-[10px] leading-5 text-[#65738b]"
                  title={selectedStudy.relativePath}
                >
                  {selectedStudy.relativePath}
                </p>
              </div>
              <div className="w-full shrink-0 2xl:w-[360px]">
                <label className="block">
                  <span className="mb-2 block text-[10px] font-semibold uppercase tracking-[0.12em] text-[#79869e]">
                    Inspect parameter
                  </span>
                  <select
                    value={selectedParameter.name}
                    onChange={(event) => chooseParameter(event.target.value)}
                    className={inputClass}
                  >
                    {selectedStudy.parameters.map((parameter) => (
                      <option key={parameter.name} value={parameter.name}>
                        #{parameter.rank} · {parameterLabel(parameter.name)} ·{" "}
                        {(parameter.importance * 100).toFixed(1)}%
                      </option>
                    ))}
                  </select>
                </label>
              </div>
            </div>

            <div className="mt-6 grid gap-3 sm:grid-cols-2 2xl:grid-cols-4">
              <div className="surface-soft px-4 py-3.5">
                <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-[#6d7b93]">
                  Objective
                </p>
                <p className="mt-2 truncate text-sm font-semibold text-white" title={selectedStudy.target}>
                  {objectiveLabel(selectedStudy.target)}
                </p>
              </div>
              <div className="surface-soft px-4 py-3.5">
                <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-[#6d7b93]">
                  Direction
                </p>
                <p className="mt-2 text-sm font-semibold capitalize text-[#c9f18a]">
                  {selectedStudy.direction}
                </p>
              </div>
              <div className="surface-soft px-4 py-3.5">
                <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-[#6d7b93]">
                  Search scale
                </p>
                <p className="mt-2 text-sm font-semibold capitalize text-white">
                  {selectedParameter.kind === "categorical"
                    ? `${selectedParameter.choices?.length ?? 0} choices`
                    : selectedParameter.log
                      ? "Logarithmic"
                      : "Linear"}
                </p>
              </div>
              <div className="surface-soft px-4 py-3.5">
                <p className="text-[9px] font-semibold uppercase tracking-[0.12em] text-[#6d7b93]">
                  Trial coverage
                </p>
                <p className="mt-2 text-sm font-semibold text-[#70ddeb]">
                  {parameterPoints.length} / {selectedStudy.completeTrials}
                </p>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex items-center justify-center p-8 text-sm text-[#7f8ca2]">
            Choose a study to inspect its parameter behavior.
          </div>
        )}
      </section>

      {selectedStudy && selectedParameter && (
        <>
          <div className="grid gap-6 2xl:grid-cols-[minmax(0,0.82fr)_minmax(0,1.3fr)]">
            <DiagramCard
              title="Parameter importance"
              description="Normalized fANOVA contribution. Select a bar to update the response view."
              exportName={`${slugify(selectedStudy.displayName)}-parameter-importance`}
            >
              <ImportanceChart
                study={selectedStudy}
                selectedParameter={selectedParameter.name}
                onSelect={chooseParameter}
              />
            </DiagramCard>
            <DiagramCard
              title={`${parameterLabel(selectedParameter.name)} response`}
              description={
                selectedParameter.name === "__joint_hparam__.two_stream_batch_sampler"
                  ? "Each column is one sampler / labeled-batch combination. Boxes show the middle half of scores; green dots mark the best quarter."
                  : selectedParameter.kind === "categorical"
                    ? "Every completed trial by choice. Boxes show the middle half of scores; green dots mark the best quarter."
                  : "Every completed trial across the sampled range, with a binned median trend."
              }

              exportName={`${slugify(selectedStudy.displayName)}-${slugify(
                selectedParameter.name,
              )}-response`}
            >
              <BehaviorChart
                study={selectedStudy}
                parameter={selectedParameter}
                points={parameterPoints}
              />
            </DiagramCard>
          </div>

          <ResponseSurfaceSection study={selectedStudy} />

          <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
            <MetricCard
              label="fANOVA importance"
              value={`${(selectedParameter.importance * 100).toFixed(1)}%`}
              context={`Rank #${selectedParameter.rank} of ${selectedStudy.parameters.length}`}
              icon={<BarChart3 size={17} />}
              accent="violet"
            />
            <MetricCard
              label="Best observed setting"
              value={displayParamValue(bestTrialWithParameter?.value, selectedParameter)}
              context={
                bestTrialWithParameter
                  ? `Trial #${bestTrialWithParameter.trial.number} · objective ${formatMetric(
                      bestTrialWithParameter.objective,
                    )}`
                  : "No plottable trials"
              }
              icon={<Target size={17} />}
              accent="lime"
            />
            <MetricCard
              label={selectedParameter.kind === "numeric" ? "Rank association" : "Choice effect"}
              value={relationship.value}
              context={relationship.context}
              icon={<Activity size={17} />}
              accent="cyan"
            />
            <MetricCard
              label="Search distribution"
              value={
                selectedParameter.kind === "categorical"
                  ? "Categorical"
                  : selectedParameter.distribution === "int"
                    ? "Integer"
                    : "Continuous"
              }
              context={
                selectedParameter.kind === "numeric" &&
                selectedParameter.low != null &&
                selectedParameter.high != null
                  ? `${displayParamValue(
                      selectedParameter.low,
                      selectedParameter,
                    )} to ${displayParamValue(
                      selectedParameter.high,
                      selectedParameter,
                    )}${selectedParameter.log ? " · log" : ""}`
                  : `${parameterPoints.length} observed values`
              }
              icon={<SlidersHorizontal size={17} />}
              accent="amber"
            />
          </div>

          <BehaviorSummaryTable
            study={selectedStudy}
            parameter={selectedParameter}
            points={parameterPoints}
          />
        </>
      )}
    </div>
  );
}
