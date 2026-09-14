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

  const useLog =
    parameter.log && numericPoints.every((point) => point.numericValue > 0);
  const transform = (value: number) => (useLog ? Math.log10(value) : value);
  const invert = (value: number) => (useLog ? 10 ** value : value);
  const observedMinimum = Math.min(...numericPoints.map((point) => point.numericValue));
  const observedMaximum = Math.max(...numericPoints.map((point) => point.numericValue));
  const configuredMinimum =
    parameter.low != null && (!useLog || parameter.low > 0)
      ? Math.min(parameter.low, observedMinimum)
      : observedMinimum;
  const configuredMaximum =
    parameter.high != null
      ? Math.max(parameter.high, observedMaximum)
      : observedMaximum;
  let transformedDomain: [number, number] = [
    transform(configuredMinimum),
    transform(configuredMaximum),
  ];
  if (transformedDomain[0] === transformedDomain[1]) {
    transformedDomain = [
      transformedDomain[0] - Math.abs(transformedDomain[0] * 0.05 || 1),
      transformedDomain[1] + Math.abs(transformedDomain[1] * 0.05 || 1),
    ];
  }
  const yDomain = extent(numericPoints.map((point) => point.objective), 0.12);
  const xTransformed = linearScale(transformedDomain, [
    margin.left,
    width - margin.right,
  ]);
  const x = (value: number) => xTransformed(transform(value));
  const y = linearScale(yDomain, [height - margin.bottom, margin.top]);
  const xTicks = ticks(transformedDomain, 6).map(invert);
  const yTicks = ticks(yDomain, 5);
  const objectiveValues = numericPoints.map((point) => point.objective);
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
        `${index === 0 ? "M" : "L"}${x(group.center)},${y(group.median)}`,
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
        return (
          <g key={point.trial.number}>
            {isBest && (
              <circle
                cx={x(point.numericValue)}
                cy={y(point.objective)}
                r="10"
                fill="none"
                stroke={CHART.amber}
                strokeWidth="2"
              />
            )}
            <circle
              className="chart-point"
              cx={x(point.numericValue)}
              cy={y(point.objective)}
              r={isBest ? 6.5 : 5}
              fill={isBest ? CHART.amber : inBestQuartile ? CHART.lime : CHART.violet}
              fillOpacity={inBestQuartile || isBest ? 0.94 : 0.68}
              stroke="#10182a"
              strokeWidth="1.75"
            >
              <title>
                Trial #{point.trial.number} · {parameterLabel(parameter.name)}{" "}
                {displayParamValue(point.numericValue, parameter)} · {objectiveLabel(study.target)}{" "}
                {formatMetric(point.objective)}
              </title>
            </circle>
          </g>
        );
      })}
      <text
        x={margin.left + (width - margin.right - margin.left) / 2}
        y={height - 13}
        fill={CHART.muted}
        fontSize="12"
        textAnchor="middle"
      >
        {parameterLabel(parameter.name)}
        {useLog ? " · log scale" : ""}
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
  const yDomain = extent(points.map((point) => point.objective), 0.12);
  const y = linearScale(yDomain, [height - margin.bottom, margin.top]);
  const yTicks = ticks(yDomain, 5);
  const objectiveValues = points.map((point) => point.objective);
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
                y1={y(group.minimum)}
                y2={y(group.maximum)}
                stroke={CHART.cyan}
                strokeOpacity="0.5"
              />
              <rect
                x={centerX - boxWidth / 2}
                y={y(thirdQuartile)}
                width={boxWidth}
                height={Math.max(2, y(firstQuartile) - y(thirdQuartile))}
                rx="6"
                fill="rgba(79,213,231,0.12)"
                stroke={CHART.cyan}
                strokeOpacity="0.72"
              />
              <line
                x1={centerX - boxWidth / 2}
                x2={centerX + boxWidth / 2}
                y1={y(group.median)}
                y2={y(group.median)}
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
                return (
                  <circle
                    key={point.trial.number}
                    className="chart-point"
                    cx={centerX + jitter}
                    cy={y(point.objective)}
                    r={isBest ? 6.5 : 5}
                    fill={isBest ? CHART.amber : inBestQuartile ? CHART.lime : CHART.violet}
                    fillOpacity={inBestQuartile || isBest ? 0.94 : 0.7}
                    stroke="#10182a"
                    strokeWidth="1.75"
                  >
                    <title>
                      Trial #{point.trial.number} · {parameterLabel(parameter.name)}{" "}
                      {displayParamValue(point.value, parameter)} · {objectiveLabel(study.target)}{" "}
                      {formatMetric(point.objective)}
                    </title>
                  </circle>
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
  const filteredStudies = useMemo(() => {
    if (!snapshot) return [];
    const normalizedQuery = query.trim().toLowerCase();
    return snapshot.studies.filter((study) => {
      if (collection !== "all" && study.collection !== collection) return false;
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
  }, [collection, query, snapshot]);
  const selectedStudy =
    snapshot?.studies.find((study) => study.id === selectedStudyId) ?? null;

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
                return (
                  <button
                    key={study.id}
                    type="button"
                    role="option"
                    aria-selected={selected}
                    onClick={() => setSelectedStudyId(study.id)}
                    className={`focus-ring w-full rounded-xl px-3.5 py-3 text-left transition ${
                      selected
                        ? "bg-[#9b8cff]/[0.11] text-white ring-1 ring-inset ring-[#9b8cff]/20"
                        : "text-[#94a0b5] hover:bg-white/[0.035] hover:text-white"
                    }`}
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
