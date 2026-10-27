"use client";

import { useEffect, useMemo, useState, type ReactNode } from "react";
import {
  ArrowUpRight,
  Check,
  CircleGauge,
  Clock3,
  FlaskConical,
  GitCompareArrows,
  Layers3,
  Medal,
  MousePointer2,
  Sparkles,
  Target,
  Trophy,
} from "lucide-react";
import { ChartBoard } from "@/components/ChartBoard";
import { FinalTestComparison } from "@/components/FinalTestComparison";
import type { TestResultFile } from "@/lib/test-results";
import {
  columnLabel,
  formatDuration,
  formatMetric,
  formatNumber,
  getStudyStats,
  numericValue,
  type Study,
} from "@/lib/trials";

const SELECTION_STORAGE_KEY = "trial-atlas:comparison-selection";
const PALETTE = [
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
const CHART = {
  width: 960,
  height: 360,
  left: 72,
  right: 34,
  top: 32,
  bottom: 58,
};
const GRID = "#27334c";
const AXIS = "#77859f";
const TEXT = "#dce3f1";
const MUTED = "#8f9bb2";

type RankMode = "best" | "top5" | "median";

interface ComparisonRecord {
  study: Study;
  color: string;
  values: number[];
  sortedValues: number[];
  best: number | null;
  top5: number | null;
  median: number | null;
  q1: number | null;
  q3: number | null;
  minimum: number | null;
  maximum: number | null;
  standardDeviation: number | null;
  medianDuration: number | null;
  totalDuration: number | null;
  complete: number;
  bestTrial: string | null;
}

function mean(values: number[]): number | null {
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function quantile(sortedValues: number[], probability: number): number | null {
  if (sortedValues.length === 0) return null;
  if (sortedValues.length === 1) return sortedValues[0];
  const position = (sortedValues.length - 1) * probability;
  const lowerIndex = Math.floor(position);
  const upperIndex = Math.ceil(position);
  const weight = position - lowerIndex;
  return sortedValues[lowerIndex] * (1 - weight) + sortedValues[upperIndex] * weight;
}

function buildRecord(study: Study, color: string): ComparisonRecord {
  const values = study.rows
    .map((row) => numericValue(row[study.config.metric]))
    .filter((value): value is number => value != null);
  const sortedValues = [...values].sort((a, b) => a - b);
  const descending = [...sortedValues].reverse();
  const durations = study.config.duration
    ? study.rows
        .map((row) => numericValue(row[study.config.duration as string]))
        .filter((value): value is number => value != null && value >= 0)
        .sort((a, b) => a - b)
    : [];
  const average = mean(values);
  const variance =
    average == null ? null : mean(values.map((value) => (value - average) ** 2));
  const stats = getStudyStats(study);

  return {
    study,
    color,
    values,
    sortedValues,
    best: descending[0] ?? null,
    top5: mean(descending.slice(0, 5)),
    median: quantile(sortedValues, 0.5),
    q1: quantile(sortedValues, 0.25),
    q3: quantile(sortedValues, 0.75),
    minimum: sortedValues[0] ?? null,
    maximum: sortedValues.at(-1) ?? null,
    standardDeviation: variance == null ? null : Math.sqrt(variance),
    medianDuration: quantile(durations, 0.5),
    totalDuration: durations.length ? durations.reduce((sum, value) => sum + value, 0) : null,
    complete: stats.complete,
    bestTrial: stats.bestTrial,
  };
}

function scoreFor(record: ComparisonRecord, mode: RankMode): number | null {
  if (mode === "top5") return record.top5;
  if (mode === "median") return record.median;
  return record.best;
}

function modeLabel(mode: RankMode): string {
  if (mode === "top5") return "Top 5 average";
  if (mode === "median") return "Median trial";
  return "Best trial";
}

function shortLabel(value: string, maximum = 30): string {
  return value.length <= maximum ? value : `${value.slice(0, maximum - 1)}…`;
}

function extent(values: number[], padding = 0.06): [number, number] {
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

function ticks(domain: [number, number], count = 5): number[] {
  return Array.from(
    { length: count },
    (_, index) => domain[0] + ((domain[1] - domain[0]) * index) / (count - 1),
  );
}

function axisNumber(value: number): string {
  const absolute = Math.abs(value);
  if (value !== 0 && (absolute < 0.001 || absolute >= 10_000)) {
    return value.toExponential(1);
  }
  if (absolute >= 100) return Math.round(value).toLocaleString("en-US");
  if (absolute >= 1) return Number(value.toFixed(1)).toString();
  return Number(value.toFixed(3)).toString();
}

function EmptyChart({ label }: { label: string }) {
  return (
    <svg
      data-export-root
      viewBox={`0 0 ${CHART.width} ${CHART.height}`}
      className="block w-full"
      role="img"
      aria-label={label}
    >
      <title>{label}</title>
      <rect
        x="240"
        y="118"
        width="480"
        height="108"
        rx="18"
        fill="rgba(255,255,255,0.025)"
        stroke={GRID}
      />
      <text x="480" y="164" fill={TEXT} fontSize="15" textAnchor="middle">
        Select at least one compatible run
      </text>
      <text x="480" y="190" fill={MUTED} fontSize="12" textAnchor="middle">
        Use the run selector above to add data to this diagram.
      </text>
    </svg>
  );
}

function ChartLegend({ records }: { records: ComparisonRecord[] }) {
  return (
    <div className="mb-2 flex flex-wrap gap-x-4 gap-y-2 px-1" aria-label="Run colors">
      {records.map((record) => (
        <span
          key={record.study.id}
          className="flex min-w-0 items-center gap-2 text-[10px] text-[#8d9ab0]"
          title={record.study.name}
        >
          <span
            className="h-2 w-2 shrink-0 rounded-full"
            style={{ backgroundColor: record.color }}
          />
          <span className="max-w-52 truncate">{record.study.name}</span>
        </span>
      ))}
    </div>
  );
}

function LeaderboardChart({
  records,
  rankMode,
}: {
  records: ComparisonRecord[];
  rankMode: RankMode;
}) {
  const ranked = records
    .map((record) => ({ record, score: scoreFor(record, rankMode) }))
    .filter(
      (entry): entry is { record: ComparisonRecord; score: number } =>
        entry.score != null,
    )
    .sort((a, b) => b.score - a.score);

  if (ranked.length === 0) return <EmptyChart label="Run leaderboard" />;

  const height = Math.max(310, 92 + ranked.length * 50);
  const left = 236;
  const right = 78;
  const top = 24;
  const bottom = 50;
  const scores = ranked.map((entry) => entry.score);
  const maximum = Math.max(...scores);
  const minimum = Math.min(...scores);
  const domainMinimum = minimum >= 0 ? 0 : minimum * 1.08;
  const domain: [number, number] = [domainMinimum, maximum * 1.08 || 1];
  const x = linearScale(domain, [left, CHART.width - right]);
  const innerHeight = height - top - bottom;
  const rowHeight = innerHeight / ranked.length;
  const barHeight = Math.min(26, rowHeight * 0.55);

  return (
    <svg
      data-export-root
      viewBox={`0 0 ${CHART.width} ${height}`}
      className="block w-full"
      role="img"
      aria-label={`${modeLabel(rankMode)} by selected run`}
    >
      <title>{modeLabel(rankMode)} by selected run</title>
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
            {axisNumber(value)}
          </text>
        </g>
      ))}
      {ranked.map(({ record, score }, index) => {
        const centerY = top + rowHeight * index + rowHeight / 2;
        const barStart = x(domainMinimum);
        const barEnd = x(score);
        return (
          <g key={record.study.id}>
            <circle cx="20" cy={centerY} r="11" fill={record.color} fillOpacity="0.13" />
            <text
              x="20"
              y={centerY + 4}
              fill={record.color}
              fontSize="10"
              fontWeight="700"
              textAnchor="middle"
            >
              {index + 1}
            </text>
            <text
              x={left - 16}
              y={centerY + 4}
              fill={TEXT}
              fontSize="12"
              textAnchor="end"
            >
              {shortLabel(record.study.name)}
            </text>
            <rect
              x={barStart}
              y={centerY - barHeight / 2}
              width={Math.max(2, barEnd - barStart)}
              height={barHeight}
              rx={barHeight / 2}
              fill={record.color}
              fillOpacity={index === 0 ? 0.9 : 0.68}
            >
              <title>
                {record.study.name} · {modeLabel(rankMode).toLowerCase()}{" "}
                {formatMetric(score)}
              </title>
            </rect>
            <text
              x={Math.min(CHART.width - 12, barEnd + 10)}
              y={centerY + 4}
              fill={index === 0 ? record.color : TEXT}
              fontSize="12"
              fontWeight="700"
            >
              {formatMetric(score)}
            </text>
          </g>
        );
      })}
      <text
        x={left + (CHART.width - left - right) / 2}
        y={height - 6}
        fill={MUTED}
        fontSize="12"
        textAnchor="middle"
      >
        {modeLabel(rankMode)}
      </text>
    </svg>
  );
}

function TrajectoryChart({ records }: { records: ComparisonRecord[] }) {
  const series = records.flatMap((record) => {
    const raw = record.study.rows
      .map((row, index) => {
        const fallbackOrder = index + 1;
        return {
          order:
            numericValue(row[record.study.config.trial]) ?? fallbackOrder,
          value: numericValue(row[record.study.config.metric]),
        };
      })
      .filter(
        (point): point is { order: number; value: number } => point.value != null,
      )
      .sort((a, b) => a.order - b.order);
    let best = Number.NEGATIVE_INFINITY;
    const points = raw.map((point) => {
      best = Math.max(best, point.value);
      return { x: point.order, y: best };
    });
    return points.length ? [{ record, points }] : [];
  });

  if (series.length === 0) return <EmptyChart label="Optimization trajectories" />;

  const xDomain = extent(series.flatMap((entry) => entry.points.map((point) => point.x)), 0);
  const yDomain = extent(series.flatMap((entry) => entry.points.map((point) => point.y)), 0.08);
  const x = linearScale(xDomain, [CHART.left, CHART.width - CHART.right]);
  const y = linearScale(yDomain, [CHART.height - CHART.bottom, CHART.top]);

  return (
    <div>
      <ChartLegend records={series.map((entry) => entry.record)} />
      <svg
        data-export-root
        viewBox={`0 0 ${CHART.width} ${CHART.height}`}
        className="block w-full"
        role="img"
        aria-label="Cumulative best metric over the trial sequence"
      >
        <title>Cumulative best metric over the trial sequence</title>
        {ticks(yDomain, 5).map((value) => (
          <g key={`y-${value}`} aria-hidden="true">
            <line
              x1={CHART.left}
              x2={CHART.width - CHART.right}
              y1={y(value)}
              y2={y(value)}
              stroke={GRID}
            />
            <text
              x={CHART.left - 12}
              y={y(value) + 4}
              fill={AXIS}
              fontSize="11"
              textAnchor="end"
            >
              {axisNumber(value)}
            </text>
          </g>
        ))}
        {ticks(xDomain, 6).map((value) => (
          <g key={`x-${value}`} aria-hidden="true">
            <line
              x1={x(value)}
              x2={x(value)}
              y1={CHART.top}
              y2={CHART.height - CHART.bottom}
              stroke={GRID}
            />
            <text
              x={x(value)}
              y={CHART.height - CHART.bottom + 22}
              fill={AXIS}
              fontSize="11"
              textAnchor="middle"
            >
              {axisNumber(value)}
            </text>
          </g>
        ))}
        {series.map(({ record, points }) => {
          const path = points
            .map(
              (point, index) =>
                `${index === 0 ? "M" : "L"}${x(point.x)},${y(point.y)}`,
            )
            .join(" ");
          const last = points.at(-1);
          return (
            <g key={record.study.id}>
              <path
                d={path}
                fill="none"
                stroke={record.color}
                strokeWidth="2.4"
                strokeLinejoin="round"
                strokeLinecap="round"
                opacity="0.88"
              />
              {last && (
                <circle
                  cx={x(last.x)}
                  cy={y(last.y)}
                  r="4"
                  fill={record.color}
                  stroke="#10182a"
                  strokeWidth="2"
                >
                  <title>
                    {record.study.name} · best {formatMetric(last.y)} after{" "}
                    {formatNumber(last.x)} trials
                  </title>
                </circle>
              )}
            </g>
          );
        })}
        <text
          x={CHART.left + (CHART.width - CHART.left - CHART.right) / 2}
          y={CHART.height - 10}
          fill={MUTED}
          fontSize="12"
          textAnchor="middle"
        >
          Trial sequence
        </text>
        <text
          x="17"
          y={CHART.top + (CHART.height - CHART.top - CHART.bottom) / 2}
          fill={MUTED}
          fontSize="12"
          textAnchor="middle"
          transform={`rotate(-90 17 ${
            CHART.top + (CHART.height - CHART.top - CHART.bottom) / 2
          })`}
        >
          Cumulative best metric
        </text>
      </svg>
    </div>
  );
}

function DistributionChart({ records }: { records: ComparisonRecord[] }) {
  const plottable = records.filter(
    (
      record,
    ): record is ComparisonRecord & {
      minimum: number;
      q1: number;
      median: number;
      q3: number;
      maximum: number;
    } =>
      record.minimum != null &&
      record.q1 != null &&
      record.median != null &&
      record.q3 != null &&
      record.maximum != null,
  );

  if (plottable.length === 0) return <EmptyChart label="Metric distributions" />;

  const height = Math.max(320, 100 + plottable.length * 58);
  const left = 238;
  const right = 38;
  const top = 28;
  const bottom = 52;
  const domain = extent(
    plottable.flatMap((record) => [record.minimum, record.maximum]),
    0.08,
  );
  const x = linearScale(domain, [left, CHART.width - right]);
  const rowHeight = (height - top - bottom) / plottable.length;

  return (
    <svg
      data-export-root
      viewBox={`0 0 ${CHART.width} ${height}`}
      className="block w-full"
      role="img"
      aria-label="Metric distributions by selected run"
    >
      <title>Metric distributions by selected run</title>
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
            {axisNumber(value)}
          </text>
        </g>
      ))}
      {plottable.map((record, index) => {
        const centerY = top + rowHeight * index + rowHeight / 2;
        const sampleStep = Math.max(1, Math.ceil(record.values.length / 42));
        const samples = record.values.filter((_, valueIndex) => valueIndex % sampleStep === 0);
        return (
          <g key={record.study.id}>
            <text
              x={left - 16}
              y={centerY + 4}
              fill={TEXT}
              fontSize="12"
              textAnchor="end"
            >
              {shortLabel(record.study.name)}
            </text>
            <line
              x1={x(record.minimum)}
              x2={x(record.maximum)}
              y1={centerY}
              y2={centerY}
              stroke={record.color}
              strokeWidth="1.5"
              opacity="0.7"
            />
            <line
              x1={x(record.minimum)}
              x2={x(record.minimum)}
              y1={centerY - 7}
              y2={centerY + 7}
              stroke={record.color}
              strokeWidth="1.5"
            />
            <line
              x1={x(record.maximum)}
              x2={x(record.maximum)}
              y1={centerY - 7}
              y2={centerY + 7}
              stroke={record.color}
              strokeWidth="1.5"
            />
            <rect
              x={x(record.q1)}
              y={centerY - 13}
              width={Math.max(2, x(record.q3) - x(record.q1))}
              height="26"
              rx="7"
              fill={record.color}
              fillOpacity="0.17"
              stroke={record.color}
              strokeOpacity="0.85"
            >
              <title>
                {record.study.name} · Q1 {formatMetric(record.q1)} · median{" "}
                {formatMetric(record.median)} · Q3 {formatMetric(record.q3)}
              </title>
            </rect>
            {samples.map((value, sampleIndex) => {
              const jitter = ((sampleIndex * 17) % 11) - 5;
              return (
                <circle
                  key={`${value}-${sampleIndex}`}
                  cx={x(value)}
                  cy={centerY + jitter}
                  r="1.8"
                  fill={record.color}
                  fillOpacity="0.42"
                />
              );
            })}
            <line
              x1={x(record.median)}
              x2={x(record.median)}
              y1={centerY - 14}
              y2={centerY + 14}
              stroke={record.color}
              strokeWidth="3"
            />
          </g>
        );
      })}
      <text
        x={left + (CHART.width - left - right) / 2}
        y={height - 7}
        fill={MUTED}
        fontSize="12"
        textAnchor="middle"
      >
        Configured primary metric
      </text>
    </svg>
  );
}

function EfficiencyChart({
  records,
  rankMode,
}: {
  records: ComparisonRecord[];
  rankMode: RankMode;
}) {
  const points = records
    .map((record) => ({
      record,
      runtime: record.medianDuration,
      score: scoreFor(record, rankMode),
    }))
    .filter(
      (
        point,
      ): point is {
        record: ComparisonRecord;
        runtime: number;
        score: number;
      } => point.runtime != null && point.runtime > 0 && point.score != null,
    );

  if (points.length === 0) return <EmptyChart label="Quality and runtime trade-off" />;

  const logRuntimeDomain = extent(
    points.map((point) => Math.log10(point.runtime)),
    0.08,
  );
  const scoreDomain = extent(
    points.map((point) => point.score),
    0.12,
  );
  const xLog = linearScale(logRuntimeDomain, [
    CHART.left,
    CHART.width - CHART.right,
  ]);
  const x = (value: number) => xLog(Math.log10(value));
  const y = linearScale(scoreDomain, [
    CHART.height - CHART.bottom,
    CHART.top,
  ]);
  const paretoIds = new Set(
    points
      .filter(
        (candidate) =>
          !points.some(
            (other) =>
              other !== candidate &&
              other.runtime <= candidate.runtime &&
              other.score >= candidate.score &&
              (other.runtime < candidate.runtime || other.score > candidate.score),
          ),
      )
      .map((point) => point.record.study.id),
  );

  return (
    <div>
      <div className="mb-2 flex flex-wrap gap-x-4 gap-y-2 px-1">
        {points.map((point, index) => (
          <span
            key={point.record.study.id}
            className="flex min-w-0 items-center gap-2 text-[10px] text-[#8d9ab0]"
            title={point.record.study.name}
          >
            <span
              className="flex h-4 w-4 shrink-0 items-center justify-center rounded-full text-[8px] font-bold text-[#080d18]"
              style={{ backgroundColor: point.record.color }}
            >
              {index + 1}
            </span>
            <span className="max-w-44 truncate">{point.record.study.name}</span>
          </span>
        ))}
      </div>
      <svg
        data-export-root
        viewBox={`0 0 ${CHART.width} ${CHART.height}`}
        className="block w-full"
        role="img"
        aria-label={`${modeLabel(rankMode)} compared with median runtime`}
      >
        <title>{modeLabel(rankMode)} compared with median runtime</title>
        {ticks(scoreDomain, 5).map((value) => (
          <g key={`y-${value}`} aria-hidden="true">
            <line
              x1={CHART.left}
              x2={CHART.width - CHART.right}
              y1={y(value)}
              y2={y(value)}
              stroke={GRID}
            />
            <text
              x={CHART.left - 12}
              y={y(value) + 4}
              fill={AXIS}
              fontSize="11"
              textAnchor="end"
            >
              {axisNumber(value)}
            </text>
          </g>
        ))}
        {ticks(logRuntimeDomain, 5).map((logValue) => {
          const value = 10 ** logValue;
          return (
            <g key={`x-${logValue}`} aria-hidden="true">
              <line
                x1={x(value)}
                x2={x(value)}
                y1={CHART.top}
                y2={CHART.height - CHART.bottom}
                stroke={GRID}
              />
              <text
                x={x(value)}
                y={CHART.height - CHART.bottom + 22}
                fill={AXIS}
                fontSize="11"
                textAnchor="middle"
              >
                {formatDuration(value)}
              </text>
            </g>
          );
        })}
        {points.map((point, index) => {
          const pareto = paretoIds.has(point.record.study.id);
          return (
            <g key={point.record.study.id}>
              {pareto && (
                <circle
                  cx={x(point.runtime)}
                  cy={y(point.score)}
                  r="14"
                  fill={point.record.color}
                  fillOpacity="0.11"
                  stroke={point.record.color}
                  strokeOpacity="0.5"
                  strokeDasharray="3 3"
                />
              )}
              <circle
                cx={x(point.runtime)}
                cy={y(point.score)}
                r="8"
                fill={point.record.color}
                stroke="#10182a"
                strokeWidth="2"
              >
                <title>
                  {point.record.study.name} · {modeLabel(rankMode).toLowerCase()}{" "}
                  {formatMetric(point.score)} · median runtime{" "}
                  {formatDuration(point.runtime)}
                  {pareto ? " · Pareto efficient" : ""}
                </title>
              </circle>
              <text
                x={x(point.runtime)}
                y={y(point.score) + 3}
                fill="#080d18"
                fontSize="8"
                fontWeight="800"
                textAnchor="middle"
              >
                {index + 1}
              </text>
            </g>
          );
        })}
        <text
          x={CHART.left + (CHART.width - CHART.left - CHART.right) / 2}
          y={CHART.height - 10}
          fill={MUTED}
          fontSize="12"
          textAnchor="middle"
        >
          Median trial runtime · logarithmic scale
        </text>
        <text
          x="17"
          y={CHART.top + (CHART.height - CHART.top - CHART.bottom) / 2}
          fill={MUTED}
          fontSize="12"
          textAnchor="middle"
          transform={`rotate(-90 17 ${
            CHART.top + (CHART.height - CHART.top - CHART.bottom) / 2
          })`}
        >
          {modeLabel(rankMode)}
        </text>
      </svg>
      {paretoIds.size > 0 && (
        <p className="mt-1 px-1 text-[10px] text-[#728097]">
          Dashed rings mark Pareto-efficient runs: no selected run is both faster
          and better.
        </p>
      )}
    </div>
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
      <div className="absolute -right-8 -top-10 h-24 w-24 rounded-full bg-[#9b8cff]/[0.05] blur-2xl" />
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
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#9b8cff]/10 text-[#b5aaff] ring-1 ring-[#9b8cff]/15">
          {icon}
        </span>
      </div>
    </article>
  );
}

function HpoComparisonWorkspace({
  studies,
  onOpenStudy,
}: {
  studies: Study[];
  onOpenStudy: (study: Study) => void;
}) {
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [rankMode, setRankMode] = useState<RankMode>("best");
  const [storageReady, setStorageReady] = useState(false);
  const studySignature = studies.map((study) => study.id).join("\u001f");

  useEffect(() => {
    try {
      const raw = window.localStorage.getItem(SELECTION_STORAGE_KEY);
      const parsed = raw
        ? (JSON.parse(raw) as { selectedIds?: unknown; rankMode?: unknown })
        : null;
      const validIds = new Set(studies.map((study) => study.id));
      const storedIds = Array.isArray(parsed?.selectedIds)
        ? parsed.selectedIds.filter(
            (id): id is string => typeof id === "string" && validIds.has(id),
          )
        : null;
      setSelectedIds(storedIds ?? studies.map((study) => study.id));
      if (
        parsed?.rankMode === "best" ||
        parsed?.rankMode === "top5" ||
        parsed?.rankMode === "median"
      ) {
        setRankMode(parsed.rankMode);
      }
    } catch {
      setSelectedIds(studies.map((study) => study.id));
    }
    setStorageReady(true);
    // ComparisonWorkspace remounts when entering the page, so this intentionally
    // restores the saved selection once rather than after every workspace change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!storageReady) return;
    const validIds = new Set(studies.map((study) => study.id));
    setSelectedIds((current) => current.filter((id) => validIds.has(id)));
  }, [storageReady, studySignature, studies]);

  useEffect(() => {
    if (!storageReady) return;
    try {
      window.localStorage.setItem(
        SELECTION_STORAGE_KEY,
        JSON.stringify({ selectedIds, rankMode }),
      );
    } catch {
      // Comparison state remains available for the current session.
    }
  }, [rankMode, selectedIds, storageReady]);

  const records = useMemo(
    () =>
      studies.map((study, index) =>
        buildRecord(study, PALETTE[index % PALETTE.length]),
      ),
    [studies],
  );
  const selectedSet = useMemo(() => new Set(selectedIds), [selectedIds]);
  const selectedRecords = records.filter((record) =>
    selectedSet.has(record.study.id),
  );
  const ranked = [...selectedRecords]
    .map((record) => ({ record, score: scoreFor(record, rankMode) }))
    .filter(
      (entry): entry is { record: ComparisonRecord; score: number } =>
        entry.score != null,
    )
    .sort((a, b) => b.score - a.score);
  const leader = ranked[0] ?? null;
  const runnerUp = ranked[1] ?? null;
  const advantage =
    leader && runnerUp ? leader.score - runnerUp.score : null;
  const selectedTrialCount = selectedRecords.reduce(
    (sum, record) => sum + record.study.rows.length,
    0,
  );
  const totalDuration = selectedRecords
    .map((record) => record.totalDuration)
    .filter((value): value is number => value != null)
    .reduce((sum, value) => sum + value, 0);
  const metricColumns = new Set(
    selectedRecords.map((record) => record.study.config.metric),
  );

  const toggleStudy = (studyId: string) => {
    setSelectedIds((current) =>
      current.includes(studyId)
        ? current.filter((id) => id !== studyId)
        : [...current, studyId],
    );
  };

  return (
    <div className="fade-up space-y-6">
      <section className="relative overflow-hidden rounded-[1.5rem] border border-[#9b8cff]/15 bg-[linear-gradient(135deg,rgba(155,140,255,0.11),rgba(79,213,231,0.035)_45%,rgba(16,24,42,0.82))] px-5 py-6 shadow-[0_24px_70px_rgba(0,0,0,.2)] sm:px-7 sm:py-7">
        <div className="absolute -right-20 -top-24 h-64 w-64 rounded-full bg-[#9b8cff]/10 blur-3xl" />
        <div className="relative flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
          <div>
            <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.17em] text-[#9b90d8]">
              <GitCompareArrows size={13} /> Multi-run comparison
            </div>
            <h1 className="mt-3 text-3xl font-semibold tracking-[-0.045em] text-white sm:text-4xl">
              See which run really wins.
            </h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-[#909db2]">
              Select any number of studies and compare search quality, stability,
              convergence, and runtime in one shared view.
            </p>
          </div>
          <div className="flex shrink-0 items-center gap-2 rounded-2xl border border-white/[0.08] bg-black/15 p-1.5">
            {(["best", "top5", "median"] as RankMode[]).map((mode) => (
              <button
                key={mode}
                type="button"
                onClick={() => setRankMode(mode)}
                className={`focus-ring rounded-xl px-3 py-2 text-[11px] font-semibold transition ${
                  rankMode === mode
                    ? "bg-[#9b8cff] text-white shadow-[0_8px_24px_rgba(118,91,255,.24)]"
                    : "text-[#8f9bb0] hover:bg-white/[0.05] hover:text-white"
                }`}
                aria-pressed={rankMode === mode}
              >
                {mode === "best" ? "Best" : mode === "top5" ? "Top 5 avg" : "Median"}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="surface overflow-hidden">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-white/[0.07] px-5 py-4 sm:px-6">
          <div>
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-semibold text-white">Runs in comparison</h2>
              <span className="rounded-full bg-[#9b8cff]/10 px-2 py-0.5 text-[10px] font-semibold text-[#b6acff]">
                {selectedRecords.length} / {studies.length}
              </span>
            </div>
            <p className="mt-1.5 text-xs text-[#8290a7]">
              Click a run to include or exclude it from every diagram.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setSelectedIds(studies.map((study) => study.id))}
              className="focus-ring rounded-lg px-2.5 py-1.5 text-[11px] font-semibold text-[#9faae0] hover:bg-white/[0.05] hover:text-white"
            >
              Select all
            </button>
            <span className="h-4 w-px bg-white/[0.09]" />
            <button
              type="button"
              onClick={() => setSelectedIds([])}
              className="focus-ring rounded-lg px-2.5 py-1.5 text-[11px] font-semibold text-[#7f8da4] hover:bg-white/[0.05] hover:text-white"
            >
              Clear
            </button>
          </div>
        </header>
        <div className="grid gap-2.5 p-4 sm:grid-cols-2 sm:p-5 xl:grid-cols-3">
          {records.map((record) => {
            const selected = selectedSet.has(record.study.id);
            return (
              <button
                key={record.study.id}
                type="button"
                onClick={() => toggleStudy(record.study.id)}
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
                    color: record.color,
                    backgroundColor: `${record.color}18`,
                    boxShadow: selected
                      ? `inset 0 0 0 1px ${record.color}40`
                      : undefined,
                  }}
                >
                  {selected ? <Check size={15} strokeWidth={2.5} /> : <Layers3 size={14} />}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-xs font-semibold text-[#dce3ee]">
                    {record.study.name}
                  </span>
                  <span className="mt-1 block truncate text-[10px] text-[#6e7c94]">
                    {record.study.dataset} · {record.study.rows.length} trials · best{" "}
                    {formatMetric(record.best)}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
      </section>

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label={`${modeLabel(rankMode)} leader`}
          value={leader ? formatMetric(leader.score) : "—"}
          context={leader?.record.study.name ?? "Select runs to compare"}
          icon={<Trophy size={17} />}
        />
        <MetricCard
          label="Lead over #2"
          value={advantage == null ? "—" : `+${formatMetric(advantage)}`}
          context={runnerUp ? `Ahead of ${runnerUp.record.study.name}` : "Choose at least two runs"}
          icon={<Medal size={17} />}
        />
        <MetricCard
          label="Trials compared"
          value={selectedTrialCount.toLocaleString()}
          context={`Across ${selectedRecords.length} selected ${selectedRecords.length === 1 ? "run" : "runs"}`}
          icon={<Target size={17} />}
        />
        <MetricCard
          label="Total trial time"
          value={totalDuration > 0 ? formatDuration(totalDuration) : "—"}
          context="Sum of reported trial runtimes"
          icon={<Clock3 size={17} />}
        />
      </div>

      {selectedRecords.length === 0 ? (
        <section className="surface flex min-h-72 items-center justify-center px-6 py-10 text-center">
          <div>
            <span className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-[#9b8cff]/10 text-[#b5aaff]">
              <MousePointer2 size={21} />
            </span>
            <h2 className="mt-4 text-lg font-semibold text-white">Choose the runs to compare</h2>
            <p className="mx-auto mt-2 max-w-md text-xs leading-5 text-[#7f8ca2]">
              Select any run above. Every chart and the scorecard will update
              together.
            </p>
          </div>
        </section>
      ) : (
        <>
          {metricColumns.size > 1 && (
            <div className="flex items-start gap-3 rounded-xl border border-[#f4c86f]/15 bg-[#f4c86f]/[0.055] px-4 py-3 text-xs text-[#c8b889]">
              <CircleGauge size={16} className="mt-0.5 shrink-0 text-[#f4c86f]" />
              <p className="leading-5">
                The selected runs use different configured metric columns. Values
                are still shown together, so confirm those columns are semantically
                comparable before drawing conclusions.
              </p>
            </div>
          )}

          <ChartBoard
            storageKey="trial-atlas:charts:comparison"
            items={[
              {
                id: "comparison-leaderboard",
                title: `${modeLabel(rankMode)} leaderboard`,
                description:
                  rankMode === "best"
                    ? "The strongest observed result in each selected run."
                    : rankMode === "top5"
                      ? "Average of each run’s five strongest trials, reducing one-off luck."
                      : "Typical trial quality across the full search.",
                exportName: `run-comparison-${rankMode}-leaderboard`,
                content: (
                  <LeaderboardChart records={selectedRecords} rankMode={rankMode} />
                ),
              },
              {
                id: "runtime-efficiency",
                title: "Quality vs. runtime",
                description:
                  "Higher and farther left is better. Dashed rings identify efficient trade-offs.",
                exportName: `run-comparison-${rankMode}-runtime`,
                content: (
                  <EfficiencyChart records={selectedRecords} rankMode={rankMode} />
                ),
              },
              {
                id: "optimization-race",
                title: "Optimization race",
                description:
                  "Cumulative best metric over the trial sequence shows how quickly each search found a strong result.",
                exportName: "run-comparison-optimization-race",
                className: "xl:col-span-2",
                content: <TrajectoryChart records={selectedRecords} />,
              },
              {
                id: "distribution-comparison",
                title: "Trial score distributions",
                description:
                  "Boxes show the interquartile range and median; whiskers span the observed range.",
                exportName: "run-comparison-score-distributions",
                className: "xl:col-span-2",
                content: <DistributionChart records={selectedRecords} />,
              },
            ]}
          />

          <section className="surface overflow-hidden">
            <header className="flex flex-wrap items-start justify-between gap-3 border-b border-white/[0.07] px-5 py-4 sm:px-6">
              <div>
                <div className="flex items-center gap-2">
                  <Sparkles size={14} className="text-[#b8e56e]" />
                  <h2 className="text-[15px] font-semibold text-white">Run scorecard</h2>
                </div>
                <p className="mt-1.5 text-xs text-[#8290a7]">
                  Sorted by {modeLabel(rankMode).toLowerCase()}. Open a run for
                  trial-level inspection.
                </p>
              </div>
              <span className="rounded-full border border-white/[0.08] bg-white/[0.03] px-2.5 py-1 text-[10px] text-[#8794aa]">
                {columnLabel(selectedRecords[0]?.study.config.metric ?? null)}
              </span>
            </header>
            <div className="overflow-x-auto">
              <table className="w-full min-w-[1040px] text-left text-xs">
                <thead className="border-b border-white/[0.07] bg-black/10 text-[10px] uppercase tracking-[0.11em] text-[#6f7d95]">
                  <tr>
                    <th className="px-5 py-3.5 text-center font-semibold">Rank</th>
                    <th className="px-4 py-3.5 font-semibold">Run</th>
                    <th className="px-4 py-3.5 font-semibold">Best</th>
                    <th className="px-4 py-3.5 font-semibold">Top 5 avg</th>
                    <th className="px-4 py-3.5 font-semibold">Median</th>
                    <th className="px-4 py-3.5 font-semibold">Std. dev.</th>
                    <th className="px-4 py-3.5 font-semibold">Median runtime</th>
                    <th className="px-4 py-3.5 font-semibold">Completed</th>
                    <th className="px-5 py-3.5 text-right font-semibold">Details</th>
                  </tr>
                </thead>
                <tbody>
                  {ranked.map(({ record }, index) => (
                    <tr
                      key={record.study.id}
                      className="border-b border-white/[0.055] text-[#aeb8ca] last:border-0 hover:bg-white/[0.025]"
                    >
                      <td className="px-5 py-4 text-center">
                        <span
                          className="inline-flex h-6 min-w-6 items-center justify-center rounded-lg px-1.5 text-[10px] font-bold"
                          style={{
                            color: record.color,
                            backgroundColor: `${record.color}16`,
                          }}
                        >
                          {index + 1}
                        </span>
                      </td>
                      <td className="max-w-72 px-4 py-4">
                        <span className="block truncate font-semibold text-white" title={record.study.name}>
                          {record.study.name}
                        </span>
                        <span className="mt-1 block truncate text-[10px] text-[#6d7a92]">
                          {record.study.dataset} · best trial #{record.bestTrial ?? "—"}
                        </span>
                      </td>
                      <td className="px-4 py-4 font-semibold tabular-nums text-[#d5cffd]">
                        {formatMetric(record.best)}
                      </td>
                      <td className="px-4 py-4 tabular-nums">
                        {formatMetric(record.top5)}
                      </td>
                      <td className="px-4 py-4 tabular-nums">
                        {formatMetric(record.median)}
                      </td>
                      <td className="px-4 py-4 tabular-nums">
                        {formatMetric(record.standardDeviation)}
                      </td>
                      <td className="px-4 py-4 tabular-nums">
                        {formatDuration(record.medianDuration)}
                      </td>
                      <td className="px-4 py-4 tabular-nums">
                        {record.complete} / {record.study.rows.length}
                      </td>
                      <td className="px-5 py-4 text-right">
                        <button
                          type="button"
                          onClick={() => onOpenStudy(record.study)}
                          className="focus-ring inline-flex items-center gap-1.5 rounded-lg px-2 py-1.5 font-semibold text-[#a99dff] hover:bg-[#9b8cff]/10"
                        >
                          Open <ArrowUpRight size={13} />
                        </button>
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

export function ComparisonWorkspace({
  studies,
  initialTestResultsFile,
  onOpenStudy,
}: {
  studies: Study[];
  initialTestResultsFile: TestResultFile | null;
  onOpenStudy: (study: Study) => void;
}) {
  const [source, setSource] = useState<"hpo" | "test">("hpo");

  return (
    <div className="space-y-4">
      <section className="surface-soft flex flex-col gap-3 p-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="px-1">
          <p className="text-xs font-semibold text-[#dce3ee]">Comparison source</p>
          <p className="mt-1 text-[10px] text-[#6f7d94]">
            Separate model selection evidence from held-out generalization.
          </p>
        </div>
        <div className="flex rounded-xl border border-white/[0.08] bg-[#0a1120] p-1">
          <button
            type="button"
            onClick={() => setSource("hpo")}
            aria-pressed={source === "hpo"}
            className={`focus-ring flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-2 text-[11px] font-semibold transition sm:flex-none ${
              source === "hpo"
                ? "bg-[#9b8cff] text-white shadow-[0_8px_24px_rgba(118,91,255,.2)]"
                : "text-[#7f8ca2] hover:text-white"
            }`}
          >
            <GitCompareArrows size={14} />
            HPO searches
          </button>
          <button
            type="button"
            onClick={() => setSource("test")}
            aria-pressed={source === "test"}
            className={`focus-ring flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-2 text-[11px] font-semibold transition sm:flex-none ${
              source === "test"
                ? "bg-[#4fd5e7] text-[#07131d] shadow-[0_8px_24px_rgba(79,213,231,.16)]"
                : "text-[#7f8ca2] hover:text-white"
            }`}
          >
            <FlaskConical size={14} />
            Final test
          </button>
        </div>
      </section>

      {source === "hpo" ? (
        <HpoComparisonWorkspace studies={studies} onOpenStudy={onOpenStudy} />
      ) : (
        <FinalTestComparison initialFile={initialTestResultsFile} />
      )}
    </div>
  );
}
