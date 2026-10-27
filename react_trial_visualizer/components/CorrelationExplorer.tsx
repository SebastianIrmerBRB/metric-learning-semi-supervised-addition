"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Network, RotateCcw, Search } from "lucide-react";
import { ChartBoard } from "@/components/ChartBoard";
import {
  columnLabel,
  formatNumber,
  normalizeColumn,
  numericColumns,
  numericValue,
  type Study,
  type TrialRow,
} from "@/lib/trials";

type CorrelationMethod = "pearson" | "spearman";

interface PairPoint {
  x: number;
  y: number;
  row: TrialRow;
  index: number;
}

interface CorrelationResult {
  value: number | null;
  count: number;
}

const MATRIX_CELL = 46;
const MATRIX_LEFT = 176;
const MATRIX_TOP = 132;
const SCATTER_WIDTH = 960;
const SCATTER_HEIGHT = 360;
const SCATTER_MARGIN = { top: 42, right: 38, bottom: 64, left: 82 };
const GRID = "#27334c";
const AXIS = "#77859f";
const TEXT = "#dce3f1";
const VIOLET = "#9b8cff";
const CYAN = "#4fd5e7";
const LIME = "#b8e56e";

const DERIVED_BATCH_SIZE_PREFIX = "__trial_atlas_batch_size__:";
const DERIVED_SAMPLER_M_PREFIX = "__trial_atlas_sampler_m__:";

interface BatchSamplerParts {
  batchSize: number;
  samplerM: number;
}

function batchSamplerParts(value: string | null | undefined): BatchSamplerParts | null {
  if (value == null) return null;
  const normalized = value.trim().replace(/^(["'])(.*)\1$/, "$2").trim();
  const match = normalized.match(/^(\d+)\s*:\s*(\d+)$/);
  if (!match) return null;
  const batchSize = Number(match[1]);
  const samplerM = Number(match[2]);
  return batchSize > 0 && samplerM > 0 ? { batchSize, samplerM } : null;
}

function batchSamplerColumn(study: Study): string | null {
  return study.columns.find((column) => {
    const normalized = normalizeColumn(column);
    return normalized === "batch_sampler" || normalized === "param_batch_sampler";
  }) ?? null;
}

function derivedBatchSizeColumn(sourceColumn: string): string {
  return `${DERIVED_BATCH_SIZE_PREFIX}${sourceColumn}`;
}

function derivedSamplerMColumn(sourceColumn: string): string {
  return `${DERIVED_SAMPLER_M_PREFIX}${sourceColumn}`;
}

function derivedSourceColumn(column: string, prefix: string): string | null {
  return column.startsWith(prefix) ? column.slice(prefix.length) : null;
}

function relationshipValue(row: TrialRow, column: string): number | null {
  const batchSizeSource = derivedSourceColumn(column, DERIVED_BATCH_SIZE_PREFIX);
  if (batchSizeSource) return batchSamplerParts(row[batchSizeSource])?.batchSize ?? null;

  const samplerMSource = derivedSourceColumn(column, DERIVED_SAMPLER_M_PREFIX);
  if (samplerMSource) return batchSamplerParts(row[samplerMSource])?.samplerM ?? null;

  return numericValue(row[column]);
}

function relationshipLabel(column: string): string {
  if (column.startsWith(DERIVED_BATCH_SIZE_PREFIX)) return "Batch Size";
  if (column.startsWith(DERIVED_SAMPLER_M_PREFIX)) return "Sampler M";
  return columnLabel(column);
}

function relationshipSource(column: string): string {
  const batchSizeSource = derivedSourceColumn(column, DERIVED_BATCH_SIZE_PREFIX);
  if (batchSizeSource) return `${batchSizeSource} → batch_size`;

  const samplerMSource = derivedSourceColumn(column, DERIVED_SAMPLER_M_PREFIX);
  if (samplerMSource) return `${samplerMSource} → sampler_m`;

  return column;
}

function isBatchSizeColumn(column: string): boolean {
  if (column.startsWith(DERIVED_BATCH_SIZE_PREFIX)) return true;
  const normalized = normalizeColumn(column);
  return normalized === "batch_size" || normalized === "param_batch_size";
}

function isSamplerMColumn(column: string): boolean {
  if (column.startsWith(DERIVED_SAMPLER_M_PREFIX)) return true;
  const normalized = normalizeColumn(column);
  return normalized === "sampler_m" || normalized === "param_sampler_m";
}

function usableRelationshipColumns(study: Study): string[] {
  const numeric = numericColumns(study);
  const samplerColumn = batchSamplerColumn(study);
  const candidates = [...numeric];

  if (samplerColumn && !numeric.some(isBatchSizeColumn)) {
    candidates.push(derivedBatchSizeColumn(samplerColumn));
  }
  if (samplerColumn && !numeric.some(isSamplerMColumn)) {
    candidates.push(derivedSamplerMColumn(samplerColumn));
  }

  return candidates.filter((column) => {
    const values = study.rows
      .map((row) => relationshipValue(row, column))
      .filter((value): value is number => value != null);
    return values.length >= 3 && new Set(values).size > 1;
  });
}

function pairPoints(study: Study, xColumn: string, yColumn: string): PairPoint[] {
  return study.rows.flatMap((row, index) => {
    const x = relationshipValue(row, xColumn);
    const y = relationshipValue(row, yColumn);
    return x == null || y == null ? [] : [{ x, y, row, index }];
  });
}

function averageRanks(values: number[]): number[] {
  const ranked = values.map((value, index) => ({ value, index })).sort((a, b) => a.value - b.value);
  const result = new Array<number>(values.length);
  let start = 0;
  while (start < ranked.length) {
    let end = start + 1;
    while (end < ranked.length && ranked[end].value === ranked[start].value) end += 1;
    const rank = (start + 1 + end) / 2;
    for (let index = start; index < end; index += 1) result[ranked[index].index] = rank;
    start = end;
  }
  return result;
}

function pearson(xValues: number[], yValues: number[]): number | null {
  if (xValues.length < 3 || xValues.length !== yValues.length) return null;
  const xMean = xValues.reduce((sum, value) => sum + value, 0) / xValues.length;
  const yMean = yValues.reduce((sum, value) => sum + value, 0) / yValues.length;
  let numerator = 0;
  let xSquares = 0;
  let ySquares = 0;
  for (let index = 0; index < xValues.length; index += 1) {
    const xDelta = xValues[index] - xMean;
    const yDelta = yValues[index] - yMean;
    numerator += xDelta * yDelta;
    xSquares += xDelta * xDelta;
    ySquares += yDelta * yDelta;
  }
  const denominator = Math.sqrt(xSquares * ySquares);
  if (denominator === 0) return null;
  return Math.max(-1, Math.min(1, numerator / denominator));
}

function correlation(points: PairPoint[], method: CorrelationMethod): CorrelationResult {
  const xValues = points.map((point) => point.x);
  const yValues = points.map((point) => point.y);
  return {
    value: method === "spearman"
      ? pearson(averageRanks(xValues), averageRanks(yValues))
      : pearson(xValues, yValues),
    count: points.length,
  };
}

function extent(values: number[], padding = 0.08): [number, number] {
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (minimum === maximum) {
    const delta = Math.abs(minimum) * 0.06 || 1;
    return [minimum - delta, maximum + delta];
  }
  const extra = (maximum - minimum) * padding;
  return [minimum - extra, maximum + extra];
}

function scale(domain: [number, number], range: [number, number]) {
  const span = domain[1] - domain[0] || 1;
  return (value: number) => range[0] + ((value - domain[0]) / span) * (range[1] - range[0]);
}

function ticks(domain: [number, number], count = 6): number[] {
  return Array.from({ length: count }, (_, index) => domain[0] + ((domain[1] - domain[0]) * index) / (count - 1));
}

function axisNumber(value: number): string {
  const absolute = Math.abs(value);
  if (value !== 0 && (absolute < 0.001 || absolute >= 10_000)) return value.toExponential(1);
  if (absolute >= 100) return Math.round(value).toLocaleString("en-US");
  if (absolute >= 1) return Number(value.toFixed(2)).toString();
  return Number(value.toFixed(4)).toString();
}

function shortLabel(column: string, length = 23): string {
  const label = relationshipLabel(column);
  return label.length > length ? `${label.slice(0, length - 1)}…` : label;
}

function matrixColor(value: number | null): string {
  if (value == null) return "rgba(255,255,255,.025)";
  const opacity = 0.1 + Math.abs(value) * 0.78;
  return value < 0 ? `rgba(79,213,231,${opacity})` : `rgba(155,140,255,${opacity})`;
}

function methodName(method: CorrelationMethod): string {
  return method === "pearson" ? "Pearson r" : "Spearman ρ";
}

function CorrelationMatrix({
  study,
  columns,
  method,
  xColumn,
  yColumn,
  onSelectPair,
}: {
  study: Study;
  columns: string[];
  method: CorrelationMethod;
  xColumn: string;
  yColumn: string;
  onSelectPair: (xColumn: string, yColumn: string) => void;
}) {
  if (columns.length === 0) {
    return (
      <svg data-export-root viewBox="0 0 960 300" className="block w-full" role="img" aria-label="No attributes selected for correlation matrix">
        <title>No attributes selected for correlation matrix</title>
        <text x="480" y="142" fill={TEXT} fontSize="15" textAnchor="middle">Select at least two matrix attributes</text>
        <text x="480" y="168" fill={AXIS} fontSize="12" textAnchor="middle">Use the Attributes control above.</text>
      </svg>
    );
  }

  const matrix = columns.map((rowColumn) =>
    columns.map((column) => correlation(pairPoints(study, column, rowColumn), method)),
  );
  const width = MATRIX_LEFT + columns.length * MATRIX_CELL + 36;
  const height = MATRIX_TOP + columns.length * MATRIX_CELL + 54;
  const gradientId = `correlation-gradient-${study.id.replace(/[^a-zA-Z0-9]/g, "")}-${method}`;

  return (
    <div className="overflow-x-auto">
      <svg
        data-export-root
        viewBox={`0 0 ${width} ${height}`}
        className="block w-full"
        style={{ minWidth: `${Math.max(720, width)}px` }}
        role="img"
        aria-label={`${methodName(method)} correlation matrix for ${study.name}`}
      >
        <title>{methodName(method)} correlation matrix for {study.name}</title>
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stopColor={CYAN} />
            <stop offset="50%" stopColor="#27334c" />
            <stop offset="100%" stopColor={VIOLET} />
          </linearGradient>
        </defs>
        {columns.map((column, index) => {
          const x = MATRIX_LEFT + index * MATRIX_CELL + MATRIX_CELL / 2;
          return (
            <text
              key={`top-${column}`}
              x={x}
              y={MATRIX_TOP - 12}
              fill={AXIS}
              fontSize="10"
              textAnchor="start"
              transform={`rotate(-45 ${x} ${MATRIX_TOP - 12})`}
            >
              {shortLabel(column)}
            </text>
          );
        })}
        {columns.map((rowColumn, rowIndex) => (
          <g key={rowColumn}>
            <text
              x={MATRIX_LEFT - 12}
              y={MATRIX_TOP + rowIndex * MATRIX_CELL + MATRIX_CELL / 2 + 4}
              fill={AXIS}
              fontSize="10"
              textAnchor="end"
            >
              {shortLabel(rowColumn)}
            </text>
            {columns.map((column, columnIndex) => {
              const result = matrix[rowIndex][columnIndex];
              const selected = xColumn === column && yColumn === rowColumn;
              const x = MATRIX_LEFT + columnIndex * MATRIX_CELL;
              const y = MATRIX_TOP + rowIndex * MATRIX_CELL;
              return (
                <g key={`${rowColumn}-${column}`} className="cursor-pointer" onClick={() => onSelectPair(column, rowColumn)}>
                  <rect
                    x={x + 1}
                    y={y + 1}
                    width={MATRIX_CELL - 2}
                    height={MATRIX_CELL - 2}
                    rx="6"
                    fill={matrixColor(result.value)}
                    stroke={selected ? LIME : "rgba(255,255,255,.045)"}
                    strokeWidth={selected ? "3" : "1"}
                  >
                    <title>
                      {relationshipLabel(column)} × {relationshipLabel(rowColumn)} · {methodName(method)} {result.value == null ? "undefined" : result.value.toFixed(4)} · n={result.count}
                    </title>
                  </rect>
                  <text
                    x={x + MATRIX_CELL / 2}
                    y={y + MATRIX_CELL / 2 + 4}
                    fill={Math.abs(result.value ?? 0) > 0.58 ? "#08101e" : TEXT}
                    fontSize="10"
                    fontWeight="600"
                    textAnchor="middle"
                    pointerEvents="none"
                  >
                    {result.value == null ? "—" : result.value.toFixed(2)}
                  </text>
                </g>
              );
            })}
          </g>
        ))}
        <g transform={`translate(${MATRIX_LEFT} ${height - 25})`} aria-hidden="true">
          <text x="0" y="4" fill={AXIS} fontSize="10">−1</text>
          <rect x="24" y="-7" width="130" height="12" rx="6" fill={`url(#${gradientId})`} />
          <text x="161" y="4" fill={AXIS} fontSize="10">+1</text>
          <text x="197" y="4" fill={AXIS} fontSize="10">Click a cell to inspect the pair</text>
        </g>
      </svg>
    </div>
  );
}

function regression(points: PairPoint[]): { slope: number; intercept: number } | null {
  if (points.length < 3) return null;
  const xMean = points.reduce((sum, point) => sum + point.x, 0) / points.length;
  const yMean = points.reduce((sum, point) => sum + point.y, 0) / points.length;
  const denominator = points.reduce((sum, point) => sum + (point.x - xMean) ** 2, 0);
  if (denominator === 0) return null;
  const numerator = points.reduce((sum, point) => sum + (point.x - xMean) * (point.y - yMean), 0);
  const slope = numerator / denominator;
  return { slope, intercept: yMean - slope * xMean };
}

function CorrelationScatter({
  study,
  xColumn,
  yColumn,
  method,
  selectedRow,
  onSelect,
}: {
  study: Study;
  xColumn: string;
  yColumn: string;
  method: CorrelationMethod;
  selectedRow: TrialRow | null;
  onSelect: (row: TrialRow) => void;
}) {
  const points = pairPoints(study, xColumn, yColumn);
  if (points.length === 0) {
    return (
      <svg data-export-root viewBox={`0 0 ${SCATTER_WIDTH} ${SCATTER_HEIGHT}`} className="block w-full" role="img" aria-label="No paired values available">
        <title>No paired values available</title>
        <text x="480" y="170" fill={TEXT} fontSize="15" textAnchor="middle">No paired numeric values</text>
        <text x="480" y="196" fill={AXIS} fontSize="12" textAnchor="middle">Choose another attribute pair.</text>
      </svg>
    );
  }

  const result = correlation(points, method);
  const xDomain = extent(points.map((point) => point.x));
  const yDomain = extent(points.map((point) => point.y));
  const x = scale(xDomain, [SCATTER_MARGIN.left, SCATTER_WIDTH - SCATTER_MARGIN.right]);
  const y = scale(yDomain, [SCATTER_HEIGHT - SCATTER_MARGIN.bottom, SCATTER_MARGIN.top]);
  const fit = regression(points);
  const clipId = `correlation-scatter-${study.id.replace(/[^a-zA-Z0-9]/g, "")}`;

  return (
    <svg
      data-export-root
      viewBox={`0 0 ${SCATTER_WIDTH} ${SCATTER_HEIGHT}`}
      className="block w-full"
      role="img"
      aria-label={`${relationshipLabel(yColumn)} by ${relationshipLabel(xColumn)} scatter plot`}
    >
      <title>{relationshipLabel(yColumn)} by {relationshipLabel(xColumn)} · {methodName(method)} {result.value == null ? "undefined" : result.value.toFixed(4)}</title>
      <defs>
        <clipPath id={clipId}>
          <rect x={SCATTER_MARGIN.left} y={SCATTER_MARGIN.top} width={SCATTER_WIDTH - SCATTER_MARGIN.left - SCATTER_MARGIN.right} height={SCATTER_HEIGHT - SCATTER_MARGIN.top - SCATTER_MARGIN.bottom} />
        </clipPath>
      </defs>
      {ticks(yDomain, 5).map((value) => (
        <g key={`y-${value}`} aria-hidden="true">
          <line x1={SCATTER_MARGIN.left} x2={SCATTER_WIDTH - SCATTER_MARGIN.right} y1={y(value)} y2={y(value)} stroke={GRID} />
          <text x={SCATTER_MARGIN.left - 12} y={y(value) + 4} fill={AXIS} fontSize="11" textAnchor="end">{axisNumber(value)}</text>
        </g>
      ))}
      {ticks(xDomain, 6).map((value) => (
        <g key={`x-${value}`} aria-hidden="true">
          <line x1={x(value)} x2={x(value)} y1={SCATTER_MARGIN.top} y2={SCATTER_HEIGHT - SCATTER_MARGIN.bottom} stroke={GRID} />
          <text x={x(value)} y={SCATTER_HEIGHT - SCATTER_MARGIN.bottom + 22} fill={AXIS} fontSize="11" textAnchor="middle">{axisNumber(value)}</text>
        </g>
      ))}
      <g clipPath={`url(#${clipId})`}>
        {fit && (
          <line
            x1={x(xDomain[0])}
            y1={y(fit.intercept + fit.slope * xDomain[0])}
            x2={x(xDomain[1])}
            y2={y(fit.intercept + fit.slope * xDomain[1])}
            stroke={CYAN}
            strokeWidth="2"
            strokeDasharray="7 6"
            opacity="0.78"
          />
        )}
        {points.map((point) => {
          const selected = point.row === selectedRow;
          return (
            <circle
              key={`${point.index}-${point.x}-${point.y}`}
              cx={x(point.x)}
              cy={y(point.y)}
              r={selected ? "7" : "4.5"}
              fill={selected ? LIME : VIOLET}
              fillOpacity={selected ? "1" : "0.72"}
              stroke="#10182a"
              strokeWidth="2"
              className="chart-point cursor-pointer"
              onClick={() => onSelect(point.row)}
            >
              <title>
                Trial {point.row[study.config.trial] ?? point.index} · {relationshipLabel(xColumn)} {formatNumber(point.x, 7)} · {relationshipLabel(yColumn)} {formatNumber(point.y, 7)}
              </title>
            </circle>
          );
        })}
      </g>
      <text x={SCATTER_MARGIN.left + 8} y="22" fill={TEXT} fontSize="12" fontWeight="600">
        {methodName(method)} = {result.value == null ? "undefined" : result.value.toFixed(3)} · n={result.count}
      </text>
      <line x1={SCATTER_WIDTH - 164} x2={SCATTER_WIDTH - 142} y1="18" y2="18" stroke={CYAN} strokeWidth="2" strokeDasharray="6 5" />
      <text x={SCATTER_WIDTH - 134} y="22" fill={AXIS} fontSize="10">Linear fit</text>
      <text x={SCATTER_MARGIN.left + (SCATTER_WIDTH - SCATTER_MARGIN.left - SCATTER_MARGIN.right) / 2} y={SCATTER_HEIGHT - 11} fill={AXIS} fontSize="12" textAnchor="middle">{relationshipLabel(xColumn)}</text>
      <text
        x="18"
        y={SCATTER_MARGIN.top + (SCATTER_HEIGHT - SCATTER_MARGIN.top - SCATTER_MARGIN.bottom) / 2}
        fill={AXIS}
        fontSize="12"
        textAnchor="middle"
        transform={`rotate(-90 18 ${SCATTER_MARGIN.top + (SCATTER_HEIGHT - SCATTER_MARGIN.top - SCATTER_MARGIN.bottom) / 2})`}
      >
        {relationshipLabel(yColumn)}
      </text>
    </svg>
  );
}

function preferredColumns(study: Study, usable: string[]): string[] {
  const batchSize = usable.find(isBatchSizeColumn);
  const samplerM = usable.find(isSamplerMColumn);
  const preferred = [
    batchSize,
    study.config.metric,
    samplerM,
    study.config.learningRate,
    study.config.epoch,
    study.config.selectedEpoch,
    study.config.duration,
    ...usable.filter((column) => column.toLowerCase().startsWith("param:")),
    study.config.trial,
    ...usable,
  ].filter((column): column is string => Boolean(column) && usable.includes(column as string));
  return [...new Set(preferred)].slice(0, 8);
}

export function CorrelationExplorer({
  study,
  selectedRow,
  onSelect,
}: {
  study: Study;
  selectedRow: TrialRow | null;
  onSelect: (row: TrialRow) => void;
}) {
  const usable = useMemo(() => usableRelationshipColumns(study), [study]);
  const defaults = useMemo(() => preferredColumns(study, usable), [study, usable]);
  const defaultY = usable.includes(study.config.metric) ? study.config.metric : usable[1] ?? usable[0] ?? "";
  const defaultX = usable.find(isBatchSizeColumn)
    ?? (study.config.learningRate && usable.includes(study.config.learningRate) ? study.config.learningRate : null)
    ?? usable.find((column) => column !== defaultY)
    ?? usable[0]
    ?? "";
  const [method, setMethod] = useState<CorrelationMethod>("spearman");
  const [matrixColumns, setMatrixColumns] = useState<string[]>(defaults);
  const [xColumn, setXColumn] = useState(defaultX);
  const [yColumn, setYColumn] = useState(defaultY);
  const [storageReady, setStorageReady] = useState(false);
  const [attributesOpen, setAttributesOpen] = useState(false);
  const [attributeQuery, setAttributeQuery] = useState("");
  const attributesRef = useRef<HTMLDivElement>(null);
  const storageKey = `trial-atlas:correlations:v2:${study.id}`;
  const usableSignature = usable.join("\u001f");
  const defaultsSignature = defaults.join("\u001f");

  useEffect(() => {
    setStorageReady(false);
    let stored: {
      method?: CorrelationMethod;
      matrixColumns?: string[];
      xColumn?: string;
      yColumn?: string;
    } | null = null;
    try {
      const raw = window.localStorage.getItem(storageKey);
      stored = raw ? JSON.parse(raw) : null;
    } catch {
      stored = null;
    }
    const storedMatrix = Array.isArray(stored?.matrixColumns)
      ? stored.matrixColumns.filter((column): column is string => typeof column === "string" && usable.includes(column))
      : [];
    setMethod(stored?.method === "pearson" ? "pearson" : "spearman");
    setMatrixColumns(storedMatrix.length >= 2 ? storedMatrix : defaults);
    setXColumn(stored?.xColumn && usable.includes(stored.xColumn) ? stored.xColumn : defaultX);
    setYColumn(stored?.yColumn && usable.includes(stored.yColumn) ? stored.yColumn : defaultY);
    setAttributesOpen(false);
    setAttributeQuery("");
    setStorageReady(true);
  }, [defaultsSignature, defaultX, defaultY, storageKey, usableSignature]);

  useEffect(() => {
    if (!storageReady) return;
    try {
      window.localStorage.setItem(storageKey, JSON.stringify({ method, matrixColumns, xColumn, yColumn }));
    } catch {
      // Correlations remain interactive when browser storage is unavailable.
    }
  }, [method, matrixColumns, storageKey, storageReady, xColumn, yColumn]);

  useEffect(() => {
    if (!attributesOpen) return;
    const closeOnPointerDown = (event: PointerEvent) => {
      if (!attributesRef.current?.contains(event.target as Node)) setAttributesOpen(false);
    };
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setAttributesOpen(false);
    };
    document.addEventListener("pointerdown", closeOnPointerDown);
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointerDown);
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [attributesOpen]);

  const filteredAttributes = useMemo(() => {
    const normalized = attributeQuery.trim().toLowerCase();
    if (!normalized) return usable;
    return usable.filter(
      (column) =>
        relationshipSource(column).toLowerCase().includes(normalized)
        || relationshipLabel(column).toLowerCase().includes(normalized),
    );
  }, [attributeQuery, usable]);

  if (usable.length < 2) {
    return (
      <section className="surface px-6 py-10 text-center">
        <Network className="mx-auto text-[#69778f]" size={23} />
        <h2 className="mt-3 text-sm font-semibold text-white">Not enough variation for relationships</h2>
        <p className="mt-2 text-xs text-[#8290a7]">At least two attributes need three paired values and more than one distinct value.</p>
      </section>
    );
  }

  const pairResult = correlation(pairPoints(study, xColumn, yColumn), method);
  const methodDescription = method === "pearson"
    ? "Linear correlation on the original numeric values."
    : "Rank correlation for monotonic relationships and skewed hyperparameters.";
  const hasDecodedBatchSampler = usable.some(
    (column) => column.startsWith(DERIVED_BATCH_SIZE_PREFIX) || column.startsWith(DERIVED_SAMPLER_M_PREFIX),
  );

  return (
    <section className="space-y-6" aria-label="Attribute correlation explorer">
      <div className="surface">
        <div className="flex flex-col gap-4 px-5 py-4 lg:flex-row lg:items-center lg:justify-between sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#9b8cff]/10 text-[#b7adff] ring-1 ring-[#9b8cff]/15">
              <Network size={17} />
            </span>
            <div className="min-w-0">
              <h2 className="text-sm font-semibold text-white">Attribute relationships</h2>
              <p className="mt-1 text-[10px] text-[#74829a]">
                {methodDescription} Missing values are removed separately for each pair.
                {hasDecodedBatchSampler ? " Batch sampler values are split into batch size and sampler m." : ""}
              </p>
            </div>
          </div>
          <div className="flex w-fit items-center rounded-xl border border-white/[0.08] bg-black/10 p-1" aria-label="Correlation method">
            {(["spearman", "pearson"] as CorrelationMethod[]).map((candidate) => (
              <button
                key={candidate}
                type="button"
                onClick={() => setMethod(candidate)}
                className={`focus-ring h-8 rounded-lg px-3 text-[10px] font-semibold transition ${
                  method === candidate ? "bg-[#9b8cff]/16 text-[#c9c2ff]" : "text-[#7f8da4] hover:text-white"
                }`}
                aria-pressed={method === candidate}
              >
                {candidate === "spearman" ? "Spearman" : "Pearson"}
              </button>
            ))}
          </div>
        </div>
        <div className="grid gap-3 border-t border-white/[0.07] p-5 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_auto] sm:p-6">
          <label className="block min-w-0">
            <span className="mb-2 block text-[10px] font-semibold uppercase tracking-[0.12em] text-[#77859d]">Horizontal attribute</span>
            <select value={xColumn} onChange={(event) => setXColumn(event.target.value)} className="focus-ring h-10 w-full appearance-none rounded-xl border border-white/[0.09] bg-[#0c1425] px-3 text-xs text-[#d9e0ed]">
              {usable.map((column) => <option key={column} value={column}>{relationshipLabel(column)}</option>)}
            </select>
          </label>
          <label className="block min-w-0">
            <span className="mb-2 block text-[10px] font-semibold uppercase tracking-[0.12em] text-[#77859d]">Vertical attribute</span>
            <select value={yColumn} onChange={(event) => setYColumn(event.target.value)} className="focus-ring h-10 w-full appearance-none rounded-xl border border-white/[0.09] bg-[#0c1425] px-3 text-xs text-[#d9e0ed]">
              {usable.map((column) => <option key={column} value={column}>{relationshipLabel(column)}</option>)}
            </select>
          </label>
          <div className="relative self-end" ref={attributesRef}>
            <button
              type="button"
              onClick={() => setAttributesOpen((open) => !open)}
              className={`focus-ring flex h-10 w-full items-center justify-center gap-2 rounded-xl border px-3.5 text-xs font-semibold transition md:w-auto ${
                attributesOpen ? "border-[#9b8cff]/35 bg-[#9b8cff]/12 text-[#c8c1ff]" : "border-white/[0.09] bg-white/[0.04] text-[#d8dfec] hover:bg-white/[0.08] hover:text-white"
              }`}
              aria-expanded={attributesOpen}
              aria-haspopup="dialog"
            >
              Matrix attributes
              <span className="rounded-full bg-white/[0.07] px-1.5 py-0.5 text-[9px] tabular-nums text-[#9da9bd]">{matrixColumns.length}/{usable.length}</span>
            </button>
            {attributesOpen && (
              <div role="dialog" aria-label="Choose correlation matrix attributes" className="absolute right-0 top-12 z-50 w-[min(370px,calc(100vw-2rem))] overflow-hidden rounded-2xl border border-white/[0.11] bg-[#111a2e]/98 shadow-[0_24px_80px_rgba(0,0,0,.5)] backdrop-blur-xl">
                <div className="border-b border-white/[0.07] px-4 py-3.5">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <p className="text-xs font-semibold text-white">Matrix attributes</p>
                      <p className="mt-1 text-[10px] text-[#77859d]">Keep at least two attributes selected.</p>
                    </div>
                    <span className="text-[10px] tabular-nums text-[#8d99ad]">{matrixColumns.length} selected</span>
                  </div>
                  <label className="relative mt-3 block">
                    <span className="sr-only">Filter relationship attributes</span>
                    <Search className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#65738b]" size={13} />
                    <input type="search" value={attributeQuery} onChange={(event) => setAttributeQuery(event.target.value)} placeholder="Find an attribute…" className="focus-ring h-9 w-full rounded-xl border border-white/[0.09] bg-black/15 pl-8 pr-3 text-[11px] text-white placeholder:text-[#637087]" />
                  </label>
                </div>
                <div className="max-h-80 overflow-y-auto p-2">
                  {filteredAttributes.map((column) => {
                    const checked = matrixColumns.includes(column);
                    const locked = checked && matrixColumns.length <= 2;
                    return (
                      <label key={column} className={`flex items-start gap-3 rounded-xl px-3 py-2.5 transition ${checked ? "bg-[#9b8cff]/[0.09]" : "hover:bg-white/[0.035]"} ${locked ? "cursor-not-allowed opacity-60" : "cursor-pointer"}`}>
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={locked}
                          onChange={() => setMatrixColumns((current) => current.includes(column) ? current.filter((candidate) => candidate !== column) : [...current, column])}
                          className="mt-0.5 h-4 w-4 shrink-0 rounded border-white/20 bg-black/20 accent-[#9b8cff]"
                        />
                        <span className="min-w-0">
                          <span className="block text-[11px] font-semibold text-[#dce3ef]">{relationshipLabel(column)}</span>
                          <span className="mt-0.5 block truncate text-[9px] text-[#68768e]" title={relationshipSource(column)}>{relationshipSource(column)}</span>
                        </span>
                      </label>
                    );
                  })}
                </div>
                <div className="flex items-center justify-between gap-2 border-t border-white/[0.07] px-3 py-3">
                  <button type="button" onClick={() => setMatrixColumns(defaults)} className="focus-ring inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-[10px] font-semibold text-[#9ca8bb] transition hover:bg-white/[0.05] hover:text-white">
                    <RotateCcw size={12} /> Defaults
                  </button>
                  <button type="button" onClick={() => setMatrixColumns([...usable])} className="focus-ring h-8 rounded-lg bg-[#9b8cff]/12 px-2.5 text-[10px] font-semibold text-[#c5bdff] transition hover:bg-[#9b8cff]/20">
                    Select all
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      <ChartBoard
        storageKey={`trial-atlas:charts:${study.id}:correlations`}
        items={[
          {
            id: "correlation-matrix",
            title: "Correlation matrix",
            description: `${methodName(method)} across the selected numeric and sampler-derived attributes. Click a cell to open that pair below.`,
            exportName: `${study.name}-${method}-correlation-matrix`,
            className: "xl:col-span-2",
            content: (
              <CorrelationMatrix
                study={study}
                columns={matrixColumns}
                method={method}
                xColumn={xColumn}
                yColumn={yColumn}
                onSelectPair={(nextX, nextY) => {
                  setXColumn(nextX);
                  setYColumn(nextY);
                }}
              />
            ),
          },
          {
            id: "correlation-scatter",
            title: `${relationshipLabel(yColumn)} × ${relationshipLabel(xColumn)}`,
            description: `${methodName(method)} ${pairResult.value == null ? "is undefined" : `= ${pairResult.value.toFixed(3)}`} across ${pairResult.count} paired trials. The dashed line is the least-squares fit.`,
            exportName: `${study.name}-${yColumn}-by-${xColumn}`,
            className: "xl:col-span-2",
            content: (
              <CorrelationScatter
                study={study}
                xColumn={xColumn}
                yColumn={yColumn}
                method={method}
                selectedRow={selectedRow}
                onSelect={onSelect}
              />
            ),
          },
        ]}
      />
    </section>
  );
}
