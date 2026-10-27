"use client";

import { useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";
import { ChevronDown, Download, FileImage, FileType2, GripVertical, LoaderCircle, Maximize2, Minimize2 } from "lucide-react";
import { exportPng, exportSvg } from "@/lib/export";
import {
  columnLabel,
  formatMetric,
  formatNumber,
  getStudyStats,
  numericValue,
  type Study,
  type TrialRow,
} from "@/lib/trials";

const COLOR = {
  grid: "#27334c",
  axis: "#77859f",
  text: "#dce3f1",
  muted: "#8f9bb2",
  violet: "#9b8cff",
  violetSoft: "rgba(155, 140, 255, 0.16)",
  cyan: "#4fd5e7",
  lime: "#b8e56e",
  amber: "#f4c86f",
};

const WIDTH = 960;
const HEIGHT = 340;
const MARGIN = { top: 28, right: 28, bottom: 58, left: 76 };
const INNER_WIDTH = WIDTH - MARGIN.left - MARGIN.right;
const INNER_HEIGHT = HEIGHT - MARGIN.top - MARGIN.bottom;

interface Point {
  x: number;
  y: number;
  row: TrialRow;
  index: number;
}

function pointsFor(study: Study, xColumn: string | null, yColumn: string | null): Point[] {
  if (!xColumn || !yColumn) return [];
  return study.rows.flatMap((row, index) => {
    const x = numericValue(row[xColumn]);
    const y = numericValue(row[yColumn]);
    return x == null || y == null ? [] : [{ x, y, row, index }];
  });
}

function extent(values: number[], padding = 0.06): [number, number] {
  if (values.length === 0) return [0, 1];
  let minimum = Math.min(...values);
  let maximum = Math.max(...values);
  if (minimum === maximum) {
    const delta = Math.abs(minimum) * 0.05 || 1;
    return [minimum - delta, maximum + delta];
  }
  const delta = (maximum - minimum) * padding;
  minimum -= delta;
  maximum += delta;
  return [minimum, maximum];
}

function linearScale(domain: [number, number], range: [number, number]) {
  const span = domain[1] - domain[0] || 1;
  return (value: number) => range[0] + ((value - domain[0]) / span) * (range[1] - range[0]);
}

function logScale(domain: [number, number], range: [number, number]) {
  const safeDomain: [number, number] = [Math.log10(domain[0]), Math.log10(domain[1])];
  return linearScale(safeDomain, range);
}

function ticks(domain: [number, number], count = 5): number[] {
  return Array.from({ length: count }, (_, index) => domain[0] + ((domain[1] - domain[0]) * index) / (count - 1));
}

function logTicks(domain: [number, number], count = 6): number[] {
  const low = Math.log10(domain[0]);
  const high = Math.log10(domain[1]);
  return Array.from({ length: count }, (_, index) => 10 ** (low + ((high - low) * index) / (count - 1)));
}

function axisNumber(value: number): string {
  const absolute = Math.abs(value);
  if (value !== 0 && (absolute < 0.001 || absolute >= 10_000)) return value.toExponential(1);
  if (absolute >= 100) return Math.round(value).toLocaleString("en-US");
  if (absolute >= 1) return Number(value.toFixed(1)).toString();
  return Number(value.toFixed(3)).toString();
}

function linePath(points: Point[], x: (value: number) => number, y: (value: number) => number): string {
  return points.map((point, index) => `${index === 0 ? "M" : "L"}${x(point.x)},${y(point.y)}`).join(" ");
}

interface FrameProps {
  xValues: number[];
  yValues: number[];
  x: (value: number) => number;
  y: (value: number) => number;
  xLabel: string;
  yLabel: string;
}

function CartesianFrame({ xValues, yValues, x, y, xLabel, yLabel }: FrameProps) {
  return (
    <g aria-hidden="true">
      {yValues.map((value) => (
        <g key={`y-${value}`}>
          <line
            x1={MARGIN.left}
            x2={WIDTH - MARGIN.right}
            y1={y(value)}
            y2={y(value)}
            stroke={COLOR.grid}
            strokeWidth="1"
          />
          <text x={MARGIN.left - 12} y={y(value) + 4} fill={COLOR.axis} fontSize="11" textAnchor="end">
            {axisNumber(value)}
          </text>
        </g>
      ))}
      {xValues.map((value) => (
        <g key={`x-${value}`}>
          <line
            x1={x(value)}
            x2={x(value)}
            y1={MARGIN.top}
            y2={HEIGHT - MARGIN.bottom}
            stroke={COLOR.grid}
            strokeWidth="1"
          />
          <text x={x(value)} y={HEIGHT - MARGIN.bottom + 22} fill={COLOR.axis} fontSize="11" textAnchor="middle">
            {axisNumber(value)}
          </text>
        </g>
      ))}
      <text x={MARGIN.left + INNER_WIDTH / 2} y={HEIGHT - 10} fill={COLOR.muted} fontSize="12" textAnchor="middle">
        {columnLabel(xLabel)}
      </text>
      <text
        x="17"
        y={MARGIN.top + INNER_HEIGHT / 2}
        fill={COLOR.muted}
        fontSize="12"
        textAnchor="middle"
        transform={`rotate(-90 17 ${MARGIN.top + INNER_HEIGHT / 2})`}
      >
        {columnLabel(yLabel)}
      </text>
    </g>
  );
}

function EmptyDiagram({ label }: { label: string }) {
  return (
    <svg
      data-export-root
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="block w-full"
      role="img"
      aria-label={label}
    >
      <title>{label}</title>
      <rect x="240" y="115" width="480" height="108" rx="18" fill="rgba(255,255,255,0.025)" stroke={COLOR.grid} />
      <text x="480" y="164" fill={COLOR.text} fontSize="15" textAnchor="middle">
        No plottable values
      </text>
      <text x="480" y="190" fill={COLOR.muted} fontSize="12" textAnchor="middle">
        Choose a compatible column in Configure columns.
      </text>
    </svg>
  );
}

interface DiagramCardProps {
  title: string;
  description: string;
  exportName: string;
  children: ReactNode;
  className?: string;
  collapsed?: boolean;
  onCollapsedChange?: (collapsed: boolean) => void;
  reorderable?: boolean;
  onMove?: (direction: -1 | 1) => void;
  canMoveUp?: boolean;
  canMoveDown?: boolean;
}

export function DiagramCard({
  title,
  description,
  exportName,
  children,
  className = "",
  collapsed,
  onCollapsedChange,
  reorderable = false,
  onMove,
  canMoveUp = false,
  canMoveDown = false,
}: DiagramCardProps) {
  const diagramRef = useRef<HTMLDivElement>(null);
  const cardRef = useRef<HTMLElement>(null);
  const [exporting, setExporting] = useState<"svg" | "png" | null>(null);
  const [localCollapsed, setLocalCollapsed] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [placeholderHeight, setPlaceholderHeight] = useState(80);
  const isCollapsed = collapsed ?? localCollapsed;

  const setCollapsed = (value: boolean) => {
    if (collapsed === undefined) setLocalCollapsed(value);
    onCollapsedChange?.(value);
  };

  const handleExport = async (format: "svg" | "png") => {
    if (!diagramRef.current) return;
    setExporting(format);
    try {
      if (format === "svg") exportSvg(diagramRef.current, exportName);
      else await exportPng(diagramRef.current, exportName);
    } finally {
      setExporting(null);
    }
  };

  useEffect(() => {
    if (!expanded) return;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key === "Escape") setExpanded(false);
    };
    document.addEventListener("keydown", closeOnEscape);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", closeOnEscape);
    };
  }, [expanded]);

  const openExpanded = () => {
    setPlaceholderHeight(cardRef.current?.getBoundingClientRect().height ?? 80);
    if (isCollapsed) setCollapsed(false);
    setExpanded(true);
  };

  const renderCard = (expandedMode: boolean) => (
    <section
      ref={cardRef}
      role={expandedMode ? "dialog" : undefined}
      aria-modal={expandedMode ? true : undefined}
      aria-label={expandedMode ? `${title} expanded diagram` : undefined}
      className={`surface overflow-hidden ${
        expandedMode
          ? "fixed inset-2 z-[100] flex min-h-0 flex-col shadow-[0_36px_120px_rgba(0,0,0,.72)] sm:inset-5"
          : className
      }`}
    >
      <header className={`flex shrink-0 flex-wrap items-start justify-between gap-4 px-5 py-4 sm:px-6 ${isCollapsed ? "" : "border-b border-white/[0.07]"}`}>
        <div className="flex min-w-0 items-start gap-2">
          {reorderable && !expandedMode && (
            <button
              type="button"
              data-chart-drag-handle
              draggable
              onKeyDown={(event) => {
                if (event.key === "ArrowUp" && canMoveUp) {
                  event.preventDefault();
                  onMove?.(-1);
                }
                if (event.key === "ArrowDown" && canMoveDown) {
                  event.preventDefault();
                  onMove?.(1);
                }
              }}
              className="focus-ring -ml-2 flex h-8 w-7 shrink-0 cursor-grab items-center justify-center rounded-lg text-[#596780] transition hover:bg-white/[0.05] hover:text-[#aeb8ca] active:cursor-grabbing"
              aria-label={`Drag ${title} to reorder. Use the up and down arrow keys to move it.`}
              title="Drag to reorder"
            >
              <GripVertical size={15} aria-hidden="true" />
            </button>
          )}
          <div className="min-w-0">
            <div className="flex items-center gap-2.5">
              <span className="h-2 w-2 rounded-full bg-[#9b8cff] shadow-[0_0_16px_rgba(155,140,255,.8)]" />
              <h2 className="text-[15px] font-semibold tracking-[-0.01em] text-white">{title}</h2>
            </div>
            {!isCollapsed && <p className="mt-1.5 text-xs leading-5 text-[#8f9bb2]">{description}</p>}
          </div>
        </div>
        <div className="flex items-center gap-1">
          {!isCollapsed && (
            <div className="flex items-center gap-1 rounded-xl border border-white/[0.08] bg-black/10 p-1" aria-label="Export diagram">
              <span className="hidden items-center gap-1.5 px-2 text-[10px] font-semibold uppercase tracking-[0.13em] text-[#78859d] sm:flex">
                <Download size={12} aria-hidden="true" /> Export
              </span>
              <button
                type="button"
                onClick={() => void handleExport("svg")}
                className="focus-ring flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-xs font-medium text-[#bac4d7] transition hover:bg-white/[0.07] hover:text-white"
                aria-label={`Export ${title} as SVG`}
              >
                {exporting === "svg" ? <LoaderCircle size={13} className="animate-spin" /> : <FileType2 size={13} />}
                SVG
              </button>
              <button
                type="button"
                onClick={() => void handleExport("png")}
                className="focus-ring flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-xs font-medium text-[#bac4d7] transition hover:bg-white/[0.07] hover:text-white"
                aria-label={`Export ${title} as PNG`}
              >
                {exporting === "png" ? <LoaderCircle size={13} className="animate-spin" /> : <FileImage size={13} />}
                PNG
              </button>
            </div>
          )}
          <button
            type="button"
            autoFocus={expandedMode}
            onClick={() => expandedMode ? setExpanded(false) : openExpanded()}
            className="focus-ring flex h-10 w-10 items-center justify-center rounded-xl border border-white/[0.08] bg-black/10 text-[#8f9bb2] transition hover:bg-white/[0.06] hover:text-white"
            aria-label={`${expandedMode ? "Exit full screen" : "Open full screen"} for ${title}`}
            title={expandedMode ? "Exit full screen" : "Make chart bigger"}
          >
            {expandedMode ? <Minimize2 size={16} aria-hidden="true" /> : <Maximize2 size={16} aria-hidden="true" />}
          </button>
          {!expandedMode && (
            <button
              type="button"
              onClick={() => setCollapsed(!isCollapsed)}
              className="focus-ring flex h-10 w-10 items-center justify-center rounded-xl border border-white/[0.08] bg-black/10 text-[#8f9bb2] transition hover:bg-white/[0.06] hover:text-white"
              aria-label={`${isCollapsed ? "Show" : "Hide"} ${title} content`}
              aria-expanded={!isCollapsed}
              title={isCollapsed ? "Expand chart content" : "Collapse chart content"}
            >
              <ChevronDown size={16} className={`transition-transform ${isCollapsed ? "" : "rotate-180"}`} aria-hidden="true" />
            </button>
          )}
        </div>
      </header>
      {!isCollapsed && (
        <div
          ref={diagramRef}
          className={expandedMode ? "min-h-0 flex-1 overflow-auto px-3 py-4 sm:px-6 sm:py-5" : "px-2 pb-2 pt-3 sm:px-4"}
        >
          <div className={expandedMode ? "mx-auto w-full max-w-[1900px]" : ""}>{children}</div>
        </div>
      )}
    </section>
  );

  if (expanded && typeof document !== "undefined") {
    return (
      <>
        <div className="surface" style={{ height: `${placeholderHeight}px` }} aria-hidden="true" />
        {createPortal(
          <>
            <button
              type="button"
              className="fixed inset-0 z-[90] cursor-default bg-[#03060d]/80 backdrop-blur-md"
              onClick={() => setExpanded(false)}
              aria-label={`Close expanded ${title} diagram`}
            />
            {renderCard(true)}
          </>,
          document.body,
        )}
      </>
    );
  }

  return renderCard(false);
}

interface StudyChartProps {
  study: Study;
  selectedRow: TrialRow | null;
  selectedRows?: TrialRow[];
  onSelect: (row: TrialRow) => void;
}

export function DevelopmentChart({ study, selectedRow, selectedRows = [], onSelect }: StudyChartProps) {
  const points = useMemo(
    () => pointsFor(study, study.config.trial, study.config.metric).sort((a, b) => a.x - b.x),
    [study],
  );

  if (points.length === 0) return <EmptyDiagram label="Trial development chart with no plottable values" />;

  const bestPoints: Point[] = [];
  let best = Number.NEGATIVE_INFINITY;
  for (const point of points) {
    best = Math.max(best, point.y);
    bestPoints.push({ ...point, y: best });
  }
  const xDomain = extent(points.map((point) => point.x), 0.02);
  const yDomain = extent([...points, ...bestPoints].map((point) => point.y), 0.12);
  const x = linearScale(xDomain, [MARGIN.left, WIDTH - MARGIN.right]);
  const y = linearScale(yDomain, [HEIGHT - MARGIN.bottom, MARGIN.top]);
  const metricPath = linePath(points, x, y);
  const bestPath = linePath(bestPoints, x, y);
  const areaPath = `${metricPath} L${x(points.at(-1)!.x)},${HEIGHT - MARGIN.bottom} L${x(points[0].x)},${HEIGHT - MARGIN.bottom} Z`;
  const gradientId = `metric-area-${study.id.replace(/[^a-zA-Z0-9]/g, "")}`;

  return (
    <svg
      data-export-root
      viewBox={`0 0 ${WIDTH} ${HEIGHT}`}
      className="block w-full"
      role="img"
      aria-labelledby={`development-title-${study.id}`}
    >
      <title id={`development-title-${study.id}`}>Metric and cumulative best by trial for {study.name}</title>
      <defs>
        <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={COLOR.violet} stopOpacity="0.28" />
          <stop offset="100%" stopColor={COLOR.violet} stopOpacity="0" />
        </linearGradient>
      </defs>
      <CartesianFrame
        xValues={ticks(xDomain, 6)}
        yValues={ticks(yDomain, 5)}
        x={x}
        y={y}
        xLabel={study.config.trial}
        yLabel={study.config.metric}
      />
      <path d={areaPath} fill={`url(#${gradientId})`} />
      <path d={metricPath} fill="none" stroke={COLOR.violet} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" />
      <path d={bestPath} fill="none" stroke={COLOR.lime} strokeWidth="2" strokeDasharray="7 7" strokeLinecap="round" />
      {points.map((point) => {
        const active = selectedRow === point.row;
        const included = selectedRows.includes(point.row);
        return (
          <g key={`${point.x}-${point.index}`} onClick={() => onSelect(point.row)} className="cursor-pointer">
            {included && <circle cx={x(point.x)} cy={y(point.y)} r={active ? 10 : 8.5} fill="none" stroke={active ? COLOR.lime : COLOR.cyan} strokeWidth="2" />}
            <circle
              className="chart-point"
              cx={x(point.x)}
              cy={y(point.y)}
              r="4.5"
              fill={active ? COLOR.lime : included ? COLOR.cyan : COLOR.violet}
              stroke="#10182a"
              strokeWidth="2"
            >
              <title>
                Trial {point.row[study.config.trial]} · {columnLabel(study.config.metric)} {formatMetric(point.y)}
              </title>
            </circle>
          </g>
        );
      })}
      <g transform={`translate(${MARGIN.left + 8} ${MARGIN.top + 9})`} aria-hidden="true">
        <circle r="4" fill={COLOR.violet} />
        <text x="11" y="4" fill={COLOR.text} fontSize="11">Metric</text>
        <line x1="74" x2="96" y1="0" y2="0" stroke={COLOR.lime} strokeWidth="2" strokeDasharray="6 5" />
        <text x="104" y="4" fill={COLOR.text} fontSize="11">Best so far</text>
      </g>
    </svg>
  );
}

export function EpochChart({ study, selectedRow, selectedRows = [], onSelect }: StudyChartProps) {
  const epochPoints = useMemo(() => pointsFor(study, study.config.trial, study.config.epoch), [study]);
  const selectedPoints = useMemo(() => pointsFor(study, study.config.trial, study.config.selectedEpoch), [study]);
  const allPoints = [...epochPoints, ...selectedPoints];
  if (allPoints.length === 0) return <EmptyDiagram label="Epoch chart with no plottable values" />;

  const xDomain = extent(allPoints.map((point) => point.x), 0.02);
  const yDomain = extent(allPoints.map((point) => point.y), 0.08);
  const x = linearScale(xDomain, [MARGIN.left, WIDTH - MARGIN.right]);
  const y = linearScale(yDomain, [HEIGHT - MARGIN.bottom, MARGIN.top]);
  const sortedEpochs = [...epochPoints].sort((a, b) => a.x - b.x);

  return (
    <svg data-export-root viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="block w-full" role="img" aria-label={`Epochs by trial for ${study.name}`}>
      <title>Epoch count and selected epoch by trial for {study.name}</title>
      <CartesianFrame
        xValues={ticks(xDomain, 6)}
        yValues={ticks(yDomain, 5)}
        x={x}
        y={y}
        xLabel={study.config.trial}
        yLabel="Epoch"
      />
      {sortedEpochs.length > 1 && (
        <path d={linePath(sortedEpochs, x, y)} fill="none" stroke={COLOR.cyan} strokeWidth="2.25" strokeLinejoin="round" />
      )}
      {epochPoints.map((point) => (
        <circle
          key={`epoch-${point.index}`}
          className="chart-point cursor-pointer"
          cx={x(point.x)}
          cy={y(point.y)}
          r={selectedRow === point.row ? 7 : selectedRows.includes(point.row) ? 6 : 4.5}
          fill={selectedRow === point.row ? COLOR.lime : selectedRows.includes(point.row) ? COLOR.violet : COLOR.cyan}
          stroke="#10182a"
          strokeWidth="2"
          onClick={() => onSelect(point.row)}
        >
          <title>Trial {point.row[study.config.trial]} · {columnLabel(study.config.epoch)} {formatNumber(point.y, 6)}</title>
        </circle>
      ))}
      {selectedPoints.map((point) => {
        const centerX = x(point.x);
        const centerY = y(point.y);
        const radius = selectedRow === point.row ? 8 : selectedRows.includes(point.row) ? 7 : 5.5;
        return (
          <path
            key={`selected-${point.index}`}
            d={`M${centerX},${centerY - radius} L${centerX + radius},${centerY} L${centerX},${centerY + radius} L${centerX - radius},${centerY} Z`}
            fill={COLOR.violet}
            stroke="#10182a"
            strokeWidth="2"
            className="cursor-pointer"
            onClick={() => onSelect(point.row)}
          >
            <title>Trial {point.row[study.config.trial]} · Selected epoch {formatNumber(point.y, 6)}</title>
          </path>
        );
      })}
      <g transform={`translate(${MARGIN.left + 8} ${MARGIN.top + 9})`} aria-hidden="true">
        <circle r="4" fill={COLOR.cyan} />
        <text x="11" y="4" fill={COLOR.text} fontSize="11">Last epoch</text>
        <path d="M94,-5 L99,0 L94,5 L89,0 Z" fill={COLOR.violet} />
        <text x="107" y="4" fill={COLOR.text} fontSize="11">Selected epoch</text>
      </g>
    </svg>
  );
}

export function LearningRateChart({ study, selectedRow, selectedRows = [], onSelect }: StudyChartProps) {
  const points = useMemo(() => pointsFor(study, study.config.learningRate, study.config.metric).filter((point) => point.x > 0), [study]);
  if (points.length === 0) return <EmptyDiagram label="Learning-rate chart with no plottable values" />;

  const rawX: [number, number] = [Math.min(...points.map((point) => point.x)), Math.max(...points.map((point) => point.x))];
  const xDomain: [number, number] = rawX[0] === rawX[1] ? [rawX[0] / 10, rawX[1] * 10] : rawX;
  const yDomain = extent(points.map((point) => point.y), 0.12);
  const xLog = logScale(xDomain, [MARGIN.left, WIDTH - MARGIN.right]);
  const x = (value: number) => xLog(Math.log10(value));
  const y = linearScale(yDomain, [HEIGHT - MARGIN.bottom, MARGIN.top]);

  return (
    <svg data-export-root viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="block w-full" role="img" aria-label={`Metric by learning rate for ${study.name}`}>
      <title>{columnLabel(study.config.metric)} by {columnLabel(study.config.learningRate)} for {study.name}</title>
      <CartesianFrame
        xValues={logTicks(xDomain, 6)}
        yValues={ticks(yDomain, 5)}
        x={x}
        y={y}
        xLabel={study.config.learningRate ?? "Learning rate"}
        yLabel={study.config.metric}
      />
      {points.map((point) => {
        const active = selectedRow === point.row;
        const included = selectedRows.includes(point.row);
        return (
          <g key={`lr-${point.index}`} className="cursor-pointer" onClick={() => onSelect(point.row)}>
            {included && <circle cx={x(point.x)} cy={y(point.y)} r={active ? 11 : 9.5} fill={COLOR.violetSoft} stroke={active ? COLOR.lime : COLOR.cyan} strokeWidth="1.5" />}
            <circle
              className="chart-point"
              cx={x(point.x)}
              cy={y(point.y)}
              r={active ? 7 : included ? 6.5 : 5.5}
              fill={active ? COLOR.lime : included ? COLOR.violet : COLOR.cyan}
              fillOpacity={included ? 1 : 0.78}
              stroke="#10182a"
              strokeWidth="2"
            >
              <title>Trial {point.row[study.config.trial]} · LR {formatNumber(point.x)} · Metric {formatMetric(point.y)}</title>
            </circle>
          </g>
        );
      })}
    </svg>
  );
}

export function DistributionChart({ study }: { study: Study }) {
  const values = useMemo(
    () => study.rows.map((row) => numericValue(row[study.config.metric])).filter((value): value is number => value != null),
    [study],
  );
  if (values.length === 0) return <EmptyDiagram label="Metric distribution with no plottable values" />;

  const rawMinimum = Math.min(...values);
  const rawMaximum = Math.max(...values);
  const xDomain = extent(values, 0.03);
  const binCount = Math.min(14, Math.max(7, Math.round(Math.sqrt(values.length) * 2)));
  const span = rawMaximum - rawMinimum || 1;
  const binWidth = span / binCount;
  const bins = Array.from({ length: binCount }, (_, index) => ({
    from: rawMinimum + index * binWidth,
    to: rawMinimum + (index + 1) * binWidth,
    count: 0,
  }));
  for (const value of values) {
    const index = Math.min(binCount - 1, Math.max(0, Math.floor((value - rawMinimum) / binWidth)));
    bins[index].count += 1;
  }
  const yDomain: [number, number] = [0, Math.max(...bins.map((bin) => bin.count), 1) * 1.12];
  const x = linearScale(xDomain, [MARGIN.left, WIDTH - MARGIN.right]);
  const y = linearScale(yDomain, [HEIGHT - MARGIN.bottom, MARGIN.top]);
  const gap = 3;

  return (
    <svg data-export-root viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="block w-full" role="img" aria-label={`Distribution of ${columnLabel(study.config.metric)} for ${study.name}`}>
      <title>Distribution of {columnLabel(study.config.metric)} for {study.name}</title>
      <CartesianFrame
        xValues={ticks(xDomain, 6)}
        yValues={ticks(yDomain, 5)}
        x={x}
        y={y}
        xLabel={study.config.metric}
        yLabel="Trial count"
      />
      {bins.map((bin, index) => {
        const left = x(bin.from) + gap / 2;
        const right = x(bin.to) - gap / 2;
        const top = y(bin.count);
        return (
          <rect
            key={index}
            x={left}
            y={top}
            width={Math.max(1, right - left)}
            height={HEIGHT - MARGIN.bottom - top}
            rx="3"
            fill={COLOR.violet}
            fillOpacity="0.72"
          >
            <title>{bin.count} trials · {formatMetric(bin.from)} to {formatMetric(bin.to)}</title>
          </rect>
        );
      })}
    </svg>
  );
}

export function StudyComparisonChart({ studies, onSelect }: { studies: Study[]; onSelect: (study: Study) => void }) {
  const records = studies
    .map((study) => ({ study, value: getStudyStats(study).bestMetric }))
    .filter((record): record is { study: Study; value: number } => record.value != null)
    .sort((a, b) => b.value - a.value);
  const chartHeight = Math.max(300, records.length * 54 + 94);
  const left = 190;
  const right = 72;
  const top = 24;
  const bottom = 54;
  const innerWidth = WIDTH - left - right;
  const innerHeight = chartHeight - top - bottom;

  if (records.length === 0) return <EmptyDiagram label="Study comparison with no plottable values" />;

  const maximum = Math.max(...records.map((record) => record.value));
  const minimum = Math.min(...records.map((record) => record.value));
  const domainMinimum = minimum >= 0 && maximum <= 1 ? 0 : Math.min(0, minimum * 1.05);
  const domain: [number, number] = [domainMinimum, maximum * 1.1 || 1];
  const x = linearScale(domain, [left, WIDTH - right]);
  const barHeight = Math.min(28, (innerHeight / records.length) * 0.56);
  const rowHeight = innerHeight / records.length;

  return (
    <svg data-export-root viewBox={`0 0 ${WIDTH} ${chartHeight}`} className="block w-full" role="img" aria-label="Best observed metric by study">
      <title>Best observed metric by study</title>
      {ticks(domain, 6).map((value) => (
        <g key={value} aria-hidden="true">
          <line x1={x(value)} x2={x(value)} y1={top} y2={chartHeight - bottom} stroke={COLOR.grid} />
          <text x={x(value)} y={chartHeight - 24} fill={COLOR.axis} fontSize="11" textAnchor="middle">{axisNumber(value)}</text>
        </g>
      ))}
      {records.map((record, index) => {
        const centerY = top + rowHeight * index + rowHeight / 2;
        const barStart = x(domainMinimum);
        const barEnd = x(record.value);
        const isBest = index === 0;
        return (
          <g key={record.study.id} className="cursor-pointer" onClick={() => onSelect(record.study)}>
            <text x={left - 16} y={centerY + 4} fill={COLOR.text} fontSize="12" textAnchor="end">{record.study.name}</text>
            <rect x={barStart} y={centerY - barHeight / 2} width={Math.max(2, barEnd - barStart)} height={barHeight} rx={barHeight / 2} fill={isBest ? COLOR.lime : COLOR.violet} fillOpacity={isBest ? 0.9 : 0.72}>
              <title>{record.study.name} · best metric {formatMetric(record.value)}</title>
            </rect>
            <text x={Math.min(WIDTH - 12, barEnd + 10)} y={centerY + 4} fill={isBest ? COLOR.lime : COLOR.text} fontSize="12" fontWeight="600">
              {formatMetric(record.value)}
            </text>
          </g>
        );
      })}
      <text x={left + innerWidth / 2} y={chartHeight - 8} fill={COLOR.muted} fontSize="12" textAnchor="middle">Best observed metric</text>
    </svg>
  );
}
