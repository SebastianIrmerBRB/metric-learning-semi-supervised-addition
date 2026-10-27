"use client";

import { useCallback, useEffect, useMemo, useState, type PointerEvent as ReactPointerEvent } from "react";
import { Database, LoaderCircle, RefreshCw, Server, SlidersHorizontal, TriangleAlert, Wifi, X } from "lucide-react";
import { ChartBoard } from "@/components/ChartBoard";
import {
  fetchTensorboardTrial,
  trialLogDirectory,
  type ScalarPoint,
  type TensorboardSeries,
  type TensorboardTrialData,
} from "@/lib/tensorboard";
import { formatDuration, formatNumber, type Study, type TrialRow } from "@/lib/trials";

const WIDTH = 960;
const HEIGHT = 320;
const MARGIN = { top: 38, right: 50, bottom: 54, left: 76 };
const COLORS = ["#9b8cff", "#4fd5e7", "#b8e56e", "#f4c86f", "#fb8ca2", "#7aa7ff"];
const GRID = "#27334c";
const AXIS = "#77859f";
const TEXT = "#dce3f1";

interface DisplaySeries {
  label: string;
  source: TensorboardSeries;
  points: ScalarPoint[];
}

interface HoveredPoint {
  seriesIndex: number;
  pointIndex: number;
}

function extent(values: number[], padding = 0.08): [number, number] {
  if (values.length === 0) return [0, 1];
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  if (minimum === maximum) {
    const delta = Math.abs(minimum) * 0.06 || 1;
    return [minimum - delta, maximum + delta];
  }
  const delta = (maximum - minimum) * padding;
  return [minimum - delta, maximum + delta];
}

function scale(domain: [number, number], range: [number, number]) {
  const span = domain[1] - domain[0] || 1;
  return (value: number) => range[0] + ((value - domain[0]) / span) * (range[1] - range[0]);
}

function ticks(domain: [number, number], count = 6) {
  return Array.from({ length: count }, (_, index) => domain[0] + ((domain[1] - domain[0]) * index) / (count - 1));
}

function axisNumber(value: number): string {
  const absolute = Math.abs(value);
  if (value !== 0 && (absolute < 0.001 || absolute >= 10_000)) return value.toExponential(1);
  if (absolute >= 100) return Math.round(value).toLocaleString("en-US");
  if (absolute >= 1) return Number(value.toFixed(2)).toString();
  return Number(value.toFixed(4)).toString();
}

function smooth(points: ScalarPoint[], windowSize: number): ScalarPoint[] {
  if (windowSize <= 1 || points.length < 3) return points;
  let sum = 0;
  return points.map((point, index) => {
    sum += point[1];
    if (index >= windowSize) sum -= points[index - windowSize][1];
    const count = Math.min(index + 1, windowSize);
    return [point[0], sum / count, point[2]];
  });
}

function shortText(value: string, length = 38): string {
  return value.length > length ? `${value.slice(0, length - 1)}…` : value;
}

function wallTimeLabel(wallTime: number): string {
  if (!Number.isFinite(wallTime) || wallTime <= 0) return "—";
  return new Date(wallTime * 1_000).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function CurveSvg({ series, yLabel, smoothing = 1 }: { series: DisplaySeries[]; yLabel: string; smoothing?: number }) {
  const [hovered, setHovered] = useState<HoveredPoint | null>(null);
  const visible = useMemo(
    () =>
      series
        .map((entry) => ({ ...entry, points: smooth(entry.points, smoothing) }))
        .filter((entry) => entry.points.length > 0),
    [series, smoothing],
  );

  useEffect(() => setHovered(null), [series, smoothing]);

  if (visible.length === 0) {
    return (
      <svg data-export-root viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="block w-full" role="img" aria-label={`${yLabel} is not available`}>
        <title>{yLabel} is not available for this trial</title>
        <rect x="238" y="105" width="484" height="112" rx="18" fill="rgba(255,255,255,.025)" stroke={GRID} />
        <text x="480" y="155" fill={TEXT} fontSize="15" textAnchor="middle">No {yLabel.toLowerCase()} scalar was logged</text>
        <text x="480" y="181" fill={AXIS} fontSize="12" textAnchor="middle">The other available training curves remain usable.</text>
      </svg>
    );
  }

  const allPoints = visible.flatMap((entry) => entry.points);
  const xDomain = extent(allPoints.map((point) => point[0]), 0.015);
  const yDomain = extent(allPoints.map((point) => point[1]), 0.1);
  const x = scale(xDomain, [MARGIN.left, WIDTH - MARGIN.right]);
  const y = scale(yDomain, [HEIGHT - MARGIN.bottom, MARGIN.top]);
  const xLabel = visible[0].source.x_label;
  const hoveredSeries = hovered ? visible[hovered.seriesIndex] : null;
  const hoveredPoint = hoveredSeries && hovered ? hoveredSeries.points[hovered.pointIndex] : null;
  const rawPoint = hoveredSeries && hovered ? hoveredSeries.source.points[hovered.pointIndex] : null;
  const hoverColor = hovered ? COLORS[hovered.seriesIndex % COLORS.length] : COLORS[0];

  const inspectPoint = (event: ReactPointerEvent<SVGRectElement>) => {
    const svg = event.currentTarget.ownerSVGElement;
    if (!svg) return;
    const bounds = svg.getBoundingClientRect();
    if (bounds.width === 0 || bounds.height === 0) return;
    const pointerX = ((event.clientX - bounds.left) / bounds.width) * WIDTH;
    const pointerY = ((event.clientY - bounds.top) / bounds.height) * HEIGHT;
    let nearest: HoveredPoint | null = null;
    let nearestDistance = Number.POSITIVE_INFINITY;

    for (let seriesIndex = 0; seriesIndex < visible.length; seriesIndex += 1) {
      const entry = visible[seriesIndex];
      for (let pointIndex = 0; pointIndex < entry.points.length; pointIndex += 1) {
        const point = entry.points[pointIndex];
        const distanceX = x(point[0]) - pointerX;
        const distanceY = y(point[1]) - pointerY;
        const distance = distanceX * distanceX + distanceY * distanceY;
        if (distance < nearestDistance) {
          nearestDistance = distance;
          nearest = { seriesIndex, pointIndex };
        }
      }
    }

    if (nearest) {
      const next = nearest;
      setHovered((current) =>
        current?.seriesIndex === next.seriesIndex && current?.pointIndex === next.pointIndex ? current : next,
      );
    }
  };

  const tooltipWidth = 286;
  const tooltipHeight = smoothing > 1 ? 148 : 130;
  const hoverX = hoveredPoint ? x(hoveredPoint[0]) : 0;
  const hoverY = hoveredPoint ? y(hoveredPoint[1]) : 0;
  const tooltipX = Math.min(
    WIDTH - MARGIN.right - tooltipWidth,
    Math.max(MARGIN.left, hoverX + 18 + tooltipWidth > WIDTH - MARGIN.right ? hoverX - tooltipWidth - 18 : hoverX + 18),
  );
  const tooltipY = Math.min(
    HEIGHT - MARGIN.bottom - tooltipHeight,
    Math.max(MARGIN.top, hoverY - tooltipHeight / 2),
  );
  const firstWallTime = hoveredSeries?.source.points[0]?.[2];
  const elapsed = rawPoint && firstWallTime != null ? Math.max(0, rawPoint[2] - firstWallTime) : null;

  return (
    <svg data-export-root viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="block w-full" role="img" aria-label={`${yLabel} training curves`}>
      <title>{yLabel} across {visible.length} selected run{visible.length === 1 ? "" : "s"}</title>
      {ticks(yDomain, 5).map((value) => (
        <g key={`y-${value}`} aria-hidden="true">
          <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={y(value)} y2={y(value)} stroke={GRID} />
          <text x={MARGIN.left - 12} y={y(value) + 4} fill={AXIS} fontSize="11" textAnchor="end">{axisNumber(value)}</text>
        </g>
      ))}
      {ticks(xDomain, 6).map((value) => (
        <g key={`x-${value}`} aria-hidden="true">
          <line x1={x(value)} x2={x(value)} y1={MARGIN.top} y2={HEIGHT - MARGIN.bottom} stroke={GRID} />
          <text x={x(value)} y={HEIGHT - MARGIN.bottom + 22} fill={AXIS} fontSize="11" textAnchor="middle">{axisNumber(value)}</text>
        </g>
      ))}
      {visible.map((entry, index) => {
        const color = COLORS[index % COLORS.length];
        const path = entry.points
          .map((point, pointIndex) => `${pointIndex === 0 ? "M" : "L"}${x(point[0])},${y(point[1])}`)
          .join(" ");
        const last = entry.points.at(-1)!;
        return (
          <g key={entry.label}>
            <path
              d={path}
              fill="none"
              stroke={color}
              strokeWidth={hovered?.seriesIndex === index ? "3" : "2"}
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity={hovered && hovered.seriesIndex !== index ? "0.3" : "0.88"}
            >
              <title>{entry.label} · latest {formatNumber(last[1], 6)} · {entry.source.point_count.toLocaleString()} source points</title>
            </path>
            <circle cx={x(last[0])} cy={y(last[1])} r="4" fill={color} stroke="#10182a" strokeWidth="2">
              <title>{entry.label}: {formatNumber(last[1], 6)}</title>
            </circle>
          </g>
        );
      })}
      <text x={MARGIN.left + (WIDTH - MARGIN.left - MARGIN.right) / 2} y={HEIGHT - 9} fill={AXIS} fontSize="12" textAnchor="middle">{xLabel}</text>
      <text x="17" y={MARGIN.top + (HEIGHT - MARGIN.top - MARGIN.bottom) / 2} fill={AXIS} fontSize="12" textAnchor="middle" transform={`rotate(-90 17 ${MARGIN.top + (HEIGHT - MARGIN.top - MARGIN.bottom) / 2})`}>
        {yLabel}
      </text>
      <g transform={`translate(${MARGIN.left + 8} 18)`} aria-hidden="true">
        {visible.map((entry, index) => {
          const offset = index * 112;
          return (
            <g key={entry.label} transform={`translate(${offset} 0)`}>
              <line x1="0" x2="18" y1="0" y2="0" stroke={COLORS[index % COLORS.length]} strokeWidth="3" strokeLinecap="round" />
              <text x="25" y="4" fill={TEXT} fontSize="10">{entry.label}</text>
            </g>
          );
        })}
      </g>
      <rect
        x={MARGIN.left}
        y={MARGIN.top}
        width={WIDTH - MARGIN.left - MARGIN.right}
        height={HEIGHT - MARGIN.top - MARGIN.bottom}
        fill="transparent"
        className="cursor-crosshair"
        onPointerMove={inspectPoint}
        onPointerLeave={() => setHovered(null)}
        aria-label="Hover the plot to inspect the nearest logged point"
      />
      {hoveredSeries && hoveredPoint && rawPoint && (
        <g data-export-ignore pointerEvents="none" aria-hidden="true">
          <line x1={hoverX} x2={hoverX} y1={MARGIN.top} y2={HEIGHT - MARGIN.bottom} stroke={hoverColor} strokeWidth="1" strokeDasharray="4 4" opacity="0.72" />
          <line x1={MARGIN.left} x2={WIDTH - MARGIN.right} y1={hoverY} y2={hoverY} stroke={hoverColor} strokeWidth="1" opacity="0.28" />
          <circle cx={hoverX} cy={hoverY} r="6" fill="#10182a" stroke={hoverColor} strokeWidth="3" />
          <g transform={`translate(${tooltipX} ${tooltipY})`}>
            <rect width={tooltipWidth} height={tooltipHeight} rx="12" fill="#111a2e" stroke={hoverColor} strokeOpacity="0.68" />
            <text x="14" y="21" fill={TEXT} fontSize="12" fontWeight="600">{shortText(hoveredSeries.label, 34)}</text>
            <text x="14" y="40" fill={AXIS} fontSize="10">{shortText(hoveredSeries.source.tag)}</text>
            <text x="14" y="61" fill={AXIS} fontSize="10">{smoothing > 1 ? `Smoothed (${smoothing})` : "Value"}</text>
            <text x="112" y="61" fill={TEXT} fontSize="11" fontWeight="600">{formatNumber(hoveredPoint[1], 8)}</text>
            {smoothing > 1 && (
              <>
                <text x="14" y="79" fill={AXIS} fontSize="10">Raw value</text>
                <text x="112" y="79" fill={TEXT} fontSize="11">{formatNumber(rawPoint[1], 8)}</text>
              </>
            )}
            <text x="14" y={smoothing > 1 ? 97 : 79} fill={AXIS} fontSize="10">Step</text>
            <text x="112" y={smoothing > 1 ? 97 : 79} fill={TEXT} fontSize="11">{formatNumber(rawPoint[0], 10)}</text>
            <text x="14" y={smoothing > 1 ? 115 : 97} fill={AXIS} fontSize="10">Wall time</text>
            <text x="112" y={smoothing > 1 ? 115 : 97} fill={TEXT} fontSize="10">{wallTimeLabel(rawPoint[2])}</text>
            <text x="14" y={smoothing > 1 ? 133 : 115} fill={AXIS} fontSize="10">Relative</text>
            <text x="112" y={smoothing > 1 ? 133 : 115} fill={TEXT} fontSize="10">{formatDuration(elapsed)}</text>
          </g>
        </g>
      )}
    </svg>
  );
}

function ServiceMessage({ kind, message, onRetry }: { kind: "missing" | "offline"; message: string; onRetry?: () => void }) {
  return (
    <section className="surface overflow-hidden" id="training-curves">
      <div className="flex flex-col items-center px-6 py-12 text-center">
        <span className={`flex h-12 w-12 items-center justify-center rounded-2xl ring-1 ${kind === "offline" ? "bg-amber-300/10 text-amber-200 ring-amber-300/20" : "bg-white/[0.04] text-[#78869e] ring-white/[0.08]"}`}>
          {kind === "offline" ? <Server size={21} /> : <Database size={21} />}
        </span>
        <h2 className="mt-4 text-base font-semibold text-white">{kind === "offline" ? "TensorBoard service unavailable" : "No TensorBoard run linked"}</h2>
        <p className="mt-2 max-w-xl text-xs leading-5 text-[#8290a7]">{message}</p>
        {kind === "offline" && (
          <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
            <code className="rounded-lg border border-white/[0.08] bg-black/15 px-3 py-2 text-[11px] text-[#b9c3d5]">.\Start-Trial-Atlas.ps1 -SkipBuild</code>
            {onRetry && (
              <button type="button" onClick={onRetry} className="focus-ring inline-flex h-9 items-center gap-2 rounded-lg bg-[#9b8cff]/12 px-3 text-[11px] font-semibold text-[#c1b9ff] hover:bg-[#9b8cff]/20">
                <RefreshCw size={13} /> Retry
              </button>
            )}
          </div>
        )}
      </div>
    </section>
  );
}

interface LoadedTrial {
  row: TrialRow;
  trialNumber: string;
  logDirectory: string | null;
  data: TensorboardTrialData | null;
  error: string | null;
}

export function TrainingCurves({
  study,
  rows,
  onRemove,
}: {
  study: Study;
  rows: TrialRow[];
  onRemove: (row: TrialRow) => void;
}) {
  const selections = useMemo(
    () =>
      rows.map((row) => ({
        row,
        trialNumber: row[study.config.trial] ?? "unknown",
        logDirectory: trialLogDirectory(study, row),
      })),
    [rows, study],
  );
  const [loadedTrials, setLoadedTrials] = useState<LoadedTrial[]>([]);
  const [loading, setLoading] = useState(false);
  const [fold, setFold] = useState("all");
  const [smoothing, setSmoothing] = useState(20);
  const [refresh, setRefresh] = useState(0);
  const retry = useCallback(() => setRefresh((value) => value + 1), []);

  useEffect(() => {
    if (selections.length === 0) {
      setLoadedTrials([]);
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    setLoadedTrials([]);
    setLoading(true);
    Promise.all(
      selections.map(async (selection): Promise<LoadedTrial> => {
        if (!selection.logDirectory) {
          return { ...selection, data: null, error: "No attr:log_dir value is available." };
        }
        try {
          const data = await fetchTensorboardTrial(selection.logDirectory, controller.signal);
          return { ...selection, data, error: null };
        } catch (reason) {
          if (reason instanceof DOMException && reason.name === "AbortError") throw reason;
          return {
            ...selection,
            data: null,
            error: reason instanceof Error ? reason.message : "TensorBoard data could not be loaded.",
          };
        }
      }),
    )
      .then((results) => setLoadedTrials(results))
      .catch((reason) => {
        if (!(reason instanceof DOMException && reason.name === "AbortError")) {
          setLoadedTrials(
            selections.map((selection) => ({
              ...selection,
              data: null,
              error: reason instanceof Error ? reason.message : "TensorBoard data could not be loaded.",
            })),
          );
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [selections, refresh]);

  const successful = useMemo(
    () =>
      loadedTrials.filter(
        (trial): trial is LoadedTrial & { data: TensorboardTrialData } => trial.data !== null,
      ),
    [loadedTrials],
  );
  const foldOptions = useMemo(() => {
    const byId = new Map<string, string>();
    successful.forEach((trial) => trial.data.folds.forEach((candidate) => byId.set(candidate.id, candidate.label)));
    return [...byId.entries()].map(([id, label]) => ({ id, label }));
  }, [successful]);

  useEffect(() => {
    if (successful.length <= 1) {
      if (fold !== "all" && !foldOptions.some((option) => option.id === fold)) setFold("all");
      return;
    }
    if (fold === "all" || !foldOptions.some((option) => option.id === fold)) {
      setFold(foldOptions[0]?.id ?? "all");
    }
  }, [fold, foldOptions, successful.length]);

  if (rows.length === 0) {
    return <ServiceMessage kind="missing" message="Use chart points or the Curves checkboxes in the trial table to select one or more trials." />;
  }

  if (loading) {
    return (
      <section id="training-curves" className="surface flex min-h-48 items-center justify-center">
        <div className="text-center">
          <LoaderCircle className="mx-auto animate-spin text-[#9b8cff]" size={24} />
          <p className="mt-3 text-xs font-medium text-[#9ca8bc]">Reading {rows.length} TensorBoard trial{rows.length === 1 ? "" : "s"}…</p>
          <p className="mt-1 text-[10px] text-[#67758d]">Dense batch curves are reduced before they reach the browser.</p>
        </div>
      </section>
    );
  }

  if (successful.length === 0) {
    return <ServiceMessage kind="offline" message={loadedTrials[0]?.error ?? "Start the local Trial Atlas service to read TensorBoard event files."} onRetry={retry} />;
  }

  const displaySeries = (key: "validation" | "batch_loss" | "epoch_loss"): DisplaySeries[] =>
    successful.flatMap((trial) => {
      const candidates =
        fold === "all"
          ? trial.data.folds
          : [trial.data.folds.find((candidate) => candidate.id === fold) ?? trial.data.folds[0]].filter(Boolean);
      return candidates.flatMap((candidate) => {
        const source = candidate?.series[key];
        if (!source) return [];
        const label =
          successful.length > 1
            ? `Trial #${trial.trialNumber}${fold === "all" ? ` · ${candidate.label}` : ""}`
            : candidate.label;
        return [{ label, source, points: source.points }];
      });
    });
  const validation = displaySeries("validation");
  const epochLoss = displaySeries("epoch_loss");
  const batchLoss = displaySeries("batch_loss");
  const validationLabel = validation[0]?.source.label ?? "Validation loss";
  const eventFileCount = successful.reduce((sum, trial) => sum + trial.data.event_file_count, 0);
  const firstWarning = successful.flatMap((trial) => trial.data.warnings)[0] ?? loadedTrials.find((trial) => trial.error)?.error;
  const exportStem = `${study.name}-${successful.map((trial) => `trial-${trial.trialNumber}`).join("-")}`;

  return (
    <section id="training-curves" className="scroll-mt-24 space-y-6">
      <div className="surface overflow-hidden">
        <div className="flex flex-col gap-4 px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6">
          <div className="flex min-w-0 items-center gap-3">
            <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl bg-[#4fd5e7]/10 text-[#6de0ec] ring-1 ring-[#4fd5e7]/15"><Wifi size={17} /></span>
            <div className="min-w-0">
              <div className="flex flex-wrap items-center gap-2">
                <h2 className="text-sm font-semibold text-white">TensorBoard comparison · {rows.length} trial{rows.length === 1 ? "" : "s"}</h2>
                <span className="rounded-full border border-[#b8e56e]/20 bg-[#b8e56e]/10 px-2 py-0.5 text-[9px] font-semibold uppercase tracking-[0.1em] text-[#c9f18a]">Connected</span>
              </div>
              <p className="mt-1 truncate text-[10px] text-[#6f7d94]">{eventFileCount} event file{eventFileCount === 1 ? "" : "s"} · one line per selected trial</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <label className="flex items-center gap-2 rounded-xl border border-white/[0.08] bg-black/10 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#77859d]">
              Compare fold
              <select value={fold} onChange={(event) => setFold(event.target.value)} className="focus-ring bg-transparent text-xs font-medium normal-case tracking-normal text-[#d5dce8]">
                {successful.length === 1 && <option value="all" className="bg-[#111a2e]">All folds</option>}
                {foldOptions.map((candidate) => <option key={candidate.id} value={candidate.id} className="bg-[#111a2e]">{candidate.label}</option>)}
              </select>
            </label>
            <label className="flex items-center gap-2 rounded-xl border border-white/[0.08] bg-black/10 px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#77859d]">
              <SlidersHorizontal size={12} /> Batch smoothing
              <select value={smoothing} onChange={(event) => setSmoothing(Number(event.target.value))} className="focus-ring bg-transparent text-xs font-medium normal-case tracking-normal text-[#d5dce8]">
                <option value="1" className="bg-[#111a2e]">Off</option>
                <option value="5" className="bg-[#111a2e]">5</option>
                <option value="20" className="bg-[#111a2e]">20</option>
                <option value="50" className="bg-[#111a2e]">50</option>
              </select>
            </label>
            <button type="button" onClick={retry} className="focus-ring flex h-9 items-center gap-2 rounded-xl border border-white/[0.08] px-3 text-[10px] font-semibold uppercase tracking-[0.1em] text-[#9ba7ba] transition hover:bg-white/[0.05] hover:text-white">
              <RefreshCw size={13} /> Refresh
            </button>
          </div>
        </div>
        <div className="flex flex-wrap gap-2 border-t border-white/[0.07] px-5 py-3 sm:px-6">
          {rows.map((row, index) => (
            <span key={`${row[study.config.trial] ?? index}-${index}`} className="inline-flex items-center gap-2 rounded-full border border-white/[0.08] bg-black/10 py-1 pl-2.5 pr-1.5 text-[10px] font-semibold text-[#c5cede]">
              <span className="h-2 w-2 rounded-full" style={{ backgroundColor: COLORS[index % COLORS.length] }} />
              Trial #{row[study.config.trial] ?? "—"}
              <button type="button" onClick={() => onRemove(row)} className="focus-ring flex h-5 w-5 items-center justify-center rounded-full text-[#71809a] hover:bg-white/[0.08] hover:text-white" aria-label={`Remove trial ${row[study.config.trial] ?? ""} from learning curves`}>
                <X size={11} />
              </button>
            </span>
          ))}
        </div>
        {firstWarning && (
          <div className="flex items-start gap-2.5 border-t border-amber-200/10 bg-amber-200/[0.045] px-5 py-3 text-[11px] leading-5 text-[#d3bb83] sm:px-6">
            <TriangleAlert size={14} className="mt-0.5 shrink-0" /> {firstWarning}
          </div>
        )}
      </div>

      <ChartBoard
        storageKey={`trial-atlas:charts:${study.id}:tensorboard`}
        items={[
          {
            id: "validation",
            title: validationLabel,
            description: validation[0] ? `${validation[0].source.tag} · comparing the selected trials on the same fold.` : "No validation scalar is available.",
            exportName: `${exportStem}-validation`,
            content: <CurveSvg series={validation} yLabel={validationLabel} />,
          },
          {
            id: "epoch-loss",
            title: "Epoch loss",
            description: epochLoss[0] ? `${epochLoss[0].source.tag} · one line per selected trial.` : "No epoch-loss scalar is available.",
            exportName: `${exportStem}-epoch-loss`,
            content: <CurveSvg series={epochLoss} yLabel="Epoch loss" />,
          },
          {
            id: "batch-loss",
            title: "Batch loss",
            description: batchLoss[0] ? `${batchLoss.reduce((sum, entry) => sum + entry.source.point_count, 0).toLocaleString()} source points across the selected trials, shape-preserving downsampled by the local service.` : "No batch-loss scalar is available.",
            exportName: `${exportStem}-batch-loss`,
            className: "xl:col-span-2",
            content: <CurveSvg series={batchLoss} yLabel="Batch loss" smoothing={smoothing} />,
          },
        ]}
      />
    </section>
  );
}
