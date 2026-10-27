"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, Crosshair, Info, Pause, Pin, PinOff, Play, Radio, RefreshCw, RotateCcw, Search, Upload, X } from "lucide-react";
import {
  EmbeddingCanvas,
  type CanvasEdge,
  type EmbeddingCanvasHandle,
  type TooltipContent,
  type ViewMode,
} from "@/components/EmbeddingCanvas";
import {
  CANVAS_COLORS,
  CATEGORICAL,
  OUTLIER_SHARE,
  PREDICTOR_LABELS,
  confidenceHistogram,
  confidentConfusions,
  frameAccuracy,
  predictorAt,
  topConfidenceAccuracy,
  type Predictor,
  CHANGE_BETTER_INDEX,
  CHANGE_NONE,
  CHANGE_NONE_INDEX,
  CHANGE_WORSE_INDEX,
  CLASS_PALETTE,
  PURITY_CHANGE_THRESHOLD,
  MAX_PINNED_CLASSES,
  NEUTRAL,
  RAMP,
  TRACKING_METRICS,
  assignClassColors,
  assignGroupColors,
  deriveGraph,
  encodePoints,
  formatCount,
  imageUrl,
  parseProjectionCsv,
  trackingTimeline,
  prepareEmbedding,
  subsetColor,
  subsetStyle,
  type ColorMode,
  type EmbeddingViewFile,
  type EpochEvidence,
  type EvidenceMap,
  type PseudoLabelFrame,
  type EmbeddingViewIndex,
  type EmbeddingViewIndexEntry,
  type MarkerShape,
  type PreparedEmbedding,
  type SplitEmphasis,
  type TimelineSeries,
  type ViewSource,
} from "@/lib/embedding-views";
import { ExportMenu } from "@/components/ExportMenu";
import { downloadBlob, type ExportRow } from "@/lib/export";
import { slugify } from "@/lib/trials";
import { embeddingRunLookup, looksLikeEmbeddingPath, normalizeEmbeddingLookup } from "@/lib/embedding-paths";
import { describeStage, useLiveRuns, useLiveTrackingView, type LiveRunEntry } from "@/lib/live-tracking";
import { ShareNav } from "@/components/ShareNav";

const STORAGE_KEY = "trial-atlas:embedding-viewer";
const LINK_SAME = "#c8d2e4";
const LINK_DIFFERENT = CATEGORICAL[1];
const MAX_PULL_EDGES = 400;
const MAX_GALLERY_IMAGES = 60;
/** Without a graph the gallery can show up to this many nearest points on screen. */
const LAYOUT_NEIGHBORS_MAX = 50;
const DEFAULT_NEIGHBOR_COUNT = 10;
/** From this width the gallery gets its own column beside the canvas; below it, it tops the side panel. */
const GALLERY_COLUMN_QUERY = "(min-width: 1280px)";

type SourceKey = ViewSource | "upload" | "live";

const SOURCE_LABELS: Record<SourceKey, string> = {
  live: "Live runs",
  head_state: "Saved heads",
  epoch_tracking: "Epoch tracking",
  projection_csv: "Test projections",
  upload: "Uploaded",
};
const SOURCE_ORDER: SourceKey[] = ["live", "head_state", "epoch_tracking", "projection_csv", "upload"];

const COLOR_MODE_LABELS: Record<ColorMode, string> = {
  subset: "Subset",
  class: "Class",
  pinned: "Pinned classes",
  agreement: "Nearest labeled agrees",
  purity: "Neighbor purity",
  cluster: "Cluster",
  purityChange: "Neighbor purity change",
  agreementChange: "Agreement change",
  lpPrediction: "Label propagation · right or wrong",
  lpWeight: "Label propagation · weight ω",
  hofferPrediction: "Hoffer · right or wrong",
  hofferCertainty: "Hoffer · certainty",
  pseudoLabel: "Pseudo labels",
  prototypePrediction: "Network prediction · right or wrong",
  propagationClass: "Propagation · class held at this step",
  propagationCorrect: "Propagation · right or wrong at this step",
  propagationHop: "Propagation · edges from the labeled node",
  outlier: "Outliers",
  centerSimilarity: "SLADE · similarity to cluster center",
};

const PREDICTOR_COLORS: Record<Predictor, { color: string; dashed?: boolean }> = {
  lp: { color: CATEGORICAL[0] },
  prototype: { color: CATEGORICAL[1] },
  hoffer: { color: CATEGORICAL[2] },
  pseudo: { color: "#c8d2e4", dashed: true },
};
const CONFIDENCE_SHARES = [0.1, 0.25, 0.5, 1];
const CONFUSION_TOP_SHARE = 0.25;
const MAX_CONFUSIONS = 8;
const HISTOGRAM_BINS = 10;
type LineSource = "graph" | "mined" | "propagation";
/** Arrival edges drawn at once when no point is selected. */
const MAX_PROPAGATION_EDGES = 4000;
/** How far the edges of earlier steps fade behind the step being shown. */
const PAST_FRONT_ALPHA = 0.22;

/** A #rrggbb color at the given opacity. */
function fade(color: string, alpha: number): string {
  const value = Number.parseInt(color.slice(1), 16);
  return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}

interface ViewOption {
  id: string;
  source: SourceKey;
  label: string;
  runLabel: string;
  variant: string;
  dataset: string | null;
  entry?: EmbeddingViewIndexEntry;
  uploaded?: PreparedEmbedding;
  /** A tracking directory the dev server reads while its run trains. */
  live?: LiveRunEntry;
}

interface OpenedEmbeddingPath {
  ok: boolean;
  error?: string;
  sourcePath?: string;
  runDir?: string;
  label?: string;
  csv?: string;
  csv3d?: string | null;
}

function viewSearchScore(option: ViewOption, query: string): number | null {
  const needle = normalizeEmbeddingLookup(query);
  const runNeedle = embeddingRunLookup(query);
  if (!needle) return null;
  const runDir = normalizeEmbeddingLookup(option.entry?.runDir ?? option.uploaded?.runDir ?? option.live?.runDir ?? "");
  const sourcePath = normalizeEmbeddingLookup(option.entry?.sourcePath ?? option.live?.dir ?? "");
  const fields = [
    normalizeEmbeddingLookup(option.id),
    normalizeEmbeddingLookup(option.label),
    normalizeEmbeddingLookup(option.runLabel),
    normalizeEmbeddingLookup(option.variant),
    runDir,
    sourcePath,
  ].filter(Boolean);

  if (fields.includes(needle) || (runNeedle && (runDir === runNeedle || sourcePath === needle))) return 0;
  if (runNeedle && runDir && (runDir.endsWith(runNeedle) || runNeedle.endsWith(runDir))) return 1;
  if (fields.some((field) => field.startsWith(needle) || field.endsWith(needle))) return 2;
  if (fields.some((field) => field.includes(needle))) return 3;
  const words = needle.split(/\s+/).filter(Boolean);
  return words.length > 0 && words.every((word) => fields.some((field) => field.includes(word))) ? 4 : null;
}

function readStored(): { viewId?: string; mode?: ViewMode } {
  try {
    return JSON.parse(window.localStorage.getItem(STORAGE_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function writeStored(value: { viewId?: string; mode?: ViewMode }) {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(value));
  } catch {
    // Remembering the last view is a convenience; the page works without it.
  }
}

function percent(value: number | null | undefined, digits = 0): string {
  return value == null || !Number.isFinite(value) ? "—" : `${(value * 100).toFixed(digits)}%`;
}

function formatNumber(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return "—";
  return Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(3);
}

function formatDelta(value: number | null, higherIsBetter: boolean | null): string {
  if (value == null || !Number.isFinite(value)) return "—";
  const sign = value > 0 ? "+" : "";
  const arrow = higherIsBetter == null || Math.abs(value) < 1e-9 ? "" : (value > 0) === higherIsBetter ? " ↑" : " ↓";
  return `${sign}${Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(3)}${arrow}`;
}

/** t-SNE is the default projection everywhere; PaCMAP and PCA views sort after it. */
function projectionRank(option: { entry?: EmbeddingViewIndexEntry }): number {
  const projection = option.entry?.projection ?? "tsne";
  return projection === "tsne" ? 0 : projection === "pca" ? 1 : 2;
}

function liveVariantLabel(run: LiveRunEntry): string {
  const state = run.running ? "running" : run.status?.state === "finished" ? "finished" : "stopped";
  return `${run.fold == null ? "no CV" : `fold ${run.fold}`} · ${run.epochCount} epochs · ${state}`;
}

/** "12 s ago", "4 min ago": how fresh a live view is. */
function ago(timestamp: number | null, now: number): string {
  if (timestamp == null) return "never";
  const seconds = Math.max(0, Math.round((now - timestamp) / 1000));
  if (seconds < 60) return `${seconds} s ago`;
  const minutes = Math.round(seconds / 60);
  return minutes < 90 ? `${minutes} min ago` : `${Math.round(minutes / 60)} h ago`;
}

function variantLabel(entry: EmbeddingViewIndexEntry): string {
  const parts = [
    entry.concatenatedFolds ? `${entry.concatenatedFolds} folds concatenated` : entry.fold == null ? "no CV" : `fold ${entry.fold}`,
    entry.projection,
    entry.epochs ? `${entry.epochs.length} epochs` : entry.has3d ? "2D + 3D" : "2D",
  ];
  return `${parts.join(" · ")} · ${formatCount(entry.pointCount)} points`;
}

function ShapeKey({ shape, color, size = 12 }: { shape: MarkerShape; color: string; size?: number }) {
  const half = size / 2;
  const r = size * 0.32;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden className="shrink-0">
      {shape === "circle" && <circle cx={half} cy={half} r={r} fill={color} />}
      {shape === "square" && <rect x={half - r * 0.9} y={half - r * 0.9} width={r * 1.8} height={r * 1.8} fill={color} />}
      {shape === "triangle" && (
        <polygon points={`${half},${half - r * 1.15} ${half + r * 1.05},${half + r * 0.8} ${half - r * 1.05},${half + r * 0.8}`} fill={color} />
      )}
      {shape === "diamond" && (
        <polygon points={`${half},${half - r * 1.3} ${half + r * 1.3},${half} ${half},${half + r * 1.3} ${half - r * 1.3},${half}`} fill={color} />
      )}
    </svg>
  );
}

function ControlLabel({ children }: { children: React.ReactNode }) {
  return <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">{children}</p>;
}

function Panel({ title, children, action }: { title: string; children: React.ReactNode; action?: React.ReactNode }) {
  return (
    <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
      <div className="flex items-center justify-between gap-2">
        <ControlLabel>{title}</ControlLabel>
        {action}
      </div>
      {children}
    </div>
  );
}

function Segmented<T extends string>({
  options,
  value,
  onChange,
  label,
}: {
  options: { value: T; label: string; disabled?: boolean; title?: string }[];
  value: T;
  onChange(value: T): void;
  label: string;
}) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap rounded-xl border border-white/[0.08] bg-white/[0.02] p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          disabled={option.disabled}
          title={option.title}
          onClick={() => onChange(option.value)}
          className={`focus-ring rounded-[10px] px-3 py-1.5 text-xs font-semibold transition disabled:cursor-not-allowed disabled:opacity-40 ${
            value === option.value ? "bg-white/[0.09] text-white" : "text-[#8895aa] hover:text-white"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

const selectClass =
  "focus-ring mt-1.5 w-full rounded-xl border border-white/[0.08] bg-[#10182a] px-3 py-2 text-sm text-white";

/** A subset's legend swatch: its own color where color means subset, otherwise its shape in a light ink. */
function legendColor(name: string, colorMode: ColorMode): string {
  if (colorMode === "subset") return subsetColor(name);
  return colorMode === "pinned" ? NEUTRAL : "#c8d2e4";
}

/** Per-subset counts of some classes, as a compact table (the chart's table view). */
function ClassSubsetTable({ data, classes }: { data: PreparedEmbedding; classes: number[] }) {
  if (classes.length === 0) return null;
  return (
    <table className="mt-2 w-full text-[11px]">
      <thead>
        <tr className="text-left text-[#7f8ba2]">
          <th className="py-1 pr-2 font-semibold">Class</th>
          {data.subsets.map((name) => (
            <th key={name} className="py-1 pl-2 text-right font-semibold">
              {subsetStyle(name).label}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {classes.map((classIndex) => (
          <tr key={classIndex} className="border-t border-white/[0.05] text-[#c8d2e4]">
            <td className="max-w-[110px] truncate py-1 pr-2" title={data.classNames[classIndex]}>
              {data.classNames[classIndex]}
            </td>
            {data.subsets.map((name, subsetIndex) => (
              <td key={name} className="py-1 pl-2 text-right tabular-nums">
                {formatCount(data.classSubsetCounts[classIndex * data.subsets.length + subsetIndex])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/** A dataset image, or a quiet note when the image server cannot provide it. */
function SampleImage({
  url,
  caption,
  size = "large",
}: {
  url: string | null;
  caption: string;
  size?: "large" | "medium" | "small";
}) {
  // Keyed by URL, so a new image starts out loading without an effect that could
  // land after a cached image's load event.
  const [status, setStatus] = useState<{ url: string | null; state: "loading" | "loaded" | "failed" }>({
    url,
    state: "loading",
  });
  const state = status.url === url ? status.state : "loading";
  const setState = (next: "loaded" | "failed") => setStatus({ url, state: next });
  const box = size === "large" ? "h-52" : size === "medium" ? "h-28" : "h-16";
  return (
    <figure className="min-w-0 flex-1">
      {url && state !== "failed" ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={url}
          alt={caption}
          onLoad={() => setState("loaded")}
          onError={() => setState("failed")}
          className={`${box} w-full rounded-lg border border-white/[0.07] object-contain ${
            state === "loading" ? "animate-pulse bg-white/[0.06]" : "bg-black/30"
          }`}
        />
      ) : (
        <div
          className={`${box} flex w-full items-center justify-center rounded-lg border border-dashed border-white/[0.1] px-2 text-center text-[10px] text-[#65748e]`}
        >
          {url
            ? "No image: the app's image server (next dev or the Trial Atlas service) could not find it in data/"
            : "No run to read images from (uploaded view)"}
        </div>
      )}
      {size === "large" && <figcaption className="mt-1 truncate text-[10px] text-[#7f8ba2]">{caption}</figcaption>}
    </figure>
  );
}

/** One evidence map's top classes as bars, the true class marked. */
function EvidenceBars({
  data,
  row,
  title,
  detail,
  map,
}: {
  data: PreparedEmbedding;
  row: number;
  title: string;
  detail: string;
  map: EvidenceMap;
}) {
  const slots = Array.from({ length: map.topK }, (_, slot) => ({
    classIndex: map.topClass[row * map.topK + slot],
    probability: map.topProbability[row * map.topK + slot],
  })).filter((slot) => slot.classIndex >= 0);
  const largest = Math.max(1e-9, ...slots.map((slot) => slot.probability));
  const trueClass = data.classIndex[row];
  return (
    <div>
      <p className="flex items-baseline justify-between gap-2 text-[11px]">
        <span className="font-semibold text-[#c8d2e4]">{title}</span>
        <span className="tabular-nums text-[#7f8ba2]">{detail}</span>
      </p>
      {slots.length === 0 ? (
        <p className="mt-1 text-[11px] text-[#7f8ba2]">No labeled point reaches it: no class mass.</p>
      ) : (
        <ul className="mt-1 space-y-1">
          {slots.map((slot) => (
            <li key={slot.classIndex} className="text-[10px] text-[#c8d2e4]">
              <span className="flex justify-between gap-2">
                <span className="truncate" title={data.classNames[slot.classIndex]}>
                  {data.classNames[slot.classIndex]}
                  {slot.classIndex === trueClass && <span className="text-[#98a4ba]"> · true class</span>}
                </span>
                <span className="tabular-nums">{percent(slot.probability, 1)}</span>
              </span>
              <span className="mt-0.5 block h-[4px] rounded-full bg-white/[0.05]">
                <span
                  className="block h-full rounded-full"
                  style={{
                    width: `${Math.max(2, (slot.probability / largest) * 100)}%`,
                    background: slot.classIndex === trueClass ? CATEGORICAL[0] : "#56607a",
                  }}
                />
              </span>
            </li>
          ))}
        </ul>
      )}
      {slots.length > 0 && !slots.some((slot) => slot.classIndex === trueClass) && (
        <p className="mt-1 text-[10px] text-[#7f8ba2]">The true class is not among its top {map.topK}.</p>
      )}
    </div>
  );
}

/** How label propagation, Hoffer's reference softmax and the run's pseudo label read one point at one epoch. */
function ClassEvidence({
  data,
  row,
  evidence,
  pseudo,
}: {
  data: PreparedEmbedding;
  row: number;
  evidence: EpochEvidence;
  pseudo: PseudoLabelFrame | null;
}) {
  return (
    <div className="mt-3 space-y-3 border-t border-white/[0.06] pt-3">
      <p className="text-[11px] text-[#7f8ba2]">
        Class evidence at this epoch. Bars are scaled to the top class; percentages are probabilities.
      </p>
      {evidence.lp && (
        <EvidenceBars
          data={data}
          row={row}
          title="Label propagation"
          detail={`ω ${(evidence.lp.weight?.[row] ?? 0).toFixed(2)} · certainty ${evidence.lp.certainty[row].toFixed(3)}`}
          map={evidence.lp}
        />
      )}
      {evidence.hoffer && (
        <EvidenceBars
          data={data}
          row={row}
          title="Hoffer reference softmax"
          detail={`certainty ${evidence.hoffer.certainty[row].toFixed(3)}`}
          map={evidence.hoffer}
        />
      )}
      {evidence.prototype && (
        <EvidenceBars
          data={data}
          row={row}
          title="Network (class prototypes)"
          detail={`certainty ${evidence.prototype.certainty[row].toFixed(3)}`}
          map={evidence.prototype}
        />
      )}
      {pseudo && (
        <p className="text-[11px] text-[#c8d2e4]">
          {pseudo.label[row] >= 0 ? (
            <>
              Pseudo label (generation {pseudo.generation}, epoch {pseudo.generationEpoch}): {data.classNames[pseudo.label[row]]} ·{" "}
              {pseudo.accepted[row] ? "accepted" : "rejected"} · confidence {pseudo.confidence[row].toFixed(2)}
              {pseudo.label[row] === data.classIndex[row] ? " · right" : " · wrong"}
            </>
          ) : (
            <>Not a pseudo-label candidate in generation {pseudo.generation}.</>
          )}
        </p>
      )}
    </div>
  );
}

function ChartBlock({ title, note, children }: { title: string; note?: string; children: React.ReactNode }) {
  return (
    <div className="mt-4">
      <p className="text-[11px] font-semibold text-[#c8d2e4]">{title}</p>
      {note && <p className="mt-0.5 text-[10px] leading-snug text-[#7f8ba2]">{note}</p>}
      <div className="mt-2">{children}</div>
    </div>
  );
}

interface ChartSeries {
  label: string;
  color: string;
  dashed?: boolean;
  values: (number | null)[];
}

/** Several series over the tracked epochs on one axis, with a hover readout and click-to-jump. */
function LineChart({
  epochs,
  series,
  epochIndex,
  referenceIndex,
  format,
  onPick,
  unit = "epoch",
}: {
  epochs: number[];
  series: ChartSeries[];
  epochIndex: number;
  referenceIndex: number;
  format: (value: number) => string;
  onPick: (index: number) => void;
  /** What the x axis counts, for the readout under the chart. */
  unit?: string;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const width = 300;
  const height = 110;
  const left = 34;
  const right = 6;
  const top = 6;
  const bottom = 16;
  const finite = series.flatMap((entry) => entry.values.filter((value): value is number => value != null && Number.isFinite(value)));
  if (!finite.length) return <p className="text-[11px] text-[#7f8ba2]">Nothing recorded yet.</p>;
  let low = Math.min(...finite);
  let high = Math.max(...finite);
  if (high - low < 1e-9) {
    high += 0.05;
    low -= 0.05;
  }
  const x = (index: number) => left + (epochs.length <= 1 ? (width - left - right) / 2 : (index / (epochs.length - 1)) * (width - left - right));
  const y = (value: number) => top + (1 - (value - low) / (high - low)) * (height - top - bottom);
  const indexAt = (clientX: number, rect: DOMRect) => {
    const position = ((clientX - rect.left) / rect.width) * width;
    const fraction = (position - left) / (width - left - right);
    return Math.max(0, Math.min(epochs.length - 1, Math.round(fraction * (epochs.length - 1))));
  };
  const shown = hover ?? epochIndex;
  return (
    <div>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        className="w-full cursor-crosshair"
        role="img"
        aria-label={series.map((entry) => entry.label).join(", ")}
        onMouseMove={(event) => setHover(indexAt(event.clientX, event.currentTarget.getBoundingClientRect()))}
        onMouseLeave={() => setHover(null)}
        onClick={(event) => onPick(indexAt(event.clientX, event.currentTarget.getBoundingClientRect()))}
      >
        {[low, (low + high) / 2, high].map((value) => (
          <g key={value}>
            <line x1={left} x2={width - right} y1={y(value)} y2={y(value)} stroke="rgba(255,255,255,0.06)" />
            <text x={left - 4} y={y(value) + 3} textAnchor="end" fontSize="8" fill="#7f8ba2">
              {format(value)}
            </text>
          </g>
        ))}
        <text x={left} y={height - 3} fontSize="8" fill="#7f8ba2">
          {epochs[0]}
        </text>
        <text x={width - right} y={height - 3} fontSize="8" fill="#7f8ba2" textAnchor="end">
          {epochs[epochs.length - 1]}
        </text>
        {referenceIndex !== epochIndex && (
          <line x1={x(referenceIndex)} x2={x(referenceIndex)} y1={top} y2={height - bottom} stroke="#7f8ba2" strokeDasharray="2 2" />
        )}
        <line x1={x(epochIndex)} x2={x(epochIndex)} y1={top} y2={height - bottom} stroke="#ffffff" strokeOpacity="0.7" />
        {hover != null && hover !== epochIndex && (
          <line x1={x(hover)} x2={x(hover)} y1={top} y2={height - bottom} stroke="#ffffff" strokeOpacity="0.25" />
        )}
        {series.map((entry) => {
          const segments: string[] = [];
          let current = "";
          entry.values.forEach((value, index) => {
            if (value == null || !Number.isFinite(value)) {
              if (current) segments.push(current);
              current = "";
              return;
            }
            current += `${current ? "L" : "M"}${x(index).toFixed(1)},${y(value).toFixed(1)}`;
          });
          if (current) segments.push(current);
          return (
            <g key={entry.label}>
              {segments.map((segment) => (
                <path
                  key={segment}
                  d={segment}
                  fill="none"
                  stroke={entry.color}
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeDasharray={entry.dashed ? "4 3" : undefined}
                />
              ))}
              {entry.values.map((value, index) =>
                value != null && Number.isFinite(value) && index === shown ? (
                  <circle key={index} cx={x(index)} cy={y(value)} r="3" fill={entry.color} stroke="#0b1222" strokeWidth="1.5" />
                ) : null,
              )}
            </g>
          );
        })}
      </svg>
      <ul className="mt-1 space-y-0.5 text-[10px] text-[#c8d2e4]">
        {series.map((entry) => (
          <li key={entry.label} className="flex items-center justify-between gap-2">
            <span className="flex min-w-0 items-center gap-1.5">
              <span
                aria-hidden
                className="w-3 shrink-0 border-t-2"
                style={{ borderColor: entry.color, borderStyle: entry.dashed ? "dashed" : "solid" }}
              />
              <span className="truncate">{entry.label}</span>
            </span>
            <span className="shrink-0 tabular-nums">
              {entry.values[shown] != null && Number.isFinite(entry.values[shown]!) ? format(entry.values[shown]!) : "—"}
            </span>
          </li>
        ))}
        <li className="text-[#7f8ba2]">
          Values at {unit} {epochs[shown]}
        </li>
      </ul>
    </div>
  );
}

/** Confidence of right and wrong predictions at one epoch, each as shares of its own group. */
function ConfidenceHistogram({
  label,
  top,
  right,
  wrong,
  rightTotal,
  wrongTotal,
}: {
  label: string;
  top: number;
  right: number[];
  wrong: number[];
  rightTotal: number;
  wrongTotal: number;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const width = 140;
  const height = 70;
  const bottom = 12;
  const largest = Math.max(1e-9, ...right, ...wrong);
  const band = width / right.length;
  const bar = Math.max(1, (band - 2) / 2);
  return (
    <div>
      <p className="text-[10px] text-[#98a4ba]">{label}</p>
      <svg viewBox={`0 0 ${width} ${height}`} className="mt-1 w-full" role="img" aria-label={`${label} confidence histogram`}>
        {right.map((share, bin) => {
          const rightHeight = (share / largest) * (height - bottom);
          const wrongHeight = (wrong[bin] / largest) * (height - bottom);
          return (
            <g key={bin} onMouseEnter={() => setHover(bin)} onMouseLeave={() => setHover(null)}>
              <rect x={bin * band} y={0} width={band} height={height - bottom} fill="transparent" />
              <rect x={bin * band + 1} y={height - bottom - rightHeight} width={bar} height={rightHeight} rx="1" fill={CATEGORICAL[0]} />
              <rect
                x={bin * band + 1 + bar}
                y={height - bottom - wrongHeight}
                width={bar}
                height={wrongHeight}
                rx="1"
                fill={CATEGORICAL[1]}
              />
            </g>
          );
        })}
        <text x={0} y={height - 2} fontSize="8" fill="#7f8ba2">
          0
        </text>
        <text x={width} y={height - 2} fontSize="8" fill="#7f8ba2" textAnchor="end">
          {top < 0.1 ? top.toFixed(3) : top.toFixed(2)}
        </text>
      </svg>
      <p className="text-[10px] text-[#7f8ba2]">
        {hover != null
          ? `${((hover / right.length) * top).toFixed(3)}–${(((hover + 1) / right.length) * top).toFixed(3)}: right ${percent(right[hover], 1)}, wrong ${percent(wrong[hover], 1)}`
          : (
            <>
              <span className="inline-block h-[6px] w-[6px] rounded-sm" style={{ background: CATEGORICAL[0] }} /> right{" "}
              {formatCount(rightTotal)} ·{" "}
              <span className="inline-block h-[6px] w-[6px] rounded-sm" style={{ background: CATEGORICAL[1] }} /> wrong{" "}
              {formatCount(wrongTotal)}
            </>
          )}
      </p>
    </div>
  );
}

function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);
  useEffect(() => {
    const list = window.matchMedia(query);
    const update = () => setMatches(list.matches);
    update();
    list.addEventListener("change", update);
    return () => list.removeEventListener("change", update);
  }, [query]);
  return matches;
}

interface GalleryContent {
  title: string;
  note: string;
  rows: number[];
  total: number;
  /** A short per-image detail, such as the cosine similarity. */
  details?: Map<number, string>;
}

interface NeighborControl {
  value: number;
  max: number;
  /** One notch past max stands for every edge of the graph row. */
  allowsAll: boolean;
  onChange: (value: number) => void;
}

/** The selected point's image beside its neighbors' (or a region's images), next to the canvas. */
function NeighborGallery({
  data,
  selectedIndex,
  gallery,
  onPick,
  neighborControl,
  header,
}: {
  data: PreparedEmbedding;
  selectedIndex: number | null;
  gallery: GalleryContent | null;
  onPick: (row: number) => void;
  neighborControl: NeighborControl | null;
  header?: React.ReactNode;
}) {
  if (!gallery) {
    return (
      <div>
        {header}
        <p className="text-[11px] leading-snug text-[#7f8ba2]">
          Click a point to see its image beside its neighbors&apos;, or shift-drag a region to see its images.
        </p>
      </div>
    );
  }
  const ownClass = selectedIndex == null ? -1 : data.classIndex[selectedIndex];
  return (
    <div>
      {selectedIndex != null && (
        <div className="mb-3">
          <SampleImage url={imageUrl(data, selectedIndex)} caption={data.classNames[ownClass]} size="large" />
          <p className="mt-1.5 truncate text-sm font-semibold text-white" title={data.classNames[ownClass]}>
            {data.classNames[ownClass]}
          </p>
          <p className="text-[11px] text-[#98a4ba]">
            {subsetStyle(data.subsets[data.subset[selectedIndex]]).label} · sample #{data.sample[selectedIndex]}
          </p>
        </div>
      )}
      {header}
      {neighborControl && selectedIndex != null && (
        <label className="mb-2 block">
          <span className="flex items-center justify-between text-[11px] text-[#98a4ba]">
            Neighbors
            <span className="tabular-nums text-[#c8d2e4]">
              {neighborControl.allowsAll && neighborControl.value > neighborControl.max
                ? "all graph edges"
                : `${neighborControl.value} of ${neighborControl.max}`}
            </span>
          </span>
          <input
            type="range"
            min={1}
            max={neighborControl.max + (neighborControl.allowsAll ? 1 : 0)}
            value={neighborControl.value}
            onChange={(event) => neighborControl.onChange(Number(event.target.value))}
            className="mt-1 w-full accent-[#3987e5]"
            aria-label="How many neighbors to draw and show"
          />
        </label>
      )}
      <p className="text-xs font-semibold text-[#c8d2e4]">{gallery.title}</p>
      <p className="mt-0.5 text-[11px] leading-snug text-[#7f8ba2]">{gallery.note}</p>
      <div className="mt-2 grid grid-cols-2 gap-2">
        {gallery.rows.map((row) => {
          const className = data.classNames[data.classIndex[row]];
          const sameClass = selectedIndex == null || data.classIndex[row] === ownClass;
          return (
            <button
              key={row}
              type="button"
              onClick={() => onPick(row)}
              className={`focus-ring rounded-lg border p-1 text-left hover:bg-white/[0.04] ${
                sameClass ? "border-white/[0.08]" : "border-[#d95926]/70"
              }`}
              title={`${className} · ${subsetStyle(data.subsets[data.subset[row]]).label} · sample #${data.sample[row]}`}
            >
              <SampleImage url={imageUrl(data, row)} caption={className} size="medium" />
              <span className="mt-1 block truncate text-[10px] text-[#c8d2e4]">{className}</span>
              <span className="flex items-center gap-1 text-[10px] text-[#7f8ba2]">
                {selectedIndex != null && (
                  <span
                    aria-hidden
                    className="h-[6px] w-[6px] shrink-0 rounded-full"
                    style={{ background: sameClass ? "#c8d2e4" : CATEGORICAL[1] }}
                  />
                )}
                {subsetStyle(data.subsets[data.subset[row]]).label}
                {selectedIndex != null && (sameClass ? " · same class" : " · other class")}
                {gallery.details?.get(row) ? ` · ${gallery.details.get(row)}` : ""}
              </span>
            </button>
          );
        })}
      </div>
      {gallery.total > gallery.rows.length && (
        <p className="mt-2 text-[11px] text-[#7f8ba2]">
          Showing {gallery.rows.length} of {formatCount(gallery.total)}.
        </p>
      )}
    </div>
  );
}

/** One metric across every tracked epoch, with the current and reference epochs marked. */
function Sparkline({
  series,
  epochIndex,
  referenceIndex,
  onPick,
}: {
  series: TimelineSeries;
  epochIndex: number;
  referenceIndex: number;
  onPick(index: number): void;
}) {
  const width = 130;
  const height = 30;
  const values = series.values;
  const finite = values.filter((value): value is number => value != null && Number.isFinite(value));
  if (finite.length === 0) return null;
  const low = Math.min(...finite);
  const high = Math.max(...finite);
  const span = high - low || 1;
  const x = (index: number) => (values.length === 1 ? width / 2 : 2 + (index / (values.length - 1)) * (width - 4));
  const y = (value: number) => height - 4 - ((value - low) / span) * (height - 8);
  let path = "";
  values.forEach((value, index) => {
    if (value == null || !Number.isFinite(value)) return;
    path += `${path ? "L" : "M"}${x(index).toFixed(1)},${y(value).toFixed(1)}`;
  });
  const marker = (index: number, fill: string) => {
    const value = values[index];
    return value == null || !Number.isFinite(value) ? null : <circle cx={x(index)} cy={y(value)} r={3} fill={fill} />;
  };
  return (
    <svg
      width={width}
      height={height}
      className="cursor-pointer"
      role="img"
      aria-label={`${series.label} across epochs`}
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const share = (event.clientX - rect.left - 2) / (width - 4);
        onPick(Math.max(0, Math.min(values.length - 1, Math.round(share * (values.length - 1)))));
      }}
    >
      <path d={path} fill="none" stroke="#8f9bb2" strokeWidth={1.5} />
      {marker(referenceIndex, NEUTRAL)}
      {marker(epochIndex, "#f7f8fc")}
    </svg>
  );
}

/** Every tracked epoch as a small frame, so a run reads as a sequence rather than its last state. */
function EpochFilmstrip({
  data,
  frames,
  epochIndex,
  referenceIndex,
  selectedIndex,
  visibleFlags,
  onPick,
}: {
  data: PreparedEmbedding;
  frames: Uint8Array[];
  epochIndex: number;
  referenceIndex: number;
  selectedIndex: number | null;
  visibleFlags: boolean[];
  onPick(index: number): void;
}) {
  const tracking = data.tracking!;
  return (
    <div className="flex gap-2 overflow-x-auto pb-1">
      {tracking.epochs.map((epoch, index) => (
        <FilmstripFrame
          key={epoch}
          data={data}
          layout={tracking.layouts[index]}
          colors={frames[index]}
          selectedIndex={selectedIndex}
          visibleFlags={visibleFlags}
          label={`${epoch}`}
          active={index === epochIndex}
          reference={index === referenceIndex}
          onClick={() => onPick(index)}
        />
      ))}
    </div>
  );
}

function FilmstripFrame({
  data,
  layout,
  colors,
  selectedIndex,
  visibleFlags,
  label,
  active,
  reference,
  onClick,
}: {
  data: PreparedEmbedding;
  layout: Float32Array;
  colors: Uint8Array;
  selectedIndex: number | null;
  visibleFlags: boolean[];
  label: string;
  active: boolean;
  reference: boolean;
  onClick(): void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const width = 132;
    const height = 96;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#0b1222";
    ctx.fillRect(0, 0, width, height);
    const scale = Math.min(width, height) * 0.46;
    ctx.globalAlpha = 0.85;
    for (let row = 0; row < data.count; row += 1) {
      if (!visibleFlags[data.subset[row]]) continue;
      ctx.fillStyle = CANVAS_COLORS[colors[row]];
      ctx.fillRect(width / 2 + layout[row * 2] * scale - 0.75, height / 2 - layout[row * 2 + 1] * scale - 0.75, 1.5, 1.5);
    }
    ctx.globalAlpha = 1;
    if (selectedIndex != null) {
      ctx.beginPath();
      ctx.arc(width / 2 + layout[selectedIndex * 2] * scale, height / 2 - layout[selectedIndex * 2 + 1] * scale, 4, 0, Math.PI * 2);
      ctx.strokeStyle = "#f7f8fc";
      ctx.lineWidth = 1.5;
      ctx.stroke();
    }
  }, [data, layout, colors, selectedIndex, visibleFlags]);
  return (
    <button
      type="button"
      onClick={onClick}
      className={`focus-ring shrink-0 rounded-lg border p-1 text-left transition ${
        active ? "border-[#9b8cff]" : reference ? "border-white/[0.3]" : "border-white/[0.07] hover:border-white/[0.2]"
      }`}
      aria-label={`Show epoch ${label}`}
    >
      <canvas ref={canvasRef} style={{ width: 132, height: 96 }} className="block rounded" />
      <span className="mt-0.5 block px-0.5 text-[10px] tabular-nums text-[#98a4ba]">
        epoch {label}
        {reference ? " · reference" : ""}
      </span>
    </button>
  );
}

function EmptyExportHint() {
  return (
    <div className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-5 text-sm leading-relaxed text-[#98a4ba]">
      <p className="font-semibold text-white">No exported views yet</p>
      <p className="mt-2">
        Open a <code className="rounded bg-white/[0.06] px-1.5 py-0.5">test_tsne.csv</code> above for a 2D view of D_test, or
        export runs into <code className="rounded bg-white/[0.06] px-1.5 py-0.5">public/embedding-views/</code>:
      </p>
      <pre className="mt-3 overflow-x-auto rounded-xl border border-white/[0.07] bg-black/40 p-3 text-[11px] leading-relaxed text-[#c8d2e4]">
{`# saved heads (--save_head_state): labeled/unlabeled, 2D + 3D, kNN graph, clusters
python evaluate/export_embedding_views.py --root logs/ablations/<dimension>
# epoch tracking: one aligned layout per tracked epoch, with the run's graph metrics
python evaluate/export_embedding_views.py --tracking-root logs/final_tracking
# test_tsne.csv: D_test only, 2D
python evaluate/export_embedding_views.py --tsne-csv-root logs/<replay>`}
      </pre>
    </div>
  );
}

/** Fetch a view's JSON, joining the pieces the hosted build splits large views into. */
async function fetchViewFile(entry: EmbeddingViewIndexEntry): Promise<EmbeddingViewFile> {
  const urls = entry.parts
    ? Array.from({ length: entry.parts }, (_, part) => `/embedding-views/${entry.file}.${part}`)
    : [`/embedding-views/${entry.file}`];
  const texts = await Promise.all(
    urls.map(async (url) => {
      const response = await fetch(url);
      if (!response.ok) throw new Error(`Could not load ${entry.file} (${response.status}).`);
      return response.text();
    }),
  );
  return JSON.parse(texts.join("")) as EmbeddingViewFile;
}

/** `shared` is the hosted build (npm run build:share): exported views only, no dev-server routes. */
export function EmbeddingExplorer({ index, shared = false }: { index: EmbeddingViewIndex | null; shared?: boolean }) {
  const canvasRef = useRef<EmbeddingCanvasHandle>(null);
  const [uploads, setUploads] = useState<PreparedEmbedding[]>([]);
  const [source, setSource] = useState<SourceKey | null>(null);
  const [runLabel, setRunLabel] = useState<string | null>(null);
  const [viewId, setViewId] = useState<string | null>(null);
  const [viewLookup, setViewLookup] = useState("");
  const [lookupFocused, setLookupFocused] = useState(false);
  const [lookupLoading, setLookupLoading] = useState(false);
  const [loaded, setLoaded] = useState<Map<string, PreparedEmbedding>>(() => new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<ViewMode>("2d");
  const [visibleSubsets, setVisibleSubsets] = useState<Record<string, boolean>>({});
  const [colorMode, setColorMode] = useState<ColorMode>("subset");
  const [splitEmphasis, setSplitEmphasis] = useState<SplitEmphasis>("none");
  const [outlineLabeled, setOutlineLabeled] = useState(true);
  const [showCenters, setShowCenters] = useState(false);
  const [pinned, setPinned] = useState<number[]>([]);
  const [selectedIndex, setSelectedIndex] = useState<number | null>(null);
  const [boxSelection, setBoxSelection] = useState<Int32Array | null>(null);
  const [pointSize, setPointSize] = useState(2);
  const [classQuery, setClassQuery] = useState("");
  const [dragActive, setDragActive] = useState(false);
  const [epochIndex, setEpochIndex] = useState(0);
  const [referenceIndex, setReferenceIndex] = useState(0);
  const [showTrails, setShowTrails] = useState(false);
  const [neighborCount, setNeighborCount] = useState(DEFAULT_NEIGHBOR_COUNT);
  const [lineSource, setLineSource] = useState<LineSource>("graph");
  const [propagationStep, setPropagationStep] = useState(-1);
  const [stepPlaying, setStepPlaying] = useState(false);
  const [diagnosticPredictor, setDiagnosticPredictor] = useState<Predictor>("lp");
  const [playing, setPlaying] = useState(false);
  const liveRuns = useLiveRuns(10_000, !shared);
  const [followNewest, setFollowNewest] = useState(false);
  const [now, setNow] = useState(() => Date.now());

  const options = useMemo<ViewOption[]>(() => {
    // A run the dev server reads live is not listed again under its exported snapshot.
    const liveDirs = new Set(liveRuns.runs.map((run) => run.dir));
    return [
      ...liveRuns.runs.map((run) => ({
        id: run.id,
        source: "live" as SourceKey,
        label: run.label,
        runLabel: run.runLabel,
        variant: liveVariantLabel(run),
        dataset: run.dataset,
        live: run,
      })),
      ...(index?.views ?? [])
        .filter((entry) => !(entry.source === "epoch_tracking" && entry.sourcePath && liveDirs.has(entry.sourcePath)))
        .map((entry) => ({
        id: entry.id,
        source: entry.source as SourceKey,
        label: entry.label,
        runLabel: entry.runLabel || entry.label,
        variant: variantLabel(entry),
        dataset: entry.dataset,
        entry,
      })),
      ...uploads.map((view) => ({
        id: view.id,
        source: "upload" as SourceKey,
        label: view.label,
        runLabel: view.label,
        variant: `${formatCount(view.count)} points · ${view.pos3 ? "2D + 3D" : "2D"}`,
        dataset: null,
        uploaded: view,
      })),
    ];
  }, [index, uploads, liveRuns.runs]);

  const sources = SOURCE_ORDER.filter((key) => options.some((option) => option.source === key));
  const viewMatches = useMemo(
    () =>
      options
        .map((candidate, order) => ({ candidate, order, score: viewSearchScore(candidate, viewLookup) }))
        .filter((match): match is { candidate: ViewOption; order: number; score: number } => match.score != null)
        .sort((left, right) => left.score - right.score || left.order - right.order)
        .slice(0, 8)
        .map((match) => match.candidate),
    [options, viewLookup],
  );
  const runsOfSource = useMemo(() => {
    const runs = new Map<string, ViewOption[]>();
    options
      .filter((option) => option.source === source)
      .forEach((option) => runs.set(option.runLabel, [...(runs.get(option.runLabel) ?? []), option]));
    // t-SNE first, so a run opens on its t-SNE view.
    runs.forEach((list) => list.sort((a, b) => projectionRank(a) - projectionRank(b)));
    return runs;
  }, [options, source]);
  const variants = runLabel ? runsOfSource.get(runLabel) ?? [] : [];

  const selectView = useCallback(
    (id: string | null) => {
      const option = options.find((candidate) => candidate.id === id);
      if (!option) return;
      setSource(option.source);
      setRunLabel(option.runLabel);
      setViewId(option.id);
    },
    [options],
  );

  // Restore the last view once the index is known.
  useEffect(() => {
    // Wait for the first live listing, so a remembered live run can be restored.
    if (viewId != null || options.length === 0 || liveRuns.available === null) return;
    // Links can pick the view: `?view=<id>`, or `?live=newest` for the most recently written live run.
    const linked = new URLSearchParams(window.location.search);
    const linkedId =
      linked.get("live") === "newest"
        ? options.find((candidate) => candidate.source === "live")?.id
        : options.find((candidate) => candidate.id === linked.get("view"))?.id;
    if (linkedId) {
      selectView(linkedId);
      return;
    }
    const stored = readStored();
    selectView(
      options.find((option) => option.id === stored.viewId)?.id ??
        (options.find((option) => projectionRank(option) === 0) ?? options[0]).id,
    );
    if (stored.mode) setMode(stored.mode);
  }, [options, selectView, viewId, liveRuns.available]);

  const option = options.find((candidate) => candidate.id === viewId) ?? null;
  const liveView = useLiveTrackingView(option?.live?.dir ?? null);
  const data = option?.uploaded ?? (option?.live ? liveView.data : option ? loaded.get(option.id) ?? null : null);

  // Follow newest run: whenever a run starts writing, switch to it.
  const newestRunning = liveRuns.runs.find((run) => run.running)?.id ?? null;
  useEffect(() => {
    if (followNewest && newestRunning && newestRunning !== viewId) selectView(newestRunning);
  }, [followNewest, newestRunning, selectView, viewId]);

  useEffect(() => {
    if (!option?.live) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [option?.live]);

  useEffect(() => {
    if (!option?.entry || loaded.has(option.id)) return;
    const entry = option.entry;
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetchViewFile(entry)
      .then((file) => {
        if (!cancelled) setLoaded((current) => new Map(current).set(entry.id, prepareEmbedding(file)));
      })
      .catch((reason: unknown) => {
        if (!cancelled) setError(reason instanceof Error ? reason.message : String(reason));
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [option, loaded]);

  // A new view starts with every subset visible, no selection, and its last epoch.
  // A live view that only grew keeps all of that; if it was showing its newest
  // epoch, it moves on to the epoch that just arrived.
  const resetKeyRef = useRef<string | null>(null);
  const epochCountRef = useRef(0);
  useEffect(() => {
    if (!data) return;
    const resetKey = `${data.id}|${data.count}|${data.subsets.join(",")}`;
    const count = data.tracking?.epochs.length ?? 0;
    if (resetKeyRef.current === resetKey) {
      const previous = epochCountRef.current;
      epochCountRef.current = count;
      if (count > previous) setEpochIndex((current) => (current >= previous - 1 ? count - 1 : current));
      return;
    }
    resetKeyRef.current = resetKey;
    epochCountRef.current = count;
    setVisibleSubsets(Object.fromEntries(data.subsets.map((name) => [name, true])));
    setPinned([]);
    setSelectedIndex(null);
    setBoxSelection(null);
    setSplitEmphasis("none");
    setPlaying(false);
    setEpochIndex(data.tracking ? data.tracking.epochs.length - 1 : 0);
    setReferenceIndex(0);
    // -1 is the settled diffusion: the step slider starts where the evidence is.
    setPropagationStep(-1);
    setStepPlaying(false);
    setLineSource("graph");
    setShowTrails(false);
    if (!data.pos3) setMode("2d");
  }, [data]);

  useEffect(() => {
    if (viewId) writeStored({ viewId, mode });
  }, [viewId, mode]);

  const epochCount = data?.tracking?.epochs.length ?? 0;
  // Every tracked epoch records the same number of steps, so the slider's range
  // does not move when the epoch does.
  const stepTotal = data?.tracking?.propagation?.find((frame) => frame)?.steps ?? 0;
  useEffect(() => {
    if (!playing || epochCount === 0) return;
    const timer = window.setInterval(() => {
      setEpochIndex((current) => {
        if (current + 1 >= epochCount) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 900);
    return () => window.clearInterval(timer);
  }, [playing, epochCount]);

  useEffect(() => {
    if (!stepPlaying || stepTotal === 0) return;
    const timer = window.setInterval(() => {
      setPropagationStep((current) => {
        const next = (current < 0 ? stepTotal - 1 : current) + 1;
        if (next >= stepTotal) {
          setStepPlaying(false);
          return stepTotal - 1;
        }
        return next;
      });
    }, 700);
    return () => window.clearInterval(timer);
  }, [stepPlaying, stepTotal]);

  const safeEpochIndex = Math.min(epochIndex, Math.max(0, epochCount - 1));
  const safeReferenceIndex = Math.min(referenceIndex, Math.max(0, epochCount - 1));
  const layout2d = data ? (data.tracking ? data.tracking.layouts[safeEpochIndex] : data.pos2) : null;
  const layout3d = data ? (data.tracking ? data.tracking.layouts3d?.[safeEpochIndex] ?? null : data.pos3) : null;
  // A live epoch can have its 2D projection before its 3D one.
  const has3d = data ? (data.tracking ? Boolean(data.tracking.layouts3d?.some(Boolean)) : data.pos3 != null) : false;
  const propagationFrame = data?.tracking?.propagation?.[safeEpochIndex] ?? null;
  const stepNumbers = useMemo(() => Array.from({ length: stepTotal }, (_, step) => step), [stepTotal]);
  const stepCount = propagationFrame?.steps ?? 0;
  const safeStep = stepCount ? (propagationStep < 0 ? stepCount - 1 : Math.min(propagationStep, stepCount - 1)) : 0;
  const derived = useMemo(
    () => (data ? deriveGraph(data, safeEpochIndex, propagationStep) : null),
    [data, safeEpochIndex, propagationStep],
  );
  const referenceDerived = useMemo(
    () => (data?.tracking ? deriveGraph(data, safeReferenceIndex) : null),
    [data, safeReferenceIndex],
  );
  const cluster = data ? data.tracking?.clusters?.[safeEpochIndex] ?? data.graph?.cluster ?? null : null;
  const clusterCount = data ? data.tracking?.clusterK || data.graph?.clusterK || 0 : 0;
  const timeline = useMemo(() => (data ? trackingTimeline(data) : []), [data]);

  const diagnosticPredictors = useMemo<Predictor[]>(() => {
    const tracking = data?.tracking;
    if (!tracking) return [];
    const available: Predictor[] = [];
    (["lp", "prototype", "hoffer"] as const).forEach((predictor) => {
      if (tracking.evidence?.some((evidence) => evidence?.[predictor])) available.push(predictor);
    });
    if (tracking.pseudoLabels) available.push("pseudo");
    return available;
  }, [data]);
  const activePredictor = diagnosticPredictors.includes(diagnosticPredictor) ? diagnosticPredictor : diagnosticPredictors[0] ?? "lp";

  /** Every predictor's frame at every epoch, computed once per view. */
  const predictorFrames = useMemo(() => {
    const frames: Partial<Record<Predictor, ReturnType<typeof predictorAt>[]>> = {};
    if (!data?.tracking) return frames;
    diagnosticPredictors.forEach((predictor) => {
      frames[predictor] = data.tracking!.epochs.map((_, index) => predictorAt(data, index, predictor));
    });
    return frames;
  }, [data, diagnosticPredictors]);
  const predictorAccuracy = useMemo(() => {
    const accuracy: Partial<Record<Predictor, (number | null)[]>> = {};
    diagnosticPredictors.forEach((predictor) => {
      accuracy[predictor] = (predictorFrames[predictor] ?? []).map((frame) => frameAccuracy(frame));
    });
    return accuracy;
  }, [diagnosticPredictors, predictorFrames]);
  const conditionalEntropy = useMemo(() => {
    const tracking = data?.tracking;
    if (!tracking) return [];
    const byEpoch = new Map(tracking.epochMetrics.map((row) => [row.epoch, row]));
    return (
      [
        ["labeled", "Labeled", CATEGORICAL[0]],
        ["unlabeled", "Unlabeled", CATEGORICAL[1]],
        ["test", "Test", CATEGORICAL[2]],
      ] as const
    )
      .map(([split, label, color]) => ({
        label,
        color,
        values: tracking.epochs.map((epoch) => byEpoch.get(epoch)?.[`conditional_entropy_${split}`] ?? null),
      }))
      .filter((series) => series.values.some((value) => value != null));
  }, [data]);
  const confidenceShareSeries = useMemo(
    () =>
      CONFIDENCE_SHARES.map((share, index) => ({
        label: share === 1 ? "All" : `Top ${percent(share)}`,
        // Ordinal: the more selective the share, the darker the step.
        color: RAMP[Math.min(RAMP.length - 1, 1 + index)],
        values: (predictorFrames[activePredictor] ?? []).map((frame) => topConfidenceAccuracy(frame, share)),
      })),
    [predictorFrames, activePredictor],
  );
  const confidenceHistograms = useMemo(() => {
    const frames = predictorFrames[activePredictor];
    if (!data?.tracking || !frames) return null;
    const indices = safeReferenceIndex === safeEpochIndex ? [safeEpochIndex] : [safeReferenceIndex, safeEpochIndex];
    const chosen = indices.map((index) => ({ index, frame: frames[index] })).filter((entry) => entry.frame);
    if (!chosen.length) return null;
    let top = 0;
    chosen.forEach(({ frame }) =>
      frame!.correct.forEach((value, row) => {
        if (value >= 0) top = Math.max(top, frame!.confidence[row]);
      }),
    );
    top = top > 0 ? top : 1;
    return chosen.map(({ index, frame }) => ({
      label: `Epoch ${data.tracking!.epochs[index]}`,
      top,
      ...confidenceHistogram(frame!, HISTOGRAM_BINS, top),
    }));
  }, [data, predictorFrames, activePredictor, safeEpochIndex, safeReferenceIndex]);
  const confusions = useMemo(() => {
    const frame = predictorFrames[activePredictor]?.[safeEpochIndex];
    return data && frame ? confidentConfusions(frame, data, CONFUSION_TOP_SHARE) : null;
  }, [data, predictorFrames, activePredictor, safeEpochIndex]);

  const availableModes = useMemo<ColorMode[]>(() => {
    const modes: ColorMode[] = ["subset", "class", "pinned"];
    if (derived?.agreement) modes.push("agreement");
    if (derived?.purity) modes.push("purity");
    if (data?.graph?.cluster || data?.tracking?.clusters) modes.push("cluster");
    if (data?.tracking && data.tracking.epochs.length > 1) {
      if (derived?.purity) modes.push("purityChange");
      if (derived?.agreement) modes.push("agreementChange");
    }
    if (data?.tracking?.evidence?.some((evidence) => evidence?.lp)) modes.push("lpPrediction", "lpWeight");
    if (data?.tracking?.evidence?.some((evidence) => evidence?.hoffer)) modes.push("hofferPrediction", "hofferCertainty");
    if (data?.tracking?.pseudoLabels) modes.push("pseudoLabel");
    if (data?.tracking?.evidence?.some((evidence) => evidence?.prototype)) modes.push("prototypePrediction");
    if (data?.tracking?.propagation?.some((frame) => frame)) {
      modes.push("propagationClass", "propagationCorrect", "propagationHop");
    }
    if (derived?.outlierScore) modes.push("outlier");
    if (data?.tracking?.centerSimilarity) modes.push("centerSimilarity");
    return modes;
  }, [data, derived]);
  const effectiveColorMode = availableModes.includes(colorMode) ? colorMode : "subset";

  const classColors = useMemo(
    () => (data && (effectiveColorMode === "class" || effectiveColorMode === "propagationClass") ? assignClassColors(data) : null),
    [data, effectiveColorMode],
  );
  const clusterColors = useMemo(
    () =>
      data && cluster && layout2d && clusterCount > 0 && effectiveColorMode === "cluster"
        ? assignGroupColors(cluster, clusterCount, layout2d, data.count)
        : null,
    [data, cluster, clusterCount, layout2d, effectiveColorMode],
  );

  const visibleFlags = useMemo(
    () => (data ? data.subsets.map((name) => visibleSubsets[name] !== false) : []),
    [data, visibleSubsets],
  );

  const selectionMask = useMemo(() => {
    if (!data || !boxSelection) return null;
    const mask = new Uint8Array(data.count);
    boxSelection.forEach((row) => {
      mask[row] = 1;
    });
    return mask;
  }, [data, boxSelection]);

  const encoding = useMemo(
    () =>
      data && derived
        ? encodePoints({
            view: data,
            colorMode: effectiveColorMode,
            classColors,
            clusterColors,
            clusters: cluster,
            derived,
            reference: referenceDerived,
            pinnedClasses: pinned,
            splitEmphasis,
            selectedIndex,
            selectionMask,
          })
        : null,
    [
      data,
      derived,
      referenceDerived,
      cluster,
      effectiveColorMode,
      classColors,
      clusterColors,
      pinned,
      splitEmphasis,
      selectedIndex,
      selectionMask,
    ],
  );

  // The same encoding for every tracked epoch, for the filmstrip. Per-epoch
  // quantities (purity, agreement, clusters) are read at each frame's own epoch.
  const filmstripFrames = useMemo(() => {
    if (!data?.tracking || !derived) return null;
    return data.tracking.epochs.map((_, index) => {
      const frameDerived = deriveGraph(data, index, propagationStep);
      const frameClusters = data.tracking!.clusters?.[index] ?? null;
      return encodePoints({
        view: data,
        colorMode: effectiveColorMode,
        classColors,
        clusterColors:
          effectiveColorMode === "cluster" && frameClusters && clusterCount > 0
            ? assignGroupColors(frameClusters, clusterCount, data.tracking!.layouts[index], data.count)
            : null,
        clusters: frameClusters,
        derived: frameDerived,
        reference: referenceDerived,
        pinnedClasses: pinned,
        splitEmphasis: "none",
        selectedIndex: null,
        selectionMask: null,
      }).color;
    });
  }, [data, derived, referenceDerived, effectiveColorMode, classColors, clusterCount, pinned, propagationStep]);

  const changeSummary = useMemo(() => {
    if (!data || !encoding || (effectiveColorMode !== "purityChange" && effectiveColorMode !== "agreementChange")) return null;
    const unlabeledSubset = data.subsets.indexOf("unlabeled");
    let better = 0;
    let worse = 0;
    let same = 0;
    for (let row = 0; row < data.count; row += 1) {
      if (unlabeledSubset >= 0 && data.subset[row] !== unlabeledSubset) continue;
      const color = encoding.color[row];
      if (color === CHANGE_BETTER_INDEX) better += 1;
      else if (color === CHANGE_WORSE_INDEX) worse += 1;
      else if (color === CHANGE_NONE_INDEX) same += 1;
    }
    return { better, worse, same };
  }, [data, encoding, effectiveColorMode]);

  const trails = useMemo(() => {
    if (!data?.tracking || !showTrails || safeReferenceIndex === safeEpochIndex) return null;
    let rows: number[] = [];
    if (boxSelection) rows = Array.from(boxSelection);
    else if (selectedIndex != null) {
      const ownClass = data.classIndex[selectedIndex];
      for (let row = 0; row < data.count; row += 1) if (data.classIndex[row] === ownClass) rows.push(row);
    } else if (pinned.length > 0) {
      for (let row = 0; row < data.count; row += 1) if (pinned.includes(data.classIndex[row])) rows.push(row);
    }
    return rows.length ? { rows: Int32Array.from(rows.slice(0, 2000)), from: data.tracking.layouts[safeReferenceIndex] } : null;
  }, [data, showTrails, safeReferenceIndex, safeEpochIndex, boxSelection, selectedIndex, pinned]);

  /** Per cluster: size, largest class and its share, over every point of the view. */
  const clusterSummary = useMemo(() => {
    if (!data || !cluster || clusterCount <= 0) return null;
    const size = new Int32Array(clusterCount);
    const classCounts = Array.from({ length: clusterCount }, () => new Map<number, number>());
    for (let row = 0; row < data.count; row += 1) {
      const group = cluster[row];
      if (group < 0) continue;
      size[group] += 1;
      const counts = classCounts[group];
      counts.set(data.classIndex[row], (counts.get(data.classIndex[row]) ?? 0) + 1);
    }
    const majorityClass = new Int32Array(clusterCount).fill(-1);
    const majorityShare = new Float32Array(clusterCount);
    classCounts.forEach((counts, group) => {
      let best = -1;
      let bestCount = 0;
      counts.forEach((count, classIndex) => {
        if (count > bestCount) {
          best = classIndex;
          bestCount = count;
        }
      });
      majorityClass[group] = best;
      majorityShare[group] = size[group] ? bestCount / size[group] : 0;
    });
    return { size, majorityClass, majorityShare };
  }, [data, cluster, clusterCount]);

  const labeledRows = useMemo(() => {
    if (!data) return [];
    const labeledSubset = data.subsets.indexOf("labeled");
    const rows: number[] = [];
    for (let row = 0; row < data.count; row += 1) if (data.subset[row] === labeledSubset) rows.push(row);
    return rows;
  }, [data]);

  const anchors = useMemo(() => {
    if (!derived?.pulls || !derived.pullsSameClass) return [];
    return labeledRows
      .filter((row) => derived.pulls![row] > 0)
      .sort((left, right) => derived.pulls![right] - derived.pulls![left])
      .slice(0, 8)
      .map((row) => ({ row, pulls: derived.pulls![row], same: derived.pullsSameClass![row] }));
  }, [derived, labeledRows]);

  /** Unlabeled points at this epoch: how many the chosen prediction mode gets right, and pseudo-label counts. */
  const evidenceSummary = useMemo(() => {
    if (!data || !derived) return null;
    const correct =
      effectiveColorMode === "lpPrediction"
        ? derived.lpCorrect
        : effectiveColorMode === "hofferPrediction"
          ? derived.hofferCorrect
          : effectiveColorMode === "prototypePrediction"
            ? derived.prototypeCorrect
            : null;
    if (correct) {
      let right = 0;
      let total = 0;
      correct.forEach((value) => {
        if (value < 0) return;
        total += 1;
        right += value;
      });
      return total ? { kind: "prediction" as const, right, total } : null;
    }
    if (effectiveColorMode === "pseudoLabel" && derived.pseudo) {
      const counts = [0, 0, 0];
      derived.pseudo.forEach((value) => {
        if (value >= 0) counts[value] += 1;
      });
      return { kind: "pseudo" as const, rejected: counts[0], wrong: counts[1], right: counts[2] };
    }
    return null;
  }, [data, derived, effectiveColorMode]);

  const agreementSummary = useMemo(() => {
    if (!data || !derived?.agreement) return null;
    const unlabeledSubset = data.subsets.indexOf("unlabeled");
    let same = 0;
    let total = 0;
    for (let row = 0; row < data.count; row += 1) {
      if (unlabeledSubset >= 0 && data.subset[row] !== unlabeledSubset) continue;
      const value = derived.agreement[row];
      if (value < 0) continue;
      total += 1;
      same += value;
    }
    return total ? { same, total } : null;
  }, [data, derived]);

  // How many neighbors to draw and show: a node's own ranked search (up to the
  // graph's k, plus "all edges" for the whole graph row), a saved head's kNN, or,
  // without a graph, the nearest points on screen.
  const epochNeighbors = data?.tracking?.neighbors?.[safeEpochIndex] ?? null;
  const neighborMax = epochNeighbors
    ? epochNeighbors.knn
      ? epochNeighbors.k
      : 0
    : data?.graph
      ? data.graph.k
      : LAYOUT_NEIGHBORS_MAX;
  const neighborAllowsAll = Boolean(epochNeighbors?.knn);
  const shownNeighbors = Math.max(1, Math.min(neighborCount, neighborMax + (neighborAllowsAll ? 1 : 0)));
  const showAllEdges = epochNeighbors != null && (!epochNeighbors.knn || shownNeighbors > neighborMax);

  /** The selected point's graph neighbors at this epoch: ranked by similarity where the run saved its search. */
  const selectedNeighbors = useMemo(() => {
    const lists = epochNeighbors;
    if (!data || !lists || selectedIndex == null) return null;
    const labeledSubset = data.subsets.indexOf("labeled");
    const ownClass = data.classIndex[selectedIndex];
    const graphRows = Array.from(lists.targets.subarray(lists.offsets[selectedIndex], lists.offsets[selectedIndex + 1]));
    const similarity = new Map<number, number>();
    const ranked: number[] = [];
    if (lists.knn) {
      for (let slot = 0; slot < lists.k; slot += 1) {
        const row = lists.knn[selectedIndex * lists.k + slot];
        if (row < 0) continue;
        ranked.push(row);
        if (lists.knnSimilarity) similarity.set(row, lists.knnSimilarity[selectedIndex * lists.k + slot]);
      }
    }
    // The whole graph row: its own ranked neighbors first, then the points that
    // listed it (labeled first, then its class).
    const inRanking = new Set(ranked);
    const rest = graphRows.filter((row) => !inRanking.has(row));
    const rank = (row: number) => (data.subset[row] === labeledSubset ? 0 : 2) + (data.classIndex[row] === ownClass ? 0 : 1);
    rest.sort((a, b) => rank(a) - rank(b));
    const shown = showAllEdges ? [...ranked.filter((row) => graphRows.includes(row)), ...rest] : ranked.slice(0, shownNeighbors);
    const count = (rows: number[], test: (row: number) => boolean) => rows.filter(test).length;
    return {
      shown,
      graphCount: graphRows.length,
      rankedCount: ranked.length,
      similarity,
      sameClass: count(shown, (row) => data.classIndex[row] === ownClass),
      labeled: count(shown, (row) => data.subset[row] === labeledSubset),
      labeledSameClass: count(shown, (row) => data.subset[row] === labeledSubset && data.classIndex[row] === ownClass),
    };
  }, [data, epochNeighbors, selectedIndex, shownNeighbors, showAllEdges]);

  const epochMinedPairs = data?.tracking?.minedPairs?.[safeEpochIndex] ?? null;
  const showMinedPairs = lineSource === "mined" && epochMinedPairs != null;

  /** SLADE: the sampled pairs this epoch mined with the selected point, positives first. */
  const selectedMinedPairs = useMemo(() => {
    if (!data || !epochMinedPairs || selectedIndex == null || !showMinedPairs) return null;
    const partners: { row: number; positive: boolean; similarity: number; sameCluster: number }[] = [];
    for (let pair = 0; pair < epochMinedPairs.count; pair += 1) {
      const from = epochMinedPairs.from[pair];
      const to = epochMinedPairs.to[pair];
      if (from !== selectedIndex && to !== selectedIndex) continue;
      partners.push({
        row: from === selectedIndex ? to : from,
        positive: epochMinedPairs.positive[pair] === 1,
        similarity: epochMinedPairs.similarity[pair],
        sameCluster: epochMinedPairs.sameCluster[pair],
      });
    }
    partners.sort((a, b) =>
      a.positive !== b.positive ? (a.positive ? -1 : 1) : a.positive ? b.similarity - a.similarity : a.similarity - b.similarity,
    );
    return partners;
  }, [data, epochMinedPairs, selectedIndex, showMinedPairs]);

  const showPropagation = lineSource === "propagation" && propagationFrame != null;

  /**
   * The selected point's place in the propagation: the chain of edges its mass
   * came over, back to the labeled node it started from; the points that were
   * reached through it; and the neighbors holding the most of its class now.
   */
  const selectedPropagation = useMemo(() => {
    if (!data || !propagationFrame || selectedIndex == null) return null;
    const frame = propagationFrame;
    const route: number[] = [selectedIndex];
    const seen = new Set<number>([selectedIndex]);
    let node = selectedIndex;
    while (frame.parent[node] >= 0 && !seen.has(frame.parent[node])) {
      node = frame.parent[node];
      route.push(node);
      seen.add(node);
    }
    const children: number[] = [];
    for (let row = 0; row < data.count && children.length < MAX_PULL_EDGES; row += 1) {
      if (frame.parent[row] === selectedIndex) children.push(row);
    }
    const contributors: { row: number; share: number }[] = [];
    for (let slot = 0; slot < frame.influenceK; slot += 1) {
      const row = frame.influence ? frame.influence[selectedIndex * frame.influenceK + slot] : -1;
      if (row < 0) continue;
      contributors.push({ row, share: frame.influenceShare ? frame.influenceShare[selectedIndex * frame.influenceK + slot] : 0 });
    }
    return {
      frame,
      // Nearest first: the point itself, its parent, ..., the labeled node.
      route,
      children,
      contributors,
      arrivalStep: frame.arrivalStep[selectedIndex],
      hop: frame.hop[selectedIndex],
      source: frame.source[selectedIndex],
      parent: frame.parent[selectedIndex],
      parentShare: frame.parentShare[selectedIndex],
      heldClass: frame.stepClass[selectedIndex * frame.steps + safeStep],
    };
  }, [data, propagationFrame, selectedIndex, safeStep]);

  const edges = useMemo<CanvasEdge[]>(() => {
    if (!data) return [];
    const list: CanvasEdge[] = [];
    if (showPropagation && propagationFrame) {
      const frame = propagationFrame;
      const edge = (from: number, to: number, width: number) => {
        if (from < 0 || to < 0 || from === to) return;
        list.push({
          from,
          to,
          width,
          // A label that arrived over an edge between two classes is the whole
          // question, so the line says which kind of edge carried it.
          color: data.classIndex[from] === data.classIndex[to] ? LINK_SAME : LINK_DIFFERENT,
        });
      };
      if (selectedPropagation) {
        selectedPropagation.route.forEach((row, position) => {
          if (position + 1 < selectedPropagation.route.length) edge(row, selectedPropagation.route[position + 1], 2.5);
        });
        selectedPropagation.children.forEach((row) => edge(row, selectedIndex!, 1));
        selectedPropagation.contributors.forEach(({ row, share }) =>
          list.push({ from: selectedIndex!, to: row, width: 1 + 3 * share, color: CATEGORICAL[0] }),
        );
        return list;
      }
      // Nothing selected: every edge the mass has travelled so far, with the
      // ones that carried it in this step at full strength over the earlier ones.
      for (let row = 0; row < data.count && list.length < MAX_PROPAGATION_EDGES; row += 1) {
        const step = frame.arrivalStep[row];
        const parent = frame.parent[row];
        if (step < 0 || step > safeStep || parent < 0) continue;
        const arriving = step === safeStep;
        list.push({
          from: row,
          to: parent,
          width: arriving ? 1.4 : 0.6,
          color: (() => {
            const base = data.classIndex[row] === data.classIndex[parent] ? LINK_SAME : LINK_DIFFERENT;
            return arriving ? base : fade(base, PAST_FRONT_ALPHA);
          })(),
        });
      }
      return list;
    }
    if (selectedIndex == null) return [];
    const ownClass = data.classIndex[selectedIndex];
    const link = (to: number, width: number, sameColor = LINK_SAME) => {
      if (to < 0 || to === selectedIndex) return;
      list.push({
        from: selectedIndex,
        to,
        width,
        color: data.classIndex[to] === ownClass ? sameColor : LINK_DIFFERENT,
      });
    };
    if (data.graph) {
      const isLabeled = data.subsets[data.subset[selectedIndex]] === "labeled";
      if (isLabeled) {
        const unlabeledSubset = data.subsets.indexOf("unlabeled");
        for (let row = 0; row < data.count && list.length < MAX_PULL_EDGES; row += 1) {
          if (data.subset[row] === unlabeledSubset && data.graph.nearestLabeled[row] === selectedIndex) link(row, 1);
        }
      } else {
        const { k, knn, nearestLabeled } = data.graph;
        for (let slot = 0; slot < Math.min(k, shownNeighbors); slot += 1) link(knn[selectedIndex * k + slot], 1);
        link(nearestLabeled[selectedIndex], 2.5, CATEGORICAL[0]);
      }
    } else if (data.tracking) {
      if (selectedMinedPairs) {
        // Mined pairs are colored by what Eq 7 made of them, not by class.
        selectedMinedPairs.forEach((partner) =>
          list.push({
            from: selectedIndex,
            to: partner.row,
            width: partner.positive ? 1.6 : 1,
            color: partner.positive ? CATEGORICAL[0] : CATEGORICAL[1],
          }),
        );
      } else if (selectedNeighbors) {
        // Edges to labeled samples are thicker: they are where the labels come from.
        const labeledSubset = data.subsets.indexOf("labeled");
        selectedNeighbors.shown.forEach((to) => link(to, data.subset[to] === labeledSubset ? 2 : 1));
      } else {
        const strongest = data.tracking.nodeMetrics[safeEpochIndex]?.strongest_neighbor;
        if (strongest) link(strongest[selectedIndex], 2.5, CATEGORICAL[0]);
      }
    }
    return list;
  }, [data, selectedIndex, safeEpochIndex, selectedNeighbors, shownNeighbors, selectedMinedPairs, showPropagation, propagationFrame, selectedPropagation, safeStep]);

  const galleryAsColumn = useMediaQuery(GALLERY_COLUMN_QUERY);
  const gallery = useMemo<GalleryContent | null>(() => {
    if (!data) return null;
    if (boxSelection) {
      return {
        title: `Region · ${formatCount(boxSelection.length)} points`,
        note: "In selection order; click an image to select that point.",
        rows: Array.from(boxSelection.subarray(0, MAX_GALLERY_IMAGES)),
        total: boxSelection.length,
      };
    }
    if (selectedIndex == null) return null;
    if (showPropagation && selectedPropagation) {
      const { route, children, contributors, hop, source } = selectedPropagation;
      const details = new Map<number, string>();
      route.forEach((row, position) => {
        if (position === 0) return;
        details.set(
          row,
          position === 1 ? `delivered it · ${percent(selectedPropagation.parentShare)} of the arriving mass` : `${position} edges back`,
        );
      });
      contributors.forEach(({ row, share }) => {
        const already = details.get(row);
        details.set(row, `${already ? `${already} · ` : ""}holds ${percent(share)} of its class's incoming mass`);
      });
      children.forEach((row) => details.set(row, details.get(row) ?? "reached through this point"));
      const rows = [...route.slice(1), ...contributors.map(({ row }) => row), ...children].filter(
        (row, position, all) => all.indexOf(row) === position,
      );
      return {
        title:
          hop < 0
            ? "Never reached by the propagation"
            : hop === 0
              ? `A labeled seed of ${data.classNames[data.classIndex[selectedIndex]]}`
              : `Its label came ${hop} edge${hop === 1 ? "" : "s"} from ${
                  source >= 0 ? `a labeled point of ${data.classNames[data.classIndex[source]]}` : "nowhere"
                }`,
        note: "The chain back to the labeled node the mass started from, the neighbors holding the most of its class, then the points reached through it. Orange lines cross a class boundary.",
        rows: rows.slice(0, MAX_GALLERY_IMAGES),
        total: rows.length,
        details,
      };
    }
    if (selectedMinedPairs && epochMinedPairs) {
      const details = new Map<number, string>();
      selectedMinedPairs.forEach((partner) =>
        details.set(
          partner.row,
          `${partner.positive ? "mined +" : "mined −"} · cos ${partner.similarity.toFixed(2)}${
            partner.sameCluster < 0 ? "" : partner.sameCluster ? " · same cluster" : " · other cluster"
          }`,
        ),
      );
      const positives = selectedMinedPairs.filter((partner) => partner.positive).length;
      return {
        title: `${positives} positive · ${selectedMinedPairs.length - positives} negative mined pairs · epoch ${
          data.tracking?.epochs[safeEpochIndex]
        }`,
        note: `A sample of this epoch's mining (${formatCount(epochMinedPairs.sampledPairs)} of ${formatCount(
          Math.round(epochMinedPairs.minedPositivePairs + epochMinedPairs.minedNegativePairs),
        )} pairs saved). Blue lines: mined positives; orange: negatives. The frame shows whether the partner's true class matches.`,
        rows: selectedMinedPairs.map((partner) => partner.row).slice(0, MAX_GALLERY_IMAGES),
        total: selectedMinedPairs.length,
        details,
      };
    }
    if (selectedNeighbors) {
      const epoch = data.tracking?.epochs[safeEpochIndex];
      const details = new Map<number, string>();
      selectedNeighbors.similarity.forEach((value, row) => details.set(row, `cos ${value.toFixed(2)}`));
      return {
        title: showAllEdges
          ? `All ${selectedNeighbors.shown.length} graph neighbors · epoch ${epoch}`
          : `${selectedNeighbors.shown.length} nearest of ${selectedNeighbors.rankedCount} searched · epoch ${epoch}`,
        note: showAllEdges
          ? selectedNeighbors.rankedCount
            ? "Its own ranked neighbors first, then the points whose search found it (labeled first)."
            : "This run saved no ranking: every edge of its graph, labeled neighbors first."
          : "Its own neighbor search at this epoch, most similar first.",
        rows: selectedNeighbors.shown.slice(0, MAX_GALLERY_IMAGES),
        total: selectedNeighbors.shown.length,
        details,
      };
    }
    if (data.graph) {
      const { k, knn, nearestLabeled } = data.graph;
      const rows: number[] = [];
      const add = (row: number) => {
        if (row >= 0 && row !== selectedIndex && !rows.includes(row)) rows.push(row);
      };
      if (data.subsets[data.subset[selectedIndex]] !== "labeled") add(nearestLabeled[selectedIndex]);
      for (let slot = 0; slot < Math.min(k, shownNeighbors); slot += 1) add(knn[selectedIndex * k + slot]);
      return {
        title: `${rows.length} nearest train points`,
        note: "Nearest in the trained embedding (cosine), its nearest labeled sample first.",
        rows,
        total: rows.length,
      };
    }
    // No graph for this view: the points nearest on screen, in the layout being drawn.
    const layout = mode === "3d" && layout3d ? layout3d : layout2d;
    if (!layout) return null;
    const dimensions = layout === layout3d && mode === "3d" ? 3 : 2;
    const nearest: [number, number][] = [];
    for (let row = 0; row < data.count; row += 1) {
      if (row === selectedIndex || !visibleFlags[data.subset[row]]) continue;
      let distance = 0;
      for (let axis = 0; axis < dimensions; axis += 1) {
        const delta = layout[row * dimensions + axis] - layout[selectedIndex * dimensions + axis];
        distance += delta * delta;
      }
      if (nearest.length < shownNeighbors || distance < nearest[nearest.length - 1][0]) {
        nearest.push([distance, row]);
        nearest.sort((a, b) => a[0] - b[0]);
        if (nearest.length > shownNeighbors) nearest.pop();
      }
    }
    return {
      title: `${nearest.length} nearest in the ${data.projection} layout`,
      note:
        data.tracking != null
          ? "This run saved no graph edges, so these are the nearest points on screen, not in the embedding."
          : "This view has no graph, so these are the nearest points on screen, not in the embedding.",
      rows: nearest.map(([, row]) => row),
      total: nearest.length,
    };
  }, [
    data,
    boxSelection,
    selectedIndex,
    selectedNeighbors,
    safeEpochIndex,
    mode,
    layout2d,
    layout3d,
    visibleFlags,
    shownNeighbors,
    showAllEdges,
    selectedMinedPairs,
    epochMinedPairs,
    showPropagation,
    selectedPropagation,
  ]);

  /** What the lines and the image column follow: the graph, SLADE's mining, or the propagation. */
  const lineSourceControl =
    data?.tracking?.minedPairs || data?.tracking?.propagation ? (
      <div className="mb-2">
        <Segmented<LineSource>
          label="Lines and images for the selected point"
          value={lineSource}
          onChange={setLineSource}
          options={[
            { value: "graph", label: "Graph neighbors" },
            ...(data?.tracking?.minedPairs
              ? [
                  {
                    value: "mined" as const,
                    label: "SLADE mined pairs",
                    disabled: !epochMinedPairs,
                    title: epochMinedPairs ? undefined : "No mined pairs saved for this epoch",
                  },
                ]
              : []),
            ...(data?.tracking?.propagation
              ? [
                  {
                    value: "propagation" as const,
                    label: "Label propagation",
                    disabled: !propagationFrame,
                    title: propagationFrame ? undefined : "No propagation saved for this epoch",
                  },
                ]
              : []),
          ]}
        />
      </div>
    ) : null;

  const pinClass = useCallback((classIndex: number) => {
    setPinned((current) => {
      if (current.includes(classIndex)) return current;
      // Three identities is what a scatter's all-pairs color check allows; the
      // oldest pin makes room for the newest.
      return [...current, classIndex].slice(-MAX_PINNED_CLASSES);
    });
    setColorMode("pinned");
  }, []);

  const selectPoint = useCallback((row: number) => {
    setBoxSelection(null);
    setSelectedIndex(row);
  }, []);

  const goToPoint = useCallback((row: number) => {
    selectPoint(row);
    canvasRef.current?.focusPoint(row);
  }, [selectPoint]);

  const onUpload = useCallback(async (files: FileList | File[]) => {
    const prepared: PreparedEmbedding[] = [];
    const problems: string[] = [];
    for (const file of Array.from(files)) {
      try {
        const text = await file.text();
        const view = file.name.toLowerCase().endsWith(".json")
          ? (JSON.parse(text) as EmbeddingViewFile)
          : parseProjectionCsv(text, file.name);
        prepared.push(prepareEmbedding(view));
      } catch (reason) {
        problems.push(`${file.name}: ${reason instanceof Error ? reason.message : String(reason)}`);
      }
    }
    if (prepared.length > 0) {
      setUploads((current) => [...prepared, ...current.filter((view) => !prepared.some((next) => next.id === view.id))]);
      setSource("upload");
      setRunLabel(prepared[0].label);
      setViewId(prepared[0].id);
    }
    setError(problems.length > 0 ? problems.join(" ") : null);
  }, []);

  const openViewLookup = useCallback(async () => {
    const query = viewLookup.trim();
    if (!query) return;
    if (viewMatches.length > 0) {
      selectView(viewMatches[0].id);
      setLookupFocused(false);
      setError(null);
      return;
    }
    if (shared || !looksLikeEmbeddingPath(query)) {
      setError(`No exported embedding view matches “${query}”.`);
      return;
    }

    setLookupLoading(true);
    setError(null);
    try {
      const response = await fetch(`/api/embedding-view/?path=${encodeURIComponent(query)}`);
      const result = (await response.json()) as OpenedEmbeddingPath;
      if (!response.ok || !result.ok || !result.csv || !result.sourcePath || !result.runDir) {
        throw new Error(result.error || `Could not open that embedding path (${response.status}).`);
      }

      const view = parseProjectionCsv(result.csv, result.sourcePath.split("/").pop() || "test_tsne.csv");
      if (result.csv3d) {
        const twin = parseProjectionCsv(result.csv3d, "test_tsne_3d.csv");
        const sameSamples =
          twin.points.sample.length === view.points.sample.length &&
          twin.points.sample.every((sample, index) => sample === view.points.sample[index]);
        if (sameSamples && twin.layouts["3d"]) view.layouts["3d"] = twin.layouts["3d"];
      }
      view.id = `path:${result.sourcePath}`;
      view.label = result.label || result.runDir;
      view.runDir = result.runDir;
      view.sourcePath = result.sourcePath;
      view.notes = [
        `Opened ${result.sourcePath} from the pasted local path.`,
        ...(view.notes ?? []).filter((note) => !note.startsWith("Uploaded ")),
      ];
      const prepared = prepareEmbedding(view);
      setUploads((current) => [prepared, ...current.filter((candidate) => candidate.id !== prepared.id)]);
      setSource("upload");
      setRunLabel(prepared.label);
      setViewId(prepared.id);
      setLookupFocused(false);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : String(reason));
    } finally {
      setLookupLoading(false);
    }
  }, [selectView, shared, viewLookup, viewMatches]);

  const classNameIndex = useMemo(() => {
    const lookup = new Map<string, number>();
    data?.classNames.forEach((name, classIndex) => lookup.set(name.toLowerCase(), classIndex));
    return lookup;
  }, [data]);

  const classMatches = useMemo(() => {
    if (!data || !classQuery.trim()) return [];
    const needle = classQuery.trim().toLowerCase();
    const matches: number[] = [];
    for (let classIndex = 0; classIndex < data.classNames.length && matches.length < 8; classIndex += 1) {
      if (data.classNames[classIndex].toLowerCase().includes(needle)) matches.push(classIndex);
    }
    return matches;
  }, [data, classQuery]);

  const describePoint = useCallback(
    (row: number): TooltipContent => {
      if (!data) return { title: "", lines: [] };
      const subsetName = data.subsets[data.subset[row]];
      const lines = [
        subsetName === "unlabeled"
          ? "Unlabeled · class shown is its true label, hidden during training"
          : `${subsetStyle(subsetName).label}`,
        `Sample #${data.sample[row]}`,
      ];
      if (data.graph) {
        const nearest = data.graph.nearestLabeled[row];
        if (nearest >= 0 && subsetName !== "labeled") {
          const same = data.classIndex[nearest] === data.classIndex[row];
          lines.push(
            `Nearest labeled: ${data.classNames[data.classIndex[nearest]]} · ${same ? "same class" : "different class"} · cos ${data.graph.nearestLabeledSimilarity[row].toFixed(2)}`,
          );
        }
        const purity = derived?.purity?.[row];
        if (purity != null && Number.isFinite(purity)) {
          lines.push(`${Math.round(purity * data.graph.k)} of ${data.graph.k} nearest train points share its class`);
        }
        if (subsetName === "labeled" && derived?.pulls) {
          lines.push(`Nearest labeled for ${derived.pulls[row]} unlabeled · ${derived.pullsSameClass![row]} of its class`);
        }
        const group = cluster ? cluster[row] : -1;
        if (group >= 0 && clusterSummary) {
          lines.push(`Cluster ${group} · ${percent(clusterSummary.majorityShare[group])} one class`);
        }
      } else if (data.tracking) {
        const metrics = data.tracking.nodeMetrics[safeEpochIndex] ?? {};
        const fraction = metrics.same_label_neighbor_fraction?.[row];
        if (fraction != null && Number.isFinite(fraction)) lines.push(`Graph neighbors of its class: ${percent(fraction)}`);
        const group = cluster ? cluster[row] : -1;
        if (group >= 0 && clusterSummary) {
          const refresh = data.tracking.clusterRefreshEpochs[safeEpochIndex];
          lines.push(
            `SLADE cluster ${group}${refresh != null ? ` (refresh at epoch ${refresh})` : ""} · ${percent(clusterSummary.majorityShare[group])} one class`,
          );
        }
        const evidence = data.tracking.evidence?.[safeEpochIndex];
        const describeMap = (name: string, map: EvidenceMap | null | undefined, confidence: string) => {
          if (!map) return;
          const top = map.topClass[row * map.topK];
          if (top < 0) {
            lines.push(`${name}: no labeled point reaches it`);
            return;
          }
          const verdict = subsetName === "unlabeled" ? (top === data.classIndex[row] ? " · right" : " · wrong") : "";
          lines.push(
            `${name}: ${data.classNames[top]} (${percent(map.topProbability[row * map.topK], 1)})${verdict} · ${confidence}`,
          );
        };
        if (evidence?.lp) describeMap("Label propagation", evidence.lp, `ω ${(evidence.lp.weight?.[row] ?? 0).toFixed(2)}`);
        if (evidence?.hoffer) describeMap("Hoffer", evidence.hoffer, `certainty ${evidence.hoffer.certainty[row].toFixed(3)}`);
        if (evidence?.prototype) {
          describeMap("Network", evidence.prototype, `certainty ${evidence.prototype.certainty[row].toFixed(3)}`);
        }
        const outlierScore = derived?.outlierScore?.[row];
        if (outlierScore != null && Number.isFinite(outlierScore)) {
          lines.push(`kNN distance ${outlierScore.toFixed(3)}${derived?.outlier?.[row] ? ` · an outlier (top ${percent(OUTLIER_SHARE)})` : ""}`);
        }
        const centerSimilarity = data.tracking.centerSimilarity?.[safeEpochIndex]?.[row];
        if (centerSimilarity != null && Number.isFinite(centerSimilarity)) {
          lines.push(`Cosine to its SLADE cluster center: ${centerSimilarity.toFixed(2)}`);
        }
        const pseudo = data.tracking.pseudoLabels?.[safeEpochIndex];
        if (pseudo && pseudo.label[row] >= 0) {
          lines.push(
            `Pseudo label: ${data.classNames[pseudo.label[row]]} · ${pseudo.accepted[row] ? "accepted" : "rejected"} · confidence ${pseudo.confidence[row].toFixed(2)}`,
          );
        }
        const hops = metrics.distance_to_labeled_node?.[row];
        const correctHops = metrics.distance_to_correct_label_labeled_node?.[row];
        if (hops != null && Number.isFinite(hops)) {
          lines.push(
            `Hops to a labeled node: ${hops}${correctHops != null && Number.isFinite(correctHops) ? ` · to one of its class: ${correctHops}` : ""}`,
          );
        }
      }
      return {
        title: data.classNames[data.classIndex[row]],
        lines,
        swatch: encoding ? CANVAS_COLORS[encoding.color[row]] : undefined,
      };
    },
    [data, derived, cluster, clusterSummary, encoding, safeEpochIndex],
  );

  const describeCenter = useCallback(
    (group: number, members: number): TooltipContent => {
      if (!data || !clusterSummary) return { title: `Cluster ${group}`, lines: [] };
      const majority = clusterSummary.majorityClass[group];
      return {
        title: `Cluster ${group} · most central member`,
        lines: [
          `${formatCount(members)} visible of ${formatCount(clusterSummary.size[group])} points`,
          majority >= 0
            ? `Largest class: ${data.classNames[majority]} (${percent(clusterSummary.majorityShare[group])})`
            : "No labeled class",
        ],
        swatch: clusterColors ? CLASS_PALETTE[clusterColors[group]] : undefined,
      };
    },
    [data, clusterSummary, clusterColors],
  );

  const visibleCount = data
    ? data.subsets.reduce((sum, name, subsetIndex) => sum + (visibleFlags[subsetIndex] ? data.subsetCounts[subsetIndex] : 0), 0)
    : 0;
  const selectedClass = data && selectedIndex != null ? data.classIndex[selectedIndex] : null;
  const hasLabeledSplit = data?.subsets.includes("labeled") ?? false;
  const trackingEpoch = data?.tracking?.epochs[safeEpochIndex] ?? null;

  const boxSummary = useMemo(() => {
    if (!data || !boxSelection || boxSelection.length === 0) return null;
    const perSubset = new Array<number>(data.subsets.length).fill(0);
    const perClass = new Map<number, number>();
    boxSelection.forEach((row) => {
      perSubset[data.subset[row]] += 1;
      perClass.set(data.classIndex[row], (perClass.get(data.classIndex[row]) ?? 0) + 1);
    });
    const topClasses = [...perClass.entries()].sort((left, right) => right[1] - left[1]).slice(0, 8);
    return { perSubset, topClasses, classCount: perClass.size };
  }, [data, boxSelection]);

  return (
    <main className="mx-auto max-w-[1500px] px-5 pb-16 pt-8">
      {shared ? (
        <ShareNav current="/embeddings/" />
      ) : (
        <Link href="/" className="focus-ring text-xs font-semibold text-[#8895aa] hover:text-white">
          <ArrowLeft size={14} className="mr-1 inline" />
          Back to Trial Atlas
        </Link>
      )}

      <header className={shared ? "" : "mt-5"}>
        <h1 className="text-2xl font-semibold text-white">Embeddings</h1>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#98a4ba]">
          One run&apos;s points in 2D or 3D. Compare the labeled support with the unlabeled pool, color unlabeled points by
          their true class or by whether their nearest labeled sample agrees, follow cluster centers, and step through the
          epochs of a tracked run.
        </p>
      </header>

      {/* What to look at. */}
      <section className="mt-6 grid gap-4 rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4 lg:grid-cols-[auto_minmax(0,1.4fr)_minmax(0,1fr)_auto]">
        <div className="relative z-20 lg:col-span-4">
          <ControlLabel>Find a run or artifact</ControlLabel>
          <div className="mt-1.5 flex gap-2">
            <div className="relative min-w-0 flex-1">
              <Search
                size={14}
                className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-[#6f7c92]"
              />
              <input
                type="text"
                value={viewLookup}
                onChange={(event) => setViewLookup(event.target.value)}
                onFocus={() => setLookupFocused(true)}
                onBlur={() => setLookupFocused(false)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault();
                    void openViewLookup();
                  }
                  if (event.key === "Escape") setLookupFocused(false);
                }}
                placeholder={shared ? "Search runs" : "Search runs, or paste a WSL path to test_embeddings.npz / head_state.pt"}
                aria-label="Find an embedding run or artifact"
                aria-expanded={lookupFocused && viewLookup.trim().length > 0}
                aria-controls="embedding-view-matches"
                className="focus-ring w-full rounded-xl border border-white/[0.1] bg-[#0b1222] py-2 pl-9 pr-9 text-xs text-white placeholder:text-[#5f6b80]"
              />
              {viewLookup && (
                <button
                  type="button"
                  onClick={() => {
                    setViewLookup("");
                    setLookupFocused(false);
                  }}
                  aria-label="Clear embedding search"
                  className="focus-ring absolute right-2.5 top-1/2 -translate-y-1/2 text-[#6f7c92] hover:text-white"
                >
                  <X size={14} />
                </button>
              )}
              {lookupFocused && viewLookup.trim() && (
                <div
                  id="embedding-view-matches"
                  className="absolute left-0 right-0 top-[calc(100%+0.35rem)] max-h-80 overflow-y-auto rounded-xl border border-white/[0.1] bg-[#101827] p-1 shadow-2xl"
                >
                  {viewMatches.length > 0 ? (
                    viewMatches.map((match) => (
                      <button
                        key={match.id}
                        type="button"
                        onMouseDown={(event) => event.preventDefault()}
                        onClick={() => {
                          selectView(match.id);
                          setLookupFocused(false);
                          setError(null);
                        }}
                        className={`focus-ring block w-full rounded-lg px-3 py-2 text-left hover:bg-white/[0.06] ${
                          match.id === viewId ? "bg-[#9b8cff]/10" : ""
                        }`}
                      >
                        <span className="block truncate text-xs font-semibold text-[#e4e9f2]">{match.label}</span>
                        <span className="mt-0.5 block truncate text-[10px] text-[#7f8ba2]">
                          {SOURCE_LABELS[match.source]} · {match.entry?.runDir ?? match.uploaded?.runDir ?? match.variant}
                        </span>
                      </button>
                    ))
                  ) : (
                    <p className="px-3 py-2 text-xs leading-relaxed text-[#7f8ba2]">
                      {looksLikeEmbeddingPath(viewLookup)
                        ? "Press Enter to open the local run projection. A test_embeddings.npz path uses test_tsne.csv beside it."
                        : "No exported view matches this search."}
                    </p>
                  )}
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={() => void openViewLookup()}
              disabled={!viewLookup.trim() || lookupLoading}
              className="focus-ring rounded-xl border border-[#9b8cff]/35 bg-[#9b8cff]/10 px-4 py-2 text-xs font-semibold text-[#d8d2ff] transition hover:bg-[#9b8cff]/20 disabled:cursor-not-allowed disabled:opacity-40"
            >
              {lookupLoading ? "Opening…" : viewMatches.length > 0 || shared ? "Select" : "Open path"}
            </button>
          </div>
          {!shared && (
            <p className="mt-1.5 text-[10px] text-[#6f7c92]">
              Accepts Linux paths and Windows WSL paths such as <code>\\wsl.localhost\Ubuntu-22.04\home\…</code>.
            </p>
          )}
        </div>
        <div>
          <ControlLabel>Source</ControlLabel>
          <div className="mt-1.5">
            {sources.length > 0 ? (
              <Segmented<SourceKey>
                label="Kind of view"
                value={source ?? sources[0]}
                onChange={(next) => {
                  const first = options.find((candidate) => candidate.source === next);
                  if (first) selectView(first.id);
                }}
                options={sources.map((key) => ({
                  value: key,
                  label: `${SOURCE_LABELS[key]} · ${new Set(options.filter((o) => o.source === key).map((o) => o.runLabel)).size}`,
                }))}
              />
            ) : (
              <p className="py-2 text-xs text-[#7f8ba2]">Nothing exported yet</p>
            )}
          </div>
        </div>
        <div>
          <ControlLabel>Run</ControlLabel>
          <select
            value={runLabel ?? ""}
            onChange={(event) => {
              const first = runsOfSource.get(event.target.value)?.[0];
              if (first) selectView(first.id);
            }}
            className={selectClass}
            disabled={runsOfSource.size === 0}
          >
            {[...new Set([...runsOfSource.values()].map((group) => group[0].dataset ?? "Other"))].map((dataset) => (
              <optgroup key={dataset} label={dataset}>
                {[...runsOfSource.entries()]
                  .filter(([, group]) => (group[0].dataset ?? "Other") === dataset)
                  .map(([label, group]) => (
                    <option key={label} value={label}>
                      {label} ({group.length})
                    </option>
                  ))}
              </optgroup>
            ))}
          </select>
        </div>
        <div>
          <ControlLabel>Fold / variant</ControlLabel>
          <select value={viewId ?? ""} onChange={(event) => selectView(event.target.value)} className={selectClass} disabled={variants.length < 2}>
            {variants.map((variant) => (
              <option key={variant.id} value={variant.id}>
                {variant.variant}
              </option>
            ))}
          </select>
        </div>
        <label
          onDragOver={(event) => {
            event.preventDefault();
            setDragActive(true);
          }}
          onDragLeave={() => setDragActive(false)}
          onDrop={(event) => {
            event.preventDefault();
            setDragActive(false);
            void onUpload(event.dataTransfer.files);
          }}
          className={`focus-ring flex cursor-pointer items-center gap-2 self-end rounded-xl border border-dashed px-3 py-2 text-xs font-semibold transition ${
            dragActive ? "border-[#9b8cff] text-white" : "border-white/[0.12] text-[#8895aa] hover:text-white"
          }`}
        >
          <Upload size={14} />
          Open CSV / JSON
          <input
            type="file"
            accept=".csv,.json"
            multiple
            className="sr-only"
            onChange={(event) => {
              if (event.target.files) void onUpload(event.target.files);
              event.target.value = "";
            }}
          />
        </label>
      </section>

      {option?.live && (
        <section
          aria-live="polite"
          className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 rounded-2xl border border-[#4fd5e7]/20 bg-[#4fd5e7]/[0.04] px-4 py-2.5 text-xs text-[#c8d2e4]"
        >
          <span className="flex items-center gap-2 font-semibold text-white">
            <Radio size={14} className={option.live.running ? "animate-pulse text-[#7fe3f0]" : "text-[#65748e]"} />
            {option.live.running ? "Live" : "Live · not running"}
          </span>
          <span>{describeStage(liveView.manifest?.status ?? option.live.status) ?? "no status.json (run tracked before live status existed)"}</span>
          <span className="text-[#98a4ba]">
            {liveView.progress.loaded} of {liveView.progress.available} epochs loaded
            {liveView.progress.fetching ? " · loading…" : ""}
          </span>
          <span className="text-[#98a4ba]">view updated {ago(liveView.lastUpdate, now)}</span>
          {mode === "3d" && has3d && !layout3d && trackingEpoch != null && (
            <span className="text-[#f4c86f]">3D projection of epoch {trackingEpoch} not written yet: showing 2D</span>
          )}
          {liveView.error && <span className="text-[#f4b4b4]">{liveView.error}</span>}
          <span className="ml-auto flex items-center gap-3">
            <label className="flex cursor-pointer items-center gap-2 font-semibold" title="Switch to a run as soon as it starts writing epoch tracking">
              <input type="checkbox" checked={followNewest} onChange={(event) => setFollowNewest(event.target.checked)} />
              Follow newest run
            </label>
            <button
              type="button"
              onClick={liveRuns.refresh}
              className="focus-ring flex items-center gap-1.5 rounded-lg border border-white/[0.1] px-2.5 py-1 font-semibold text-[#c8d2e4] hover:text-white"
            >
              <RefreshCw size={12} />
              Rescan runs
            </button>
          </span>
        </section>
      )}

      {/* How to show it. */}
      {data && (
        <section className="mt-3 flex flex-wrap items-end gap-4">
          <div>
            <ControlLabel>Projection</ControlLabel>
            <div className="mt-1.5">
              <Segmented<ViewMode>
                label="Projection dimensions"
                value={mode}
                onChange={setMode}
                options={[
                  { value: "2d", label: "2D" },
                  {
                    value: "3d",
                    label: "3D",
                    disabled: !has3d,
                    title: has3d ? undefined : "No 3D layout: re-run with --projection_3d or export from a saved head",
                  },
                ]}
              />
            </div>
          </div>
          <div className="min-w-[200px]">
            <ControlLabel>Color by</ControlLabel>
            <select
              value={effectiveColorMode}
              onChange={(event) => setColorMode(event.target.value as ColorMode)}
              className={selectClass}
            >
              {availableModes.map((key) => (
                <option key={key} value={key}>
                  {COLOR_MODE_LABELS[key]}
                </option>
              ))}
            </select>
          </div>
          <div>
            <ControlLabel>Split</ControlLabel>
            <div className="mt-1.5">
              <Segmented<SplitEmphasis>
                label="Emphasize labeled or unlabeled points"
                value={splitEmphasis}
                onChange={setSplitEmphasis}
                options={[
                  { value: "none", label: "Both" },
                  { value: "labeled", label: "Labeled", disabled: !hasLabeledSplit },
                  { value: "unlabeled", label: "Unlabeled", disabled: !hasLabeledSplit },
                ]}
              />
            </div>
          </div>
          <div>
            <ControlLabel>Point size</ControlLabel>
            <input
              type="range"
              min={0.8}
              max={6}
              step={0.2}
              value={pointSize}
              onChange={(event) => setPointSize(Number(event.target.value))}
              aria-label="Point size"
              className="mt-2.5 w-24 accent-[#9b8cff]"
            />
          </div>
          {cluster && (
            <label className="flex cursor-pointer items-center gap-2 pb-2 text-xs font-semibold text-[#c8d2e4]">
              <input
                type="checkbox"
                checked={showCenters}
                onChange={() => setShowCenters((current) => !current)}
                className="accent-[#9b8cff]"
              />
              Cluster centers
            </label>
          )}
          {data && layout2d && (
            <div className="ml-auto" title="Aktueller Ausschnitt; PNG hell in 4-facher Auflösung, CSV mit den sichtbaren Punkten">
              <ExportMenu
                name={`embedding-${data.label}${trackingEpoch != null ? `-epoche-${trackingEpoch}` : ""}-${colorMode}`}
                png={async () => {
                  const canvas = canvasRef.current?.renderImage({ scale: 4, print: true });
                  if (!canvas) throw new Error("Die Ansicht ist noch nicht gezeichnet.");
                  const blob = await new Promise<Blob>((resolve, reject) =>
                    canvas.toBlob((result) => (result ? resolve(result) : reject(new Error("PNG konnte nicht erzeugt werden."))), "image/png"),
                  );
                  downloadBlob(blob, `${slugify(`embedding-${data.label}${trackingEpoch != null ? `-epoche-${trackingEpoch}` : ""}-${colorMode}`)}.png`);
                }}
                rows={() => {
                  const out: ExportRow[] = [];
                  const use3d = mode === "3d" && layout3d != null;
                  for (let index = 0; index < data.count; index += 1) {
                    if (!visibleFlags[data.subset[index]]) continue;
                    out.push({
                      row: index,
                      sample: data.sample[index],
                      class: data.classNames[data.classIndex[index]] ?? data.classIndex[index],
                      subset: data.subsets[data.subset[index]],
                      x: use3d ? layout3d![index * 3] : layout2d[index * 2],
                      y: use3d ? layout3d![index * 3 + 1] : layout2d[index * 2 + 1],
                      ...(use3d ? { z: layout3d![index * 3 + 2] } : {}),
                      epoch: trackingEpoch ?? data.selectedEpoch,
                    });
                  }
                  return out;
                }}
              />
            </div>
          )}
          <button
            type="button"
            onClick={() => canvasRef.current?.resetView()}
            className="focus-ring flex items-center gap-1.5 rounded-xl border border-white/[0.08] px-3 py-2 text-xs font-semibold text-[#8895aa] hover:text-white"
          >
            <RotateCcw size={13} />
            Reset view
          </button>
        </section>
      )}

      {data?.tracking && trackingEpoch != null && (
        <section className="mt-3 flex items-center gap-3 rounded-2xl border border-white/[0.07] bg-white/[0.02] px-4 py-2.5">
          <button
            type="button"
            onClick={() => {
              if (!playing && safeEpochIndex >= epochCount - 1) setEpochIndex(0);
              setPlaying((current) => !current);
            }}
            className="focus-ring flex h-8 w-8 items-center justify-center rounded-full border border-white/[0.1] text-white"
            aria-label={playing ? "Pause epochs" : "Play epochs"}
          >
            {playing ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <span className="w-24 text-sm font-semibold tabular-nums text-white">
            Epoch {trackingEpoch}
            {trackingEpoch < 0 ? " (untrained)" : ""}
          </span>
          <input
            type="range"
            min={0}
            max={epochCount - 1}
            step={1}
            value={safeEpochIndex}
            onChange={(event) => {
              setPlaying(false);
              setEpochIndex(Number(event.target.value));
            }}
            aria-label="Tracked epoch"
            className="flex-1 accent-[#9b8cff]"
          />
          <label className="flex items-center gap-2 text-[11px] text-[#98a4ba]">
            Compare with
            <select
              value={safeReferenceIndex}
              onChange={(event) => setReferenceIndex(Number(event.target.value))}
              className="focus-ring rounded-lg border border-white/[0.08] bg-[#10182a] px-2 py-1 text-xs text-white"
            >
              {data.tracking.epochs.map((epoch, index) => (
                <option key={epoch} value={index}>
                  epoch {epoch}
                </option>
              ))}
            </select>
          </label>
          <label
            className="flex cursor-pointer items-center gap-1.5 text-[11px] text-[#98a4ba]"
            title="Lines from where points sat at the reference epoch to where they are now: the region, the selected point's class, or the pinned classes"
          >
            <input type="checkbox" checked={showTrails} onChange={() => setShowTrails((current) => !current)} className="accent-[#9b8cff]" />
            Movement
          </label>
        </section>
      )}

      {data?.tracking && propagationFrame && stepTotal > 0 && (
        <section className="mt-2 flex flex-wrap items-center gap-3 rounded-2xl border border-white/[0.07] bg-white/[0.02] px-4 py-2.5">
          <button
            type="button"
            onClick={() => {
              if (!stepPlaying && safeStep >= stepTotal - 1) setPropagationStep(0);
              setStepPlaying((current) => !current);
            }}
            className="focus-ring flex h-8 w-8 items-center justify-center rounded-full border border-white/[0.1] text-white"
            aria-label={stepPlaying ? "Pause the propagation" : "Play the propagation"}
          >
            {stepPlaying ? <Pause size={14} /> : <Play size={14} />}
          </button>
          <span className="w-44 text-sm font-semibold tabular-nums text-white">
            Propagation step {safeStep}
            {safeStep === 0 ? " (seeds)" : safeStep === stepTotal - 1 ? " (settled)" : ""}
          </span>
          <input
            type="range"
            min={0}
            max={stepTotal - 1}
            step={1}
            value={safeStep}
            onChange={(event) => {
              setStepPlaying(false);
              setPropagationStep(Number(event.target.value));
            }}
            aria-label="Propagation step"
            className="min-w-[160px] flex-1 accent-[#9b8cff]"
          />
          <span className="text-[11px] tabular-nums text-[#98a4ba]">
            {percent(propagationFrame.stepReached[safeStep] ?? null)} of unlabeled reached
            {propagationFrame.stepAccuracy[safeStep] != null
              ? ` · ${percent(propagationFrame.stepAccuracy[safeStep])} of them right`
              : ""}
          </span>
          <button
            type="button"
            onClick={() => {
              setStepPlaying(false);
              setPropagationStep(-1);
            }}
            className="focus-ring rounded-lg border border-white/[0.08] px-2 py-1 text-[11px] text-[#98a4ba] hover:text-white"
            title="Follow the settled diffusion, the one class_evidence.npz records"
          >
            Settled
          </button>
        </section>
      )}

      {data?.tracking && filmstripFrames && (
        <section className="mt-2">
          <EpochFilmstrip
            data={data}
            frames={filmstripFrames}
            epochIndex={safeEpochIndex}
            referenceIndex={safeReferenceIndex}
            selectedIndex={selectedIndex}
            visibleFlags={visibleFlags}
            onPick={(index) => {
              setPlaying(false);
              setEpochIndex(index);
            }}
          />
        </section>
      )}

      {error && (
        <p role="alert" className="mt-3 rounded-xl border border-[#d03b3b]/40 bg-[#d03b3b]/10 px-3 py-2 text-xs text-[#f4b4b4]">
          {error}
        </p>
      )}

      {options.length === 0 && (
        <div className="mt-6">
          <EmptyExportHint />
        </div>
      )}

      {option && (
        <div className="mt-4 grid gap-4 lg:grid-cols-[minmax(0,1fr)_340px] xl:grid-cols-[minmax(0,1fr)_300px_340px]">
          <section className="relative h-[72vh] min-h-[520px] rounded-2xl border border-white/[0.07] bg-[#0b1222]">
            {data && encoding && layout2d ? (
              <EmbeddingCanvas
                ref={canvasRef}
                data={data}
                mode={mode}
                layout2d={layout2d}
                layout3d={layout3d}
                trails={trails}
                visibleSubsets={visibleFlags}
                pointColors={encoding.color}
                pointTiers={encoding.tier}
                outlineLabeled={outlineLabeled}
                pointSize={pointSize}
                selectedIndex={selectedIndex}
                edges={edges}
                centerGroups={showCenters ? cluster : null}
                centerCount={showCenters ? clusterCount : 0}
                describePoint={describePoint}
                describeCenter={describeCenter}
                onSelect={(row) => {
                  setSelectedIndex(row);
                  setBoxSelection(null);
                }}
                onBoxSelect={(rows) => {
                  setBoxSelection(rows.length > 0 ? rows : null);
                  setSelectedIndex(null);
                }}
              />
            ) : (
              <p className="p-6 text-sm text-[#7f8ba2]">{option?.live
                  ? liveView.error ?? (liveView.progress.available === 0 ? "Waiting for the first tracked epoch…" : "Loading points…")
                  : loading
                    ? "Loading points…"
                    : "Select a view."}</p>
            )}
            {data && (
              <p className="pointer-events-none absolute bottom-3 right-4 text-[11px] text-[#7f8ba2]">
                {formatCount(visibleCount)} of {formatCount(data.count)} points ·{" "}
                {mode === "3d" ? "drag rotate · right-drag pan" : "drag pan"} · scroll zoom · shift-drag select · click a point
              </p>
            )}
          </section>

          {data && galleryAsColumn && (
            <section className="h-[72vh] min-h-[520px] overflow-y-auto rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
              <ControlLabel>Images</ControlLabel>
              <div className="mt-2">
                <NeighborGallery
                  data={data}
                  selectedIndex={selectedIndex}
                  gallery={gallery}
                  onPick={selectPoint}
                  neighborControl={
                    boxSelection || neighborMax === 0 || showMinedPairs || showPropagation
                      ? null
                      : { value: shownNeighbors, max: neighborMax, allowsAll: neighborAllowsAll, onChange: setNeighborCount }
                  }
                  header={lineSourceControl}
                />
              </div>
            </section>
          )}

          <aside className="space-y-4">
            {data && !galleryAsColumn && (
              <div className="max-h-[72vh] overflow-y-auto rounded-2xl border border-white/[0.07] bg-white/[0.02] p-4">
                <ControlLabel>Images</ControlLabel>
                <div className="mt-2">
                  <NeighborGallery
                  data={data}
                  selectedIndex={selectedIndex}
                  gallery={gallery}
                  onPick={selectPoint}
                  neighborControl={
                    boxSelection || neighborMax === 0 || showMinedPairs || showPropagation
                      ? null
                      : { value: shownNeighbors, max: neighborMax, allowsAll: neighborAllowsAll, onChange: setNeighborCount }
                  }
                  header={lineSourceControl}
                />
                </div>
              </div>
            )}
            {data && encoding && (
              <>
                <Panel title="Subsets & color">
                  <ul className="mt-2 space-y-1">
                    {data.subsets.map((name, subsetIndex) => {
                      const style = subsetStyle(name);
                      const checked = visibleFlags[subsetIndex];
                      return (
                        <li key={name}>
                          <label
                            className={`flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-xs transition hover:bg-white/[0.04] ${
                              checked ? "text-[#e3e8f2]" : "text-[#65748e]"
                            }`}
                            title={style.description}
                          >
                            <input
                              type="checkbox"
                              checked={checked}
                              onChange={() => setVisibleSubsets((current) => ({ ...current, [name]: !checked }))}
                              className="accent-[#9b8cff]"
                            />
                            <ShapeKey shape={style.shape} color={checked ? legendColor(name, effectiveColorMode) : "#3a4458"} />
                            <span className="font-semibold">{style.label}</span>
                            <span className="ml-auto tabular-nums text-[#7f8ba2]">{formatCount(data.subsetCounts[subsetIndex])}</span>
                          </label>
                        </li>
                      );
                    })}
                  </ul>
                  {hasLabeledSplit ? (
                    <label className="mt-2 flex cursor-pointer items-center gap-2 px-2 text-[11px] text-[#98a4ba]">
                      <input
                        type="checkbox"
                        checked={outlineLabeled}
                        onChange={() => setOutlineLabeled((current) => !current)}
                        className="accent-[#9b8cff]"
                      />
                      Outline labeled points
                    </label>
                  ) : (
                    <p className="mt-2 flex gap-1.5 text-[11px] leading-relaxed text-[#7f8ba2]">
                      <Info size={12} className="mt-0.5 shrink-0" />
                      No labeled/unlabeled split in this view. Saved heads and epoch tracking carry one.
                    </p>
                  )}

                  <div className="mt-3 border-t border-white/[0.06] pt-3 text-[11px] leading-relaxed text-[#7f8ba2]">
                    {effectiveColorMode === "subset" && <p>Color and shape both mark the subset.</p>}
                    {(effectiveColorMode === "class" || effectiveColorMode === "cluster") && (
                      <>
                        <div className="flex items-center gap-1" aria-hidden>
                          {CLASS_PALETTE.map((color) => (
                            <span key={color} className="h-[3px] flex-1 rounded-full" style={{ background: color }} />
                          ))}
                        </div>
                        <p className="mt-2">
                          {effectiveColorMode === "class"
                            ? `Color marks the class — for unlabeled points, their true label, which training never saw. ${formatCount(data.classNames.length)} classes share ${CLASS_PALETTE.length} colors, assigned so touching classes never share one.`
                            : `Color marks the cluster (${formatCount(clusterCount)} clusters, touching clusters never share a color). ${
                              data.graph?.clusterSource ??
                              (data.tracking?.clusters
                                ? `The run's own SLADE refresh${data.tracking.clusterRefreshEpochs[safeEpochIndex] != null ? ` at epoch ${data.tracking.clusterRefreshEpochs[safeEpochIndex]}` : ""}.`
                                : "")
                            }`}{" "}
                          Shape marks the subset.
                        </p>
                      </>
                    )}
                    {effectiveColorMode === "pinned" && (
                      <p>Up to {MAX_PINNED_CLASSES} pinned classes get a color; every other point is gray.</p>
                    )}
                    {effectiveColorMode === "agreement" && (
                      <>
                        <p className="flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CATEGORICAL[0] }} /> Same true class as{" "}
                          {derived?.agreementSource}
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CATEGORICAL[1] }} /> A different class
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: NEUTRAL }} /> Not applicable (labeled,
                          or a class with no labeled sample)
                        </p>
                        {agreementSummary && (
                          <p className="mt-2 text-[#c8d2e4]">
                            {percent(agreementSummary.same / agreementSummary.total, 1)} of{" "}
                            {formatCount(agreementSummary.total)} unlabeled points agree.
                          </p>
                        )}
                      </>
                    )}
                    {(effectiveColorMode === "purityChange" || effectiveColorMode === "agreementChange") && (
                      <>
                        <p className="flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CLASS_PALETTE[CHANGE_BETTER_INDEX] }} />
                          {effectiveColorMode === "purityChange"
                            ? `Purer neighborhood than at epoch ${data.tracking?.epochs[safeReferenceIndex]} (+${PURITY_CHANGE_THRESHOLD * 100}% or more)`
                            : `Strongest neighbor became its own class since epoch ${data.tracking?.epochs[safeReferenceIndex]}`}
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CLASS_PALETTE[CHANGE_WORSE_INDEX] }} />
                          {effectiveColorMode === "purityChange" ? "Less pure" : "Strongest neighbor stopped being its own class"}
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CHANGE_NONE }} /> Unchanged
                        </p>
                        {changeSummary && (
                          <p className="mt-2 text-[#c8d2e4]">
                            Unlabeled: {formatCount(changeSummary.better)} better · {formatCount(changeSummary.worse)} worse ·{" "}
                            {formatCount(changeSummary.same)} unchanged
                          </p>
                        )}
                      </>
                    )}
                    {effectiveColorMode === "outlier" && (
                      <>
                        <p className="flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CATEGORICAL[1] }} /> The{" "}
                          {percent(OUTLIER_SHARE)} of points farthest from their own nearest neighbors
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: NEUTRAL }} /> Everything else
                        </p>
                        <p className="mt-2">
                          kNN distance: 1 − the mean cosine to its {data.tracking?.neighbors?.[safeEpochIndex]?.k ?? "k"} nearest
                          neighbors in the embedding at this epoch (not on screen). Outliers are picked per epoch.
                        </p>
                      </>
                    )}
                    {effectiveColorMode === "centerSimilarity" && (
                      <>
                        <div className="flex items-center gap-1" aria-hidden>
                          {RAMP.map((color) => (
                            <span key={color} className="h-[6px] flex-1 rounded-sm" style={{ background: color }} />
                          ))}
                        </div>
                        <div className="mt-1 flex justify-between tabular-nums">
                          <span>≤ 0</span>
                          <span>1</span>
                        </div>
                        <p className="mt-1">
                          Cosine between a point&apos;s teacher embedding and its cluster&apos;s mean, at the refresh this epoch trained
                          on. SLADE never uses the centers; this shows how central each pseudo label&apos;s members were. Gray: no
                          cluster (labeled).
                        </p>
                      </>
                    )}
                    {(effectiveColorMode === "lpPrediction" ||
                      effectiveColorMode === "hofferPrediction" ||
                      effectiveColorMode === "prototypePrediction") && (
                      <>
                        <p className="flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CATEGORICAL[0] }} /> Its most likely class is
                          its true class
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CATEGORICAL[1] }} />
                          {effectiveColorMode === "lpPrediction"
                            ? "Another class, or no labeled point reaches it"
                            : "Another class"}
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: NEUTRAL }} /> Labeled (not scored)
                        </p>
                        <p className="mt-2">
                          {effectiveColorMode === "lpPrediction"
                            ? "Iscen et al.'s label spreading from the labeled points over this epoch's embedding (kNN graph, α-diffusion)."
                            : effectiveColorMode === "prototypePrediction"
                              ? `The network's own prediction: a softmax over cosine similarity to each labeled class's mean embedding (temperature ${
                                  data.tracking?.evidence?.[safeEpochIndex]?.prototypeTemperature ?? 0.05
                                }).`
                              : `Hoffer & Ailon's softmax over distances to one labeled reference per class, averaged over random reference draws${
                                data.tracking?.evidence?.[safeEpochIndex]?.hofferDistanceScale != null
                                  ? ` (distance scale ${data.tracking.evidence[safeEpochIndex]!.hofferDistanceScale})`
                                  : ""
                              }.`}{" "}
                          Computed for every method; hidden labels only score it.
                        </p>
                        {evidenceSummary?.kind === "prediction" && (
                          <p className="mt-2 text-[#c8d2e4]">
                            {percent(evidenceSummary.right / evidenceSummary.total, 1)} of {formatCount(evidenceSummary.total)}{" "}
                            unlabeled points right.
                          </p>
                        )}
                      </>
                    )}
                    {effectiveColorMode === "propagationClass" && propagationFrame && (
                      <>
                        <p>
                          The class each point holds after {safeStep} step{safeStep === 1 ? "" : "s"} of the diffusion, in the
                          class colors. Dimmed points hold nothing yet: no labeled point is that close.
                        </p>
                        <p className="mt-2">
                          Play the step slider to watch the labels leave the seeds and spread over the graph.
                        </p>
                      </>
                    )}
                    {effectiveColorMode === "propagationCorrect" && (
                      <>
                        <p className="flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CATEGORICAL[0] }} /> Holds its true class
                          at this step
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CATEGORICAL[1] }} /> Holds another class
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: NEUTRAL }} /> Nothing yet, or labeled
                        </p>
                        <p className="mt-2">
                          Where the front is already wrong at a step, more steps rarely fix it: the mass arrives over the same
                          edges.
                        </p>
                      </>
                    )}
                    {effectiveColorMode === "propagationHop" && (
                      <>
                        <p className="flex items-center gap-2">
                          <span className="flex">
                            {RAMP.map((color) => (
                              <span key={color} className="h-[10px] w-3" style={{ background: color }} />
                            ))}
                          </span>
                          <span className="tabular-nums">0 → {RAMP.length - 1}+ edges</span>
                        </p>
                        <p className="mt-2">
                          Edges between a point and the labeled node its mass came from: 0 is a labeled point itself. A graph
                          with a large k puts almost everything one edge away, which is itself the finding.
                        </p>
                      </>
                    )}
                    {(effectiveColorMode === "lpWeight" || effectiveColorMode === "hofferCertainty") && (
                      <>
                        <div className="flex items-center gap-1" aria-hidden>
                          {RAMP.map((color) => (
                            <span key={color} className="h-[6px] flex-1 rounded-sm" style={{ background: color }} />
                          ))}
                        </div>
                        <div className="mt-1 flex justify-between tabular-nums">
                          <span>0</span>
                          <span>
                            {effectiveColorMode === "lpWeight" ? "1" : (data.tracking?.hofferCertaintyMax ?? 0).toFixed(3)}
                          </span>
                        </div>
                        <p className="mt-1">
                          {effectiveColorMode === "lpWeight"
                            ? "Iscen's weight ω: 1 − normalized entropy of the propagated distribution, over this epoch's most certain point. Labeled points are 1. An Iscen run weights each pseudo label's loss by it."
                            : "1 − normalized entropy of Hoffer's reference softmax: 0 is uniform over the classes, 1 one-hot. The entropy term pushes it up. The ramp tops out at the largest unlabeled value of any epoch, so frames compare."}
                        </p>
                      </>
                    )}
                    {effectiveColorMode === "pseudoLabel" && (
                      <>
                        <p className="flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CATEGORICAL[0] }} /> Accepted and right
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CATEGORICAL[1] }} /> Accepted and wrong
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: CHANGE_NONE }} /> Rejected by the confidence
                          filter
                        </p>
                        <p className="mt-1 flex items-center gap-2">
                          <span className="h-[2px] w-3 rounded-full" style={{ background: NEUTRAL }} /> Not a candidate (labeled)
                        </p>
                        {data.tracking?.pseudoLabels?.[safeEpochIndex] && (
                          <p className="mt-2">
                            Generation {data.tracking.pseudoLabels[safeEpochIndex]!.generation}, made at epoch{" "}
                            {data.tracking.pseudoLabels[safeEpochIndex]!.generationEpoch}.
                          </p>
                        )}
                        {evidenceSummary?.kind === "pseudo" && (
                          <p className="mt-2 text-[#c8d2e4]">
                            {formatCount(evidenceSummary.right)} right · {formatCount(evidenceSummary.wrong)} wrong ·{" "}
                            {formatCount(evidenceSummary.rejected)} rejected
                          </p>
                        )}
                      </>
                    )}
                    {effectiveColorMode === "purity" && (
                      <>
                        <div className="flex items-center gap-1" aria-hidden>
                          {RAMP.map((color) => (
                            <span key={color} className="h-[6px] flex-1 rounded-sm" style={{ background: color }} />
                          ))}
                        </div>
                        <div className="mt-1 flex justify-between tabular-nums">
                          <span>0%</span>
                          <span>100%</span>
                        </div>
                        <p className="mt-1">
                          Share of a point&apos;s {data.graph ? `${data.graph.k} nearest train points` : "graph neighbors at this epoch"} with its
                          own class. Gray: no neighbors recorded.
                        </p>
                      </>
                    )}
                  </div>
                </Panel>

                {data.graph && data.graph.metrics.length > 0 && (
                  <Panel title="Graph metrics">
                    <table className="mt-2 w-full text-[11px]">
                      <thead>
                        <tr className="text-left text-[#7f8ba2]">
                          <th className="py-1 font-semibold">Metric</th>
                          <th className="py-1 pl-2 text-right font-semibold">Trained</th>
                          <th className="py-1 pl-2 text-right font-semibold">Backbone</th>
                          <th className="py-1 pl-2 text-right font-semibold">Δ</th>
                        </tr>
                      </thead>
                      <tbody>
                        {data.graph.metrics.map((metric) => (
                          <tr key={metric.name} className="border-t border-white/[0.05] text-[#c8d2e4]" title={metric.description}>
                            <td className="py-1 pr-1">{metric.label}</td>
                            <td className="py-1 text-right tabular-nums">{formatNumber(metric.trained)}</td>
                            <td className="py-1 text-right tabular-nums">{formatNumber(metric.pretrained)}</td>
                            <td className="py-1 text-right tabular-nums">
                              {formatDelta(
                                metric.trained != null && metric.pretrained != null ? metric.trained - metric.pretrained : null,
                                metric.higherIsBetter,
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p className="mt-2 text-[11px] leading-relaxed text-[#7f8ba2]">
                      Over every train sample; Backbone is DINOv2 without the trained head. Hover a row for its definition.
                    </p>
                  </Panel>
                )}

                {timeline.length > 0 && data.tracking && (
                  <Panel title="Across all tracked epochs">
                    <p className="mt-2 text-[11px] leading-relaxed text-[#7f8ba2]">
                      Each line is one metric over every tracked epoch: white marks epoch {trackingEpoch}, gray the reference epoch{" "}
                      {data.tracking.epochs[safeReferenceIndex]}. Click a line to jump.
                    </p>
                    <table className="mt-2 w-full text-[11px]">
                      <tbody>
                        {timeline.map((series) => {
                          const now = series.values[safeEpochIndex];
                          const before = series.values[safeReferenceIndex];
                          return (
                            <tr key={series.name} className="border-t border-white/[0.05] text-[#c8d2e4]">
                              <td className="py-1.5 pr-2 align-middle leading-tight">{series.label}</td>
                              <td className="py-1.5 align-middle">
                                <Sparkline
                                  series={series}
                                  epochIndex={safeEpochIndex}
                                  referenceIndex={safeReferenceIndex}
                                  onPick={(index) => {
                                    setPlaying(false);
                                    setEpochIndex(index);
                                  }}
                                />
                              </td>
                              <td className="py-1.5 pl-2 text-right align-middle tabular-nums">
                                {formatNumber(now)}
                                <span className="block text-[10px] text-[#7f8ba2]">
                                  {formatDelta(now != null && before != null ? now - before : null, series.higherIsBetter)}
                                </span>
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </Panel>
                )}

                {data.tracking && diagnosticPredictors.length > 0 && (
                  <Panel title="Mechanism diagnostics">
                    <p className="mt-2 text-[11px] leading-relaxed text-[#7f8ba2]">
                      Unlabeled points only. Their hidden labels score these after the fact; nothing here reaches training. White marks
                      epoch {trackingEpoch}, gray the comparison epoch {data.tracking.epochs[safeReferenceIndex]}.
                    </p>
                    <ChartBlock title="Accuracy on unlabeled over training">
                      <LineChart
                        epochs={data.tracking.epochs}
                        epochIndex={safeEpochIndex}
                        referenceIndex={safeReferenceIndex}
                        series={diagnosticPredictors.map((predictor) => ({
                          label: PREDICTOR_LABELS[predictor],
                          ...PREDICTOR_COLORS[predictor],
                          values: predictorAccuracy[predictor] ?? [],
                        }))}
                        format={(value) => percent(value, 1)}
                        onPick={(index) => {
                          setPlaying(false);
                          setEpochIndex(index);
                        }}
                      />
                    </ChartBlock>
                    {conditionalEntropy.length > 0 && (
                      <ChartBlock
                        title="Conditional entropy H(y|x), nats"
                        note="Entropy of the network's prototype softmax: labeled and unlabeled points against the labeled classes, D_test against its own classes. Pseudo-labeling acting as entropy regularization shows as the unlabeled line falling."
                      >
                        <LineChart
                          epochs={data.tracking.epochs}
                          epochIndex={safeEpochIndex}
                          referenceIndex={safeReferenceIndex}
                          series={conditionalEntropy}
                          format={(value) => value.toFixed(3)}
                          onPick={(index) => {
                            setPlaying(false);
                            setEpochIndex(index);
                          }}
                        />
                      </ChartBlock>
                    )}
                    <label className="mt-4 block">
                      <ControlLabel>Predictor</ControlLabel>
                      <select
                        value={activePredictor}
                        onChange={(event) => setDiagnosticPredictor(event.target.value as Predictor)}
                        className={`${selectClass} mt-1 w-full`}
                      >
                        {diagnosticPredictors.map((predictor) => (
                          <option key={predictor} value={predictor}>
                            {PREDICTOR_LABELS[predictor]}
                          </option>
                        ))}
                      </select>
                    </label>
                    <ChartBlock
                      title="Accuracy of the most confident"
                      note={`Confidence is ${
                        activePredictor === "lp"
                          ? "Iscen's weight ω"
                          : activePredictor === "pseudo"
                            ? "the run's own pseudo-label confidence"
                            : "1 − normalized entropy"
                      }. Lines that fan out mean confidence ranks correctness.`}
                    >
                      <LineChart
                        epochs={data.tracking.epochs}
                        epochIndex={safeEpochIndex}
                        referenceIndex={safeReferenceIndex}
                        series={confidenceShareSeries}
                        format={(value) => percent(value, 1)}
                        onPick={(index) => {
                          setPlaying(false);
                          setEpochIndex(index);
                        }}
                      />
                    </ChartBlock>
                    {confidenceHistograms && (
                      <ChartBlock
                        title="Confidence of right vs wrong predictions"
                        note="Each bar is a share of its own group, so the two shapes compare even when wrong predictions are few."
                      >
                        <div className="grid grid-cols-2 gap-3">
                          {confidenceHistograms.map((histogram) => (
                            <ConfidenceHistogram key={histogram.label} {...histogram} />
                          ))}
                        </div>
                      </ChartBlock>
                    )}
                    {confusions && (
                      <ChartBlock
                        title="Confident mistakes"
                        note={`Wrong predictions in the most confident ${percent(CONFUSION_TOP_SHARE)} at epoch ${trackingEpoch}, grouped by true → predicted class. Click one to see its images.`}
                      >
                        {confusions.length === 0 ? (
                          <p className="text-[11px] text-[#7f8ba2]">None: every confident prediction is right.</p>
                        ) : (
                          <ul className="space-y-1">
                            {confusions.slice(0, MAX_CONFUSIONS).map((confusion) => (
                              <li key={`${confusion.trueClass}:${confusion.predictedClass}`}>
                                <button
                                  type="button"
                                  onClick={() => {
                                    setSelectedIndex(null);
                                    setBoxSelection(Int32Array.from(confusion.rows));
                                  }}
                                  className="focus-ring flex w-full items-baseline justify-between gap-2 rounded px-1 py-0.5 text-left text-[11px] text-[#c8d2e4] hover:bg-white/[0.04]"
                                >
                                  <span className="min-w-0 truncate">
                                    {data.classNames[confusion.trueClass]}{" "}
                                    <span className="text-[#7f8ba2]">→</span> {data.classNames[confusion.predictedClass]}
                                  </span>
                                  <span className="shrink-0 tabular-nums">{confusion.rows.length}</span>
                                </button>
                              </li>
                            ))}
                          </ul>
                        )}
                      </ChartBlock>
                    )}
                  </Panel>
                )}

                {data.tracking?.propagation && propagationFrame && (
                  <Panel title="Label propagation, step by step">
                    <p className="mt-2 text-[11px] leading-relaxed text-[#7f8ba2]">
                      The diffusion as a series: F<sub>0</sub> holds only the labeled seeds, and each step lets the mass move one
                      more edge (F<sub>t</sub> = α·S·F<sub>t−1</sub> + Y). Step {stepTotal - 1} is the answer the solve gives.
                      {propagationFrame.graphSource === "label_propagation"
                        ? " Over label propagation's own graph, so the last step is what the label-propagation color modes show."
                        : " Over the run's own tracked graph, backfilled from its saved edges."}{" "}
                      {propagationFrame.alpha != null ? ` α = ${propagationFrame.alpha.toFixed(3)}` : ""}
                      {propagationFrame.nNeighbors > 0 ? `, k = ${propagationFrame.nNeighbors}` : ""}
                      {propagationFrame.gamma != null && Number.isFinite(propagationFrame.gamma)
                        ? `, γ = ${propagationFrame.gamma}`
                        : ""}
                      .
                    </p>
                    <ChartBlock
                      title={`Along the diffusion at epoch ${trackingEpoch}`}
                      note="Unlabeled points only. Click to move the step slider. Accuracy is over the points reached so far, so the first step is scored on the few nodes that touch a labeled one."
                    >
                      <LineChart
                        epochs={stepNumbers}
                        epochIndex={safeStep}
                        referenceIndex={safeStep}
                        unit="step"
                        series={[
                          { label: "Reached", color: CATEGORICAL[1], values: propagationFrame.stepReached },
                          { label: "Right among them", color: CATEGORICAL[0], values: propagationFrame.stepAccuracy },
                        ]}
                        format={(value) => percent(value, 1)}
                        onPick={(index) => {
                          setStepPlaying(false);
                          setPropagationStep(index);
                        }}
                      />
                    </ChartBlock>
                    <dl className="mt-3 space-y-1 text-[11px]">
                      {(
                        [
                          ["lp_reached_share", "Unlabeled reached", (value: number) => percent(value, 1)],
                          ["lp_mean_hops", "Edges travelled, mean", (value: number) => value.toFixed(2)],
                          ["lp_tree_edge_purity", "Arrival edge stays in class", (value: number) => percent(value, 1)],
                          ["lp_source_accuracy", "Mass came from the true class", (value: number) => percent(value, 1)],
                          ["lp_influence_share", "Carried by the strongest neighbor", (value: number) => percent(value, 1)],
                          ["lp_influence_purity", "That neighbor is the true class", (value: number) => percent(value, 1)],
                        ] as const
                      ).map(([key, label, format]) => {
                        const value = propagationFrame.summary[key];
                        if (value == null) return null;
                        return (
                          <div key={key} className="flex items-baseline justify-between gap-2">
                            <dt className="min-w-0 truncate text-[#98a4ba]">{label}</dt>
                            <dd className="shrink-0 tabular-nums text-[#e3e8f2]">{format(value)}</dd>
                          </div>
                        );
                      })}
                    </dl>
                    {selectedPropagation ? (
                      <div className="mt-3 border-t border-white/[0.07] pt-3">
                        <p className="text-[11px] font-semibold text-[#c8d2e4]">The selected point</p>
                        {selectedPropagation.arrivalStep < 0 ? (
                          <p className="mt-1 text-[11px] leading-relaxed text-[#7f8ba2]">
                            No labeled point reaches it on this graph: it holds no class mass at any step.
                          </p>
                        ) : (
                          <>
                            <p className="mt-1 text-[11px] leading-relaxed text-[#98a4ba]">
                              {selectedPropagation.hop === 0 ? (
                                <>It is a labeled seed: the propagation starts here.</>
                              ) : (
                                <>
                                  Reached at step {selectedPropagation.arrivalStep}, {selectedPropagation.hop} edge
                                  {selectedPropagation.hop === 1 ? "" : "s"} from{" "}
                                  <button
                                    type="button"
                                    onClick={() => selectPoint(selectedPropagation.source)}
                                    className="focus-ring rounded underline decoration-dotted hover:text-white"
                                  >
                                    a labeled point of {data.classNames[data.classIndex[selectedPropagation.source]]}
                                  </button>
                                  {data.classIndex[selectedPropagation.source] === data.classIndex[selectedIndex!] ? (
                                    <span className="text-[#c8d2e4]"> · its own class</span>
                                  ) : (
                                    <span style={{ color: CATEGORICAL[1] }}> · another class</span>
                                  )}
                                  . The edge that delivered it carried {percent(selectedPropagation.parentShare, 1)} of the
                                  mass arriving that step.
                                </>
                              )}
                            </p>
                            <p className="mt-1 text-[11px] text-[#98a4ba]">
                              At step {safeStep} it holds{" "}
                              {selectedPropagation.heldClass >= 0 ? (
                                <span className="text-[#e3e8f2]">{data.classNames[selectedPropagation.heldClass]}</span>
                              ) : (
                                "nothing yet"
                              )}
                              {selectedPropagation.heldClass >= 0 &&
                                (selectedPropagation.heldClass === data.classIndex[selectedIndex!] ? " · right" : " · wrong")}
                              . It passed the label on to {selectedPropagation.children.length} point
                              {selectedPropagation.children.length === 1 ? "" : "s"}.
                            </p>
                            {selectedPropagation.contributors.length > 0 && (
                              <ul className="mt-2 space-y-1">
                                {selectedPropagation.contributors.map(({ row, share }) => (
                                  <li key={row} className="text-[10px] text-[#c8d2e4]">
                                    <button
                                      type="button"
                                      onClick={() => selectPoint(row)}
                                      className="focus-ring flex w-full justify-between gap-2 rounded text-left hover:bg-white/[0.04]"
                                    >
                                      <span className="min-w-0 truncate">
                                        {data.classNames[data.classIndex[row]]}
                                        {data.subsets[data.subset[row]] === "labeled" && (
                                          <span className="text-[#98a4ba]"> · labeled</span>
                                        )}
                                      </span>
                                      <span className="shrink-0 tabular-nums">{percent(share, 1)}</span>
                                    </button>
                                    <span className="mt-0.5 block h-[4px] rounded-full bg-white/[0.05]">
                                      <span
                                        className="block h-full rounded-full"
                                        style={{
                                          width: `${Math.max(2, share * 100)}%`,
                                          background:
                                            data.classIndex[row] === data.classIndex[selectedIndex!]
                                              ? CATEGORICAL[0]
                                              : CATEGORICAL[1],
                                        }}
                                      />
                                    </span>
                                  </li>
                                ))}
                              </ul>
                            )}
                            <p className="mt-1 text-[10px] leading-relaxed text-[#7f8ba2]">
                              The bars are the neighbors holding the most of its settled class: no single edge carries a
                              label, so this is where its mass is coming from now.
                            </p>
                          </>
                        )}
                      </div>
                    ) : (
                      <p className="mt-3 text-[11px] leading-relaxed text-[#7f8ba2]">
                        Select a point to see the chain its label came over. With the image column set to{" "}
                        <span className="text-[#c8d2e4]">Label propagation</span> the canvas draws that chain; with nothing
                        selected it draws every arrival edge up to the current step.
                      </p>
                    )}
                  </Panel>
                )}

                {selectedIndex != null && selectedClass != null && (
                  <Panel
                    title="Selected point"
                    action={
                      <button
                        type="button"
                        onClick={() => setSelectedIndex(null)}
                        className="focus-ring rounded p-1 text-[#8895aa] hover:text-white"
                        aria-label="Clear selection"
                      >
                        <X size={13} />
                      </button>
                    }
                  >
                    <p className="mt-2 truncate text-base font-semibold text-white" title={data.classNames[selectedClass]}>
                      {data.classNames[selectedClass]}
                    </p>
                    <p className="mt-1 flex items-center gap-2 text-xs text-[#98a4ba]">
                      <ShapeKey
                        shape={subsetStyle(data.subsets[data.subset[selectedIndex]]).shape}
                        color={CANVAS_COLORS[encoding.color[selectedIndex]]}
                      />
                      {subsetStyle(data.subsets[data.subset[selectedIndex]]).label} · sample #{data.sample[selectedIndex]}
                    </p>
                    {data.subsets[data.subset[selectedIndex]] === "unlabeled" && (
                      <p className="mt-1 text-[11px] text-[#7f8ba2]">The class is its true label, which training never saw.</p>
                    )}
                    {selectedNeighbors && (
                      <div className="mt-3">
                        <p className="text-[11px] leading-snug text-[#c8d2e4]">
                          {selectedNeighbors.shown.length} of {selectedNeighbors.graphCount} graph neighbors shown at epoch{" "}
                          {data.tracking?.epochs[safeEpochIndex]}: {selectedNeighbors.sameClass} same class,{" "}
                          {selectedNeighbors.labeled} labeled ({selectedNeighbors.labeledSameClass} of them same class).
                        </p>
                        <p className="mt-1 text-[11px] text-[#7f8ba2]">
                          Lines: light same class, orange another class, thicker a labeled neighbor.
                        </p>
                      </div>
                    )}
                    {data.tracking?.evidence?.[safeEpochIndex] && (
                      <ClassEvidence
                        data={data}
                        row={selectedIndex}
                        evidence={data.tracking.evidence[safeEpochIndex]!}
                        pseudo={data.tracking.pseudoLabels?.[safeEpochIndex] ?? null}
                      />
                    )}
                    {describePoint(selectedIndex)
                      .lines.slice(2)
                      .map((line) => (
                        <p key={line} className="mt-1 text-[11px] leading-snug text-[#c8d2e4]">
                          {line}
                        </p>
                      ))}
                    {data.graph && data.graph.nearestLabeled[selectedIndex] >= 0 && data.subsets[data.subset[selectedIndex]] !== "labeled" && (
                      <p className="mt-2 text-[11px] text-[#7f8ba2]">
                        Lines: its {data.graph.k} nearest train points (light: same class, orange: other) and, thicker, its
                        nearest labeled sample.
                      </p>
                    )}
                    {data.graph && data.subsets[data.subset[selectedIndex]] === "labeled" && (
                      <p className="mt-2 text-[11px] text-[#7f8ba2]">
                        Lines: the unlabeled points whose nearest labeled sample this is (light: same class, orange: other).
                      </p>
                    )}
                    <div className="mt-3 flex flex-wrap gap-2">
                      <button
                        type="button"
                        onClick={() => pinClass(selectedClass)}
                        disabled={pinned.includes(selectedClass)}
                        className="focus-ring flex items-center gap-1.5 rounded-lg border border-white/[0.08] px-2.5 py-1.5 text-xs font-semibold text-[#c8d2e4] hover:text-white disabled:opacity-40"
                      >
                        <Pin size={12} />
                        Pin class
                      </button>
                      <button
                        type="button"
                        onClick={() => canvasRef.current?.focusClass(selectedClass)}
                        className="focus-ring flex items-center gap-1.5 rounded-lg border border-white/[0.08] px-2.5 py-1.5 text-xs font-semibold text-[#c8d2e4] hover:text-white"
                      >
                        <Crosshair size={12} />
                        Zoom to class
                      </button>
                      {data.graph && data.graph.nearestLabeled[selectedIndex] >= 0 && (
                        <button
                          type="button"
                          onClick={() => goToPoint(data.graph!.nearestLabeled[selectedIndex])}
                          className="focus-ring flex items-center gap-1.5 rounded-lg border border-white/[0.08] px-2.5 py-1.5 text-xs font-semibold text-[#c8d2e4] hover:text-white"
                        >
                          Go to nearest labeled
                        </button>
                      )}
                    </div>
                    <ClassSubsetTable data={data} classes={[selectedClass]} />
                    {data.tracking && (
                      <div className="mt-3 max-h-56 overflow-y-auto">
                        <p className="text-[11px] text-[#7f8ba2]">This point at every tracked epoch (click a row to jump):</p>
                        <table className="mt-1 w-full text-[11px]">
                          <thead>
                            <tr className="text-left text-[#7f8ba2]">
                              <th className="py-1 font-semibold">Epoch</th>
                              <th className="py-1 pl-1 text-right font-semibold">Purity</th>
                              <th className="py-1 pl-1 text-right font-semibold">Hops</th>
                              <th className="py-1 pl-1 text-right font-semibold">Strongest</th>
                              {data.tracking.clusters && <th className="py-1 pl-1 text-right font-semibold">Cluster</th>}
                            </tr>
                          </thead>
                          <tbody>
                            {data.tracking.epochs.map((epoch, index) => {
                              const metrics = data.tracking!.nodeMetrics[index] ?? {};
                              const purity = metrics.same_label_neighbor_fraction?.[selectedIndex];
                              const hops = metrics.distance_to_labeled_node?.[selectedIndex];
                              const strongest = metrics.strongest_neighbor_same_label?.[selectedIndex];
                              const group = data.tracking!.clusters?.[index]?.[selectedIndex];
                              return (
                                <tr
                                  key={epoch}
                                  onClick={() => {
                                    setPlaying(false);
                                    setEpochIndex(index);
                                  }}
                                  className={`cursor-pointer border-t border-white/[0.05] ${
                                    index === safeEpochIndex ? "bg-white/[0.06] text-white" : "text-[#c8d2e4] hover:bg-white/[0.03]"
                                  }`}
                                >
                                  <td className="py-1 tabular-nums">{epoch}</td>
                                  <td className="py-1 pl-1 text-right tabular-nums">{percent(purity)}</td>
                                  <td className="py-1 pl-1 text-right tabular-nums">
                                    {hops != null && Number.isFinite(hops) ? hops : "—"}
                                  </td>
                                  <td className="py-1 pl-1 text-right">
                                    {strongest == null || Number.isNaN(strongest) ? "—" : strongest > 0.5 ? "same" : "other"}
                                  </td>
                                  {data.tracking!.clusters && (
                                    <td className="py-1 pl-1 text-right tabular-nums">{group != null && group >= 0 ? group : "—"}</td>
                                  )}
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </Panel>
                )}

                {anchors.length > 0 && (
                  <Panel title="Labeled anchors">
                    <p className="mt-2 text-[11px] leading-relaxed text-[#7f8ba2]">
                      Labeled samples that are the nearest labeled sample for the most unlabeled points in the trained space,
                      and how many of those share their class. This is where the unlabeled pool gathered, not proof that a
                      sample caused it.
                    </p>
                    <table className="mt-2 w-full text-[11px]">
                      <thead>
                        <tr className="text-left text-[#7f8ba2]">
                          <th className="py-1 font-semibold">Class</th>
                          <th className="py-1 text-right font-semibold">Pulls</th>
                          <th className="py-1 text-right font-semibold">Same class</th>
                          <th className="py-1" />
                        </tr>
                      </thead>
                      <tbody>
                        {anchors.map((anchor) => (
                          <tr key={anchor.row} className="border-t border-white/[0.05] text-[#c8d2e4]">
                            <td className="max-w-[110px] truncate py-1" title={data.classNames[data.classIndex[anchor.row]]}>
                              {data.classNames[data.classIndex[anchor.row]]}
                            </td>
                            <td className="py-1 text-right tabular-nums">{anchor.pulls}</td>
                            <td className="py-1 text-right tabular-nums">{percent(anchor.same / anchor.pulls)}</td>
                            <td className="py-1 text-right">
                              <button
                                type="button"
                                onClick={() => goToPoint(anchor.row)}
                                className="focus-ring rounded p-1 text-[#8895aa] hover:text-white"
                                aria-label={`Select this labeled ${data.classNames[data.classIndex[anchor.row]]} sample`}
                              >
                                <Crosshair size={11} />
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Panel>
                )}

                {boxSummary && boxSelection && (
                  <Panel
                    title="Region"
                    action={
                      <button
                        type="button"
                        onClick={() => setBoxSelection(null)}
                        className="focus-ring rounded p-1 text-[#8895aa] hover:text-white"
                        aria-label="Clear region"
                      >
                        <X size={13} />
                      </button>
                    }
                  >
                    <p className="mt-2 text-base font-semibold text-white">
                      {formatCount(boxSelection.length)} points · {formatCount(boxSummary.classCount)} classes
                    </p>
                    <p className="mt-1 text-xs text-[#98a4ba]">
                      Largest class holds {percent(boxSummary.topClasses[0][1] / boxSelection.length, 1)}
                    </p>
                    <ul className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[11px] text-[#98a4ba]">
                      {data.subsets.map((name, subsetIndex) =>
                        boxSummary.perSubset[subsetIndex] > 0 ? (
                          <li key={name} className="flex items-center gap-1.5">
                            <ShapeKey shape={subsetStyle(name).shape} color={legendColor(name, effectiveColorMode)} size={10} />
                            {subsetStyle(name).label} {formatCount(boxSummary.perSubset[subsetIndex])}
                          </li>
                        ) : null,
                      )}
                    </ul>
                    <table className="mt-3 w-full text-[11px]">
                      <thead>
                        <tr className="text-left text-[#7f8ba2]">
                          <th className="py-1 font-semibold">Class</th>
                          <th className="py-1 text-right font-semibold">Points</th>
                          <th className="py-1 text-right font-semibold">Share</th>
                          <th className="py-1" />
                        </tr>
                      </thead>
                      <tbody>
                        {boxSummary.topClasses.map(([classIndex, count]) => (
                          <tr key={classIndex} className="border-t border-white/[0.05] text-[#c8d2e4]">
                            <td className="max-w-[120px] truncate py-1" title={data.classNames[classIndex]}>
                              {data.classNames[classIndex]}
                            </td>
                            <td className="py-1 text-right tabular-nums">{formatCount(count)}</td>
                            <td className="py-1 text-right tabular-nums">{percent(count / boxSelection.length, 1)}</td>
                            <td className="py-1 text-right">
                              <button
                                type="button"
                                onClick={() => pinClass(classIndex)}
                                disabled={pinned.includes(classIndex)}
                                className="focus-ring rounded p-1 text-[#8895aa] hover:text-white disabled:opacity-40"
                                aria-label={`Pin ${data.classNames[classIndex]}`}
                              >
                                <Pin size={11} />
                              </button>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </Panel>
                )}

                <Panel title="Pinned classes">
                  <div className="relative mt-2">
                    <Search size={13} className="pointer-events-none absolute left-2.5 top-2.5 text-[#65748e]" />
                    <input
                      value={classQuery}
                      onChange={(event) => setClassQuery(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key !== "Enter") return;
                        const target = classNameIndex.get(classQuery.trim().toLowerCase()) ?? classMatches[0];
                        if (target != null) {
                          pinClass(target);
                          setClassQuery("");
                        }
                      }}
                      placeholder={`Find a class (${formatCount(data.classNames.length)})`}
                      aria-label="Find a class to pin"
                      className="focus-ring w-full rounded-lg border border-white/[0.08] bg-[#10182a] py-1.5 pl-8 pr-2 text-xs text-white placeholder:text-[#596780]"
                    />
                    {classMatches.length > 0 && (
                      <ul className="absolute z-20 mt-1 w-full overflow-hidden rounded-lg border border-white/[0.08] bg-[#10182a] shadow-xl">
                        {classMatches.map((classIndex) => (
                          <li key={classIndex}>
                            <button
                              type="button"
                              onClick={() => {
                                pinClass(classIndex);
                                setClassQuery("");
                              }}
                              className="focus-ring flex w-full items-center gap-2 px-2.5 py-1.5 text-left text-xs text-[#c8d2e4] hover:bg-white/[0.05]"
                            >
                              <Pin size={11} className="text-[#65748e]" />
                              <span className="truncate">{data.classNames[classIndex]}</span>
                            </button>
                          </li>
                        ))}
                      </ul>
                    )}
                  </div>
                  {pinned.length === 0 ? (
                    <p className="mt-2 text-[11px] leading-relaxed text-[#7f8ba2]">
                      Click a point, then pin its class, or search above.
                    </p>
                  ) : (
                    <>
                      <ul className="mt-2 space-y-1">
                        {pinned.map((classIndex, slot) => (
                          <li key={classIndex} className="flex items-center gap-2 text-xs text-[#e3e8f2]">
                            <span aria-hidden className="h-[2px] w-3 rounded-full" style={{ background: CATEGORICAL[slot] }} />
                            <span className="truncate font-semibold" title={data.classNames[classIndex]}>
                              {data.classNames[classIndex]}
                            </span>
                            <button
                              type="button"
                              onClick={() => canvasRef.current?.focusClass(classIndex)}
                              className="focus-ring ml-auto rounded p-1 text-[#8895aa] hover:text-white"
                              aria-label={`Zoom to ${data.classNames[classIndex]}`}
                            >
                              <Crosshair size={12} />
                            </button>
                            <button
                              type="button"
                              onClick={() => setPinned((current) => current.filter((value) => value !== classIndex))}
                              className="focus-ring rounded p-1 text-[#8895aa] hover:text-white"
                              aria-label={`Unpin ${data.classNames[classIndex]}`}
                            >
                              <PinOff size={12} />
                            </button>
                          </li>
                        ))}
                      </ul>
                      <ClassSubsetTable data={data} classes={pinned} />
                    </>
                  )}
                </Panel>

                {data.notes.length > 0 && (
                  <Panel title="About this view">
                    <p className="mt-2 text-[11px] leading-relaxed text-[#7f8ba2]">
                      {data.projection} projection
                      {data.selectedEpoch != null ? ` of the head selected at epoch ${data.selectedEpoch}` : ""}. Axes carry no
                      units; only neighborhoods mean something.
                    </p>
                    <ul className="mt-1 list-disc space-y-0.5 pl-4 text-[11px] leading-relaxed text-[#7f8ba2]">
                      {data.notes.map((note) => (
                        <li key={note}>{note}</li>
                      ))}
                    </ul>
                  </Panel>
                )}
              </>
            )}
          </aside>
        </div>
      )}
    </main>
  );
}
