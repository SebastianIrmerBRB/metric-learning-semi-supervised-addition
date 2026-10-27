"use client";

import Link from "next/link";
import { useMemo, useRef, useState, type PointerEvent as ReactPointerEvent, type ReactNode } from "react";
import { ArrowLeft, Cpu, Info, Timer, TriangleAlert } from "lucide-react";
import { FigureCard } from "@/components/ExportMenu";
import { toLatexTable, type ExportLegendItem, type ExportRow } from "@/lib/export";
import {
  AGGREGATION_LABEL,
  DIMENSION_LABEL,
  METRICS,
  PHASE_LABEL,
  TIME_MODEL_LABEL,
  aggregate,
  dimensionLabel,
  dimensionValue,
  formatMetric,
  formatNumber,
  mainEpochSeconds,
  makeContext,
  metricByKey,
  orderedValues,
  type CurvePoint,
  type Dimension,
  type MetricContext,
  type MetricDef,
  type MetricKey,
  type PhaseRate,
  type TimingCase,
  type TimingSnapshotEntry,
  type TimingStudy,
} from "@/lib/timings";

/* Categorical slots 1-3 of the reference palette (dark steps), validated all-pairs on the card surface. */
const COMPONENTS = [
  { key: "train", label: "Training", color: "#3987e5" },
  { key: "between", label: "Zwischen den Epochen", color: "#d95926" },
  { key: "valid", label: "Validierung", color: "#199e70" },
] as const;
type ComponentKey = (typeof COMPONENTS)[number]["key"];

const COMPONENT_LEGEND: ExportLegendItem[] = COMPONENTS.map(({ label, color }) => ({ name: label, color }));

/* Sequential blue ramp, dark mode: small values recede toward the surface, large ones light up. */
const RAMP = [
  "#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5",
  "#5598e7", "#6da7ec", "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb",
];
const FIRST_LIGHT_STEP = 6;

const MUTED = "#7f8ba2";
const GRID = "rgba(148, 163, 184, 0.14)";

interface Layout {
  key: string;
  label: string;
  rows: Dimension;
  cols: Dimension;
  filter: Dimension;
}

const LAYOUTS: Layout[] = [
  { key: "method-dataset", label: "Verfahren × Datensatz", rows: "method", cols: "dataset", filter: "loss" },
  { key: "method-loss", label: "Verfahren × Verlust", rows: "method", cols: "loss", filter: "dataset" },
  { key: "loss-dataset", label: "Verlust × Datensatz", rows: "loss", cols: "dataset", filter: "method" },
];

const ALL = "__all__";
const ALL_LABEL: Record<Dimension, string> = { method: "alle Verfahren", dataset: "alle Datensätze", loss: "alle Verluste" };

/* ------------------------------------------------------------------ shared UI */

function Select({
  label,
  value,
  options,
  onChange,
}: {
  label: string;
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">{label}</span>
      <select
        className="focus-ring rounded-lg border border-white/[0.08] bg-[#0f1522] px-2.5 py-1.5 text-xs text-white"
        value={value}
        onChange={(event) => onChange(event.target.value)}
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

function Toggle({
  value,
  options,
  onChange,
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <div className="inline-flex rounded-lg border border-white/[0.08] bg-[#0f1522] p-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          onClick={() => onChange(option.value)}
          className={`focus-ring rounded-md px-2.5 py-1 text-[11px] font-semibold transition ${
            value === option.value ? "bg-white/[0.09] text-white" : "text-[#8895aa] hover:text-white"
          }`}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function StatTile({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="rounded-xl border border-white/[0.07] bg-white/[0.02] px-3.5 py-3">
      <p className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">{label}</p>
      <p className="mt-1 text-lg font-semibold text-white">{value}</p>
      {note && <p className="mt-0.5 text-[11px] text-[#7f8ba2]">{note}</p>}
    </div>
  );
}

function LegendRow({ items }: { items: ExportLegendItem[] }) {
  return (
    <div className="flex flex-wrap gap-x-4 gap-y-1 text-[11px] text-[#b3bdd0]">
      {items.map((item) => (
        <span key={item.name} className="inline-flex items-center gap-1.5">
          <span className="inline-block h-2.5 w-2.5 rounded-[3px]" style={{ background: item.color }} />
          {item.name}
        </span>
      ))}
    </div>
  );
}

/** One floating readout per chart; marks report their content on hover and focus. */
function useTooltip() {
  const container = useRef<HTMLDivElement>(null);
  const [tip, setTip] = useState<{ x: number; y: number; content: ReactNode } | null>(null);
  const place = (clientX: number, clientY: number, content: ReactNode) => {
    const box = container.current?.getBoundingClientRect();
    if (!box) return;
    setTip({ x: clientX - box.left, y: clientY - box.top, content });
  };
  const handlers = (content: ReactNode) => ({
    onPointerMove: (event: ReactPointerEvent<Element>) => place(event.clientX, event.clientY, content),
    onPointerLeave: () => setTip(null),
    onFocus: (event: { currentTarget: Element }) => {
      const rect = event.currentTarget.getBoundingClientRect();
      place(rect.left + rect.width / 2, rect.top, content);
    },
    onBlur: () => setTip(null),
  });
  const node = tip ? (
    <div
      className="pointer-events-none absolute z-20 w-max min-w-[10rem] max-w-[20rem] -translate-x-1/2 -translate-y-[calc(100%+12px)] rounded-lg border border-white/[0.1] bg-[#0b1120]/95 px-3 py-2 text-[11px] text-[#b3bdd0] shadow-xl"
      style={{ left: tip.x, top: tip.y }}
    >
      {tip.content}
    </div>
  ) : null;
  return { container, handlers, node };
}

function TipRow({ color, label, value }: { color?: string; label: string; value: string }) {
  return (
    <div className="flex items-center justify-between gap-3 whitespace-nowrap">
      <span className="inline-flex items-center gap-1.5">
        {color && <span className="inline-block h-[2px] w-3 rounded-full" style={{ background: color }} />}
        {label}
      </span>
      <span className="font-semibold tabular-nums text-white">{value}</span>
    </div>
  );
}

/* --------------------------------------------------------------- heat pivot */

interface PivotCell {
  row: string;
  col: string;
  studies: TimingStudy[];
  value: number | null;
}

function rampColor(value: number, min: number, max: number): { fill: string; ink: string } {
  const t = max > min ? (value - min) / (max - min) : 0.5;
  const index = Math.max(0, Math.min(RAMP.length - 1, Math.round(t * (RAMP.length - 1))));
  return { fill: RAMP[index], ink: index >= FIRST_LIGHT_STEP ? "#0b0b0b" : "#ffffff" };
}

function HeatPivot({
  layout,
  rows,
  cols,
  labels,
  cells,
  rowTotals,
  colTotals,
  total,
  metric,
  selected,
  onSelect,
}: {
  layout: Layout;
  rows: string[];
  cols: string[];
  labels: Record<Dimension, Record<string, string>>;
  cells: Map<string, PivotCell>;
  rowTotals: Map<string, number | null>;
  colTotals: Map<string, number | null>;
  total: number | null;
  metric: MetricDef;
  selected: { row: string; col: string } | null;
  onSelect: (cell: PivotCell) => void;
}) {
  const { container, handlers, node } = useTooltip();
  const values = Array.from(cells.values())
    .map((cell) => cell.value)
    .filter((value): value is number => value != null);
  const min = values.length ? Math.min(...values) : 0;
  const max = values.length ? Math.max(...values) : 1;
  const marginLabel = AGGREGATION_LABEL[metric.aggregation];

  return (
    <div ref={container} className="relative">
      <div className="overflow-x-auto">
        <table className="w-full border-separate border-spacing-[2px] text-xs">
          <thead>
            <tr>
              <th className="w-44 px-2 py-1.5 text-left text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">
                {DIMENSION_LABEL[layout.rows]} \ {DIMENSION_LABEL[layout.cols]}
              </th>
              {cols.map((col) => (
                <th key={col} className="px-2 py-1.5 text-center font-semibold text-[#c8d2e4]">
                  {labels[layout.cols][col] ?? col}
                </th>
              ))}
              <th className="w-24 px-2 py-1.5 text-center text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">
                {marginLabel}
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row}>
                <th className="whitespace-nowrap px-2 py-1.5 text-left font-semibold text-[#c8d2e4]">
                  {labels[layout.rows][row] ?? row}
                </th>
                {cols.map((col) => {
                  const cell = cells.get(`${row}|${col}`);
                  if (!cell || cell.studies.length === 0) {
                    return (
                      <td key={col} className="rounded-md bg-white/[0.02] px-2 py-2.5 text-center text-[#4f5d75]">
                        ·
                      </td>
                    );
                  }
                  const colors = cell.value == null ? null : rampColor(cell.value, min, max);
                  const isSelected = selected?.row === row && selected?.col === col;
                  const estimated = cell.studies.reduce((sum, study) => sum + study.trials_estimated, 0);
                  const trials = cell.studies.reduce((sum, study) => sum + study.trials, 0);
                  return (
                    <td key={col} className="p-0">
                      <button
                        type="button"
                        onClick={() => onSelect(cell)}
                        {...handlers(
                          <>
                            <p className="font-semibold text-white">
                              {labels[layout.rows][row] ?? row} · {labels[layout.cols][col] ?? col}
                            </p>
                            <TipRow label={`${metric.label}${cell.studies.length > 1 ? ` (${marginLabel})` : ""}`} value={formatMetric(metric, cell.value)} />
                            <TipRow label="Studien" value={String(cell.studies.length)} />
                            <TipRow label="Trials geschätzt" value={`${estimated}/${trials}`} />
                          </>,
                        )}
                        className={`focus-ring w-full rounded-md px-2 py-2.5 text-center font-semibold tabular-nums transition hover:brightness-110 ${
                          isSelected ? "ring-2 ring-white/80" : ""
                        }`}
                        style={colors ? { background: colors.fill, color: colors.ink } : { background: "rgba(255,255,255,0.03)", color: MUTED }}
                      >
                        {formatNumber(cell.value, metric.digits)}
                      </button>
                    </td>
                  );
                })}
                <td className="rounded-md bg-white/[0.03] px-2 py-2.5 text-center font-semibold tabular-nums text-[#c8d2e4]">
                  {formatNumber(rowTotals.get(row), metric.digits)}
                </td>
              </tr>
            ))}
            <tr>
              <th className="px-2 py-1.5 text-left text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">
                {marginLabel}
              </th>
              {cols.map((col) => (
                <td key={col} className="rounded-md bg-white/[0.03] px-2 py-2.5 text-center font-semibold tabular-nums text-[#c8d2e4]">
                  {formatNumber(colTotals.get(col), metric.digits)}
                </td>
              ))}
              <td className="rounded-md bg-white/[0.06] px-2 py-2.5 text-center font-semibold tabular-nums text-white">
                {formatNumber(total, metric.digits)}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
      <div className="mt-3 flex flex-wrap items-center gap-3 text-[11px] text-[#7f8ba2]">
        <span>{formatMetric(metric, min)}</span>
        <span
          className="h-2 w-40 rounded-full"
          style={{ background: `linear-gradient(90deg, ${RAMP[0]}, ${RAMP[Math.floor(RAMP.length / 2)]}, ${RAMP[RAMP.length - 1]})` }}
        />
        <span>{formatMetric(metric, max)}</span>
        <span className="ml-2">Randspalten: {marginLabel} über die Studien der Zeile bzw. Spalte, nicht eingefärbt.</span>
      </div>
      {node}
    </div>
  );
}

/* ------------------------------------------------------------ phase bars */

function componentsOf(rate: { train_s: number; valid_s: number; gap_s: number; other_s: number }): Record<ComponentKey, number> {
  return {
    train: Math.max(0, rate.train_s),
    between: Math.max(0, rate.gap_s + rate.other_s),
    valid: Math.max(0, rate.valid_s),
  };
}

function PhaseBars({ timingCase, supervised }: { timingCase: TimingCase; supervised: boolean }) {
  const { container, handlers, node } = useTooltip();
  const bars: { label: string; rate: PhaseRate }[] = [];
  if (timingCase.warm) bars.push({ label: supervised ? "Überwachte Epoche" : "Warm-up-Epoche", rate: timingCase.warm });
  if (timingCase.ssl_first) bars.push({ label: "Erste SSL-Epoche", rate: timingCase.ssl_first });
  if (timingCase.ssl) bars.push({ label: "SSL-Epoche (Mittel)", rate: timingCase.ssl });
  if (bars.length === 0) return <p className="text-xs text-[#7f8ba2]">Keine Epochenraten gemessen.</p>;

  const width = 640;
  const labelWidth = 150;
  const valueWidth = 70;
  const plot = width - labelWidth - valueWidth;
  const band = 36;
  const thickness = 20;
  const maxTotal = Math.max(...bars.map((bar) => bar.rate.total_s));
  const scale = (seconds: number) => (maxTotal > 0 ? (seconds / maxTotal) * plot : 0);
  const height = bars.length * band + 8;

  return (
    <div ref={container} className="relative">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Zusammensetzung einer Epoche je Phase">
        {bars.map((bar, index) => {
          const y = 4 + index * band + (band - thickness) / 2;
          const parts = componentsOf(bar.rate);
          let x = labelWidth;
          const segments = COMPONENTS.map((component) => {
            const w = scale(parts[component.key]);
            const segment = { ...component, x, w, seconds: parts[component.key] };
            x += w;
            return segment;
          }).filter((segment) => segment.w > 0);
          const tip = (
            <>
              <p className="font-semibold text-white">{bar.label}</p>
              <TipRow label="Gesamt" value={`${formatNumber(bar.rate.total_s, 3)} s`} />
              {COMPONENTS.map((component) => (
                <TipRow key={component.key} color={component.color} label={component.label} value={`${formatNumber(parts[component.key], 3)} s`} />
              ))}
              <TipRow label="Schritte je Epoche" value={formatNumber(bar.rate.steps, 0)} />
              <TipRow label="gemessene Epochen" value={String(bar.rate.n)} />
            </>
          );
          return (
            <g key={bar.label} tabIndex={0} className="focus-ring outline-none" {...handlers(tip)}>
              <rect x={0} y={y - 8} width={width} height={thickness + 16} fill="transparent" />
              <text x={0} y={y + thickness / 2 + 4} fill="#c8d2e4" fontSize={12}>
                {bar.label}
              </text>
              {segments.map((segment, segmentIndex) => {
                const last = segmentIndex === segments.length - 1;
                // 2px surface gap between touching segments; only the data end is rounded.
                const w = Math.max(0, segment.w - (last ? 0 : 2));
                return last ? (
                  <path
                    key={segment.key}
                    d={roundedEndBar(segment.x, y, w, thickness, 4)}
                    fill={segment.color}
                  />
                ) : (
                  <rect key={segment.key} x={segment.x} y={y} width={w} height={thickness} fill={segment.color} />
                );
              })}
              <text x={labelWidth + scale(bar.rate.total_s) + 8} y={y + thickness / 2 + 4} fill="#f7f8fc" fontSize={12} fontWeight={600}>
                {formatNumber(bar.rate.total_s, 2)} s
              </text>
            </g>
          );
        })}
      </svg>
      {node}
    </div>
  );
}

/** A horizontal bar square at the baseline (left) with a rounded data end (right). */
function roundedEndBar(x: number, y: number, w: number, h: number, r: number): string {
  const radius = Math.min(r, w, h / 2);
  return [
    `M${x},${y}`,
    `H${x + w - radius}`,
    `Q${x + w},${y} ${x + w},${y + radius}`,
    `V${y + h - radius}`,
    `Q${x + w},${y + h} ${x + w - radius},${y + h}`,
    `H${x}`,
    "Z",
  ].join(" ");
}

/** A column square at the baseline (bottom) with a rounded data end (top). */
function roundedTopColumn(x: number, y: number, w: number, h: number, r: number): string {
  const radius = Math.min(r, w / 2, h);
  return [
    `M${x},${y + h}`,
    `V${y + radius}`,
    `Q${x},${y} ${x + radius},${y}`,
    `H${x + w - radius}`,
    `Q${x + w},${y} ${x + w},${y + radius}`,
    `V${y + h}`,
    "Z",
  ].join(" ");
}

/* ------------------------------------------------------------ epoch curve */

function niceTicks(max: number, count = 4): number[] {
  if (max <= 0) return [0];
  const raw = max / count;
  const power = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * power).find((candidate) => candidate >= raw) ?? raw;
  const ticks = [];
  for (let value = 0; value <= max + step * 0.001; value += step) ticks.push(Number(value.toFixed(10)));
  return ticks;
}

function EpochColumns({ curve, warmup, sslEnabled }: { curve: CurvePoint[]; warmup: number; sslEnabled: boolean }) {
  const { container, handlers, node } = useTooltip();
  if (curve.length === 0) return <p className="text-xs text-[#7f8ba2]">Kein Epochenverlauf gespeichert.</p>;
  const width = 640;
  const height = 220;
  const left = 44;
  const right = 8;
  const top = 34;
  const bottom = 26;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const totals = curve.map((point) => point.gap_s + point.other_s + point.train_s + point.valid_s);
  const ticks = niceTicks(Math.max(...totals));
  const yMax = ticks[ticks.length - 1] || 1;
  const slot = plotWidth / curve.length;
  const columnWidth = Math.min(24, Math.max(2, slot - 2));
  const y = (seconds: number) => top + plotHeight - (seconds / yMax) * plotHeight;
  const boundary = sslEnabled && warmup > 0 ? left + warmup * slot : null;
  const labelEvery = Math.max(1, Math.ceil(curve.length / 12));

  return (
    <div ref={container} className="relative">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Zeit je Epoche über den Lauf">
        {ticks.map((tick) => (
          <g key={tick}>
            <line x1={left} x2={width - right} y1={y(tick)} y2={y(tick)} stroke={GRID} strokeWidth={1} />
            <text x={left - 6} y={y(tick) + 3} textAnchor="end" fill={MUTED} fontSize={10} style={{ fontVariantNumeric: "tabular-nums" }}>
              {formatNumber(tick, tick < 1 ? 2 : 1)}
            </text>
          </g>
        ))}
        <text x={4} y={12} fill={MUTED} fontSize={10}>
          s je Epoche
        </text>
        {curve.map((point, index) => {
          const parts = componentsOf(point);
          const x = left + index * slot + (slot - columnWidth) / 2;
          let base = 0;
          const stack = COMPONENTS.map((component) => {
            const from = base;
            base += parts[component.key];
            return { ...component, from, to: base, seconds: parts[component.key] };
          }).filter((segment) => segment.seconds > 0);
          const tip = (
            <>
              <p className="font-semibold text-white">
                Epoche {point.epoch} · {PHASE_LABEL[point.phase]}
              </p>
              <TipRow label="Gesamt" value={`${formatNumber(base, 3)} s`} />
              {COMPONENTS.map((component) => (
                <TipRow key={component.key} color={component.color} label={component.label} value={`${formatNumber(parts[component.key], 3)} s`} />
              ))}
            </>
          );
          return (
            <g key={point.epoch} tabIndex={0} className="outline-none" {...handlers(tip)}>
              <rect x={left + index * slot} y={top} width={slot} height={plotHeight} fill="transparent" />
              {stack.map((segment, segmentIndex) => {
                const last = segmentIndex === stack.length - 1;
                const yTop = y(segment.to);
                // 2px surface gap between stacked segments.
                const h = Math.max(0, y(segment.from) - yTop - (segmentIndex > 0 ? 2 : 0));
                return last ? (
                  <path key={segment.key} d={roundedTopColumn(x, yTop, columnWidth, h, Math.min(4, columnWidth / 2))} fill={segment.color} />
                ) : (
                  <rect key={segment.key} x={x} y={yTop} width={columnWidth} height={h} fill={segment.color} />
                );
              })}
              {index % labelEvery === 0 && (
                <text x={left + index * slot + slot / 2} y={height - 10} textAnchor="middle" fill={MUTED} fontSize={10}>
                  {point.epoch}
                </text>
              )}
            </g>
          );
        })}
        <line x1={left} x2={width - right} y1={y(0)} y2={y(0)} stroke="rgba(148,163,184,0.35)" strokeWidth={1} />
        {boundary != null && (
          <g>
            <line x1={boundary} x2={boundary} y1={top - 4} y2={top + plotHeight} stroke="#c8d2e4" strokeWidth={1} />
            <text
              x={boundary > width / 2 ? boundary - 4 : boundary + 4}
              y={top - 8}
              textAnchor={boundary > width / 2 ? "end" : "start"}
              fill="#c8d2e4"
              fontSize={10}
            >
              Ende Warm-up (Epoche {warmup})
            </text>
          </g>
        )}
      </svg>
      <p className="mt-1 text-center text-[10px] text-[#7f8ba2]">Epoche (Mittel über Folds und Wiederholungen)</p>
      {node}
    </div>
  );
}

/* ------------------------------------------------------------ study detail */

function StudyDetail({ study, timingCase, context }: { study: TimingStudy; timingCase: TimingCase | undefined; context: MetricContext }) {
  const supervised = study.method === "supervised";
  const reference = context.supervisedOf(study);
  const relative = supervised ? null : metricByKey("epochVsSup").value(study, context);
  const models = Object.entries(study.time_models).sort((a, b) => b[1] - a[1]);

  return (
    <div className="mt-4 space-y-4">
      <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h3 className="text-base font-semibold text-white">
          {study.method_label} · {study.dataset_label} · {study.loss_label}
        </h3>
        <span className="text-[11px] text-[#7f8ba2]" title={study.study_key}>
          {study.name ?? study.study_key.split("/").slice(-4).join("/")}
        </span>
      </div>
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <StatTile
          label={supervised ? "Epoche" : "SSL-Epoche"}
          value={`${formatNumber(mainEpochSeconds(study), 2)} s`}
          note={relative != null ? `${formatNumber(relative, 2)}× Ü (${reference?.loss_label ?? ""})` : undefined}
        />
        {!supervised && <StatTile label="Warm-up-Epoche" value={`${formatNumber(study.selected_warm_epoch_s, 2)} s`} />}
        {!supervised && <StatTile label="Erste SSL-Epoche" value={`${formatNumber(study.selected_ssl_first_epoch_s, 2)} s`} />}
        <StatTile label="Fixkosten je Fold" value={`${formatNumber(study.selected_fold_fixed_s, 2)} s`} note="Setup, Basisvalidierung, Abschluss" />
        <StatTile
          label="Epochen je Fold"
          value={formatNumber(study.epochs_per_fold_mean, 1)}
          note={supervised ? "Mittel über alle Trials" : `davon SSL ${formatNumber(study.ssl_epochs_per_fold_mean, 1)}, Warm-up ${formatNumber(study.warmup_epochs_mean, 1)}`}
        />
        <StatTile
          label="HPO-Kosten"
          value={`${formatNumber(study.estimated_gpu_hours_imputed, 2)} GPU-h`}
          note={`${study.trials_estimated}/${study.trials} Trials geschätzt`}
        />
        <StatTile label="Trial-Dauer" value={`${formatNumber(study.trial_minutes_median, 1)} min`} note="Median, allein auf der GPU" />
        <StatTile label="Beobachtet" value={`${formatNumber(study.observed_hours, 1)} h`} note="Optuna, parallel unter Last" />
      </div>

      {timingCase ? (
        <div className="grid gap-4 lg:grid-cols-2">
          <FigureCard
            title="Zusammensetzung einer Epoche"
            description={`Trial ${timingCase.trial}, Batchgröße ${timingCase.batch_size ?? "–"}, ${formatNumber(timingCase.steps_per_epoch, 0)} Schritte je Epoche`}
            name={`timings-phasen-${study.method}-${study.dataset}-${study.loss}`}
            legend={COMPONENT_LEGEND}
            rows={() => phaseRows(timingCase)}
          >
            <div className="mt-3">
              <LegendRow items={COMPONENT_LEGEND} />
              <div className="mt-2">
                <PhaseBars timingCase={timingCase} supervised={supervised} />
              </div>
            </div>
          </FigureCard>
          <FigureCard
            title="Zeit je Epoche über den Lauf"
            description={
              timingCase.update_mode === "every_n_epochs" && (timingCase.update_interval_epochs ?? 1) > 1
                ? `Refresh alle ${timingCase.update_interval_epochs} Epochen`
                : timingCase.update_mode
                  ? `Aktualisierung: ${timingCase.update_mode}`
                  : undefined
            }
            name={`timings-verlauf-${study.method}-${study.dataset}-${study.loss}`}
            legend={COMPONENT_LEGEND}
            rows={() => timingCase.curve.map((point) => ({ ...point }))}
          >
            <div className="mt-3">
              <LegendRow items={COMPONENT_LEGEND} />
              <div className="mt-2">
                <EpochColumns curve={timingCase.curve} warmup={timingCase.warmup_epochs} sslEnabled={timingCase.ssl_enabled} />
              </div>
            </div>
          </FigureCard>
        </div>
      ) : (
        <p className="rounded-xl border border-white/[0.07] bg-white/[0.02] px-4 py-3 text-xs text-[#98a4ba]">
          Für diese Studie wurde keine Konfiguration gemessen; ihre Kosten stammen aus einer anderen Studie derselben Kombination oder fehlen.
        </p>
      )}

      <div className="flex flex-wrap gap-x-6 gap-y-2 text-[11px] text-[#98a4ba]">
        <span className="font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">Zeitmodell der Trials</span>
        {models.map(([model, count]) => (
          <span key={model}>
            {TIME_MODEL_LABEL[model] ?? model}: <span className="font-semibold text-white">{count}</span>
          </span>
        ))}
        {timingCase && (
          <span>
            Messung: {timingCase.folds} Folds × {timingCase.epochs_per_fold} Epochen, {timingCase.repeats}× wiederholt,
            GPU-Auslastung ∅ {formatNumber(timingCase.gpu_utilization_mean, 0)} %
            {timingCase.contaminated && (
              <span className="ml-2 inline-flex items-center gap-1 font-semibold text-[#fab219]">
                <TriangleAlert size={12} /> GPU war geteilt
              </span>
            )}
          </span>
        )}
      </div>
    </div>
  );
}

function phaseRows(timingCase: TimingCase): ExportRow[] {
  const phases: [string, PhaseRate | null][] = [
    ["warm", timingCase.warm],
    ["ssl_first", timingCase.ssl_first],
    ["ssl", timingCase.ssl],
  ];
  return phases
    .filter((entry): entry is [string, PhaseRate] => entry[1] != null)
    .map(([phase, rate]) => ({
      phase,
      total_s: rate.total_s,
      train_s: rate.train_s,
      between_s: rate.gap_s + rate.other_s,
      valid_s: rate.valid_s,
      steps: rate.steps,
      epochs_measured: rate.n,
    }));
}

/* ------------------------------------------------------------ study table */

type SortKey = "dataset" | "method" | "loss" | MetricKey;

const TABLE_METRICS: MetricKey[] = ["epoch", "warmEpoch", "firstSslEpoch", "epochsPerFold", "trialMinutes", "gpuHours", "observedHours", "gpuHoursVsSup"];

function StudyTable({
  studies,
  context,
  onSelect,
  selectedKey,
}: {
  studies: TimingStudy[];
  context: MetricContext;
  onSelect: (study: TimingStudy) => void;
  selectedKey: string | null;
}) {
  const [sort, setSort] = useState<{ key: SortKey; descending: boolean }>({ key: "dataset", descending: false });
  const orders = useMemo(
    () => ({
      dataset: orderedValues(studies, "dataset"),
      method: orderedValues(studies, "method"),
      loss: orderedValues(studies, "loss"),
    }),
    [studies],
  );
  const sorted = useMemo(() => {
    const rank = (study: TimingStudy, dimension: Dimension) => orders[dimension].indexOf(dimensionValue(study, dimension));
    const byDimensions = (a: TimingStudy, b: TimingStudy) =>
      rank(a, "dataset") - rank(b, "dataset") || rank(a, "method") - rank(b, "method") || rank(a, "loss") - rank(b, "loss");
    const list = [...studies];
    list.sort((a, b) => {
      let result: number;
      if (sort.key === "dataset" || sort.key === "method" || sort.key === "loss") {
        result = rank(a, sort.key) - rank(b, sort.key) || byDimensions(a, b);
      } else {
        const metric = metricByKey(sort.key);
        const va = metric.value(a, context);
        const vb = metric.value(b, context);
        result = va == null ? 1 : vb == null ? -1 : va - vb;
      }
      return sort.descending ? -result : result;
    });
    return list;
  }, [studies, sort, orders, context]);

  const header = (key: SortKey, label: string, align: "left" | "right" = "right") => (
    <th key={key} className={`px-2 py-2 font-semibold text-[#8895aa] ${align === "left" ? "text-left" : "text-right"}`}>
      <button
        type="button"
        className="focus-ring hover:text-white"
        onClick={() => setSort((current) => ({ key, descending: current.key === key ? !current.descending : key !== "dataset" && key !== "method" && key !== "loss" }))}
      >
        {label}
        {sort.key === key ? (sort.descending ? " ↓" : " ↑") : ""}
      </button>
    </th>
  );

  return (
    <div className="max-h-[32rem] overflow-auto">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-[#0d1424]">
          <tr>
            {header("dataset", "Datensatz", "left")}
            {header("method", "Verfahren", "left")}
            {header("loss", "Verlust", "left")}
            {TABLE_METRICS.map((key) => {
              const metric = metricByKey(key);
              return header(key, metric.unit ? `${metric.label} (${metric.unit})` : metric.label);
            })}
            <th className="px-2 py-2 text-right font-semibold text-[#8895aa]">Trials</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((study) => (
            <tr
              key={study.study_key}
              onClick={() => onSelect(study)}
              className={`cursor-pointer border-t border-white/[0.05] hover:bg-white/[0.03] ${selectedKey === study.study_key ? "bg-white/[0.05]" : ""}`}
            >
              <td className="px-2 py-1.5 text-[#c8d2e4]">{study.dataset_label}</td>
              <td className="px-2 py-1.5 text-[#c8d2e4]">{study.method_label}</td>
              <td className="px-2 py-1.5 text-[#c8d2e4]">{study.loss_label}</td>
              {TABLE_METRICS.map((key) => {
                const metric = metricByKey(key);
                return (
                  <td key={key} className="px-2 py-1.5 text-right tabular-nums text-white">
                    {formatNumber(metric.value(study, context), metric.digits)}
                  </td>
                );
              })}
              <td className="px-2 py-1.5 text-right tabular-nums text-[#98a4ba]">
                {study.trials_estimated}/{study.trials}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ page */

function EmptyState() {
  return (
    <main className="mx-auto max-w-[900px] px-5 pb-20 pt-8">
      <Link href="/" className="focus-ring text-xs font-semibold text-[#8895aa] hover:text-white">
        <ArrowLeft size={14} className="mr-1 inline" />
        Back to Trial Atlas
      </Link>
      <div className="surface mt-6 p-6">
        <div className="flex items-center gap-3">
          <Timer size={20} className="text-[#4fd5e7]" />
          <h1 className="text-xl font-semibold text-white">Timings</h1>
        </div>
        <p className="mt-3 text-sm leading-relaxed text-[#98a4ba]">
          Noch keine Messung gefunden. Die Seite liest jede <code className="text-white">timings.json</code> unter{" "}
          <code className="text-white">/mnt/f/Studien/Timings</code> (oder <code className="text-white">TIMINGS_DIR</code>).
        </p>
        <pre className="mt-4 overflow-x-auto rounded-xl bg-black/30 p-4 text-[11px] leading-relaxed text-[#c8d2e4]">
{`# Server
venv/bin/python scripts/bench_epoch_cost.py run --gpu 0 --output-dir timings/phase1 \\
    --manifest configs/final_manifests/cars_all.json configs/final_manifests/cub_all.json \\
               configs/final_manifests/in_shop_all.json configs/final_manifests/in_shop_all_10.json

# Ergebnisordner nach F:\\Studien\\Timings\\ kopieren, dann Seite neu laden`}
        </pre>
      </div>
    </main>
  );
}

export function TimingsExplorer({ snapshots }: { snapshots: TimingSnapshotEntry[] }) {
  const [snapshotId, setSnapshotId] = useState(snapshots[0]?.id ?? "");
  const [metricKey, setMetricKey] = useState<MetricKey>("epoch");
  const [layoutKey, setLayoutKey] = useState(LAYOUTS[0].key);
  const [filters, setFilters] = useState<Record<Dimension, string>>({ method: ALL, dataset: ALL, loss: ALL });
  const [selectedCell, setSelectedCell] = useState<{ row: string; col: string } | null>(null);
  const [selectedStudyKey, setSelectedStudyKey] = useState<string | null>(null);

  const entry = snapshots.find((candidate) => candidate.id === snapshotId) ?? snapshots[0];
  const snapshot = entry?.snapshot;
  const studies = useMemo(() => snapshot?.studies ?? [], [snapshot]);
  const casesById = useMemo(() => new Map((snapshot?.cases ?? []).map((item) => [item.case_id, item])), [snapshot]);
  const context = useMemo(() => makeContext(studies), [studies]);
  const layout = LAYOUTS.find((candidate) => candidate.key === layoutKey) ?? LAYOUTS[0];
  const metric = metricByKey(metricKey);

  const labels = useMemo(() => {
    const result: Record<Dimension, Record<string, string>> = { method: {}, dataset: {}, loss: {} };
    for (const study of studies) {
      for (const dimension of ["method", "dataset", "loss"] as Dimension[]) {
        result[dimension][dimensionValue(study, dimension)] = dimensionLabel(study, dimension);
      }
    }
    return result;
  }, [studies]);

  const filterValue = filters[layout.filter];
  const visible = useMemo(
    () => studies.filter((study) => filterValue === ALL || dimensionValue(study, layout.filter) === filterValue),
    [studies, filterValue, layout.filter],
  );
  const rows = useMemo(() => orderedValues(visible, layout.rows), [visible, layout.rows]);
  const cols = useMemo(() => orderedValues(visible, layout.cols), [visible, layout.cols]);

  const pivot = useMemo(() => {
    const cells = new Map<string, PivotCell>();
    for (const row of rows) {
      for (const col of cols) {
        const members = visible.filter(
          (study) => dimensionValue(study, layout.rows) === row && dimensionValue(study, layout.cols) === col,
        );
        cells.set(`${row}|${col}`, {
          row,
          col,
          studies: members,
          value: aggregate(members.map((study) => metric.value(study, context)), metric.aggregation),
        });
      }
    }
    const over = (members: TimingStudy[]) => aggregate(members.map((study) => metric.value(study, context)), metric.aggregation);
    const rowTotals = new Map(rows.map((row) => [row, over(visible.filter((study) => dimensionValue(study, layout.rows) === row))]));
    const colTotals = new Map(cols.map((col) => [col, over(visible.filter((study) => dimensionValue(study, layout.cols) === col))]));
    return { cells, rowTotals, colTotals, total: over(visible) };
  }, [rows, cols, visible, layout, metric, context]);

  if (!snapshot) return <EmptyState />;

  const activeCell = selectedCell ? pivot.cells.get(`${selectedCell.row}|${selectedCell.col}`) : undefined;
  const cellStudies = activeCell?.studies ?? [];
  const selectedStudy =
    studies.find((study) => study.study_key === selectedStudyKey) ?? (cellStudies.length === 1 ? cellStudies[0] : undefined);
  const selectedCase = selectedStudy
    ? selectedStudy.cases
        .map((id) => casesById.get(id))
        .find((item): item is TimingCase => item != null && item.trial === selectedStudy.selected_trial) ??
      selectedStudy.cases.map((id) => casesById.get(id)).find((item): item is TimingCase => item != null)
    : undefined;

  const run = snapshot.run ?? {};
  const contaminated = snapshot.cases.filter((item) => item.contaminated).length;
  const unverified = snapshot.cases.filter((item) => !item.gpu_process_visibility).length;
  const totalGpuHours = aggregate(studies.map((study) => study.estimated_gpu_hours_imputed), "sum");
  const totalObserved = aggregate(studies.map((study) => study.observed_hours), "sum");
  const trialsEstimated = studies.reduce((sum, study) => sum + study.trials_estimated, 0);
  const trialsTotal = studies.reduce((sum, study) => sum + study.trials, 0);

  const filterOptions = [
    { value: ALL, label: `Alle (${AGGREGATION_LABEL[metric.aggregation]})` },
    ...orderedValues(studies, layout.filter).map((value) => ({ value, label: labels[layout.filter][value] ?? value })),
  ];
  const filterLabel = filterValue === ALL ? ALL_LABEL[layout.filter] : (labels[layout.filter][filterValue] ?? filterValue);

  const pivotRows = (): ExportRow[] =>
    rows.map((row) => {
      const record: ExportRow = { [DIMENSION_LABEL[layout.rows]]: labels[layout.rows][row] ?? row };
      for (const col of cols) record[labels[layout.cols][col] ?? col] = pivot.cells.get(`${row}|${col}`)?.value ?? null;
      record[AGGREGATION_LABEL[metric.aggregation]] = pivot.rowTotals.get(row) ?? null;
      return record;
    });
  const pivotLatex = () =>
    toLatexTable(
      [
        { header: DIMENSION_LABEL[layout.rows], align: "l" },
        ...cols.map((col) => ({ header: labels[layout.cols][col] ?? col, align: "r" as const })),
        { header: AGGREGATION_LABEL[metric.aggregation], align: "r" },
      ],
      [
        ...rows.map((row) => [
          labels[layout.rows][row] ?? row,
          ...cols.map((col) => formatNumber(pivot.cells.get(`${row}|${col}`)?.value, metric.digits)),
          formatNumber(pivot.rowTotals.get(row), metric.digits),
        ]),
        [
          AGGREGATION_LABEL[metric.aggregation],
          ...cols.map((col) => formatNumber(pivot.colTotals.get(col), metric.digits)),
          formatNumber(pivot.total, metric.digits),
        ],
      ],
      `${metric.label}${metric.unit ? ` (${metric.unit})` : ""}, ${filterLabel}. Gemessen allein auf ${run.device_name ?? run.gpu?.name ?? "einer GPU"}.`,
      `tab:timings-${metric.key}`,
    );

  return (
    <main className="mx-auto max-w-[1180px] px-5 pb-20 pt-8">
      <Link href="/" className="focus-ring text-xs font-semibold text-[#8895aa] hover:text-white">
        <ArrowLeft size={14} className="mr-1 inline" />
        Back to Trial Atlas
      </Link>
      <header className="mt-5">
        <div className="flex flex-wrap items-center gap-3">
          <Timer size={20} className="text-[#4fd5e7]" />
          <h1 className="text-2xl font-semibold text-white">Timings</h1>
          <span className="inline-flex items-center gap-1.5 rounded-full bg-white/[0.05] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#b3bdd0]">
            <Cpu size={11} />
            {run.device_name ?? run.gpu?.name ?? "GPU unbekannt"}
            {run.torch ? ` · torch ${run.torch}` : ""}
          </span>
          <span className="rounded-full bg-white/[0.05] px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#b3bdd0]">
            Stand {snapshot.generated.slice(0, 16).replace("T", " ")}
          </span>
          {contaminated > 0 && (
            <span className="inline-flex items-center gap-1.5 rounded-full bg-[#fab219]/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#fab219]">
              <TriangleAlert size={11} />
              {contaminated} Fälle mit geteilter GPU
            </span>
          )}
        </div>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#98a4ba]">
          Die HPO-Läufe teilten sich die GPUs mit wechselnder Last, ihre Laufzeiten sind daher nicht vergleichbar. Hier ist die
          Zeit je Epoche für die ausgewählte Konfiguration jeder Studie allein auf einer GPU gemessen und mit den protokollierten
          Epochen aller Trials zu HPO-Kosten hochgerechnet.
        </p>
        {unverified > 0 && (
          <p className="mt-2 inline-flex items-center gap-1.5 text-[11px] text-[#98a4ba]">
            <Info size={12} />
            Bei {unverified} Fällen zeigte nvidia-smi keine Prozesse; fremde GPU-Nutzung war dort nicht prüfbar.
          </p>
        )}
      </header>

      <div className="mt-6 grid grid-cols-2 gap-2 md:grid-cols-4">
        <StatTile label="HPO gesamt, geschätzt" value={`${formatNumber(totalGpuHours, 1)} GPU-h`} note={`${studies.length} Studien`} />
        <StatTile label="HPO gesamt, beobachtet" value={`${formatNumber(totalObserved, 1)} h`} note="parallel unter Last" />
        <StatTile label="Trials mit Schätzung" value={`${trialsEstimated}/${trialsTotal}`} />
        <StatTile
          label="Gemessene Fälle"
          value={String(snapshot.cases.length)}
          note={run.settings ? `Auswahl ${run.settings.select}, Fenster ${run.settings.window} Epochen` : undefined}
        />
      </div>

      <div className="mt-6 flex flex-wrap items-end gap-4">
        {snapshots.length > 1 && (
          <Select
            label="Messung"
            value={entry.id}
            options={snapshots.map((candidate) => ({ value: candidate.id, label: `${candidate.id} (${candidate.snapshot.generated.slice(0, 10)})` }))}
            onChange={(value) => {
              setSnapshotId(value);
              setSelectedCell(null);
              setSelectedStudyKey(null);
            }}
          />
        )}
        <Select
          label="Kennzahl"
          value={metricKey}
          options={METRICS.map((candidate) => ({ value: candidate.key, label: candidate.unit ? `${candidate.label} (${candidate.unit})` : candidate.label }))}
          onChange={(value) => setMetricKey(value as MetricKey)}
        />
        <label className="flex flex-col gap-1">
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">Ansicht</span>
          <Toggle
            value={layoutKey}
            options={LAYOUTS.map((candidate) => ({ value: candidate.key, label: candidate.label }))}
            onChange={(value) => {
              setLayoutKey(value);
              setSelectedCell(null);
            }}
          />
        </label>
        <Select
          label={DIMENSION_LABEL[layout.filter]}
          value={filterValue}
          options={filterOptions}
          onChange={(value) => {
            setFilters((current) => ({ ...current, [layout.filter]: value }));
            setSelectedCell(null);
          }}
        />
      </div>

      <FigureCard
        className="mt-4"
        title={`${metric.label}${metric.unit ? ` (${metric.unit})` : ""} · ${filterLabel}`}
        description={metric.description}
        name={`timings-${metric.key}-${layout.key}-${filterValue === ALL ? "alle" : filterValue}`}
        rows={pivotRows}
        latex={pivotLatex}
      >
        <div className="mt-4">
          <HeatPivot
            layout={layout}
            rows={rows}
            cols={cols}
            labels={labels}
            cells={pivot.cells}
            rowTotals={pivot.rowTotals}
            colTotals={pivot.colTotals}
            total={pivot.total}
            metric={metric}
            selected={selectedCell}
            onSelect={(cell) => {
              setSelectedCell({ row: cell.row, col: cell.col });
              setSelectedStudyKey(cell.studies.length === 1 ? cell.studies[0].study_key : null);
            }}
          />
        </div>
      </FigureCard>

      {activeCell && cellStudies.length > 1 && (
        <div className="mt-4 flex flex-wrap items-center gap-2">
          <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">Studien in der Zelle</span>
          {cellStudies.map((study) => (
            <button
              key={study.study_key}
              type="button"
              onClick={() => setSelectedStudyKey(study.study_key)}
              className={`focus-ring rounded-full px-2.5 py-1 text-[11px] font-semibold transition ${
                selectedStudy?.study_key === study.study_key ? "bg-white/[0.12] text-white" : "bg-white/[0.04] text-[#98a4ba] hover:text-white"
              }`}
            >
              {study.method_label} · {study.dataset_label} · {study.loss_label}
              <span className="ml-1.5 font-normal tabular-nums text-[#7f8ba2]">{formatMetric(metric, metric.value(study, context))}</span>
            </button>
          ))}
        </div>
      )}

      {selectedStudy ? (
        <section className="surface mt-4 p-5">
          <StudyDetail study={selectedStudy} timingCase={selectedCase} context={context} />
        </section>
      ) : (
        <p className="mt-4 text-xs text-[#7f8ba2]">Eine Zelle oder Tabellenzeile wählen, um die Messung einer Studie zu sehen.</p>
      )}

      <FigureCard
        className="mt-6"
        title="Alle Studien"
        description="Klick auf eine Zeile zeigt die Messung. Spaltenköpfe sortieren."
        name="timings-studien"
        exportFigure={false}
        rows={() =>
          studies.map((study) => ({
            dataset: study.dataset_label,
            method: study.method_label,
            loss: study.loss_label,
            ...Object.fromEntries(TABLE_METRICS.map((key) => [key, metricByKey(key).value(study, context)])),
            trials: study.trials,
            trials_estimated: study.trials_estimated,
            selected_trial: study.selected_trial,
            selected_batch_size: study.selected_batch_size,
            study_key: study.study_key,
          }))
        }
      >
        <div className="mt-3">
          <StudyTable
            studies={studies}
            context={context}
            selectedKey={selectedStudy?.study_key ?? null}
            onSelect={(study) => {
              setSelectedStudyKey(study.study_key);
              setSelectedCell(null);
            }}
          />
        </div>
      </FigureCard>
    </main>
  );
}
