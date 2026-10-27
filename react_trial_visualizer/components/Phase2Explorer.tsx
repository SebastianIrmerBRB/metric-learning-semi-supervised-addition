"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, FlaskConical, Info, Radio, ShieldAlert } from "lucide-react";
import {
  CELLS,
  EFFECTS,
  fmt,
  fmtP,
  fmtSigned,
  type AblationSnapshot,
  type AblationVariant,
  type Aggregation,
  type GeometrySnapshot,
  type GeometrySpectrum,
  type TwoByTwoBlock,
} from "@/lib/phase2";
import { toLatexTable, type ExportRow } from "@/lib/export";
import { ExportMenu, FigureCard } from "@/components/ExportMenu";
import { ShareNav } from "@/components/ShareNav";

const METHOD_COLOR: Record<string, string> = {
  sup: "#8895aa",
  lp: "#4fd5e7",
  lrml: "#9b8cff",
  hoffer: "#f4c86f",
  slade: "#b8e56e",
};
const METHOD_ORDER = ["sup", "lp", "lrml", "hoffer", "slade"];
const COMBO_METHOD: Record<string, string> = { sup: "sup", iscen: "lp", lrml: "lrml", hoffer: "hoffer", slade: "slade" };

function methodOfCombo(combo: string): string {
  return COMBO_METHOD[combo.split("_")[0]] ?? combo;
}

function colorOf(method: string): string {
  return METHOD_COLOR[method] ?? "#c8d2e4";
}

type Tab = "findings" | "twoByTwo" | "variants" | "geometry";

/* ------------------------------------------------------------------ shared UI */

function Select<T extends string | number>({
  label,
  value,
  options,
  onChange,
  render,
}: {
  label: string;
  value: T;
  options: T[];
  onChange: (value: T) => void;
  render?: (value: T) => string;
}) {
  return (
    <label className="flex flex-col gap-1">
      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">{label}</span>
      <select
        className="focus-ring rounded-lg border border-white/[0.08] bg-[#0f1522] px-2.5 py-1.5 text-xs text-white"
        value={String(value)}
        onChange={(event) => {
          const raw = event.target.value;
          const next = options.find((option) => String(option) === raw);
          if (next !== undefined) onChange(next);
        }}
      >
        {options.map((option) => (
          <option key={String(option)} value={String(option)}>
            {render ? render(option) : String(option)}
          </option>
        ))}
      </select>
    </label>
  );
}

function Toggle<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
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

function unique<T>(values: T[]): T[] {
  return Array.from(new Set(values));
}

/* ------------------------------------------------------------------ charts */

interface ChartSeries {
  name: string;
  color: string;
  points: [number, number][];
  dashed?: boolean;
  marker?: number | null;
  /** Optional tooltip lines per point (same order as `points`). */
  tips?: string[][];
}

interface HoverTarget {
  px: number;
  py: number;
  color: string;
  lines: string[];
}

function formatValue(v: number): string {
  if (!Number.isFinite(v)) return "–";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 10) return v.toFixed(1);
  if (a >= 0.01 || a === 0) return v.toFixed(3);
  return v.toPrecision(2);
}

/** Nearest-point hover for an SVG chart whose viewBox is width x height. */
function useHover(width: number, height: number, targets: HoverTarget[], radius = 28) {
  const [hover, setHover] = useState<HoverTarget | null>(null);
  const onMouseMove = (event: React.MouseEvent<SVGSVGElement>) => {
    const box = event.currentTarget.getBoundingClientRect();
    if (box.width === 0 || box.height === 0) return;
    const x = ((event.clientX - box.left) / box.width) * width;
    const y = ((event.clientY - box.top) / box.height) * height;
    let best: HoverTarget | null = null;
    let bestDistance = radius * radius;
    for (const target of targets) {
      const d = (target.px - x) ** 2 + (target.py - y) ** 2;
      if (d < bestDistance) {
        bestDistance = d;
        best = target;
      }
    }
    setHover(best);
  };
  return { hover, onMouseMove, onMouseLeave: () => setHover(null) };
}

function Tooltip({ hover, width, height }: { hover: HoverTarget | null; width: number; height: number }) {
  if (!hover) return null;
  const lineHeight = 12;
  const boxWidth = Math.min(width - 8, Math.max(...hover.lines.map((l) => l.length)) * 5.6 + 14);
  const boxHeight = hover.lines.length * lineHeight + 8;
  let x = hover.px + 10;
  let y = hover.py - boxHeight - 8;
  if (x + boxWidth > width - 4) x = hover.px - boxWidth - 10;
  if (x < 4) x = 4;
  if (y < 4) y = hover.py + 10;
  if (y + boxHeight > height - 4) y = height - boxHeight - 4;
  return (
    <g pointerEvents="none">
      <circle cx={hover.px} cy={hover.py} r={5} fill="none" stroke={hover.color} strokeWidth={1.5} />
      <rect x={x} y={y} width={boxWidth} height={boxHeight} rx={4} fill="#0b101b" fillOpacity={0.96} stroke="#ffffff30" />
      {hover.lines.map((line, i) => (
        <text key={i} x={x + 7} y={y + 14 + i * lineHeight} fontSize="10" fill={i === 0 ? "#ffffff" : "#c8d2e4"} fontWeight={i === 0 ? 600 : 400}>
          {line}
        </text>
      ))}
    </g>
  );
}

function LineChart({
  series,
  xLabel,
  yLabel,
  logX = false,
  logY = false,
  zeroLine = false,
  height = 220,
  width = 520,
}: {
  series: ChartSeries[];
  xLabel: string;
  yLabel: string;
  logX?: boolean;
  logY?: boolean;
  zeroLine?: boolean;
  height?: number;
  width?: number;
}) {
  const pad = { left: 52, right: 12, top: 10, bottom: 34 };
  const tx = (x: number) => (logX ? Math.log2(x) : x);
  const ty = (y: number) => (logY ? Math.log10(Math.max(y, 1e-12)) : y);
  const all = series.flatMap((s) => s.points.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)));
  const xs = all.map(([x]) => tx(x));
  const ys = all.map(([, y]) => ty(y)).concat(zeroLine ? [0] : []);
  let [x0, x1] = all.length ? [Math.min(...xs), Math.max(...xs)] : [0, 1];
  let [y0, y1] = all.length ? [Math.min(...ys), Math.max(...ys)] : [0, 1];
  if (x0 === x1) [x0, x1] = [x0 - 1, x1 + 1];
  if (y0 === y1) [y0, y1] = [y0 - 1, y1 + 1];
  const yPad = (y1 - y0) * 0.08;
  y0 -= yPad;
  y1 += yPad;
  const sx = (x: number) => pad.left + ((tx(x) - x0) / (x1 - x0)) * (width - pad.left - pad.right);
  const sy = (y: number) => pad.top + (1 - (ty(y) - y0) / (y1 - y0)) * (height - pad.top - pad.bottom);
  const targets: HoverTarget[] = series.flatMap((s) =>
    s.points.flatMap(([x, y], i) =>
      Number.isFinite(x) && Number.isFinite(y)
        ? [{ px: sx(x), py: sy(y), color: s.color, lines: s.tips?.[i] ?? [s.name, `${xLabel}: ${formatValue(x)}`, `${yLabel}: ${formatValue(y)}`] }]
        : [],
    ),
  );
  const { hover, onMouseMove, onMouseLeave } = useHover(width, height, targets);
  if (all.length === 0) {
    return <p className="py-10 text-center text-xs text-[#7f8ba2]">Keine Daten für diese Auswahl.</p>;
  }
  const yTicks = Array.from({ length: 5 }, (_, i) => y0 + ((y1 - y0) * i) / 4);
  // Log axes label the measured x values, skipping any closer than ~30 px to the previous label.
  const xTickValues = logX
    ? unique(all.map(([x]) => x))
        .sort((a, b) => a - b)
        .reduce<number[]>((kept, v) => (kept.length === 0 || sx(v) - sx(kept[kept.length - 1]) >= 30 ? [...kept, v] : kept), [])
    : Array.from({ length: 5 }, (_, i) => x0 + ((x1 - x0) * i) / 4);
  const xSpan = logX ? 0 : x1 - x0;
  const formatXTick = (v: number) =>
    logX ? (Number.isInteger(v) ? String(v) : formatValue(v)) : xSpan >= 20 ? v.toFixed(0) : xSpan >= 2 ? v.toFixed(1) : v.toFixed(2);
  const formatTick = (v: number, log: boolean) => {
    const raw = log ? 10 ** v : v;
    return Math.abs(raw) >= 100 ? raw.toFixed(0) : Math.abs(raw) >= 1 ? raw.toFixed(1) : raw.toPrecision(2);
  };
  const hoveredName = hover?.lines[0];
  return (
    <svg
      data-export-root
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label={`${yLabel} gegen ${xLabel}`}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
    >
      {yTicks.map((v) => (
        <g key={`y${v}`}>
          <line x1={pad.left} x2={width - pad.right} y1={sy(logY ? 10 ** v : v)} y2={sy(logY ? 10 ** v : v)} stroke="#ffffff10" />
          <text x={pad.left - 6} y={sy(logY ? 10 ** v : v) + 3} textAnchor="end" fontSize="9" fill="#7f8ba2">
            {formatTick(v, logY)}
          </text>
        </g>
      ))}
      {xTickValues.map((v) => (
        <text key={`x${v}`} x={sx(v)} y={height - pad.bottom + 14} textAnchor="middle" fontSize="9" fill="#7f8ba2">
          {formatXTick(v)}
        </text>
      ))}
      {zeroLine && <line x1={pad.left} x2={width - pad.right} y1={sy(0)} y2={sy(0)} stroke="#ffffff55" />}
      {series.map((s) => {
        const pts = s.points.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y));
        if (pts.length === 0) return null;
        const d = pts.map(([x, y], i) => `${i ? "L" : "M"}${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join("");
        const dim = hoveredName != null && !(s.tips?.some((tip) => tip[0] === hoveredName) || s.name === hoveredName);
        return (
          <g key={s.name} opacity={dim ? 0.25 : 1}>
            <path d={d} fill="none" stroke={s.color} strokeWidth={1.6} strokeDasharray={s.dashed ? "4 3" : undefined} />
            {pts.length <= 12 && pts.map(([x, y], i) => <circle key={i} cx={sx(x)} cy={sy(y)} r={2.2} fill={s.color} />)}
            {s.marker != null && Number.isFinite(s.marker) && (
              <line x1={sx(s.marker)} x2={sx(s.marker)} y1={pad.top} y2={height - pad.bottom} stroke={s.color} strokeOpacity={0.35} strokeDasharray="2 3" />
            )}
          </g>
        );
      })}
      <text x={(width + pad.left) / 2} y={height - 4} textAnchor="middle" fontSize="10" fill="#98a4ba">
        {xLabel}
      </text>
      <text x={12} y={height / 2} textAnchor="middle" fontSize="10" fill="#98a4ba" transform={`rotate(-90 12 ${height / 2})`}>
        {yLabel}
      </text>
      <Tooltip hover={hover} width={width} height={height} />
    </svg>
  );
}

function ScatterChart({
  series,
  xLabel,
  yLabel,
  note,
  height = 240,
  width = 520,
}: {
  series: ChartSeries[];
  xLabel: string;
  yLabel: string;
  note?: string;
  height?: number;
  width?: number;
}) {
  const pad = { left: 52, right: 12, top: 18, bottom: 34 };
  const all = series.flatMap((s) => s.points.filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y)));
  const span = (values: number[]) => {
    if (values.length === 0) return [0, 1];
    let [lo, hi] = [Math.min(...values), Math.max(...values)];
    if (lo === hi) [lo, hi] = [lo - 1, hi + 1];
    const margin = (hi - lo) * 0.08;
    return [lo - margin, hi + margin];
  };
  const [x0, x1] = span(all.map(([x]) => x));
  const [y0, y1] = span(all.map(([, y]) => y));
  const sx = (x: number) => pad.left + ((x - x0) / (x1 - x0)) * (width - pad.left - pad.right);
  const sy = (y: number) => pad.top + (1 - (y - y0) / (y1 - y0)) * (height - pad.top - pad.bottom);
  const targets: HoverTarget[] = series.flatMap((s) =>
    s.points.flatMap(([x, y], i) =>
      Number.isFinite(x) && Number.isFinite(y)
        ? [{ px: sx(x), py: sy(y), color: s.color, lines: s.tips?.[i] ?? [s.name, `${xLabel}: ${formatValue(x)}`, `${yLabel}: ${formatValue(y)}`] }]
        : [],
    ),
  );
  const { hover, onMouseMove, onMouseLeave } = useHover(width, height, targets, 18);
  if (all.length === 0) {
    return <p className="py-10 text-center text-xs text-[#7f8ba2]">Keine Daten für diese Auswahl (Läufe noch nicht fertig?).</p>;
  }
  const ticks = (lo: number, hi: number) => Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  const label = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(2));
  // Least-squares line over all points, as in Roth et al.'s panels.
  const n = all.length;
  const mx = all.reduce((a, [x]) => a + x, 0) / n;
  const my = all.reduce((a, [, y]) => a + y, 0) / n;
  const sxx = all.reduce((a, [x]) => a + (x - mx) ** 2, 0);
  const slope = sxx > 0 ? all.reduce((a, [x, y]) => a + (x - mx) * (y - my), 0) / sxx : 0;
  return (
    <svg
      data-export-root
      viewBox={`0 0 ${width} ${height}`}
      className="w-full"
      role="img"
      aria-label={`${yLabel} gegen ${xLabel}`}
      onMouseMove={onMouseMove}
      onMouseLeave={onMouseLeave}
    >
      {ticks(y0, y1).map((v) => (
        <g key={`y${v}`}>
          <line x1={pad.left} x2={width - pad.right} y1={sy(v)} y2={sy(v)} stroke="#ffffff10" />
          <text x={pad.left - 6} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="#7f8ba2">
            {label(v)}
          </text>
        </g>
      ))}
      {ticks(x0, x1).map((v) => (
        <text key={`x${v}`} x={sx(v)} y={height - pad.bottom + 14} textAnchor="middle" fontSize="9" fill="#7f8ba2">
          {label(v)}
        </text>
      ))}
      {n >= 3 && sxx > 0 && (
        <line x1={sx(x0)} x2={sx(x1)} y1={sy(my + slope * (x0 - mx))} y2={sy(my + slope * (x1 - mx))} stroke="#ffffff55" strokeDasharray="4 3" />
      )}
      {series.map((s) =>
        s.points
          .filter(([x, y]) => Number.isFinite(x) && Number.isFinite(y))
          .map(([x, y], i) => <circle key={`${s.name}-${i}`} cx={sx(x)} cy={sy(y)} r={3.2} fill={s.color} fillOpacity={0.9} />),
      )}
      {note && (
        <text x={width - pad.right} y={12} textAnchor="end" fontSize="10" fill="#c8d2e4">
          {note}
        </text>
      )}
      <text x={(width + pad.left) / 2} y={height - 4} textAnchor="middle" fontSize="10" fill="#98a4ba">
        {xLabel}
      </text>
      <text x={12} y={height / 2} textAnchor="middle" fontSize="10" fill="#98a4ba" transform={`rotate(-90 12 ${height / 2})`}>
        {yLabel}
      </text>
      <Tooltip hover={hover} width={width} height={height} />
    </svg>
  );
}

/** Horizontal bars for signed values in [-1, 1], e.g. correlations; each row may carry a second, lighter bar. */
function CorrelationBars({
  rows,
  primaryLabel,
  secondaryLabel,
  highlight,
  onSelect,
}: {
  rows: { key: string; label: string; primary: number | null; secondary: number | null; n: number }[];
  primaryLabel: string;
  secondaryLabel: string;
  highlight?: string;
  onSelect?: (key: string) => void;
}) {
  const width = 1080;
  const rowHeight = 20;
  const labelWidth = 230;
  const height = rows.length * rowHeight + 30;
  const mid = labelWidth + (width - labelWidth - 20) / 2;
  const half = (width - labelWidth - 20) / 2;
  const targets: HoverTarget[] = rows.map((row, i) => ({
    px: mid + (row.primary ?? 0) * half,
    py: 22 + i * rowHeight + 6,
    color: "#ffffff",
    lines: [row.label, `${primaryLabel}: ${fmt(row.primary)}`, `${secondaryLabel}: ${fmt(row.secondary)}`, `n = ${row.n}`],
  }));
  const { hover, onMouseMove, onMouseLeave } = useHover(width, height, targets, 400);
  return (
    <svg data-export-root viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Korrelationen" onMouseMove={onMouseMove} onMouseLeave={onMouseLeave}>
      {[-1, -0.5, 0, 0.5, 1].map((v) => (
        <g key={v}>
          <line x1={mid + v * half} x2={mid + v * half} y1={14} y2={height - 6} stroke={v === 0 ? "#ffffff55" : "#ffffff12"} />
          <text x={mid + v * half} y={10} textAnchor="middle" fontSize="9" fill="#7f8ba2">
            {v}
          </text>
        </g>
      ))}
      {rows.map((row, i) => {
        const y = 22 + i * rowHeight;
        const bar = (value: number | null, offset: number, color: string, thickness: number) =>
          value == null ? null : (
            <rect
              x={Math.min(mid, mid + value * half)}
              y={y + offset}
              width={Math.abs(value * half)}
              height={thickness}
              fill={color}
              rx={1.5}
            />
          );
        const selected = row.key === highlight;
        return (
          <g key={row.key} onClick={() => onSelect?.(row.key)} style={{ cursor: onSelect ? "pointer" : undefined }}>
            <rect x={0} y={y - 3} width={width} height={rowHeight} fill={selected ? "#ffffff0d" : "transparent"} />
            <text x={labelWidth - 8} y={y + 10} textAnchor="end" fontSize="10" fill={selected ? "#ffffff" : "#c8d2e4"} fontWeight={selected ? 600 : 400}>
              {row.label}
            </text>
            {bar(row.primary, 0, (row.primary ?? 0) >= 0 ? "#4fd5e7" : "#f08a7e", 8)}
            {bar(row.secondary, 9, (row.secondary ?? 0) >= 0 ? "#4fd5e766" : "#f08a7e66", 4)}
          </g>
        );
      })}
      <Tooltip hover={hover} width={width} height={height} />
    </svg>
  );
}

function ChipFilter({
  label,
  options,
  selected,
  onChange,
  render,
  colorOf: optionColor,
}: {
  label: string;
  options: string[];
  selected: string[];
  onChange: (next: string[]) => void;
  render?: (value: string) => string;
  colorOf?: (value: string) => string;
}) {
  const active = (option: string) => selected.length === 0 || selected.includes(option);
  const toggle = (option: string) => {
    const current = selected.length === 0 ? options : selected;
    const next = current.includes(option) ? current.filter((o) => o !== option) : [...current, option];
    onChange(next.length === options.length ? [] : next);
  };
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] font-semibold uppercase tracking-[0.12em] text-[#7f8ba2]">{label}</span>
      <div className="flex flex-wrap items-center gap-1">
        {options.map((option) => (
          <button
            key={option}
            type="button"
            onClick={() => toggle(option)}
            onDoubleClick={() => onChange([option])}
            title="Klick: an/aus · Doppelklick: nur diese"
            className={`focus-ring flex items-center gap-1.5 rounded-md border px-2 py-1 text-[11px] font-semibold transition ${
              active(option) ? "border-white/[0.14] bg-white/[0.08] text-white" : "border-white/[0.05] text-[#5f6b82]"
            }`}
          >
            {optionColor && <span className="h-2 w-2 rounded-full" style={{ background: optionColor(option), opacity: active(option) ? 1 : 0.3 }} />}
            {render ? render(option) : option}
          </button>
        ))}
        {selected.length > 0 && (
          <button type="button" onClick={() => onChange([])} className="focus-ring px-1.5 text-[11px] text-[#8895aa] hover:text-white">
            alle
          </button>
        )}
      </div>
    </div>
  );
}

function Legend({ items }: { items: { name: string; color: string; dashed?: boolean }[] }) {
  return (
    <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
      {items.map((item) => (
        <span key={item.name} className="flex items-center gap-1.5 text-[11px] text-[#c8d2e4]">
          <svg width="18" height="6">
            <line x1="0" x2="18" y1="3" y2="3" stroke={item.color} strokeWidth="2" strokeDasharray={item.dashed ? "4 3" : undefined} />
          </svg>
          {item.name}
        </span>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ 2x2 */

function EffectBar({ label, hint, value, p, n, scale }: { label: string; hint: string; value?: number; p?: number | null; n: number; scale: number }) {
  const width = value == null ? 0 : Math.min(50, (Math.abs(value) / scale) * 50);
  const positive = (value ?? 0) >= 0;
  return (
    <div className="grid grid-cols-[150px_1fr_110px] items-center gap-3 py-1" title={hint}>
      <span className="text-[11px] text-[#c8d2e4]">{label}</span>
      <div className="relative h-3 rounded bg-white/[0.04]">
        <div className="absolute left-1/2 top-0 h-3 w-px bg-white/30" />
        {value != null && (
          <div
            className={`absolute top-0.5 h-2 rounded ${positive ? "bg-[#4fd5e7]" : "bg-[#f28b82]"}`}
            style={positive ? { left: "50%", width: `${width}%` } : { right: "50%", width: `${width}%` }}
          />
        )}
      </div>
      <span className="text-right text-[11px] tabular-nums text-white">
        {fmtSigned(value)} <span className="text-[#7f8ba2]">{n > 1 ? fmtP(p) : n === 1 ? "1 Seed" : ""}</span>
      </span>
    </div>
  );
}

const CELL_LABEL: Record<string, string> = {
  "sup_hp/no_data": "überwachte HPs, ohne Daten",
  "sup_hp/with_data": "überwachte HPs, mit Daten",
  "ssl_hp/no_data": "SSL-HPs, ohne Daten",
  "ssl_hp/with_data": "SSL-HPs, mit Daten",
};

function blockRows(block: TwoByTwoBlock, aggregation: Aggregation, labels: AblationSnapshot["labels"]): ExportRow[] {
  const data = block[aggregation];
  if (!data) return [];
  const base = { dataset: block.dataset, loss: block.loss, k: block.k, method: labels.method[block.method] ?? block.method, aggregation };
  return [
    ...CELLS.map((c) => {
      const v = data.cells[c.key];
      return { ...base, kind: "cell", name: CELL_LABEL[c.key], key: c.key, mean: v?.mean, sd: v?.sd, n: v?.n };
    }),
    ...EFFECTS.map((e) => {
      const d = data[e.key];
      return { ...base, kind: "effect", name: e.label, key: e.key, mean: d?.mean, sd: d?.sd, se: d?.se, n: d?.n, p: d?.p, hint: e.hint };
    }),
  ];
}

function twoByTwoLatex(blocks: TwoByTwoBlock[], aggregation: Aggregation, labels: AblationSnapshot["labels"], metricLabel: string): string {
  const cellText = (v?: { mean?: number; sd?: number }) => (v?.mean == null ? "–" : v.sd != null ? `${fmt(v.mean)} ± ${fmt(v.sd)}` : fmt(v.mean));
  const effectText = (d?: { mean?: number; p?: number | null; n: number }) =>
    d?.mean == null ? "–" : `${fmtSigned(d.mean)}${d.n > 1 && d.p != null ? ` (${fmtP(d.p)})` : ""}`;
  const first = blocks[0];
  return toLatexTable(
    [
      { header: "Verfahren" },
      ...CELLS.map((c) => ({ header: CELL_LABEL[c.key], align: "r" as const })),
      ...["total", "data_effect_ssl_hp", "hp_effect_no_data", "interaction"].map((key) => ({
        header: EFFECTS.find((e) => e.key === key)?.label ?? key,
        align: "r" as const,
      })),
    ],
    blocks.map((block) => {
      const data = block[aggregation];
      return [
        labels.method[block.method] ?? block.method,
        ...CELLS.map((c) => cellText(data?.cells[c.key])),
        ...(["total", "data_effect_ssl_hp", "hp_effect_no_data", "interaction"] as const).map((key) => effectText(data?.[key])),
      ];
    }),
    `2×2-Ablation (${labels.dataset[first.dataset] ?? first.dataset}, ${labels.loss[first.loss] ?? first.loss}, k=${first.k}): Test-${metricLabel} in %, Mittel ± SD über Seeds; Effekte gepaart über Seeds.`,
    `tab:2x2-${first.dataset}-${first.loss}-k${first.k}`,
  );
}

/** Interaction plot (cells) and effect bars with 95% intervals, as one exportable figure. */
function TwoByTwoFigure({ block, aggregation, metricLabel }: { block: TwoByTwoBlock; aggregation: Aggregation; metricLabel: string }) {
  const data = block[aggregation];
  if (!data) return null;
  const width = 640;
  const height = 230;
  const left = { x0: 52, x1: 270, y0: 16, y1: 190 };
  const cells = CELLS.map((c) => data.cells[c.key]).filter(Boolean);
  const lows = cells.map((v) => v.mean - (v.sd ?? 0));
  const highs = cells.map((v) => v.mean + (v.sd ?? 0));
  let [lo, hi] = [Math.min(...lows), Math.max(...highs)];
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return null;
  if (lo === hi) [lo, hi] = [lo - 1, hi + 1];
  const margin = (hi - lo) * 0.12;
  lo -= margin;
  hi += margin;
  const sx = (col: "no_data" | "with_data") => (col === "no_data" ? left.x0 + 40 : left.x1 - 40);
  const sy = (v: number) => left.y0 + (1 - (v - lo) / (hi - lo)) * (left.y1 - left.y0);
  const rows = [
    { row: "sup_hp" as const, label: "überwachte HPs", color: "#8895aa", dashed: true },
    { row: "ssl_hp" as const, label: "SSL-HPs", color: colorOf(block.method), dashed: false },
  ];
  const effects = EFFECTS.map((e) => ({ ...e, d: data[e.key] }));
  const extent = Math.max(0.5, ...effects.map(({ d }) => Math.abs(d?.mean ?? 0) + 1.96 * (d?.se ?? 0)));
  const bars = { x0: 420, x1: width - 14, y0: 16, row: 28 };
  const bx = (v: number) => bars.x0 + ((v + extent) / (2 * extent)) * (bars.x1 - bars.x0);
  const ticks = Array.from({ length: 5 }, (_, i) => lo + ((hi - lo) * i) / 4);
  return (
    <svg data-export-root viewBox={`0 0 ${width} ${height}`} className="mt-4 w-full" role="img" aria-label="2×2-Interaktionsdiagramm">
      {ticks.map((v) => (
        <g key={v}>
          <line x1={left.x0} x2={left.x1} y1={sy(v)} y2={sy(v)} stroke="#ffffff10" />
          <text x={left.x0 - 6} y={sy(v) + 3} textAnchor="end" fontSize="9" fill="#7f8ba2">
            {v.toFixed(1)}
          </text>
        </g>
      ))}
      {rows.map(({ row, label, color, dashed }) => {
        const a = data.cells[`${row}/no_data`];
        const b = data.cells[`${row}/with_data`];
        if (!a || !b) return null;
        return (
          <g key={row}>
            <line x1={sx("no_data")} x2={sx("with_data")} y1={sy(a.mean)} y2={sy(b.mean)} stroke={color} strokeWidth={1.8} strokeDasharray={dashed ? "4 3" : undefined} />
            {[
              ["no_data", a],
              ["with_data", b],
            ].map(([col, v]) => {
              const value = v as typeof a;
              const x = sx(col as "no_data" | "with_data");
              return (
                <g key={col as string}>
                  {value.sd != null && <line x1={x} x2={x} y1={sy(value.mean - value.sd)} y2={sy(value.mean + value.sd)} stroke={color} strokeWidth={1} />}
                  <circle cx={x} cy={sy(value.mean)} r={3.5} fill={color} />
                </g>
              );
            })}
            <text x={sx("with_data") + 8} y={sy(b.mean) + 3} fontSize="9" fill={color}>
              {label}
            </text>
          </g>
        );
      })}
      <text x={sx("no_data")} y={left.y1 + 14} textAnchor="middle" fontSize="9" fill="#98a4ba">
        ohne ungelabelte Daten
      </text>
      <text x={sx("with_data")} y={left.y1 + 14} textAnchor="middle" fontSize="9" fill="#98a4ba">
        mit ungelabelten Daten
      </text>
      <text x={12} y={(left.y0 + left.y1) / 2} textAnchor="middle" fontSize="10" fill="#98a4ba" transform={`rotate(-90 12 ${(left.y0 + left.y1) / 2})`}>
        Test-{metricLabel}
      </text>
      <line x1={bx(0)} x2={bx(0)} y1={bars.y0 - 6} y2={bars.y0 + bars.row * effects.length - 8} stroke="#ffffff55" />
      {effects.map(({ key, label, d }, i) => {
        const y = bars.y0 + i * bars.row + 6;
        const mean = d?.mean;
        return (
          <g key={key}>
            <text x={bars.x0 - 8} y={y + 3} textAnchor="end" fontSize="9" fill="#c8d2e4">
              {label}
            </text>
            {mean != null && (
              <>
                <rect
                  x={Math.min(bx(0), bx(mean))}
                  y={y - 5}
                  width={Math.abs(bx(mean) - bx(0))}
                  height={10}
                  rx={2}
                  fill={mean >= 0 ? "#4fd5e7" : "#f28b82"}
                  fillOpacity={0.8}
                />
                {d?.se != null && d.n > 1 && (
                  <line x1={bx(mean - 1.96 * d.se)} x2={bx(mean + 1.96 * d.se)} y1={y} y2={y} stroke="#ffffff" strokeWidth={1} />
                )}
                <text x={bars.x1} y={y + 14} textAnchor="end" fontSize="8" fill="#7f8ba2">
                  {fmtSigned(mean)} {d && d.n > 1 ? fmtP(d.p) : ""}
                </text>
              </>
            )}
          </g>
        );
      })}
      <text x={(bars.x0 + bars.x1) / 2} y={height - 4} textAnchor="middle" fontSize="9" fill="#98a4ba">
        Effekt (Prozentpunkte), Linie: 95 %-Intervall über Seeds
      </text>
    </svg>
  );
}

function TwoByTwoCard({
  block,
  aggregation,
  labels,
  metricLabel,
}: {
  block: TwoByTwoBlock;
  aggregation: Aggregation;
  labels: AblationSnapshot["labels"];
  metricLabel: string;
}) {
  const figure = useRef<HTMLDivElement>(null);
  const data = block[aggregation];
  if (!data) return null;
  const scale = Math.max(0.5, ...EFFECTS.map((e) => Math.abs(data[e.key]?.mean ?? 0)));
  const cell = (key: string) => data.cells[key];
  const name = `2x2-${block.dataset}-${block.loss}-k${block.k}-${block.method}-${aggregation}`;
  return (
    <section className="rounded-2xl border border-white/[0.07] bg-white/[0.02] p-5">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <h3 className="flex items-center gap-2 text-sm font-semibold text-white">
          <span className="h-2.5 w-2.5 rounded-full" style={{ background: colorOf(block.method) }} />
          {labels.method[block.method] ?? block.method}
          <span className="text-xs font-normal text-[#7f8ba2]">
            · {labels.loss[block.loss] ?? block.loss} · k={block.k}
          </span>
        </h3>
        <ExportMenu
          name={name}
          figure={figure}
          rows={() => blockRows(block, aggregation, labels)}
          latex={() => twoByTwoLatex([block], aggregation, labels, metricLabel)}
        />
      </div>
      <table className="mt-4 w-full text-xs">
        <thead>
          <tr className="text-[10px] uppercase tracking-[0.1em] text-[#7f8ba2]">
            <th className="pb-2 text-left font-semibold" />
            <th className="pb-2 text-right font-semibold">ohne ungelabelte Daten</th>
            <th className="pb-2 text-right font-semibold">mit ungelabelten Daten</th>
          </tr>
        </thead>
        <tbody>
          {(["sup_hp", "ssl_hp"] as const).map((row) => (
            <tr key={row} className="border-t border-white/[0.06]">
              <td className="py-2 text-[#c8d2e4]">{row === "sup_hp" ? "Überwachte HPs" : "SSL-HPs"}</td>
              {CELLS.filter((c) => c.row === row).map((c) => {
                const v = cell(c.key);
                return (
                  <td key={c.key} className="py-2 text-right tabular-nums text-white">
                    {fmt(v?.mean)}
                    {v?.sd != null && <span className="text-[#7f8ba2]"> ± {fmt(v.sd)}</span>}
                    {c.key === "sup_hp/no_data" && <span className="ml-1 text-[10px] text-[#7f8ba2]">(überwacht)</span>}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4">
        {EFFECTS.map((effect) => {
          const d = data[effect.key];
          return <EffectBar key={effect.key} label={effect.label} hint={effect.hint} value={d?.mean} p={d?.p} n={d?.n ?? 0} scale={scale} />;
        })}
      </div>
      <div ref={figure}>
        <TwoByTwoFigure block={block} aggregation={aggregation} metricLabel={metricLabel} />
      </div>
    </section>
  );
}

function TwoByTwoView({ snapshot, dataset, k, loss, aggregation }: { snapshot: AblationSnapshot; dataset: string; k: number; loss: string; aggregation: Aggregation }) {
  const blocks = snapshot.twoByTwo
    .filter((b) => b.dataset === dataset && b.k === k && b.loss === loss)
    .sort((a, b) => METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method));
  if (blocks.length === 0) {
    return <p className="mt-8 text-sm text-[#98a4ba]">Für diese Auswahl ist noch keine vollständige 2×2-Tabelle vorhanden.</p>;
  }
  return (
    <div className="mt-6">
      <div className="mb-3 flex items-center justify-end gap-2 text-[11px] text-[#7f8ba2]">
        Alle Verfahren dieser Auswahl als eine Tabelle:
        <ExportMenu
          name={`2x2-${dataset}-${loss}-k${k}-${aggregation}`}
          rows={() => blocks.flatMap((block) => blockRows(block, aggregation, snapshot.labels))}
          latex={() => twoByTwoLatex(blocks, aggregation, snapshot.labels, snapshot.metricLabel)}
        />
      </div>
      <div className="grid gap-4 lg:grid-cols-2">
        {blocks.map((block) => (
          <TwoByTwoCard key={`${block.method}`} block={block} aggregation={aggregation} labels={snapshot.labels} metricLabel={snapshot.metricLabel} />
        ))}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ variants */

function DiffCell({ diff }: { diff?: { mean?: number; p?: number | null; n: number } }) {
  if (!diff || diff.n === 0) return <td className="px-3 py-2 text-right text-[#596780]">–</td>;
  const tone = (diff.mean ?? 0) > 0 ? "text-[#7fe3f0]" : (diff.mean ?? 0) < 0 ? "text-[#f5a39c]" : "text-white";
  return (
    <td className={`px-3 py-2 text-right tabular-nums ${tone}`}>
      {fmtSigned(diff.mean)}
      <span className="ml-1 text-[10px] text-[#7f8ba2]">{diff.n > 1 ? fmtP(diff.p) : ""}</span>
    </td>
  );
}

function VariantsView({ snapshot, dataset, k, loss, aggregation }: { snapshot: AblationSnapshot; dataset: string; k: number; loss: string; aggregation: Aggregation }) {
  const rows = snapshot.variants
    .filter((v) => v.dataset === dataset && v.k === k && v.loss === loss)
    .sort((a, b) => METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method) || a.group.localeCompare(b.group) || a.label_de.localeCompare(b.label_de));
  const value = (v: AblationVariant) => v[aggregation];
  const baselineDiff = (v: AblationVariant) => (aggregation === "mean" ? v.mean_vs_baseline : v.concat_vs_baseline);
  const supervisedDiff = (v: AblationVariant) => (aggregation === "mean" ? v.mean_vs_supervised : v.concat_vs_supervised);
  const diffText = (d?: { mean?: number; p?: number | null; n: number }) =>
    !d || d.n === 0 ? "–" : `${fmtSigned(d.mean)}${d.n > 1 && d.p != null ? ` (${fmtP(d.p)})` : ""}`;
  return (
    <>
    <div className="mt-6 flex items-center justify-end gap-2 text-[11px] text-[#7f8ba2]">
      Tabelle exportieren:
      <ExportMenu
        name={`ablationen-${dataset}-${loss}-k${k}-${aggregation}`}
        rows={() =>
          rows.map((v) => ({
            dataset: v.dataset, loss: v.loss, k: v.k, method: snapshot.labels.method[v.method] ?? v.method, variant: v.variant || "baseline",
            label: v.label_de, group: v.group, uses_hidden_labels: v.uses_hidden_labels, mean: value(v)?.mean, sd: value(v)?.sd, seeds: value(v)?.n,
            delta_baseline: baselineDiff(v)?.mean, p_baseline: baselineDiff(v)?.p, delta_supervised: supervisedDiff(v)?.mean, p_supervised: supervisedDiff(v)?.p,
          }))
        }
        latex={() =>
          toLatexTable(
            [
              { header: "Verfahren" },
              { header: "Ablation" },
              { header: snapshot.metricLabel, align: "r" },
              { header: "Seeds", align: "r" },
              { header: "Δ Baseline", align: "r" },
              { header: "Δ überwacht", align: "r" },
            ],
            rows.map((v) => {
              const summary = value(v);
              return [
                snapshot.labels.method[v.method] ?? v.method,
                v.label_de + (v.uses_hidden_labels ? " (verborgene Labels)" : ""),
                summary ? `${fmt(summary.mean)}${summary.sd != null ? ` ± ${fmt(summary.sd)}` : ""}` : "–",
                String(summary?.n ?? 0),
                diffText(baselineDiff(v)),
                diffText(supervisedDiff(v)),
              ];
            }),
            `Ablationen (${snapshot.labels.dataset[dataset] ?? dataset}, ${snapshot.labels.loss[loss] ?? loss}, k=${k}): Test-${snapshot.metricLabel}, Mittel ± SD über Seeds; Differenzen gepaart über Seeds.`,
            `tab:ablationen-${dataset}-${loss}-k${k}`,
          )
        }
      />
    </div>
    <div className="mt-2 overflow-x-auto rounded-2xl border border-white/[0.07]">
      <table className="w-full min-w-[820px] text-xs">
        <thead className="bg-white/[0.03] text-[10px] uppercase tracking-[0.1em] text-[#7f8ba2]">
          <tr>
            <th className="px-3 py-2 text-left">Verfahren</th>
            <th className="px-3 py-2 text-left">Ablation</th>
            <th className="px-3 py-2 text-left">Gruppe</th>
            <th className="px-3 py-2 text-right">{snapshot.metricLabel}</th>
            <th className="px-3 py-2 text-right">Seeds</th>
            <th className="px-3 py-2 text-right">Δ Baseline</th>
            <th className="px-3 py-2 text-right">Δ überwacht</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((v) => {
            const summary = value(v);
            return (
              <tr key={`${v.method}-${v.variant}`} className="border-t border-white/[0.06]" title={v.description}>
                <td className="px-3 py-2">
                  <span className="flex items-center gap-2 text-white">
                    <span className="h-2 w-2 rounded-full" style={{ background: colorOf(v.method) }} />
                    {snapshot.labels.method[v.method] ?? v.method}
                  </span>
                </td>
                <td className="px-3 py-2 text-white">
                  {v.label_de}
                  <span className="ml-2 font-mono text-[10px] text-[#596780]">{v.variant || "baseline"}</span>
                </td>
                <td className="px-3 py-2">
                  <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${v.uses_hidden_labels ? "bg-[#f28b82]/10 text-[#f5a39c]" : "bg-white/[0.05] text-[#98a4ba]"}`}>
                    {v.uses_hidden_labels && <ShieldAlert size={10} className="mr-1 inline" />}
                    {v.group}
                  </span>
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-white">
                  {fmt(summary?.mean)}
                  {summary?.sd != null && <span className="text-[#7f8ba2]"> ± {fmt(summary.sd)}</span>}
                </td>
                <td className="px-3 py-2 text-right tabular-nums text-[#98a4ba]">{summary?.n ?? 0}</td>
                <DiffCell diff={aggregation === "mean" ? v.mean_vs_baseline : v.concat_vs_baseline} />
                <DiffCell diff={aggregation === "mean" ? v.mean_vs_supervised : v.concat_vs_supervised} />
              </tr>
            );
          })}
        </tbody>
      </table>
      <p className="border-t border-white/[0.06] px-3 py-2 text-[11px] text-[#7f8ba2]">
        <ShieldAlert size={11} className="mr-1 inline text-[#f5a39c]" />
        Rot markierte Varianten nutzen verborgene Labels: Diagnose, nie als SSL-Ergebnis berichten.
      </p>
    </div>
    </>
  );
}

/* ------------------------------------------------------------------ geometry */

function pearson(xs: number[], ys: number[]): number | null {
  const n = xs.length;
  if (n < 3) return null;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let sxy = 0;
  let sxx = 0;
  let syy = 0;
  for (let i = 0; i < n; i += 1) {
    sxy += (xs[i] - mx) * (ys[i] - my);
    sxx += (xs[i] - mx) ** 2;
    syy += (ys[i] - my) ** 2;
  }
  return sxx > 0 && syy > 0 ? sxy / Math.sqrt(sxx * syy) : null;
}

function ranks(values: number[]): number[] {
  const order = values.map((v, i) => [v, i] as [number, number]).sort((a, b) => a[0] - b[0]);
  const out = new Array<number>(values.length);
  for (let i = 0; i < order.length; ) {
    let j = i;
    while (j + 1 < order.length && order[j + 1][0] === order[i][0]) j += 1;
    for (let t = i; t <= j; t += 1) out[order[t][1]] = (i + j) / 2;
    i = j + 1;
  }
  return out;
}

function seriesRows(series: ChartSeries[], x: string, y: string): ExportRow[] {
  return series.flatMap((s) => s.points.map(([px, py]) => ({ series: s.name, [x]: px, [y]: Number.isFinite(py) ? py : null })));
}

const LOSS_COLOR: Record<string, string> = {
  arc: "#f08a7e",
  cir: "#4fd5e7",
  ms: "#b8e56e",
  nt: "#f4c86f",
  prx: "#9b8cff",
};
const LOSS_LABEL: Record<string, string> = { arc: "ArcFace", cir: "Circle", ms: "Multi-Similarity", nt: "NT-Xent", prx: "Proxy-Anchor" };
const METHOD_LABEL: Record<string, string> = { sup: "Überwacht", lp: "LP", lrml: "LRML", hoffer: "Hoffer", slade: "SLADE" };
const K_COLOR = ["#8fb8ff", "#4fd5e7", "#b8e56e", "#f4c86f", "#f08a7e", "#9b8cff"];

function lossOfCombo(combo: string): string {
  return combo.split("_")[1] ?? "";
}

function comboLabel(combo: string): string {
  const method = methodOfCombo(combo);
  const loss = lossOfCombo(combo);
  return `${METHOD_LABEL[method] ?? method} · ${LOSS_LABEL[loss] ?? loss}`;
}

type ColorBy = "method" | "loss" | "k";

/** Pearson after removing each k-group's mean from x and y: the association within a label budget. */
function withinGroupPearson(xs: number[], ys: number[], groups: number[]): number | null {
  const byGroup = new Map<number, number[]>();
  groups.forEach((g, i) => byGroup.set(g, [...(byGroup.get(g) ?? []), i]));
  const rx: number[] = [];
  const ry: number[] = [];
  for (const indices of byGroup.values()) {
    if (indices.length < 2) continue;
    const mx = indices.reduce((a, i) => a + xs[i], 0) / indices.length;
    const my = indices.reduce((a, i) => a + ys[i], 0) / indices.length;
    for (const i of indices) {
      rx.push(xs[i] - mx);
      ry.push(ys[i] - my);
    }
  }
  return pearson(rx, ry);
}

function GeometryView({ snapshot }: { snapshot: GeometrySnapshot }) {
  const roots = unique(snapshot.curves.map((c) => c.root)).sort();
  const [rootChoice, setRoot] = useState(roots[0] ?? "");
  const root = roots.includes(rootChoice) ? rootChoice : (roots[0] ?? "");
  const ks = unique(snapshot.curves.filter((c) => c.root === root).map((c) => c.k)).sort((a, b) => a - b);
  const [k, setK] = useState<number>(ks[0] ?? 0);
  // Roth et al. measure on the embedded training data, so that split opens first.
  const [split, setSplit] = useState(snapshot.splits.includes("train") ? "train" : snapshot.splits[0]);
  const [metric, setMetric] = useState(snapshot.metrics[0]);
  const [dropFirst, setDropFirst] = useState<"all" | "drop">("drop");
  const [spectrumKind, setSpectrumKind] = useState<"roth" | "centered">(snapshot.spectraRoth?.length ? "roth" : "centered");
  const [corrScope, setCorrScope] = useState<"all" | "k">("all");
  const [colorBy, setColorBy] = useState<ColorBy>("method");
  const [methodSel, setMethodSel] = useState<string[]>([]);
  const [lossSel, setLossSel] = useState<string[]>([]);
  const targets = snapshot.targets ?? { r1: "R@1" };
  const [targetChoice, setTarget] = useState<string>("map_r" in targets ? "map_r" : Object.keys(targets)[0]);
  const target = targetChoice in targets ? targetChoice : Object.keys(targets)[0];
  const targetLabel = `Test-${targets[target] ?? target}`;
  const kValue = ks.includes(k) ? k : ks[0];
  const splitLabel = snapshot.splitLabels[split] ?? split;
  const metricLabel = snapshot.metricLabels[metric] ?? metric;
  const lossFilterSlug = lossSel.length ? `-${lossSel.join("+")}` : "";
  const methodFilterSlug = methodSel.length ? `-${methodSel.join("+")}` : "";
  const slug = `${root}-${split}-k${kValue}${methodFilterSlug}${lossFilterSlug}`;

  const methodsAvailable = unique(snapshot.curves.filter((c) => c.root === root).map((c) => c.method)).sort(
    (a, b) => METHOD_ORDER.indexOf(a) - METHOD_ORDER.indexOf(b),
  );
  const lossesAvailable = unique(snapshot.curves.filter((c) => c.root === root).map((c) => c.loss)).sort();
  const keep = (method: string, loss: string) =>
    (methodSel.length === 0 || methodSel.includes(method)) && (lossSel.length === 0 || lossSel.includes(loss));
  const kIndex = (value: number) => Math.max(0, ks.indexOf(value));
  const colorFor = (method: string, loss: string, kk: number) =>
    colorBy === "loss" ? (LOSS_COLOR[loss] ?? "#c8d2e4") : colorBy === "k" ? K_COLOR[kIndex(kk) % K_COLOR.length] : colorOf(method);

  const curves = snapshot.curves
    .filter((c) => c.root === root && c.k === kValue && c.split === split && keep(c.method, c.loss))
    .sort((a, b) => METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method) || a.loss.localeCompare(b.loss));
  const curveSeries: ChartSeries[] = curves.map((c) => {
    const name = comboLabel(c.combo);
    const values = c.epochs.map((_, i) => c.series[metric]?.[i] ?? Number.NaN);
    return {
      name,
      color: colorFor(c.method, c.loss, c.k),
      points: c.epochs.map((e, i) => [e, values[i]] as [number, number]),
      marker: c.selectedEpoch,
      tips: c.epochs.map((e, i) => [
        name,
        `Epoche ${e}${c.selectedEpoch != null && Math.abs(e - c.selectedEpoch) < 2.5 ? " (≈ gewählte Epoche)" : ""}`,
        `${metricLabel}: ${formatValue(values[i])}`,
        `gewählte Epoche: ${c.selectedEpoch != null ? c.selectedEpoch.toFixed(1) : "–"} · ${c.folds} Folds`,
      ]),
    };
  });
  const backbone = curves.find((c) => c.backbone[metric] != null);
  if (backbone && curves.length) {
    const epochs = curves.flatMap((c) => c.epochs);
    const value = backbone.backbone[metric] as number;
    curveSeries.push({
      name: "Backbone",
      color: "#ffffff",
      dashed: true,
      points: [[Math.min(...epochs), value], [Math.max(...epochs), value]],
      tips: [["Backbone (eingefroren, ohne Kopf)", `${metricLabel}: ${formatValue(value)}`], ["Backbone (eingefroren, ohne Kopf)", `${metricLabel}: ${formatValue(value)}`]],
    });
  }

  // One line per method: supervised minus SSL at the chosen epoch, averaged over the selected losses at each k.
  const hypothesisRows = snapshot.hypothesis.filter((h) => h.root === root && h.split === split && keep(h.method, String(h.loss)));
  const hypothesisSeries: ChartSeries[] = unique(hypothesisRows.map((h) => h.method))
    .sort((a, b) => METHOD_ORDER.indexOf(a) - METHOD_ORDER.indexOf(b))
    .map((method) => {
      const perK = unique(hypothesisRows.filter((h) => h.method === method).map((h) => h.k)).sort((a, b) => a - b);
      const points: [number, number][] = [];
      const tips: string[][] = [];
      for (const kk of perK) {
        const rows = hypothesisRows.filter((h) => h.method === method && h.k === kk);
        const values = rows.map((h) => Number(h[`sup_minus_ssl_${metric}`])).filter((v) => Number.isFinite(v));
        const mean = values.length ? values.reduce((a, b) => a + b, 0) / values.length : Number.NaN;
        points.push([kk, mean]);
        tips.push([
          `${METHOD_LABEL[method] ?? method}, k=${kk}`,
          `Δ ${metricLabel} (überwacht − SSL): ${formatValue(mean)}`,
          `Mittel über ${values.length} Loss${values.length === 1 ? "" : "es"}: ${rows
            .map((h) => `${LOSS_LABEL[String(h.loss)] ?? h.loss} ${formatValue(Number(h[`sup_minus_ssl_${metric}`]))}`)
            .join(", ")}`,
        ]);
      }
      return { name: METHOD_LABEL[method] ?? method, color: colorOf(method), points, tips };
    });

  const spectrumSeriesOf = (source: GeometrySpectrum[] | undefined, renormalize: boolean): ChartSeries[] =>
    (source ?? [])
      .filter(
        (s) =>
          s.root === root &&
          s.split === split &&
          (s.combo === "backbone" || (s.k === kValue && keep(methodOfCombo(s.combo), lossOfCombo(s.combo)))),
      )
      .map((s) => {
        const shares = renormalize && dropFirst === "drop" ? s.shares.slice(1) : s.shares;
        const total = shares.reduce((acc, v) => acc + v, 0) || 1;
        const offset = renormalize && dropFirst === "drop" ? 2 : 1;
        const name = s.combo === "backbone" ? "Backbone" : comboLabel(s.combo);
        const points = shares.map((v, i) => [i + offset, renormalize && dropFirst === "drop" ? v / total : v] as [number, number]);
        return {
          name,
          color: s.combo === "backbone" ? "#ffffff" : colorFor(methodOfCombo(s.combo), lossOfCombo(s.combo), s.k),
          dashed: s.combo === "backbone",
          points,
          tips: points.map(([x, y]) => [name, `Richtung ${x}`, `Anteil: ${formatValue(y)}`]),
        };
      });
  const spectrumSeries = spectrumSeriesOf(spectrumKind === "roth" && snapshot.spectraRoth?.length ? snapshot.spectraRoth : snapshot.spectra, true);
  const classSpectrumSeries = spectrumSeriesOf(snapshot.classSpectraRoth, false);

  // Correlation with the test metric (Roth et al., Fig. 5), recomputed for the current selection.
  const isNumber = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);
  const basePoints = (snapshot.correlationPoints ?? []).filter(
    (p) => p.root === root && p.split === split && isNumber(p[target]) && keep(p.method, lossOfCombo(p.combo)),
  );
  const corrPoints = basePoints.filter((p) => isNumber(p[metric]) && (corrScope === "all" || p.k === kValue));
  const corrX = corrPoints.map((p) => p[metric] as number);
  const corrY = corrPoints.map((p) => p[target] as number);
  const corrPearson = pearson(corrX, corrY);
  const corrSpearman = corrX.length >= 3 ? pearson(ranks(corrX), ranks(corrY)) : null;
  const corrWithin = corrScope === "all" ? withinGroupPearson(corrX, corrY, corrPoints.map((p) => p.k)) : null;
  const groupKey = (p: (typeof corrPoints)[number]) =>
    colorBy === "loss" ? lossOfCombo(p.combo) : colorBy === "k" ? String(p.k) : p.method;
  const groupLabel = (key: string) =>
    colorBy === "loss" ? (LOSS_LABEL[key] ?? key) : colorBy === "k" ? `k=${key}` : (METHOD_LABEL[key] ?? key);
  const groupOrder = (a: string, b: string) =>
    colorBy === "k" ? Number(a) - Number(b) : colorBy === "method" ? METHOD_ORDER.indexOf(a) - METHOD_ORDER.indexOf(b) : a.localeCompare(b);
  const corrSeries: ChartSeries[] = unique(corrPoints.map(groupKey))
    .sort(groupOrder)
    .map((key) => {
      const members = corrPoints.filter((p) => groupKey(p) === key);
      return {
        name: groupLabel(key),
        color: colorFor(members[0].method, lossOfCombo(members[0].combo), members[0].k),
        points: members.map((p) => [p[metric] as number, p[target] as number] as [number, number]),
        tips: members.map((p) => [
          `${comboLabel(p.combo)}, k=${p.k}`,
          `${metricLabel}: ${formatValue(p[metric] as number)}`,
          `${targetLabel}: ${formatValue((p[target] as number) * 100)} %`,
          `Seed ${p.seed} · ${splitLabel}`,
        ]),
      };
    });
  const corrText =
    corrPoints.length < 3
      ? `n = ${corrPoints.length} fertige Läufe: r braucht mindestens 3`
      : `r = ${fmt(corrPearson)} · ρ_s = ${fmt(corrSpearman)}${corrWithin != null ? ` · r innerhalb k = ${fmt(corrWithin)}` : ""} · n = ${corrPoints.length}`;

  // Every measure against the test metric for the current filter: overall and within k.
  const barRows = snapshot.metrics
    .map((m) => {
      const pts = basePoints.filter((p) => isNumber(p[m]) && (corrScope === "all" || p.k === kValue));
      const xs = pts.map((p) => p[m] as number);
      const ys = pts.map((p) => p[target] as number);
      return {
        key: m,
        label: snapshot.metricLabels[m] ?? m,
        primary: corrScope === "all" ? withinGroupPearson(xs, ys, pts.map((p) => p.k)) : pearson(xs, ys),
        secondary: corrScope === "all" ? pearson(xs, ys) : xs.length >= 3 ? pearson(ranks(xs), ranks(ys)) : null,
        n: pts.length,
      };
    })
    .sort((a, b) => Math.abs(b.primary ?? 0) - Math.abs(a.primary ?? 0));
  const barPrimaryLabel = corrScope === "all" ? "r innerhalb k" : `r (k=${kValue})`;
  const barSecondaryLabel = corrScope === "all" ? "r über alle k" : `ρ_s (k=${kValue})`;

  const correlationScopes: (number | "all")[] = ["all", ...ks];
  const correlationCell = (m: string, scope: number | "all") => {
    const pts = basePoints.filter((p) => isNumber(p[m]) && (scope === "all" || p.k === scope));
    const xs = pts.map((p) => p[m] as number);
    const ys = pts.map((p) => p[target] as number);
    return { n: pts.length, pearson: pearson(xs, ys), spearman: xs.length >= 3 ? pearson(ranks(xs), ranks(ys)) : null };
  };
  const scopeLabel = (scope: number | "all") => (scope === "all" ? "alle k" : `k=${scope}`);
  const legendOf = (series: ChartSeries[]) => series.map((s) => ({ name: s.name, color: s.color, dashed: s.dashed }));
  const colorToggle = (
    <Toggle
      value={colorBy}
      onChange={setColorBy}
      options={[
        { value: "method", label: "Farbe: Verfahren" },
        { value: "loss", label: "Loss" },
        { value: "k", label: "k" },
      ]}
    />
  );

  return (
    <div className="mt-6">
      <div className="flex flex-wrap items-end gap-4">
        <Select label="Lauf" value={root} options={roots} onChange={setRoot} />
        <Select label="k" value={kValue} options={ks} onChange={setK} />
        <Select label="Split" value={split} options={snapshot.splits} onChange={setSplit} render={(s) => snapshot.splitLabels[s] ?? s} />
        <Select label="Kennzahl" value={metric} options={snapshot.metrics} onChange={setMetric} render={(m) => snapshot.metricLabels[m] ?? m} />
      </div>
      <div className="mt-3 flex flex-wrap items-end gap-5">
        <ChipFilter
          label="Verfahren"
          options={methodsAvailable}
          selected={methodSel}
          onChange={setMethodSel}
          render={(m) => METHOD_LABEL[m] ?? m}
          colorOf={colorOf}
        />
        <ChipFilter
          label="Loss"
          options={lossesAvailable}
          selected={lossSel}
          onChange={setLossSel}
          render={(l) => LOSS_LABEL[l] ?? l}
          colorOf={(l) => LOSS_COLOR[l] ?? "#c8d2e4"}
        />
        <span className="pb-1 text-[11px] text-[#7f8ba2]">Filter gelten für alle Grafiken · Doppelklick wählt nur diesen Eintrag</span>
      </div>
      <div className="mt-5 grid gap-4 lg:grid-cols-2">
        <FigureCard
          title={`${metricLabel} über das Training`}
          description="Mittel über Folds; gestrichelt vertikal: gewählte Epoche; weiß gestrichelt: Backbone. Maus über eine Kurve zeigt Lauf, Epoche und Wert."
          name={`geometrie-verlauf-${metric}-${slug}`}
          legend={legendOf(curveSeries)}
          rows={() => seriesRows(curveSeries, "epoch", metric)}
          controls={colorToggle}
        >
          <LineChart series={curveSeries} xLabel="Epoche" yLabel={metricLabel} />
          <Legend items={legendOf(curveSeries)} />
        </FigureCard>
        <FigureCard
          title={`Korrelation mit ${targetLabel}`}
          description={
            <>
              Ein Punkt pro Lauf (Maus darüber zeigt welcher): {metricLabel} an der validierungsgewählten Epoche ({splitLabel}) gegen{" "}
              {targetLabel}, wie Roth et al. (Fig. 5; dort R@1). Nur Läufe mit allen Folds fertig. <strong>Über alle k gemischt</strong> treibt
              vor allem die Labelzahl r: mehr Labels verändern sowohl die Geometrie als auch die Testleistung. Aussagekräftiger ist daher
              „r innerhalb k“ (je k zentriert) oder die Auswahl „nur k={kValue}“.
            </>
          }
          name={`korrelation-${metric}-vs-${target}-${slug}-${corrScope === "all" ? "alle-k" : `k${kValue}`}`}
          legend={legendOf(corrSeries)}
          rows={() =>
            corrPoints.map((p) => ({
              combo: p.combo,
              method: p.method,
              loss: lossOfCombo(p.combo),
              k: p.k,
              seed: p.seed,
              split: p.split,
              [metric]: p[metric] as number,
              [`test_${target}`]: p[target] as number,
            }))
          }
          controls={
            <>
              <Toggle value={target} onChange={setTarget} options={Object.entries(targets).map(([value, label]) => ({ value, label }))} />
              <Toggle
                value={corrScope}
                onChange={setCorrScope}
                options={[
                  { value: "all", label: "alle k" },
                  { value: "k", label: `nur k=${kValue}` },
                ]}
              />
              {colorToggle}
            </>
          }
        >
          <p className="mb-1 text-[11px] tabular-nums text-[#c8d2e4]">{corrText}</p>
          <ScatterChart series={corrSeries} xLabel={metricLabel} yLabel={targetLabel} />
          <Legend items={legendOf(corrSeries)} />
        </FigureCard>
        <FigureCard
          className="lg:col-span-2"
          exportFigure
          title={`Alle Kennzahlen gegen ${targetLabel} (${corrScope === "all" ? "innerhalb k" : `k=${kValue}`})`}
          description={
            corrScope === "all"
              ? "Dicker Balken: Pearson-r nach Abzug des k-Mittels (Zusammenhang bei gleicher Labelzahl). Dünner Balken: r über alle k gemischt. Sortiert nach |r innerhalb k|; Klick wählt die Kennzahl für die Grafiken oben."
              : `Dicker Balken: Pearson-r bei k=${kValue}; dünner Balken: Spearman. Klick wählt die Kennzahl.`
          }
          name={`korrelation-alle-${target}-${slug}-${corrScope}`}
          rows={() => barRows.map((r) => ({ metric: r.key, label: r.label, [barPrimaryLabel]: r.primary, [barSecondaryLabel]: r.secondary, n: r.n }))}
        >
          <CorrelationBars rows={barRows} primaryLabel={barPrimaryLabel} secondaryLabel={barSecondaryLabel} highlight={metric} onSelect={setMetric} />
        </FigureCard>
        <FigureCard
          title="Hypothese: überwacht − SSL über k"
          description="An der gewählten Epoche, je Verfahren gemittelt über die ausgewählten Losses (Tooltip zeigt die Einzelwerte). Für ρ gilt: > 0 heißt, der überwachte Kopf komprimiert stärker; die Hypothese erwartet einen mit k schrumpfenden positiven Abstand."
          name={`hypothese-${metric}-${root}-${split}${methodFilterSlug}${lossFilterSlug}`}
          legend={legendOf(hypothesisSeries)}
          rows={() => seriesRows(hypothesisSeries, "k", `sup_minus_ssl_${metric}`)}
        >
          <LineChart series={hypothesisSeries} xLabel="gelabelte Beispiele pro Klasse k" yLabel={`Δ ${metricLabel}`} logX zeroLine />
          <Legend items={legendOf(hypothesisSeries)} />
        </FigureCard>
        <FigureCard
          title="Klassenbedingtes Singulärwertspektrum (Roth et al., Anhang E)"
          description={`SVD je Klasse auf den unzentrierten Embeddings, die ersten 35 Singulärwerte, gemittelt über Klassen (wie Roths Fig. 11). k=${kValue}, ${splitLabel}.`}
          name={`klassenspektrum-${slug}`}
          legend={legendOf(classSpectrumSeries)}
          rows={() => seriesRows(classSpectrumSeries, "index", "singular_value")}
          controls={colorToggle}
        >
          {classSpectrumSeries.length ? (
            <LineChart series={classSpectrumSeries} xLabel="Index (sortiert)" yLabel="Singulärwert" />
          ) : (
            <p className="py-10 text-center text-xs text-[#7f8ba2]">Erst in Läufen ab dem 24.09.2026, ca. 18 Uhr, vorhanden.</p>
          )}
          <Legend items={legendOf(classSpectrumSeries)} />
        </FigureCard>
        <FigureCard
          className="lg:col-span-2"
          title="Singulärwertspektrum an der gewählten Epoche"
          description={
            <>
              {spectrumKind === "roth"
                ? "Normierte Singulärwerte der unzentrierten Embeddings wie bei Roth et al. (Fig. 7); ohne größten Wert = Grundlage von ρ."
                : "Varianzanteil je Richtung der zentrierten Embeddings (eigene Variante)."}{" "}
              Log-Skala, k={kValue}, {splitLabel}.
            </>
          }
          name={`spektrum-${spectrumKind}-${dropFirst}-${slug}`}
          legend={legendOf(spectrumSeries)}
          rows={() => seriesRows(spectrumSeries, "direction", spectrumKind === "roth" ? "singular_value_share" : "variance_share")}
          controls={
            <>
              <Toggle
                value={spectrumKind}
                onChange={setSpectrumKind}
                options={[
                  { value: "roth", label: "Roth et al." },
                  { value: "centered", label: "zentriert" },
                ]}
              />
              <Toggle
                value={dropFirst}
                onChange={setDropFirst}
                options={[
                  { value: "all", label: "alle Richtungen" },
                  { value: "drop", label: "ohne größten Singulärwert" },
                ]}
              />
              {colorToggle}
            </>
          }
        >
          <LineChart
            series={spectrumSeries}
            xLabel="Richtung"
            yLabel={spectrumKind === "roth" ? "Singulärwertanteil" : "Varianzanteil"}
            logY
            height={260}
            width={1080}
          />
          <Legend items={legendOf(spectrumSeries)} />
        </FigureCard>
        {basePoints.length > 0 && (
          <FigureCard
            className="lg:col-span-2"
            exportFigure={false}
            title={`Korrelation aller Kennzahlen mit ${targetLabel}`}
            description={`Pearson / Spearman · n, ${splitLabel}, für die aktuelle Auswahl an Verfahren und Losses. Die gewählte Kennzahl ist hervorgehoben.`}
            name={`korrelationstabelle-${target}-${slug}`}
            rows={() =>
              snapshot.metrics.flatMap((m) =>
                correlationScopes.map((scope) => ({ metric: m, metric_label: snapshot.metricLabels[m] ?? m, k: String(scope), ...correlationCell(m, scope) })),
              )
            }
            latex={() =>
              toLatexTable(
                [{ header: "Kennzahl" }, ...correlationScopes.map((scope) => ({ header: scopeLabel(scope), align: "r" as const }))],
                snapshot.metrics.map((m) => [
                  snapshot.metricLabels[m] ?? m,
                  ...correlationScopes.map((scope) => {
                    const c = correlationCell(m, scope);
                    return c.n >= 3 ? `${fmt(c.pearson)} / ${fmt(c.spearman)}` : "–";
                  }),
                ]),
                `Pearson- / Spearman-Korrelation der Geometriekennzahlen mit ${targetLabel} (${splitLabel}), ein Punkt pro Lauf.`,
                `tab:geometrie-korrelation-${target}-${split}`,
              )
            }
          >
            <div className="overflow-x-auto">
              <table className="mt-3 w-full min-w-[640px] text-xs">
                <thead>
                  <tr className="text-[10px] uppercase tracking-[0.1em] text-[#7f8ba2]">
                    <th className="pb-2 text-left font-semibold">Kennzahl</th>
                    {correlationScopes.map((scope) => (
                      <th key={String(scope)} className="pb-2 text-right font-semibold">
                        {scopeLabel(scope)}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {snapshot.metrics.map((m) => (
                    <tr
                      key={m}
                      onClick={() => setMetric(m)}
                      className={`cursor-pointer border-t border-white/[0.06] hover:bg-white/[0.03] ${m === metric ? "bg-white/[0.04]" : ""}`}
                    >
                      <td className={`py-1.5 ${m === metric ? "font-semibold text-white" : "text-[#c8d2e4]"}`}>{snapshot.metricLabels[m] ?? m}</td>
                      {correlationScopes.map((scope) => {
                        const c = correlationCell(m, scope);
                        return (
                          <td key={String(scope)} className="py-1.5 text-right tabular-nums text-white">
                            {c.n >= 3 ? fmt(c.pearson) : "–"}
                            {c.n >= 3 && <span className="text-[#7f8ba2]"> / {fmt(c.spearman)} · n={c.n}</span>}
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </FigureCard>
        )}
      </div>
      <p className="mt-4 flex items-start gap-2 text-[11px] text-[#7f8ba2]">
        <Info size={12} className="mt-0.5 shrink-0" />
        Vergleiche nur innerhalb eines Splits lesen: die Stichproben sind je Fold fest, aber die Splits unterscheiden sich in der Größe,
        und die Spektrumsschätzung hängt von der Stichprobengröße ab. Export: SVG/PNG im hellen Druckschema mit Legende im Bild, CSV mit den
        dargestellten Werten.
      </p>
    </div>
  );
}

/* ------------------------------------------------------------------ findings */

const LOSS_FALLBACK_COLORS = ["#ff9ecd", "#7dd3a8", "#ffb86b", "#6fa8ff", "#d4a5ff", "#e0e07a"];

/** Stable colour for any loss key, including losses added later. */
function lossColor(loss: string): string {
  if (LOSS_COLOR[loss]) return LOSS_COLOR[loss];
  let hash = 0;
  for (const ch of loss) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
  return LOSS_FALLBACK_COLORS[hash % LOSS_FALLBACK_COLORS.length];
}

const EFFECT_COLOR: Record<string, string> = {
  total: "#ffffff",
  data_effect_ssl_hp: "#4fd5e7",
  data_effect_sup_hp: "#7fb3ff",
  hp_effect_no_data: "#f4c86f",
  interaction: "#b8a0ff",
};

const CONTROL_ORDER: { key: string; label: string }[] = [
  { key: "pool_labeled", label: "Ungelabelte Bilder → gelabelte" },
  { key: "zero_weight", label: "SSL-Gewicht 0" },
  { key: "sup_all_shared", label: "Überwachte HPs + SSL" },
  { key: "no_warmup", label: "Ohne Warm-up" },
  { key: "pl_once", label: "Pseudo-Labels eingefroren" },
  { key: "graph_once", label: "Graph eingefroren" },
  { key: "graph_rewired", label: "Graph umverdrahtet" },
  { key: "graph_random_pruned", label: "Graph zufällig ausgedünnt" },
  { key: "graph_oracle_pruned", label: "Graph nur gleichklassig (Oracle)" },
  { key: "oracle_noise_0", label: "Oracle-Labels (0 % Rauschen)" },
  { key: "diag", label: "Diagnose (unverändert)" },
];

const PARAM_LABEL: Record<string, string> = {
  n_neighbors: "Anzahl Nachbarn",
  confidence_threshold: "Konfidenzschwelle",
  warmup_epochs: "Warm-up-Epochen",
  regularizer_target_ratio: "Zielverhältnis η",
  distance_scale: "Distanzskala s",
  alpha: "Propagationsstärke α",
  cluster_ratio: "Clusterverhältnis",
  update_interval_epochs: "Aktualisierungsintervall",
  basis_target_ratio: "Basis-Zielverhältnis",
};

interface DotPoint {
  row: number;
  x: number;
  color: string;
  slot: number;
  slots: number;
  tip: string[];
  hollow?: boolean;
}

/** One row per control, one dot per method × loss; x is the difference to the supervised reference. */
function DotPlot({
  rows,
  points,
  references,
  xLabel,
}: {
  rows: string[];
  points: DotPoint[];
  references: { x: number; color: string; label: string }[];
  xLabel: string;
}) {
  const width = 1080;
  const labelWidth = 250;
  const rowHeight = 26;
  const top = 18;
  const height = top + rows.length * rowHeight + 34;
  const xs = [0, ...points.map((p) => p.x), ...references.map((r) => r.x)];
  let [lo, hi] = [Math.min(...xs), Math.max(...xs)];
  const margin = Math.max(0.5, (hi - lo) * 0.06);
  lo -= margin;
  hi += margin;
  const sx = (x: number) => labelWidth + ((x - lo) / (hi - lo)) * (width - labelWidth - 16);
  const rowY = (row: number) => top + row * rowHeight + rowHeight / 2;
  const py = (p: DotPoint) => rowY(p.row) + (p.slots > 1 ? ((p.slot - (p.slots - 1) / 2) * (rowHeight - 8)) / p.slots : 0);
  const targets: HoverTarget[] = [
    ...points.map((p) => ({ px: sx(p.x), py: py(p), color: p.color, lines: p.tip })),
    ...references.map((r) => ({ px: sx(r.x), py: top + 4, color: r.color, lines: [r.label, `${fmtSigned(r.x)} PP ggü. überwacht`] })),
  ];
  const { hover, onMouseMove, onMouseLeave } = useHover(width, height, targets, 16);
  const step = Math.max(1, Math.round((hi - lo) / 8));
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  return (
    <svg data-export-root viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={xLabel} onMouseMove={onMouseMove} onMouseLeave={onMouseLeave}>
      {rows.map((label, i) => (
        <g key={label}>
          {i % 2 === 0 && <rect x={0} y={top + i * rowHeight} width={width} height={rowHeight} fill="#ffffff05" />}
          <text x={labelWidth - 10} y={rowY(i) + 3.5} textAnchor="end" fontSize="10.5" fill="#c8d2e4">
            {label}
          </text>
        </g>
      ))}
      {ticks.map((v) => (
        <g key={v}>
          <line x1={sx(v)} x2={sx(v)} y1={top} y2={height - 30} stroke={v === 0 ? "#ffffff66" : "#ffffff10"} />
          <text x={sx(v)} y={height - 18} textAnchor="middle" fontSize="9" fill="#7f8ba2">
            {v > 0 ? `+${v}` : v}
          </text>
        </g>
      ))}
      {references.map((r) => (
        <line key={r.label} x1={sx(r.x)} x2={sx(r.x)} y1={top} y2={height - 30} stroke={r.color} strokeOpacity={0.55} strokeDasharray="3 3" />
      ))}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={sx(p.x)}
          cy={py(p)}
          r={4.2}
          fill={p.hollow ? "transparent" : p.color}
          stroke={p.color}
          strokeWidth={1.5}
        />
      ))}
      <text x={sx(0)} y={top - 6} textAnchor="middle" fontSize="9" fill="#98a4ba">
        überwacht
      </text>
      <text x={(width + labelWidth) / 2} y={height - 3} textAnchor="middle" fontSize="10" fill="#98a4ba">
        {xLabel}
      </text>
      <Tooltip hover={hover} width={width} height={height} />
    </svg>
  );
}

/** Horizontal grouped bars: one row per method × loss, one bar per effect. */
function EffectRows({
  rows,
  effects,
}: {
  rows: { label: string; color: string; values: Record<string, { mean?: number; n: number } | undefined> }[];
  effects: { key: string; label: string; hint: string }[];
}) {
  const width = 1080;
  const labelWidth = 200;
  const bar = 7;
  const gap = 2;
  const rowHeight = effects.length * (bar + gap) + 12;
  const top = 18;
  const height = top + rows.length * rowHeight + 30;
  const values = rows.flatMap((r) => effects.map((e) => r.values[e.key]?.mean ?? 0));
  let [lo, hi] = [Math.min(0, ...values), Math.max(0, ...values)];
  const margin = Math.max(0.3, (hi - lo) * 0.05);
  lo -= margin;
  hi += margin;
  const sx = (x: number) => labelWidth + ((x - lo) / (hi - lo)) * (width - labelWidth - 16);
  const targets: HoverTarget[] = rows.flatMap((r, i) =>
    effects.map((e, j) => {
      const v = r.values[e.key]?.mean;
      return {
        px: sx(v ?? 0),
        py: top + i * rowHeight + 6 + j * (bar + gap) + bar / 2,
        color: EFFECT_COLOR[e.key] ?? "#fff",
        lines: [r.label, `${e.label}: ${v == null ? "–" : `${fmtSigned(v)} PP`}`, e.hint],
      };
    }),
  );
  const { hover, onMouseMove, onMouseLeave } = useHover(width, height, targets, 30);
  const step = Math.max(1, Math.round((hi - lo) / 8));
  const ticks: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) ticks.push(v);
  return (
    <svg data-export-root viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label="Gewinnzerlegung" onMouseMove={onMouseMove} onMouseLeave={onMouseLeave}>
      {ticks.map((v) => (
        <g key={v}>
          <line x1={sx(v)} x2={sx(v)} y1={top - 4} y2={height - 26} stroke={v === 0 ? "#ffffff66" : "#ffffff10"} />
          <text x={sx(v)} y={height - 14} textAnchor="middle" fontSize="9" fill="#7f8ba2">
            {v > 0 ? `+${v}` : v}
          </text>
        </g>
      ))}
      {rows.map((r, i) => {
        const y0 = top + i * rowHeight;
        return (
          <g key={r.label}>
            {i % 2 === 0 && <rect x={0} y={y0} width={width} height={rowHeight} fill="#ffffff05" />}
            <circle cx={10} cy={y0 + rowHeight / 2} r={4} fill={r.color} />
            <text x={20} y={y0 + rowHeight / 2 + 3.5} fontSize="10.5" fill="#e3e8f2">
              {r.label}
            </text>
            {effects.map((e, j) => {
              const v = r.values[e.key]?.mean;
              if (v == null) return null;
              return (
                <rect
                  key={e.key}
                  x={Math.min(sx(0), sx(v))}
                  y={y0 + 6 + j * (bar + gap)}
                  width={Math.max(1, Math.abs(sx(v) - sx(0)))}
                  height={bar}
                  rx={1.5}
                  fill={EFFECT_COLOR[e.key] ?? "#fff"}
                  fillOpacity={e.key === "total" ? 0.9 : 0.75}
                />
              );
            })}
          </g>
        );
      })}
      <text x={(width + labelWidth) / 2} y={height - 1} textAnchor="middle" fontSize="10" fill="#98a4ba">
        Effekt in Prozentpunkten Test-MAP@R
      </text>
      <Tooltip hover={hover} width={width} height={height} />
    </svg>
  );
}

function FindingsView({
  snapshot,
  dataset,
  k,
  aggregation,
}: {
  snapshot: AblationSnapshot;
  dataset: string;
  k: number;
  aggregation: Aggregation;
}) {
  const inScope = snapshot.variants.filter((v) => v.dataset === dataset && v.k === k);
  const lossesAvailable = unique(inScope.map((v) => v.loss)).sort();
  const methodsAvailable = unique(inScope.map((v) => v.method)).sort((a, b) => METHOD_ORDER.indexOf(a) - METHOD_ORDER.indexOf(b));
  const [lossSel, setLossSel] = useState<string[]>([]);
  const [methodSel, setMethodSel] = useState<string[]>([]);
  const keep = (method: string, loss: string) =>
    (lossSel.length === 0 || lossSel.includes(loss)) && (methodSel.length === 0 || methodSel.includes(method));
  const variants = inScope.filter((v) => keep(v.method, v.loss));
  const losses = unique(variants.map((v) => v.loss)).sort();
  const multiLoss = losses.length > 1;
  const labelOf = (method: string, loss: string) =>
    `${snapshot.labels.method[method] ?? method}${multiLoss ? ` · ${snapshot.labels.loss[loss] ?? loss}` : ""}`;
  const valueOf = (v: AblationVariant | undefined) => v?.[aggregation]?.mean;
  const find = (method: string, loss: string, key: string) => variants.find((v) => v.method === method && v.loss === loss && v.key === key);
  const supervisedOf = (loss: string) => valueOf(find("sup", loss, "baseline"));
  const baselineOf = (method: string, loss: string) => valueOf(find(method, loss, "baseline"));
  const slug = `${dataset}-k${k}${lossSel.length ? `-${lossSel.join("+")}` : ""}${methodSel.length ? `-${methodSel.join("+")}` : ""}-${aggregation}`;
  const datasetLabel = snapshot.labels.dataset[dataset] ?? dataset;
  const metric = snapshot.metricLabel;

  // 1. Gain decomposition from the complete 2x2 tables.
  const effects = EFFECTS.filter((e) => ["total", "data_effect_ssl_hp", "data_effect_sup_hp", "hp_effect_no_data", "interaction"].includes(e.key));
  const blocks = snapshot.twoByTwo
    .filter((b) => b.dataset === dataset && b.k === k && keep(b.method, b.loss) && b[aggregation])
    .sort((a, b) => METHOD_ORDER.indexOf(a.method) - METHOD_ORDER.indexOf(b.method) || a.loss.localeCompare(b.loss));
  const effectRows = blocks.map((b) => ({
    label: labelOf(b.method, b.loss),
    color: multiLoss ? lossColor(b.loss) : colorOf(b.method),
    values: Object.fromEntries(effects.map((e) => [e.key, b[aggregation]?.[e.key] as { mean?: number; n: number } | undefined])),
  }));
  const incomplete = unique(
    variants.filter((v) => v.method !== "sup" && v.key === "baseline").map((v) => `${v.method}|${v.loss}`),
  ).filter((key) => !blocks.some((b) => `${b.method}|${b.loss}` === key));

  // 2. Controls against the supervised reference.
  const combos = unique(variants.filter((v) => v.method !== "sup").map((v) => `${v.method}|${v.loss}`)).sort((a, b) => {
    const [ma, la] = a.split("|");
    const [mb, lb] = b.split("|");
    return METHOD_ORDER.indexOf(ma) - METHOD_ORDER.indexOf(mb) || la.localeCompare(lb);
  });
  const controlRows = CONTROL_ORDER.filter((c) => variants.some((v) => v.key === c.key));
  const dotPoints: DotPoint[] = controlRows.flatMap((control, row) => {
    const present = combos.filter((combo) => {
      const [method, loss] = combo.split("|");
      return valueOf(find(method, loss, control.key)) != null && supervisedOf(loss) != null;
    });
    return present.map((combo, slot) => {
      const [method, loss] = combo.split("|");
      const value = valueOf(find(method, loss, control.key)) as number;
      const sup = supervisedOf(loss) as number;
      const base = baselineOf(method, loss);
      return {
        row,
        x: value - sup,
        color: colorOf(method),
        slot,
        slots: present.length,
        hollow: multiLoss && losses.indexOf(loss) % 2 === 1,
        tip: [
          `${labelOf(method, loss)}: ${control.label}`,
          `${metric} ${fmt(value)} · ${fmtSigned(value - sup)} PP ggü. überwacht`,
          base != null ? `${fmtSigned(value - base)} PP ggü. eigener SSL-Baseline (${fmt(base)})` : "keine SSL-Baseline in diesem Ordner",
        ],
      };
    });
  });
  const references = combos.flatMap((combo) => {
    const [method, loss] = combo.split("|");
    const base = baselineOf(method, loss);
    const sup = supervisedOf(loss);
    return base != null && sup != null ? [{ x: base - sup, color: colorOf(method), label: `SSL-Baseline ${labelOf(method, loss)}` }] : [];
  });

  // 3. LP label quality: dose-response and every LP variant with diagnostics.
  const lpVariants = variants.filter((v) => v.method === "lp");
  const noiseSeries: ChartSeries[] = losses.flatMap((loss) => {
    const oracle = lpVariants
      .filter((v) => v.loss === loss && /^oracle_noise_\d+$/.test(v.key) && valueOf(v) != null)
      .map((v) => {
        const nominal = Number(v.key.split("_").pop());
        const measured = v.pseudo_label_quality?.accuracy_last;
        return { v, x: measured != null ? 100 * (1 - measured) : nominal, nominal };
      })
      .sort((a, b) => a.x - b.x);
    const out: ChartSeries[] = [];
    if (oracle.length) {
      out.push({
        name: `Oracle + zufälliges Rauschen${multiLoss ? ` · ${snapshot.labels.loss[loss] ?? loss}` : ""}`,
        color: multiLoss ? lossColor(loss) : "#9fb0cc",
        points: oracle.map((o) => [o.x, valueOf(o.v) as number] as [number, number]),
        tips: oracle.map((o) => [
          `Oracle, ${o.nominal} % Rauschen (nominell)`,
          `gemessene Fehlerrate ${fmt(o.x, 1)} % · Abdeckung ${o.v.pseudo_label_quality?.accepted_share_last != null ? `${fmt(100 * o.v.pseudo_label_quality.accepted_share_last, 0)} %` : "–"}`,
          `${metric} ${fmt(valueOf(o.v))}`,
        ]),
      });
    }
    const measured = lpVariants.find((v) => v.loss === loss && (v.key === "diag" || v.key === "baseline") && v.pseudo_label_quality?.accuracy_last != null);
    const lpValue = baselineOf("lp", loss) ?? valueOf(measured);
    if (measured?.pseudo_label_quality?.accuracy_last != null && lpValue != null) {
      const x = 100 * (1 - measured.pseudo_label_quality.accuracy_last);
      out.push({
        name: `LP (gemessene Fehlerrate)${multiLoss ? ` · ${snapshot.labels.loss[loss] ?? loss}` : ""}`,
        color: "#4fd5e7",
        points: [[x, lpValue]],
        tips: [[
          "LP, echte Pseudo-Labels",
          `Fehlerrate ${fmt(x, 1)} % (letzte Aktualisierung) · Abdeckung ${fmt(100 * (measured.pseudo_label_quality.accepted_share_last ?? Number.NaN), 0)} %`,
          `${metric} ${fmt(lpValue)}`,
        ]],
      });
    }
    const sup = supervisedOf(loss);
    const xsAll = [...oracle.map((o) => o.x), ...(out[out.length - 1]?.points.map((p) => p[0]) ?? [])];
    if (sup != null && xsAll.length) {
      out.push({
        name: `Überwacht${multiLoss ? ` · ${snapshot.labels.loss[loss] ?? loss}` : ""}`,
        color: "#8895aa",
        dashed: true,
        points: [[Math.min(0, ...xsAll), sup], [Math.max(...xsAll), sup]],
        tips: [[`Überwacht: ${metric} ${fmt(sup)}`], [`Überwacht: ${metric} ${fmt(sup)}`]],
      });
    }
    return out;
  });

  const qualityGroups: { name: string; color: string; test: (v: AblationVariant) => boolean }[] = [
    { name: "LP (Baseline/Diagnose)", color: "#4fd5e7", test: (v) => v.key === "diag" || v.key === "baseline" },
    { name: "Oracle-Rauschen", color: "#9fb0cc", test: (v) => v.key.startsWith("oracle_noise") },
    { name: "Konfidenzschwelle", color: "#f4c86f", test: (v) => v.sweep?.param === "confidence_threshold" },
    { name: "Nachbarn / Warm-up", color: "#b8e56e", test: (v) => v.sweep != null && v.sweep.param !== "confidence_threshold" },
    { name: "Kontrollen", color: "#f08a7e", test: () => true },
  ];
  const withQuality = lpVariants.filter((v) => v.pseudo_label_quality?.accuracy_last != null && valueOf(v) != null);
  const qualitySeries: ChartSeries[] = qualityGroups
    .map((group, i) => {
      const members = withQuality.filter((v) => group.test(v) && !qualityGroups.slice(0, i).some((g) => g.test(v)));
      return {
        name: group.name,
        color: group.color,
        points: members.map((v) => [100 * (v.pseudo_label_quality!.accuracy_last as number), valueOf(v) as number] as [number, number]),
        tips: members.map((v) => [
          `${v.label_de}${multiLoss ? ` · ${snapshot.labels.loss[v.loss] ?? v.loss}` : ""}`,
          `Label-Genauigkeit ${fmt(100 * (v.pseudo_label_quality!.accuracy_last as number), 1)} % · Abdeckung ${fmt(100 * (v.pseudo_label_quality!.accepted_share_last ?? Number.NaN), 0)} %`,
          `${metric} ${fmt(valueOf(v))}`,
        ]),
      };
    })
    .filter((s) => s.points.length > 0);

  // 4. One-at-a-time sweeps: one panel per method and parameter, one line per loss.
  const panels = unique(variants.filter((v) => v.sweep).map((v) => `${v.method}|${v.sweep!.param}`)).sort((a, b) => {
    const [ma, pa] = a.split("|");
    const [mb, pb] = b.split("|");
    return METHOD_ORDER.indexOf(ma) - METHOD_ORDER.indexOf(mb) || pa.localeCompare(pb);
  });
  const panelOf = (panel: string) => {
    const [method, param] = panel.split("|");
    const series: ChartSeries[] = [];
    let tunedMarker: number | null = null;
    const allX: number[] = [];
    for (const loss of losses) {
      const points = variants
        .filter((v) => v.method === method && v.loss === loss && v.sweep?.param === param && valueOf(v) != null)
        .map((v) => ({ x: v.sweep!.value, y: valueOf(v) as number, tip: v.label_de }));
      if (!points.length) continue;
      const tuned = variants.find((v) => v.method === method && v.loss === loss && v.sweep?.param === param)?.sweep?.tuned ?? null;
      const base = baselineOf(method, loss);
      if (tuned != null && base != null && !points.some((p) => Math.abs(p.x - tuned) <= 1e-9 * Math.max(1, Math.abs(tuned)))) {
        points.push({ x: tuned, y: base, tip: "getunter Wert (SSL-Baseline)" });
      }
      if (tuned != null) tunedMarker = tuned;
      points.sort((a, b) => a.x - b.x);
      allX.push(...points.map((p) => p.x));
      const name = multiLoss ? (snapshot.labels.loss[loss] ?? loss) : (snapshot.labels.method[method] ?? method);
      series.push({
        name,
        color: multiLoss ? lossColor(loss) : colorOf(method),
        marker: tuned,
        points: points.map((p) => [p.x, p.y] as [number, number]),
        tips: points.map((p) => [
          `${labelOf(method, loss)}: ${PARAM_LABEL[param] ?? param} = ${formatValue(p.x)}`,
          p.tip,
          `${metric} ${fmt(p.y)}${base != null ? ` · ${fmtSigned(p.y - base)} PP ggü. Baseline` : ""}`,
        ]),
      });
    }
    const sup = losses.map(supervisedOf).find((v) => v != null);
    if (sup != null && allX.length) {
      series.push({
        name: "Überwacht",
        color: "#8895aa",
        dashed: true,
        points: [[Math.min(...allX), sup], [Math.max(...allX), sup]],
        tips: [[`Überwacht: ${metric} ${fmt(sup)}`], [`Überwacht: ${metric} ${fmt(sup)}`]],
      });
    }
    const positive = allX.length > 0 && Math.min(...allX) > 0;
    const logX = positive && Math.max(...allX) / Math.min(...allX) >= 20;
    return { method, param, series, logX, tunedMarker };
  };

  const legendOf = (series: ChartSeries[]) => series.map((s) => ({ name: s.name, color: s.color, dashed: s.dashed }));
  const methodLegend = combos
    .map((c) => c.split("|")[0])
    .filter((m, i, all) => all.indexOf(m) === i)
    .map((m) => ({ name: snapshot.labels.method[m] ?? m, color: colorOf(m) }));

  return (
    <div className="mt-6">
      <div className="flex flex-wrap items-end gap-5">
        <ChipFilter
          label="Verfahren"
          options={methodsAvailable}
          selected={methodSel}
          onChange={setMethodSel}
          render={(m) => snapshot.labels.method[m] ?? m}
          colorOf={colorOf}
        />
        <ChipFilter
          label="Loss"
          options={lossesAvailable}
          selected={lossSel}
          onChange={setLossSel}
          render={(l) => snapshot.labels.loss[l] ?? l}
          colorOf={lossColor}
        />
        <span className="pb-1 text-[11px] text-[#7f8ba2]">
          {datasetLabel}, k={k} · ein Seed je Variante · Werte: Test-{metric} ({aggregation === "mean" ? "Mittel über Folds" : "Concat"}) · Maus über
          Balken/Punkte für Details
        </span>
      </div>

      <div className="mt-5 grid gap-4">
        <FigureCard
          title="1 · Woher kommt der Gewinn? (2×2-Zerlegung)"
          description={
            <>
              Weiß: Gesamtgewinn ggü. überwacht. Cyan: Effekt der ungelabelten Bilder mit SSL-HPs, hellblau mit überwachten HPs. Gelb: Effekt der
              SSL-HPs ohne ungelabelte Bilder. Violett: Interaktion. Liegt Gelb bei 0 und Cyan nahe am Gesamtgewinn, kommt der Gewinn aus den Daten.
              {incomplete.length > 0 && (
                <> Noch unvollständig (fehlende Zellen): {incomplete.map((key) => labelOf(key.split("|")[0], key.split("|")[1])).join(", ")}.</>
              )}
            </>
          }
          name={`befunde-2x2-${slug}`}
          legend={effects.map((e) => ({ name: e.label, color: EFFECT_COLOR[e.key] ?? "#fff" }))}
          rows={() => effectRows.flatMap((r) => effects.map((e) => ({ row: r.label, effect: e.label, value: r.values[e.key]?.mean ?? null })))}
        >
          {effectRows.length ? (
            <>
              <EffectRows rows={effectRows} effects={effects} />
              <Legend items={effects.map((e) => ({ name: e.label, color: EFFECT_COLOR[e.key] ?? "#fff" }))} />
            </>
          ) : (
            <p className="py-8 text-center text-xs text-[#7f8ba2]">Keine vollständige 2×2-Tabelle für diese Auswahl.</p>
          )}
        </FigureCard>

        <FigureCard
          title="2 · Kontrollen: was kostet welcher Eingriff?"
          description={
            <>
              Jeder Punkt ist ein Lauf, x = Differenz zur überwachten Referenz (0-Linie). Gestrichelte senkrechte Linien: die unveränderte
              SSL-Baseline je Verfahren. Ein Punkt links seiner Linie heißt: der Eingriff kostet Leistung.
              {multiLoss && " Bei mehreren Losses sind die Punkte je Zeile gestaffelt; hohl = jeder zweite Loss (Tooltip nennt ihn)."} Oracle-Zeilen
              nutzen verdeckte Labels und sind nur Diagnose.
            </>
          }
          name={`befunde-kontrollen-${slug}`}
          legend={methodLegend}
          rows={() => dotPoints.map((p) => ({ control: controlRows[p.row].label, run: p.tip[0], diff_vs_supervised: p.x }))}
        >
          {dotPoints.length ? (
            <>
              <DotPlot rows={controlRows.map((c) => c.label)} points={dotPoints} references={references} xLabel={`Differenz zu überwacht (PP Test-${metric})`} />
              <Legend items={methodLegend} />
            </>
          ) : (
            <p className="py-8 text-center text-xs text-[#7f8ba2]">Keine Kontrollen für diese Auswahl.</p>
          )}
        </FigureCard>

        <div className="grid gap-4 lg:grid-cols-2">
          <FigureCard
            title="3a · LP: Dosis-Wirkung der Label-Fehler"
            description="Oracle-Labels mit zufällig vertauschtem Anteil, x = gemessene Fehlerrate der übernommenen Labels. Der LP-Punkt steht bei seiner gemessenen Fehlerrate. Liegt er über der Kurve, schaden LPs strukturierte Fehler weniger als zufällige."
            name={`befunde-dosis-${slug}`}
            legend={legendOf(noiseSeries)}
            rows={() => seriesRows(noiseSeries, "error_rate_percent", metric)}
          >
            {noiseSeries.length ? (
              <>
                <LineChart series={noiseSeries} xLabel="Fehlerrate der übernommenen Labels (%)" yLabel={`Test-${metric}`} />
                <Legend items={legendOf(noiseSeries)} />
              </>
            ) : (
              <p className="py-8 text-center text-xs text-[#7f8ba2]">Keine Oracle-Läufe für diese Auswahl.</p>
            )}
          </FigureCard>
          <FigureCard
            title="3b · LP: Label-Genauigkeit gegen Leistung"
            description="Alle LP-Varianten mit Pseudo-Label-Diagnose. Tooltip zeigt zusätzlich die Abdeckung: Hohe Schwellen erhöhen die Genauigkeit, aber übernehmen kaum noch Labels."
            name={`befunde-genauigkeit-${slug}`}
            legend={legendOf(qualitySeries)}
            rows={() => seriesRows(qualitySeries, "label_accuracy_percent", metric)}
          >
            {qualitySeries.length ? (
              <>
                <ScatterChart series={qualitySeries} xLabel="Genauigkeit der übernommenen Pseudo-Labels (%)" yLabel={`Test-${metric}`} />
                <Legend items={legendOf(qualitySeries)} />
              </>
            ) : (
              <p className="py-8 text-center text-xs text-[#7f8ba2]">Keine LP-Diagnosen für diese Auswahl.</p>
            )}
          </FigureCard>
        </div>

        {panels.length > 0 && (
          <div>
            <h3 className="mt-2 text-sm font-semibold text-white">4 · Einzelparameter-Sweeps</h3>
            <p className="mt-1 text-[11px] text-[#7f8ba2]">
              Je Panel ein Verfahren und ein Hyperparameter, alle übrigen auf dem getunten Wert. Gestrichelt senkrecht: getunter Wert (dort liegt der
              Baseline-Punkt); grau gestrichelt: überwacht.{multiLoss ? " Eine Linie je Loss." : ""}
            </p>
            <div className="mt-3 grid gap-4 lg:grid-cols-2">
              {panels.map((panel) => {
                const { method, param, series, logX } = panelOf(panel);
                return (
                  <FigureCard
                    key={panel}
                    title={`${snapshot.labels.method[method] ?? method}: ${PARAM_LABEL[param] ?? param}`}
                    description={logX ? "Log-Achse." : undefined}
                    name={`befunde-sweep-${method}-${param}-${slug}`}
                    legend={legendOf(series)}
                    rows={() => seriesRows(series, param, metric)}
                  >
                    <LineChart series={series} xLabel={PARAM_LABEL[param] ?? param} yLabel={`Test-${metric}`} logX={logX} />
                    <Legend items={legendOf(series)} />
                  </FigureCard>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ live data */

const POLL_MS = 60_000;

/**
 * Poll the dev server's /api/phase2 so finished runs appear without a manual export.
 * In the static export the route does not exist; the page then keeps the build-time snapshot.
 */
function useLiveSnapshot<T>(kind: "ablations" | "geometry", initial: T | null, enabled: boolean) {
  const [data, setData] = useState<T | null>(initial);
  const [live, setLive] = useState(false);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);
  useEffect(() => {
    if (!enabled) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const poll = async () => {
      try {
        const response = await fetch(`/api/phase2/?kind=${kind}`, { cache: "no-store" });
        if (!response.ok) throw new Error(String(response.status));
        const next = (await response.json()) as T | null;
        if (cancelled) return;
        if (next) setData(next);
        setLive(true);
        setCheckedAt(new Date());
      } catch {
        if (!cancelled) setLive(false);
      }
      if (!cancelled) timer = setTimeout(poll, POLL_MS);
    };
    void poll();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [kind, enabled]);
  return { data, live, checkedAt };
}

/* ------------------------------------------------------------------ page */

function Missing({ what, command }: { what: string; command: string }) {
  return (
    <div className="mt-8 max-w-3xl">
      <p className="text-sm text-[#98a4ba]">{what}</p>
      <pre className="mt-3 overflow-x-auto rounded-xl border border-white/[0.07] bg-black/40 p-4 text-[11px] leading-relaxed text-[#c8d2e4]">{command}</pre>
    </div>
  );
}

/**
 * `shared` is the hosted build (npm run build:share): no dev-server polling, no link back
 * to the local atlas, and only the tabs whose snapshot was bundled.
 */
export function Phase2Explorer({
  ablations: initialAblations,
  geometry: initialGeometry,
  shared = false,
}: {
  ablations: AblationSnapshot | null;
  geometry: GeometrySnapshot | null;
  shared?: boolean;
}) {
  const { data: ablations, live, checkedAt } = useLiveSnapshot<AblationSnapshot>("ablations", initialAblations, !shared);
  const { data: geometry } = useLiveSnapshot<GeometrySnapshot>("geometry", initialGeometry, !shared);
  const [tab, setTab] = useState<Tab>(initialAblations ? "findings" : "geometry");
  const [aggregation, setAggregation] = useState<Aggregation>("mean");
  const variants = ablations?.variants ?? [];
  const datasets = useMemo(() => unique(variants.map((v) => v.dataset)).sort(), [variants]);
  const [dataset, setDataset] = useState(datasets[0] ?? "");
  const datasetValue = datasets.includes(dataset) ? dataset : (datasets[0] ?? "");
  const ks = unique(variants.filter((v) => v.dataset === datasetValue).map((v) => v.k)).sort((a, b) => a - b);
  const [k, setK] = useState<number>(ks[0] ?? 0);
  const losses = unique(variants.filter((v) => v.dataset === datasetValue).map((v) => v.loss)).sort();
  const [loss, setLoss] = useState(losses[0] ?? "");
  const kValue = ks.includes(k) ? k : ks[0];
  const lossValue = losses.includes(loss) ? loss : losses[0];

  const tabs: { value: Tab; label: string }[] = [
    { value: "findings" as Tab, label: "Befunde" },
    { value: "twoByTwo" as Tab, label: "2×2-Ablation" },
    { value: "variants" as Tab, label: "Alle Varianten" },
    { value: "geometry" as Tab, label: "Geometrie (B6)" },
  ].filter((t) => !shared || (t.value === "geometry" ? geometry != null : ablations != null));
  const geometryOnly = shared && ablations == null;

  return (
    <main className="mx-auto max-w-[1180px] px-5 pb-20 pt-8">
      {shared ? (
        <ShareNav current="/" />
      ) : (
        <Link href="/" className="focus-ring text-xs font-semibold text-[#8895aa] hover:text-white">
          <ArrowLeft size={14} className="mr-1 inline" />
          Back to Trial Atlas
        </Link>
      )}
      <header className={shared ? "" : "mt-5"}>
        <div className="flex flex-wrap items-center gap-3">
          <FlaskConical size={20} className="text-[#b8e56e]" />
          <h1 className="text-2xl font-semibold text-white">{geometryOnly ? "Phase 2 · Embedding-Geometrie" : "Phase 2"}</h1>
          {ablations && (
            <span className="rounded-full bg-[#b8e56e]/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#c7ef85]">
              Test-{ablations.metricLabel} · Stand {ablations.generatedAt.slice(0, 16).replace("T", " ")} UTC
            </span>
          )}
          {geometryOnly && geometry && (
            <span className="rounded-full bg-[#b8e56e]/10 px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] text-[#c7ef85]">
              Stand {geometry.generatedAt.slice(0, 16).replace("T", " ")} UTC
            </span>
          )}
          {!shared && (
          <span
            className={`flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.08em] ${
              live ? "bg-[#4fd5e7]/10 text-[#7fe3f0]" : "bg-white/[0.05] text-[#7f8ba2]"
            }`}
            title={live ? "Aktualisiert sich jede Minute aus den Logs (Dev-Server)" : "Statischer Snapshot aus public/"}
          >
            <Radio size={11} />
            {live ? `Live · geprüft ${checkedAt?.toLocaleTimeString() ?? ""}` : "Snapshot"}
          </span>
          )}
        </div>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-[#98a4ba]">
          {geometryOnly ? (
            "Was verändern die SSL-Signale im Embedding-Raum? Roth-ρ, π-Verhältnisse, Spektren und Klassenabstände je Verfahren, Verlust und Label-Budget über die Epochen."
          ) : (
            <>
          Woher kommt der Gewinn der SSL-Verfahren? Die 2×2-Ablation trennt ungelabelte Daten von den Hyperparametern,
          die benannten Kontrollen prüfen einzelne Mechanismen, und die Geometrie zeigt, was die SSL-Signale im Embedding-Raum verändern.
            </>
          )}
        </p>
      </header>

      <div className="mt-6 flex flex-wrap items-end gap-4">
        {tabs.length > 1 && <Toggle value={tab} onChange={setTab} options={tabs} />}
        {tab !== "geometry" && ablations && (
          <>
            <Select label="Datensatz" value={datasetValue} options={datasets} onChange={setDataset} render={(d) => ablations.labels.dataset[d] ?? d} />
            <Select label="k" value={kValue} options={ks} onChange={setK} />
            {tab !== "findings" && (
              <Select label="Verlust" value={lossValue} options={losses} onChange={setLoss} render={(l) => ablations.labels.loss[l] ?? l} />
            )}
            <Toggle
              value={aggregation}
              onChange={setAggregation}
              options={[
                { value: "mean", label: "Mittel über Folds" },
                { value: "concat", label: "Concat" },
              ]}
            />
          </>
        )}
      </div>

      {tab === "findings" &&
        (ablations ? (
          <FindingsView snapshot={ablations} dataset={datasetValue} k={kValue} aggregation={aggregation} />
        ) : (
          <Missing what="Noch kein Ablations-Snapshot (public/ablations.json)." command="python main_eval/ablation_report.py --app react_trial_visualizer" />
        ))}
      {tab === "twoByTwo" &&
        (ablations ? (
          <TwoByTwoView snapshot={ablations} dataset={datasetValue} k={kValue} loss={lossValue} aggregation={aggregation} />
        ) : (
          <Missing what="Noch kein Ablations-Snapshot (public/ablations.json)." command="python main_eval/ablation_report.py --app react_trial_visualizer" />
        ))}
      {tab === "variants" &&
        (ablations ? (
          <VariantsView snapshot={ablations} dataset={datasetValue} k={kValue} loss={lossValue} aggregation={aggregation} />
        ) : (
          <Missing what="Noch kein Ablations-Snapshot (public/ablations.json)." command="python main_eval/ablation_report.py --app react_trial_visualizer" />
        ))}
      {tab === "geometry" &&
        (geometry ? (
          <GeometryView snapshot={geometry} />
        ) : (
          <Missing
            what="Noch kein Geometrie-Snapshot (public/geometry.json). Erst die Läufe mit Geometrie-Tracking starten, dann exportieren:"
            command={`python -m training.run_scheduler --manifest configs/run_manifests/phase2/cars_cir_geometry.json --gpus 0 --max-total-runs 4
python main_eval/geometry_tracking.py --roots logs/final/phase_2/cars_geometry --app react_trial_visualizer`}
          />
        ))}
    </main>
  );
}
